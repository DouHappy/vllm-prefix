# Adapted from
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
import asyncio
import json
import time
from http import HTTPStatus
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import fastapi

import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from packaging import version

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse,
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import random_uuid
from .prefix_group import PrefixGroup
from vllm.entrypoints.openai.protocol import SchedulePrefixRequest, SchedulePrefixResponse

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False

TIMEOUT_KEEP_ALIVE = 5  # seconds

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()
engine = None


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


async def get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")

    conv = get_conversation_template(request.model)
    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    return prompt


async def check_length(
    request: Union[ChatCompletionRequest, CompletionRequest],
    prompt: Optional[str] = None,
    prompt_ids: Optional[List[int]] = None
) -> Tuple[List[int], Optional[JSONResponse]]:
    assert (not (prompt is None and prompt_ids is None)
            and not (prompt is not None and prompt_ids is not None)
            ), "Either prompt or prompt_ids should be provided."
    if prompt_ids is not None:
        input_ids = prompt_ids
    else:
        input_ids = tokenizer(prompt).input_ids
    token_num = len(input_ids)

    if request.max_tokens is None:
        request.max_tokens = max_model_len - token_num
    if token_num + request.max_tokens > max_model_len:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {max_model_len} tokens. "
            f"However, you requested {request.max_tokens + token_num} tokens "
            f"({token_num} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )
    else:
        return input_ids, None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    model_cards = [
        ModelCard(id=served_model,
                  root=served_model,
                  permission=[ModelPermission()])
    ]
    return ModelList(data=model_cards)


def create_logprobs(token_ids: List[int],
                    id_logprobs: List[Dict[int, float]],
                    initial_text_offset: int = 0) -> LogProbs:
    """Create OpenAI-style logprobs."""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] +
                                        last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append({
            tokenizer.convert_ids_to_tokens(i): p
            for i, p in id_logprob.items()
        })
    return logprobs


@app.post("/delete_prefix/")
async def delete_prefix(request: ChatCompletionRequest) -> Response:
    logger.info(f"Received prefix deletion request: {request}")
    prompt = await get_gen_prompt(request)
    prefix_tokens = tokenizer(prompt).input_ids
    logger.info(f"prefix tokens going to delete: {prefix_tokens}")
    if engine.delete_prefix(prefix_tokens) != None:
        JSONResponse({"response": "ok"})
    else:
        JSONResponse({"response": "delete error"})

@app.post("/schedule_prefix")
async def schedule_prefix(request: SchedulePrefixRequest,
                          raw_request: Request):
    logger.info(f"free blocks number {engine.engine.scheduler.block_manager.get_num_free_gpu_blocks()}")
    logger.info(f"Received schedule prefix request: {request}")
    warmup_time = 0
    query_time = 0
    start_t = time.monotonic()

    # sub_messages = request.sub_message
    # messages = request.messages_list
    prefix_groups:List[PrefixGroup] = []
    # count total number of request for each prefix
    messages_dict:Dict[str,List[Tuple[int, List]]] = {}
    current_prefixes_token_list: List[List[int]] = []
    prefix_trie = engine.engine.scheduler.prefix_trie
    logger.info(f"prefix_tree's size {prefix_trie.root.size}. as follow:")
    last_prefixes_list = deepcopy(prefix_trie.get_prefix_list())
    for prefix in last_prefixes_list:
        logger.info(f"{prefix}\n")
    for ind, (message, sub_message_list) in enumerate(
        request.message_and_submessage):

        # put them all in prefix_trie but share block
        # block will be shared by prefix_trie.update_block()
        if not isinstance(sub_message_list[0], list):
            sub_message_list = [sub_message_list]
        
        real_sub_message = ([], 0)
        for sub_message in sub_message_list:
            # to using ready-made function,
            # simply put sub_message in request's message
            request.messages = sub_message
            prefix_str = await get_gen_prompt(request)
            prefix_tokens = tokenizer(prefix_str).input_ids
            current_prefixes_token_list.append(prefix_tokens)
            prefix_trie.add_prefix(prefix_tokens)
            # using the longest prefix
            if real_sub_message[1] < len(prefix_tokens):
                real_sub_message = (sub_message, len(prefix_tokens))

        str_sub_message = str(real_sub_message[0])
        if str_sub_message in messages_dict:
            messages_dict[str_sub_message].append((ind, message))
        else:
            messages_dict[str_sub_message] = [(ind, message)]

    # if prefix not used in current request list, we consider to delete.
    for prefix in last_prefixes_list:
        if prefix.token_ids not in current_prefixes_token_list:
            engine.delete_prefix(prefix.token_ids)

    response: SchedulePrefixResponse = SchedulePrefixResponse(outputs=[])
    response.outputs = [None] * len(request.message_and_submessage)
    # init prefix group
    for str_sub_message, messages in messages_dict.items():
        sub_message = eval(str_sub_message)
        request.messages = sub_message
        prefix = await get_gen_prompt(request)
        prefix_tokens = tokenizer(prefix).input_ids
        prompts_tokens_list = []
        query_ids = []
        messages_list = []
        for ind, message in messages:
            request.messages = message
            query_ids.append(ind)
            messages_list.append(message)
            message_prompt = await get_gen_prompt(request)
            prompt_tokens = tokenizer(message_prompt).input_ids
            prompts_tokens_list.append(prompt_tokens)
            
        prefix_groups.append(PrefixGroup(
            sub_message=sub_message,
            messages_list=messages_list,
            block_size=engine.engine.cache_config.block_size,
            prefix_tokens=prefix_tokens,
            prompts_tokens_list=prompts_tokens_list,
            query_ids=query_ids,
        ))

    logger.info(f"init prefix_group: {prefix_groups}")
    # sort by needed block number
    sorted(prefix_groups, key=lambda x:x.get_block_num(), reverse=True)
    
    async def batch_query(query_list: List[PrefixGroup],
                          request: SchedulePrefixRequest,
                          warmup=False):
        logger.info(f"free blocks number {engine.engine.scheduler.block_manager.get_num_free_gpu_blocks()}")
        var_dict = vars(request)
        # logger.info(f"at batch_query schedule prefix request: {request}")
        # maybe message_and_submessage not in var_dict. Strange.
        # Maybe some bugs i didn't noticed.
        if 'message_and_submessage' in var_dict:
            var_dict.pop('message_and_submessage')
        request_temp = ChatCompletionRequest(**var_dict)
        tasks = []
        for query in query_list:
            temp_request = deepcopy(request_temp)
            # we don't need to generate anything here but max_token must > 0
            if warmup:
                temp_request.max_tokens = 1
                temp_request.messages = query.sub_message
                temp_request.prefix_pos = query.prefix_pos
                task = asyncio.create_task(
                    create_chat_completion(temp_request, raw_request)
                    )
                tasks.append(task)
            else:
                for message in query.messages_list:
                    temp_request.messages = message
                    temp_request.prefix_pos = query.prefix_pos
                    task = asyncio.create_task(
                        create_chat_completion(temp_request, raw_request)
                        )
                    tasks.append(task)
        results = await asyncio.gather(*tasks)

        return results

    logger.info(f"ready for warmup and query batch")
    while prefix_groups:
        query_list:List[PrefixGroup] = []
        cur_free_block_num = \
            engine.engine.scheduler.block_manager.get_num_free_gpu_blocks()
        for group in prefix_groups:
            if group.can_alloc(cur_free_block_num):
                query_list.append(group)
                cur_free_block_num -= group.block_num
        
        if query_list == []:
            break

        logger.info(f"fill query list with {len(query_list)} batch")
        end_t1 = time.monotonic()
        # warmup batch
        results = await batch_query(query_list, request, warmup=True)
        end_t2 = time.monotonic()
        warmup_time += end_t2 - end_t1
        end_t1 = time.monotonic()
        # query_batch
        results = await batch_query(query_list, request, warmup=False)
        query_ids:List[int] = []
        end_t2 = time.monotonic()
        query_time += end_t2 - end_t1

        for query in query_list:
            for ind, message in zip(query.query_ids, query.messages_list):
                query_ids.append(ind)

        # handle output
        for result, ind in zip(results, query_ids):
            response.outputs[ind] = result
            logger.info(f"gather query {ind}'s response")
            
        # delete_prefix
        for query in query_list:
            # logger.info(f"doing delete prefix after query batch: \n"
            #             f"prefix_tokens:{query.prefix_tokens}")
            engine.delete_prefix(query.prefix_tokens)
            prefix_groups.remove(query)
            logger.info(f"after free,free blocks number {engine.engine.scheduler.block_manager.get_num_free_gpu_blocks()}")
    
    # handle unable to warmup-batched query
    logger.info(f"handle unwarmuped batch")
    if prefix_groups:
        tasks = []
        query_ids = []
        for query in prefix_groups:
            
            end_t1 = time.monotonic()

            results = await batch_query([query], request, warmup=True)

            end_t2 = time.monotonic()
            warmup_time += end_t2 - end_t1

            var_dict = vars(request)
            var_dict.pop('message_and_submessage')
            request_temp = ChatCompletionRequest(**var_dict)
            for ind, message in zip(query.query_ids, query.messages_list):
                temp_request = deepcopy(request_temp)
                temp_request.messages = message
                temp_request.prefix_pos = query.prefix_pos
                task = asyncio.create_task(
                    create_chat_completion(temp_request, raw_request)
                    )
                tasks.append(task)
                query_ids.append(ind)
                
            end_t1 = time.monotonic()
            results = await asyncio.gather(*tasks)
            end_t2 = time.monotonic()
            query_time += end_t2 - end_t1
            
            # logger.info(f"last batch query_ids: {query_ids}")
            for ind, result in zip(query_ids, results):
                response.outputs[ind] = result
    
    end_t = time.monotonic()
    logger.info(f"total time: {end_t - start_t}"
                f"warmup time: {warmup_time}"
                f"query time: {query_time}")
    return response


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received chat completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    prompt = await get_gen_prompt(request)
    token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret
    
    
    prefix_trie = engine.engine.scheduler.prefix_trie
    prefix_pos = request.prefix_pos
    if prefix_pos == None:
        if request.sub_message is not None:
            sub_request = deepcopy(request)
            sub_request.messages = sub_request.sub_message
            substr = await get_gen_prompt(sub_request)
            substr_ids, error_check_ret = await check_length(request, 
                                                             prompt=substr
                                                             )
            if error_check_ret is not None:
                return error_check_ret
            for i in range(0, min(len(substr_ids), len(token_ids))):
                if substr_ids[i] != token_ids[i]:
                    prefix_pos = i
                    break
                if i == len(substr_ids) - 1:
                    prefix_pos = i + 1

            if prefix_pos == 0:
                logger.info(f"prefix is too short to fill one physics block."
                            "So will not be cached")
                prefix_pos = None
        else:
            prefix = prefix_trie.find_longest_prefix(token_ids)
            logger.info(f"try auto prefix detect, get:\n{prefix}")
            if prefix != None:
                prefix_pos = len(prefix.token_ids)
                logger.info(f"auto found a prefix of length {prefix_pos}")
    

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            max_tokens=request.max_tokens,
            best_of=request.best_of,
            top_k=request.top_k,
            ignore_eos=request.ignore_eos,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, prefix_pos, sampling_params, request_id,
                                       token_ids)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = ChatCompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = ChatCompletionResponseStreamChoice(
                index=i,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(id=request_id,
                                                 choices=[choice_data],
                                                 model=model_name)
            data = chunk.json(exclude_unset=True, ensure_ascii=False)
            yield f"data: {data}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = ChatCompletionResponseChoice(
            index=output.index,
            message=ChatMessage(role="assistant", content=output.text),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    """Completion API similar to OpenAI's API.

    See https://platform.openai.com/docs/api-reference/completions/create
    for the API specification. This API mimics the OpenAI Completion API.

    NOTE: Currently we do not support the following features:
        - echo (since the vLLM engine does not currently support
          getting the logprobs of prompt tokens)
        - suffix (the language models we currently support do not support
          suffix)
        - logit_bias (to be supported by vLLM engine)
    """
    logger.info(f"Received completion request: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.echo:
        # We do not support echo since the vLLM engine does not
        # currently support getting the logprobs of prompt tokens.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo is not currently supported")

    if request.suffix is not None:
        # The language models we currently support do not support suffix.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "suffix is not currently supported")

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"

    use_token_ids = False
    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, int):
            use_token_ids = True
            prompt = request.prompt
        elif isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            use_token_ids = not isinstance(first_element, str)
            prompt = request.prompt[0]
    else:
        prompt = request.prompt

    if use_token_ids:
        _, error_check_ret = await check_length(request, prompt_ids=prompt)
    else:
        token_ids, error_check_ret = await check_length(request, prompt=prompt)
    if error_check_ret is not None:
        return error_check_ret

    created_time = int(time.monotonic())
    try:
        spaces_between_special_tokens = request.spaces_between_special_tokens
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            stop_token_ids=request.stop_token_ids,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
            skip_special_tokens=request.skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    if use_token_ids:
        result_generator = engine.generate(None,
                                           sampling_params,
                                           request_id,
                                           prompt_token_ids=prompt)
    else:
        result_generator = engine.generate(prompt, sampling_params, request_id,
                                           token_ids)

    # Similar to the OpenAI API, when n != best_of, we do not stream the
    # results. In addition, we do not stream the results when use beam search.
    stream = (request.stream
              and (request.best_of is None or request.n == request.best_of)
              and not request.use_beam_search)

    def create_stream_response_json(
        index: int,
        text: str,
        logprobs: Optional[LogProbs] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i]:],
                        output.logprobs[previous_num_tokens[i]:],
                        len(previous_texts[i]))
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = (LogProbs()
                                if request.logprobs is not None else None)
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    # Streaming response
    if stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs)
        else:
            logprobs = None
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output.text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default=None, help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument("--served-model-name",
                        type=str,
                        default=None,
                        help="The model name used in the API. If not "
                        "specified, the model name will be the same as "
                        "the huggingface name.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())
    max_model_len = engine_model_config.max_model_len

    # A separate tokenizer to map token IDs to strings.
    tokenizer = get_tokenizer(engine_args.tokenizer,
                              tokenizer_mode=engine_args.tokenizer_mode,
                              trust_remote_code=engine_args.trust_remote_code)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
