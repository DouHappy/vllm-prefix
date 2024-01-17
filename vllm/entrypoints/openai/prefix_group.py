from typing import List, Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)
class PrefixGroup:
    prefix_tokens:List[int]
    prompts_tokens_list:List[List[int]]
    block_num:Optional[int] = None
    query_ids:List[int]
    prefix_pos:Optional[int] = None
    def __init__(self, sub_message, messages_list, block_size, prefix_tokens, prompts_tokens_list, query_ids):
        self.sub_message = sub_message
        self.messages_list = messages_list
        self.block_size = block_size
        self.prefix_tokens = prefix_tokens
        self.prompts_tokens_list = prompts_tokens_list
        self.query_ids = query_ids
        min_prompts_len = min([len(x) for x in prompts_tokens_list])
        max_equal_len = min(min_prompts_len, len(prefix_tokens))
        for i in range(0, max_equal_len):
            equal_flag = 1
            for j in range(0, len(prompts_tokens_list)):
                if prefix_tokens[i] != prompts_tokens_list[j][i]:
                    equal_flag = 0
                    break
            if equal_flag == 0:
                max_equal_len = max(0, i - 1)
                break
        self.prefix_pos = max_equal_len // block_size * block_size
        self.prefix_tokens = self.prefix_tokens[:self.prefix_pos]
        if self.prefix_pos <= 0:
            self.prefix_pos = None

    def dict(self)->Dict:
        raise NotImplementedError
        return {}
    
    def get_block_num(self)->int:
        if self.block_num != None:
            return self.block_num
        prefix_block_num = len(self.prefix_tokens) // self.block_size
        total_block_num = sum([(len(x) + self.block_size - 1) // self.block_size for x in self.prompts_tokens_list])
        self.block_num = total_block_num - prefix_block_num * (len(self.prompts_tokens_list) - 1)
        # debuge info
        logger.info(f"prefix_block_num: {prefix_block_num}\n"
                    f"total_block_num: {total_block_num}\n"
                    f"self.block_num: {self.block_num}\n")
        return self.block_num

    def can_alloc(self, free_block_num)->bool:
        # currently we only consider the block number limitation
        logger.info(f"need {self.get_block_num()} "
                    f"have {free_block_num} "
                    f"{'yes' if free_block_num - self.get_block_num() >= 15 else 'no'}")
        return free_block_num - self.get_block_num() >= 15
    
    def __repr__(self) -> str:
        res_list = []
        res_list.append(f'messages_list {self.messages_list}')
        res_list.append(f'sub_message {self.sub_message}')
        res_list.append(f'prefix tokens {self.prefix_tokens}')
        res_list.append(f'prompts_tokens_list {self.prompts_tokens_list}')
        res_list.append(f'block_size {self.block_size}')
        res_list.append(f'query_ids {self.query_ids}')
        res_list.append(f'prefix pos {self.prefix_pos}')
        return '\n'.join(res_list)
        

        
        
