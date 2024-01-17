from sys import prefix
from typing import Dict, List, Optional, Union
from vllm.block import PhysicalTokenBlock

from vllm.logger import init_logger

logger = init_logger(__name__)
# Define the prefix class, which is a collection of prefix (a sequence of tokens).
# The class contains the following main methods:
# 1. A match method that checks if a prefix matches a given sequence of tokens.
# 2. A swapping method that can load or offload the prefix to or from GPU
# 3. An update_frequency method that updates the frequency of the prefix.
# 4. A get_status method that tells if the prefix is on GPU or not.


class Prefix:
    def __init__(self, prefix_id, token_ids, block_size):
        self.prefix_id = prefix_id
        self.token_ids = token_ids
        self.length = len(token_ids)
        print("prefix length: ", self.length)
        print("block size: ", block_size)
        print("prefix id: ", self.prefix_id)
        assert self.length % block_size == 0
        self.on_gpu = False
        self.on_cpu = False
        self.block_table = None
        # a lock to prevent multiple sequence from calculating the same prefix
        self.swap_to_gpu = False

        # freq-related
        self.freq = 1
        self.alpha = 0.8
        self.beta = 0.5
    
    def get_block_table_num(self) -> List[int]:
        return [block.block_number for block in self.block_table]
    
    def match(self, tokens):
        return tokens[:self.length] == self.token_ids
    
    # should be called if the prefix is hit for this iteration
    def update_freq(self, new_hit_rate):
        self.freq = self.alpha * self.freq + (1 - self.alpha) * new_hit_rate
        self.alpha = 0.8
    
    # should be called if the prefix is not hit for this iteration
    def punish_freq(self):
        self.alpha = self.beta * self.alpha if self.alpha > 0.1 else 0.1
   
    # whether the prefix is on GPU or not
    def get_status(self):
        return self.on_gpu
    
    def get_length(self):
        return self.length
    
    def __repr__(self) -> str:
        return (f"prefix_id: {self.prefix_id}\n"
               f"token_ids: {self.token_ids}\n"
               f"block_table: {self.block_table}\n")

# Define the prefix pool class, which is a collection of prefixes.
# The class contains the following main methods:
# 1. add a prefix to the pool, with a computed hash
# 2. TODO: create subprefix, if one is a prefix of the other: they can share some memory blocks
# 3. efficient_search: given a sequence of tokens, find the longest prefix in the pool that matches the sequence
# 4. fixed_search: given the prefix's hash, find the prefix in the pool
# 5. TODO: approximate_search: given a sequence of tokens, find the similar prefixes in the pool


class PrefixPool:
    def __init__(self, block_size):
        self.prefixes = []
        self.prefixes_hash = {}
        self.block_size = block_size
    
    def add_prefix(self, token_ids: List[int]):
        # generate prefix_id
        prefix_id = len(self.prefixes)
        # create a new prefix
        prefix = Prefix(prefix_id, token_ids, self.block_size)
        self.prefixes.append(prefix)
        # @TODO: compute the hash of the prefix
        prefix_hash = hash(tuple(prefix.token_ids))
        # self.prefixes_hash[prefix.prefix_id] = prefix_hash
        self.prefixes_hash[prefix_hash] = prefix.prefix_id
        return prefix
        
    # @TODO: this one should also come with a method to identify the prefix
    def efficient_search(self, token_ids: List[int]):
        # improve this search
        for prefix in self.prefixes:
            if prefix.match(token_ids):
                return prefix
        return None
    
    # use this first, if we already know from the application which part of the tokens are prefix.
    def fixed_search(self, prefix_hash):
        if prefix_hash not in self.prefixes_hash:
            return None
        # print("Found prefix in the pool.")
        prefix_id = self.prefixes_hash[prefix_hash]
        return self.prefixes[prefix_id]

    def delete_prefix(self, prefix_hash: int) -> Optional[int]:
        if prefix_hash not in self.prefixes_hash:
            return None
        
        prefix_id = self.prefixes_hash[prefix_hash]
        # physics block will be deleted in block_manager outside this function
        # del prefix
        self.prefixes_hash.pop(prefix_hash)
        for key, value in self.prefixes_hash.items():
            if value > prefix_id:
                self.prefixes_hash[key] -= 1

        del self.prefixes[prefix_id]
        
        return prefix_id

    
'''
This class is the node for trie
'''
class TrieNode:
    def __init__(self, value = None):
        self.value: Optional[Prefix] = value
        self.next_dict: Dict[int, TrieNode] = {}
        self.size = 0 if value is None else 1

'''
Using Trie to auto-detect available prefix for prompt
'''
class PrefixTrie:
    '''
    Manage prefix by Trie
    '''
    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = TrieNode()
        self.prefixes_list = []
        self.prefix_id = 0
    
    # add a new prefix
    def add_prefix(self, token_ids: List[int]):
        '''
        Input: tokens of prefix
        Output: None
        '''
        assert len(self.prefixes_list) == self.root.size, f"list length{len(self.prefixes_list)} != trie size{self.root.size}"
        
        truncated_len = len(token_ids) // self.block_size * self.block_size
        token_ids = token_ids[:truncated_len]
        if self.match(token_ids) != None:
            return
        cur = self.root
        cur.size += 1
        for token in token_ids:
            if token not in cur.next_dict:
                cur.next_dict[token] = TrieNode()
            
            cur = cur.next_dict[token]
            cur.size += 1
        
        cur.value = Prefix(self.prefix_id, token_ids, self.block_size)
        self.prefixes_list.append(cur.value)
        assert len(self.prefixes_list) == self.root.size, f"list length{len(self.prefixes_list)} != trie size{self.root.size}"
        self.prefix_id += 1

    def delete_prefix(self, token_ids: List[int]) -> Union[bool, int]:
        '''
        Input: tokens of prefix
        Output: None
        '''
        # not in prefix_trie
        if len(token_ids) == 0:
            return False
        truncated_len = len(token_ids) // self.block_size * self.block_size
        token_ids = token_ids[:truncated_len]
        deleted_prefix = self.match(token_ids)
        if deleted_prefix == None:
            logger.warning(f"no matched prefix to delete")
            return False

        cur = self.root
        path_node: List[TrieNode] = [cur]
        for token in token_ids:
            if token in cur.next_dict:
                cur = cur.next_dict[token]
            else:
                return False
            path_node.append(cur)
            
        # remove from prefixes_list
        # delete value
        
        self.prefixes_list.remove(path_node[-1].value)
        path_node[-1].value = None

        # remove related node and key-value in dict
        # blocks will be freed ouside here (in block_manager)
        # can not call block_manager in PrefixTrie
        father_id, son_id = len(path_node) - 2, len(path_node) - 1

        while father_id >= 0:
            path_node[son_id].size -= 1
            # delete path if path is empty
            if path_node[son_id].size == 0:
                path_node[father_id].next_dict.pop(token_ids[father_id])
                
            father_id -= 1
            son_id -= 1
        path_node[0].size -= 1

        assert self.match(token_ids) == None, f"not delete completly"
        return deleted_prefix.prefix_id

    # find the longest cached prefix of specific token_ids 
    def find_longest_prefix(self, token_ids: List[int]) -> Optional[Prefix]:
        '''
        Input: tokens of prompt
        Output: The prefix which has the longest prefix-tokens with given
                or None if not exist.
        '''
        cur = self.root
        res = None
        for token in token_ids:
            if token in cur.next_dict:
                cur = cur.next_dict[token]
                if cur.value != None:
                    res = cur.value
            else:
               break
        
        return res
    
    # finding prefix has exactly specific token_ids
    def match(self, token_ids: List[int]) -> Optional[Prefix]:
        '''
        Input: tokens of prefix
        Output: The prefix which exactly match given tokens
                or None if not exist
        '''
        assert len(self.prefixes_list) == self.root.size, f"list length{len(self.prefixes_list)} != trie size{self.root.size}"
        truncated_len = len(token_ids) // self.block_size * self.block_size
        token_ids = token_ids[:truncated_len]
        cur = self.root
        for token in token_ids:
            if token in cur.next_dict:
                cur = cur.next_dict[token]
            else:
                return None
        
        return cur.value
    
    # using same physice block as many as possible
    # update blocks from leaves to root
    # return blocks need to be freed
    def _update_block(self, prefix: Prefix) -> List[PhysicalTokenBlock]:
        '''
        Share blocks of a prefix
        Input: the prefix you want to share blocks
        Output: The List of old blocks. We need to free them latter.
        '''
        free_blocks: List[PhysicalTokenBlock] = []
        if not prefix.block_table:
            # didn't allocated block, no need to update
            return free_blocks
        
        cur = self.root
        for token in prefix.token_ids:
            assert token in cur.next_dict, f"prefix not found in prefix_trie.\n"\
                                           f"prefix: {prefix.token_ids}"
            cur = cur.next_dict[token]
            if cur.value != None:
                # free old block and replace by new block
                if cur.value.block_table != None:
                    free_blocks += cur.value.block_table
                block_len = len(cur.value.token_ids) // self.block_size
                assert len(cur.value.token_ids) % self.block_size == 0, f"len(cur.value.token_ids): {len(cur.value.token_ids)}"
                cur.value.block_table = prefix.block_table[:block_len]
                # update ref_count for each block
                for block in cur.value.block_table:
                    block.ref_count += 1
        return free_blocks

    def update_block(self) -> List[PhysicalTokenBlock]:
        '''
        Share blocks of prefixes on Trie.
        Input: None
        Output: The List of old blocks. We need to free them latter.
        '''
        free_blocks: List[PhysicalTokenBlock] = []
        sorted_prefixes_list = sorted(self.prefixes_list, key=lambda x:len(x.token_ids))
        for prefix in sorted_prefixes_list:
            free_blocks += self._update_block(prefix)
    
        return free_blocks

    def get_prefix_list(self) -> List[Prefix]:
        '''
        Get prefixes we now have, may not cached.
        '''
        return self.prefixes_list
        
               
        