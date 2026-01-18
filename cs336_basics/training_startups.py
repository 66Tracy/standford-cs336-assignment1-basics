from typing import List, Tuple, Dict, ByteString
from pretokenization_example import find_chunk_boundaries
import regex as re
from collections import defaultdict
from functools import cmp_to_key
import json
import time

class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        """初始化必要参数"""
        self.file_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        ## Step-1: 预分词
        self.pre_tokenized_text_list = self.parallel_pre_tokenization()
        self.bytepair_counting_dict = defaultdict(int) # 维护一个统计表

        ## Step-2: 初始化词表
        self.BPE_TOKEN_TABLE = {}    
        for i in range(256):
            self.BPE_TOKEN_TABLE[i] = bytes([i])
        # print("#########【测试点位-1】：初始化词表\n", self.BPE_TOKEN_TABLE, "\n\n\n")

        ## Step-3: 迭代训练
        self.train_bpe()

    def split_with_special_tokens(self, chunk: str, special_tokens:List[str]):
        """根据special tokens将chunk划分成一篇篇独立的文章"""
        texts = re.split("|".join(special_tokens), chunk)
        return texts

    def pre_tokenization(self, text: str):
        """对一段text文本使用GPT2规则分词"""
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenization_res = re.findall(PAT, text)
        return pre_tokenization_res

    def parallel_pre_tokenization(self):
        """英文预分词，对每篇text并行"""
        ## Step-1: Pre-Tokenization
        pre_tokenized_text_list = []
        with open(self.file_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            print("boundaries: ", boundaries)

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.\
            # TODO: parallelization
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                texts = self.split_with_special_tokens(chunk, self.special_tokens)
                for text in texts:
                    pre_tokenized_text = self.pre_tokenization(text)
                    pre_tokenized_text_list.append(pre_tokenized_text)

        return pre_tokenized_text_list


    def merge_bytepair_to_source_word(self, word_tokenizing:list):
        """从一个分词状态，反推出原单词"""
        source_word = word_tokenizing[0][0].decode("utf-8") + word_tokenizing[0][1].decode("utf-8")
        for i in range(1,len(word_tokenizing)):
            source_word += word_tokenizing[i][1].decode("utf-8")
        
        return source_word


    def merge_and_update_bytepair(self, most_frequent_pair: Tuple[bytes, bytes]):
        """根据传入的bytepair"""
        # 先对自身做更新
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        # print(f"#########【测试点位-6】：新token：{new_token}\n\n\n")

        # 源词的序列列表
        source_word_index_list = self.bytepair_from_words_index[most_frequent_pair]
        # print(f"#########【测试点位-7】：源词的序列列表：{source_word_index_list}\n\n\n")
        for pos, index in enumerate(source_word_index_list):
            # 获取源词的分词状态
            word_tokenizing = self.words_tokenizing_states[index]
            # 反查i,j
            j = 0
            while j < len(word_tokenizing):
                if most_frequent_pair == word_tokenizing[j]:
                    break
                j += 1
            # print(f"#########【测试点位-8】：源词的分词状态：{word_tokenizing}：{index}\n\n\n")
            # 需要合并词元，反推出当前词的个数
            source_word = self.merge_bytepair_to_source_word(word_tokenizing)
            source_word_count = self.words_counting[source_word]
            # 更新左边
            if j >= 1:
                left_pair = word_tokenizing[j-1]
                self.bytepair_counting_dict[left_pair] -= source_word_count
                new_left_pair = (left_pair[0], new_token)
                self.bytepair_counting_dict[new_left_pair] += source_word_count
                # 反查列表，new_left_pair要新增，left_pair要删减
                self.bytepair_from_words_index[new_left_pair].append(index)
                # 更新word_tokenizing状态，left_pair -> new_left_pair
                word_tokenizing[j-1] = new_left_pair
                del_idx = 0
                while del_idx < len(self.bytepair_from_words_index[left_pair]):
                    if self.bytepair_from_words_index[left_pair][del_idx] == index:
                        break
                    del_idx += 1
                self.bytepair_from_words_index[left_pair].pop(del_idx)
                # print(f"#########【测试点位-9】：左边新pair和更新后的统计 {new_left_pair}\n{self.bytepair_counting_dict}\n\n\n")

            if j <= len(word_tokenizing)-2:
                right_pair = word_tokenizing[j+1]
                self.bytepair_counting_dict[right_pair] -= source_word_count
                new_right_pair = (new_token, right_pair[1])
                self.bytepair_counting_dict[new_right_pair] += source_word_count
                # 反查index的列表，新增new pair
                self.bytepair_from_words_index[new_right_pair].append(index)
                word_tokenizing[j+1] = new_right_pair
                # 考虑一种情况，如果right_pair == middle pair的时候，会出现索引到同一张表，所以要从右边一位查起
                # 删去旧的
                del_idx = 0
                if right_pair == most_frequent_pair:
                    print("触发midle和right pair一样的情况: ", word_tokenizing)
                    del_idx = pos+1
                while del_idx < len(self.bytepair_from_words_index[right_pair]):
                    if self.bytepair_from_words_index[right_pair][del_idx] == index:
                        break
                    del_idx += 1
                self.bytepair_from_words_index[right_pair].pop(del_idx)

            word_tokenizing.pop(j)
            self.words_tokenizing_states[index] = word_tokenizing

        # 删除被合并的词组的倒查索引列表
        del self.bytepair_from_words_index[most_frequent_pair]
        # 删除字节对
        del self.bytepair_counting_dict[most_frequent_pair]

        # 更新词表
        self.BPE_TOKEN_TABLE[len(self.BPE_TOKEN_TABLE)] = new_token
    

    def train_bpe(self):
        """训练bpe流程"""
        ## sub-step-1: 预统计，开始循环merge之前，获取第一版统计信息
        ## TODO: 这里是不是也可以并行
        self.words_counting = defaultdict(int)
        for pre_tokenized_text in self.pre_tokenized_text_list:
            for word in pre_tokenized_text:
                self.words_counting[word] += 1
        # print("#########【测试点位-2】：词统计\n", self.words_counting, "\n\n\n")
        
        # 2025.12.28 写完了初始化
        # 维护一个word列表，使得byte-pair可以反查出自哪个index
        self.bytepair_from_words_index = defaultdict(list)
        self.words_tokenizing_states = []
        for word, counts in self.words_counting.items():
            # 对每个word进行encoding -> 其实是基于已有的词表进行分词
            word_split = [ch.encode("utf-8") for ch in word]
            if len(word_split) <= 1: # 只有一个字母的没办法构成byte-pair
                continue
            word_tokenizing = []
            for i in range(len(word_split)-1):
                byte_pair = (word_split[i], word_split[i+1])
                word_tokenizing.append(byte_pair)
                self.bytepair_counting_dict[byte_pair] += counts
                self.bytepair_from_words_index[byte_pair].append(len(self.words_tokenizing_states))
            self.words_tokenizing_states.append(word_tokenizing)
        # print("#########【测试点位-3】：分词状态列表\n", self.words_tokenizing_states, "\n\n\n")
        # print("#########【测试点位-4】：bytepair倒查索引列表\n", self.bytepair_from_words_index, "\n\n\n")
        
        # 2025.12.29 基于已经初始化的内容
        # 反复merge
        TABLE_SIZE = len(self.BPE_TOKEN_TABLE)
        idx = 256
        iter_time = 0
        print(f"##### 初始化的bytepair统计表：{self.bytepair_counting_dict}\n\n\n")
        while TABLE_SIZE <= self.vocab_size:
            iter_time += 1
            # 找到统计次数最多的词表
            # 找出所有counts最多的byte pairs
            max_count = max(self.bytepair_counting_dict.values())
            most_frequent_pairs = [pair for pair, count in self.bytepair_counting_dict.items() if count == max_count]
            ## 根据字典序找出其中一个pair，max天然就是字典序找最大
            # 先比较第一个元素，在比较第二个
            most_frequent_pair = max(most_frequent_pairs)
            print(f"Iter-{iter_time} 当前最频繁pair: ", most_frequent_pair, max_count)
            # print(f"#########【测试点位-5】：最频繁的bytepair\nIter-{iter_time}, most frequent pairs: ({most_frequent_pair}), count: {max_count}\n\n")
            self.merge_and_update_bytepair(most_frequent_pair)
            print(f"##### Iter-{iter_time}：\n最频繁的pair是：{most_frequent_pair}\n\n\n")
            TABLE_SIZE += 1
        print("完成后的词表：\n", self.BPE_TOKEN_TABLE)

if __name__ == "__main__":
    start_time = time.time()
    vocab_size = 1000
    obj = BPETrainer(input_path="datasets/TinyStoriesV2-GPT4-valid.txt", vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    print(f"目标词表大小：{vocab_size}, 耗时{time.time() - start_time}")

    ## 测试样例
    # test_obj = BPETrainer(input_path="datasets/test-sentences.txt", vocab_size=260, special_tokens=["<|endoftext|>"])