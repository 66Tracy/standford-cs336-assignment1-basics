from typing import List, Tuple
import regex as re
from collections import defaultdict
import time
from sortedcontainers import SortedList
from collections import Counter
import os, json
from typing import BinaryIO
import heapq
from typing import Tuple, Optional, List
from collections import Counter
from itertools import chain

class _HeapItem:
    """
    堆元素包装器，实现以下排序逻辑：
    1. count降序
    2. pair字典序降序
    """
    __slots__ = ['neg_count', 'pair', 'count']

    def __init__(self, pair: Tuple[bytes, bytes], count: int):
        self.pair = pair
        self.count = count
        self.neg_count = -count
    
    def __lt__(self, other: '_HeapItem') -> bool:
        if self.neg_count != other.neg_count:
            return self.neg_count < other.neg_count
        
        return self.pair > other.pair

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _HeapItem):
            return NotImplemented
        return self.neg_count == other.neg_count and self.pari == other.pair

class BytePairMaxHeap:
    """
    大顶堆，维护(pair, count)的降序，count相同时按pair的字典序排序
    """

    __slots__ = ['_heap']

    def __init__(self):
        self._heap: List[_HeapItem] = []

    def push(self, pair: Tuple[bytes, bytes], count: int) -> None:
        """
        压入一个新的pair
        """
        heapq.heappush(self._heap, _HeapItem(pair, count))

    def pop_most_frequent(self) -> Optional[Tuple[Tuple[bytes, bytes], int]]:
        """
        弹出最频繁的pair
        返回：pair, count
        """
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        return item.pair, item.count      

    def __len__(self) -> int:
        return len(self._heap)
    
    def is_empty(self) -> bool:
        return len(self._heap) == 0

class BPEVocab(BytePairMaxHeap):
    def __init__(self):
        """
        继承__slots__类型的堆结构
        外层构造dict类型，提高内存效率
        """
        super().__init__()
        self.valid_counts = {}
    
    def push(self, pair, count):
        self.valid_counts[pair] = count
        super().push(pair, count)
    
    def pop_most_frequent_valid(self):
        while self._heap:
            pair, count = super().pop_most_frequent()
            if self.valid_counts.get(pair, 0) == count:
                return pair, count
        return None

    def add_count(self, pair, count):
        """基于某个pair增加count，可以是负数"""
        old_count = self.valid_counts.get(pair, 0)
        new_count = old_count + count if old_count + count > 0 else 0
        self.push(pair,new_count)



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class BPETrainer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: List[str]):
        """初始化必要参数"""
        self.file_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        
        ## Step-1: 预分词
        self.pre_tokenized_text_list = self.parallel_pre_tokenization()
        self.bytepair_counting_dict = BPEVocab()

        ## Step-2: 初始化词表
        self.vocab = {}
        self.merges = []
        # add special tokens to vacab firstly
        for special_token in special_tokens:
            self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        for i in range(256):
            self.vocab[len(self.vocab)] = bytes([i])

        ## Step-3: 迭代训练
        # self.train_bpe()

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
        word_encoding = word_tokenizing[0][0] + word_tokenizing[0][1]
        for i in range(1,len(word_tokenizing)):
            word_encoding += word_tokenizing[i][1]
        source_word = word_encoding.decode("utf-8")
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
                # self.bytepair_counting_dict[left_pair] -= source_word_count
                self.bytepair_counting_dict.add_count(left_pair, -source_word_count)
                new_left_pair = (left_pair[0], new_token)
                # self.bytepair_counting_dict[new_left_pair] += source_word_count
                self.bytepair_counting_dict.add_count(new_left_pair, source_word_count)
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
                # self.bytepair_counting_dict[right_pair] -= source_word_count
                self.bytepair_counting_dict.add_count(right_pair, -source_word_count)
                new_right_pair = (new_token, right_pair[1])
                # self.bytepair_counting_dict[new_right_pair] += source_word_count
                self.bytepair_counting_dict.add_count(new_right_pair, source_word_count)
                # 反查index的列表，新增new pair
                self.bytepair_from_words_index[new_right_pair].append(index)
                word_tokenizing[j+1] = new_right_pair
                # 考虑一种情况，如果right_pair == middle pair的时候，会出现索引到同一张表，所以要从右边一位查起
                # 删去旧的
                del_idx = 0
                if right_pair == most_frequent_pair:
                    # print("触发midle和right pair一样的情况: ", word_tokenizing)
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
        self.bytepair_counting_dict.push(most_frequent_pair, 0)

        # 更新词表
        self.vocab[len(self.vocab)] = new_token
    
    def _init_words_counting(self):
        """
        统计语料片段里每个词的统计次数
        """
        ## TODO: 这里是不是也可以并行
        self.words_counting = defaultdict(int)
        for pre_tokenized_text in self.pre_tokenized_text_list:
            for word in pre_tokenized_text:
                self.words_counting[word] += 1

    # def _init_words_counting(self):
    #     # 将嵌套列表展平后一次性计数，避免 Python 层级的显式循环
    #     self.words_counting = Counter(chain.from_iterable(self.pre_tokenized_text_list))

    def _inital_states(self):
        """
        构建每个word的分词状态记忆列表：self.words_tokenizing_states
        构建根据词索引反查bytepair所在word：self.bytepair_from_words_index
        """
        # 2025.12.28 写完了初始化
        # 维护一个word列表，使得byte-pair可以反查出自哪个index
        self.bytepair_from_words_index = defaultdict(list)
        self.words_tokenizing_states = []
        for word, counts in self.words_counting.items():
            # 对每个word进行encoding -> 其实是基于已有的词表进行分词
            # word_split = [ch.encode("utf-8") for ch in word] # 基于词做encoding - 失败
            word_encoding = word.encode("utf-8")
            word_split = [word_encoding[i:i+1] for i in range(len(word_encoding))]# 使用切片保持 bytes 类型
            
            if len(word_split) <= 1: # 只有一个字母的没办法构成byte-pair
                continue
            word_tokenizing = []
            for i in range(len(word_split)-1):
                byte_pair = (word_split[i], word_split[i+1])
                word_tokenizing.append(byte_pair)
                self.bytepair_counting_dict.add_count(byte_pair, counts)
                # self.bytepair_counting_dict[byte_pair] += counts
                self.bytepair_from_words_index[byte_pair].append(len(self.words_tokenizing_states))
            self.words_tokenizing_states.append(word_tokenizing)
    

    def train_bpe(self):
        """训练bpe流程"""
        ## sub-step-1: 预统计，开始循环merge之前，获取第一版统计信息
        self._init_words_counting()
        
        self._inital_states()
        
        # 2025.12.29 基于已经初始化的内容
        # 反复merge
        iter_time = 0
        while len(self.vocab) < self.vocab_size:
            iter_time += 1
            # 找到最frequent的bytepair
            most_frequent_pair, max_count = self.bytepair_counting_dict.pop_most_frequent()
            self.merges.append(most_frequent_pair)
            # 完成合并
            self.merge_and_update_bytepair(most_frequent_pair)


if __name__ == "__main__":

    start_time = time.time()
    vocab_size = 1000
    obj = BPETrainer(input_path="tests/fixtures/tinystories_sample_5M.txt", vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    print(f"目标词表大小：{vocab_size}, 耗时{time.time() - start_time}")