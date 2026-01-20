from typing import List, ByteString
from functools import cmp_to_key

class BytePair:
    """字节对类型"""
    left: ByteString
    right: ByteString

    def __init__(self, left: ByteString, right: ByteString):
        self.left = left
        self.right = right
    
    def __getitem__(self, index: int) -> ByteString:
        """支持索引访问"""
        if index == 0:
            return self.left
        elif index == 1:
            return self.right
        elif isinstance(index, slice):
            # 支持切片
            return (self.left, self.right)[index]
        else:
            raise IndexError(f"Index {index} out of range for BytePair")
    
    def __len__(self) -> int:
        """支持 len() 函数"""
        return 2
    
    def __repr__(self):
        return f"BytePair({self.left!r}, {self.right!r})"
    
    def __str__(self):
        return f"({self.left},{self.right})"


def merge_bytepair_to_source_word(word_tokening:list):
    """从一个分词状态，反推出原单词"""
    source_word = bytes(word_tokening[0][0]).decode("utf-8") + bytes(word_tokening[0][1]).decode("utf-8")
    for i in range(1,len(word_tokening)):
        source_word += bytes(word_tokening[i][1]).decode("utf-8")
    
    return source_word


from sortedcontainers import SortedList
from collections import Counter

class BPEVocab:
    def __init__(self):
        self.sorted_pairs = SortedList(key=lambda x: (-x[1], x[0]))
        self.pair_counts = Counter()
        
    def push(self, pair, count):
        # 如果已存在，先删除
        if pair in self.pair_counts:
            old_count = self.pair_counts[pair]
            self.sorted_pairs.remove((pair, old_count))
        
        # 添加新的
        self.pair_counts[pair] = count
        self.sorted_pairs.add((pair, count))
        
    def get_most_frequent(self):
        if self.sorted_pairs:
            return self.sorted_pairs[0]
        return None

    def get_count_by_pair(self, pair):
        """输入pair获取count"""
        if pair in self.pair_counts:
            return self.pair_counts[pair]
        return 0

    def add_count(self, pair, count):
        """基于某个pair增加coun，可以是负数"""
        old_count = self.get_count_by_pair(pair)
        new_count = old_count + count if old_count + count > 0 else 0
        self.push(pair,new_count)


if __name__ == "__main__":
    # candidate_list = [("A".encode("utf-8"), "B".encode("utf-8")), ("B".encode("utf-8"), "C".encode("utf-8")), ("C".encode("utf-8"), "ZZ".encode("utf-8")), ("ZZ".encode("utf-8"), "A".encode("utf-8"))]
    # print(merge_bytepair_to_source_word(candidate_list))
    # “A”, “B”), (“A”, “C”), (“B”, “ZZ”),and (“BA”, “A”)

    # word = "widest"
    # word_encoding = word.encode("utf-8")
    # word_split = [word_encoding[i:i+1] for i in range(len(word_encoding))]

    # word_tokenizing = []
    # for i in range(len(word_split)-1):
    #     print(word_split[i], type(word_split[i]))
    #     byte_pair = BytePair(word_split[i], word_split[i+1])
    #     word_tokenizing.append(byte_pair)
    
    # print(word_tokenizing)

    # 按字符拆分而不是按字节拆分
    # word = "café"
    #     # 直接按字符拆分
    # word_split = [ch.encode("utf-8") for ch in word]
    # print(word_split)
        # 或者如果你需要字符本身而不是编码
        # word_split = list(word)  # ['c', 'a', 'f', 'é']

    # a = BPEVocab()
    # a.add_count(pair=(b' th', b'e'), count=5)
    # print(a.get_most_frequent())

    # a.add_count(pair=(b' th', b'e'), count=-5)
    # print(a.get_most_frequent())

    # a.add_count(pair=(b' th', b'e'), count=7)
    # print(a.get_most_frequent())

    # print(a)
    # sorted_pairs = SortedList(key=lambda x: (x[1], x[0]))
    # sorted_pairs.add(((b' a', b'nd'), 609))
    # print(sorted_pairs[-1]) # (b' a', b'nd')
    # sorted_pairs.add(((b' ', b'd'), 609))
    # print(sorted_pairs[-1]) # (b' ', b'd')
    # sorted_pairs.add(((b' ', b'cd'), 608))
    # print(sorted_pairs[-1]) # (b' ', b'd')
    # sorted_pairs.add(((b'a ', b'd'), 606))
    # print(sorted_pairs[-1]) # (b' ', b'd')
    # sorted_pairs.add(((b' d', b'd'), 64))
    # print(sorted_pairs[-1]) # (b' ', b'd')
    # sorted_pairs.add(((b'ss ', b'd'), 60))
    # print(sorted_pairs[-1]) # (b' ', b'd')

    print("<|endoftext|>".encode("utf-8"))