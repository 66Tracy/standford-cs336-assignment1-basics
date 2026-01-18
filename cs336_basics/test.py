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
    word = "café"
        # 直接按字符拆分
    word_split = [ch.encode("utf-8") for ch in word]
    print(word_split)
        # 或者如果你需要字符本身而不是编码
        # word_split = list(word)  # ['c', 'a', 'f', 'é']