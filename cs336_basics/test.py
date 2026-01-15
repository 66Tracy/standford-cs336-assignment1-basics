from typing import List
from functools import cmp_to_key

def merge_bytepair_to_source_word(word_tokening:list):
    """从一个分词状态，反推出原单词"""
    source_word = bytes(word_tokening[0][0]).decode("utf-8") + bytes(word_tokening[0][1]).decode("utf-8")
    for i in range(1,len(word_tokening)):
        source_word += bytes(word_tokening[i][1]).decode("utf-8")
    
    return source_word


if __name__ == "__main__":
    candidate_list = [("A".encode("utf-8"), "B".encode("utf-8")), ("B".encode("utf-8"), "C".encode("utf-8")), ("C".encode("utf-8"), "ZZ".encode("utf-8")), ("ZZ".encode("utf-8"), "A".encode("utf-8"))]
    print(merge_bytepair_to_source_word(candidate_list))
    # “A”, “B”), (“A”, “C”), (“B”, “ZZ”),and (“BA”, “A”)