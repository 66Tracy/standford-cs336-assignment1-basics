# CS336 Spring 2025 Assignment 1: Basics

### Personal Log
2025.12 ~ 2026.01 断断续续写了基本的BPE训练，测试ing

2025.01.18 写出了version-1，在TinyStoriesV2-GPT4-valid.txt上合并到词表1000耗时3.16，有很多可以优化的地方；先校验正确性。
测试使用Soted_list维护一个排序，速度反而更慢，3.64秒

2025.01.20 修改部分bug，重新提交，通过train_bpe测试corpus.en 通过

2025.01.21 修改部分bug，word.encoding拆分的时候，就是完全基于byte拆分；原来特殊多字节的词没有拆分导致失败。test_train_bpe_special_tokens通过