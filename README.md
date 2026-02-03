# CS336 Spring 2025 Assignment 1: Basics

### Personal Log
**2025.12 ~ 2026.01** 断断续续写了基本的BPE训练，测试ing

**2025.01.18** 写出了version-1，在TinyStoriesV2-GPT4-valid.txt上合并到词表1000耗时3.16，有很多可以优化的地方；先校验正确性。
测试使用Soted_list维护一个排序，速度反而更慢，3.64秒

**2025.01.20** 修改部分bug，重新提交，通过train_bpe测试corpus.en 通过

**2025.01.21** 修改部分bug，word.encoding拆分的时候，就是完全基于byte拆分；原来特殊多字节的词没有拆分导致失败。test_train_bpe_special_tokens通过
Problem (train_bpe): BPE Tokenizer Training (15 points)
============================================= test session starts ==============================================
platform darwin -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0
rootdir: .../standford-cs336-assignment1-basics-master
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 3 items

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

**2025.01.25** 测试在TinyStoriesV2-GPT4-train.txt上不优化原版代码时，cProfile测试性能瓶颈在哪里
==================================================
性能分析结果：
==================================================
         99086060 function calls in 67.808 seconds

   Ordered by: internal time
   List reduced from 28 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   42.183   42.183   67.808   67.808 training_startups.py:230(train_bpe)
      743   13.885    0.019   23.259    0.031 training_startups.py:154(merge_and_update_bytepair)
 82828491    5.466    0.000    5.466    0.000 {built-in method builtins.len}
   955591    1.088    0.000    2.436    0.000 sortedlist.py:1778(add)
   932095    1.016    0.000    2.419    0.000 sortedlist.py:2001(remove)
   955591    0.653    0.000    5.507    0.000 training_startups.py:61(push)
   932095    0.547    0.000    0.612    0.000 sortedlist.py:2054(_delete)
  1908128    0.501    0.000    0.501    0.000 {built-in method _bisect.bisect_right}
  1864190    0.494    0.000    0.494    0.000 {built-in method _bisect.bisect_left}
  1905127    0.428    0.000    0.428    0.000 {method 'insert' of 'list' objects}
   954848    0.372    0.000    6.063    0.000 training_startups.py:82(add_count)
   197150    0.235    0.000    0.280    0.000 training_startups.py:145(merge_bytepair_to_source_word)
   955590    0.215    0.000    0.280    0.000 sortedlist.py:1822(_expand)
   954848    0.190    0.000    0.190    0.000 training_startups.py:76(get_count_by_pair)
   486996    0.180    0.000    0.180    0.000 {method 'pop' of 'list' objects}
  1887686    0.169    0.000    0.169    0.000 training_startups.py:58(<lambda>)
  1106842    0.140    0.000    0.140    0.000 {method 'append' of 'list' objects}
   197150    0.035    0.000    0.035    0.000 {method 'decode' of 'bytes' objects}
    59921    0.011    0.000    0.011    0.000 {method 'encode' of 'str' objects}
      743    0.001    0.000    0.001    0.000 training_startups.py:71(get_most_frequent)


**2025.01.29** 常使用堆优化SortedList，出于两个原因：
1）不想要使用外部类
2）不需要完全顺序化的list，我们只需要读取最大的那个，用堆来维护更合适。效果很显著merge_and_update_bytepair从23秒降到1秒以内
3）细化了train_bpe，以便查看哪个函数是瓶颈
优化策略：
重新实现了BytePairMaxHeap，用__slots__实现静态的数组，读取比较快

==================================================
性能分析结果：
==================================================
         5819346 function calls in 49.868 seconds

   Ordered by: internal time
   List reduced from 24 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1   44.493   44.493   44.493   44.493 training_startups.py:284(_init_words_counting)
   379651    4.007    0.000    4.402    0.000 training_startups.py:78(push)
        1    0.342    0.342    5.066    5.066 training_startups.py:294(_inital_states)
      743    0.214    0.000    0.305    0.000 training_startups.py:209(merge_and_update_bytepair)    
   379651    0.159    0.000    0.395    0.000 training_startups.py:47(push)
   378908    0.147    0.000    4.595    0.000 training_startups.py:89(add_count)
   379651    0.140    0.000    0.204    0.000 {built-in method _heapq.heappush}
  1770727    0.123    0.000    0.123    0.000 {built-in method builtins.len}
   812765    0.082    0.000    0.082    0.000 {method 'append' of 'list' objects}
   889769    0.065    0.000    0.065    0.000 training_startups.py:26(__lt__)
   378908    0.047    0.000    0.047    0.000 {method 'get' of 'dict' objects}
   379651    0.031    0.000    0.031    0.000 training_startups.py:21(__init__)
    59921    0.010    0.000    0.010    0.000 {method 'encode' of 'str' objects}
     3753    0.002    0.000    0.002    0.000 {method 'pop' of 'list' objects}
      743    0.002    0.000    0.003    0.000 {built-in method _heapq.heappop}
     1877    0.002    0.000    0.002    0.000 training_startups.py:200(merge_bytepair_to_source_word)
        1    0.001    0.001   49.868   49.868 training_startups.py:321(train_bpe)
     1877    0.000    0.000    0.000    0.000 {method 'decode' of 'bytes' objects}
      743    0.000    0.000    0.003    0.000 training_startups.py:53(pop_most_frequent)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}

测试耗时: 49.88