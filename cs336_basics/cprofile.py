import cProfile
import pstats
import contextlib
from training_startups import BPETrainer
import time

@contextlib.contextmanager
def profile_context(sort_by='cumulative', print_top=20):
    """上下文管理器，自动开始和结束分析"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield
    finally:
        profiler.disable()
        
        # 创建统计对象
        stats = pstats.Stats(profiler)
        
        # 清理和排序
        stats.strip_dirs().sort_stats(sort_by)
        
        # 打印结果
        print("\n" + "="*50)
        print("性能分析结果：")
        print("="*50)
        stats.print_stats(print_top)

if __name__ == "__main__":
    vocab_size = 1000
    obj = BPETrainer(input_path="data\TinyStoriesV2-GPT4-train.txt", vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    
    start_time = time.time()
    # 使用with语句，自动分析train_bpe方法
    with profile_context(sort_by='time', print_top=20):
        obj.train_bpe()  #
    print(f"测试耗时: {(time.time()-start_time):.2f}")