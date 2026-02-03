import heapq
from typing import Tuple, Optional, List

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
    



if __name__ == "__main__":

    heap = BPEVocab() 

    # 模拟 BPE 场景中的 n-gram pairs
    data = [
        ((b' a', b'nd'), 10),
        ((b'e', b'r '), 5),
        ((b' ', b'd'), 10),
    ]
    
    for pair, count in data:
        heap.push(pair, count)
    
    print("弹出顺序（应为先按 count 降序，count 相同按 pair 降序）：")
    while not heap.is_empty():
        pair, count = heap.pop_most_frequent()
        print(f"  {pair} -> count={count}")