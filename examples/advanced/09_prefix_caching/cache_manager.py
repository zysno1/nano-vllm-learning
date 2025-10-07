#!/usr/bin/env python3
"""
高级缓存管理器实现

本模块实现了生产级的 Prefix Cache 管理器，包括：
1. LRU 缓存淘汰策略
2. 内存使用监控和控制
3. 缓存压缩和优化
4. 并发安全访问
5. 详细的统计和监控
"""

import time
import threading
import hashlib
import pickle
import zlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import torch
import numpy as np
import psutil
import logging


class EvictionPolicy(Enum):
    """缓存淘汰策略"""
    LRU = "lru"          # 最近最少使用
    LFU = "lfu"          # 最少使用频率
    FIFO = "fifo"        # 先进先出
    RANDOM = "random"    # 随机淘汰


@dataclass
class CacheConfig:
    """缓存配置"""
    max_memory_mb: int = 1024                    # 最大内存使用 (MB)
    max_entries: int = 1000                      # 最大条目数
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression_enabled: bool = True             # 启用压缩
    compression_threshold: int = 1024            # 压缩阈值 (bytes)
    auto_cleanup: bool = True                    # 自动清理
    cleanup_interval: int = 300                  # 清理间隔 (秒)
    stats_enabled: bool = True                   # 启用统计
    thread_safe: bool = True                     # 线程安全


@dataclass
class DetailedCacheStats:
    """详细的缓存统计信息"""
    # 基础统计
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    compressions: int = 0
    decompressions: int = 0
    
    # 时间统计
    total_get_time: float = 0.0
    total_put_time: float = 0.0
    total_compression_time: float = 0.0
    total_decompression_time: float = 0.0
    
    # 内存统计
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    total_entries: int = 0
    compressed_entries: int = 0
    
    # 性能统计
    avg_get_time: float = field(init=False)
    avg_put_time: float = field(init=False)
    hit_rate: float = field(init=False)
    compression_ratio: float = field(init=False)
    memory_efficiency: float = field(init=False)
    
    def __post_init__(self):
        self._update_derived_stats()
    
    def _update_derived_stats(self):
        """更新派生统计信息"""
        total_requests = self.hits + self.misses
        
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        self.avg_get_time = self.total_get_time / total_requests if total_requests > 0 else 0.0
        self.avg_put_time = self.total_put_time / self.evictions if self.evictions > 0 else 0.0
        self.compression_ratio = self.compressed_entries / self.total_entries if self.total_entries > 0 else 0.0
        self.memory_efficiency = self.current_memory_bytes / (1024 * 1024)  # MB


@dataclass
class CacheEntry:
    """缓存条目"""
    data: Any                           # 缓存数据
    timestamp: float                    # 创建时间
    last_access: float                  # 最后访问时间
    access_count: int = 0              # 访问次数
    size_bytes: int = 0                # 数据大小
    is_compressed: bool = False        # 是否压缩
    original_size: int = 0             # 原始大小
    
    def update_access(self):
        """更新访问信息"""
        self.last_access = time.time()
        self.access_count += 1


class CompressedStorage:
    """压缩存储管理器"""
    
    def __init__(self, compression_threshold: int = 1024):
        self.compression_threshold = compression_threshold
    
    def should_compress(self, data: Any) -> bool:
        """判断是否应该压缩"""
        try:
            serialized = pickle.dumps(data)
            return len(serialized) > self.compression_threshold
        except:
            return False
    
    def compress_data(self, data: Any) -> Tuple[bytes, int, bool]:
        """
        压缩数据
        
        Returns:
            compressed_data: 压缩后的数据
            original_size: 原始大小
            is_compressed: 是否已压缩
        """
        try:
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            if original_size > self.compression_threshold:
                compressed = zlib.compress(serialized, level=6)
                return compressed, original_size, True
            else:
                return serialized, original_size, False
        except Exception as e:
            logging.warning(f"压缩失败: {e}")
            return pickle.dumps(data), len(pickle.dumps(data)), False
    
    def decompress_data(self, compressed_data: bytes, is_compressed: bool) -> Any:
        """解压缩数据"""
        try:
            if is_compressed:
                decompressed = zlib.decompress(compressed_data)
                return pickle.loads(decompressed)
            else:
                return pickle.loads(compressed_data)
        except Exception as e:
            logging.error(f"解压缩失败: {e}")
            raise


class AdvancedCacheManager:
    """高级缓存管理器"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = DetailedCacheStats()
        self.compressor = CompressedStorage(config.compression_threshold)
        
        # 线程安全
        self._lock = threading.RLock() if config.thread_safe else None
        
        # 自动清理
        self._cleanup_timer = None
        if config.auto_cleanup:
            self._start_cleanup_timer()
        
        # 内存监控
        self.max_memory_bytes = config.max_memory_mb * 1024 * 1024
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _with_lock(self, func: Callable):
        """线程安全装饰器"""
        def wrapper(*args, **kwargs):
            if self._lock:
                with self._lock:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    
    def _hash_key(self, key: Any) -> str:
        """生成缓存键的哈希值"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            key_bytes = pickle.dumps(key)
            return hashlib.md5(key_bytes).hexdigest()
    
    @_with_lock
    def get(self, key: Any) -> Optional[Any]:
        """获取缓存项"""
        start_time = time.time()
        
        try:
            key_hash = self._hash_key(key)
            
            if key_hash in self.cache:
                entry = self.cache[key_hash]
                entry.update_access()
                
                # 移动到末尾（LRU）
                if self.config.eviction_policy == EvictionPolicy.LRU:
                    self.cache.move_to_end(key_hash)
                
                # 解压缩数据
                if entry.is_compressed:
                    decomp_start = time.time()
                    data = self.compressor.decompress_data(entry.data, True)
                    self.stats.total_decompression_time += time.time() - decomp_start
                    self.stats.decompressions += 1
                else:
                    data = self.compressor.decompress_data(entry.data, False)
                
                self.stats.hits += 1
                return data
            else:
                self.stats.misses += 1
                return None
                
        finally:
            self.stats.total_get_time += time.time() - start_time
    
    @_with_lock
    def put(self, key: Any, value: Any) -> bool:
        """存储缓存项"""
        start_time = time.time()
        
        try:
            key_hash = self._hash_key(key)
            
            # 压缩数据
            comp_start = time.time()
            compressed_data, original_size, is_compressed = self.compressor.compress_data(value)
            if is_compressed:
                self.stats.total_compression_time += time.time() - comp_start
                self.stats.compressions += 1
            
            # 计算大小
            size_bytes = len(compressed_data)
            
            # 检查内存限制
            if not self._ensure_memory_available(size_bytes):
                return False
            
            # 创建缓存条目
            entry = CacheEntry(
                data=compressed_data,
                timestamp=time.time(),
                last_access=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                is_compressed=is_compressed,
                original_size=original_size
            )
            
            # 如果键已存在，更新统计
            if key_hash in self.cache:
                old_entry = self.cache[key_hash]
                self.stats.current_memory_bytes -= old_entry.size_bytes
                if old_entry.is_compressed:
                    self.stats.compressed_entries -= 1
                self.stats.total_entries -= 1
            
            # 存储条目
            self.cache[key_hash] = entry
            self.stats.current_memory_bytes += size_bytes
            self.stats.total_entries += 1
            if is_compressed:
                self.stats.compressed_entries += 1
            
            # 更新峰值内存
            if self.stats.current_memory_bytes > self.stats.peak_memory_bytes:
                self.stats.peak_memory_bytes = self.stats.current_memory_bytes
            
            return True
            
        finally:
            self.stats.total_put_time += time.time() - start_time
    
    def _ensure_memory_available(self, required_bytes: int) -> bool:
        """确保有足够的内存空间"""
        # 检查条目数限制
        while len(self.cache) >= self.config.max_entries:
            if not self._evict_one():
                return False
        
        # 检查内存限制
        while (self.stats.current_memory_bytes + required_bytes > self.max_memory_bytes and 
               len(self.cache) > 0):
            if not self._evict_one():
                return False
        
        return True
    
    def _evict_one(self) -> bool:
        """淘汰一个缓存条目"""
        if not self.cache:
            return False
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # 淘汰最近最少使用的
            key_to_evict = next(iter(self.cache))
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # 淘汰使用频率最低的
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.cache[k].access_count)
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # 淘汰最早的
            key_to_evict = min(self.cache.keys(), 
                             key=lambda k: self.cache[k].timestamp)
        else:  # RANDOM
            import random
            key_to_evict = random.choice(list(self.cache.keys()))
        
        # 执行淘汰
        entry = self.cache.pop(key_to_evict)
        self.stats.current_memory_bytes -= entry.size_bytes
        self.stats.total_entries -= 1
        if entry.is_compressed:
            self.stats.compressed_entries -= 1
        self.stats.evictions += 1
        
        return True
    
    def _start_cleanup_timer(self):
        """启动自动清理定时器"""
        def cleanup():
            self._cleanup_expired_entries()
            # 重新设置定时器
            self._cleanup_timer = threading.Timer(
                self.config.cleanup_interval, cleanup
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
        
        self._cleanup_timer = threading.Timer(
            self.config.cleanup_interval, cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    @_with_lock
    def _cleanup_expired_entries(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = []
        
        # 找出长时间未访问的条目
        for key, entry in self.cache.items():
            if current_time - entry.last_access > self.config.cleanup_interval * 2:
                expired_keys.append(key)
        
        # 删除过期条目
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.stats.current_memory_bytes -= entry.size_bytes
            self.stats.total_entries -= 1
            if entry.is_compressed:
                self.stats.compressed_entries -= 1
        
        if expired_keys:
            self.logger.info(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def get_stats(self) -> DetailedCacheStats:
        """获取详细统计信息"""
        self.stats._update_derived_stats()
        return self.stats
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'cache_memory_mb': self.stats.current_memory_bytes / (1024 * 1024),
            'cache_memory_bytes': self.stats.current_memory_bytes,
            'peak_cache_memory_mb': self.stats.peak_memory_bytes / (1024 * 1024),
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'memory_utilization': self.stats.current_memory_bytes / self.max_memory_bytes,
            'entries_count': len(self.cache),
            'max_entries': self.config.max_entries,
            'entries_utilization': len(self.cache) / self.config.max_entries
        }
    
    @_with_lock
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = DetailedCacheStats()
    
    def shutdown(self):
        """关闭缓存管理器"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        self.clear()


class PrefixCacheManager(AdvancedCacheManager):
    """专门用于前缀缓存的管理器"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self.prefix_trie = {}  # 前缀树用于快速匹配
    
    def find_longest_prefix_match(self, tokens: List[int]) -> Tuple[int, Optional[Any]]:
        """找到最长前缀匹配"""
        longest_match_len = 0
        longest_match_data = None
        
        # 尝试不同长度的前缀
        for i in range(1, len(tokens) + 1):
            prefix = tokens[:i]
            data = self.get(prefix)
            if data is not None:
                longest_match_len = i
                longest_match_data = data
        
        return longest_match_len, longest_match_data
    
    def store_prefix_kv(self, tokens: List[int], kv_cache: Tuple[torch.Tensor, torch.Tensor]):
        """存储前缀的 KV Cache"""
        return self.put(tokens, kv_cache)
    
    def get_prefix_kv(self, tokens: List[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取前缀的 KV Cache"""
        return self.get(tokens)


def demo_advanced_cache_manager():
    """演示高级缓存管理器"""
    print("🚀 高级缓存管理器演示")
    print("=" * 50)
    
    # 创建配置
    config = CacheConfig(
        max_memory_mb=100,
        max_entries=50,
        eviction_policy=EvictionPolicy.LRU,
        compression_enabled=True,
        compression_threshold=512,
        auto_cleanup=True,
        cleanup_interval=10,
        stats_enabled=True,
        thread_safe=True
    )
    
    # 创建缓存管理器
    cache_manager = PrefixCacheManager(config)
    
    print(f"📋 缓存配置:")
    print(f"  最大内存: {config.max_memory_mb} MB")
    print(f"  最大条目: {config.max_entries}")
    print(f"  淘汰策略: {config.eviction_policy.value}")
    print(f"  压缩启用: {config.compression_enabled}")
    
    # 模拟存储一些 KV Cache
    print(f"\n💾 存储测试数据...")
    
    for i in range(20):
        # 创建模拟的 tokens 和 KV cache
        tokens = list(range(i * 10, (i + 1) * 10))
        k_cache = torch.randn(1, 8, 100, 64)  # [batch, heads, seq, head_dim]
        v_cache = torch.randn(1, 8, 100, 64)
        kv_cache = (k_cache, v_cache)
        
        success = cache_manager.store_prefix_kv(tokens, kv_cache)
        if i % 5 == 0:
            print(f"  存储条目 {i}: {'成功' if success else '失败'}")
    
    # 测试检索
    print(f"\n🔍 检索测试...")
    
    hit_count = 0
    for i in range(0, 20, 2):
        tokens = list(range(i * 10, (i + 1) * 10))
        kv_cache = cache_manager.get_prefix_kv(tokens)
        if kv_cache is not None:
            hit_count += 1
            if i < 10:
                print(f"  检索条目 {i}: 命中")
    
    print(f"  总命中数: {hit_count}")
    
    # 显示统计信息
    stats = cache_manager.get_stats()
    print(f"\n📊 缓存统计:")
    print(f"  命中率: {stats.hit_rate:.2%}")
    print(f"  总条目: {stats.total_entries}")
    print(f"  压缩条目: {stats.compressed_entries}")
    print(f"  压缩比例: {stats.compression_ratio:.2%}")
    print(f"  当前内存: {stats.memory_efficiency:.2f} MB")
    print(f"  峰值内存: {stats.peak_memory_bytes / (1024*1024):.2f} MB")
    print(f"  淘汰次数: {stats.evictions}")
    
    # 显示内存使用情况
    memory_usage = cache_manager.get_memory_usage()
    print(f"\n🧠 内存使用:")
    print(f"  缓存内存: {memory_usage['cache_memory_mb']:.2f} MB")
    print(f"  进程内存: {memory_usage['process_memory_mb']:.2f} MB")
    print(f"  内存利用率: {memory_usage['memory_utilization']:.2%}")
    print(f"  条目利用率: {memory_usage['entries_utilization']:.2%}")
    
    # 测试前缀匹配
    print(f"\n🎯 前缀匹配测试...")
    
    test_tokens = list(range(0, 25))  # 跨越多个已存储的前缀
    match_len, match_data = cache_manager.find_longest_prefix_match(test_tokens)
    print(f"  测试序列长度: {len(test_tokens)}")
    print(f"  最长匹配长度: {match_len}")
    print(f"  匹配数据: {'找到' if match_data else '未找到'}")
    
    # 关闭缓存管理器
    cache_manager.shutdown()
    print(f"\n✅ 演示完成！")


if __name__ == "__main__":
    demo_advanced_cache_manager()