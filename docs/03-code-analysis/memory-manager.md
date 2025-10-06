# 内存管理器分析

## 🎯 内存管理器概览

内存管理器是 nano-vllm 中负责高效内存分配、回收和优化的核心组件。它管理模型权重、KV缓存、激活值等各种内存资源，确保系统在有限内存下实现最大吞吐量。

## 🏗️ 核心架构

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
import gc
import psutil
from collections import defaultdict, deque
import numpy as np

from nano_vllm.config import MemoryConfig, CacheConfig
from nano_vllm.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryType(Enum):
    """内存类型枚举"""
    MODEL_WEIGHTS = "model_weights"
    KV_CACHE = "kv_cache"
    ACTIVATIONS = "activations"
    WORKSPACE = "workspace"
    SYSTEM = "system"

class AllocationStrategy(Enum):
    """内存分配策略"""
    EAGER = "eager"          # 立即分配
    LAZY = "lazy"            # 延迟分配
    POOLED = "pooled"        # 池化分配
    PAGED = "paged"          # 分页分配

@dataclass
class MemoryBlock:
    """内存块"""
    ptr: int                    # 内存指针
    size: int                   # 块大小
    memory_type: MemoryType     # 内存类型
    device: torch.device        # 设备
    allocated: bool = False     # 是否已分配
    ref_count: int = 0         # 引用计数
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.id = f"{self.memory_type.value}_{self.ptr}_{self.size}"

@dataclass
class MemoryStats:
    """内存统计信息"""
    total_allocated: int = 0
    total_reserved: int = 0
    peak_allocated: int = 0
    peak_reserved: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    cache_hit_rate: float = 0.0
    fragmentation_rate: float = 0.0
    
    def update_peak(self):
        """更新峰值"""
        self.peak_allocated = max(self.peak_allocated, self.total_allocated)
        self.peak_reserved = max(self.peak_reserved, self.total_reserved)

class BaseMemoryManager:
    """基础内存管理器"""
    
    def __init__(
        self,
        device: torch.device,
        memory_config: MemoryConfig,
    ):
        self.device = device
        self.config = memory_config
        
        # 内存统计
        self.stats = MemoryStats()
        
        # 内存块管理
        self.allocated_blocks: Dict[int, MemoryBlock] = {}
        self.free_blocks: Dict[int, List[MemoryBlock]] = defaultdict(list)
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 内存监控
        self.enable_monitoring = memory_config.enable_monitoring
        self.monitoring_thread = None
        
        logger.info(f"Initialized BaseMemoryManager for device: {device}")
    
    def allocate(
        self, 
        size: int, 
        memory_type: MemoryType = MemoryType.WORKSPACE,
        alignment: int = 256
    ) -> Optional[MemoryBlock]:
        """分配内存"""
        with self.lock:
            # 对齐大小
            aligned_size = self._align_size(size, alignment)
            
            # 尝试从空闲块中分配
            block = self._find_free_block(aligned_size, memory_type)
            
            if block is None:
                # 分配新块
                block = self._allocate_new_block(aligned_size, memory_type)
            
            if block:
                block.allocated = True
                block.ref_count = 1
                block.last_accessed = time.time()
                
                self.allocated_blocks[block.ptr] = block
                self.stats.total_allocated += block.size
                self.stats.allocation_count += 1
                self.stats.update_peak()
                
                logger.debug(f"Allocated {aligned_size} bytes for {memory_type.value}")
            
            return block
    
    def deallocate(self, block: MemoryBlock):
        """释放内存"""
        with self.lock:
            if block.ptr not in self.allocated_blocks:
                logger.warning(f"Attempting to deallocate untracked block: {block.id}")
                return
            
            block.ref_count -= 1
            
            if block.ref_count <= 0:
                # 移除分配记录
                del self.allocated_blocks[block.ptr]
                
                # 添加到空闲列表
                self.free_blocks[block.size].append(block)
                
                # 更新统计
                self.stats.total_allocated -= block.size
                self.stats.deallocation_count += 1
                
                block.allocated = False
                
                logger.debug(f"Deallocated {block.size} bytes for {block.memory_type.value}")
                
                # 尝试合并相邻块
                self._try_merge_blocks()
    
    def _align_size(self, size: int, alignment: int) -> int:
        """对齐内存大小"""
        return ((size + alignment - 1) // alignment) * alignment
    
    def _find_free_block(
        self, 
        size: int, 
        memory_type: MemoryType
    ) -> Optional[MemoryBlock]:
        """查找合适的空闲块"""
        
        # 查找完全匹配的块
        if size in self.free_blocks and self.free_blocks[size]:
            block = self.free_blocks[size].pop()
            if block.memory_type == memory_type:
                return block
            else:
                # 类型不匹配，放回
                self.free_blocks[size].append(block)
        
        # 查找更大的块
        for block_size in sorted(self.free_blocks.keys()):
            if block_size >= size and self.free_blocks[block_size]:
                blocks = self.free_blocks[block_size]
                for i, block in enumerate(blocks):
                    if block.memory_type == memory_type:
                        return blocks.pop(i)
        
        return None
    
    def _allocate_new_block(
        self, 
        size: int, 
        memory_type: MemoryType
    ) -> Optional[MemoryBlock]:
        """分配新的内存块"""
        try:
            # 检查内存限制
            if not self._check_memory_limit(size):
                logger.warning(f"Memory limit exceeded, cannot allocate {size} bytes")
                return None
            
            # 分配PyTorch张量
            tensor = torch.empty(size // 4, dtype=torch.float32, device=self.device)
            
            # 创建内存块
            block = MemoryBlock(
                ptr=tensor.data_ptr(),
                size=size,
                memory_type=memory_type,
                device=self.device,
            )
            
            return block
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            self._handle_oom()
            return None
        except Exception as e:
            logger.error(f"Failed to allocate memory: {e}")
            return None
    
    def _check_memory_limit(self, size: int) -> bool:
        """检查内存限制"""
        if self.device.type == "cuda":
            available = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            
            # 保留一定的安全边距
            safety_margin = self.config.memory_safety_margin
            max_usable = available * (1.0 - safety_margin)
            
            return allocated + size <= max_usable
        
        return True
    
    def _handle_oom(self):
        """处理内存不足"""
        logger.warning("Handling out of memory situation")
        
        # 强制垃圾回收
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # 清理最久未使用的块
        self._cleanup_lru_blocks()
    
    def _cleanup_lru_blocks(self, target_ratio: float = 0.2):
        """清理最久未使用的块"""
        with self.lock:
            # 按最后访问时间排序
            blocks = sorted(
                self.allocated_blocks.values(),
                key=lambda b: b.last_accessed
            )
            
            # 清理目标比例的块
            cleanup_count = int(len(blocks) * target_ratio)
            
            for block in blocks[:cleanup_count]:
                if block.ref_count == 0:
                    self.deallocate(block)
    
    def _try_merge_blocks(self):
        """尝试合并相邻的空闲块"""
        # 简化实现：只合并相同大小的块
        for size, blocks in self.free_blocks.items():
            if len(blocks) >= 2:
                # 按地址排序
                blocks.sort(key=lambda b: b.ptr)
                
                merged = []
                i = 0
                while i < len(blocks):
                    current = blocks[i]
                    
                    # 查找相邻块
                    j = i + 1
                    while j < len(blocks) and blocks[j].ptr == current.ptr + current.size:
                        current.size += blocks[j].size
                        j += 1
                    
                    merged.append(current)
                    i = j
                
                self.free_blocks[size] = merged
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        with self.lock:
            device_info = {}
            
            if self.device.type == "cuda":
                device_info.update({
                    "device_total": torch.cuda.get_device_properties(self.device).total_memory,
                    "device_allocated": torch.cuda.memory_allocated(self.device),
                    "device_reserved": torch.cuda.memory_reserved(self.device),
                })
            
            return {
                "stats": self.stats,
                "device_info": device_info,
                "allocated_blocks": len(self.allocated_blocks),
                "free_blocks": sum(len(blocks) for blocks in self.free_blocks.values()),
                "fragmentation_rate": self._calculate_fragmentation_rate(),
            }
    
    def _calculate_fragmentation_rate(self) -> float:
        """计算内存碎片率"""
        if not self.free_blocks:
            return 0.0
        
        total_free = sum(
            sum(block.size for block in blocks)
            for blocks in self.free_blocks.values()
        )
        
        if total_free == 0:
            return 0.0
        
        # 最大连续空闲块
        max_free_block = max(
            max(block.size for block in blocks) if blocks else 0
            for blocks in self.free_blocks.values()
        )
        
        return 1.0 - (max_free_block / total_free)

class KVCacheManager(BaseMemoryManager):
    """KV缓存管理器"""
    
    def __init__(
        self,
        device: torch.device,
        memory_config: MemoryConfig,
        cache_config: CacheConfig,
    ):
        super().__init__(device, memory_config)
        
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.max_num_blocks = cache_config.max_num_blocks
        
        # KV缓存块
        self.key_cache_blocks: List[torch.Tensor] = []
        self.value_cache_blocks: List[torch.Tensor] = []
        
        # 块分配状态
        self.block_allocator = BlockAllocator(self.max_num_blocks)
        
        # 预分配缓存块
        self._preallocate_cache_blocks()
        
        logger.info(f"Initialized KVCacheManager with {self.max_num_blocks} blocks")
    
    def _preallocate_cache_blocks(self):
        """预分配KV缓存块"""
        
        num_heads = self.cache_config.num_heads
        head_dim = self.cache_config.head_dim
        num_layers = self.cache_config.num_layers
        
        # 计算每个块的形状
        block_shape = (num_layers, num_heads, self.block_size, head_dim)
        
        try:
            for i in range(self.max_num_blocks):
                # 分配key缓存块
                key_block = torch.empty(
                    block_shape,
                    dtype=self.cache_config.cache_dtype,
                    device=self.device,
                )
                self.key_cache_blocks.append(key_block)
                
                # 分配value缓存块
                value_block = torch.empty(
                    block_shape,
                    dtype=self.cache_config.cache_dtype,
                    device=self.device,
                )
                self.value_cache_blocks.append(value_block)
                
                logger.debug(f"Preallocated cache block {i}")
                
        except torch.cuda.OutOfMemoryError:
            logger.error("Failed to preallocate all cache blocks due to OOM")
            # 调整块数量
            self.max_num_blocks = len(self.key_cache_blocks)
            self.block_allocator = BlockAllocator(self.max_num_blocks)
    
    def allocate_sequence_blocks(self, seq_len: int) -> List[int]:
        """为序列分配缓存块"""
        
        # 计算需要的块数
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # 分配块
        block_ids = []
        for _ in range(num_blocks):
            block_id = self.block_allocator.allocate()
            if block_id is None:
                # 回收已分配的块
                for bid in block_ids:
                    self.block_allocator.free(bid)
                raise RuntimeError("No available cache blocks")
            
            block_ids.append(block_id)
        
        logger.debug(f"Allocated {num_blocks} blocks for sequence length {seq_len}")
        return block_ids
    
    def free_sequence_blocks(self, block_ids: List[int]):
        """释放序列的缓存块"""
        for block_id in block_ids:
            self.block_allocator.free(block_id)
        
        logger.debug(f"Freed {len(block_ids)} cache blocks")
    
    def get_cache_blocks(
        self, 
        block_ids: List[int]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """获取缓存块"""
        
        key_blocks = [self.key_cache_blocks[bid] for bid in block_ids]
        value_blocks = [self.value_cache_blocks[bid] for bid in block_ids]
        
        return key_blocks, value_blocks
    
    def update_cache_block(
        self,
        block_id: int,
        layer_idx: int,
        start_pos: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """更新缓存块"""
        
        end_pos = start_pos + key.size(1)
        
        # 更新key缓存
        self.key_cache_blocks[block_id][layer_idx, :, start_pos:end_pos] = key
        
        # 更新value缓存
        self.value_cache_blocks[block_id][layer_idx, :, start_pos:end_pos] = value
    
    def get_cache_utilization(self) -> float:
        """获取缓存利用率"""
        allocated_blocks = self.block_allocator.get_allocated_count()
        return allocated_blocks / self.max_num_blocks

class BlockAllocator:
    """块分配器"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        
        # 使用位图管理块状态
        self.allocated_blocks = set()
        self.free_blocks = deque(range(num_blocks))
        
        self.lock = threading.Lock()
    
    def allocate(self) -> Optional[int]:
        """分配一个块"""
        with self.lock:
            if not self.free_blocks:
                return None
            
            block_id = self.free_blocks.popleft()
            self.allocated_blocks.add(block_id)
            
            return block_id
    
    def free(self, block_id: int):
        """释放一个块"""
        with self.lock:
            if block_id in self.allocated_blocks:
                self.allocated_blocks.remove(block_id)
                self.free_blocks.append(block_id)
    
    def get_allocated_count(self) -> int:
        """获取已分配块数"""
        return len(self.allocated_blocks)
    
    def get_free_count(self) -> int:
        """获取空闲块数"""
        return len(self.free_blocks)

class MemoryPool:
    """内存池"""
    
    def __init__(
        self,
        device: torch.device,
        pool_size: int,
        block_sizes: List[int],
    ):
        self.device = device
        self.pool_size = pool_size
        self.block_sizes = sorted(block_sizes)
        
        # 为每种大小创建池
        self.pools: Dict[int, deque] = {size: deque() for size in block_sizes}
        
        # 预分配内存
        self._preallocate_pools()
        
        self.lock = threading.Lock()
    
    def _preallocate_pools(self):
        """预分配内存池"""
        
        blocks_per_size = self.pool_size // len(self.block_sizes)
        
        for size in self.block_sizes:
            pool = self.pools[size]
            
            for _ in range(blocks_per_size):
                try:
                    tensor = torch.empty(
                        size // 4,  # 假设float32
                        dtype=torch.float32,
                        device=self.device,
                    )
                    pool.append(tensor)
                    
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"Failed to preallocate pool for size {size}")
                    break
    
    def get_tensor(self, size: int) -> Optional[torch.Tensor]:
        """从池中获取张量"""
        with self.lock:
            # 找到最小的合适大小
            suitable_size = None
            for pool_size in self.block_sizes:
                if pool_size >= size:
                    suitable_size = pool_size
                    break
            
            if suitable_size is None:
                return None
            
            pool = self.pools[suitable_size]
            if pool:
                return pool.popleft()
            
            return None
    
    def return_tensor(self, tensor: torch.Tensor):
        """将张量返回池中"""
        with self.lock:
            size = tensor.numel() * tensor.element_size()
            
            if size in self.pools:
                # 清零张量（可选）
                tensor.zero_()
                self.pools[size].append(tensor)

class UnifiedMemoryManager:
    """统一内存管理器"""
    
    def __init__(
        self,
        device: torch.device,
        memory_config: MemoryConfig,
        cache_config: CacheConfig,
    ):
        self.device = device
        self.memory_config = memory_config
        self.cache_config = cache_config
        
        # 子管理器
        self.base_manager = BaseMemoryManager(device, memory_config)
        self.kv_cache_manager = KVCacheManager(device, memory_config, cache_config)
        
        # 内存池
        self.memory_pool = MemoryPool(
            device=device,
            pool_size=memory_config.pool_size,
            block_sizes=memory_config.pool_block_sizes,
        )
        
        # 全局统计
        self.global_stats = MemoryStats()
        
        # 监控线程
        self.monitoring_enabled = memory_config.enable_monitoring
        self.monitoring_thread = None
        
        if self.monitoring_enabled:
            self._start_monitoring()
        
        logger.info("Initialized UnifiedMemoryManager")
    
    def allocate_model_weights(self, size: int) -> Optional[torch.Tensor]:
        """分配模型权重内存"""
        
        # 尝试从内存池获取
        tensor = self.memory_pool.get_tensor(size)
        if tensor is not None:
            return tensor
        
        # 从基础管理器分配
        block = self.base_manager.allocate(size, MemoryType.MODEL_WEIGHTS)
        if block is None:
            return None
        
        # 创建张量视图
        tensor = torch.empty(
            size // 4,  # 假设float32
            dtype=torch.float32,
            device=self.device,
        )
        
        return tensor
    
    def allocate_kv_cache(self, seq_len: int) -> List[int]:
        """分配KV缓存"""
        return self.kv_cache_manager.allocate_sequence_blocks(seq_len)
    
    def free_kv_cache(self, block_ids: List[int]):
        """释放KV缓存"""
        self.kv_cache_manager.free_sequence_blocks(block_ids)
    
    def allocate_workspace(self, size: int) -> Optional[torch.Tensor]:
        """分配工作空间"""
        
        # 尝试从内存池获取
        tensor = self.memory_pool.get_tensor(size)
        if tensor is not None:
            return tensor
        
        # 直接分配
        try:
            tensor = torch.empty(
                size // 4,
                dtype=torch.float32,
                device=self.device,
            )
            return tensor
            
        except torch.cuda.OutOfMemoryError:
            logger.error("Failed to allocate workspace due to OOM")
            return None
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        
        base_info = self.base_manager.get_memory_info()
        kv_cache_info = {
            "cache_utilization": self.kv_cache_manager.get_cache_utilization(),
            "allocated_blocks": self.kv_cache_manager.block_allocator.get_allocated_count(),
            "free_blocks": self.kv_cache_manager.block_allocator.get_free_count(),
        }
        
        system_info = {}
        if self.device.type == "cuda":
            system_info.update({
                "cuda_memory_allocated": torch.cuda.memory_allocated(self.device),
                "cuda_memory_reserved": torch.cuda.memory_reserved(self.device),
                "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(self.device),
            })
        
        # 系统内存信息
        system_memory = psutil.virtual_memory()
        system_info.update({
            "system_memory_total": system_memory.total,
            "system_memory_available": system_memory.available,
            "system_memory_percent": system_memory.percent,
        })
        
        return {
            "base_manager": base_info,
            "kv_cache_manager": kv_cache_info,
            "system_info": system_info,
            "global_stats": self.global_stats,
        }
    
    def _start_monitoring(self):
        """启动内存监控"""
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    # 收集内存信息
                    summary = self.get_memory_summary()
                    
                    # 检查内存使用情况
                    self._check_memory_health(summary)
                    
                    # 等待下次检查
                    time.sleep(self.memory_config.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Started memory monitoring thread")
    
    def _check_memory_health(self, summary: Dict[str, Any]):
        """检查内存健康状况"""
        
        # 检查CUDA内存使用率
        if self.device.type == "cuda":
            device_props = torch.cuda.get_device_properties(self.device)
            total_memory = device_props.total_memory
            allocated_memory = summary["system_info"]["cuda_memory_allocated"]
            
            usage_ratio = allocated_memory / total_memory
            
            if usage_ratio > self.memory_config.high_memory_threshold:
                logger.warning(f"High CUDA memory usage: {usage_ratio:.2%}")
                
                # 触发内存清理
                self._trigger_memory_cleanup()
        
        # 检查缓存利用率
        cache_utilization = summary["kv_cache_manager"]["cache_utilization"]
        if cache_utilization > 0.9:
            logger.warning(f"High KV cache utilization: {cache_utilization:.2%}")
    
    def _trigger_memory_cleanup(self):
        """触发内存清理"""
        logger.info("Triggering memory cleanup")
        
        # 清理基础管理器
        self.base_manager._cleanup_lru_blocks()
        
        # 强制垃圾回收
        gc.collect()
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def shutdown(self):
        """关闭内存管理器"""
        self.monitoring_enabled = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("UnifiedMemoryManager shutdown complete")

# 内存管理工具函数
def get_memory_usage(device: torch.device) -> Dict[str, int]:
    """获取设备内存使用情况"""
    
    if device.type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "max_allocated": torch.cuda.max_memory_allocated(device),
            "max_reserved": torch.cuda.max_memory_reserved(device),
        }
    else:
        # CPU内存使用psutil
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
        }

def format_bytes(bytes_value: int) -> str:
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

# 使用示例
def example_memory_manager_usage():
    """内存管理器使用示例"""
    
    # 配置
    memory_config = MemoryConfig(
        pool_size=1024 * 1024 * 1024,  # 1GB
        pool_block_sizes=[1024, 4096, 16384, 65536],
        memory_safety_margin=0.1,
        enable_monitoring=True,
        monitoring_interval=5.0,
        high_memory_threshold=0.8,
    )
    
    cache_config = CacheConfig(
        block_size=16,
        max_num_blocks=1000,
        num_heads=32,
        head_dim=128,
        num_layers=32,
        cache_dtype=torch.float16,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建内存管理器
    memory_manager = UnifiedMemoryManager(
        device=device,
        memory_config=memory_config,
        cache_config=cache_config,
    )
    
    try:
        # 分配模型权重
        model_weights = memory_manager.allocate_model_weights(1024 * 1024)  # 1MB
        print(f"Allocated model weights: {model_weights.shape if model_weights else None}")
        
        # 分配KV缓存
        kv_blocks = memory_manager.allocate_kv_cache(seq_len=512)
        print(f"Allocated KV cache blocks: {kv_blocks}")
        
        # 分配工作空间
        workspace = memory_manager.allocate_workspace(512 * 1024)  # 512KB
        print(f"Allocated workspace: {workspace.shape if workspace else None}")
        
        # 获取内存摘要
        summary = memory_manager.get_memory_summary()
        print("\nMemory Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # 释放KV缓存
        memory_manager.free_kv_cache(kv_blocks)
        print("\nFreed KV cache blocks")
        
    finally:
        # 关闭内存管理器
        memory_manager.shutdown()

if __name__ == "__main__":
    example_memory_manager_usage()
```

## 🔧 关键特性分析

### 1. 多层次内存管理

- **基础内存管理**：通用内存分配和回收
- **KV缓存管理**：专门的缓存块管理
- **内存池**：预分配的内存复用
- **统一管理**：协调各种内存资源

### 2. 智能分配策略

- **对齐分配**：确保内存对齐以提高性能
- **块合并**：减少内存碎片
- **LRU清理**：最久未使用块的自动清理
- **预分配**：减少运行时分配开销

### 3. 内存监控和优化

- **实时监控**：持续跟踪内存使用情况
- **健康检查**：自动检测内存压力
- **自动清理**：智能触发内存回收
- **统计分析**：详细的内存使用统计

## 📊 性能优化技术

### 内存预分配优化

```python
class OptimizedMemoryPreallocator:
    """优化的内存预分配器"""
    
    def __init__(self, device: torch.device, config: MemoryConfig):
        self.device = device
        self.config = config
        
        # 预分配不同大小的内存块
        self.preallocated_blocks = {}
        self._preallocate_common_sizes()
    
    def _preallocate_common_sizes(self):
        """预分配常用大小的内存块"""
        
        # 常用的内存大小（字节）
        common_sizes = [
            1024,      # 1KB
            4096,      # 4KB
            16384,     # 16KB
            65536,     # 64KB
            262144,    # 256KB
            1048576,   # 1MB
            4194304,   # 4MB
        ]
        
        for size in common_sizes:
            blocks = []
            
            # 为每种大小预分配多个块
            for _ in range(self.config.blocks_per_size):
                try:
                    tensor = torch.empty(
                        size // 4,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    blocks.append(tensor)
                    
                except torch.cuda.OutOfMemoryError:
                    break
            
            self.preallocated_blocks[size] = deque(blocks)
            logger.info(f"Preallocated {len(blocks)} blocks of size {size}")
    
    def get_block(self, size: int) -> Optional[torch.Tensor]:
        """获取预分配的块"""
        
        # 找到最小的合适大小
        for block_size in sorted(self.preallocated_blocks.keys()):
            if block_size >= size:
                blocks = self.preallocated_blocks[block_size]
                if blocks:
                    return blocks.popleft()
        
        return None
    
    def return_block(self, tensor: torch.Tensor):
        """返回块到预分配池"""
        
        size = tensor.numel() * tensor.element_size()
        
        if size in self.preallocated_blocks:
            # 清零并返回
            tensor.zero_()
            self.preallocated_blocks[size].append(tensor)
```

### 内存碎片整理

```python
class MemoryDefragmenter:
    """内存碎片整理器"""
    
    def __init__(self, memory_manager: BaseMemoryManager):
        self.memory_manager = memory_manager
        self.defrag_threshold = 0.3  # 碎片率阈值
    
    def should_defragment(self) -> bool:
        """判断是否需要碎片整理"""
        
        fragmentation_rate = self.memory_manager._calculate_fragmentation_rate()
        return fragmentation_rate > self.defrag_threshold
    
    def defragment(self):
        """执行碎片整理"""
        
        logger.info("Starting memory defragmentation")
        
        with self.memory_manager.lock:
            # 收集所有空闲块
            all_free_blocks = []
            for blocks in self.memory_manager.free_blocks.values():
                all_free_blocks.extend(blocks)
            
            # 按地址排序
            all_free_blocks.sort(key=lambda b: b.ptr)
            
            # 合并相邻块
            merged_blocks = []
            current_block = None
            
            for block in all_free_blocks:
                if current_block is None:
                    current_block = block
                elif current_block.ptr + current_block.size == block.ptr:
                    # 相邻块，合并
                    current_block.size += block.size
                else:
                    # 不相邻，保存当前块
                    merged_blocks.append(current_block)
                    current_block = block
            
            if current_block:
                merged_blocks.append(current_block)
            
            # 重建空闲块字典
            self.memory_manager.free_blocks.clear()
            for block in merged_blocks:
                self.memory_manager.free_blocks[block.size].append(block)
            
            logger.info(f"Defragmentation complete: {len(all_free_blocks)} -> {len(merged_blocks)} blocks")
```

### 自适应内存管理

```python
class AdaptiveMemoryManager:
    """自适应内存管理器"""
    
    def __init__(self, base_manager: UnifiedMemoryManager):
        self.base_manager = base_manager
        
        # 使用模式统计
        self.allocation_patterns = defaultdict(list)
        self.usage_history = deque(maxlen=1000)
        
        # 自适应参数
        self.adaptive_pool_sizes = {}
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0  # 60秒调整一次
    
    def record_allocation(self, size: int, memory_type: MemoryType):
        """记录分配模式"""
        
        timestamp = time.time()
        self.allocation_patterns[memory_type].append((timestamp, size))
        self.usage_history.append((timestamp, memory_type, size))
        
        # 定期调整
        if timestamp - self.last_adjustment > self.adjustment_interval:
            self._adjust_pool_sizes()
            self.last_adjustment = timestamp
    
    def _adjust_pool_sizes(self):
        """调整内存池大小"""
        
        logger.info("Adjusting memory pool sizes based on usage patterns")
        
        # 分析最近的使用模式
        recent_usage = defaultdict(list)
        cutoff_time = time.time() - self.adjustment_interval
        
        for timestamp, memory_type, size in self.usage_history:
            if timestamp > cutoff_time:
                recent_usage[memory_type].append(size)
        
        # 为每种类型计算推荐的池大小
        for memory_type, sizes in recent_usage.items():
            if sizes:
                # 计算统计信息
                avg_size = np.mean(sizes)
                std_size = np.std(sizes)
                max_size = np.max(sizes)
                
                # 推荐池大小：平均值 + 2倍标准差
                recommended_size = int(avg_size + 2 * std_size)
                
                # 更新池大小
                self.adaptive_pool_sizes[memory_type] = {
                    'recommended_size': recommended_size,
                    'max_size': max_size,
                    'allocation_count': len(sizes),
                }
                
                logger.debug(f"Memory type {memory_type}: "
                           f"avg={avg_size:.0f}, std={std_size:.0f}, "
                           f"recommended={recommended_size}")
    
    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """获取自适应建议"""
        
        return {
            'pool_size_recommendations': self.adaptive_pool_sizes,
            'allocation_patterns': dict(self.allocation_patterns),
            'recent_usage_count': len(self.usage_history),
        }
```

## 🚀 使用最佳实践

### 1. 内存配置优化

```python
def create_optimized_memory_config(
    device: torch.device,
    model_size: int,
    max_batch_size: int,
    max_seq_len: int,
) -> MemoryConfig:
    """创建优化的内存配置"""
    
    if device.type == "cuda":
        # GPU内存配置
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # 为模型权重预留40%内存
        model_memory = int(total_memory * 0.4)
        
        # 为KV缓存预留30%内存
        cache_memory = int(total_memory * 0.3)
        
        # 为工作空间预留20%内存
        workspace_memory = int(total_memory * 0.2)
        
        # 系统预留10%内存
        safety_margin = 0.1
        
    else:
        # CPU内存配置
        system_memory = psutil.virtual_memory().total
        
        model_memory = min(model_size * 2, int(system_memory * 0.3))
        cache_memory = int(system_memory * 0.2)
        workspace_memory = int(system_memory * 0.1)
        safety_margin = 0.2
    
    return MemoryConfig(
        pool_size=workspace_memory,
        pool_block_sizes=[1024, 4096, 16384, 65536, 262144],
        memory_safety_margin=safety_margin,
        enable_monitoring=True,
        monitoring_interval=5.0,
        high_memory_threshold=0.8,
        blocks_per_size=10,
    )
```

### 2. 内存监控和调试

```python
class MemoryProfiler:
    """内存性能分析器"""
    
    def __init__(self, memory_manager: UnifiedMemoryManager):
        self.memory_manager = memory_manager
        self.profile_data = []
    
    def start_profiling(self):
        """开始性能分析"""
        self.profile_data.clear()
        
        def profile_loop():
            while self.profiling_enabled:
                summary = self.memory_manager.get_memory_summary()
                timestamp = time.time()
                
                self.profile_data.append({
                    'timestamp': timestamp,
                    'summary': summary,
                })
                
                time.sleep(1.0)  # 每秒采样
        
        self.profiling_enabled = True
        self.profile_thread = threading.Thread(target=profile_loop, daemon=True)
        self.profile_thread.start()
    
    def stop_profiling(self):
        """停止性能分析"""
        self.profiling_enabled = False
        if hasattr(self, 'profile_thread'):
            self.profile_thread.join()
    
    def generate_report(self) -> Dict[str, Any]:
        """生成分析报告"""
        
        if not self.profile_data:
            return {}
        
        # 提取时间序列数据
        timestamps = [d['timestamp'] for d in self.profile_data]
        
        if self.memory_manager.device.type == "cuda":
            allocated_memory = [
                d['summary']['system_info']['cuda_memory_allocated']
                for d in self.profile_data
            ]
            
            peak_memory = max(allocated_memory)
            avg_memory = np.mean(allocated_memory)
            
        else:
            allocated_memory = []
            peak_memory = 0
            avg_memory = 0
        
        cache_utilization = [
            d['summary']['kv_cache_manager']['cache_utilization']
            for d in self.profile_data
        ]
        
        return {
            'duration': timestamps[-1] - timestamps[0],
            'sample_count': len(self.profile_data),
            'memory_stats': {
                'peak_allocated': peak_memory,
                'avg_allocated': avg_memory,
                'peak_cache_utilization': max(cache_utilization) if cache_utilization else 0,
                'avg_cache_utilization': np.mean(cache_utilization) if cache_utilization else 0,
            },
            'recommendations': self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        
        recommendations = []
        
        if not self.profile_data:
            return recommendations
        
        # 分析缓存利用率
        cache_utilizations = [
            d['summary']['kv_cache_manager']['cache_utilization']
            for d in self.profile_data
        ]
        
        avg_cache_util = np.mean(cache_utilizations)
        
        if avg_cache_util > 0.9:
            recommendations.append("考虑增加KV缓存块数量")
        elif avg_cache_util < 0.3:
            recommendations.append("可以减少KV缓存块数量以节省内存")
        
        # 分析内存使用模式
        if self.memory_manager.device.type == "cuda":
            memory_usage = [
                d['summary']['system_info']['cuda_memory_allocated']
                for d in self.profile_data
            ]
            
            memory_variance = np.var(memory_usage)
            if memory_variance > (np.mean(memory_usage) * 0.1) ** 2:
                recommendations.append("内存使用波动较大，考虑启用内存预分配")
        
        return recommendations
```

### 3. 错误处理和恢复

```python
class MemoryErrorHandler:
    """内存错误处理器"""
    
    def __init__(self, memory_manager: UnifiedMemoryManager):
        self.memory_manager = memory_manager
        self.error_count = 0
        self.last_oom_time = 0
        self.oom_recovery_strategies = [
            self._strategy_garbage_collection,
            self._strategy_cache_cleanup,
            self._strategy_reduce_batch_size,
            self._strategy_emergency_cleanup,
        ]
    
    def handle_oom_error(self, error: Exception) -> bool:
        """处理内存不足错误"""
        
        self.error_count += 1
        current_time = time.time()
        
        logger.error(f"OOM error #{self.error_count}: {error}")
        
        # 防止频繁的OOM恢复
        if current_time - self.last_oom_time < 5.0:
            logger.warning("Frequent OOM errors detected")
            return False
        
        self.last_oom_time = current_time
        
        # 尝试恢复策略
        for i, strategy in enumerate(self.oom_recovery_strategies):
            logger.info(f"Trying recovery strategy {i+1}")
            
            try:
                if strategy():
                    logger.info(f"Recovery strategy {i+1} succeeded")
                    return True
                    
            except Exception as e:
                logger.error(f"Recovery strategy {i+1} failed: {e}")
        
        logger.error("All recovery strategies failed")
        return False
    
    def _strategy_garbage_collection(self) -> bool:
        """策略1：垃圾回收"""
        
        gc.collect()
        
        if self.memory_manager.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return True
    
    def _strategy_cache_cleanup(self) -> bool:
        """策略2：清理缓存"""
        
        # 清理LRU块
        self.memory_manager.base_manager._cleanup_lru_blocks(target_ratio=0.5)
        
        # 清理内存池
        for pool in self.memory_manager.memory_pool.pools.values():
            pool.clear()
        
        return True
    
    def _strategy_reduce_batch_size(self) -> bool:
        """策略3：减少批处理大小"""
        
        # 这里需要与调度器协调
        # 实际实现中会通知调度器减少批处理大小
        logger.info("Requesting batch size reduction")
        return True
    
    def _strategy_emergency_cleanup(self) -> bool:
        """策略4：紧急清理"""
        
        # 释放所有非必要内存
        self.memory_manager.kv_cache_manager.block_allocator.free_blocks.clear()
        
        # 重新初始化内存池
        self.memory_manager.memory_pool._preallocate_pools()
        
        return True
```

---

*内存管理器是 nano-vllm 性能的关键基础设施，合理的内存管理策略能够显著提升系统的稳定性和吞吐量。*