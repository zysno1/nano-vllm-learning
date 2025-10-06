# 🧠 内存优化 (Memory Optimization)

## 📖 概述

内存优化是nano-vllm性能调优的核心环节，直接影响模型推理的效率和可扩展性。本文档详细介绍内存管理策略、优化技术和最佳实践，帮助开发者构建内存高效的推理系统。

## 🏗️ 内存管理架构

### 1. 统一内存管理器

```python
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading
import time
import logging
from collections import defaultdict
import gc
import psutil

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """内存类型"""
    GPU_MEMORY = "gpu_memory"
    CPU_MEMORY = "cpu_memory"
    SHARED_MEMORY = "shared_memory"
    PINNED_MEMORY = "pinned_memory"

class AllocationStrategy(Enum):
    """分配策略"""
    EAGER = "eager"           # 立即分配
    LAZY = "lazy"             # 延迟分配
    POOLED = "pooled"         # 池化分配
    STREAMING = "streaming"   # 流式分配

@dataclass
class MemoryConfig:
    """内存配置"""
    # GPU内存配置
    gpu_memory_fraction: float = 0.9
    gpu_memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    enable_memory_pool: bool = True
    
    # CPU内存配置
    cpu_memory_limit: int = 8 * 1024 * 1024 * 1024  # 8GB
    enable_cpu_offload: bool = True
    
    # KV缓存配置
    kv_cache_size: int = 512 * 1024 * 1024  # 512MB
    kv_cache_block_size: int = 16
    enable_kv_cache_compression: bool = False
    
    # 优化配置
    enable_gradient_checkpointing: bool = True
    enable_activation_checkpointing: bool = True
    memory_defrag_threshold: float = 0.8
    gc_frequency: int = 100

@dataclass
class MemoryStats:
    """内存统计"""
    total_allocated: int = 0
    total_reserved: int = 0
    peak_allocated: int = 0
    allocation_count: int = 0
    deallocation_count: int = 0
    fragmentation_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_allocated_mb': self.total_allocated / (1024 * 1024),
            'total_reserved_mb': self.total_reserved / (1024 * 1024),
            'peak_allocated_mb': self.peak_allocated / (1024 * 1024),
            'allocation_count': self.allocation_count,
            'deallocation_count': self.deallocation_count,
            'fragmentation_ratio': self.fragmentation_ratio,
        }

class UnifiedMemoryManager:
    """统一内存管理器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.stats = MemoryStats()
        self.lock = threading.RLock()
        
        # 内存池
        self.gpu_memory_pool = {}
        self.cpu_memory_pool = {}
        self.pinned_memory_pool = {}
        
        # 分配记录
        self.allocations = {}
        self.allocation_history = []
        
        # 监控
        self.monitoring_enabled = True
        self.last_gc_time = time.time()
        
        # 初始化
        self._initialize_memory_pools()
        self._setup_monitoring()
    
    def _initialize_memory_pools(self):
        """初始化内存池"""
        
        if torch.cuda.is_available() and self.config.enable_memory_pool:
            # 设置GPU内存分数
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # 预分配GPU内存池
            try:
                pool_tensor = torch.empty(
                    self.config.gpu_memory_pool_size // 4,  # float32
                    dtype=torch.float32,
                    device='cuda'
                )
                self.gpu_memory_pool['main'] = pool_tensor
                logger.info(f"GPU memory pool initialized: {self.config.gpu_memory_pool_size / (1024**3):.2f}GB")
            except RuntimeError as e:
                logger.warning(f"Failed to initialize GPU memory pool: {e}")
        
        # 初始化CPU内存池
        if self.config.enable_cpu_offload:
            self.cpu_memory_pool['main'] = {}
            logger.info("CPU memory pool initialized")
    
    def _setup_monitoring(self):
        """设置内存监控"""
        
        if self.monitoring_enabled:
            # 启动监控线程
            import threading
            monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
            monitor_thread.start()
    
    def allocate(
        self,
        size: int,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
        strategy: AllocationStrategy = AllocationStrategy.POOLED
    ) -> torch.Tensor:
        """分配内存"""
        
        with self.lock:
            start_time = time.time()
            
            try:
                if strategy == AllocationStrategy.POOLED:
                    tensor = self._allocate_from_pool(size, dtype, device)
                elif strategy == AllocationStrategy.EAGER:
                    tensor = self._allocate_eager(size, dtype, device)
                elif strategy == AllocationStrategy.LAZY:
                    tensor = self._allocate_lazy(size, dtype, device)
                else:
                    tensor = torch.empty(size, dtype=dtype, device=device)
                
                # 记录分配
                allocation_id = id(tensor)
                self.allocations[allocation_id] = {
                    'tensor': tensor,
                    'size': tensor.numel() * tensor.element_size(),
                    'device': str(tensor.device),
                    'timestamp': time.time(),
                    'strategy': strategy
                }
                
                # 更新统计
                self.stats.allocation_count += 1
                self.stats.total_allocated += tensor.numel() * tensor.element_size()
                self.stats.peak_allocated = max(
                    self.stats.peak_allocated,
                    self.stats.total_allocated
                )
                
                allocation_time = time.time() - start_time
                logger.debug(f"Memory allocated: {tensor.numel() * tensor.element_size() / (1024**2):.2f}MB "
                           f"in {allocation_time*1000:.2f}ms")
                
                return tensor
                
            except RuntimeError as e:
                logger.error(f"Memory allocation failed: {e}")
                # 尝试垃圾回收后重试
                self._emergency_cleanup()
                raise
    
    def deallocate(self, tensor: torch.Tensor):
        """释放内存"""
        
        with self.lock:
            allocation_id = id(tensor)
            
            if allocation_id in self.allocations:
                allocation_info = self.allocations[allocation_id]
                
                # 更新统计
                self.stats.deallocation_count += 1
                self.stats.total_allocated -= allocation_info['size']
                
                # 移除记录
                del self.allocations[allocation_id]
                
                logger.debug(f"Memory deallocated: {allocation_info['size'] / (1024**2):.2f}MB")
            
            # 删除张量引用
            del tensor
    
    def _allocate_from_pool(
        self,
        size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device]
    ) -> torch.Tensor:
        """从内存池分配"""
        
        device_str = str(device)
        
        if device_str.startswith('cuda') and 'main' in self.gpu_memory_pool:
            # 从GPU内存池分配
            pool_tensor = self.gpu_memory_pool['main']
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_elements = size // element_size
            
            if required_elements <= pool_tensor.numel():
                # 创建视图
                return pool_tensor[:required_elements].view(-1).to(dtype)
        
        # 回退到直接分配
        return torch.empty(size // torch.tensor([], dtype=dtype).element_size(), dtype=dtype, device=device)
    
    def _allocate_eager(
        self,
        size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device]
    ) -> torch.Tensor:
        """立即分配"""
        
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size // element_size
        return torch.empty(num_elements, dtype=dtype, device=device)
    
    def _allocate_lazy(
        self,
        size: int,
        dtype: torch.dtype,
        device: Union[str, torch.device]
    ) -> torch.Tensor:
        """延迟分配"""
        
        # 创建延迟分配的占位符
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size // element_size
        
        # 这里可以实现更复杂的延迟分配逻辑
        return torch.empty(num_elements, dtype=dtype, device=device)
    
    def _emergency_cleanup(self):
        """紧急清理"""
        
        logger.warning("Performing emergency memory cleanup")
        
        # 强制垃圾回收
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 清理过期分配
        current_time = time.time()
        expired_allocations = []
        
        for allocation_id, info in self.allocations.items():
            if current_time - info['timestamp'] > 300:  # 5分钟
                expired_allocations.append(allocation_id)
        
        for allocation_id in expired_allocations:
            if allocation_id in self.allocations:
                del self.allocations[allocation_id]
        
        logger.info(f"Emergency cleanup completed, removed {len(expired_allocations)} expired allocations")
    
    def _memory_monitor_loop(self):
        """内存监控循环"""
        
        while self.monitoring_enabled:
            try:
                self._update_memory_stats()
                
                # 检查是否需要垃圾回收
                if time.time() - self.last_gc_time > 60:  # 每分钟检查一次
                    if self._should_trigger_gc():
                        self._perform_gc()
                
                time.sleep(10)  # 每10秒监控一次
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(30)
    
    def _update_memory_stats(self):
        """更新内存统计"""
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_reserved = torch.cuda.memory_reserved()
            
            self.stats.total_reserved = gpu_reserved
            
            # 计算碎片率
            if gpu_reserved > 0:
                self.stats.fragmentation_ratio = 1.0 - (gpu_allocated / gpu_reserved)
    
    def _should_trigger_gc(self) -> bool:
        """判断是否应该触发垃圾回收"""
        
        # 基于内存使用率和碎片率判断
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            return (memory_usage > self.config.memory_defrag_threshold or
                    self.stats.fragmentation_ratio > 0.3)
        
        return False
    
    def _perform_gc(self):
        """执行垃圾回收"""
        
        logger.info("Performing scheduled garbage collection")
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.last_gc_time = time.time()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        
        info = {
            'stats': self.stats.to_dict(),
            'active_allocations': len(self.allocations),
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'gpu_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'gpu_max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
            })
        
        # CPU内存信息
        process = psutil.Process()
        memory_info = process.memory_info()
        info.update({
            'cpu_rss_mb': memory_info.rss / (1024**2),
            'cpu_vms_mb': memory_info.vms / (1024**2),
        })
        
        return info
    
    def optimize_memory_layout(self):
        """优化内存布局"""
        
        logger.info("Optimizing memory layout")
        
        with self.lock:
            # 整理内存碎片
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 重新组织内存池
            self._reorganize_memory_pools()
    
    def _reorganize_memory_pools(self):
        """重新组织内存池"""
        
        # 这里可以实现更复杂的内存池重组逻辑
        pass
    
    def cleanup(self):
        """清理资源"""
        
        self.monitoring_enabled = False
        
        with self.lock:
            # 清理所有分配
            self.allocations.clear()
            
            # 清理内存池
            self.gpu_memory_pool.clear()
            self.cpu_memory_pool.clear()
            self.pinned_memory_pool.clear()
        
        # 最终垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory manager cleaned up")
```

### 2. KV缓存优化器

```python
class KVCacheOptimizer:
    """KV缓存优化器"""
    
    def __init__(self, config: MemoryConfig, memory_manager: UnifiedMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        
        # 缓存配置
        self.block_size = config.kv_cache_block_size
        self.cache_size = config.kv_cache_size
        self.enable_compression = config.enable_kv_cache_compression
        
        # 缓存存储
        self.key_cache = {}
        self.value_cache = {}
        self.cache_metadata = {}
        
        # 压缩器
        self.compressor = KVCacheCompressor() if self.enable_compression else None
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_ratio = 0.0
    
    def allocate_kv_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """分配KV缓存"""
        
        # 计算缓存大小
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        
        # 分配key缓存
        key_cache = self.memory_manager.allocate(
            size=np.prod(cache_shape) * torch.tensor([], dtype=dtype).element_size(),
            dtype=dtype,
            strategy=AllocationStrategy.POOLED
        ).view(cache_shape)
        
        # 分配value缓存
        value_cache = self.memory_manager.allocate(
            size=np.prod(cache_shape) * torch.tensor([], dtype=dtype).element_size(),
            dtype=dtype,
            strategy=AllocationStrategy.POOLED
        ).view(cache_shape)
        
        # 记录缓存元数据
        cache_id = f"{batch_size}_{num_heads}_{head_dim}_{max_seq_len}"
        self.cache_metadata[cache_id] = {
            'shape': cache_shape,
            'dtype': dtype,
            'allocated_time': time.time(),
            'access_count': 0,
        }
        
        logger.debug(f"KV cache allocated: {cache_shape}, dtype: {dtype}")
        
        return key_cache, value_cache
    
    def get_cached_kv(
        self,
        sequence_id: str,
        layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存的KV"""
        
        cache_key = f"{sequence_id}_{layer_idx}"
        
        if cache_key in self.key_cache and cache_key in self.value_cache:
            self.cache_hits += 1
            
            # 更新访问统计
            if cache_key in self.cache_metadata:
                self.cache_metadata[cache_key]['access_count'] += 1
            
            key_cache = self.key_cache[cache_key]
            value_cache = self.value_cache[cache_key]
            
            # 如果启用压缩，需要解压缩
            if self.enable_compression and self.compressor:
                key_cache = self.compressor.decompress(key_cache)
                value_cache = self.compressor.decompress(value_cache)
            
            return key_cache, value_cache
        
        self.cache_misses += 1
        return None
    
    def cache_kv(
        self,
        sequence_id: str,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ):
        """缓存KV状态"""
        
        cache_key = f"{sequence_id}_{layer_idx}"
        
        # 如果启用压缩，先压缩
        if self.enable_compression and self.compressor:
            compressed_key = self.compressor.compress(key_states)
            compressed_value = self.compressor.compress(value_states)
            
            self.key_cache[cache_key] = compressed_key
            self.value_cache[cache_key] = compressed_value
            
            # 更新压缩比
            original_size = key_states.numel() * key_states.element_size() + \
                          value_states.numel() * value_states.element_size()
            compressed_size = compressed_key.numel() * compressed_key.element_size() + \
                            compressed_value.numel() * compressed_value.element_size()
            self.compression_ratio = compressed_size / original_size
        else:
            self.key_cache[cache_key] = key_states.clone()
            self.value_cache[cache_key] = value_states.clone()
        
        # 更新元数据
        self.cache_metadata[cache_key] = {
            'shape': key_states.shape,
            'dtype': key_states.dtype,
            'cached_time': time.time(),
            'access_count': 0,
        }
    
    def evict_cache(self, sequence_id: str):
        """驱逐缓存"""
        
        keys_to_remove = []
        for cache_key in self.key_cache.keys():
            if cache_key.startswith(f"{sequence_id}_"):
                keys_to_remove.append(cache_key)
        
        for cache_key in keys_to_remove:
            if cache_key in self.key_cache:
                del self.key_cache[cache_key]
            if cache_key in self.value_cache:
                del self.value_cache[cache_key]
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]
        
        logger.debug(f"Evicted cache for sequence: {sequence_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cached_sequences': len(self.key_cache),
            'compression_ratio': self.compression_ratio,
        }

class KVCacheCompressor:
    """KV缓存压缩器"""
    
    def __init__(self, compression_method: str = "quantization"):
        self.compression_method = compression_method
    
    def compress(self, tensor: torch.Tensor) -> torch.Tensor:
        """压缩张量"""
        
        if self.compression_method == "quantization":
            return self._quantize_tensor(tensor)
        elif self.compression_method == "pruning":
            return self._prune_tensor(tensor)
        else:
            return tensor
    
    def decompress(self, compressed_tensor: torch.Tensor) -> torch.Tensor:
        """解压缩张量"""
        
        if self.compression_method == "quantization":
            return self._dequantize_tensor(compressed_tensor)
        elif self.compression_method == "pruning":
            return self._unprune_tensor(compressed_tensor)
        else:
            return compressed_tensor
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """量化张量"""
        
        # 简单的8位量化
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        
        scale = (tensor_max - tensor_min) / 255.0
        zero_point = tensor_min
        
        quantized = ((tensor - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)
        
        # 保存量化参数
        quantized_with_params = torch.cat([
            scale.unsqueeze(0).to(torch.float32).view(-1),
            zero_point.unsqueeze(0).to(torch.float32).view(-1),
            quantized.view(-1).to(torch.float32)
        ])
        
        return quantized_with_params
    
    def _dequantize_tensor(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """反量化张量"""
        
        # 提取量化参数
        scale = quantized_tensor[0]
        zero_point = quantized_tensor[1]
        quantized_data = quantized_tensor[2:].to(torch.uint8)
        
        # 反量化
        dequantized = quantized_data.to(torch.float32) * scale + zero_point
        
        return dequantized
    
    def _prune_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """剪枝张量"""
        
        # 简单的幅度剪枝
        threshold = tensor.abs().quantile(0.1)  # 保留90%的权重
        mask = tensor.abs() > threshold
        
        pruned_tensor = tensor * mask
        
        return pruned_tensor
    
    def _unprune_tensor(self, pruned_tensor: torch.Tensor) -> torch.Tensor:
        """恢复剪枝张量"""
        
        # 剪枝是不可逆的，这里直接返回
        return pruned_tensor
```

### 3. 梯度检查点优化器

```python
class GradientCheckpointOptimizer:
    """梯度检查点优化器"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.enable_checkpointing = config.enable_gradient_checkpointing
        self.enable_activation_checkpointing = config.enable_activation_checkpointing
        
        # 检查点策略
        self.checkpoint_layers = []
        self.activation_checkpoints = {}
        
        # 统计信息
        self.memory_saved = 0
        self.recomputation_overhead = 0.0
    
    def setup_gradient_checkpointing(self, model: torch.nn.Module):
        """设置梯度检查点"""
        
        if not self.enable_checkpointing:
            return model
        
        # 为Transformer层设置检查点
        for name, module in model.named_modules():
            if self._should_checkpoint_layer(name, module):
                self._apply_gradient_checkpointing(module)
                self.checkpoint_layers.append(name)
        
        logger.info(f"Gradient checkpointing applied to {len(self.checkpoint_layers)} layers")
        
        return model
    
    def _should_checkpoint_layer(self, name: str, module: torch.nn.Module) -> bool:
        """判断是否应该对层应用检查点"""
        
        # 通常对Transformer块应用检查点
        checkpoint_patterns = [
            'transformer.h.',
            'layers.',
            'decoder.layers.',
            'encoder.layers.'
        ]
        
        return any(pattern in name for pattern in checkpoint_patterns)
    
    def _apply_gradient_checkpointing(self, module: torch.nn.Module):
        """应用梯度检查点"""
        
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            if self.enable_checkpointing and any(arg.requires_grad for arg in args if isinstance(arg, torch.Tensor)):
                return torch.utils.checkpoint.checkpoint(
                    original_forward,
                    *args,
                    **kwargs,
                    use_reentrant=False
                )
            else:
                return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
    
    def checkpoint_activations(
        self,
        layer_name: str,
        activations: torch.Tensor,
        checkpoint_ratio: float = 0.5
    ) -> torch.Tensor:
        """检查点激活"""
        
        if not self.enable_activation_checkpointing:
            return activations
        
        # 决定是否检查点这个激活
        if torch.rand(1).item() < checkpoint_ratio:
            # 将激活移到CPU
            cpu_activations = activations.cpu()
            
            # 记录检查点
            checkpoint_id = f"{layer_name}_{time.time()}"
            self.activation_checkpoints[checkpoint_id] = {
                'activations': cpu_activations,
                'device': activations.device,
                'shape': activations.shape,
                'dtype': activations.dtype,
            }
            
            # 创建一个需要时重新加载的占位符
            placeholder = self._create_activation_placeholder(
                checkpoint_id,
                activations.shape,
                activations.dtype,
                activations.device
            )
            
            # 计算节省的内存
            memory_saved = activations.numel() * activations.element_size()
            self.memory_saved += memory_saved
            
            return placeholder
        
        return activations
    
    def _create_activation_placeholder(
        self,
        checkpoint_id: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """创建激活占位符"""
        
        class ActivationPlaceholder(torch.Tensor):
            def __new__(cls, checkpoint_id, shape, dtype, device, optimizer):
                return torch.empty(shape, dtype=dtype, device=device)
            
            def __init__(self, checkpoint_id, shape, dtype, device, optimizer):
                self.checkpoint_id = checkpoint_id
                self.optimizer = optimizer
                self._loaded = False
            
            def __getattr__(self, name):
                if not self._loaded:
                    self._load_from_checkpoint()
                return super().__getattr__(name)
            
            def _load_from_checkpoint(self):
                if self.checkpoint_id in self.optimizer.activation_checkpoints:
                    checkpoint_data = self.optimizer.activation_checkpoints[self.checkpoint_id]
                    cpu_activations = checkpoint_data['activations']
                    
                    # 重新加载到GPU
                    gpu_activations = cpu_activations.to(checkpoint_data['device'])
                    self.data = gpu_activations.data
                    self._loaded = True
                    
                    # 清理检查点
                    del self.optimizer.activation_checkpoints[self.checkpoint_id]
        
        return ActivationPlaceholder(checkpoint_id, shape, dtype, device, self)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计"""
        
        return {
            'checkpointed_layers': len(self.checkpoint_layers),
            'active_checkpoints': len(self.activation_checkpoints),
            'memory_saved_mb': self.memory_saved / (1024**2),
            'recomputation_overhead_ms': self.recomputation_overhead * 1000,
        }
    
    def cleanup_checkpoints(self):
        """清理检查点"""
        
        self.activation_checkpoints.clear()
        self.memory_saved = 0
        logger.info("Gradient checkpoints cleaned up")
```

### 4. 内存碎片整理器

```python
class MemoryDefragmenter:
    """内存碎片整理器"""
    
    def __init__(self, memory_manager: UnifiedMemoryManager):
        self.memory_manager = memory_manager
        self.defrag_threshold = 0.3  # 碎片率阈值
        self.last_defrag_time = time.time()
        self.defrag_interval = 300  # 5分钟
    
    def should_defragment(self) -> bool:
        """判断是否需要碎片整理"""
        
        current_time = time.time()
        
        # 检查时间间隔
        if current_time - self.last_defrag_time < self.defrag_interval:
            return False
        
        # 检查碎片率
        fragmentation_ratio = self.memory_manager.stats.fragmentation_ratio
        
        return fragmentation_ratio > self.defrag_threshold
    
    def defragment_memory(self):
        """执行内存碎片整理"""
        
        logger.info("Starting memory defragmentation")
        start_time = time.time()
        
        try:
            # 1. 收集所有活跃的张量
            active_tensors = self._collect_active_tensors()
            
            # 2. 创建新的连续内存区域
            consolidated_tensors = self._consolidate_tensors(active_tensors)
            
            # 3. 更新引用
            self._update_tensor_references(active_tensors, consolidated_tensors)
            
            # 4. 清理旧内存
            self._cleanup_old_memory()
            
            # 5. 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            defrag_time = time.time() - start_time
            self.last_defrag_time = time.time()
            
            logger.info(f"Memory defragmentation completed in {defrag_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Memory defragmentation failed: {e}")
    
    def _collect_active_tensors(self) -> List[Dict[str, Any]]:
        """收集活跃张量"""
        
        active_tensors = []
        
        for allocation_id, allocation_info in self.memory_manager.allocations.items():
            tensor = allocation_info['tensor']
            
            if tensor is not None and tensor.is_cuda:
                active_tensors.append({
                    'id': allocation_id,
                    'tensor': tensor,
                    'size': allocation_info['size'],
                    'device': allocation_info['device'],
                })
        
        # 按大小排序，大的张量优先
        active_tensors.sort(key=lambda x: x['size'], reverse=True)
        
        return active_tensors
    
    def _consolidate_tensors(self, active_tensors: List[Dict[str, Any]]) -> List[torch.Tensor]:
        """整合张量到连续内存"""
        
        consolidated_tensors = []
        
        # 按设备分组
        device_groups = defaultdict(list)
        for tensor_info in active_tensors:
            device = tensor_info['device']
            device_groups[device].append(tensor_info)
        
        for device, tensor_infos in device_groups.items():
            # 计算总大小
            total_size = sum(info['size'] for info in tensor_infos)
            
            # 分配连续内存
            consolidated_memory = self.memory_manager.allocate(
                size=total_size,
                dtype=torch.uint8,  # 使用字节类型
                device=device,
                strategy=AllocationStrategy.EAGER
            )
            
            # 复制张量数据
            offset = 0
            for tensor_info in tensor_infos:
                tensor = tensor_info['tensor']
                tensor_bytes = tensor.numel() * tensor.element_size()
                
                # 复制数据
                consolidated_memory[offset:offset + tensor_bytes] = tensor.view(-1).to(torch.uint8)
                
                # 创建新的张量视图
                new_tensor = consolidated_memory[offset:offset + tensor_bytes].view(tensor.shape).to(tensor.dtype)
                consolidated_tensors.append(new_tensor)
                
                offset += tensor_bytes
        
        return consolidated_tensors
    
    def _update_tensor_references(
        self,
        active_tensors: List[Dict[str, Any]],
        consolidated_tensors: List[torch.Tensor]
    ):
        """更新张量引用"""
        
        for i, tensor_info in enumerate(active_tensors):
            allocation_id = tensor_info['id']
            new_tensor = consolidated_tensors[i]
            
            # 更新分配记录
            if allocation_id in self.memory_manager.allocations:
                self.memory_manager.allocations[allocation_id]['tensor'] = new_tensor
    
    def _cleanup_old_memory(self):
        """清理旧内存"""
        
        # 这里可以实现更复杂的清理逻辑
        pass
    
    def get_fragmentation_info(self) -> Dict[str, Any]:
        """获取碎片信息"""
        
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        
        fragmentation_ratio = 1.0 - (allocated / reserved) if reserved > 0 else 0.0
        
        return {
            'allocated_mb': allocated / (1024**2),
            'reserved_mb': reserved / (1024**2),
            'fragmentation_ratio': fragmentation_ratio,
            'fragmentation_mb': (reserved - allocated) / (1024**2),
            'last_defrag_time': self.last_defrag_time,
        }
```

## 🚀 使用示例

### 1. 基本内存管理

```python
def main():
    # 配置内存管理
    config = MemoryConfig(
        gpu_memory_fraction=0.8,
        enable_memory_pool=True,
        enable_gradient_checkpointing=True,
        kv_cache_size=1024 * 1024 * 1024,  # 1GB
    )
    
    # 创建内存管理器
    memory_manager = UnifiedMemoryManager(config)
    
    try:
        # 分配内存
        tensor1 = memory_manager.allocate(
            size=1024 * 1024 * 4,  # 4MB
            dtype=torch.float32,
            device='cuda'
        )
        
        tensor2 = memory_manager.allocate(
            size=2048 * 1024 * 4,  # 8MB
            dtype=torch.float16,
            device='cuda'
        )
        
        # 使用张量进行计算
        result = torch.matmul(tensor1.view(1024, 1024), tensor2.view(2048, 1024).T)
        
        # 获取内存信息
        memory_info = memory_manager.get_memory_info()
        print(f"Memory info: {memory_info}")
        
        # 释放内存
        memory_manager.deallocate(tensor1)
        memory_manager.deallocate(tensor2)
        
    finally:
        # 清理资源
        memory_manager.cleanup()

if __name__ == "__main__":
    main()
```

### 2. KV缓存优化

```python
def kv_cache_example():
    config = MemoryConfig(
        kv_cache_size=512 * 1024 * 1024,  # 512MB
        enable_kv_cache_compression=True
    )
    
    memory_manager = UnifiedMemoryManager(config)
    kv_optimizer = KVCacheOptimizer(config, memory_manager)
    
    # 分配KV缓存
    batch_size, num_heads, head_dim, max_seq_len = 4, 32, 128, 2048
    
    key_cache, value_cache = kv_optimizer.allocate_kv_cache(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=torch.float16
    )
    
    # 模拟KV缓存使用
    for layer_idx in range(24):  # 24层
        sequence_id = "seq_001"
        
        # 生成随机KV状态
        key_states = torch.randn(batch_size, num_heads, 128, head_dim, dtype=torch.float16, device='cuda')
        value_states = torch.randn(batch_size, num_heads, 128, head_dim, dtype=torch.float16, device='cuda')
        
        # 缓存KV状态
        kv_optimizer.cache_kv(sequence_id, layer_idx, key_states, value_states)
        
        # 获取缓存的KV状态
        cached_kv = kv_optimizer.get_cached_kv(sequence_id, layer_idx)
        
        if cached_kv is not None:
            cached_key, cached_value = cached_kv
            print(f"Layer {layer_idx}: Cache hit, shapes: {cached_key.shape}, {cached_value.shape}")
    
    # 获取缓存统计
    cache_stats = kv_optimizer.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    
    # 清理
    kv_optimizer.evict_cache("seq_001")
    memory_manager.cleanup()

if __name__ == "__main__":
    kv_cache_example()
```

## 🎯 最佳实践

### 1. 内存分配策略
- **预分配**: 在推理开始前预分配大块内存
- **池化管理**: 使用内存池减少分配/释放开销
- **延迟分配**: 对于不确定大小的张量使用延迟分配
- **分层管理**: 根据数据访问模式选择不同存储层级

### 2. KV缓存优化
- **合理设置缓存大小**: 根据模型和序列长度配置
- **启用压缩**: 在内存受限时使用KV缓存压缩
- **及时清理**: 定期清理过期的缓存条目
- **监控命中率**: 跟踪缓存效果并调整策略

### 3. 梯度检查点
- **选择性应用**: 只对内存消耗大的层应用检查点
- **平衡权衡**: 在内存节省和计算开销间找平衡
- **动态调整**: 根据可用内存动态启用/禁用检查点

### 4. 内存监控
- **实时监控**: 持续跟踪内存使用情况
- **设置告警**: 在内存使用率过高时及时告警
- **定期整理**: 定期执行内存碎片整理
- **性能分析**: 使用工具分析内存瓶颈

## 📈 总结

内存优化是nano-vllm性能调优的核心，通过统一内存管理、KV缓存优化、梯度检查点和碎片整理等技术，可以显著提升内存效率和系统性能。

关键要点：
1. **统一管理**: 建立统一的内存管理框架
2. **智能分配**: 根据使用模式选择最优分配策略
3. **缓存优化**: 通过KV缓存减少重复计算
4. **动态调整**: 根据运行时情况动态优化内存使用

通过合理应用这些内存优化技术，可以在有限的硬件资源下支持更大的模型和更高的并发量。