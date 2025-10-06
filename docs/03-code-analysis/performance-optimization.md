# 性能优化分析

## 🎯 性能优化概览

性能优化是 nano-vllm 的核心竞争力之一。本文档深入分析各种性能优化技术的实现原理，包括计算优化、内存优化、通信优化、以及系统级优化策略。

## 🏗️ 核心优化架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import threading
import multiprocessing
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from contextlib import contextmanager

from nano_vllm.config import ModelConfig, GenerationConfig
from nano_vllm.utils.logger import get_logger

logger = get_logger(__name__)

class OptimizationType(Enum):
    """优化类型枚举"""
    COMPUTE = "compute"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    SYSTEM = "system"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    throughput: float = 0.0  # tokens/second
    latency: float = 0.0     # seconds
    memory_usage: float = 0.0  # MB
    gpu_utilization: float = 0.0  # %
    cpu_utilization: float = 0.0  # %
    cache_hit_rate: float = 0.0  # %
    batch_efficiency: float = 0.0  # %
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'throughput': self.throughput,
            'latency': self.latency,
            'memory_usage': self.memory_usage,
            'gpu_utilization': self.gpu_utilization,
            'cpu_utilization': self.cpu_utilization,
            'cache_hit_rate': self.cache_hit_rate,
            'batch_efficiency': self.batch_efficiency,
        }

@dataclass
class OptimizationConfig:
    """优化配置"""
    enable_compute_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_communication_optimization: bool = True
    enable_system_optimization: bool = True
    
    # 计算优化
    use_flash_attention: bool = True
    use_fused_kernels: bool = True
    use_mixed_precision: bool = True
    
    # 内存优化
    use_kv_cache: bool = True
    use_memory_pool: bool = True
    use_gradient_checkpointing: bool = False
    
    # 通信优化
    use_tensor_parallelism: bool = False
    use_pipeline_parallelism: bool = False
    
    # 系统优化
    use_async_processing: bool = True
    use_batch_optimization: bool = True
    prefetch_factor: int = 2

class BaseOptimizer(ABC):
    """基础优化器抽象类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.optimization_history = []
        
    @abstractmethod
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行优化"""
        pass
    
    @abstractmethod
    def get_optimization_type(self) -> OptimizationType:
        """获取优化类型"""
        pass
    
    def update_metrics(self, metrics: PerformanceMetrics):
        """更新性能指标"""
        self.metrics = metrics
        self.optimization_history.append(metrics)

class ComputeOptimizer(BaseOptimizer):
    """计算优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
        # 融合算子缓存
        self.fused_ops_cache = {}
        
        # 编译缓存
        self.compiled_models = {}
        
        logger.info("Initialized ComputeOptimizer")
    
    def get_optimization_type(self) -> OptimizationType:
        return OptimizationType.COMPUTE
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行计算优化"""
        
        optimized_model = model
        
        # 1. 融合算子优化
        if self.config.use_fused_kernels:
            optimized_model = self._apply_kernel_fusion(optimized_model)
        
        # 2. 混合精度优化
        if self.config.use_mixed_precision:
            optimized_model = self._apply_mixed_precision(optimized_model)
        
        # 3. 编译优化
        optimized_model = self._apply_compilation_optimization(optimized_model)
        
        # 4. Flash Attention优化
        if self.config.use_flash_attention:
            optimized_model = self._apply_flash_attention(optimized_model)
        
        logger.info("Applied compute optimizations")
        return optimized_model
    
    def _apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """应用算子融合优化"""
        
        class FusedLinearGELU(nn.Module):
            """融合的Linear+GELU层"""
            
            def __init__(self, linear: nn.Linear):
                super().__init__()
                self.weight = linear.weight
                self.bias = linear.bias
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 融合Linear和GELU操作
                return F.gelu(F.linear(x, self.weight, self.bias))
        
        class FusedLayerNormLinear(nn.Module):
            """融合的LayerNorm+Linear层"""
            
            def __init__(self, layer_norm: nn.LayerNorm, linear: nn.Linear):
                super().__init__()
                self.layer_norm = layer_norm
                self.linear = linear
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 融合LayerNorm和Linear操作
                normalized = self.layer_norm(x)
                return self.linear(normalized)
        
        # 遍历模型并替换可融合的层
        def replace_fusable_modules(module: nn.Module) -> nn.Module:
            for name, child in module.named_children():
                if isinstance(child, nn.Sequential):
                    # 检查Sequential中的连续层
                    new_layers = []
                    i = 0
                    while i < len(child):
                        current_layer = child[i]
                        
                        # 检查Linear+GELU模式
                        if (i + 1 < len(child) and 
                            isinstance(current_layer, nn.Linear) and
                            isinstance(child[i + 1], nn.GELU)):
                            
                            fused_layer = FusedLinearGELU(current_layer)
                            new_layers.append(fused_layer)
                            i += 2  # 跳过下一层
                        
                        # 检查LayerNorm+Linear模式
                        elif (i + 1 < len(child) and 
                              isinstance(current_layer, nn.LayerNorm) and
                              isinstance(child[i + 1], nn.Linear)):
                            
                            fused_layer = FusedLayerNormLinear(current_layer, child[i + 1])
                            new_layers.append(fused_layer)
                            i += 2  # 跳过下一层
                        
                        else:
                            new_layers.append(current_layer)
                            i += 1
                    
                    # 替换Sequential
                    setattr(module, name, nn.Sequential(*new_layers))
                
                else:
                    # 递归处理子模块
                    replace_fusable_modules(child)
            
            return module
        
        return replace_fusable_modules(model)
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """应用混合精度优化"""
        
        class MixedPrecisionWrapper(nn.Module):
            """混合精度包装器"""
            
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model
                
                # 自动混合精度scaler
                self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            
            def forward(self, *args, **kwargs):
                if self.scaler is not None and torch.cuda.is_available():
                    # 使用自动混合精度
                    with torch.cuda.amp.autocast():
                        return self.model(*args, **kwargs)
                else:
                    return self.model(*args, **kwargs)
        
        # 转换模型参数到半精度（推理时）
        if not model.training:
            model = model.half()
        
        return MixedPrecisionWrapper(model)
    
    def _apply_compilation_optimization(self, model: nn.Module) -> nn.Module:
        """应用编译优化"""
        
        model_id = id(model)
        
        if model_id in self.compiled_models:
            return self.compiled_models[model_id]
        
        try:
            # 使用torch.compile进行优化（PyTorch 2.0+）
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    mode='max-autotune',  # 最大优化模式
                    fullgraph=True,       # 完整图优化
                )
                
                self.compiled_models[model_id] = compiled_model
                logger.info("Applied torch.compile optimization")
                return compiled_model
            
            else:
                # 使用TorchScript优化
                traced_model = torch.jit.trace(model, example_inputs)
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                
                self.compiled_models[model_id] = optimized_model
                logger.info("Applied TorchScript optimization")
                return optimized_model
                
        except Exception as e:
            logger.warning(f"Compilation optimization failed: {e}")
            return model
    
    def _apply_flash_attention(self, model: nn.Module) -> nn.Module:
        """应用Flash Attention优化"""
        
        class OptimizedAttention(nn.Module):
            """优化的注意力机制"""
            
            def __init__(self, original_attention: nn.Module):
                super().__init__()
                self.original_attention = original_attention
                
                # 检查是否支持Flash Attention
                self.use_flash_attention = self._check_flash_attention_support()
            
            def _check_flash_attention_support(self) -> bool:
                """检查Flash Attention支持"""
                try:
                    # 检查是否安装了flash-attn
                    import flash_attn
                    return True
                except ImportError:
                    return False
            
            def forward(self, query, key, value, attention_mask=None, **kwargs):
                if self.use_flash_attention and attention_mask is None:
                    # 使用Flash Attention
                    return self._flash_attention_forward(query, key, value)
                else:
                    # 使用原始注意力
                    return self.original_attention(query, key, value, attention_mask, **kwargs)
            
            def _flash_attention_forward(self, query, key, value):
                """Flash Attention前向传播"""
                try:
                    from flash_attn import flash_attn_func
                    
                    # 重塑张量以适应Flash Attention
                    batch_size, seq_len, num_heads, head_dim = query.shape
                    
                    # Flash Attention需要 (batch, seq_len, num_heads, head_dim) 格式
                    output = flash_attn_func(
                        query, key, value,
                        dropout_p=0.0,
                        softmax_scale=1.0 / (head_dim ** 0.5),
                        causal=True,
                    )
                    
                    return output
                    
                except Exception as e:
                    logger.warning(f"Flash Attention failed, falling back: {e}")
                    return self.original_attention(query, key, value)
        
        # 替换注意力层
        def replace_attention_layers(module: nn.Module):
            for name, child in module.named_children():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    # 替换注意力层
                    optimized_attention = OptimizedAttention(child)
                    setattr(module, name, optimized_attention)
                else:
                    # 递归处理子模块
                    replace_attention_layers(child)
        
        replace_attention_layers(model)
        return model

class MemoryOptimizer(BaseOptimizer):
    """内存优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
        # 内存池
        self.memory_pools = {
            'activations': [],
            'gradients': [],
            'kv_cache': [],
        }
        
        # 内存统计
        self.memory_stats = {
            'peak_memory': 0,
            'current_memory': 0,
            'pool_hits': 0,
            'pool_misses': 0,
        }
        
        logger.info("Initialized MemoryOptimizer")
    
    def get_optimization_type(self) -> OptimizationType:
        return OptimizationType.MEMORY
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行内存优化"""
        
        optimized_model = model
        
        # 1. 梯度检查点优化
        if self.config.use_gradient_checkpointing:
            optimized_model = self._apply_gradient_checkpointing(optimized_model)
        
        # 2. 内存池优化
        if self.config.use_memory_pool:
            optimized_model = self._apply_memory_pooling(optimized_model)
        
        # 3. KV缓存优化
        if self.config.use_kv_cache:
            optimized_model = self._apply_kv_cache_optimization(optimized_model)
        
        # 4. 激活重计算优化
        optimized_model = self._apply_activation_recomputation(optimized_model)
        
        logger.info("Applied memory optimizations")
        return optimized_model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """应用梯度检查点优化"""
        
        class CheckpointedModule(nn.Module):
            """带检查点的模块"""
            
            def __init__(self, module: nn.Module):
                super().__init__()
                self.module = module
            
            def forward(self, *args, **kwargs):
                if self.training:
                    # 训练时使用梯度检查点
                    return torch.utils.checkpoint.checkpoint(
                        self.module, *args, **kwargs
                    )
                else:
                    # 推理时直接调用
                    return self.module(*args, **kwargs)
        
        # 为大型层添加检查点
        def add_checkpoints(module: nn.Module, threshold_params: int = 1000000):
            for name, child in module.named_children():
                # 计算参数数量
                num_params = sum(p.numel() for p in child.parameters())
                
                if num_params > threshold_params:
                    # 为大型模块添加检查点
                    checkpointed_module = CheckpointedModule(child)
                    setattr(module, name, checkpointed_module)
                else:
                    # 递归处理子模块
                    add_checkpoints(child, threshold_params)
        
        add_checkpoints(model)
        return model
    
    def _apply_memory_pooling(self, model: nn.Module) -> nn.Module:
        """应用内存池优化"""
        
        class MemoryPooledModule(nn.Module):
            """使用内存池的模块"""
            
            def __init__(self, module: nn.Module, memory_optimizer: 'MemoryOptimizer'):
                super().__init__()
                self.module = module
                self.memory_optimizer = memory_optimizer
            
            def forward(self, *args, **kwargs):
                # 从内存池获取张量
                pooled_tensors = self.memory_optimizer._get_pooled_tensors(args)
                
                try:
                    # 执行前向传播
                    output = self.module(*args, **kwargs)
                    return output
                    
                finally:
                    # 返回张量到内存池
                    self.memory_optimizer._return_pooled_tensors(pooled_tensors)
        
        return MemoryPooledModule(model, self)
    
    def _get_pooled_tensors(self, tensors: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
        """从内存池获取张量"""
        
        pooled_tensors = []
        
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                # 尝试从池中获取相同大小的张量
                pooled_tensor = self._get_tensor_from_pool(tensor.shape, tensor.dtype, tensor.device)
                
                if pooled_tensor is not None:
                    pooled_tensor.copy_(tensor)
                    pooled_tensors.append(pooled_tensor)
                    self.memory_stats['pool_hits'] += 1
                else:
                    pooled_tensors.append(tensor)
                    self.memory_stats['pool_misses'] += 1
        
        return pooled_tensors
    
    def _return_pooled_tensors(self, tensors: List[torch.Tensor]):
        """返回张量到内存池"""
        
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                self._return_tensor_to_pool(tensor)
    
    def _get_tensor_from_pool(
        self, 
        shape: torch.Size, 
        dtype: torch.dtype, 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """从池中获取张量"""
        
        pool_key = f"{shape}_{dtype}_{device}"
        
        if pool_key in self.memory_pools['activations']:
            pool = self.memory_pools['activations'][pool_key]
            if pool:
                return pool.pop()
        
        return None
    
    def _return_tensor_to_pool(self, tensor: torch.Tensor):
        """返回张量到池中"""
        
        pool_key = f"{tensor.shape}_{tensor.dtype}_{tensor.device}"
        
        if pool_key not in self.memory_pools['activations']:
            self.memory_pools['activations'][pool_key] = []
        
        pool = self.memory_pools['activations'][pool_key]
        
        # 限制池大小
        if len(pool) < 10:
            tensor.zero_()  # 清零张量
            pool.append(tensor)
    
    def _apply_kv_cache_optimization(self, model: nn.Module) -> nn.Module:
        """应用KV缓存优化"""
        
        class OptimizedKVCache:
            """优化的KV缓存"""
            
            def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int):
                self.max_batch_size = max_batch_size
                self.max_seq_len = max_seq_len
                self.num_heads = num_heads
                self.head_dim = head_dim
                
                # 预分配缓存空间
                self.key_cache = torch.zeros(
                    max_batch_size, num_heads, max_seq_len, head_dim,
                    dtype=torch.float16
                )
                self.value_cache = torch.zeros(
                    max_batch_size, num_heads, max_seq_len, head_dim,
                    dtype=torch.float16
                )
                
                # 缓存状态
                self.cache_lengths = torch.zeros(max_batch_size, dtype=torch.long)
            
            def update_cache(
                self,
                batch_idx: int,
                new_keys: torch.Tensor,
                new_values: torch.Tensor,
                start_pos: int,
            ):
                """更新缓存"""
                
                seq_len = new_keys.size(-2)
                end_pos = start_pos + seq_len
                
                # 更新key缓存
                self.key_cache[batch_idx, :, start_pos:end_pos, :] = new_keys
                
                # 更新value缓存
                self.value_cache[batch_idx, :, start_pos:end_pos, :] = new_values
                
                # 更新缓存长度
                self.cache_lengths[batch_idx] = end_pos
            
            def get_cache(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
                """获取缓存"""
                
                cache_len = self.cache_lengths[batch_idx]
                
                keys = self.key_cache[batch_idx, :, :cache_len, :]
                values = self.value_cache[batch_idx, :, :cache_len, :]
                
                return keys, values
            
            def clear_cache(self, batch_idx: Optional[int] = None):
                """清空缓存"""
                
                if batch_idx is not None:
                    self.cache_lengths[batch_idx] = 0
                else:
                    self.cache_lengths.zero_()
        
        # 为模型添加KV缓存
        if not hasattr(model, 'kv_cache'):
            model.kv_cache = OptimizedKVCache(
                max_batch_size=32,
                max_seq_len=2048,
                num_heads=32,  # 根据实际模型配置
                head_dim=128,  # 根据实际模型配置
            )
        
        return model
    
    def _apply_activation_recomputation(self, model: nn.Module) -> nn.Module:
        """应用激活重计算优化"""
        
        class RecomputationModule(nn.Module):
            """激活重计算模块"""
            
            def __init__(self, module: nn.Module, recompute_ratio: float = 0.5):
                super().__init__()
                self.module = module
                self.recompute_ratio = recompute_ratio
                
                # 标记需要重计算的层
                self.recompute_layers = self._select_recompute_layers()
            
            def _select_recompute_layers(self) -> List[str]:
                """选择需要重计算的层"""
                
                layers = []
                total_params = sum(p.numel() for p in self.module.parameters())
                
                for name, child in self.module.named_modules():
                    child_params = sum(p.numel() for p in child.parameters())
                    
                    # 选择参数较多的层进行重计算
                    if child_params > total_params * self.recompute_ratio / 10:
                        layers.append(name)
                
                return layers
            
            def forward(self, *args, **kwargs):
                # 实现选择性激活重计算
                return self.module(*args, **kwargs)
        
        return RecomputationModule(model)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            peak_memory = current_memory
        
        self.memory_stats.update({
            'current_memory': current_memory,
            'peak_memory': max(peak_memory, self.memory_stats['peak_memory']),
        })
        
        return self.memory_stats.copy()
    
    def clear_memory_pools(self):
        """清空内存池"""
        
        for pool_type in self.memory_pools:
            self.memory_pools[pool_type].clear()
        
        # 强制垃圾回收
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory pools cleared")

class CommunicationOptimizer(BaseOptimizer):
    """通信优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
        # 通信配置
        self.world_size = 1
        self.rank = 0
        
        # 通信组
        self.process_groups = {}
        
        # 通信统计
        self.comm_stats = {
            'total_comm_time': 0.0,
            'total_comm_volume': 0,
            'avg_bandwidth': 0.0,
        }
        
        logger.info("Initialized CommunicationOptimizer")
    
    def get_optimization_type(self) -> OptimizationType:
        return OptimizationType.COMMUNICATION
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行通信优化"""
        
        optimized_model = model
        
        # 1. 张量并行优化
        if self.config.use_tensor_parallelism:
            optimized_model = self._apply_tensor_parallelism(optimized_model)
        
        # 2. 流水线并行优化
        if self.config.use_pipeline_parallelism:
            optimized_model = self._apply_pipeline_parallelism(optimized_model)
        
        # 3. 通信压缩优化
        optimized_model = self._apply_communication_compression(optimized_model)
        
        # 4. 异步通信优化
        optimized_model = self._apply_async_communication(optimized_model)
        
        logger.info("Applied communication optimizations")
        return optimized_model
    
    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """应用张量并行优化"""
        
        class TensorParallelLinear(nn.Module):
            """张量并行线性层"""
            
            def __init__(self, original_linear: nn.Linear, world_size: int, rank: int):
                super().__init__()
                
                self.world_size = world_size
                self.rank = rank
                
                # 分割权重
                original_weight = original_linear.weight
                input_dim, output_dim = original_weight.shape
                
                # 按输出维度分割
                chunk_size = output_dim // world_size
                start_idx = rank * chunk_size
                end_idx = (rank + 1) * chunk_size if rank < world_size - 1 else output_dim
                
                self.weight = nn.Parameter(original_weight[start_idx:end_idx, :])
                
                if original_linear.bias is not None:
                    self.bias = nn.Parameter(original_linear.bias[start_idx:end_idx])
                else:
                    self.bias = None
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 本地计算
                local_output = F.linear(x, self.weight, self.bias)
                
                # 全收集结果
                if self.world_size > 1:
                    output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
                    torch.distributed.all_gather(output_list, local_output)
                    output = torch.cat(output_list, dim=-1)
                else:
                    output = local_output
                
                return output
        
        # 替换线性层为张量并行版本
        def replace_linear_layers(module: nn.Module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # 替换为张量并行层
                    tp_layer = TensorParallelLinear(child, self.world_size, self.rank)
                    setattr(module, name, tp_layer)
                else:
                    # 递归处理子模块
                    replace_linear_layers(child)
        
        if self.world_size > 1:
            replace_linear_layers(model)
        
        return model
    
    def _apply_pipeline_parallelism(self, model: nn.Module) -> nn.Module:
        """应用流水线并行优化"""
        
        class PipelineStage(nn.Module):
            """流水线阶段"""
            
            def __init__(self, layers: nn.ModuleList, stage_id: int):
                super().__init__()
                self.layers = layers
                self.stage_id = stage_id
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x
        
        class PipelineParallelModel(nn.Module):
            """流水线并行模型"""
            
            def __init__(self, original_model: nn.Module, num_stages: int):
                super().__init__()
                
                self.num_stages = num_stages
                
                # 分割模型为多个阶段
                self.stages = self._split_model_into_stages(original_model, num_stages)
            
            def _split_model_into_stages(self, model: nn.Module, num_stages: int) -> nn.ModuleList:
                """将模型分割为多个阶段"""
                
                # 获取所有层
                layers = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # 叶子节点
                        layers.append(module)
                
                # 按层数平均分割
                layers_per_stage = len(layers) // num_stages
                stages = nn.ModuleList()
                
                for i in range(num_stages):
                    start_idx = i * layers_per_stage
                    end_idx = (i + 1) * layers_per_stage if i < num_stages - 1 else len(layers)
                    
                    stage_layers = nn.ModuleList(layers[start_idx:end_idx])
                    stage = PipelineStage(stage_layers, i)
                    stages.append(stage)
                
                return stages
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 顺序执行各个阶段
                for stage in self.stages:
                    x = stage(x)
                return x
        
        if self.world_size > 1:
            return PipelineParallelModel(model, self.world_size)
        
        return model
    
    def _apply_communication_compression(self, model: nn.Module) -> nn.Module:
        """应用通信压缩优化"""
        
        class CompressedCommunication:
            """压缩通信"""
            
            @staticmethod
            def compress_tensor(tensor: torch.Tensor, compression_ratio: float = 0.1) -> Tuple[torch.Tensor, Dict]:
                """压缩张量"""
                
                # 简单的Top-K压缩
                numel = tensor.numel()
                k = int(numel * compression_ratio)
                
                # 获取Top-K值和索引
                flat_tensor = tensor.flatten()
                top_values, top_indices = torch.topk(torch.abs(flat_tensor), k)
                
                # 保留符号
                compressed_values = torch.gather(flat_tensor, 0, top_indices)
                
                metadata = {
                    'shape': tensor.shape,
                    'indices': top_indices,
                    'compression_ratio': compression_ratio,
                }
                
                return compressed_values, metadata
            
            @staticmethod
            def decompress_tensor(compressed_values: torch.Tensor, metadata: Dict) -> torch.Tensor:
                """解压缩张量"""
                
                # 重建张量
                shape = metadata['shape']
                indices = metadata['indices']
                
                # 创建零张量
                decompressed = torch.zeros(torch.prod(torch.tensor(shape)), 
                                         dtype=compressed_values.dtype,
                                         device=compressed_values.device)
                
                # 填充压缩值
                decompressed[indices] = compressed_values
                
                return decompressed.reshape(shape)
        
        # 为模型添加压缩通信功能
        model.compressed_comm = CompressedCommunication()
        
        return model
    
    def _apply_async_communication(self, model: nn.Module) -> nn.Module:
        """应用异步通信优化"""
        
        class AsyncCommunicationModule(nn.Module):
            """异步通信模块"""
            
            def __init__(self, original_model: nn.Module):
                super().__init__()
                self.model = original_model
                
                # 异步通信队列
                self.comm_queue = []
                self.comm_futures = []
            
            def forward(self, *args, **kwargs):
                # 启动异步通信
                self._start_async_communication()
                
                # 执行计算
                output = self.model(*args, **kwargs)
                
                # 等待通信完成
                self._wait_communication()
                
                return output
            
            def _start_async_communication(self):
                """启动异步通信"""
                
                # 这里可以启动预取、预发送等异步操作
                pass
            
            def _wait_communication(self):
                """等待通信完成"""
                
                # 等待所有异步通信完成
                for future in self.comm_futures:
                    future.wait()
                
                self.comm_futures.clear()
        
        return AsyncCommunicationModule(model)

class SystemOptimizer(BaseOptimizer):
    """系统级优化器"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        
        # 系统资源监控
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.cpu_count)
        
        # 系统统计
        self.system_stats = {
            'cpu_utilization': 0.0,
            'memory_utilization': 0.0,
            'io_wait': 0.0,
            'context_switches': 0,
        }
        
        logger.info("Initialized SystemOptimizer")
    
    def get_optimization_type(self) -> OptimizationType:
        return OptimizationType.SYSTEM
    
    def optimize(self, model: nn.Module, **kwargs) -> nn.Module:
        """执行系统级优化"""
        
        optimized_model = model
        
        # 1. 异步处理优化
        if self.config.use_async_processing:
            optimized_model = self._apply_async_processing(optimized_model)
        
        # 2. 批量优化
        if self.config.use_batch_optimization:
            optimized_model = self._apply_batch_optimization(optimized_model)
        
        # 3. 预取优化
        optimized_model = self._apply_prefetching(optimized_model)
        
        # 4. 资源调度优化
        optimized_model = self._apply_resource_scheduling(optimized_model)
        
        logger.info("Applied system optimizations")
        return optimized_model
    
    def _apply_async_processing(self, model: nn.Module) -> nn.Module:
        """应用异步处理优化"""
        
        class AsyncProcessingModule(nn.Module):
            """异步处理模块"""
            
            def __init__(self, original_model: nn.Module, thread_pool: ThreadPoolExecutor):
                super().__init__()
                self.model = original_model
                self.thread_pool = thread_pool
                
                # 异步任务队列
                self.pending_tasks = []
            
            def forward_async(self, *args, **kwargs):
                """异步前向传播"""
                
                future = self.thread_pool.submit(self.model, *args, **kwargs)
                self.pending_tasks.append(future)
                
                return future
            
            def forward(self, *args, **kwargs):
                """同步前向传播"""
                return self.model(*args, **kwargs)
            
            def wait_all_tasks(self):
                """等待所有异步任务完成"""
                
                results = []
                for task in self.pending_tasks:
                    results.append(task.result())
                
                self.pending_tasks.clear()
                return results
        
        return AsyncProcessingModule(model, self.thread_pool)
    
    def _apply_batch_optimization(self, model: nn.Module) -> nn.Module:
        """应用批量优化"""
        
        class BatchOptimizedModule(nn.Module):
            """批量优化模块"""
            
            def __init__(self, original_model: nn.Module):
                super().__init__()
                self.model = original_model
                
                # 批量配置
                self.optimal_batch_size = self._determine_optimal_batch_size()
                self.batch_buffer = []
            
            def _determine_optimal_batch_size(self) -> int:
                """确定最优批量大小"""
                
                # 基于可用内存和模型大小估算
                model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
                available_memory = psutil.virtual_memory().available
                
                # 保守估计：使用可用内存的1/4
                max_batch_memory = available_memory // 4
                estimated_batch_size = max_batch_memory // (model_memory * 10)  # 10倍安全系数
                
                return max(1, min(64, estimated_batch_size))  # 限制在1-64之间
            
            def add_to_batch(self, *args, **kwargs):
                """添加到批量缓冲区"""
                
                self.batch_buffer.append((args, kwargs))
                
                if len(self.batch_buffer) >= self.optimal_batch_size:
                    return self.process_batch()
                
                return None
            
            def process_batch(self):
                """处理批量数据"""
                
                if not self.batch_buffer:
                    return []
                
                # 合并批量输入
                batch_args, batch_kwargs = self._merge_batch_inputs()
                
                # 批量处理
                batch_output = self.model(*batch_args, **batch_kwargs)
                
                # 分解批量输出
                outputs = self._split_batch_outputs(batch_output)
                
                # 清空缓冲区
                self.batch_buffer.clear()
                
                return outputs
            
            def _merge_batch_inputs(self):
                """合并批量输入"""
                
                # 简化实现：假设所有输入都是张量
                merged_args = []
                merged_kwargs = {}
                
                if self.batch_buffer:
                    first_args, first_kwargs = self.batch_buffer[0]
                    
                    # 合并位置参数
                    for i in range(len(first_args)):
                        arg_list = [item[0][i] for item in self.batch_buffer]
                        if isinstance(arg_list[0], torch.Tensor):
                            merged_arg = torch.stack(arg_list, dim=0)
                            merged_args.append(merged_arg)
                        else:
                            merged_args.append(arg_list)
                    
                    # 合并关键字参数
                    for key in first_kwargs:
                        value_list = [item[1][key] for item in self.batch_buffer]
                        if isinstance(value_list[0], torch.Tensor):
                            merged_kwargs[key] = torch.stack(value_list, dim=0)
                        else:
                            merged_kwargs[key] = value_list
                
                return tuple(merged_args), merged_kwargs
            
            def _split_batch_outputs(self, batch_output):
                """分解批量输出"""
                
                if isinstance(batch_output, torch.Tensor):
                    return [batch_output[i] for i in range(batch_output.size(0))]
                else:
                    return [batch_output] * len(self.batch_buffer)
            
            def forward(self, *args, **kwargs):
                """前向传播"""
                return self.model(*args, **kwargs)
        
        return BatchOptimizedModule(model)
    
    def _apply_prefetching(self, model: nn.Module) -> nn.Module:
        """应用预取优化"""
        
        class PrefetchingModule(nn.Module):
            """预取模块"""
            
            def __init__(self, original_model: nn.Module, prefetch_factor: int = 2):
                super().__init__()
                self.model = original_model
                self.prefetch_factor = prefetch_factor
                
                # 预取缓存
                self.prefetch_cache = {}
                self.cache_lock = threading.Lock()
            
            def prefetch_data(self, data_loader, num_batches: int = None):
                """预取数据"""
                
                if num_batches is None:
                    num_batches = self.prefetch_factor
                
                def prefetch_worker():
                    for i, batch in enumerate(data_loader):
                        if i >= num_batches:
                            break
                        
                        with self.cache_lock:
                            self.prefetch_cache[i] = batch
                
                # 启动预取线程
                prefetch_thread = threading.Thread(target=prefetch_worker)
                prefetch_thread.daemon = True
                prefetch_thread.start()
            
            def get_prefetched_batch(self, batch_idx: int):
                """获取预取的批次"""
                
                with self.cache_lock:
                    return self.prefetch_cache.pop(batch_idx, None)
            
            def forward(self, *args, **kwargs):
                """前向传播"""
                return self.model(*args, **kwargs)
        
        return PrefetchingModule(model, self.config.prefetch_factor)
    
    def _apply_resource_scheduling(self, model: nn.Module) -> nn.Module:
        """应用资源调度优化"""
        
        class ResourceScheduledModule(nn.Module):
            """资源调度模块"""
            
            def __init__(self, original_model: nn.Module):
                super().__init__()
                self.model = original_model
                
                # 资源监控
                self.resource_monitor = ResourceMonitor()
                
                # 调度策略
                self.scheduling_policy = 'adaptive'
            
            def forward(self, *args, **kwargs):
                """前向传播"""
                
                # 监控资源使用
                resource_info = self.resource_monitor.get_current_usage()
                
                # 根据资源情况调整执行策略
                if resource_info['cpu_usage'] > 80:
                    # CPU使用率过高，减少并行度
                    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
                
                if resource_info['memory_usage'] > 80:
                    # 内存使用率过高，启用内存优化
                    with torch.cuda.amp.autocast():
                        output = self.model(*args, **kwargs)
                else:
                    output = self.model(*args, **kwargs)
                
                return output
        
        return ResourceScheduledModule(model)
    
    def get_system_stats(self) -> Dict[str, float]:
        """获取系统统计"""
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        self.system_stats.update({
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory_info.percent,
            'available_memory_gb': memory_info.available / 1024 / 1024 / 1024,
        })
        
        return self.system_stats.copy()

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.stats_history = []
        
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        
        self.monitoring = True
        
        def monitor_worker():
            while self.monitoring:
                stats = self.get_current_usage()
                self.stats_history.append({
                    'timestamp': time.time(),
                    **stats
                })
                
                # 限制历史记录长度
                if len(self.stats_history) > 1000:
                    self.stats_history = self.stats_history[-500:]
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_current_usage(self) -> Dict[str, float]:
        """获取当前资源使用情况"""
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # 内存使用情况
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # GPU使用情况（如果可用）
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_usage = (gpu_memory_used / gpu_memory_total) * 100
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'gpu_usage': gpu_usage,
            'gpu_memory_usage': gpu_memory_usage,
        }
    
    def get_stats_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        
        if not self.stats_history:
            return {}
        
        cpu_usages = [stat['cpu_usage'] for stat in self.stats_history]
        memory_usages = [stat['memory_usage'] for stat in self.stats_history]
        
        return {
            'avg_cpu_usage': np.mean(cpu_usages),
            'max_cpu_usage': np.max(cpu_usages),
            'avg_memory_usage': np.mean(memory_usages),
            'max_memory_usage': np.max(memory_usages),
        }

# 性能分析器
class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.profiling_data = {}
        self.profiling_active = False
        
    @contextmanager
    def profile(self, operation_name: str):
        """性能分析上下文管理器"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # 记录性能数据
            self.profiling_data[operation_name] = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': start_time,
            }
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """获取性能分析摘要"""
        
        if not self.profiling_data:
            return {}
        
        total_time = sum(data['duration'] for data in self.profiling_data.values())
        
        summary = {
            'total_operations': len(self.profiling_data),
            'total_time': total_time,
            'operations': {}
        }
        
        for op_name, data in self.profiling_data.items():
            summary['operations'][op_name] = {
                'duration': data['duration'],
                'memory_delta': data['memory_delta'],
                'percentage': (data['duration'] / total_time) * 100,
            }
        
        return summary
    
    def clear_profile_data(self):
        """清空性能分析数据"""
        self.profiling_data.clear()

# 使用示例
def example_optimization_usage():
    """性能优化使用示例"""
    
    # 创建优化配置
    optimization_config = OptimizationConfig(
        enable_compute_optimization=True,
        enable_memory_optimization=True,
        enable_communication_optimization=False,  # 单机环境
        enable_system_optimization=True,
        use_flash_attention=True,
        use_fused_kernels=True,
        use_mixed_precision=True,
        use_kv_cache=True,
        use_memory_pool=True,
        use_async_processing=True,
        use_batch_optimization=True,
    )
    
    # 创建优化器
    compute_optimizer = ComputeOptimizer(optimization_config)
    memory_optimizer = MemoryOptimizer(optimization_config)
    system_optimizer = SystemOptimizer(optimization_config)
    
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 1024)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(1024, 512)
            self.layer_norm = nn.LayerNorm(512)
        
        def forward(self, x):
            x = self.linear1(x)
            x = self.gelu(x)
            x = self.linear2(x)
            x = self.layer_norm(x)
            return x
    
    model = SimpleModel()
    
    # 应用优化
    print("Applying optimizations...")
    
    # 计算优化
    model = compute_optimizer.optimize(model)
    print("✓ Compute optimization applied")
    
    # 内存优化
    model = memory_optimizer.optimize(model)
    print("✓ Memory optimization applied")
    
    # 系统优化
    model = system_optimizer.optimize(model)
    print("✓ System optimization applied")
    
    # 性能测试
    profiler = PerformanceProfiler()
    
    # 测试数据
    batch_size = 32
    seq_len = 128
    hidden_dim = 512
    
    input_data = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 性能分析
    with profiler.profile("model_forward"):
        output = model(input_data)
    
    # 获取性能摘要
    profile_summary = profiler.get_profile_summary()
    print(f"\nPerformance Summary:")
    print(f"Model forward time: {profile_summary['operations']['model_forward']['duration']:.4f}s")
    
    # 获取内存统计
    memory_stats = memory_optimizer.get_memory_stats()
    print(f"Memory usage: {memory_stats['current_memory']:.2f} MB")
    print(f"Pool hits: {memory_stats['pool_hits']}")
    
    # 获取系统统计
    system_stats = system_optimizer.get_system_stats()
    print(f"CPU utilization: {system_stats['cpu_utilization']:.1f}%")
    print(f"Memory utilization: {system_stats['memory_utilization']:.1f}%")

if __name__ == "__main__":
    example_optimization_usage()
```

## 🔧 关键优化技术分析

### 1. 计算优化技术

- **算子融合**：将多个连续操作合并为单个算子
- **混合精度**：使用FP16/BF16减少计算量和内存使用
- **Flash Attention**：优化的注意力机制实现
- **编译优化**：使用torch.compile或TorchScript优化

### 2. 内存优化技术

- **梯度检查点**：用计算换内存的策略
- **内存池**：重用张量减少内存分配开销
- **KV缓存优化**：高效的键值缓存管理
- **激活重计算**：选择性重计算减少内存占用

### 3. 通信优化技术

- **张量并行**：将大张量分割到多个设备
- **流水线并行**：将模型分层到多个设备
- **通信压缩**：减少通信数据量
- **异步通信**：重叠计算和通信

### 4. 系统优化技术

- **异步处理**：并行执行多个任务
- **批量优化**：动态调整批量大小
- **预取机制**：提前加载数据
- **资源调度**：智能分配系统资源

## 📊 性能监控和调优

### 性能指标监控

```python
class AdvancedPerformanceMonitor:
    """高级性能监控器"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'latency': 1.0,      # 1秒
            'memory_usage': 80,   # 80%
            'throughput': 10,     # 10 tokens/s
        }
        
    def collect_metrics(self, model_output, execution_time: float) -> PerformanceMetrics:
        """收集性能指标"""
        
        # 计算吞吐量
        num_tokens = model_output.numel() if hasattr(model_output, 'numel') else 0
        throughput = num_tokens / execution_time if execution_time > 0 else 0
        
        # 获取内存使用
        memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # 获取GPU利用率
        gpu_utilization = self._get_gpu_utilization()
        
        # 获取CPU利用率
        cpu_utilization = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            throughput=throughput,
            latency=execution_time,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
        )
        
        self.metrics_history.append(metrics)
        
        # 检查告警
        self._check_alerts(metrics)
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """检查性能告警"""
        
        if metrics.latency > self.alert_thresholds['latency']:
            logger.warning(f"High latency detected: {metrics.latency:.3f}s")
        
        if metrics.memory_usage > self.alert_thresholds['memory_usage']:
            logger.warning(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if metrics.throughput < self.alert_thresholds['throughput']:
            logger.warning(f"Low throughput: {metrics.throughput:.1f} tokens/s")
    
    def get_performance_trends(self) -> Dict[str, List[float]]:
        """获取性能趋势"""
        
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # 最近100个数据点
        
        return {
            'throughput_trend': [m.throughput for m in recent_metrics],
            'latency_trend': [m.latency for m in recent_metrics],
            'memory_trend': [m.memory_usage for m in recent_metrics],
        }
    
    def suggest_optimizations(self) -> List[str]:
        """建议优化措施"""
        
        if not self.metrics_history:
            return []
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个数据点
        avg_metrics = PerformanceMetrics(
            throughput=np.mean([m.throughput for m in recent_metrics]),
            latency=np.mean([m.latency for m in recent_metrics]),
            memory_usage=np.mean([m.memory_usage for m in recent_metrics]),
            gpu_utilization=np.mean([m.gpu_utilization for m in recent_metrics]),
            cpu_utilization=np.mean([m.cpu_utilization for m in recent_metrics]),
        )
        
        suggestions = []
        
        # 基于性能指标给出建议
        if avg_metrics.latency > 0.5:
            suggestions.append("Consider using faster sampling strategies")
            suggestions.append("Enable mixed precision training")
        
        if avg_metrics.memory_usage > 70:
            suggestions.append("Enable gradient checkpointing")
            suggestions.append("Reduce batch size")
            suggestions.append("Use memory pooling")
        
        if avg_metrics.throughput < 20:
            suggestions.append("Increase batch size if memory allows")
            suggestions.append("Use tensor parallelism")
            suggestions.append("Enable kernel fusion")
        
        if avg_metrics.gpu_utilization < 50:
            suggestions.append("Increase model complexity or batch size")
            suggestions.append("Check for CPU bottlenecks")
        
        if avg_metrics.cpu_utilization > 80:
            suggestions.append("Reduce CPU preprocessing")
            suggestions.append("Use GPU for data loading")
        
        return suggestions
```

### 自动调优系统

```python
class AutoTuner:
    """自动调优系统"""
    
    def __init__(self, optimization_config: OptimizationConfig):
        self.config = optimization_config
        self.performance_monitor = AdvancedPerformanceMonitor()
        self.tuning_history = []
        
        # 调优参数空间
        self.param_space = {
            'batch_size': [1, 2, 4, 8, 16, 32, 64],
            'max_seq_len': [128, 256, 512, 1024, 2048],
            'use_mixed_precision': [True, False],
            'use_flash_attention': [True, False],
            'num_threads': [1, 2, 4, 8, 16],
        }
        
        # 当前最佳配置
        self.best_config = None
        self.best_performance = None
    
    def auto_tune(self, model: nn.Module, test_data: torch.Tensor, max_trials: int = 50):
        """自动调优"""
        
        logger.info(f"Starting auto-tuning with {max_trials} trials")
        
        for trial in range(max_trials):
            # 生成随机配置
            trial_config = self._generate_trial_config()
            
            # 测试配置
            performance = self._evaluate_config(model, test_data, trial_config)
            
            # 更新最佳配置
            if self._is_better_performance(performance, self.best_performance):
                self.best_config = trial_config.copy()
                self.best_performance = performance
                
                logger.info(f"Trial {trial}: New best performance - "
                          f"Throughput: {performance.throughput:.2f}, "
                          f"Latency: {performance.latency:.4f}")
            
            # 记录调优历史
            self.tuning_history.append({
                'trial': trial,
                'config': trial_config,
                'performance': performance,
            })
        
        logger.info(f"Auto-tuning completed. Best throughput: {self.best_performance.throughput:.2f}")
        return self.best_config, self.best_performance
    
    def _generate_trial_config(self) -> Dict[str, Any]:
        """生成试验配置"""
        
        config = {}
        
        for param, values in self.param_space.items():
            config[param] = np.random.choice(values)
        
        return config
    
    def _evaluate_config(
        self, 
        model: nn.Module, 
        test_data: torch.Tensor, 
        config: Dict[str, Any]
    ) -> PerformanceMetrics:
        """评估配置性能"""
        
        # 应用配置
        original_threads = torch.get_num_threads()
        torch.set_num_threads(config['num_threads'])
        
        try:
            # 执行测试
            start_time = time.time()
            
            with torch.no_grad():
                if config['use_mixed_precision']:
                    with torch.cuda.amp.autocast():
                        output = model(test_data)
                else:
                    output = model(test_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 收集性能指标
            performance = self.performance_monitor.collect_metrics(output, execution_time)
            
            return performance
            
        except Exception as e:
            logger.warning(f"Config evaluation failed: {e}")
            # 返回最差性能
            return PerformanceMetrics(
                throughput=0.0,
                latency=float('inf'),
                memory_usage=100.0,
            )
        
        finally:
            # 恢复原始设置
            torch.set_num_threads(original_threads)
    
    def _is_better_performance(
        self, 
        new_perf: PerformanceMetrics, 
        best_perf: Optional[PerformanceMetrics]
    ) -> bool:
        """判断性能是否更好"""
        
        if best_perf is None:
            return True
        
        # 综合评分：吞吐量权重0.5，延迟权重0.3，内存权重0.2
        new_score = (new_perf.throughput * 0.5 - 
                    new_perf.latency * 0.3 - 
                    new_perf.memory_usage * 0.002)
        
        best_score = (best_perf.throughput * 0.5 - 
                     best_perf.latency * 0.3 - 
                     best_perf.memory_usage * 0.002)
        
        return new_score > best_score
    
    def get_tuning_report(self) -> Dict[str, Any]:
        """获取调优报告"""
        
        if not self.tuning_history:
            return {}
        
        # 统计信息
        throughputs = [h['performance'].throughput for h in self.tuning_history]
        latencies = [h['performance'].latency for h in self.tuning_history]
        
        report = {
            'total_trials': len(self.tuning_history),
            'best_config': self.best_config,
            'best_performance': self.best_performance.to_dict() if self.best_performance else None,
            'performance_stats': {
                'avg_throughput': np.mean(throughputs),
                'max_throughput': np.max(throughputs),
                'avg_latency': np.mean(latencies),
                'min_latency': np.min(latencies),
            },
            'parameter_analysis': self._analyze_parameters(),
        }
        
        return report
    
    def _analyze_parameters(self) -> Dict[str, Any]:
        """分析参数影响"""
        
        param_analysis = {}
        
        for param in self.param_space.keys():
            param_values = []
            param_performances = []
            
            for history in self.tuning_history:
                param_values.append(history['config'][param])
                param_performances.append(history['performance'].throughput)
            
            # 计算参数与性能的相关性
            if len(set(param_values)) > 1:
                correlation = np.corrcoef(param_values, param_performances)[0, 1]
                param_analysis[param] = {
                    'correlation_with_throughput': correlation,
                    'best_value': self.best_config[param] if self.best_config else None,
                }
        
        return param_analysis
```

## 🎯 最佳实践和建议

### 1. 优化策略选择

- **计算密集型任务**：优先使用计算优化（Flash Attention、算子融合）
- **内存受限场景**：重点关注内存优化（梯度检查点、内存池）
- **多设备部署**：考虑通信优化（张量并行、流水线并行）
- **系统瓶颈**：应用系统优化（异步处理、批量优化）

### 2. 性能监控要点

- **关键指标**：吞吐量、延迟、内存使用率、GPU利用率
- **监控频率**：实时监控关键指标，定期分析趋势
- **告警机制**：设置合理的性能告警阈值
- **自动调优**：使用自动调优系统持续优化性能

### 3. 调试和故障排除

- **性能分析**：使用profiler定位性能瓶颈
- **内存分析**：监控内存泄漏和峰值使用
- **错误处理**：实现robust的错误恢复机制
- **日志记录**：详细记录优化过程和效果

## 📈 总结

nano-vllm的性能优化是一个多层次、多维度的系统工程，涵盖了从底层算子优化到系统级资源调度的各个方面。通过合理选择和组合各种优化技术，可以显著提升模型推理的效率和用户体验。

关键要点：
1. **分层优化**：从计算、内存、通信、系统四个层面系统性优化
2. **动态调整**：根据实际负载和资源情况动态调整优化策略
3. **持续监控**：建立完善的性能监控和自动调优机制
4. **平衡权衡**：在性能、内存、复杂度之间找到最佳平衡点

通过深入理解和应用这些优化技术，开发者可以构建出高性能、高效率的大语言模型推理系统。