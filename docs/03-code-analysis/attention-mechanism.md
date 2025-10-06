# 注意力机制分析

## 🎯 注意力机制概览

注意力机制是 Transformer 模型的核心组件，也是 nano-vllm 中最重要的计算模块之一。本文档深入分析 nano-vllm 中注意力机制的实现，包括标准注意力、优化版本（如 Flash Attention）、KV 缓存管理等关键技术。

## 🏗️ 核心架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from nano_vllm.kernels import flash_attention, paged_attention
from nano_vllm.memory import KVCache, PagedKVCache
from nano_vllm.utils.tensor_parallel import tensor_parallel_linear
from nano_vllm.config import AttentionConfig

logger = logging.getLogger(__name__)

class AttentionBackend(Enum):
    """注意力后端类型"""
    TORCH = "torch"
    FLASH_ATTENTION = "flash_attention"
    PAGED_ATTENTION = "paged_attention"
    XFORMERS = "xformers"

@dataclass
class AttentionMetadata:
    """注意力元数据"""
    seq_lens: List[int]
    max_seq_len: int
    num_prefill_tokens: int
    num_decode_tokens: int
    slot_mapping: torch.Tensor
    context_lens: List[int]
    block_tables: Optional[torch.Tensor] = None
    use_cuda_graph: bool = False

class BaseAttention(nn.Module):
    """基础注意力模块"""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        num_kv_heads: Optional[int] = None,
        sliding_window: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = num_heads * head_dim
        self.kv_hidden_size = self.num_kv_heads * head_dim
        
        # 缩放因子
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        
        # 滑动窗口注意力
        self.sliding_window = sliding_window
        
        # ALiBi位置编码
        self.alibi_slopes = alibi_slopes
        
        # 缓存配置
        self.cache_config = cache_config or {}
        
        logger.info(f"Initialized attention: heads={num_heads}, head_dim={head_dim}, "
                   f"kv_heads={self.num_kv_heads}, sliding_window={sliding_window}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """前向传播"""
        raise NotImplementedError

class TorchAttention(BaseAttention):
    """PyTorch原生注意力实现"""
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, seq_len, num_heads, head_dim]
            key: [batch_size, seq_len, num_kv_heads, head_dim]
            value: [batch_size, seq_len, num_kv_heads, head_dim]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = query.shape[:2]
        
        # 重塑张量形状
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 处理KV缓存
        if kv_cache is not None:
            key, value = self._update_kv_cache(key, value, kv_cache, attn_metadata)
        
        # 扩展KV头以匹配查询头数
        if self.num_kv_heads != self.num_heads:
            key = self._repeat_kv(key, self.num_queries_per_kv)
            value = self._repeat_kv(value, self.num_queries_per_kv)
        
        # 转置以适应注意力计算
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch, num_heads, kv_len, head_dim]
        value = value.transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用位置编码
        if self.alibi_slopes is not None:
            attn_scores = self._apply_alibi(attn_scores, seq_len)
        
        # 应用注意力掩码
        attn_mask = self._create_attention_mask(seq_len, key.size(-2), query.device)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        # 应用滑动窗口
        if self.sliding_window is not None:
            attn_scores = self._apply_sliding_window(attn_scores, self.sliding_window)
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 应用dropout（训练时）
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)
        
        # 计算输出
        output = torch.matmul(attn_weights, value)
        
        # 重塑输出形状
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        
        return output
    
    def _repeat_kv(self, tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
        """重复KV张量以匹配查询头数"""
        batch, num_kv_heads, seq_len, head_dim = tensor.shape
        if n_rep == 1:
            return tensor
        
        tensor = tensor[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return tensor.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def _update_kv_cache(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新KV缓存"""
        if attn_metadata is None:
            return key, value
        
        # 获取缓存的KV
        cached_key, cached_value = kv_cache.get_kv_cache()
        
        if cached_key is not None and cached_value is not None:
            # 拼接新的KV
            key = torch.cat([cached_key, key], dim=1)
            value = torch.cat([cached_value, value], dim=1)
        
        # 更新缓存
        kv_cache.update(key, value, attn_metadata.slot_mapping)
        
        return key, value
    
    def _create_attention_mask(
        self, 
        seq_len: int, 
        kv_len: int, 
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """创建注意力掩码"""
        if seq_len == 1:
            # 解码阶段，不需要掩码
            return None
        
        # 创建因果掩码
        mask = torch.triu(
            torch.full((seq_len, kv_len), float('-inf'), device=device),
            diagonal=kv_len - seq_len + 1
        )
        
        return mask
    
    def _apply_alibi(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """应用ALiBi位置编码"""
        if self.alibi_slopes is None:
            return attn_scores
        
        batch_size, num_heads = attn_scores.shape[:2]
        
        # 创建位置偏置
        position_ids = torch.arange(seq_len, device=attn_scores.device)
        relative_pos = position_ids[None, :] - position_ids[:, None]
        
        # 应用ALiBi斜率
        alibi_bias = torch.zeros_like(attn_scores)
        for i, slope in enumerate(self.alibi_slopes[:num_heads]):
            alibi_bias[:, i] = relative_pos * slope
        
        return attn_scores + alibi_bias
    
    def _apply_sliding_window(
        self, 
        attn_scores: torch.Tensor, 
        window_size: int
    ) -> torch.Tensor:
        """应用滑动窗口注意力"""
        seq_len = attn_scores.size(-2)
        kv_len = attn_scores.size(-1)
        
        # 创建滑动窗口掩码
        mask = torch.triu(
            torch.full((seq_len, kv_len), float('-inf'), device=attn_scores.device),
            diagonal=window_size + 1
        )
        
        return attn_scores + mask

class FlashAttention(BaseAttention):
    """Flash Attention实现"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 检查Flash Attention可用性
        try:
            import flash_attn
            self.flash_attn_available = True
            logger.info("Flash Attention is available")
        except ImportError:
            self.flash_attn_available = False
            logger.warning("Flash Attention not available, falling back to torch implementation")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """Flash Attention前向传播"""
        
        if not self.flash_attn_available:
            # 回退到PyTorch实现
            torch_attn = TorchAttention(
                self.num_heads, self.head_dim, self.scale,
                self.num_kv_heads, self.sliding_window, self.alibi_slopes
            )
            return torch_attn.forward(query, key, value, kv_cache, attn_metadata)
        
        batch_size, seq_len = query.shape[:2]
        
        # 重塑张量
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 处理KV缓存
        if kv_cache is not None:
            key, value = self._update_kv_cache(key, value, kv_cache, attn_metadata)
        
        # 调用Flash Attention内核
        output = self._flash_attention_forward(query, key, value, attn_metadata)
        
        return output.view(batch_size, seq_len, self.hidden_size)
    
    def _flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """Flash Attention内核调用"""
        from flash_attn import flash_attn_func
        
        # Flash Attention参数
        dropout_p = 0.0 if not self.training else 0.1
        causal = True  # 因果注意力
        
        # 处理滑动窗口
        window_size = (-1, -1)  # 默认无限窗口
        if self.sliding_window is not None:
            window_size = (-1, self.sliding_window)
        
        # 调用Flash Attention
        output = flash_attn_func(
            query, key, value,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=self.alibi_slopes,
            return_attn_probs=False
        )
        
        return output

class PagedAttention(BaseAttention):
    """分页注意力实现"""
    
    def __init__(self, *args, block_size: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        
        logger.info(f"Initialized PagedAttention with block_size={block_size}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[PagedKVCache] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """分页注意力前向传播"""
        
        if kv_cache is None or attn_metadata is None:
            # 回退到标准注意力
            torch_attn = TorchAttention(
                self.num_heads, self.head_dim, self.scale,
                self.num_kv_heads, self.sliding_window, self.alibi_slopes
            )
            return torch_attn.forward(query, key, value, kv_cache, attn_metadata)
        
        batch_size, seq_len = query.shape[:2]
        
        # 重塑查询张量
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 分离预填充和解码
        if attn_metadata.num_prefill_tokens > 0:
            # 预填充阶段
            output = self._paged_prefill_attention(
                query, key, value, kv_cache, attn_metadata
            )
        else:
            # 解码阶段
            output = self._paged_decode_attention(
                query, kv_cache, attn_metadata
            )
        
        return output.view(batch_size, seq_len, self.hidden_size)
    
    def _paged_prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: PagedKVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """分页预填充注意力"""
        
        # 更新KV缓存
        kv_cache.update_prefill(key, value, attn_metadata.slot_mapping)
        
        # 使用标准注意力进行预填充
        torch_attn = TorchAttention(
            self.num_heads, self.head_dim, self.scale,
            self.num_kv_heads, self.sliding_window, self.alibi_slopes
        )
        
        return torch_attn.forward(query, key, value, None, attn_metadata)
    
    def _paged_decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: PagedKVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """分页解码注意力"""
        
        # 调用分页注意力内核
        output = paged_attention.paged_attention_v1(
            query=query,
            key_cache=kv_cache.key_cache,
            value_cache=kv_cache.value_cache,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            block_tables=attn_metadata.block_tables,
            context_lens=torch.tensor(attn_metadata.context_lens, device=query.device),
            block_size=self.block_size,
            max_context_len=attn_metadata.max_seq_len,
            alibi_slopes=self.alibi_slopes,
        )
        
        return output

class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        bias: bool = True,
        sliding_window: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        attention_backend: str = "torch",
        cache_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # 线性投影层
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)
        
        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_config=rope_scaling,
        )
        
        # 注意力实现
        self.attention_backend = attention_backend
        if attention_backend == "flash_attention":
            self.attn = FlashAttention(
                num_heads=num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                sliding_window=sliding_window,
                cache_config=cache_config,
            )
        elif attention_backend == "paged_attention":
            self.attn = PagedAttention(
                num_heads=num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                sliding_window=sliding_window,
                block_size=cache_config.get("block_size", 16) if cache_config else 16,
            )
        else:
            self.attn = TorchAttention(
                num_heads=num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                sliding_window=sliding_window,
            )
        
        logger.info(f"Initialized MultiHeadAttention with backend: {attention_backend}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """前向传播"""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑形状
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        if position_ids is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max().item() + 1)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
        
        # 注意力计算
        attn_output = self.attn(
            query=query_states,
            key=key_states,
            value=value_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output

class RotaryEmbedding(nn.Module):
    """旋转位置编码"""
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_config = scaling_config
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # 应用缩放
        if scaling_config is not None:
            self._apply_scaling(scaling_config)
    
    def _apply_scaling(self, scaling_config: Dict[str, Any]):
        """应用RoPE缩放"""
        scaling_type = scaling_config.get("type", "linear")
        scaling_factor = scaling_config.get("factor", 1.0)
        
        if scaling_type == "linear":
            self.inv_freq = self.inv_freq / scaling_factor
        elif scaling_type == "dynamic":
            # 动态缩放实现
            pass
        elif scaling_type == "yarn":
            # YaRN缩放实现
            pass
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算cos和sin值"""
        
        # 生成位置序列
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        # 计算频率
        freqs = torch.outer(t, self.inv_freq)
        
        # 计算cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """应用旋转位置编码"""
    
    # 获取cos和sin值
    cos = cos[position_ids].unsqueeze(2)  # [batch_size, seq_len, 1, head_dim]
    sin = sin[position_ids].unsqueeze(2)
    
    # 旋转函数
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# 注意力工厂函数
def create_attention(
    config: AttentionConfig,
    hidden_size: int,
    num_heads: int,
    **kwargs
) -> MultiHeadAttention:
    """创建注意力模块"""
    
    return MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        bias=config.bias,
        sliding_window=config.sliding_window,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
        max_position_embeddings=config.max_position_embeddings,
        attention_backend=config.backend,
        cache_config=config.cache_config,
        **kwargs
    )

# 使用示例
def example_attention_usage():
    """注意力机制使用示例"""
    
    # 配置
    config = AttentionConfig(
        backend="flash_attention",
        num_kv_heads=8,
        head_dim=128,
        sliding_window=4096,
        rope_theta=10000.0,
        max_position_embeddings=8192,
    )
    
    # 创建注意力模块
    attention = create_attention(
        config=config,
        hidden_size=4096,
        num_heads=32,
    )
    
    # 输入数据
    batch_size, seq_len = 2, 1024
    hidden_states = torch.randn(batch_size, seq_len, 4096)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 前向传播
    output = attention(
        hidden_states=hidden_states,
        position_ids=position_ids,
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    example_attention_usage()
```

## 🔧 关键特性分析

### 1. 多种注意力后端

- **PyTorch原生**：标准实现，兼容性好
- **Flash Attention**：内存高效，速度快
- **Paged Attention**：支持动态批处理
- **XFormers**：Facebook的优化实现

### 2. KV缓存优化

- **增量更新**：只计算新token的KV
- **内存复用**：高效的缓存管理
- **分页存储**：支持长序列处理

### 3. 位置编码支持

- **RoPE**：旋转位置编码
- **ALiBi**：注意力偏置
- **缩放策略**：支持长序列扩展

## 📊 性能优化技术

### Flash Attention优化

```python
class OptimizedFlashAttention(FlashAttention):
    """优化的Flash Attention"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 预分配内存
        self.workspace = None
        self.max_batch_size = 32
        self.max_seq_len = 4096
    
    def _preallocate_workspace(self, device: torch.device):
        """预分配工作空间"""
        if self.workspace is None:
            workspace_size = self._calculate_workspace_size()
            self.workspace = torch.empty(
                workspace_size, 
                dtype=torch.uint8, 
                device=device
            )
    
    def _calculate_workspace_size(self) -> int:
        """计算工作空间大小"""
        # 基于batch size和序列长度估算
        return self.max_batch_size * self.max_seq_len * self.head_dim * 4
    
    def forward(self, *args, **kwargs):
        """优化的前向传播"""
        # 预分配工作空间
        if len(args) > 0:
            self._preallocate_workspace(args[0].device)
        
        # 调用父类方法
        return super().forward(*args, **kwargs)
```

### 内存池管理

```python
class AttentionMemoryPool:
    """注意力内存池"""
    
    def __init__(self, device: torch.device, max_blocks: int = 1000):
        self.device = device
        self.max_blocks = max_blocks
        
        # 内存池
        self.free_blocks = []
        self.used_blocks = {}
        
        # 预分配内存块
        self._preallocate_blocks()
    
    def _preallocate_blocks(self):
        """预分配内存块"""
        block_size = 16 * 128  # 16 tokens * 128 head_dim
        
        for i in range(self.max_blocks):
            block = torch.empty(
                (block_size,), 
                dtype=torch.float16, 
                device=self.device
            )
            self.free_blocks.append(block)
    
    def allocate_block(self, request_id: str) -> torch.Tensor:
        """分配内存块"""
        if not self.free_blocks:
            raise RuntimeError("No free memory blocks available")
        
        block = self.free_blocks.pop()
        self.used_blocks[request_id] = block
        
        return block
    
    def free_block(self, request_id: str):
        """释放内存块"""
        if request_id in self.used_blocks:
            block = self.used_blocks.pop(request_id)
            self.free_blocks.append(block)
```

### 批处理优化

```python
class BatchedAttention:
    """批处理注意力"""
    
    def __init__(self, attention: BaseAttention):
        self.attention = attention
        self.batch_optimizer = AttentionBatchOptimizer()
    
    def forward_batch(
        self, 
        queries: List[torch.Tensor],
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """批处理前向传播"""
        
        # 优化批处理
        batches = self.batch_optimizer.create_batches(queries, keys, values)
        
        outputs = []
        for batch in batches:
            # 合并批次
            batch_query = torch.cat([b['query'] for b in batch], dim=0)
            batch_key = torch.cat([b['key'] for b in batch], dim=0)
            batch_value = torch.cat([b['value'] for b in batch], dim=0)
            
            # 执行注意力
            batch_output = self.attention(batch_query, batch_key, batch_value, **kwargs)
            
            # 分割输出
            start_idx = 0
            for b in batch:
                seq_len = b['query'].size(1)
                output = batch_output[start_idx:start_idx+1, :seq_len]
                outputs.append(output)
                start_idx += 1
        
        return outputs

class AttentionBatchOptimizer:
    """注意力批处理优化器"""
    
    def create_batches(
        self, 
        queries: List[torch.Tensor],
        keys: List[torch.Tensor],
        values: List[torch.Tensor],
        max_batch_size: int = 32,
        max_tokens: int = 2048
    ) -> List[List[Dict]]:
        """创建优化的批次"""
        
        # 按序列长度排序
        items = list(zip(queries, keys, values))
        items.sort(key=lambda x: x[0].size(1))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for query, key, value in items:
            seq_len = query.size(1)
            
            if (len(current_batch) < max_batch_size and 
                current_tokens + seq_len <= max_tokens):
                current_batch.append({
                    'query': query,
                    'key': key,
                    'value': value
                })
                current_tokens += seq_len
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [{
                    'query': query,
                    'key': key,
                    'value': value
                }]
                current_tokens = seq_len
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
```

## 🚀 使用最佳实践

### 1. 后端选择策略

```python
def select_attention_backend(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: str
) -> str:
    """选择最优的注意力后端"""
    
    # GPU且支持Flash Attention
    if device == "cuda" and seq_len > 512:
        return "flash_attention"
    
    # 长序列使用分页注意力
    if seq_len > 4096:
        return "paged_attention"
    
    # 小批次使用PyTorch
    if batch_size <= 4:
        return "torch"
    
    # 默认使用Flash Attention
    return "flash_attention"
```

### 2. 内存优化配置

```python
# 内存受限环境
memory_optimized_config = AttentionConfig(
    backend="paged_attention",
    block_size=16,           # 小块大小
    max_blocks_per_seq=256,  # 限制每序列块数
    enable_chunked_prefill=True,  # 分块预填充
    chunk_size=512,          # 块大小
)

# 性能优先环境
performance_config = AttentionConfig(
    backend="flash_attention",
    enable_torch_compile=True,    # 启用编译
    use_cuda_graph=True,          # 使用CUDA图
    preallocate_workspace=True,   # 预分配工作空间
)
```

### 3. 监控和调试

```python
class AttentionProfiler:
    """注意力性能分析器"""
    
    def __init__(self):
        self.metrics = {
            'forward_time': [],
            'memory_usage': [],
            'cache_hit_rate': [],
        }
    
    def profile_attention(self, attention_fn, *args, **kwargs):
        """分析注意力性能"""
        
        # 记录开始时间和内存
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 执行注意力
        output = attention_fn(*args, **kwargs)
        
        # 记录结束时间和内存
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 更新指标
        self.metrics['forward_time'].append(end_time - start_time)
        self.metrics['memory_usage'].append(end_memory - start_memory)
        
        return output
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'avg_forward_time': np.mean(self.metrics['forward_time']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'max_memory_usage': np.max(self.metrics['memory_usage']),
        }
```

---

*注意力机制是 nano-vllm 的性能核心，选择合适的实现和优化策略对系统性能至关重要。*