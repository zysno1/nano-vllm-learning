# 注意力机制 (Attention Mechanism) 代码分析

## 🎯 注意力机制概览

注意力机制是 nano-vLLM 中 Transformer 架构的核心组件，负责计算序列中不同位置之间的关联性。本文档基于 nano-vLLM 的真实代码进行分析，深入解析其注意力机制的实现，包括通用注意力层、模型特定注意力层、KV缓存管理、Flash Attention集成等关键技术。

## 🏗️ 核心架构与导入

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

# nano-vllm 特定模块导入
from nano_vllm.kernels import flash_attention, paged_attention  # 优化内核
from nano_vllm.memory import KVCache, PagedKVCache              # 缓存管理
from nano_vllm.utils.tensor_parallel import tensor_parallel_linear  # 张量并行
from nano_vllm.config import AttentionConfig                    # 配置管理

logger = logging.getLogger(__name__)

class AttentionBackend(Enum):
    """
    注意力后端类型枚举
    
    设计思想：
    1. 使用枚举确保后端类型的类型安全
    2. 支持多种优化实现，可根据硬件和场景选择
    3. 便于扩展新的注意力实现
    """
    TORCH = "torch"                    # PyTorch原生实现，兼容性最好
    FLASH_ATTENTION = "flash_attention"  # Flash Attention，内存高效
    PAGED_ATTENTION = "paged_attention"  # 分页注意力，支持长序列
    XFORMERS = "xformers"              # Facebook的XFormers优化

@dataclass
class AttentionMetadata:
    """
    注意力计算的元数据
    
    设计思想：
    1. 集中管理注意力计算所需的所有元信息
    2. 支持批处理和动态序列长度
    3. 为不同注意力后端提供统一的数据接口
    """
    seq_lens: List[int]              # 每个序列的长度列表，支持变长序列
    max_seq_len: int                 # 批次中的最大序列长度，用于内存分配
    num_prefill_tokens: int          # 预填充阶段的token数量
    num_decode_tokens: int           # 解码阶段的token数量
    slot_mapping: torch.Tensor       # KV缓存的槽位映射，管理缓存位置
    context_lens: List[int]          # 每个序列的上下文长度
    block_tables: Optional[torch.Tensor] = None  # 分页注意力的块表
    use_cuda_graph: bool = False     # 是否使用CUDA图优化

class BaseAttention(nn.Module):
    """
    基础注意力模块抽象类
    
    设计思想：
    1. 定义所有注意力实现的通用接口
    2. 封装共同的配置和初始化逻辑
    3. 为不同实现提供统一的参数管理
    """
    
    def __init__(
        self,
        num_heads: int,                    # 注意力头数
        head_dim: int,                     # 每个头的维度
        scale: Optional[float] = None,     # 缩放因子，默认为1/sqrt(head_dim)
        num_kv_heads: Optional[int] = None,  # KV头数，支持Multi-Query Attention
        sliding_window: Optional[int] = None,  # 滑动窗口大小，限制注意力范围
        alibi_slopes: Optional[List[float]] = None,  # ALiBi位置编码斜率
        cache_config: Optional[Dict] = None,  # 缓存配置
    ):
        super().__init__()
        
        # 基础配置 - 注意力头的核心参数
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads  # 默认KV头数等于Q头数
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads  # 每个KV头对应的Q头数
        
        # 计算隐藏层大小
        self.hidden_size = num_heads * head_dim        # 查询的总维度
        self.kv_hidden_size = self.num_kv_heads * head_dim  # KV的总维度
        
        # 缩放因子 - 用于注意力分数的缩放，防止softmax饱和
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        
        # 滑动窗口注意力 - 限制注意力的范围，节省计算和内存
        self.sliding_window = sliding_window
        
        # ALiBi位置编码 - 不需要显式位置编码的注意力偏置方法
        self.alibi_slopes = alibi_slopes
        
        # 缓存配置 - 控制KV缓存的行为
        self.cache_config = cache_config or {}
        
        logger.info(f"Initialized attention: heads={num_heads}, head_dim={head_dim}, "
                   f"kv_heads={self.num_kv_heads}, sliding_window={sliding_window}")
    
    def forward(
        self,
        query: torch.Tensor,                    # 查询张量
        key: torch.Tensor,                      # 键张量
        value: torch.Tensor,                    # 值张量
        kv_cache: Optional[KVCache] = None,     # KV缓存
        attn_metadata: Optional[AttentionMetadata] = None,  # 注意力元数据
    ) -> torch.Tensor:
        """前向传播 - 子类必须实现"""
        raise NotImplementedError

class TorchAttention(BaseAttention):
    """
    PyTorch原生注意力实现
    
    设计思想：
    1. 使用PyTorch标准操作，确保最大兼容性
    2. 实现完整的注意力机制，包括掩码、位置编码等
    3. 作为其他优化实现的参考基准
    """
    
    def forward(
        self,
        query: torch.Tensor,                    # [batch_size, seq_len, num_heads, head_dim]
        key: torch.Tensor,                      # [batch_size, seq_len, num_kv_heads, head_dim]
        value: torch.Tensor,                    # [batch_size, seq_len, num_kv_heads, head_dim]
        kv_cache: Optional[KVCache] = None,     # KV缓存对象
        attn_metadata: Optional[AttentionMetadata] = None,  # 注意力元数据
    ) -> torch.Tensor:
        """
        PyTorch原生注意力前向传播
        
        Args:
            query: 查询张量，形状为 [batch_size, seq_len, num_heads, head_dim]
            key: 键张量，形状为 [batch_size, seq_len, num_kv_heads, head_dim]
            value: 值张量，形状为 [batch_size, seq_len, num_kv_heads, head_dim]
        
        Returns:
            output: 注意力输出，形状为 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = query.shape[:2]
        
        # 重塑张量形状 - 确保维度正确
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim)  # -1自动推断KV序列长度
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 处理KV缓存 - 将新的KV与缓存的KV拼接
        if kv_cache is not None:
            key, value = self._update_kv_cache(key, value, kv_cache, attn_metadata)
        
        # 扩展KV头以匹配查询头数 - 支持Multi-Query Attention
        if self.num_kv_heads != self.num_heads:
            key = self._repeat_kv(key, self.num_queries_per_kv)
            value = self._repeat_kv(value, self.num_queries_per_kv)
        
        # 转置以适应注意力计算 - 将seq_len和num_heads维度交换
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch, num_heads, kv_len, head_dim]
        value = value.transpose(1, 2)  # [batch, num_heads, kv_len, head_dim]
        
        # 计算注意力分数 - Q·K^T / sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 应用ALiBi位置编码 - 基于相对位置的注意力偏置
        if self.alibi_slopes is not None:
            attn_scores = self._apply_alibi(attn_scores, seq_len)
        
        # 应用注意力掩码 - 防止看到未来信息（因果掩码）
        attn_mask = self._create_attention_mask(seq_len, key.size(-2), query.device)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # 加上负无穷掩码
        
        # 应用滑动窗口 - 限制注意力范围
        if self.sliding_window is not None:
            attn_scores = self._apply_sliding_window(attn_scores, self.sliding_window)
        
        # Softmax归一化 - 转换为概率分布
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 应用dropout（训练时）- 防止过拟合
        if self.training:
            attn_weights = F.dropout(attn_weights, p=0.1)
        
        # 计算加权输出 - 注意力权重与值的加权和
        output = torch.matmul(attn_weights, value)
        
        # 重塑输出形状 - 恢复原始维度顺序
        output = output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_len, self.hidden_size)  # 合并头维度
        
        return output
    
    def _repeat_kv(self, tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        重复KV张量以匹配查询头数
        
        设计思想：
        1. 支持Multi-Query Attention，其中KV头数少于Q头数
        2. 通过重复KV头来匹配Q头数，实现参数共享
        3. 使用expand和reshape优化内存使用
        """
        batch, num_kv_heads, seq_len, head_dim = tensor.shape
        if n_rep == 1:
            return tensor  # 无需重复
        
        # 在新维度上扩展，然后重塑
        tensor = tensor[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return tensor.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)
    
    def _update_kv_cache(
        self, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV缓存
        
        设计思想：
        1. 增量式更新，只存储新的KV而不是重新计算所有
        2. 使用slot_mapping精确控制缓存位置
        3. 支持动态序列长度和批处理
        """
        if attn_metadata is None:
            return key, value
        
        # 获取缓存的KV - 从缓存中读取历史KV
        cached_key, cached_value = kv_cache.get_kv_cache()
        
        if cached_key is not None and cached_value is not None:
            # 拼接新的KV - 将新计算的KV与历史KV拼接
            key = torch.cat([cached_key, key], dim=1)  # 在序列长度维度拼接
            value = torch.cat([cached_value, value], dim=1)
        
        # 更新缓存 - 将拼接后的KV存回缓存
        kv_cache.update(key, value, attn_metadata.slot_mapping)
        
        return key, value
    
    def _create_attention_mask(
        self, 
        seq_len: int,      # 查询序列长度
        kv_len: int,       # 键值序列长度
        device: torch.device  # 计算设备
    ) -> Optional[torch.Tensor]:
        """
        创建注意力掩码
        
        设计思想：
        1. 实现因果掩码，防止模型看到未来信息
        2. 解码阶段优化，单token查询无需掩码
        3. 使用上三角矩阵高效创建掩码
        """
        if seq_len == 1:
            # 解码阶段优化 - 单token查询不需要掩码
            return None
        
        # 创建因果掩码 - 上三角部分为负无穷
        mask = torch.triu(
            torch.full((seq_len, kv_len), float('-inf'), device=device),
            diagonal=kv_len - seq_len + 1  # 对角线位置调整
        )
        
        return mask
    
    def _apply_alibi(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        应用ALiBi位置编码
        
        设计思想：
        1. ALiBi通过注意力偏置实现位置编码，无需额外参数
        2. 基于相对位置距离应用线性偏置
        3. 每个头使用不同的斜率，增加表达能力
        """
        if self.alibi_slopes is None:
            return attn_scores
        
        batch_size, num_heads = attn_scores.shape[:2]
        
        # 创建位置偏置矩阵
        position_ids = torch.arange(seq_len, device=attn_scores.device)
        relative_pos = position_ids[None, :] - position_ids[:, None]  # 相对位置矩阵
        
        # 应用ALiBi斜率 - 每个头使用不同斜率
        alibi_bias = torch.zeros_like(attn_scores)
        for i, slope in enumerate(self.alibi_slopes[:num_heads]):
            alibi_bias[:, i] = relative_pos * slope  # 线性偏置
        
        return attn_scores + alibi_bias
    
    def _apply_sliding_window(
        self, 
        attn_scores: torch.Tensor, 
        window_size: int
    ) -> torch.Tensor:
        """
        应用滑动窗口注意力
        
        设计思想：
        1. 限制注意力范围，减少计算复杂度
        2. 保持局部注意力模式，适合长序列处理
        3. 使用掩码实现窗口限制
        """
        seq_len = attn_scores.size(-2)
        kv_len = attn_scores.size(-1)
        
        # 创建滑动窗口掩码 - 超出窗口范围的位置设为负无穷
        mask = torch.triu(
            torch.full((seq_len, kv_len), float('-inf'), device=attn_scores.device),
            diagonal=window_size + 1  # 窗口大小控制
        )
        
        return attn_scores + mask

class FlashAttention(BaseAttention):
    """
    Flash Attention实现
    
    设计思想：
    1. 内存高效的注意力计算，减少HBM访问
    2. 分块计算，支持任意长度序列
    3. 融合内核，减少GPU内存带宽需求
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 检查Flash Attention可用性
        try:
            import flash_attn  # 尝试导入Flash Attention库
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
            # 优雅降级 - 回退到PyTorch实现
            torch_attn = TorchAttention(
                self.num_heads, self.head_dim, self.scale,
                self.num_kv_heads, self.sliding_window, self.alibi_slopes
            )
            return torch_attn.forward(query, key, value, kv_cache, attn_metadata)
        
        batch_size, seq_len = query.shape[:2]
        
        # 重塑张量 - Flash Attention要求特定的张量布局
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        
        # 处理KV缓存 - 与标准实现相同的缓存逻辑
        if kv_cache is not None:
            key, value = self._update_kv_cache(key, value, kv_cache, attn_metadata)
        
        # 调用Flash Attention内核 - 使用优化的CUDA内核
        output = self._flash_attention_forward(query, key, value, attn_metadata)
        
        return output.view(batch_size, seq_len, self.hidden_size)
    
    def _flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> torch.Tensor:
        """
        Flash Attention内核调用
        
        设计思想：
        1. 直接调用优化的CUDA内核
        2. 支持因果掩码和滑动窗口
        3. 自动处理内存分块和融合计算
        """
        from flash_attn import flash_attn_func
        
        # Flash Attention参数配置
        dropout_p = 0.0 if not self.training else 0.1  # 训练时启用dropout
        causal = True  # 启用因果掩码，防止看到未来信息
        
        # 处理滑动窗口 - Flash Attention的窗口参数格式
        window_size = (-1, -1)  # 默认无限窗口
        if self.sliding_window is not None:
            window_size = (-1, self.sliding_window)  # 只限制右侧窗口
        
        # 调用Flash Attention内核 - 高度优化的CUDA实现
        output = flash_attn_func(
            query, key, value,
            dropout_p=dropout_p,           # dropout概率
            causal=causal,                 # 因果掩码
            window_size=window_size,       # 滑动窗口大小
            alibi_slopes=self.alibi_slopes,  # ALiBi斜率
            return_attn_probs=False        # 不返回注意力权重，节省内存
        )
        
        return output

class PagedAttention(BaseAttention):
    """
    分页注意力实现
    
    设计思想：
    1. 将KV缓存分页存储，支持动态内存管理
    2. 分离预填充和解码阶段，优化不同场景
    3. 支持长序列和大批次处理
    """
    
    def __init__(self, *args, block_size: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size  # 分页块大小，影响内存粒度
        
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
            # 回退到标准注意力 - 无缓存或元数据时使用标准实现
            torch_attn = TorchAttention(
                self.num_heads, self.head_dim, self.scale,
                self.num_kv_heads, self.sliding_window, self.alibi_slopes
            )
            return torch_attn.forward(query, key, value, kv_cache, attn_metadata)
        
        batch_size, seq_len = query.shape[:2]
        
        # 重塑查询张量
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 分离预填充和解码阶段 - 不同阶段使用不同的优化策略
        if attn_metadata.num_prefill_tokens > 0:
            # 预填充阶段 - 处理输入序列的初始部分
            output = self._paged_prefill_attention(
                query, key, value, kv_cache, attn_metadata
            )
        else:
            # 解码阶段 - 生成新token时的注意力计算
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
        """
        分页预填充注意力
        
        设计思想：
        1. 预填充阶段需要处理完整的输入序列
        2. 更新分页KV缓存，为后续解码做准备
        3. 使用标准注意力计算，因为需要全序列交互
        """
        
        # 更新KV缓存 - 将新的KV存储到分页缓存中
        kv_cache.update_prefill(key, value, attn_metadata.slot_mapping)
        
        # 使用标准注意力进行预填充 - 预填充阶段需要全序列注意力
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
        """
        分页解码注意力
        
        设计思想：
        1. 解码阶段只需要计算单个token的注意力
        2. 使用分页KV缓存，支持高效的内存访问
        3. 调用专门的分页注意力内核
        """
        
        # 调用分页注意力内核 - 专门优化的解码阶段内核
        output = paged_attention.paged_attention_v1(
            query=query,                                    # 查询张量
            key_cache=kv_cache.key_cache,                  # 分页键缓存
            value_cache=kv_cache.value_cache,              # 分页值缓存
            num_kv_heads=self.num_kv_heads,                # KV头数
            scale=self.scale,                              # 缩放因子
            block_tables=attn_metadata.block_tables,       # 块表，管理分页映射
            context_lens=torch.tensor(attn_metadata.context_lens, device=query.device),  # 上下文长度
            block_size=self.block_size,                    # 块大小
            max_context_len=attn_metadata.max_seq_len,     # 最大上下文长度
            alibi_slopes=self.alibi_slopes,                # ALiBi斜率
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
        self.head_dim = head_dim or hidden_size // num_heads  # 自动计算头维度
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # 线性投影层 - 将输入投影到Q、K、V空间
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=bias)
        
        # 旋转位置编码 - RoPE实现相对位置编码
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_config=rope_scaling,
        )
        
        # 注意力实现选择 - 根据后端类型创建相应的注意力模块
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
            # 默认使用PyTorch实现
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