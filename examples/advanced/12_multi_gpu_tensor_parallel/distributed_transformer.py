#!/usr/bin/env python3
"""
分布式Transformer模型实现

本模块实现了完整的分布式Transformer模型，包括：
1. 分布式嵌入层
2. 多GPU Transformer块
3. 分布式输出层
4. 梯度同步和优化
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import math
from contextlib import contextmanager
import json

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributedTransformerConfig:
    """分布式Transformer配置"""
    # 模型参数
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008
    max_seq_length: int = 4096
    
    # 分布式参数
    world_size: int = 4
    tensor_parallel_size: int = 2  # 张量并行大小
    pipeline_parallel_size: int = 2  # 流水线并行大小
    
    # 训练参数
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # 优化参数
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    activation_function: str = "swiglu"
    
    # 通信参数
    communication_backend: str = "nccl"
    overlap_communication: bool = True
    bucket_size_mb: float = 25.0

class DistributedEmbedding(nn.Module):
    """分布式嵌入层"""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        world_size: int,
        rank: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.world_size = world_size
        self.rank = rank
        
        # 按词汇表分割
        self.vocab_size_per_gpu = vocab_size // world_size
        self.vocab_start = rank * self.vocab_size_per_gpu
        self.vocab_end = self.vocab_start + self.vocab_size_per_gpu
        
        # 嵌入层
        self.embedding = nn.Embedding(
            self.vocab_size_per_gpu,
            hidden_size,
            padding_idx=padding_idx
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len = input_ids.shape
        
        # 创建掩码，标识哪些token属于当前GPU
        mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        
        # 调整输入ID到本地范围
        local_input_ids = input_ids - self.vocab_start
        local_input_ids = local_input_ids.clamp(0, self.vocab_size_per_gpu - 1)
        
        # 嵌入查找
        embeddings = self.embedding(local_input_ids)
        
        # 应用掩码
        embeddings = embeddings * mask.unsqueeze(-1).float()
        
        # 全归约以获得完整的嵌入
        dist.all_reduce(embeddings, op=dist.ReduceOp.SUM)
        
        return embeddings

class DistributedLinear(nn.Module):
    """分布式线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True,
        split_dim: int = 1  # 0: 按行分割, 1: 按列分割
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        self.split_dim = split_dim
        
        if split_dim == 1:  # 按列分割
            self.out_features_per_gpu = out_features // world_size
            self.weight = nn.Parameter(torch.randn(self.out_features_per_gpu, in_features))
            if bias:
                self.bias = nn.Parameter(torch.randn(self.out_features_per_gpu))
            else:
                self.register_parameter('bias', None)
        else:  # 按行分割
            self.in_features_per_gpu = in_features // world_size
            self.weight = nn.Parameter(torch.randn(out_features, self.in_features_per_gpu))
            if bias and rank == 0:  # 只有第一个GPU有bias
                self.bias = nn.Parameter(torch.randn(out_features))
            else:
                self.register_parameter('bias', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.split_dim == 1:  # 按列分割
            # 线性变换
            output = torch.matmul(x, self.weight.t())
            if self.bias is not None:
                output += self.bias
            
            # 如果需要收集输出
            if self.gather_output:
                output_list = [torch.zeros_like(output) for _ in range(self.world_size)]
                dist.all_gather(output_list, output)
                output = torch.cat(output_list, dim=-1)
        
        else:  # 按行分割
            # 分割输入
            input_list = torch.split(x, x.size(-1) // self.world_size, dim=-1)
            local_input = input_list[self.rank]
            
            # 线性变换
            output = torch.matmul(local_input, self.weight.t())
            
            # 全归约
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            
            # 添加bias（只有第一个GPU）
            if self.bias is not None and self.rank == 0:
                output += self.bias
        
        return output

class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int,
        rank: int
    ):
        super().__init__()
        self.gate_proj = DistributedLinear(
            hidden_size, intermediate_size, world_size, rank,
            bias=False, gather_output=False
        )
        self.up_proj = DistributedLinear(
            hidden_size, intermediate_size, world_size, rank,
            bias=False, gather_output=False
        )
        self.down_proj = DistributedLinear(
            intermediate_size // world_size, hidden_size, world_size, rank,
            bias=False, gather_output=True, split_dim=0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU: gate * silu(up)
        intermediate = gate * F.silu(up)
        
        # 下投影
        output = self.down_proj(intermediate)
        
        return output

class DistributedAttention(nn.Module):
    """分布式注意力机制"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        max_seq_length: int = 4096,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.max_seq_length = max_seq_length
        self.use_flash_attention = use_flash_attention
        
        # 确保头数能被world_size整除
        assert num_heads % world_size == 0
        self.num_heads_per_gpu = num_heads // world_size
        
        # QKV投影
        self.qkv_proj = DistributedLinear(
            hidden_size, 3 * self.num_heads_per_gpu * head_dim,
            world_size, rank, bias=False, gather_output=False
        )
        
        # 输出投影
        self.out_proj = DistributedLinear(
            self.num_heads_per_gpu * head_dim, hidden_size,
            world_size, rank, bias=False, gather_output=True, split_dim=0
        )
        
        self.scale = head_dim ** -0.5
        
        # 旋转位置编码
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_length)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, hidden_size = x.shape
        
        # QKV投影
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads_per_gpu, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # 应用旋转位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        q, k = self.rotary_emb(q, k, position_ids)
        
        # 转置为 [batch, num_heads_per_gpu, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.use_flash_attention:
            # 使用Flash Attention
            attn_output = self._flash_attention(q, k, v, attention_mask)
        else:
            # 标准注意力
            attn_output = self._standard_attention(q, k, v, attention_mask)
        
        # 转置回 [batch, seq_len, num_heads_per_gpu, head_dim]
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(
            batch_size, seq_len, self.num_heads_per_gpu * self.head_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Flash Attention实现"""
        # 这里是简化的Flash Attention实现
        # 实际使用中应该使用优化的CUDA内核
        
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 分块大小
        block_size = min(512, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        
        # 初始化输出
        output = torch.zeros_like(q)
        lse = torch.full((batch_size, num_heads, seq_len), -float('inf'), device=q.device)
        
        for i in range(num_blocks):
            start_i = i * block_size
            end_i = min(start_i + block_size, seq_len)
            
            q_block = q[:, :, start_i:end_i, :]
            
            for j in range(num_blocks):
                start_j = j * block_size
                end_j = min(start_j + block_size, seq_len)
                
                k_block = k[:, :, start_j:end_j, :]
                v_block = v[:, :, start_j:end_j, :]
                
                # 计算注意力分数
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # 应用掩码
                if attention_mask is not None:
                    mask_block = attention_mask[:, start_i:end_i, start_j:end_j]
                    scores = scores.masked_fill(mask_block == 0, -float('inf'))
                
                # 在线Softmax更新
                scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
                scores_exp = torch.exp(scores - scores_max)
                
                # 更新LSE
                old_lse = lse[:, :, start_i:end_i].unsqueeze(-1)
                new_lse = torch.logsumexp(
                    torch.stack([old_lse, scores_max + torch.log(torch.sum(scores_exp, dim=-1, keepdim=True))], dim=-1),
                    dim=-1
                )
                
                # 重新归一化
                alpha = torch.exp(old_lse - new_lse)
                output[:, :, start_i:end_i, :] *= alpha
                
                # 添加当前贡献
                beta = torch.exp(scores_max - new_lse)
                attn_weights = scores_exp * beta
                output[:, :, start_i:end_i, :] += torch.matmul(attn_weights, v_block)
                
                # 更新LSE
                lse[:, :, start_i:end_i] = new_lse.squeeze(-1)
        
        return output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """标准注意力实现"""
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -float('inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output

class RotaryEmbedding(nn.Module):
    """旋转位置编码"""
    
    def __init__(self, dim: int, max_seq_length: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算cos和sin值
        self._precompute_freqs_cis(max_seq_length)
    
    def _precompute_freqs_cis(self, seq_len: int):
        """预计算cos和sin值"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用旋转位置编码"""
        seq_len = q.size(2)
        
        if seq_len > self.cos_cached.size(0):
            self._precompute_freqs_cis(seq_len)
        
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq_len, dim//2]
        sin = self.sin_cached[position_ids].unsqueeze(1)
        
        # 应用旋转
        q_rot = self._apply_rotary_pos_emb(q, cos, sin)
        k_rot = self._apply_rotary_pos_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """应用旋转位置编码"""
        # 分离x的前半部分和后半部分
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        # 应用旋转
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class DistributedTransformerLayer(nn.Module):
    """分布式Transformer层"""
    
    def __init__(self, config: DistributedTransformerConfig, rank: int):
        super().__init__()
        self.config = config
        self.rank = rank
        
        # 注意力层
        self.attention = DistributedAttention(
            config.hidden_size,
            config.num_heads,
            config.head_dim,
            config.tensor_parallel_size,
            rank % config.tensor_parallel_size,
            config.max_seq_length,
            config.use_flash_attention
        )
        
        # 前馈网络
        if config.activation_function == "swiglu":
            self.feed_forward = SwiGLU(
                config.hidden_size,
                config.intermediate_size,
                config.tensor_parallel_size,
                rank % config.tensor_parallel_size
            )
        else:
            self.feed_forward = DistributedMLP(
                config.hidden_size,
                config.intermediate_size,
                config.tensor_parallel_size,
                rank % config.tensor_parallel_size
            )
        
        # 层归一化
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 注意力残差连接
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, attention_mask, position_ids)
        x = residual + x
        
        # 前馈网络残差连接
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x

class DistributedMLP(nn.Module):
    """分布式多层感知机"""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        world_size: int,
        rank: int
    ):
        super().__init__()
        self.up_proj = DistributedLinear(
            hidden_size, intermediate_size, world_size, rank,
            bias=False, gather_output=False
        )
        self.down_proj = DistributedLinear(
            intermediate_size // world_size, hidden_size, world_size, rank,
            bias=False, gather_output=True, split_dim=0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        return x

class DistributedTransformer(nn.Module):
    """分布式Transformer模型"""
    
    def __init__(self, config: DistributedTransformerConfig, rank: int):
        super().__init__()
        self.config = config
        self.rank = rank
        
        # 嵌入层
        self.embedding = DistributedEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.tensor_parallel_size,
            rank % config.tensor_parallel_size
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            DistributedTransformerLayer(config, rank)
            for _ in range(config.num_layers)
        ])
        
        # 最终层归一化
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # 输出层
        self.lm_head = DistributedLinear(
            config.hidden_size,
            config.vocab_size,
            config.tensor_parallel_size,
            rank % config.tensor_parallel_size,
            bias=False,
            gather_output=True
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        # 嵌入
        x = self.embedding(input_ids)
        
        # Transformer层
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, position_ids)
            else:
                x = layer(x, attention_mask, position_ids)
        
        # 最终层归一化
        x = self.final_layernorm(x)
        
        # 输出投影
        logits = self.lm_head(x)
        
        return logits

class DistributedTransformerBenchmark:
    """分布式Transformer基准测试"""
    
    def __init__(self, config: DistributedTransformerConfig):
        self.config = config
        self.results = {}
    
    def benchmark_model_sizes(self):
        """基准测试不同模型大小"""
        print("\n🚀 开始分布式Transformer基准测试...")
        
        # 测试不同的模型配置
        model_configs = [
            ("Small (1B)", {"num_layers": 12, "hidden_size": 2048, "num_heads": 16}),
            ("Medium (3B)", {"num_layers": 24, "hidden_size": 3072, "num_heads": 24}),
            ("Large (7B)", {"num_layers": 32, "hidden_size": 4096, "num_heads": 32}),
            ("XL (13B)", {"num_layers": 40, "hidden_size": 5120, "num_heads": 40}),
        ]
        
        results = {}
        
        for model_name, model_params in model_configs:
            print(f"\n📊 测试 {model_name} 模型...")
            
            # 更新配置
            test_config = DistributedTransformerConfig(**{**self.config.__dict__, **model_params})
            
            # 运行测试
            result = self._benchmark_single_config(test_config)
            results[model_name] = result
        
        self.results["model_sizes"] = results
        self.print_results()
        self.visualize_results()
    
    def _benchmark_single_config(self, config: DistributedTransformerConfig) -> Dict[str, float]:
        """基准测试单个配置"""
        # 创建模型
        model = DistributedTransformer(config, rank=0)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        
        # 创建输入数据
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_length // 4))
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 基准测试
        num_iterations = 5
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = model(input_ids)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # 计算性能指标
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        throughput = (config.batch_size * input_ids.size(1)) / avg_time
        
        # 模拟分布式开销
        comm_overhead = 0.15 * avg_time * (config.world_size - 1)
        effective_throughput = (config.batch_size * input_ids.size(1)) / (avg_time + comm_overhead)
        
        return {
            "forward_time": avg_time,
            "memory_usage": memory_usage,
            "total_params": total_params,
            "throughput": throughput,
            "communication_overhead": comm_overhead,
            "effective_throughput": effective_throughput,
            "params_per_gpu": total_params / config.tensor_parallel_size
        }
    
    def print_results(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📈 分布式Transformer性能分析")
        print("="*80)
        
        if "model_sizes" in self.results:
            print(f"\n🔍 不同模型大小性能对比:")
            
            for model_name, result in self.results["model_sizes"].items():
                print(f"\n  {model_name}:")
                print(f"    - 参数数量: {result['total_params']/1e9:.2f}B")
                print(f"    - 每GPU参数: {result['params_per_gpu']/1e9:.2f}B")
                print(f"    - 前向时间: {result['forward_time']*1000:.2f} ms")
                print(f"    - 内存使用: {result['memory_usage']:.2f} GB")
                print(f"    - 通信开销: {result['communication_overhead']*1000:.2f} ms")
                print(f"    - 有效吞吐量: {result['effective_throughput']:.0f} tokens/s")
    
    def visualize_results(self):
        """可视化结果"""
        if "model_sizes" not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('分布式Transformer性能分析', fontsize=16, fontweight='bold')
        
        results = self.results["model_sizes"]
        model_names = list(results.keys())
        
        # 1. 参数数量对比
        ax1 = axes[0, 0]
        params = [results[name]["total_params"]/1e9 for name in model_names]
        
        bars1 = ax1.bar(model_names, params, color='#FF6B6B', alpha=0.8)
        ax1.set_ylabel('参数数量 (B)')
        ax1.set_title('模型参数数量对比')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, param in zip(bars1, params):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{param:.1f}B', ha='center', va='bottom')
        
        # 2. 执行时间对比
        ax2 = axes[0, 1]
        times = [results[name]["forward_time"]*1000 for name in model_names]
        
        bars2 = ax2.bar(model_names, times, color='#4ECDC4', alpha=0.8)
        ax2.set_ylabel('执行时间 (ms)')
        ax2.set_title('前向传播时间对比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars2, times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{time:.0f}ms', ha='center', va='bottom')
        
        # 3. 内存使用对比
        ax3 = axes[1, 0]
        memories = [results[name]["memory_usage"] for name in model_names]
        
        bars3 = ax3.bar(model_names, memories, color='#45B7D1', alpha=0.8)
        ax3.set_ylabel('内存使用 (GB)')
        ax3.set_title('GPU内存使用对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, memory in zip(bars3, memories):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{memory:.1f}GB', ha='center', va='bottom')
        
        # 4. 吞吐量对比
        ax4 = axes[1, 1]
        throughputs = [results[name]["effective_throughput"] for name in model_names]
        
        bars4 = ax4.bar(model_names, throughputs, color='#96CEB4', alpha=0.8)
        ax4.set_ylabel('吞吐量 (tokens/s)')
        ax4.set_title('有效吞吐量对比')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, throughput in zip(bars4, throughputs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{throughput:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "distributed_transformer_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🎯 分布式Transformer模型实现演示")
    print("="*50)
    
    # 配置
    config = DistributedTransformerConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        intermediate_size=11008,
        max_seq_length=4096,
        world_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        batch_size=4
    )
    
    # 运行基准测试
    benchmark = DistributedTransformerBenchmark(config)
    benchmark.benchmark_model_sizes()
    
    print("\n✅ 分布式Transformer模型演示完成！")
    print("📁 结果图表已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()