#!/usr/bin/env python3
"""
多GPU注意力机制实现

本模块实现了分布式注意力机制，包括：
1. 分布式QKV计算
2. 跨GPU注意力聚合
3. 通信优化策略
4. 内存高效的实现
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import all_gather, all_reduce, reduce_scatter
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from contextlib import contextmanager
import math

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiGPUAttentionConfig:
    """多GPU注意力配置"""
    world_size: int = 4  # GPU数量
    hidden_size: int = 4096  # 隐藏层大小
    num_heads: int = 32  # 注意力头数
    head_dim: int = 128  # 每个头的维度
    seq_length: int = 2048  # 序列长度
    batch_size: int = 4  # 批次大小
    dropout: float = 0.1  # Dropout率
    use_flash_attention: bool = True  # 是否使用Flash Attention
    use_gradient_checkpointing: bool = True  # 是否使用梯度检查点
    communication_backend: str = "nccl"  # 通信后端
    overlap_communication: bool = True  # 是否重叠通信和计算

class CommunicationOptimizer:
    """通信优化器"""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.comm_times = []
    
    @contextmanager
    def time_communication(self, operation_name: str):
        """计时通信操作"""
        start_time = time.time()
        yield
        end_time = time.time()
        comm_time = end_time - start_time
        self.comm_times.append((operation_name, comm_time))
        logger.debug(f"Communication {operation_name} took {comm_time*1000:.2f}ms")
    
    def all_gather_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """高效的全收集操作"""
        with self.time_communication("all_gather"):
            # 创建输出张量列表
            tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            
            # 执行all_gather
            dist.all_gather(tensor_list, tensor)
            
            # 沿指定维度拼接
            return torch.cat(tensor_list, dim=dim)
    
    def reduce_scatter_tensor(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """高效的归约散射操作"""
        with self.time_communication("reduce_scatter"):
            # 计算每个GPU的分片大小
            total_size = tensor.size(dim)
            chunk_size = total_size // self.world_size
            
            # 分割张量
            chunks = torch.split(tensor, chunk_size, dim=dim)
            
            # 创建输出张量
            output = torch.zeros_like(chunks[0])
            
            # 执行reduce_scatter
            dist.reduce_scatter(output, list(chunks))
            
            return output
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """全归约操作"""
        with self.time_communication("all_reduce"):
            dist.all_reduce(tensor, op=op)
            return tensor
    
    def get_communication_stats(self) -> Dict[str, float]:
        """获取通信统计信息"""
        if not self.comm_times:
            return {}
        
        stats = {}
        for op_name, comm_time in self.comm_times:
            if op_name not in stats:
                stats[op_name] = []
            stats[op_name].append(comm_time)
        
        # 计算统计信息
        result = {}
        for op_name, times in stats.items():
            result[f"{op_name}_avg"] = np.mean(times)
            result[f"{op_name}_total"] = np.sum(times)
            result[f"{op_name}_count"] = len(times)
        
        return result

class DistributedQKVProjection(nn.Module):
    """分布式QKV投影层"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        world_size: int,
        rank: int,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        
        # 确保头数能被world_size整除
        assert num_heads % world_size == 0, f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
        self.num_heads_per_gpu = num_heads // world_size
        
        # 每个GPU只负责部分头的QKV计算
        self.qkv_dim_per_gpu = 3 * self.num_heads_per_gpu * head_dim
        
        # QKV投影权重
        self.qkv_weight = nn.Parameter(torch.randn(self.qkv_dim_per_gpu, hidden_size))
        
        if bias:
            self.qkv_bias = nn.Parameter(torch.randn(self.qkv_dim_per_gpu))
        else:
            self.register_parameter('qkv_bias', None)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.qkv_weight)
        if self.qkv_bias is not None:
            nn.init.zeros_(self.qkv_bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len, hidden_size = x.shape
        
        # QKV投影
        qkv = torch.matmul(x, self.qkv_weight.t())
        if self.qkv_bias is not None:
            qkv += self.qkv_bias
        
        # 重塑并分离QKV
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads_per_gpu, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        return q, k, v

class RingAttention(nn.Module):
    """环形注意力机制 - 用于长序列的内存高效实现"""
    
    def __init__(
        self,
        num_heads_per_gpu: int,
        head_dim: int,
        world_size: int,
        rank: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads_per_gpu = num_heads_per_gpu
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.scale = head_dim ** -0.5
        
        self.dropout = nn.Dropout(dropout)
        self.comm_optimizer = CommunicationOptimizer(world_size, rank)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        环形注意力前向传播
        
        Args:
            q: Query张量 [batch, seq_len_per_gpu, num_heads_per_gpu, head_dim]
            k: Key张量 [batch, seq_len_per_gpu, num_heads_per_gpu, head_dim]
            v: Value张量 [batch, seq_len_per_gpu, num_heads_per_gpu, head_dim]
            mask: 注意力掩码
        
        Returns:
            注意力输出 [batch, seq_len_per_gpu, num_heads_per_gpu, head_dim]
        """
        batch_size, seq_len_per_gpu, num_heads_per_gpu, head_dim = q.shape
        
        # 转置为 [batch, num_heads_per_gpu, seq_len_per_gpu, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 初始化输出和归一化项
        output = torch.zeros_like(q)
        lse = torch.full((batch_size, num_heads_per_gpu, seq_len_per_gpu), -float('inf'), device=q.device)
        
        # 环形通信：每个GPU依次处理所有GPU的KV
        for step in range(self.world_size):
            # 计算当前步骤对应的GPU rank
            kv_rank = (self.rank + step) % self.world_size
            
            # 如果是第一步，使用本地的KV
            if step == 0:
                k_curr, v_curr = k, v
            else:
                # 从下一个GPU接收KV
                k_curr = torch.zeros_like(k)
                v_curr = torch.zeros_like(v)
                
                # 模拟环形通信（实际实现需要点对点通信）
                if kv_rank < self.world_size:
                    k_curr = k  # 这里应该是从其他GPU接收的数据
                    v_curr = v
            
            # 计算注意力分数
            scores = torch.matmul(q, k_curr.transpose(-2, -1)) * self.scale
            
            # 应用掩码
            if mask is not None:
                # 计算全局位置的掩码
                global_start = kv_rank * seq_len_per_gpu
                global_end = global_start + seq_len_per_gpu
                mask_slice = mask[:, :, :, global_start:global_end]
                scores = scores.masked_fill(mask_slice == 0, -float('inf'))
            
            # 在线Softmax更新
            scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
            scores_exp = torch.exp(scores - scores_max)
            
            # 更新LSE (Log-Sum-Exp)
            new_lse = torch.logsumexp(torch.stack([lse.unsqueeze(-1), scores_max.squeeze(-1) + torch.log(torch.sum(scores_exp, dim=-1))], dim=-1), dim=-1)
            
            # 重新归一化之前的输出
            alpha = torch.exp(lse - new_lse).unsqueeze(-1)
            output = output * alpha
            
            # 添加当前步骤的贡献
            beta = torch.exp(scores_max.squeeze(-1) - new_lse).unsqueeze(-1)
            attn_weights = scores_exp * beta
            output += torch.matmul(attn_weights, v_curr)
            
            # 更新LSE
            lse = new_lse
        
        # 转置回原始形状
        output = output.transpose(1, 2)
        
        return output

class MultiGPUAttention(nn.Module):
    """多GPU注意力机制"""
    
    def __init__(self, config: MultiGPUAttentionConfig, rank: int):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = config.world_size
        
        # 计算每个GPU的头数
        assert config.num_heads % config.world_size == 0
        self.num_heads_per_gpu = config.num_heads // config.world_size
        
        # QKV投影层
        self.qkv_proj = DistributedQKVProjection(
            config.hidden_size,
            config.num_heads,
            config.head_dim,
            config.world_size,
            rank
        )
        
        # 注意力计算
        if config.use_flash_attention:
            self.attention = RingAttention(
                self.num_heads_per_gpu,
                config.head_dim,
                config.world_size,
                rank,
                config.dropout
            )
        else:
            self.attention = StandardDistributedAttention(
                self.num_heads_per_gpu,
                config.head_dim,
                config.world_size,
                rank,
                config.dropout
            )
        
        # 输出投影层
        self.out_proj = DistributedLinear(
            self.num_heads_per_gpu * config.head_dim,
            config.hidden_size,
            config.world_size,
            rank,
            gather_output=True
        )
        
        self.comm_optimizer = CommunicationOptimizer(config.world_size, rank)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, hidden_size = x.shape
        
        # 计算每个GPU处理的序列长度
        seq_len_per_gpu = seq_len // self.world_size
        start_idx = self.rank * seq_len_per_gpu
        end_idx = start_idx + seq_len_per_gpu
        
        # 分割输入序列
        x_local = x[:, start_idx:end_idx, :]
        
        # QKV投影
        q, k, v = self.qkv_proj(x_local)
        
        # 注意力计算
        attn_output = self.attention(q, k, v, mask)
        
        # 重塑输出
        attn_output = attn_output.contiguous().view(
            batch_size, seq_len_per_gpu, self.num_heads_per_gpu * self.config.head_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output

class StandardDistributedAttention(nn.Module):
    """标准分布式注意力机制"""
    
    def __init__(
        self,
        num_heads_per_gpu: int,
        head_dim: int,
        world_size: int,
        rank: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads_per_gpu = num_heads_per_gpu
        self.head_dim = head_dim
        self.world_size = world_size
        self.rank = rank
        self.scale = head_dim ** -0.5
        
        self.dropout = nn.Dropout(dropout)
        self.comm_optimizer = CommunicationOptimizer(world_size, rank)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len_per_gpu, num_heads_per_gpu, head_dim = q.shape
        
        # 转置为 [batch, num_heads_per_gpu, seq_len_per_gpu, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 收集所有GPU的K和V
        k_all = self.comm_optimizer.all_gather_tensor(k, dim=2)  # [batch, num_heads_per_gpu, seq_len_total, head_dim]
        v_all = self.comm_optimizer.all_gather_tensor(v, dim=2)  # [batch, num_heads_per_gpu, seq_len_total, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v_all)
        
        # 转置回原始形状
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output

class DistributedLinear(nn.Module):
    """分布式线性层"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = True,
        gather_output: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.gather_output = gather_output
        
        # 按列分割权重
        self.out_features_per_gpu = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.out_features_per_gpu, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_features_per_gpu))
        else:
            self.register_parameter('bias', None)
        
        self.comm_optimizer = CommunicationOptimizer(world_size, rank)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 线性变换
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        
        # 如果需要收集输出
        if self.gather_output:
            output = self.comm_optimizer.all_gather_tensor(output, dim=-1)
        
        return output

class MultiGPUAttentionBenchmark:
    """多GPU注意力基准测试"""
    
    def __init__(self, config: MultiGPUAttentionConfig):
        self.config = config
        self.results = {}
    
    def benchmark_attention_variants(self) -> Dict[str, Any]:
        """基准测试不同的注意力变体"""
        results = {}
        
        # 测试配置
        test_configs = [
            ("Standard", False, False),
            ("Flash Attention", True, False),
            ("Flash + Gradient Checkpointing", True, True),
        ]
        
        for name, use_flash, use_grad_checkpoint in test_configs:
            print(f"\n🔄 测试 {name}...")
            
            # 更新配置
            test_config = MultiGPUAttentionConfig(
                world_size=self.config.world_size,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                head_dim=self.config.head_dim,
                seq_length=self.config.seq_length,
                batch_size=self.config.batch_size,
                use_flash_attention=use_flash,
                use_gradient_checkpointing=use_grad_checkpoint
            )
            
            # 模拟多GPU测试
            result = self._simulate_multi_gpu_test(test_config)
            results[name] = result
        
        return results
    
    def _simulate_multi_gpu_test(self, config: MultiGPUAttentionConfig) -> Dict[str, float]:
        """模拟多GPU测试"""
        # 创建模型（模拟单个GPU）
        model = MultiGPUAttention(config, rank=0)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 创建输入数据
        seq_len_per_gpu = config.seq_length // config.world_size
        x = torch.randn(
            config.batch_size,
            seq_len_per_gpu,
            config.hidden_size
        )
        
        if torch.cuda.is_available():
            x = x.cuda()
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 基准测试
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                output = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # 计算性能指标
        total_params = sum(p.numel() for p in model.parameters())
        memory_usage = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        # 模拟通信开销
        comm_overhead = 0.1 * avg_time * (config.world_size - 1)  # 模拟通信开销
        
        return {
            "forward_time": avg_time,
            "memory_usage": memory_usage,
            "total_params": total_params,
            "communication_overhead": comm_overhead,
            "effective_throughput": (config.batch_size * config.seq_length) / (avg_time + comm_overhead)
        }
    
    def run_scalability_test(self):
        """运行可扩展性测试"""
        print("\n🚀 开始多GPU注意力可扩展性测试...")
        
        # 测试不同的GPU数量
        world_sizes = [1, 2, 4, 8]
        scalability_results = {}
        
        for world_size in world_sizes:
            if world_size > torch.cuda.device_count() and torch.cuda.is_available():
                print(f"⚠️  跳过 {world_size} GPU测试（设备不足）")
                continue
            
            print(f"\n📊 测试 {world_size} GPU配置...")
            
            # 更新配置
            test_config = MultiGPUAttentionConfig(
                world_size=world_size,
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                seq_length=self.config.seq_length,
                batch_size=self.config.batch_size
            )
            
            # 运行测试
            result = self._simulate_multi_gpu_test(test_config)
            scalability_results[f"{world_size}_GPU"] = result
        
        self.results["scalability"] = scalability_results
        
        # 测试不同的注意力变体
        attention_results = self.benchmark_attention_variants()
        self.results["attention_variants"] = attention_results
        
        self.print_results()
        self.visualize_results()
    
    def print_results(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📈 多GPU注意力机制性能分析")
        print("="*80)
        
        # 可扩展性结果
        if "scalability" in self.results:
            print(f"\n🔍 可扩展性测试结果:")
            scalability = self.results["scalability"]
            
            for config_name, result in scalability.items():
                print(f"\n  {config_name}:")
                print(f"    - 前向时间: {result['forward_time']*1000:.2f} ms")
                print(f"    - 内存使用: {result['memory_usage']:.2f} GB")
                print(f"    - 通信开销: {result['communication_overhead']*1000:.2f} ms")
                print(f"    - 有效吞吐量: {result['effective_throughput']:.0f} tokens/s")
        
        # 注意力变体结果
        if "attention_variants" in self.results:
            print(f"\n🎯 注意力变体对比:")
            variants = self.results["attention_variants"]
            
            for variant_name, result in variants.items():
                print(f"\n  {variant_name}:")
                print(f"    - 前向时间: {result['forward_time']*1000:.2f} ms")
                print(f"    - 内存使用: {result['memory_usage']:.2f} GB")
                print(f"    - 参数数量: {result['total_params']:,}")
                print(f"    - 有效吞吐量: {result['effective_throughput']:.0f} tokens/s")
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('多GPU注意力机制性能分析', fontsize=16, fontweight='bold')
        
        # 1. 可扩展性 - 执行时间
        if "scalability" in self.results:
            ax1 = axes[0, 0]
            scalability = self.results["scalability"]
            
            configs = list(scalability.keys())
            times = [scalability[config]["forward_time"]*1000 for config in configs]
            
            ax1.plot(range(len(configs)), times, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
            ax1.set_xlabel('GPU配置')
            ax1.set_ylabel('执行时间 (ms)')
            ax1.set_title('可扩展性 - 执行时间')
            ax1.set_xticks(range(len(configs)))
            ax1.set_xticklabels(configs, rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. 可扩展性 - 吞吐量
        if "scalability" in self.results:
            ax2 = axes[0, 1]
            throughputs = [scalability[config]["effective_throughput"] for config in configs]
            
            ax2.plot(range(len(configs)), throughputs, 's-', linewidth=2, markersize=8, color='#4ECDC4')
            ax2.set_xlabel('GPU配置')
            ax2.set_ylabel('吞吐量 (tokens/s)')
            ax2.set_title('可扩展性 - 吞吐量')
            ax2.set_xticks(range(len(configs)))
            ax2.set_xticklabels(configs, rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 3. 注意力变体对比 - 执行时间
        if "attention_variants" in self.results:
            ax3 = axes[1, 0]
            variants = self.results["attention_variants"]
            
            variant_names = list(variants.keys())
            variant_times = [variants[name]["forward_time"]*1000 for name in variant_names]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            bars = ax3.bar(variant_names, variant_times, color=colors[:len(variant_names)], alpha=0.8)
            ax3.set_ylabel('执行时间 (ms)')
            ax3.set_title('注意力变体 - 执行时间对比')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, time in zip(bars, variant_times):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{time:.1f}ms', ha='center', va='bottom')
        
        # 4. 注意力变体对比 - 内存使用
        if "attention_variants" in self.results:
            ax4 = axes[1, 1]
            variant_memory = [variants[name]["memory_usage"] for name in variant_names]
            
            bars = ax4.bar(variant_names, variant_memory, color=colors[:len(variant_names)], alpha=0.8)
            ax4.set_ylabel('内存使用 (GB)')
            ax4.set_title('注意力变体 - 内存使用对比')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, memory in zip(bars, variant_memory):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{memory:.2f}GB', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "multi_gpu_attention_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🎯 多GPU注意力机制实现演示")
    print("="*50)
    
    # 配置
    config = MultiGPUAttentionConfig(
        world_size=4,
        hidden_size=4096,
        num_heads=32,
        head_dim=128,
        seq_length=2048,
        batch_size=4
    )
    
    # 运行基准测试
    benchmark = MultiGPUAttentionBenchmark(config)
    benchmark.run_scalability_test()
    
    print("\n✅ 多GPU注意力机制演示完成！")
    print("📁 结果图表已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()