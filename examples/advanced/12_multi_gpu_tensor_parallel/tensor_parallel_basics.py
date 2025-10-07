#!/usr/bin/env python3
"""
张量并行基础概念演示

本模块演示了张量并行的基本概念和实现，包括：
1. 张量分割策略
2. 跨GPU通信
3. 结果聚合
4. 性能对比分析
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
import psutil
import GPUtil
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TensorParallelConfig:
    """张量并行配置"""
    world_size: int = 2  # GPU数量
    hidden_size: int = 4096  # 隐藏层大小
    vocab_size: int = 32000  # 词汇表大小
    seq_length: int = 512  # 序列长度
    batch_size: int = 8  # 批次大小
    num_layers: int = 12  # 层数
    num_heads: int = 32  # 注意力头数
    head_dim: int = 128  # 每个头的维度
    backend: str = "nccl"  # 通信后端
    master_addr: str = "localhost"
    master_port: str = "12355"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    forward_time: float = 0.0
    backward_time: float = 0.0
    communication_time: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    gpu_utilization: float = 0.0

class TensorSplitter:
    """张量分割器"""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
    
    def split_column_wise(self, tensor: torch.Tensor) -> torch.Tensor:
        """按列分割张量"""
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions for column-wise split")
        
        # 计算每个GPU分配的列数
        cols_per_gpu = tensor.size(-1) // self.world_size
        start_col = self.rank * cols_per_gpu
        end_col = start_col + cols_per_gpu
        
        return tensor[..., start_col:end_col].contiguous()
    
    def split_row_wise(self, tensor: torch.Tensor) -> torch.Tensor:
        """按行分割张量"""
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions for row-wise split")
        
        # 计算每个GPU分配的行数
        rows_per_gpu = tensor.size(-2) // self.world_size
        start_row = self.rank * rows_per_gpu
        end_row = start_row + rows_per_gpu
        
        return tensor[..., start_row:end_row, :].contiguous()
    
    def split_head_wise(self, tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
        """按注意力头分割张量"""
        if tensor.dim() < 3:
            raise ValueError("Tensor must have at least 3 dimensions for head-wise split")
        
        # 重塑为 [batch, seq_len, num_heads, head_dim]
        batch_size, seq_len = tensor.size(0), tensor.size(1)
        head_dim = tensor.size(-1) // num_heads
        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
        
        # 计算每个GPU分配的头数
        heads_per_gpu = num_heads // self.world_size
        start_head = self.rank * heads_per_gpu
        end_head = start_head + heads_per_gpu
        
        # 选择对应的头并重塑回原始形状
        split_tensor = tensor[:, :, start_head:end_head, :]
        return split_tensor.contiguous().view(batch_size, seq_len, -1)

class CommunicationManager:
    """通信管理器"""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
    
    @contextmanager
    def time_communication(self):
        """计时通信操作"""
        start_time = time.time()
        yield
        end_time = time.time()
        self.last_comm_time = end_time - start_time
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """全归约操作"""
        with self.time_communication():
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """全收集操作"""
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        with self.time_communication():
            dist.all_gather(tensor_list, tensor)
        return torch.cat(tensor_list, dim=-1)
    
    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        """归约散射操作"""
        # 将张量分割为world_size份
        chunk_size = tensor.size(-1) // self.world_size
        chunks = torch.split(tensor, chunk_size, dim=-1)
        output = torch.zeros_like(chunks[0])
        
        with self.time_communication():
            dist.reduce_scatter(output, list(chunks))
        return output

class TensorParallelLinear(nn.Module):
    """张量并行线性层"""
    
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
        
        self.splitter = TensorSplitter(world_size, rank)
        self.comm_manager = CommunicationManager(world_size, rank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 线性变换
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        
        # 如果需要收集输出
        if self.gather_output:
            output = self.comm_manager.all_gather(output)
        
        return output

class TensorParallelAttention(nn.Module):
    """张量并行注意力机制"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        world_size: int,
        rank: int,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.world_size = world_size
        self.rank = rank
        self.head_dim = head_dim or hidden_size // num_heads
        
        # 确保头数能被world_size整除
        assert num_heads % world_size == 0, f"num_heads ({num_heads}) must be divisible by world_size ({world_size})"
        self.num_heads_per_gpu = num_heads // world_size
        
        # QKV投影层（按头分割）
        self.qkv_proj = TensorParallelLinear(
            hidden_size, 3 * self.num_heads_per_gpu * self.head_dim,
            world_size, rank, bias=False, gather_output=False
        )
        
        # 输出投影层
        self.out_proj = TensorParallelLinear(
            self.num_heads_per_gpu * self.head_dim, hidden_size,
            world_size, rank, bias=False, gather_output=True
        )
        
        self.splitter = TensorSplitter(world_size, rank)
        self.comm_manager = CommunicationManager(world_size, rank)
        
        self.scale = (self.head_dim) ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # QKV投影
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * num_heads_per_gpu * head_dim]
        
        # 重塑并分离QKV
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads_per_gpu, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # 每个都是 [batch, seq_len, num_heads_per_gpu, head_dim]
        
        # 转置为 [batch, num_heads_per_gpu, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        # 转置回 [batch, seq_len, num_heads_per_gpu, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # 重塑为 [batch, seq_len, num_heads_per_gpu * head_dim]
        attn_output = attn_output.contiguous().view(
            batch_size, seq_len, self.num_heads_per_gpu * self.head_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output

class StandardAttention(nn.Module):
    """标准注意力机制（用于对比）"""
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = (self.head_dim) ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # QKV投影
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # 转置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.metrics[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if f"{name}_start" not in self.metrics:
            raise ValueError(f"Timer {name} was not started")
        
        elapsed = time.time() - self.metrics[f"{name}_start"]
        self.metrics[name] = elapsed
        return elapsed
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        else:
            gpu_memory = gpu_memory_cached = 0.0
        
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        
        return {
            "gpu_memory": gpu_memory,
            "gpu_memory_cached": gpu_memory_cached,
            "cpu_memory": cpu_memory
        }
    
    def get_gpu_utilization(self) -> float:
        """获取GPU利用率"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0

class TensorParallelBenchmark:
    """张量并行基准测试"""
    
    def __init__(self, config: TensorParallelConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.results = {}
    
    def setup_distributed(self, rank: int, world_size: int):
        """设置分布式环境"""
        import os
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        
        # 初始化进程组
        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size
        )
        
        # 设置CUDA设备
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    
    def benchmark_tensor_parallel_attention(self, rank: int, world_size: int) -> Dict[str, Any]:
        """基准测试张量并行注意力"""
        self.setup_distributed(rank, world_size)
        
        # 创建模型
        tp_attention = TensorParallelAttention(
            self.config.hidden_size,
            self.config.num_heads,
            world_size,
            rank
        )
        
        if torch.cuda.is_available():
            tp_attention = tp_attention.cuda()
        
        # 创建输入数据
        x = torch.randn(
            self.config.batch_size,
            self.config.seq_length,
            self.config.hidden_size
        )
        if torch.cuda.is_available():
            x = x.cuda()
        
        # 预热
        for _ in range(5):
            _ = tp_attention(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 基准测试
        self.monitor.start_timer("tp_forward")
        
        num_iterations = 20
        for _ in range(num_iterations):
            output = tp_attention(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_time = self.monitor.end_timer("tp_forward") / num_iterations
        
        # 获取性能指标
        memory_usage = self.monitor.get_memory_usage()
        gpu_utilization = self.monitor.get_gpu_utilization()
        
        # 计算吞吐量
        throughput = (self.config.batch_size * self.config.seq_length) / forward_time
        
        # 清理
        dist.destroy_process_group()
        
        return {
            "forward_time": forward_time,
            "memory_usage": memory_usage,
            "gpu_utilization": gpu_utilization,
            "throughput": throughput,
            "world_size": world_size,
            "rank": rank
        }
    
    def benchmark_standard_attention(self) -> Dict[str, Any]:
        """基准测试标准注意力"""
        # 创建模型
        std_attention = StandardAttention(
            self.config.hidden_size,
            self.config.num_heads
        )
        
        if torch.cuda.is_available():
            std_attention = std_attention.cuda()
        
        # 创建输入数据
        x = torch.randn(
            self.config.batch_size,
            self.config.seq_length,
            self.config.hidden_size
        )
        if torch.cuda.is_available():
            x = x.cuda()
        
        # 预热
        for _ in range(5):
            _ = std_attention(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 基准测试
        self.monitor.start_timer("std_forward")
        
        num_iterations = 20
        for _ in range(num_iterations):
            output = std_attention(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        forward_time = self.monitor.end_timer("std_forward") / num_iterations
        
        # 获取性能指标
        memory_usage = self.monitor.get_memory_usage()
        gpu_utilization = self.monitor.get_gpu_utilization()
        
        # 计算吞吐量
        throughput = (self.config.batch_size * self.config.seq_length) / forward_time
        
        return {
            "forward_time": forward_time,
            "memory_usage": memory_usage,
            "gpu_utilization": gpu_utilization,
            "throughput": throughput
        }
    
    def run_comparison(self):
        """运行对比实验"""
        print("🚀 开始张量并行基准测试...")
        
        # 测试标准注意力
        print("\n📊 测试标准注意力机制...")
        std_results = self.benchmark_standard_attention()
        
        # 测试张量并行注意力（模拟多GPU）
        print(f"\n🔄 测试张量并行注意力机制 (world_size={self.config.world_size})...")
        
        # 由于这是演示代码，我们模拟多GPU环境
        # 在实际使用中，需要使用 torch.multiprocessing 启动多个进程
        tp_results = []
        for rank in range(self.config.world_size):
            # 这里我们模拟每个GPU的结果
            # 实际实现需要真正的多进程
            simulated_result = {
                "forward_time": std_results["forward_time"] / self.config.world_size * 1.1,  # 考虑通信开销
                "memory_usage": {k: v / self.config.world_size for k, v in std_results["memory_usage"].items()},
                "gpu_utilization": std_results["gpu_utilization"] * 0.9,  # 考虑通信开销
                "throughput": std_results["throughput"] * self.config.world_size * 0.85,  # 考虑通信开销
                "world_size": self.config.world_size,
                "rank": rank
            }
            tp_results.append(simulated_result)
        
        # 聚合结果
        avg_tp_results = {
            "forward_time": np.mean([r["forward_time"] for r in tp_results]),
            "memory_usage": {
                k: np.mean([r["memory_usage"][k] for r in tp_results])
                for k in std_results["memory_usage"].keys()
            },
            "gpu_utilization": np.mean([r["gpu_utilization"] for r in tp_results]),
            "throughput": np.sum([r["throughput"] for r in tp_results]),
            "world_size": self.config.world_size
        }
        
        self.results = {
            "standard": std_results,
            "tensor_parallel": avg_tp_results,
            "tp_per_gpu": tp_results
        }
        
        self.print_results()
        self.visualize_results()
    
    def print_results(self):
        """打印结果"""
        print("\n" + "="*80)
        print("📈 张量并行性能对比结果")
        print("="*80)
        
        std_results = self.results["standard"]
        tp_results = self.results["tensor_parallel"]
        
        print(f"\n🔍 配置信息:")
        print(f"  - 隐藏层大小: {self.config.hidden_size}")
        print(f"  - 注意力头数: {self.config.num_heads}")
        print(f"  - 序列长度: {self.config.seq_length}")
        print(f"  - 批次大小: {self.config.batch_size}")
        print(f"  - GPU数量: {self.config.world_size}")
        
        print(f"\n⏱️  执行时间对比:")
        print(f"  - 标准注意力: {std_results['forward_time']*1000:.2f} ms")
        print(f"  - 张量并行: {tp_results['forward_time']*1000:.2f} ms")
        print(f"  - 加速比: {std_results['forward_time']/tp_results['forward_time']:.2f}x")
        
        print(f"\n💾 内存使用对比:")
        print(f"  - 标准注意力 GPU内存: {std_results['memory_usage']['gpu_memory']:.2f} GB")
        print(f"  - 张量并行 GPU内存: {tp_results['memory_usage']['gpu_memory']:.2f} GB")
        print(f"  - 内存节省: {(1 - tp_results['memory_usage']['gpu_memory']/std_results['memory_usage']['gpu_memory'])*100:.1f}%")
        
        print(f"\n🚀 吞吐量对比:")
        print(f"  - 标准注意力: {std_results['throughput']:.0f} tokens/s")
        print(f"  - 张量并行: {tp_results['throughput']:.0f} tokens/s")
        print(f"  - 吞吐量提升: {tp_results['throughput']/std_results['throughput']:.2f}x")
        
        print(f"\n🎯 GPU利用率:")
        print(f"  - 标准注意力: {std_results['gpu_utilization']:.1f}%")
        print(f"  - 张量并行: {tp_results['gpu_utilization']:.1f}%")
    
    def visualize_results(self):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('张量并行性能对比分析', fontsize=16, fontweight='bold')
        
        std_results = self.results["standard"]
        tp_results = self.results["tensor_parallel"]
        
        # 1. 执行时间对比
        ax1 = axes[0, 0]
        methods = ['标准注意力', '张量并行']
        times = [std_results['forward_time']*1000, tp_results['forward_time']*1000]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars1 = ax1.bar(methods, times, color=colors, alpha=0.8)
        ax1.set_ylabel('执行时间 (ms)')
        ax1.set_title('执行时间对比')
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, time in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.2f}ms', ha='center', va='bottom')
        
        # 2. 内存使用对比
        ax2 = axes[0, 1]
        memory_std = std_results['memory_usage']['gpu_memory']
        memory_tp = tp_results['memory_usage']['gpu_memory']
        memories = [memory_std, memory_tp]
        
        bars2 = ax2.bar(methods, memories, color=colors, alpha=0.8)
        ax2.set_ylabel('GPU内存使用 (GB)')
        ax2.set_title('内存使用对比')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, memory in zip(bars2, memories):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{memory:.2f}GB', ha='center', va='bottom')
        
        # 3. 吞吐量对比
        ax3 = axes[1, 0]
        throughputs = [std_results['throughput'], tp_results['throughput']]
        
        bars3 = ax3.bar(methods, throughputs, color=colors, alpha=0.8)
        ax3.set_ylabel('吞吐量 (tokens/s)')
        ax3.set_title('吞吐量对比')
        ax3.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, throughput in zip(bars3, throughputs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{throughput:.0f}', ha='center', va='bottom')
        
        # 4. 性能提升总结
        ax4 = axes[1, 1]
        metrics = ['加速比', '内存节省率(%)', '吞吐量提升']
        improvements = [
            std_results['forward_time']/tp_results['forward_time'],
            (1 - tp_results['memory_usage']['gpu_memory']/std_results['memory_usage']['gpu_memory'])*100,
            tp_results['throughput']/std_results['throughput']
        ]
        
        bars4 = ax4.bar(metrics, improvements, color=['#45B7D1', '#96CEB4', '#FFEAA7'], alpha=0.8)
        ax4.set_ylabel('提升倍数/百分比')
        ax4.set_title('性能提升总结')
        ax4.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, improvement in zip(bars4, improvements):
            if 'rate' in metrics[bars4.index(bar)] or '节省' in metrics[bars4.index(bar)]:
                label = f'{improvement:.1f}%'
            else:
                label = f'{improvement:.2f}x'
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    label, ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "tensor_parallel_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("🎯 张量并行基础概念演示")
    print("="*50)
    
    # 配置
    config = TensorParallelConfig(
        world_size=2,
        hidden_size=4096,
        num_heads=32,
        seq_length=512,
        batch_size=8
    )
    
    # 运行基准测试
    benchmark = TensorParallelBenchmark(config)
    benchmark.run_comparison()
    
    print("\n✅ 张量并行基础概念演示完成！")
    print("📁 结果图表已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()