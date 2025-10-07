"""
Flash Attention 基础概念演示
对比标准注意力和 Flash Attention 的性能差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import gc
import psutil
import logging
from dataclasses import dataclass
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """注意力配置"""
    batch_size: int = 4
    num_heads: int = 8
    head_dim: int = 64
    seq_len: int = 1024
    dropout: float = 0.0
    scale: Optional[float] = None
    
    def __post_init__(self):
        if self.scale is None:
            self.scale = 1.0 / math.sqrt(self.head_dim)

class StandardAttention(nn.Module):
    """标准注意力实现"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.scale
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        标准注意力前向传播
        
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, head_dim]
            mask: [batch_size, num_heads, seq_len, seq_len] 或 None
        
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        # 计算注意力分数 - O(N²) 内存
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax - 需要存储完整的注意力矩阵
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights

class FlashAttentionNaive(nn.Module):
    """朴素的 Flash Attention 实现（用于教学）"""
    
    def __init__(self, config: AttentionConfig, block_size: int = 64):
        super().__init__()
        self.config = config
        self.scale = config.scale
        self.block_size = block_size
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Flash Attention 前向传播（简化版）
        
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, head_dim]
            mask: [batch_size, num_heads, seq_len, seq_len] 或 None
        
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        B, H, N, D = Q.shape
        
        # 初始化输出和统计量
        O = torch.zeros_like(Q)
        l = torch.zeros(B, H, N, 1, device=Q.device)  # 行和
        m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)  # 行最大值
        
        # 分块处理
        for j in range(0, N, self.block_size):
            # 获取 K, V 块
            K_j = K[:, :, j:j+self.block_size, :]
            V_j = V[:, :, j:j+self.block_size, :]
            
            for i in range(0, N, self.block_size):
                # 获取 Q 块
                Q_i = Q[:, :, i:i+self.block_size, :]
                
                # 计算注意力分数
                S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * self.scale
                
                if mask is not None:
                    mask_ij = mask[:, :, i:i+self.block_size, j:j+self.block_size]
                    S_ij = S_ij.masked_fill(mask_ij == 0, float('-inf'))
                
                # 在线 Softmax 更新
                m_i_old = m[:, :, i:i+self.block_size, :]
                l_i_old = l[:, :, i:i+self.block_size, :]
                O_i_old = O[:, :, i:i+self.block_size, :]
                
                # 计算新的最大值
                m_ij = torch.max(S_ij, dim=-1, keepdim=True)[0]
                m_i_new = torch.maximum(m_i_old, m_ij)
                
                # 计算概率
                P_ij = torch.exp(S_ij - m_i_new)
                
                # 更新行和
                l_i_new = torch.exp(m_i_old - m_i_new) * l_i_old + \
                         torch.sum(P_ij, dim=-1, keepdim=True)
                
                # 更新输出
                O_i_new = (torch.exp(m_i_old - m_i_new) * l_i_old * O_i_old + \
                          torch.matmul(P_ij, V_j)) / l_i_new
                
                # 保存更新的值
                O[:, :, i:i+self.block_size, :] = O_i_new
                l[:, :, i:i+self.block_size, :] = l_i_new
                m[:, :, i:i+self.block_size, :] = m_i_new
        
        return O, None  # Flash Attention 不返回注意力权重

class FlashAttentionBasicsDemo:
    """Flash Attention 基础演示"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        
    @contextmanager
    def memory_monitor(self, name: str):
        """内存监控上下文管理器"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            start_memory = 0
        
        start_time = time.time()
        
        yield
        
        end_time = time.time()
        
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0
            peak_memory = 0
        
        self.results[name] = {
            "time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "peak_memory": peak_memory,
            "start_memory": start_memory,
            "end_memory": end_memory
        }
    
    def create_sample_inputs(self, config: AttentionConfig) -> Tuple[torch.Tensor, ...]:
        """创建示例输入"""
        Q = torch.randn(config.batch_size, config.num_heads, 
                       config.seq_len, config.head_dim, device=self.device)
        K = torch.randn(config.batch_size, config.num_heads, 
                       config.seq_len, config.head_dim, device=self.device)
        V = torch.randn(config.batch_size, config.num_heads, 
                       config.seq_len, config.head_dim, device=self.device)
        
        # 创建因果掩码
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len, device=self.device))
        mask = mask.unsqueeze(0).unsqueeze(0).expand(config.batch_size, config.num_heads, -1, -1)
        
        return Q, K, V, mask
    
    def compare_attention_implementations(self, seq_lengths: List[int] = [256, 512, 1024, 2048]):
        """对比不同注意力实现"""
        logger.info("开始注意力实现对比...")
        
        comparison_results = {}
        
        for seq_len in seq_lengths:
            logger.info(f"测试序列长度: {seq_len}")
            
            config = AttentionConfig(
                batch_size=2,
                num_heads=8,
                head_dim=64,
                seq_len=seq_len
            )
            
            # 创建输入
            Q, K, V, mask = self.create_sample_inputs(config)
            
            # 测试标准注意力
            standard_attn = StandardAttention(config).to(self.device)
            standard_attn.eval()
            
            try:
                with torch.no_grad():
                    with self.memory_monitor(f"standard_{seq_len}"):
                        standard_output, standard_weights = standard_attn(Q, K, V, mask)
                
                standard_success = True
            except RuntimeError as e:
                logger.warning(f"标准注意力在序列长度 {seq_len} 失败: {e}")
                standard_success = False
                standard_output = None
            
            # 测试 Flash Attention
            flash_attn = FlashAttentionNaive(config, block_size=64).to(self.device)
            flash_attn.eval()
            
            try:
                with torch.no_grad():
                    with self.memory_monitor(f"flash_{seq_len}"):
                        flash_output, _ = flash_attn(Q, K, V, mask)
                
                flash_success = True
            except RuntimeError as e:
                logger.warning(f"Flash Attention 在序列长度 {seq_len} 失败: {e}")
                flash_success = False
                flash_output = None
            
            # 验证数值正确性
            numerical_error = None
            if standard_success and flash_success:
                numerical_error = torch.mean(torch.abs(standard_output - flash_output)).item()
            
            comparison_results[seq_len] = {
                "standard_success": standard_success,
                "flash_success": flash_success,
                "numerical_error": numerical_error,
                "standard_stats": self.results.get(f"standard_{seq_len}", {}),
                "flash_stats": self.results.get(f"flash_{seq_len}", {})
            }
            
            # 清理内存
            del Q, K, V, mask
            if standard_output is not None:
                del standard_output
            if flash_output is not None:
                del flash_output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["comparison"] = comparison_results
        return comparison_results
    
    def analyze_memory_complexity(self, seq_lengths: List[int] = [128, 256, 512, 1024]):
        """分析内存复杂度"""
        logger.info("开始内存复杂度分析...")
        
        memory_analysis = {}
        
        for seq_len in seq_lengths:
            config = AttentionConfig(
                batch_size=1,
                num_heads=1,
                head_dim=64,
                seq_len=seq_len
            )
            
            # 理论内存使用
            theoretical_standard = seq_len * seq_len * 4  # float32, 注意力矩阵
            theoretical_flash = seq_len * 64 * 4  # 分块大小 64
            
            # 实际测试
            Q, K, V, mask = self.create_sample_inputs(config)
            
            # 标准注意力内存使用
            try:
                standard_attn = StandardAttention(config).to(self.device)
                with torch.no_grad():
                    with self.memory_monitor(f"memory_standard_{seq_len}"):
                        _ = standard_attn(Q, K, V, mask)
                standard_memory = self.results[f"memory_standard_{seq_len}"]["peak_memory"]
            except:
                standard_memory = float('inf')
            
            # Flash Attention 内存使用
            try:
                flash_attn = FlashAttentionNaive(config, block_size=64).to(self.device)
                with torch.no_grad():
                    with self.memory_monitor(f"memory_flash_{seq_len}"):
                        _ = flash_attn(Q, K, V, mask)
                flash_memory = self.results[f"memory_flash_{seq_len}"]["peak_memory"]
            except:
                flash_memory = float('inf')
            
            memory_analysis[seq_len] = {
                "theoretical_standard": theoretical_standard,
                "theoretical_flash": theoretical_flash,
                "actual_standard": standard_memory,
                "actual_flash": flash_memory,
                "memory_ratio": flash_memory / standard_memory if standard_memory > 0 else float('inf')
            }
            
            # 清理
            del Q, K, V, mask
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["memory_analysis"] = memory_analysis
        return memory_analysis
    
    def benchmark_block_sizes(self, seq_len: int = 1024, 
                            block_sizes: List[int] = [32, 64, 128, 256]):
        """基准测试不同块大小"""
        logger.info("开始块大小基准测试...")
        
        config = AttentionConfig(
            batch_size=2,
            num_heads=8,
            head_dim=64,
            seq_len=seq_len
        )
        
        Q, K, V, mask = self.create_sample_inputs(config)
        
        block_size_results = {}
        
        for block_size in block_sizes:
            if block_size > seq_len:
                continue
                
            logger.info(f"测试块大小: {block_size}")
            
            flash_attn = FlashAttentionNaive(config, block_size=block_size).to(self.device)
            flash_attn.eval()
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    try:
                        _ = flash_attn(Q, K, V, mask)
                    except:
                        break
            
            # 基准测试
            times = []
            for _ in range(10):
                try:
                    with torch.no_grad():
                        start_time = time.time()
                        _ = flash_attn(Q, K, V, mask)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        end_time = time.time()
                        times.append(end_time - start_time)
                except:
                    times.append(float('inf'))
            
            valid_times = [t for t in times if t != float('inf')]
            
            block_size_results[block_size] = {
                "mean_time": np.mean(valid_times) if valid_times else float('inf'),
                "std_time": np.std(valid_times) if valid_times else 0,
                "success_rate": len(valid_times) / len(times)
            }
        
        self.results["block_sizes"] = block_size_results
        return block_size_results
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            logger.warning("没有结果可以可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Flash Attention vs 标准注意力对比", fontsize=16)
        
        # 1. 内存使用对比
        if "comparison" in self.results:
            ax = axes[0, 0]
            data = self.results["comparison"]
            
            seq_lens = []
            standard_memory = []
            flash_memory = []
            
            for seq_len, stats in data.items():
                if stats["standard_success"] and stats["flash_success"]:
                    seq_lens.append(seq_len)
                    standard_memory.append(stats["standard_stats"].get("peak_memory", 0) / 1e9)
                    flash_memory.append(stats["flash_stats"].get("peak_memory", 0) / 1e9)
            
            if seq_lens:
                x = np.arange(len(seq_lens))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, standard_memory, width, label="标准注意力", alpha=0.7)
                bars2 = ax.bar(x + width/2, flash_memory, width, label="Flash Attention", alpha=0.7)
                
                ax.set_xlabel("序列长度")
                ax.set_ylabel("峰值内存使用 (GB)")
                ax.set_title("内存使用对比")
                ax.set_xticks(x)
                ax.set_xticklabels(seq_lens)
                ax.legend()
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 执行时间对比
        if "comparison" in self.results:
            ax = axes[0, 1]
            data = self.results["comparison"]
            
            seq_lens = []
            standard_times = []
            flash_times = []
            
            for seq_len, stats in data.items():
                if stats["standard_success"] and stats["flash_success"]:
                    seq_lens.append(seq_len)
                    standard_times.append(stats["standard_stats"].get("time", 0))
                    flash_times.append(stats["flash_stats"].get("time", 0))
            
            if seq_lens:
                ax.plot(seq_lens, standard_times, 'o-', label="标准注意力", linewidth=2)
                ax.plot(seq_lens, flash_times, 's-', label="Flash Attention", linewidth=2)
                
                ax.set_xlabel("序列长度")
                ax.set_ylabel("执行时间 (秒)")
                ax.set_title("执行时间对比")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 3. 内存复杂度分析
        if "memory_analysis" in self.results:
            ax = axes[1, 0]
            data = self.results["memory_analysis"]
            
            seq_lens = list(data.keys())
            memory_ratios = [data[seq_len]["memory_ratio"] for seq_len in seq_lens 
                           if data[seq_len]["memory_ratio"] != float('inf')]
            valid_seq_lens = [seq_len for seq_len in seq_lens 
                            if data[seq_len]["memory_ratio"] != float('inf')]
            
            if valid_seq_lens:
                ax.plot(valid_seq_lens, memory_ratios, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel("序列长度")
                ax.set_ylabel("内存使用比率 (Flash/Standard)")
                ax.set_title("内存效率分析")
                ax.grid(True, alpha=0.3)
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="相等线")
                ax.legend()
        
        # 4. 块大小影响
        if "block_sizes" in self.results:
            ax = axes[1, 1]
            data = self.results["block_sizes"]
            
            block_sizes = list(data.keys())
            mean_times = [data[bs]["mean_time"] for bs in block_sizes 
                         if data[bs]["mean_time"] != float('inf')]
            valid_block_sizes = [bs for bs in block_sizes 
                               if data[bs]["mean_time"] != float('inf')]
            
            if valid_block_sizes:
                bars = ax.bar(range(len(valid_block_sizes)), mean_times, alpha=0.7)
                ax.set_xlabel("块大小")
                ax.set_ylabel("平均执行时间 (秒)")
                ax.set_title("块大小对性能的影响")
                ax.set_xticks(range(len(valid_block_sizes)))
                ax.set_xticklabels(valid_block_sizes)
                
                # 添加数值标签
                for bar, time_val in zip(bars, mean_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("flash_attention_basics_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细结果
        self.print_detailed_results()
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("Flash Attention 基础对比详细结果")
        print("="*80)
        
        if "comparison" in self.results:
            print("\n📊 性能对比总结:")
            data = self.results["comparison"]
            
            for seq_len, stats in data.items():
                print(f"\n序列长度 {seq_len}:")
                
                if stats["standard_success"] and stats["flash_success"]:
                    standard_time = stats["standard_stats"].get("time", 0)
                    flash_time = stats["flash_stats"].get("time", 0)
                    speedup = standard_time / flash_time if flash_time > 0 else 0
                    
                    standard_memory = stats["standard_stats"].get("peak_memory", 0) / 1e9
                    flash_memory = stats["flash_stats"].get("peak_memory", 0) / 1e9
                    memory_saving = (1 - flash_memory / standard_memory) * 100 if standard_memory > 0 else 0
                    
                    print(f"  ⏱️  执行时间: 标准={standard_time:.4f}s, Flash={flash_time:.4f}s, 加速比={speedup:.2f}x")
                    print(f"  💾 内存使用: 标准={standard_memory:.2f}GB, Flash={flash_memory:.2f}GB, 节省={memory_saving:.1f}%")
                    
                    if stats["numerical_error"] is not None:
                        print(f"  🔍 数值误差: {stats['numerical_error']:.2e}")
                else:
                    print(f"  ❌ 测试失败: 标准注意力={'成功' if stats['standard_success'] else '失败'}, "
                          f"Flash Attention={'成功' if stats['flash_success'] else '失败'}")
        
        if "memory_analysis" in self.results:
            print("\n💾 内存复杂度分析:")
            data = self.results["memory_analysis"]
            
            for seq_len, stats in data.items():
                if stats["memory_ratio"] != float('inf'):
                    print(f"  序列长度 {seq_len:4d}: 内存比率 = {stats['memory_ratio']:.3f}")
        
        if "block_sizes" in self.results:
            print("\n🔧 块大小优化:")
            data = self.results["block_sizes"]
            
            best_block_size = min(data.keys(), 
                                key=lambda x: data[x]["mean_time"] if data[x]["mean_time"] != float('inf') else float('inf'))
            
            for block_size, stats in data.items():
                if stats["mean_time"] != float('inf'):
                    marker = " ⭐" if block_size == best_block_size else ""
                    print(f"  块大小 {block_size:3d}: 平均时间={stats['mean_time']:.4f}s, "
                          f"成功率={stats['success_rate']*100:.1f}%{marker}")

def main():
    """主函数"""
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("⚠️  警告: 在 CPU 上运行，Flash Attention 的优势可能不明显")
    
    # 创建演示
    demo = FlashAttentionBasicsDemo(device)
    
    try:
        # 1. 对比不同注意力实现
        print("🔄 开始注意力实现对比...")
        demo.compare_attention_implementations([256, 512, 1024])
        
        # 2. 内存复杂度分析
        print("💾 开始内存复杂度分析...")
        demo.analyze_memory_complexity([128, 256, 512, 1024])
        
        # 3. 块大小基准测试
        print("🔧 开始块大小基准测试...")
        demo.benchmark_block_sizes(seq_len=1024, block_sizes=[32, 64, 128, 256])
        
        # 4. 可视化结果
        print("📊 生成可视化结果...")
        demo.visualize_results()
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Flash Attention 基础演示完成!")

if __name__ == "__main__":
    main()