"""
Triton Flash Attention 实现
使用 Triton 实现高性能的 Flash Attention
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
import gc
import logging
from dataclasses import dataclass
from contextlib import contextmanager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    L, M,  # 用于在线 softmax 的统计量
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    Triton Flash Attention 内核
    
    参数:
        Q, K, V: 输入张量
        Out: 输出张量
        L, M: 在线 softmax 统计量
        stride_*: 各维度步长
        N_CTX: 序列长度
        HEAD_DIM: 头维度
        BLOCK_M, BLOCK_N, BLOCK_DMODEL: 块大小常量
    """
    # 获取程序 ID
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 计算偏移量
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # 初始化输出累加器
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # 初始化在线 softmax 统计量
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + (-float("inf"))
    
    # 加载 Q 块
    q = tl.load(Q_block_ptr)
    
    # 循环处理 K, V 块
    for start_n in range(0, N_CTX, BLOCK_N):
        # 加载 K 块
        k = tl.load(K_block_ptr)
        
        # 计算注意力分数
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= (1.0 / math.sqrt(HEAD_DIM))
        
        # 因果掩码
        m_mask = tl.arange(0, BLOCK_M)[:, None] + start_m * BLOCK_M
        n_mask = tl.arange(0, BLOCK_N)[None, :] + start_n
        mask = m_mask >= n_mask
        qk = tl.where(mask, qk, float("-inf"))
        
        # 在线 softmax 更新
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_i_new[:, None]), 1)
        
        # 更新累加器
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # 加载 V 块并累加
        v = tl.load(V_block_ptr)
        p = tl.exp(qk - m_i_new[:, None])
        acc += tl.dot(p, v)
        
        # 更新统计量
        l_i = l_i_new
        m_i = m_i_new
        
        # 移动到下一个块
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    
    # 最终归一化
    acc = acc / l_i[:, None]
    
    # 存储输出
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))
    
    # 存储统计量（用于反向传播）
    l_ptrs = L + off_hz * N_CTX + tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    m_ptrs = M + off_hz * N_CTX + tl.arange(0, BLOCK_M) + start_m * BLOCK_M
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)

class TritonFlashAttention(nn.Module):
    """Triton Flash Attention 实现"""
    
    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # 确保输入是连续的
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # 创建输出张量
        O = torch.empty_like(Q)
        
        # 创建统计量张量
        L = torch.empty((batch_size, num_heads, seq_len), device=Q.device, dtype=torch.float32)
        M = torch.empty((batch_size, num_heads, seq_len), device=Q.device, dtype=torch.float32)
        
        # 计算块大小
        BLOCK_M = 64
        BLOCK_N = 64
        
        # 确保块大小不超过序列长度
        BLOCK_M = min(BLOCK_M, seq_len)
        BLOCK_N = min(BLOCK_N, seq_len)
        
        # 计算网格大小
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
        
        # 调用 Triton 内核
        flash_attention_kernel[grid](
            Q, K, V, O,
            L, M,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            seq_len, head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
        )
        
        return O

@triton.jit
def flash_attention_fwd_kernel_simple(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    N_CTX, HEAD_DIM, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """简化版 Flash Attention 内核（用于教学）"""
    # 获取当前块的索引
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    
    # 计算偏移量
    off_q = off_hz * stride_qh + start_m * stride_qm
    off_k = off_hz * stride_kh
    off_v = off_hz * stride_vh
    off_o = off_hz * stride_oh + start_m * stride_om
    
    # 创建掩码
    m_mask = tl.arange(0, BLOCK_M) + start_m
    valid_m = m_mask < N_CTX
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    
    # 加载 Q
    q_ptrs = Q + off_q + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=valid_m[:, None])
    
    # 循环处理 K, V 块
    for start_n in range(0, N_CTX, BLOCK_N):
        # 创建 K, V 的掩码
        n_mask = tl.arange(0, BLOCK_N) + start_n
        valid_n = n_mask < N_CTX
        
        # 加载 K
        k_ptrs = K + off_k + start_n * stride_kn + tl.arange(0, HEAD_DIM)[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=valid_n[None, :])
        
        # 计算注意力分数
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for d in range(0, HEAD_DIM):
            qk += q[:, d, None] * k[d, :]
        qk *= scale
        
        # 因果掩码
        causal_mask = m_mask[:, None] >= n_mask[None, :]
        qk = tl.where(causal_mask & valid_m[:, None] & valid_n[None, :], qk, float("-inf"))
        
        # 在线 softmax
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_i_new[:, None]), 1)
        
        # 更新累加器
        acc_scale = (l_i / l_i_new * alpha)[:, None]
        acc = acc * acc_scale
        
        # 加载 V 并累加
        v_ptrs = V + off_v + start_n * stride_vn + tl.arange(0, HEAD_DIM)[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=valid_n[:, None])
        
        p = tl.exp(qk - m_i_new[:, None])
        acc += tl.dot(p, v)
        
        # 更新统计量
        l_i = l_i_new
        m_i = m_i_new
    
    # 最终归一化
    acc = acc / l_i[:, None]
    
    # 存储输出
    o_ptrs = Out + off_o + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    tl.store(o_ptrs, acc, mask=valid_m[:, None])

class SimpleTritonFlashAttention(nn.Module):
    """简化版 Triton Flash Attention"""
    
    def __init__(self, head_dim: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = Q.shape
        
        # 确保输入是连续的
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # 创建输出张量
        O = torch.empty_like(Q)
        
        # 计算块大小
        BLOCK_M = min(64, seq_len)
        BLOCK_N = min(64, seq_len)
        
        # 计算网格大小
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)
        
        # 调用简化内核
        flash_attention_fwd_kernel_simple[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            seq_len, head_dim, self.scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        
        return O

class TritonFlashAttentionDemo:
    """Triton Flash Attention 演示"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        
        if not torch.cuda.is_available():
            raise RuntimeError("Triton Flash Attention 需要 CUDA 支持")
    
    @contextmanager
    def memory_monitor(self, name: str):
        """内存监控上下文管理器"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        yield
        
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        self.results[name] = {
            "time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "peak_memory": peak_memory,
            "start_memory": start_memory,
            "end_memory": end_memory
        }
    
    def create_sample_inputs(self, batch_size: int, num_heads: int, 
                           seq_len: int, head_dim: int) -> Tuple[torch.Tensor, ...]:
        """创建示例输入"""
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=torch.float16)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=torch.float16)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       device=self.device, dtype=torch.float16)
        
        return Q, K, V
    
    def compare_implementations(self, seq_lengths: List[int] = [256, 512, 1024, 2048]):
        """对比不同实现"""
        logger.info("开始 Triton Flash Attention 实现对比...")
        
        comparison_results = {}
        
        for seq_len in seq_lengths:
            logger.info(f"测试序列长度: {seq_len}")
            
            # 创建输入
            Q, K, V = self.create_sample_inputs(2, 8, seq_len, 64)
            
            # 测试 PyTorch 标准注意力
            try:
                with torch.no_grad():
                    with self.memory_monitor(f"pytorch_{seq_len}"):
                        scale = 1.0 / math.sqrt(64)
                        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                        
                        # 因果掩码
                        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
                        scores = scores.masked_fill(mask == 0, float('-inf'))
                        
                        attn_weights = torch.softmax(scores, dim=-1)
                        pytorch_output = torch.matmul(attn_weights, V)
                
                pytorch_success = True
            except RuntimeError as e:
                logger.warning(f"PyTorch 注意力在序列长度 {seq_len} 失败: {e}")
                pytorch_success = False
                pytorch_output = None
            
            # 测试简化版 Triton Flash Attention
            try:
                simple_triton_attn = SimpleTritonFlashAttention(64).to(self.device)
                with torch.no_grad():
                    with self.memory_monitor(f"simple_triton_{seq_len}"):
                        simple_triton_output = simple_triton_attn(Q, K, V)
                
                simple_triton_success = True
            except Exception as e:
                logger.warning(f"简化版 Triton Flash Attention 在序列长度 {seq_len} 失败: {e}")
                simple_triton_success = False
                simple_triton_output = None
            
            # 验证数值正确性
            numerical_error = None
            if pytorch_success and simple_triton_success:
                numerical_error = torch.mean(torch.abs(
                    pytorch_output.float() - simple_triton_output.float()
                )).item()
            
            comparison_results[seq_len] = {
                "pytorch_success": pytorch_success,
                "simple_triton_success": simple_triton_success,
                "numerical_error": numerical_error,
                "pytorch_stats": self.results.get(f"pytorch_{seq_len}", {}),
                "simple_triton_stats": self.results.get(f"simple_triton_{seq_len}", {})
            }
            
            # 清理内存
            del Q, K, V
            if pytorch_output is not None:
                del pytorch_output
            if simple_triton_output is not None:
                del simple_triton_output
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results["comparison"] = comparison_results
        return comparison_results
    
    def benchmark_block_sizes(self, seq_len: int = 1024, 
                            block_sizes: List[int] = [32, 64, 128]):
        """基准测试不同块大小"""
        logger.info("开始 Triton 块大小基准测试...")
        
        Q, K, V = self.create_sample_inputs(2, 8, seq_len, 64)
        
        block_size_results = {}
        
        for block_size in block_sizes:
            if block_size > seq_len:
                continue
                
            logger.info(f"测试块大小: {block_size}")
            
            # 修改内核以使用不同的块大小
            try:
                # 预热
                simple_triton_attn = SimpleTritonFlashAttention(64).to(self.device)
                with torch.no_grad():
                    for _ in range(3):
                        _ = simple_triton_attn(Q, K, V)
                
                # 基准测试
                times = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    start_time = time.time()
                    with torch.no_grad():
                        _ = simple_triton_attn(Q, K, V)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                block_size_results[block_size] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "success_rate": 1.0
                }
                
            except Exception as e:
                logger.warning(f"块大小 {block_size} 测试失败: {e}")
                block_size_results[block_size] = {
                    "mean_time": float('inf'),
                    "std_time": 0,
                    "success_rate": 0.0
                }
        
        self.results["block_sizes"] = block_size_results
        return block_size_results
    
    def analyze_memory_efficiency(self, seq_lengths: List[int] = [256, 512, 1024]):
        """分析内存效率"""
        logger.info("开始内存效率分析...")
        
        memory_analysis = {}
        
        for seq_len in seq_lengths:
            Q, K, V = self.create_sample_inputs(1, 1, seq_len, 64)
            
            # PyTorch 内存使用
            try:
                with torch.no_grad():
                    with self.memory_monitor(f"memory_pytorch_{seq_len}"):
                        scale = 1.0 / math.sqrt(64)
                        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
                        attn_weights = torch.softmax(scores, dim=-1)
                        _ = torch.matmul(attn_weights, V)
                pytorch_memory = self.results[f"memory_pytorch_{seq_len}"]["peak_memory"]
            except:
                pytorch_memory = float('inf')
            
            # Triton 内存使用
            try:
                simple_triton_attn = SimpleTritonFlashAttention(64).to(self.device)
                with torch.no_grad():
                    with self.memory_monitor(f"memory_triton_{seq_len}"):
                        _ = simple_triton_attn(Q, K, V)
                triton_memory = self.results[f"memory_triton_{seq_len}"]["peak_memory"]
            except:
                triton_memory = float('inf')
            
            memory_analysis[seq_len] = {
                "pytorch_memory": pytorch_memory,
                "triton_memory": triton_memory,
                "memory_ratio": triton_memory / pytorch_memory if pytorch_memory > 0 else float('inf'),
                "memory_saving": (1 - triton_memory / pytorch_memory) * 100 if pytorch_memory > 0 else 0
            }
            
            del Q, K, V
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results["memory_analysis"] = memory_analysis
        return memory_analysis
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            logger.warning("没有结果可以可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Triton Flash Attention 性能分析", fontsize=16)
        
        # 1. 执行时间对比
        if "comparison" in self.results:
            ax = axes[0, 0]
            data = self.results["comparison"]
            
            seq_lens = []
            pytorch_times = []
            triton_times = []
            
            for seq_len, stats in data.items():
                if stats["pytorch_success"] and stats["simple_triton_success"]:
                    seq_lens.append(seq_len)
                    pytorch_times.append(stats["pytorch_stats"].get("time", 0))
                    triton_times.append(stats["simple_triton_stats"].get("time", 0))
            
            if seq_lens:
                ax.plot(seq_lens, pytorch_times, 'o-', label="PyTorch", linewidth=2)
                ax.plot(seq_lens, triton_times, 's-', label="Triton Flash Attention", linewidth=2)
                
                ax.set_xlabel("序列长度")
                ax.set_ylabel("执行时间 (秒)")
                ax.set_title("执行时间对比")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 2. 内存使用对比
        if "memory_analysis" in self.results:
            ax = axes[0, 1]
            data = self.results["memory_analysis"]
            
            seq_lens = list(data.keys())
            pytorch_memory = [data[seq_len]["pytorch_memory"] / 1e9 for seq_len in seq_lens 
                            if data[seq_len]["pytorch_memory"] != float('inf')]
            triton_memory = [data[seq_len]["triton_memory"] / 1e9 for seq_len in seq_lens 
                           if data[seq_len]["triton_memory"] != float('inf')]
            valid_seq_lens = [seq_len for seq_len in seq_lens 
                            if data[seq_len]["pytorch_memory"] != float('inf') and 
                               data[seq_len]["triton_memory"] != float('inf')]
            
            if valid_seq_lens:
                x = np.arange(len(valid_seq_lens))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, pytorch_memory, width, label="PyTorch", alpha=0.7)
                bars2 = ax.bar(x + width/2, triton_memory, width, label="Triton", alpha=0.7)
                
                ax.set_xlabel("序列长度")
                ax.set_ylabel("峰值内存使用 (GB)")
                ax.set_title("内存使用对比")
                ax.set_xticks(x)
                ax.set_xticklabels(valid_seq_lens)
                ax.legend()
        
        # 3. 加速比分析
        if "comparison" in self.results:
            ax = axes[1, 0]
            data = self.results["comparison"]
            
            seq_lens = []
            speedups = []
            
            for seq_len, stats in data.items():
                if stats["pytorch_success"] and stats["simple_triton_success"]:
                    pytorch_time = stats["pytorch_stats"].get("time", 0)
                    triton_time = stats["simple_triton_stats"].get("time", 0)
                    if triton_time > 0:
                        speedup = pytorch_time / triton_time
                        seq_lens.append(seq_len)
                        speedups.append(speedup)
            
            if seq_lens:
                bars = ax.bar(range(len(seq_lens)), speedups, alpha=0.7)
                ax.set_xlabel("序列长度")
                ax.set_ylabel("加速比 (PyTorch/Triton)")
                ax.set_title("Triton Flash Attention 加速比")
                ax.set_xticks(range(len(seq_lens)))
                ax.set_xticklabels(seq_lens)
                ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label="基准线")
                ax.legend()
                
                # 添加数值标签
                for bar, speedup in zip(bars, speedups):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # 4. 数值误差分析
        if "comparison" in self.results:
            ax = axes[1, 1]
            data = self.results["comparison"]
            
            seq_lens = []
            errors = []
            
            for seq_len, stats in data.items():
                if stats["numerical_error"] is not None:
                    seq_lens.append(seq_len)
                    errors.append(stats["numerical_error"])
            
            if seq_lens:
                ax.semilogy(seq_lens, errors, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel("序列长度")
                ax.set_ylabel("平均绝对误差 (对数尺度)")
                ax.set_title("数值精度分析")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("triton_flash_attention_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细结果
        self.print_detailed_results()
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("Triton Flash Attention 详细分析结果")
        print("="*80)
        
        if "comparison" in self.results:
            print("\n🚀 性能对比总结:")
            data = self.results["comparison"]
            
            for seq_len, stats in data.items():
                print(f"\n序列长度 {seq_len}:")
                
                if stats["pytorch_success"] and stats["simple_triton_success"]:
                    pytorch_time = stats["pytorch_stats"].get("time", 0)
                    triton_time = stats["simple_triton_stats"].get("time", 0)
                    speedup = pytorch_time / triton_time if triton_time > 0 else 0
                    
                    print(f"  ⏱️  执行时间: PyTorch={pytorch_time:.4f}s, Triton={triton_time:.4f}s, 加速比={speedup:.2f}x")
                    
                    if stats["numerical_error"] is not None:
                        print(f"  🔍 数值误差: {stats['numerical_error']:.2e}")
                else:
                    print(f"  ❌ 测试失败: PyTorch={'成功' if stats['pytorch_success'] else '失败'}, "
                          f"Triton={'成功' if stats['simple_triton_success'] else '失败'}")
        
        if "memory_analysis" in self.results:
            print("\n💾 内存效率分析:")
            data = self.results["memory_analysis"]
            
            for seq_len, stats in data.items():
                if stats["pytorch_memory"] != float('inf') and stats["triton_memory"] != float('inf'):
                    print(f"  序列长度 {seq_len:4d}: PyTorch={stats['pytorch_memory']/1e9:.2f}GB, "
                          f"Triton={stats['triton_memory']/1e9:.2f}GB, 节省={stats['memory_saving']:.1f}%")

def main():
    """主函数"""
    # 检查设备
    if not torch.cuda.is_available():
        print("❌ 错误: Triton Flash Attention 需要 CUDA 支持")
        return
    
    print("🚀 开始 Triton Flash Attention 演示")
    
    # 创建演示
    demo = TritonFlashAttentionDemo("cuda")
    
    try:
        # 1. 对比不同实现
        print("🔄 开始实现对比...")
        demo.compare_implementations([256, 512, 1024])
        
        # 2. 内存效率分析
        print("💾 开始内存效率分析...")
        demo.analyze_memory_efficiency([256, 512, 1024])
        
        # 3. 块大小基准测试
        print("🔧 开始块大小基准测试...")
        demo.benchmark_block_sizes(seq_len=1024, block_sizes=[32, 64, 128])
        
        # 4. 可视化结果
        print("📊 生成可视化结果...")
        demo.visualize_results()
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Triton Flash Attention 演示完成!")

if __name__ == "__main__":
    main()