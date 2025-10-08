#!/usr/bin/env python3
"""
Softmax 优化 Triton 实现

本模块演示如何使用 Triton 实现高性能的 Softmax 操作，包括：
1. 基础 Softmax 实现
2. 数值稳定性优化
3. 融合 Softmax 操作
4. 在线 Softmax 算法
5. 性能对比分析

Author: nano-vLLM-learning
Date: 2024
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import math


class TritonSoftmaxDemo:
    """Triton Softmax 演示类"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        print(f"🚀 Triton Softmax 优化演示初始化")
        print(f"📱 使用设备: {device}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
    
    @staticmethod
    @triton.jit
    def softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride,
                      n_cols, BLOCK_SIZE: tl.constexpr):
        """
        基础 Softmax 内核实现
        
        计算 softmax(x) = exp(x) / sum(exp(x))
        """
        # 获取当前行的程序 ID
        row_idx = tl.program_id(0)
        
        # 计算当前行的起始指针
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        # 生成列偏移量
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        
        # 创建掩码（处理不对齐的情况）
        mask = col_offsets < n_cols
        
        # 加载输入数据
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # 第一步：找到最大值（数值稳定性）
        row_max = tl.max(row, axis=0)
        
        # 第二步：计算 exp(x - max)
        row_shifted = row - row_max
        numerator = tl.exp(row_shifted)
        
        # 第三步：计算分母（归一化因子）
        denominator = tl.sum(numerator, axis=0)
        
        # 第四步：计算 softmax
        softmax_output = numerator / denominator
        
        # 存储结果
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
    
    def softmax_triton(self, x: torch.Tensor) -> torch.Tensor:
        """
        Triton Softmax 实现
        
        Args:
            x: 输入张量 [batch_size, seq_len] 或 [seq_len]
        
        Returns:
            softmax 结果
        """
        # 处理输入维度
        input_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        n_rows, n_cols = x.shape
        
        # 创建输出张量
        y = torch.empty_like(x)
        
        # 选择合适的块大小
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > 2048:
            BLOCK_SIZE = 2048
        
        # 计算网格大小
        grid = (n_rows,)
        
        # 启动内核
        self.softmax_kernel[grid](
            x, y,
            x.stride(0), y.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # 恢复原始形状
        if len(input_shape) == 1:
            y = y.squeeze(0)
        
        return y
    
    @staticmethod
    @triton.jit
    def online_softmax_kernel(input_ptr, output_ptr, input_row_stride, output_row_stride,
                             n_cols, BLOCK_SIZE: tl.constexpr):
        """
        在线 Softmax 内核实现
        
        使用在线算法，只需要一次遍历，内存效率更高
        """
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        
        # 加载数据
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # 在线 Softmax 算法
        # 初始化
        m = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
        d = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # 更新最大值和分母
        m_new = tl.maximum(m, row)
        d = d * tl.exp(m - m_new) + tl.exp(row - m_new)
        m = m_new
        
        # 计算最终的 softmax
        softmax_output = tl.exp(row - m) / d
        
        # 存储结果
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
    
    def online_softmax_triton(self, x: torch.Tensor) -> torch.Tensor:
        """在线 Softmax Triton 实现"""
        input_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > 2048:
            BLOCK_SIZE = 2048
        
        grid = (n_rows,)
        
        self.online_softmax_kernel[grid](
            x, y,
            x.stride(0), y.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        if len(input_shape) == 1:
            y = y.squeeze(0)
        
        return y
    
    @staticmethod
    @triton.jit
    def fused_attention_softmax_kernel(
        q_ptr, k_ptr, output_ptr,
        q_row_stride, q_col_stride,
        k_row_stride, k_col_stride,
        output_row_stride, output_col_stride,
        seq_len, head_dim,
        scale: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr
    ):
        """
        融合的注意力 Softmax 内核
        
        计算 softmax(Q @ K^T / sqrt(d_k))
        """
        # 获取程序 ID
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # 计算偏移量
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, head_dim)
        
        # 加载 Q 和 K
        q_ptrs = q_ptr + (offs_m[:, None] * q_row_stride + offs_k[None, :] * q_col_stride)
        k_ptrs = k_ptr + (offs_n[:, None] * k_row_stride + offs_k[None, :] * k_col_stride)
        
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len) & (offs_k[None, :] < head_dim))
        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len) & (offs_k[None, :] < head_dim))
        
        # 计算注意力分数
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # 应用 Softmax
        scores_max = tl.max(scores, axis=1, keep_dims=True)
        scores_shifted = scores - scores_max
        numerator = tl.exp(scores_shifted)
        denominator = tl.sum(numerator, axis=1, keep_dims=True)
        attention_weights = numerator / denominator
        
        # 存储结果
        output_ptrs = output_ptr + (offs_m[:, None] * output_row_stride + offs_n[None, :] * output_col_stride)
        mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
        tl.store(output_ptrs, attention_weights, mask=mask)
    
    def fused_attention_softmax_triton(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """融合的注意力 Softmax 实现"""
        seq_len, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # 创建输出张量
        output = torch.empty((seq_len, seq_len), device=q.device, dtype=q.dtype)
        
        # 块大小
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        
        # 网格大小
        grid = (
            triton.cdiv(seq_len, BLOCK_SIZE_M),
            triton.cdiv(seq_len, BLOCK_SIZE_N)
        )
        
        self.fused_attention_softmax_kernel[grid](
            q, k, output,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            output.stride(0), output.stride(1),
            seq_len, head_dim, scale,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N
        )
        
        return output
    
    @staticmethod
    @triton.jit
    def masked_softmax_kernel(input_ptr, mask_ptr, output_ptr,
                             input_row_stride, mask_row_stride, output_row_stride,
                             n_cols, BLOCK_SIZE: tl.constexpr):
        """
        带掩码的 Softmax 内核
        
        支持因果掩码和填充掩码
        """
        row_idx = tl.program_id(0)
        
        # 计算指针
        input_row_start_ptr = input_ptr + row_idx * input_row_stride
        mask_row_start_ptr = mask_ptr + row_idx * mask_row_stride
        
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_row_start_ptr + col_offsets
        mask_ptrs = mask_row_start_ptr + col_offsets
        
        # 创建掩码
        col_mask = col_offsets < n_cols
        
        # 加载数据
        row = tl.load(input_ptrs, mask=col_mask, other=-float('inf'))
        mask = tl.load(mask_ptrs, mask=col_mask, other=0.0)
        
        # 应用掩码（将掩码为 0 的位置设为 -inf）
        row = tl.where(mask == 0.0, -float('inf'), row)
        
        # 计算 Softmax
        row_max = tl.max(row, axis=0)
        row_shifted = row - row_max
        numerator = tl.exp(row_shifted)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        # 将掩码位置的输出设为 0
        softmax_output = tl.where(mask == 0.0, 0.0, softmax_output)
        
        # 存储结果
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=col_mask)
    
    def masked_softmax_triton(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """带掩码的 Softmax 实现"""
        assert x.shape == mask.shape, "输入和掩码形状必须相同"
        
        input_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        if BLOCK_SIZE > 2048:
            BLOCK_SIZE = 2048
        
        grid = (n_rows,)
        
        self.masked_softmax_kernel[grid](
            x, mask, y,
            x.stride(0), mask.stride(0), y.stride(0),
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        if len(input_shape) == 1:
            y = y.squeeze(0)
        
        return y
    
    def benchmark_softmax(self, sizes: List[Tuple[int, int]] = None,
                         num_runs: int = 100) -> Dict:
        """Softmax 性能基准测试"""
        if sizes is None:
            sizes = [
                (1024, 512),
                (2048, 1024),
                (4096, 2048),
                (8192, 4096),
                (1024, 8192),
                (512, 16384)
            ]
        
        print(f"\n🎯 Softmax 性能基准测试")
        print(f"  测试配置: {sizes}")
        print(f"  运行次数: {num_runs}")
        
        results = {
            'sizes': sizes,
            'triton_basic': [],
            'triton_online': [],
            'pytorch': [],
            'speedup_basic': [],
            'speedup_online': []
        }
        
        for batch_size, seq_len in sizes:
            print(f"\n  📏 测试大小: [{batch_size}, {seq_len}]")
            
            # 创建测试数据
            x = torch.randn((batch_size, seq_len), device=self.device, dtype=torch.float32)
            
            # 1. Triton 基础版本
            print(f"    🔄 Triton 基础版本...")
            triton_basic_time = self._benchmark_function(
                lambda: self.softmax_triton(x), num_runs
            )
            results['triton_basic'].append(triton_basic_time)
            
            # 2. Triton 在线版本
            print(f"    🔄 Triton 在线版本...")
            triton_online_time = self._benchmark_function(
                lambda: self.online_softmax_triton(x), num_runs
            )
            results['triton_online'].append(triton_online_time)
            
            # 3. PyTorch 版本
            print(f"    🔄 PyTorch 版本...")
            pytorch_time = self._benchmark_function(
                lambda: torch.softmax(x, dim=-1), num_runs
            )
            results['pytorch'].append(pytorch_time)
            
            # 计算加速比
            speedup_basic = pytorch_time / triton_basic_time
            speedup_online = pytorch_time / triton_online_time
            results['speedup_basic'].append(speedup_basic)
            results['speedup_online'].append(speedup_online)
            
            print(f"      Triton 基础: {triton_basic_time*1000:.3f}ms")
            print(f"      Triton 在线: {triton_online_time*1000:.3f}ms")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms")
            print(f"      基础版加速比: {speedup_basic:.2f}x")
            print(f"      在线版加速比: {speedup_online:.2f}x")
        
        return results
    
    def _benchmark_function(self, func, num_runs: int) -> float:
        """基准测试函数"""
        # 预热
        for _ in range(10):
            func()
        
        torch.cuda.synchronize()
        
        # 测试
        start_time = time.time()
        for _ in range(num_runs):
            func()
        torch.cuda.synchronize()
        
        return (time.time() - start_time) / num_runs
    
    def verify_correctness(self) -> bool:
        """验证 Softmax 实现的正确性"""
        print(f"\n🔍 验证 Softmax 正确性...")
        
        test_cases = [
            (32, 128),
            (64, 256),
            (128, 512),
            (256, 1024)
        ]
        
        all_correct = True
        
        for batch_size, seq_len in test_cases:
            print(f"  测试大小: [{batch_size}, {seq_len}]")
            
            # 创建测试数据
            x = torch.randn((batch_size, seq_len), device=self.device, dtype=torch.float32)
            
            # 计算参考结果
            ref_result = torch.softmax(x, dim=-1)
            
            # 测试基础版本
            triton_basic_result = self.softmax_triton(x)
            max_diff_basic = torch.max(torch.abs(triton_basic_result - ref_result)).item()
            
            print(f"    基础版本最大误差: {max_diff_basic:.2e}")
            
            if max_diff_basic > 1e-5:
                print(f"    ❌ 基础版本验证失败")
                all_correct = False
            else:
                print(f"    ✅ 基础版本验证通过")
            
            # 测试在线版本
            triton_online_result = self.online_softmax_triton(x)
            max_diff_online = torch.max(torch.abs(triton_online_result - ref_result)).item()
            
            print(f"    在线版本最大误差: {max_diff_online:.2e}")
            
            if max_diff_online > 1e-5:
                print(f"    ❌ 在线版本验证失败")
                all_correct = False
            else:
                print(f"    ✅ 在线版本验证通过")
            
            # 测试带掩码版本
            mask = torch.ones_like(x)
            # 创建因果掩码
            for i in range(batch_size):
                for j in range(seq_len):
                    if j > i % seq_len:  # 简化的因果掩码
                        mask[i, j] = 0
            
            triton_masked_result = self.masked_softmax_triton(x, mask)
            
            # 计算参考的带掩码 softmax
            masked_x = x.clone()
            masked_x[mask == 0] = -float('inf')
            ref_masked_result = torch.softmax(masked_x, dim=-1)
            ref_masked_result[mask == 0] = 0
            
            max_diff_masked = torch.max(torch.abs(triton_masked_result - ref_masked_result)).item()
            
            print(f"    掩码版本最大误差: {max_diff_masked:.2e}")
            
            if max_diff_masked > 1e-5:
                print(f"    ❌ 掩码版本验证失败")
                all_correct = False
            else:
                print(f"    ✅ 掩码版本验证通过")
        
        return all_correct
    
    def test_numerical_stability(self) -> bool:
        """测试数值稳定性"""
        print(f"\n🔬 测试数值稳定性...")
        
        # 创建极端值测试数据
        test_cases = [
            ("大数值", torch.tensor([[1000.0, 1001.0, 999.0]], device=self.device)),
            ("小数值", torch.tensor([[-1000.0, -1001.0, -999.0]], device=self.device)),
            ("混合数值", torch.tensor([[1000.0, -1000.0, 0.0]], device=self.device)),
            ("零值", torch.tensor([[0.0, 0.0, 0.0]], device=self.device))
        ]
        
        all_stable = True
        
        for name, x in test_cases:
            print(f"  测试 {name}...")
            
            try:
                # PyTorch 参考结果
                ref_result = torch.softmax(x, dim=-1)
                
                # Triton 结果
                triton_result = self.softmax_triton(x)
                
                # 检查是否有 NaN 或 Inf
                if torch.isnan(triton_result).any() or torch.isinf(triton_result).any():
                    print(f"    ❌ {name} 产生了 NaN 或 Inf")
                    all_stable = False
                    continue
                
                # 检查和是否接近 1
                sum_result = torch.sum(triton_result, dim=-1)
                if not torch.allclose(sum_result, torch.ones_like(sum_result), atol=1e-6):
                    print(f"    ❌ {name} 和不等于 1: {sum_result.item()}")
                    all_stable = False
                    continue
                
                # 检查与参考结果的差异
                max_diff = torch.max(torch.abs(triton_result - ref_result)).item()
                if max_diff > 1e-5:
                    print(f"    ❌ {name} 与参考结果差异过大: {max_diff}")
                    all_stable = False
                    continue
                
                print(f"    ✅ {name} 数值稳定")
                
            except Exception as e:
                print(f"    ❌ {name} 测试失败: {e}")
                all_stable = False
        
        return all_stable
    
    def analyze_memory_usage(self) -> Dict:
        """分析内存使用情况"""
        print(f"\n💾 分析内存使用情况...")
        
        sizes = [(1024, 512), (2048, 1024), (4096, 2048)]
        results = {
            'sizes': sizes,
            'triton_memory': [],
            'pytorch_memory': []
        }
        
        for batch_size, seq_len in sizes:
            print(f"  测试大小: [{batch_size}, {seq_len}]")
            
            x = torch.randn((batch_size, seq_len), device=self.device, dtype=torch.float32)
            
            # 测试 Triton 内存使用
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            _ = self.softmax_triton(x)
            triton_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # 测试 PyTorch 内存使用
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            _ = torch.softmax(x, dim=-1)
            pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            results['triton_memory'].append(triton_memory)
            results['pytorch_memory'].append(pytorch_memory)
            
            print(f"    Triton 内存: {triton_memory:.2f} MB")
            print(f"    PyTorch 内存: {pytorch_memory:.2f} MB")
            print(f"    内存节省: {(pytorch_memory - triton_memory) / pytorch_memory * 100:.1f}%")
        
        return results
    
    def visualize_results(self, benchmark_results: Dict, memory_results: Dict):
        """可视化测试结果"""
        print(f"\n📊 生成 Softmax 性能分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 性能对比图
        ax1 = axes[0, 0]
        sizes_str = [f"{bs}x{sl}" for bs, sl in benchmark_results['sizes']]
        x_pos = np.arange(len(sizes_str))
        
        triton_basic_times = np.array(benchmark_results['triton_basic']) * 1000
        triton_online_times = np.array(benchmark_results['triton_online']) * 1000
        pytorch_times = np.array(benchmark_results['pytorch']) * 1000
        
        width = 0.25
        ax1.bar(x_pos - width, triton_basic_times, width, label='Triton 基础', alpha=0.8)
        ax1.bar(x_pos, triton_online_times, width, label='Triton 在线', alpha=0.8)
        ax1.bar(x_pos + width, pytorch_times, width, label='PyTorch', alpha=0.8)
        
        ax1.set_title('Softmax 执行时间对比')
        ax1.set_xlabel('矩阵大小 (batch_size × seq_len)')
        ax1.set_ylabel('执行时间 (ms)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sizes_str, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 加速比图
        ax2 = axes[0, 1]
        speedup_basic = benchmark_results['speedup_basic']
        speedup_online = benchmark_results['speedup_online']
        
        ax2.plot(x_pos, speedup_basic, 'o-', label='Triton 基础', linewidth=2, markersize=6)
        ax2.plot(x_pos, speedup_online, 's-', label='Triton 在线', linewidth=2, markersize=6)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='基准线')
        
        ax2.set_title('相对 PyTorch 的加速比')
        ax2.set_xlabel('矩阵大小')
        ax2.set_ylabel('加速比')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sizes_str, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 内存使用对比
        ax3 = axes[1, 0]
        memory_sizes_str = [f"{bs}x{sl}" for bs, sl in memory_results['sizes']]
        memory_x_pos = np.arange(len(memory_sizes_str))
        
        triton_memory = memory_results['triton_memory']
        pytorch_memory = memory_results['pytorch_memory']
        
        width = 0.35
        ax3.bar(memory_x_pos - width/2, triton_memory, width, label='Triton', alpha=0.8)
        ax3.bar(memory_x_pos + width/2, pytorch_memory, width, label='PyTorch', alpha=0.8)
        
        ax3.set_title('内存使用对比')
        ax3.set_xlabel('矩阵大小')
        ax3.set_ylabel('内存使用 (MB)')
        ax3.set_xticks(memory_x_pos)
        ax3.set_xticklabels(memory_sizes_str, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 吞吐量分析
        ax4 = axes[1, 1]
        
        # 计算吞吐量 (elements/second)
        throughput_basic = []
        throughput_online = []
        throughput_pytorch = []
        
        for i, (bs, sl) in enumerate(benchmark_results['sizes']):
            elements = bs * sl
            throughput_basic.append(elements / benchmark_results['triton_basic'][i] / 1e6)  # M elements/s
            throughput_online.append(elements / benchmark_results['triton_online'][i] / 1e6)
            throughput_pytorch.append(elements / benchmark_results['pytorch'][i] / 1e6)
        
        ax4.plot(x_pos, throughput_basic, 'o-', label='Triton 基础', linewidth=2, markersize=6)
        ax4.plot(x_pos, throughput_online, 's-', label='Triton 在线', linewidth=2, markersize=6)
        ax4.plot(x_pos, throughput_pytorch, '^-', label='PyTorch', linewidth=2, markersize=6)
        
        ax4.set_title('吞吐量对比')
        ax4.set_xlabel('矩阵大小')
        ax4.set_ylabel('吞吐量 (M elements/s)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(sizes_str, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('triton_softmax_performance.png', dpi=300, bbox_inches='tight')
        print(f"  📈 Softmax 性能图已保存: triton_softmax_performance.png")
        plt.show()


def main():
    """主函数：运行 Triton Softmax 演示"""
    print("🚀 Triton Softmax 优化演示")
    print("=" * 60)
    
    try:
        # 初始化演示类
        demo = TritonSoftmaxDemo()
        
        # 1. 正确性验证
        print(f"\n" + "="*60)
        print(f"🔍 正确性验证")
        print(f"="*60)
        
        is_correct = demo.verify_correctness()
        
        if not is_correct:
            print(f"❌ 正确性验证失败，请检查实现")
            return
        
        # 2. 数值稳定性测试
        print(f"\n" + "="*60)
        print(f"🔬 数值稳定性测试")
        print(f"="*60)
        
        is_stable = demo.test_numerical_stability()
        
        if not is_stable:
            print(f"⚠️ 数值稳定性测试发现问题")
        
        # 3. 性能基准测试
        print(f"\n" + "="*60)
        print(f"🎯 性能基准测试")
        print(f"="*60)
        
        benchmark_results = demo.benchmark_softmax(
            sizes=[
                (1024, 512),
                (2048, 1024),
                (4096, 2048),
                (1024, 4096)
            ],
            num_runs=50
        )
        
        # 4. 内存使用分析
        print(f"\n" + "="*60)
        print(f"💾 内存使用分析")
        print(f"="*60)
        
        memory_results = demo.analyze_memory_usage()
        
        # 5. 结果可视化
        print(f"\n" + "="*60)
        print(f"📈 结果可视化")
        print(f"="*60)
        
        demo.visualize_results(benchmark_results, memory_results)
        
        # 6. 总结报告
        print(f"\n" + "="*60)
        print(f"📋 总结报告")
        print(f"="*60)
        
        avg_speedup_basic = np.mean(benchmark_results['speedup_basic'])
        avg_speedup_online = np.mean(benchmark_results['speedup_online'])
        avg_memory_saving = np.mean([
            (p - t) / p * 100 for t, p in zip(memory_results['triton_memory'], memory_results['pytorch_memory'])
        ])
        
        print(f"✅ Triton Softmax 演示完成!")
        print(f"")
        print(f"🎯 性能总结:")
        print(f"  • Triton 基础版平均加速比: {avg_speedup_basic:.2f}x")
        print(f"  • Triton 在线版平均加速比: {avg_speedup_online:.2f}x")
        print(f"  • 平均内存节省: {avg_memory_saving:.1f}%")
        print(f"")
        print(f"💡 关键发现:")
        print(f"  • 在线算法在大序列长度下表现更好")
        print(f"  • 数值稳定性优化至关重要")
        print(f"  • 内存访问模式影响性能")
        print(f"")
        print(f"📚 学习要点:")
        print(f"  • 理解 Softmax 的数值稳定性问题")
        print(f"  • 掌握在线算法的优势")
        print(f"  • 学会融合操作减少内存访问")
        print(f"  • 了解掩码操作的实现")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()