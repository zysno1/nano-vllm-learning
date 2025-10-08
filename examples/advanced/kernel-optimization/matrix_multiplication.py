#!/usr/bin/env python3
"""
高性能矩阵乘法 Triton 实现

本模块演示如何使用 Triton 实现高性能的矩阵乘法，包括：
1. 基础矩阵乘法实现
2. 分块优化策略
3. 内存访问优化
4. 与 PyTorch/cuBLAS 的性能对比

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


class TritonMatMulDemo:
    """Triton 矩阵乘法演示类"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        print(f"🚀 Triton 矩阵乘法演示初始化")
        print(f"📱 使用设备: {device}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
    
    @staticmethod
    @triton.jit
    def matmul_kernel(
        # 指针参数
        a_ptr, b_ptr, c_ptr,
        # 矩阵维度
        M, N, K,
        # 步长参数
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # 块大小（编译时常量）
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        # 激活函数选项
        ACTIVATION: tl.constexpr
    ):
        """
        高性能矩阵乘法内核
        
        计算 C = A @ B，其中：
        - A: [M, K]
        - B: [K, N]  
        - C: [M, N]
        """
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        
        # 计算当前块在 2D 网格中的位置
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        
        # 计算当前块的偏移量
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # 计算指针偏移
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # 主计算循环：沿 K 维度分块
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # 加载 A 和 B 的块
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            
            # 执行矩阵乘法累加
            accumulator += tl.dot(a, b)
            
            # 更新指针到下一个 K 块
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # 应用激活函数
        if ACTIVATION == "relu":
            accumulator = tl.maximum(accumulator, 0)
        elif ACTIVATION == "gelu":
            # 近似 GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            x = accumulator
            accumulator = 0.5 * x * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
        
        # 转换为输出数据类型
        c = accumulator.to(tl.float16)
        
        # 计算输出偏移
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        
        # 创建掩码并存储结果
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    
    def matmul_triton(self, a: torch.Tensor, b: torch.Tensor, 
                     activation: str = "none") -> torch.Tensor:
        """
        Triton 矩阵乘法实现
        
        Args:
            a: 输入矩阵 A [M, K]
            b: 输入矩阵 B [K, N]
            activation: 激活函数 ("none", "relu", "gelu")
        
        Returns:
            输出矩阵 C [M, N]
        """
        # 检查输入
        assert a.shape[1] == b.shape[0], f"矩阵维度不匹配: {a.shape} @ {b.shape}"
        assert a.is_contiguous(), "矩阵 A 必须是连续的"
        assert b.is_contiguous(), "矩阵 B 必须是连续的"
        
        M, K = a.shape
        K, N = b.shape
        
        # 创建输出张量
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        
        # 选择最优的块大小
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        # 计算网格大小
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        # 启动内核
        self.matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            ACTIVATION=activation
        )
        
        return c
    
    @staticmethod
    @triton.jit
    def matmul_kernel_optimized(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        """
        优化的矩阵乘法内核，使用更好的块调度策略
        """
        # 获取程序 ID
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        
        # 计算偏移量
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        # 计算指针
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # 主计算循环
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # 加载数据
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            
            # 矩阵乘法
            accumulator += tl.dot(a, b)
            
            # 更新指针
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        
        # 存储结果
        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
    
    def matmul_triton_optimized(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """优化的 Triton 矩阵乘法实现"""
        assert a.shape[1] == b.shape[0]
        assert a.is_contiguous() and b.is_contiguous()
        
        M, K = a.shape
        K, N = b.shape
        
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        
        # 优化的块大小
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        GROUP_SIZE_M = 8
        
        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
        
        self.matmul_kernel_optimized[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M
        )
        
        return c
    
    def benchmark_matmul(self, sizes: List[Tuple[int, int, int]] = None,
                        num_runs: int = 100) -> Dict:
        """矩阵乘法性能基准测试"""
        if sizes is None:
            sizes = [
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (512, 4096, 512),
                (2048, 512, 2048)
            ]
        
        print(f"\n🎯 矩阵乘法性能基准测试")
        print(f"  测试配置: {sizes}")
        print(f"  运行次数: {num_runs}")
        
        results = {
            'sizes': sizes,
            'triton_basic': [],
            'triton_optimized': [],
            'pytorch': [],
            'cublas': [],
            'speedup_basic': [],
            'speedup_optimized': []
        }
        
        for M, N, K in sizes:
            print(f"\n  📏 测试矩阵大小: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")
            
            # 创建测试数据
            a = torch.randn((M, K), device=self.device, dtype=torch.float16)
            b = torch.randn((K, N), device=self.device, dtype=torch.float16)
            
            # 1. Triton 基础版本
            print(f"    🔄 Triton 基础版本...")
            triton_basic_time = self._benchmark_function(
                lambda: self.matmul_triton(a, b), num_runs
            )
            results['triton_basic'].append(triton_basic_time)
            
            # 2. Triton 优化版本
            print(f"    🔄 Triton 优化版本...")
            triton_opt_time = self._benchmark_function(
                lambda: self.matmul_triton_optimized(a, b), num_runs
            )
            results['triton_optimized'].append(triton_opt_time)
            
            # 3. PyTorch 版本
            print(f"    🔄 PyTorch 版本...")
            pytorch_time = self._benchmark_function(
                lambda: torch.mm(a, b), num_runs
            )
            results['pytorch'].append(pytorch_time)
            
            # 4. cuBLAS 版本（通过 torch.mm 调用）
            print(f"    🔄 cuBLAS 版本...")
            with torch.backends.cudnn.flags(enabled=False):
                cublas_time = self._benchmark_function(
                    lambda: torch.mm(a, b), num_runs
                )
            results['cublas'].append(cublas_time)
            
            # 计算加速比
            speedup_basic = pytorch_time / triton_basic_time
            speedup_opt = pytorch_time / triton_opt_time
            results['speedup_basic'].append(speedup_basic)
            results['speedup_optimized'].append(speedup_opt)
            
            # 计算 TFLOPS
            flops = 2 * M * N * K  # 矩阵乘法的浮点运算数
            triton_basic_tflops = flops / (triton_basic_time * 1e12)
            triton_opt_tflops = flops / (triton_opt_time * 1e12)
            pytorch_tflops = flops / (pytorch_time * 1e12)
            
            print(f"      Triton 基础: {triton_basic_time*1000:.3f}ms ({triton_basic_tflops:.2f} TFLOPS)")
            print(f"      Triton 优化: {triton_opt_time*1000:.3f}ms ({triton_opt_tflops:.2f} TFLOPS)")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms ({pytorch_tflops:.2f} TFLOPS)")
            print(f"      基础版加速比: {speedup_basic:.2f}x")
            print(f"      优化版加速比: {speedup_opt:.2f}x")
        
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
        """验证矩阵乘法实现的正确性"""
        print(f"\n🔍 验证矩阵乘法正确性...")
        
        test_cases = [
            (128, 256, 128),
            (256, 512, 256),
            (512, 1024, 512)
        ]
        
        all_correct = True
        
        for M, K, N in test_cases:
            print(f"  测试大小: [{M}, {K}] @ [{K}, {N}]")
            
            # 创建测试数据
            a = torch.randn((M, K), device=self.device, dtype=torch.float16)
            b = torch.randn((K, N), device=self.device, dtype=torch.float16)
            
            # 计算参考结果
            ref_result = torch.mm(a, b)
            
            # 测试基础版本
            triton_basic_result = self.matmul_triton(a, b)
            max_diff_basic = torch.max(torch.abs(triton_basic_result - ref_result)).item()
            relative_diff_basic = max_diff_basic / torch.max(torch.abs(ref_result)).item()
            
            print(f"    基础版本相对误差: {relative_diff_basic:.2e}")
            
            if relative_diff_basic > 1e-2:  # float16 精度较低
                print(f"    ❌ 基础版本验证失败")
                all_correct = False
            else:
                print(f"    ✅ 基础版本验证通过")
            
            # 测试优化版本
            triton_opt_result = self.matmul_triton_optimized(a, b)
            max_diff_opt = torch.max(torch.abs(triton_opt_result - ref_result)).item()
            relative_diff_opt = max_diff_opt / torch.max(torch.abs(ref_result)).item()
            
            print(f"    优化版本相对误差: {relative_diff_opt:.2e}")
            
            if relative_diff_opt > 1e-2:
                print(f"    ❌ 优化版本验证失败")
                all_correct = False
            else:
                print(f"    ✅ 优化版本验证通过")
        
        return all_correct
    
    def analyze_block_size_performance(self) -> Dict:
        """分析不同块大小对性能的影响"""
        print(f"\n📊 分析块大小对矩阵乘法性能的影响")
        
        # 测试不同的块大小组合
        block_configs = [
            (64, 64, 32),
            (128, 128, 32),
            (256, 256, 32),
            (128, 64, 32),
            (64, 128, 32),
            (128, 128, 64)
        ]
        
        test_size = (2048, 2048, 2048)
        M, K, N = test_size
        
        print(f"  测试矩阵大小: [{M}, {K}] @ [{K}, {N}]")
        
        # 创建测试数据
        a = torch.randn((M, K), device=self.device, dtype=torch.float16)
        b = torch.randn((K, N), device=self.device, dtype=torch.float16)
        
        results = {
            'configs': block_configs,
            'times': [],
            'tflops': []
        }
        
        for bm, bn, bk in block_configs:
            print(f"    测试块大小: M={bm}, N={bn}, K={bk}")
            
            # 创建自定义内核
            @triton.jit
            def custom_matmul_kernel(
                a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
            ):
                pid = tl.program_id(axis=0)
                num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
                num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
                
                pid_m = pid // num_pid_n
                pid_n = pid % num_pid_n
                
                offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
                offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
                
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                
                for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    a_block = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                    b_block = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
                    accumulator += tl.dot(a_block, b_block)
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    b_ptrs += BLOCK_SIZE_K * stride_bk
                
                c = accumulator.to(tl.float16)
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)
            
            def matmul_with_config(a, b, bm, bn, bk):
                c = torch.empty((M, N), device=a.device, dtype=torch.float16)
                grid = lambda META: (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
                
                custom_matmul_kernel[grid](
                    a, b, c, M, N, K,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk
                )
                return c
            
            # 测试性能
            exec_time = self._benchmark_function(
                lambda: matmul_with_config(a, b, bm, bn, bk), 50
            )
            
            flops = 2 * M * N * K
            tflops = flops / (exec_time * 1e12)
            
            results['times'].append(exec_time)
            results['tflops'].append(tflops)
            
            print(f"      执行时间: {exec_time*1000:.3f}ms")
            print(f"      性能: {tflops:.2f} TFLOPS")
        
        return results
    
    def visualize_results(self, benchmark_results: Dict, block_analysis_results: Dict):
        """可视化测试结果"""
        print(f"\n📊 生成矩阵乘法性能分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 性能对比图
        ax1 = axes[0, 0]
        sizes_str = [f"{M}x{K}x{N}" for M, K, N in benchmark_results['sizes']]
        x_pos = np.arange(len(sizes_str))
        
        triton_basic_times = np.array(benchmark_results['triton_basic']) * 1000
        triton_opt_times = np.array(benchmark_results['triton_optimized']) * 1000
        pytorch_times = np.array(benchmark_results['pytorch']) * 1000
        
        width = 0.25
        ax1.bar(x_pos - width, triton_basic_times, width, label='Triton 基础', alpha=0.8)
        ax1.bar(x_pos, triton_opt_times, width, label='Triton 优化', alpha=0.8)
        ax1.bar(x_pos + width, pytorch_times, width, label='PyTorch', alpha=0.8)
        
        ax1.set_title('矩阵乘法执行时间对比')
        ax1.set_xlabel('矩阵大小')
        ax1.set_ylabel('执行时间 (ms)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sizes_str, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 加速比图
        ax2 = axes[0, 1]
        speedup_basic = benchmark_results['speedup_basic']
        speedup_opt = benchmark_results['speedup_optimized']
        
        ax2.plot(x_pos, speedup_basic, 'o-', label='Triton 基础', linewidth=2, markersize=6)
        ax2.plot(x_pos, speedup_opt, 's-', label='Triton 优化', linewidth=2, markersize=6)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='基准线')
        
        ax2.set_title('相对 PyTorch 的加速比')
        ax2.set_xlabel('矩阵大小')
        ax2.set_ylabel('加速比')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sizes_str, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. TFLOPS 性能图
        ax3 = axes[1, 0]
        flops_data = []
        for i, (M, K, N) in enumerate(benchmark_results['sizes']):
            flops = 2 * M * N * K
            triton_basic_tflops = flops / (benchmark_results['triton_basic'][i] * 1e12)
            triton_opt_tflops = flops / (benchmark_results['triton_optimized'][i] * 1e12)
            pytorch_tflops = flops / (benchmark_results['pytorch'][i] * 1e12)
            flops_data.append([triton_basic_tflops, triton_opt_tflops, pytorch_tflops])
        
        flops_data = np.array(flops_data)
        
        ax3.bar(x_pos - width, flops_data[:, 0], width, label='Triton 基础', alpha=0.8)
        ax3.bar(x_pos, flops_data[:, 1], width, label='Triton 优化', alpha=0.8)
        ax3.bar(x_pos + width, flops_data[:, 2], width, label='PyTorch', alpha=0.8)
        
        ax3.set_title('计算性能 (TFLOPS)')
        ax3.set_xlabel('矩阵大小')
        ax3.set_ylabel('TFLOPS')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(sizes_str, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 块大小影响分析
        if block_analysis_results:
            ax4 = axes[1, 1]
            configs = block_analysis_results['configs']
            config_labels = [f"{bm}x{bn}x{bk}" for bm, bn, bk in configs]
            tflops = block_analysis_results['tflops']
            
            bars = ax4.bar(range(len(config_labels)), tflops, alpha=0.8)
            ax4.set_title('不同块大小的性能影响')
            ax4.set_xlabel('块大小配置 (M×N×K)')
            ax4.set_ylabel('TFLOPS')
            ax4.set_xticks(range(len(config_labels)))
            ax4.set_xticklabels(config_labels, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 标注最佳配置
            best_idx = np.argmax(tflops)
            ax4.annotate(f'最佳: {tflops[best_idx]:.2f}', 
                        xy=(best_idx, tflops[best_idx]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('triton_matmul_performance.png', dpi=300, bbox_inches='tight')
        print(f"  📈 矩阵乘法性能图已保存: triton_matmul_performance.png")
        plt.show()


def main():
    """主函数：运行 Triton 矩阵乘法演示"""
    print("🚀 Triton 高性能矩阵乘法演示")
    print("=" * 60)
    
    try:
        # 初始化演示类
        demo = TritonMatMulDemo()
        
        # 1. 正确性验证
        print(f"\n" + "="*60)
        print(f"🔍 正确性验证")
        print(f"="*60)
        
        is_correct = demo.verify_correctness()
        
        if not is_correct:
            print(f"❌ 正确性验证失败，请检查实现")
            return
        
        # 2. 性能基准测试
        print(f"\n" + "="*60)
        print(f"🎯 性能基准测试")
        print(f"="*60)
        
        benchmark_results = demo.benchmark_matmul(
            sizes=[
                (512, 512, 512),
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (512, 4096, 512)
            ],
            num_runs=50
        )
        
        # 3. 块大小影响分析
        print(f"\n" + "="*60)
        print(f"📊 块大小影响分析")
        print(f"="*60)
        
        block_analysis_results = demo.analyze_block_size_performance()
        
        # 4. 结果可视化
        print(f"\n" + "="*60)
        print(f"📈 结果可视化")
        print(f"="*60)
        
        demo.visualize_results(benchmark_results, block_analysis_results)
        
        # 5. 总结报告
        print(f"\n" + "="*60)
        print(f"📋 总结报告")
        print(f"="*60)
        
        avg_speedup_basic = np.mean(benchmark_results['speedup_basic'])
        avg_speedup_opt = np.mean(benchmark_results['speedup_optimized'])
        
        print(f"✅ Triton 矩阵乘法演示完成!")
        print(f"")
        print(f"🎯 性能总结:")
        print(f"  • Triton 基础版平均加速比: {avg_speedup_basic:.2f}x")
        print(f"  • Triton 优化版平均加速比: {avg_speedup_opt:.2f}x")
        print(f"")
        print(f"💡 关键发现:")
        print(f"  • 优化的块调度策略显著提升性能")
        print(f"  • 块大小对性能有重要影响")
        print(f"  • 大矩阵的加速效果更明显")
        print(f"")
        print(f"📚 学习要点:")
        print(f"  • 理解矩阵乘法的分块策略")
        print(f"  • 掌握内存访问模式优化")
        print(f"  • 学会调优块大小参数")
        print(f"  • 了解 GPU 内存层次结构")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()