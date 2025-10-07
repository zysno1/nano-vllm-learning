#!/usr/bin/env python3
"""
Triton 基础概念演示

本模块演示 Triton 的基本编程概念，包括：
1. 基础语法和编程模型
2. 内存访问模式
3. 向量化计算
4. 性能对比分析

Author: nano-vLLM-learning
Date: 2024
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict


class TritonBasicsDemo:
    """Triton 基础演示类"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics = defaultdict(list)
        
        # 检查 Triton 支持
        try:
            import triton
            print(f"✅ Triton 支持检查通过")
            print(f"📱 使用设备: {device}")
            print(f"🔧 Triton 版本: {triton.__version__}")
            print(f"🐍 PyTorch 版本: {torch.__version__}")
        except ImportError:
            raise RuntimeError("Triton 未安装，请运行: pip install triton")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
    
    @staticmethod
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """向量加法 Triton 内核"""
        # 获取程序 ID（类似 CUDA 的 blockIdx.x）
        pid = tl.program_id(axis=0)
        
        # 计算当前块的起始位置
        block_start = pid * BLOCK_SIZE
        
        # 生成偏移量（类似 CUDA 的 threadIdx.x）
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        
        # 边界检查掩码
        mask = offsets < n_elements
        
        # 加载数据（向量化加载）
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        
        # 向量化计算
        output = x + y
        
        # 存储结果（向量化存储）
        tl.store(output_ptr + offsets, output, mask=mask)
    
    def vector_add_triton(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """向量加法的 Triton 实现"""
        output = torch.empty_like(x)
        n_elements = output.numel()
        
        # 选择合适的块大小（必须是 2 的幂）
        BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
        
        # 计算网格大小
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        # 启动内核
        self.vector_add_kernel[grid](
            x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
    
    def vector_add_pytorch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """向量加法的 PyTorch 实现"""
        return x + y
    
    @staticmethod
    @triton.jit
    def vector_multiply_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """向量乘法 Triton 内核"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        
        output = x * y
        
        tl.store(output_ptr + offsets, output, mask=mask)
    
    def vector_multiply_triton(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """向量乘法的 Triton 实现"""
        output = torch.empty_like(x)
        n_elements = output.numel()
        
        BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        self.vector_multiply_kernel[grid](
            x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
    
    @staticmethod
    @triton.jit
    def fused_operations_kernel(x_ptr, y_ptr, z_ptr, output_ptr, n_elements, 
                               alpha: tl.constexpr, beta: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        """融合操作内核：output = alpha * x * y + beta * z"""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # 加载数据
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        z = tl.load(z_ptr + offsets, mask=mask)
        
        # 融合计算
        output = alpha * x * y + beta * z
        
        tl.store(output_ptr + offsets, output, mask=mask)
    
    def fused_operations_triton(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                               alpha: float = 2.0, beta: float = 1.0) -> torch.Tensor:
        """融合操作的 Triton 实现"""
        output = torch.empty_like(x)
        n_elements = output.numel()
        
        BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        self.fused_operations_kernel[grid](
            x, y, z, output, n_elements, alpha, beta, BLOCK_SIZE=BLOCK_SIZE
        )
        
        return output
    
    def fused_operations_pytorch(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                                alpha: float = 2.0, beta: float = 1.0) -> torch.Tensor:
        """融合操作的 PyTorch 实现"""
        return alpha * x * y + beta * z
    
    @staticmethod
    @triton.jit
    def reduction_sum_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """归约求和内核"""
        pid = tl.program_id(axis=0)
        
        # 每个程序处理一个块
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # 加载数据
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # 块内归约
        block_sum = tl.sum(x, axis=0)
        
        # 存储块的部分和
        tl.store(output_ptr + pid, block_sum)
    
    def reduction_sum_triton(self, x: torch.Tensor) -> torch.Tensor:
        """归约求和的 Triton 实现"""
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        
        # 计算需要的块数
        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        
        # 创建输出张量存储每个块的部分和
        partial_sums = torch.zeros(num_blocks, dtype=x.dtype, device=x.device)
        
        # 第一阶段：块内归约
        grid = (num_blocks,)
        self.reduction_sum_kernel[grid](
            x, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        
        # 第二阶段：对部分和进行最终归约（使用 PyTorch）
        return partial_sums.sum()
    
    def benchmark_operations(self, sizes: List[int] = [1024, 4096, 16384, 65536, 262144],
                           num_runs: int = 100) -> Dict:
        """对比不同操作的性能"""
        print(f"\n🎯 Triton vs PyTorch 性能对比")
        print(f"  测试大小: {sizes}")
        print(f"  运行次数: {num_runs}")
        
        results = {
            'sizes': sizes,
            'vector_add': {'triton': [], 'pytorch': [], 'speedup': []},
            'vector_multiply': {'triton': [], 'pytorch': [], 'speedup': []},
            'fused_operations': {'triton': [], 'pytorch': [], 'speedup': []},
            'reduction_sum': {'triton': [], 'pytorch': [], 'speedup': []}
        }
        
        for size in sizes:
            print(f"\n  📏 测试大小: {size}")
            
            # 创建测试数据
            x = torch.randn(size, device=self.device, dtype=torch.float32)
            y = torch.randn(size, device=self.device, dtype=torch.float32)
            z = torch.randn(size, device=self.device, dtype=torch.float32)
            
            # 1. 向量加法测试
            print(f"    🔄 向量加法测试...")
            
            # Triton 版本
            triton_time = self._benchmark_function(
                lambda: self.vector_add_triton(x, y), num_runs
            )
            
            # PyTorch 版本
            pytorch_time = self._benchmark_function(
                lambda: self.vector_add_pytorch(x, y), num_runs
            )
            
            speedup = pytorch_time / triton_time
            results['vector_add']['triton'].append(triton_time)
            results['vector_add']['pytorch'].append(pytorch_time)
            results['vector_add']['speedup'].append(speedup)
            
            print(f"      Triton: {triton_time*1000:.3f}ms")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms")
            print(f"      加速比: {speedup:.2f}x")
            
            # 2. 向量乘法测试
            print(f"    🔄 向量乘法测试...")
            
            triton_time = self._benchmark_function(
                lambda: self.vector_multiply_triton(x, y), num_runs
            )
            
            pytorch_time = self._benchmark_function(
                lambda: x * y, num_runs
            )
            
            speedup = pytorch_time / triton_time
            results['vector_multiply']['triton'].append(triton_time)
            results['vector_multiply']['pytorch'].append(pytorch_time)
            results['vector_multiply']['speedup'].append(speedup)
            
            print(f"      Triton: {triton_time*1000:.3f}ms")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms")
            print(f"      加速比: {speedup:.2f}x")
            
            # 3. 融合操作测试
            print(f"    🔄 融合操作测试...")
            
            triton_time = self._benchmark_function(
                lambda: self.fused_operations_triton(x, y, z), num_runs
            )
            
            pytorch_time = self._benchmark_function(
                lambda: self.fused_operations_pytorch(x, y, z), num_runs
            )
            
            speedup = pytorch_time / triton_time
            results['fused_operations']['triton'].append(triton_time)
            results['fused_operations']['pytorch'].append(pytorch_time)
            results['fused_operations']['speedup'].append(speedup)
            
            print(f"      Triton: {triton_time*1000:.3f}ms")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms")
            print(f"      加速比: {speedup:.2f}x")
            
            # 4. 归约求和测试
            print(f"    🔄 归约求和测试...")
            
            triton_time = self._benchmark_function(
                lambda: self.reduction_sum_triton(x), num_runs
            )
            
            pytorch_time = self._benchmark_function(
                lambda: x.sum(), num_runs
            )
            
            speedup = pytorch_time / triton_time
            results['reduction_sum']['triton'].append(triton_time)
            results['reduction_sum']['pytorch'].append(pytorch_time)
            results['reduction_sum']['speedup'].append(speedup)
            
            print(f"      Triton: {triton_time*1000:.3f}ms")
            print(f"      PyTorch: {pytorch_time*1000:.3f}ms")
            print(f"      加速比: {speedup:.2f}x")
        
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
        """验证 Triton 实现的正确性"""
        print(f"\n🔍 验证 Triton 实现正确性...")
        
        # 创建测试数据
        size = 1000
        x = torch.randn(size, device=self.device, dtype=torch.float32)
        y = torch.randn(size, device=self.device, dtype=torch.float32)
        z = torch.randn(size, device=self.device, dtype=torch.float32)
        
        all_correct = True
        
        # 1. 验证向量加法
        triton_result = self.vector_add_triton(x, y)
        pytorch_result = self.vector_add_pytorch(x, y)
        
        max_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
        print(f"  向量加法最大差异: {max_diff:.2e}")
        
        if max_diff > 1e-5:
            print(f"  ❌ 向量加法验证失败")
            all_correct = False
        else:
            print(f"  ✅ 向量加法验证通过")
        
        # 2. 验证向量乘法
        triton_result = self.vector_multiply_triton(x, y)
        pytorch_result = x * y
        
        max_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
        print(f"  向量乘法最大差异: {max_diff:.2e}")
        
        if max_diff > 1e-5:
            print(f"  ❌ 向量乘法验证失败")
            all_correct = False
        else:
            print(f"  ✅ 向量乘法验证通过")
        
        # 3. 验证融合操作
        triton_result = self.fused_operations_triton(x, y, z)
        pytorch_result = self.fused_operations_pytorch(x, y, z)
        
        max_diff = torch.max(torch.abs(triton_result - pytorch_result)).item()
        print(f"  融合操作最大差异: {max_diff:.2e}")
        
        if max_diff > 1e-5:
            print(f"  ❌ 融合操作验证失败")
            all_correct = False
        else:
            print(f"  ✅ 融合操作验证通过")
        
        # 4. 验证归约求和
        triton_result = self.reduction_sum_triton(x)
        pytorch_result = x.sum()
        
        diff = torch.abs(triton_result - pytorch_result).item()
        relative_diff = diff / torch.abs(pytorch_result).item()
        print(f"  归约求和相对差异: {relative_diff:.2e}")
        
        if relative_diff > 1e-4:
            print(f"  ❌ 归约求和验证失败")
            all_correct = False
        else:
            print(f"  ✅ 归约求和验证通过")
        
        return all_correct
    
    def analyze_block_size_impact(self, sizes: List[int] = [1024, 4096, 16384]) -> Dict:
        """分析块大小对性能的影响"""
        print(f"\n📊 分析块大小对性能的影响")
        
        block_sizes = [64, 128, 256, 512, 1024]
        results = {
            'block_sizes': block_sizes,
            'performance': {}
        }
        
        for size in sizes:
            print(f"\n  📏 数据大小: {size}")
            results['performance'][size] = []
            
            x = torch.randn(size, device=self.device, dtype=torch.float32)
            y = torch.randn(size, device=self.device, dtype=torch.float32)
            
            for block_size in block_sizes:
                # 修改内核以接受动态块大小
                @triton.jit
                def dynamic_vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
                    pid = tl.program_id(axis=0)
                    block_start = pid * BLOCK_SIZE
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements
                    
                    x = tl.load(x_ptr + offsets, mask=mask)
                    y = tl.load(y_ptr + offsets, mask=mask)
                    output = x + y
                    
                    tl.store(output_ptr + offsets, output, mask=mask)
                
                def vector_add_with_block_size(x, y, block_size):
                    output = torch.empty_like(x)
                    n_elements = output.numel()
                    grid = (triton.cdiv(n_elements, block_size),)
                    
                    dynamic_vector_add_kernel[grid](
                        x, y, output, n_elements, BLOCK_SIZE=block_size
                    )
                    return output
                
                # 测试性能
                exec_time = self._benchmark_function(
                    lambda: vector_add_with_block_size(x, y, block_size), 50
                )
                
                results['performance'][size].append(exec_time)
                print(f"    块大小 {block_size}: {exec_time*1000:.3f}ms")
        
        return results
    
    def visualize_results(self, benchmark_results: Dict, block_size_results: Dict):
        """可视化测试结果"""
        print(f"\n📊 生成性能分析图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 性能对比图
        operations = ['vector_add', 'vector_multiply', 'fused_operations', 'reduction_sum']
        operation_names = ['向量加法', '向量乘法', '融合操作', '归约求和']
        
        for i, (op, name) in enumerate(zip(operations, operation_names)):
            ax = axes[i // 2, i % 2]
            
            sizes = benchmark_results['sizes']
            triton_times = np.array(benchmark_results[op]['triton']) * 1000  # 转换为毫秒
            pytorch_times = np.array(benchmark_results[op]['pytorch']) * 1000
            
            ax.loglog(sizes, triton_times, 'o-', label='Triton', linewidth=2, markersize=6)
            ax.loglog(sizes, pytorch_times, 's--', label='PyTorch', linewidth=2, markersize=6)
            
            ax.set_title(f'{name} 性能对比')
            ax.set_xlabel('数据大小')
            ax.set_ylabel('执行时间 (ms)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加加速比注释
            for j, (size, speedup) in enumerate(zip(sizes, benchmark_results[op]['speedup'])):
                if j % 2 == 0:  # 只在部分点添加注释以避免拥挤
                    ax.annotate(f'{speedup:.1f}x', 
                               xy=(size, triton_times[j]), 
                               xytext=(5, 5), 
                               textcoords='offset points',
                               fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('triton_performance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  📈 性能对比图已保存: triton_performance_comparison.png")
        
        # 2. 块大小影响分析图
        if block_size_results:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
            
            block_sizes = block_size_results['block_sizes']
            
            for size in block_size_results['performance']:
                times = np.array(block_size_results['performance'][size]) * 1000
                ax2.plot(block_sizes, times, 'o-', label=f'数据大小 {size}', linewidth=2, markersize=6)
            
            ax2.set_title('块大小对性能的影响')
            ax2.set_xlabel('块大小')
            ax2.set_ylabel('执行时间 (ms)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)
            
            plt.tight_layout()
            plt.savefig('triton_block_size_analysis.png', dpi=300, bbox_inches='tight')
            print(f"  📈 块大小分析图已保存: triton_block_size_analysis.png")
        
        plt.show()


def main():
    """主函数：运行 Triton 基础演示"""
    print("🚀 Triton 基础概念演示")
    print("=" * 60)
    
    try:
        # 初始化演示类
        demo = TritonBasicsDemo()
        
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
        
        benchmark_results = demo.benchmark_operations(
            sizes=[1024, 4096, 16384, 65536],
            num_runs=50
        )
        
        # 3. 块大小影响分析
        print(f"\n" + "="*60)
        print(f"📊 块大小影响分析")
        print(f"="*60)
        
        block_size_results = demo.analyze_block_size_impact(
            sizes=[4096, 16384]
        )
        
        # 4. 结果可视化
        print(f"\n" + "="*60)
        print(f"📈 结果可视化")
        print(f"="*60)
        
        demo.visualize_results(benchmark_results, block_size_results)
        
        # 5. 总结报告
        print(f"\n" + "="*60)
        print(f"📋 总结报告")
        print(f"="*60)
        
        # 计算平均加速比
        avg_speedups = {}
        for op in ['vector_add', 'vector_multiply', 'fused_operations', 'reduction_sum']:
            avg_speedups[op] = np.mean(benchmark_results[op]['speedup'])
        
        overall_avg_speedup = np.mean(list(avg_speedups.values()))
        
        print(f"✅ Triton 基础演示完成!")
        print(f"")
        print(f"🎯 性能总结:")
        print(f"  • 向量加法平均加速比: {avg_speedups['vector_add']:.2f}x")
        print(f"  • 向量乘法平均加速比: {avg_speedups['vector_multiply']:.2f}x")
        print(f"  • 融合操作平均加速比: {avg_speedups['fused_operations']:.2f}x")
        print(f"  • 归约求和平均加速比: {avg_speedups['reduction_sum']:.2f}x")
        print(f"  • 总体平均加速比: {overall_avg_speedup:.2f}x")
        print(f"")
        print(f"💡 关键发现:")
        print(f"  • 融合操作通常有最好的加速效果")
        print(f"  • 块大小对性能有显著影响")
        print(f"  • 数据大小越大，加速效果越明显")
        print(f"")
        print(f"📚 学习要点:")
        print(f"  • Triton 语法类似 Python，易于学习")
        print(f"  • 向量化操作是性能的关键")
        print(f"  • 内存访问模式影响性能")
        print(f"  • 块大小需要根据数据大小调优")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()