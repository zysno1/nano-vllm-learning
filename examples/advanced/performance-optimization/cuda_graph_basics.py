#!/usr/bin/env python3
"""
CUDA Graph 基础概念演示

本模块演示 CUDA Graph 的基本工作原理，包括：
1. 简单矩阵运算的图捕获
2. 图重放的性能对比
3. 内存使用分析
4. 基础调试技巧

Author: nano-vLLM-learning
Date: 2024
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict


class BasicCUDAGraphDemo:
    """CUDA Graph 基础演示类"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.metrics = defaultdict(list)
        
        # 检查 CUDA Graph 支持
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        
        if not hasattr(torch.cuda, 'CUDAGraph'):
            raise RuntimeError("当前 PyTorch 版本不支持 CUDA Graph")
        
        print(f"✅ CUDA Graph 支持检查通过")
        print(f"📱 使用设备: {device}")
        print(f"🔧 CUDA 版本: {torch.version.cuda}")
        print(f"🐍 PyTorch 版本: {torch.__version__}")
    
    def simple_matrix_operations(self, x: torch.Tensor) -> torch.Tensor:
        """简单的矩阵运算序列"""
        # 模拟一系列常见的神经网络操作
        y = torch.matmul(x, x.transpose(-2, -1))  # 注意力计算
        y = torch.softmax(y, dim=-1)              # Softmax
        y = torch.matmul(y, x)                    # 加权求和
        y = torch.relu(y)                         # 激活函数
        y = torch.layer_norm(y, y.shape[-1:])    # Layer Norm
        return y
    
    def benchmark_traditional_execution(self, input_tensor: torch.Tensor, 
                                      num_runs: int = 1000) -> Dict[str, float]:
        """传统执行方式的性能基准测试"""
        print(f"\n🔄 开始传统执行方式基准测试 (运行 {num_runs} 次)")
        
        # 预热
        for _ in range(10):
            _ = self.simple_matrix_operations(input_tensor)
        
        torch.cuda.synchronize()
        
        # 性能测试
        start_time = time.time()
        
        for i in range(num_runs):
            result = self.simple_matrix_operations(input_tensor)
            
            # 每100次打印进度
            if (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{num_runs}")
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        metrics = {
            'total_time': total_time,
            'avg_time': total_time / num_runs,
            'throughput': num_runs / total_time,
            'result_shape': result.shape
        }
        
        print(f"📊 传统方式结果:")
        print(f"  总时间: {metrics['total_time']:.4f}s")
        print(f"  平均时间: {metrics['avg_time']*1000:.4f}ms")
        print(f"  吞吐量: {metrics['throughput']:.2f} ops/s")
        
        return metrics
    
    def benchmark_cuda_graph_execution(self, input_tensor: torch.Tensor, 
                                     num_runs: int = 1000) -> Dict[str, float]:
        """CUDA Graph 执行方式的性能基准测试"""
        print(f"\n🚀 开始 CUDA Graph 执行方式基准测试")
        
        # 1. 创建静态张量（形状必须固定）
        static_input = torch.zeros_like(input_tensor)
        static_output = None
        
        # 2. 预热
        print("  🔥 预热阶段...")
        for _ in range(3):
            _ = self.simple_matrix_operations(static_input)
        
        torch.cuda.synchronize()
        
        # 3. 捕获 CUDA Graph
        print("  📸 捕获 CUDA Graph...")
        cuda_graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(cuda_graph):
            static_output = self.simple_matrix_operations(static_input)
        
        print(f"  ✅ CUDA Graph 捕获完成")
        
        # 4. 性能测试
        print(f"  🔄 开始性能测试 (运行 {num_runs} 次)")
        
        start_time = time.time()
        
        for i in range(num_runs):
            # 复制输入数据到静态张量
            static_input.copy_(input_tensor)
            
            # 重放图
            cuda_graph.replay()
            
            # 每100次打印进度
            if (i + 1) % 100 == 0:
                print(f"    进度: {i + 1}/{num_runs}")
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        metrics = {
            'total_time': total_time,
            'avg_time': total_time / num_runs,
            'throughput': num_runs / total_time,
            'result_shape': static_output.shape
        }
        
        print(f"📊 CUDA Graph 结果:")
        print(f"  总时间: {metrics['total_time']:.4f}s")
        print(f"  平均时间: {metrics['avg_time']*1000:.4f}ms")
        print(f"  吞吐量: {metrics['throughput']:.2f} ops/s")
        
        return metrics, static_output.clone()
    
    def verify_correctness(self, input_tensor: torch.Tensor, 
                          cuda_graph_output: torch.Tensor) -> bool:
        """验证 CUDA Graph 输出的正确性"""
        print(f"\n🔍 验证输出正确性...")
        
        # 传统方式计算参考结果
        reference_output = self.simple_matrix_operations(input_tensor)
        
        # 计算差异
        max_diff = torch.max(torch.abs(reference_output - cuda_graph_output)).item()
        mean_diff = torch.mean(torch.abs(reference_output - cuda_graph_output)).item()
        
        # 相对误差
        relative_error = max_diff / (torch.max(torch.abs(reference_output)).item() + 1e-8)
        
        print(f"  最大绝对差异: {max_diff:.2e}")
        print(f"  平均绝对差异: {mean_diff:.2e}")
        print(f"  相对误差: {relative_error:.2e}")
        
        # 判断是否正确（允许小的数值误差）
        is_correct = relative_error < 1e-5
        
        if is_correct:
            print(f"  ✅ 输出正确性验证通过")
        else:
            print(f"  ❌ 输出正确性验证失败")
        
        return is_correct
    
    def analyze_memory_usage(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """分析内存使用情况"""
        print(f"\n💾 分析内存使用情况...")
        
        # 记录初始内存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"  初始内存使用: {initial_memory / 1024**2:.2f} MB")
        
        # 传统方式内存使用
        torch.cuda.reset_peak_memory_stats()
        _ = self.simple_matrix_operations(input_tensor)
        traditional_peak = torch.cuda.max_memory_allocated()
        
        print(f"  传统方式峰值内存: {traditional_peak / 1024**2:.2f} MB")
        
        # CUDA Graph 内存使用
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建静态张量
        static_input = torch.zeros_like(input_tensor)
        
        # 捕获图
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            static_output = self.simple_matrix_operations(static_input)
        
        cuda_graph_peak = torch.cuda.max_memory_allocated()
        
        print(f"  CUDA Graph 峰值内存: {cuda_graph_peak / 1024**2:.2f} MB")
        
        # 计算内存开销
        memory_overhead = cuda_graph_peak - traditional_peak
        overhead_ratio = memory_overhead / traditional_peak * 100
        
        print(f"  内存开销: {memory_overhead / 1024**2:.2f} MB ({overhead_ratio:.1f}%)")
        
        return {
            'initial_memory': initial_memory,
            'traditional_peak': traditional_peak,
            'cuda_graph_peak': cuda_graph_peak,
            'memory_overhead': memory_overhead,
            'overhead_ratio': overhead_ratio
        }
    
    def run_comprehensive_benchmark(self, batch_sizes: List[int] = [1, 4, 8, 16, 32],
                                  seq_lengths: List[int] = [128, 256, 512, 1024]) -> Dict:
        """运行综合性能基准测试"""
        print(f"\n🎯 开始综合性能基准测试")
        print(f"  批次大小: {batch_sizes}")
        print(f"  序列长度: {seq_lengths}")
        
        results = {
            'batch_sizes': batch_sizes,
            'seq_lengths': seq_lengths,
            'traditional_times': [],
            'cuda_graph_times': [],
            'speedup_ratios': []
        }
        
        for batch_size in batch_sizes:
            batch_traditional_times = []
            batch_cuda_graph_times = []
            batch_speedup_ratios = []
            
            for seq_len in seq_lengths:
                print(f"\n  📏 测试配置: batch_size={batch_size}, seq_len={seq_len}")
                
                # 创建测试输入
                input_tensor = torch.randn(batch_size, seq_len, 512, device=self.device)
                
                # 传统方式测试
                traditional_metrics = self.benchmark_traditional_execution(
                    input_tensor, num_runs=100
                )
                
                # CUDA Graph 方式测试
                cuda_graph_metrics, _ = self.benchmark_cuda_graph_execution(
                    input_tensor, num_runs=100
                )
                
                # 计算加速比
                speedup = traditional_metrics['avg_time'] / cuda_graph_metrics['avg_time']
                
                batch_traditional_times.append(traditional_metrics['avg_time'])
                batch_cuda_graph_times.append(cuda_graph_metrics['avg_time'])
                batch_speedup_ratios.append(speedup)
                
                print(f"    🚀 加速比: {speedup:.2f}x")
            
            results['traditional_times'].append(batch_traditional_times)
            results['cuda_graph_times'].append(batch_cuda_graph_times)
            results['speedup_ratios'].append(batch_speedup_ratios)
        
        return results
    
    def visualize_results(self, results: Dict):
        """可视化测试结果"""
        print(f"\n📊 生成性能对比图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 加速比热力图
        speedup_matrix = np.array(results['speedup_ratios'])
        im1 = axes[0, 0].imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_title('CUDA Graph 加速比热力图')
        axes[0, 0].set_xlabel('序列长度')
        axes[0, 0].set_ylabel('批次大小')
        axes[0, 0].set_xticks(range(len(results['seq_lengths'])))
        axes[0, 0].set_xticklabels(results['seq_lengths'])
        axes[0, 0].set_yticks(range(len(results['batch_sizes'])))
        axes[0, 0].set_yticklabels(results['batch_sizes'])
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 添加数值标注
        for i in range(len(results['batch_sizes'])):
            for j in range(len(results['seq_lengths'])):
                axes[0, 0].text(j, i, f'{speedup_matrix[i, j]:.2f}x',
                               ha='center', va='center', color='black', fontweight='bold')
        
        # 2. 不同批次大小下的性能对比
        for i, batch_size in enumerate(results['batch_sizes']):
            axes[0, 1].plot(results['seq_lengths'], 
                           np.array(results['traditional_times'][i]) * 1000,
                           'o-', label=f'传统方式 (batch={batch_size})', alpha=0.7)
            axes[0, 1].plot(results['seq_lengths'], 
                           np.array(results['cuda_graph_times'][i]) * 1000,
                           's--', label=f'CUDA Graph (batch={batch_size})', alpha=0.7)
        
        axes[0, 1].set_title('执行时间对比')
        axes[0, 1].set_xlabel('序列长度')
        axes[0, 1].set_ylabel('执行时间 (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 加速比趋势
        for i, batch_size in enumerate(results['batch_sizes']):
            axes[1, 0].plot(results['seq_lengths'], results['speedup_ratios'][i],
                           'o-', label=f'batch_size={batch_size}', linewidth=2)
        
        axes[1, 0].set_title('加速比趋势')
        axes[1, 0].set_xlabel('序列长度')
        axes[1, 0].set_ylabel('加速比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='基准线')
        
        # 4. 平均加速比统计
        avg_speedups = [np.mean(speedups) for speedups in results['speedup_ratios']]
        bars = axes[1, 1].bar(range(len(results['batch_sizes'])), avg_speedups, 
                             color='skyblue', alpha=0.7)
        axes[1, 1].set_title('平均加速比 (按批次大小)')
        axes[1, 1].set_xlabel('批次大小')
        axes[1, 1].set_ylabel('平均加速比')
        axes[1, 1].set_xticks(range(len(results['batch_sizes'])))
        axes[1, 1].set_xticklabels(results['batch_sizes'])
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标注
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cuda_graph_performance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  📈 图表已保存为: cuda_graph_performance_analysis.png")
        plt.show()


def main():
    """主函数：运行 CUDA Graph 基础演示"""
    print("🚀 CUDA Graph 基础概念演示")
    print("=" * 60)
    
    try:
        # 初始化演示类
        demo = BasicCUDAGraphDemo()
        
        # 创建测试输入
        batch_size, seq_len, hidden_size = 8, 512, 512
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        
        print(f"\n📊 测试配置:")
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  数据类型: {input_tensor.dtype}")
        print(f"  设备: {input_tensor.device}")
        
        # 1. 基础性能对比
        print(f"\n" + "="*60)
        print(f"🎯 基础性能对比测试")
        print(f"="*60)
        
        traditional_metrics = demo.benchmark_traditional_execution(input_tensor, num_runs=500)
        cuda_graph_metrics, cuda_graph_output = demo.benchmark_cuda_graph_execution(input_tensor, num_runs=500)
        
        # 计算加速比
        speedup = traditional_metrics['avg_time'] / cuda_graph_metrics['avg_time']
        print(f"\n🚀 性能提升总结:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  时间节省: {(1 - cuda_graph_metrics['avg_time']/traditional_metrics['avg_time'])*100:.1f}%")
        
        # 2. 正确性验证
        print(f"\n" + "="*60)
        print(f"🔍 正确性验证")
        print(f"="*60)
        
        is_correct = demo.verify_correctness(input_tensor, cuda_graph_output)
        
        # 3. 内存使用分析
        print(f"\n" + "="*60)
        print(f"💾 内存使用分析")
        print(f"="*60)
        
        memory_metrics = demo.analyze_memory_usage(input_tensor)
        
        # 4. 综合性能测试
        print(f"\n" + "="*60)
        print(f"📊 综合性能基准测试")
        print(f"="*60)
        
        # 使用较小的测试规模以节省时间
        results = demo.run_comprehensive_benchmark(
            batch_sizes=[1, 4, 8, 16],
            seq_lengths=[128, 256, 512]
        )
        
        # 5. 结果可视化
        print(f"\n" + "="*60)
        print(f"📈 结果可视化")
        print(f"="*60)
        
        demo.visualize_results(results)
        
        # 6. 总结报告
        print(f"\n" + "="*60)
        print(f"📋 总结报告")
        print(f"="*60)
        
        avg_speedup = np.mean([np.mean(speedups) for speedups in results['speedup_ratios']])
        
        print(f"✅ CUDA Graph 基础演示完成!")
        print(f"")
        print(f"🎯 关键发现:")
        print(f"  • 平均加速比: {avg_speedup:.2f}x")
        print(f"  • 内存开销: {memory_metrics['overhead_ratio']:.1f}%")
        print(f"  • 输出正确性: {'✅ 通过' if is_correct else '❌ 失败'}")
        print(f"")
        print(f"💡 适用场景:")
        print(f"  • 重复执行的固定计算模式")
        print(f"  • 批处理推理任务")
        print(f"  • 对延迟敏感的应用")
        print(f"")
        print(f"⚠️  使用注意事项:")
        print(f"  • 输入形状必须固定")
        print(f"  • 需要额外的内存开销")
        print(f"  • 不支持动态控制流")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()