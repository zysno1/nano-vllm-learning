#!/usr/bin/env python3
"""
torch.compile 基础概念演示

本模块演示 torch.compile 的基本概念和使用方法，包括：
1. 基础编译功能
2. 不同编译模式的对比
3. 性能基准测试
4. 编译开销分析
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
import warnings

# 忽略编译警告
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    function_name: str
    mode: str
    execution_time: float
    memory_usage: float
    speedup: float = 1.0
    compilation_time: float = 0.0


class CompilationBenchmark:
    """编译基准测试类"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.results: List[BenchmarkResult] = []
        
        print(f"🚀 使用设备: {device}")
        if device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def benchmark_function(self, 
                          func: Callable, 
                          args: Tuple, 
                          name: str,
                          modes: List[str] = None,
                          warmup_runs: int = 3,
                          benchmark_runs: int = 10) -> Dict[str, BenchmarkResult]:
        """
        对函数进行基准测试
        
        Args:
            func: 要测试的函数
            args: 函数参数
            name: 函数名称
            modes: 编译模式列表
            warmup_runs: 预热运行次数
            benchmark_runs: 基准测试运行次数
        """
        if modes is None:
            modes = ["original", "default", "reduce-overhead", "max-autotune"]
        
        results = {}
        
        print(f"\n📊 基准测试: {name}")
        print("-" * 50)
        
        for mode in modes:
            if mode == "original":
                test_func = func
                compilation_time = 0.0
            else:
                # 编译函数并测量编译时间
                compile_start = time.time()
                test_func = torch.compile(func, mode=mode)
                
                # 预热编译（触发实际编译）
                for _ in range(warmup_runs):
                    with torch.no_grad():
                        _ = test_func(*args)
                
                compilation_time = time.time() - compile_start
            
            # 清理 GPU 缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 测量内存使用
            if self.device == "cuda":
                start_memory = torch.cuda.memory_allocated()
            else:
                start_memory = 0
            
            # 基准测试
            times = []
            for _ in range(benchmark_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                with torch.no_grad():
                    result = test_func(*args)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # 计算统计信息
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            if self.device == "cuda":
                end_memory = torch.cuda.memory_allocated()
                memory_usage = (end_memory - start_memory) / 1e6  # MB
            else:
                memory_usage = 0
            
            # 计算加速比
            if mode == "original":
                baseline_time = avg_time
                speedup = 1.0
            else:
                speedup = baseline_time / avg_time if avg_time > 0 else 1.0
            
            # 存储结果
            result = BenchmarkResult(
                function_name=name,
                mode=mode,
                execution_time=avg_time,
                memory_usage=memory_usage,
                speedup=speedup,
                compilation_time=compilation_time
            )
            results[mode] = result
            self.results.append(result)
            
            # 打印结果
            print(f"  {mode:15s}: {avg_time*1000:6.2f}ms ± {std_time*1000:4.2f}ms "
                  f"(加速: {speedup:4.2f}x, 内存: {memory_usage:5.1f}MB)")
            
            if compilation_time > 0:
                print(f"                   编译时间: {compilation_time:.2f}s")
        
        return results


def simple_operations():
    """简单数学操作"""
    
    def add_mul(x, y):
        """加法和乘法"""
        return x + y * 2.0
    
    def matrix_ops(a, b):
        """矩阵操作"""
        return torch.matmul(a, b) + torch.relu(a)
    
    def complex_ops(x):
        """复杂操作"""
        x = torch.sin(x)
        x = torch.exp(x)
        x = torch.log(x + 1e-8)
        return torch.sum(x, dim=-1)
    
    return add_mul, matrix_ops, complex_ops


def attention_operations():
    """注意力相关操作"""
    
    def scaled_dot_product_attention(q, k, v, mask=None):
        """缩放点积注意力"""
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    def multi_head_attention(x, num_heads=8):
        """多头注意力（简化版）"""
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // num_heads
        
        # 简化的 QKV 投影
        q = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.view(batch_size, seq_len, d_model)
    
    def feedforward_network(x, hidden_dim=2048):
        """前馈网络"""
        # 第一层
        hidden = F.relu(F.linear(x, torch.randn(hidden_dim, x.size(-1), device=x.device)))
        # 第二层
        output = F.linear(hidden, torch.randn(x.size(-1), hidden_dim, device=x.device))
        return output
    
    return scaled_dot_product_attention, multi_head_attention, feedforward_network


class SimpleTransformerLayer(nn.Module):
    """简化的 Transformer 层"""
    
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


def run_basic_compilation_demo():
    """运行基础编译演示"""
    print("🎯 torch.compile 基础概念演示")
    print("=" * 60)
    
    benchmark = CompilationBenchmark()
    
    # 1. 简单操作测试
    print("\n📝 测试 1: 简单数学操作")
    add_mul, matrix_ops, complex_ops = simple_operations()
    
    # 测试数据
    x = torch.randn(1000, 1000, device=benchmark.device)
    y = torch.randn(1000, 1000, device=benchmark.device)
    
    # 基准测试
    benchmark.benchmark_function(add_mul, (x, y), "加法乘法")
    benchmark.benchmark_function(matrix_ops, (x, y), "矩阵操作")
    benchmark.benchmark_function(complex_ops, (x,), "复杂操作")
    
    # 2. 注意力操作测试
    print("\n📝 测试 2: 注意力机制操作")
    scaled_attn, multi_head_attn, ffn = attention_operations()
    
    # 注意力测试数据
    batch_size, seq_len, d_model = 32, 128, 512
    q = torch.randn(batch_size, seq_len, d_model, device=benchmark.device)
    k = torch.randn(batch_size, seq_len, d_model, device=benchmark.device)
    v = torch.randn(batch_size, seq_len, d_model, device=benchmark.device)
    x_attn = torch.randn(batch_size, seq_len, d_model, device=benchmark.device)
    
    benchmark.benchmark_function(scaled_attn, (q, k, v), "缩放点积注意力")
    benchmark.benchmark_function(multi_head_attn, (x_attn,), "多头注意力")
    benchmark.benchmark_function(ffn, (x_attn,), "前馈网络")
    
    # 3. Transformer 层测试
    print("\n📝 测试 3: Transformer 层")
    transformer_layer = SimpleTransformerLayer().to(benchmark.device)
    
    def transformer_forward(x):
        return transformer_layer(x)
    
    benchmark.benchmark_function(transformer_forward, (x_attn,), "Transformer层")
    
    return benchmark


def analyze_compilation_overhead():
    """分析编译开销"""
    print("\n🔍 编译开销分析")
    print("=" * 40)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_function(x):
        return torch.matmul(x, x.T) + torch.relu(x)
    
    # 不同大小的输入
    sizes = [100, 500, 1000, 2000]
    modes = ["default", "reduce-overhead", "max-autotune"]
    
    compilation_times = {mode: [] for mode in modes}
    execution_times = {mode: [] for mode in modes}
    
    for size in sizes:
        print(f"\n测试输入大小: {size}x{size}")
        x = torch.randn(size, size, device=device)
        
        for mode in modes:
            # 测量编译时间
            start_time = time.time()
            compiled_func = torch.compile(test_function, mode=mode)
            
            # 触发编译
            with torch.no_grad():
                _ = compiled_func(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            compile_time = time.time() - start_time
            compilation_times[mode].append(compile_time)
            
            # 测量执行时间
            start_time = time.time()
            with torch.no_grad():
                _ = compiled_func(x)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            exec_time = time.time() - start_time
            execution_times[mode].append(exec_time)
            
            print(f"  {mode:15s}: 编译 {compile_time:.3f}s, 执行 {exec_time*1000:.2f}ms")
    
    return sizes, compilation_times, execution_times


def visualize_results(benchmark: CompilationBenchmark, 
                     sizes: List[int], 
                     compilation_times: Dict[str, List[float]],
                     execution_times: Dict[str, List[float]]):
    """可视化结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 加速比对比
    functions = {}
    for result in benchmark.results:
        if result.function_name not in functions:
            functions[result.function_name] = {}
        functions[result.function_name][result.mode] = result.speedup
    
    modes = ["default", "reduce-overhead", "max-autotune"]
    x_pos = np.arange(len(functions))
    width = 0.25
    
    for i, mode in enumerate(modes):
        speedups = [functions[func].get(mode, 1.0) for func in functions.keys()]
        ax1.bar(x_pos + i * width, speedups, width, label=mode, alpha=0.8)
    
    ax1.set_xlabel('函数')
    ax1.set_ylabel('加速比')
    ax1.set_title('不同编译模式的加速比对比')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(list(functions.keys()), rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 内存使用对比
    memory_usage = {}
    for result in benchmark.results:
        if result.function_name not in memory_usage:
            memory_usage[result.function_name] = {}
        memory_usage[result.function_name][result.mode] = result.memory_usage
    
    for i, mode in enumerate(modes):
        memory_vals = [memory_usage[func].get(mode, 0) for func in memory_usage.keys()]
        ax2.bar(x_pos + i * width, memory_vals, width, label=mode, alpha=0.8)
    
    ax2.set_xlabel('函数')
    ax2.set_ylabel('内存使用 (MB)')
    ax2.set_title('不同编译模式的内存使用对比')
    ax2.set_xticks(x_pos + width)
    ax2.set_xticklabels(list(memory_usage.keys()), rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 编译时间 vs 输入大小
    for mode in compilation_times.keys():
        ax3.plot(sizes, compilation_times[mode], 'o-', label=f'{mode} 编译时间', linewidth=2)
    
    ax3.set_xlabel('输入大小')
    ax3.set_ylabel('编译时间 (秒)')
    ax3.set_title('编译时间 vs 输入大小')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # 4. 执行时间 vs 输入大小
    for mode in execution_times.keys():
        ax4.plot(sizes, [t*1000 for t in execution_times[mode]], 'o-', 
                label=f'{mode} 执行时间', linewidth=2)
    
    ax4.set_xlabel('输入大小')
    ax4.set_ylabel('执行时间 (毫秒)')
    ax4.set_title('执行时间 vs 输入大小')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('torch_compile_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 分析图表已保存为 'torch_compile_analysis.png'")


def print_summary(benchmark: CompilationBenchmark):
    """打印总结"""
    print("\n📋 性能总结")
    print("=" * 50)
    
    # 按函数分组统计
    functions = {}
    for result in benchmark.results:
        if result.function_name not in functions:
            functions[result.function_name] = []
        functions[result.function_name].append(result)
    
    for func_name, results in functions.items():
        print(f"\n🔧 {func_name}:")
        
        # 找到最佳模式
        best_result = max(results, key=lambda r: r.speedup if r.mode != "original" else 0)
        
        if best_result.mode != "original":
            print(f"   最佳模式: {best_result.mode}")
            print(f"   最大加速: {best_result.speedup:.2f}x")
            print(f"   执行时间: {best_result.execution_time*1000:.2f}ms")
            print(f"   内存使用: {best_result.memory_usage:.1f}MB")
            
            if best_result.compilation_time > 0:
                print(f"   编译时间: {best_result.compilation_time:.2f}s")
    
    # 整体统计
    compiled_results = [r for r in benchmark.results if r.mode != "original"]
    if compiled_results:
        avg_speedup = np.mean([r.speedup for r in compiled_results])
        max_speedup = max([r.speedup for r in compiled_results])
        
        print(f"\n🎯 整体性能:")
        print(f"   平均加速比: {avg_speedup:.2f}x")
        print(f"   最大加速比: {max_speedup:.2f}x")
        print(f"   测试函数数: {len(functions)}")
        print(f"   编译模式数: {len(set(r.mode for r in compiled_results))}")


def main():
    """主函数"""
    print("🎯 torch.compile 基础概念演示")
    print("本演示将展示 PyTorch 2.0 编译功能的基本用法和性能优势")
    
    # 检查 PyTorch 版本
    print(f"\nPyTorch 版本: {torch.__version__}")
    if torch.__version__ < "2.0":
        print("⚠️  警告: torch.compile 需要 PyTorch 2.0 或更高版本")
        return
    
    # 运行基础编译演示
    benchmark = run_basic_compilation_demo()
    
    # 分析编译开销
    sizes, compilation_times, execution_times = analyze_compilation_overhead()
    
    # 可视化结果
    visualize_results(benchmark, sizes, compilation_times, execution_times)
    
    # 打印总结
    print_summary(benchmark)
    
    print("\n✅ 演示完成！")
    print("\n💡 关键观察:")
    print("1. torch.compile 在大多数情况下都能提供显著的性能提升")
    print("2. 'max-autotune' 模式通常提供最佳性能，但编译时间较长")
    print("3. 'reduce-overhead' 模式在减少 Python 开销方面效果显著")
    print("4. 编译开销在大型模型和长时间运行中是值得的")
    print("5. 内存使用可能会有所增加，但通常换来更好的性能")


if __name__ == "__main__":
    main()