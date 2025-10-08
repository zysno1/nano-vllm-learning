"""
动态形状处理的 torch.compile 实现
演示如何处理动态输入形状的编译优化挑战
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import gc
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import random

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DynamicShapeConfig:
    """动态形状配置"""
    min_batch_size: int = 1
    max_batch_size: int = 16
    min_seq_len: int = 64
    max_seq_len: int = 2048
    hidden_size: int = 768
    num_heads: int = 12
    vocab_size: int = 32000

class DynamicAttention(nn.Module):
    """支持动态形状的注意力层"""
    
    def __init__(self, config: DynamicShapeConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # 计算 QKV
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # 重塑输出
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)

class DynamicTransformerBlock(nn.Module):
    """支持动态形状的 Transformer 块"""
    
    def __init__(self, config: DynamicShapeConfig):
        super().__init__()
        self.attention = DynamicAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力 + 残差
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # 前馈 + 残差
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x

class DynamicTransformer(nn.Module):
    """支持动态形状的 Transformer 模型"""
    
    def __init__(self, config: DynamicShapeConfig, num_layers: int = 6):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        
        self.blocks = nn.ModuleList([
            DynamicTransformerBlock(config) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        
        # 位置编码
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        x = x + pos_emb
        
        # Transformer 块
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 输出
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

class DynamicShapeBenchmark:
    """动态形状基准测试"""
    
    def __init__(self, config: DynamicShapeConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.results = {}
        
    def generate_random_inputs(self, num_samples: int = 10) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """生成随机形状的输入"""
        inputs = []
        
        for _ in range(num_samples):
            batch_size = random.randint(self.config.min_batch_size, self.config.max_batch_size)
            seq_len = random.randint(self.config.min_seq_len, self.config.max_seq_len)
            
            input_ids = torch.randint(0, self.config.vocab_size, 
                                    (batch_size, seq_len), device=self.device)
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
            
            inputs.append((input_ids, attention_mask))
        
        return inputs
    
    def benchmark_static_vs_dynamic_compilation(self):
        """对比静态和动态编译"""
        logger.info("开始静态 vs 动态编译基准测试...")
        
        model = DynamicTransformer(self.config).to(self.device)
        model.eval()
        
        # 静态编译 (固定形状)
        static_model = torch.compile(model, dynamic=False)
        
        # 动态编译
        dynamic_model = torch.compile(model, dynamic=True)
        
        # 生成测试输入
        test_inputs = self.generate_random_inputs(20)
        
        results = {
            "original": {"times": [], "recompilations": 0},
            "static": {"times": [], "recompilations": 0},
            "dynamic": {"times": [], "recompilations": 0}
        }
        
        models = {
            "original": model,
            "static": static_model,
            "dynamic": dynamic_model
        }
        
        for model_name, compiled_model in models.items():
            logger.info(f"测试 {model_name} 模型...")
            
            # 预热
            warmup_input = test_inputs[0]
            with torch.no_grad():
                for _ in range(3):
                    try:
                        _ = compiled_model(warmup_input[0], warmup_input[1])
                    except Exception as e:
                        logger.warning(f"{model_name} 预热失败: {e}")
                        break
            
            # 基准测试
            recompilation_count = 0
            
            for i, (input_ids, attention_mask) in enumerate(test_inputs):
                try:
                    with torch.no_grad():
                        start_time = time.time()
                        
                        # 监控是否发生重编译
                        if model_name != "original":
                            torch._dynamo.reset()
                            compilation_start = time.time()
                        
                        _ = compiled_model(input_ids, attention_mask)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        
                        # 检测重编译
                        if model_name != "original":
                            compilation_time = time.time() - compilation_start
                            if compilation_time > 0.1:  # 假设编译时间超过100ms表示重编译
                                recompilation_count += 1
                        
                        results[model_name]["times"].append(end_time - start_time)
                
                except Exception as e:
                    logger.warning(f"{model_name} 在输入 {i} 上失败: {e}")
                    results[model_name]["times"].append(float('inf'))
                    if "shape" in str(e).lower() or "size" in str(e).lower():
                        recompilation_count += 1
            
            results[model_name]["recompilations"] = recompilation_count
            
            # 清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["static_vs_dynamic"] = results
        return results
    
    def benchmark_shape_variation_impact(self):
        """测试形状变化对性能的影响"""
        logger.info("开始形状变化影响测试...")
        
        model = DynamicTransformer(self.config).to(self.device)
        model.eval()
        
        # 动态编译
        dynamic_model = torch.compile(model, dynamic=True)
        
        # 测试不同的形状变化模式
        shape_patterns = {
            "fixed": [(4, 512)] * 20,  # 固定形状
            "batch_varying": [(i % 8 + 1, 512) for i in range(20)],  # 批次大小变化
            "seq_varying": [(4, (i % 4 + 1) * 256) for i in range(20)],  # 序列长度变化
            "both_varying": [(i % 4 + 1, (i % 3 + 1) * 256) for i in range(20)]  # 都变化
        }
        
        results = {}
        
        for pattern_name, shapes in shape_patterns.items():
            logger.info(f"测试形状模式: {pattern_name}")
            
            times = []
            memory_usage = []
            
            for batch_size, seq_len in shapes:
                # 确保形状在配置范围内
                batch_size = min(max(batch_size, self.config.min_batch_size), self.config.max_batch_size)
                seq_len = min(max(seq_len, self.config.min_seq_len), self.config.max_seq_len)
                
                input_ids = torch.randint(0, self.config.vocab_size, 
                                        (batch_size, seq_len), device=self.device)
                attention_mask = torch.ones(batch_size, seq_len, device=self.device)
                
                try:
                    with torch.no_grad():
                        start_time = time.time()
                        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        
                        _ = dynamic_model(input_ids, attention_mask)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                        
                        times.append(end_time - start_time)
                        memory_usage.append(end_memory - start_memory)
                
                except Exception as e:
                    logger.warning(f"形状 ({batch_size}, {seq_len}) 失败: {e}")
                    times.append(float('inf'))
                    memory_usage.append(0)
            
            results[pattern_name] = {
                "times": times,
                "mean_time": np.mean([t for t in times if t != float('inf')]),
                "std_time": np.std([t for t in times if t != float('inf')]),
                "memory_usage": memory_usage,
                "shapes": shapes
            }
            
            # 清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["shape_variation"] = results
        return results
    
    def benchmark_compilation_strategies(self):
        """测试不同的编译策略"""
        logger.info("开始编译策略基准测试...")
        
        model = DynamicTransformer(self.config).to(self.device)
        model.eval()
        
        # 不同的编译策略
        strategies = {
            "no_compile": {"model": model, "config": {}},
            "dynamic_default": {"model": torch.compile(model, dynamic=True), "config": {"dynamic": True}},
            "dynamic_reduce_overhead": {"model": torch.compile(model, dynamic=True, mode="reduce-overhead"), 
                                      "config": {"dynamic": True, "mode": "reduce-overhead"}},
            "dynamic_max_autotune": {"model": torch.compile(model, dynamic=True, mode="max-autotune"), 
                                   "config": {"dynamic": True, "mode": "max-autotune"}},
        }
        
        # 生成多样化的测试输入
        test_inputs = []
        for _ in range(15):
            batch_size = random.choice([1, 2, 4, 8])
            seq_len = random.choice([128, 256, 512, 1024])
            
            input_ids = torch.randint(0, self.config.vocab_size, 
                                    (batch_size, seq_len), device=self.device)
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
            test_inputs.append((input_ids, attention_mask))
        
        results = {}
        
        for strategy_name, strategy_info in strategies.items():
            logger.info(f"测试策略: {strategy_name}")
            
            compiled_model = strategy_info["model"]
            times = []
            successful_runs = 0
            
            # 预热
            if test_inputs:
                warmup_input = test_inputs[0]
                with torch.no_grad():
                    for _ in range(2):
                        try:
                            _ = compiled_model(warmup_input[0], warmup_input[1])
                        except Exception as e:
                            logger.warning(f"{strategy_name} 预热失败: {e}")
                            break
            
            # 基准测试
            for input_ids, attention_mask in test_inputs:
                try:
                    with torch.no_grad():
                        start_time = time.time()
                        _ = compiled_model(input_ids, attention_mask)
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        end_time = time.time()
                        times.append(end_time - start_time)
                        successful_runs += 1
                
                except Exception as e:
                    logger.warning(f"{strategy_name} 运行失败: {e}")
                    times.append(float('inf'))
            
            results[strategy_name] = {
                "times": times,
                "mean_time": np.mean([t for t in times if t != float('inf')]),
                "std_time": np.std([t for t in times if t != float('inf')]),
                "success_rate": successful_runs / len(test_inputs),
                "config": strategy_info["config"]
            }
            
            # 清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["compilation_strategies"] = results
        return results
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            logger.warning("没有结果可以可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("动态形状 torch.compile 优化结果", fontsize=16)
        
        # 1. 静态 vs 动态编译对比
        if "static_vs_dynamic" in self.results:
            ax = axes[0, 0]
            data = self.results["static_vs_dynamic"]
            
            models = list(data.keys())
            mean_times = [np.mean([t for t in data[model]["times"] if t != float('inf')]) 
                         for model in models]
            recompilations = [data[model]["recompilations"] for model in models]
            
            x = np.arange(len(models))
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - 0.2, mean_times, 0.4, label="平均延迟", alpha=0.7)
            bars2 = ax2.bar(x + 0.2, recompilations, 0.4, label="重编译次数", 
                           alpha=0.7, color='red')
            
            ax.set_xlabel("编译类型")
            ax.set_ylabel("平均延迟 (秒)", color='blue')
            ax2.set_ylabel("重编译次数", color='red')
            ax.set_title("静态 vs 动态编译对比")
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            
            # 添加数值标签
            for bar, time_val in zip(bars1, mean_times):
                if time_val != float('inf'):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 形状变化影响
        if "shape_variation" in self.results:
            ax = axes[0, 1]
            data = self.results["shape_variation"]
            
            patterns = list(data.keys())
            mean_times = [data[pattern]["mean_time"] for pattern in patterns]
            std_times = [data[pattern]["std_time"] for pattern in patterns]
            
            bars = ax.bar(patterns, mean_times, yerr=std_times, capsize=5, alpha=0.7)
            ax.set_ylabel("平均延迟 (秒)")
            ax.set_title("不同形状变化模式的性能")
            ax.set_xticklabels(patterns, rotation=45)
            
            # 添加数值标签
            for bar, time_val in zip(bars, mean_times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. 编译策略对比
        if "compilation_strategies" in self.results:
            ax = axes[1, 0]
            data = self.results["compilation_strategies"]
            
            strategies = list(data.keys())
            mean_times = [data[strategy]["mean_time"] for strategy in strategies]
            success_rates = [data[strategy]["success_rate"] * 100 for strategy in strategies]
            
            x = np.arange(len(strategies))
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - 0.2, mean_times, 0.4, label="平均延迟", alpha=0.7)
            bars2 = ax2.bar(x + 0.2, success_rates, 0.4, label="成功率 (%)", 
                           alpha=0.7, color='green')
            
            ax.set_xlabel("编译策略")
            ax.set_ylabel("平均延迟 (秒)", color='blue')
            ax2.set_ylabel("成功率 (%)", color='green')
            ax.set_title("编译策略性能对比")
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45)
        
        # 4. 时间序列分析
        if "shape_variation" in self.results:
            ax = axes[1, 1]
            data = self.results["shape_variation"]
            
            for pattern_name, pattern_data in data.items():
                times = [t for t in pattern_data["times"] if t != float('inf')]
                if times:
                    ax.plot(range(len(times)), times, 'o-', label=pattern_name, alpha=0.7)
            
            ax.set_xlabel("测试步骤")
            ax.set_ylabel("延迟 (秒)")
            ax.set_title("不同形状模式的时间序列")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("dynamic_shapes_benchmark.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细结果
        self.print_detailed_results()
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("动态形状 torch.compile 优化详细结果")
        print("="*80)
        
        if "static_vs_dynamic" in self.results:
            print("\n🔄 静态 vs 动态编译对比:")
            data = self.results["static_vs_dynamic"]
            
            for model_type, stats in data.items():
                valid_times = [t for t in stats["times"] if t != float('inf')]
                if valid_times:
                    mean_time = np.mean(valid_times)
                    success_rate = len(valid_times) / len(stats["times"]) * 100
                    print(f"  {model_type:15s}: "
                          f"平均延迟={mean_time:.4f}s, "
                          f"重编译={stats['recompilations']}次, "
                          f"成功率={success_rate:.1f}%")
        
        if "shape_variation" in self.results:
            print("\n📏 形状变化影响分析:")
            data = self.results["shape_variation"]
            
            for pattern, stats in data.items():
                print(f"  {pattern:15s}: "
                      f"平均延迟={stats['mean_time']:.4f}s, "
                      f"标准差={stats['std_time']:.4f}s")
        
        if "compilation_strategies" in self.results:
            print("\n🚀 编译策略对比:")
            data = self.results["compilation_strategies"]
            
            baseline_time = data.get("no_compile", {}).get("mean_time", 1.0)
            
            for strategy, stats in data.items():
                speedup = baseline_time / stats["mean_time"] if stats["mean_time"] > 0 else 0
                print(f"  {strategy:25s}: "
                      f"延迟={stats['mean_time']:.4f}s, "
                      f"成功率={stats['success_rate']*100:.1f}%, "
                      f"加速比={speedup:.2f}x")

def main():
    """主函数"""
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("⚠️  警告: 在 CPU 上运行，动态形状优化效果可能不明显")
    
    # 配置
    config = DynamicShapeConfig(
        min_batch_size=1,
        max_batch_size=8,
        min_seq_len=128,
        max_seq_len=1024,
        hidden_size=512,  # 减小模型以加快测试
        num_heads=8,
        vocab_size=10000
    )
    
    # 创建基准测试
    benchmark = DynamicShapeBenchmark(config, device)
    
    try:
        # 1. 静态 vs 动态编译对比
        print("🔄 开始静态 vs 动态编译基准测试...")
        benchmark.benchmark_static_vs_dynamic_compilation()
        
        # 2. 形状变化影响测试
        print("📏 开始形状变化影响测试...")
        benchmark.benchmark_shape_variation_impact()
        
        # 3. 编译策略对比
        print("🚀 开始编译策略基准测试...")
        benchmark.benchmark_compilation_strategies()
        
        # 4. 可视化结果
        print("📊 生成可视化结果...")
        benchmark.visualize_results()
        
    except Exception as e:
        logger.error(f"基准测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ 动态形状处理基准测试完成!")

if __name__ == "__main__":
    main()