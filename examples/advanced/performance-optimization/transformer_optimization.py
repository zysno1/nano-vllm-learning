"""
Transformer 模型编译优化实现
演示如何使用 torch.compile 优化 Transformer 模型的推理性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
import gc
import psutil
import os
from dataclasses import dataclass
from contextlib import contextmanager
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12

@dataclass
class CompilationConfig:
    """编译配置"""
    mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    backend: str = "inductor"  # "inductor", "aot_eager", "cudagraphs"
    dynamic: bool = False
    fullgraph: bool = False
    disable: bool = False

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算 Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 处理 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        present_key_value = (key, value) if use_cache else None
        
        # 计算注意力分数
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.config.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU 激活函数
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))

class TransformerLayer(nn.Module):
    """Transformer 层"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        # 注意力层 + 残差连接
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attn_output, present_key_value = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + attn_output
        
        # 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(hidden_states)
        hidden_states = residual + ffn_output
        
        return hidden_states, present_key_value

class SimpleTransformer(nn.Module):
    """简化的 Transformer 模型"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
                use_cache: bool = False) -> Dict[str, Any]:
        
        batch_size, seq_len = input_ids.shape
        
        # 嵌入层
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 位置编码
        if past_key_values is not None:
            position_ids = torch.arange(
                past_key_values[0][0].shape[-2], 
                past_key_values[0][0].shape[-2] + seq_len,
                device=input_ids.device
            ).unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Transformer 层
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states, attention_mask, past_key_value, use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # 输出层
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None
        }

class TransformerCompilationBenchmark:
    """Transformer 编译基准测试"""
    
    def __init__(self, model_config: ModelConfig, device: str = "cuda"):
        self.model_config = model_config
        self.device = device
        self.results = {}
        
    @contextmanager
    def timer(self, name: str):
        """计时器上下文管理器"""
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        yield
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        self.results[name] = {
            "time": end_time - start_time,
            "memory": end_memory - start_memory,
            "peak_memory": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def create_sample_inputs(self, batch_size: int = 4, seq_len: int = 512) -> Dict[str, torch.Tensor]:
        """创建示例输入"""
        return {
            "input_ids": torch.randint(0, self.model_config.vocab_size, 
                                     (batch_size, seq_len), device=self.device),
            "attention_mask": torch.ones(batch_size, seq_len, device=self.device)
        }
    
    def benchmark_compilation_modes(self, num_warmup: int = 5, num_runs: int = 20):
        """基准测试不同编译模式"""
        logger.info("开始 Transformer 编译模式基准测试...")
        
        # 测试配置
        compilation_configs = [
            CompilationConfig(mode="default", backend="inductor"),
            CompilationConfig(mode="reduce-overhead", backend="inductor"),
            CompilationConfig(mode="max-autotune", backend="inductor"),
            CompilationConfig(disable=True)  # 不编译作为基准
        ]
        
        batch_size, seq_len = 4, 512
        inputs = self.create_sample_inputs(batch_size, seq_len)
        
        results = {}
        
        for config in compilation_configs:
            config_name = f"{config.mode}_{config.backend}" if not config.disable else "no_compile"
            logger.info(f"测试配置: {config_name}")
            
            # 创建模型
            model = SimpleTransformer(self.model_config).to(self.device)
            model.eval()
            
            # 应用编译
            if not config.disable:
                compiled_model = torch.compile(
                    model,
                    mode=config.mode,
                    backend=config.backend,
                    dynamic=config.dynamic,
                    fullgraph=config.fullgraph
                )
            else:
                compiled_model = model
            
            # 预热
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = compiled_model(**inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 基准测试
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = compiled_model(**inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            results[config_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "throughput": batch_size / np.mean(times),  # samples/sec
                "memory_usage": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
            
            # 清理内存
            del model, compiled_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        
        self.results["compilation_modes"] = results
        return results
    
    def benchmark_sequence_lengths(self, compilation_config: CompilationConfig):
        """基准测试不同序列长度"""
        logger.info("开始序列长度基准测试...")
        
        seq_lengths = [128, 256, 512, 1024, 2048]
        batch_size = 2
        
        # 创建编译模型
        model = SimpleTransformer(self.model_config).to(self.device)
        model.eval()
        
        if not compilation_config.disable:
            compiled_model = torch.compile(
                model,
                mode=compilation_config.mode,
                backend=compilation_config.backend,
                dynamic=compilation_config.dynamic
            )
        else:
            compiled_model = model
        
        results = {}
        
        for seq_len in seq_lengths:
            logger.info(f"测试序列长度: {seq_len}")
            
            inputs = self.create_sample_inputs(batch_size, seq_len)
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    try:
                        _ = compiled_model(**inputs)
                    except RuntimeError as e:
                        logger.warning(f"序列长度 {seq_len} 预热失败: {e}")
                        break
                else:
                    # 基准测试
                    times = []
                    for _ in range(10):
                        start_time = time.time()
                        _ = compiled_model(**inputs)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        end_time = time.time()
                        times.append(end_time - start_time)
                    
                    results[seq_len] = {
                        "mean_time": np.mean(times),
                        "throughput": batch_size / np.mean(times),
                        "memory_usage": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
                    }
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        
        self.results["sequence_lengths"] = results
        return results
    
    def benchmark_kv_cache_optimization(self):
        """基准测试 KV Cache 优化"""
        logger.info("开始 KV Cache 优化基准测试...")
        
        batch_size, seq_len = 1, 512
        generation_steps = 100
        
        # 创建模型
        model = SimpleTransformer(self.model_config).to(self.device)
        model.eval()
        
        # 编译不同的推理函数
        @torch.compile(mode="reduce-overhead")
        def compiled_generate_step(model, input_ids, past_key_values=None):
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                return outputs["logits"], outputs["past_key_values"]
        
        def regular_generate_step(model, input_ids, past_key_values=None):
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                return outputs["logits"], outputs["past_key_values"]
        
        results = {}
        
        for use_compiled, generate_fn, name in [
            (False, regular_generate_step, "regular"),
            (True, compiled_generate_step, "compiled")
        ]:
            logger.info(f"测试 {name} 生成...")
            
            # 初始输入
            input_ids = torch.randint(0, self.model_config.vocab_size, 
                                    (batch_size, seq_len), device=self.device)
            
            # 预热
            past_key_values = None
            for _ in range(5):
                next_token_id = torch.randint(0, self.model_config.vocab_size, 
                                            (batch_size, 1), device=self.device)
                logits, past_key_values = generate_fn(model, next_token_id, past_key_values)
            
            # 重置
            past_key_values = None
            
            # 基准测试
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            for step in range(generation_steps):
                if step == 0:
                    current_input = input_ids
                else:
                    # 模拟下一个 token
                    current_input = torch.randint(0, self.model_config.vocab_size, 
                                                (batch_size, 1), device=self.device)
                
                logits, past_key_values = generate_fn(model, current_input, past_key_values)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            results[name] = {
                "total_time": end_time - start_time,
                "time_per_step": (end_time - start_time) / generation_steps,
                "tokens_per_second": generation_steps / (end_time - start_time),
                "memory_usage": end_memory - start_memory
            }
            
            # 清理
            del past_key_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.results["kv_cache"] = results
        return results
    
    def visualize_results(self):
        """可视化结果"""
        if not self.results:
            logger.warning("没有结果可以可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Transformer torch.compile 优化结果", fontsize=16)
        
        # 1. 编译模式对比
        if "compilation_modes" in self.results:
            ax = axes[0, 0]
            data = self.results["compilation_modes"]
            
            modes = list(data.keys())
            times = [data[mode]["mean_time"] for mode in modes]
            throughputs = [data[mode]["throughput"] for mode in modes]
            
            x = np.arange(len(modes))
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - 0.2, times, 0.4, label="延迟 (s)", alpha=0.7)
            bars2 = ax2.bar(x + 0.2, throughputs, 0.4, label="吞吐量 (samples/s)", 
                           alpha=0.7, color='orange')
            
            ax.set_xlabel("编译模式")
            ax.set_ylabel("延迟 (秒)", color='blue')
            ax2.set_ylabel("吞吐量 (samples/s)", color='orange')
            ax.set_title("编译模式性能对比")
            ax.set_xticks(x)
            ax.set_xticklabels(modes, rotation=45)
            
            # 添加数值标签
            for bar, time_val in zip(bars1, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{time_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. 序列长度影响
        if "sequence_lengths" in self.results:
            ax = axes[0, 1]
            data = self.results["sequence_lengths"]
            
            seq_lens = list(data.keys())
            times = [data[seq_len]["mean_time"] for seq_len in seq_lens]
            
            ax.plot(seq_lens, times, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel("序列长度")
            ax.set_ylabel("延迟 (秒)")
            ax.set_title("序列长度 vs 推理时间")
            ax.grid(True, alpha=0.3)
            
            # 添加数值标签
            for seq_len, time_val in zip(seq_lens, times):
                ax.annotate(f'{time_val:.3f}', (seq_len, time_val), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # 3. KV Cache 优化
        if "kv_cache" in self.results:
            ax = axes[1, 0]
            data = self.results["kv_cache"]
            
            methods = list(data.keys())
            tokens_per_sec = [data[method]["tokens_per_second"] for method in methods]
            
            bars = ax.bar(methods, tokens_per_sec, alpha=0.7, 
                         color=['skyblue', 'lightcoral'])
            ax.set_ylabel("Tokens/秒")
            ax.set_title("KV Cache 生成速度对比")
            
            # 添加数值标签
            for bar, tps in zip(bars, tokens_per_sec):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{tps:.1f}', ha='center', va='bottom')
            
            # 计算加速比
            if len(tokens_per_sec) >= 2:
                speedup = tokens_per_sec[1] / tokens_per_sec[0]
                ax.text(0.5, max(tokens_per_sec) * 0.8, 
                       f'加速比: {speedup:.2f}x', 
                       ha='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 4. 内存使用对比
        ax = axes[1, 1]
        if "compilation_modes" in self.results:
            data = self.results["compilation_modes"]
            modes = list(data.keys())
            memory_usage = [data[mode]["memory_usage"] / (1024**3) for mode in modes]  # GB
            
            bars = ax.bar(modes, memory_usage, alpha=0.7, color='lightgreen')
            ax.set_ylabel("内存使用 (GB)")
            ax.set_title("编译模式内存使用对比")
            ax.set_xticklabels(modes, rotation=45)
            
            # 添加数值标签
            for bar, mem in zip(bars, memory_usage):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mem:.2f}GB', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("transformer_compilation_benchmark.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细结果
        self.print_detailed_results()
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("Transformer torch.compile 优化详细结果")
        print("="*80)
        
        if "compilation_modes" in self.results:
            print("\n📊 编译模式性能对比:")
            data = self.results["compilation_modes"]
            
            baseline_time = data.get("no_compile", {}).get("mean_time", 1.0)
            
            for mode, stats in data.items():
                speedup = baseline_time / stats["mean_time"] if stats["mean_time"] > 0 else 0
                print(f"  {mode:20s}: "
                      f"延迟={stats['mean_time']:.4f}s, "
                      f"吞吐量={stats['throughput']:.1f} samples/s, "
                      f"加速比={speedup:.2f}x")
        
        if "sequence_lengths" in self.results:
            print("\n📏 序列长度性能分析:")
            data = self.results["sequence_lengths"]
            
            for seq_len, stats in data.items():
                print(f"  序列长度 {seq_len:4d}: "
                      f"延迟={stats['mean_time']:.4f}s, "
                      f"吞吐量={stats['throughput']:.1f} samples/s")
        
        if "kv_cache" in self.results:
            print("\n🚀 KV Cache 优化结果:")
            data = self.results["kv_cache"]
            
            for method, stats in data.items():
                print(f"  {method:10s}: "
                      f"生成速度={stats['tokens_per_second']:.1f} tokens/s, "
                      f"每步耗时={stats['time_per_step']:.4f}s")
            
            if "regular" in data and "compiled" in data:
                speedup = data["compiled"]["tokens_per_second"] / data["regular"]["tokens_per_second"]
                print(f"  🎯 KV Cache 编译加速比: {speedup:.2f}x")

def main():
    """主函数"""
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if device == "cpu":
        print("⚠️  警告: 在 CPU 上运行，性能提升可能不明显")
    
    # 配置
    model_config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=6,  # 减少层数以加快测试
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048
    )
    
    # 创建基准测试
    benchmark = TransformerCompilationBenchmark(model_config, device)
    
    try:
        # 1. 编译模式对比
        print("🔄 开始编译模式基准测试...")
        benchmark.benchmark_compilation_modes(num_warmup=3, num_runs=10)
        
        # 2. 序列长度测试
        print("📏 开始序列长度基准测试...")
        compilation_config = CompilationConfig(mode="reduce-overhead", backend="inductor")
        benchmark.benchmark_sequence_lengths(compilation_config)
        
        # 3. KV Cache 优化测试
        print("🚀 开始 KV Cache 优化测试...")
        benchmark.benchmark_kv_cache_optimization()
        
        # 4. 可视化结果
        print("📊 生成可视化结果...")
        benchmark.visualize_results()
        
    except Exception as e:
        logger.error(f"基准测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Transformer 编译优化基准测试完成!")

if __name__ == "__main__":
    main()