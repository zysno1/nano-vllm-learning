#!/usr/bin/env python3
"""
LLM 推理 CUDA Graph 优化实现

本模块演示如何在实际的 LLM 推理中应用 CUDA Graph 优化，包括：
1. 解码阶段的 CUDA Graph 优化
2. 批处理推理优化
3. 动态批处理的处理策略
4. 与其他优化技术的结合

Author: nano-vLLM-learning
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class InferenceConfig:
    """推理配置"""
    model_name: str = "llama-7b"
    max_seq_len: int = 2048
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    intermediate_size: int = 11008
    max_batch_size: int = 32
    use_cuda_graph: bool = True
    enable_kv_cache: bool = True
    dtype: torch.dtype = torch.float16


class SimplifiedLLMLayer(nn.Module):
    """简化的 LLM 层实现"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        
        # 注意力层
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # MLP 层
        self.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, self.hidden_size, bias=False)
        
        # Layer Norm
        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 注意力层
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 计算 Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV Cache 处理
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.o_proj(attn_output)
        
        # 残差连接
        hidden_states = residual + attn_output
        
        # 2. MLP 层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # SwiGLU 激活
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states = self.down_proj(hidden_states)
        
        # 残差连接
        hidden_states = residual + hidden_states
        
        # 返回新的 KV Cache
        new_kv_cache = (k, v)
        
        return hidden_states, new_kv_cache


class SimplifiedLLM(nn.Module):
    """简化的 LLM 模型"""
    
    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            SimplifiedLLMLayer(config) for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """前向传播"""
        # 嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 初始化 KV Cache
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        
        new_kv_caches = []
        
        # 通过所有层
        for i, layer in enumerate(self.layers):
            hidden_states, new_kv_cache = layer(
                hidden_states, attention_mask, kv_caches[i]
            )
            new_kv_caches.append(new_kv_cache)
        
        # 最终归一化和输出
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, new_kv_caches


class CUDAGraphInferenceEngine:
    """CUDA Graph 推理引擎"""
    
    def __init__(self, model: SimplifiedLLM, config: InferenceConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # CUDA Graph 相关
        self.cuda_graphs = {}  # 存储不同配置的 CUDA Graph
        self.static_inputs = {}  # 静态输入张量
        self.static_outputs = {}  # 静态输出张量
        self.static_kv_caches = {}  # 静态 KV Cache
        
        # 性能监控
        self.metrics = defaultdict(list)
        
        print(f"🚀 CUDA Graph 推理引擎初始化完成")
        print(f"  模型: {config.model_name}")
        print(f"  设备: {self.device}")
        print(f"  最大批次大小: {config.max_batch_size}")
        print(f"  最大序列长度: {config.max_seq_len}")
    
    def _get_cache_key(self, batch_size: int, seq_len: int, is_prefill: bool) -> str:
        """生成缓存键"""
        return f"b{batch_size}_s{seq_len}_{'prefill' if is_prefill else 'decode'}"
    
    def _create_static_tensors(self, batch_size: int, seq_len: int, is_prefill: bool):
        """创建静态张量"""
        cache_key = self._get_cache_key(batch_size, seq_len, is_prefill)
        
        # 输入张量
        static_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=torch.long, device=self.device
        )
        
        static_attention_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len), dtype=self.config.dtype, device=self.device
        )
        
        # KV Cache 张量
        static_kv_caches = []
        if not is_prefill:  # 解码阶段需要 KV Cache
            for _ in range(self.config.num_layers):
                # 假设已有的序列长度
                past_seq_len = self.config.max_seq_len - seq_len
                k_cache = torch.zeros(
                    (batch_size, self.config.num_heads, past_seq_len, self.config.head_dim),
                    dtype=self.config.dtype, device=self.device
                )
                v_cache = torch.zeros(
                    (batch_size, self.config.num_heads, past_seq_len, self.config.head_dim),
                    dtype=self.config.dtype, device=self.device
                )
                static_kv_caches.append((k_cache, v_cache))
        else:
            static_kv_caches = None
        
        self.static_inputs[cache_key] = {
            'input_ids': static_input_ids,
            'attention_mask': static_attention_mask,
            'kv_caches': static_kv_caches
        }
        
        return cache_key
    
    def capture_cuda_graph(self, batch_size: int, seq_len: int, is_prefill: bool = False):
        """捕获 CUDA Graph"""
        cache_key = self._get_cache_key(batch_size, seq_len, is_prefill)
        
        if cache_key in self.cuda_graphs:
            print(f"  ⚡ CUDA Graph 已存在: {cache_key}")
            return cache_key
        
        print(f"  📸 捕获 CUDA Graph: {cache_key}")
        
        # 创建静态张量
        self._create_static_tensors(batch_size, seq_len, is_prefill)
        
        static_inputs = self.static_inputs[cache_key]
        
        # 预热
        print(f"    🔥 预热阶段...")
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(
                    static_inputs['input_ids'],
                    static_inputs['attention_mask'],
                    static_inputs['kv_caches']
                )
        
        torch.cuda.synchronize()
        
        # 捕获图
        print(f"    📸 捕获计算图...")
        cuda_graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(cuda_graph):
            with torch.no_grad():
                static_logits, static_new_kv_caches = self.model(
                    static_inputs['input_ids'],
                    static_inputs['attention_mask'],
                    static_inputs['kv_caches']
                )
        
        # 存储图和输出
        self.cuda_graphs[cache_key] = cuda_graph
        self.static_outputs[cache_key] = {
            'logits': static_logits,
            'kv_caches': static_new_kv_caches
        }
        
        print(f"    ✅ CUDA Graph 捕获完成")
        return cache_key
    
    def inference_with_cuda_graph(self, input_ids: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 kv_caches: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """使用 CUDA Graph 进行推理"""
        batch_size, seq_len = input_ids.shape
        is_prefill = kv_caches is None
        
        cache_key = self._get_cache_key(batch_size, seq_len, is_prefill)
        
        # 如果没有对应的图，先捕获
        if cache_key not in self.cuda_graphs:
            self.capture_cuda_graph(batch_size, seq_len, is_prefill)
        
        # 复制输入数据到静态张量
        static_inputs = self.static_inputs[cache_key]
        static_inputs['input_ids'].copy_(input_ids)
        
        if attention_mask is not None:
            static_inputs['attention_mask'].copy_(attention_mask)
        
        if kv_caches is not None and static_inputs['kv_caches'] is not None:
            for i, (k, v) in enumerate(kv_caches):
                static_inputs['kv_caches'][i][0].copy_(k)
                static_inputs['kv_caches'][i][1].copy_(v)
        
        # 重放图
        self.cuda_graphs[cache_key].replay()
        
        # 获取输出
        static_outputs = self.static_outputs[cache_key]
        
        # 克隆输出以避免被后续操作覆盖
        output_logits = static_outputs['logits'].clone()
        output_kv_caches = []
        
        if static_outputs['kv_caches'] is not None:
            for k, v in static_outputs['kv_caches']:
                output_kv_caches.append((k.clone(), v.clone()))
        
        return output_logits, output_kv_caches
    
    def inference_traditional(self, input_ids: torch.Tensor,
                            attention_mask: Optional[torch.Tensor] = None,
                            kv_caches: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        """传统推理方式"""
        with torch.no_grad():
            return self.model(input_ids, attention_mask, kv_caches)
    
    def benchmark_inference_methods(self, batch_sizes: List[int] = [1, 4, 8, 16],
                                  seq_lengths: List[int] = [1, 32, 128],
                                  num_runs: int = 100) -> Dict:
        """对比不同推理方法的性能"""
        print(f"\n🎯 推理方法性能对比")
        print(f"  批次大小: {batch_sizes}")
        print(f"  序列长度: {seq_lengths}")
        print(f"  运行次数: {num_runs}")
        
        results = {
            'batch_sizes': batch_sizes,
            'seq_lengths': seq_lengths,
            'traditional_times': [],
            'cuda_graph_times': [],
            'speedup_ratios': [],
            'memory_usage': []
        }
        
        for batch_size in batch_sizes:
            batch_traditional_times = []
            batch_cuda_graph_times = []
            batch_speedup_ratios = []
            batch_memory_usage = []
            
            for seq_len in seq_lengths:
                print(f"\n  📏 测试配置: batch_size={batch_size}, seq_len={seq_len}")
                
                # 创建测试输入
                input_ids = torch.randint(
                    0, self.config.vocab_size, (batch_size, seq_len), device=self.device
                )
                
                # 创建注意力掩码
                attention_mask = torch.zeros(
                    (batch_size, 1, seq_len, seq_len), 
                    dtype=self.config.dtype, device=self.device
                )
                
                # 1. 传统方式性能测试
                print(f"    🔄 传统方式测试...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 预热
                for _ in range(10):
                    _ = self.inference_traditional(input_ids, attention_mask)
                
                torch.cuda.synchronize()
                
                # 测试
                start_time = time.time()
                for _ in range(num_runs):
                    _ = self.inference_traditional(input_ids, attention_mask)
                torch.cuda.synchronize()
                
                traditional_time = (time.time() - start_time) / num_runs
                traditional_memory = torch.cuda.max_memory_allocated()
                
                # 2. CUDA Graph 方式性能测试
                print(f"    🚀 CUDA Graph 测试...")
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 预热（包括图捕获）
                cache_key = self.capture_cuda_graph(batch_size, seq_len, is_prefill=True)
                
                for _ in range(10):
                    _ = self.inference_with_cuda_graph(input_ids, attention_mask)
                
                torch.cuda.synchronize()
                
                # 测试
                start_time = time.time()
                for _ in range(num_runs):
                    _ = self.inference_with_cuda_graph(input_ids, attention_mask)
                torch.cuda.synchronize()
                
                cuda_graph_time = (time.time() - start_time) / num_runs
                cuda_graph_memory = torch.cuda.max_memory_allocated()
                
                # 计算指标
                speedup = traditional_time / cuda_graph_time
                memory_overhead = (cuda_graph_memory - traditional_memory) / traditional_memory * 100
                
                batch_traditional_times.append(traditional_time)
                batch_cuda_graph_times.append(cuda_graph_time)
                batch_speedup_ratios.append(speedup)
                batch_memory_usage.append(memory_overhead)
                
                print(f"      传统方式: {traditional_time*1000:.3f}ms")
                print(f"      CUDA Graph: {cuda_graph_time*1000:.3f}ms")
                print(f"      加速比: {speedup:.2f}x")
                print(f"      内存开销: {memory_overhead:.1f}%")
            
            results['traditional_times'].append(batch_traditional_times)
            results['cuda_graph_times'].append(batch_cuda_graph_times)
            results['speedup_ratios'].append(batch_speedup_ratios)
            results['memory_usage'].append(batch_memory_usage)
        
        return results
    
    def simulate_text_generation(self, prompt_tokens: torch.Tensor, 
                               max_new_tokens: int = 50) -> Dict:
        """模拟文本生成过程"""
        print(f"\n📝 模拟文本生成")
        print(f"  输入长度: {prompt_tokens.shape[1]}")
        print(f"  生成长度: {max_new_tokens}")
        
        batch_size = prompt_tokens.shape[0]
        
        # 1. Prefill 阶段（传统方式）
        print(f"  🔄 Prefill 阶段 (传统方式)...")
        start_time = time.time()
        
        logits, kv_caches = self.inference_traditional(prompt_tokens)
        next_token = torch.argmax(logits[:, -1:, :], dim=-1)
        
        prefill_time = time.time() - start_time
        
        # 2. Decode 阶段（CUDA Graph 优化）
        print(f"  🚀 Decode 阶段 (CUDA Graph 优化)...")
        
        # 捕获解码图
        decode_cache_key = self.capture_cuda_graph(batch_size, 1, is_prefill=False)
        
        decode_times = []
        generated_tokens = [next_token]
        
        for i in range(max_new_tokens - 1):
            start_time = time.time()
            
            # 使用 CUDA Graph 进行解码
            logits, kv_caches = self.inference_with_cuda_graph(
                next_token, kv_caches=kv_caches
            )
            next_token = torch.argmax(logits[:, -1:, :], dim=-1)
            
            decode_time = time.time() - start_time
            decode_times.append(decode_time)
            generated_tokens.append(next_token)
            
            if (i + 1) % 10 == 0:
                print(f"    生成进度: {i + 1}/{max_new_tokens - 1}")
        
        # 统计结果
        total_decode_time = sum(decode_times)
        avg_decode_time = total_decode_time / len(decode_times)
        
        results = {
            'prefill_time': prefill_time,
            'total_decode_time': total_decode_time,
            'avg_decode_time': avg_decode_time,
            'decode_times': decode_times,
            'tokens_per_second': len(decode_times) / total_decode_time,
            'generated_tokens': torch.cat(generated_tokens, dim=1)
        }
        
        print(f"  📊 生成统计:")
        print(f"    Prefill 时间: {prefill_time*1000:.2f}ms")
        print(f"    总解码时间: {total_decode_time*1000:.2f}ms")
        print(f"    平均解码时间: {avg_decode_time*1000:.2f}ms")
        print(f"    生成速度: {results['tokens_per_second']:.1f} tokens/s")
        
        return results


def main():
    """主函数：运行 LLM 推理优化演示"""
    print("🚀 LLM 推理 CUDA Graph 优化演示")
    print("=" * 60)
    
    try:
        # 配置
        config = InferenceConfig(
            model_name="simplified-llm",
            max_seq_len=1024,
            vocab_size=32000,
            hidden_size=2048,  # 减小模型以节省内存
            num_layers=8,      # 减少层数
            num_heads=16,
            head_dim=128,
            intermediate_size=5504,
            max_batch_size=16,
            use_cuda_graph=True,
            dtype=torch.float16
        )
        
        print(f"📊 模型配置:")
        print(f"  模型大小: {config.hidden_size}d, {config.num_layers}层")
        print(f"  词汇表大小: {config.vocab_size}")
        print(f"  最大序列长度: {config.max_seq_len}")
        print(f"  数据类型: {config.dtype}")
        
        # 创建模型
        print(f"\n🏗️  创建模型...")
        model = SimplifiedLLM(config).cuda().to(config.dtype)
        model.eval()
        
        # 计算模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params / 1e6:.1f}M")
        
        # 创建推理引擎
        print(f"\n🚀 创建推理引擎...")
        engine = CUDAGraphInferenceEngine(model, config)
        
        # 1. 基础性能对比
        print(f"\n" + "="*60)
        print(f"🎯 基础推理性能对比")
        print(f"="*60)
        
        results = engine.benchmark_inference_methods(
            batch_sizes=[1, 4, 8],
            seq_lengths=[1, 16, 64],
            num_runs=50
        )
        
        # 计算平均加速比
        avg_speedup = np.mean([np.mean(speedups) for speedups in results['speedup_ratios']])
        print(f"\n📊 性能总结:")
        print(f"  平均加速比: {avg_speedup:.2f}x")
        
        # 2. 文本生成模拟
        print(f"\n" + "="*60)
        print(f"📝 文本生成性能测试")
        print(f"="*60)
        
        # 创建模拟输入
        batch_size = 4
        prompt_length = 128
        prompt_tokens = torch.randint(
            0, config.vocab_size, (batch_size, prompt_length), device='cuda'
        )
        
        generation_results = engine.simulate_text_generation(
            prompt_tokens, max_new_tokens=30
        )
        
        # 3. 保存结果
        print(f"\n" + "="*60)
        print(f"💾 保存测试结果")
        print(f"="*60)
        
        # 保存性能数据
        performance_data = {
            'config': {
                'model_name': config.model_name,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'total_params': total_params
            },
            'benchmark_results': {
                'batch_sizes': results['batch_sizes'],
                'seq_lengths': results['seq_lengths'],
                'avg_speedup': avg_speedup,
                'speedup_matrix': results['speedup_ratios']
            },
            'generation_results': {
                'prefill_time': generation_results['prefill_time'],
                'avg_decode_time': generation_results['avg_decode_time'],
                'tokens_per_second': generation_results['tokens_per_second']
            }
        }
        
        with open('llm_inference_performance.json', 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        print(f"  📄 性能数据已保存: llm_inference_performance.json")
        
        # 4. 总结报告
        print(f"\n" + "="*60)
        print(f"📋 优化效果总结")
        print(f"="*60)
        
        print(f"✅ LLM 推理 CUDA Graph 优化完成!")
        print(f"")
        print(f"🎯 关键性能指标:")
        print(f"  • 平均推理加速比: {avg_speedup:.2f}x")
        print(f"  • 解码阶段加速: {generation_results['tokens_per_second']:.1f} tokens/s")
        print(f"  • Prefill 时间: {generation_results['prefill_time']*1000:.2f}ms")
        print(f"  • 平均解码时间: {generation_results['avg_decode_time']*1000:.2f}ms")
        print(f"")
        print(f"💡 优化效果最佳场景:")
        print(f"  • 解码阶段 (单 token 生成)")
        print(f"  • 固定批次大小的推理")
        print(f"  • 重复的推理模式")
        print(f"")
        print(f"⚠️  实际部署建议:")
        print(f"  • Prefill 阶段使用传统方式")
        print(f"  • Decode 阶段使用 CUDA Graph")
        print(f"  • 预先捕获常用配置的图")
        print(f"  • 监控内存使用量")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()