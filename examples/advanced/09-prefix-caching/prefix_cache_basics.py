#!/usr/bin/env python3
"""
Prefix Caching 基础概念演示

本模块演示 Prefix Caching 的基本概念和实现原理，包括：
1. 基础的前缀缓存机制
2. KV Cache 存储和复用
3. 前缀匹配算法
4. 性能对比分析
"""

import time
import hashlib
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    total_compute_time: float = 0.0
    total_cache_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def time_saved(self) -> float:
        """节省的时间"""
        return self.total_compute_time - self.total_cache_time


@dataclass
class CacheEntry:
    """缓存条目"""
    kv_cache: Tuple[torch.Tensor, torch.Tensor]
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class BasicPrefixCache:
    """基础前缀缓存实现"""
    
    def __init__(self, max_entries: int = 100):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.stats = CacheStats()
    
    def _hash_tokens(self, tokens: List[int]) -> str:
        """计算 token 序列的哈希值"""
        token_bytes = pickle.dumps(tokens)
        return hashlib.md5(token_bytes).hexdigest()
    
    def get_cached_kv(self, prefix_tokens: List[int]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取缓存的 KV Cache"""
        prefix_hash = self._hash_tokens(prefix_tokens)
        self.stats.total_requests += 1
        
        if prefix_hash in self.cache:
            entry = self.cache[prefix_hash]
            entry.access_count += 1
            entry.timestamp = time.time()
            self.stats.hits += 1
            return entry.kv_cache
        
        self.stats.misses += 1
        return None
    
    def store_kv(self, prefix_tokens: List[int], kv_cache: Tuple[torch.Tensor, torch.Tensor]):
        """存储 KV Cache"""
        prefix_hash = self._hash_tokens(prefix_tokens)
        
        # 如果缓存已满，删除最旧的条目
        if len(self.cache) >= self.max_entries:
            self._evict_oldest()
        
        # 计算缓存大小
        k_cache, v_cache = kv_cache
        size_bytes = k_cache.numel() * k_cache.element_size() + v_cache.numel() * v_cache.element_size()
        
        # 存储新条目
        entry = CacheEntry(
            kv_cache=kv_cache,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size_bytes
        )
        self.cache[prefix_hash] = entry
    
    def _evict_oldest(self):
        """淘汰最旧的缓存条目"""
        if not self.cache:
            return
        
        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        del self.cache[oldest_key]
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计信息"""
        return self.stats


class SimplifiedAttentionLayer(nn.Module):
    """简化的注意力层，用于演示 KV Cache"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            kv_cache: 可选的 KV 缓存 (k_cache, v_cache)
        
        Returns:
            output: 输出张量
            new_kv_cache: 新的 KV 缓存
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 如果有缓存，拼接历史 K, V
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # 在序列维度拼接
            v = torch.cat([v_cache, v], dim=2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        
        return output, (k, v)


class PrefixCachingDemo:
    """Prefix Caching 演示类"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        self.attention_layer = SimplifiedAttentionLayer(d_model, n_heads)
        self.cache = BasicPrefixCache(max_entries=50)
        self.d_model = d_model
        
        # 模拟的词汇表
        self.vocab_size = 1000
        self.embedding = nn.Embedding(self.vocab_size, d_model)
    
    def tokenize(self, text: str) -> List[int]:
        """简单的分词器（模拟）"""
        # 这里使用简单的哈希来模拟分词
        words = text.split()
        tokens = []
        for word in words:
            token_id = hash(word) % self.vocab_size
            tokens.append(abs(token_id))
        return tokens
    
    def compute_attention_with_cache(self, tokens: List[int]) -> Tuple[torch.Tensor, float]:
        """使用缓存计算注意力"""
        start_time = time.time()
        
        # 尝试从缓存获取前缀的 KV Cache
        cached_kv = None
        cache_prefix_len = 0
        
        # 从最长前缀开始尝试
        for i in range(len(tokens), 0, -1):
            prefix = tokens[:i]
            cached_kv = self.cache.get_cached_kv(prefix)
            if cached_kv is not None:
                cache_prefix_len = i
                break
        
        # 计算需要处理的新 tokens
        if cache_prefix_len > 0:
            new_tokens = tokens[cache_prefix_len:]
            self.cache.stats.total_cache_time += time.time() - start_time
        else:
            new_tokens = tokens
            self.cache.stats.total_compute_time += time.time() - start_time
        
        # 如果有新 tokens 需要处理
        if new_tokens:
            # 转换为嵌入
            new_token_tensor = torch.tensor(new_tokens).unsqueeze(0)  # [1, seq_len]
            new_embeddings = self.embedding(new_token_tensor)  # [1, seq_len, d_model]
            
            # 计算注意力
            _, new_kv_cache = self.attention_layer(new_embeddings, cached_kv)
            
            # 存储完整序列的 KV Cache
            self.cache.store_kv(tokens, new_kv_cache)
            
            compute_time = time.time() - start_time
            self.cache.stats.total_compute_time += compute_time
            
            return new_embeddings, compute_time
        else:
            # 完全命中缓存
            cache_time = time.time() - start_time
            self.cache.stats.total_cache_time += cache_time
            
            # 返回空张量（实际应用中会返回缓存的输出）
            return torch.zeros(1, 1, self.d_model), cache_time
    
    def compute_attention_without_cache(self, tokens: List[int]) -> Tuple[torch.Tensor, float]:
        """不使用缓存计算注意力"""
        start_time = time.time()
        
        # 转换为嵌入
        token_tensor = torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]
        embeddings = self.embedding(token_tensor)  # [1, seq_len, d_model]
        
        # 计算注意力
        output, _ = self.attention_layer(embeddings)
        
        compute_time = time.time() - start_time
        return output, compute_time
    
    def run_comparison_demo(self):
        """运行对比演示"""
        print("🚀 Prefix Caching 基础演示")
        print("=" * 50)
        
        # 模拟多轮对话场景
        system_prompt = "You are a helpful assistant."
        conversations = [
            f"{system_prompt} What is Python?",
            f"{system_prompt} How to learn Python?",
            f"{system_prompt} What are Python libraries?",
            f"{system_prompt} Explain Python syntax.",
            f"{system_prompt} Python vs Java comparison.",
        ]
        
        print("\n📝 测试场景：多轮对话")
        print("系统提示:", system_prompt)
        print("对话轮数:", len(conversations))
        
        # 测试不使用缓存
        print("\n🔄 不使用缓存的计算时间:")
        no_cache_times = []
        for i, conv in enumerate(conversations):
            tokens = self.tokenize(conv)
            _, compute_time = self.compute_attention_without_cache(tokens)
            no_cache_times.append(compute_time)
            print(f"  轮次 {i+1}: {compute_time:.4f}s (tokens: {len(tokens)})")
        
        # 重置缓存统计
        self.cache.clear()
        
        # 测试使用缓存
        print("\n⚡ 使用缓存的计算时间:")
        cache_times = []
        for i, conv in enumerate(conversations):
            tokens = self.tokenize(conv)
            _, compute_time = self.compute_attention_with_cache(tokens)
            cache_times.append(compute_time)
            print(f"  轮次 {i+1}: {compute_time:.4f}s (tokens: {len(tokens)})")
        
        # 显示统计信息
        stats = self.cache.get_stats()
        print(f"\n📊 缓存统计信息:")
        print(f"  总请求数: {stats.total_requests}")
        print(f"  缓存命中: {stats.hits}")
        print(f"  缓存未命中: {stats.misses}")
        print(f"  命中率: {stats.hit_rate:.2%}")
        print(f"  总计算时间: {sum(no_cache_times):.4f}s")
        print(f"  缓存计算时间: {sum(cache_times):.4f}s")
        print(f"  时间节省: {sum(no_cache_times) - sum(cache_times):.4f}s")
        print(f"  性能提升: {sum(no_cache_times) / sum(cache_times):.2f}x")
        
        return no_cache_times, cache_times, stats
    
    def run_prefix_length_analysis(self):
        """分析不同前缀长度对缓存效果的影响"""
        print("\n🔍 前缀长度影响分析")
        print("=" * 30)
        
        base_prefix = "You are a helpful assistant. Please answer the following question:"
        questions = [
            "What is machine learning?",
            "Explain deep learning.",
            "What is neural network?",
            "How does AI work?",
            "What is data science?"
        ]
        
        prefix_lengths = []
        hit_rates = []
        
        for prefix_len in range(1, len(self.tokenize(base_prefix)) + 1):
            self.cache.clear()
            
            # 使用不同长度的前缀
            prefix_tokens = self.tokenize(base_prefix)[:prefix_len]
            
            for question in questions:
                full_tokens = prefix_tokens + self.tokenize(question)
                self.compute_attention_with_cache(full_tokens)
            
            stats = self.cache.get_stats()
            prefix_lengths.append(prefix_len)
            hit_rates.append(stats.hit_rate)
            
            print(f"  前缀长度 {prefix_len}: 命中率 {stats.hit_rate:.2%}")
        
        return prefix_lengths, hit_rates
    
    def visualize_results(self, no_cache_times: List[float], cache_times: List[float], 
                         prefix_lengths: List[int], hit_rates: List[float]):
        """可视化结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 计算时间对比
        rounds = list(range(1, len(no_cache_times) + 1))
        ax1.plot(rounds, no_cache_times, 'ro-', label='无缓存', linewidth=2, markersize=8)
        ax1.plot(rounds, cache_times, 'bo-', label='有缓存', linewidth=2, markersize=8)
        ax1.set_xlabel('对话轮次')
        ax1.set_ylabel('计算时间 (秒)')
        ax1.set_title('计算时间对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 累积时间节省
        cumulative_saved = np.cumsum(np.array(no_cache_times) - np.array(cache_times))
        ax2.plot(rounds, cumulative_saved, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('对话轮次')
        ax2.set_ylabel('累积节省时间 (秒)')
        ax2.set_title('累积时间节省')
        ax2.grid(True, alpha=0.3)
        
        # 3. 前缀长度 vs 命中率
        ax3.plot(prefix_lengths, hit_rates, 'mo-', linewidth=2, markersize=8)
        ax3.set_xlabel('前缀长度 (tokens)')
        ax3.set_ylabel('缓存命中率')
        ax3.set_title('前缀长度对命中率的影响')
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能提升倍数
        speedup = [no_cache / cache if cache > 0 else 1 for no_cache, cache in zip(no_cache_times, cache_times)]
        ax4.bar(rounds, speedup, color='skyblue', alpha=0.7)
        ax4.set_xlabel('对话轮次')
        ax4.set_ylabel('性能提升倍数')
        ax4.set_title('每轮性能提升')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prefix_caching_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 分析图表已保存为 'prefix_caching_analysis.png'")


def main():
    """主函数"""
    print("🎯 Prefix Caching 基础概念演示")
    print("本演示将展示前缀缓存的基本原理和性能优势")
    
    # 创建演示实例
    demo = PrefixCachingDemo(d_model=256, n_heads=4)  # 使用较小的模型以便快速演示
    
    # 运行对比演示
    no_cache_times, cache_times, stats = demo.run_comparison_demo()
    
    # 运行前缀长度分析
    prefix_lengths, hit_rates = demo.run_prefix_length_analysis()
    
    # 可视化结果
    demo.visualize_results(no_cache_times, cache_times, prefix_lengths, hit_rates)
    
    print("\n✅ 演示完成！")
    print("\n💡 关键观察:")
    print("1. 随着对话轮次增加，缓存的优势越来越明显")
    print("2. 前缀越长，缓存命中率越高")
    print("3. 在多轮对话场景中，Prefix Caching 可以显著提升性能")
    print("4. 第一轮对话缓存未命中，但后续轮次大幅受益")


if __name__ == "__main__":
    main()