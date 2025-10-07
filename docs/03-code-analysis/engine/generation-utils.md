# 生成工具分析

## 🎯 生成工具概览

生成工具是 nano-vllm 中负责文本生成逻辑的核心组件。它包含各种采样策略、生成控制机制、以及输出后处理功能。高效的生成工具实现直接影响生成质量和推理性能。

**设计思想：**
- **多样化策略**：提供多种采样策略满足不同生成需求
- **高效实现**：优化采样算法提升生成速度
- **可控生成**：支持多种停止条件和生成控制
- **质量保证**：通过后处理确保输出质量

## 🏗️ 核心架构

```python
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import threading
import math
import random
from concurrent.futures import ThreadPoolExecutor
import logging

from nano_vllm.config import GenerationConfig
from nano_vllm.utils.logger import get_logger

logger = get_logger(__name__)

class SamplingStrategy(Enum):
    """采样策略枚举
    
    设计思想：
    - 策略多样性：涵盖从确定性到随机性的各种采样方法
    - 质量控制：不同策略适用于不同的生成质量要求
    - 性能平衡：在生成质量和计算效率间找到平衡
    """
    GREEDY = "greedy"          # 贪心采样：选择概率最高的token
    MULTINOMIAL = "multinomial" # 多项式采样：按概率分布随机采样
    TOP_K = "top_k"            # Top-K采样：从概率最高的K个token中采样
    TOP_P = "top_p"            # Top-P采样：从累积概率达到P的token中采样
    TEMPERATURE = "temperature" # 温度采样：通过温度调节概率分布
    BEAM_SEARCH = "beam_search" # 束搜索：保持多个候选序列
    NUCLEUS = "nucleus"         # 核采样：Top-P的改进版本
    TYPICAL = "typical"         # 典型采样：基于信息论的采样策略

class StoppingCriteria(Enum):
    """停止条件枚举
    
    设计思想：
    - 灵活控制：提供多种停止条件满足不同需求
    - 安全保障：防止无限生成消耗资源
    - 质量保证：在合适的时机停止生成
    """
    MAX_LENGTH = "max_length"   # 最大长度：达到指定长度时停止
    EOS_TOKEN = "eos_token"     # 结束标记：遇到EOS token时停止
    STOP_WORDS = "stop_words"   # 停止词：遇到指定词汇时停止
    CUSTOM = "custom"           # 自定义：用户定义的停止条件

@dataclass
class GenerationStats:
    """生成统计信息
    
    设计思想：
    - 全面监控：记录生成过程的各项指标
    - 性能分析：提供性能优化的数据支持
    - 质量评估：帮助评估生成质量和效率
    """
    total_tokens_generated: int = 0      # 总生成token数
    total_sequences_generated: int = 0   # 总生成序列数
    generation_time: float = 0.0         # 总生成时间
    average_tokens_per_sequence: float = 0.0  # 平均每序列token数
    sampling_time: float = 0.0           # 采样时间
    postprocessing_time: float = 0.0     # 后处理时间
    cache_hits: int = 0                  # 缓存命中次数
    cache_misses: int = 0                # 缓存未命中次数

@dataclass
class GenerationResult:
    """生成结果
    
    设计思想：
    - 完整信息：包含生成序列及相关元信息
    - 调试支持：提供logprobs和attention权重用于分析
    - 状态跟踪：记录生成状态和停止原因
    """
    sequences: List[torch.Tensor]                    # 生成的序列
    scores: Optional[List[float]] = None             # 序列得分
    logprobs: Optional[List[torch.Tensor]] = None    # 对数概率
    attention_weights: Optional[List[torch.Tensor]] = None  # 注意力权重
    generation_time: float = 0.0                    # 生成时间
    num_tokens: int = 0                             # token数量
    finished: bool = False                          # 是否完成
    stop_reason: Optional[str] = None               # 停止原因

@dataclass
class BeamSearchState:
    """束搜索状态
    
    设计思想：
    - 状态管理：维护束搜索过程中的所有状态信息
    - 内存效率：紧凑的状态表示减少内存占用
    - 并行友好：支持批量处理和并行计算
    """
    sequences: torch.Tensor  # [beam_size, seq_len] 候选序列
    scores: torch.Tensor     # [beam_size] 序列得分
    finished: torch.Tensor   # [beam_size] 完成标记
    
class BaseSampler(ABC):
    """基础采样器抽象类
    
    设计思想：
    - 统一接口：为所有采样策略提供统一的抽象接口
    - 可扩展性：便于添加新的采样策略
    - 统计支持：内置统计信息收集功能
    - 配置驱动：通过配置控制采样行为
    """
    
    def __init__(self, config: GenerationConfig):
        """初始化采样器
        
        设计思想：
        - 配置中心化：通过配置对象统一管理参数
        - 统计初始化：准备性能监控基础设施
        - 资源准备：预分配必要的计算资源
        """
        self.config = config
        self.stats = GenerationStats()
        
    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """采样下一个token
        
        设计思想：
        - 抽象接口：定义所有采样器必须实现的核心方法
        - 灵活参数：支持传递额外的采样参数
        - 上下文感知：可以利用历史token信息
        """
        pass
    
    def update_stats(self, sampling_time: float, num_samples: int = 1):
        """更新统计信息
        
        设计思想：
        - 实时统计：及时更新性能指标
        - 累积计算：维护全局统计信息
        - 性能监控：为性能优化提供数据支持
        """
        self.stats.sampling_time += sampling_time
        self.stats.total_tokens_generated += num_samples

class GreedySampler(BaseSampler):
    """贪心采样器
    
    设计思想：
    - 确定性采样：总是选择概率最高的token，保证输出的一致性
    - 高效实现：简单的argmax操作，计算开销最小
    - 适用场景：事实性问答、代码生成等需要准确性的任务
    - 缺点分析：缺乏多样性，容易产生重复内容
    """
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """贪心采样实现
        
        设计思想：
        - 简单高效：直接使用argmax选择最大概率token
        - 性能监控：记录采样时间用于性能分析
        - 批量支持：支持批量输入的并行处理
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的logits
            past_tokens: 历史token序列（贪心采样中未使用）
            
        Returns:
            next_tokens: [batch_size] 采样得到的下一个token
        """
        start_time = time.time()
        
        # 选择概率最大的token - 核心贪心策略
        # argmax操作：O(vocab_size)时间复杂度，内存友好
        next_tokens = torch.argmax(logits, dim=-1)
        
        # 性能统计：记录采样时间和token数量
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class MultinomialSampler(BaseSampler):
    """多项式采样器
    
    设计思想：
    - 概率采样：按照模型输出的概率分布进行随机采样
    - 温度控制：通过temperature参数调节采样的随机性
    - 多样性保证：相比贪心采样提供更多的输出多样性
    - 平衡策略：在确定性和随机性之间找到平衡
    """
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """多项式采样实现
        
        设计思想：
        - 温度缩放：temperature控制概率分布的尖锐程度
          * temperature < 1.0：分布更尖锐，更确定性
          * temperature > 1.0：分布更平滑，更随机性
          * temperature = 1.0：保持原始分布
        - 概率归一化：使用softmax确保概率和为1
        - 随机采样：使用multinomial进行概率采样
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的logits
            past_tokens: 历史token序列（当前未使用）
            temperature: 温度参数，控制采样随机性
            
        Returns:
            next_tokens: [batch_size] 采样得到的下一个token
        """
        start_time = time.time()
        
        # 温度缩放：调节概率分布的尖锐程度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 概率归一化：将logits转换为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 多项式采样：按概率分布随机选择token
        # multinomial采样保证了概率较高的token有更大被选中概率
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 性能统计
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TopKSampler(BaseSampler):
    """Top-K采样器
    
    设计思想：
    - 候选限制：只从概率最高的K个token中采样
    - 质量保证：过滤掉低概率的不合理token
    - 可控多样性：通过K值控制候选范围大小
    - 计算效率：相比全词汇表采样减少了计算量
    """
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-K采样实现
        
        设计思想：
        - 两阶段过程：
          1. 选择top-k个最高概率的token
          2. 在这k个token中进行概率采样
        - 质量控制：避免选择极低概率的无意义token
        - 参数平衡：k值需要在质量和多样性间平衡
          * k太小：输出过于确定，缺乏多样性
          * k太大：可能包含低质量token
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的logits
            past_tokens: 历史token序列（当前未使用）
            top_k: 候选token数量
            temperature: 温度参数
            
        Returns:
            next_tokens: [batch_size] 采样得到的下一个token
        """
        start_time = time.time()
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 获取top-k值和对应的索引
        # topk操作：O(vocab_size * log(k))时间复杂度
        top_k_values, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        
        # 计算top-k token的概率分布
        top_k_probs = F.softmax(top_k_values, dim=-1)
        
        # 在top-k候选中进行多项式采样
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        
        # 将采样索引映射回原始词汇表索引
        next_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        
        # 性能统计
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TopPSampler(BaseSampler):
    """Top-P (Nucleus) 采样器
    
    设计思想：
    - 动态候选集：根据累积概率动态确定候选token数量
    - 自适应性：不同时刻的候选集大小可能不同
    - 概率阈值：只考虑累积概率达到p的最小token集合
    - 质量保证：自动过滤掉概率过低的token
    """
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-P采样实现
        
        设计思想：
        - 核心算法：
          1. 计算所有token的概率并排序
          2. 计算累积概率分布
          3. 找到累积概率刚好超过p的位置
          4. 在这个"核"集合中进行采样
        - 自适应特性：候选集大小根据概率分布自动调整
        - 边界处理：确保至少保留一个token避免空集合
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的logits
            past_tokens: 历史token序列（当前未使用）
            top_p: 累积概率阈值 (0.0, 1.0]
            temperature: 温度参数
            
        Returns:
            next_tokens: [batch_size] 采样得到的下一个token
        """
        start_time = time.time()
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算概率分布并按降序排序
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率分布
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到需要移除的token（累积概率超过top_p的部分）
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 边界处理：保留第一个超过阈值的token，确保至少有一个候选
        # 这是一个重要的技巧，避免在极端情况下候选集为空
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 创建原始索引的mask
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        # 应用mask：将被移除的token的logits设为负无穷
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        # 在过滤后的分布中重新计算概率并采样
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        # 性能统计
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TypicalSampler(BaseSampler):
    """Typical采样器
    
    设计思想：
    - 信息论基础：基于token的"典型性"进行采样
    - 质量优先：选择信息量适中的token，避免过于平凡或异常的选择
    - 自适应阈值：根据分布的熵动态调整采样范围
    - 平衡策略：在信息量和概率之间找到最佳平衡点
    """
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        typical_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Typical采样实现
        
        设计思想：
        - 信息论原理：选择信息量接近期望值的token
        - 质量保证：避免选择过于平凡或异常的token
        - 动态阈值：根据分布特性自适应调整
        
        典型采样的核心思想：
        1. 计算每个token的信息量：-log(p_i)
        2. 计算期望信息量：-sum(p_i * log(p_i))
        3. 选择信息量接近期望值的token集合
        4. 在该集合中进行概率采样
        
        Args:
            logits: [batch_size, vocab_size] 模型输出的logits
            past_tokens: 历史token序列（当前未使用）
            typical_p: 典型性阈值，控制候选集大小
            temperature: 温度参数
            
        Returns:
            next_tokens: [batch_size] 采样得到的下一个token
        """
        start_time = time.time()
        
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算每个token的信息量：-log(p_i)
        # 添加小常数避免log(0)
        epsilon = 1e-10
        information = -torch.log(probs + epsilon)
        
        # 计算期望信息量（熵）：H = -sum(p_i * log(p_i))
        entropy = torch.sum(probs * information, dim=-1, keepdim=True)
        
        # 计算每个token与期望信息量的差异
        # 典型性度量：|信息量 - 期望信息量|
        typicality = torch.abs(information - entropy)
        
        # 按典型性排序（差异越小越典型）
        sorted_typicality, sorted_indices = torch.sort(typicality, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        
        # 计算累积概率，找到典型性阈值
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过typical_p的位置
        cutoff_index = torch.searchsorted(
            cumulative_probs, 
            typical_p * torch.ones_like(cumulative_probs[..., :1])
        )
        
        # 确保至少保留一个token
        cutoff_index = torch.clamp(cutoff_index, min=1)
        
        # 创建mask，保留典型的token
        batch_size, vocab_size = logits.shape
        indices_to_remove = torch.arange(vocab_size, device=logits.device).expand(batch_size, -1)
        indices_to_remove = indices_to_remove >= cutoff_index
        
        # 将非典型token的概率设为0
        filtered_probs = probs.clone()
        filtered_probs[torch.gather(indices_to_remove, -1, sorted_indices)] = 0
        
        # 重新归一化概率分布
        filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + epsilon)
        
        # 在过滤后的分布中采样
        next_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        # 性能统计
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class BeamSearchSampler(BaseSampler):
    """束搜索采样器
    
    设计思想：
    - 全局优化：维护多个候选序列，寻找全局最优解
    - 分支管理：在每步保持固定数量的最优候选
    - 长度归一化：避免短序列的偏好
    - 多样性控制：通过分组束搜索增加输出多样性
    """
    
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.beam_size = getattr(config, 'beam_size', 4)
        self.length_penalty = getattr(config, 'length_penalty', 1.0)
        self.early_stopping = getattr(config, 'early_stopping', True)
        self.num_beam_groups = getattr(config, 'num_beam_groups', 1)
        self.diversity_penalty = getattr(config, 'diversity_penalty', 0.0)
        
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        beam_state: Optional[BeamSearchState] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, BeamSearchState]:
        """束搜索采样实现
        
        设计思想：
        - 候选扩展：为每个候选序列生成所有可能的下一个token
        - 得分计算：结合概率和长度惩罚计算序列得分
        - 候选选择：选择得分最高的beam_size个候选
        - 状态更新：更新束搜索状态用于下一步
        
        Args:
            logits: [batch_size * beam_size, vocab_size] 模型输出
            past_tokens: 历史token序列
            beam_state: 当前束搜索状态
            
        Returns:
            next_tokens: [batch_size * beam_size] 下一个token
            new_beam_state: 更新后的束搜索状态
        """
        start_time = time.time()
        
        batch_size = logits.size(0) // self.beam_size
        vocab_size = logits.size(-1)
        
        # 计算对数概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        if beam_state is None:
            # 初始化束搜索状态
            beam_state = BeamSearchState(
                sequences=torch.zeros(batch_size, self.beam_size, 1, dtype=torch.long, device=logits.device),
                scores=torch.zeros(batch_size, self.beam_size, device=logits.device),
                finished=torch.zeros(batch_size, self.beam_size, dtype=torch.bool, device=logits.device)
            )
            
            # 第一步：只从第一个beam选择top-k
            first_beam_log_probs = log_probs[::self.beam_size]  # [batch_size, vocab_size]
            top_k_scores, top_k_indices = torch.topk(first_beam_log_probs, self.beam_size, dim=-1)
            
            # 更新状态
            beam_state.sequences = top_k_indices.unsqueeze(-1)  # [batch_size, beam_size, 1]
            beam_state.scores = top_k_scores
            
            next_tokens = top_k_indices.flatten()  # [batch_size * beam_size]
        else:
            # 后续步骤：扩展所有候选
            # 重塑logits为 [batch_size, beam_size, vocab_size]
            log_probs = log_probs.view(batch_size, self.beam_size, vocab_size)
            
            # 计算新的候选得分
            # [batch_size, beam_size, 1] + [batch_size, beam_size, vocab_size]
            candidate_scores = beam_state.scores.unsqueeze(-1) + log_probs
            
            # 应用长度惩罚
            current_length = beam_state.sequences.size(-1)
            length_penalty = ((5 + current_length) / 6) ** self.length_penalty
            candidate_scores = candidate_scores / length_penalty
            
            # 将候选得分展平并选择top-k
            candidate_scores_flat = candidate_scores.view(batch_size, -1)  # [batch_size, beam_size * vocab_size]
            top_k_scores, top_k_indices = torch.topk(candidate_scores_flat, self.beam_size, dim=-1)
            
            # 计算beam索引和token索引
            beam_indices = top_k_indices // vocab_size  # 来自哪个beam
            token_indices = top_k_indices % vocab_size   # 选择的token
            
            # 更新序列
            # 选择对应的历史序列
            batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
            selected_sequences = beam_state.sequences[batch_indices, beam_indices]  # [batch_size, beam_size, seq_len]
            
            # 添加新token
            new_sequences = torch.cat([
                selected_sequences,
                token_indices.unsqueeze(-1)
            ], dim=-1)
            
            # 更新状态
            beam_state.sequences = new_sequences
            beam_state.scores = top_k_scores
            
            next_tokens = token_indices.flatten()  # [batch_size * beam_size]
        
        # 性能统计
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens, beam_state

class SamplerManager:
    """采样器管理器
    
    设计思想：
    - 策略工厂：根据配置创建相应的采样器
    - 统一接口：为不同采样策略提供统一的调用接口
    - 动态切换：支持运行时切换采样策略
    - 性能监控：统一收集各采样器的性能指标
    """
    
    def __init__(self, config: GenerationConfig):
        """初始化采样器管理器
        
        设计思想：
        - 延迟初始化：采样器在需要时才创建
        - 配置驱动：通过配置决定使用的采样策略
        - 缓存机制：避免重复创建相同的采样器
        """
        self.config = config
        self.samplers: Dict[SamplingStrategy, BaseSampler] = {}
        self.current_strategy = SamplingStrategy(config.sampling_strategy)
        
    def get_sampler(self, strategy: SamplingStrategy) -> BaseSampler:
        """获取指定策略的采样器
        
        设计思想：
        - 单例模式：每种策略只创建一个采样器实例
        - 工厂模式：根据策略类型创建相应的采样器
        - 错误处理：对不支持的策略抛出异常
        """
        if strategy not in self.samplers:
            self.samplers[strategy] = self._create_sampler(strategy)
        return self.samplers[strategy]
    
    def _create_sampler(self, strategy: SamplingStrategy) -> BaseSampler:
        """创建采样器实例
        
        设计思想：
        - 策略映射：将枚举值映射到具体的采样器类
        - 配置传递：将配置对象传递给采样器构造函数
        - 扩展性：便于添加新的采样策略
        """
        sampler_map = {
            SamplingStrategy.GREEDY: GreedySampler,
            SamplingStrategy.MULTINOMIAL: MultinomialSampler,
            SamplingStrategy.TOP_K: TopKSampler,
            SamplingStrategy.TOP_P: TopPSampler,
            SamplingStrategy.NUCLEUS: TopPSampler,  # Nucleus是Top-P的别名
            SamplingStrategy.TYPICAL: TypicalSampler,
            SamplingStrategy.BEAM_SEARCH: BeamSearchSampler,
        }
        
        if strategy not in sampler_map:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
        
        return sampler_map[strategy](self.config)
    
    def sample(
        self,
        logits: torch.Tensor,
        strategy: Optional[SamplingStrategy] = None,
        **kwargs
    ) -> torch.Tensor:
        """执行采样
        
        设计思想：
        - 策略选择：使用指定策略或默认策略
        - 参数传递：将额外参数传递给具体的采样器
        - 统一接口：为所有采样策略提供统一的调用方式
        """
        if strategy is None:
            strategy = self.current_strategy
        
        sampler = self.get_sampler(strategy)
        return sampler.sample(logits, **kwargs)
    
    def get_stats(self) -> Dict[str, GenerationStats]:
        """获取所有采样器的统计信息
        
        设计思想：
        - 全面监控：收集所有已创建采样器的统计信息
        - 性能分析：为性能优化提供数据支持
        - 调试支持：帮助分析不同策略的性能表现
        """
        return {
            strategy.value: sampler.stats 
            for strategy, sampler in self.samplers.items()
        }

class StoppingCriteriaChecker:
    """停止条件检查器
    
    设计思想：
    - 多条件支持：支持多种停止条件的组合
    - 高效检查：优化检查算法减少计算开销
    - 可扩展性：便于添加新的停止条件
    - 状态管理：维护检查过程中的状态信息
    """
    
    def __init__(self, config: GenerationConfig, tokenizer):
        """初始化停止条件检查器
        
        设计思想：
        - 配置解析：从配置中提取停止条件参数
        - 预处理：预先处理停止词等条件以提高检查效率
        - 分词器集成：利用分词器进行文本处理
        """
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = getattr(config, 'max_length', 2048)
        self.stop_words = getattr(config, 'stop_words', [])
        self.eos_token_id = getattr(config, 'eos_token_id', None)
        
        # 预处理停止词：转换为token序列
        self.stop_token_sequences = []
        for stop_word in self.stop_words:
            tokens = self.tokenizer.encode(stop_word, add_special_tokens=False)
            self.stop_token_sequences.append(tokens)
    
    def should_stop(
        self,
        sequences: torch.Tensor,
        current_length: int,
        **kwargs
    ) -> torch.Tensor:
        """检查是否应该停止生成
        
        设计思想：
        - 多条件检查：依次检查各种停止条件
        - 批量处理：支持批量序列的并行检查
        - 早期退出：一旦满足条件立即返回
        - 优先级：按重要性顺序检查条件
        
        Args:
            sequences: [batch_size, seq_len] 当前生成的序列
            current_length: 当前序列长度
            
        Returns:
            should_stop: [batch_size] 每个序列是否应该停止
        """
        batch_size = sequences.size(0)
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=sequences.device)
        
        # 1. 检查最大长度
        if current_length >= self.max_length:
            should_stop.fill_(True)
            return should_stop
        
        # 2. 检查EOS token
        if self.eos_token_id is not None:
            last_tokens = sequences[:, -1]
            eos_mask = (last_tokens == self.eos_token_id)
            should_stop = should_stop | eos_mask
        
        # 3. 检查停止词
        if self.stop_token_sequences:
            for i in range(batch_size):
                if should_stop[i]:
                    continue
                
                sequence = sequences[i].tolist()
                for stop_tokens in self.stop_token_sequences:
                    if self._sequence_ends_with(sequence, stop_tokens):
                        should_stop[i] = True
                        break
        
        return should_stop
    
    def _sequence_ends_with(self, sequence: List[int], pattern: List[int]) -> bool:
        """检查序列是否以指定模式结尾
        
        设计思想：
        - 高效匹配：使用后缀匹配算法
        - 边界处理：正确处理序列长度不足的情况
        - 精确匹配：确保完全匹配停止词序列
        """
        if len(pattern) > len(sequence):
            return False
        
        return sequence[-len(pattern):] == pattern

class TextGenerator:
    """文本生成器
    
    设计思想：
    - 端到端生成：整合模型推理、采样、停止检查等全流程
    - 流式支持：支持流式生成和批量生成
    - 状态管理：维护生成过程中的各种状态
    - 性能优化：通过缓存和并行化提升生成效率
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: GenerationConfig,
        device: torch.device = None
    ):
        """初始化文本生成器
        
        设计思想：
        - 组件集成：整合模型、分词器、采样器等组件
        - 设备管理：统一管理计算设备
        - 配置中心化：通过配置对象管理所有参数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.sampler_manager = SamplerManager(config)
        self.stopping_checker = StoppingCriteriaChecker(config, tokenizer)
        
        # 统计信息
        self.stats = GenerationStats()
        
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_strategy: Optional[SamplingStrategy] = None,
        **kwargs
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """生成文本
        
        设计思想：
        - 输入标准化：统一处理单个和批量输入
        - 策略选择：支持指定采样策略或使用默认策略
        - 结果封装：将生成结果封装为标准格式
        - 异常处理：妥善处理生成过程中的异常
        
        Args:
            prompts: 输入提示文本（单个或批量）
            sampling_strategy: 采样策略
            **kwargs: 额外的生成参数
            
        Returns:
            生成结果（单个或批量）
        """
        start_time = time.time()
        
        # 输入标准化
        if isinstance(prompts, str):
            prompts = [prompts]
            single_input = True
        else:
            single_input = False
        
        batch_size = len(prompts)
        
        try:
            # 编码输入
            input_ids = []
            for prompt in prompts:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                input_ids.append(torch.tensor(tokens, device=self.device))
            
            # 填充到相同长度
            max_input_length = max(len(ids) for ids in input_ids)
            padded_input_ids = torch.zeros(
                batch_size, max_input_length, 
                dtype=torch.long, device=self.device
            )
            
            for i, ids in enumerate(input_ids):
                padded_input_ids[i, :len(ids)] = ids
            
            # 生成序列
            generated_sequences = self._generate_sequences(
                padded_input_ids, 
                sampling_strategy,
                **kwargs
            )
            
            # 解码结果
            results = []
            for i, sequence in enumerate(generated_sequences):
                # 移除输入部分，只保留生成部分
                input_length = len(input_ids[i])
                generated_tokens = sequence[input_length:]
                
                # 解码文本
                generated_text = self.tokenizer.decode(
                    generated_tokens.tolist(), 
                    skip_special_tokens=True
                )
                
                # 创建结果对象
                result = GenerationResult(
                    sequences=[sequence],
                    generation_time=time.time() - start_time,
                    num_tokens=len(generated_tokens),
                    finished=True
                )
                results.append(result)
            
            # 更新统计信息
            total_tokens = sum(result.num_tokens for result in results)
            self.stats.total_tokens_generated += total_tokens
            self.stats.total_sequences_generated += len(results)
            self.stats.generation_time += time.time() - start_time
            
            # 返回结果
            return results[0] if single_input else results
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_sequences(
        self,
        input_ids: torch.Tensor,
        sampling_strategy: Optional[SamplingStrategy] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """生成序列的核心逻辑
        
        设计思想：
        - 自回归生成：逐步生成每个token
        - 状态维护：维护生成过程中的各种状态
        - 并行处理：支持批量序列的并行生成
        - 动态停止：根据停止条件动态结束生成
        """
        batch_size, seq_len = input_ids.shape
        max_new_tokens = kwargs.get('max_new_tokens', self.config.max_length - seq_len)
        
        # 初始化生成状态
        current_sequences = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 束搜索状态（如果使用束搜索）
        beam_state = None
        if sampling_strategy == SamplingStrategy.BEAM_SEARCH:
            beam_state = None  # 将在第一次采样时初始化
        
        # 逐步生成
        for step in range(max_new_tokens):
            if finished.all():
                break
            
            # 模型前向传播
            with torch.no_grad():
                outputs = self.model(current_sequences)
                logits = outputs.logits[:, -1, :]  # 取最后一个位置的logits
            
            # 采样下一个token
            if sampling_strategy == SamplingStrategy.BEAM_SEARCH:
                next_tokens, beam_state = self.sampler_manager.sample(
                    logits, 
                    strategy=sampling_strategy,
                    beam_state=beam_state,
                    **kwargs
                )
                # 束搜索的序列更新在采样器内部完成
                current_sequences = beam_state.sequences.view(-1, beam_state.sequences.size(-1))
            else:
                next_tokens = self.sampler_manager.sample(
                    logits, 
                    strategy=sampling_strategy,
                    **kwargs
                )
                
                # 添加新token到序列
                current_sequences = torch.cat([
                    current_sequences,
                    next_tokens.unsqueeze(-1)
                ], dim=-1)
            
            # 检查停止条件
            should_stop = self.stopping_checker.should_stop(
                current_sequences,
                current_sequences.size(-1)
            )
            
            finished = finished | should_stop
        
        # 返回生成的序列
        return [current_sequences[i] for i in range(batch_size)]
    
    def generate_stream(
        self,
        prompt: str,
        sampling_strategy: Optional[SamplingStrategy] = None,
        **kwargs
    ) -> Iterator[str]:
        """流式生成文本
        
        设计思想：
        - 实时输出：每生成一个token就返回对应的文本
        - 增量解码：只解码新生成的部分
        - 状态保持：维护生成过程中的状态
        - 异常处理：优雅处理生成过程中的异常
        
        Args:
            prompt: 输入提示文本
            sampling_strategy: 采样策略
            **kwargs: 额外的生成参数
            
        Yields:
            每次生成的增量文本
        """
        # 编码输入
        input_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_tokens], device=self.device)
        
        current_sequence = input_ids.clone()
        input_length = len(input_tokens)
        max_new_tokens = kwargs.get('max_new_tokens', self.config.max_length - input_length)
        
        try:
            for step in range(max_new_tokens):
                # 模型前向传播
                with torch.no_grad():
                    outputs = self.model(current_sequence)
                    logits = outputs.logits[:, -1, :]
                
                # 采样下一个token
                next_token = self.sampler_manager.sample(
                    logits,
                    strategy=sampling_strategy,
                    **kwargs
                )
                
                # 添加到序列
                current_sequence = torch.cat([
                    current_sequence,
                    next_token.unsqueeze(-1)
                ], dim=-1)
                
                # 解码新token
                new_token_text = self.tokenizer.decode(
                    [next_token.item()],
                    skip_special_tokens=True
                )
                
                yield new_token_text
                
                # 检查停止条件
                should_stop = self.stopping_checker.should_stop(
                    current_sequence,
                    current_sequence.size(-1)
                )
                
                if should_stop.item():
                    break
                    
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取生成统计信息
        
        设计思想：
        - 全面统计：包含生成器和采样器的统计信息
        - 性能指标：计算关键性能指标
        - 调试支持：提供详细的调试信息
        """
        sampler_stats = self.sampler_manager.get_stats()
        
        # 计算平均指标
        if self.stats.total_sequences_generated > 0:
            self.stats.average_tokens_per_sequence = (
                self.stats.total_tokens_generated / self.stats.total_sequences_generated
            )
        
        return {
            'generator_stats': self.stats,
            'sampler_stats': sampler_stats,
            'performance_metrics': {
                'tokens_per_second': (
                    self.stats.total_tokens_generated / self.stats.generation_time
                    if self.stats.generation_time > 0 else 0
                ),
                'sequences_per_second': (
                    self.stats.total_sequences_generated / self.stats.generation_time
                    if self.stats.generation_time > 0 else 0
                )
            }
        }

# 工厂函数和实用工具

def create_text_generator(
    model,
    tokenizer,
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None
) -> TextGenerator:
    """创建文本生成器的工厂函数
    
    设计思想：
    - 简化创建：提供便捷的创建接口
    - 默认配置：为常见场景提供合理的默认配置
    - 设备自动检测：自动选择最佳的计算设备
    """
    if config is None:
        config = GenerationConfig()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return TextGenerator(model, tokenizer, config, device)

def example_generation_usage():
    """生成工具使用示例
    
    设计思想：
    - 完整示例：展示从初始化到生成的完整流程
    - 多种策略：演示不同采样策略的使用
    - 最佳实践：展示推荐的使用模式
    """
    # 这里是示例代码，实际使用时需要替换为真实的模型和分词器
    print("=== 生成工具使用示例 ===")
    
    # 1. 创建配置
    config = GenerationConfig(
        max_length=100,
        sampling_strategy="top_p",
        temperature=0.8,
        top_p=0.9,
        stop_words=["什么", "\n\n"]
    )
    
    # 2. 创建生成器（需要真实的模型和分词器）
    # generator = create_text_generator(model, tokenizer, config)
    
    # 3. 单次生成
    # result = generator.generate("请解释什么是人工智能")
    # print(f"生成结果: {result}")
    
    # 4. 批量生成
    # prompts = ["什么是机器学习？", "深度学习的原理是什么？"]
    # results = generator.generate(prompts)
    # for i, result in enumerate(results):
    #     print(f"结果 {i+1}: {result}")
    
    # 5. 流式生成
    # for token in generator.generate_stream("请写一首关于春天的诗"):
    #     print(token, end='', flush=True)
    
    # 6. 获取统计信息
    # stats = generator.get_stats()
    # print(f"生成统计: {stats}")
    
    print("示例代码展示了生成工具的主要功能和使用方法")

# 关键特性分析

"""
## 🎯 关键特性分析

### 1. 采样策略多样性
- **贪心采样**：确定性输出，适合事实性任务
- **随机采样**：增加输出多样性，适合创意任务
- **Top-K/Top-P**：平衡质量和多样性
- **典型采样**：基于信息论的高质量采样
- **束搜索**：全局优化，适合翻译等任务

### 2. 性能优化策略
- **批量处理**：支持多序列并行生成
- **缓存机制**：减少重复计算
- **早期停止**：避免不必要的计算
- **内存管理**：优化内存使用效率

### 3. 质量控制机制
- **多种停止条件**：灵活控制生成长度
- **温度调节**：精确控制随机性
- **长度惩罚**：避免长度偏好
- **重复惩罚**：减少重复内容

### 4. 可扩展性设计
- **插件化架构**：易于添加新的采样策略
- **配置驱动**：通过配置文件控制行为
- **统一接口**：所有采样器共享相同接口
- **模块化设计**：各组件独立可测试

## 🏗️ 设计模式应用

### 1. 策略模式 (Strategy Pattern)
- **BaseSampler抽象类**：定义采样策略接口
- **具体采样器**：实现不同的采样算法
- **SamplerManager**：管理和切换策略

### 2. 工厂模式 (Factory Pattern)
- **SamplerManager._create_sampler()**：根据策略创建采样器
- **create_text_generator()**：创建文本生成器

### 3. 单例模式 (Singleton Pattern)
- **采样器缓存**：每种策略只创建一个实例
- **资源复用**：避免重复创建相同对象

### 4. 观察者模式 (Observer Pattern)
- **统计信息收集**：各组件更新统计信息
- **性能监控**：实时跟踪生成性能

## 🚀 最佳实践建议

### 1. 策略选择指南
- **事实性任务**：使用贪心采样或低温度采样
- **创意任务**：使用Top-P或典型采样
- **翻译任务**：使用束搜索
- **对话系统**：使用Top-P + 温度调节

### 2. 性能优化建议
- **批量处理**：尽可能使用批量生成
- **合理设置停止条件**：避免过长生成
- **监控内存使用**：及时释放不需要的资源
- **选择合适的设备**：GPU加速计算密集型操作

### 3. 质量保证措施
- **多次采样**：对重要任务进行多次生成
- **后处理过滤**：对生成结果进行质量检查
- **A/B测试**：比较不同策略的效果
- **用户反馈**：收集用户对生成质量的反馈

### 4. 调试和监控
- **详细日志**：记录生成过程的关键信息
- **性能指标**：监控生成速度和质量
- **异常处理**：妥善处理生成过程中的异常
- **资源监控**：跟踪内存和GPU使用情况
"""

if __name__ == "__main__":
    example_generation_usage()