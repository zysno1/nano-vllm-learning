# 生成工具分析

## 🎯 生成工具概览

生成工具是 nano-vllm 中负责文本生成逻辑的核心组件。它包含各种采样策略、生成控制机制、以及输出后处理功能。高效的生成工具实现直接影响生成质量和推理性能。

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
    """采样策略枚举"""
    GREEDY = "greedy"
    MULTINOMIAL = "multinomial"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"
    BEAM_SEARCH = "beam_search"
    NUCLEUS = "nucleus"
    TYPICAL = "typical"

class StoppingCriteria(Enum):
    """停止条件枚举"""
    MAX_LENGTH = "max_length"
    EOS_TOKEN = "eos_token"
    STOP_WORDS = "stop_words"
    CUSTOM = "custom"

@dataclass
class GenerationStats:
    """生成统计信息"""
    total_tokens_generated: int = 0
    total_sequences_generated: int = 0
    generation_time: float = 0.0
    average_tokens_per_sequence: float = 0.0
    sampling_time: float = 0.0
    postprocessing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class GenerationResult:
    """生成结果"""
    sequences: List[torch.Tensor]
    scores: Optional[List[float]] = None
    logprobs: Optional[List[torch.Tensor]] = None
    attention_weights: Optional[List[torch.Tensor]] = None
    generation_time: float = 0.0
    num_tokens: int = 0
    finished: bool = False
    stop_reason: Optional[str] = None

@dataclass
class BeamSearchState:
    """束搜索状态"""
    sequences: torch.Tensor  # [beam_size, seq_len]
    scores: torch.Tensor     # [beam_size]
    finished: torch.Tensor   # [beam_size]
    
class BaseSampler(ABC):
    """基础采样器抽象类"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.stats = GenerationStats()
        
    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """采样下一个token"""
        pass
    
    def update_stats(self, sampling_time: float, num_samples: int = 1):
        """更新统计信息"""
        self.stats.sampling_time += sampling_time
        self.stats.total_tokens_generated += num_samples

class GreedySampler(BaseSampler):
    """贪心采样器"""
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """贪心采样"""
        
        start_time = time.time()
        
        # 选择概率最大的token
        next_tokens = torch.argmax(logits, dim=-1)
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class MultinomialSampler(BaseSampler):
    """多项式采样器"""
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """多项式采样"""
        
        start_time = time.time()
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 多项式采样
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TopKSampler(BaseSampler):
    """Top-K采样器"""
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        top_k: int = 50,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-K采样"""
        
        start_time = time.time()
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 获取top-k值和索引
        top_k_values, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
        
        # 计算top-k概率
        top_k_probs = F.softmax(top_k_values, dim=-1)
        
        # 从top-k中采样
        sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
        
        # 获取实际的token索引
        next_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TopPSampler(BaseSampler):
    """Top-P (Nucleus) 采样器"""
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        top_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Top-P采样"""
        
        start_time = time.time()
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算概率并排序
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过top_p的位置
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # 保留第一个超过阈值的token（确保至少有一个token）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # 创建mask
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        # 应用mask
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        # 重新计算概率并采样
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class TypicalSampler(BaseSampler):
    """Typical采样器"""
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        typical_p: float = 0.9,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Typical采样"""
        
        start_time = time.time()
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        # 计算信息量 -log(p)
        log_probs = F.log_softmax(logits, dim=-1)
        neg_log_probs = -log_probs
        
        # 计算熵
        entropy = torch.sum(probs * neg_log_probs, dim=-1, keepdim=True)
        
        # 计算与熵的差异
        deviation = torch.abs(neg_log_probs - entropy)
        
        # 排序并找到typical tokens
        sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
        sorted_probs = torch.gather(probs, -1, sorted_indices)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过typical_p的位置
        cutoff = cumulative_probs > typical_p
        cutoff[..., 0] = False  # 确保至少保留一个token
        
        # 创建mask
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, cutoff)
        
        # 应用mask
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        # 重新计算概率并采样
        filtered_probs = F.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens

class BeamSearchSampler(BaseSampler):
    """束搜索采样器"""
    
    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.beam_size = config.num_beams
        self.length_penalty = config.length_penalty
        self.early_stopping = config.early_stopping
    
    def sample(
        self,
        logits: torch.Tensor,
        past_tokens: Optional[torch.Tensor] = None,
        beam_state: Optional[BeamSearchState] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, BeamSearchState]:
        """束搜索采样"""
        
        start_time = time.time()
        
        batch_size = logits.size(0) // self.beam_size
        vocab_size = logits.size(-1)
        
        # 计算log概率
        log_probs = F.log_softmax(logits, dim=-1)
        
        if beam_state is None:
            # 初始化束搜索状态
            beam_state = self._initialize_beam_state(batch_size, logits.device)
        
        # 计算候选分数
        candidate_scores = beam_state.scores.unsqueeze(-1) + log_probs
        
        # 应用长度惩罚
        if self.length_penalty != 1.0:
            current_length = beam_state.sequences.size(-1)
            length_penalty = ((5 + current_length) / 6) ** self.length_penalty
            candidate_scores = candidate_scores / length_penalty
        
        # 重塑为 [batch_size, beam_size * vocab_size]
        candidate_scores = candidate_scores.view(batch_size, -1)
        
        # 选择top-k候选
        top_scores, top_indices = torch.topk(
            candidate_scores, 
            k=self.beam_size, 
            dim=-1
        )
        
        # 计算beam索引和token索引
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size
        
        # 更新序列
        new_sequences = []
        new_scores = []
        new_finished = []
        
        for batch_idx in range(batch_size):
            batch_sequences = []
            batch_scores = []
            batch_finished = []
            
            for beam_idx in range(self.beam_size):
                # 获取父beam和新token
                parent_beam = beam_indices[batch_idx, beam_idx]
                new_token = token_indices[batch_idx, beam_idx]
                
                # 构建新序列
                parent_sequence = beam_state.sequences[batch_idx * self.beam_size + parent_beam]
                new_sequence = torch.cat([parent_sequence, new_token.unsqueeze(0)])
                
                batch_sequences.append(new_sequence)
                batch_scores.append(top_scores[batch_idx, beam_idx])
                
                # 检查是否结束
                is_finished = (new_token == self.config.eos_token_id) or \
                             (len(new_sequence) >= self.config.max_length)
                batch_finished.append(is_finished)
            
            new_sequences.extend(batch_sequences)
            new_scores.extend(batch_scores)
            new_finished.extend(batch_finished)
        
        # 更新beam状态
        new_beam_state = BeamSearchState(
            sequences=torch.stack(new_sequences),
            scores=torch.tensor(new_scores, device=logits.device),
            finished=torch.tensor(new_finished, device=logits.device)
        )
        
        # 选择下一个token（用于继续生成）
        next_tokens = torch.stack([seq[-1] for seq in new_sequences])
        
        sampling_time = time.time() - start_time
        self.update_stats(sampling_time, next_tokens.numel())
        
        return next_tokens, new_beam_state
    
    def _initialize_beam_state(self, batch_size: int, device: torch.device) -> BeamSearchState:
        """初始化束搜索状态"""
        
        # 初始序列（空序列或BOS token）
        if self.config.bos_token_id is not None:
            initial_sequences = torch.full(
                (batch_size * self.beam_size, 1),
                self.config.bos_token_id,
                device=device
            )
        else:
            initial_sequences = torch.empty(
                (batch_size * self.beam_size, 0),
                dtype=torch.long,
                device=device
            )
        
        # 初始分数
        initial_scores = torch.zeros(batch_size * self.beam_size, device=device)
        
        # 只有第一个beam有非零分数
        for i in range(batch_size):
            initial_scores[i * self.beam_size + 1:(i + 1) * self.beam_size] = float('-inf')
        
        # 初始完成状态
        initial_finished = torch.zeros(
            batch_size * self.beam_size,
            dtype=torch.bool,
            device=device
        )
        
        return BeamSearchState(
            sequences=initial_sequences,
            scores=initial_scores,
            finished=initial_finished
        )

class SamplerManager:
    """采样器管理器"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        # 注册采样器
        self.samplers = {
            SamplingStrategy.GREEDY: GreedySampler(config),
            SamplingStrategy.MULTINOMIAL: MultinomialSampler(config),
            SamplingStrategy.TOP_K: TopKSampler(config),
            SamplingStrategy.TOP_P: TopPSampler(config),
            SamplingStrategy.TYPICAL: TypicalSampler(config),
            SamplingStrategy.BEAM_SEARCH: BeamSearchSampler(config),
        }
        
        logger.info("Initialized SamplerManager")
    
    def get_sampler(self, strategy: SamplingStrategy) -> BaseSampler:
        """获取采样器"""
        
        if strategy not in self.samplers:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
        
        return self.samplers[strategy]
    
    def sample(
        self,
        logits: torch.Tensor,
        strategy: SamplingStrategy,
        **kwargs
    ) -> torch.Tensor:
        """执行采样"""
        
        sampler = self.get_sampler(strategy)
        return sampler.sample(logits, **kwargs)

class StoppingCriteriaChecker:
    """停止条件检查器"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        # 停止词列表
        self.stop_words = config.stop_words or []
        self.stop_word_ids = self._convert_stop_words_to_ids()
        
        logger.info(f"Initialized StoppingCriteriaChecker with {len(self.stop_words)} stop words")
    
    def _convert_stop_words_to_ids(self) -> List[List[int]]:
        """将停止词转换为token ID"""
        
        # 这里需要tokenizer，实际实现中应该从外部传入
        # 暂时返回空列表
        return []
    
    def should_stop(
        self,
        sequences: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """检查是否应该停止生成"""
        
        batch_size, seq_len = sequences.shape
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=sequences.device)
        
        for i in range(batch_size):
            sequence = sequences[i]
            
            # 检查最大长度
            if seq_len >= self.config.max_length:
                should_stop[i] = True
                continue
            
            # 检查EOS token
            if self.config.eos_token_id is not None:
                if sequence[-1] == self.config.eos_token_id:
                    should_stop[i] = True
                    continue
            
            # 检查停止词
            if self._contains_stop_word(sequence):
                should_stop[i] = True
                continue
        
        return should_stop
    
    def _contains_stop_word(self, sequence: torch.Tensor) -> bool:
        """检查序列是否包含停止词"""
        
        sequence_list = sequence.tolist()
        
        for stop_word_ids in self.stop_word_ids:
            if len(stop_word_ids) <= len(sequence_list):
                # 检查序列末尾是否匹配停止词
                if sequence_list[-len(stop_word_ids):] == stop_word_ids:
                    return True
        
        return False

class TextGenerator:
    """文本生成器"""
    
    def __init__(
        self,
        config: GenerationConfig,
        sampler_manager: SamplerManager,
        stopping_checker: StoppingCriteriaChecker,
    ):
        self.config = config
        self.sampler_manager = sampler_manager
        self.stopping_checker = stopping_checker
        
        # 统计信息
        self.stats = GenerationStats()
        
        # 缓存
        self.generation_cache = {}
        self.cache_lock = threading.RLock()
        
        logger.info("Initialized TextGenerator")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        model_forward_fn: Optional[Callable] = None,
        **kwargs
    ) -> GenerationResult:
        """生成文本"""
        
        start_time = time.time()
        
        # 合并配置参数
        generation_config = self._merge_generation_config(kwargs)
        
        # 选择生成策略
        if generation_config.num_beams > 1:
            result = self._beam_search_generate(
                input_ids, attention_mask, model_forward_fn, generation_config
            )
        else:
            result = self._sampling_generate(
                input_ids, attention_mask, model_forward_fn, generation_config
            )
        
        generation_time = time.time() - start_time
        result.generation_time = generation_time
        
        # 更新统计信息
        self._update_stats(result)
        
        return result
    
    def _merge_generation_config(self, kwargs: Dict[str, Any]) -> GenerationConfig:
        """合并生成配置"""
        
        # 创建配置副本
        config_dict = self.config.__dict__.copy()
        
        # 更新参数
        config_dict.update(kwargs)
        
        return GenerationConfig(**config_dict)
    
    def _sampling_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        model_forward_fn: Callable,
        config: GenerationConfig,
    ) -> GenerationResult:
        """采样生成"""
        
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # 初始化生成序列
        generated_sequences = input_ids.clone()
        
        # 初始化attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        current_attention_mask = attention_mask.clone()
        
        # 生成循环
        for step in range(config.max_new_tokens):
            # 前向传播
            outputs = model_forward_fn(
                input_ids=generated_sequences,
                attention_mask=current_attention_mask,
            )
            
            # 获取logits
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 应用生成约束
            logits = self._apply_generation_constraints(logits, generated_sequences, config)
            
            # 采样下一个token
            next_tokens = self.sampler_manager.sample(
                logits,
                strategy=SamplingStrategy(config.sampling_strategy),
                past_tokens=generated_sequences,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                typical_p=getattr(config, 'typical_p', 0.9),
            )
            
            # 更新序列
            generated_sequences = torch.cat([
                generated_sequences,
                next_tokens.unsqueeze(-1)
            ], dim=-1)
            
            # 更新attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, 1, device=device)
            ], dim=-1)
            
            # 检查停止条件
            should_stop = self.stopping_checker.should_stop(generated_sequences)
            
            if should_stop.all():
                break
        
        # 提取生成的部分
        generated_tokens = generated_sequences[:, input_length:]
        
        return GenerationResult(
            sequences=[generated_tokens[i] for i in range(batch_size)],
            num_tokens=generated_tokens.numel(),
            finished=True,
            stop_reason="max_length" if step == config.max_new_tokens - 1 else "eos_token"
        )
    
    def _beam_search_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        model_forward_fn: Callable,
        config: GenerationConfig,
    ) -> GenerationResult:
        """束搜索生成"""
        
        batch_size, input_length = input_ids.shape
        beam_size = config.num_beams
        device = input_ids.device
        
        # 扩展输入以适应beam search
        expanded_input_ids = input_ids.unsqueeze(1).expand(
            batch_size, beam_size, input_length
        ).reshape(batch_size * beam_size, input_length)
        
        if attention_mask is not None:
            expanded_attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, beam_size, input_length
            ).reshape(batch_size * beam_size, input_length)
        else:
            expanded_attention_mask = torch.ones_like(expanded_input_ids)
        
        # 初始化beam search状态
        beam_sampler = self.sampler_manager.get_sampler(SamplingStrategy.BEAM_SEARCH)
        beam_state = None
        
        generated_sequences = expanded_input_ids.clone()
        current_attention_mask = expanded_attention_mask.clone()
        
        # 生成循环
        for step in range(config.max_new_tokens):
            # 前向传播
            outputs = model_forward_fn(
                input_ids=generated_sequences,
                attention_mask=current_attention_mask,
            )
            
            # 获取logits
            logits = outputs.logits[:, -1, :]  # [batch_size * beam_size, vocab_size]
            
            # 应用生成约束
            logits = self._apply_generation_constraints(logits, generated_sequences, config)
            
            # 束搜索采样
            next_tokens, beam_state = beam_sampler.sample(
                logits,
                beam_state=beam_state,
            )
            
            # 更新序列
            generated_sequences = beam_state.sequences
            
            # 更新attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size * beam_size, 1, device=device)
            ], dim=-1)
            
            # 检查是否所有beam都完成
            if beam_state.finished.all():
                break
        
        # 选择最佳序列
        best_sequences = []
        best_scores = []
        
        for batch_idx in range(batch_size):
            batch_start = batch_idx * beam_size
            batch_end = (batch_idx + 1) * beam_size
            
            batch_scores = beam_state.scores[batch_start:batch_end]
            best_beam_idx = torch.argmax(batch_scores)
            
            best_sequence = beam_state.sequences[batch_start + best_beam_idx]
            best_score = batch_scores[best_beam_idx]
            
            # 提取生成的部分
            generated_part = best_sequence[input_length:]
            
            best_sequences.append(generated_part)
            best_scores.append(best_score.item())
        
        return GenerationResult(
            sequences=best_sequences,
            scores=best_scores,
            num_tokens=sum(len(seq) for seq in best_sequences),
            finished=True,
            stop_reason="beam_search_complete"
        )
    
    def _apply_generation_constraints(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        config: GenerationConfig,
    ) -> torch.Tensor:
        """应用生成约束"""
        
        # 重复惩罚
        if config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(
                logits, past_tokens, config.repetition_penalty
            )
        
        # 长度惩罚（在采样阶段应用）
        if config.length_penalty != 1.0:
            # 对短序列给予奖励，对长序列给予惩罚
            current_length = past_tokens.size(-1)
            if current_length > config.min_length:
                eos_penalty = -config.length_penalty
                if config.eos_token_id is not None:
                    logits[:, config.eos_token_id] += eos_penalty
        
        # 禁用token
        if hasattr(config, 'bad_words_ids') and config.bad_words_ids:
            for bad_word_ids in config.bad_words_ids:
                if len(bad_word_ids) == 1:
                    logits[:, bad_word_ids[0]] = float('-inf')
        
        return logits
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚"""
        
        if penalty == 1.0:
            return logits
        
        batch_size = logits.size(0)
        
        for batch_idx in range(batch_size):
            # 获取已生成的token
            generated_tokens = past_tokens[batch_idx]
            
            # 计算token频率
            unique_tokens = torch.unique(generated_tokens)
            
            for token in unique_tokens:
                # 应用惩罚
                if logits[batch_idx, token] > 0:
                    logits[batch_idx, token] /= penalty
                else:
                    logits[batch_idx, token] *= penalty
        
        return logits
    
    def _update_stats(self, result: GenerationResult):
        """更新统计信息"""
        
        self.stats.total_sequences_generated += len(result.sequences)
        self.stats.total_tokens_generated += result.num_tokens
        self.stats.generation_time += result.generation_time
        
        if self.stats.total_sequences_generated > 0:
            self.stats.average_tokens_per_sequence = (
                self.stats.total_tokens_generated / 
                self.stats.total_sequences_generated
            )
    
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        model_forward_fn: Optional[Callable] = None,
        **kwargs
    ) -> Iterator[torch.Tensor]:
        """流式生成"""
        
        # 合并配置参数
        generation_config = self._merge_generation_config(kwargs)
        
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # 初始化生成序列
        generated_sequences = input_ids.clone()
        
        # 初始化attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        current_attention_mask = attention_mask.clone()
        
        # 生成循环
        for step in range(generation_config.max_new_tokens):
            # 前向传播
            outputs = model_forward_fn(
                input_ids=generated_sequences,
                attention_mask=current_attention_mask,
            )
            
            # 获取logits
            logits = outputs.logits[:, -1, :]
            
            # 应用生成约束
            logits = self._apply_generation_constraints(
                logits, generated_sequences, generation_config
            )
            
            # 采样下一个token
            next_tokens = self.sampler_manager.sample(
                logits,
                strategy=SamplingStrategy(generation_config.sampling_strategy),
                past_tokens=generated_sequences,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )
            
            # 更新序列
            generated_sequences = torch.cat([
                generated_sequences,
                next_tokens.unsqueeze(-1)
            ], dim=-1)
            
            # 更新attention mask
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, 1, device=device)
            ], dim=-1)
            
            # 产出新token
            yield next_tokens
            
            # 检查停止条件
            should_stop = self.stopping_checker.should_stop(generated_sequences)
            
            if should_stop.all():
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        
        return {
            "total_tokens_generated": self.stats.total_tokens_generated,
            "total_sequences_generated": self.stats.total_sequences_generated,
            "generation_time": self.stats.generation_time,
            "average_tokens_per_sequence": self.stats.average_tokens_per_sequence,
            "generation_throughput": (
                self.stats.total_tokens_generated / 
                max(self.stats.generation_time, 0.001)
            ),
        }

class PostProcessor:
    """后处理器"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
        # 后处理规则
        self.cleanup_rules = [
            self._remove_special_tokens,
            self._clean_whitespace,
            self._fix_punctuation,
            self._apply_custom_rules,
        ]
        
        logger.info("Initialized PostProcessor")
    
    def process(
        self,
        generated_text: str,
        input_text: Optional[str] = None,
    ) -> str:
        """后处理生成的文本"""
        
        processed_text = generated_text
        
        # 应用后处理规则
        for rule in self.cleanup_rules:
            processed_text = rule(processed_text, input_text)
        
        return processed_text
    
    def _remove_special_tokens(self, text: str, input_text: Optional[str] = None) -> str:
        """移除特殊token"""
        
        # 移除常见的特殊token
        special_tokens = ['<pad>', '<eos>', '<bos>', '<unk>', '<|endoftext|>']
        
        for token in special_tokens:
            text = text.replace(token, '')
        
        return text
    
    def _clean_whitespace(self, text: str, input_text: Optional[str] = None) -> str:
        """清理空白字符"""
        
        import re
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除行首行尾空格
        text = text.strip()
        
        return text
    
    def _fix_punctuation(self, text: str, input_text: Optional[str] = None) -> str:
        """修复标点符号"""
        
        import re
        
        # 修复标点符号前的空格
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # 修复引号
        text = re.sub(r'\s+"([^"]*?)"\s+', r' "\1" ', text)
        
        return text
    
    def _apply_custom_rules(self, text: str, input_text: Optional[str] = None) -> str:
        """应用自定义规则"""
        
        # 这里可以添加特定的后处理规则
        # 例如：特定格式的修复、领域特定的清理等
        
        return text
    
    def batch_process(self, texts: List[str], input_texts: Optional[List[str]] = None) -> List[str]:
        """批量后处理"""
        
        if input_texts is None:
            input_texts = [None] * len(texts)
        
        processed_texts = []
        
        for text, input_text in zip(texts, input_texts):
            processed_text = self.process(text, input_text)
            processed_texts.append(processed_text)
        
        return processed_texts

# 性能优化组件
class OptimizedGenerator:
    """优化的生成器"""
    
    def __init__(
        self,
        base_generator: TextGenerator,
        cache_size: int = 10000,
        num_workers: int = 4,
    ):
        self.base_generator = base_generator
        self.cache_size = cache_size
        self.num_workers = num_workers
        
        # 生成缓存
        self.generation_cache = {}
        self.cache_lock = threading.RLock()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(f"Initialized OptimizedGenerator with cache_size={cache_size}")
    
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> GenerationResult:
        """带缓存的生成"""
        
        # 生成缓存键
        cache_key = self._generate_cache_key(input_ids, kwargs)
        
        # 检查缓存
        with self.cache_lock:
            if cache_key in self.generation_cache:
                logger.debug("Cache hit for generation")
                return self.generation_cache[cache_key]
        
        # 执行生成
        result = self.base_generator.generate(input_ids, **kwargs)
        
        # 更新缓存
        with self.cache_lock:
            if len(self.generation_cache) >= self.cache_size:
                # 移除最旧的条目
                oldest_key = next(iter(self.generation_cache))
                del self.generation_cache[oldest_key]
            
            self.generation_cache[cache_key] = result
        
        return result
    
    def generate_batch_parallel(
        self,
        input_ids_list: List[torch.Tensor],
        **kwargs
    ) -> List[GenerationResult]:
        """并行批量生成"""
        
        # 提交任务
        futures = []
        for input_ids in input_ids_list:
            future = self.thread_pool.submit(
                self.generate_with_cache,
                input_ids,
                **kwargs
            )
            futures.append(future)
        
        # 收集结果
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
        
        return results
    
    def _generate_cache_key(self, input_ids: torch.Tensor, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        
        import hashlib
        import json
        
        # 组合输入和参数
        key_data = {
            'input_ids': input_ids.tolist(),
            'kwargs': kwargs,
        }
        
        # 生成哈希
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear_cache(self):
        """清空缓存"""
        
        with self.cache_lock:
            self.generation_cache.clear()
            logger.info("Generation cache cleared")
    
    def shutdown(self):
        """关闭线程池"""
        self.thread_pool.shutdown(wait=True)

# 使用示例
def example_generation_usage():
    """生成工具使用示例"""
    
    # 配置
    generation_config = GenerationConfig(
        max_length=512,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        num_beams=1,
        repetition_penalty=1.1,
        length_penalty=1.0,
        early_stopping=True,
        sampling_strategy="top_p",
    )
    
    # 创建组件
    sampler_manager = SamplerManager(generation_config)
    stopping_checker = StoppingCriteriaChecker(generation_config)
    text_generator = TextGenerator(
        generation_config,
        sampler_manager,
        stopping_checker,
    )
    
    # 模拟输入
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # 示例输入
    
    # 模拟模型前向传播函数
    def mock_model_forward(input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        vocab_size = 1000
        
        # 生成随机logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)
    
    # 生成文本
    result = text_generator.generate(
        input_ids=input_ids,
        model_forward_fn=mock_model_forward,
        temperature=0.8,
        top_p=0.9,
    )
    
    print(f"Generated {len(result.sequences)} sequences")
    print(f"Total tokens: {result.num_tokens}")
    print(f"Generation time: {result.generation_time:.4f}s")
    
    # 流式生成示例
    print("\nStreaming generation:")
    for i, token in enumerate(text_generator.generate_streaming(
        input_ids=input_ids,
        model_forward_fn=mock_model_forward,
        max_new_tokens=10,
    )):
        print(f"Step {i}: {token}")
    
    # 束搜索生成示例
    beam_result = text_generator.generate(
        input_ids=input_ids,
        model_forward_fn=mock_model_forward,
        num_beams=4,
        max_new_tokens=20,
    )
    
    print(f"\nBeam search generated {len(beam_result.sequences)} sequences")
    print(f"Best score: {beam_result.scores[0] if beam_result.scores else 'N/A'}")
    
    # 后处理示例
    post_processor = PostProcessor(generation_config)
    
    raw_text = "  Hello , world !  <eos>  "
    processed_text = post_processor.process(raw_text)
    print(f"\nRaw text: '{raw_text}'")
    print(f"Processed text: '{processed_text}'")
    
    # 获取统计信息
    stats = text_generator.get_stats()
    print(f"\nGeneration stats: {stats}")

if __name__ == "__main__":
    example_generation_usage()
```

## 🔧 关键特性分析

### 1. 多种采样策略

- **贪心采样**：选择概率最大的token
- **多项式采样**：基于概率分布随机采样
- **Top-K采样**：从概率最高的K个token中采样
- **Top-P采样**：从累积概率达到P的token集合中采样
- **Typical采样**：基于信息熵的采样策略
- **束搜索**：维护多个候选序列的搜索算法

### 2. 智能停止条件

- **最大长度限制**：防止无限生成
- **EOS token检测**：识别结束标记
- **停止词匹配**：自定义停止条件
- **自定义规则**：灵活的停止逻辑

### 3. 生成约束控制

- **重复惩罚**：减少重复内容
- **长度惩罚**：控制生成长度
- **禁用词过滤**：避免不当内容
- **格式约束**：确保输出格式

## 📊 性能优化技术

### 预计算优化

```python
class PrecomputedSampler:
    """预计算采样器"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        
        # 预计算常用的概率分布
        self.precomputed_distributions = {}
        
        # 预计算Top-K索引
        self.topk_cache = {}
    
    def precompute_topk_indices(self, k_values: List[int]):
        """预计算Top-K索引"""
        
        for k in k_values:
            # 为不同的k值预计算索引模板
            indices = torch.arange(self.vocab_size)
            self.topk_cache[k] = indices[:k]
    
    def fast_topk_sample(
        self,
        logits: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """快速Top-K采样"""
        
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 使用预计算的索引（如果可用）
        if k in self.topk_cache and k <= 100:  # 只对小k值使用缓存
            # 快速路径：使用预计算的索引
            top_values, _ = torch.topk(logits, k=k)
            top_probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(top_probs, num_samples=1)
            
            # 获取实际token索引需要重新计算
            _, top_indices = torch.topk(logits, k=k)
            return torch.gather(top_indices, -1, sampled_idx).squeeze(-1)
        
        else:
            # 标准路径
            top_values, top_indices = torch.topk(logits, k=k)
            top_probs = F.softmax(top_values, dim=-1)
            sampled_idx = torch.multinomial(top_probs, num_samples=1)
            return torch.gather(top_indices, -1, sampled_idx).squeeze(-1)
```

### 批量优化

```python
class BatchOptimizedGenerator:
    """批量优化生成器"""
    
    def __init__(self, base_generator: TextGenerator):
        self.base_generator = base_generator
        
        # 批量配置
        self.optimal_batch_sizes = {
            'short': 64,   # < 128 tokens
            'medium': 32,  # 128-512 tokens
            'long': 16,    # > 512 tokens
        }
    
    def generate_dynamic_batch(
        self,
        inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """动态批量生成"""
        
        # 按序列长度分组
        grouped_inputs = self._group_inputs_by_length(inputs)
        
        results = []
        
        for length_category, group_inputs in grouped_inputs.items():
            batch_size = self.optimal_batch_sizes[length_category]
            
            # 分批处理
            for i in range(0, len(group_inputs), batch_size):
                batch_inputs = group_inputs[i:i + batch_size]
                
                # 批量生成
                batch_results = self._generate_batch(batch_inputs, **kwargs)
                results.extend(batch_results)
        
        return results
    
    def _group_inputs_by_length(self, inputs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """按长度分组输入"""
        
        groups = {
            'short': [],
            'medium': [],
            'long': [],
        }
        
        for input_data in inputs:
            input_ids = input_data['input_ids']
            seq_len = input_ids.size(-1)
            
            if seq_len < 128:
                groups['short'].append(input_data)
            elif seq_len < 512:
                groups['medium'].append(input_data)
            else:
                groups['long'].append(input_data)
        
        return groups
    
    def _generate_batch(
        self,
        batch_inputs: List[Dict[str, Any]],
        **kwargs
    ) -> List[GenerationResult]:
        """批量生成"""
        
        # 合并输入
        input_ids_list = [inp['input_ids'] for inp in batch_inputs]
        attention_mask_list = [inp.get('attention_mask') for inp in batch_inputs]
        
        # 填充到相同长度
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=0
        )
        
        if attention_mask_list[0] is not None:
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask_list, batch_first=True, padding_value=0
            )
        else:
            padded_attention_mask = None
        
        # 批量生成
        batch_result = self.base_generator.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            **kwargs
        )
        
        # 分解批量结果
        results = []
        for i in range(len(batch_inputs)):
            result = GenerationResult(
                sequences=[batch_result.sequences[i]],
                scores=[batch_result.scores[i]] if batch_result.scores else None,
                generation_time=batch_result.generation_time / len(batch_inputs),
                num_tokens=len(batch_result.sequences[i]),
                finished=batch_result.finished,
                stop_reason=batch_result.stop_reason,
            )
            results.append(result)
        
        return results
```

### 内存优化

```python
class MemoryEfficientGenerator:
    """内存高效生成器"""
    
    def __init__(self, base_generator: TextGenerator):
        self.base_generator = base_generator
        
        # 内存池
        self.tensor_pool = {
            'logits': [],
            'sequences': [],
            'attention_mask': [],
        }
        
        self.pool_lock = threading.Lock()
    
    def generate_with_memory_pool(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> GenerationResult:
        """使用内存池的生成"""
        
        # 从池中获取tensor
        logits_tensor = self._get_tensor_from_pool('logits', input_ids.device)
        sequence_tensor = self._get_tensor_from_pool('sequences', input_ids.device)
        
        try:
            # 执行生成
            result = self.base_generator.generate(input_ids, **kwargs)
            
            return result
            
        finally:
            # 返回tensor到池中
            if logits_tensor is not None:
                self._return_tensor_to_pool('logits', logits_tensor)
            if sequence_tensor is not None:
                self._return_tensor_to_pool('sequences', sequence_tensor)
    
    def _get_tensor_from_pool(self, tensor_type: str, device: torch.device) -> Optional[torch.Tensor]:
        """从池中获取tensor"""
        
        with self.pool_lock:
            pool = self.tensor_pool[tensor_type]
            
            for i, tensor in enumerate(pool):
                if tensor.device == device:
                    return pool.pop(i)
        
        return None
    
    def _return_tensor_to_pool(self, tensor_type: str, tensor: torch.Tensor):
        """返回tensor到池中"""
        
        with self.pool_lock:
            pool = self.tensor_pool[tensor_type]
            
            # 限制池大小
            if len(pool) < 10:
                tensor.zero_()  # 清零tensor
                pool.append(tensor)
    
    def generate_with_gradient_checkpointing(
        self,
        input_ids: torch.Tensor,
        model_forward_fn: Callable,
        **kwargs
    ) -> GenerationResult:
        """使用梯度检查点的生成"""
        
        def checkpointed_forward(input_ids, attention_mask):
            # 使用梯度检查点减少内存使用
            return torch.utils.checkpoint.checkpoint(
                model_forward_fn,
                input_ids,
                attention_mask,
            )
        
        return self.base_generator.generate(
            input_ids=input_ids,
            model_forward_fn=checkpointed_forward,
            **kwargs
        )
```

## 🚀 使用最佳实践

### 1. 采样策略选择指南

```python
def choose_sampling_strategy(
    task_type: str,
    creativity_level: str,
    quality_priority: str,
) -> Dict[str, Any]:
    """选择最适合的采样策略"""
    
    if task_type == "factual_qa":
        # 事实问答：优先准确性
        return {
            "sampling_strategy": "greedy",
            "temperature": 0.1,
        }
    
    elif task_type == "creative_writing":
        # 创意写作：平衡创造性和质量
        if creativity_level == "high":
            return {
                "sampling_strategy": "top_p",
                "temperature": 0.9,
                "top_p": 0.9,
            }
        else:
            return {
                "sampling_strategy": "top_k",
                "temperature": 0.7,
                "top_k": 40,
            }
    
    elif task_type == "code_generation":
        # 代码生成：优先正确性
        return {
            "sampling_strategy": "top_p",
            "temperature": 0.2,
            "top_p": 0.95,
        }
    
    elif quality_priority == "high":
        # 高质量要求：使用束搜索
        return {
            "sampling_strategy": "beam_search",
            "num_beams": 4,
            "length_penalty": 1.2,
        }
    
    else:
        # 默认配置
        return {
            "sampling_strategy": "top_p",
            "temperature": 0.8,
            "top_p": 0.9,
        }
```

### 2. 性能监控和调优

```python
class GenerationProfiler:
    """生成性能分析器"""
    
    def __init__(self):
        self.metrics = {
            'sampling_times': [],
            'generation_times': [],
            'memory_usage': [],
            'throughput': [],
        }
    
    def profile_generation(
        self,
        generator: TextGenerator,
        input_ids: torch.Tensor,
        **kwargs
    ) -> Tuple[GenerationResult, Dict[str, float]]:
        """分析生成性能"""
        
        import psutil
        import torch
        
        # 记录初始状态
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        start_time = time.time()
        
        # 执行生成
        result = generator.generate(input_ids, **kwargs)
        
        end_time = time.time()
        
        # 记录结束状态
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # 计算指标
        generation_time = end_time - start_time
        memory_increase = end_memory - start_memory
        gpu_memory_increase = end_gpu_memory - start_gpu_memory
        throughput = result.num_tokens / generation_time
        
        profile_data = {
            'generation_time': generation_time,
            'memory_increase_mb': memory_increase,
            'gpu_memory_increase_mb': gpu_memory_increase,
            'throughput_tokens_per_sec': throughput,
            'tokens_generated': result.num_tokens,
        }
        
        # 更新历史指标
        self.metrics['generation_times'].append(generation_time)
        self.metrics['memory_usage'].append(memory_increase)
        self.metrics['throughput'].append(throughput)
        
        return result, profile_data
    
    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        
        if not self.metrics['generation_times']:
            return {}
        
        return {
            'avg_generation_time': np.mean(self.metrics['generation_times']),
            'avg_memory_usage': np.mean(self.metrics['memory_usage']),
            'avg_throughput': np.mean(self.metrics['throughput']),
            'max_generation_time': np.max(self.metrics['generation_times']),
            'min_generation_time': np.min(self.metrics['generation_times']),
        }
    
    def suggest_optimizations(self) -> List[str]:
        """建议优化措施"""
        
        suggestions = []
        
        if not self.metrics['generation_times']:
            return suggestions
        
        avg_time = np.mean(self.metrics['generation_times'])
        avg_memory = np.mean(self.metrics['memory_usage'])
        avg_throughput = np.mean(self.metrics['throughput'])
        
        if avg_time > 5.0:  # 生成时间过长
            suggestions.append("Consider using faster sampling strategies (greedy/top-k)")
            suggestions.append("Reduce max_new_tokens or use early stopping")
        
        if avg_memory > 1000:  # 内存使用过多
            suggestions.append("Enable gradient checkpointing")
            suggestions.append("Use smaller batch sizes")
            suggestions.append("Consider model quantization")
        
        if avg_throughput < 10:  # 吞吐量过低
            suggestions.append("Optimize batch processing")
            suggestions.append("Use tensor parallelism")
            suggestions.append("Consider using faster hardware")
        
        return suggestions
```

### 3. 错误处理和恢复

```python
class RobustGenerator:
    """健壮的生成器"""
    
    def __init__(self, base_generator: TextGenerator):
        self.base_generator = base_generator
        
        # 错误统计
        self.error_counts = {
            'oom_errors': 0,
            'timeout_errors': 0,
            'generation_errors': 0,
        }
        
        # 恢复策略
        self.recovery_strategies = {
            'oom': self._handle_oom_error,
            'timeout': self._handle_timeout_error,
            'generation': self._handle_generation_error,
        }
    
    def generate_with_recovery(
        self,
        input_ids: torch.Tensor,
        max_retries: int = 3,
        **kwargs
    ) -> Optional[GenerationResult]:
        """带恢复机制的生成"""
        
        for attempt in range(max_retries):
            try:
                return self.base_generator.generate(input_ids, **kwargs)
                
            except torch.cuda.OutOfMemoryError as e:
                self.error_counts['oom_errors'] += 1
                logger.warning(f"OOM error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    # 尝试恢复
                    recovery_kwargs = self.recovery_strategies['oom'](kwargs)
                    kwargs.update(recovery_kwargs)
                else:
                    logger.error("Max retries reached for OOM error")
                    return None
                    
            except TimeoutError as e:
                self.error_counts['timeout_errors'] += 1
                logger.warning(f"Timeout error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    recovery_kwargs = self.recovery_strategies['timeout'](kwargs)
                    kwargs.update(recovery_kwargs)
                else:
                    logger.error("Max retries reached for timeout error")
                    return None
                    
            except Exception as e:
                self.error_counts['generation_errors'] += 1
                logger.warning(f"Generation error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    recovery_kwargs = self.recovery_strategies['generation'](kwargs)
                    kwargs.update(recovery_kwargs)
                else:
                    logger.error("Max retries reached for generation error")
                    return None
        
        return None
    
    def _handle_oom_error(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """处理OOM错误"""
        
        recovery_kwargs = {}
        
        # 减少批量大小
        if 'batch_size' in kwargs:
            recovery_kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
        
        # 减少最大长度
        if 'max_new_tokens' in kwargs:
            recovery_kwargs['max_new_tokens'] = max(10, kwargs['max_new_tokens'] // 2)
        
        # 启用梯度检查点
        recovery_kwargs['use_gradient_checkpointing'] = True
        
        logger.info(f"OOM recovery: {recovery_kwargs}")
        return recovery_kwargs
    
    def _handle_timeout_error(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """处理超时错误"""
        
        recovery_kwargs = {}
        
        # 减少生成长度
        if 'max_new_tokens' in kwargs:
            recovery_kwargs['max_new_tokens'] = max(10, kwargs['max_new_tokens'] // 2)
        
        # 使用更快的采样策略
        recovery_kwargs['sampling_strategy'] = 'greedy'
        recovery_kwargs['num_beams'] = 1
        
        logger.info(f"Timeout recovery: {recovery_kwargs}")
        return recovery_kwargs
    
    def _handle_generation_error(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """处理生成错误"""
        
        recovery_kwargs = {}
        
        # 使用保守的参数
        recovery_kwargs['temperature'] = 0.7
        recovery_kwargs['top_p'] = 0.9
        recovery_kwargs['repetition_penalty'] = 1.0
        
        logger.info(f"Generation recovery: {recovery_kwargs}")
        return recovery_kwargs
    
    def get_error_stats(self) -> Dict[str, int]:
        """获取错误统计"""
        return self.error_counts.copy()
```

## 🎯 总结

生成工具是 nano-vllm 中最复杂的组件之一，它集成了多种采样策略、停止条件、性能优化和错误处理机制。通过合理配置和使用这些工具，可以实现高质量、高效率的文本生成。

### 关键要点

1. **采样策略选择**：根据任务类型选择合适的采样方法
2. **性能优化**：使用批量处理、内存池、缓存等技术
3. **错误处理**：实现健壮的错误恢复机制
4. **监控调优**：持续监控性能并进行优化

### 使用建议

- 对于事实性任务使用贪心或低温度采样
- 对于创意性任务使用Top-P或Top-K采样
- 合理设置停止条件避免过度生成
- 监控内存使用和生成速度
- 实现错误恢复机制提高系统稳定性

通过深入理解这些生成工具的实现原理和使用方法，可以更好地优化和扩展 nano-vllm 的文本生成能力。