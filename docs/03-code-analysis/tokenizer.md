# 分词器分析

## 🎯 分词器概览

分词器是 nano-vllm 中负责文本预处理和后处理的核心组件。它将输入文本转换为模型可理解的token序列，并将模型输出的token序列转换回可读文本。高效的分词器实现对于提升推理性能至关重要。

## 🏗️ 核心架构

```python
import torch
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

from transformers import AutoTokenizer, PreTrainedTokenizer
from nano_vllm.config import TokenizerConfig
from nano_vllm.utils.logger import get_logger

logger = get_logger(__name__)

class TokenizerType(Enum):
    """分词器类型枚举"""
    HUGGINGFACE = "huggingface"
    SENTENCEPIECE = "sentencepiece"
    TIKTOKEN = "tiktoken"
    CUSTOM = "custom"

class EncodingMode(Enum):
    """编码模式枚举"""
    STANDARD = "standard"      # 标准编码
    FAST = "fast"             # 快速编码
    BATCH = "batch"           # 批量编码
    STREAMING = "streaming"    # 流式编码

@dataclass
class TokenizerStats:
    """分词器统计信息"""
    total_tokens_processed: int = 0
    total_texts_processed: int = 0
    encoding_time: float = 0.0
    decoding_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    average_tokens_per_text: float = 0.0
    peak_memory_usage: int = 0

@dataclass
class EncodingResult:
    """编码结果"""
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    special_tokens_mask: Optional[torch.Tensor] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None
    encoding_time: float = 0.0
    num_tokens: int = 0

@dataclass
class DecodingResult:
    """解码结果"""
    text: str
    tokens: List[str] = field(default_factory=list)
    token_ids: List[int] = field(default_factory=list)
    decoding_time: float = 0.0
    skip_special_tokens: bool = True

class BaseTokenizer(ABC):
    """基础分词器抽象类"""
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        vocab_size: Optional[int] = None,
    ):
        self.config = tokenizer_config
        self.vocab_size = vocab_size
        
        # 统计信息
        self.stats = TokenizerStats()
        
        # 缓存
        self.encoding_cache: Dict[str, EncodingResult] = {}
        self.decoding_cache: Dict[Tuple[int, ...], DecodingResult] = {}
        self.cache_lock = threading.RLock()
        
        # 特殊token
        self.pad_token_id = None
        self.eos_token_id = None
        self.bos_token_id = None
        self.unk_token_id = None
        
        logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[EncodingResult, List[EncodingResult]]:
        """编码文本为token序列"""
        pass
    
    @abstractmethod
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[DecodingResult, List[DecodingResult]]:
        """解码token序列为文本"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        pass
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[EncodingResult]:
        """批量编码"""
        
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = self.encode(batch_texts, **kwargs)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def decode_batch(
        self,
        token_ids_list: List[List[int]],
        batch_size: int = 32,
        **kwargs
    ) -> List[DecodingResult]:
        """批量解码"""
        
        results = []
        
        for i in range(0, len(token_ids_list), batch_size):
            batch_token_ids = token_ids_list[i:i + batch_size]
            batch_results = self.decode(batch_token_ids, **kwargs)
            
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def encode_streaming(
        self,
        text_stream: Iterator[str],
        **kwargs
    ) -> Iterator[EncodingResult]:
        """流式编码"""
        
        for text in text_stream:
            yield self.encode(text, **kwargs)
    
    def decode_streaming(
        self,
        token_stream: Iterator[List[int]],
        **kwargs
    ) -> Iterator[DecodingResult]:
        """流式解码"""
        
        for tokens in token_stream:
            yield self.decode(tokens, **kwargs)
    
    def _update_stats(
        self,
        operation: str,
        processing_time: float,
        num_items: int = 1,
        num_tokens: int = 0,
    ):
        """更新统计信息"""
        
        if operation == "encode":
            self.stats.total_texts_processed += num_items
            self.stats.total_tokens_processed += num_tokens
            self.stats.encoding_time += processing_time
        elif operation == "decode":
            self.stats.total_texts_processed += num_items
            self.stats.decoding_time += processing_time
        
        if num_tokens > 0:
            self.stats.average_tokens_per_text = (
                self.stats.total_tokens_processed / 
                max(self.stats.total_texts_processed, 1)
            )
    
    def _check_cache(self, key: Any, cache_type: str) -> Optional[Any]:
        """检查缓存"""
        
        with self.cache_lock:
            if cache_type == "encode" and key in self.encoding_cache:
                self.stats.cache_hits += 1
                return self.encoding_cache[key]
            elif cache_type == "decode" and key in self.decoding_cache:
                self.stats.cache_hits += 1
                return self.decoding_cache[key]
            
            self.stats.cache_misses += 1
            return None
    
    def _update_cache(self, key: Any, value: Any, cache_type: str):
        """更新缓存"""
        
        with self.cache_lock:
            if cache_type == "encode":
                # 限制缓存大小
                if len(self.encoding_cache) >= self.config.max_cache_size:
                    # 删除最旧的条目
                    oldest_key = next(iter(self.encoding_cache))
                    del self.encoding_cache[oldest_key]
                
                self.encoding_cache[key] = value
                
            elif cache_type == "decode":
                if len(self.decoding_cache) >= self.config.max_cache_size:
                    oldest_key = next(iter(self.decoding_cache))
                    del self.decoding_cache[oldest_key]
                
                self.decoding_cache[key] = value
    
    def clear_cache(self):
        """清空缓存"""
        
        with self.cache_lock:
            self.encoding_cache.clear()
            self.decoding_cache.clear()
            logger.info("Tokenizer cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        
        return {
            "total_tokens_processed": self.stats.total_tokens_processed,
            "total_texts_processed": self.stats.total_texts_processed,
            "encoding_time": self.stats.encoding_time,
            "decoding_time": self.stats.decoding_time,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": (
                self.stats.cache_hits / 
                max(self.stats.cache_hits + self.stats.cache_misses, 1)
            ),
            "average_tokens_per_text": self.stats.average_tokens_per_text,
            "encoding_throughput": (
                self.stats.total_texts_processed / 
                max(self.stats.encoding_time, 0.001)
            ),
            "decoding_throughput": (
                self.stats.total_texts_processed / 
                max(self.stats.decoding_time, 0.001)
            ),
        }

class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace分词器实现"""
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_name_or_path: str,
    ):
        super().__init__(tokenizer_config)
        
        self.model_name_or_path = model_name_or_path
        
        # 加载分词器
        self.tokenizer = self._load_tokenizer()
        
        # 设置特殊token
        self._setup_special_tokens()
        
        logger.info(f"Loaded HuggingFace tokenizer: {model_name_or_path}")
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """加载HuggingFace分词器"""
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=self.config.use_fast,
                revision=self.config.revision,
                use_auth_token=self.config.use_auth_token,
            )
            
            # 设置padding token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _setup_special_tokens(self):
        """设置特殊token"""
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        self.unk_token_id = getattr(self.tokenizer, 'unk_token_id', None)
        
        self.vocab_size = len(self.tokenizer)
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[EncodingResult, List[EncodingResult]]:
        """编码文本"""
        
        start_time = time.time()
        
        # 检查缓存
        if isinstance(text, str):
            cache_key = f"{text}_{add_special_tokens}_{max_length}_{padding}_{truncation}"
            cached_result = self._check_cache(cache_key, "encode")
            if cached_result is not None:
                return cached_result
        
        try:
            # 使用HuggingFace tokenizer编码
            encoded = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_offsets_mapping=self.config.return_offsets,
            )
            
            encoding_time = time.time() - start_time
            
            # 处理单个文本
            if isinstance(text, str):
                result = EncodingResult(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded.get('attention_mask'),
                    offset_mapping=encoded.get('offset_mapping'),
                    encoding_time=encoding_time,
                    num_tokens=len(encoded['input_ids'][0]) if encoded['input_ids'].dim() > 1 else len(encoded['input_ids']),
                )
                
                # 更新缓存
                self._update_cache(cache_key, result, "encode")
                
                # 更新统计
                self._update_stats("encode", encoding_time, 1, result.num_tokens)
                
                return result
            
            # 处理批量文本
            else:
                results = []
                batch_size = len(text)
                
                for i in range(batch_size):
                    result = EncodingResult(
                        input_ids=encoded['input_ids'][i],
                        attention_mask=encoded['attention_mask'][i] if encoded.get('attention_mask') is not None else None,
                        offset_mapping=encoded['offset_mapping'][i] if encoded.get('offset_mapping') is not None else None,
                        encoding_time=encoding_time / batch_size,
                        num_tokens=len(encoded['input_ids'][i]),
                    )
                    results.append(result)
                
                # 更新统计
                total_tokens = sum(result.num_tokens for result in results)
                self._update_stats("encode", encoding_time, batch_size, total_tokens)
                
                return results
                
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[DecodingResult, List[DecodingResult]]:
        """解码token序列"""
        
        start_time = time.time()
        
        # 处理输入格式
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                token_ids = token_ids.tolist()
            else:
                token_ids = token_ids.tolist()
        
        # 检查缓存
        if isinstance(token_ids[0], int):  # 单个序列
            cache_key = (tuple(token_ids), skip_special_tokens)
            cached_result = self._check_cache(cache_key, "decode")
            if cached_result is not None:
                return cached_result
        
        try:
            # 使用HuggingFace tokenizer解码
            if isinstance(token_ids[0], int):  # 单个序列
                text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                
                decoding_time = time.time() - start_time
                
                result = DecodingResult(
                    text=text,
                    token_ids=token_ids,
                    decoding_time=decoding_time,
                    skip_special_tokens=skip_special_tokens,
                )
                
                # 更新缓存
                self._update_cache(cache_key, result, "decode")
                
                # 更新统计
                self._update_stats("decode", decoding_time, 1)
                
                return result
            
            else:  # 批量序列
                texts = self.tokenizer.batch_decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                
                decoding_time = time.time() - start_time
                batch_size = len(token_ids)
                
                results = []
                for i, text in enumerate(texts):
                    result = DecodingResult(
                        text=text,
                        token_ids=token_ids[i],
                        decoding_time=decoding_time / batch_size,
                        skip_special_tokens=skip_special_tokens,
                    )
                    results.append(result)
                
                # 更新统计
                self._update_stats("decode", decoding_time, batch_size)
                
                return results
                
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            raise
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.tokenizer)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """将token转换为ID"""
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """将ID转换为token"""
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def tokenize(self, text: str) -> List[str]:
        """分词（不转换为ID）"""
        return self.tokenizer.tokenize(text)

class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece分词器实现"""
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_path: str,
    ):
        super().__init__(tokenizer_config)
        
        self.model_path = model_path
        
        # 加载SentencePiece模型
        self.sp_model = self._load_sentencepiece_model()
        
        # 设置特殊token
        self._setup_special_tokens()
        
        logger.info(f"Loaded SentencePiece tokenizer: {model_path}")
    
    def _load_sentencepiece_model(self):
        """加载SentencePiece模型"""
        
        try:
            import sentencepiece as spm
            
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(self.model_path)
            
            return sp_model
            
        except ImportError:
            raise ImportError("sentencepiece is required for SentencePieceTokenizer")
        except Exception as e:
            logger.error(f"Failed to load SentencePiece model: {e}")
            raise
    
    def _setup_special_tokens(self):
        """设置特殊token"""
        
        self.vocab_size = self.sp_model.get_piece_size()
        self.pad_token_id = self.sp_model.pad_id()
        self.eos_token_id = self.sp_model.eos_id()
        self.bos_token_id = self.sp_model.bos_id()
        self.unk_token_id = self.sp_model.unk_id()
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[EncodingResult, List[EncodingResult]]:
        """编码文本"""
        
        start_time = time.time()
        
        if isinstance(text, str):
            # 单个文本编码
            token_ids = self.sp_model.encode_as_ids(text)
            
            # 添加特殊token
            if add_special_tokens:
                if self.bos_token_id is not None:
                    token_ids = [self.bos_token_id] + token_ids
                if self.eos_token_id is not None:
                    token_ids = token_ids + [self.eos_token_id]
            
            # 截断
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            
            # 填充
            if padding and max_length is not None:
                if len(token_ids) < max_length:
                    pad_length = max_length - len(token_ids)
                    token_ids = token_ids + [self.pad_token_id] * pad_length
            
            # 创建attention mask
            attention_mask = [1] * len(token_ids)
            if padding and self.pad_token_id is not None:
                for i, token_id in enumerate(token_ids):
                    if token_id == self.pad_token_id:
                        attention_mask[i] = 0
            
            # 转换为tensor
            if return_tensors == "pt":
                input_ids = torch.tensor([token_ids])
                attention_mask = torch.tensor([attention_mask])
            else:
                input_ids = torch.tensor(token_ids)
                attention_mask = torch.tensor(attention_mask)
            
            encoding_time = time.time() - start_time
            
            result = EncodingResult(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoding_time=encoding_time,
                num_tokens=len(token_ids),
            )
            
            self._update_stats("encode", encoding_time, 1, len(token_ids))
            
            return result
        
        else:
            # 批量文本编码
            results = []
            
            for single_text in text:
                result = self.encode(
                    single_text,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None,  # 单独处理
                )
                results.append(result)
            
            # 批量转换为tensor
            if return_tensors == "pt":
                input_ids = torch.stack([r.input_ids for r in results])
                attention_mask = torch.stack([r.attention_mask for r in results])
                
                for i, result in enumerate(results):
                    result.input_ids = input_ids[i]
                    result.attention_mask = attention_mask[i]
            
            return results
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[DecodingResult, List[DecodingResult]]:
        """解码token序列"""
        
        start_time = time.time()
        
        # 处理输入格式
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], int):  # 单个序列
            # 过滤特殊token
            if skip_special_tokens:
                filtered_ids = []
                for token_id in token_ids:
                    if token_id not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                        filtered_ids.append(token_id)
                token_ids = filtered_ids
            
            # 解码
            text = self.sp_model.decode_ids(token_ids)
            
            decoding_time = time.time() - start_time
            
            result = DecodingResult(
                text=text,
                token_ids=token_ids,
                decoding_time=decoding_time,
                skip_special_tokens=skip_special_tokens,
            )
            
            self._update_stats("decode", decoding_time, 1)
            
            return result
        
        else:  # 批量序列
            results = []
            
            for single_token_ids in token_ids:
                result = self.decode(
                    single_token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                results.append(result)
            
            return results
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.sp_model.get_piece_size()

class TikTokenTokenizer(BaseTokenizer):
    """TikToken分词器实现（用于OpenAI模型）"""
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        encoding_name: str = "cl100k_base",
    ):
        super().__init__(tokenizer_config)
        
        self.encoding_name = encoding_name
        
        # 加载tiktoken编码器
        self.encoding = self._load_tiktoken_encoding()
        
        # 设置特殊token
        self._setup_special_tokens()
        
        logger.info(f"Loaded TikToken tokenizer: {encoding_name}")
    
    def _load_tiktoken_encoding(self):
        """加载tiktoken编码器"""
        
        try:
            import tiktoken
            
            encoding = tiktoken.get_encoding(self.encoding_name)
            return encoding
            
        except ImportError:
            raise ImportError("tiktoken is required for TikTokenTokenizer")
        except Exception as e:
            logger.error(f"Failed to load tiktoken encoding: {e}")
            raise
    
    def _setup_special_tokens(self):
        """设置特殊token"""
        
        self.vocab_size = self.encoding.n_vocab
        
        # TikToken的特殊token（根据具体编码器设置）
        special_tokens = self.encoding.special_tokens_set
        
        # 尝试获取常见的特殊token
        try:
            self.eos_token_id = self.encoding.encode("<|endoftext|>")[0]
        except:
            self.eos_token_id = None
        
        # 其他特殊token需要根据具体模型设置
        self.pad_token_id = self.eos_token_id  # 通常使用EOS作为PAD
        self.bos_token_id = None
        self.unk_token_id = None
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[EncodingResult, List[EncodingResult]]:
        """编码文本"""
        
        start_time = time.time()
        
        if isinstance(text, str):
            # 单个文本编码
            token_ids = self.encoding.encode(text)
            
            # 截断
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            
            # 填充
            if padding and max_length is not None and self.pad_token_id is not None:
                if len(token_ids) < max_length:
                    pad_length = max_length - len(token_ids)
                    token_ids = token_ids + [self.pad_token_id] * pad_length
            
            # 创建attention mask
            attention_mask = [1] * len(token_ids)
            if padding and self.pad_token_id is not None:
                for i, token_id in enumerate(token_ids):
                    if token_id == self.pad_token_id:
                        attention_mask[i] = 0
            
            # 转换为tensor
            if return_tensors == "pt":
                input_ids = torch.tensor([token_ids])
                attention_mask = torch.tensor([attention_mask])
            else:
                input_ids = torch.tensor(token_ids)
                attention_mask = torch.tensor(attention_mask)
            
            encoding_time = time.time() - start_time
            
            result = EncodingResult(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoding_time=encoding_time,
                num_tokens=len(token_ids),
            )
            
            self._update_stats("encode", encoding_time, 1, len(token_ids))
            
            return result
        
        else:
            # 批量文本编码
            results = []
            
            for single_text in text:
                result = self.encode(
                    single_text,
                    add_special_tokens=add_special_tokens,
                    max_length=max_length,
                    padding=padding,
                    truncation=truncation,
                    return_tensors=None,
                )
                results.append(result)
            
            return results
    
    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[DecodingResult, List[DecodingResult]]:
        """解码token序列"""
        
        start_time = time.time()
        
        # 处理输入格式
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], int):  # 单个序列
            # 解码
            text = self.encoding.decode(token_ids)
            
            decoding_time = time.time() - start_time
            
            result = DecodingResult(
                text=text,
                token_ids=token_ids,
                decoding_time=decoding_time,
                skip_special_tokens=skip_special_tokens,
            )
            
            self._update_stats("decode", decoding_time, 1)
            
            return result
        
        else:  # 批量序列
            results = []
            
            for single_token_ids in token_ids:
                result = self.decode(
                    single_token_ids,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                results.append(result)
            
            return results
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.encoding.n_vocab

class TokenizerManager:
    """分词器管理器"""
    
    def __init__(self, tokenizer_config: TokenizerConfig):
        self.config = tokenizer_config
        self.tokenizers: Dict[str, BaseTokenizer] = {}
        self.default_tokenizer: Optional[BaseTokenizer] = None
        
        logger.info("Initialized TokenizerManager")
    
    def register_tokenizer(
        self,
        name: str,
        tokenizer: BaseTokenizer,
        set_as_default: bool = False,
    ):
        """注册分词器"""
        
        self.tokenizers[name] = tokenizer
        
        if set_as_default or self.default_tokenizer is None:
            self.default_tokenizer = tokenizer
        
        logger.info(f"Registered tokenizer: {name}")
    
    def get_tokenizer(self, name: Optional[str] = None) -> BaseTokenizer:
        """获取分词器"""
        
        if name is None:
            if self.default_tokenizer is None:
                raise ValueError("No default tokenizer set")
            return self.default_tokenizer
        
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer not found: {name}")
        
        return self.tokenizers[name]
    
    def create_huggingface_tokenizer(
        self,
        model_name_or_path: str,
        name: Optional[str] = None,
        set_as_default: bool = False,
    ) -> HuggingFaceTokenizer:
        """创建HuggingFace分词器"""
        
        tokenizer = HuggingFaceTokenizer(self.config, model_name_or_path)
        
        if name is None:
            name = f"hf_{model_name_or_path.replace('/', '_')}"
        
        self.register_tokenizer(name, tokenizer, set_as_default)
        
        return tokenizer
    
    def create_sentencepiece_tokenizer(
        self,
        model_path: str,
        name: Optional[str] = None,
        set_as_default: bool = False,
    ) -> SentencePieceTokenizer:
        """创建SentencePiece分词器"""
        
        tokenizer = SentencePieceTokenizer(self.config, model_path)
        
        if name is None:
            name = f"sp_{Path(model_path).stem}"
        
        self.register_tokenizer(name, tokenizer, set_as_default)
        
        return tokenizer
    
    def create_tiktoken_tokenizer(
        self,
        encoding_name: str = "cl100k_base",
        name: Optional[str] = None,
        set_as_default: bool = False,
    ) -> TikTokenTokenizer:
        """创建TikToken分词器"""
        
        tokenizer = TikTokenTokenizer(self.config, encoding_name)
        
        if name is None:
            name = f"tiktoken_{encoding_name}"
        
        self.register_tokenizer(name, tokenizer, set_as_default)
        
        return tokenizer
    
    def encode_with_fallback(
        self,
        text: Union[str, List[str]],
        tokenizer_names: Optional[List[str]] = None,
        **kwargs
    ) -> Union[EncodingResult, List[EncodingResult]]:
        """带回退的编码"""
        
        if tokenizer_names is None:
            tokenizer_names = list(self.tokenizers.keys())
        
        last_error = None
        
        for tokenizer_name in tokenizer_names:
            try:
                tokenizer = self.get_tokenizer(tokenizer_name)
                return tokenizer.encode(text, **kwargs)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Tokenizer {tokenizer_name} failed: {e}")
                continue
        
        raise RuntimeError(f"All tokenizers failed. Last error: {last_error}")
    
    def get_combined_stats(self) -> Dict[str, Any]:
        """获取所有分词器的统计信息"""
        
        combined_stats = {
            "tokenizers": {},
            "total_tokens_processed": 0,
            "total_texts_processed": 0,
            "total_encoding_time": 0.0,
            "total_decoding_time": 0.0,
        }
        
        for name, tokenizer in self.tokenizers.items():
            stats = tokenizer.get_stats()
            combined_stats["tokenizers"][name] = stats
            
            combined_stats["total_tokens_processed"] += stats["total_tokens_processed"]
            combined_stats["total_texts_processed"] += stats["total_texts_processed"]
            combined_stats["total_encoding_time"] += stats["encoding_time"]
            combined_stats["total_decoding_time"] += stats["decoding_time"]
        
        return combined_stats
    
    def clear_all_caches(self):
        """清空所有分词器的缓存"""
        
        for tokenizer in self.tokenizers.values():
            tokenizer.clear_cache()
        
        logger.info("Cleared all tokenizer caches")

# 性能优化组件
class FastTokenizer:
    """高性能分词器包装器"""
    
    def __init__(
        self,
        base_tokenizer: BaseTokenizer,
        num_workers: int = 4,
    ):
        self.base_tokenizer = base_tokenizer
        self.num_workers = num_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        
        logger.info(f"Initialized FastTokenizer with {num_workers} workers")
    
    def encode_parallel(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[EncodingResult]:
        """并行编码"""
        
        # 分批处理
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        
        # 提交任务
        futures = []
        for batch in batches:
            future = self.thread_pool.submit(
                self.base_tokenizer.encode,
                batch,
                **kwargs
            )
            futures.append(future)
        
        # 收集结果
        results = []
        for future in as_completed(futures):
            batch_results = future.result()
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def decode_parallel(
        self,
        token_ids_list: List[List[int]],
        batch_size: int = 32,
        **kwargs
    ) -> List[DecodingResult]:
        """并行解码"""
        
        # 分批处理
        batches = [
            token_ids_list[i:i + batch_size]
            for i in range(0, len(token_ids_list), batch_size)
        ]
        
        # 提交任务
        futures = []
        for batch in batches:
            future = self.thread_pool.submit(
                self.base_tokenizer.decode,
                batch,
                **kwargs
            )
            futures.append(future)
        
        # 收集结果
        results = []
        for future in as_completed(futures):
            batch_results = future.result()
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
        
        return results
    
    def shutdown(self):
        """关闭线程池"""
        self.thread_pool.shutdown(wait=True)

# 使用示例
def example_tokenizer_usage():
    """分词器使用示例"""
    
    # 配置
    tokenizer_config = TokenizerConfig(
        use_fast=True,
        trust_remote_code=False,
        max_cache_size=10000,
        return_offsets=True,
    )
    
    # 创建分词器管理器
    tokenizer_manager = TokenizerManager(tokenizer_config)
    
    # 创建HuggingFace分词器
    hf_tokenizer = tokenizer_manager.create_huggingface_tokenizer(
        model_name_or_path="microsoft/DialoGPT-medium",
        name="dialogpt",
        set_as_default=True,
    )
    
    # 编码示例
    text = "Hello, how are you today?"
    
    # 单个文本编码
    result = hf_tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    print(f"Input IDs: {result.input_ids}")
    print(f"Attention Mask: {result.attention_mask}")
    print(f"Encoding time: {result.encoding_time:.4f}s")
    
    # 解码示例
    decoded_result = hf_tokenizer.decode(
        result.input_ids,
        skip_special_tokens=True,
    )
    
    print(f"Decoded text: {decoded_result.text}")
    print(f"Decoding time: {decoded_result.decoding_time:.4f}s")
    
    # 批量处理示例
    texts = [
        "Hello, world!",
        "How are you?",
        "Nice to meet you.",
    ]
    
    batch_results = hf_tokenizer.encode_batch(texts, batch_size=2)
    print(f"Batch encoded {len(batch_results)} texts")
    
    # 获取统计信息
    stats = hf_tokenizer.get_stats()
    print(f"Tokenizer stats: {stats}")
    
    # 高性能分词器示例
    fast_tokenizer = FastTokenizer(hf_tokenizer, num_workers=4)
    
    large_texts = ["Sample text"] * 1000
    parallel_results = fast_tokenizer.encode_parallel(
        large_texts,
        batch_size=50,
    )
    
    print(f"Parallel encoded {len(parallel_results)} texts")
    
    # 清理
    fast_tokenizer.shutdown()

if __name__ == "__main__":
    example_tokenizer_usage()
```

## 🔧 关键特性分析

### 1. 多种分词器支持

- **HuggingFace分词器**：支持transformers库的所有分词器
- **SentencePiece分词器**：支持Google的SentencePiece模型
- **TikToken分词器**：支持OpenAI的tiktoken编码器
- **自定义分词器**：支持扩展自定义分词器

### 2. 高性能优化

- **缓存机制**：编码和解码结果缓存
- **并行处理**：多线程并行编码/解码
- **批量处理**：高效的批量操作
- **流式处理**：支持流式编码/解码

### 3. 灵活的配置

- **特殊token处理**：自动处理PAD、EOS、BOS等特殊token
- **长度控制**：支持截断和填充
- **格式转换**：支持多种输出格式
- **错误恢复**：带回退机制的编码

## 📊 性能优化技术

### 预编译正则表达式

```python
class OptimizedTokenizer:
    """优化的分词器"""
    
    def __init__(self, base_tokenizer: BaseTokenizer):
        self.base_tokenizer = base_tokenizer
        
        # 预编译常用正则表达式
        self.whitespace_pattern = re.compile(r'\s+')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.number_pattern = re.compile(r'\d+')
        
        # 预处理缓存
        self.preprocess_cache = {}
    
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        
        if text in self.preprocess_cache:
            return self.preprocess_cache[text]
        
        # 标准化空白字符
        processed = self.whitespace_pattern.sub(' ', text)
        
        # 处理标点符号
        processed = processed.strip()
        
        # 缓存结果
        if len(self.preprocess_cache) < 10000:
            self.preprocess_cache[text] = processed
        
        return processed
```

### 内存池优化

```python
class MemoryEfficientTokenizer:
    """内存高效的分词器"""
    
    def __init__(self, base_tokenizer: BaseTokenizer):
        self.base_tokenizer = base_tokenizer
        
        # 预分配tensor池
        self.tensor_pool = {
            'small': [],   # < 128 tokens
            'medium': [],  # 128-512 tokens
            'large': [],   # > 512 tokens
        }
        
        self.pool_lock = threading.Lock()
    
    def get_tensor_from_pool(self, size: int) -> Optional[torch.Tensor]:
        """从池中获取tensor"""
        
        with self.pool_lock:
            if size <= 128 and self.tensor_pool['small']:
                return self.tensor_pool['small'].pop()
            elif size <= 512 and self.tensor_pool['medium']:
                return self.tensor_pool['medium'].pop()
            elif self.tensor_pool['large']:
                return self.tensor_pool['large'].pop()
        
        return None
    
    def return_tensor_to_pool(self, tensor: torch.Tensor):
        """将tensor返回到池中"""
        
        size = tensor.numel()
        
        with self.pool_lock:
            if size <= 128 and len(self.tensor_pool['small']) < 100:
                tensor.zero_()  # 清零
                self.tensor_pool['small'].append(tensor)
            elif size <= 512 and len(self.tensor_pool['medium']) < 50:
                tensor.zero_()
                self.tensor_pool['medium'].append(tensor)
            elif len(self.tensor_pool['large']) < 20:
                tensor.zero_()
                self.tensor_pool['large'].append(tensor)
```

### 自适应批处理

```python
class AdaptiveBatchTokenizer:
    """自适应批处理分词器"""
    
    def __init__(self, base_tokenizer: BaseTokenizer):
        self.base_tokenizer = base_tokenizer
        
        # 性能统计
        self.batch_performance = {}
        self.optimal_batch_size = 32
    
    def encode_adaptive_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[EncodingResult]:
        """自适应批处理编码"""
        
        # 根据文本长度分组
        text_groups = self._group_texts_by_length(texts)
        
        results = []
        
        for length_range, group_texts in text_groups.items():
            # 选择最优批大小
            batch_size = self._get_optimal_batch_size(length_range)
            
            # 批处理编码
            group_results = self.base_tokenizer.encode_batch(
                group_texts,
                batch_size=batch_size,
                **kwargs
            )
            
            results.extend(group_results)
        
        return results
    
    def _group_texts_by_length(self, texts: List[str]) -> Dict[str, List[str]]:
        """按长度分组文本"""
        
        groups = {
            'short': [],   # < 50 chars
            'medium': [],  # 50-200 chars
            'long': [],    # > 200 chars
        }
        
        for text in texts:
            if len(text) < 50:
                groups['short'].append(text)
            elif len(text) < 200:
                groups['medium'].append(text)
            else:
                groups['long'].append(text)
        
        return groups
    
    def _get_optimal_batch_size(self, length_range: str) -> int:
        """获取最优批大小"""
        
        # 根据历史性能数据选择批大小
        if length_range in self.batch_performance:
            performance_data = self.batch_performance[length_range]
            best_batch_size = min(performance_data.keys(), 
                                key=lambda k: performance_data[k])
            return best_batch_size
        
        # 默认批大小
        default_sizes = {
            'short': 64,
            'medium': 32,
            'long': 16,
        }
        
        return default_sizes.get(length_range, 32)
```

## 🚀 使用最佳实践

### 1. 分词器选择指南

```python
def choose_tokenizer(
    model_type: str,
    performance_priority: str,
    memory_constraint: bool,
) -> str:
    """选择最适合的分词器"""
    
    if model_type.startswith("gpt"):
        return "tiktoken"
    elif model_type in ["llama", "alpaca", "vicuna"]:
        return "sentencepiece"
    elif performance_priority == "speed" and not memory_constraint:
        return "huggingface_fast"
    else:
        return "huggingface"
```

### 2. 缓存策略优化

```python
class SmartCacheTokenizer:
    """智能缓存分词器"""
    
    def __init__(self, base_tokenizer: BaseTokenizer):
        self.base_tokenizer = base_tokenizer
        
        # 多级缓存
        self.l1_cache = {}  # 热点数据
        self.l2_cache = {}  # 常用数据
        
        # 缓存统计
        self.access_count = {}
        self.cache_size_limit = {
            'l1': 1000,
            'l2': 5000,
        }
    
    def encode_with_smart_cache(self, text: str, **kwargs) -> EncodingResult:
        """智能缓存编码"""
        
        cache_key = self._generate_cache_key(text, kwargs)
        
        # 检查L1缓存
        if cache_key in self.l1_cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.l1_cache[cache_key]
        
        # 检查L2缓存
        if cache_key in self.l2_cache:
            result = self.l2_cache[cache_key]
            
            # 提升到L1缓存
            self._promote_to_l1(cache_key, result)
            
            return result
        
        # 执行编码
        result = self.base_tokenizer.encode(text, **kwargs)
        
        # 添加到缓存
        self._add_to_cache(cache_key, result)
        
        return result
    
    def _promote_to_l1(self, key: str, value: EncodingResult):
        """提升到L1缓存"""
        
        # 如果L1缓存满了，移除最少使用的项
        if len(self.l1_cache) >= self.cache_size_limit['l1']:
            lru_key = min(self.access_count.keys(), 
                         key=lambda k: self.access_count.get(k, 0))
            
            # 降级到L2缓存
            if lru_key in self.l1_cache:
                self.l2_cache[lru_key] = self.l1_cache.pop(lru_key)
        
        self.l1_cache[key] = value
        self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def _add_to_cache(self, key: str, value: EncodingResult):
        """添加到缓存"""
        
        # 新项目直接添加到L2缓存
        if len(self.l2_cache) >= self.cache_size_limit['l2']:
            # 移除最旧的项
            oldest_key = next(iter(self.l2_cache))
            del self.l2_cache[oldest_key]
        
        self.l2_cache[key] = value
        self.access_count[key] = 1
    
    def _generate_cache_key(self, text: str, kwargs: Dict[str, Any]) -> str:
        """生成缓存键"""
        
        import hashlib
        
        # 组合文本和参数
        key_data = f"{text}_{json.dumps(kwargs, sort_keys=True)}"
        
        # 生成哈希
        return hashlib.md5(key_data.encode()).hexdigest()
```

### 3. 错误处理和监控

```python
class RobustTokenizer:
    """健壮的分词器"""
    
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager
        
        # 错误统计
        self.error_stats = {
            'encoding_errors': 0,
            'decoding_errors': 0,
            'timeout_errors': 0,
            'memory_errors': 0,
        }
        
        # 监控指标
        self.performance_metrics = {
            'avg_encoding_time': 0.0,
            'avg_decoding_time': 0.0,
            'throughput': 0.0,
        }
    
    def safe_encode(
        self,
        text: str,
        timeout: float = 30.0,
        **kwargs
    ) -> Optional[EncodingResult]:
        """安全编码"""
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Encoding timeout")
        
        # 设置超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            start_time = time.time()
            
            # 尝试编码
            result = self.tokenizer_manager.encode_with_fallback(
                text,
                **kwargs
            )
            
            encoding_time = time.time() - start_time
            
            # 更新性能指标
            self._update_performance_metrics('encode', encoding_time)
            
            return result
            
        except TimeoutError:
            self.error_stats['timeout_errors'] += 1
            logger.error(f"Encoding timeout for text: {text[:100]}...")
            return None
            
        except MemoryError:
            self.error_stats['memory_errors'] += 1
            logger.error(f"Memory error during encoding: {text[:100]}...")
            return None
            
        except Exception as e:
            self.error_stats['encoding_errors'] += 1
            logger.error(f"Encoding error: {e}")
            return None
            
        finally:
            signal.alarm(0)  # 取消超时
    
    def _update_performance_metrics(self, operation: str, processing_time: float):
        """更新性能指标"""
        
        if operation == 'encode':
            # 使用指数移动平均
            alpha = 0.1
            self.performance_metrics['avg_encoding_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.performance_metrics['avg_encoding_time']
            )
        
        # 计算吞吐量
        self.performance_metrics['throughput'] = (
            1.0 / max(self.performance_metrics['avg_encoding_time'], 0.001)
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        
        total_errors = sum(self.error_stats.values())
        total_requests = total_errors + 1000  # 假设的总请求数
        
        error_rate = total_errors / total_requests
        
        return {
            'error_rate': error_rate,
            'error_stats': self.error_stats,
            'performance_metrics': self.performance_metrics,
            'health_score': max(0, 1 - error_rate * 10),  # 健康评分
        }
```

---

*分词器是文本处理的入口，高效的分词器实现能够显著提升整个推理系统的性能。*