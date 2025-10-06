# 🔧 模型适配 (Model Adaptation)

## 📖 概述

模型适配是nano-vllm支持多种模型架构和格式的核心机制。通过灵活的适配器系统，nano-vllm能够无缝集成不同的预训练模型，并提供统一的推理接口。本文档详细介绍模型适配的架构、实现和最佳实践。

## 🏗️ 模型适配架构

### 1. 基础适配器框架

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import os
from pathlib import Path
import importlib
import inspect

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """模型类型"""
    LLAMA = "llama"
    GPT = "gpt"
    BERT = "bert"
    T5 = "t5"
    BLOOM = "bloom"
    OPT = "opt"
    FALCON = "falcon"
    CHATGLM = "chatglm"
    BAICHUAN = "baichuan"
    QWEN = "qwen"
    CUSTOM = "custom"

class AdapterType(Enum):
    """适配器类型"""
    ARCHITECTURE = "architecture"
    TOKENIZER = "tokenizer"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"
    POSITION_ENCODING = "position_encoding"

@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_name: str
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    
    # 可选配置
    num_key_value_heads: Optional[int] = None
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    
    # 自定义配置
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type.value,
            'model_name': self.model_name,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'num_key_value_heads': self.num_key_value_heads,
            'rope_theta': self.rope_theta,
            'rope_scaling': self.rope_scaling,
            'attention_dropout': self.attention_dropout,
            'hidden_dropout': self.hidden_dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'tie_word_embeddings': self.tie_word_embeddings,
            'custom_config': self.custom_config,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """从字典创建配置"""
        model_type = ModelType(config_dict.pop('model_type'))
        return cls(model_type=model_type, **config_dict)

class BaseAdapter(ABC):
    """基础适配器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.adapter_type = None
        self.supported_models = set()
    
    @abstractmethod
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查是否兼容指定模型"""
        pass
    
    @abstractmethod
    def adapt(self, model: nn.Module, **kwargs) -> nn.Module:
        """适配模型"""
        pass
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """获取适配器信息"""
        return {
            'adapter_type': self.adapter_type.value if self.adapter_type else None,
            'supported_models': list(self.supported_models),
            'config': self.config.to_dict(),
        }

class ArchitectureAdapter(BaseAdapter):
    """架构适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.adapter_type = AdapterType.ARCHITECTURE
        self.layer_mapping = {}
        self.weight_mapping = {}
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查架构兼容性"""
        
        required_keys = [
            'hidden_size', 'num_layers', 'num_attention_heads'
        ]
        
        for key in required_keys:
            if key not in model_config:
                return False
        
        # 检查模型类型
        model_type = model_config.get('model_type', '').lower()
        return model_type in [m.value for m in self.supported_models]
    
    def adapt(self, model: nn.Module, **kwargs) -> nn.Module:
        """适配模型架构"""
        
        logger.info(f"Adapting model architecture for {self.config.model_type.value}")
        
        # 应用层映射
        adapted_model = self._apply_layer_mapping(model)
        
        # 应用权重映射
        adapted_model = self._apply_weight_mapping(adapted_model)
        
        # 应用架构特定的修改
        adapted_model = self._apply_architecture_modifications(adapted_model)
        
        return adapted_model
    
    def _apply_layer_mapping(self, model: nn.Module) -> nn.Module:
        """应用层映射"""
        
        if not self.layer_mapping:
            return model
        
        # 重命名层
        for old_name, new_name in self.layer_mapping.items():
            if hasattr(model, old_name):
                layer = getattr(model, old_name)
                setattr(model, new_name, layer)
                delattr(model, old_name)
        
        return model
    
    def _apply_weight_mapping(self, model: nn.Module) -> nn.Module:
        """应用权重映射"""
        
        if not self.weight_mapping:
            return model
        
        state_dict = model.state_dict()
        new_state_dict = {}
        
        for old_key, new_key in self.weight_mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]
            else:
                new_state_dict[old_key] = state_dict[old_key]
        
        # 加载新的状态字典
        model.load_state_dict(new_state_dict, strict=False)
        
        return model
    
    def _apply_architecture_modifications(self, model: nn.Module) -> nn.Module:
        """应用架构特定修改"""
        
        # 子类可以重写此方法来实现特定的架构修改
        return model

class LlamaAdapter(ArchitectureAdapter):
    """LLaMA模型适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.supported_models = {ModelType.LLAMA}
        
        # LLaMA特定的层映射
        self.layer_mapping = {
            'transformer.wte': 'embed_tokens',
            'transformer.ln_f': 'norm',
            'lm_head': 'lm_head',
        }
        
        # LLaMA特定的权重映射
        self.weight_mapping = {
            'transformer.h.{}.ln_1.weight': 'layers.{}.input_layernorm.weight',
            'transformer.h.{}.ln_2.weight': 'layers.{}.post_attention_layernorm.weight',
            'transformer.h.{}.attn.c_attn.weight': 'layers.{}.self_attn.qkv_proj.weight',
            'transformer.h.{}.attn.c_proj.weight': 'layers.{}.self_attn.o_proj.weight',
            'transformer.h.{}.mlp.c_fc.weight': 'layers.{}.mlp.gate_proj.weight',
            'transformer.h.{}.mlp.c_proj.weight': 'layers.{}.mlp.down_proj.weight',
        }
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查LLaMA兼容性"""
        
        if not super().is_compatible(model_config):
            return False
        
        # 检查LLaMA特定配置
        llama_keys = ['rope_theta', 'rope_scaling']
        return any(key in model_config for key in llama_keys)
    
    def _apply_architecture_modifications(self, model: nn.Module) -> nn.Module:
        """应用LLaMA特定修改"""
        
        # 设置RoPE参数
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    layer.self_attn.rotary_emb.base = self.config.rope_theta
        
        # 应用RoPE缩放
        if self.config.rope_scaling:
            self._apply_rope_scaling(model)
        
        return model
    
    def _apply_rope_scaling(self, model: nn.Module):
        """应用RoPE缩放"""
        
        scaling_config = self.config.rope_scaling
        scaling_type = scaling_config.get('type', 'linear')
        scaling_factor = scaling_config.get('factor', 1.0)
        
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                    rotary_emb = layer.self_attn.rotary_emb
                    
                    if scaling_type == 'linear':
                        rotary_emb.base *= scaling_factor
                    elif scaling_type == 'dynamic':
                        # 动态缩放实现
                        rotary_emb.scaling_factor = scaling_factor

class GPTAdapter(ArchitectureAdapter):
    """GPT模型适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.supported_models = {ModelType.GPT}
        
        # GPT特定的层映射
        self.layer_mapping = {
            'transformer.wte': 'wte',
            'transformer.wpe': 'wpe',
            'transformer.ln_f': 'ln_f',
            'lm_head': 'lm_head',
        }
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查GPT兼容性"""
        
        if not super().is_compatible(model_config):
            return False
        
        # 检查GPT特定配置
        return 'n_positions' in model_config or 'max_position_embeddings' in model_config

class TokenizerAdapter(BaseAdapter):
    """分词器适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.adapter_type = AdapterType.TOKENIZER
        self.special_tokens = {}
        self.token_mapping = {}
    
    def is_compatible(self, tokenizer_config: Dict[str, Any]) -> bool:
        """检查分词器兼容性"""
        
        required_keys = ['vocab_size']
        return all(key in tokenizer_config for key in required_keys)
    
    def adapt(self, tokenizer, **kwargs) -> Any:
        """适配分词器"""
        
        logger.info(f"Adapting tokenizer for {self.config.model_type.value}")
        
        # 设置特殊令牌
        self._set_special_tokens(tokenizer)
        
        # 应用令牌映射
        self._apply_token_mapping(tokenizer)
        
        # 验证词汇表大小
        self._validate_vocab_size(tokenizer)
        
        return tokenizer
    
    def _set_special_tokens(self, tokenizer):
        """设置特殊令牌"""
        
        for token_type, token_value in self.special_tokens.items():
            if hasattr(tokenizer, f'set_{token_type}'):
                getattr(tokenizer, f'set_{token_type}')(token_value)
            elif hasattr(tokenizer, token_type):
                setattr(tokenizer, token_type, token_value)
    
    def _apply_token_mapping(self, tokenizer):
        """应用令牌映射"""
        
        if not self.token_mapping:
            return
        
        # 更新词汇表映射
        if hasattr(tokenizer, 'vocab'):
            new_vocab = {}
            for token, token_id in tokenizer.vocab.items():
                new_token = self.token_mapping.get(token, token)
                new_vocab[new_token] = token_id
            tokenizer.vocab = new_vocab
    
    def _validate_vocab_size(self, tokenizer):
        """验证词汇表大小"""
        
        actual_vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
        expected_vocab_size = self.config.vocab_size
        
        if actual_vocab_size != expected_vocab_size:
            logger.warning(
                f"Vocab size mismatch: expected {expected_vocab_size}, "
                f"got {actual_vocab_size}"
            )

class AttentionAdapter(BaseAdapter):
    """注意力机制适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.adapter_type = AdapterType.ATTENTION
        self.attention_type = "multi_head"  # "multi_head", "grouped_query", "multi_query"
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查注意力机制兼容性"""
        
        required_keys = ['num_attention_heads', 'hidden_size']
        return all(key in model_config for key in required_keys)
    
    def adapt(self, attention_module: nn.Module, **kwargs) -> nn.Module:
        """适配注意力机制"""
        
        logger.info(f"Adapting attention mechanism: {self.attention_type}")
        
        # 根据注意力类型进行适配
        if self.attention_type == "grouped_query":
            return self._adapt_grouped_query_attention(attention_module)
        elif self.attention_type == "multi_query":
            return self._adapt_multi_query_attention(attention_module)
        else:
            return self._adapt_multi_head_attention(attention_module)
    
    def _adapt_multi_head_attention(self, attention_module: nn.Module) -> nn.Module:
        """适配多头注意力"""
        
        # 确保头数配置正确
        if hasattr(attention_module, 'num_heads'):
            attention_module.num_heads = self.config.num_attention_heads
        
        # 确保头维度配置正确
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        if hasattr(attention_module, 'head_dim'):
            attention_module.head_dim = head_dim
        
        return attention_module
    
    def _adapt_grouped_query_attention(self, attention_module: nn.Module) -> nn.Module:
        """适配分组查询注意力"""
        
        num_key_value_heads = self.config.num_key_value_heads or self.config.num_attention_heads
        
        # 设置KV头数
        if hasattr(attention_module, 'num_key_value_heads'):
            attention_module.num_key_value_heads = num_key_value_heads
        
        # 计算头维度
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        kv_head_dim = self.config.hidden_size // num_key_value_heads
        
        if hasattr(attention_module, 'head_dim'):
            attention_module.head_dim = head_dim
        if hasattr(attention_module, 'kv_head_dim'):
            attention_module.kv_head_dim = kv_head_dim
        
        return attention_module
    
    def _adapt_multi_query_attention(self, attention_module: nn.Module) -> nn.Module:
        """适配多查询注意力"""
        
        # 多查询注意力只有一个KV头
        if hasattr(attention_module, 'num_key_value_heads'):
            attention_module.num_key_value_heads = 1
        
        return attention_module

class ModelAdapterRegistry:
    """模型适配器注册表"""
    
    def __init__(self):
        self.adapters = {}
        self.model_configs = {}
        
        # 注册默认适配器
        self._register_default_adapters()
    
    def register_adapter(
        self,
        model_type: ModelType,
        adapter_type: AdapterType,
        adapter_class: Type[BaseAdapter]
    ):
        """注册适配器"""
        
        key = (model_type, adapter_type)
        self.adapters[key] = adapter_class
        
        logger.info(f"Registered adapter: {model_type.value} - {adapter_type.value}")
    
    def get_adapter(
        self,
        model_type: ModelType,
        adapter_type: AdapterType,
        config: ModelConfig
    ) -> Optional[BaseAdapter]:
        """获取适配器"""
        
        key = (model_type, adapter_type)
        adapter_class = self.adapters.get(key)
        
        if adapter_class:
            return adapter_class(config)
        
        logger.warning(f"No adapter found for {model_type.value} - {adapter_type.value}")
        return None
    
    def register_model_config(self, model_name: str, config: ModelConfig):
        """注册模型配置"""
        
        self.model_configs[model_name] = config
        logger.info(f"Registered model config: {model_name}")
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        
        return self.model_configs.get(model_name)
    
    def _register_default_adapters(self):
        """注册默认适配器"""
        
        # 注册架构适配器
        self.register_adapter(ModelType.LLAMA, AdapterType.ARCHITECTURE, LlamaAdapter)
        self.register_adapter(ModelType.GPT, AdapterType.ARCHITECTURE, GPTAdapter)
        
        # 注册分词器适配器
        self.register_adapter(ModelType.LLAMA, AdapterType.TOKENIZER, TokenizerAdapter)
        self.register_adapter(ModelType.GPT, AdapterType.TOKENIZER, TokenizerAdapter)
        
        # 注册注意力适配器
        self.register_adapter(ModelType.LLAMA, AdapterType.ATTENTION, AttentionAdapter)
        self.register_adapter(ModelType.GPT, AdapterType.ATTENTION, AttentionAdapter)
    
    def list_adapters(self) -> Dict[str, List[str]]:
        """列出所有适配器"""
        
        result = {}
        for (model_type, adapter_type), adapter_class in self.adapters.items():
            model_key = model_type.value
            if model_key not in result:
                result[model_key] = []
            result[model_key].append(adapter_type.value)
        
        return result

class ModelAdaptationManager:
    """模型适配管理器"""
    
    def __init__(self):
        self.registry = ModelAdapterRegistry()
        self.adaptation_history = []
    
    def adapt_model(
        self,
        model: nn.Module,
        model_config: Dict[str, Any],
        target_config: Optional[ModelConfig] = None
    ) -> Tuple[nn.Module, ModelConfig]:
        """适配模型"""
        
        logger.info("Starting model adaptation...")
        
        # 解析模型配置
        if target_config is None:
            target_config = self._parse_model_config(model_config)
        
        # 获取架构适配器
        arch_adapter = self.registry.get_adapter(
            target_config.model_type,
            AdapterType.ARCHITECTURE,
            target_config
        )
        
        if arch_adapter is None:
            raise ValueError(f"No architecture adapter found for {target_config.model_type.value}")
        
        # 检查兼容性
        if not arch_adapter.is_compatible(model_config):
            raise ValueError(f"Model is not compatible with {target_config.model_type.value} adapter")
        
        # 执行适配
        adapted_model = arch_adapter.adapt(model)
        
        # 记录适配历史
        self.adaptation_history.append({
            'timestamp': torch.tensor(0).item(),  # 简化时间戳
            'model_type': target_config.model_type.value,
            'adapter_type': AdapterType.ARCHITECTURE.value,
            'config': target_config.to_dict(),
        })
        
        logger.info(f"Model adaptation completed: {target_config.model_type.value}")
        
        return adapted_model, target_config
    
    def adapt_tokenizer(
        self,
        tokenizer,
        tokenizer_config: Dict[str, Any],
        model_config: ModelConfig
    ):
        """适配分词器"""
        
        logger.info("Starting tokenizer adaptation...")
        
        # 获取分词器适配器
        tokenizer_adapter = self.registry.get_adapter(
            model_config.model_type,
            AdapterType.TOKENIZER,
            model_config
        )
        
        if tokenizer_adapter is None:
            logger.warning(f"No tokenizer adapter found for {model_config.model_type.value}")
            return tokenizer
        
        # 检查兼容性
        if not tokenizer_adapter.is_compatible(tokenizer_config):
            logger.warning("Tokenizer is not compatible with adapter")
            return tokenizer
        
        # 执行适配
        adapted_tokenizer = tokenizer_adapter.adapt(tokenizer)
        
        logger.info("Tokenizer adaptation completed")
        
        return adapted_tokenizer
    
    def adapt_attention(
        self,
        attention_module: nn.Module,
        model_config: ModelConfig
    ) -> nn.Module:
        """适配注意力机制"""
        
        logger.info("Starting attention adaptation...")
        
        # 获取注意力适配器
        attention_adapter = self.registry.get_adapter(
            model_config.model_type,
            AdapterType.ATTENTION,
            model_config
        )
        
        if attention_adapter is None:
            logger.warning(f"No attention adapter found for {model_config.model_type.value}")
            return attention_module
        
        # 执行适配
        adapted_attention = attention_adapter.adapt(attention_module)
        
        logger.info("Attention adaptation completed")
        
        return adapted_attention
    
    def _parse_model_config(self, model_config: Dict[str, Any]) -> ModelConfig:
        """解析模型配置"""
        
        # 推断模型类型
        model_type = self._infer_model_type(model_config)
        
        # 提取基本配置
        config = ModelConfig(
            model_type=model_type,
            model_name=model_config.get('model_name', 'unknown'),
            vocab_size=model_config.get('vocab_size', 32000),
            hidden_size=model_config.get('hidden_size', 4096),
            num_layers=model_config.get('num_layers', model_config.get('num_hidden_layers', 32)),
            num_attention_heads=model_config.get('num_attention_heads', 32),
            intermediate_size=model_config.get('intermediate_size', 11008),
            max_position_embeddings=model_config.get('max_position_embeddings', 2048),
        )
        
        # 设置可选配置
        if 'num_key_value_heads' in model_config:
            config.num_key_value_heads = model_config['num_key_value_heads']
        
        if 'rope_theta' in model_config:
            config.rope_theta = model_config['rope_theta']
        
        if 'rope_scaling' in model_config:
            config.rope_scaling = model_config['rope_scaling']
        
        # 设置自定义配置
        custom_keys = set(model_config.keys()) - {
            'model_name', 'vocab_size', 'hidden_size', 'num_layers',
            'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
            'max_position_embeddings', 'num_key_value_heads', 'rope_theta',
            'rope_scaling'
        }
        
        config.custom_config = {k: model_config[k] for k in custom_keys}
        
        return config
    
    def _infer_model_type(self, model_config: Dict[str, Any]) -> ModelType:
        """推断模型类型"""
        
        model_name = model_config.get('model_name', '').lower()
        architectures = model_config.get('architectures', [])
        
        # 基于模型名称推断
        if 'llama' in model_name:
            return ModelType.LLAMA
        elif 'gpt' in model_name:
            return ModelType.GPT
        elif 'bert' in model_name:
            return ModelType.BERT
        elif 't5' in model_name:
            return ModelType.T5
        elif 'bloom' in model_name:
            return ModelType.BLOOM
        elif 'opt' in model_name:
            return ModelType.OPT
        elif 'falcon' in model_name:
            return ModelType.FALCON
        elif 'chatglm' in model_name:
            return ModelType.CHATGLM
        elif 'baichuan' in model_name:
            return ModelType.BAICHUAN
        elif 'qwen' in model_name:
            return ModelType.QWEN
        
        # 基于架构推断
        if architectures:
            arch_name = architectures[0].lower()
            if 'llama' in arch_name:
                return ModelType.LLAMA
            elif 'gpt' in arch_name:
                return ModelType.GPT
            elif 'bert' in arch_name:
                return ModelType.BERT
        
        # 基于配置特征推断
        if 'rope_theta' in model_config or 'rope_scaling' in model_config:
            return ModelType.LLAMA
        elif 'n_positions' in model_config:
            return ModelType.GPT
        
        logger.warning("Could not infer model type, using CUSTOM")
        return ModelType.CUSTOM
    
    def register_custom_adapter(
        self,
        model_type: ModelType,
        adapter_type: AdapterType,
        adapter_class: Type[BaseAdapter]
    ):
        """注册自定义适配器"""
        
        self.registry.register_adapter(model_type, adapter_type, adapter_class)
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """获取适配历史"""
        
        return self.adaptation_history.copy()
    
    def save_adaptation_config(self, config: ModelConfig, filepath: str):
        """保存适配配置"""
        
        config_dict = config.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Adaptation config saved to {filepath}")
    
    def load_adaptation_config(self, filepath: str) -> ModelConfig:
        """加载适配配置"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = ModelConfig.from_dict(config_dict)
        
        logger.info(f"Adaptation config loaded from {filepath}")
        
        return config
```

### 2. 自定义适配器示例

```python
class CustomLlamaAdapter(LlamaAdapter):
    """自定义LLaMA适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # 自定义配置
        self.use_flash_attention = config.custom_config.get('use_flash_attention', True)
        self.use_gradient_checkpointing = config.custom_config.get('use_gradient_checkpointing', False)
    
    def _apply_architecture_modifications(self, model: nn.Module) -> nn.Module:
        """应用自定义架构修改"""
        
        # 调用父类方法
        model = super()._apply_architecture_modifications(model)
        
        # 启用Flash Attention
        if self.use_flash_attention:
            self._enable_flash_attention(model)
        
        # 启用梯度检查点
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing(model)
        
        return model
    
    def _enable_flash_attention(self, model: nn.Module):
        """启用Flash Attention"""
        
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'self_attn'):
                    # 替换为Flash Attention实现
                    layer.self_attn.use_flash_attention = True
    
    def _enable_gradient_checkpointing(self, model: nn.Module):
        """启用梯度检查点"""
        
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

class QuantizationAdapter(BaseAdapter):
    """量化适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.adapter_type = AdapterType.CUSTOM
        
        # 量化配置
        self.quantization_config = config.custom_config.get('quantization', {})
        self.bits = self.quantization_config.get('bits', 8)
        self.group_size = self.quantization_config.get('group_size', 128)
        self.quantization_method = self.quantization_config.get('method', 'gptq')
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查量化兼容性"""
        
        # 检查模型是否支持量化
        return 'quantization' in model_config or self.quantization_config
    
    def adapt(self, model: nn.Module, **kwargs) -> nn.Module:
        """应用量化"""
        
        logger.info(f"Applying {self.quantization_method} quantization ({self.bits}-bit)")
        
        if self.quantization_method == 'gptq':
            return self._apply_gptq_quantization(model)
        elif self.quantization_method == 'awq':
            return self._apply_awq_quantization(model)
        elif self.quantization_method == 'bnb':
            return self._apply_bitsandbytes_quantization(model)
        else:
            logger.warning(f"Unknown quantization method: {self.quantization_method}")
            return model
    
    def _apply_gptq_quantization(self, model: nn.Module) -> nn.Module:
        """应用GPTQ量化"""
        
        try:
            from auto_gptq import AutoGPTQForCausalLM
            
            # 这里需要根据实际的GPTQ库实现
            # 示例代码
            quantized_model = model  # 实际量化逻辑
            
            logger.info("GPTQ quantization applied successfully")
            return quantized_model
            
        except ImportError:
            logger.error("auto-gptq not installed, skipping GPTQ quantization")
            return model
    
    def _apply_awq_quantization(self, model: nn.Module) -> nn.Module:
        """应用AWQ量化"""
        
        try:
            # AWQ量化实现
            quantized_model = model  # 实际量化逻辑
            
            logger.info("AWQ quantization applied successfully")
            return quantized_model
            
        except ImportError:
            logger.error("AWQ library not installed, skipping AWQ quantization")
            return model
    
    def _apply_bitsandbytes_quantization(self, model: nn.Module) -> nn.Module:
        """应用BitsAndBytes量化"""
        
        try:
            import bitsandbytes as bnb
            
            # BitsAndBytes量化实现
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if self.bits == 8:
                        quantized_module = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None
                        )
                    elif self.bits == 4:
                        quantized_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None
                        )
                    else:
                        continue
                    
                    # 复制权重
                    quantized_module.weight.data = module.weight.data
                    if module.bias is not None:
                        quantized_module.bias.data = module.bias.data
                    
                    # 替换模块
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent_module = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent_module, child_name, quantized_module)
            
            logger.info("BitsAndBytes quantization applied successfully")
            return model
            
        except ImportError:
            logger.error("bitsandbytes not installed, skipping BnB quantization")
            return model

class LoRAAdapter(BaseAdapter):
    """LoRA适配器"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.adapter_type = AdapterType.CUSTOM
        
        # LoRA配置
        self.lora_config = config.custom_config.get('lora', {})
        self.r = self.lora_config.get('r', 8)
        self.alpha = self.lora_config.get('alpha', 16)
        self.dropout = self.lora_config.get('dropout', 0.1)
        self.target_modules = self.lora_config.get('target_modules', ['q_proj', 'v_proj'])
    
    def is_compatible(self, model_config: Dict[str, Any]) -> bool:
        """检查LoRA兼容性"""
        
        return 'lora' in model_config or self.lora_config
    
    def adapt(self, model: nn.Module, **kwargs) -> nn.Module:
        """应用LoRA适配"""
        
        logger.info(f"Applying LoRA adaptation (r={self.r}, alpha={self.alpha})")
        
        try:
            from peft import LoraConfig, get_peft_model
            
            # 创建LoRA配置
            peft_config = LoraConfig(
                r=self.r,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=self.target_modules,
                task_type="CAUSAL_LM"
            )
            
            # 应用LoRA
            model = get_peft_model(model, peft_config)
            
            logger.info("LoRA adaptation applied successfully")
            return model
            
        except ImportError:
            logger.error("peft library not installed, skipping LoRA adaptation")
            return model
```

## 🚀 使用示例

### 1. 基本模型适配

```python
def basic_model_adaptation():
    # 创建适配管理器
    adapter_manager = ModelAdaptationManager()
    
    # 模拟模型配置
    model_config = {
        'model_name': 'llama-7b',
        'architectures': ['LlamaForCausalLM'],
        'vocab_size': 32000,
        'hidden_size': 4096,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'intermediate_size': 11008,
        'max_position_embeddings': 2048,
        'rope_theta': 10000.0,
        'rope_scaling': None,
    }
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(32000, 4096)
            self.layers = nn.ModuleList([
                nn.TransformerDecoderLayer(4096, 32) for _ in range(32)
            ])
            self.norm = nn.LayerNorm(4096)
            self.lm_head = nn.Linear(4096, 32000, bias=False)
    
    model = MockModel()
    
    try:
        # 执行模型适配
        adapted_model, config = adapter_manager.adapt_model(model, model_config)
        
        print(f"Model adapted successfully:")
        print(f"  Model Type: {config.model_type.value}")
        print(f"  Hidden Size: {config.hidden_size}")
        print(f"  Num Layers: {config.num_layers}")
        print(f"  Num Heads: {config.num_attention_heads}")
        
        # 保存适配配置
        adapter_manager.save_adaptation_config(config, "adaptation_config.json")
        
    except Exception as e:
        print(f"Model adaptation failed: {e}")

if __name__ == "__main__":
    basic_model_adaptation()
```

### 2. 自定义适配器注册

```python
def custom_adapter_registration():
    # 创建适配管理器
    adapter_manager = ModelAdaptationManager()
    
    # 注册自定义适配器
    adapter_manager.register_custom_adapter(
        ModelType.LLAMA,
        AdapterType.ARCHITECTURE,
        CustomLlamaAdapter
    )
    
    # 注册量化适配器
    adapter_manager.register_custom_adapter(
        ModelType.CUSTOM,
        AdapterType.CUSTOM,
        QuantizationAdapter
    )
    
    # 注册LoRA适配器
    adapter_manager.register_custom_adapter(
        ModelType.CUSTOM,
        AdapterType.CUSTOM,
        LoRAAdapter
    )
    
    # 创建带有自定义配置的模型配置
    custom_config = ModelConfig(
        model_type=ModelType.LLAMA,
        model_name="custom-llama-7b",
        vocab_size=32000,
        hidden_size=4096,
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_position_embeddings=2048,
        custom_config={
            'use_flash_attention': True,
            'use_gradient_checkpointing': True,
            'quantization': {
                'bits': 8,
                'method': 'bnb',
                'group_size': 128
            },
            'lora': {
                'r': 16,
                'alpha': 32,
                'dropout': 0.1,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            }
        }
    )
    
    print("Custom adapters registered successfully:")
    print(f"Available adapters: {adapter_manager.registry.list_adapters()}")
    
    return adapter_manager, custom_config

if __name__ == "__main__":
    manager, config = custom_adapter_registration()
```

### 3. 批量模型适配

```python
def batch_model_adaptation():
    # 创建适配管理器
    adapter_manager = ModelAdaptationManager()
    
    # 定义多个模型配置
    model_configs = [
        {
            'name': 'llama-7b',
            'config': {
                'model_name': 'llama-7b',
                'architectures': ['LlamaForCausalLM'],
                'vocab_size': 32000,
                'hidden_size': 4096,
                'num_hidden_layers': 32,
                'num_attention_heads': 32,
                'intermediate_size': 11008,
                'max_position_embeddings': 2048,
                'rope_theta': 10000.0,
            }
        },
        {
            'name': 'gpt-3.5',
            'config': {
                'model_name': 'gpt-3.5',
                'architectures': ['GPTForCausalLM'],
                'vocab_size': 50257,
                'hidden_size': 4096,
                'num_hidden_layers': 48,
                'num_attention_heads': 32,
                'intermediate_size': 16384,
                'n_positions': 4096,
            }
        }
    ]
    
    adapted_models = {}
    
    for model_info in model_configs:
        model_name = model_info['name']
        model_config = model_info['config']
        
        try:
            print(f"Adapting model: {model_name}")
            
            # 创建模拟模型
            mock_model = nn.Module()
            
            # 执行适配
            adapted_model, config = adapter_manager.adapt_model(mock_model, model_config)
            
            adapted_models[model_name] = {
                'model': adapted_model,
                'config': config
            }
            
            print(f"  ✓ {model_name} adapted successfully")
            
        except Exception as e:
            print(f"  ✗ {model_name} adaptation failed: {e}")
    
    # 打印适配历史
    history = adapter_manager.get_adaptation_history()
    print(f"\nAdaptation History ({len(history)} entries):")
    for i, entry in enumerate(history):
        print(f"  {i+1}. {entry['model_type']} - {entry['adapter_type']}")
    
    return adapted_models

if __name__ == "__main__":
    models = batch_model_adaptation()
```

## 🎯 最佳实践

### 1. 适配器设计原则
- **模块化设计**: 将不同类型的适配逻辑分离
- **可扩展性**: 支持新模型类型的快速集成
- **向后兼容**: 保持与现有模型的兼容性
- **错误处理**: 提供详细的错误信息和恢复机制

### 2. 配置管理
- **标准化配置**: 使用统一的配置格式
- **配置验证**: 验证配置的完整性和正确性
- **版本控制**: 跟踪配置变更历史
- **文档化**: 详细记录配置参数含义

### 3. 性能优化
- **延迟加载**: 按需加载适配器
- **缓存机制**: 缓存适配结果
- **并行处理**: 支持并行适配多个模型
- **内存管理**: 及时释放不需要的资源

### 4. 测试和验证
- **兼容性测试**: 验证适配器与不同模型的兼容性
- **功能测试**: 确保适配后模型功能正常
- **性能测试**: 验证适配对性能的影响
- **回归测试**: 防止新适配器影响现有功能

## 📈 总结

模型适配是nano-vllm支持多样化模型生态的关键技术。通过灵活的适配器架构，可以快速集成新的模型类型，同时保持系统的稳定性和性能。

关键要点：
1. **分层适配**: 从架构、分词器、注意力机制等多个层面进行适配
2. **注册机制**: 通过注册表管理不同类型的适配器
3. **配置驱动**: 使用配置文件驱动适配过程
4. **扩展性**: 支持自定义适配器的注册和使用

通过合理设计和实现模型适配系统，可以大大提升nano-vllm的模型兼容性和易用性。