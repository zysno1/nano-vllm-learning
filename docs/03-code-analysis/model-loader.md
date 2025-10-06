# 模型加载器分析

## 🎯 模型加载器概览

模型加载器是 nano-vllm 中负责加载、初始化和管理大语言模型的核心组件。它支持多种模型格式、权重分片、动态加载等功能，确保模型能够高效地加载到内存中并准备好进行推理。

## 🏗️ 核心架构

```python
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import pickle
import safetensors
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import mmap
import os

from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from nano_vllm.config import ModelConfig, LoaderConfig
from nano_vllm.utils.logger import get_logger
from nano_vllm.memory.manager import UnifiedMemoryManager

logger = get_logger(__name__)

class ModelFormat(Enum):
    """模型格式枚举"""
    HUGGINGFACE = "huggingface"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    CUSTOM = "custom"

class LoadingStrategy(Enum):
    """加载策略枚举"""
    EAGER = "eager"          # 立即加载所有权重
    LAZY = "lazy"            # 延迟加载权重
    STREAMING = "streaming"   # 流式加载
    MEMORY_MAPPED = "mmap"   # 内存映射加载

@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    format: ModelFormat
    size: int                    # 模型大小（字节）
    num_parameters: int          # 参数数量
    architecture: str            # 模型架构
    vocab_size: int             # 词汇表大小
    hidden_size: int            # 隐藏层大小
    num_layers: int             # 层数
    num_heads: int              # 注意力头数
    max_position_embeddings: int # 最大位置编码
    dtype: torch.dtype          # 数据类型
    device_map: Optional[Dict[str, str]] = None  # 设备映射
    shard_info: Optional[Dict[str, Any]] = None  # 分片信息
    checksum: Optional[str] = None               # 校验和
    created_at: float = field(default_factory=time.time)

@dataclass
class WeightShard:
    """权重分片"""
    name: str                   # 分片名称
    file_path: Path            # 文件路径
    offset: int                # 文件偏移
    size: int                  # 分片大小
    shape: Tuple[int, ...]     # 张量形状
    dtype: torch.dtype         # 数据类型
    device: torch.device       # 目标设备
    loaded: bool = False       # 是否已加载
    tensor: Optional[torch.Tensor] = None  # 加载的张量

class BaseModelLoader:
    """基础模型加载器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        memory_manager: UnifiedMemoryManager,
    ):
        self.model_config = model_config
        self.loader_config = loader_config
        self.memory_manager = memory_manager
        
        # 模型路径
        self.model_path = Path(model_config.model_path)
        
        # 加载状态
        self.loading_progress = 0.0
        self.is_loading = False
        self.load_start_time = 0.0
        
        # 权重管理
        self.weight_shards: Dict[str, WeightShard] = {}
        self.loaded_weights: Dict[str, torch.Tensor] = {}
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=loader_config.num_loading_threads
        )
        
        # 锁
        self.loading_lock = threading.RLock()
        
        logger.info(f"Initialized BaseModelLoader for {model_config.model_name}")
    
    def load_model(self) -> nn.Module:
        """加载模型"""
        
        with self.loading_lock:
            if self.is_loading:
                raise RuntimeError("Model is already being loaded")
            
            self.is_loading = True
            self.load_start_time = time.time()
            self.loading_progress = 0.0
        
        try:
            logger.info(f"Starting to load model: {self.model_config.model_name}")
            
            # 1. 加载模型元数据
            metadata = self._load_metadata()
            self.loading_progress = 0.1
            
            # 2. 创建模型架构
            model = self._create_model_architecture(metadata)
            self.loading_progress = 0.2
            
            # 3. 分析权重分片
            self._analyze_weight_shards(metadata)
            self.loading_progress = 0.3
            
            # 4. 加载权重
            self._load_weights(model, metadata)
            self.loading_progress = 0.8
            
            # 5. 后处理
            self._post_process_model(model, metadata)
            self.loading_progress = 1.0
            
            load_time = time.time() - self.load_start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            self.is_loading = False
    
    def _load_metadata(self) -> ModelMetadata:
        """加载模型元数据"""
        
        logger.info("Loading model metadata")
        
        # 检查配置文件
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 检测模型格式
        model_format = self._detect_model_format()
        
        # 计算模型大小
        model_size = self._calculate_model_size()
        
        # 创建元数据
        metadata = ModelMetadata(
            name=self.model_config.model_name,
            format=model_format,
            size=model_size,
            num_parameters=self._estimate_parameters(config_dict),
            architecture=config_dict.get("architectures", ["unknown"])[0],
            vocab_size=config_dict.get("vocab_size", 0),
            hidden_size=config_dict.get("hidden_size", 0),
            num_layers=config_dict.get("num_hidden_layers", 0),
            num_heads=config_dict.get("num_attention_heads", 0),
            max_position_embeddings=config_dict.get("max_position_embeddings", 0),
            dtype=getattr(torch, self.model_config.torch_dtype),
        )
        
        logger.info(f"Model metadata: {metadata.architecture}, "
                   f"{metadata.num_parameters:,} parameters, "
                   f"{metadata.size / 1024**3:.2f}GB")
        
        return metadata
    
    def _detect_model_format(self) -> ModelFormat:
        """检测模型格式"""
        
        # 检查safetensors文件
        if any(self.model_path.glob("*.safetensors")):
            return ModelFormat.SAFETENSORS
        
        # 检查PyTorch文件
        if any(self.model_path.glob("*.bin")) or any(self.model_path.glob("*.pth")):
            return ModelFormat.PYTORCH
        
        # 检查ONNX文件
        if any(self.model_path.glob("*.onnx")):
            return ModelFormat.ONNX
        
        # 默认为HuggingFace格式
        return ModelFormat.HUGGINGFACE
    
    def _calculate_model_size(self) -> int:
        """计算模型大小"""
        
        total_size = 0
        
        # 遍历所有权重文件
        for pattern in ["*.bin", "*.safetensors", "*.pth"]:
            for file_path in self.model_path.glob(pattern):
                total_size += file_path.stat().st_size
        
        return total_size
    
    def _estimate_parameters(self, config_dict: Dict[str, Any]) -> int:
        """估算参数数量"""
        
        # 从配置中获取参数数量
        if "num_parameters" in config_dict:
            return config_dict["num_parameters"]
        
        # 根据架构估算
        hidden_size = config_dict.get("hidden_size", 0)
        vocab_size = config_dict.get("vocab_size", 0)
        num_layers = config_dict.get("num_hidden_layers", 0)
        
        if hidden_size and vocab_size and num_layers:
            # 简化的参数估算
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (
                4 * hidden_size * hidden_size +  # 注意力权重
                8 * hidden_size * hidden_size    # MLP权重
            )
            
            return embedding_params + layer_params
        
        return 0
    
    def _create_model_architecture(self, metadata: ModelMetadata) -> nn.Module:
        """创建模型架构"""
        
        logger.info("Creating model architecture")
        
        if metadata.format == ModelFormat.HUGGINGFACE:
            # 使用transformers库创建模型
            config = AutoConfig.from_pretrained(
                self.model_path,
                torch_dtype=metadata.dtype,
                trust_remote_code=self.loader_config.trust_remote_code,
            )
            
            # 创建空模型（不加载权重）
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config)
            
        else:
            # 自定义模型架构
            model = self._create_custom_architecture(metadata)
        
        logger.info(f"Created model architecture: {type(model).__name__}")
        return model
    
    def _create_custom_architecture(self, metadata: ModelMetadata) -> nn.Module:
        """创建自定义模型架构"""
        
        # 这里实现自定义模型架构的创建
        # 根据metadata中的信息构建模型
        
        class CustomModel(nn.Module):
            def __init__(self, metadata: ModelMetadata):
                super().__init__()
                
                self.embed_tokens = nn.Embedding(
                    metadata.vocab_size,
                    metadata.hidden_size,
                    dtype=metadata.dtype,
                )
                
                self.layers = nn.ModuleList([
                    self._create_transformer_layer(metadata)
                    for _ in range(metadata.num_layers)
                ])
                
                self.norm = nn.LayerNorm(
                    metadata.hidden_size,
                    dtype=metadata.dtype,
                )
                
                self.lm_head = nn.Linear(
                    metadata.hidden_size,
                    metadata.vocab_size,
                    bias=False,
                    dtype=metadata.dtype,
                )
            
            def _create_transformer_layer(self, metadata: ModelMetadata):
                """创建Transformer层"""
                
                class TransformerLayer(nn.Module):
                    def __init__(self):
                        super().__init__()
                        
                        self.self_attn = nn.MultiheadAttention(
                            metadata.hidden_size,
                            metadata.num_heads,
                            dtype=metadata.dtype,
                        )
                        
                        self.mlp = nn.Sequential(
                            nn.Linear(
                                metadata.hidden_size,
                                metadata.hidden_size * 4,
                                dtype=metadata.dtype,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                metadata.hidden_size * 4,
                                metadata.hidden_size,
                                dtype=metadata.dtype,
                            ),
                        )
                        
                        self.input_layernorm = nn.LayerNorm(
                            metadata.hidden_size,
                            dtype=metadata.dtype,
                        )
                        
                        self.post_attention_layernorm = nn.LayerNorm(
                            metadata.hidden_size,
                            dtype=metadata.dtype,
                        )
                    
                    def forward(self, x):
                        # 简化的前向传播
                        residual = x
                        x = self.input_layernorm(x)
                        x, _ = self.self_attn(x, x, x)
                        x = residual + x
                        
                        residual = x
                        x = self.post_attention_layernorm(x)
                        x = self.mlp(x)
                        x = residual + x
                        
                        return x
                
                return TransformerLayer()
            
            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                
                for layer in self.layers:
                    x = layer(x)
                
                x = self.norm(x)
                logits = self.lm_head(x)
                
                return logits
        
        return CustomModel(metadata)
    
    def _analyze_weight_shards(self, metadata: ModelMetadata):
        """分析权重分片"""
        
        logger.info("Analyzing weight shards")
        
        if metadata.format == ModelFormat.SAFETENSORS:
            self._analyze_safetensors_shards()
        elif metadata.format == ModelFormat.PYTORCH:
            self._analyze_pytorch_shards()
        else:
            self._analyze_huggingface_shards()
        
        logger.info(f"Found {len(self.weight_shards)} weight shards")
    
    def _analyze_safetensors_shards(self):
        """分析safetensors分片"""
        
        for safetensors_file in self.model_path.glob("*.safetensors"):
            # 读取safetensors头部信息
            with open(safetensors_file, 'rb') as f:
                # 读取头部长度
                header_size = int.from_bytes(f.read(8), 'little')
                
                # 读取头部JSON
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # 分析每个张量
                for tensor_name, tensor_info in header.items():
                    if tensor_name == "__metadata__":
                        continue
                    
                    shard = WeightShard(
                        name=tensor_name,
                        file_path=safetensors_file,
                        offset=8 + header_size + tensor_info["data_offsets"][0],
                        size=tensor_info["data_offsets"][1] - tensor_info["data_offsets"][0],
                        shape=tuple(tensor_info["shape"]),
                        dtype=getattr(torch, tensor_info["dtype"]),
                        device=torch.device("cpu"),  # 初始在CPU
                    )
                    
                    self.weight_shards[tensor_name] = shard
    
    def _analyze_pytorch_shards(self):
        """分析PyTorch分片"""
        
        for bin_file in self.model_path.glob("*.bin"):
            # 加载权重文件的键信息（不加载实际数据）
            checkpoint = torch.load(bin_file, map_location="cpu", weights_only=True)
            
            for tensor_name, tensor in checkpoint.items():
                shard = WeightShard(
                    name=tensor_name,
                    file_path=bin_file,
                    offset=0,  # PyTorch文件不支持偏移
                    size=tensor.numel() * tensor.element_size(),
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    device=torch.device("cpu"),
                )
                
                self.weight_shards[tensor_name] = shard
    
    def _analyze_huggingface_shards(self):
        """分析HuggingFace分片"""
        
        # 检查分片索引文件
        index_file = self.model_path / "pytorch_model.bin.index.json"
        
        if index_file.exists():
            # 多文件分片
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            weight_map = index["weight_map"]
            
            for tensor_name, file_name in weight_map.items():
                file_path = self.model_path / file_name
                
                shard = WeightShard(
                    name=tensor_name,
                    file_path=file_path,
                    offset=0,
                    size=0,  # 需要实际加载才能知道大小
                    shape=(),
                    dtype=torch.float32,  # 默认类型
                    device=torch.device("cpu"),
                )
                
                self.weight_shards[tensor_name] = shard
        
        else:
            # 单文件
            single_file = self.model_path / "pytorch_model.bin"
            if single_file.exists():
                self._analyze_single_pytorch_file(single_file)
    
    def _analyze_single_pytorch_file(self, file_path: Path):
        """分析单个PyTorch文件"""
        
        # 只加载键信息
        checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
        
        for tensor_name, tensor in checkpoint.items():
            shard = WeightShard(
                name=tensor_name,
                file_path=file_path,
                offset=0,
                size=tensor.numel() * tensor.element_size(),
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=torch.device("cpu"),
            )
            
            self.weight_shards[tensor_name] = shard
    
    def _load_weights(self, model: nn.Module, metadata: ModelMetadata):
        """加载权重"""
        
        logger.info("Loading model weights")
        
        if self.loader_config.loading_strategy == LoadingStrategy.EAGER:
            self._load_weights_eager(model, metadata)
        elif self.loader_config.loading_strategy == LoadingStrategy.LAZY:
            self._load_weights_lazy(model, metadata)
        elif self.loader_config.loading_strategy == LoadingStrategy.STREAMING:
            self._load_weights_streaming(model, metadata)
        elif self.loader_config.loading_strategy == LoadingStrategy.MEMORY_MAPPED:
            self._load_weights_mmap(model, metadata)
        else:
            raise ValueError(f"Unknown loading strategy: {self.loader_config.loading_strategy}")
    
    def _load_weights_eager(self, model: nn.Module, metadata: ModelMetadata):
        """立即加载所有权重"""
        
        # 并行加载权重
        futures = []
        
        for shard_name, shard in self.weight_shards.items():
            future = self.thread_pool.submit(self._load_single_shard, shard)
            futures.append((shard_name, future))
        
        # 等待所有权重加载完成
        loaded_count = 0
        total_count = len(futures)
        
        for shard_name, future in futures:
            try:
                tensor = future.result(timeout=self.loader_config.loading_timeout)
                
                # 将权重分配到模型
                self._assign_weight_to_model(model, shard_name, tensor)
                
                loaded_count += 1
                self.loading_progress = 0.3 + 0.5 * (loaded_count / total_count)
                
                if loaded_count % 100 == 0:
                    logger.info(f"Loaded {loaded_count}/{total_count} weight shards")
                
            except Exception as e:
                logger.error(f"Failed to load shard {shard_name}: {e}")
                raise
        
        logger.info(f"All {total_count} weight shards loaded successfully")
    
    def _load_weights_lazy(self, model: nn.Module, metadata: ModelMetadata):
        """延迟加载权重"""
        
        # 创建延迟加载的权重占位符
        for shard_name, shard in self.weight_shards.items():
            # 创建空张量作为占位符
            placeholder = torch.empty(
                shard.shape,
                dtype=shard.dtype,
                device="meta",  # meta设备不占用实际内存
            )
            
            # 添加延迟加载钩子
            placeholder.register_hook(
                lambda grad, name=shard_name: self._lazy_load_hook(name, grad)
            )
            
            self._assign_weight_to_model(model, shard_name, placeholder)
        
        logger.info("Lazy loading setup complete")
    
    def _load_weights_streaming(self, model: nn.Module, metadata: ModelMetadata):
        """流式加载权重"""
        
        # 按层顺序流式加载
        layer_weights = self._group_weights_by_layer()
        
        for layer_idx, weights in layer_weights.items():
            logger.info(f"Loading layer {layer_idx} weights")
            
            # 加载当前层的所有权重
            for shard_name, shard in weights.items():
                tensor = self._load_single_shard(shard)
                self._assign_weight_to_model(model, shard_name, tensor)
            
            # 更新进度
            progress = 0.3 + 0.5 * (layer_idx + 1) / len(layer_weights)
            self.loading_progress = min(progress, 0.8)
        
        logger.info("Streaming loading complete")
    
    def _load_weights_mmap(self, model: nn.Module, metadata: ModelMetadata):
        """内存映射加载权重"""
        
        # 为每个权重文件创建内存映射
        mmap_files = {}
        
        for shard in self.weight_shards.values():
            file_path = str(shard.file_path)
            
            if file_path not in mmap_files:
                with open(file_path, 'rb') as f:
                    mmap_files[file_path] = mmap.mmap(
                        f.fileno(),
                        0,
                        access=mmap.ACCESS_READ
                    )
        
        # 使用内存映射加载权重
        for shard_name, shard in self.weight_shards.items():
            file_path = str(shard.file_path)
            mmap_file = mmap_files[file_path]
            
            # 从内存映射读取数据
            mmap_file.seek(shard.offset)
            data = mmap_file.read(shard.size)
            
            # 创建张量
            tensor = torch.frombuffer(
                data,
                dtype=shard.dtype
            ).reshape(shard.shape).clone()
            
            self._assign_weight_to_model(model, shard_name, tensor)
        
        # 清理内存映射
        for mmap_file in mmap_files.values():
            mmap_file.close()
        
        logger.info("Memory-mapped loading complete")
    
    def _load_single_shard(self, shard: WeightShard) -> torch.Tensor:
        """加载单个权重分片"""
        
        if shard.loaded and shard.tensor is not None:
            return shard.tensor
        
        try:
            if shard.file_path.suffix == ".safetensors":
                tensor = self._load_safetensors_shard(shard)
            else:
                tensor = self._load_pytorch_shard(shard)
            
            # 转换到目标设备和数据类型
            if self.loader_config.target_device != "auto":
                target_device = torch.device(self.loader_config.target_device)
                tensor = tensor.to(device=target_device, dtype=shard.dtype)
            
            shard.tensor = tensor
            shard.loaded = True
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard.name}: {e}")
            raise
    
    def _load_safetensors_shard(self, shard: WeightShard) -> torch.Tensor:
        """加载safetensors分片"""
        
        with open(shard.file_path, 'rb') as f:
            f.seek(shard.offset)
            data = f.read(shard.size)
            
            # 解析张量数据
            tensor = torch.frombuffer(
                data,
                dtype=shard.dtype
            ).reshape(shard.shape).clone()
            
            return tensor
    
    def _load_pytorch_shard(self, shard: WeightShard) -> torch.Tensor:
        """加载PyTorch分片"""
        
        # 加载整个文件（PyTorch格式不支持部分加载）
        checkpoint = torch.load(
            shard.file_path,
            map_location="cpu",
            weights_only=True
        )
        
        if shard.name in checkpoint:
            return checkpoint[shard.name].clone()
        else:
            raise KeyError(f"Weight {shard.name} not found in {shard.file_path}")
    
    def _assign_weight_to_model(
        self,
        model: nn.Module,
        weight_name: str,
        tensor: torch.Tensor
    ):
        """将权重分配给模型"""
        
        # 解析权重名称路径
        name_parts = weight_name.split('.')
        
        # 遍历模型结构找到对应的参数
        current_module = model
        
        for part in name_parts[:-1]:
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            elif part.isdigit():
                # 处理数字索引（如layers.0）
                current_module = current_module[int(part)]
            else:
                logger.warning(f"Cannot find module path: {'.'.join(name_parts[:-1])}")
                return
        
        # 设置参数
        param_name = name_parts[-1]
        
        if hasattr(current_module, param_name):
            param = getattr(current_module, param_name)
            
            if isinstance(param, nn.Parameter):
                # 替换参数
                with torch.no_grad():
                    param.data = tensor
            else:
                # 设置属性
                setattr(current_module, param_name, nn.Parameter(tensor))
        else:
            logger.warning(f"Cannot find parameter: {param_name}")
    
    def _group_weights_by_layer(self) -> Dict[int, Dict[str, WeightShard]]:
        """按层分组权重"""
        
        layer_weights = {}
        
        for shard_name, shard in self.weight_shards.items():
            # 解析层索引
            layer_idx = self._extract_layer_index(shard_name)
            
            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = {}
            
            layer_weights[layer_idx][shard_name] = shard
        
        return layer_weights
    
    def _extract_layer_index(self, weight_name: str) -> int:
        """从权重名称中提取层索引"""
        
        # 查找类似 "layers.0" 的模式
        parts = weight_name.split('.')
        
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        
        # 如果没找到层索引，返回0（通常是embedding或输出层）
        return 0
    
    def _lazy_load_hook(self, weight_name: str, grad: torch.Tensor):
        """延迟加载钩子"""
        
        if weight_name in self.weight_shards:
            shard = self.weight_shards[weight_name]
            
            if not shard.loaded:
                logger.info(f"Lazy loading weight: {weight_name}")
                tensor = self._load_single_shard(shard)
                
                # 更新模型中的权重
                # 这里需要根据实际情况实现权重更新逻辑
                
        return grad
    
    def _post_process_model(self, model: nn.Module, metadata: ModelMetadata):
        """模型后处理"""
        
        logger.info("Post-processing model")
        
        # 1. 权重验证
        if self.loader_config.verify_weights:
            self._verify_model_weights(model, metadata)
        
        # 2. 模型优化
        if self.loader_config.optimize_model:
            self._optimize_model(model)
        
        # 3. 设备分配
        if self.loader_config.device_map:
            self._apply_device_map(model, self.loader_config.device_map)
        
        # 4. 编译优化
        if self.loader_config.compile_model:
            model = torch.compile(model)
        
        logger.info("Model post-processing complete")
    
    def _verify_model_weights(self, model: nn.Module, metadata: ModelMetadata):
        """验证模型权重"""
        
        logger.info("Verifying model weights")
        
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()
        
        expected_params = metadata.num_parameters
        
        if abs(total_params - expected_params) > expected_params * 0.01:  # 1%容差
            logger.warning(
                f"Parameter count mismatch: expected {expected_params:,}, "
                f"got {total_params:,}"
            )
        else:
            logger.info(f"Parameter count verified: {total_params:,}")
        
        # 检查权重范围
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"NaN values found in parameter: {name}")
            
            if torch.isinf(param).any():
                logger.error(f"Inf values found in parameter: {name}")
    
    def _optimize_model(self, model: nn.Module):
        """优化模型"""
        
        logger.info("Optimizing model")
        
        # 1. 权重量化
        if self.loader_config.quantization:
            model = self._quantize_model(model)
        
        # 2. 权重融合
        if self.loader_config.fuse_weights:
            model = self._fuse_model_weights(model)
        
        # 3. 内存优化
        if self.loader_config.optimize_memory:
            self._optimize_model_memory(model)
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """量化模型"""
        
        if self.loader_config.quantization == "int8":
            # INT8量化
            model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif self.loader_config.quantization == "int4":
            # INT4量化（需要自定义实现）
            pass
        
        return model
    
    def _fuse_model_weights(self, model: nn.Module) -> nn.Module:
        """融合模型权重"""
        
        # 融合BatchNorm和Conv层
        torch.quantization.fuse_modules(
            model,
            [['conv', 'bn'], ['conv', 'bn', 'relu']],
            inplace=True
        )
        
        return model
    
    def _optimize_model_memory(self, model: nn.Module):
        """优化模型内存"""
        
        # 启用梯度检查点
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # 设置内存高效的注意力
        for module in model.modules():
            if hasattr(module, 'use_memory_efficient_attention'):
                module.use_memory_efficient_attention = True
    
    def _apply_device_map(self, model: nn.Module, device_map: Dict[str, str]):
        """应用设备映射"""
        
        logger.info("Applying device map")
        
        for module_name, device_name in device_map.items():
            device = torch.device(device_name)
            
            # 查找模块
            module = model
            for part in module_name.split('.'):
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    logger.warning(f"Module not found: {module_name}")
                    break
            else:
                # 移动模块到指定设备
                module.to(device)
                logger.info(f"Moved {module_name} to {device}")
    
    def get_loading_progress(self) -> Dict[str, Any]:
        """获取加载进度"""
        
        return {
            "progress": self.loading_progress,
            "is_loading": self.is_loading,
            "elapsed_time": time.time() - self.load_start_time if self.is_loading else 0,
            "loaded_shards": sum(1 for shard in self.weight_shards.values() if shard.loaded),
            "total_shards": len(self.weight_shards),
        }
    
    def cleanup(self):
        """清理资源"""
        
        logger.info("Cleaning up model loader")
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 清理权重缓存
        for shard in self.weight_shards.values():
            if shard.tensor is not None:
                del shard.tensor
                shard.tensor = None
                shard.loaded = False
        
        self.loaded_weights.clear()

class HuggingFaceModelLoader(BaseModelLoader):
    """HuggingFace模型加载器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        memory_manager: UnifiedMemoryManager,
    ):
        super().__init__(model_config, loader_config, memory_manager)
        
        # HuggingFace特定配置
        self.trust_remote_code = loader_config.trust_remote_code
        self.revision = loader_config.revision
        self.use_auth_token = loader_config.use_auth_token
    
    def load_model(self) -> nn.Module:
        """加载HuggingFace模型"""
        
        logger.info(f"Loading HuggingFace model: {self.model_config.model_name}")
        
        try:
            # 使用transformers库加载
            from transformers import AutoModelForCausalLM
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=getattr(torch, self.model_config.torch_dtype),
                device_map=self.loader_config.device_map,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                use_auth_token=self.use_auth_token,
                low_cpu_mem_usage=True,
            )
            
            logger.info("HuggingFace model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            # 回退到基础加载器
            return super().load_model()

class SafetensorsModelLoader(BaseModelLoader):
    """Safetensors模型加载器"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        memory_manager: UnifiedMemoryManager,
    ):
        super().__init__(model_config, loader_config, memory_manager)
    
    def _load_safetensors_shard(self, shard: WeightShard) -> torch.Tensor:
        """优化的safetensors分片加载"""
        
        try:
            from safetensors import safe_open
            
            with safe_open(shard.file_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(shard.name)
                return tensor.clone()
                
        except ImportError:
            # 回退到基础实现
            return super()._load_safetensors_shard(shard)

class ModelLoaderFactory:
    """模型加载器工厂"""
    
    @staticmethod
    def create_loader(
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        memory_manager: UnifiedMemoryManager,
    ) -> BaseModelLoader:
        """创建模型加载器"""
        
        model_path = Path(model_config.model_path)
        
        # 检测模型格式
        if any(model_path.glob("*.safetensors")):
            return SafetensorsModelLoader(model_config, loader_config, memory_manager)
        elif (model_path / "config.json").exists():
            return HuggingFaceModelLoader(model_config, loader_config, memory_manager)
        else:
            return BaseModelLoader(model_config, loader_config, memory_manager)

# 使用示例
def example_model_loader_usage():
    """模型加载器使用示例"""
    
    # 配置
    model_config = ModelConfig(
        model_name="llama-7b",
        model_path="/path/to/llama-7b",
        torch_dtype="float16",
    )
    
    loader_config = LoaderConfig(
        loading_strategy=LoadingStrategy.EAGER,
        num_loading_threads=4,
        loading_timeout=300,
        verify_weights=True,
        optimize_model=True,
        device_map="auto",
        trust_remote_code=False,
    )
    
    memory_config = MemoryConfig()
    cache_config = CacheConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建内存管理器
    memory_manager = UnifiedMemoryManager(
        device=device,
        memory_config=memory_config,
        cache_config=cache_config,
    )
    
    # 创建模型加载器
    loader = ModelLoaderFactory.create_loader(
        model_config=model_config,
        loader_config=loader_config,
        memory_manager=memory_manager,
    )
    
    try:
        # 加载模型
        model = loader.load_model()
        print(f"Model loaded: {type(model).__name__}")
        
        # 获取加载进度
        progress = loader.get_loading_progress()
        print(f"Loading progress: {progress}")
        
        # 模型推理示例
        model.eval()
        with torch.no_grad():
            # 这里添加推理代码
            pass
        
    finally:
        # 清理资源
        loader.cleanup()
        memory_manager.shutdown()

if __name__ == "__main__":
    example_model_loader_usage()
```

## 🔧 关键特性分析

### 1. 多格式支持

- **HuggingFace格式**：标准的transformers模型
- **Safetensors格式**：安全的张量存储格式
- **PyTorch格式**：原生PyTorch模型文件
- **自定义格式**：支持扩展其他格式

### 2. 灵活的加载策略

- **立即加载**：一次性加载所有权重
- **延迟加载**：按需加载权重
- **流式加载**：按层顺序加载
- **内存映射**：零拷贝的内存映射加载

### 3. 并行加载优化

- **多线程加载**：并行加载多个权重分片
- **异步处理**：非阻塞的加载过程
- **进度跟踪**：实时的加载进度反馈
- **错误恢复**：加载失败的自动重试

## 📊 性能优化技术

### 权重预处理优化

```python
class WeightPreprocessor:
    """权重预处理器"""
    
    def __init__(self, loader_config: LoaderConfig):
        self.loader_config = loader_config
    
    def preprocess_weights(
        self,
        weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """预处理权重"""
        
        processed_weights = {}
        
        for name, weight in weights.items():
            # 1. 数据类型转换
            if self.loader_config.target_dtype:
                weight = weight.to(dtype=self.loader_config.target_dtype)
            
            # 2. 权重量化
            if self.loader_config.quantization:
                weight = self._quantize_weight(weight, name)
            
            # 3. 权重重排
            if self.loader_config.reorder_weights:
                weight = self._reorder_weight(weight, name)
            
            # 4. 权重融合
            if self.loader_config.fuse_weights:
                weight = self._fuse_weight(weight, name)
            
            processed_weights[name] = weight
        
        return processed_weights
    
    def _quantize_weight(self, weight: torch.Tensor, name: str) -> torch.Tensor:
        """量化权重"""
        
        if "embed" in name or "lm_head" in name:
            # 嵌入层和输出层保持高精度
            return weight
        
        if self.loader_config.quantization == "int8":
            # INT8量化
            scale = weight.abs().max() / 127
            quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
            
            # 保存量化参数
            quantized.scale = scale
            return quantized
            
        elif self.loader_config.quantization == "int4":
            # INT4量化
            scale = weight.abs().max() / 7
            quantized = torch.round(weight / scale).clamp(-8, 7).to(torch.int8)
            
            quantized.scale = scale
            return quantized
        
        return weight
    
    def _reorder_weight(self, weight: torch.Tensor, name: str) -> torch.Tensor:
        """重排权重以优化内存访问"""
        
        if len(weight.shape) == 2:  # 线性层权重
            # 按行重排以提高缓存局部性
            if weight.shape[0] > weight.shape[1]:
                return weight.t().contiguous().t()
        
        return weight.contiguous()
    
    def _fuse_weight(self, weight: torch.Tensor, name: str) -> torch.Tensor:
        """融合权重"""
        
        # 这里实现权重融合逻辑
        # 例如：将多个小权重合并为一个大权重
        
        return weight
```

### 缓存优化加载

```python
class CachedModelLoader:
    """缓存优化的模型加载器"""
    
    def __init__(
        self,
        base_loader: BaseModelLoader,
        cache_dir: Path,
    ):
        self.base_loader = base_loader
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> nn.Module:
        """加载模型（带缓存）"""
        
        # 计算模型缓存键
        cache_key = self._compute_cache_key()
        cache_path = self.cache_dir / f"{cache_key}.cache"
        
        # 检查缓存
        if cache_path.exists() and self._is_cache_valid(cache_path):
            logger.info("Loading model from cache")
            return self._load_from_cache(cache_path)
        
        # 加载原始模型
        logger.info("Loading model from source")
        model = self.base_loader.load_model()
        
        # 保存到缓存
        self._save_to_cache(model, cache_path)
        
        return model
    
    def _compute_cache_key(self) -> str:
        """计算缓存键"""
        
        # 基于模型配置和文件修改时间计算哈希
        hasher = hashlib.sha256()
        
        # 模型配置
        config_str = json.dumps(
            self.base_loader.model_config.__dict__,
            sort_keys=True
        )
        hasher.update(config_str.encode())
        
        # 加载器配置
        loader_config_str = json.dumps(
            self.base_loader.loader_config.__dict__,
            sort_keys=True
        )
        hasher.update(loader_config_str.encode())
        
        # 文件修改时间
        for file_path in self.base_loader.model_path.glob("*"):
            if file_path.is_file():
                mtime = file_path.stat().st_mtime
                hasher.update(str(mtime).encode())
        
        return hasher.hexdigest()[:16]
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """检查缓存是否有效"""
        
        try:
            # 检查缓存文件的修改时间
            cache_mtime = cache_path.stat().st_mtime
            
            # 检查源文件是否有更新
            for file_path in self.base_loader.model_path.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime > cache_mtime:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _load_from_cache(self, cache_path: Path) -> nn.Module:
        """从缓存加载模型"""
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        model = cached_data['model']
        metadata = cached_data['metadata']
        
        logger.info(f"Loaded model from cache: {metadata['name']}")
        return model
    
    def _save_to_cache(self, model: nn.Module, cache_path: Path):
        """保存模型到缓存"""
        
        cached_data = {
            'model': model,
            'metadata': {
                'name': self.base_loader.model_config.model_name,
                'timestamp': time.time(),
                'loader_version': "1.0",
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        logger.info(f"Saved model to cache: {cache_path}")
```

### 分布式加载

```python
class DistributedModelLoader:
    """分布式模型加载器"""
    
    def __init__(
        self,
        base_loader: BaseModelLoader,
        world_size: int,
        rank: int,
    ):
        self.base_loader = base_loader
        self.world_size = world_size
        self.rank = rank
    
    def load_model(self) -> nn.Module:
        """分布式加载模型"""
        
        logger.info(f"Loading model on rank {self.rank}/{self.world_size}")
        
        # 1. 协调加载计划
        loading_plan = self._create_loading_plan()
        
        # 2. 分布式加载权重
        model = self._distributed_load_weights(loading_plan)
        
        # 3. 同步检查点
        self._synchronize_loading()
        
        return model
    
    def _create_loading_plan(self) -> Dict[str, Any]:
        """创建加载计划"""
        
        # 分析权重分片
        self.base_loader._analyze_weight_shards(
            self.base_loader._load_metadata()
        )
        
        # 按rank分配权重
        all_shards = list(self.base_loader.weight_shards.items())
        shards_per_rank = len(all_shards) // self.world_size
        
        start_idx = self.rank * shards_per_rank
        end_idx = start_idx + shards_per_rank
        
        if self.rank == self.world_size - 1:
            # 最后一个rank处理剩余的分片
            end_idx = len(all_shards)
        
        my_shards = dict(all_shards[start_idx:end_idx])
        
        return {
            'my_shards': my_shards,
            'total_shards': len(all_shards),
            'shard_range': (start_idx, end_idx),
        }
    
    def _distributed_load_weights(self, loading_plan: Dict[str, Any]) -> nn.Module:
        """分布式加载权重"""
        
        # 创建模型架构
        metadata = self.base_loader._load_metadata()
        model = self.base_loader._create_model_architecture(metadata)
        
        # 加载分配给当前rank的权重
        my_shards = loading_plan['my_shards']
        
        for shard_name, shard in my_shards.items():
            tensor = self.base_loader._load_single_shard(shard)
            self.base_loader._assign_weight_to_model(model, shard_name, tensor)
        
        logger.info(f"Rank {self.rank} loaded {len(my_shards)} shards")
        
        return model
    
    def _synchronize_loading(self):
        """同步加载进度"""
        
        # 这里实现分布式同步逻辑
        # 例如使用torch.distributed进行同步
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            logger.info(f"Rank {self.rank} synchronized")
```

## 🚀 使用最佳实践

### 1. 加载策略选择

```python
def choose_loading_strategy(
    model_size: int,
    available_memory: int,
    num_gpus: int,
) -> LoadingStrategy:
    """选择最佳加载策略"""
    
    memory_ratio = model_size / available_memory
    
    if memory_ratio <= 0.5:
        # 内存充足，使用立即加载
        return LoadingStrategy.EAGER
    elif memory_ratio <= 0.8:
        # 内存适中，使用流式加载
        return LoadingStrategy.STREAMING
    elif num_gpus > 1:
        # 多GPU环境，使用延迟加载
        return LoadingStrategy.LAZY
    else:
        # 内存紧张，使用内存映射
        return LoadingStrategy.MEMORY_MAPPED
```

### 2. 错误处理和重试

```python
class RobustModelLoader:
    """健壮的模型加载器"""
    
    def __init__(self, base_loader: BaseModelLoader):
        self.base_loader = base_loader
        self.max_retries = 3
        self.retry_delay = 5.0
    
    def load_model_with_retry(self) -> nn.Module:
        """带重试的模型加载"""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Loading attempt {attempt + 1}/{self.max_retries}")
                
                model = self.base_loader.load_model()
                logger.info("Model loaded successfully")
                return model
                
            except Exception as e:
                last_error = e
                logger.error(f"Loading attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    
                    # 清理状态
                    self._cleanup_failed_loading()
        
        raise RuntimeError(f"Failed to load model after {self.max_retries} attempts") from last_error
    
    def _cleanup_failed_loading(self):
        """清理失败的加载状态"""
        
        # 清理部分加载的权重
        for shard in self.base_loader.weight_shards.values():
            if shard.tensor is not None:
                del shard.tensor
                shard.tensor = None
                shard.loaded = False
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 3. 性能监控

```python
class LoadingProfiler:
    """加载性能分析器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_profiling(self):
        """开始性能分析"""
        self.start_time = time.time()
        self.metrics = {
            'loading_times': {},
            'memory_usage': [],
            'io_stats': {},
        }
    
    def record_shard_loading(self, shard_name: str, loading_time: float):
        """记录分片加载时间"""
        self.metrics['loading_times'][shard_name] = loading_time
    
    def record_memory_usage(self):
        """记录内存使用"""
        if torch.cuda.is_available():
            memory_info = {
                'timestamp': time.time() - self.start_time,
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
            }
            self.metrics['memory_usage'].append(memory_info)
    
    def generate_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        
        total_time = time.time() - self.start_time
        loading_times = list(self.metrics['loading_times'].values())
        
        return {
            'total_loading_time': total_time,
            'average_shard_time': np.mean(loading_times) if loading_times else 0,
            'slowest_shard_time': max(loading_times) if loading_times else 0,
            'fastest_shard_time': min(loading_times) if loading_times else 0,
            'peak_memory_usage': max(
                m['allocated'] for m in self.metrics['memory_usage']
            ) if self.metrics['memory_usage'] else 0,
            'loading_throughput': len(loading_times) / total_time if total_time > 0 else 0,
        }
```

---

*模型加载器是 nano-vllm 的入口组件，高效的模型加载能够显著减少系统启动时间，提升用户体验。*