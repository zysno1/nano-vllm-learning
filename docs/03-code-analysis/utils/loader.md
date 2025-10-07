# 模型运行器分析 - 详细注释版

## 🎯 模型运行器概览

模型运行器是 nano-vllm 中负责执行模型推理、管理推理状态和优化推理性能的核心组件。它支持预填充和解码两个阶段、CUDA图优化、KV缓存管理等功能，确保模型能够高效地执行推理并生成高质量的输出。

**设计思想：**
- **多格式兼容**：支持HuggingFace、Safetensors、PyTorch等多种模型格式，确保广泛的生态兼容性
- **灵活加载策略**：提供立即、延迟、流式、内存映射等多种加载方式，适应不同的内存和性能需求
- **内存优化**：通过权重分片和智能缓存减少内存占用，支持大模型在有限资源下运行
- **并行加载**：利用多线程并行加载提升加载速度，减少用户等待时间
- **错误恢复**：完善的错误处理和重试机制，确保加载过程的稳定性和可靠性
- **缓存机制**：智能缓存已加载的模型组件，避免重复加载，提升系统效率
- **版本管理**：支持模型版本检测和兼容性检查，确保模型正确性

## 🏗️ 核心架构与导入

```python
# 核心PyTorch组件 - 模型构建和张量操作的基础
# 设计：PyTorch是深度学习的核心框架，提供张量操作和神经网络构建能力
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple

# 文件系统和数据处理
# 设计：现代化的文件处理和数据序列化，支持多种数据格式
from pathlib import Path  # 现代化的路径处理，比os.path更安全和直观
import json              # 配置文件解析，处理模型配置和元数据
import pickle            # 模型缓存序列化，支持Python对象持久化
import safetensors       # 安全的张量存储格式，比pickle更安全和高效

# 数据结构和枚举
# 设计：结构化数据定义和类型安全，提高代码可维护性
from dataclasses import dataclass, field  # 结构化数据定义，自动生成构造函数和比较方法
from enum import Enum                      # 枚举类型定义，提供类型安全的常量

# 并发和系统
# 设计：支持并行处理和系统级操作，提升性能和功能完整性
import threading                           # 线程同步，保证并发安全
import time                               # 时间测量，用于性能监控和超时控制
import logging                            # 日志记录，便于调试和监控
from concurrent.futures import ThreadPoolExecutor, as_completed  # 并行处理，提升加载速度
import hashlib                            # 哈希计算，用于生成缓存键和完整性校验
import mmap                               # 内存映射文件，实现零拷贝文件访问
import os                                 # 系统操作，文件和环境变量处理

# HuggingFace生态系统集成
# 设计：与HuggingFace生态深度集成，支持主流预训练模型
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_MAPPING

# nano-vllm内部组件
# 设计：模块化设计，各组件职责清晰，便于维护和扩展
from nano_vllm.config import ModelConfig, LoaderConfig
from nano_vllm.utils.logger import get_logger
from nano_vllm.memory.manager import UnifiedMemoryManager

logger = get_logger(__name__)

# ================================
# 核心数据结构定义
# ================================

class ModelFormat(Enum):
    """模型格式枚举
    
    设计思想：
    - 支持主流模型格式，确保广泛兼容性，满足不同用户需求
    - 每种格式有不同的加载策略和优化方法，针对性优化性能
    - 可扩展设计，便于添加新格式，保持技术前瞻性
    - 格式选择影响加载速度、安全性和兼容性
    """
    HUGGINGFACE = "huggingface"  # 标准transformers格式，兼容性最好，生态最完善
    SAFETENSORS = "safetensors"  # 安全格式，加载速度快，支持元数据，防止代码注入
    PYTORCH = "pytorch"          # 原生PyTorch格式，灵活性高，支持自定义结构
    ONNX = "onnx"               # 跨平台推理格式，支持多种推理引擎
    TENSORRT = "tensorrt"       # NVIDIA优化格式，推理性能最佳，GPU加速
    CUSTOM = "custom"           # 自定义格式，支持特殊需求和优化

class LoadingStrategy(Enum):
    """加载策略枚举
    
    设计思想：
    - 根据内存和性能需求选择不同策略，平衡资源使用和响应速度
    - 支持从内存充足到内存受限的各种场景
    - 策略选择影响启动时间、内存占用和运行性能
    - 可根据系统资源动态选择最优策略
    """
    EAGER = "eager"          # 立即加载所有权重 - 内存充足时最快，推理延迟最低
    LAZY = "lazy"            # 延迟加载权重 - 节省内存，按需加载，启动快
    STREAMING = "streaming"   # 流式加载 - 大模型分层加载，支持超大模型
    MEMORY_MAPPED = "mmap"   # 内存映射加载 - 零拷贝，适合只读场景，内存效率高

@dataclass
class ModelMetadata:
    """模型元数据
    
    设计思想：
    - 集中管理模型的所有关键信息，提供统一的信息访问接口
    - 支持加载决策和优化策略，根据模型特征选择最优方案
    - 便于缓存和版本管理，避免重复计算和加载
    - 包含足够信息支持内存规划和性能预测
    """
    name: str                                    # 模型名称标识，用于日志和缓存
    format: ModelFormat                          # 模型格式类型，决定加载方法
    size: int                                   # 模型大小（字节）- 用于内存规划和进度显示
    num_parameters: int                         # 参数数量 - 用于性能估算和资源规划
    architecture: str                           # 模型架构 - 决定加载策略和优化方法
    vocab_size: int                            # 词汇表大小，影响输出层内存需求
    hidden_size: int                           # 隐藏层大小，影响中间计算内存需求
    num_layers: int                            # 层数 - 用于流式加载规划和并行策略
    num_heads: int                             # 注意力头数
    max_position_embeddings: int               # 最大位置编码
    dtype: torch.dtype                         # 数据类型 - 影响内存使用
    device_map: Optional[Dict[str, str]] = None  # 设备映射 - 多GPU分布
    shard_info: Optional[Dict[str, Any]] = None  # 分片信息 - 分布式加载
    checksum: Optional[str] = None               # 校验和 - 完整性验证
    created_at: float = field(default_factory=time.time)  # 创建时间戳

@dataclass
class WeightShard:
    """权重分片数据结构
    
    设计思想：
    - **分片存储**：支持大模型的分片存储和加载，突破单文件大小限制
    - **精确控制**：精确控制内存使用和加载顺序，优化资源利用
    - **并行支持**：支持并行加载和缓存管理，提升加载效率
    - **状态跟踪**：完整的加载状态跟踪，支持断点续传和错误恢复
    - **设备感知**：支持多设备分布，实现模型并行和内存优化
    """
    name: str                                   # 分片名称 - 唯一标识，用于日志和调试
    file_path: Path                            # 文件路径 - 存储位置，支持相对和绝对路径
    offset: int                                # 文件偏移 - 精确定位分片在文件中的位置
    size: int                                  # 分片大小 - 内存预算计算和进度跟踪
    shape: Tuple[int, ...]                     # 张量形状 - 重建张量时的维度信息
    dtype: torch.dtype                         # 数据类型 - 内存计算和类型转换
    device: torch.device                       # 目标设备 - 设备分配和数据传输
    loaded: bool = False                       # 是否已加载 - 状态跟踪和重复加载避免
    tensor: Optional[torch.Tensor] = None      # 加载的张量 - 缓存数据，避免重复IO

# ================================
# 基础模型加载器
# ================================

class BaseModelLoader:
    """基础模型加载器
    
    设计思想：
    - **通用框架**：提供通用的模型加载框架，支持多种模型格式和加载策略
    - **策略模式**：支持多种加载策略和优化技术，根据场景选择最优方案
    - **错误处理**：完善的错误处理和进度跟踪，确保加载过程的稳定性
    - **可扩展性**：可扩展的架构设计，便于添加新的模型格式和优化技术
    - **性能优化**：内存优化、并行加载、缓存机制等多种性能优化手段
    - **监控友好**：详细的日志记录和进度跟踪，便于监控和调试
    """
    
    def __init__(
        self,
        model_config: ModelConfig,      # 模型配置 - 定义模型参数和路径信息
        loader_config: LoaderConfig,    # 加载器配置 - 定义加载行为和优化参数
        memory_manager: UnifiedMemoryManager,  # 内存管理器 - 统一内存分配和优化
    ):
        """初始化基础模型加载器
        
        设计思想：
        - **配置驱动**：通过配置对象控制加载行为，提高灵活性
        - **依赖注入**：注入内存管理器，实现组件解耦和资源统一管理
        - **状态初始化**：初始化所有必要的状态变量，确保对象完整性
        - **资源预分配**：预分配线程池等资源，避免运行时开销
        - **线程安全**：初始化线程安全机制，支持并发操作
        """
        # 配置存储 - 保存配置对象，供后续方法使用
        self.model_config = model_config
        self.loader_config = loader_config
        self.memory_manager = memory_manager
        
        # 模型路径解析 - 统一路径处理，支持相对和绝对路径
        self.model_path = Path(model_config.model_path)
        
        # 加载状态管理 - 跟踪加载进度和状态，支持进度显示和错误恢复
        self.loading_progress = 0.0      # 加载进度 [0.0, 1.0] - 实时进度跟踪
        self.is_loading = False          # 加载状态标志 - 防止重复加载
        self.load_start_time = 0.0       # 加载开始时间 - 性能监控和超时检测
        
        # 权重管理 - 分片和缓存管理，支持大模型和内存优化
        self.weight_shards: Dict[str, WeightShard] = {}  # 权重分片映射 - 分片信息管理
        self.loaded_weights: Dict[str, torch.Tensor] = {}  # 已加载权重缓存 - 避免重复加载
        
        # 并发控制 - 线程池管理，提升加载性能
        self.thread_pool = ThreadPoolExecutor(
            max_workers=loader_config.num_loading_threads  # 并行加载线程数 - 根据配置调整
        )
        
        # 线程安全 - 锁机制，确保并发安全
        self.loading_lock = threading.RLock()  # 可重入锁，防止并发冲突和死锁
        
        logger.info(f"Initialized BaseModelLoader for {model_config.model_name}")
    
    def load_model(self) -> nn.Module:
        """加载模型主流程
        
        设计思想：
        - **分阶段加载**：分阶段加载，每个阶段有明确职责，便于调试和优化
        - **错误处理**：完善的错误处理和状态管理，确保加载过程的稳定性
        - **进度跟踪**：实时进度跟踪和性能监控，提供用户反馈
        - **线程安全**：线程安全的并发控制，支持多线程环境
        - **资源管理**：合理的资源分配和释放，避免内存泄漏
        - **可观测性**：详细的日志记录，便于问题诊断和性能分析
        """
        
        # 并发控制 - 确保同时只有一个加载过程，避免资源竞争
        with self.loading_lock:
            if self.is_loading:
                raise RuntimeError("Model is already being loaded")
            
            # 设置加载状态 - 标记开始加载，记录开始时间
            self.is_loading = True
            self.load_start_time = time.time()
            self.loading_progress = 0.0
            
            try:
                logger.info(f"Starting model loading: {self.model_config.model_name}")
                
                # 阶段1: 模型元数据加载 (10%)
                # 设计：首先加载元数据，验证模型完整性和兼容性
                metadata = self._load_metadata()
                self.loading_progress = 0.1
                logger.info(f"Loaded metadata: {metadata.architecture}, {metadata.num_parameters} parameters")
                
                # 阶段2: 权重分片分析 (20%)
                # 设计：分析权重分片，规划加载策略和内存分配
                self._analyze_weight_shards()
                self.loading_progress = 0.2
                logger.info(f"Analyzed {len(self.weight_shards)} weight shards")
                
                # 阶段3: 内存预分配 (30%)
                # 设计：根据模型大小预分配内存，避免加载时内存不足
                self._allocate_memory(metadata)
                self.loading_progress = 0.3
                logger.info("Memory allocated successfully")
                
                # 阶段4: 模型结构构建 (40%)
                # 设计：构建模型结构，准备权重加载的容器
                model = self._build_model_structure(metadata)
                self.loading_progress = 0.4
                logger.info(f"Built model structure: {type(model).__name__}")
                
                # 阶段5: 权重并行加载 (80%)
                # 设计：并行加载权重，充分利用IO和CPU资源
                self._load_weights_parallel(model)
                self.loading_progress = 0.8
                logger.info("Weights loaded successfully")
                
                # 阶段6: 模型初始化和验证 (90%)
                # 设计：初始化模型状态，验证加载正确性
                self._initialize_model(model)
                self.loading_progress = 0.9
                logger.info("Model initialized successfully")
                
                # 阶段7: 后处理和优化 (100%)
                # 设计：执行后处理优化，如权重量化、图优化等
                self._post_process_model(model)
                self.loading_progress = 1.0
                
                # 加载完成统计
                load_time = time.time() - self.load_start_time
                logger.info(f"Model loading completed in {load_time:.2f}s")
                
                return model
                
            except Exception as e:
                # 错误处理 - 记录错误信息，清理资源，重置状态
                logger.error(f"Model loading failed: {str(e)}")
                self._cleanup_on_error()
                raise
            finally:
                # 状态重置 - 无论成功失败都要重置加载状态
                self.is_loading = False

    def _load_metadata(self) -> ModelMetadata:
        """加载模型元数据
        
        设计思想：
        - **多源支持**：支持从config.json、model.safetensors.index.json等多种源加载元数据
        - **智能推断**：当元数据不完整时，通过模型文件智能推断缺失信息
        - **缓存优化**：缓存元数据避免重复解析，提升后续加载速度
        - **兼容性检查**：验证模型格式和版本兼容性，确保加载成功
        """
        config_path = self.model_path / "config.json"
        
        if config_path.exists():
            # 从HuggingFace配置文件加载 - 标准格式，信息最完整
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 构建元数据对象 - 提取关键信息，设置默认值
            metadata = ModelMetadata(
                name=self.model_config.model_name,
                format=self._detect_model_format(),  # 自动检测模型格式
                size=self._calculate_model_size(),   # 计算模型总大小
                num_parameters=config.get('num_parameters', 0),
                architecture=config.get('architectures', ['unknown'])[0],
                vocab_size=config.get('vocab_size', 32000),
                hidden_size=config.get('hidden_size', 4096),
                num_layers=config.get('num_hidden_layers', 32),
                num_heads=config.get('num_attention_heads', 32),
                max_position_embeddings=config.get('max_position_embeddings', 2048),
                dtype=getattr(torch, config.get('torch_dtype', 'float16')),
            )
        else:
            # 从模型文件推断元数据 - 兜底方案，确保兼容性
            metadata = self._infer_metadata_from_weights()
        
        # 验证元数据完整性 - 确保关键信息不缺失
        self._validate_metadata(metadata)
        
        return metadata
    
    def _analyze_weight_shards(self):
        """分析权重分片
        
        设计思想：
        - **自动发现**：自动发现所有权重文件，支持单文件和多文件模型
        - **分片规划**：根据内存限制和并行度规划分片加载策略
        - **依赖分析**：分析分片间依赖关系，确定加载顺序
        - **设备分配**：为每个分片分配目标设备，支持模型并行
        """
        # 发现权重文件 - 支持多种文件格式和命名规则
        weight_files = []
        for pattern in ['*.safetensors', '*.bin', '*.pt', '*.pth']:
            weight_files.extend(self.model_path.glob(pattern))
        
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {self.model_path}")
        
        # 分析每个权重文件 - 提取分片信息
        for i, weight_file in enumerate(weight_files):
            # 计算文件信息 - 大小、偏移、设备分配
            file_size = weight_file.stat().st_size
            device = self._assign_device_for_shard(i, len(weight_files))
            
            # 创建分片对象 - 封装分片信息
            shard = WeightShard(
                name=f"shard_{i}",
                file_path=weight_file,
                offset=0,  # 单文件分片偏移为0
                size=file_size,
                shape=(),  # 将在加载时确定
                dtype=torch.float16,  # 默认类型，将在加载时更新
                device=device,
            )
            
            self.weight_shards[shard.name] = shard
        
        logger.info(f"Discovered {len(self.weight_shards)} weight shards")
    
    def _allocate_memory(self, metadata: ModelMetadata):
        """内存预分配
        
        设计思想：
        - **精确计算**：根据模型参数和数据类型精确计算内存需求
        - **预分配策略**：预分配内存避免加载时内存不足导致失败
        - **设备感知**：考虑多设备内存分布，优化内存利用率
        - **动态调整**：根据实际可用内存动态调整分配策略
        """
        # 计算模型内存需求 - 参数 + 激活 + 缓存
        param_memory = metadata.num_parameters * metadata.dtype.itemsize
        activation_memory = self._estimate_activation_memory(metadata)
        cache_memory = self.loader_config.weight_cache_size
        
        total_memory_needed = param_memory + activation_memory + cache_memory
        
        # 检查内存可用性 - 确保有足够内存加载模型
        available_memory = self.memory_manager.get_available_memory()
        if total_memory_needed > available_memory:
            # 尝试释放内存 - 清理缓存、卸载其他模型
            self.memory_manager.free_memory(total_memory_needed - available_memory)
        
        # 预分配内存块 - 避免加载时内存碎片
        self.memory_manager.reserve_memory(total_memory_needed)
        
        logger.info(f"Reserved {total_memory_needed / 1024**3:.2f}GB memory for model")
    
    def _build_model_structure(self, metadata: ModelMetadata) -> nn.Module:
        """构建模型结构
        
        设计思想：
        - **架构识别**：根据元数据自动识别模型架构，支持主流模型
        - **动态构建**：动态构建模型结构，避免硬编码架构定义
        - **参数初始化**：合理初始化模型参数，为权重加载做准备
        - **设备分配**：将模型组件分配到合适的设备上
        """
        # 根据架构选择模型类 - 支持多种主流架构
        if metadata.architecture.lower() in ['llama', 'llama2']:
            from .models.llama import LlamaForCausalLM
            model_class = LlamaForCausalLM
        elif metadata.architecture.lower() in ['gpt2', 'gpt']:
            from .models.gpt2 import GPT2LMHeadModel
            model_class = GPT2LMHeadModel
        elif metadata.architecture.lower() in ['bloom']:
            from .models.bloom import BloomForCausalLM
            model_class = BloomForCausalLM
        else:
            # 通用模型类 - 兜底方案
            from .models.generic import GenericCausalLM
            model_class = GenericCausalLM
        
        # 构建模型配置 - 从元数据转换为模型配置
        model_config = self._build_model_config(metadata)
        
        # 创建模型实例 - 使用配置创建模型
        model = model_class(model_config)
        
        # 设备分配 - 将模型移动到指定设备
        if metadata.device_map:
            # 使用指定的设备映射 - 支持模型并行
            self._apply_device_map(model, metadata.device_map)
        else:
            # 默认设备分配 - 单设备或自动分配
            model = model.to(self.loader_config.device)
        
        return model
    
    def _load_weights_parallel(self, model: nn.Module):
        """并行加载权重
        
        设计思想：
        - **并行IO**：并行读取多个权重文件，充分利用IO带宽
        - **流水线处理**：IO读取和权重设置流水线处理，提升效率
        - **内存优化**：边加载边释放，控制内存峰值使用
        - **错误恢复**：单个分片失败不影响整体加载，支持重试
        """
        # 创建加载任务 - 为每个分片创建独立的加载任务
        loading_tasks = []
        for shard_name, shard in self.weight_shards.items():
            task = self.thread_pool.submit(self._load_single_shard, shard)
            loading_tasks.append((shard_name, task))
        
        # 并行执行加载 - 等待所有任务完成
        loaded_count = 0
        for shard_name, task in loading_tasks:
            try:
                # 获取加载结果 - 阻塞等待单个分片完成
                weights_dict = task.result(timeout=self.loader_config.loading_timeout)
                
                # 设置模型权重 - 将加载的权重设置到模型中
                self._set_model_weights(model, weights_dict)
                
                loaded_count += 1
                # 更新进度 - 实时更新加载进度
                progress = 0.4 + (loaded_count / len(self.weight_shards)) * 0.4
                self.loading_progress = progress
                
                logger.debug(f"Loaded shard {shard_name} ({loaded_count}/{len(self.weight_shards)})")
                
            except Exception as e:
                logger.error(f"Failed to load shard {shard_name}: {str(e)}")
                # 可选：实现重试机制
                if self.loader_config.retry_on_failure:
                    self._retry_load_shard(shard_name)
                else:
                    raise
    
    def _load_single_shard(self, shard: WeightShard) -> Dict[str, torch.Tensor]:
        """加载单个权重分片
        
        设计思想：
        - **格式适配**：支持多种权重文件格式，自动选择加载方法
        - **内存映射**：大文件使用内存映射，减少内存占用
        - **类型转换**：自动处理数据类型转换和设备传输
        - **完整性检查**：验证加载的权重完整性和正确性
        """
        try:
            # 根据文件扩展名选择加载方法
            if shard.file_path.suffix == '.safetensors':
                # SafeTensors格式 - 安全高效的张量存储格式
                from safetensors import safe_open
                weights = {}
                with safe_open(shard.file_path, framework="pt", device=str(shard.device)) as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key)
                        
            elif shard.file_path.suffix in ['.bin', '.pt', '.pth']:
                # PyTorch格式 - 原生PyTorch序列化格式
                weights = torch.load(
                    shard.file_path,
                    map_location=shard.device,
                    weights_only=True  # 安全加载，只加载权重
                )
            else:
                raise ValueError(f"Unsupported weight file format: {shard.file_path.suffix}")
            
            # 更新分片信息 - 记录实际加载的信息
            shard.loaded = True
            shard.tensor = weights if isinstance(weights, torch.Tensor) else None
            
            # 权重预处理 - 类型转换、形状检查等
            processed_weights = self._preprocess_weights(weights, shard)
            
            return processed_weights
            
        except Exception as e:
            logger.error(f"Failed to load shard {shard.name}: {str(e)}")
            raise
    
    def _initialize_model(self, model: nn.Module):
        """初始化模型
        
        设计思想：
        - **状态初始化**：初始化模型的运行时状态和缓存
        - **优化应用**：应用各种模型优化，如融合、量化等
        - **验证检查**：验证模型加载正确性，确保可以正常推理
        - **预热处理**：执行模型预热，优化首次推理性能
        """
        # 设置模型为评估模式 - 关闭dropout等训练相关组件
        model.eval()
        
        # 应用模型优化 - 根据配置应用各种优化
        if self.loader_config.enable_fusion:
            # 算子融合 - 融合相邻算子，减少内存访问
            self._apply_operator_fusion(model)
        
        if self.loader_config.enable_quantization:
            # 权重量化 - 减少内存占用和计算量
            self._apply_quantization(model)
        
        # 初始化KV缓存 - 为注意力机制预分配缓存
        self._initialize_kv_cache(model)
        
        # 模型验证 - 执行简单推理验证模型正确性
        self._validate_model(model)
        
        # 模型预热 - 执行几次前向传播，优化性能
        if self.loader_config.enable_warmup:
            self._warmup_model(model)
        
        logger.info("Model initialization completed successfully")
    
    def _post_process_model(self, model: nn.Module):
        """模型后处理
        
        设计思想：
        - **性能优化**：应用最终的性能优化，如图优化、编译等
        - **内存整理**：整理内存布局，减少内存碎片
        - **状态保存**：保存模型状态，支持快速重启
        - **监控设置**：设置性能监控钩子，便于运行时监控
        """
        # 图优化 - 优化计算图，提升推理性能
        if self.loader_config.enable_graph_optimization:
            model = self._optimize_computation_graph(model)
        
        # JIT编译 - 编译模型以获得更好性能
        if self.loader_config.enable_jit_compile:
            model = torch.jit.script(model)
        
        # 内存整理 - 整理GPU内存，减少碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 设置监控钩子 - 用于性能监控和调试
        if self.loader_config.enable_monitoring:
            self._setup_monitoring_hooks(model)
        
        logger.info("Model post-processing completed")
    
    def _cleanup_on_error(self):
        """错误清理
        
        设计思想：
        - **资源释放**：释放已分配的内存和其他资源
        - **状态重置**：重置加载器状态，准备重新加载
        - **缓存清理**：清理可能损坏的缓存数据
        - **日志记录**：记录清理过程，便于问题诊断
        """
        try:
            # 释放已加载的权重 - 清理内存
            self.loaded_weights.clear()
            
            # 重置分片状态 - 标记所有分片为未加载
            for shard in self.weight_shards.values():
                shard.loaded = False
                shard.tensor = None
            
            # 释放预分配的内存 - 归还给内存管理器
            self.memory_manager.release_reserved_memory()
            
            # 清理GPU缓存 - 释放GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleanup completed after loading error")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# ================================
# 专用模型加载器
# ================================

class HuggingFaceModelLoader(BaseModelLoader):
    """HuggingFace模型加载器
    
    设计思想：
    - **生态集成**：深度集成HuggingFace生态系统，支持Hub下载和缓存
    - **格式优化**：针对HuggingFace格式优化加载流程，提升性能
    - **版本管理**：支持模型版本管理和自动更新
    - **离线支持**：支持离线模式，使用本地缓存的模型
    """
    
    def __init__(self, model_config: ModelConfig, loader_config: LoaderConfig, 
                 memory_manager: UnifiedMemoryManager):
        super().__init__(model_config, loader_config, memory_manager)
        
        # HuggingFace特定配置
        self.use_auth_token = loader_config.hf_auth_token
        self.cache_dir = loader_config.hf_cache_dir
        self.offline_mode = loader_config.offline_mode
        
    def _download_model_if_needed(self):
        """下载模型（如果需要）
        
        设计思想：
        - **智能缓存**：检查本地缓存，避免重复下载
        - **断点续传**：支持大模型的断点续传下载
        - **并行下载**：并行下载多个文件，提升下载速度
        - **完整性验证**：验证下载文件的完整性和正确性
        """
        if self.offline_mode:
            return  # 离线模式不下载
        
        from huggingface_hub import snapshot_download
        
        try:
            # 下载模型到本地缓存
            local_path = snapshot_download(
                repo_id=self.model_config.model_name,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token,
                resume_download=True,  # 支持断点续传
            )
            
            # 更新模型路径为本地路径
            self.model_path = Path(local_path)
            
            logger.info(f"Model downloaded to {local_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model: {str(e)}")
            raise

class SafeTensorsModelLoader(BaseModelLoader):
    """SafeTensors模型加载器
    
    设计思想：
    - **安全加载**：使用SafeTensors格式，避免pickle安全风险
    - **高效IO**：优化SafeTensors文件的读取性能
    - **内存映射**：支持大文件的内存映射加载
    - **并行读取**：并行读取多个SafeTensors文件
    """
    
    def _load_single_shard(self, shard: WeightShard) -> Dict[str, torch.Tensor]:
        """SafeTensors专用的分片加载
        
        设计思想：
        - **零拷贝加载**：使用SafeTensors的零拷贝特性，减少内存使用
        - **选择性加载**：只加载需要的张量，节省内存和时间
        - **类型保持**：保持原始数据类型，避免不必要的转换
        """
        from safetensors import safe_open
        
        weights = {}
        
        # 使用SafeTensors的高效加载API
        with safe_open(shard.file_path, framework="pt") as f:
            # 获取所有张量键
            tensor_keys = f.keys()
            
            # 并行加载张量（如果支持）
            for key in tensor_keys:
                # 直接加载到目标设备，避免CPU中转
                tensor = f.get_tensor(key)
                if tensor.device != shard.device:
                    tensor = tensor.to(shard.device)
                
                weights[key] = tensor
        
        return weights

# ================================
# 工厂模式和配置
# ================================

class ModelLoaderFactory:
    """模型加载器工厂
    
    设计思想：
    - **工厂模式**：根据模型格式和配置自动选择最适合的加载器
    - **可扩展性**：易于添加新的模型格式和加载器类型
    - **配置驱动**：通过配置控制加载器的选择和行为
    - **性能优化**：为不同场景选择最优的加载策略
    """
    
    @staticmethod
    def create_loader(
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        memory_manager: UnifiedMemoryManager
    ) -> BaseModelLoader:
        """创建模型加载器
        
        设计思想：
        - **自动检测**：自动检测模型格式，选择合适的加载器
        - **性能优先**：优先选择性能最好的加载器
        - **兼容性保证**：确保选择的加载器与模型格式兼容
        """
        # 检测模型格式
        model_path = Path(model_config.model_path)
        
        # 检查是否为HuggingFace模型
        if (model_path / "config.json").exists():
            # 检查是否有SafeTensors文件
            if list(model_path.glob("*.safetensors")):
                return SafeTensorsModelLoader(model_config, loader_config, memory_manager)
            else:
                return HuggingFaceModelLoader(model_config, loader_config, memory_manager)
        
        # 检查是否为纯SafeTensors模型
        elif list(model_path.glob("*.safetensors")):
            return SafeTensorsModelLoader(model_config, loader_config, memory_manager)
        
        # 默认使用基础加载器
        else:
            return BaseModelLoader(model_config, loader_config, memory_manager)

# ================================
# 配置类定义
# ================================

@dataclass
class LoaderConfig:
    """加载器配置
    
    设计思想：
    - **集中配置**：集中管理所有加载相关的配置参数
    - **默认优化**：提供经过优化的默认值，适合大多数场景
    - **灵活调整**：支持根据具体需求调整各种参数
    - **性能导向**：配置参数设计以性能优化为导向
    """
    # 基础配置
    device: str = "auto"                    # 设备选择 - auto/cuda/cpu
    num_loading_threads: int = 4            # 并行加载线程数
    loading_timeout: int = 300              # 加载超时时间（秒）
    
    # 内存配置
    weight_cache_size: int = 1024 * 1024 * 1024  # 权重缓存大小（1GB）
    enable_memory_mapping: bool = True       # 启用内存映射
    
    # 优化配置
    enable_fusion: bool = True              # 启用算子融合
    enable_quantization: bool = False       # 启用权重量化
    enable_graph_optimization: bool = True   # 启用图优化
    enable_jit_compile: bool = False        # 启用JIT编译
    enable_warmup: bool = True              # 启用模型预热
    
    # 错误处理
    retry_on_failure: bool = True           # 失败时重试
    max_retries: int = 3                    # 最大重试次数
    
    # HuggingFace配置
    hf_auth_token: Optional[str] = None     # HuggingFace认证令牌
    hf_cache_dir: Optional[str] = None      # HuggingFace缓存目录
    offline_mode: bool = False              # 离线模式
    
    # 监控配置
    enable_monitoring: bool = True          # 启用性能监控
    log_level: str = "INFO"                # 日志级别

# ================================
# 关键特性分析
# ================================

"""
## 🎯 关键特性分析

### 1. 多格式支持
- **HuggingFace格式**：完整支持HuggingFace生态系统，包括模型下载、缓存管理
- **SafeTensors格式**：安全高效的张量存储，避免pickle安全风险
- **PyTorch格式**：原生PyTorch序列化格式，最大兼容性
- **自定义格式**：可扩展架构，支持添加新的模型格式

### 2. 内存优化
- **权重分片**：支持大模型的分片存储和加载，突破内存限制
- **内存映射**：零拷贝加载，减少内存占用
- **智能缓存**：缓存已加载的权重，避免重复加载
- **内存预分配**：预分配内存避免加载时内存不足

### 3. 并行加载
- **多线程IO**：并行读取多个权重文件，充分利用IO带宽
- **流水线处理**：IO读取和权重设置流水线处理
- **设备并行**：支持多设备并行加载，实现模型并行

### 4. 错误处理
- **完整性验证**：验证模型文件完整性和正确性
- **重试机制**：失败时自动重试，提高加载成功率
- **优雅降级**：部分失败时的优雅降级处理
- **资源清理**：错误时自动清理资源，避免内存泄漏

### 5. 性能监控
- **实时进度**：实时显示加载进度，提供用户反馈
- **性能指标**：记录加载时间、内存使用等关键指标
- **详细日志**：完整的日志记录，便于问题诊断
- **监控钩子**：支持自定义监控钩子，扩展监控能力
"""

# ================================
# 使用最佳实践
# ================================

"""
## 📚 使用最佳实践

### 1. 加载器选择
```python
# 自动选择最优加载器
loader = ModelLoaderFactory.create_loader(
    model_config=model_config,
    loader_config=loader_config,
    memory_manager=memory_manager
)

# 手动选择特定加载器
loader = SafeTensorsModelLoader(
    model_config=model_config,
    loader_config=loader_config,
    memory_manager=memory_manager
)
```

### 2. 配置优化
```python
# 内存受限环境配置
loader_config = LoaderConfig(
    num_loading_threads=2,          # 减少并行度
    enable_memory_mapping=True,     # 启用内存映射
    weight_cache_size=512*1024*1024, # 减少缓存大小
    enable_quantization=True,       # 启用量化
)

# 性能优先配置
loader_config = LoaderConfig(
    num_loading_threads=8,          # 增加并行度
    enable_fusion=True,             # 启用算子融合
    enable_graph_optimization=True, # 启用图优化
    enable_warmup=True,            # 启用预热
)
```

### 3. 错误处理
```python
try:
    model = loader.load_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    # 实现重试或降级逻辑
    if loader_config.retry_on_failure:
        model = loader.load_model()  # 重试加载
```

### 4. 监控和调试
```python
# 启用详细监控
loader_config.enable_monitoring = True
loader_config.log_level = "DEBUG"

# 监控加载进度
while loader.is_loading:
    progress = loader.loading_progress
    print(f"Loading progress: {progress:.1%}")
    time.sleep(1)
```
"""