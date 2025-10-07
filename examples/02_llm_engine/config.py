#!/usr/bin/env python3
"""
第二步：LLM引擎架构与初始化 - 配置文件

这个文件包含了LLM引擎相关的配置参数，包括模型配置、调度器配置、内存管理配置等。
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass

# ================================
# 引擎配置数据类
# ================================

@dataclass
class EngineConfig:
    """LLM引擎配置"""
    # 模型配置
    model_name: str = "microsoft/DialoGPT-small"
    device: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    
    # 调度器配置
    max_num_seqs: int = 32          # 最大并发序列数
    max_seq_len: int = 2048         # 最大序列长度
    max_batch_size: int = 8         # 最大批处理大小
    
    # 内存管理配置
    block_size: int = 16            # KV Cache块大小
    num_gpu_blocks: int = 1000      # GPU内存块数量
    num_cpu_blocks: int = 1000      # CPU内存块数量
    gpu_memory_utilization: float = 0.9  # GPU内存利用率
    
    # 生成配置
    max_tokens: int = 100           # 默认最大生成token数
    temperature: float = 1.0        # 采样温度
    top_p: float = 1.0             # Top-p采样
    top_k: int = -1                # Top-k采样
    
    # 性能配置
    enable_chunked_prefill: bool = True    # 启用分块预填充
    max_prefill_tokens: int = 4096         # 最大预填充token数
    enable_prefix_caching: bool = False    # 启用前缀缓存
    
    # 调试配置
    verbose: bool = True            # 详细输出
    log_level: str = "INFO"        # 日志级别

# ================================
# 预设配置
# ================================

# 默认配置
DEFAULT_ENGINE_CONFIG = EngineConfig()

# 开发测试配置（小规模，快速测试）
DEV_ENGINE_CONFIG = EngineConfig(
    model_name="microsoft/DialoGPT-small",
    max_num_seqs=4,
    max_seq_len=512,
    max_batch_size=2,
    num_gpu_blocks=100,
    num_cpu_blocks=100,
    torch_dtype=torch.float32,
    verbose=True,
)

# 生产环境配置（高性能，大规模）
PROD_ENGINE_CONFIG = EngineConfig(
    model_name="microsoft/DialoGPT-medium",
    max_num_seqs=64,
    max_seq_len=4096,
    max_batch_size=16,
    num_gpu_blocks=2000,
    num_cpu_blocks=2000,
    torch_dtype=torch.float16,
    gpu_memory_utilization=0.85,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    verbose=False,
)

# GPU优化配置
GPU_ENGINE_CONFIG = EngineConfig(
    device="cuda",
    torch_dtype=torch.float16,
    max_num_seqs=32,
    num_gpu_blocks=1500,
    gpu_memory_utilization=0.9,
    enable_chunked_prefill=True,
)

# CPU配置
CPU_ENGINE_CONFIG = EngineConfig(
    device="cpu",
    torch_dtype=torch.float32,
    max_num_seqs=8,
    max_batch_size=4,
    num_gpu_blocks=0,
    num_cpu_blocks=500,
    gpu_memory_utilization=0.0,
    enable_chunked_prefill=False,
)

# ================================
# 配置管理函数
# ================================

def get_engine_config(config_name: str = "default") -> EngineConfig:
    """
    获取指定的引擎配置
    
    Args:
        config_name: 配置名称 ("default", "dev", "prod", "gpu", "cpu")
        
    Returns:
        EngineConfig对象
    """
    configs = {
        "default": DEFAULT_ENGINE_CONFIG,
        "dev": DEV_ENGINE_CONFIG,
        "prod": PROD_ENGINE_CONFIG,
        "gpu": GPU_ENGINE_CONFIG,
        "cpu": CPU_ENGINE_CONFIG,
    }
    
    if config_name not in configs:
        print(f"⚠️  未知配置名称: {config_name}，使用默认配置")
        return DEFAULT_ENGINE_CONFIG
    
    return configs[config_name]

def print_engine_config(config: EngineConfig):
    """打印引擎配置信息"""
    print("🔧 LLM引擎配置:")
    print(f"   📦 模型: {config.model_name}")
    print(f"   🎯 设备: {config.device}")
    print(f"   📊 数据类型: {config.torch_dtype}")
    print(f"   🔢 最大序列数: {config.max_num_seqs}")
    print(f"   📏 最大序列长度: {config.max_seq_len}")
    print(f"   📦 批处理大小: {config.max_batch_size}")
    print(f"   🧱 块大小: {config.block_size}")
    print(f"   💾 GPU块数: {config.num_gpu_blocks}")
    print(f"   💿 CPU块数: {config.num_cpu_blocks}")
    print(f"   🎛️  温度: {config.temperature}")
    print(f"   🔝 Top-p: {config.top_p}")
    print(f"   🔢 Top-k: {config.top_k}")

def validate_config(config: EngineConfig) -> bool:
    """
    验证配置的合理性
    
    Args:
        config: 引擎配置
        
    Returns:
        是否有效
    """
    print("🔍 验证引擎配置...")
    
    # 基础验证
    if config.max_num_seqs <= 0:
        print("❌ max_num_seqs 必须大于 0")
        return False
    
    if config.max_seq_len <= 0:
        print("❌ max_seq_len 必须大于 0")
        return False
    
    if config.block_size <= 0:
        print("❌ block_size 必须大于 0")
        return False
    
    if config.temperature <= 0:
        print("❌ temperature 必须大于 0")
        return False
    
    if not (0 <= config.top_p <= 1):
        print("❌ top_p 必须在 [0, 1] 范围内")
        return False
    
    if not (0 <= config.gpu_memory_utilization <= 1):
        print("❌ gpu_memory_utilization 必须在 [0, 1] 范围内")
        return False
    
    # 设备相关验证
    if config.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  指定使用CUDA但CUDA不可用，将自动切换到CPU")
        config.device = "cpu"
        config.torch_dtype = torch.float32
        config.num_gpu_blocks = 0
    
    # 内存配置验证
    if config.device == "cpu" and config.num_gpu_blocks > 0:
        print("⚠️  CPU模式下GPU块数应为0")
        config.num_gpu_blocks = 0
    
    print("✅ 配置验证通过")
    return True

# ================================
# 环境适配函数
# ================================

def auto_configure() -> EngineConfig:
    """
    根据当前环境自动选择最佳配置
    
    Returns:
        自动配置的EngineConfig
    """
    print("🤖 自动配置引擎...")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"   🎯 检测到 {gpu_count} 个GPU，显存: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 8:
            print("   📈 使用高性能GPU配置")
            return get_engine_config("gpu")
        else:
            print("   📊 使用开发GPU配置")
            config = get_engine_config("dev")
            config.device = "cuda"
            config.torch_dtype = torch.float16
            return config
    else:
        print("   💻 使用CPU配置")
        return get_engine_config("cpu")

if __name__ == "__main__":
    print("=" * 60)
    print("🔧 LLM引擎配置管理")
    print("=" * 60)
    
    # 显示所有预设配置
    config_names = ["default", "dev", "prod", "gpu", "cpu"]
    
    for name in config_names:
        print(f"\n📋 {name.upper()} 配置:")
        config = get_engine_config(name)
        print_engine_config(config)
        validate_config(config)
    
    # 自动配置演示
    print(f"\n🤖 自动配置演示:")
    auto_config = auto_configure()
    print_engine_config(auto_config)