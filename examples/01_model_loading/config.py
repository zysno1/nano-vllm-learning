#!/usr/bin/env python3
"""
第一步：模型加载与基础配置 - 配置文件

这个文件包含了模型加载相关的配置参数，方便用户根据自己的环境进行调整。
"""

import torch
from typing import Dict, Any

# ================================
# 基础配置
# ================================

# 默认模型配置
DEFAULT_MODEL_CONFIG = {
    # 模型路径（可以是本地路径或HuggingFace模型名）
    "model_path": "microsoft/DialoGPT-small",  # 使用较小的模型进行演示
    
    # 设备配置
    "device": "auto",  # "auto", "cuda", "cpu"
    
    # 数据类型配置
    "torch_dtype": torch.float16,  # torch.float32, torch.float16, torch.bfloat16
    
    # 内存优化配置
    "use_cache": True,           # 是否启用KV缓存
    "low_cpu_mem_usage": True,   # 是否使用低CPU内存模式
    
    # 加载配置
    "trust_remote_code": True,   # 是否信任远程代码
    "use_fast_tokenizer": True,  # 是否使用快速分词器
}

# ================================
# 不同场景的预设配置
# ================================

# 开发测试配置（小模型，快速加载）
DEV_CONFIG = {
    **DEFAULT_MODEL_CONFIG,
    "model_path": "microsoft/DialoGPT-small",
    "torch_dtype": torch.float32,
    "low_cpu_mem_usage": False,
}

# 生产环境配置（优化内存和性能）
PROD_CONFIG = {
    **DEFAULT_MODEL_CONFIG,
    "model_path": "microsoft/DialoGPT-medium",
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
}

# GPU配置（充分利用GPU内存）
GPU_CONFIG = {
    **DEFAULT_MODEL_CONFIG,
    "device": "cuda",
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
}

# CPU配置（仅使用CPU）
CPU_CONFIG = {
    **DEFAULT_MODEL_CONFIG,
    "device": "cpu",
    "torch_dtype": torch.float32,
    "low_cpu_mem_usage": False,
}

# ================================
# 配置选择函数
# ================================

def get_config(config_name: str = "default") -> Dict[str, Any]:
    """
    获取指定的配置
    
    Args:
        config_name: 配置名称 ("default", "dev", "prod", "gpu", "cpu")
        
    Returns:
        配置字典
    """
    configs = {
        "default": DEFAULT_MODEL_CONFIG,
        "dev": DEV_CONFIG,
        "prod": PROD_CONFIG,
        "gpu": GPU_CONFIG,
        "cpu": CPU_CONFIG,
    }
    
    if config_name not in configs:
        print(f"⚠️  未知配置名称: {config_name}，使用默认配置")
        return DEFAULT_MODEL_CONFIG
    
    return configs[config_name].copy()

def print_config(config: Dict[str, Any]):
    """打印配置信息"""
    print("📋 当前配置:")
    for key, value in config.items():
        print(f"   {key}: {value}")

# ================================
# 环境检查函数
# ================================

def check_environment():
    """检查运行环境"""
    print("🔍 环境检查:")
    
    # 检查PyTorch
    print(f"   PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA可用: {cuda_available}")
    
    if cuda_available:
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 内存检查
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    except ImportError:
        print("   系统内存: 无法检测 (需要安装psutil)")

if __name__ == "__main__":
    print("=" * 50)
    print("🎯 模型加载配置文件")
    print("=" * 50)
    
    # 环境检查
    check_environment()
    print()
    
    # 显示所有配置
    config_names = ["default", "dev", "prod", "gpu", "cpu"]
    
    for name in config_names:
        print(f"\n📋 {name.upper()} 配置:")
        config = get_config(name)
        print_config(config)