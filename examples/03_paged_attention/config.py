#!/usr/bin/env python3
"""
第三步：PagedAttention与内存管理 - 配置文件

这个文件包含了PagedAttention和内存管理相关的配置参数。
"""

import torch
from typing import Dict, Any, List
from dataclasses import dataclass

# ================================
# PagedAttention配置
# ================================

@dataclass
class PagedAttentionConfig:
    """PagedAttention配置"""
    # 基础配置
    block_size: int = 16                    # 每个Block的token数量
    max_num_blocks_per_seq: int = 256       # 每个序列最大Block数
    max_num_seqs: int = 32                  # 最大并发序列数
    
    # 内存配置
    num_gpu_blocks: int = 1000              # GPU内存Block总数
    num_cpu_blocks: int = 1000              # CPU内存Block总数
    gpu_memory_utilization: float = 0.9     # GPU内存利用率
    
    # KV Cache配置
    kv_cache_dtype: torch.dtype = torch.float16  # KV Cache数据类型
    head_size: int = 64                     # 注意力头维度
    num_heads: int = 12                     # 注意力头数量
    num_kv_heads: int = 12                  # KV头数量（GQA支持）
    
    # Copy-on-Write配置
    enable_cow: bool = True                 # 启用Copy-on-Write优化
    cow_threshold: int = 4                  # COW触发阈值（共享引用数）
    
    # 内存管理策略
    eviction_policy: str = "lru"            # 驱逐策略: "lru", "fifo", "random"
    swap_space_size: int = 2048             # 交换空间大小(MB)
    enable_memory_pool: bool = True         # 启用内存池
    
    # 性能优化
    enable_attention_optimization: bool = True   # 启用注意力优化
    use_flash_attention: bool = False           # 使用Flash Attention
    attention_backend: str = "torch"            # 注意力后端: "torch", "xformers"
    
    # 调试配置
    debug_memory: bool = False              # 内存调试模式
    log_block_allocation: bool = True       # 记录Block分配
    validate_block_table: bool = False      # 验证Block Table

# ================================
# 预设配置
# ================================

# 默认配置
DEFAULT_PAGED_ATTENTION_CONFIG = PagedAttentionConfig()

# 开发测试配置（小规模，便于调试）
DEV_PAGED_ATTENTION_CONFIG = PagedAttentionConfig(
    block_size=8,
    max_num_blocks_per_seq=64,
    max_num_seqs=4,
    num_gpu_blocks=100,
    num_cpu_blocks=100,
    kv_cache_dtype=torch.float32,
    debug_memory=True,
    log_block_allocation=True,
    validate_block_table=True,
)

# 生产环境配置（高性能，大规模）
PROD_PAGED_ATTENTION_CONFIG = PagedAttentionConfig(
    block_size=16,
    max_num_blocks_per_seq=512,
    max_num_seqs=64,
    num_gpu_blocks=4000,
    num_cpu_blocks=8000,
    kv_cache_dtype=torch.float16,
    enable_cow=True,
    cow_threshold=8,
    eviction_policy="lru",
    enable_attention_optimization=True,
    debug_memory=False,
    log_block_allocation=False,
)

# 内存优化配置（适用于显存较小的环境）
MEMORY_OPTIMIZED_CONFIG = PagedAttentionConfig(
    block_size=8,
    max_num_blocks_per_seq=128,
    max_num_seqs=16,
    num_gpu_blocks=500,
    num_cpu_blocks=2000,
    gpu_memory_utilization=0.8,
    kv_cache_dtype=torch.float16,
    enable_cow=True,
    cow_threshold=4,
    swap_space_size=4096,
    enable_memory_pool=True,
)

# 高性能配置（适用于显存充足的环境）
HIGH_PERFORMANCE_CONFIG = PagedAttentionConfig(
    block_size=32,
    max_num_blocks_per_seq=1024,
    max_num_seqs=128,
    num_gpu_blocks=8000,
    num_cpu_blocks=4000,
    gpu_memory_utilization=0.95,
    kv_cache_dtype=torch.float16,
    enable_cow=True,
    cow_threshold=16,
    enable_attention_optimization=True,
    use_flash_attention=True,
    attention_backend="xformers",
)

# ================================
# 配置管理函数
# ================================

def get_paged_attention_config(config_name: str = "default") -> PagedAttentionConfig:
    """
    获取指定的PagedAttention配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        PagedAttentionConfig对象
    """
    configs = {
        "default": DEFAULT_PAGED_ATTENTION_CONFIG,
        "dev": DEV_PAGED_ATTENTION_CONFIG,
        "prod": PROD_PAGED_ATTENTION_CONFIG,
        "memory_optimized": MEMORY_OPTIMIZED_CONFIG,
        "high_performance": HIGH_PERFORMANCE_CONFIG,
    }
    
    if config_name not in configs:
        print(f"⚠️  未知配置名称: {config_name}，使用默认配置")
        return DEFAULT_PAGED_ATTENTION_CONFIG
    
    return configs[config_name]

def print_paged_attention_config(config: PagedAttentionConfig):
    """打印PagedAttention配置信息"""
    print("🧱 PagedAttention配置:")
    print(f"   📦 Block大小: {config.block_size} tokens")
    print(f"   🔢 最大序列Block数: {config.max_num_blocks_per_seq}")
    print(f"   🎯 最大并发序列: {config.max_num_seqs}")
    print(f"   💾 GPU Block数: {config.num_gpu_blocks}")
    print(f"   💿 CPU Block数: {config.num_cpu_blocks}")
    print(f"   📊 KV Cache类型: {config.kv_cache_dtype}")
    print(f"   🔄 COW优化: {config.enable_cow}")
    print(f"   📈 驱逐策略: {config.eviction_policy}")
    print(f"   ⚡ 注意力优化: {config.enable_attention_optimization}")
    print(f"   🔍 调试模式: {config.debug_memory}")

def validate_paged_attention_config(config: PagedAttentionConfig) -> bool:
    """
    验证PagedAttention配置的合理性
    
    Args:
        config: PagedAttention配置
        
    Returns:
        是否有效
    """
    print("🔍 验证PagedAttention配置...")
    
    # 基础验证
    if config.block_size <= 0 or config.block_size > 128:
        print("❌ block_size 必须在 (0, 128] 范围内")
        return False
    
    if config.max_num_blocks_per_seq <= 0:
        print("❌ max_num_blocks_per_seq 必须大于 0")
        return False
    
    if config.max_num_seqs <= 0:
        print("❌ max_num_seqs 必须大于 0")
        return False
    
    if config.num_gpu_blocks < 0 or config.num_cpu_blocks < 0:
        print("❌ Block数量不能为负数")
        return False
    
    if not (0 <= config.gpu_memory_utilization <= 1):
        print("❌ gpu_memory_utilization 必须在 [0, 1] 范围内")
        return False
    
    # 内存容量检查
    total_blocks_needed = config.max_num_seqs * config.max_num_blocks_per_seq
    total_blocks_available = config.num_gpu_blocks + config.num_cpu_blocks
    
    if total_blocks_needed > total_blocks_available:
        print(f"⚠️  总Block需求 ({total_blocks_needed}) 超过可用Block数 ({total_blocks_available})")
        print("   这可能导致内存不足，建议调整配置")
    
    # 驱逐策略验证
    valid_policies = ["lru", "fifo", "random"]
    if config.eviction_policy not in valid_policies:
        print(f"❌ 无效的驱逐策略: {config.eviction_policy}，支持: {valid_policies}")
        return False
    
    # 注意力后端验证
    valid_backends = ["torch", "xformers"]
    if config.attention_backend not in valid_backends:
        print(f"❌ 无效的注意力后端: {config.attention_backend}，支持: {valid_backends}")
        return False
    
    print("✅ PagedAttention配置验证通过")
    return True

def estimate_memory_usage(config: PagedAttentionConfig) -> Dict[str, float]:
    """
    估算内存使用量
    
    Args:
        config: PagedAttention配置
        
    Returns:
        内存使用估算（单位：MB）
    """
    # KV Cache内存计算
    # 每个token的KV Cache大小 = 2 * num_layers * head_size * num_kv_heads * dtype_size
    # 简化计算，假设24层，dtype_size根据数据类型确定
    
    dtype_size = 2 if config.kv_cache_dtype == torch.float16 else 4  # bytes
    num_layers = 24  # 假设值
    
    kv_size_per_token = 2 * num_layers * config.head_size * config.num_kv_heads * dtype_size
    kv_size_per_block = kv_size_per_token * config.block_size
    
    gpu_kv_memory = config.num_gpu_blocks * kv_size_per_block / 1024 / 1024  # MB
    cpu_kv_memory = config.num_cpu_blocks * kv_size_per_block / 1024 / 1024  # MB
    
    # Block Table内存（简化估算）
    block_table_memory = config.max_num_seqs * config.max_num_blocks_per_seq * 4 / 1024 / 1024  # MB
    
    return {
        "gpu_kv_cache": gpu_kv_memory,
        "cpu_kv_cache": cpu_kv_memory,
        "block_table": block_table_memory,
        "total": gpu_kv_memory + cpu_kv_memory + block_table_memory
    }

# ================================
# 自动配置函数
# ================================

def auto_configure_paged_attention() -> PagedAttentionConfig:
    """
    根据当前环境自动配置PagedAttention
    
    Returns:
        自动配置的PagedAttentionConfig
    """
    print("🤖 自动配置PagedAttention...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"   🎯 检测到GPU显存: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 16:
            print("   📈 使用高性能配置")
            return get_paged_attention_config("high_performance")
        elif gpu_memory >= 8:
            print("   📊 使用生产配置")
            return get_paged_attention_config("prod")
        else:
            print("   💾 使用内存优化配置")
            return get_paged_attention_config("memory_optimized")
    else:
        print("   💻 CPU环境，使用开发配置")
        config = get_paged_attention_config("dev")
        config.num_gpu_blocks = 0
        config.kv_cache_dtype = torch.float32
        return config

if __name__ == "__main__":
    print("=" * 60)
    print("🧱 PagedAttention配置管理")
    print("=" * 60)
    
    # 显示所有预设配置
    config_names = ["default", "dev", "prod", "memory_optimized", "high_performance"]
    
    for name in config_names:
        print(f"\n📋 {name.upper()} 配置:")
        config = get_paged_attention_config(name)
        print_paged_attention_config(config)
        
        if validate_paged_attention_config(config):
            memory_usage = estimate_memory_usage(config)
            print(f"   💾 内存估算: GPU={memory_usage['gpu_kv_cache']:.1f}MB, "
                  f"CPU={memory_usage['cpu_kv_cache']:.1f}MB, "
                  f"总计={memory_usage['total']:.1f}MB")
    
    # 自动配置演示
    print(f"\n🤖 自动配置演示:")
    auto_config = auto_configure_paged_attention()
    print_paged_attention_config(auto_config)