#!/usr/bin/env python3
"""
第四步：智能调度器与请求管理 - 配置文件

这个文件包含了调度器相关的配置参数，包括调度策略、内存管理、性能优化等。
"""

import torch
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# ================================
# 调度策略枚举
# ================================

class SchedulerPolicy(Enum):
    """调度策略"""
    FCFS = "fcfs"           # 先来先服务
    SJF = "sjf"             # 最短作业优先
    PRIORITY = "priority"   # 优先级调度
    ROUND_ROBIN = "rr"      # 轮转调度
    FAIR_SHARE = "fair"     # 公平共享

class PreemptionMode(Enum):
    """抢占模式"""
    SWAP = "swap"           # 换出到CPU
    RECOMPUTE = "recompute" # 重新计算

# ================================
# 调度器配置
# ================================

@dataclass
class SchedulerConfig:
    """调度器配置"""
    # 基础调度配置
    max_num_seqs: int = 32                      # 最大并发序列数
    max_batch_size: int = 8                     # 最大批处理大小
    max_waiting_time: float = 300.0             # 最大等待时间(秒)
    
    # 调度策略配置
    policy: SchedulerPolicy = SchedulerPolicy.FCFS  # 调度策略
    enable_preemption: bool = True              # 启用抢占
    preemption_mode: PreemptionMode = PreemptionMode.SWAP  # 抢占模式
    
    # 内存管理配置
    memory_threshold: float = 0.9               # 内存使用阈值
    swap_threshold: float = 0.8                 # 换出阈值
    min_free_blocks: int = 100                  # 最小空闲Block数
    
    # 优先级配置
    enable_priority: bool = True                # 启用优先级调度
    default_priority: int = 0                   # 默认优先级
    priority_levels: int = 10                   # 优先级级别数
    priority_boost_interval: float = 60.0      # 优先级提升间隔(秒)
    
    # 公平性配置
    enable_fairness: bool = False               # 启用公平调度
    fairness_window: float = 300.0              # 公平性时间窗口(秒)
    max_user_share: float = 0.5                 # 单用户最大资源占用比例
    
    # 性能优化配置
    enable_continuous_batching: bool = True     # 启用连续批处理
    enable_chunked_prefill: bool = True         # 启用分块预填充
    max_prefill_tokens: int = 4096              # 最大预填充token数
    chunk_size: int = 512                       # 分块大小
    
    # 延迟优化配置
    target_latency: float = 1.0                 # 目标延迟(秒)
    latency_sla: float = 5.0                    # 延迟SLA(秒)
    enable_speculative_decoding: bool = False   # 启用推测解码
    
    # 吞吐量优化配置
    target_throughput: float = 100.0            # 目标吞吐量(tokens/s)
    enable_dynamic_batching: bool = True        # 启用动态批处理
    batch_expansion_factor: float = 1.5         # 批处理扩展因子
    
    # 监控配置
    enable_metrics: bool = True                 # 启用指标收集
    metrics_interval: float = 10.0              # 指标收集间隔(秒)
    log_scheduling_decisions: bool = False      # 记录调度决策
    
    # 调试配置
    debug_mode: bool = False                    # 调试模式
    verbose_logging: bool = True                # 详细日志

# ================================
# 预设配置
# ================================

# 默认配置
DEFAULT_SCHEDULER_CONFIG = SchedulerConfig()

# 开发测试配置
DEV_SCHEDULER_CONFIG = SchedulerConfig(
    max_num_seqs=4,
    max_batch_size=2,
    policy=SchedulerPolicy.FCFS,
    enable_preemption=True,
    memory_threshold=0.8,
    enable_continuous_batching=False,
    debug_mode=True,
    verbose_logging=True,
    log_scheduling_decisions=True,
)

# 生产环境配置（高吞吐量）
PROD_HIGH_THROUGHPUT_CONFIG = SchedulerConfig(
    max_num_seqs=64,
    max_batch_size=16,
    policy=SchedulerPolicy.PRIORITY,
    enable_preemption=True,
    preemption_mode=PreemptionMode.SWAP,
    memory_threshold=0.95,
    enable_priority=True,
    enable_fairness=True,
    enable_continuous_batching=True,
    enable_chunked_prefill=True,
    enable_dynamic_batching=True,
    target_throughput=500.0,
    debug_mode=False,
    verbose_logging=False,
)

# 生产环境配置（低延迟）
PROD_LOW_LATENCY_CONFIG = SchedulerConfig(
    max_num_seqs=16,
    max_batch_size=4,
    policy=SchedulerPolicy.SJF,
    enable_preemption=True,
    preemption_mode=PreemptionMode.RECOMPUTE,
    memory_threshold=0.85,
    target_latency=0.5,
    latency_sla=2.0,
    enable_speculative_decoding=True,
    enable_continuous_batching=True,
    chunk_size=256,
    debug_mode=False,
    verbose_logging=False,
)

# 公平调度配置
FAIR_SCHEDULER_CONFIG = SchedulerConfig(
    max_num_seqs=32,
    max_batch_size=8,
    policy=SchedulerPolicy.FAIR_SHARE,
    enable_preemption=True,
    enable_priority=True,
    enable_fairness=True,
    fairness_window=600.0,
    max_user_share=0.3,
    priority_boost_interval=120.0,
    enable_continuous_batching=True,
)

# 内存优化配置
MEMORY_OPTIMIZED_SCHEDULER_CONFIG = SchedulerConfig(
    max_num_seqs=16,
    max_batch_size=4,
    policy=SchedulerPolicy.PRIORITY,
    enable_preemption=True,
    preemption_mode=PreemptionMode.SWAP,
    memory_threshold=0.75,
    swap_threshold=0.6,
    min_free_blocks=200,
    enable_continuous_batching=False,
    enable_chunked_prefill=True,
    chunk_size=256,
)

# ================================
# 配置管理函数
# ================================

def get_scheduler_config(config_name: str = "default") -> SchedulerConfig:
    """
    获取指定的调度器配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        SchedulerConfig对象
    """
    configs = {
        "default": DEFAULT_SCHEDULER_CONFIG,
        "dev": DEV_SCHEDULER_CONFIG,
        "prod_throughput": PROD_HIGH_THROUGHPUT_CONFIG,
        "prod_latency": PROD_LOW_LATENCY_CONFIG,
        "fair": FAIR_SCHEDULER_CONFIG,
        "memory_optimized": MEMORY_OPTIMIZED_SCHEDULER_CONFIG,
    }
    
    if config_name not in configs:
        print(f"⚠️  未知配置名称: {config_name}，使用默认配置")
        return DEFAULT_SCHEDULER_CONFIG
    
    return configs[config_name]

def print_scheduler_config(config: SchedulerConfig):
    """打印调度器配置信息"""
    print("📋 调度器配置:")
    print(f"   🎯 最大序列数: {config.max_num_seqs}")
    print(f"   📦 最大批处理: {config.max_batch_size}")
    print(f"   🔄 调度策略: {config.policy.value}")
    print(f"   ⚡ 抢占模式: {config.preemption_mode.value if config.enable_preemption else '禁用'}")
    print(f"   💾 内存阈值: {config.memory_threshold:.1%}")
    print(f"   🏆 优先级调度: {'启用' if config.enable_priority else '禁用'}")
    print(f"   ⚖️  公平调度: {'启用' if config.enable_fairness else '禁用'}")
    print(f"   🔄 连续批处理: {'启用' if config.enable_continuous_batching else '禁用'}")
    print(f"   📊 目标延迟: {config.target_latency:.1f}s")
    print(f"   🚀 目标吞吐量: {config.target_throughput:.0f} tokens/s")

def validate_scheduler_config(config: SchedulerConfig) -> bool:
    """
    验证调度器配置的合理性
    
    Args:
        config: 调度器配置
        
    Returns:
        是否有效
    """
    print("🔍 验证调度器配置...")
    
    # 基础验证
    if config.max_num_seqs <= 0:
        print("❌ max_num_seqs 必须大于 0")
        return False
    
    if config.max_batch_size <= 0 or config.max_batch_size > config.max_num_seqs:
        print("❌ max_batch_size 必须在 (0, max_num_seqs] 范围内")
        return False
    
    if not (0 <= config.memory_threshold <= 1):
        print("❌ memory_threshold 必须在 [0, 1] 范围内")
        return False
    
    if not (0 <= config.swap_threshold <= config.memory_threshold):
        print("❌ swap_threshold 必须在 [0, memory_threshold] 范围内")
        return False
    
    if config.target_latency <= 0:
        print("❌ target_latency 必须大于 0")
        return False
    
    if config.target_throughput <= 0:
        print("❌ target_throughput 必须大于 0")
        return False
    
    # 逻辑一致性检查
    if config.enable_fairness and not config.enable_priority:
        print("⚠️  启用公平调度建议同时启用优先级调度")
    
    if config.policy == SchedulerPolicy.PRIORITY and not config.enable_priority:
        print("⚠️  使用优先级调度策略但未启用优先级功能")
    
    if config.enable_speculative_decoding and config.policy != SchedulerPolicy.SJF:
        print("⚠️  推测解码通常与SJF调度策略配合使用效果更好")
    
    print("✅ 调度器配置验证通过")
    return True

def optimize_config_for_workload(
    config: SchedulerConfig,
    avg_sequence_length: int,
    request_rate: float,
    latency_sensitive: bool = False
) -> SchedulerConfig:
    """
    根据工作负载特征优化配置
    
    Args:
        config: 基础配置
        avg_sequence_length: 平均序列长度
        request_rate: 请求速率 (requests/s)
        latency_sensitive: 是否对延迟敏感
        
    Returns:
        优化后的配置
    """
    print(f"🎯 根据工作负载优化配置...")
    print(f"   📏 平均序列长度: {avg_sequence_length}")
    print(f"   📈 请求速率: {request_rate:.1f} req/s")
    print(f"   ⏱️  延迟敏感: {latency_sensitive}")
    
    optimized_config = config
    
    # 根据序列长度调整
    if avg_sequence_length > 2048:
        print("   📊 长序列优化: 减少并发数，增加批处理大小")
        optimized_config.max_num_seqs = min(config.max_num_seqs, 16)
        optimized_config.max_batch_size = min(config.max_batch_size * 2, optimized_config.max_num_seqs)
        optimized_config.enable_chunked_prefill = True
        optimized_config.chunk_size = 1024
    elif avg_sequence_length < 512:
        print("   📊 短序列优化: 增加并发数，启用动态批处理")
        optimized_config.max_num_seqs = min(config.max_num_seqs * 2, 64)
        optimized_config.enable_dynamic_batching = True
        optimized_config.batch_expansion_factor = 2.0
    
    # 根据请求速率调整
    if request_rate > 10:
        print("   📈 高请求速率优化: 启用连续批处理和抢占")
        optimized_config.enable_continuous_batching = True
        optimized_config.enable_preemption = True
        optimized_config.policy = SchedulerPolicy.PRIORITY
    elif request_rate < 1:
        print("   📉 低请求速率优化: 简化调度策略")
        optimized_config.policy = SchedulerPolicy.FCFS
        optimized_config.enable_preemption = False
    
    # 根据延迟敏感性调整
    if latency_sensitive:
        print("   ⏱️  延迟优化: 使用SJF策略，启用推测解码")
        optimized_config.policy = SchedulerPolicy.SJF
        optimized_config.preemption_mode = PreemptionMode.RECOMPUTE
        optimized_config.enable_speculative_decoding = True
        optimized_config.target_latency = 0.5
        optimized_config.max_batch_size = min(config.max_batch_size, 4)
    
    return optimized_config

# ================================
# 自动配置函数
# ================================

def auto_configure_scheduler() -> SchedulerConfig:
    """
    根据当前环境自动配置调度器
    
    Returns:
        自动配置的SchedulerConfig
    """
    print("🤖 自动配置调度器...")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        gpu_count = torch.cuda.device_count()
        
        print(f"   🎯 检测到 {gpu_count} 个GPU，显存: {gpu_memory:.1f}GB")
        
        if gpu_memory >= 24 and gpu_count >= 2:
            print("   🚀 高性能环境，使用高吞吐量配置")
            return get_scheduler_config("prod_throughput")
        elif gpu_memory >= 16:
            print("   📊 中等性能环境，使用生产配置")
            return get_scheduler_config("prod_latency")
        elif gpu_memory >= 8:
            print("   💾 标准环境，使用默认配置")
            return get_scheduler_config("default")
        else:
            print("   🔧 资源受限环境，使用内存优化配置")
            return get_scheduler_config("memory_optimized")
    else:
        print("   💻 CPU环境，使用开发配置")
        return get_scheduler_config("dev")

if __name__ == "__main__":
    print("=" * 60)
    print("📋 调度器配置管理")
    print("=" * 60)
    
    # 显示所有预设配置
    config_names = ["default", "dev", "prod_throughput", "prod_latency", "fair", "memory_optimized"]
    
    for name in config_names:
        print(f"\n📋 {name.upper()} 配置:")
        config = get_scheduler_config(name)
        print_scheduler_config(config)
        validate_scheduler_config(config)
    
    # 自动配置演示
    print(f"\n🤖 自动配置演示:")
    auto_config = auto_configure_scheduler()
    print_scheduler_config(auto_config)
    
    # 工作负载优化演示
    print(f"\n🎯 工作负载优化演示:")
    base_config = get_scheduler_config("default")
    optimized_config = optimize_config_for_workload(
        base_config, 
        avg_sequence_length=1024, 
        request_rate=5.0, 
        latency_sensitive=True
    )
    print_scheduler_config(optimized_config)