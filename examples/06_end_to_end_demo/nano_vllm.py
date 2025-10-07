#!/usr/bin/env python3
"""
NanoVLLM - 完整的轻量级vLLM推理引擎

这个文件整合了前面所有步骤的功能，提供一个完整可用的推理系统。
包含：模型加载、内存管理、调度器、PagedAttention等核心功能。
"""

import os
import time
import torch
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
from contextlib import contextmanager

# 导入transformers相关
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        GenerationConfig, StoppingCriteria, StoppingCriteriaList
    )
except ImportError:
    print("❌ 请安装transformers: pip install transformers")
    raise

# ================================
# 核心数据结构和枚举
# ================================

class RequestStatus(Enum):
    """请求状态"""
    WAITING = "waiting"      # 等待调度
    RUNNING = "running"      # 正在执行
    SWAPPED = "swapped"      # 已交换到CPU
    FINISHED = "finished"    # 已完成
    FAILED = "failed"        # 执行失败

class BlockStatus(Enum):
    """内存块状态"""
    FREE = "free"           # 空闲
    ALLOCATED = "allocated" # 已分配
    SWAPPED = "swapped"     # 已交换

class SchedulerPolicy(Enum):
    """调度策略"""
    FCFS = "fcfs"          # 先来先服务
    PRIORITY = "priority"   # 优先级调度
    SJF = "sjf"            # 最短作业优先

@dataclass
class GenerationParams:
    """生成参数"""
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False

@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    prompt: str
    params: GenerationParams
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0
    status: RequestStatus = RequestStatus.WAITING
    
    # 内部状态
    prompt_tokens: List[int] = field(default_factory=list)
    generated_tokens: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    num_blocks: int = 0
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    generated_text: str
    generated_tokens: List[int]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    generation_time: float
    tokens_per_second: float

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: float = field(default_factory=time.time)
    
    # 吞吐量指标
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # 延迟指标
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # 内存指标
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_memory_utilization: float = 0.0
    kv_cache_usage: float = 0.0
    
    # 队列指标
    waiting_requests: int = 0
    running_requests: int = 0
    swapped_requests: int = 0
    
    # 错误指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    error_rate: float = 0.0

# ================================
# 内存管理组件
# ================================

class Block:
    """KV Cache内存块"""
    
    def __init__(self, block_id: int, block_size: int, device: str = "cuda"):
        self.block_id = block_id
        self.block_size = block_size
        self.device = device
        self.status = BlockStatus.FREE
        self.ref_count = 0
        self.data = None
        self.last_accessed = time.time()
        
        print(f"🧱 创建Block {block_id} (大小: {block_size}, 设备: {device})")
    
    def allocate(self):
        """分配内存块"""
        if self.status != BlockStatus.FREE:
            raise RuntimeError(f"Block {self.block_id} 不是空闲状态")
        
        self.status = BlockStatus.ALLOCATED
        self.ref_count = 1
        self.last_accessed = time.time()
        
        print(f"📦 分配Block {self.block_id} (引用计数: {self.ref_count})")
    
    def add_ref(self):
        """增加引用计数"""
        if self.status != BlockStatus.ALLOCATED:
            raise RuntimeError(f"Block {self.block_id} 未分配")
        
        self.ref_count += 1
        self.last_accessed = time.time()
        
        print(f"📈 Block {self.block_id} 引用计数增加到 {self.ref_count}")
    
    def remove_ref(self):
        """减少引用计数"""
        if self.ref_count <= 0:
            raise RuntimeError(f"Block {self.block_id} 引用计数已为0")
        
        self.ref_count -= 1
        print(f"📉 Block {self.block_id} 引用计数减少到 {self.ref_count}")
        
        if self.ref_count == 0:
            self.free()
    
    def free(self):
        """释放内存块"""
        self.status = BlockStatus.FREE
        self.ref_count = 0
        self.data = None
        
        print(f"🗑️  释放Block {self.block_id}")

class BlockTable:
    """Block Table - 管理逻辑到物理块的映射"""
    
    def __init__(self, sequence_id: str):
        self.sequence_id = sequence_id
        self.logical_blocks: List[int] = []  # 逻辑块ID列表
        self.block_size = 16  # 每个块的token数量
        
        print(f"📋 创建BlockTable for sequence {sequence_id}")
    
    def add_block(self, physical_block_id: int):
        """添加物理块"""
        self.logical_blocks.append(physical_block_id)
        logical_block_id = len(self.logical_blocks) - 1
        
        print(f"➕ BlockTable {self.sequence_id}: 添加物理块 {physical_block_id} -> 逻辑块 {logical_block_id}")
    
    def get_physical_block_id(self, logical_block_id: int) -> int:
        """获取物理块ID"""
        if logical_block_id >= len(self.logical_blocks):
            raise IndexError(f"逻辑块 {logical_block_id} 不存在")
        
        physical_id = self.logical_blocks[logical_block_id]
        print(f"🔍 BlockTable {self.sequence_id}: 逻辑块 {logical_block_id} -> 物理块 {physical_id}")
        return physical_id
    
    def get_block_offset(self, token_position: int) -> Tuple[int, int]:
        """获取token在块中的位置"""
        logical_block_id = token_position // self.block_size
        block_offset = token_position % self.block_size
        
        print(f"📍 BlockTable {self.sequence_id}: token位置 {token_position} -> 块 {logical_block_id}, 偏移 {block_offset}")
        return logical_block_id, block_offset

class BlockAllocator:
    """内存块分配器"""
    
    def __init__(self, num_blocks: int, block_size: int, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        
        # 创建所有块
        self.blocks = [Block(i, block_size, device) for i in range(num_blocks)]
        self.free_blocks = list(range(num_blocks))
        
        print(f"🏭 创建BlockAllocator: {num_blocks}个块，每块{block_size}个token，设备{device}")
    
    def allocate(self, num_blocks: int) -> List[int]:
        """分配指定数量的块"""
        print(f"🔍 请求分配 {num_blocks} 个块 (可用: {len(self.free_blocks)})")
        
        if len(self.free_blocks) < num_blocks:
            print(f"❌ 内存不足！需要 {num_blocks} 个块，但只有 {len(self.free_blocks)} 个可用")
            return []
        
        allocated_blocks = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            self.blocks[block_id].allocate()
            allocated_blocks.append(block_id)
        
        print(f"✅ 成功分配块: {allocated_blocks}")
        print(f"📊 当前内存状态: {len(self.free_blocks)}/{self.num_blocks} 块可用")
        
        return allocated_blocks
    
    def free(self, block_ids: List[int]):
        """释放指定的块"""
        print(f"🗑️  请求释放块: {block_ids}")
        
        for block_id in block_ids:
            if block_id >= len(self.blocks):
                print(f"⚠️  无效的块ID: {block_id}")
                continue
            
            block = self.blocks[block_id]
            if block.status == BlockStatus.FREE:
                print(f"⚠️  块 {block_id} 已经是空闲状态")
                continue
            
            # 检查引用计数
            if block.ref_count > 1:
                print(f"📉 块 {block_id} 引用计数 {block.ref_count} > 1，减少引用")
                block.remove_ref()
            else:
                print(f"🧹 清理块 {block_id} 的KV Cache数据")
                block.free()
                self.free_blocks.append(block_id)
        
        print(f"📊 释放后内存状态: {len(self.free_blocks)}/{self.num_blocks} 块可用")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        used_blocks = self.num_blocks - len(self.free_blocks)
        usage_ratio = used_blocks / self.num_blocks
        
        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "usage_ratio": usage_ratio,
            "block_size": self.block_size,
            "device": self.device
        }

# ================================
# PagedAttention引擎
# ================================

class PagedAttentionEngine:
    """PagedAttention推理引擎"""
    
    def __init__(self, block_allocator: BlockAllocator):
        self.block_allocator = block_allocator
        self.sequence_tables: Dict[str, BlockTable] = {}
        
        print("🚀 初始化PagedAttentionEngine")
    
    def allocate_sequence(self, sequence_id: str, sequence_length: int) -> bool:
        """为序列分配内存"""
        print(f"🎯 为序列 {sequence_id} 分配内存 (长度: {sequence_length})")
        
        # 计算需要的块数量
        block_size = self.block_allocator.block_size
        num_blocks = (sequence_length + block_size - 1) // block_size
        
        print(f"📊 序列长度 {sequence_length}，块大小 {block_size}，需要 {num_blocks} 个块")
        
        # 分配内存块
        allocated_blocks = self.block_allocator.allocate(num_blocks)
        if not allocated_blocks:
            print(f"❌ 为序列 {sequence_id} 分配内存失败")
            return False
        
        # 创建BlockTable
        block_table = BlockTable(sequence_id)
        for block_id in allocated_blocks:
            block_table.add_block(block_id)
        
        self.sequence_tables[sequence_id] = block_table
        print(f"✅ 序列 {sequence_id} 内存分配成功，使用块: {allocated_blocks}")
        
        return True
    
    def free_sequence(self, sequence_id: str):
        """释放序列内存"""
        print(f"🗑️  释放序列 {sequence_id} 的内存")
        
        if sequence_id not in self.sequence_tables:
            print(f"⚠️  序列 {sequence_id} 不存在")
            return
        
        block_table = self.sequence_tables[sequence_id]
        self.block_allocator.free(block_table.logical_blocks)
        
        del self.sequence_tables[sequence_id]
        print(f"✅ 序列 {sequence_id} 内存释放完成")
    
    def write_kv_cache(self, sequence_id: str, token_position: int, kv_data: Any):
        """写入KV Cache数据"""
        print(f"✍️  写入KV Cache: 序列 {sequence_id}, 位置 {token_position}")
        
        if sequence_id not in self.sequence_tables:
            print(f"❌ 序列 {sequence_id} 不存在")
            return False
        
        try:
            block_table = self.sequence_tables[sequence_id]
            logical_block_id, block_offset = block_table.get_block_offset(token_position)
            physical_block_id = block_table.get_physical_block_id(logical_block_id)
            
            print(f"📝 KV Cache写入: 物理块 {physical_block_id}, 偏移 {block_offset}")
            
            # 这里应该写入实际的KV Cache数据
            # 为了演示，我们只是标记数据已写入
            block = self.block_allocator.blocks[physical_block_id]
            if block.data is None:
                block.data = {}
            block.data[block_offset] = kv_data
            
            print(f"✅ KV Cache写入成功")
            return True
            
        except Exception as e:
            print(f"❌ KV Cache写入失败: {e}")
            return False

# ================================
# 调度器
# ================================

class Scheduler:
    """请求调度器"""
    
    def __init__(self, 
                 max_num_seqs: int = 32,
                 max_batch_size: int = 8,
                 policy: SchedulerPolicy = SchedulerPolicy.FCFS,
                 memory_threshold: float = 0.9):
        
        self.max_num_seqs = max_num_seqs
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.memory_threshold = memory_threshold
        
        # 请求队列
        self.waiting_queue: deque = deque()
        self.running_requests: Dict[str, InferenceRequest] = {}
        self.swapped_requests: Dict[str, InferenceRequest] = {}
        
        print(f"📋 初始化调度器: 最大序列数={max_num_seqs}, 批大小={max_batch_size}, 策略={policy.value}")
    
    def add_request(self, request: InferenceRequest):
        """添加新请求"""
        print(f"📨 接收新请求 {request.request_id}")
        print(f"   📝 提示: {request.prompt[:50]}...")
        print(f"   🎯 最大token数: {request.params.max_tokens}")
        print(f"   ⭐ 优先级: {request.priority}")
        print(f"   ⏰ 到达时间: {request.arrival_time:.2f}")
        
        # 估算需要的内存块数
        estimated_tokens = len(request.prompt) + request.params.max_tokens
        estimated_blocks = (estimated_tokens + 15) // 16  # 假设block_size=16
        request.num_blocks = estimated_blocks
        
        print(f"   📊 估算token数: {estimated_tokens}, 需要块数: {estimated_blocks}")
        
        self.waiting_queue.append(request)
        print(f"   📋 当前等待队列长度: {len(self.waiting_queue)}")
    
    def schedule(self, paged_attention: PagedAttentionEngine) -> Dict[str, List[InferenceRequest]]:
        """执行调度决策"""
        print(f"\n🔄 开始调度循环")
        print(f"   📊 当前状态: 等待={len(self.waiting_queue)}, 运行={len(self.running_requests)}, 交换={len(self.swapped_requests)}")
        
        # 检查是否需要抢占
        if self._should_preempt(paged_attention):
            print("⚠️  内存压力过高，执行抢占")
            self._preempt_requests(paged_attention)
        
        # 处理等待队列
        prefill_requests = self._schedule_waiting(paged_attention)
        
        # 处理运行中的请求
        decode_requests = self._schedule_running()
        
        # 处理交换队列
        swap_in_requests = self._schedule_swapped(paged_attention)
        
        result = {
            "prefill": prefill_requests,
            "decode": decode_requests,
            "swap_in": swap_in_requests
        }
        
        print(f"✅ 调度完成: prefill={len(prefill_requests)}, decode={len(decode_requests)}, swap_in={len(swap_in_requests)}")
        return result
    
    def _should_preempt(self, paged_attention: PagedAttentionEngine) -> bool:
        """检查是否需要抢占"""
        memory_usage = paged_attention.block_allocator.get_memory_usage()
        usage_ratio = memory_usage["usage_ratio"]
        
        print(f"🔍 内存检查: 使用率 {usage_ratio:.2%}")
        print(f"   📊 空闲块: {memory_usage['free_blocks']}/{memory_usage['total_blocks']}")
        
        if usage_ratio > self.memory_threshold:
            print(f"⚠️  内存使用率 {usage_ratio:.2%} 超过阈值 {self.memory_threshold:.2%}，需要抢占")
            return True
        
        return False
    
    def _preempt_requests(self, paged_attention: PagedAttentionEngine):
        """抢占请求"""
        print(f"🚨 开始抢占决策")
        
        if not self.running_requests:
            print("   ℹ️  没有运行中的请求可以抢占")
            return
        
        # 根据策略选择抢占候选
        candidates = list(self.running_requests.values())
        
        if self.policy == SchedulerPolicy.PRIORITY:
            candidates.sort(key=lambda x: x.priority)  # 低优先级先抢占
            print("   📊 按优先级排序抢占候选")
        elif self.policy == SchedulerPolicy.SJF:
            candidates.sort(key=lambda x: len(x.generated_tokens), reverse=True)  # 长任务先抢占
            print("   📊 按作业长度排序抢占候选")
        else:  # FCFS
            candidates.sort(key=lambda x: x.start_time or 0, reverse=True)  # 最新的先抢占
            print("   📊 按开始时间排序抢占候选")
        
        # 抢占模式：swap或recompute
        preemption_mode = "swap"  # 简化实现，总是使用swap
        
        for request in candidates[:2]:  # 最多抢占2个请求
            print(f"   🔄 抢占请求 {request.request_id} (模式: {preemption_mode})")
            
            if preemption_mode == "swap":
                # 交换到CPU
                request.status = RequestStatus.SWAPPED
                self.swapped_requests[request.request_id] = request
                del self.running_requests[request.request_id]
                print(f"   💾 请求 {request.request_id} 已交换到CPU")
                
            else:  # recompute
                # 重新计算，放回等待队列
                request.status = RequestStatus.WAITING
                self.waiting_queue.appendleft(request)
                del self.running_requests[request.request_id]
                paged_attention.free_sequence(request.request_id)
                print(f"   🔄 请求 {request.request_id} 已移回等待队列")
            
            # 检查是否释放了足够内存
            memory_usage = paged_attention.block_allocator.get_memory_usage()
            if memory_usage["usage_ratio"] <= self.memory_threshold:
                print(f"   ✅ 内存使用率降至 {memory_usage['usage_ratio']:.2%}，停止抢占")
                break
        
        if not candidates:
            print("   ❌ 抢占失败，没有合适的候选请求")
    
    def _schedule_waiting(self, paged_attention: PagedAttentionEngine) -> List[InferenceRequest]:
        """调度等待队列"""
        scheduled = []
        
        while (self.waiting_queue and 
               len(self.running_requests) < self.max_num_seqs and
               len(scheduled) < self.max_batch_size):
            
            request = self.waiting_queue.popleft()
            
            # 尝试分配内存
            if paged_attention.allocate_sequence(request.request_id, 
                                               len(request.prompt_tokens) + request.params.max_tokens):
                request.status = RequestStatus.RUNNING
                request.start_time = time.time()
                self.running_requests[request.request_id] = request
                scheduled.append(request)
                
                print(f"✅ 调度请求 {request.request_id} 进入运行队列")
            else:
                # 内存不足，放回队列
                self.waiting_queue.appendleft(request)
                print(f"❌ 请求 {request.request_id} 内存分配失败，保持等待")
                break
        
        return scheduled
    
    def _schedule_running(self) -> List[InferenceRequest]:
        """调度运行中的请求"""
        return list(self.running_requests.values())
    
    def _schedule_swapped(self, paged_attention: PagedAttentionEngine) -> List[InferenceRequest]:
        """调度交换队列"""
        scheduled = []
        
        # 简化实现：暂时不处理swap in
        return scheduled
    
    def finish_request(self, request_id: str, paged_attention: PagedAttentionEngine):
        """完成请求"""
        if request_id in self.running_requests:
            request = self.running_requests[request_id]
            request.status = RequestStatus.FINISHED
            request.finish_time = time.time()
            
            # 释放内存
            paged_attention.free_sequence(request_id)
            del self.running_requests[request_id]
            
            print(f"✅ 请求 {request_id} 已完成")
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_requests),
            "swapped": len(self.swapped_requests)
        }

# ================================
# 指标收集器
# ================================

class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.request_latencies: deque = deque(maxlen=1000)
        self.request_timestamps: deque = deque(maxlen=1000)
        self.token_counts: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        print("📊 初始化指标收集器")
    
    def record_request(self, request: InferenceRequest, success: bool = True, error_type: str = None):
        """记录请求指标"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            if request.start_time and request.finish_time:
                latency = request.finish_time - request.start_time
                self.request_latencies.append(latency)
                self.request_timestamps.append(request.finish_time)
                self.token_counts.append(len(request.generated_tokens))
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] += 1
    
    def get_metrics(self, paged_attention: PagedAttentionEngine, scheduler: Scheduler) -> SystemMetrics:
        """获取系统指标"""
        now = time.time()
        
        # 计算吞吐量
        recent_requests = [t for t in self.request_timestamps if now - t <= 60]  # 最近1分钟
        requests_per_second = len(recent_requests) / 60.0
        
        recent_tokens = [self.token_counts[i] for i, t in enumerate(self.request_timestamps) 
                        if now - t <= 60]
        tokens_per_second = sum(recent_tokens) / 60.0 if recent_tokens else 0.0
        
        # 计算延迟
        latencies = list(self.request_latencies)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0.0
        
        # 内存指标
        memory_usage = paged_attention.block_allocator.get_memory_usage()
        
        # GPU内存（如果可用）
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        gpu_memory_utilization = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_memory_utilization = gpu_memory_used / gpu_memory_total if gpu_memory_total > 0 else 0.0
        
        # 队列状态
        queue_status = scheduler.get_queue_status()
        
        # 错误率
        error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0
        
        return SystemMetrics(
            timestamp=now,
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_usage=memory_usage["usage_ratio"],
            waiting_requests=queue_status["waiting"],
            running_requests=queue_status["running"],
            swapped_requests=queue_status["swapped"],
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            error_rate=error_rate
        )

# ================================
# 主要的NanoVLLM类
# ================================

class NanoVLLM:
    """NanoVLLM主类 - 完整的推理系统"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # 核心组件
        self.block_allocator = None
        self.paged_attention = None
        self.scheduler = None
        self.metrics_collector = None
        
        # 运行状态
        self.is_running = False
        self.engine_thread = None
        self._stop_event = threading.Event()
        
        print("🚀 初始化NanoVLLM系统")
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所有组件"""
        print("🔧 初始化系统组件...")
        
        # 设备配置
        if self.config.get("device") == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.get("device", "cpu")
        
        print(f"   🎯 使用设备: {self.device}")
        
        # 初始化内存管理
        num_gpu_blocks = self.config.get("num_gpu_blocks", 1000)
        block_size = self.config.get("block_size", 16)
        
        self.block_allocator = BlockAllocator(
            num_blocks=num_gpu_blocks,
            block_size=block_size,
            device=self.device
        )
        
        # 初始化PagedAttention
        self.paged_attention = PagedAttentionEngine(self.block_allocator)
        
        # 初始化调度器
        self.scheduler = Scheduler(
            max_num_seqs=self.config.get("max_num_seqs", 32),
            max_batch_size=self.config.get("max_batch_size", 8),
            policy=SchedulerPolicy(self.config.get("scheduler_policy", "fcfs")),
            memory_threshold=self.config.get("memory_threshold", 0.9)
        )
        
        # 初始化指标收集器
        self.metrics_collector = MetricsCollector()
        
        print("✅ 系统组件初始化完成")
    
    def load_model(self, model_name: str):
        """加载模型"""
        print(f"📦 开始加载模型: {model_name}")
        start_time = time.time()
        
        try:
            # 加载tokenizer
            print("   🔤 加载tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("   🧠 加载模型...")
            torch_dtype = self.config.get("torch_dtype", torch.float16)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            load_time = time.time() - start_time
            
            # 模型信息
            num_params = sum(p.numel() for p in self.model.parameters())
            model_size = num_params * 2 / 1024**3  # 假设float16，单位GB
            
            print(f"✅ 模型加载完成!")
            print(f"   ⏱️  加载时间: {load_time:.2f}秒")
            print(f"   📊 参数数量: {num_params:,}")
            print(f"   💾 模型大小: {model_size:.2f}GB")
            print(f"   🎯 设备: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def generate(self, prompt: str, params: GenerationParams = None) -> InferenceResponse:
        """生成文本"""
        if params is None:
            params = GenerationParams()
        
        request_id = f"req_{int(time.time() * 1000)}"
        
        print(f"\n🎯 开始处理请求 {request_id}")
        print(f"   📝 提示: {prompt[:100]}...")
        print(f"   ⚙️  参数: max_tokens={params.max_tokens}, temperature={params.temperature}")
        
        start_time = time.time()
        
        try:
            # 创建推理请求
            request = InferenceRequest(
                request_id=request_id,
                prompt=prompt,
                params=params
            )
            
            # Tokenize
            print("   🔤 Tokenizing...")
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            request.prompt_tokens = inputs["input_ids"][0].tolist()
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            print(f"   📊 输入token数: {len(request.prompt_tokens)}")
            
            # 添加到调度队列
            print("   📋 添加到调度队列...")
            self.scheduler.add_request(request)
            
            # 简化的生成过程（实际应该通过调度器）
            print("   🧠 生成文本...")
            
            with torch.no_grad():
                # 设置生成参数
                generation_config = GenerationConfig(
                    max_new_tokens=params.max_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    top_k=params.top_k if params.top_k > 0 else None,
                    do_sample=params.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )
            
            # 解码输出
            generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            request.generated_tokens = generated_ids.tolist()
            request.finish_time = time.time()
            
            generation_time = request.finish_time - start_time
            tokens_per_second = len(request.generated_tokens) / generation_time if generation_time > 0 else 0
            
            print(f"✅ 生成完成!")
            print(f"   ⏱️  生成时间: {generation_time:.2f}秒")
            print(f"   🚀 生成速度: {tokens_per_second:.1f} tokens/s")
            print(f"   📊 生成token数: {len(request.generated_tokens)}")
            print(f"   📝 生成文本: {generated_text[:100]}...")
            
            # 记录指标
            self.metrics_collector.record_request(request, success=True)
            
            # 完成请求
            self.scheduler.finish_request(request_id, self.paged_attention)
            
            return InferenceResponse(
                request_id=request_id,
                generated_text=generated_text,
                generated_tokens=request.generated_tokens,
                prompt_tokens=len(request.prompt_tokens),
                completion_tokens=len(request.generated_tokens),
                total_tokens=len(request.prompt_tokens) + len(request.generated_tokens),
                finish_reason="stop",
                generation_time=generation_time,
                tokens_per_second=tokens_per_second
            )
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            self.metrics_collector.record_request(request, success=False, error_type=str(type(e).__name__))
            
            return InferenceResponse(
                request_id=request_id,
                generated_text="",
                generated_tokens=[],
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                finish_reason="error",
                generation_time=time.time() - start_time,
                tokens_per_second=0.0
            )
    
    def start_engine_loop(self):
        """启动引擎循环"""
        if self.is_running:
            print("⚠️  引擎已在运行")
            return
        
        print("🚀 启动引擎循环...")
        self.is_running = True
        self._stop_event.clear()
        
        self.engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self.engine_thread.start()
        
        print("✅ 引擎循环已启动")
    
    def stop_engine_loop(self):
        """停止引擎循环"""
        if not self.is_running:
            print("⚠️  引擎未在运行")
            return
        
        print("🛑 停止引擎循环...")
        self.is_running = False
        self._stop_event.set()
        
        if self.engine_thread:
            self.engine_thread.join(timeout=5.0)
        
        print("✅ 引擎循环已停止")
    
    def _engine_loop(self):
        """引擎主循环"""
        print("🔄 引擎循环开始运行...")
        iteration = 0
        
        while self.is_running and not self._stop_event.is_set():
            iteration += 1
            loop_start = time.time()
            
            print(f"\n🔄 引擎循环 #{iteration}")
            
            try:
                # 执行调度
                schedule_start = time.time()
                scheduled = self.scheduler.schedule(self.paged_attention)
                schedule_time = time.time() - schedule_start
                
                prefill_count = len(scheduled["prefill"])
                decode_count = len(scheduled["decode"])
                
                print(f"   📋 调度结果: prefill={prefill_count}, decode={decode_count}")
                
                # 执行prefill
                if scheduled["prefill"]:
                    prefill_start = time.time()
                    # 这里应该执行实际的prefill操作
                    prefill_time = time.time() - prefill_start
                    print(f"   ⚡ Prefill执行时间: {prefill_time:.3f}秒")
                
                # 执行decode
                if scheduled["decode"]:
                    decode_start = time.time()
                    # 这里应该执行实际的decode操作
                    decode_time = time.time() - decode_start
                    print(f"   🔄 Decode执行时间: {decode_time:.3f}秒")
                
                # 定期输出队列状态
                if iteration % 10 == 0:
                    queue_status = self.scheduler.get_queue_status()
                    print(f"   📊 队列状态: 等待={queue_status['waiting']}, 运行={queue_status['running']}, 交换={queue_status['swapped']}")
                
            except Exception as e:
                print(f"❌ 引擎循环错误: {e}")
            
            # 控制循环频率
            loop_time = time.time() - loop_start
            sleep_time = max(0, self.config.get("engine_loop_interval", 0.01) - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("🔄 引擎循环结束")
    
    def get_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        return self.metrics_collector.get_metrics(self.paged_attention, self.scheduler)
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            metrics = self.get_metrics()
            memory_usage = self.paged_attention.block_allocator.get_memory_usage()
            
            # 健康检查
            is_healthy = True
            issues = []
            
            # 检查内存使用
            if memory_usage["usage_ratio"] > 0.95:
                is_healthy = False
                issues.append("内存使用率过高")
            
            # 检查错误率
            if metrics.error_rate > 0.1:  # 10%错误率
                is_healthy = False
                issues.append("错误率过高")
            
            # 检查队列积压
            if metrics.waiting_requests > 100:
                is_healthy = False
                issues.append("等待队列积压严重")
            
            return {
                "healthy": is_healthy,
                "issues": issues,
                "uptime": time.time() - getattr(self, '_start_time', time.time()),
                "model_loaded": self.model is not None,
                "engine_running": self.is_running,
                "memory_usage": memory_usage,
                "queue_status": self.scheduler.get_queue_status(),
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "issues": [f"健康检查失败: {e}"],
                "error": str(e)
            }
    
    def __enter__(self):
        """上下文管理器入口"""
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_engine_loop()
        print("🔚 NanoVLLM系统已关闭")

# ================================
# 工具函数
# ================================

def create_nano_vllm(config_name: str = "dev") -> NanoVLLM:
    """创建NanoVLLM实例的便捷函数"""
    from .config import get_config
    
    config = get_config(config_name)
    config_dict = {
        "device": config.device,
        "torch_dtype": config.torch_dtype,
        "max_num_seqs": config.max_num_seqs,
        "max_batch_size": config.max_batch_size,
        "block_size": config.block_size,
        "num_gpu_blocks": config.num_gpu_blocks,
        "scheduler_policy": config.scheduler_policy,
        "memory_threshold": config.memory_threshold,
        "engine_loop_interval": config.engine_loop_interval,
    }
    
    return NanoVLLM(config_dict)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NanoVLLM 完整推理系统演示")
    print("=" * 60)
    
    # 创建系统配置
    config = {
        "device": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "max_num_seqs": 4,
        "max_batch_size": 2,
        "block_size": 16,
        "num_gpu_blocks": 100,
        "scheduler_policy": "fcfs",
        "memory_threshold": 0.9,
        "engine_loop_interval": 0.01,
    }
    
    # 创建并运行系统
    with NanoVLLM(config) as nano_vllm:
        # 加载模型
        if nano_vllm.load_model("microsoft/DialoGPT-small"):
            
            # 单个请求演示
            print("\n" + "="*40)
            print("📝 单个请求演示")
            print("="*40)
            
            response = nano_vllm.generate(
                "Hello, how are you?",
                GenerationParams(max_tokens=50, temperature=0.8)
            )
            
            print(f"生成结果: {response.generated_text}")
            print(f"性能: {response.tokens_per_second:.1f} tokens/s")
            
            # 批量请求演示
            print("\n" + "="*40)
            print("📦 批量请求演示")
            print("="*40)
            
            prompts = [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "How does deep learning work?"
            ]
            
            responses = []
            for prompt in prompts:
                response = nano_vllm.generate(
                    prompt,
                    GenerationParams(max_tokens=30, temperature=0.7)
                )
                responses.append(response)
            
            for i, response in enumerate(responses):
                print(f"请求 {i+1}: {response.generated_text[:50]}...")
            
            # 系统指标
            print("\n" + "="*40)
            print("📊 系统指标")
            print("="*40)
            
            metrics = nano_vllm.get_metrics()
            print(f"总请求数: {metrics.total_requests}")
            print(f"成功率: {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
            print(f"平均延迟: {metrics.avg_latency:.2f}秒")
            print(f"吞吐量: {metrics.tokens_per_second:.1f} tokens/s")
            print(f"内存使用: {metrics.kv_cache_usage:.1%}")
            
            # 健康状态
            health = nano_vllm.get_health_status()
            print(f"系统健康: {'✅' if health['healthy'] else '❌'}")
            if health['issues']:
                print(f"问题: {', '.join(health['issues'])}")
        
        else:
            print("❌ 模型加载失败，跳过演示")
    
    print("\n🎉 演示完成！")