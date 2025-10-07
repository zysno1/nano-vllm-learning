#!/usr/bin/env python3
"""
第四步：请求调度与批处理优化

这个脚本演示了LLM推理系统中调度器的核心实现，包括：
- 请求生命周期管理
- Continuous Batching 机制
- 内存感知调度策略
- 抢占与换出机制

学习目标：
1. 理解调度器在LLM推理系统中的核心作用
2. 掌握Continuous Batching的实现原理
3. 学会设计内存感知的调度算法
4. 理解不同调度策略的权衡
"""

import os
import sys
import time
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Deque
from enum import Enum
import heapq
import uuid

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class RequestStatus(Enum):
    """请求状态枚举"""
    WAITING = "waiting"      # 等待调度
    RUNNING = "running"      # 正在执行
    SWAPPED = "swapped"      # 被换出内存
    FINISHED = "finished"    # 执行完成
    CANCELLED = "cancelled"  # 被取消


class SchedulerPolicy(Enum):
    """调度策略枚举"""
    FCFS = "fcfs"                    # 先来先服务
    SJF = "sjf"                      # 最短作业优先
    PRIORITY = "priority"            # 优先级调度
    MEMORY_AWARE = "memory_aware"    # 内存感知调度


class PreemptionMode(Enum):
    """抢占模式枚举"""
    RECOMPUTE = "recompute"  # 重计算（适用于prefill阶段）
    SWAP = "swap"            # 换出（适用于decode阶段）


@dataclass
class SequenceRequest:
    """推理请求类"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    priority: int = 0  # 优先级，数值越大优先级越高
    
    # 运行时状态
    status: RequestStatus = RequestStatus.WAITING
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    
    # 生成状态
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    
    # 内存使用
    allocated_blocks: List[int] = field(default_factory=list)
    cpu_blocks: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        # 模拟prompt tokenization
        self.prompt_tokens = len(self.prompt.split()) * 2  # 粗略估算
        self.total_tokens = self.prompt_tokens
    
    def is_finished(self) -> bool:
        """检查请求是否完成"""
        return (self.generated_tokens >= self.max_tokens or 
                self.status == RequestStatus.FINISHED)
    
    def get_estimated_remaining_tokens(self) -> int:
        """估算剩余生成token数量"""
        return max(0, self.max_tokens - self.generated_tokens)
    
    def get_memory_blocks_needed(self, block_size: int) -> int:
        """计算需要的内存块数量"""
        total_needed = self.total_tokens + self.get_estimated_remaining_tokens()
        return (total_needed + block_size - 1) // block_size
    
    def get_wait_time(self) -> float:
        """获取等待时间"""
        if self.start_time:
            return self.start_time - self.arrival_time
        return time.time() - self.arrival_time
    
    def get_execution_time(self) -> float:
        """获取执行时间"""
        if self.start_time:
            end_time = self.finish_time or time.time()
            return end_time - self.start_time
        return 0.0


@dataclass
class SchedulerOutput:
    """调度器输出"""
    scheduled_requests: List[SequenceRequest] = field(default_factory=list)
    preempted_requests: List[SequenceRequest] = field(default_factory=list)
    ignored_requests: List[SequenceRequest] = field(default_factory=list)
    finished_requests: List[SequenceRequest] = field(default_factory=list)
    
    # 内存操作
    blocks_to_swap_in: Dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: Dict[int, int] = field(default_factory=dict)
    blocks_to_copy: Dict[int, List[int]] = field(default_factory=dict)
    
    def get_batch_size(self) -> int:
        """获取批次大小"""
        return len(self.scheduled_requests)


class BlockSpaceManager:
    """内存块空间管理器"""
    
    def __init__(self, 
                 num_gpu_blocks: int,
                 num_cpu_blocks: int,
                 block_size: int):
        
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.block_size = block_size
        
        # GPU内存块管理
        self.gpu_allocator = list(range(num_gpu_blocks))
        self.gpu_allocated: Dict[str, List[int]] = {}
        
        # CPU内存块管理
        self.cpu_allocator = list(range(num_cpu_blocks))
        self.cpu_allocated: Dict[str, List[int]] = {}
        
        print(f"🧱 内存管理器初始化：")
        print(f"   GPU块数：{num_gpu_blocks}")
        print(f"   CPU块数：{num_cpu_blocks}")
        print(f"   块大小：{block_size}")
    
    def can_allocate(self, request: SequenceRequest) -> bool:
        """检查是否可以为请求分配内存"""
        needed_blocks = request.get_memory_blocks_needed(self.block_size)
        return len(self.gpu_allocator) >= needed_blocks
    
    def allocate(self, request: SequenceRequest) -> List[int]:
        """为请求分配GPU内存块"""
        needed_blocks = request.get_memory_blocks_needed(self.block_size)
        
        if len(self.gpu_allocator) < needed_blocks:
            raise RuntimeError(f"GPU内存不足：需要{needed_blocks}块，可用{len(self.gpu_allocator)}块")
        
        # 分配内存块
        allocated_blocks = []
        for _ in range(needed_blocks):
            block_id = self.gpu_allocator.pop(0)
            allocated_blocks.append(block_id)
        
        self.gpu_allocated[request.request_id] = allocated_blocks
        request.allocated_blocks = allocated_blocks
        
        print(f"   🔧 为请求 {request.request_id} 分配 {needed_blocks} 个GPU块")
        return allocated_blocks
    
    def free(self, request: SequenceRequest):
        """释放请求的GPU内存块"""
        if request.request_id in self.gpu_allocated:
            blocks = self.gpu_allocated[request.request_id]
            self.gpu_allocator.extend(blocks)
            del self.gpu_allocated[request.request_id]
            request.allocated_blocks = []
            
            print(f"   🗑️ 释放请求 {request.request_id} 的 {len(blocks)} 个GPU块")
    
    def swap_out(self, request: SequenceRequest) -> List[int]:
        """将请求的内存块从GPU换出到CPU"""
        if request.request_id not in self.gpu_allocated:
            return []
        
        gpu_blocks = self.gpu_allocated[request.request_id]
        needed_cpu_blocks = len(gpu_blocks)
        
        if len(self.cpu_allocator) < needed_cpu_blocks:
            raise RuntimeError(f"CPU内存不足：需要{needed_cpu_blocks}块，可用{len(self.cpu_allocator)}块")
        
        # 分配CPU块
        cpu_blocks = []
        for _ in range(needed_cpu_blocks):
            cpu_block = self.cpu_allocator.pop(0)
            cpu_blocks.append(cpu_block)
        
        # 释放GPU块
        self.gpu_allocator.extend(gpu_blocks)
        del self.gpu_allocated[request.request_id]
        
        # 记录CPU块
        self.cpu_allocated[request.request_id] = cpu_blocks
        request.allocated_blocks = []
        request.cpu_blocks = cpu_blocks
        
        print(f"   💾 请求 {request.request_id} 换出：GPU块 {gpu_blocks} -> CPU块 {cpu_blocks}")
        return cpu_blocks
    
    def swap_in(self, request: SequenceRequest) -> List[int]:
        """将请求的内存块从CPU换入到GPU"""
        if request.request_id not in self.cpu_allocated:
            return []
        
        cpu_blocks = self.cpu_allocated[request.request_id]
        needed_gpu_blocks = len(cpu_blocks)
        
        if len(self.gpu_allocator) < needed_gpu_blocks:
            raise RuntimeError(f"GPU内存不足：需要{needed_gpu_blocks}块，可用{len(self.gpu_allocator)}块")
        
        # 分配GPU块
        gpu_blocks = []
        for _ in range(needed_gpu_blocks):
            gpu_block = self.gpu_allocator.pop(0)
            gpu_blocks.append(gpu_block)
        
        # 释放CPU块
        self.cpu_allocator.extend(cpu_blocks)
        del self.cpu_allocated[request.request_id]
        
        # 记录GPU块
        self.gpu_allocated[request.request_id] = gpu_blocks
        request.allocated_blocks = gpu_blocks
        request.cpu_blocks = []
        
        print(f"   📥 请求 {request.request_id} 换入：CPU块 {cpu_blocks} -> GPU块 {gpu_blocks}")
        return gpu_blocks
    
    def get_num_free_gpu_blocks(self) -> int:
        """获取空闲GPU块数量"""
        return len(self.gpu_allocator)
    
    def get_num_free_cpu_blocks(self) -> int:
        """获取空闲CPU块数量"""
        return len(self.cpu_allocator)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return {
            "gpu_total": self.num_gpu_blocks,
            "gpu_free": len(self.gpu_allocator),
            "gpu_used": self.num_gpu_blocks - len(self.gpu_allocator),
            "cpu_total": self.num_cpu_blocks,
            "cpu_free": len(self.cpu_allocator),
            "cpu_used": self.num_cpu_blocks - len(self.cpu_allocator),
            "gpu_utilization": (self.num_gpu_blocks - len(self.gpu_allocator)) / self.num_gpu_blocks * 100,
        }


class Scheduler:
    """LLM推理调度器"""
    
    def __init__(self,
                 policy: SchedulerPolicy = SchedulerPolicy.FCFS,
                 max_batch_size: int = 32,
                 max_num_seqs: int = 256,
                 block_size: int = 16,
                 num_gpu_blocks: int = 1024,
                 num_cpu_blocks: int = 2048):
        
        self.policy = policy
        self.max_batch_size = max_batch_size
        self.max_num_seqs = max_num_seqs
        self.block_size = block_size
        
        # 请求队列
        self.waiting: Deque[SequenceRequest] = deque()
        self.running: List[SequenceRequest] = []
        self.swapped: List[SequenceRequest] = []
        self.finished: List[SequenceRequest] = []
        
        # 内存管理
        self.block_manager = BlockSpaceManager(
            num_gpu_blocks, num_cpu_blocks, block_size
        )
        
        # 统计信息
        self.total_requests = 0
        self.completed_requests = 0
        self.preempted_count = 0
        self.swap_in_count = 0
        self.swap_out_count = 0
        
        print(f"🚀 调度器初始化完成：")
        print(f"   调度策略：{policy.value}")
        print(f"   最大批次大小：{max_batch_size}")
        print(f"   最大序列数：{max_num_seqs}")
    
    def add_request(self, request: SequenceRequest):
        """添加新请求到等待队列"""
        print(f"\n📥 接收新请求: {request.request_id}")
        print(f"    📝 Prompt: {request.prompt[:50]}{'...' if len(request.prompt) > 50 else ''}")
        print(f"    🎯 Max tokens: {request.max_tokens}, Priority: {request.priority}")
        print(f"    ⏰ 到达时间: {time.strftime('%H:%M:%S', time.localtime(request.arrival_time))}")
        
        # 估算内存需求
        estimated_blocks = request.get_memory_blocks_needed(self.block_size)
        print(f"    💾 预估内存需求: {estimated_blocks} 块")
        
        # 添加到等待队列
        self.waiting_requests.append(request)
        
        # 更新统计
        self.stats["total_requests"] += 1
        self.stats["waiting_requests"] += 1
        
        print(f"    📊 当前队列状态: 等待 {len(self.waiting_requests)}, 运行 {len(self.running_requests)}, 换出 {len(self.swapped_requests)}")
    
    def schedule(self) -> SchedulerOutput:
        """执行调度决策"""
        print(f"\n🔄 开始调度循环 (策略: {self.policy.value})")
        print(f"    📊 当前状态: 等待 {len(self.waiting_requests)}, 运行 {len(self.running_requests)}, 换出 {len(self.swapped_requests)}")
        
        scheduler_output = SchedulerOutput()
        
        # 1. 处理正在运行的请求
        print(f"  🏃 处理运行中的请求...")
        self._schedule_running(scheduler_output)
        
        # 2. 处理被换出的请求
        print(f"  🔄 处理换出的请求...")
        self._schedule_swapped(scheduler_output)
        
        # 3. 处理等待中的请求
        print(f"  ⏳ 处理等待中的请求...")
        self._schedule_waiting(scheduler_output)
        
        # 4. 检查是否需要抢占
        if self._should_preempt():
            print(f"  ⚠️  内存压力过大，执行抢占...")
            preempted = self._preempt_requests()
            scheduler_output.preempted_requests.extend(preempted)
        
        # 5. 更新统计信息
        self._update_stats(scheduler_output)
        
        print(f"    ✅ 调度完成: 调度 {len(scheduler_output.scheduled_requests)}, 抢占 {len(scheduler_output.preempted_requests)}, 忽略 {len(scheduler_output.ignored_requests)}")
        
        return scheduler_output
    
    def _schedule_running(self, scheduler_output: SchedulerOutput):
        """调度运行中的请求"""
        print(f"   🔄 处理运行中的请求...")
        
        # 检查完成的请求
        finished_requests = []
        continuing_requests = []
        
        for request in self.running:
            # 模拟token生成
            request.generated_tokens += 1
            request.total_tokens += 1
            
            if request.is_finished():
                request.status = RequestStatus.FINISHED
                request.finish_time = time.time()
                finished_requests.append(request)
                
                # 释放内存
                self.block_manager.free(request)
                
                print(f"   ✅ 请求 {request.request_id} 完成")
            else:
                continuing_requests.append(request)
        
        # 更新运行队列
        self.running = continuing_requests
        self.finished.extend(finished_requests)
        scheduler_output.finished_requests = finished_requests
        
        # 检查是否需要抢占
        if self._should_preempt():
            preempted = self._preempt_requests()
            scheduler_output.preempted_requests = preempted
        
        # 继续运行的请求加入调度输出
        scheduler_output.scheduled_requests.extend(continuing_requests)
    
    def _schedule_swapped(self, scheduler_output: SchedulerOutput):
        """调度换出的请求"""
        if not self.swapped:
            return
        
        print(f"   💾 处理换出的请求...")
        
        # 按优先级排序换出的请求
        self.swapped.sort(key=self._get_request_priority, reverse=True)
        
        swapped_in = []
        for request in self.swapped[:]:
            if self.block_manager.can_allocate(request):
                # 换入内存
                self.block_manager.swap_in(request)
                request.status = RequestStatus.RUNNING
                
                self.swapped.remove(request)
                self.running.append(request)
                swapped_in.append(request)
                
                self.swap_in_count += 1
                print(f"   📥 请求 {request.request_id} 换入成功")
                
                # 限制批次大小
                if len(scheduler_output.scheduled_requests) + len(swapped_in) >= self.max_batch_size:
                    break
        
        scheduler_output.scheduled_requests.extend(swapped_in)
    
    def _schedule_waiting(self, scheduler_output: SchedulerOutput):
        """调度等待中的请求"""
        if not self.waiting:
            return
        
        print(f"   ⏳ 处理等待的请求...")
        
        # 根据调度策略排序
        waiting_list = list(self.waiting)
        waiting_list.sort(key=self._get_request_priority, reverse=True)
        
        scheduled = []
        current_batch_size = len(scheduler_output.scheduled_requests)
        
        for request in waiting_list:
            # 检查批次大小限制
            if current_batch_size >= self.max_batch_size:
                break
            
            # 检查内存是否足够
            if not self.block_manager.can_allocate(request):
                scheduler_output.ignored_requests.append(request)
                print(f"   ⏸️ 请求 {request.request_id} 内存不足，暂时忽略")
                continue
            
            # 分配内存并调度
            try:
                self.block_manager.allocate(request)
                request.status = RequestStatus.RUNNING
                request.start_time = time.time()
                
                self.waiting.remove(request)
                self.running.append(request)
                scheduled.append(request)
                
                current_batch_size += 1
                print(f"   ✅ 请求 {request.request_id} 开始执行")
                
            except RuntimeError as e:
                print(f"   ❌ 请求 {request.request_id} 分配失败：{e}")
                scheduler_output.ignored_requests.append(request)
        
        scheduler_output.scheduled_requests.extend(scheduled)
    
    def _should_preempt(self) -> bool:
        """判断是否需要抢占"""
        free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        total_gpu_blocks = self.block_manager.num_gpu_blocks
        memory_usage = (total_gpu_blocks - free_gpu_blocks) / total_gpu_blocks
        
        print(f"    🔍 内存检查: {free_gpu_blocks}/{total_gpu_blocks} 空闲块 (使用率: {memory_usage:.1%})")
        
        # 内存使用率超过90%时考虑抢占
        should_preempt = memory_usage > 0.9 and len(self.waiting_requests) > 0
        
        if should_preempt:
            print(f"    ⚠️  内存压力过大，需要抢占 (使用率: {memory_usage:.1%})")
        
        return should_preempt
    
    def _preempt_requests(self) -> List[SequenceRequest]:
        """执行请求抢占"""
        print(f"    🎯 开始抢占决策...")
        
        if not self.running_requests:
            print(f"    ℹ️  没有运行中的请求可以抢占")
            return []
        
        preempted_requests = []
        
        # 根据策略选择抢占目标
        if self.policy == SchedulerPolicy.PRIORITY:
            # 优先级调度：抢占低优先级请求
            candidates = sorted(self.running_requests, key=lambda r: r.priority)
            print(f"    📊 按优先级排序候选请求: {[f'{r.request_id}(P{r.priority})' for r in candidates[:3]]}")
        elif self.policy == SchedulerPolicy.SJF:
            # 最短作业优先：抢占剩余时间最长的请求
            candidates = sorted(self.running_requests, 
                              key=lambda r: r.get_estimated_remaining_tokens(), reverse=True)
            print(f"    📊 按剩余时间排序候选请求: {[f'{r.request_id}({r.get_estimated_remaining_tokens()}t)' for r in candidates[:3]]}")
        else:
            # FCFS和内存感知：抢占最近开始的请求
            candidates = sorted(self.running_requests, 
                              key=lambda r: r.start_time or 0, reverse=True)
            print(f"    📊 按开始时间排序候选请求: {[r.request_id for r in candidates[:3]]}")
        
        # 选择抢占目标
        for request in candidates:
            if self._can_preempt(request):
                preemption_mode = self._get_preemption_mode(request)
                
                print(f"    🎯 抢占请求 {request.request_id} (模式: {preemption_mode.value})")
                
                if preemption_mode == PreemptionMode.SWAP:
                    # 换出到CPU内存
                    try:
                        cpu_blocks = self.block_manager.swap_out(request)
                        request.status = RequestStatus.SWAPPED
                        self.running_requests.remove(request)
                        self.swapped_requests.append(request)
                        print(f"      💾 换出到CPU: {len(cpu_blocks)} 块")
                    except Exception as e:
                        print(f"      ❌ 换出失败: {e}")
                        continue
                        
                elif preemption_mode == PreemptionMode.RECOMPUTE:
                    # 重计算模式：直接移回等待队列
                    self.block_manager.free(request)
                    request.status = RequestStatus.WAITING
                    request.start_time = None
                    self.running_requests.remove(request)
                    self.waiting_requests.append(request)
                    print(f"      🔄 移回等待队列，稍后重计算")
                
                preempted_requests.append(request)
                
                # 检查是否释放了足够的内存
                if self.block_manager.get_num_free_gpu_blocks() > 10:  # 至少10个空闲块
                    print(f"    ✅ 抢占完成，释放了足够内存")
                    break
        
        if not preempted_requests:
            print(f"    ⚠️  没有找到合适的抢占目标")
        
        return preempted_requests
    
    def _can_preempt(self, request: SequenceRequest) -> bool:
        """判断请求是否可以被抢占"""
        # 高优先级请求不轻易抢占
        if request.priority > 5:
            return False
        
        # 即将完成的请求不抢占
        remaining_tokens = request.get_estimated_remaining_tokens()
        if remaining_tokens < 10:
            return False
        
        return True
    
    def _get_preemption_mode(self, request: SequenceRequest) -> PreemptionMode:
        """获取抢占模式"""
        # 如果已经生成了较多token，使用换出模式
        if request.generated_tokens > 10:
            return PreemptionMode.SWAP
        else:
            return PreemptionMode.RECOMPUTE
    
    def _get_request_priority(self, request: SequenceRequest) -> float:
        """计算请求的调度优先级"""
        if self.policy == SchedulerPolicy.FCFS:
            # 先来先服务：按到达时间排序
            return -request.arrival_time
        
        elif self.policy == SchedulerPolicy.SJF:
            # 最短作业优先：按预估执行时间排序
            estimated_time = request.get_estimated_remaining_tokens()
            return -estimated_time
        
        elif self.policy == SchedulerPolicy.PRIORITY:
            # 优先级调度：考虑优先级和等待时间
            base_priority = request.priority
            wait_time_bonus = request.get_wait_time() * 0.1  # 等待时间奖励
            return base_priority + wait_time_bonus
        
        elif self.policy == SchedulerPolicy.MEMORY_AWARE:
            # 内存感知调度：考虑内存效率
            memory_efficiency = self._calculate_memory_efficiency(request)
            priority_bonus = request.priority * 0.5
            return memory_efficiency + priority_bonus
        
        return 0.0
    
    def _calculate_memory_efficiency(self, request: SequenceRequest) -> float:
        """计算内存效率分数"""
        # 内存利用率：实际使用的token数 / 分配的内存块数
        allocated_blocks = request.get_memory_blocks_needed(self.block_size)
        if allocated_blocks == 0:
            return 0.0
        
        actual_tokens = request.total_tokens
        allocated_capacity = allocated_blocks * self.block_size
        
        utilization = actual_tokens / allocated_capacity
        
        # 考虑剩余生成长度
        remaining_ratio = request.get_estimated_remaining_tokens() / request.max_tokens
        
        return utilization * (1 + remaining_ratio)
    
    def _update_stats(self, scheduler_output: SchedulerOutput):
        """更新统计信息"""
        self.completed_requests += len(scheduler_output.finished_requests)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        memory_stats = self.block_manager.get_memory_stats()
        
        # 计算平均等待时间
        total_wait_time = 0
        wait_count = 0
        for request in self.finished:
            if request.start_time:
                total_wait_time += request.get_wait_time()
                wait_count += 1
        
        avg_wait_time = total_wait_time / wait_count if wait_count > 0 else 0
        
        # 计算平均执行时间
        total_exec_time = 0
        exec_count = 0
        for request in self.finished:
            if request.finish_time and request.start_time:
                total_exec_time += request.get_execution_time()
                exec_count += 1
        
        avg_exec_time = total_exec_time / exec_count if exec_count > 0 else 0
        
        return {
            "policy": self.policy.value,
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "waiting_requests": len(self.waiting),
            "running_requests": len(self.running),
            "swapped_requests": len(self.swapped),
            "preempted_count": self.preempted_count,
            "swap_in_count": self.swap_in_count,
            "swap_out_count": self.swap_out_count,
            "avg_wait_time": avg_wait_time,
            "avg_exec_time": avg_exec_time,
            "completion_rate": self.completed_requests / self.total_requests * 100 if self.total_requests > 0 else 0,
            **memory_stats
        }


def generate_test_requests(num_requests: int = 50) -> List[SequenceRequest]:
    """生成测试请求"""
    requests = []
    
    prompts = [
        "Explain the concept of machine learning",
        "Write a short story about a robot",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis",
        "How does blockchain technology work?",
        "Write a poem about the ocean",
        "Explain quantum computing in simple terms",
        "What is the history of artificial intelligence?",
        "Describe the solar system",
        "How do neural networks learn?"
    ]
    
    for i in range(num_requests):
        request = SequenceRequest(
            request_id=f"req_{i:04d}",
            prompt=random.choice(prompts),
            max_tokens=random.randint(20, 200),
            temperature=random.uniform(0.7, 1.3),
            top_p=random.uniform(0.8, 1.0),
            priority=random.randint(0, 10)
        )
        
        # 添加一些随机延迟模拟真实到达时间
        request.arrival_time = time.time() + random.uniform(0, 5)
        
        requests.append(request)
    
    return requests


def test_scheduler_policy(policy: SchedulerPolicy, requests: List[SequenceRequest]):
    """测试特定调度策略"""
    print(f"\n{'='*60}")
    print(f"🧪 测试调度策略：{policy.value.upper()}")
    print(f"{'='*60}")
    
    # 创建调度器
    scheduler = Scheduler(
        policy=policy,
        max_batch_size=8,
        max_num_seqs=100,
        num_gpu_blocks=256,
        num_cpu_blocks=512
    )
    
    # 复制请求列表以避免修改原始数据
    test_requests = []
    for req in requests:
        new_req = SequenceRequest(
            request_id=req.request_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            priority=req.priority
        )
        test_requests.append(new_req)
    
    # 模拟请求到达
    start_time = time.time()
    request_iter = iter(test_requests)
    
    # 运行调度循环
    step = 0
    while (scheduler.total_requests < len(test_requests) or 
           scheduler.waiting or scheduler.running or scheduler.swapped):
        
        step += 1
        print(f"\n--- 调度步骤 {step} ---")
        
        # 添加新到达的请求
        current_time = time.time()
        while True:
            try:
                next_request = next(request_iter)
                if next_request.arrival_time <= current_time:
                    scheduler.add_request(next_request)
                else:
                    # 将请求放回迭代器（简化处理）
                    break
            except StopIteration:
                break
        
        # 执行调度
        scheduler_output = scheduler.schedule()
        
        # 模拟执行延迟
        time.sleep(0.1)
        
        # 限制测试步数
        if step > 100:
            break
    
    # 打印最终统计
    stats = scheduler.get_stats()
    print(f"\n📊 {policy.value.upper()} 策略统计结果：")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return stats


def compare_scheduler_policies():
    """对比不同调度策略的性能"""
    print(f"\n🏆 调度策略性能对比")
    print(f"{'='*60}")
    
    # 生成测试请求
    test_requests = generate_test_requests(30)
    
    # 测试所有策略
    policies = [
        SchedulerPolicy.FCFS,
        SchedulerPolicy.SJF,
        SchedulerPolicy.PRIORITY,
        SchedulerPolicy.MEMORY_AWARE
    ]
    
    results = {}
    for policy in policies:
        stats = test_scheduler_policy(policy, test_requests)
        results[policy.value] = stats
    
    # 对比结果
    print(f"\n📈 性能对比总结：")
    print(f"{'策略':<15} {'完成率':<10} {'平均等待':<12} {'平均执行':<12} {'GPU利用率':<12}")
    print(f"{'-'*65}")
    
    for policy_name, stats in results.items():
        print(f"{policy_name:<15} "
              f"{stats['completion_rate']:<10.1f}% "
              f"{stats['avg_wait_time']:<12.2f}s "
              f"{stats['avg_exec_time']:<12.2f}s "
              f"{stats['gpu_utilization']:<12.1f}%")


def test_memory_pressure():
    """测试内存压力下的调度表现"""
    print(f"\n🧪 内存压力测试")
    print(f"{'='*60}")
    
    # 创建内存受限的调度器
    scheduler = Scheduler(
        policy=SchedulerPolicy.MEMORY_AWARE,
        max_batch_size=16,
        max_num_seqs=50,
        num_gpu_blocks=64,  # 较小的GPU内存
        num_cpu_blocks=128
    )
    
    # 生成大量请求
    requests = generate_test_requests(40)
    
    # 逐步添加请求，观察内存管理
    for i, request in enumerate(requests):
        scheduler.add_request(request)
        
        if i % 5 == 0:  # 每5个请求执行一次调度
            print(f"\n--- 添加了 {i+1} 个请求 ---")
            scheduler_output = scheduler.schedule()
            
            # 打印内存状态
            memory_stats = scheduler.block_manager.get_memory_stats()
            print(f"   GPU内存使用：{memory_stats['gpu_used']}/{memory_stats['gpu_total']} "
                  f"({memory_stats['gpu_utilization']:.1f}%)")
            print(f"   CPU内存使用：{memory_stats['cpu_used']}/{memory_stats['cpu_total']}")
            print(f"   抢占次数：{scheduler.preempted_count}")
            print(f"   换出次数：{scheduler.swap_out_count}")
            print(f"   换入次数：{scheduler.swap_in_count}")
    
    # 最终统计
    final_stats = scheduler.get_stats()
    print(f"\n📊 内存压力测试结果：")
    print(f"   总请求数：{final_stats['total_requests']}")
    print(f"   完成请求数：{final_stats['completed_requests']}")
    print(f"   抢占次数：{final_stats['preempted_count']}")
    print(f"   换出次数：{final_stats['swap_out_count']}")
    print(f"   换入次数：{final_stats['swap_in_count']}")
    print(f"   最终GPU利用率：{final_stats['gpu_utilization']:.1f}%")


def test_continuous_batching():
    """测试连续批处理的效果"""
    print(f"\n🧪 连续批处理测试")
    print(f"{'='*60}")
    
    scheduler = Scheduler(
        policy=SchedulerPolicy.FCFS,
        max_batch_size=8,
        max_num_seqs=50,
        num_gpu_blocks=128,
        num_cpu_blocks=256
    )
    
    # 创建不同长度的请求
    requests = [
        SequenceRequest(f"short_{i}", "Short prompt", 20, priority=1) 
        for i in range(5)
    ] + [
        SequenceRequest(f"medium_{i}", "Medium length prompt", 100, priority=1) 
        for i in range(5)
    ] + [
        SequenceRequest(f"long_{i}", "Very long prompt for testing", 200, priority=1) 
        for i in range(5)
    ]
    
    # 添加所有请求
    for request in requests:
        scheduler.add_request(request)
    
    # 模拟连续批处理
    step = 0
    batch_sizes = []
    
    while scheduler.waiting or scheduler.running or scheduler.swapped:
        step += 1
        print(f"\n--- 批处理步骤 {step} ---")
        
        scheduler_output = scheduler.schedule()
        batch_size = scheduler_output.get_batch_size()
        batch_sizes.append(batch_size)
        
        print(f"   当前批次大小：{batch_size}")
        print(f"   完成请求：{len(scheduler_output.finished_requests)}")
        
        # 显示批次中的请求类型
        if scheduler_output.scheduled_requests:
            request_types = {}
            for req in scheduler_output.scheduled_requests:
                req_type = req.request_id.split('_')[0]
                request_types[req_type] = request_types.get(req_type, 0) + 1
            
            print(f"   批次组成：{dict(request_types)}")
        
        time.sleep(0.1)
        
        if step > 50:  # 防止无限循环
            break
    
    # 分析批次大小变化
    print(f"\n📊 连续批处理分析：")
    print(f"   总调度步数：{len(batch_sizes)}")
    print(f"   平均批次大小：{sum(batch_sizes)/len(batch_sizes):.2f}")
    print(f"   最大批次大小：{max(batch_sizes)}")
    print(f"   最小批次大小：{min(batch_sizes)}")
    
    # 显示批次大小变化趋势
    print(f"   批次大小变化：{batch_sizes[:10]}..." if len(batch_sizes) > 10 else f"   批次大小变化：{batch_sizes}")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 第四步：请求调度与批处理优化")
    print("=" * 60)
    
    try:
        # 1. 对比不同调度策略
        compare_scheduler_policies()
        
        # 2. 内存压力测试
        test_memory_pressure()
        
        # 3. 连续批处理测试
        test_continuous_batching()
        
        print("\n🎉 所有测试完成！")
        print("\n📚 学习要点总结：")
        print("   1. 调度器是LLM推理系统的核心组件，负责资源分配和请求管理")
        print("   2. Continuous Batching 显著提升了系统吞吐量和资源利用率")
        print("   3. 不同调度策略有各自的优势和适用场景")
        print("   4. 内存感知调度能够更好地处理资源约束")
        print("   5. 抢占和换出机制是处理内存压力的重要手段")
        print("   6. 合理的批次管理策略能够平衡延迟和吞吐量")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序")
    except Exception as e:
        print(f"\n💥 程序异常：{str(e)}")
        import traceback
        traceback.print_exc()