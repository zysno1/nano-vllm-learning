# 请求调度器分析

## 🎯 调度器概览

请求调度器是 nano-vllm 的核心组件之一，负责管理和调度所有推理请求。它决定了哪些请求应该被处理、何时处理以及如何批处理，直接影响系统的吞吐量和延迟性能。

## 🏗️ 核心架构

```python
import asyncio
import heapq
import time
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

from nano_vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from nano_vllm.sampling import SamplingParams
from nano_vllm.memory import MemoryManager, KVCache
from nano_vllm.config import SchedulerConfig
from nano_vllm.utils.metrics import SchedulerMetrics

logger = logging.getLogger(__name__)

class SchedulingPolicy(Enum):
    """调度策略"""
    FCFS = "fcfs"  # 先来先服务
    PRIORITY = "priority"  # 优先级调度
    SJF = "sjf"  # 最短作业优先
    ROUND_ROBIN = "round_robin"  # 轮询调度

@dataclass
class SchedulerOutput:
    """调度器输出"""
    scheduled_seq_groups: List[SequenceGroup]
    preempted_seq_groups: List[SequenceGroup]
    ignored_seq_groups: List[SequenceGroup]
    num_batched_tokens: int
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    kv_caches: List[KVCache]

@dataclass
class RequestMetadata:
    """请求元数据"""
    request_id: str
    arrival_time: float
    priority: int
    estimated_tokens: int
    sampling_params: SamplingParams
    
    def __lt__(self, other):
        """用于优先队列排序"""
        if self.priority != other.priority:
            return self.priority > other.priority  # 高优先级优先
        return self.arrival_time < other.arrival_time  # 早到优先

class RequestScheduler:
    """请求调度器"""
    
    def __init__(self, config: SchedulerConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        
        # 调度队列
        self.waiting_queue: List[RequestMetadata] = []  # 等待队列（优先队列）
        self.running_sequences: Dict[str, SequenceGroup] = {}  # 运行中的序列
        self.swapped_sequences: Dict[str, SequenceGroup] = {}  # 交换出的序列
        
        # 调度状态
        self.current_batch_size = 0
        self.current_batch_tokens = 0
        self.last_schedule_time = 0.0
        
        # 性能指标
        self.metrics = SchedulerMetrics()
        
        # 线程安全
        self.lock = asyncio.Lock()
        
        # 调度策略
        self.policy = SchedulingPolicy(config.scheduling_policy)
        
        # 预抢占支持
        self.enable_preemption = config.enable_preemption
        self.preemption_threshold = config.preemption_threshold
        
        # 批处理优化
        self.max_batch_size = config.max_num_seqs
        self.max_batch_tokens = config.max_num_batched_tokens
        
        # 内存管理
        self.block_size = config.block_size
        self.max_blocks_per_seq = config.max_model_len // self.block_size
        
        logger.info(f"RequestScheduler initialized with policy: {self.policy.value}")
    
    async def add_request(self, request_metadata: RequestMetadata, sequence_group: SequenceGroup):
        """添加新请求"""
        async with self.lock:
            # 估算请求所需的token数
            estimated_tokens = self._estimate_request_tokens(request_metadata, sequence_group)
            request_metadata.estimated_tokens = estimated_tokens
            
            # 添加到等待队列
            heapq.heappush(self.waiting_queue, request_metadata)
            
            # 存储序列组
            self.running_sequences[request_metadata.request_id] = sequence_group
            
            # 更新指标
            self.metrics.record_request_arrival(request_metadata)
            
            logger.debug(f"Added request {request_metadata.request_id} to waiting queue")
    
    def _estimate_request_tokens(self, request_metadata: RequestMetadata, sequence_group: SequenceGroup) -> int:
        """估算请求所需的token数"""
        prompt_tokens = len(sequence_group.prompt_token_ids)
        max_new_tokens = request_metadata.sampling_params.max_tokens or self.config.default_max_tokens
        
        return prompt_tokens + max_new_tokens
    
    async def schedule(self) -> SchedulerOutput:
        """执行调度决策"""
        start_time = time.time()
        
        async with self.lock:
            # 1. 处理完成的序列
            await self._process_finished_sequences()
            
            # 2. 处理交换操作
            swap_in_requests = await self._handle_swap_operations()
            
            # 3. 调度新请求
            scheduled_requests = await self._schedule_new_requests()
            
            # 4. 处理抢占
            preempted_requests = await self._handle_preemption()
            
            # 5. 构建调度输出
            scheduler_output = await self._build_scheduler_output(
                scheduled_requests, preempted_requests, swap_in_requests
            )
            
            # 6. 更新调度状态
            await self._update_scheduler_state(scheduler_output)
            
            # 记录调度时间
            schedule_time = time.time() - start_time
            self.metrics.record_schedule_time(schedule_time)
            
            return scheduler_output
    
    async def _process_finished_sequences(self):
        """处理完成的序列"""
        finished_request_ids = []
        
        for request_id, seq_group in self.running_sequences.items():
            if seq_group.is_finished():
                finished_request_ids.append(request_id)
                
                # 释放内存块
                await self._free_sequence_blocks(seq_group)
                
                # 更新指标
                self.metrics.record_request_completion(seq_group)
        
        # 从运行队列中移除
        for request_id in finished_request_ids:
            del self.running_sequences[request_id]
            logger.debug(f"Removed finished request {request_id}")
    
    async def _handle_swap_operations(self) -> List[SequenceGroup]:
        """处理交换操作"""
        swap_in_requests = []
        
        if not self.swapped_sequences:
            return swap_in_requests
        
        # 按优先级排序交换队列
        sorted_swapped = sorted(
            self.swapped_sequences.items(),
            key=lambda x: (x[1].priority, x[1].arrival_time),
            reverse=True
        )
        
        for request_id, seq_group in sorted_swapped:
            # 检查是否有足够内存交换回来
            required_blocks = self._calculate_required_blocks(seq_group)
            
            if await self.memory_manager.can_allocate_blocks(required_blocks):
                # 执行交换
                await self._swap_in_sequence(seq_group)
                swap_in_requests.append(seq_group)
                
                # 从交换队列移除
                del self.swapped_sequences[request_id]
                self.running_sequences[request_id] = seq_group
                
                logger.debug(f"Swapped in request {request_id}")
        
        return swap_in_requests
    
    async def _schedule_new_requests(self) -> List[SequenceGroup]:
        """调度新请求"""
        scheduled_requests = []
        
        while self.waiting_queue and self._can_schedule_more():
            # 根据调度策略选择下一个请求
            next_request = await self._select_next_request()
            
            if next_request is None:
                break
            
            request_id = next_request.request_id
            seq_group = self.running_sequences.get(request_id)
            
            if seq_group is None:
                continue
            
            # 检查资源是否足够
            if await self._can_allocate_resources(seq_group):
                # 分配资源
                await self._allocate_sequence_resources(seq_group)
                scheduled_requests.append(seq_group)
                
                # 更新批处理状态
                self._update_batch_state(seq_group)
                
                logger.debug(f"Scheduled request {request_id}")
            else:
                # 资源不足，放回队列
                heapq.heappush(self.waiting_queue, next_request)
                break
        
        return scheduled_requests
    
    async def _select_next_request(self) -> Optional[RequestMetadata]:
        """根据调度策略选择下一个请求"""
        if not self.waiting_queue:
            return None
        
        if self.policy == SchedulingPolicy.FCFS:
            return heapq.heappop(self.waiting_queue)
        
        elif self.policy == SchedulingPolicy.PRIORITY:
            return heapq.heappop(self.waiting_queue)  # 已按优先级排序
        
        elif self.policy == SchedulingPolicy.SJF:
            # 找到最短的作业
            min_tokens = float('inf')
            min_index = -1
            
            for i, request in enumerate(self.waiting_queue):
                if request.estimated_tokens < min_tokens:
                    min_tokens = request.estimated_tokens
                    min_index = i
            
            if min_index >= 0:
                return self.waiting_queue.pop(min_index)
        
        elif self.policy == SchedulingPolicy.ROUND_ROBIN:
            # 轮询调度（简化实现）
            return heapq.heappop(self.waiting_queue)
        
        return None
    
    def _can_schedule_more(self) -> bool:
        """检查是否可以调度更多请求"""
        return (self.current_batch_size < self.max_batch_size and 
                self.current_batch_tokens < self.max_batch_tokens)
    
    async def _can_allocate_resources(self, seq_group: SequenceGroup) -> bool:
        """检查是否可以分配资源"""
        # 计算所需的内存块数
        required_blocks = self._calculate_required_blocks(seq_group)
        
        # 检查内存是否足够
        if not await self.memory_manager.can_allocate_blocks(required_blocks):
            return False
        
        # 检查批处理限制
        estimated_tokens = len(seq_group.prompt_token_ids)
        if (self.current_batch_size + 1 > self.max_batch_size or
            self.current_batch_tokens + estimated_tokens > self.max_batch_tokens):
            return False
        
        return True
    
    def _calculate_required_blocks(self, seq_group: SequenceGroup) -> int:
        """计算序列组所需的内存块数"""
        max_seq_len = 0
        for seq in seq_group.seqs:
            seq_len = len(seq.token_ids)
            max_seq_len = max(max_seq_len, seq_len)
        
        # 考虑未来的token生成
        max_new_tokens = seq_group.sampling_params.max_tokens or self.config.default_max_tokens
        total_len = max_seq_len + max_new_tokens
        
        return (total_len + self.block_size - 1) // self.block_size
    
    async def _allocate_sequence_resources(self, seq_group: SequenceGroup):
        """为序列组分配资源"""
        required_blocks = self._calculate_required_blocks(seq_group)
        
        # 分配内存块
        allocated_blocks = await self.memory_manager.allocate_blocks(
            seq_group.request_id, required_blocks
        )
        
        # 设置序列的内存块
        for seq in seq_group.seqs:
            seq.logical_token_blocks = allocated_blocks
    
    def _update_batch_state(self, seq_group: SequenceGroup):
        """更新批处理状态"""
        self.current_batch_size += len(seq_group.seqs)
        
        for seq in seq_group.seqs:
            self.current_batch_tokens += len(seq.token_ids)
    
    async def _handle_preemption(self) -> List[SequenceGroup]:
        """处理抢占"""
        preempted_requests = []
        
        if not self.enable_preemption:
            return preempted_requests
        
        # 检查是否需要抢占
        if (self.current_batch_tokens > self.preemption_threshold or
            len(self.running_sequences) > self.max_batch_size):
            
            # 选择要抢占的序列
            candidates = await self._select_preemption_candidates()
            
            for seq_group in candidates:
                await self._preempt_sequence(seq_group)
                preempted_requests.append(seq_group)
        
        return preempted_requests
    
    async def _select_preemption_candidates(self) -> List[SequenceGroup]:
        """选择抢占候选者"""
        candidates = []
        
        # 按优先级和到达时间排序
        sorted_sequences = sorted(
            self.running_sequences.values(),
            key=lambda x: (x.priority, x.arrival_time)
        )
        
        # 选择低优先级的序列进行抢占
        for seq_group in sorted_sequences:
            if len(candidates) >= self.config.max_preemptions_per_step:
                break
            
            # 只抢占低优先级的长序列
            if (seq_group.priority < self.config.min_preemption_priority and
                len(seq_group.seqs[0].token_ids) > self.config.min_preemption_length):
                candidates.append(seq_group)
        
        return candidates
    
    async def _preempt_sequence(self, seq_group: SequenceGroup):
        """抢占序列"""
        request_id = seq_group.request_id
        
        # 交换到CPU内存
        await self._swap_out_sequence(seq_group)
        
        # 从运行队列移除
        if request_id in self.running_sequences:
            del self.running_sequences[request_id]
        
        # 添加到交换队列
        self.swapped_sequences[request_id] = seq_group
        
        # 更新批处理状态
        self._remove_from_batch_state(seq_group)
        
        logger.debug(f"Preempted request {request_id}")
    
    async def _swap_out_sequence(self, seq_group: SequenceGroup):
        """将序列交换到CPU内存"""
        # 获取KV缓存
        kv_caches = []
        for seq in seq_group.seqs:
            if hasattr(seq, 'kv_cache'):
                kv_caches.append(seq.kv_cache)
        
        # 交换到CPU
        await self.memory_manager.swap_out(seq_group.request_id, kv_caches)
    
    async def _swap_in_sequence(self, seq_group: SequenceGroup):
        """将序列交换回GPU内存"""
        # 分配GPU内存
        await self._allocate_sequence_resources(seq_group)
        
        # 交换KV缓存
        await self.memory_manager.swap_in(seq_group.request_id)
    
    def _remove_from_batch_state(self, seq_group: SequenceGroup):
        """从批处理状态中移除"""
        self.current_batch_size -= len(seq_group.seqs)
        
        for seq in seq_group.seqs:
            self.current_batch_tokens -= len(seq.token_ids)
    
    async def _build_scheduler_output(
        self, 
        scheduled_requests: List[SequenceGroup],
        preempted_requests: List[SequenceGroup],
        swap_in_requests: List[SequenceGroup]
    ) -> SchedulerOutput:
        """构建调度器输出"""
        
        # 收集所有调度的序列组
        all_scheduled = scheduled_requests + swap_in_requests
        
        # 计算批处理token数
        num_batched_tokens = sum(
            len(seq.token_ids) for seq_group in all_scheduled 
            for seq in seq_group.seqs
        )
        
        # 获取内存操作
        blocks_to_swap_in = await self._get_swap_in_blocks(swap_in_requests)
        blocks_to_swap_out = await self._get_swap_out_blocks(preempted_requests)
        blocks_to_copy = await self._get_copy_blocks(all_scheduled)
        
        # 获取KV缓存
        kv_caches = await self._get_kv_caches(all_scheduled)
        
        return SchedulerOutput(
            scheduled_seq_groups=all_scheduled,
            preempted_seq_groups=preempted_requests,
            ignored_seq_groups=[],
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            kv_caches=kv_caches
        )
    
    async def _get_swap_in_blocks(self, swap_in_requests: List[SequenceGroup]) -> Dict[int, int]:
        """获取需要交换入的内存块"""
        blocks_to_swap_in = {}
        
        for seq_group in swap_in_requests:
            for seq in seq_group.seqs:
                if hasattr(seq, 'logical_token_blocks'):
                    for logical_block, physical_block in seq.logical_token_blocks.items():
                        blocks_to_swap_in[logical_block] = physical_block
        
        return blocks_to_swap_in
    
    async def _get_swap_out_blocks(self, preempted_requests: List[SequenceGroup]) -> Dict[int, int]:
        """获取需要交换出的内存块"""
        blocks_to_swap_out = {}
        
        for seq_group in preempted_requests:
            for seq in seq_group.seqs:
                if hasattr(seq, 'logical_token_blocks'):
                    for logical_block, physical_block in seq.logical_token_blocks.items():
                        blocks_to_swap_out[logical_block] = physical_block
        
        return blocks_to_swap_out
    
    async def _get_copy_blocks(self, scheduled_requests: List[SequenceGroup]) -> Dict[int, List[int]]:
        """获取需要复制的内存块"""
        blocks_to_copy = {}
        
        for seq_group in scheduled_requests:
            # 处理beam search等需要复制KV缓存的情况
            if len(seq_group.seqs) > 1:
                parent_seq = seq_group.seqs[0]
                for child_seq in seq_group.seqs[1:]:
                    if hasattr(parent_seq, 'logical_token_blocks') and hasattr(child_seq, 'logical_token_blocks'):
                        for logical_block in parent_seq.logical_token_blocks:
                            if logical_block not in blocks_to_copy:
                                blocks_to_copy[logical_block] = []
                            blocks_to_copy[logical_block].append(
                                child_seq.logical_token_blocks[logical_block]
                            )
        
        return blocks_to_copy
    
    async def _get_kv_caches(self, scheduled_requests: List[SequenceGroup]) -> List[KVCache]:
        """获取KV缓存"""
        kv_caches = []
        
        for seq_group in scheduled_requests:
            for seq in seq_group.seqs:
                if hasattr(seq, 'kv_cache'):
                    kv_caches.append(seq.kv_cache)
        
        return kv_caches
    
    async def _update_scheduler_state(self, scheduler_output: SchedulerOutput):
        """更新调度器状态"""
        # 重置批处理状态
        self.current_batch_size = 0
        self.current_batch_tokens = 0
        
        # 重新计算当前批处理状态
        for seq_group in scheduler_output.scheduled_seq_groups:
            self._update_batch_state(seq_group)
        
        # 更新调度时间
        self.last_schedule_time = time.time()
    
    async def _free_sequence_blocks(self, seq_group: SequenceGroup):
        """释放序列的内存块"""
        for seq in seq_group.seqs:
            if hasattr(seq, 'logical_token_blocks'):
                await self.memory_manager.free_blocks(
                    seq_group.request_id, 
                    list(seq.logical_token_blocks.keys())
                )
    
    async def abort_request(self, request_id: str):
        """中止请求"""
        async with self.lock:
            # 从等待队列中移除
            self.waiting_queue = [req for req in self.waiting_queue if req.request_id != request_id]
            heapq.heapify(self.waiting_queue)
            
            # 从运行队列中移除
            if request_id in self.running_sequences:
                seq_group = self.running_sequences.pop(request_id)
                await self._free_sequence_blocks(seq_group)
                self._remove_from_batch_state(seq_group)
            
            # 从交换队列中移除
            if request_id in self.swapped_sequences:
                seq_group = self.swapped_sequences.pop(request_id)
                await self.memory_manager.free_swap_space(request_id)
            
            logger.debug(f"Aborted request {request_id}")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        async with self.lock:
            return {
                'waiting_requests': len(self.waiting_queue),
                'running_requests': len(self.running_sequences),
                'swapped_requests': len(self.swapped_sequences),
                'current_batch_size': self.current_batch_size,
                'current_batch_tokens': self.current_batch_tokens,
                'last_schedule_time': self.last_schedule_time
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取调度器指标"""
        return self.metrics.get_metrics()
    
    async def shutdown(self):
        """关闭调度器"""
        logger.info("Shutting down request scheduler...")
        
        # 中止所有请求
        all_request_ids = (
            [req.request_id for req in self.waiting_queue] +
            list(self.running_sequences.keys()) +
            list(self.swapped_sequences.keys())
        )
        
        for request_id in all_request_ids:
            await self.abort_request(request_id)
        
        logger.info("Request scheduler shutdown complete")

# 调度器指标收集
class SchedulerMetrics:
    """调度器指标"""
    
    def __init__(self):
        self.request_count = 0
        self.completed_requests = 0
        self.total_wait_time = 0.0
        self.total_schedule_time = 0.0
        self.preemption_count = 0
        self.swap_count = 0
        
        # 队列长度历史
        self.queue_length_history = deque(maxlen=1000)
        
        # 延迟统计
        self.latency_history = deque(maxlen=1000)
    
    def record_request_arrival(self, request_metadata: RequestMetadata):
        """记录请求到达"""
        self.request_count += 1
    
    def record_request_completion(self, seq_group: SequenceGroup):
        """记录请求完成"""
        self.completed_requests += 1
        
        # 计算等待时间
        wait_time = time.time() - seq_group.arrival_time
        self.total_wait_time += wait_time
        self.latency_history.append(wait_time)
    
    def record_schedule_time(self, schedule_time: float):
        """记录调度时间"""
        self.total_schedule_time += schedule_time
    
    def record_preemption(self):
        """记录抢占事件"""
        self.preemption_count += 1
    
    def record_swap(self):
        """记录交换事件"""
        self.swap_count += 1
    
    def record_queue_length(self, queue_length: int):
        """记录队列长度"""
        self.queue_length_history.append(queue_length)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        avg_wait_time = (self.total_wait_time / self.completed_requests 
                        if self.completed_requests > 0 else 0.0)
        
        avg_schedule_time = (self.total_schedule_time / self.request_count 
                           if self.request_count > 0 else 0.0)
        
        avg_queue_length = (sum(self.queue_length_history) / len(self.queue_length_history)
                          if self.queue_length_history else 0.0)
        
        return {
            'request_count': self.request_count,
            'completed_requests': self.completed_requests,
            'avg_wait_time': avg_wait_time,
            'avg_schedule_time': avg_schedule_time,
            'avg_queue_length': avg_queue_length,
            'preemption_count': self.preemption_count,
            'swap_count': self.swap_count,
            'throughput': self.completed_requests / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0.0
        }

# 使用示例
async def example_scheduler_usage():
    """调度器使用示例"""
    
    # 创建配置
    config = SchedulerConfig(
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        scheduling_policy="priority",
        enable_preemption=True,
        preemption_threshold=1500
    )
    
    # 创建内存管理器
    memory_manager = MemoryManager()
    await memory_manager.initialize()
    
    # 创建调度器
    scheduler = RequestScheduler(config, memory_manager)
    
    try:
        # 添加请求
        for i in range(10):
            request_metadata = RequestMetadata(
                request_id=f"req_{i}",
                arrival_time=time.time(),
                priority=i % 3,  # 0-2优先级
                estimated_tokens=100 + i * 10,
                sampling_params=SamplingParams(max_tokens=50)
            )
            
            # 创建序列组（简化）
            seq_group = SequenceGroup(
                request_id=f"req_{i}",
                seqs=[],  # 实际使用中需要创建Sequence对象
                sampling_params=request_metadata.sampling_params,
                arrival_time=request_metadata.arrival_time,
                priority=request_metadata.priority
            )
            
            await scheduler.add_request(request_metadata, seq_group)
        
        # 执行调度
        for step in range(5):
            scheduler_output = await scheduler.schedule()
            
            print(f"Step {step}:")
            print(f"  Scheduled: {len(scheduler_output.scheduled_seq_groups)}")
            print(f"  Preempted: {len(scheduler_output.preempted_seq_groups)}")
            print(f"  Batch tokens: {scheduler_output.num_batched_tokens}")
            
            # 获取队列状态
            status = await scheduler.get_queue_status()
            print(f"  Queue status: {status}")
            
            await asyncio.sleep(0.1)
    
    finally:
        await scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(example_scheduler_usage())
```

## 🔧 关键特性分析

### 1. 多种调度策略

- **FCFS (First Come First Serve)**：先来先服务
- **Priority**：基于优先级的调度
- **SJF (Shortest Job First)**：最短作业优先
- **Round Robin**：轮询调度

### 2. 内存感知调度

- **内存预估**：预估请求所需内存
- **动态分配**：根据可用内存动态调度
- **交换机制**：支持内存不足时的交换

### 3. 抢占式调度

- **优先级抢占**：高优先级任务可抢占低优先级任务
- **资源回收**：及时回收被抢占任务的资源
- **公平性保证**：避免饥饿现象

## 📊 性能优化技术

### 批处理优化

```python
class BatchOptimizer:
    """批处理优化器"""
    
    def __init__(self, max_batch_size: int, max_batch_tokens: int):
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
    
    def optimize_batch(self, requests: List[SequenceGroup]) -> List[List[SequenceGroup]]:
        """优化批处理"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        # 按序列长度排序，相似长度的放在一起
        sorted_requests = sorted(requests, key=lambda x: len(x.seqs[0].token_ids))
        
        for request in sorted_requests:
            request_tokens = sum(len(seq.token_ids) for seq in request.seqs)
            
            if (len(current_batch) < self.max_batch_size and 
                current_tokens + request_tokens <= self.max_batch_tokens):
                current_batch.append(request)
                current_tokens += request_tokens
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [request]
                current_tokens = request_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
```

### 预测性调度

```python
class PredictiveScheduler:
    """预测性调度器"""
    
    def __init__(self):
        self.completion_time_predictor = CompletionTimePredictor()
        self.memory_usage_predictor = MemoryUsagePredictor()
    
    async def predict_and_schedule(self, requests: List[RequestMetadata]) -> List[RequestMetadata]:
        """基于预测进行调度"""
        # 预测每个请求的完成时间和内存使用
        predictions = []
        for request in requests:
            completion_time = await self.completion_time_predictor.predict(request)
            memory_usage = await self.memory_usage_predictor.predict(request)
            
            predictions.append({
                'request': request,
                'completion_time': completion_time,
                'memory_usage': memory_usage
            })
        
        # 基于预测结果优化调度顺序
        optimized_order = self._optimize_schedule_order(predictions)
        
        return [pred['request'] for pred in optimized_order]
    
    def _optimize_schedule_order(self, predictions: List[Dict]) -> List[Dict]:
        """优化调度顺序"""
        # 使用启发式算法优化调度顺序
        # 目标：最小化平均等待时间，最大化资源利用率
        
        # 简单的贪心算法：优先调度短任务
        return sorted(predictions, key=lambda x: x['completion_time'])
```

## 🚀 使用最佳实践

### 1. 调度策略选择

```python
# 高吞吐量场景
high_throughput_config = SchedulerConfig(
    scheduling_policy="sjf",  # 最短作业优先
    max_num_seqs=64,         # 大批处理
    max_num_batched_tokens=4096,
    enable_preemption=False   # 减少抢占开销
)

# 低延迟场景
low_latency_config = SchedulerConfig(
    scheduling_policy="priority",  # 优先级调度
    max_num_seqs=16,              # 小批处理
    max_num_batched_tokens=1024,
    enable_preemption=True,       # 启用抢占
    preemption_threshold=512
)

# 公平性场景
fairness_config = SchedulerConfig(
    scheduling_policy="round_robin",  # 轮询调度
    max_num_seqs=32,
    max_num_batched_tokens=2048,
    enable_preemption=True,
    preemption_threshold=1024
)
```

### 2. 内存管理优化

```python
# 内存优化配置
memory_optimized_config = SchedulerConfig(
    max_num_seqs=16,              # 减少并发数
    max_num_batched_tokens=1024,  # 减少批处理大小
    enable_preemption=True,       # 启用抢占释放内存
    swap_space=8,                 # 8GB交换空间
    block_size=16                 # 16 token块大小
)
```

### 3. 监控和调试

```python
async def monitor_scheduler(scheduler: RequestScheduler):
    """监控调度器性能"""
    while True:
        # 获取队列状态
        status = await scheduler.get_queue_status()
        
        # 获取性能指标
        metrics = await scheduler.get_metrics()
        
        # 检查异常情况
        if status['waiting_requests'] > 100:
            logger.warning("High queue length detected")
        
        if metrics['avg_wait_time'] > 10.0:
            logger.warning("High average wait time")
        
        # 记录指标
        logger.info(f"Scheduler status: {status}")
        logger.info(f"Scheduler metrics: {metrics}")
        
        await asyncio.sleep(5)
```

---

*请求调度器是 nano-vllm 性能的关键组件，合理的调度策略和优化可以显著提升系统的吞吐量和响应时间。*