#!/usr/bin/env python3
"""
第五步：完整推理流程与系统集成

这个文件实现了一个完整的nano-vLLM推理系统，整合了前面所有步骤的组件：
- 模型加载与配置
- LLM引擎初始化
- PagedAttention内存管理
- 智能调度器
- 端到端推理流程

学习重点：
1. 理解完整推理系统的架构设计
2. 掌握组件间的协调与集成
3. 学习性能优化和监控技巧
4. 了解生产环境的部署考虑
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# 核心数据结构定义
# ================================

class RequestStatus(Enum):
    """请求状态枚举"""
    WAITING = "waiting"      # 等待调度
    RUNNING = "running"      # 正在执行
    SWAPPED = "swapped"      # 已换出到CPU
    FINISHED = "finished"    # 已完成
    FAILED = "failed"        # 执行失败

class GenerationPhase(Enum):
    """生成阶段枚举"""
    PREFILL = "prefill"      # 预填充阶段
    DECODE = "decode"        # 解码阶段

@dataclass
class InferenceRequest:
    """推理请求定义"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    
    # 元数据
    user_id: Optional[str] = None
    priority: int = 0
    timeout: float = 300.0
    created_at: float = field(default_factory=time.time)

@dataclass
class InferenceResponse:
    """推理响应定义"""
    request_id: str
    generated_text: str
    tokens_generated: int
    total_time: float
    prefill_time: float
    decode_time: float
    tokens_per_second: float
    finish_reason: str
    
    # 统计信息
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: float
    requests_per_second: float
    tokens_per_second: float
    average_latency: float
    p95_latency: float
    p99_latency: float
    
    # 资源使用
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    cpu_utilization: float
    
    # 队列状态
    waiting_requests: int
    running_requests: int
    swapped_requests: int

# ================================
# 序列管理
# ================================

class Sequence:
    """序列对象，表示一个推理请求的完整生命周期"""
    
    def __init__(self, request: InferenceRequest, tokenizer):
        self.request = request
        self.tokenizer = tokenizer
        
        # 基本信息
        self.seq_id = request.request_id
        self.status = RequestStatus.WAITING
        self.phase = GenerationPhase.PREFILL
        
        # Token信息
        self.prompt_tokens = tokenizer.encode(request.prompt)
        self.generated_tokens = []
        self.current_length = len(self.prompt_tokens)
        
        # 时间统计
        self.created_at = time.time()
        self.prefill_start_time = None
        self.prefill_end_time = None
        self.decode_start_time = None
        self.last_token_time = None
        
        # 内存管理
        self.gpu_blocks = []
        self.cpu_blocks = []
        
        # 生成控制
        self.finished = False
        self.finish_reason = None
        
        print(f"🔄 创建序列 {self.seq_id[:8]}...")
        print(f"   📝 提示词长度: {len(self.prompt_tokens)} tokens")
        print(f"   🎯 最大生成长度: {request.max_tokens} tokens")
    
    def get_all_tokens(self) -> List[int]:
        """获取所有token（提示词+生成的）"""
        return self.prompt_tokens + self.generated_tokens
    
    def get_last_token(self) -> Optional[int]:
        """获取最后一个token"""
        if self.generated_tokens:
            return self.generated_tokens[-1]
        return None
    
    def add_token(self, token: int):
        """添加新生成的token"""
        self.generated_tokens.append(token)
        self.current_length += 1
        self.last_token_time = time.time()
        
        # 检查是否完成
        if self.current_length >= len(self.prompt_tokens) + self.request.max_tokens:
            self.finish_reason = "length"
            self.finished = True
        
        # 检查停止序列
        if self.request.stop_sequences:
            generated_text = self.tokenizer.decode(self.generated_tokens)
            for stop_seq in self.request.stop_sequences:
                if stop_seq in generated_text:
                    self.finish_reason = "stop"
                    self.finished = True
                    break
    
    def is_prefill(self) -> bool:
        """是否处于预填充阶段"""
        return self.phase == GenerationPhase.PREFILL
    
    def is_decode(self) -> bool:
        """是否处于解码阶段"""
        return self.phase == GenerationPhase.DECODE
    
    def start_prefill(self):
        """开始预填充阶段"""
        self.phase = GenerationPhase.PREFILL
        self.prefill_start_time = time.time()
        print(f"🚀 序列 {self.seq_id[:8]} 开始预填充阶段")
    
    def finish_prefill(self):
        """完成预填充阶段"""
        self.phase = GenerationPhase.DECODE
        self.prefill_end_time = time.time()
        self.decode_start_time = time.time()
        prefill_time = self.prefill_end_time - self.prefill_start_time
        print(f"✅ 序列 {self.seq_id[:8]} 预填充完成，耗时: {prefill_time:.3f}s")
    
    def get_generated_text(self) -> str:
        """获取生成的文本"""
        if not self.generated_tokens:
            return ""
        return self.tokenizer.decode(self.generated_tokens, skip_special_tokens=True)

# ================================
# 内存管理（简化版PagedAttention）
# ================================

class Block:
    """内存块"""
    def __init__(self, block_id: int, block_size: int, device: str = "cuda"):
        self.block_id = block_id
        self.block_size = block_size
        self.device = device
        self.ref_count = 0
        self.is_free = True
        
        # KV缓存存储
        self.key_cache = None
        self.value_cache = None

class BlockAllocator:
    """内存块分配器"""
    
    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int, block_size: int):
        self.block_size = block_size
        
        # GPU块池
        self.gpu_blocks = [
            Block(i, block_size, "cuda") 
            for i in range(num_gpu_blocks)
        ]
        self.free_gpu_blocks = list(self.gpu_blocks)
        
        # CPU块池
        self.cpu_blocks = [
            Block(i + num_gpu_blocks, block_size, "cpu") 
            for i in range(num_cpu_blocks)
        ]
        self.free_cpu_blocks = list(self.cpu_blocks)
        
        print(f"🧠 初始化内存分配器:")
        print(f"   📦 GPU块数量: {num_gpu_blocks}, 每块大小: {block_size}")
        print(f"   💾 CPU块数量: {num_cpu_blocks}, 每块大小: {block_size}")
    
    def allocate_gpu_blocks(self, num_blocks: int) -> List[Block]:
        """分配GPU内存块"""
        if len(self.free_gpu_blocks) < num_blocks:
            return []
        
        allocated = []
        for _ in range(num_blocks):
            block = self.free_gpu_blocks.pop()
            block.is_free = False
            block.ref_count = 1
            allocated.append(block)
        
        print(f"📦 分配 {num_blocks} 个GPU块，剩余: {len(self.free_gpu_blocks)}")
        return allocated
    
    def free_gpu_blocks(self, blocks: List[Block]):
        """释放GPU内存块"""
        for block in blocks:
            block.is_free = True
            block.ref_count = 0
            self.free_gpu_blocks.append(block)
        
        print(f"🗑️ 释放 {len(blocks)} 个GPU块，剩余: {len(self.free_gpu_blocks)}")
    
    def get_available_gpu_blocks(self) -> int:
        """获取可用GPU块数量"""
        return len(self.free_gpu_blocks)

# ================================
# 调度器
# ================================

class Scheduler:
    """智能调度器"""
    
    def __init__(self, max_num_seqs: int, block_allocator: BlockAllocator):
        self.max_num_seqs = max_num_seqs
        self.block_allocator = block_allocator
        
        # 请求队列
        self.waiting_queue: List[Sequence] = []
        self.running_queue: List[Sequence] = []
        self.swapped_queue: List[Sequence] = []
        self.finished_queue: List[Sequence] = []
        
        print(f"📋 初始化调度器，最大并发序列数: {max_num_seqs}")
    
    def add_sequence(self, sequence: Sequence):
        """添加新序列到等待队列"""
        sequence.status = RequestStatus.WAITING
        self.waiting_queue.append(sequence)
        print(f"➕ 序列 {sequence.seq_id[:8]} 加入等待队列")
    
    def schedule(self) -> Dict[str, List[Sequence]]:
        """调度序列执行"""
        print(f"    🎯 开始调度决策")
        
        prefill_seqs = []
        decode_seqs = []
        
        # 处理等待队列中的序列
        waiting_count = len(self.waiting)
        if waiting_count > 0:
            print(f"      📋 处理等待队列: {waiting_count} 个序列")
            
            # 尝试分配内存并启动新序列
            while self.waiting and len(self.running) < self.max_num_seqs:
                sequence = self.waiting.pop(0)
                
                # 估算需要的block数量
                estimated_blocks = (len(sequence.get_all_tokens()) + sequence.max_tokens) // 16 + 1
                
                if self.block_allocator.get_available_gpu_blocks() >= estimated_blocks:
                    # 分配内存
                    blocks = self.block_allocator.allocate_gpu_blocks(estimated_blocks)
                    sequence.blocks = blocks
                    sequence.status = RequestStatus.RUNNING
                    self.running.append(sequence)
                    
                    if sequence.is_prefill():
                        prefill_seqs.append(sequence)
                        print(f"        ✅ 启动 Prefill: {sequence.request_id} (需要 {estimated_blocks} blocks)")
                    else:
                        decode_seqs.append(sequence)
                        print(f"        ✅ 启动 Decode: {sequence.request_id} (需要 {estimated_blocks} blocks)")
                else:
                    # 内存不足，放回队列
                    self.waiting.insert(0, sequence)
                    print(f"        ❌ 内存不足，无法启动: {sequence.request_id} (需要 {estimated_blocks} blocks)")
                    break
        
        # 处理运行中的序列
        running_count = len(self.running)
        if running_count > 0:
            print(f"      🏃 处理运行队列: {running_count} 个序列")
            
            for sequence in self.running[:]:  # 使用切片避免修改列表时的问题
                if sequence.is_finished():
                    # 释放内存
                    if hasattr(sequence, 'blocks'):
                        self.block_allocator.free_gpu_blocks(sequence.blocks)
                    self.running.remove(sequence)
                    print(f"        🏁 序列完成: {sequence.request_id}")
                elif sequence.is_decode():
                    decode_seqs.append(sequence)
        
        # 处理换出队列
        swapped_count = len(self.swapped)
        if swapped_count > 0:
            print(f"      💾 检查换出队列: {swapped_count} 个序列")
            # 尝试换入一些序列（简化实现）
            if len(self.running) < self.max_num_seqs and self.swapped:
                sequence = self.swapped.pop(0)
                # 尝试重新分配GPU内存
                estimated_blocks = (len(sequence.get_all_tokens()) + sequence.max_tokens) // 16 + 1
                if self.block_allocator.get_available_gpu_blocks() >= estimated_blocks:
                    blocks = self.block_allocator.allocate_gpu_blocks(estimated_blocks)
                    sequence.blocks = blocks
                    sequence.status = RequestStatus.RUNNING
                    self.running.append(sequence)
                    decode_seqs.append(sequence)
                    print(f"        🔄 换入序列: {sequence.request_id}")
                else:
                    self.swapped.insert(0, sequence)
        
        result = {
            "prefill": prefill_seqs,
            "decode": decode_seqs
        }
        
        total_scheduled = len(prefill_seqs) + len(decode_seqs)
        if total_scheduled > 0:
            print(f"      📊 调度完成: {len(prefill_seqs)} prefill + {len(decode_seqs)} decode = {total_scheduled} 总计")
        
        return result
    
    def _try_swap_out(self) -> bool:
        """尝试换出序列以释放内存"""
        # 简化实现：换出最老的decode序列
        decode_seqs = [seq for seq in self.running_queue if seq.is_decode()]
        if decode_seqs:
            # 按创建时间排序，换出最老的
            decode_seqs.sort(key=lambda x: x.created_at)
            seq_to_swap = decode_seqs[0]
            
            # 换出到CPU
            self.running_queue.remove(seq_to_swap)
            self.swapped_queue.append(seq_to_swap)
            seq_to_swap.status = RequestStatus.SWAPPED
            
            # 释放GPU内存
            if seq_to_swap.gpu_blocks:
                self.block_allocator.free_gpu_blocks(seq_to_swap.gpu_blocks)
                seq_to_swap.gpu_blocks = []
            
            print(f"💾 序列 {seq_to_swap.seq_id[:8]} 换出到CPU")
            return True
        
        return False
    
    def get_queue_status(self) -> Dict[str, int]:
        """获取队列状态"""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_queue),
            "swapped": len(self.swapped_queue),
            "finished": len(self.finished_queue)
        }

# ================================
# 模型执行器
# ================================

class ModelExecutor:
    """模型执行器"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 性能统计
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0
        self.total_tokens_generated = 0
        
        print(f"🤖 初始化模型执行器，设备: {device}")
    
    def execute_prefill(self, sequences: List[Sequence]) -> Dict[str, torch.Tensor]:
        """执行预填充阶段"""
        if not sequences:
            return {}
        
        start_time = time.time()
        
        # 准备输入
        input_ids_list = []
        attention_masks = []
        
        for seq in sequences:
            tokens = seq.get_all_tokens()
            input_ids_list.append(torch.tensor(tokens, device=self.device))
            attention_masks.append(torch.ones(len(tokens), device=self.device))
        
        # 批处理
        max_len = max(len(ids) for ids in input_ids_list)
        batch_input_ids = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
        batch_attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)
        
        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_masks)):
            batch_input_ids[i, :len(ids)] = ids
            batch_attention_mask[i, :len(mask)] = mask
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=True
            )
        
        # 更新序列状态
        for seq in sequences:
            seq.finish_prefill()
        
        prefill_time = time.time() - start_time
        self.total_prefill_time += prefill_time
        
        print(f"⚡ 预填充批次完成: {len(sequences)} 个序列, 耗时: {prefill_time:.3f}s")
        
        return {
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values
        }
    
    def execute_decode(self, sequences: List[Sequence]) -> Dict[str, torch.Tensor]:
        """执行解码阶段"""
        if not sequences:
            return {}
        
        start_time = time.time()
        
        # 准备输入（只需要最后一个token）
        input_ids = torch.tensor(
            [[seq.get_last_token() or seq.prompt_tokens[-1]] for seq in sequences],
            device=self.device
        )
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True
            )
        
        # 采样新token
        logits = outputs.logits[:, -1, :]  # 取最后一个位置的logits
        
        new_tokens = []
        for i, seq in enumerate(sequences):
            # 应用温度和top-p采样
            token = self._sample_token(
                logits[i], 
                seq.request.temperature, 
                seq.request.top_p,
                seq.request.top_k
            )
            new_tokens.append(token)
            seq.add_token(token)
        
        decode_time = time.time() - start_time
        self.total_decode_time += decode_time
        self.total_tokens_generated += len(sequences)
        
        print(f"🔤 解码批次完成: {len(sequences)} 个token, 耗时: {decode_time:.3f}s")
        
        return {
            "new_tokens": new_tokens,
            "logits": outputs.logits
        }
    
    def _sample_token(self, logits: torch.Tensor, temperature: float, 
                     top_p: float, top_k: int) -> int:
        """采样下一个token"""
        # 应用温度
        if temperature > 0:
            logits = logits / temperature
        
        # Top-k过滤
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(0, top_k_indices, top_k_logits)
        
        # Top-p过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 找到累积概率超过top_p的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1).item()
        
        return token
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        return {
            "total_prefill_time": self.total_prefill_time,
            "total_decode_time": self.total_decode_time,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_prefill_time": self.total_prefill_time / max(1, self.total_tokens_generated),
            "avg_decode_time": self.total_decode_time / max(1, self.total_tokens_generated),
            "tokens_per_second": self.total_tokens_generated / max(0.001, self.total_decode_time)
        }

# ================================
# 性能监控器
# ================================

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.request_times = []
        self.token_counts = []
        self.latencies = []
        self.start_time = time.time()
        
        # GPU监控
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()
        
        print("📊 性能监控器已启动")
    
    def record_request(self, tokens_generated: int, latency: float):
        """记录请求性能"""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_counts.append(tokens_generated)
        self.latencies.append(latency)
        
        # 保持最近1000条记录
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
            self.token_counts = self.token_counts[-1000:]
            self.latencies = self.latencies[-1000:]
    
    def get_current_metrics(self) -> SystemMetrics:
        """获取当前系统指标"""
        current_time = time.time()
        
        # 计算吞吐量（最近60秒）
        recent_cutoff = current_time - 60.0
        recent_requests = [t for t in self.request_times if t > recent_cutoff]
        recent_tokens = [self.token_counts[i] for i, t in enumerate(self.request_times) if t > recent_cutoff]
        
        rps = len(recent_requests) / 60.0 if recent_requests else 0.0
        tps = sum(recent_tokens) / 60.0 if recent_tokens else 0.0
        
        # 计算延迟统计
        if self.latencies:
            sorted_latencies = sorted(self.latencies[-100:])  # 最近100个请求
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))] if sorted_latencies else 0.0
            p99_latency = sorted_latencies[int(0.99 * len(sorted_latencies))] if sorted_latencies else 0.0
        else:
            avg_latency = p95_latency = p99_latency = 0.0
        
        # GPU内存使用
        gpu_memory_used = gpu_memory_total = gpu_utilization = 0.0
        if self.gpu_available:
            gpu_memory_used = torch.cuda.memory_allocated(self.gpu_device) / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(self.gpu_device).total_memory / 1024**3  # GB
            gpu_utilization = torch.cuda.utilization(self.gpu_device) if hasattr(torch.cuda, 'utilization') else 0.0
        
        # CPU使用率
        cpu_utilization = psutil.cpu_percent()
        
        return SystemMetrics(
            timestamp=current_time,
            requests_per_second=rps,
            tokens_per_second=tps,
            average_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization,
            waiting_requests=0,  # 将由调度器填充
            running_requests=0,
            swapped_requests=0
        )
    
    def print_metrics(self, metrics: SystemMetrics, queue_status: Dict[str, int]):
        """打印性能指标"""
        print("\n" + "="*60)
        print("📊 系统性能监控")
        print("="*60)
        print(f"🚀 吞吐量:")
        print(f"   请求/秒: {metrics.requests_per_second:.2f}")
        print(f"   Token/秒: {metrics.tokens_per_second:.2f}")
        print(f"⏱️ 延迟统计:")
        print(f"   平均延迟: {metrics.average_latency:.3f}s")
        print(f"   P95延迟: {metrics.p95_latency:.3f}s")
        print(f"   P99延迟: {metrics.p99_latency:.3f}s")
        print(f"💾 资源使用:")
        print(f"   GPU内存: {metrics.gpu_memory_used:.2f}GB / {metrics.gpu_memory_total:.2f}GB")
        print(f"   GPU利用率: {metrics.gpu_utilization:.1f}%")
        print(f"   CPU利用率: {metrics.cpu_utilization:.1f}%")
        print(f"📋 队列状态:")
        print(f"   等待中: {queue_status['waiting']}")
        print(f"   运行中: {queue_status['running']}")
        print(f"   已换出: {queue_status['swapped']}")
        print(f"   已完成: {queue_status['finished']}")
        print("="*60)

# ================================
# 完整推理引擎
# ================================

class NanoVLLMEngine:
    """完整的nano-vLLM推理引擎"""
    
    def __init__(self, model_name: str, max_num_seqs: int = 32, 
                 block_size: int = 16, num_gpu_blocks: int = 1000):
        print("🚀 初始化 nano-vLLM 推理引擎...")
        
        # 加载模型和分词器
        print("📚 加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()
        
        # 初始化组件
        self.block_allocator = BlockAllocator(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_gpu_blocks // 2,
            block_size=block_size
        )
        
        self.scheduler = Scheduler(max_num_seqs, self.block_allocator)
        self.model_executor = ModelExecutor(self.model, self.tokenizer)
        self.performance_monitor = PerformanceMonitor()
        
        # 运行状态
        self.running = False
        self.engine_loop_task = None
        
        print("✅ nano-vLLM 引擎初始化完成!")
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """生成响应（非流式）"""
        print(f"\n🎯 开始处理请求: {request.request_id}")
        print(f"    📝 Prompt: {request.prompt[:100]}{'...' if len(request.prompt) > 100 else ''}")
        print(f"    🎛️  参数: max_tokens={request.max_tokens}, temp={request.temperature}, top_p={request.top_p}")
        
        start_time = time.time()
        
        # 创建序列
        sequence = Sequence(request, self.tokenizer)
        print(f"    🔤 Tokenization: {len(sequence.prompt_tokens)} tokens")
        
        # 添加到调度器
        self.scheduler.add_sequence(sequence)
        print(f"    📋 已添加到调度队列")
        
        # 等待完成
        while not sequence.is_finished():
            await asyncio.sleep(0.01)  # 让出控制权
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 构建响应
        generated_text = sequence.get_generated_text()
        tokens_generated = len(sequence.output_tokens)
        
        print(f"    ✅ 生成完成:")
        print(f"      📊 生成 {tokens_generated} tokens，耗时 {total_time:.2f}s")
        print(f"      🚀 速度: {tokens_generated/total_time:.1f} tokens/s")
        print(f"      📝 结果: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
        
        response = InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            tokens_generated=tokens_generated,
            total_time=total_time,
            prefill_time=sequence.prefill_time or 0,
            decode_time=sequence.decode_time or 0,
            tokens_per_second=tokens_generated / total_time if total_time > 0 else 0,
            finish_reason="length" if tokens_generated >= request.max_tokens else "stop",
            prompt_tokens=len(sequence.prompt_tokens),
            completion_tokens=tokens_generated,
            total_tokens=len(sequence.prompt_tokens) + tokens_generated
        )
        
        # 记录性能指标
        self.performance_monitor.record_request(tokens_generated, total_time)
        
        return response
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """流式生成"""
        # 创建序列
        sequence = Sequence(request, self.tokenizer)
        self.scheduler.add_sequence(sequence)
        
        last_generated_length = 0
        
        while not sequence.finished:
            await asyncio.sleep(0.01)
            
            # 检查是否有新token
            current_length = len(sequence.generated_tokens)
            if current_length > last_generated_length:
                # 获取新生成的token
                new_tokens = sequence.generated_tokens[last_generated_length:]
                new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                yield new_text
                last_generated_length = current_length
    
    async def start_engine_loop(self):
        """启动引擎主循环"""
        print("\n🚀 启动 NanoVLLM 引擎主循环")
        print("=" * 60)
        
        self.running = True
        iteration = 0
        
        while self.running:
            try:
                iteration += 1
                print(f"\n🔄 引擎循环 #{iteration}")
                
                # 调度序列
                scheduled = self.scheduler.schedule()
                
                # 显示调度结果
                prefill_count = len(scheduled.get('prefill', []))
                decode_count = len(scheduled.get('decode', []))
                
                if prefill_count > 0 or decode_count > 0:
                    print(f"    📋 调度结果: {prefill_count} prefill, {decode_count} decode")
                
                # 执行prefill
                if scheduled['prefill']:
                    print(f"    🔥 执行 Prefill: {len(scheduled['prefill'])} 序列")
                    start_time = time.time()
                    self.model_executor.execute_prefill(scheduled['prefill'])
                    prefill_time = time.time() - start_time
                    print(f"      ⏱️  Prefill 耗时: {prefill_time:.3f}s")
                
                # 执行decode
                if scheduled['decode']:
                    print(f"    🎯 执行 Decode: {len(scheduled['decode'])} 序列")
                    start_time = time.time()
                    self.model_executor.execute_decode(scheduled['decode'])
                    decode_time = time.time() - start_time
                    print(f"      ⏱️  Decode 耗时: {decode_time:.3f}s")
                
                # 显示队列状态（每10次循环显示一次）
                if iteration % 10 == 0:
                    queue_status = self.scheduler.get_queue_status()
                    if any(queue_status.values()):
                        print(f"    📊 队列状态: waiting={queue_status['waiting']}, "
                              f"running={queue_status['running']}, swapped={queue_status['swapped']}")
                
                await asyncio.sleep(0.01)  # 避免CPU占用过高
                
            except Exception as e:
                logger.error(f"引擎循环出错: {e}")
                await asyncio.sleep(0.1)
    
    def stop_engine_loop(self):
        """停止引擎循环"""
        self.running = False
        print("🛑 停止引擎循环")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        metrics = self.performance_monitor.get_current_metrics()
        queue_status = self.scheduler.get_queue_status()
        executor_stats = self.model_executor.get_performance_stats()
        
        return {
            "performance": metrics,
            "queue_status": queue_status,
            "executor_stats": executor_stats
        }
    
    def print_status(self):
        """打印引擎状态"""
        metrics = self.performance_monitor.get_current_metrics()
        queue_status = self.scheduler.get_queue_status()
        self.performance_monitor.print_metrics(metrics, queue_status)

# ================================
# 测试和演示函数
# ================================

async def test_single_request():
    """测试单个请求"""
    print("\n🧪 测试单个推理请求")
    print("="*50)
    
    # 初始化引擎
    engine = NanoVLLMEngine(
        model_name="gpt2",  # 使用较小的模型进行测试
        max_num_seqs=4,
        num_gpu_blocks=100
    )
    
    # 启动引擎循环
    engine_task = asyncio.create_task(engine.start_engine_loop())
    
    try:
        # 创建测试请求
        request = InferenceRequest(
            request_id="test-001",
            prompt="The future of artificial intelligence is",
            max_tokens=50,
            temperature=0.8,
            top_p=0.9
        )
        
        print(f"📝 发送请求: {request.prompt}")
        
        # 执行推理
        response = await engine.generate(request)
        
        # 打印结果
        print(f"\n✅ 推理完成!")
        print(f"🎯 生成文本: {response.generated_text}")
        print(f"📊 统计信息:")
        print(f"   总耗时: {response.total_time:.3f}s")
        print(f"   预填充耗时: {response.prefill_time:.3f}s")
        print(f"   解码耗时: {response.decode_time:.3f}s")
        print(f"   生成速度: {response.tokens_per_second:.2f} tokens/s")
        print(f"   Token数量: {response.tokens_generated}")
        
    finally:
        engine.stop_engine_loop()
        await asyncio.sleep(0.1)  # 等待循环停止

async def test_concurrent_requests():
    """测试并发请求"""
    print("\n🧪 测试并发推理请求")
    print("="*50)
    
    # 初始化引擎
    engine = NanoVLLMEngine(
        model_name="gpt2",
        max_num_seqs=8,
        num_gpu_blocks=200
    )
    
    # 启动引擎循环
    engine_task = asyncio.create_task(engine.start_engine_loop())
    
    try:
        # 创建多个测试请求
        prompts = [
            "The benefits of renewable energy include",
            "Machine learning algorithms can help",
            "Climate change is affecting",
            "The development of quantum computing",
            "Artificial intelligence will transform"
        ]
        
        requests = [
            InferenceRequest(
                request_id=f"concurrent-{i:03d}",
                prompt=prompt,
                max_tokens=30,
                temperature=0.7
            )
            for i, prompt in enumerate(prompts)
        ]
        
        print(f"🚀 发送 {len(requests)} 个并发请求...")
        
        # 并发执行
        start_time = time.time()
        tasks = [engine.generate(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 打印结果
        print(f"\n✅ 所有请求完成!")
        print(f"⏱️ 总耗时: {total_time:.3f}s")
        print(f"🚀 平均吞吐量: {len(requests)/total_time:.2f} requests/s")
        
        total_tokens = sum(r.tokens_generated for r in responses)
        print(f"🔤 总Token数: {total_tokens}")
        print(f"📈 Token吞吐量: {total_tokens/total_time:.2f} tokens/s")
        
        print(f"\n📋 详细结果:")
        for i, (req, resp) in enumerate(zip(requests, responses)):
            print(f"  {i+1}. [{resp.request_id}] {resp.tokens_generated} tokens, "
                  f"{resp.tokens_per_second:.1f} t/s")
        
        # 打印引擎状态
        engine.print_status()
        
    finally:
        engine.stop_engine_loop()
        await asyncio.sleep(0.1)

async def test_streaming_generation():
    """测试流式生成"""
    print("\n🧪 测试流式生成")
    print("="*50)
    
    # 初始化引擎
    engine = NanoVLLMEngine(
        model_name="gpt2",
        max_num_seqs=2,
        num_gpu_blocks=100
    )
    
    # 启动引擎循环
    engine_task = asyncio.create_task(engine.start_engine_loop())
    
    try:
        # 创建流式请求
        request = InferenceRequest(
            request_id="stream-001",
            prompt="Once upon a time in a distant galaxy",
            max_tokens=80,
            temperature=0.8,
            stream=True
        )
        
        print(f"📝 开始流式生成: {request.prompt}")
        print("🔄 生成中...")
        print("-" * 50)
        
        # 流式接收结果
        full_text = ""
        async for text_chunk in engine.generate_stream(request):
            print(text_chunk, end="", flush=True)
            full_text += text_chunk
        
        print("\n" + "-" * 50)
        print(f"✅ 流式生成完成!")
        print(f"📝 完整文本: {request.prompt}{full_text}")
        
    finally:
        engine.stop_engine_loop()
        await asyncio.sleep(0.1)

def test_performance_benchmark():
    """性能基准测试"""
    print("\n🧪 性能基准测试")
    print("="*50)
    
    async def run_benchmark():
        # 初始化引擎
        engine = NanoVLLMEngine(
            model_name="gpt2",
            max_num_seqs=16,
            num_gpu_blocks=500
        )
        
        # 启动引擎循环
        engine_task = asyncio.create_task(engine.start_engine_loop())
        
        try:
            # 预热
            print("🔥 预热阶段...")
            warmup_requests = [
                InferenceRequest(
                    request_id=f"warmup-{i}",
                    prompt="This is a warmup request",
                    max_tokens=10
                )
                for i in range(5)
            ]
            
            await asyncio.gather(*[engine.generate(req) for req in warmup_requests])
            
            # 基准测试
            print("📊 开始基准测试...")
            
            test_configs = [
                {"batch_size": 1, "max_tokens": 50},
                {"batch_size": 4, "max_tokens": 50},
                {"batch_size": 8, "max_tokens": 50},
                {"batch_size": 16, "max_tokens": 50},
            ]
            
            for config in test_configs:
                batch_size = config["batch_size"]
                max_tokens = config["max_tokens"]
                
                print(f"\n🔬 测试配置: 批次大小={batch_size}, 最大Token={max_tokens}")
                
                # 创建请求
                requests = [
                    InferenceRequest(
                        request_id=f"bench-{batch_size}-{i}",
                        prompt=f"Benchmark test prompt number {i} with batch size {batch_size}",
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    for i in range(batch_size)
                ]
                
                # 执行测试
                start_time = time.time()
                responses = await asyncio.gather(*[engine.generate(req) for req in requests])
                end_time = time.time()
                
                # 计算指标
                total_time = end_time - start_time
                total_tokens = sum(r.tokens_generated for r in responses)
                avg_latency = sum(r.total_time for r in responses) / len(responses)
                
                print(f"   ⏱️ 总耗时: {total_time:.3f}s")
                print(f"   📈 吞吐量: {batch_size/total_time:.2f} req/s, {total_tokens/total_time:.2f} tokens/s")
                print(f"   🎯 平均延迟: {avg_latency:.3f}s")
                
                # 等待一下再进行下一个测试
                await asyncio.sleep(1.0)
            
            # 最终状态
            print(f"\n📊 最终引擎状态:")
            engine.print_status()
            
        finally:
            engine.stop_engine_loop()
            await asyncio.sleep(0.1)
    
    # 运行基准测试
    asyncio.run(run_benchmark())

def main():
    """主函数"""
    print("🎉 欢迎使用 nano-vLLM 完整推理系统!")
    print("="*60)
    
    # 检查环境
    print("🔍 环境检查:")
    print(f"   Python版本: {torch.__version__}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    print("\n🎯 学习要点总结:")
    print("1. 端到端推理流程的完整实现")
    print("2. 多组件协调与集成架构")
    print("3. 性能监控与优化策略")
    print("4. 并发处理与资源管理")
    print("5. 生产环境部署考虑")
    
    # 运行测试
    print("\n🧪 开始测试演示...")
    
    try:
        # 测试1: 单个请求
        asyncio.run(test_single_request())
        
        # 测试2: 并发请求
        asyncio.run(test_concurrent_requests())
        
        # 测试3: 流式生成
        asyncio.run(test_streaming_generation())
        
        # 测试4: 性能基准
        test_performance_benchmark()
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎓 学习总结:")
    print("✅ 完成了完整LLM推理系统的学习")
    print("✅ 理解了端到端推理流程")
    print("✅ 掌握了性能优化技巧")
    print("✅ 学会了系统监控和调试")
    print("\n🚀 恭喜！你已经掌握了构建高性能LLM推理系统的核心技能！")

if __name__ == "__main__":
    main()