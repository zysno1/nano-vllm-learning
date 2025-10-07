#!/usr/bin/env python3
"""
第二步：LLM 引擎初始化与架构理解

这个脚本演示了如何构建和初始化一个简化版的 LLM 引擎，
展示各个组件（Scheduler、CacheEngine、Worker）如何协同工作。

学习目标：
1. 理解 LLM 引擎的整体架构
2. 掌握各组件的初始化流程
3. 学会配置和优化引擎参数
4. 理解组件间的协调机制
"""

import os
import sys
import time
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import psutil
import gc
from enum import Enum

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"


@dataclass
class EngineArgs:
    """引擎配置参数"""
    model: str = "microsoft/DialoGPT-small"
    max_model_len: int = 2048
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    block_size: int = 16
    swap_space: int = 4  # GB
    device: str = "auto"
    dtype: str = "float16"
    seed: int = 0
    
    def __post_init__(self):
        """配置后处理和验证"""
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的合理性"""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1.0:
            raise ValueError(f"GPU 内存利用率必须在 (0, 1] 范围内，当前值：{self.gpu_memory_utilization}")
        
        if self.max_model_len <= 0:
            raise ValueError(f"最大模型长度必须大于 0，当前值：{self.max_model_len}")
        
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"最大批次 token 数必须大于 0，当前值：{self.max_num_batched_tokens}")
        
        if self.block_size <= 0 or self.block_size > 128:
            raise ValueError(f"Block 大小必须在 (0, 128] 范围内，当前值：{self.block_size}")


@dataclass
class ModelConfig:
    """模型配置"""
    model: str
    max_model_len: int
    dtype: torch.dtype
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_attention_heads: int = 0
    
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """从引擎参数创建模型配置"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        return cls(
            model=engine_args.model,
            max_model_len=engine_args.max_model_len,
            dtype=dtype_map.get(engine_args.dtype, torch.float16)
        )


@dataclass
class CacheConfig:
    """缓存配置"""
    block_size: int
    gpu_memory_utilization: float
    swap_space: int
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0
    
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """从引擎参数创建缓存配置"""
        return cls(
            block_size=engine_args.block_size,
            gpu_memory_utilization=engine_args.gpu_memory_utilization,
            swap_space=engine_args.swap_space
        )
    
    def get_block_size_bytes(self, model_config: ModelConfig) -> int:
        """计算每个块的字节大小"""
        # 简化计算：每个 token 的 KV cache 大小
        # 实际计算需要考虑 hidden_size, num_layers, num_heads 等
        bytes_per_token = model_config.hidden_size * 2 * 2  # K + V, float16
        return self.block_size * bytes_per_token


@dataclass
class SchedulerConfig:
    """调度器配置"""
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """从引擎参数创建调度器配置"""
        return cls(
            max_num_batched_tokens=engine_args.max_num_batched_tokens,
            max_num_seqs=engine_args.max_num_seqs,
            max_model_len=engine_args.max_model_len
        )


class Sequence:
    """序列类，表示一个生成请求"""
    
    def __init__(self, seq_id: str, prompt: str, seq_len: int):
        self.seq_id = seq_id
        self.prompt = prompt
        self.seq_len = seq_len
        self.status = SequenceStatus.WAITING
        self.block_table: List[int] = []  # 分配的内存块列表
        self.logical_token_blocks = 0
        
    def get_len(self) -> int:
        """获取序列长度"""
        return self.seq_len
    
    def is_finished(self) -> bool:
        """检查序列是否已完成"""
        return self.status in [
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED
        ]


class SequenceGroup:
    """序列组，包含一个或多个相关序列"""
    
    def __init__(self, request_id: str, seqs: List[Sequence]):
        self.request_id = request_id
        self.seqs = seqs
        self.arrival_time = time.time()
    
    def get_seqs(self) -> List[Sequence]:
        """获取所有序列"""
        return self.seqs
    
    def get_max_num_running_seqs(self) -> int:
        """获取最大运行序列数"""
        return len([seq for seq in self.seqs if seq.status == SequenceStatus.RUNNING])


@dataclass
class SchedulerOutputs:
    """调度器输出"""
    scheduled_seq_groups: List[SequenceGroup]
    prompt_run: bool
    num_batched_tokens: int
    blocks_to_swap_in: Dict[int, int]
    blocks_to_swap_out: Dict[int, int]
    blocks_to_copy: Dict[int, List[int]]
    ignored_seq_groups: List[SequenceGroup]


class BlockAllocator:
    """内存块分配器"""
    
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.allocated_blocks = set()
    
    def allocate(self, num_blocks: int) -> List[int]:
        """分配指定数量的内存块"""
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError(f"内存不足：需要 {num_blocks} 块，可用 {len(self.free_blocks)} 块")
        
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop(0)
            self.allocated_blocks.add(block_id)
            allocated.append(block_id)
        
        return allocated
    
    def free(self, block_ids: List[int]):
        """释放内存块"""
        for block_id in block_ids:
            if block_id in self.allocated_blocks:
                self.allocated_blocks.remove(block_id)
                self.free_blocks.append(block_id)
    
    def get_num_free_blocks(self) -> int:
        """获取空闲块数量"""
        return len(self.free_blocks)


class CacheEngine:
    """缓存引擎，管理 KV Cache 的内存分配"""
    
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cache_config = cache_config
        self.model_config = model_config
        
        # 初始化内存块分配器
        self.gpu_allocator = BlockAllocator(cache_config.num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(cache_config.num_cpu_blocks)
        
        print(f"🧠 缓存引擎初始化完成：")
        print(f"   GPU 块数：{cache_config.num_gpu_blocks}")
        print(f"   CPU 块数：{cache_config.num_cpu_blocks}")
        print(f"   块大小：{cache_config.block_size} tokens")
    
    def allocate(self, seq_group: SequenceGroup) -> None:
        """为序列组分配内存块"""
        for seq in seq_group.get_seqs():
            # 计算需要的块数量
            num_blocks = (seq.get_len() + self.cache_config.block_size - 1) // self.cache_config.block_size
            
            # 分配 GPU 内存块
            try:
                block_table = self.gpu_allocator.allocate(num_blocks)
                seq.block_table = block_table
                seq.logical_token_blocks = num_blocks
                
                print(f"   ✅ 为序列 {seq.seq_id} 分配了 {num_blocks} 个 GPU 块：{block_table}")
                
            except RuntimeError as e:
                print(f"   ❌ GPU 内存分配失败：{e}")
                # 可以尝试 CPU 分配或交换策略
                raise
    
    def free(self, seq_group: SequenceGroup) -> None:
        """释放序列组的内存块"""
        for seq in seq_group.get_seqs():
            if seq.block_table:
                self.gpu_allocator.free(seq.block_table)
                print(f"   🗑️ 释放序列 {seq.seq_id} 的 {len(seq.block_table)} 个块")
                seq.block_table = []
    
    def swap_in(self, blocks_to_swap_in: Dict[int, int]) -> None:
        """将内存块从 CPU 交换到 GPU"""
        if blocks_to_swap_in:
            print(f"🔄 交换入 GPU：{len(blocks_to_swap_in)} 个块")
    
    def swap_out(self, blocks_to_swap_out: Dict[int, int]) -> None:
        """将内存块从 GPU 交换到 CPU"""
        if blocks_to_swap_out:
            print(f"🔄 交换出 GPU：{len(blocks_to_swap_out)} 个块")
    
    def get_num_free_gpu_blocks(self) -> int:
        """获取空闲 GPU 块数量"""
        return self.gpu_allocator.get_num_free_blocks()
    
    def get_num_free_cpu_blocks(self) -> int:
        """获取空闲 CPU 块数量"""
        return self.cpu_allocator.get_num_free_blocks()


class Scheduler:
    """调度器，负责请求调度和批次组织"""
    
    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig):
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        
        # 请求队列
        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []
        
        print(f"📋 调度器初始化完成：")
        print(f"   最大批次 tokens：{scheduler_config.max_num_batched_tokens}")
        print(f"   最大序列数：{scheduler_config.max_num_seqs}")
    
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """添加新的序列组到等待队列"""
        self.waiting.append(seq_group)
        print(f"📥 添加序列组 {seq_group.request_id} 到等待队列")
    
    def schedule(self) -> SchedulerOutputs:
        """执行调度，决定这一步处理哪些序列"""
        print(f"\n🎯 开始调度：等待 {len(self.waiting)}, 运行 {len(self.running)}, 交换 {len(self.swapped)}")
        
        # 1. 处理完成的序列
        self._process_finished_seqs()
        
        # 2. 调度等待中的序列
        scheduled_seq_groups = []
        num_batched_tokens = 0
        
        # 优先处理正在运行的序列
        for seq_group in self.running[:]:
            if self._can_schedule_seq_group(seq_group, num_batched_tokens):
                scheduled_seq_groups.append(seq_group)
                num_batched_tokens += self._get_seq_group_tokens(seq_group)
            else:
                break
        
        # 尝试调度等待中的序列
        for seq_group in self.waiting[:]:
            if (len(scheduled_seq_groups) < self.scheduler_config.max_num_seqs and
                self._can_schedule_seq_group(seq_group, num_batched_tokens)):
                
                scheduled_seq_groups.append(seq_group)
                num_batched_tokens += self._get_seq_group_tokens(seq_group)
                
                # 从等待队列移到运行队列
                self.waiting.remove(seq_group)
                self.running.append(seq_group)
                
                # 更新序列状态
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.RUNNING
                
                print(f"   ✅ 调度序列组 {seq_group.request_id}")
            else:
                break
        
        print(f"   📊 本轮调度：{len(scheduled_seq_groups)} 个序列组，{num_batched_tokens} 个 tokens")
        
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            prompt_run=True,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            ignored_seq_groups=[]
        )
    
    def _process_finished_seqs(self):
        """处理已完成的序列"""
        finished_seq_groups = []
        for seq_group in self.running[:]:
            if all(seq.is_finished() for seq in seq_group.get_seqs()):
                finished_seq_groups.append(seq_group)
                self.running.remove(seq_group)
        
        if finished_seq_groups:
            print(f"   ✅ 完成 {len(finished_seq_groups)} 个序列组")
    
    def _can_schedule_seq_group(self, seq_group: SequenceGroup, current_tokens: int) -> bool:
        """检查是否可以调度序列组"""
        seq_group_tokens = self._get_seq_group_tokens(seq_group)
        return current_tokens + seq_group_tokens <= self.scheduler_config.max_num_batched_tokens
    
    def _get_seq_group_tokens(self, seq_group: SequenceGroup) -> int:
        """获取序列组的 token 数量"""
        return sum(seq.get_len() for seq in seq_group.get_seqs())
    
    def get_num_unfinished_seq_groups(self) -> int:
        """获取未完成的序列组数量"""
        return len(self.waiting) + len(self.running) + len(self.swapped)


class ModelExecutor:
    """模型执行器，负责实际的推理计算"""
    
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = self._determine_device()
        
        print(f"🚀 模型执行器初始化完成：")
        print(f"   设备：{self.device}")
        print(f"   数据类型：{model_config.dtype}")
    
    def _determine_device(self) -> str:
        """确定执行设备"""
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def execute_model(self, 
                     seq_groups: List[SequenceGroup],
                     blocks_to_swap_in: Dict[int, int],
                     blocks_to_swap_out: Dict[int, int],
                     blocks_to_copy: Dict[int, List[int]]) -> Dict[str, Any]:
        """执行模型推理"""
        if not seq_groups:
            return {}
        
        print(f"🔥 执行模型推理：{len(seq_groups)} 个序列组")
        
        # 模拟推理过程
        time.sleep(0.1)  # 模拟计算时间
        
        # 模拟输出
        outputs = {}
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs():
                # 模拟生成一个新 token
                seq.seq_len += 1
                
                # 检查是否达到最大长度
                if seq.seq_len >= self.model_config.max_model_len:
                    seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
                
                outputs[seq.seq_id] = {
                    "token_id": 1,  # 模拟生成的 token
                    "logprob": -0.1,
                    "finished": seq.is_finished()
                }
        
        return outputs
    
    def get_memory_usage(self) -> int:
        """获取模型内存使用量（字节）"""
        # 简化计算，实际需要根据模型参数计算
        return 1024 * 1024 * 1024  # 1GB


class LLMEngine:
    """LLM 引擎，协调所有组件的工作"""
    
    def __init__(self, engine_args: EngineArgs):
        print("🏗️ 初始化 LLM 引擎...")
        
        # 1. 创建配置对象
        self.model_config = ModelConfig.from_engine_args(engine_args)
        self.cache_config = CacheConfig.from_engine_args(engine_args)
        self.scheduler_config = SchedulerConfig.from_engine_args(engine_args)
        
        print(f"📋 配置创建完成：")
        print(f"   模型：{self.model_config.model}")
        print(f"   最大长度：{self.model_config.max_model_len}")
        print(f"   数据类型：{self.model_config.dtype}")
        
        # 2. 初始化模型执行器
        self.model_executor = ModelExecutor(self.model_config, self.cache_config)
        
        # 3. 确定缓存块数量
        self._determine_num_blocks()
        
        # 4. 初始化缓存引擎
        self.cache_engine = CacheEngine(self.cache_config, self.model_config)
        
        # 5. 初始化调度器
        self.scheduler = Scheduler(self.scheduler_config, self.cache_config)
        
        # 统计信息
        self.request_counter = 0
        
        print("✅ LLM 引擎初始化完成！\n")
    
    def _determine_num_blocks(self):
        """计算可用的内存块数量"""
        print("🧮 计算内存块数量...")
        
        # 获取可用内存
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_gpu_memory * self.cache_config.gpu_memory_utilization
        else:
            # CPU 内存
            total_memory = psutil.virtual_memory().total
            available_memory = total_memory * 0.5  # 使用 50% CPU 内存
        
        # 减去模型占用的内存
        model_memory = self.model_executor.get_memory_usage()
        cache_memory = available_memory - model_memory
        
        # 计算块大小（简化计算）
        # 实际需要根据模型的 hidden_size, num_layers 等参数精确计算
        bytes_per_block = self.cache_config.block_size * 1024  # 简化为每块 1KB
        
        # 计算块数量
        num_gpu_blocks = max(0, int(cache_memory // bytes_per_block))
        num_cpu_blocks = int(self.cache_config.swap_space * 1024**3 // bytes_per_block)
        
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        print(f"   总内存：{available_memory / 1024**3:.2f} GB")
        print(f"   模型内存：{model_memory / 1024**3:.2f} GB")
        print(f"   缓存内存：{cache_memory / 1024**3:.2f} GB")
        print(f"   GPU 块数：{num_gpu_blocks}")
        print(f"   CPU 块数：{num_cpu_blocks}")
        print(f"   块大小：{bytes_per_block} bytes")
    
    def add_request(self, 
                   request_id: str,
                   prompt: str,
                   max_tokens: int = 100) -> None:
        """添加新的生成请求"""
        # 创建序列
        seq_id = f"{request_id}_seq_0"
        seq = Sequence(seq_id, prompt, len(prompt.split()))  # 简化的 token 计算
        
        # 创建序列组
        seq_group = SequenceGroup(request_id, [seq])
        
        # 分配内存
        try:
            self.cache_engine.allocate(seq_group)
        except RuntimeError as e:
            print(f"❌ 请求 {request_id} 内存分配失败：{e}")
            return
        
        # 添加到调度器
        self.scheduler.add_seq_group(seq_group)
        
        print(f"📝 添加请求：{request_id}")
        print(f"   提示词：{prompt}")
        print(f"   序列长度：{seq.get_len()} tokens")
    
    def step(self) -> List[Dict[str, Any]]:
        """执行一步推理"""
        # 1. 调度
        scheduler_outputs = self.scheduler.schedule()
        
        if not scheduler_outputs.scheduled_seq_groups:
            return []
        
        # 2. 缓存操作
        self.cache_engine.swap_in(scheduler_outputs.blocks_to_swap_in)
        self.cache_engine.swap_out(scheduler_outputs.blocks_to_swap_out)
        
        # 3. 执行推理
        model_outputs = self.model_executor.execute_model(
            scheduler_outputs.scheduled_seq_groups,
            scheduler_outputs.blocks_to_swap_in,
            scheduler_outputs.blocks_to_swap_out,
            scheduler_outputs.blocks_to_copy
        )
        
        # 4. 处理输出
        request_outputs = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            for seq in seq_group.get_seqs():
                if seq.seq_id in model_outputs:
                    output = model_outputs[seq.seq_id]
                    request_outputs.append({
                        "request_id": seq_group.request_id,
                        "seq_id": seq.seq_id,
                        "token_id": output["token_id"],
                        "finished": output["finished"]
                    })
                    
                    # 如果序列完成，释放内存
                    if output["finished"]:
                        self.cache_engine.free(seq_group)
        
        return request_outputs
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            "num_waiting": len(self.scheduler.waiting),
            "num_running": len(self.scheduler.running),
            "num_swapped": len(self.scheduler.swapped),
            "num_free_gpu_blocks": self.cache_engine.get_num_free_gpu_blocks(),
            "num_free_cpu_blocks": self.cache_engine.get_num_free_cpu_blocks(),
        }


def test_engine_initialization():
    """测试引擎初始化"""
    print("🧪 测试引擎初始化...\n")
    
    try:
        # 创建引擎配置
        engine_args = EngineArgs(
            model="microsoft/DialoGPT-small",
            max_model_len=512,
            max_num_batched_tokens=1024,
            max_num_seqs=16,
            gpu_memory_utilization=0.8,
            block_size=16
        )
        
        # 初始化引擎
        engine = LLMEngine(engine_args)
        
        # 打印初始状态
        stats = engine.get_stats()
        print("📊 引擎初始状态：")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ 引擎初始化测试成功！")
        return engine
        
    except Exception as e:
        print(f"❌ 引擎初始化失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_request_processing(engine: LLMEngine):
    """测试请求处理"""
    print("\n🧪 测试请求处理...\n")
    
    # 添加测试请求
    test_requests = [
        ("req_1", "Hello, how are you?"),
        ("req_2", "What is the weather like?"),
        ("req_3", "Tell me a joke"),
    ]
    
    for request_id, prompt in test_requests:
        engine.add_request(request_id, prompt)
    
    # 执行多步推理
    max_steps = 10
    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        # 执行一步
        outputs = engine.step()
        
        # 打印输出
        if outputs:
            for output in outputs:
                print(f"🔤 {output['request_id']}: token_{output['token_id']} "
                      f"({'完成' if output['finished'] else '继续'})")
        
        # 打印统计信息
        stats = engine.get_stats()
        print(f"📊 状态：等待 {stats['num_waiting']}, 运行 {stats['num_running']}, "
              f"GPU块 {stats['num_free_gpu_blocks']}")
        
        # 检查是否所有请求都完成
        if stats['num_waiting'] == 0 and stats['num_running'] == 0:
            print("🎉 所有请求处理完成！")
            break
        
        time.sleep(0.5)  # 模拟处理间隔


def test_memory_management():
    """测试内存管理"""
    print("\n🧪 测试内存管理...\n")
    
    # 创建小内存配置
    engine_args = EngineArgs(
        model="microsoft/DialoGPT-small",
        max_model_len=128,
        max_num_batched_tokens=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,  # 较小的内存利用率
        block_size=8
    )
    
    try:
        engine = LLMEngine(engine_args)
        
        # 添加多个请求，测试内存分配
        for i in range(6):  # 超过最大序列数
            engine.add_request(f"mem_test_{i}", f"This is test request number {i}")
        
        # 执行几步，观察内存管理
        for step in range(5):
            print(f"\n--- 内存测试步骤 {step + 1} ---")
            outputs = engine.step()
            stats = engine.get_stats()
            
            print(f"📊 内存状态：")
            print(f"   等待队列：{stats['num_waiting']}")
            print(f"   运行队列：{stats['num_running']}")
            print(f"   空闲 GPU 块：{stats['num_free_gpu_blocks']}")
            
            time.sleep(0.3)
        
        print("✅ 内存管理测试完成！")
        
    except Exception as e:
        print(f"❌ 内存管理测试失败：{str(e)}")


def demonstrate_config_impact():
    """演示不同配置对性能的影响"""
    print("\n🧪 演示配置参数影响...\n")
    
    configs = [
        {
            "name": "小批次配置",
            "max_num_batched_tokens": 512,
            "max_num_seqs": 4,
            "block_size": 8
        },
        {
            "name": "大批次配置", 
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 16,
            "block_size": 32
        }
    ]
    
    for config in configs:
        print(f"📊 测试配置：{config['name']}")
        
        try:
            engine_args = EngineArgs(
                model="microsoft/DialoGPT-small",
                max_model_len=256,
                max_num_batched_tokens=config["max_num_batched_tokens"],
                max_num_seqs=config["max_num_seqs"],
                gpu_memory_utilization=0.7,
                block_size=config["block_size"]
            )
            
            start_time = time.time()
            engine = LLMEngine(engine_args)
            init_time = time.time() - start_time
            
            print(f"   初始化时间：{init_time:.3f} 秒")
            
            stats = engine.get_stats()
            print(f"   可用 GPU 块：{stats['num_free_gpu_blocks']}")
            print(f"   可用 CPU 块：{stats['num_free_cpu_blocks']}")
            
        except Exception as e:
            print(f"   ❌ 配置测试失败：{str(e)}")
        
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 第二步：LLM 引擎初始化与架构理解")
    print("=" * 60)
    
    # 环境检查
    print(f"🔍 环境检查：")
    print(f"   Python 版本：{sys.version}")
    print(f"   PyTorch 版本：{torch.__version__}")
    print(f"   CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU 数量：{torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}：{torch.cuda.get_device_name(i)}")
    print()
    
    try:
        # 1. 引擎初始化测试
        engine = test_engine_initialization()
        
        if engine:
            # 2. 请求处理测试
            test_request_processing(engine)
            
            # 3. 内存管理测试
            test_memory_management()
            
            # 4. 配置影响演示
            demonstrate_config_impact()
        
        print("\n🎉 所有测试完成！")
        print("\n📚 学习要点总结：")
        print("   1. LLM 引擎协调 Scheduler、CacheEngine、Worker 三大组件")
        print("   2. 内存块管理是高效推理的关键")
        print("   3. 调度策略直接影响系统吞吐量")
        print("   4. 配置参数需要根据硬件资源合理设置")
        print("   5. 组件间的协调机制确保系统稳定运行")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序")
    except Exception as e:
        print(f"\n💥 程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 资源清理完成")