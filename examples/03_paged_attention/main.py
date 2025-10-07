#!/usr/bin/env python3
"""
第三步：PagedAttention 机制与内存优化

这个脚本演示了 PagedAttention 的核心实现，包括：
- Block Table 的设计与管理
- 内存分配与回收策略
- 分页注意力计算
- Copy-on-Write 优化

学习目标：
1. 理解 PagedAttention 如何解决内存碎片问题
2. 掌握 Block Table 的映射机制
3. 学会高效的内存管理策略
4. 理解分块存储下的注意力计算
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import gc
from enum import Enum

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class BlockStatus(Enum):
    """内存块状态"""
    FREE = "free"
    ALLOCATED = "allocated"
    SWAPPED = "swapped"


@dataclass
class Block:
    """内存块类"""
    block_id: int
    ref_count: int = 0
    status: BlockStatus = BlockStatus.FREE
    last_accessed: float = 0.0
    
    # KV Cache 数据 [block_size, num_heads, head_dim]
    key_data: Optional[torch.Tensor] = None
    value_data: Optional[torch.Tensor] = None
    
    def is_free(self) -> bool:
        return self.status == BlockStatus.FREE and self.ref_count == 0
    
    def allocate(self):
        """分配内存块"""
        if self.status == BlockStatus.FREE:
            self.status = BlockStatus.ALLOCATED
            self.ref_count = 1
            self.last_accessed = time.time()
            print(f"    🔵 Block {self.block_id} 已分配 (ref_count: {self.ref_count})")
        else:
            print(f"    ⚠️  Block {self.block_id} 已被占用，状态: {self.status.value}")
    
    def add_ref(self):
        """增加引用计数"""
        self.ref_count += 1
        self.last_accessed = time.time()
        print(f"    📈 Block {self.block_id} 引用计数增加到 {self.ref_count}")
    
    def remove_ref(self):
        """减少引用计数"""
        if self.ref_count > 0:
            self.ref_count -= 1
            print(f"    📉 Block {self.block_id} 引用计数减少到 {self.ref_count}")
            if self.ref_count == 0:
                self.status = BlockStatus.FREE
                print(f"    🟢 Block {self.block_id} 已释放 (ref_count: 0)")
        else:
            print(f"    ⚠️  Block {self.block_id} 引用计数已为 0")
    
    def add_ref(self):
        """增加引用计数"""
        self.ref_count += 1
        self.last_accessed = time.time()
    
    def remove_ref(self):
        """减少引用计数"""
        self.ref_count = max(0, self.ref_count - 1)
        if self.ref_count == 0:
            self.status = BlockStatus.FREE


class BlockTable:
    """Block Table：逻辑地址到物理地址的映射表"""
    
    def __init__(self, seq_id: str, block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.blocks: List[int] = []  # 物理块ID列表
        print(f"  📋 为序列 {seq_id} 创建 Block Table (block_size: {block_size})")
    
    def add_block(self, physical_block_id: int):
        """添加物理块"""
        self.blocks.append(physical_block_id)
        print(f"    ➕ Block Table[{self.seq_id}] 添加物理块 {physical_block_id} (总块数: {len(self.blocks)})")
    
    def get_physical_block_id(self, logical_token_idx: int) -> int:
        """根据逻辑token索引获取物理块ID"""
        block_idx = logical_token_idx // self.block_size
        if block_idx >= len(self.blocks):
            raise IndexError(f"逻辑索引 {logical_token_idx} 超出范围")
        physical_id = self.blocks[block_idx]
        print(f"    🔍 映射: token[{logical_token_idx}] -> block[{block_idx}] -> 物理块 {physical_id}")
        return physical_id
    
    def get_block_offset(self, logical_token_idx: int) -> int:
        """获取在块内的偏移量"""
        offset = logical_token_idx % self.block_size
        print(f"    📍 token[{logical_token_idx}] 在块内偏移: {offset}")
        return offset
    
    def get_num_blocks(self) -> int:
        return len(self.blocks)
    
    def get_all_blocks(self) -> List[int]:
        return self.blocks.copy()


class BlockAllocator:
    """内存块分配器"""
    
    def __init__(self, num_blocks: int, block_size: int, num_heads: int, head_dim: int, device: str = "cuda"):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # 初始化所有块
        self.blocks: List[Block] = []
        self.free_blocks: List[int] = []
        
        self._initialize_blocks()
        
        print(f"🧱 块分配器初始化完成：")
        print(f"   总块数：{num_blocks}")
        print(f"   块大小：{block_size} tokens")
        print(f"   注意力头数：{num_heads}")
        print(f"   头维度：{head_dim}")
        print(f"   设备：{device}")
    
    def _initialize_blocks(self):
        """初始化所有内存块"""
        for i in range(self.num_blocks):
            block = Block(block_id=i)
            
            # 预分配 KV Cache 存储空间
            if self.device == "cuda" and torch.cuda.is_available():
                block.key_data = torch.zeros(
                    (self.block_size, self.num_heads, self.head_dim),
                    dtype=torch.float16, device="cuda"
                )
                block.value_data = torch.zeros(
                    (self.block_size, self.num_heads, self.head_dim),
                    dtype=torch.float16, device="cuda"
                )
            else:
                block.key_data = torch.zeros(
                    (self.block_size, self.num_heads, self.head_dim),
                    dtype=torch.float16, device="cpu"
                )
                block.value_data = torch.zeros(
                    (self.block_size, self.num_heads, self.head_dim),
                    dtype=torch.float16, device="cpu"
                )
            
            self.blocks.append(block)
            self.free_blocks.append(i)
    
    def allocate(self, num_blocks: int) -> List[int]:
        """分配指定数量的内存块"""
        print(f"  🔄 请求分配 {num_blocks} 个内存块...")
        
        if num_blocks > self.get_free_blocks_count():
            print(f"    ❌ 内存不足！需要 {num_blocks} 块，可用 {self.get_free_blocks_count()} 块")
            # 尝试垃圾回收
            self._garbage_collect()
            if num_blocks > self.get_free_blocks_count():
                raise RuntimeError(f"内存不足：需要 {num_blocks} 块，可用 {self.get_free_blocks_count()} 块")
        
        allocated_blocks = []
        for block_id in self.free_blocks:
            if len(allocated_blocks) >= num_blocks:
                break
            
            block = self.blocks[block_id]
            if block.is_free():
                block.allocate()
                allocated_blocks.append(block_id)
                self.allocated_blocks.add(block_id)
        
        # 从空闲列表中移除已分配的块
        for block_id in allocated_blocks:
            self.free_blocks.remove(block_id)
        
        print(f"    ✅ 成功分配块: {allocated_blocks}")
        print(f"    📊 内存状态: {self.get_free_blocks_count()} 空闲 / {len(self.allocated_blocks)} 已分配")
        return allocated_blocks
    
    def free(self, block_ids: List[int]):
        """释放指定的内存块"""
        print(f"  🗑️  释放内存块: {block_ids}")
        
        for block_id in block_ids:
            if block_id in self.allocated_blocks:
                block = self.blocks[block_id]
                block.remove_ref()
                
                if block.ref_count == 0:
                    self.allocated_blocks.remove(block_id)
                    self.free_blocks.add(block_id)
                    
                    # 清理 KV Cache 数据
                    if block.key_data is not None:
                        del block.key_data
                        block.key_data = None
                    if block.value_data is not None:
                        del block.value_data
                        block.value_data = None
                    
                    print(f"    🟢 Block {block_id} 已完全释放")
                else:
                    print(f"    🔄 Block {block_id} 仍有引用 (ref_count: {block.ref_count})")
            else:
                print(f"    ⚠️  Block {block_id} 未在已分配列表中")
        
        print(f"    📊 释放后内存状态: {self.get_free_blocks_count()} 空闲 / {len(self.allocated_blocks)} 已分配")
    
    def add_ref(self, block_id: int):
        """增加块的引用计数"""
        if 0 <= block_id < len(self.blocks):
            self.blocks[block_id].add_ref()
    
    def get_block(self, block_id: int) -> Block:
        """获取指定的块"""
        if 0 <= block_id < len(self.blocks):
            return self.blocks[block_id]
        raise IndexError(f"块 ID {block_id} 超出范围")
    
    def get_free_blocks_count(self) -> int:
        """获取空闲块数量"""
        return len(self.free_blocks)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用统计"""
        allocated = sum(1 for block in self.blocks if block.status == BlockStatus.ALLOCATED)
        free = len(self.free_blocks)
        
        return {
            "total": self.num_blocks,
            "allocated": allocated,
            "free": free,
            "utilization": allocated / self.num_blocks * 100
        }
    
    def _garbage_collect(self):
        """垃圾回收：释放引用计数为 0 的块"""
        collected = 0
        for block in self.blocks:
            if block.ref_count == 0 and block.status == BlockStatus.ALLOCATED:
                block.status = BlockStatus.FREE
                self.free_blocks.append(block.block_id)
                collected += 1
        
        if collected > 0:
            print(f"   🧹 垃圾回收：释放 {collected} 个块")


class PagedAttentionEngine:
    """PagedAttention 引擎"""
    
    def __init__(self, 
                 num_blocks: int,
                 block_size: int,
                 num_heads: int,
                 head_dim: int,
                 device: str = "cuda"):
        
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # 初始化块分配器
        self.block_allocator = BlockAllocator(
            num_blocks, block_size, num_heads, head_dim, device
        )
        
        # 序列管理
        self.sequence_block_tables: Dict[str, BlockTable] = {}
        
        print(f"🚀 PagedAttention 引擎初始化完成")
    
    def allocate_sequence(self, seq_id: str, seq_len: int) -> BlockTable:
        """为序列分配内存块"""
        print(f"\n🎯 为序列 '{seq_id}' 分配内存 (长度: {seq_len})")
        
        # 计算需要的块数
        num_blocks_needed = (seq_len + self.block_size - 1) // self.block_size
        print(f"  📐 计算所需块数: ceil({seq_len} / {self.block_size}) = {num_blocks_needed}")
        
        # 分配物理块
        try:
            physical_blocks = self.block_allocator.allocate(num_blocks_needed)
        except RuntimeError as e:
            print(f"  ❌ 分配失败: {e}")
            raise
        
        # 创建 Block Table
        block_table = BlockTable(seq_id, self.block_size)
        for block_id in physical_blocks:
            block_table.add_block(block_id)
        
        # 存储映射关系
        self.block_tables[seq_id] = block_table
        print(f"  ✅ 序列 '{seq_id}' 分配完成，使用块: {physical_blocks}")
        
        return block_table
    
    def free_sequence(self, seq_id: str):
        """释放序列占用的内存"""
        print(f"\n🗑️  释放序列 '{seq_id}' 的内存")
        
        if seq_id not in self.block_tables:
            print(f"  ⚠️  序列 '{seq_id}' 不存在")
            return
        
        block_table = self.block_tables[seq_id]
        physical_blocks = block_table.get_all_blocks()
        
        # 释放物理块
        self.block_allocator.free(physical_blocks)
        
        # 移除映射关系
        del self.block_tables[seq_id]
        print(f"  ✅ 序列 '{seq_id}' 内存释放完成")
    
    def write_kv_cache(self, 
                      seq_id: str,
                      token_positions: List[int],
                      keys: torch.Tensor,
                      values: torch.Tensor):
        """写入 KV Cache"""
        print(f"\n📝 写入 KV Cache: 序列 '{seq_id}', 位置 {token_positions}")
        
        if seq_id not in self.block_tables:
            raise ValueError(f"序列 {seq_id} 未分配内存")
        
        block_table = self.block_tables[seq_id]
        
        for i, token_pos in enumerate(token_positions):
            # 获取物理块和偏移
            physical_block_id = block_table.get_physical_block_id(token_pos)
            block_offset = block_table.get_block_offset(token_pos)
            
            # 获取物理块
            block = self.block_allocator.get_block(physical_block_id)
            
            # 写入数据
            if block.key_data is None:
                print(f"    🔧 初始化 Block {physical_block_id} 的 KV Cache")
                block.key_data = torch.zeros(self.block_size, self.num_heads, self.head_dim, 
                                           device=self.device, dtype=keys.dtype)
                block.value_data = torch.zeros(self.block_size, self.num_heads, self.head_dim, 
                                             device=self.device, dtype=values.dtype)
            
            block.key_data[block_offset] = keys[i]
            block.value_data[block_offset] = values[i]
            print(f"    ✏️  写入 token[{token_pos}] -> Block[{physical_block_id}][{block_offset}]")
    
    def read_kv_cache(self, seq_id: str, context_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """读取 KV Cache 数据"""
        if seq_id not in self.sequence_block_tables:
            raise ValueError(f"序列 {seq_id} 未分配内存")
        
        block_table = self.sequence_block_tables[seq_id]
        
        print(f"📖 读取序列 {seq_id} 的 KV Cache (长度: {context_len})")
        
        keys = []
        values = []
        
        # 按块读取数据
        num_blocks = (context_len + self.block_size - 1) // self.block_size
        
        for block_idx in range(num_blocks):
            physical_block_id = block_table.logical_blocks[block_idx]
            block = self.block_allocator.get_block(physical_block_id)
            
            # 计算这个块中有效的 token 数量
            start_token = block_idx * self.block_size
            end_token = min(start_token + self.block_size, context_len)
            valid_tokens = end_token - start_token
            
            # 读取有效数据
            block_keys = block.key_data[:valid_tokens]
            block_values = block.value_data[:valid_tokens]
            
            keys.append(block_keys)
            values.append(block_values)
            
            print(f"   块 {physical_block_id}: {valid_tokens} 个有效 tokens")
        
        # 拼接所有块的数据
        full_keys = torch.cat(keys, dim=0)  # [context_len, num_heads, head_dim]
        full_values = torch.cat(values, dim=0)
        
        return full_keys, full_values
    
    def compute_attention(self,
                         query: torch.Tensor,  # [1, num_heads, head_dim]
                         seq_id: str,
                         context_len: int) -> torch.Tensor:
        """计算 PagedAttention"""
        
        print(f"\n🧠 计算序列 {seq_id} 的注意力 (上下文长度: {context_len})")
        
        # 1. 读取 KV Cache
        keys, values = self.read_kv_cache(seq_id, context_len)
        
        # 2. 重塑张量维度以适配注意力计算
        # query: [1, num_heads, head_dim]
        # keys: [context_len, num_heads, head_dim] -> [num_heads, context_len, head_dim]
        # values: [context_len, num_heads, head_dim] -> [num_heads, context_len, head_dim]
        
        query = query.unsqueeze(0)  # [1, 1, num_heads, head_dim]
        keys = keys.transpose(0, 1).unsqueeze(0)  # [1, num_heads, context_len, head_dim]
        values = values.transpose(0, 1).unsqueeze(0)  # [1, num_heads, context_len, head_dim]
        
        # 3. 计算注意力分数
        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Q * K^T
        attention_scores = torch.matmul(query, keys.transpose(-2, -1)) * scale
        # [1, num_heads, 1, context_len]
        
        # 4. 应用 softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 5. 加权求和
        attention_output = torch.matmul(attention_weights, values)
        # [1, num_heads, 1, head_dim]
        
        # 6. 重塑输出
        output = attention_output.squeeze(0).squeeze(1)  # [num_heads, head_dim]
        
        print(f"   ✅ 注意力计算完成，输出形状：{output.shape}")
        
        return output
    
    def fork_sequence(self, parent_seq_id: str, child_seq_id: str):
        """序列分叉（Copy-on-Write）"""
        if parent_seq_id not in self.sequence_block_tables:
            raise ValueError(f"父序列 {parent_seq_id} 不存在")
        
        print(f"\n🍴 序列分叉：{parent_seq_id} -> {child_seq_id}")
        
        parent_table = self.sequence_block_tables[parent_seq_id]
        
        # 创建子序列的 Block Table（共享物理块）
        child_table = BlockTable(child_seq_id, self.block_size)
        child_table.logical_blocks = parent_table.logical_blocks.copy()
        
        # 增加所有共享块的引用计数
        for block_id in child_table.logical_blocks:
            self.block_allocator.add_ref(block_id)
        
        self.sequence_block_tables[child_seq_id] = child_table
        
        print(f"   📎 共享 {len(child_table.logical_blocks)} 个块")
    
    def copy_on_write(self, seq_id: str, token_pos: int):
        """Copy-on-Write 处理"""
        if seq_id not in self.sequence_block_tables:
            raise ValueError(f"序列 {seq_id} 不存在")
        
        block_table = self.sequence_block_tables[seq_id]
        physical_block_id = block_table.get_physical_block_id(token_pos)
        block = self.block_allocator.get_block(physical_block_id)
        
        # 检查是否需要 COW
        if block.ref_count > 1:
            print(f"\n📝 COW 触发：序列 {seq_id}, token {token_pos}")
            
            # 分配新块
            new_block_ids = self.block_allocator.allocate(1)
            new_block_id = new_block_ids[0]
            new_block = self.block_allocator.get_block(new_block_id)
            
            # 复制数据
            new_block.key_data.copy_(block.key_data)
            new_block.value_data.copy_(block.value_data)
            
            # 更新 Block Table
            block_idx = token_pos // self.block_size
            block_table.logical_blocks[block_idx] = new_block_id
            
            # 更新引用计数
            block.remove_ref()
            
            print(f"   📋 复制块：{physical_block_id} -> {new_block_id}")
            print(f"   📊 原块引用计数：{block.ref_count}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        stats = self.block_allocator.get_memory_usage()
        stats["sequences"] = len(self.sequence_block_tables)
        stats["block_size"] = self.block_size
        
        return stats


def test_basic_allocation():
    """测试基本的内存分配功能"""
    print("🧪 测试基本内存分配...\n")
    
    # 创建 PagedAttention 引擎
    engine = PagedAttentionEngine(
        num_blocks=64,
        block_size=16,
        num_heads=8,
        head_dim=64,
        device="cpu"  # 使用 CPU 以确保兼容性
    )
    
    # 测试序列分配
    sequences = [
        ("seq_1", 100),  # 需要 7 个块
        ("seq_2", 50),   # 需要 4 个块
        ("seq_3", 200),  # 需要 13 个块
    ]
    
    for seq_id, seq_len in sequences:
        try:
            block_table = engine.allocate_sequence(seq_id, seq_len)
            print(f"   ✅ 序列 {seq_id} 分配成功")
        except Exception as e:
            print(f"   ❌ 序列 {seq_id} 分配失败：{e}")
    
    # 打印内存统计
    stats = engine.get_memory_stats()
    print(f"\n📊 内存统计：")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return engine


def test_kv_cache_operations(engine: PagedAttentionEngine):
    """测试 KV Cache 读写操作"""
    print("\n🧪 测试 KV Cache 读写操作...\n")
    
    seq_id = "seq_1"
    context_len = 48  # 3 个块
    
    # 生成测试数据
    keys = torch.randn(context_len, engine.num_heads, engine.head_dim, dtype=torch.float16)
    values = torch.randn(context_len, engine.num_heads, engine.head_dim, dtype=torch.float16)
    
    # 写入 KV Cache
    token_positions = list(range(context_len))
    engine.write_kv_cache(seq_id, token_positions, keys, values)
    
    # 读取 KV Cache
    read_keys, read_values = engine.read_kv_cache(seq_id, context_len)
    
    # 验证数据一致性
    key_diff = torch.abs(keys - read_keys).max().item()
    value_diff = torch.abs(values - read_values).max().item()
    
    print(f"📊 数据一致性检查：")
    print(f"   Key 差异：{key_diff:.10f}")
    print(f"   Value 差异：{value_diff:.10f}")
    print(f"   ✅ 数据{'一致' if key_diff < 1e-6 and value_diff < 1e-6 else '不一致'}")


def test_attention_computation(engine: PagedAttentionEngine):
    """测试注意力计算"""
    print("\n🧪 测试注意力计算...\n")
    
    seq_id = "seq_1"
    context_len = 48
    
    # 生成查询向量
    query = torch.randn(1, engine.num_heads, engine.head_dim, dtype=torch.float16)
    
    # 计算注意力
    try:
        attention_output = engine.compute_attention(query, seq_id, context_len)
        print(f"   ✅ 注意力计算成功，输出形状：{attention_output.shape}")
        
        # 验证输出形状
        expected_shape = (engine.num_heads, engine.head_dim)
        if attention_output.shape == expected_shape:
            print(f"   ✅ 输出形状正确：{expected_shape}")
        else:
            print(f"   ❌ 输出形状错误：期望 {expected_shape}，实际 {attention_output.shape}")
            
    except Exception as e:
        print(f"   ❌ 注意力计算失败：{e}")


def test_copy_on_write(engine: PagedAttentionEngine):
    """测试 Copy-on-Write 功能"""
    print("\n🧪 测试 Copy-on-Write...\n")
    
    parent_seq = "seq_1"
    child_seq = "seq_1_fork"
    
    # 序列分叉
    try:
        engine.fork_sequence(parent_seq, child_seq)
        print(f"   ✅ 序列分叉成功")
        
        # 触发 COW
        engine.copy_on_write(child_seq, 10)  # 修改第 10 个 token
        print(f"   ✅ COW 处理成功")
        
        # 检查内存统计
        stats = engine.get_memory_stats()
        print(f"   📊 COW 后内存统计：已分配 {stats['allocated']} 块")
        
    except Exception as e:
        print(f"   ❌ COW 测试失败：{e}")


def test_memory_efficiency():
    """测试内存效率对比"""
    print("\n🧪 测试内存效率对比...\n")
    
    sequences = [
        ("seq_1", 100),
        ("seq_2", 50),
        ("seq_3", 200),
        ("seq_4", 75),
        ("seq_5", 150),
    ]
    
    block_size = 16
    max_seq_len = 512
    
    # 计算传统方式的内存使用
    traditional_memory = len(sequences) * max_seq_len
    
    # 计算 PagedAttention 的内存使用
    paged_memory = 0
    for seq_id, actual_len in sequences:
        num_blocks = (actual_len + block_size - 1) // block_size
        paged_memory += num_blocks * block_size
    
    # 计算节省的内存
    saved_memory = traditional_memory - paged_memory
    efficiency = saved_memory / traditional_memory * 100
    
    print(f"📊 内存效率对比：")
    print(f"   传统方式：{traditional_memory:,} tokens")
    print(f"   PagedAttention：{paged_memory:,} tokens")
    print(f"   节省内存：{saved_memory:,} tokens ({efficiency:.1f}%)")
    
    # 计算内存利用率
    actual_tokens = sum(seq_len for _, seq_len in sequences)
    traditional_utilization = actual_tokens / traditional_memory * 100
    paged_utilization = actual_tokens / paged_memory * 100
    
    print(f"   传统利用率：{traditional_utilization:.1f}%")
    print(f"   PagedAttention 利用率：{paged_utilization:.1f}%")


def test_block_fragmentation():
    """测试内存碎片处理"""
    print("\n🧪 测试内存碎片处理...\n")
    
    engine = PagedAttentionEngine(
        num_blocks=32,  # 较小的内存池
        block_size=16,
        num_heads=8,
        head_dim=64,
        device="cpu"
    )
    
    # 分配多个序列
    sequences = []
    for i in range(8):
        seq_id = f"frag_seq_{i}"
        seq_len = 30 + i * 10  # 不同长度的序列
        try:
            engine.allocate_sequence(seq_id, seq_len)
            sequences.append(seq_id)
            print(f"   ✅ 分配序列 {seq_id} ({seq_len} tokens)")
        except RuntimeError as e:
            print(f"   ❌ 分配序列 {seq_id} 失败：{e}")
            break
    
    # 释放部分序列（模拟碎片化）
    for i in range(0, len(sequences), 2):  # 释放偶数索引的序列
        engine.free_sequence(sequences[i])
        print(f"   🗑️ 释放序列 {sequences[i]}")
    
    # 尝试分配新序列
    try:
        engine.allocate_sequence("new_seq", 100)
        print(f"   ✅ 碎片化后成功分配新序列")
    except RuntimeError as e:
        print(f"   ❌ 碎片化后分配失败：{e}")
    
    # 打印最终统计
    stats = engine.get_memory_stats()
    print(f"   📊 最终内存统计：{stats}")


def benchmark_attention_performance():
    """基准测试注意力计算性能"""
    print("\n🧪 基准测试注意力计算性能...\n")
    
    # 测试不同的配置
    configs = [
        {"block_size": 16, "context_len": 128},
        {"block_size": 32, "context_len": 256},
        {"block_size": 64, "context_len": 512},
    ]
    
    for config in configs:
        print(f"📊 测试配置：块大小 {config['block_size']}, 上下文长度 {config['context_len']}")
        
        engine = PagedAttentionEngine(
            num_blocks=128,
            block_size=config["block_size"],
            num_heads=8,
            head_dim=64,
            device="cpu"
        )
        
        # 分配序列
        seq_id = "benchmark_seq"
        engine.allocate_sequence(seq_id, config["context_len"])
        
        # 准备测试数据
        keys = torch.randn(config["context_len"], 8, 64, dtype=torch.float16)
        values = torch.randn(config["context_len"], 8, 64, dtype=torch.float16)
        query = torch.randn(1, 8, 64, dtype=torch.float16)
        
        # 写入 KV Cache
        token_positions = list(range(config["context_len"]))
        engine.write_kv_cache(seq_id, token_positions, keys, values)
        
        # 性能测试
        num_iterations = 10
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = engine.compute_attention(query, seq_id, config["context_len"])
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000  # 毫秒
        
        print(f"   ⏱️ 平均计算时间：{avg_time:.2f} ms")
        print(f"   🚀 吞吐量：{config['context_len'] / avg_time * 1000:.0f} tokens/s")
        
        # 清理
        engine.free_sequence(seq_id)
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 第三步：PagedAttention 机制与内存优化")
    print("=" * 60)
    
    # 环境检查
    print(f"🔍 环境检查：")
    print(f"   PyTorch 版本：{torch.__version__}")
    print(f"   CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU 数量：{torch.cuda.device_count()}")
    print()
    
    try:
        # 1. 基本分配测试
        engine = test_basic_allocation()
        
        # 2. KV Cache 操作测试
        test_kv_cache_operations(engine)
        
        # 3. 注意力计算测试
        test_attention_computation(engine)
        
        # 4. Copy-on-Write 测试
        test_copy_on_write(engine)
        
        # 5. 内存效率对比
        test_memory_efficiency()
        
        # 6. 内存碎片测试
        test_block_fragmentation()
        
        # 7. 性能基准测试
        benchmark_attention_performance()
        
        print("🎉 所有测试完成！")
        print("\n📚 学习要点总结：")
        print("   1. PagedAttention 通过分块存储解决内存碎片问题")
        print("   2. Block Table 实现逻辑地址到物理地址的映射")
        print("   3. Copy-on-Write 优化减少内存复制开销")
        print("   4. 合适的块大小对性能和内存利用率都很重要")
        print("   5. 内存分配策略直接影响系统的可扩展性")
        
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