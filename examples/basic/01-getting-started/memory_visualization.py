#!/usr/bin/env python3
"""
内存使用可视化演示
展示 PagedAttention 的内存管理机制

基于费曼学习法：通过可视化理解复杂概念
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import time
import random

class MemoryBlock:
    """内存块类"""
    
    def __init__(self, block_id: int, size: int = 16):
        self.block_id = block_id
        self.size = size
        self.tokens = []
        self.is_allocated = False
        self.sequence_id = None
        
    def allocate(self, sequence_id: int):
        """分配给特定序列"""
        self.is_allocated = True
        self.sequence_id = sequence_id
        
    def deallocate(self):
        """释放内存块"""
        self.is_allocated = False
        self.sequence_id = None
        self.tokens.clear()
        
    def add_token(self, token: str):
        """添加token"""
        if len(self.tokens) < self.size:
            self.tokens.append(token)
            return True
        return False
        
    def is_full(self) -> bool:
        """检查是否已满"""
        return len(self.tokens) >= self.size

class TraditionalMemoryManager:
    """传统内存管理器"""
    
    def __init__(self, max_seq_len: int = 2048):
        self.max_seq_len = max_seq_len
        self.allocated_sequences = {}
        self.total_allocated = 0
        
    def allocate_sequence(self, seq_id: int, estimated_len: int) -> Dict:
        """为序列分配内存（预分配最大长度）"""
        # 传统方法：总是分配最大可能长度
        allocated_size = self.max_seq_len
        
        self.allocated_sequences[seq_id] = {
            'allocated_size': allocated_size,
            'used_size': 0,
            'estimated_len': estimated_len,
            'tokens': []
        }
        
        self.total_allocated += allocated_size
        
        return {
            'seq_id': seq_id,
            'allocated': allocated_size,
            'utilization': 0.0
        }
    
    def add_token(self, seq_id: int, token: str):
        """添加token"""
        if seq_id in self.allocated_sequences:
            seq_info = self.allocated_sequences[seq_id]
            seq_info['tokens'].append(token)
            seq_info['used_size'] = len(seq_info['tokens'])
    
    def get_memory_stats(self) -> Dict:
        """获取内存统计"""
        total_used = sum(seq['used_size'] for seq in self.allocated_sequences.values())
        utilization = total_used / self.total_allocated if self.total_allocated > 0 else 0
        
        return {
            'total_allocated': self.total_allocated,
            'total_used': total_used,
            'utilization': utilization,
            'sequences': len(self.allocated_sequences)
        }

class PagedAttentionMemoryManager:
    """PagedAttention 内存管理器"""
    
    def __init__(self, total_blocks: int = 1000, block_size: int = 16):
        self.block_size = block_size
        self.blocks = [MemoryBlock(i, block_size) for i in range(total_blocks)]
        self.free_blocks = list(range(total_blocks))
        self.sequence_blocks = {}  # seq_id -> [block_ids]
        
    def allocate_sequence(self, seq_id: int, estimated_len: int) -> Dict:
        """为序列分配内存（按需分配）"""
        # 计算初始需要的块数（可以动态增长）
        initial_blocks_needed = max(1, (estimated_len + self.block_size - 1) // self.block_size)
        initial_blocks_needed = min(initial_blocks_needed, len(self.free_blocks))
        
        if initial_blocks_needed == 0:
            return {'error': 'No free blocks available'}
        
        # 分配初始块
        allocated_blocks = []
        for _ in range(initial_blocks_needed):
            if self.free_blocks:
                block_id = self.free_blocks.pop(0)
                self.blocks[block_id].allocate(seq_id)
                allocated_blocks.append(block_id)
        
        self.sequence_blocks[seq_id] = allocated_blocks
        
        return {
            'seq_id': seq_id,
            'allocated_blocks': len(allocated_blocks),
            'allocated_size': len(allocated_blocks) * self.block_size
        }
    
    def add_token(self, seq_id: int, token: str) -> bool:
        """添加token，必要时分配新块"""
        if seq_id not in self.sequence_blocks:
            return False
        
        blocks = self.sequence_blocks[seq_id]
        
        # 尝试添加到现有块
        for block_id in blocks:
            if self.blocks[block_id].add_token(token):
                return True
        
        # 需要新块
        if self.free_blocks:
            new_block_id = self.free_blocks.pop(0)
            self.blocks[new_block_id].allocate(seq_id)
            self.blocks[new_block_id].add_token(token)
            blocks.append(new_block_id)
            return True
        
        return False  # 内存不足
    
    def deallocate_sequence(self, seq_id: int):
        """释放序列内存"""
        if seq_id in self.sequence_blocks:
            for block_id in self.sequence_blocks[seq_id]:
                self.blocks[block_id].deallocate()
                self.free_blocks.append(block_id)
            del self.sequence_blocks[seq_id]
    
    def get_memory_stats(self) -> Dict:
        """获取内存统计"""
        allocated_blocks = len([b for b in self.blocks if b.is_allocated])
        total_tokens = sum(len(b.tokens) for b in self.blocks if b.is_allocated)
        
        return {
            'total_blocks': len(self.blocks),
            'allocated_blocks': allocated_blocks,
            'free_blocks': len(self.free_blocks),
            'total_tokens': total_tokens,
            'utilization': allocated_blocks / len(self.blocks),
            'sequences': len(self.sequence_blocks)
        }

def visualize_memory_comparison():
    """可视化内存使用对比"""
    
    print("🎨 内存使用可视化对比")
    print("=" * 50)
    
    # 创建测试场景
    sequences = [
        {'id': 1, 'estimated_len': 50, 'actual_tokens': 45},
        {'id': 2, 'estimated_len': 200, 'actual_tokens': 180},
        {'id': 3, 'estimated_len': 100, 'actual_tokens': 120},
        {'id': 4, 'estimated_len': 300, 'actual_tokens': 250},
    ]
    
    # 初始化管理器
    traditional = TraditionalMemoryManager(max_seq_len=512)
    paged = PagedAttentionMemoryManager(total_blocks=200, block_size=16)
    
    # 分配内存并添加tokens
    for seq in sequences:
        # 传统方法
        traditional.allocate_sequence(seq['id'], seq['estimated_len'])
        for i in range(seq['actual_tokens']):
            traditional.add_token(seq['id'], f"token_{i}")
        
        # PagedAttention
        paged.allocate_sequence(seq['id'], seq['estimated_len'])
        for i in range(seq['actual_tokens']):
            paged.add_token(seq['id'], f"token_{i}")
    
    # 获取统计信息
    trad_stats = traditional.get_memory_stats()
    paged_stats = paged.get_memory_stats()
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 传统方法可视化
    ax1.set_title('传统内存管理\n(预分配固定大小)', fontsize=14, fontweight='bold')
    
    y_pos = 0
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, seq in enumerate(sequences):
        # 分配的内存（灰色背景）
        allocated_rect = patches.Rectangle(
            (0, y_pos), traditional.max_seq_len, 0.8,
            linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.5
        )
        ax1.add_patch(allocated_rect)
        
        # 实际使用的内存（彩色）
        used_rect = patches.Rectangle(
            (0, y_pos), seq['actual_tokens'], 0.8,
            linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.8
        )
        ax1.add_patch(used_rect)
        
        # 添加标签
        ax1.text(traditional.max_seq_len/2, y_pos + 0.4, 
                f"Seq {seq['id']}: {seq['actual_tokens']}/{traditional.max_seq_len}",
                ha='center', va='center', fontweight='bold')
        
        y_pos += 1
    
    ax1.set_xlim(0, traditional.max_seq_len)
    ax1.set_ylim(-0.5, len(sequences))
    ax1.set_xlabel('内存单位 (tokens)')
    ax1.set_ylabel('序列')
    ax1.grid(True, alpha=0.3)
    
    # PagedAttention 可视化
    ax2.set_title('PagedAttention 内存管理\n(按需分配)', fontsize=14, fontweight='bold')
    
    # 计算每个序列使用的块
    block_width = 16
    max_blocks_per_row = 20
    
    y_pos = 0
    for i, seq in enumerate(sequences):
        seq_blocks = paged.sequence_blocks[seq['id']]
        
        for j, block_id in enumerate(seq_blocks):
            block = paged.blocks[block_id]
            
            # 计算块的位置
            row = j // max_blocks_per_row
            col = j % max_blocks_per_row
            
            x_pos = col * (block_width + 1)
            y_block_pos = y_pos - row * 0.3
            
            # 绘制块
            if len(block.tokens) == block.size:
                # 满块
                block_rect = patches.Rectangle(
                    (x_pos, y_block_pos), block_width, 0.25,
                    linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.8
                )
            else:
                # 部分填充的块
                block_rect = patches.Rectangle(
                    (x_pos, y_block_pos), block_width, 0.25,
                    linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.5
                )
                # 显示实际使用部分
                used_width = (len(block.tokens) / block.size) * block_width
                used_rect = patches.Rectangle(
                    (x_pos, y_block_pos), used_width, 0.25,
                    linewidth=0, facecolor=colors[i], alpha=0.8
                )
                ax2.add_patch(used_rect)
            
            ax2.add_patch(block_rect)
        
        # 添加序列标签
        ax2.text(-20, y_pos, f"Seq {seq['id']}", ha='right', va='center', fontweight='bold')
        y_pos += 1
    
    ax2.set_xlim(-30, max_blocks_per_row * (block_width + 1))
    ax2.set_ylim(-0.5, len(sequences))
    ax2.set_xlabel('内存块 (每块16 tokens)')
    ax2.set_ylabel('序列')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memory_visualization.png', dpi=300, bbox_inches='tight')
    
    # 打印统计信息
    print(f"\n📊 内存使用统计对比:")
    print(f"{'方法':<15} {'总分配':<10} {'实际使用':<10} {'利用率':<10} {'内存节省'}")
    print("-" * 60)
    
    memory_saved = (trad_stats['total_allocated'] - paged_stats['allocated_blocks'] * 16) / trad_stats['total_allocated']
    
    print(f"{'传统方法':<15} {trad_stats['total_allocated']:<10} {trad_stats['total_used']:<10} {trad_stats['utilization']:.1%:<10} -")
    print(f"{'PagedAttention':<15} {paged_stats['allocated_blocks']*16:<10} {paged_stats['total_tokens']:<10} {paged_stats['total_tokens']/(paged_stats['allocated_blocks']*16):.1%:<10} {memory_saved:.1%}")
    
    print(f"\n📈 可视化图表已保存为 memory_visualization.png")

def demonstrate_dynamic_allocation():
    """演示动态内存分配过程"""
    
    print("\n🔄 动态内存分配演示")
    print("=" * 50)
    
    paged = PagedAttentionMemoryManager(total_blocks=50, block_size=8)
    
    # 模拟序列生成过程
    seq_id = 1
    paged.allocate_sequence(seq_id, estimated_len=10)
    
    print("初始分配后的状态:")
    stats = paged.get_memory_stats()
    print(f"  已分配块数: {stats['allocated_blocks']}")
    print(f"  空闲块数: {stats['free_blocks']}")
    
    # 模拟token生成过程
    tokens_to_generate = 25
    print(f"\n开始生成 {tokens_to_generate} 个tokens...")
    
    for i in range(tokens_to_generate):
        success = paged.add_token(seq_id, f"token_{i}")
        
        if (i + 1) % 8 == 0:  # 每8个token打印一次状态
            stats = paged.get_memory_stats()
            print(f"  生成 {i+1} tokens后:")
            print(f"    已分配块数: {stats['allocated_blocks']}")
            print(f"    总token数: {stats['total_tokens']}")
            print(f"    内存利用率: {stats['total_tokens']/(stats['allocated_blocks']*8):.1%}")
        
        if not success:
            print(f"    ❌ 第 {i+1} 个token分配失败（内存不足）")
            break
    
    print(f"\n✅ 动态分配演示完成")

def analyze_memory_fragmentation():
    """分析内存碎片化问题"""
    
    print("\n🧩 内存碎片化分析")
    print("=" * 50)
    
    paged = PagedAttentionMemoryManager(total_blocks=100, block_size=16)
    
    # 创建多个不同长度的序列
    sequences = [
        {'id': 1, 'tokens': 10},
        {'id': 2, 'tokens': 25},
        {'id': 3, 'tokens': 8},
        {'id': 4, 'tokens': 30},
        {'id': 5, 'tokens': 15},
    ]
    
    print("分配多个序列...")
    for seq in sequences:
        paged.allocate_sequence(seq['id'], seq['tokens'])
        for i in range(seq['tokens']):
            paged.add_token(seq['id'], f"token_{i}")
    
    stats_before = paged.get_memory_stats()
    print(f"分配后状态:")
    print(f"  已分配块数: {stats_before['allocated_blocks']}")
    print(f"  总token数: {stats_before['total_tokens']}")
    print(f"  平均块利用率: {stats_before['total_tokens']/(stats_before['allocated_blocks']*16):.1%}")
    
    # 释放部分序列（模拟完成的请求）
    print(f"\n释放序列 2 和 4...")
    paged.deallocate_sequence(2)
    paged.deallocate_sequence(4)
    
    stats_after = paged.get_memory_stats()
    print(f"释放后状态:")
    print(f"  已分配块数: {stats_after['allocated_blocks']}")
    print(f"  空闲块数: {stats_after['free_blocks']}")
    print(f"  总token数: {stats_after['total_tokens']}")
    
    # 分析碎片化
    allocated_blocks = [b for b in paged.blocks if b.is_allocated]
    fragmentation_info = []
    
    for block in allocated_blocks:
        utilization = len(block.tokens) / block.size
        fragmentation_info.append({
            'block_id': block.block_id,
            'seq_id': block.sequence_id,
            'utilization': utilization,
            'wasted_space': block.size - len(block.tokens)
        })
    
    total_wasted = sum(info['wasted_space'] for info in fragmentation_info)
    print(f"\n碎片化分析:")
    print(f"  总浪费空间: {total_wasted} tokens")
    print(f"  平均块利用率: {np.mean([info['utilization'] for info in fragmentation_info]):.1%}")
    
    print(f"\n💡 PagedAttention 优势:")
    print(f"  1. 动态分配减少预分配浪费")
    print(f"  2. 块级管理便于内存回收")
    print(f"  3. 支持不同长度序列混合处理")

def main():
    """主函数"""
    
    print("🎯 PagedAttention 内存管理可视化演示")
    print("基于费曼学习法：通过可视化理解抽象概念")
    print("=" * 60)
    
    try:
        # 内存使用对比可视化
        visualize_memory_comparison()
        
        # 动态分配演示
        demonstrate_dynamic_allocation()
        
        # 碎片化分析
        analyze_memory_fragmentation()
        
        print("\n🎉 内存管理演示完成！")
        print("\n💡 关键理解：")
        print("1. PagedAttention 通过块级管理实现高效内存利用")
        print("2. 动态分配避免传统方法的预分配浪费")
        print("3. 支持不同长度序列的灵活混合处理")
        print("4. 内存回收机制减少碎片化问题")
        
        print("\n📚 深入学习建议：")
        print("1. 理解注意力机制原理 → docs/concepts.md")
        print("2. 学习系统架构设计 → docs/architecture.md")
        print("3. 实践性能优化 → examples/advanced/")
        
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("💡 请安装: pip install matplotlib numpy")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("💡 请查看 docs/faq.md 获取帮助")

if __name__ == "__main__":
    main()