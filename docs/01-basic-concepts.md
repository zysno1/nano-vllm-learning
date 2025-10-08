# 📚 基础概念

本章节介绍 nano-vLLM 推理引擎的基础概念和核心技术原理。通过学习这些基础概念，你将建立对 vLLM 技术的整体认知。

## 🏗️ 系统架构概览

在深入学习具体概念之前，让我们先了解 nano-vLLM 的整体架构：

![nano-vLLM 系统架构](assets/system-architecture-diagram.svg)

上图展示了 nano-vLLM 推理引擎的核心组件和数据流向。主要包括：
- **请求处理层**：接收和预处理用户请求
- **调度器**：智能调度和批处理管理
- **内存管理**：PagedAttention 和 KV Cache 管理
- **推理引擎**：模型推理和张量并行处理
- **输出生成**：结果后处理和返回

## 🎯 学习目标

通过本章节的学习，你将能够：

1. **理解推理引擎的基本概念** - 掌握大语言模型推理的核心原理
2. **掌握张量并行技术** - 理解如何通过并行化提升推理性能  
3. **了解注意力机制优化** - 学习 KV Cache 等关键优化技术
4. **理解内存管理策略** - 掌握高效的内存使用和管理方法

---

## 1. 推理引擎基础

### 1.1 大语言模型推理概述

大语言模型（LLM）推理是指使用已训练好的模型对输入文本进行处理，生成相应输出的过程。与训练过程不同，推理过程具有以下特点：

#### 推理 vs 训练的区别

| 特性 | 训练 | 推理 |
|------|------|------|
| **目标** | 学习参数 | 生成输出 |
| **计算模式** | 批量并行 | 序列生成 |
| **内存使用** | 需要梯度 | 只需前向传播 |
| **延迟要求** | 可接受较高延迟 | 要求低延迟 |
| **吞吐量** | 重视训练效率 | 重视推理吞吐量 |

### 1.2 推理过程的主要挑战

#### 内存挑战
- **KV Cache 增长**：随着序列长度增加，注意力机制的 Key-Value 缓存呈线性增长
- **内存碎片化**：动态序列长度导致内存分配不规则
- **峰值内存**：长序列推理时内存需求激增

#### 计算挑战  
- **序列依赖**：自回归生成无法完全并行化
- **计算不均衡**：不同序列长度的计算量差异很大
- **硬件利用率**：GPU 在处理可变长度序列时利用率不高

#### 延迟挑战
- **首 Token 延迟**：从输入到第一个 Token 输出的时间
- **后续 Token 延迟**：每个后续 Token 的生成时间
- **批处理权衡**：批大小与延迟之间的平衡

### 1.3 推理引擎的作用

推理引擎通过以下技术解决上述挑战：

![数据流程图](assets/data-flow-diagram.svg)

上图展示了推理引擎的完整数据处理流程，从请求接收到结果返回的全过程。

1. **内存优化**
   - 分页注意力（Paged Attention）
   - 内存池化管理
   - KV Cache 压缩

2. **计算优化**
   - 连续批处理（Continuous Batching）
   - 动态调度算法
   - 算子融合优化

3. **系统优化**
   - 异步处理流水线
   - 多 GPU 协调
   - 负载均衡策略

---

## 2. 张量并行技术

### 2.1 张量并行基本原理

张量并行是一种模型并行技术，将模型的参数矩阵按维度切分到多个设备上，实现并行计算。

#### 与数据并行的区别

| 并行类型 | 数据并行 | 张量并行 |
|----------|----------|----------|
| **切分对象** | 输入数据 | 模型参数 |
| **通信模式** | AllReduce | AllGather/ReduceScatter |
| **内存占用** | 每设备存储完整模型 | 每设备存储部分模型 |
| **适用场景** | 模型较小 | 模型较大 |

### 2.2 张量并行的实现方式

#### 线性层并行化
```python
# 原始线性层
y = xW + b  # x: [batch, in_features], W: [in_features, out_features]

# 按列切分（Column Parallel）
W = [W1, W2, ..., Wn]  # 按输出维度切分
y_i = xW_i + b_i       # 每个设备计算部分输出
y = concat([y1, y2, ..., yn])  # 拼接结果

# 按行切分（Row Parallel）  
W = [W1; W2; ...; Wn]  # 按输入维度切分
x = [x1, x2, ..., xn]  # 输入也需要切分
y_i = x_i W_i          # 每个设备计算部分结果
y = sum([y1, y2, ..., yn])  # 求和得到最终结果
```

#### 注意力机制并行化
```python
# Multi-Head Attention 并行化
# 将不同的 attention head 分配到不同设备
heads_per_device = num_heads // world_size
local_heads = heads[device_id * heads_per_device : (device_id + 1) * heads_per_device]

# 每个设备计算分配给它的 heads
local_output = multi_head_attention(x, local_heads)

# AllGather 收集所有设备的结果
all_outputs = all_gather(local_output)
final_output = concat(all_outputs, dim=head_dim)
```

### 2.3 性能优势和适用场景

#### 性能优势
- **内存效率**：单设备内存需求降低 1/N（N 为并行度）
- **计算并行**：多设备同时计算，理论上加速 N 倍
- **扩展性好**：支持超大模型的推理

#### 适用场景
- **大模型推理**：单设备无法容纳完整模型
- **高吞吐需求**：需要处理大量并发请求
- **资源充足**：有多个 GPU 设备可用

---

## 3. 注意力机制优化

### 3.1 注意力机制在推理中的特点

#### 计算复杂度
- **时间复杂度**：O(n²d)，其中 n 是序列长度，d 是隐藏维度
- **空间复杂度**：O(n²) 用于存储注意力矩阵，O(nd) 用于 KV Cache

#### 内存访问模式
```python
# 标准注意力计算
Q = x @ W_q  # Query: [batch, seq_len, d_model]
K = x @ W_k  # Key: [batch, seq_len, d_model]  
V = x @ W_v  # Value: [batch, seq_len, d_model]

# 注意力分数计算
scores = Q @ K.T / sqrt(d_k)  # [batch, seq_len, seq_len]
attn = softmax(scores)        # [batch, seq_len, seq_len]
output = attn @ V             # [batch, seq_len, d_model]
```

### 3.2 KV Cache 原理和实现

#### KV Cache 的必要性
在自回归生成中，每个新 Token 的生成都需要计算与之前所有 Token 的注意力。KV Cache 避免了重复计算：

```python
# 不使用 KV Cache（低效）
for i in range(max_length):
    # 每次都重新计算所有位置的 K, V
    K = tokens[:i+1] @ W_k  # 重复计算
    V = tokens[:i+1] @ W_v  # 重复计算
    output_i = attention(Q_i, K, V)

# 使用 KV Cache（高效）
K_cache, V_cache = [], []
for i in range(max_length):
    # 只计算新位置的 K, V
    K_i = token_i @ W_k
    V_i = token_i @ W_v
    
    # 追加到缓存
    K_cache.append(K_i)
    V_cache.append(V_i)
    
    # 使用完整缓存计算注意力
    output_i = attention(Q_i, K_cache, V_cache)
```

#### KV Cache 的内存管理
```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, num_heads, head_dim):
        # 预分配内存
        self.k_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.v_cache = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim)
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32)
    
    def append(self, batch_idx, k, v):
        seq_len = self.seq_lens[batch_idx]
        self.k_cache[batch_idx, :, seq_len] = k
        self.v_cache[batch_idx, :, seq_len] = v
        self.seq_lens[batch_idx] += 1
    
    def get(self, batch_idx):
        seq_len = self.seq_lens[batch_idx]
        return (
            self.k_cache[batch_idx, :, :seq_len],
            self.v_cache[batch_idx, :, :seq_len]
        )
```

### 3.3 内存优化策略

#### 分页注意力（Paged Attention）
将 KV Cache 分割成固定大小的页面，类似操作系统的虚拟内存管理：

![PagedAttention 可视化](assets/paged-attention-visualization.svg)

上图对比了传统注意力机制与 PagedAttention 的内存使用方式。PagedAttention 通过分页管理显著提升了内存利用率。

```python
class PagedKVCache:
    def __init__(self, page_size=16, num_pages=1000):
        self.page_size = page_size
        self.pages = torch.zeros(num_pages, 2, num_heads, page_size, head_dim)
        self.free_pages = list(range(num_pages))
        self.sequence_pages = {}  # seq_id -> [page_ids]
    
    def allocate_sequence(self, seq_id, seq_len):
        num_pages_needed = (seq_len + self.page_size - 1) // self.page_size
        pages = [self.free_pages.pop() for _ in range(num_pages_needed)]
        self.sequence_pages[seq_id] = pages
        return pages
    
    def free_sequence(self, seq_id):
        pages = self.sequence_pages.pop(seq_id)
        self.free_pages.extend(pages)
```
```

#### 注意力计算优化
```python
def paged_attention(q, k_pages, v_pages, page_table):
    """分页注意力计算"""
    outputs = []
    
    for page_id in page_table:
        k_page = k_pages[page_id]  # [num_heads, page_size, head_dim]
        v_page = v_pages[page_id]  # [num_heads, page_size, head_dim]
        
        # 计算当前页面的注意力
        scores = q @ k_page.transpose(-2, -1)  # [num_heads, 1, page_size]
        attn = F.softmax(scores, dim=-1)
        output = attn @ v_page  # [num_heads, 1, head_dim]
        outputs.append(output)
    
    return torch.cat(outputs, dim=-2)
```

---

## 4. 内存管理策略

### 4.1 推理过程中的内存使用模式

#### 内存使用特点
- **动态性**：序列长度在推理过程中动态变化
- **不规律性**：不同请求的序列长度差异很大
- **峰值性**：某些时刻内存需求激增

#### 性能对比分析

![性能对比图表](assets/performance-comparison-chart.svg)

上图展示了不同内存管理策略的性能对比，包括内存利用率、吞吐量和延迟等关键指标。

#### 内存分配模式
```python
# 传统内存分配（低效）
class NaiveMemoryManager:
    def allocate_kv_cache(self, batch_size, seq_len):
        # 为每个序列分配独立的内存
        return torch.zeros(batch_size, num_heads, seq_len, head_dim)
    
    def problems(self):
        # 1. 内存碎片化严重
        # 2. 无法处理动态长度
        # 3. 内存利用率低
        pass

# 改进的内存分配
class SmartMemoryManager:
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.block_size = 16  # 固定块大小
    
    def allocate_kv_cache(self, seq_len):
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        blocks = [self.memory_pool.allocate_block() for _ in range(num_blocks)]
        return blocks
```

### 4.2 内存池化技术

#### 内存池设计
```python
class MemoryPool:
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.blocks = torch.zeros(num_blocks, 2, num_heads, block_size, head_dim)
        self.free_blocks = set(range(num_blocks))
        self.used_blocks = set()
    
    def allocate_block(self):
        if not self.free_blocks:
            raise OutOfMemoryError("No free blocks available")
        
        block_id = self.free_blocks.pop()
        self.used_blocks.add(block_id)
        return block_id
    
    def free_block(self, block_id):
        if block_id in self.used_blocks:
            self.used_blocks.remove(block_id)
            self.free_blocks.add(block_id)
            # 清零块内容
            self.blocks[block_id].zero_()
```

#### 内存回收策略
```python
class MemoryRecycler:
    def __init__(self, memory_pool):
        self.memory_pool = memory_pool
        self.sequence_blocks = {}  # seq_id -> [block_ids]
    
    def allocate_sequence(self, seq_id, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            block_id = self.memory_pool.allocate_block()
            blocks.append(block_id)
        
        self.sequence_blocks[seq_id] = blocks
        return blocks
    
    def free_sequence(self, seq_id):
        if seq_id in self.sequence_blocks:
            blocks = self.sequence_blocks.pop(seq_id)
            for block_id in blocks:
                self.memory_pool.free_block(block_id)
    
    def garbage_collect(self):
        # 定期回收未使用的内存
        for seq_id in list(self.sequence_blocks.keys()):
            if self.is_sequence_finished(seq_id):
                self.free_sequence(seq_id)
```

### 4.3 动态内存分配策略

#### 按需分配
```python
class DynamicAllocator:
    def __init__(self):
        self.allocations = {}
        self.growth_factor = 1.5
    
    def allocate_or_grow(self, seq_id, required_size):
        if seq_id not in self.allocations:
            # 首次分配，预留一些空间
            initial_size = max(required_size, 32)
            self.allocations[seq_id] = self.allocate_blocks(initial_size)
        
        current_size = len(self.allocations[seq_id])
        if required_size > current_size:
            # 需要扩容
            new_size = int(required_size * self.growth_factor)
            additional_blocks = new_size - current_size
            
            new_blocks = self.allocate_blocks(additional_blocks)
            self.allocations[seq_id].extend(new_blocks)
        
        return self.allocations[seq_id][:required_size]
```

#### 内存压缩和整理
```python
class MemoryCompactor:
    def __init__(self, memory_pool):
        self.memory_pool = memory_pool
    
    def compact(self):
        """整理内存碎片"""
        # 1. 识别碎片化的分配
        fragmented_sequences = self.find_fragmented_sequences()
        
        # 2. 重新分配连续内存
        for seq_id in fragmented_sequences:
            old_blocks = self.sequence_blocks[seq_id]
            new_blocks = self.allocate_contiguous_blocks(len(old_blocks))
            
            # 3. 复制数据
            self.copy_blocks(old_blocks, new_blocks)
            
            # 4. 释放旧内存
            self.free_blocks(old_blocks)
            
            # 5. 更新映射
            self.sequence_blocks[seq_id] = new_blocks
    
    def find_fragmented_sequences(self):
        """找出内存碎片化严重的序列"""
        fragmented = []
        for seq_id, blocks in self.sequence_blocks.items():
            if self.is_fragmented(blocks):
                fragmented.append(seq_id)
        return fragmented
```

---

## 📖 学习建议

### 💡 理解要点
1. **推理引擎的核心价值**：解决内存、计算、延迟三大挑战
2. **张量并行的本质**：通过参数切分实现模型并行
3. **KV Cache 的重要性**：避免重复计算，提升推理效率
4. **内存管理的关键**：动态分配、池化管理、碎片整理

### 🔧 实践建议
1. **动手实验**：运行相关示例代码，观察内存使用情况
2. **参数调优**：尝试不同的批大小、序列长度，观察性能变化
3. **性能分析**：使用性能分析工具了解瓶颈所在
4. **代码阅读**：深入阅读 vLLM 源码，理解实现细节

### 📚 延伸阅读
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原理
- [Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180) - PagedAttention 论文
- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) - NVIDIA 的推理优化库

---

通过掌握这些基础概念，你已经建立了对 nano-vLLM 技术的整体认知。接下来可以深入学习 [架构设计](02-architecture.md) 和 [代码分析](03-code-analysis.md)，进一步理解系统的实现细节。