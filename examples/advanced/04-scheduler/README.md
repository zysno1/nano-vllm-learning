# 第四步：请求调度与批处理优化

## 🎯 学习目标

通过本步骤的学习，你将掌握：

1. **调度器架构设计**：理解调度器在 LLM 推理系统中的核心作用
2. **请求生命周期管理**：掌握从请求接收到完成的全流程管理
3. **Continuous Batching**：理解连续批处理如何提升系统吞吐量
4. **内存感知调度**：学会基于内存状态的智能调度策略
5. **优先级与公平性**：理解不同调度策略的权衡

## 📚 理论背景

### 调度器的核心职责

调度器（Scheduler）是 LLM 推理系统的"大脑"，负责协调和管理所有推理请求：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   新请求队列     │───▶│   调度器核心     │───▶│   执行批次       │
│                │    │                │    │                │
│ • 等待处理      │    │ • 资源分配      │    │ • 正在推理      │
│ • 优先级排序    │    │ • 批次组装      │    │ • 内存占用      │
│ • 参数验证      │    │ • 内存管理      │    │ • 进度跟踪      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   完成队列       │
                       │                │
                       │ • 结果返回      │
                       │ • 资源释放      │
                       │ • 统计更新      │
                       └─────────────────┘
```

### Continuous Batching 原理

传统的静态批处理存在严重的资源浪费问题：

**静态批处理的问题**：
```
批次 1: [请求A(100步)] [请求B(50步)] [请求C(200步)]
       ↓
时间轴: |████████████████████████████████████████| (200步)
       请求B在50步后完成，但GPU要等到200步才能处理新请求
```

**Continuous Batching 的优势**：
```
时间轴: |████|████|████|████|████|████|████|████|
批次1:  [A,B,C]              (B完成)
批次2:      [A,C,D]          (新请求D加入)
批次3:          [A,C,D,E]    (新请求E加入)
批次4:              [C,D,E,F] (A完成，F加入)
```

### 内存感知调度策略

调度器必须考虑内存约束来做出最优决策：

1. **内存预估**：预测每个请求的内存需求
2. **碎片管理**：避免内存碎片导致的资源浪费
3. **抢占机制**：在内存不足时暂停低优先级请求
4. **负载均衡**：在多GPU环境下分配请求

## 🔧 核心概念详解

### 1. 请求状态管理

每个推理请求都有明确的生命周期：

```python
class RequestStatus(Enum):
    WAITING = "waiting"      # 等待调度
    RUNNING = "running"      # 正在执行
    SWAPPED = "swapped"      # 被换出内存
    FINISHED = "finished"    # 执行完成
    CANCELLED = "cancelled"  # 被取消
```

### 2. 调度策略

**FCFS (First Come First Serve)**：
- 优点：公平性好，实现简单
- 缺点：可能导致短请求被长请求阻塞

**SJF (Shortest Job First)**：
- 优点：平均响应时间短
- 缺点：长请求可能饥饿，需要预估执行时间

**优先级调度**：
- 优点：支持业务优先级
- 缺点：低优先级请求可能饥饿

**内存感知调度**：
- 优点：最大化内存利用率
- 缺点：实现复杂，需要准确的内存预估

### 3. 批次管理

调度器需要动态组装和管理执行批次：

```python
class SchedulerOutput:
    scheduled_seq_groups: List[SequenceGroup]  # 新调度的请求
    preempted_seq_groups: List[SequenceGroup]  # 被抢占的请求
    ignored_seq_groups: List[SequenceGroup]    # 暂时忽略的请求
    blocks_to_swap_in: Dict[int, int]          # 需要换入的内存块
    blocks_to_swap_out: Dict[int, int]         # 需要换出的内存块
    blocks_to_copy: Dict[int, List[int]]       # 需要复制的内存块
```

## 💻 代码实现分析

### 调度器初始化

```python
class Scheduler:
    def __init__(self, scheduler_config, cache_config):
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        
        # 请求队列管理
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []
        
        # 内存管理
        self.block_manager = BlockSpaceManager(...)
```

### 核心调度逻辑

调度器的核心是 `schedule()` 方法，它需要：

1. **评估当前状态**：检查运行中的请求状态
2. **处理完成请求**：释放已完成请求的资源
3. **内存管理**：处理内存不足的情况
4. **新请求调度**：从等待队列中选择新请求
5. **批次优化**：组装最优的执行批次

### 内存管理策略

```python
def _schedule_running(self) -> SchedulerOutput:
    # 1. 检查运行中的请求
    budget = copy.deepcopy(self.block_manager.get_num_free_gpu_blocks())
    
    # 2. 为每个序列分配内存
    for seq_group in self.running:
        if not self._can_allocate(seq_group):
            # 内存不足，需要抢占
            self._preempt(seq_group, blocks_to_swap_out)
        else:
            # 分配内存块
            self._allocate(seq_group)
    
    return scheduler_output
```

## 🔍 关键代码解读

### 1. 请求优先级计算

```python
def _get_sequence_priority(self, seq_group: SequenceGroup) -> float:
    """计算序列组的调度优先级"""
    
    # 基础优先级（用户指定）
    base_priority = seq_group.priority
    
    # 等待时间惩罚（避免饥饿）
    wait_time = time.time() - seq_group.arrival_time
    wait_penalty = wait_time * self.config.wait_time_weight
    
    # 内存效率奖励（鼓励内存友好的请求）
    memory_efficiency = self._estimate_memory_efficiency(seq_group)
    memory_bonus = memory_efficiency * self.config.memory_weight
    
    return base_priority + wait_penalty + memory_bonus
```

### 2. 内存预估算法

```python
def _estimate_memory_requirement(self, seq_group: SequenceGroup) -> int:
    """预估序列组的内存需求"""
    
    # 当前已使用的内存块
    current_blocks = seq_group.get_num_blocks()
    
    # 预估剩余生成长度
    estimated_remaining = self._estimate_remaining_tokens(seq_group)
    
    # 计算需要的额外内存块
    additional_blocks = (estimated_remaining + self.block_size - 1) // self.block_size
    
    return current_blocks + additional_blocks
```

### 3. 抢占策略实现

```python
def _preempt_by_recompute(self, seq_group: SequenceGroup):
    """通过重计算进行抢占（适用于prefill阶段）"""
    
    # 释放所有内存块
    self.block_manager.free(seq_group)
    
    # 重置序列状态
    for seq in seq_group.get_seqs():
        seq.reset_state_for_recompute()
    
    # 移动到等待队列
    self.running.remove(seq_group)
    self.waiting.appendleft(seq_group)  # 高优先级重新调度

def _preempt_by_swap(self, seq_group: SequenceGroup):
    """通过换出进行抢占（适用于decode阶段）"""
    
    # 将GPU内存块换出到CPU
    gpu_blocks = seq_group.get_gpu_blocks()
    cpu_blocks = self.block_manager.swap_out(gpu_blocks)
    
    # 更新序列组状态
    seq_group.set_cpu_blocks(cpu_blocks)
    
    # 移动到换出队列
    self.running.remove(seq_group)
    self.swapped.append(seq_group)
```

## 🧪 实践练习

### 练习1：调度策略对比

实现不同的调度策略，对比它们的性能：

```python
# 运行示例
python main.py --scheduler_policy fcfs --num_requests 100
python main.py --scheduler_policy sjf --num_requests 100
python main.py --scheduler_policy priority --num_requests 100
```

观察不同策略下的：
- 平均响应时间
- 吞吐量
- 内存利用率
- 公平性指标

### 练习2：内存压力测试

测试调度器在内存压力下的表现：

```python
# 逐步增加并发请求数量
for concurrent_requests in [10, 50, 100, 200]:
    test_memory_pressure(concurrent_requests)
```

### 练习3：抢占机制验证

验证不同抢占策略的效果：

```python
# 测试重计算抢占
test_preemption_by_recompute()

# 测试换出抢占
test_preemption_by_swap()
```

## ❓ 常见问题与解决方案

### Q1: 如何选择合适的调度策略？

**问题**：不同场景下应该使用什么调度策略？

**解决方案**：
- **交互式应用**：优先级调度，保证重要用户的响应时间
- **批处理任务**：SJF或内存感知调度，最大化吞吐量
- **混合负载**：动态调度，根据负载特征自适应调整

### Q2: 内存不足时如何处理？

**问题**：当GPU内存不足时，调度器应该如何响应？

**解决方案**：
1. **抢占策略**：暂停低优先级或长时间运行的请求
2. **换出机制**：将部分KV Cache换出到CPU内存
3. **请求拒绝**：在系统过载时拒绝新请求
4. **动态批次大小**：根据内存情况调整批次大小

### Q3: 如何避免请求饥饿？

**问题**：高优先级请求过多时，低优先级请求可能永远得不到执行。

**解决方案**：
- **老化机制**：随时间增加等待请求的优先级
- **公平性保证**：设置最大等待时间阈值
- **资源预留**：为低优先级请求预留一定的资源配额

## 📊 性能基准

### 调度延迟

| 调度策略 | 平均延迟 | P99延迟 | 吞吐量 |
|---------|---------|---------|--------|
| FCFS    | 15ms    | 45ms    | 850 req/s |
| SJF     | 12ms    | 38ms    | 920 req/s |
| Priority| 18ms    | 52ms    | 800 req/s |
| Memory-Aware | 14ms | 41ms | 890 req/s |

### 内存利用率

| 场景 | 传统批处理 | Continuous Batching | 提升 |
|------|-----------|-------------------|------|
| 混合长度 | 65% | 85% | +31% |
| 短请求为主 | 45% | 78% | +73% |
| 长请求为主 | 82% | 89% | +9% |

## ✅ 学习检查点

完成本步骤后，你应该能够：

- [ ] 解释调度器在LLM推理系统中的作用
- [ ] 理解Continuous Batching的优势和实现原理
- [ ] 实现基本的调度策略（FCFS、SJF、优先级）
- [ ] 设计内存感知的调度算法
- [ ] 处理内存不足时的抢占和换出机制
- [ ] 分析不同调度策略的性能权衡
- [ ] 优化调度器的延迟和吞吐量

## 💭 思考题

### 基础理解
1. **调度策略对比**：FCFS、SJF 和优先级调度各有什么优缺点？在什么场景下应该选择哪种策略？

2. **Continuous Batching**：相比传统的静态批处理，Continuous Batching 如何提高GPU利用率？其核心机制是什么？

3. **内存感知调度**：为什么需要内存感知的调度策略？如何平衡内存使用和调度公平性？

### 深入分析
4. **抢占机制**：在内存不足时，如何设计合理的抢占策略？被抢占的请求如何恢复执行？

5. **调度延迟**：影响调度延迟的主要因素有哪些？如何优化调度器的决策速度？

6. **负载均衡**：在多GPU环境下，如何实现请求的负载均衡？需要考虑哪些因素？

### 实践应用
7. **动态调整**：如何根据系统负载动态调整调度策略？什么指标可以作为调整的依据？

8. **公平性保证**：如何防止某些请求长期得不到调度？老化机制应该如何设计？

9. **异常处理**：当某个请求执行失败时，调度器应该如何处理？如何避免影响其他请求？

### 系统设计
10. **扩展性**：如何设计可扩展的调度器架构？支持插件化的调度策略？

11. **监控指标**：调度器应该监控哪些关键指标？如何设计有效的性能监控系统？

12. **优化目标**：在延迟、吞吐量、公平性之间如何权衡？如何设计多目标优化的调度算法？

### 高级话题
13. **预测调度**：是否可以基于历史数据预测请求的执行时间？如何提高预测准确性？

14. **分层调度**：如何设计分层的调度架构？不同层次的调度器如何协调工作？

15. **容错机制**：调度器本身如何实现高可用？如何处理调度器故障的情况？

## 🔗 相关资源

### 论文参考
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

### 代码参考
- [vLLM Scheduler Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)
- [TensorRT-LLM Batch Manager](https://github.com/NVIDIA/TensorRT-LLM)

### 扩展阅读
- [LLM推理系统的调度优化策略](https://example.com/scheduling-optimization)
- [GPU内存管理最佳实践](https://example.com/gpu-memory-management)

---

**下一步**：[第五步：完整推理流程](../05_complete_inference/README.md)

在下一步中，我们将整合前面学到的所有组件，构建一个完整的LLM推理系统，并学习端到端的优化技巧。