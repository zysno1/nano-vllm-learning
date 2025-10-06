# 张量并行 (Tensor Parallelism)

## 🎯 什么是张量并行？

张量并行是一种模型并行技术，通过将模型的权重张量在多个设备（通常是GPU）之间进行分割，使得每个设备只需要存储和计算模型的一部分，从而实现大模型的分布式推理。

### 核心思想
- **权重分割**：将大的权重矩阵按某个维度分割到多个GPU
- **计算分布**：每个GPU只计算自己负责的部分
- **结果聚合**：通过通信将各部分结果合并得到最终输出

## 🔄 张量并行 vs 数据并行

| 特征 | 数据并行 | 张量并行 |
|------|----------|----------|
| **分割对象** | 输入数据 | 模型权重 |
| **内存需求** | 每个GPU存储完整模型 | 每个GPU存储部分模型 |
| **通信模式** | AllReduce梯度 | AllReduce激活值 |
| **适用场景** | 模型较小，数据量大 | 模型很大，单GPU放不下 |
| **扩展性** | 受模型大小限制 | 可以支持超大模型 |

## 🏗️ 张量并行的实现方式

### 1. 行并行 (Row Parallelism)
将权重矩阵按行分割：

```python
# 原始计算: Y = XW (X: [batch, input_dim], W: [input_dim, output_dim])
# 行并行分割W为W1, W2, ..., Wn

class RowParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim_per_partition = output_dim // world_size
        self.weight = nn.Parameter(torch.randn(input_dim, self.output_dim_per_partition))
        
    def forward(self, x):
        # 每个GPU计算部分输出
        local_output = torch.matmul(x, self.weight)  # [batch, output_dim_per_partition]
        
        # AllGather收集所有GPU的输出
        output = all_gather(local_output)  # [batch, output_dim]
        return output
```

### 2. 列并行 (Column Parallelism)
将权重矩阵按列分割：

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        self.input_dim_per_partition = input_dim // world_size
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(self.input_dim_per_partition, output_dim))
        
    def forward(self, x):
        # 输入需要先分割
        local_x = x[:, self.rank * self.input_dim_per_partition:
                     (self.rank + 1) * self.input_dim_per_partition]
        
        # 每个GPU计算部分结果
        local_output = torch.matmul(local_x, self.weight)
        
        # AllReduce求和得到最终结果
        output = all_reduce(local_output)  # [batch, output_dim]
        return output
```

## 🧠 Transformer中的张量并行

### 1. 注意力机制的并行化

```python
class ParallelAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, world_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_heads_per_partition = num_heads // world_size
        
        # Q, K, V投影使用列并行
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, world_size
        )
        
        # 输出投影使用行并行
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, world_size
        )
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # 计算Q, K, V (每个GPU只计算部分头)
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * hidden_size_per_partition]
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        
        # 计算注意力 (每个GPU独立计算自己的头)
        attention_output = self.compute_attention(q, k, v)
        
        # 输出投影
        output = self.out_proj(attention_output)
        return output
```

### 2. MLP层的并行化

```python
class ParallelMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size):
        super().__init__()
        # 第一层使用列并行 (扩展维度)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size, 2 * intermediate_size, world_size
        )
        
        # 第二层使用行并行 (收缩维度)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size, world_size
        )
        
    def forward(self, x):
        # 第一层：hidden_size -> intermediate_size_per_partition
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        
        # 激活函数 (每个GPU独立计算)
        intermediate = F.silu(gate) * up
        
        # 第二层：intermediate_size_per_partition -> hidden_size
        output = self.down_proj(intermediate)
        return output
```

## 📊 nano-vllm中的张量并行实现

### 1. 并行配置

```python
class TensorParallelConfig:
    def __init__(self, tensor_parallel_size: int = 1):
        self.tensor_parallel_size = tensor_parallel_size
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        
    def is_tensor_parallel_enabled(self) -> bool:
        return self.tensor_parallel_size > 1
```

### 2. 权重分割策略

```python
def split_weight_for_tensor_parallel(weight: torch.Tensor, 
                                   dim: int, 
                                   rank: int, 
                                   world_size: int) -> torch.Tensor:
    """按指定维度分割权重张量"""
    size_per_partition = weight.size(dim) // world_size
    start_idx = rank * size_per_partition
    end_idx = (rank + 1) * size_per_partition
    
    if dim == 0:
        return weight[start_idx:end_idx, ...]
    elif dim == 1:
        return weight[:, start_idx:end_idx, ...]
    else:
        raise ValueError(f"Unsupported split dimension: {dim}")
```

### 3. 通信原语

```python
def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """所有GPU求和并广播结果"""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor
    
    torch.distributed.all_reduce(tensor, group=get_tensor_model_parallel_group())
    return tensor

def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """收集所有GPU的张量并拼接"""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor
    
    world_size = get_tensor_model_parallel_world_size()
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    
    torch.distributed.all_gather(gathered_tensors, tensor, 
                                group=get_tensor_model_parallel_group())
    
    return torch.cat(gathered_tensors, dim=dim)
```

## ⚡ 性能优化技巧

### 1. 通信优化
```python
class OptimizedTensorParallel:
    def __init__(self):
        # 重叠计算和通信
        self.async_comm = True
        # 使用高效的通信后端
        self.backend = "nccl"
        
    def forward_with_overlap(self, x):
        # 启动异步通信
        comm_handle = all_reduce_async(x)
        
        # 在通信进行时执行其他计算
        other_computation()
        
        # 等待通信完成
        result = comm_handle.wait()
        return result
```

### 2. 内存优化
```python
def optimize_memory_layout(weight: torch.Tensor) -> torch.Tensor:
    """优化权重的内存布局以提高访问效率"""
    # 确保权重是连续的
    if not weight.is_contiguous():
        weight = weight.contiguous()
    
    # 使用更高效的数据类型
    if weight.dtype == torch.float32:
        weight = weight.half()  # 转换为fp16
    
    return weight
```

### 3. 负载均衡
```python
def balance_tensor_parallel_load(hidden_size: int, world_size: int) -> int:
    """确保张量并行的负载均衡"""
    # 确保hidden_size能被world_size整除
    if hidden_size % world_size != 0:
        # 填充到最近的可整除大小
        padded_size = ((hidden_size + world_size - 1) // world_size) * world_size
        return padded_size
    return hidden_size
```

## 🎯 使用场景和最佳实践

### 1. 适用场景
- **超大模型**：单GPU内存无法容纳完整模型
- **高吞吐量需求**：需要同时服务多个请求
- **低延迟要求**：相比数据并行有更低的通信开销

### 2. 最佳实践
```python
# 1. 合理选择并行度
def choose_tensor_parallel_size(model_size_gb: float, gpu_memory_gb: float) -> int:
    """根据模型大小和GPU内存选择合适的并行度"""
    min_parallel_size = math.ceil(model_size_gb / (gpu_memory_gb * 0.8))  # 留20%余量
    return min_parallel_size

# 2. 优化通信拓扑
def setup_optimal_process_group():
    """设置最优的进程组拓扑"""
    # 优先使用同一节点内的GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 创建张量并行组
    for i in range(0, world_size, tensor_parallel_size):
        ranks = list(range(i, min(i + tensor_parallel_size, world_size)))
        group = torch.distributed.new_group(ranks)
        if local_rank in ranks:
            tensor_parallel_group = group
```

## 📈 性能分析

### 1. 理论分析
```python
def analyze_tensor_parallel_performance(model_params: int, 
                                      world_size: int,
                                      bandwidth_gbps: float) -> dict:
    """分析张量并行的理论性能"""
    
    # 计算量分析
    compute_per_gpu = model_params / world_size
    
    # 通信量分析 (主要是AllReduce)
    comm_volume = model_params * 4  # float32, 4 bytes per param
    comm_time = comm_volume / (bandwidth_gbps * 1e9)
    
    # 内存使用分析
    memory_per_gpu = model_params * 4 / world_size  # 权重内存
    
    return {
        "compute_per_gpu": compute_per_gpu,
        "communication_time": comm_time,
        "memory_per_gpu": memory_per_gpu,
        "efficiency": compute_per_gpu / (compute_per_gpu + comm_time)
    }
```

### 2. 实际测试
```python
def benchmark_tensor_parallel():
    """张量并行性能基准测试"""
    import time
    
    model = create_parallel_model()
    input_data = create_test_input()
    
    # 预热
    for _ in range(10):
        _ = model(input_data)
    
    # 测试
    start_time = time.time()
    for _ in range(100):
        output = model(input_data)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / 100
    throughput = input_data.size(0) / avg_latency  # samples per second
    
    print(f"Average latency: {avg_latency:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
```

## 🚀 总结

张量并行是实现大模型高效推理的关键技术：

1. **核心优势**
   - 突破单GPU内存限制
   - 提高计算并行度
   - 降低推理延迟

2. **实现要点**
   - 合理的权重分割策略
   - 高效的通信原语
   - 优化的内存管理

3. **性能关键**
   - 通信和计算的重叠
   - 负载均衡
   - 内存访问优化

在nano-vllm中，张量并行的实现简洁而高效，为理解和学习这一重要技术提供了很好的参考。

---

*接下来，让我们学习注意力机制的优化技术！*