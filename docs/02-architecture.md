# 🏗️ 架构设计

本章节深入分析 nano-vLLM 推理引擎的架构设计，帮助你理解系统的整体设计思路和各组件间的协作关系。

## 🎯 学习目标

通过本章学习，你将能够：

1. **理解整体架构** - 掌握 nano-vLLM 的整体设计思路和模块关系
2. **分析核心组件** - 深入理解每个核心组件的职责和交互机制
3. **掌握数据流程** - 了解数据在系统中的流转过程和处理机制
4. **理解性能设计** - 掌握性能相关的架构设计和优化策略

---

## 1. 整体架构设计

### 1.1 架构概览

nano-vLLM 采用分层架构设计，从上到下分为以下几个层次：

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│                     (FastAPI/HTTP)                          │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                            │
│              (Request Handler & Router)                     │
├─────────────────────────────────────────────────────────────┤
│                     Engine Layer                            │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│    │  Scheduler  │  │   Memory    │  │   Model     │       │
│    │             │  │   Manager   │  │   Manager   │       │
│    └─────────────┘  └─────────────┘  └─────────────┘       │
├─────────────────────────────────────────────────────────────┤
│                   Execution Layer                           │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│    │   Worker    │  │    Cache    │  │  Attention  │       │
│    │   Process   │  │   Manager   │  │   Kernel    │       │
│    └─────────────┘  └─────────────┘  └─────────────┘       │
├─────────────────────────────────────────────────────────────┤
│                   Hardware Layer                            │
│              (GPU/CPU/Memory/Network)                       │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

#### 模块化设计
- **单一职责**：每个模块专注于特定功能
- **松耦合**：模块间通过接口交互，降低依赖
- **高内聚**：相关功能集中在同一模块内

#### 可扩展性
- **水平扩展**：支持多 GPU、多节点部署
- **垂直扩展**：支持不同规模的模型和硬件
- **插件化**：支持自定义调度器、内存管理器等

#### 高性能
- **异步处理**：采用异步 I/O 和并发处理
- **内存优化**：高效的内存管理和缓存策略
- **计算优化**：充分利用硬件加速能力

### 1.3 核心模块关系

```python
# 模块依赖关系图
class ArchitectureDependencies:
    """
    API Layer
      ↓
    Service Layer (FastAPI Router)
      ↓
    Engine Layer
      ├── LLMEngine
      │   ├── Scheduler (调度器)
      │   ├── ModelExecutor (模型执行器)
      │   └── CacheEngine (缓存引擎)
      │
      ├── AsyncLLMEngine (异步引擎)
      │   └── LLMEngine
      │
      └── Worker (工作进程)
          ├── ModelRunner (模型运行器)
          ├── CacheEngine (缓存引擎)
          └── AttentionBackend (注意力后端)
    """
```

---

## 2. 核心组件分析

### 2.1 推理引擎 (LLMEngine)

推理引擎是系统的核心组件，负责协调各个子模块完成推理任务。

#### 主要职责
```python
class LLMEngine:
    def __init__(self, model_config, cache_config, parallel_config):
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        
        # 初始化核心组件
        self.scheduler = Scheduler(...)
        self.model_executor = ModelExecutor(...)
        self.cache_engine = CacheEngine(...)
    
    def add_request(self, request):
        """添加推理请求"""
        self.scheduler.add_request(request)
    
    def step(self):
        """执行一步推理"""
        # 1. 调度器选择要处理的请求
        scheduler_outputs = self.scheduler.schedule()
        
        # 2. 执行模型推理
        model_outputs = self.model_executor.execute_model(
            scheduler_outputs.scheduled_seq_groups
        )
        
        # 3. 更新缓存和状态
        self.cache_engine.update_cache(scheduler_outputs, model_outputs)
        
        # 4. 处理输出结果
        return self.process_outputs(model_outputs)
```

#### 生命周期管理
```python
class EngineLifecycle:
    def startup(self):
        """引擎启动流程"""
        # 1. 加载模型配置
        self.load_model_config()
        
        # 2. 初始化硬件资源
        self.initialize_hardware()
        
        # 3. 创建工作进程
        self.create_workers()
        
        # 4. 预热模型
        self.warmup_model()
    
    def shutdown(self):
        """引擎关闭流程"""
        # 1. 停止接收新请求
        self.stop_accepting_requests()
        
        # 2. 完成处理中的请求
        self.finish_pending_requests()
        
        # 3. 释放资源
        self.cleanup_resources()
```

### 2.2 请求调度器 (Scheduler)

调度器负责管理推理请求的生命周期，决定何时处理哪些请求。

#### 调度策略
```python
class Scheduler:
    def __init__(self, cache_config, parallel_config):
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        
        # 请求队列
        self.waiting_queue = []      # 等待队列
        self.running_queue = []      # 运行队列
        self.swapped_queue = []      # 交换队列
    
    def schedule(self):
        """核心调度逻辑"""
        scheduled_seq_groups = []
        
        # 1. 处理运行中的请求
        self._schedule_running(scheduled_seq_groups)
        
        # 2. 处理交换队列中的请求
        self._schedule_swapped(scheduled_seq_groups)
        
        # 3. 处理等待队列中的请求
        self._schedule_waiting(scheduled_seq_groups)
        
        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_batched_tokens=self._get_num_batched_tokens(),
            blocks_to_swap_in=self.blocks_to_swap_in,
            blocks_to_swap_out=self.blocks_to_swap_out,
            blocks_to_copy=self.blocks_to_copy
        )
```

#### 内存感知调度
```python
class MemoryAwareScheduler:
    def can_allocate(self, seq_group):
        """检查是否有足够内存分配给序列组"""
        num_required_blocks = self._get_num_required_blocks(seq_group)
        num_free_blocks = self.cache_engine.get_num_free_gpu_blocks()
        
        return num_free_blocks >= num_required_blocks
    
    def _schedule_waiting(self, scheduled_seq_groups):
        """调度等待队列中的请求"""
        while self.waiting_queue:
            seq_group = self.waiting_queue[0]
            
            # 检查内存是否足够
            if not self.can_allocate(seq_group):
                break
            
            # 分配内存块
            self._allocate_blocks(seq_group)
            
            # 移动到运行队列
            self.waiting_queue.pop(0)
            self.running_queue.append(seq_group)
            scheduled_seq_groups.append(seq_group)
```

### 2.3 内存管理器 (CacheEngine)

内存管理器负责 KV Cache 的分配、回收和优化。

#### 块级内存管理
```python
class CacheEngine:
    def __init__(self, cache_config, model_config, parallel_config):
        self.cache_config = cache_config
        self.model_config = model_config
        
        # 初始化 GPU 和 CPU 缓存
        self.gpu_cache = self._initialize_gpu_cache()
        self.cpu_cache = self._initialize_cpu_cache()
        
        # 块分配器
        self.block_allocator = BlockAllocator(
            num_gpu_blocks=cache_config.num_gpu_blocks,
            num_cpu_blocks=cache_config.num_cpu_blocks,
            block_size=cache_config.block_size
        )
    
    def allocate(self, seq_group):
        """为序列组分配内存块"""
        num_blocks = self._get_num_required_blocks(seq_group)
        blocks = self.block_allocator.allocate(num_blocks)
        
        # 记录分配信息
        for seq in seq_group.seqs:
            seq.logical_token_blocks = blocks
        
        return blocks
    
    def free(self, seq_group):
        """释放序列组的内存块"""
        for seq in seq_group.seqs:
            if seq.logical_token_blocks:
                self.block_allocator.free(seq.logical_token_blocks)
                seq.logical_token_blocks = None
```

#### 内存交换机制
```python
class SwapManager:
    def swap_out(self, seq_group):
        """将序列组从 GPU 交换到 CPU"""
        gpu_blocks = seq_group.get_gpu_blocks()
        cpu_blocks = self.allocate_cpu_blocks(len(gpu_blocks))
        
        # 执行数据传输
        self._copy_blocks(gpu_blocks, cpu_blocks, 'gpu_to_cpu')
        
        # 更新映射关系
        seq_group.set_cpu_blocks(cpu_blocks)
        self.free_gpu_blocks(gpu_blocks)
    
    def swap_in(self, seq_group):
        """将序列组从 CPU 交换到 GPU"""
        cpu_blocks = seq_group.get_cpu_blocks()
        gpu_blocks = self.allocate_gpu_blocks(len(cpu_blocks))
        
        # 执行数据传输
        self._copy_blocks(cpu_blocks, gpu_blocks, 'cpu_to_gpu')
        
        # 更新映射关系
        seq_group.set_gpu_blocks(gpu_blocks)
        self.free_cpu_blocks(cpu_blocks)
```

### 2.4 模型执行器 (ModelExecutor)

模型执行器负责实际的模型推理计算。

#### 并行执行
```python
class ModelExecutor:
    def __init__(self, model_config, cache_config, parallel_config):
        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        
        # 创建工作进程
        self.workers = self._create_workers()
        
        # 加载模型
        self._load_model()
    
    def execute_model(self, seq_group_metadata_list):
        """执行模型推理"""
        # 准备输入数据
        model_input = self._prepare_model_input(seq_group_metadata_list)
        
        # 分布式执行
        if self.parallel_config.tensor_parallel_size > 1:
            output = self._execute_distributed(model_input)
        else:
            output = self._execute_single(model_input)
        
        return output
    
    def _execute_distributed(self, model_input):
        """分布式执行"""
        # 广播输入到所有工作进程
        broadcast_tensor_dict(model_input, src=0)
        
        # 并行执行
        outputs = []
        for worker in self.workers:
            output = worker.execute_model(model_input)
            outputs.append(output)
        
        # 聚合结果
        return self._aggregate_outputs(outputs)
```

---

## 3. 数据流程分析

### 3.1 请求处理流程

```python
class RequestProcessingFlow:
    """
    完整的请求处理流程：
    
    1. API 接收请求
       ↓
    2. 请求验证和预处理
       ↓
    3. 添加到调度器队列
       ↓
    4. 调度器选择处理请求
       ↓
    5. 分配内存资源
       ↓
    6. 模型执行推理
       ↓
    7. 生成输出 Token
       ↓
    8. 更新缓存状态
       ↓
    9. 返回结果给客户端
    """
    
    async def process_request(self, request):
        # 1. 请求预处理
        processed_request = await self.preprocess_request(request)
        
        # 2. 添加到引擎
        request_id = await self.engine.add_request(processed_request)
        
        # 3. 异步等待结果
        async for output in self.engine.generate(request_id):
            yield output
```

### 3.2 数据传输路径

#### 内存层次结构
```
CPU Memory (Host)
      ↕ PCIe
GPU Memory (Device)
      ↕ High Bandwidth Memory
GPU Cache (L2/L1)
      ↕ Memory Bus
GPU Compute Units
```

#### 数据流向
```python
class DataFlow:
    def forward_pass(self, input_tokens):
        """前向传播数据流"""
        # 1. 输入 Token 嵌入
        embeddings = self.embedding_layer(input_tokens)
        
        # 2. 逐层 Transformer 计算
        hidden_states = embeddings
        for layer in self.transformer_layers:
            # 从 KV Cache 读取历史状态
            past_kv = self.kv_cache.get(layer.layer_id)
            
            # 计算当前层输出
            hidden_states, present_kv = layer(hidden_states, past_kv)
            
            # 更新 KV Cache
            self.kv_cache.update(layer.layer_id, present_kv)
        
        # 3. 输出层计算
        logits = self.output_layer(hidden_states)
        
        return logits
```

### 3.3 并发处理机制

#### 批处理策略
```python
class BatchProcessor:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.current_batch = []
    
    def add_to_batch(self, request):
        """添加请求到当前批次"""
        self.current_batch.append(request)
        
        # 检查是否需要执行批处理
        if len(self.current_batch) >= self.max_batch_size:
            return self.execute_batch()
        
        return None
    
    def execute_batch(self):
        """执行批处理"""
        if not self.current_batch:
            return None
        
        # 准备批处理输入
        batch_input = self.prepare_batch_input(self.current_batch)
        
        # 执行推理
        batch_output = self.model.forward(batch_input)
        
        # 分发结果
        results = self.distribute_results(batch_output, self.current_batch)
        
        # 清空当前批次
        self.current_batch = []
        
        return results
```

#### 流水线并行
```python
class PipelineParallel:
    def __init__(self, num_stages=4):
        self.num_stages = num_stages
        self.stages = [Stage(i) for i in range(num_stages)]
        self.pipeline_queue = Queue()
    
    async def process_pipeline(self, inputs):
        """流水线处理"""
        # 将输入分割成微批次
        micro_batches = self.split_into_micro_batches(inputs)
        
        # 启动流水线
        tasks = []
        for i, micro_batch in enumerate(micro_batches):
            task = asyncio.create_task(
                self.process_micro_batch(micro_batch, i)
            )
            tasks.append(task)
        
        # 等待所有微批次完成
        results = await asyncio.gather(*tasks)
        
        return self.merge_results(results)
    
    async def process_micro_batch(self, micro_batch, batch_id):
        """处理单个微批次"""
        current_data = micro_batch
        
        for stage_id, stage in enumerate(self.stages):
            # 等待前一个微批次在当前阶段完成
            await self.wait_for_stage(stage_id, batch_id - 1)
            
            # 在当前阶段处理数据
            current_data = await stage.process(current_data)
            
            # 标记当前阶段完成
            self.mark_stage_complete(stage_id, batch_id)
        
        return current_data
```

---

## 4. 接口设计分析

### 4.1 API 接口设计

#### RESTful API
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="nano-vLLM API")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    usage: dict

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """生成文本接口"""
    try:
        # 创建推理请求
        inference_request = create_inference_request(request)
        
        # 提交到引擎
        request_id = await engine.add_request(inference_request)
        
        # 获取结果
        if request.stream:
            return StreamingResponse(
                stream_generate(request_id),
                media_type="text/plain"
            )
        else:
            result = await engine.generate(request_id)
            return GenerateResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 流式接口
```python
async def stream_generate(request_id: str):
    """流式生成接口"""
    try:
        async for output in engine.generate_stream(request_id):
            # 格式化输出
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [{
                    "text": output.text,
                    "index": 0,
                    "finish_reason": output.finish_reason
                }]
            }
            
            # 发送 SSE 格式数据
            yield f"data: {json.dumps(chunk)}\n\n"
    
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    finally:
        yield "data: [DONE]\n\n"
```

### 4.2 内部接口规范

#### 组件间通信接口
```python
from abc import ABC, abstractmethod

class SchedulerInterface(ABC):
    """调度器接口"""
    
    @abstractmethod
    def add_request(self, request: Request) -> None:
        """添加请求到调度队列"""
        pass
    
    @abstractmethod
    def schedule(self) -> SchedulerOutputs:
        """执行调度逻辑"""
        pass
    
    @abstractmethod
    def free_finished_seq_groups(self) -> None:
        """释放已完成的序列组"""
        pass

class CacheEngineInterface(ABC):
    """缓存引擎接口"""
    
    @abstractmethod
    def allocate(self, seq_group: SequenceGroup) -> List[int]:
        """分配内存块"""
        pass
    
    @abstractmethod
    def free(self, seq_group: SequenceGroup) -> None:
        """释放内存块"""
        pass
    
    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup) -> None:
        """交换到 GPU"""
        pass
    
    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> None:
        """交换到 CPU"""
        pass
```

### 4.3 扩展接口机制

#### 插件系统
```python
class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.hooks = defaultdict(list)
    
    def register_plugin(self, name: str, plugin: Plugin):
        """注册插件"""
        self.plugins[name] = plugin
        
        # 注册钩子函数
        for hook_name in plugin.get_hooks():
            self.hooks[hook_name].append(plugin)
    
    def call_hook(self, hook_name: str, *args, **kwargs):
        """调用钩子函数"""
        results = []
        for plugin in self.hooks[hook_name]:
            result = plugin.call_hook(hook_name, *args, **kwargs)
            results.append(result)
        return results

class CustomSchedulerPlugin(Plugin):
    """自定义调度器插件"""
    
    def get_hooks(self):
        return ['before_schedule', 'after_schedule']
    
    def call_hook(self, hook_name, *args, **kwargs):
        if hook_name == 'before_schedule':
            return self.before_schedule(*args, **kwargs)
        elif hook_name == 'after_schedule':
            return self.after_schedule(*args, **kwargs)
    
    def before_schedule(self, scheduler_state):
        """调度前的自定义逻辑"""
        # 实现自定义调度策略
        pass
    
    def after_schedule(self, scheduler_outputs):
        """调度后的自定义逻辑"""
        # 实现调度结果后处理
        pass
```

---

## 5. 性能架构设计

### 5.1 并行处理架构

#### 多级并行策略
```python
class ParallelArchitecture:
    """
    多级并行处理架构：
    
    1. 数据并行 (Data Parallel)
       - 不同请求在不同设备上并行处理
    
    2. 张量并行 (Tensor Parallel)  
       - 单个模型的参数分布在多个设备上
    
    3. 流水线并行 (Pipeline Parallel)
       - 模型的不同层在不同设备上
    
    4. 序列并行 (Sequence Parallel)
       - 长序列分割到多个设备上处理
    """
    
    def __init__(self, parallel_config):
        self.data_parallel_size = parallel_config.data_parallel_size
        self.tensor_parallel_size = parallel_config.tensor_parallel_size
        self.pipeline_parallel_size = parallel_config.pipeline_parallel_size
        
        # 初始化通信组
        self.init_process_groups()
    
    def init_process_groups(self):
        """初始化进程组"""
        # 张量并行组
        self.tensor_parallel_group = torch.distributed.new_group(
            ranks=list(range(self.tensor_parallel_size))
        )
        
        # 流水线并行组
        self.pipeline_parallel_group = torch.distributed.new_group(
            ranks=list(range(self.pipeline_parallel_size))
        )
```

#### 负载均衡策略
```python
class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.worker_loads = {worker.id: 0 for worker in workers}
    
    def assign_request(self, request):
        """分配请求到负载最低的工作进程"""
        # 选择负载最低的工作进程
        min_load_worker = min(
            self.workers, 
            key=lambda w: self.worker_loads[w.id]
        )
        
        # 更新负载统计
        estimated_load = self.estimate_request_load(request)
        self.worker_loads[min_load_worker.id] += estimated_load
        
        return min_load_worker
    
    def estimate_request_load(self, request):
        """估算请求的计算负载"""
        # 基于输入长度和预期输出长度估算
        input_length = len(request.prompt_tokens)
        max_output_length = request.max_tokens
        
        # 简单的负载估算公式
        return input_length + max_output_length * 2
```

### 5.2 内存优化架构

#### 分层内存管理
```python
class HierarchicalMemoryManager:
    """
    分层内存管理架构：
    
    L1: GPU 寄存器 (最快，容量最小)
    L2: GPU 共享内存 (快，容量小)  
    L3: GPU 全局内存 (中等，容量中等)
    L4: CPU 内存 (慢，容量大)
    L5: 磁盘存储 (最慢，容量最大)
    """
    
    def __init__(self):
        self.gpu_memory_pool = GPUMemoryPool()
        self.cpu_memory_pool = CPUMemoryPool()
        self.disk_cache = DiskCache()
        
        # 内存使用统计
        self.memory_stats = MemoryStats()
    
    def allocate_kv_cache(self, seq_group, priority='high'):
        """分配 KV Cache 内存"""
        required_size = self.calculate_kv_cache_size(seq_group)
        
        # 根据优先级选择内存层级
        if priority == 'high':
            # 优先使用 GPU 内存
            if self.gpu_memory_pool.has_free_space(required_size):
                return self.gpu_memory_pool.allocate(required_size)
            else:
                # GPU 内存不足，使用 CPU 内存
                return self.cpu_memory_pool.allocate(required_size)
        else:
            # 低优先级直接使用 CPU 内存
            return self.cpu_memory_pool.allocate(required_size)
```

#### 内存压缩技术
```python
class MemoryCompression:
    def __init__(self):
        self.compression_ratio = 0.5  # 压缩比
        self.compression_threshold = 0.8  # 内存使用阈值
    
    def should_compress(self):
        """判断是否需要压缩"""
        memory_usage = self.get_memory_usage_ratio()
        return memory_usage > self.compression_threshold
    
    def compress_kv_cache(self, kv_cache):
        """压缩 KV Cache"""
        # 使用量化压缩
        compressed_k = self.quantize_tensor(kv_cache.k, bits=8)
        compressed_v = self.quantize_tensor(kv_cache.v, bits=8)
        
        return CompressedKVCache(compressed_k, compressed_v)
    
    def quantize_tensor(self, tensor, bits=8):
        """张量量化"""
        # 计算量化参数
        min_val = tensor.min()
        max_val = tensor.max()
        scale = (max_val - min_val) / (2**bits - 1)
        
        # 执行量化
        quantized = torch.round((tensor - min_val) / scale)
        quantized = torch.clamp(quantized, 0, 2**bits - 1)
        
        return QuantizedTensor(quantized, scale, min_val)
```

### 5.3 计算优化设计

#### 算子融合
```python
class OperatorFusion:
    """算子融合优化"""
    
    def fuse_attention_ops(self, q, k, v, mask=None):
        """融合注意力计算操作"""
        # 传统方式：多个独立操作
        # scores = torch.matmul(q, k.transpose(-2, -1))
        # scores = scores / math.sqrt(q.size(-1))
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, -1e9)
        # attn_weights = torch.softmax(scores, dim=-1)
        # output = torch.matmul(attn_weights, v)
        
        # 融合方式：单个 kernel 完成所有操作
        output = fused_attention_kernel(q, k, v, mask)
        return output
    
    def fuse_linear_ops(self, x, weights, biases):
        """融合多个线性层操作"""
        # 将多个小的矩阵乘法合并为一个大的矩阵乘法
        fused_weight = torch.cat(weights, dim=0)
        fused_bias = torch.cat(biases, dim=0)
        
        fused_output = torch.matmul(x, fused_weight.T) + fused_bias
        
        # 分割输出
        outputs = torch.split(fused_output, [w.size(0) for w in weights], dim=-1)
        return outputs
```

#### 动态批处理优化
```python
class DynamicBatching:
    def __init__(self, max_batch_size=32, max_wait_time=10):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.batch_timer = None
    
    async def add_request(self, request):
        """添加请求到动态批处理"""
        self.pending_requests.append(request)
        
        # 检查是否需要立即处理
        if len(self.pending_requests) >= self.max_batch_size:
            return await self.process_batch()
        
        # 设置定时器
        if self.batch_timer is None:
            self.batch_timer = asyncio.create_task(
                self.wait_and_process()
            )
        
        return None
    
    async def wait_and_process(self):
        """等待并处理批次"""
        await asyncio.sleep(self.max_wait_time / 1000)  # 转换为秒
        
        if self.pending_requests:
            await self.process_batch()
        
        self.batch_timer = None
    
    async def process_batch(self):
        """处理当前批次"""
        if not self.pending_requests:
            return
        
        # 获取当前批次
        current_batch = self.pending_requests[:]
        self.pending_requests.clear()
        
        # 取消定时器
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # 执行批处理
        return await self.execute_batch(current_batch)
```

---

## 📖 学习建议

### 💡 理解要点
1. **分层架构的优势**：模块化、可扩展、易维护
2. **组件协作机制**：接口设计、数据流转、状态管理
3. **性能优化策略**：并行处理、内存优化、计算融合
4. **扩展性设计**：插件机制、配置化、热更新

### 🔧 实践建议
1. **架构分析**：绘制系统架构图，理解模块关系
2. **代码追踪**：跟踪一个完整的请求处理流程
3. **性能测试**：测试不同配置下的性能表现
4. **扩展实验**：尝试实现自定义调度器或内存管理器

### 📚 延伸阅读
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)

---

通过深入理解架构设计，你已经掌握了 nano-vLLM 系统的核心设计思路。接下来可以学习 [代码分析](03-code-analysis.md)，深入了解具体的实现细节。