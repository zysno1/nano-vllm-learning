# 推理引擎分析

## 🚀 推理引擎概览

推理引擎是 nano-vllm 的核心组件，负责执行大语言模型的推理计算。它集成了模型加载、内存管理、注意力计算、文本生成等关键功能，为上层应用提供高效的推理服务。

## 🏗️ 核心架构

```python
import asyncio
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

from nano_vllm.models import ModelRegistry
from nano_vllm.attention import AttentionBackend
from nano_vllm.memory import MemoryManager, KVCache
from nano_vllm.scheduler import RequestScheduler
from nano_vllm.tokenizer import TokenizerWrapper
from nano_vllm.sampling import SamplingParams, SamplingEngine
from nano_vllm.config import ModelConfig, EngineConfig
from nano_vllm.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class GenerationRequest:
    """生成请求"""
    request_id: str
    prompt: str
    sampling_params: SamplingParams
    arrival_time: float
    priority: int = 0

@dataclass
class GenerationResult:
    """生成结果"""
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    finish_reason: str
    logprobs: Optional[List[Dict[str, float]]] = None

class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig):
        self.model_config = model_config
        self.engine_config = engine_config
        
        # 核心组件
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[TokenizerWrapper] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.attention_backend: Optional[AttentionBackend] = None
        self.scheduler: Optional[RequestScheduler] = None
        self.sampling_engine: Optional[SamplingEngine] = None
        
        # 状态管理
        self.is_initialized = False
        self.is_running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 性能监控
        self.metrics = MetricsCollector()
        
        # 并发控制
        self.request_queue = asyncio.Queue()
        self.result_futures: Dict[str, asyncio.Future] = {}
        self.processing_lock = asyncio.Lock()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=engine_config.max_worker_threads
        )
    
    async def initialize(self):
        """初始化推理引擎"""
        if self.is_initialized:
            return
        
        logger.info("Initializing inference engine...")
        
        try:
            # 1. 加载分词器
            await self._load_tokenizer()
            
            # 2. 初始化内存管理器
            await self._initialize_memory_manager()
            
            # 3. 加载模型
            await self._load_model()
            
            # 4. 初始化注意力后端
            await self._initialize_attention_backend()
            
            # 5. 初始化调度器
            await self._initialize_scheduler()
            
            # 6. 初始化采样引擎
            await self._initialize_sampling_engine()
            
            # 7. 启动处理循环
            await self._start_processing_loop()
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Inference engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            await self.shutdown()
            raise
    
    async def _load_tokenizer(self):
        """加载分词器"""
        logger.info("Loading tokenizer...")
        
        self.tokenizer = TokenizerWrapper(
            model_path=self.model_config.model_path,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        await self.tokenizer.initialize()
        
        logger.info(f"Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
    
    async def _initialize_memory_manager(self):
        """初始化内存管理器"""
        logger.info("Initializing memory manager...")
        
        self.memory_manager = MemoryManager(
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            swap_space=self.engine_config.swap_space,
            cpu_offload_gb=self.engine_config.cpu_offload_gb
        )
        
        await self.memory_manager.initialize()
        
        logger.info("Memory manager initialized")
    
    async def _load_model(self):
        """加载模型"""
        logger.info("Loading model...")
        
        # 从模型注册表获取模型类
        model_class = ModelRegistry.get_model_class(self.model_config.model_type)
        
        # 创建模型实例
        self.model = model_class(
            config=self.model_config,
            device=self.device,
            dtype=getattr(torch, self.model_config.dtype)
        )
        
        # 加载权重
        await self.model.load_weights(self.model_config.model_path)
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 编译模型（如果启用）
        if self.engine_config.enable_torch_compile:
            self.model = torch.compile(self.model)
        
        logger.info(f"Model loaded: {self.model_config.model_type}")
    
    async def _initialize_attention_backend(self):
        """初始化注意力后端"""
        logger.info("Initializing attention backend...")
        
        self.attention_backend = AttentionBackend(
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            scale=1.0 / (self.model.config.hidden_size // self.model.config.num_attention_heads) ** 0.5,
            num_kv_heads=getattr(self.model.config, 'num_key_value_heads', self.model.config.num_attention_heads),
            sliding_window=getattr(self.model.config, 'sliding_window', None),
            dtype=getattr(torch, self.model_config.dtype),
            device=self.device
        )
        
        await self.attention_backend.initialize()
        
        logger.info("Attention backend initialized")
    
    async def _initialize_scheduler(self):
        """初始化调度器"""
        logger.info("Initializing scheduler...")
        
        self.scheduler = RequestScheduler(
            max_num_seqs=self.engine_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.engine_config.max_num_batched_tokens,
            memory_manager=self.memory_manager
        )
        
        await self.scheduler.initialize()
        
        logger.info("Scheduler initialized")
    
    async def _initialize_sampling_engine(self):
        """初始化采样引擎"""
        logger.info("Initializing sampling engine...")
        
        self.sampling_engine = SamplingEngine(
            vocab_size=self.tokenizer.vocab_size,
            device=self.device,
            dtype=getattr(torch, self.model_config.dtype)
        )
        
        await self.sampling_engine.initialize()
        
        logger.info("Sampling engine initialized")
    
    async def _start_processing_loop(self):
        """启动处理循环"""
        logger.info("Starting processing loop...")
        
        # 启动主处理循环
        asyncio.create_task(self._main_processing_loop())
        
        logger.info("Processing loop started")
    
    async def _main_processing_loop(self):
        """主处理循环"""
        while self.is_running:
            try:
                # 获取调度决策
                scheduler_output = await self.scheduler.schedule()
                
                if scheduler_output.scheduled_seq_groups:
                    # 执行推理步骤
                    await self._execute_inference_step(scheduler_output)
                else:
                    # 没有任务时短暂休眠
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_inference_step(self, scheduler_output):
        """执行推理步骤"""
        start_time = time.time()
        
        try:
            # 1. 准备输入数据
            model_input = await self._prepare_model_input(scheduler_output)
            
            # 2. 执行前向传播
            with torch.no_grad():
                model_output = await self._forward_pass(model_input)
            
            # 3. 执行采样
            sampling_output = await self._execute_sampling(model_output, scheduler_output)
            
            # 4. 更新序列状态
            await self._update_sequences(sampling_output, scheduler_output)
            
            # 5. 处理完成的序列
            await self._process_finished_sequences(scheduler_output)
            
            # 记录性能指标
            step_time = time.time() - start_time
            self.metrics.record_inference_step(
                batch_size=len(scheduler_output.scheduled_seq_groups),
                step_time=step_time,
                num_tokens=sum(len(seq_group.seqs) for seq_group in scheduler_output.scheduled_seq_groups)
            )
            
        except Exception as e:
            logger.error(f"Error in inference step: {e}")
            # 处理错误的序列
            await self._handle_inference_error(scheduler_output, str(e))
    
    async def _prepare_model_input(self, scheduler_output) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        # 收集所有序列的token IDs
        input_ids = []
        position_ids = []
        attention_mask = []
        
        for seq_group in scheduler_output.scheduled_seq_groups:
            for seq in seq_group.seqs:
                input_ids.append(seq.get_token_ids())
                position_ids.append(seq.get_position_ids())
                attention_mask.append(seq.get_attention_mask())
        
        # 批处理和填充
        max_len = max(len(ids) for ids in input_ids)
        
        batched_input_ids = torch.zeros(
            (len(input_ids), max_len), 
            dtype=torch.long, 
            device=self.device
        )
        
        batched_position_ids = torch.zeros(
            (len(input_ids), max_len), 
            dtype=torch.long, 
            device=self.device
        )
        
        batched_attention_mask = torch.zeros(
            (len(input_ids), max_len), 
            dtype=torch.bool, 
            device=self.device
        )
        
        for i, (ids, pos_ids, mask) in enumerate(zip(input_ids, position_ids, attention_mask)):
            seq_len = len(ids)
            batched_input_ids[i, :seq_len] = torch.tensor(ids, device=self.device)
            batched_position_ids[i, :seq_len] = torch.tensor(pos_ids, device=self.device)
            batched_attention_mask[i, :seq_len] = torch.tensor(mask, device=self.device)
        
        return {
            'input_ids': batched_input_ids,
            'position_ids': batched_position_ids,
            'attention_mask': batched_attention_mask,
            'kv_caches': scheduler_output.kv_caches
        }
    
    async def _forward_pass(self, model_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """执行前向传播"""
        # 在线程池中执行计算密集型操作
        def _forward():
            return self.model(
                input_ids=model_input['input_ids'],
                position_ids=model_input['position_ids'],
                attention_mask=model_input['attention_mask'],
                kv_caches=model_input['kv_caches']
            )
        
        # 异步执行前向传播
        loop = asyncio.get_event_loop()
        model_output = await loop.run_in_executor(self.thread_pool, _forward)
        
        return model_output
    
    async def _execute_sampling(self, model_output: Dict[str, torch.Tensor], scheduler_output) -> Dict[str, Any]:
        """执行采样"""
        logits = model_output['logits']  # [batch_size, seq_len, vocab_size]
        
        # 获取最后一个位置的logits
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # 收集采样参数
        sampling_params_list = []
        for seq_group in scheduler_output.scheduled_seq_groups:
            sampling_params_list.append(seq_group.sampling_params)
        
        # 执行采样
        sampling_output = await self.sampling_engine.sample(
            logits=next_token_logits,
            sampling_params_list=sampling_params_list
        )
        
        return sampling_output
    
    async def _update_sequences(self, sampling_output: Dict[str, Any], scheduler_output):
        """更新序列状态"""
        next_tokens = sampling_output['next_tokens']  # [batch_size]
        
        seq_idx = 0
        for seq_group in scheduler_output.scheduled_seq_groups:
            for seq in seq_group.seqs:
                next_token = next_tokens[seq_idx].item()
                
                # 添加新token
                seq.append_token(next_token)
                
                # 检查是否完成
                if self._is_sequence_finished(seq, seq_group.sampling_params):
                    seq.set_finished()
                
                seq_idx += 1
    
    def _is_sequence_finished(self, seq, sampling_params: SamplingParams) -> bool:
        """检查序列是否完成"""
        # 检查最大长度
        if len(seq.token_ids) >= sampling_params.max_tokens:
            return True
        
        # 检查停止词
        if sampling_params.stop:
            text = self.tokenizer.decode(seq.token_ids)
            for stop_word in sampling_params.stop:
                if stop_word in text:
                    return True
        
        # 检查EOS token
        if seq.token_ids[-1] == self.tokenizer.eos_token_id:
            return True
        
        return False
    
    async def _process_finished_sequences(self, scheduler_output):
        """处理完成的序列"""
        for seq_group in scheduler_output.scheduled_seq_groups:
            if seq_group.is_finished():
                # 生成结果
                result = await self._create_generation_result(seq_group)
                
                # 返回结果
                request_id = seq_group.request_id
                if request_id in self.result_futures:
                    future = self.result_futures.pop(request_id)
                    if not future.done():
                        future.set_result(result)
                
                # 释放资源
                await self.scheduler.free_sequence_group(seq_group)
    
    async def _create_generation_result(self, seq_group) -> GenerationResult:
        """创建生成结果"""
        # 获取最佳序列（通常是第一个）
        best_seq = seq_group.seqs[0]
        
        # 解码文本
        prompt_tokens = seq_group.prompt_token_ids
        completion_tokens = best_seq.token_ids[len(prompt_tokens):]
        
        prompt_text = self.tokenizer.decode(prompt_tokens)
        completion_text = self.tokenizer.decode(completion_tokens)
        
        # 计算生成时间
        generation_time = time.time() - seq_group.arrival_time
        
        return GenerationResult(
            request_id=seq_group.request_id,
            text=completion_text,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(completion_tokens),
            total_tokens=len(prompt_tokens) + len(completion_tokens),
            generation_time=generation_time,
            finish_reason=best_seq.finish_reason or "length"
        )
    
    async def _handle_inference_error(self, scheduler_output, error_message: str):
        """处理推理错误"""
        for seq_group in scheduler_output.scheduled_seq_groups:
            request_id = seq_group.request_id
            if request_id in self.result_futures:
                future = self.result_futures.pop(request_id)
                if not future.done():
                    future.set_exception(RuntimeError(f"Inference error: {error_message}"))
            
            # 释放资源
            await self.scheduler.free_sequence_group(seq_group)
    
    async def generate(
        self, 
        prompt: str, 
        sampling_params: Optional[SamplingParams] = None,
        priority: int = 0
    ) -> GenerationResult:
        """生成文本"""
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        # 创建默认采样参数
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # 创建请求
        request_id = f"req_{int(time.time() * 1000000)}"
        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            arrival_time=time.time(),
            priority=priority
        )
        
        # 创建结果Future
        future = asyncio.Future()
        self.result_futures[request_id] = future
        
        try:
            # 添加到调度器
            await self.scheduler.add_request(request)
            
            # 等待结果
            result = await future
            
            # 更新指标
            self.metrics.record_request(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                generation_time=result.generation_time
            )
            
            return result
            
        except Exception as e:
            # 清理Future
            if request_id in self.result_futures:
                self.result_futures.pop(request_id)
            raise
    
    async def generate_stream(
        self, 
        prompt: str, 
        sampling_params: Optional[SamplingParams] = None,
        priority: int = 0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式生成文本"""
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        # 创建默认采样参数
        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # 启用流式输出
        sampling_params.stream = True
        
        # 创建请求
        request_id = f"req_{int(time.time() * 1000000)}"
        request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            arrival_time=time.time(),
            priority=priority
        )
        
        # 创建流式队列
        stream_queue = asyncio.Queue()
        self.result_futures[request_id] = stream_queue
        
        try:
            # 添加到调度器
            await self.scheduler.add_request(request)
            
            # 流式返回结果
            while True:
                try:
                    chunk = await asyncio.wait_for(stream_queue.get(), timeout=30.0)
                    
                    if chunk is None:  # 结束标志
                        break
                    
                    yield chunk
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Stream timeout for request {request_id}")
                    break
                    
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise
        finally:
            # 清理资源
            if request_id in self.result_futures:
                self.result_futures.pop(request_id)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized")
        
        return {
            'model_type': self.model_config.model_type,
            'model_path': self.model_config.model_path,
            'vocab_size': self.tokenizer.vocab_size,
            'max_model_len': self.model_config.max_model_len,
            'tensor_parallel_size': self.model_config.tensor_parallel_size,
            'dtype': self.model_config.dtype,
            'device': str(self.device)
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.is_initialized:
            return {}
        
        engine_metrics = self.metrics.get_metrics()
        memory_metrics = await self.memory_manager.get_metrics()
        scheduler_metrics = await self.scheduler.get_metrics()
        
        return {
            'engine': engine_metrics,
            'memory': memory_metrics,
            'scheduler': scheduler_metrics,
            'timestamp': time.time()
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'active_requests': len(self.result_futures),
            'device': str(self.device),
            'memory_usage': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        }
    
    async def shutdown(self):
        """关闭推理引擎"""
        logger.info("Shutting down inference engine...")
        
        self.is_running = False
        
        # 取消所有待处理的请求
        for future in self.result_futures.values():
            if not future.done():
                future.set_exception(RuntimeError("Engine shutdown"))
        
        self.result_futures.clear()
        
        # 关闭组件
        if self.scheduler:
            await self.scheduler.shutdown()
        
        if self.memory_manager:
            await self.memory_manager.shutdown()
        
        if self.attention_backend:
            await self.attention_backend.shutdown()
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        
        logger.info("Inference engine shutdown complete")

# 工厂函数
async def create_inference_engine(
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    **kwargs
) -> InferenceEngine:
    """创建推理引擎"""
    
    model_config = ModelConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs
    )
    
    engine_config = EngineConfig()
    
    engine = InferenceEngine(model_config, engine_config)
    await engine.initialize()
    
    return engine

# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建推理引擎
    engine = await create_inference_engine(
        model_path="/path/to/model",
        tensor_parallel_size=1,
        max_model_len=2048
    )
    
    try:
        # 单次生成
        result = await engine.generate(
            prompt="Hello, how are you?",
            sampling_params=SamplingParams(
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
        )
        
        print(f"Generated text: {result.text}")
        print(f"Tokens: {result.total_tokens}")
        print(f"Time: {result.generation_time:.2f}s")
        
        # 流式生成
        print("\nStreaming generation:")
        async for chunk in engine.generate_stream(
            prompt="Tell me a story about",
            sampling_params=SamplingParams(max_tokens=200)
        ):
            print(chunk.get('text', ''), end='', flush=True)
        
        print("\n")
        
        # 获取指标
        metrics = await engine.get_metrics()
        print(f"Engine metrics: {metrics}")
        
    finally:
        # 关闭引擎
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(example_usage())
```

## 🔧 关键特性分析

### 1. 异步架构

- **异步初始化**：支持异步组件初始化
- **并发处理**：支持多请求并发处理
- **非阻塞I/O**：避免阻塞主线程

### 2. 内存优化

- **KV缓存管理**：高效的键值缓存机制
- **内存池**：预分配内存池减少碎片
- **动态调度**：根据内存使用动态调度请求

### 3. 性能监控

- **实时指标**：实时收集性能指标
- **资源监控**：监控GPU/CPU/内存使用
- **请求追踪**：追踪每个请求的生命周期

### 4. 错误处理

- **异常捕获**：全面的异常处理机制
- **资源清理**：确保资源正确释放
- **故障恢复**：支持从错误中恢复

## 📊 性能优化技术

### 批处理优化

```python
class BatchOptimizer:
    """批处理优化器"""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.batch_timeout = 0.01  # 10ms
    
    async def optimize_batch(self, requests: List[GenerationRequest]) -> List[List[GenerationRequest]]:
        """优化批处理"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        for request in requests:
            request_tokens = len(self.tokenizer.encode(request.prompt))
            
            # 检查是否可以加入当前批次
            if (len(current_batch) < self.max_batch_size and 
                current_tokens + request_tokens <= self.max_batch_tokens):
                current_batch.append(request)
                current_tokens += request_tokens
            else:
                # 开始新批次
                if current_batch:
                    batches.append(current_batch)
                current_batch = [request]
                current_tokens = request_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
```

### 内存预分配

```python
class MemoryPreallocator:
    """内存预分配器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.preallocated_tensors = {}
    
    def preallocate_tensors(self, max_batch_size: int, max_seq_len: int, hidden_size: int):
        """预分配张量"""
        self.preallocated_tensors.update({
            'input_ids': torch.zeros(
                (max_batch_size, max_seq_len), 
                dtype=torch.long, 
                device=self.device
            ),
            'attention_mask': torch.zeros(
                (max_batch_size, max_seq_len), 
                dtype=torch.bool, 
                device=self.device
            ),
            'hidden_states': torch.zeros(
                (max_batch_size, max_seq_len, hidden_size), 
                dtype=torch.float16, 
                device=self.device
            )
        })
    
    def get_tensor(self, name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """获取预分配的张量"""
        if name in self.preallocated_tensors:
            tensor = self.preallocated_tensors[name]
            if tensor.shape[:len(shape)] == shape:
                return tensor[:shape[0], :shape[1]] if len(shape) == 2 else tensor
        
        # 如果没有合适的预分配张量，创建新的
        return torch.zeros(shape, device=self.device)
```

## 🚀 使用最佳实践

### 1. 引擎配置优化

```python
# 生产环境配置
production_config = ModelConfig(
    model_path="/path/to/model",
    tensor_parallel_size=4,  # 根据GPU数量调整
    max_model_len=4096,      # 根据需求调整
    gpu_memory_utilization=0.85,  # 留出一些内存余量
    dtype="float16",         # 使用半精度节省内存
    trust_remote_code=False  # 安全考虑
)

engine_config = EngineConfig(
    max_num_seqs=256,        # 最大并发序列数
    max_num_batched_tokens=8192,  # 最大批处理token数
    max_worker_threads=8,    # 工作线程数
    enable_torch_compile=True,    # 启用编译优化
    swap_space=4             # 4GB交换空间
)
```

### 2. 请求处理优化

```python
# 优化采样参数
optimal_sampling = SamplingParams(
    temperature=0.7,         # 平衡创造性和一致性
    top_p=0.9,              # 核采样
    top_k=50,               # 限制候选token数
    repetition_penalty=1.1,  # 避免重复
    max_tokens=512,         # 合理的最大长度
    stop=["</s>", "\n\n"]   # 适当的停止条件
)
```

### 3. 监控和调试

```python
# 性能监控
async def monitor_performance(engine: InferenceEngine):
    """监控引擎性能"""
    while True:
        metrics = await engine.get_metrics()
        
        # 检查关键指标
        if metrics['engine']['avg_latency'] > 5.0:
            logger.warning("High latency detected")
        
        if metrics['memory']['gpu_utilization'] > 0.95:
            logger.warning("High GPU memory usage")
        
        await asyncio.sleep(10)  # 每10秒检查一次
```

---

*推理引擎是 nano-vllm 的核心，通过深入理解其架构和实现，你可以更好地优化和扩展系统功能。*