# 核心组件详解

## 🎯 组件概览

nano-vllm 的核心组件设计遵循单一职责原则，每个组件专注于特定的功能领域，通过清晰的接口进行协作。

## 🚀 推理引擎 (Inference Engine)

### 设计目标
- 高效执行模型前向推理
- 支持多种并行策略
- 优化内存和计算资源使用
- 提供灵活的生成控制

### 核心架构

```python
class InferenceEngine:
    """推理引擎核心实现"""
    
    def __init__(self, model_config, device_config):
        self.model_config = model_config
        self.device_config = device_config
        
        # 核心组件初始化
        self.model_executor = ModelExecutor(model_config)
        self.attention_backend = self._init_attention_backend()
        self.generation_sampler = GenerationSampler()
        self.parallel_context = ParallelContext(device_config)
        
        # 性能优化组件
        self.kernel_cache = KernelCache()
        self.memory_optimizer = MemoryOptimizer()
        
    def _init_attention_backend(self):
        """初始化注意力计算后端"""
        if self.device_config.supports_flash_attention:
            return FlashAttentionBackend()
        elif self.device_config.supports_xformers:
            return XFormersBackend()
        else:
            return StandardAttentionBackend()
    
    async def execute_batch(self, batch_request):
        """执行批量推理"""
        try:
            # 1. 预处理和内存分配
            execution_context = await self._prepare_execution(batch_request)
            
            # 2. 模型前向推理
            model_outputs = await self._forward_pass(
                execution_context, batch_request
            )
            
            # 3. 生成和采样
            generated_tokens = await self._generation_step(
                model_outputs, batch_request.generation_configs
            )
            
            # 4. 后处理和清理
            results = await self._postprocess_results(
                generated_tokens, execution_context
            )
            
            return results
            
        except Exception as e:
            await self._handle_execution_error(e, batch_request)
            raise
        finally:
            await self._cleanup_execution(execution_context)
    
    async def _prepare_execution(self, batch_request):
        """准备执行环境"""
        context = ExecutionContext()
        
        # 内存分配
        context.memory_allocation = await self.memory_optimizer.allocate_for_batch(
            batch_request
        )
        
        # KV Cache准备
        context.kv_cache_blocks = await self._allocate_kv_cache(batch_request)
        
        # 并行上下文设置
        context.parallel_state = self.parallel_context.create_state(batch_request)
        
        return context
    
    async def _forward_pass(self, context, batch_request):
        """模型前向推理"""
        # 输入嵌入
        input_embeddings = self.model_executor.embed_tokens(
            batch_request.input_tokens
        )
        
        # Transformer层推理
        hidden_states = input_embeddings
        for layer_idx, transformer_layer in enumerate(self.model_executor.layers):
            # 注意力计算
            attention_output = await self._compute_attention(
                transformer_layer.attention,
                hidden_states,
                context.kv_cache_blocks[layer_idx],
                context.parallel_state
            )
            
            # MLP计算
            mlp_output = transformer_layer.mlp(attention_output)
            
            # 残差连接和层归一化
            hidden_states = transformer_layer.post_attention_layernorm(
                attention_output + hidden_states
            )
            hidden_states = transformer_layer.post_mlp_layernorm(
                mlp_output + hidden_states
            )
        
        # 输出投影
        logits = self.model_executor.lm_head(hidden_states)
        
        return ModelOutputs(
            logits=logits,
            hidden_states=hidden_states,
            attention_weights=None  # 推理时通常不需要
        )
    
    async def _compute_attention(self, attention_layer, hidden_states, 
                                kv_cache, parallel_state):
        """计算注意力"""
        # 选择最优的注意力实现
        if self._should_use_flash_attention(hidden_states.shape):
            return await self._flash_attention_compute(
                attention_layer, hidden_states, kv_cache, parallel_state
            )
        else:
            return await self._standard_attention_compute(
                attention_layer, hidden_states, kv_cache, parallel_state
            )
    
    async def _flash_attention_compute(self, attention_layer, hidden_states,
                                     kv_cache, parallel_state):
        """Flash Attention计算"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # QKV投影
        qkv = attention_layer.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.model_config.num_attention_heads, -1)
        k = k.view(batch_size, seq_len, self.model_config.num_key_value_heads, -1)
        v = v.view(batch_size, seq_len, self.model_config.num_key_value_heads, -1)
        
        # KV Cache更新
        k, v = kv_cache.update(k, v)
        
        # Flash Attention计算
        attention_output = self.attention_backend.flash_attention(
            q, k, v, 
            causal=True,
            softmax_scale=1.0 / math.sqrt(q.size(-1))
        )
        
        # 输出投影
        output = attention_layer.o_proj(
            attention_output.view(batch_size, seq_len, hidden_size)
        )
        
        return output
```

### 性能优化特性

```python
class PerformanceOptimizer:
    """推理引擎性能优化器"""
    
    def __init__(self, engine_config):
        self.config = engine_config
        self.profiler = InferenceProfiler()
        self.adaptive_scheduler = AdaptiveScheduler()
        
    def optimize_batch_execution(self, batch_request):
        """优化批次执行策略"""
        # 1. 批次大小优化
        optimal_batch_size = self._calculate_optimal_batch_size(batch_request)
        
        # 2. 序列长度分组
        grouped_requests = self._group_by_sequence_length(batch_request)
        
        # 3. 内存布局优化
        memory_layout = self._optimize_memory_layout(grouped_requests)
        
        return OptimizedBatchPlan(
            batch_size=optimal_batch_size,
            grouped_requests=grouped_requests,
            memory_layout=memory_layout
        )
    
    def _calculate_optimal_batch_size(self, batch_request):
        """计算最优批次大小"""
        # 基于GPU内存和计算能力的动态计算
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory
        
        # 估算单个请求的内存需求
        avg_seq_len = sum(len(req.tokens) for req in batch_request.requests) / len(batch_request.requests)
        memory_per_request = self._estimate_memory_per_request(avg_seq_len)
        
        # 计算最大可支持的批次大小
        max_batch_size = int(free_memory * 0.8 / memory_per_request)  # 留20%余量
        
        # 考虑计算效率的最优批次大小
        compute_optimal = self._get_compute_optimal_batch_size()
        
        return min(max_batch_size, compute_optimal, self.config.max_batch_size)
    
    def _estimate_memory_per_request(self, seq_len):
        """估算单个请求的内存需求"""
        # 模型参数内存（共享）
        model_memory = 0  # 在批处理中共享
        
        # KV Cache内存
        kv_cache_memory = (
            seq_len * self.config.num_layers * 
            self.config.hidden_size * 2 * 2  # K+V, FP16
        )
        
        # 激活值内存
        activation_memory = seq_len * self.config.hidden_size * 4  # 估算
        
        return kv_cache_memory + activation_memory
```

## 📋 请求调度器 (Request Scheduler)

### 设计目标
- 智能的请求队列管理
- 动态批处理优化
- 公平性和优先级调度
- 负载均衡和资源利用最大化

### 核心实现

```python
class RequestScheduler:
    """请求调度器"""
    
    def __init__(self, scheduler_config):
        self.config = scheduler_config
        
        # 队列管理
        self.pending_queue = PriorityQueue()
        self.running_batches = {}
        self.completed_requests = {}
        
        # 调度策略
        self.batch_scheduler = BatchScheduler(scheduler_config)
        self.priority_manager = PriorityManager()
        self.load_balancer = LoadBalancer()
        
        # 性能监控
        self.metrics_collector = MetricsCollector()
        self.adaptive_tuner = AdaptiveTuner()
        
    async def schedule_request(self, request):
        """调度单个请求"""
        # 1. 请求预处理
        processed_request = await self._preprocess_request(request)
        
        # 2. 优先级计算
        priority = self.priority_manager.calculate_priority(processed_request)
        processed_request.priority = priority
        
        # 3. 加入队列
        await self.pending_queue.put((priority, processed_request))
        
        # 4. 触发调度决策
        await self._trigger_scheduling_decision()
        
        # 5. 等待结果
        return await self._wait_for_completion(processed_request.request_id)
    
    async def _trigger_scheduling_decision(self):
        """触发调度决策"""
        # 检查是否应该创建新批次
        if await self._should_create_batch():
            batch = await self._create_optimal_batch()
            if batch:
                await self._execute_batch(batch)
    
    async def _should_create_batch(self):
        """判断是否应该创建批次"""
        # 1. 队列长度检查
        if self.pending_queue.qsize() >= self.config.min_batch_size:
            return True
        
        # 2. 等待时间检查
        if self.pending_queue.qsize() > 0:
            oldest_request = await self.pending_queue.peek()
            wait_time = time.time() - oldest_request[1].arrival_time
            if wait_time > self.config.max_wait_time:
                return True
        
        # 3. 资源利用率检查
        if self._get_resource_utilization() < self.config.min_utilization:
            return self.pending_queue.qsize() > 0
        
        return False
    
    async def _create_optimal_batch(self):
        """创建最优批次"""
        batch_requests = []
        total_tokens = 0
        max_seq_len = 0
        
        # 从队列中选择请求
        while (len(batch_requests) < self.config.max_batch_size and 
               not self.pending_queue.empty()):
            
            priority, request = await self.pending_queue.get()
            
            # 检查是否可以加入当前批次
            if self._can_add_to_batch(request, batch_requests, total_tokens):
                batch_requests.append(request)
                total_tokens += len(request.tokens)
                max_seq_len = max(max_seq_len, len(request.tokens))
            else:
                # 放回队列
                await self.pending_queue.put((priority, request))
                break
        
        if batch_requests:
            return BatchRequest(
                requests=batch_requests,
                batch_id=self._generate_batch_id(),
                total_tokens=total_tokens,
                max_sequence_length=max_seq_len
            )
        
        return None
    
    def _can_add_to_batch(self, request, current_batch, current_tokens):
        """检查请求是否可以加入当前批次"""
        # 1. 令牌数量限制
        if current_tokens + len(request.tokens) > self.config.max_tokens_per_batch:
            return False
        
        # 2. 序列长度兼容性
        if current_batch:
            max_current_len = max(len(req.tokens) for req in current_batch)
            if abs(len(request.tokens) - max_current_len) > self.config.max_length_diff:
                return False
        
        # 3. 生成配置兼容性
        if current_batch:
            if not self._compatible_generation_configs(request, current_batch[0]):
                return False
        
        return True
    
    async def _execute_batch(self, batch):
        """执行批次"""
        batch_id = batch.batch_id
        self.running_batches[batch_id] = batch
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 执行推理
            results = await self.inference_engine.execute_batch(batch)
            
            # 记录完成时间
            end_time = time.time()
            
            # 更新指标
            self.metrics_collector.record_batch_completion(
                batch_id, end_time - start_time, len(batch.requests)
            )
            
            # 处理结果
            await self._process_batch_results(batch, results)
            
        except Exception as e:
            await self._handle_batch_error(batch, e)
        finally:
            # 清理
            if batch_id in self.running_batches:
                del self.running_batches[batch_id]
```

### 调度策略

```python
class SchedulingStrategies:
    """调度策略集合"""
    
    @staticmethod
    def fair_share_scheduling(requests, available_resources):
        """公平共享调度"""
        # 按用户分组
        user_groups = {}
        for request in requests:
            user_id = request.user_id
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(request)
        
        # 计算每个用户的资源配额
        num_users = len(user_groups)
        resource_per_user = available_resources / num_users
        
        # 为每个用户分配资源
        scheduled_requests = []
        for user_id, user_requests in user_groups.items():
            user_allocation = min(len(user_requests), resource_per_user)
            scheduled_requests.extend(user_requests[:int(user_allocation)])
        
        return scheduled_requests
    
    @staticmethod
    def shortest_job_first(requests):
        """最短作业优先调度"""
        return sorted(requests, key=lambda r: r.estimated_completion_time)
    
    @staticmethod
    def priority_based_scheduling(requests):
        """基于优先级的调度"""
        return sorted(requests, key=lambda r: r.priority, reverse=True)
    
    @staticmethod
    def adaptive_scheduling(requests, system_state):
        """自适应调度"""
        # 根据系统状态选择调度策略
        if system_state.cpu_utilization > 0.8:
            # 高CPU使用率时，优先短任务
            return SchedulingStrategies.shortest_job_first(requests)
        elif system_state.memory_utilization > 0.8:
            # 高内存使用率时，优先小内存任务
            return sorted(requests, key=lambda r: r.estimated_memory_usage)
        else:
            # 正常情况下，公平调度
            return SchedulingStrategies.fair_share_scheduling(
                requests, system_state.available_resources
            )
```

## 💾 内存管理器 (Memory Manager)

### 设计目标
- 高效的GPU/CPU内存分配
- 智能的KV Cache管理
- 内存碎片最小化
- 动态内存回收

### 核心架构

```python
class MemoryManager:
    """统一内存管理器"""
    
    def __init__(self, memory_config):
        self.config = memory_config
        
        # 内存池管理
        self.gpu_memory_pool = GPUMemoryPool(memory_config.gpu_memory_gb)
        self.cpu_memory_pool = CPUMemoryPool(memory_config.cpu_memory_gb)
        
        # 专用管理器
        self.kv_cache_manager = KVCacheManager(memory_config.kv_cache_config)
        self.activation_manager = ActivationMemoryManager()
        self.weight_manager = WeightMemoryManager()
        
        # 优化器
        self.memory_optimizer = MemoryOptimizer()
        self.garbage_collector = MemoryGarbageCollector()
        
        # 监控
        self.memory_monitor = MemoryMonitor()
        
    async def allocate_for_request(self, request):
        """为请求分配内存"""
        allocation_plan = await self._create_allocation_plan(request)
        
        try:
            # 分配KV Cache
            kv_allocation = await self.kv_cache_manager.allocate(
                request.seq_id,
                allocation_plan.kv_cache_size
            )
            
            # 分配激活值内存
            activation_allocation = await self.activation_manager.allocate(
                request.seq_id,
                allocation_plan.activation_size
            )
            
            # 分配权重内存（如果需要）
            weight_allocation = None
            if allocation_plan.needs_weight_loading:
                weight_allocation = await self.weight_manager.allocate(
                    request.model_id,
                    allocation_plan.weight_size
                )
            
            return MemoryAllocation(
                request_id=request.seq_id,
                kv_cache=kv_allocation,
                activations=activation_allocation,
                weights=weight_allocation
            )
            
        except OutOfMemoryError:
            # 内存不足时的处理策略
            await self._handle_out_of_memory(request, allocation_plan)
            raise
    
    async def _create_allocation_plan(self, request):
        """创建内存分配计划"""
        plan = AllocationPlan()
        
        # 估算KV Cache需求
        plan.kv_cache_size = self._estimate_kv_cache_size(request)
        
        # 估算激活值内存需求
        plan.activation_size = self._estimate_activation_size(request)
        
        # 检查是否需要加载权重
        plan.needs_weight_loading = not self.weight_manager.is_loaded(request.model_id)
        if plan.needs_weight_loading:
            plan.weight_size = self._get_model_weight_size(request.model_id)
        
        # 优化分配计划
        plan = await self.memory_optimizer.optimize_plan(plan)
        
        return plan
    
    def _estimate_kv_cache_size(self, request):
        """估算KV Cache大小"""
        seq_len = request.max_tokens
        num_layers = self.config.model_config.num_layers
        hidden_size = self.config.model_config.hidden_size
        
        # K和V各占一份，FP16格式
        kv_size = seq_len * num_layers * hidden_size * 2 * 2
        
        return kv_size
    
    async def _handle_out_of_memory(self, request, allocation_plan):
        """处理内存不足情况"""
        # 1. 尝试垃圾回收
        freed_memory = await self.garbage_collector.collect()
        
        if freed_memory >= allocation_plan.total_size:
            return  # 回收后可以满足需求
        
        # 2. 尝试内存压缩
        compressed_memory = await self.memory_optimizer.compress_memory()
        
        if freed_memory + compressed_memory >= allocation_plan.total_size:
            return
        
        # 3. 尝试offloading
        await self._offload_low_priority_data(allocation_plan.total_size)
        
        # 4. 如果仍然不足，抛出异常
        available = self._get_available_memory()
        if available < allocation_plan.total_size:
            raise OutOfMemoryError(
                f"Cannot allocate {allocation_plan.total_size} bytes, "
                f"only {available} bytes available"
            )
```

### KV Cache管理

```python
class KVCacheManager:
    """KV Cache专用管理器"""
    
    def __init__(self, cache_config):
        self.config = cache_config
        
        # 分页式管理
        self.page_size = cache_config.page_size  # 每页token数
        self.max_pages = cache_config.max_pages
        
        # 物理页面存储
        self.key_pages = torch.zeros(
            self.max_pages, self.page_size, cache_config.hidden_size,
            dtype=torch.float16, device='cuda'
        )
        self.value_pages = torch.zeros_like(self.key_pages)
        
        # 页面管理
        self.free_pages = set(range(self.max_pages))
        self.page_table = {}  # seq_id -> [page_ids]
        self.page_ref_count = torch.zeros(self.max_pages, dtype=torch.int32)
        
        # LRU管理
        self.lru_tracker = LRUTracker()
        
    async def allocate(self, seq_id, estimated_tokens):
        """为序列分配KV Cache"""
        num_pages = (estimated_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages:
            # 尝试回收页面
            await self._evict_pages(num_pages - len(self.free_pages))
        
        if len(self.free_pages) < num_pages:
            raise OutOfMemoryError(f"Cannot allocate {num_pages} pages for KV cache")
        
        # 分配页面
        allocated_pages = []
        for _ in range(num_pages):
            page_id = self.free_pages.pop()
            allocated_pages.append(page_id)
            self.page_ref_count[page_id] = 1
        
        self.page_table[seq_id] = allocated_pages
        self.lru_tracker.access(seq_id)
        
        return KVCacheAllocation(
            seq_id=seq_id,
            pages=allocated_pages,
            page_size=self.page_size
        )
    
    def get_kv_cache(self, seq_id, layer_id):
        """获取指定层的KV Cache"""
        if seq_id not in self.page_table:
            raise ValueError(f"Sequence {seq_id} not found in KV cache")
        
        pages = self.page_table[seq_id]
        self.lru_tracker.access(seq_id)
        
        # 收集页面数据
        key_data = []
        value_data = []
        
        for page_id in pages:
            key_data.append(self.key_pages[page_id])
            value_data.append(self.value_pages[page_id])
        
        keys = torch.cat(key_data, dim=0)
        values = torch.cat(value_data, dim=0)
        
        return keys, values
    
    def update_kv_cache(self, seq_id, layer_id, new_keys, new_values):
        """更新KV Cache"""
        if seq_id not in self.page_table:
            raise ValueError(f"Sequence {seq_id} not found in KV cache")
        
        pages = self.page_table[seq_id]
        self.lru_tracker.access(seq_id)
        
        # 计算需要更新的页面
        tokens_to_add = new_keys.size(0)
        start_page = len(pages) - 1  # 从最后一页开始
        
        # 更新页面数据
        current_token = 0
        for page_idx in range(start_page, len(pages)):
            page_id = pages[page_idx]
            tokens_in_page = min(self.page_size, tokens_to_add - current_token)
            
            if tokens_in_page > 0:
                self.key_pages[page_id][:tokens_in_page] = new_keys[current_token:current_token + tokens_in_page]
                self.value_pages[page_id][:tokens_in_page] = new_values[current_token:current_token + tokens_in_page]
                current_token += tokens_in_page
    
    async def _evict_pages(self, num_pages_needed):
        """驱逐页面"""
        evicted = 0
        
        # 按LRU顺序驱逐
        for seq_id in self.lru_tracker.get_lru_sequences():
            if evicted >= num_pages_needed:
                break
            
            if seq_id in self.page_table:
                pages_to_evict = self.page_table[seq_id]
                
                # 可选：将数据offload到CPU
                if self.config.enable_cpu_offload:
                    await self._offload_to_cpu(seq_id, pages_to_evict)
                
                # 释放页面
                for page_id in pages_to_evict:
                    self.free_pages.add(page_id)
                    self.page_ref_count[page_id] = 0
                
                del self.page_table[seq_id]
                evicted += len(pages_to_evict)
```

## 🎮 模型管理器 (Model Manager)

### 设计目标
- 高效的模型加载和卸载
- 多模型并发支持
- 权重共享和复用
- 动态模型切换

### 核心实现

```python
class ModelManager:
    """模型管理器"""
    
    def __init__(self, model_config):
        self.config = model_config
        
        # 模型存储
        self.loaded_models = {}  # model_id -> ModelInstance
        self.model_metadata = {}  # model_id -> ModelMetadata
        
        # 权重管理
        self.weight_loader = WeightLoader()
        self.weight_cache = WeightCache(model_config.weight_cache_size)
        
        # 模型优化
        self.model_optimizer = ModelOptimizer()
        self.quantization_manager = QuantizationManager()
        
    async def load_model(self, model_id, model_path):
        """加载模型"""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]
        
        # 1. 加载模型元数据
        metadata = await self._load_model_metadata(model_path)
        self.model_metadata[model_id] = metadata
        
        # 2. 检查内存需求
        memory_required = self._calculate_memory_requirement(metadata)
        if not self._check_memory_availability(memory_required):
            await self._free_memory_for_model(memory_required)
        
        # 3. 加载权重
        weights = await self.weight_loader.load_weights(model_path)
        
        # 4. 应用优化
        if self.config.enable_quantization:
            weights = await self.quantization_manager.quantize(weights, metadata)
        
        # 5. 创建模型实例
        model_instance = await self._create_model_instance(
            model_id, metadata, weights
        )
        
        # 6. 模型优化
        optimized_model = await self.model_optimizer.optimize(model_instance)
        
        self.loaded_models[model_id] = optimized_model
        
        return optimized_model
    
    async def _create_model_instance(self, model_id, metadata, weights):
        """创建模型实例"""
        # 根据模型类型创建相应的实例
        if metadata.model_type == "llama":
            return LlamaModel(model_id, metadata, weights)
        elif metadata.model_type == "gpt":
            return GPTModel(model_id, metadata, weights)
        else:
            raise ValueError(f"Unsupported model type: {metadata.model_type}")
    
    def get_model(self, model_id):
        """获取已加载的模型"""
        if model_id not in self.loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        return self.loaded_models[model_id]
    
    async def unload_model(self, model_id):
        """卸载模型"""
        if model_id in self.loaded_models:
            model_instance = self.loaded_models[model_id]
            
            # 清理GPU内存
            await model_instance.cleanup()
            
            # 从缓存中移除
            del self.loaded_models[model_id]
            
            # 强制垃圾回收
            torch.cuda.empty_cache()
    
    async def switch_model(self, from_model_id, to_model_id):
        """模型切换"""
        # 预加载目标模型
        if to_model_id not in self.loaded_models:
            await self.load_model(to_model_id)
        
        # 可选：卸载源模型以释放内存
        if self.config.auto_unload_on_switch:
            await self.unload_model(from_model_id)
        
        return self.loaded_models[to_model_id]

class WeightLoader:
    """权重加载器"""
    
    def __init__(self):
        self.loading_cache = {}
        self.loading_locks = {}
    
    async def load_weights(self, model_path):
        """异步加载权重"""
        # 避免重复加载
        if model_path in self.loading_cache:
            return self.loading_cache[model_path]
        
        # 使用锁避免并发加载同一模型
        if model_path not in self.loading_locks:
            self.loading_locks[model_path] = asyncio.Lock()
        
        async with self.loading_locks[model_path]:
            if model_path in self.loading_cache:
                return self.loading_cache[model_path]
            
            # 实际加载权重
            weights = await self._load_weights_from_disk(model_path)
            self.loading_cache[model_path] = weights
            
            return weights
    
    async def _load_weights_from_disk(self, model_path):
        """从磁盘加载权重"""
        weights = {}
        
        # 支持多种格式
        if model_path.endswith('.safetensors'):
            weights = await self._load_safetensors(model_path)
        elif model_path.endswith('.bin'):
            weights = await self._load_pytorch_bin(model_path)
        else:
            raise ValueError(f"Unsupported weight format: {model_path}")
        
        return weights
    
    async def _load_safetensors(self, file_path):
        """加载safetensors格式权重"""
        # 使用异步IO避免阻塞
        loop = asyncio.get_event_loop()
        
        def load_sync():
            from safetensors import safe_open
            weights = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            return weights
        
        return await loop.run_in_executor(None, load_sync)
```

## 🔗 组件协作机制

### 组件间通信

```python
class ComponentCommunicator:
    """组件间通信协调器"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.message_queue = MessageQueue()
        self.component_registry = ComponentRegistry()
    
    def register_component(self, component_name, component_instance):
        """注册组件"""
        self.component_registry.register(component_name, component_instance)
        
        # 订阅相关事件
        if hasattr(component_instance, 'event_handlers'):
            for event_type, handler in component_instance.event_handlers.items():
                self.event_bus.subscribe(event_type, handler)
    
    async def send_message(self, from_component, to_component, message):
        """发送组件间消息"""
        await self.message_queue.send(
            Message(
                sender=from_component,
                receiver=to_component,
                content=message,
                timestamp=time.time()
            )
        )
    
    async def broadcast_event(self, event_type, event_data):
        """广播事件"""
        await self.event_bus.publish(
            Event(
                type=event_type,
                data=event_data,
                timestamp=time.time()
            )
        )

# 使用示例
async def coordinate_inference_request(request):
    """协调推理请求的处理"""
    communicator = ComponentCommunicator()
    
    # 1. 调度器接收请求
    await communicator.send_message(
        "api_gateway", "scheduler", 
        {"type": "new_request", "request": request}
    )
    
    # 2. 内存管理器分配资源
    await communicator.send_message(
        "scheduler", "memory_manager",
        {"type": "allocate_memory", "request": request}
    )
    
    # 3. 推理引擎执行
    await communicator.send_message(
        "scheduler", "inference_engine",
        {"type": "execute_batch", "batch": batch}
    )
    
    # 4. 广播完成事件
    await communicator.broadcast_event(
        "request_completed", 
        {"request_id": request.id, "result": result}
    )
```

## 📊 性能监控

```python
class ComponentMonitor:
    """组件性能监控"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = AlertManager()
        
    def record_component_metric(self, component_name, metric_name, value):
        """记录组件指标"""
        key = f"{component_name}.{metric_name}"
        if key not in self.metrics:
            self.metrics[key] = []
        
        self.metrics[key].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # 检查告警条件
        self._check_alerts(component_name, metric_name, value)
    
    def get_component_health(self, component_name):
        """获取组件健康状态"""
        health_score = 100
        
        # 检查各项指标
        for metric_name in ['latency', 'throughput', 'error_rate']:
            metric_key = f"{component_name}.{metric_name}"
            if metric_key in self.metrics:
                recent_values = self.metrics[metric_key][-10:]  # 最近10个值
                health_score *= self._calculate_metric_health(metric_name, recent_values)
        
        return min(health_score, 100)
    
    def _check_alerts(self, component_name, metric_name, value):
        """检查告警条件"""
        alert_rules = {
            'latency': lambda v: v > 1000,  # 延迟超过1秒
            'error_rate': lambda v: v > 0.05,  # 错误率超过5%
            'memory_usage': lambda v: v > 0.9  # 内存使用率超过90%
        }
        
        if metric_name in alert_rules and alert_rules[metric_name](value):
            self.alerts.trigger_alert(
                f"{component_name}.{metric_name}",
                f"Metric {metric_name} value {value} exceeds threshold"
            )
```

---

*通过深入理解这些核心组件的设计和实现，我们可以更好地掌握 nano-vllm 的工作原理和优化策略。每个组件都经过精心设计，以实现高性能、高可靠性的大模型推理服务。*