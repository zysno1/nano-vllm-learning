# 内存管理优化

## 🎯 推理中的内存挑战

大语言模型推理面临着严峻的内存管理挑战：

### 主要问题
1. **模型权重占用**：大模型权重可达数十GB甚至数百GB
2. **KV Cache增长**：随着序列长度线性增长，可能占用大量内存
3. **激活值存储**：中间激活值需要临时存储空间
4. **内存碎片化**：频繁的分配和释放导致内存碎片
5. **动态批处理**：不同请求的内存需求差异很大

### 内存使用分析
```python
def analyze_memory_usage(model_config, batch_size, seq_len):
    """分析模型推理的内存使用情况"""
    
    # 模型权重内存 (FP16)
    model_params = model_config.num_layers * (
        model_config.hidden_size * model_config.intermediate_size * 2 +  # MLP
        model_config.hidden_size * model_config.hidden_size * 4  # Attention
    )
    model_memory = model_params * 2  # FP16, 2 bytes per param
    
    # KV Cache内存
    kv_cache_memory = (
        batch_size * seq_len * model_config.num_layers * 
        model_config.hidden_size * 2 * 2  # K + V, FP16
    )
    
    # 激活值内存
    activation_memory = (
        batch_size * seq_len * model_config.hidden_size * 
        model_config.num_layers * 4  # 估算中间激活值
    )
    
    total_memory = model_memory + kv_cache_memory + activation_memory
    
    return {
        "model_memory_gb": model_memory / (1024**3),
        "kv_cache_memory_gb": kv_cache_memory / (1024**3),
        "activation_memory_gb": activation_memory / (1024**3),
        "total_memory_gb": total_memory / (1024**3)
    }
```

## 🏗️ 内存管理架构

### 1. 分层内存管理

```python
class HierarchicalMemoryManager:
    def __init__(self, gpu_memory_gb, cpu_memory_gb):
        self.gpu_memory = gpu_memory_gb * 1024**3
        self.cpu_memory = cpu_memory_gb * 1024**3
        
        # GPU内存池
        self.gpu_pool = GPUMemoryPool(self.gpu_memory * 0.9)  # 留10%余量
        
        # CPU内存池 (用于offloading)
        self.cpu_pool = CPUMemoryPool(self.cpu_memory * 0.5)
        
        # 内存使用统计
        self.memory_stats = MemoryStats()
    
    def allocate(self, size, priority="normal", device="gpu"):
        """智能内存分配"""
        if device == "gpu":
            try:
                return self.gpu_pool.allocate(size)
            except OutOfMemoryError:
                if priority == "low":
                    # 低优先级数据可以offload到CPU
                    return self.cpu_pool.allocate(size)
                else:
                    # 尝试释放一些低优先级内存
                    self._evict_low_priority_data()
                    return self.gpu_pool.allocate(size)
        else:
            return self.cpu_pool.allocate(size)
    
    def _evict_low_priority_data(self):
        """驱逐低优先级数据到CPU或释放"""
        # 实现LRU或其他驱逐策略
        pass
```

### 2. 块状内存管理

```python
class BlockMemoryManager:
    def __init__(self, block_size=16, max_blocks=1000):
        self.block_size = block_size  # 每个块的token数
        self.max_blocks = max_blocks
        
        # 预分配内存块
        self.memory_blocks = torch.zeros(
            max_blocks, 2, block_size, 128,  # 2 for K,V; 128 for head_dim
            dtype=torch.float16, device='cuda'
        )
        
        # 块状态管理
        self.free_blocks = set(range(max_blocks))
        self.allocated_blocks = {}  # seq_id -> [block_ids]
        self.block_ref_count = torch.zeros(max_blocks, dtype=torch.int32)
    
    def allocate_sequence(self, seq_id, num_blocks):
        """为序列分配内存块"""
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemoryError(f"Not enough free blocks: {len(self.free_blocks)} < {num_blocks}")
        
        # 分配块
        allocated = []
        for _ in range(num_blocks):
            block_id = self.free_blocks.pop()
            allocated.append(block_id)
            self.block_ref_count[block_id] = 1
        
        self.allocated_blocks[seq_id] = allocated
        return allocated
    
    def extend_sequence(self, seq_id, additional_blocks):
        """扩展序列的内存块"""
        if seq_id not in self.allocated_blocks:
            raise ValueError(f"Sequence {seq_id} not found")
        
        new_blocks = []
        for _ in range(additional_blocks):
            if not self.free_blocks:
                # 尝试回收一些块
                self._garbage_collect()
            
            if self.free_blocks:
                block_id = self.free_blocks.pop()
                new_blocks.append(block_id)
                self.block_ref_count[block_id] = 1
        
        self.allocated_blocks[seq_id].extend(new_blocks)
        return new_blocks
    
    def free_sequence(self, seq_id):
        """释放序列的所有内存块"""
        if seq_id in self.allocated_blocks:
            for block_id in self.allocated_blocks[seq_id]:
                self.block_ref_count[block_id] -= 1
                if self.block_ref_count[block_id] == 0:
                    self.free_blocks.add(block_id)
            
            del self.allocated_blocks[seq_id]
    
    def _garbage_collect(self):
        """垃圾回收：释放引用计数为0的块"""
        for block_id in range(self.max_blocks):
            if self.block_ref_count[block_id] == 0 and block_id not in self.free_blocks:
                self.free_blocks.add(block_id)
```

## 🚀 KV Cache优化策略

### 1. 分页式KV Cache

```python
class PagedKVCache:
    def __init__(self, page_size=16, max_pages=1000):
        self.page_size = page_size
        self.max_pages = max_pages
        
        # 物理页面存储
        self.key_pages = torch.zeros(
            max_pages, page_size, 128,  # 128 = num_heads * head_dim
            dtype=torch.float16, device='cuda'
        )
        self.value_pages = torch.zeros_like(self.key_pages)
        
        # 页面管理
        self.free_pages = set(range(max_pages))
        self.page_table = {}  # seq_id -> [page_ids]
        self.page_usage = torch.zeros(max_pages, dtype=torch.int32)
    
    def allocate_pages(self, seq_id, num_tokens):
        """为序列分配页面"""
        num_pages = (num_tokens + self.page_size - 1) // self.page_size
        
        if len(self.free_pages) < num_pages:
            # 尝试页面换出
            self._evict_pages(num_pages - len(self.free_pages))
        
        allocated_pages = []
        for _ in range(num_pages):
            page_id = self.free_pages.pop()
            allocated_pages.append(page_id)
            self.page_usage[page_id] = 1
        
        self.page_table[seq_id] = allocated_pages
        return allocated_pages
    
    def get_kv_data(self, seq_id, start_pos=0, end_pos=None):
        """获取序列的KV数据"""
        if seq_id not in self.page_table:
            raise ValueError(f"Sequence {seq_id} not found")
        
        pages = self.page_table[seq_id]
        
        # 收集相关页面的数据
        key_data = []
        value_data = []
        
        for page_id in pages:
            key_data.append(self.key_pages[page_id])
            value_data.append(self.value_pages[page_id])
        
        # 拼接并截取所需部分
        keys = torch.cat(key_data, dim=0)
        values = torch.cat(value_data, dim=0)
        
        if end_pos is None:
            end_pos = keys.size(0)
        
        return keys[start_pos:end_pos], values[start_pos:end_pos]
    
    def _evict_pages(self, num_pages_needed):
        """页面换出策略 (LRU)"""
        # 找到最少使用的页面
        usage_sorted = torch.argsort(self.page_usage)
        
        evicted = 0
        for page_id in usage_sorted:
            if self.page_usage[page_id] > 0 and evicted < num_pages_needed:
                # 将页面数据移到CPU或磁盘
                self._offload_page(page_id)
                self.free_pages.add(page_id.item())
                self.page_usage[page_id] = 0
                evicted += 1
    
    def _offload_page(self, page_id):
        """将页面数据offload到CPU"""
        # 实现页面数据的CPU存储
        pass
```

### 2. 压缩式KV Cache

```python
class CompressedKVCache:
    def __init__(self, compression_ratio=0.5):
        self.compression_ratio = compression_ratio
        self.compressor = KVCompressor()
        
        # 原始和压缩数据存储
        self.raw_cache = {}
        self.compressed_cache = {}
        self.compression_metadata = {}
    
    def store_kv(self, seq_id, keys, values, compress=True):
        """存储KV数据，可选择是否压缩"""
        if compress and keys.size(0) > 32:  # 只对长序列压缩
            # 压缩KV数据
            compressed_keys, key_metadata = self.compressor.compress(keys)
            compressed_values, value_metadata = self.compressor.compress(values)
            
            self.compressed_cache[seq_id] = (compressed_keys, compressed_values)
            self.compression_metadata[seq_id] = (key_metadata, value_metadata)
        else:
            # 直接存储原始数据
            self.raw_cache[seq_id] = (keys, values)
    
    def get_kv(self, seq_id):
        """获取KV数据，自动解压缩"""
        if seq_id in self.raw_cache:
            return self.raw_cache[seq_id]
        elif seq_id in self.compressed_cache:
            compressed_keys, compressed_values = self.compressed_cache[seq_id]
            key_metadata, value_metadata = self.compression_metadata[seq_id]
            
            # 解压缩
            keys = self.compressor.decompress(compressed_keys, key_metadata)
            values = self.compressor.decompress(compressed_values, value_metadata)
            
            return keys, values
        else:
            raise ValueError(f"Sequence {seq_id} not found")

class KVCompressor:
    def compress(self, tensor):
        """压缩张量 (简化实现)"""
        # 使用量化压缩
        scale = tensor.abs().max() / 127
        quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
        
        metadata = {"scale": scale, "shape": tensor.shape, "dtype": tensor.dtype}
        return quantized, metadata
    
    def decompress(self, compressed_tensor, metadata):
        """解压缩张量"""
        scale = metadata["scale"]
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        
        # 反量化
        decompressed = compressed_tensor.float() * scale
        return decompressed.to(dtype).view(shape)
```

## 💾 内存池化技术

### 1. 预分配内存池

```python
class PreallocatedMemoryPool:
    def __init__(self, pool_size_gb=8):
        self.pool_size = pool_size_gb * 1024**3
        
        # 预分配大块内存
        self.memory_pool = torch.zeros(
            self.pool_size // 2,  # FP16, 2 bytes per element
            dtype=torch.float16, device='cuda'
        )
        
        # 内存块管理
        self.free_blocks = [(0, self.pool_size // 2)]  # (start, size)
        self.allocated_blocks = {}  # request_id -> (start, size)
        self.fragmentation_threshold = 0.1  # 10%碎片率触发整理
    
    def allocate(self, request_id, size_bytes):
        """分配内存块"""
        size_elements = size_bytes // 2  # FP16
        
        # 寻找合适的空闲块
        for i, (start, block_size) in enumerate(self.free_blocks):
            if block_size >= size_elements:
                # 分配内存
                self.allocated_blocks[request_id] = (start, size_elements)
                
                # 更新空闲块
                if block_size > size_elements:
                    self.free_blocks[i] = (start + size_elements, block_size - size_elements)
                else:
                    del self.free_blocks[i]
                
                return self.memory_pool[start:start + size_elements]
        
        # 尝试内存整理
        if self._get_fragmentation_ratio() > self.fragmentation_threshold:
            self._defragment()
            return self.allocate(request_id, size_bytes)  # 递归重试
        
        raise OutOfMemoryError("Cannot allocate memory block")
    
    def deallocate(self, request_id):
        """释放内存块"""
        if request_id in self.allocated_blocks:
            start, size = self.allocated_blocks[request_id]
            del self.allocated_blocks[request_id]
            
            # 添加到空闲块列表
            self.free_blocks.append((start, size))
            self.free_blocks.sort()
            
            # 合并相邻空闲块
            self._merge_free_blocks()
    
    def _get_fragmentation_ratio(self):
        """计算内存碎片率"""
        total_free = sum(size for _, size in self.free_blocks)
        if total_free == 0:
            return 0.0
        
        largest_free = max(size for _, size in self.free_blocks) if self.free_blocks else 0
        return 1.0 - (largest_free / total_free)
    
    def _defragment(self):
        """内存碎片整理"""
        # 简化实现：重新组织已分配的内存块
        active_blocks = list(self.allocated_blocks.items())
        
        # 按大小排序，大块优先
        active_blocks.sort(key=lambda x: x[1][1], reverse=True)
        
        # 重新分配位置
        current_pos = 0
        new_allocated = {}
        
        for request_id, (old_start, size) in active_blocks:
            # 移动数据到新位置
            old_data = self.memory_pool[old_start:old_start + size].clone()
            self.memory_pool[current_pos:current_pos + size] = old_data
            
            new_allocated[request_id] = (current_pos, size)
            current_pos += size
        
        # 更新分配表和空闲块
        self.allocated_blocks = new_allocated
        remaining_size = len(self.memory_pool) - current_pos
        self.free_blocks = [(current_pos, remaining_size)] if remaining_size > 0 else []
```

### 2. 自适应内存管理

```python
class AdaptiveMemoryManager:
    def __init__(self):
        self.memory_usage_history = []
        self.allocation_patterns = {}
        self.prediction_model = MemoryUsagePredictor()
    
    def predict_memory_need(self, request_info):
        """预测请求的内存需求"""
        # 基于历史数据预测
        seq_len = request_info.get('seq_len', 100)
        batch_size = request_info.get('batch_size', 1)
        
        # 简单的线性预测模型
        base_memory = seq_len * batch_size * 1024  # 基础内存需求
        
        # 考虑历史模式
        if request_info.get('user_id') in self.allocation_patterns:
            pattern = self.allocation_patterns[request_info['user_id']]
            adjustment_factor = pattern.get('avg_memory_multiplier', 1.0)
            base_memory *= adjustment_factor
        
        return int(base_memory)
    
    def update_usage_pattern(self, request_info, actual_memory_used):
        """更新内存使用模式"""
        user_id = request_info.get('user_id')
        if user_id:
            if user_id not in self.allocation_patterns:
                self.allocation_patterns[user_id] = {
                    'total_requests': 0,
                    'total_memory': 0,
                    'avg_memory_multiplier': 1.0
                }
            
            pattern = self.allocation_patterns[user_id]
            pattern['total_requests'] += 1
            pattern['total_memory'] += actual_memory_used
            
            # 更新平均倍数
            predicted = self.predict_memory_need(request_info)
            if predicted > 0:
                pattern['avg_memory_multiplier'] = (
                    pattern['total_memory'] / pattern['total_requests'] / predicted
                )
    
    def optimize_allocation_strategy(self):
        """优化分配策略"""
        # 分析内存使用模式
        recent_usage = self.memory_usage_history[-100:]  # 最近100次
        
        if len(recent_usage) > 10:
            avg_usage = sum(recent_usage) / len(recent_usage)
            peak_usage = max(recent_usage)
            
            # 调整预分配策略
            if peak_usage > avg_usage * 2:
                # 峰值较高，增加预分配
                self.preallocation_factor = 1.5
            else:
                # 使用较为平稳，减少预分配
                self.preallocation_factor = 1.2
```

## 🔧 内存优化技术

### 1. 梯度检查点 (在推理中不适用，但了解概念)

```python
# 推理中不需要梯度，但理解概念有助于优化
def inference_with_checkpointing(model, input_data):
    """
    推理中的内存优化：重计算vs缓存权衡
    """
    with torch.no_grad():  # 推理不需要梯度
        # 对于内存受限的情况，可以选择重计算某些中间结果
        # 而不是全部缓存
        
        intermediate_results = {}
        
        for layer_idx, layer in enumerate(model.layers):
            if layer_idx % 2 == 0:  # 每隔一层保存中间结果
                intermediate_results[layer_idx] = input_data.clone()
            
            input_data = layer(input_data)
        
        return input_data
```

### 2. 内存映射文件

```python
class MemoryMappedWeights:
    def __init__(self, weight_file_path):
        self.weight_file = weight_file_path
        self.mmap_weights = {}
        self.loaded_weights = {}  # 缓存在GPU上的权重
        
    def load_weight(self, layer_name):
        """按需加载权重到GPU"""
        if layer_name in self.loaded_weights:
            return self.loaded_weights[layer_name]
        
        # 从内存映射文件读取
        if layer_name not in self.mmap_weights:
            self.mmap_weights[layer_name] = np.memmap(
                f"{self.weight_file}_{layer_name}.bin",
                dtype=np.float16, mode='r'
            )
        
        # 加载到GPU
        weight_tensor = torch.from_numpy(self.mmap_weights[layer_name]).cuda()
        
        # 缓存管理：如果GPU内存不足，释放最少使用的权重
        if self._gpu_memory_usage() > 0.8:  # 80%阈值
            self._evict_least_used_weight()
        
        self.loaded_weights[layer_name] = weight_tensor
        return weight_tensor
    
    def _evict_least_used_weight(self):
        """驱逐最少使用的权重"""
        # 简化实现：随机驱逐一个权重
        if self.loaded_weights:
            layer_to_evict = next(iter(self.loaded_weights))
            del self.loaded_weights[layer_to_evict]
            torch.cuda.empty_cache()
```

## 📊 内存监控和调试

### 1. 内存使用监控

```python
class MemoryMonitor:
    def __init__(self):
        self.memory_snapshots = []
        self.allocation_tracker = {}
        
    def take_snapshot(self, tag=""):
        """拍摄内存使用快照"""
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        else:
            gpu_memory = {}
        
        import psutil
        cpu_memory = {
            'used': psutil.virtual_memory().used,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        }
        
        snapshot = {
            'timestamp': time.time(),
            'tag': tag,
            'gpu_memory': gpu_memory,
            'cpu_memory': cpu_memory
        }
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def analyze_memory_trend(self):
        """分析内存使用趋势"""
        if len(self.memory_snapshots) < 2:
            return "Not enough data"
        
        recent = self.memory_snapshots[-10:]  # 最近10个快照
        
        gpu_trend = []
        for snapshot in recent:
            if 'allocated' in snapshot['gpu_memory']:
                gpu_trend.append(snapshot['gpu_memory']['allocated'])
        
        if len(gpu_trend) > 1:
            trend = "increasing" if gpu_trend[-1] > gpu_trend[0] else "decreasing"
            return f"GPU memory trend: {trend}"
        
        return "No clear trend"
    
    def detect_memory_leak(self):
        """检测内存泄漏"""
        if len(self.memory_snapshots) < 10:
            return False
        
        recent_gpu_usage = [
            s['gpu_memory'].get('allocated', 0) 
            for s in self.memory_snapshots[-10:]
        ]
        
        # 简单的泄漏检测：内存持续增长
        increasing_count = 0
        for i in range(1, len(recent_gpu_usage)):
            if recent_gpu_usage[i] > recent_gpu_usage[i-1]:
                increasing_count += 1
        
        return increasing_count > 7  # 70%的时间在增长
```

### 2. 内存调试工具

```python
def debug_memory_usage():
    """调试内存使用情况"""
    print("=== GPU Memory Debug Info ===")
    
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
        
        # 详细的内存分配信息
        print("\n=== Memory Allocation Details ===")
        print(torch.cuda.memory_summary())
    
    print("\n=== CPU Memory Info ===")
    import psutil
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / 1024**3:.2f} GB")
    print(f"Used: {mem.used / 1024**3:.2f} GB")
    print(f"Available: {mem.available / 1024**3:.2f} GB")
    print(f"Percentage: {mem.percent:.1f}%")

def profile_memory_usage(func):
    """内存使用分析装饰器"""
    def wrapper(*args, **kwargs):
        # 记录开始状态
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录结束状态
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"Function {func.__name__} memory usage: {(end_memory - start_memory) / 1024**2:.2f} MB")
        
        return result
    return wrapper
```

## 🎯 nano-vllm中的内存管理

### 1. 统一内存管理接口

```python
class NanoVLLMMemoryManager:
    def __init__(self, config):
        self.config = config
        
        # 不同类型的内存管理器
        self.kv_cache_manager = BlockMemoryManager(
            block_size=config.block_size,
            max_blocks=config.max_kv_blocks
        )
        
        self.activation_pool = PreallocatedMemoryPool(
            pool_size_gb=config.activation_memory_gb
        )
        
        self.weight_manager = MemoryMappedWeights(config.weight_path)
        
        # 内存监控
        self.monitor = MemoryMonitor()
    
    def allocate_for_request(self, request):
        """为请求分配所需的所有内存资源"""
        resources = {}
        
        # 分配KV缓存
        kv_blocks = self.kv_cache_manager.allocate_sequence(
            request.seq_id, 
            request.estimated_blocks
        )
        resources['kv_cache'] = kv_blocks
        
        # 分配激活值内存
        activation_memory = self.activation_pool.allocate(
            request.seq_id,
            request.estimated_activation_size
        )
        resources['activations'] = activation_memory
        
        return resources
    
    def cleanup_request(self, request_id):
        """清理请求的所有内存资源"""
        self.kv_cache_manager.free_sequence(request_id)
        self.activation_pool.deallocate(request_id)
        
        # 记录内存快照
        self.monitor.take_snapshot(f"cleanup_{request_id}")
```

## 🚀 总结

内存管理是大语言模型推理性能的关键因素：

1. **分层管理**：GPU/CPU/磁盘的分层内存架构
2. **块状管理**：减少碎片化，提高分配效率
3. **智能缓存**：KV Cache的优化存储和访问
4. **动态调度**：根据负载动态调整内存分配策略
5. **监控调试**：实时监控内存使用，及时发现问题

nano-vllm通过这些内存管理技术，实现了高效的内存使用，为大模型推理提供了坚实的基础。

---

*至此，我们已经完成了基础概念的学习。接下来让我们深入了解nano-vllm的架构设计！*