# torch.compile 编译优化专题

## 📚 学习目标

通过本教程，你将深入理解并掌握：

1. **torch.compile 基础概念**
   - PyTorch 2.0 编译系统架构
   - 动态图到静态图的转换
   - 编译优化的原理和优势

2. **编译后端和优化器**
   - TorchInductor 后端
   - TorchScript 后端
   - 自定义编译后端
   - 优化策略选择

3. **LLM 推理优化**
   - Transformer 模型编译
   - 注意力机制优化
   - KV Cache 编译优化
   - 批处理优化

4. **性能调优技巧**
   - 编译模式选择
   - 动态形状处理
   - 内存优化
   - 调试和性能分析

## 🎯 理论基础

### torch.compile 概述

`torch.compile` 是 PyTorch 2.0 引入的革命性功能，通过即时编译（JIT）将 PyTorch 代码转换为优化的机器代码，显著提升推理和训练性能。

### 核心优势

| 优势 | 描述 | 性能提升 |
|------|------|----------|
| **算子融合** | 将多个操作融合为单个内核 | 20-40% |
| **内存优化** | 减少中间张量分配 | 30-50% |
| **并行优化** | 更好的 GPU 利用率 | 1.5-3x |
| **图优化** | 消除冗余计算 | 10-30% |

### 编译流程

```
Python 代码 → TorchDynamo → 图捕获 → 编译后端 → 优化代码
     ↓              ↓           ↓          ↓           ↓
  动态执行      字节码分析    计算图    TorchInductor   机器代码
```

### 编译模式

1. **default**: 平衡编译时间和性能
2. **reduce-overhead**: 最小化 Python 开销
3. **max-autotune**: 最大化性能优化

## 🏗️ 项目结构

```
10_torch_compile/
├── README.md                    # 本文档
├── requirements.txt             # 依赖包
├── compile_basics.py            # 基础编译概念
├── model_compilation.py         # 模型编译实战
├── attention_optimization.py    # 注意力机制优化
├── kv_cache_compilation.py      # KV Cache 编译优化
├── batch_optimization.py        # 批处理优化
├── dynamic_shapes.py            # 动态形状处理
├── memory_optimization.py       # 内存优化技巧
├── debugging_tools.py           # 调试和分析工具
├── benchmarks/                  # 性能测试
│   ├── compilation_overhead.py # 编译开销分析
│   ├── inference_speedup.py    # 推理加速测试
│   └── memory_usage_test.py    # 内存使用测试
├── backends/                    # 自定义后端
│   ├── custom_backend.py       # 自定义编译后端
│   └── backend_comparison.py   # 后端性能对比
└── docs/                       # 详细文档
    ├── compilation_guide.md    # 编译指南
    ├── optimization_tips.md    # 优化技巧
    └── troubleshooting.md      # 故障排除
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd examples/advanced/10_torch_compile
pip install -r requirements.txt
```

### 2. 基础概念演示

```bash
# 运行基础编译演示
python compile_basics.py

# 模型编译实战
python model_compilation.py

# 注意力机制优化
python attention_optimization.py
```

### 3. 高级优化技巧

```bash
# KV Cache 编译优化
python kv_cache_compilation.py

# 批处理优化
python batch_optimization.py

# 动态形状处理
python dynamic_shapes.py
```

### 4. 性能分析

```bash
# 编译开销分析
python benchmarks/compilation_overhead.py

# 推理加速测试
python benchmarks/inference_speedup.py

# 内存使用测试
python benchmarks/memory_usage_test.py
```

## 🧪 实验内容

### 实验 1：基础编译概念

**目标**：理解 torch.compile 的基本用法和效果

**内容**：
- 简单函数编译
- 编译模式对比
- 性能基准测试
- 编译开销分析

**关键代码**：
```python
import torch

# 基础函数
def simple_function(x, y):
    return torch.matmul(x, y) + torch.relu(x)

# 编译函数
compiled_fn = torch.compile(simple_function, mode="default")

# 性能对比
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')

# 原始执行
start = time.time()
result1 = simple_function(x, y)
original_time = time.time() - start

# 编译执行
start = time.time()
result2 = compiled_fn(x, y)
compiled_time = time.time() - start

print(f"加速比: {original_time / compiled_time:.2f}x")
```

### 实验 2：Transformer 模型编译

**目标**：优化完整的 Transformer 模型

**内容**：
- 注意力层编译
- MLP 层优化
- 层归一化融合
- 残差连接优化

**性能提升**：
- 推理速度：1.5-2.5x
- 内存使用：减少 20-40%
- GPU 利用率：提升 30-50%

### 实验 3：注意力机制深度优化

**目标**：专门优化注意力计算

**内容**：
- 多头注意力融合
- Softmax 优化
- 矩阵乘法优化
- 掩码处理优化

**关键优化**：
```python
class OptimizedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 融合的 QKV 投影
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # 一次性计算 QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 融合的注意力计算
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=True
        )
        
        # 输出投影
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn)

# 编译优化
compiled_attention = torch.compile(
    OptimizedAttention(512, 8), 
    mode="max-autotune"
)
```

### 实验 4：KV Cache 编译优化

**目标**：优化 KV Cache 的使用和管理

**内容**：
- 增量计算优化
- 缓存更新融合
- 内存访问优化
- 批处理缓存管理

**优化策略**：
- 融合 KV 更新操作
- 优化内存布局
- 减少数据拷贝
- 批量缓存操作

### 实验 5：动态形状处理

**目标**：处理变长序列的编译优化

**内容**：
- 动态形状编译
- 填充策略优化
- 掩码处理
- 性能权衡分析

**技术要点**：
```python
# 动态形状编译
@torch.compile(dynamic=True)
def dynamic_attention(q, k, v, seq_lens):
    # 处理变长序列
    max_len = seq_lens.max()
    batch_size = q.size(0)
    
    # 创建因果掩码
    causal_mask = torch.triu(
        torch.ones(max_len, max_len, dtype=torch.bool),
        diagonal=1
    )
    
    # 创建填充掩码
    padding_mask = torch.arange(max_len)[None, :] >= seq_lens[:, None]
    
    # 组合掩码
    combined_mask = causal_mask | padding_mask.unsqueeze(1)
    
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=combined_mask
    )
```

### 实验 6：内存优化技巧

**目标**：最小化内存使用并提高缓存效率

**内容**：
- 梯度检查点
- 激活重计算
- 内存池优化
- 垃圾回收控制

**优化技术**：
- 就地操作优化
- 内存布局优化
- 临时张量管理
- 内存碎片减少

## 💡 核心实现代码

### 1. 智能编译装饰器

```python
import functools
import torch
from typing import Optional, Dict, Any

class SmartCompiler:
    """智能编译管理器"""
    
    def __init__(self):
        self.compiled_cache = {}
        self.compilation_stats = {}
    
    def compile_with_cache(self, 
                          mode: str = "default",
                          dynamic: bool = False,
                          backend: str = "inductor"):
        """带缓存的编译装饰器"""
        def decorator(func):
            cache_key = f"{func.__name__}_{mode}_{dynamic}_{backend}"
            
            if cache_key not in self.compiled_cache:
                compiled_func = torch.compile(
                    func, 
                    mode=mode, 
                    dynamic=dynamic,
                    backend=backend
                )
                self.compiled_cache[cache_key] = compiled_func
                self.compilation_stats[cache_key] = {
                    'compile_count': 1,
                    'last_compiled': time.time()
                }
            else:
                self.compilation_stats[cache_key]['compile_count'] += 1
            
            return self.compiled_cache[cache_key]
        
        return decorator

# 使用示例
compiler = SmartCompiler()

@compiler.compile_with_cache(mode="max-autotune")
def optimized_matmul(a, b):
    return torch.matmul(a, b)
```

### 2. 自适应编译策略

```python
class AdaptiveCompiler:
    """自适应编译策略"""
    
    def __init__(self):
        self.performance_history = {}
        self.compilation_overhead = {}
    
    def adaptive_compile(self, func, input_shapes, device):
        """根据输入特征自适应选择编译策略"""
        # 分析输入特征
        total_elements = sum(torch.prod(torch.tensor(shape)) for shape in input_shapes)
        complexity_score = self._calculate_complexity(func, input_shapes)
        
        # 选择编译模式
        if total_elements > 1e6 and complexity_score > 0.8:
            mode = "max-autotune"
        elif total_elements > 1e4:
            mode = "reduce-overhead"
        else:
            mode = "default"
        
        # 动态形状检测
        dynamic = self._needs_dynamic_shapes(input_shapes)
        
        return torch.compile(func, mode=mode, dynamic=dynamic)
    
    def _calculate_complexity(self, func, input_shapes):
        """计算函数复杂度"""
        # 简化的复杂度计算
        return min(1.0, len(input_shapes) * 0.2)
    
    def _needs_dynamic_shapes(self, input_shapes):
        """判断是否需要动态形状"""
        # 检查形状变化模式
        return any(dim < 0 for shape in input_shapes for dim in shape)
```

### 3. 编译性能监控

```python
class CompilationProfiler:
    """编译性能监控器"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_compilation(self, func_name: str):
        """编译性能分析装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 记录编译前状态
                start_memory = torch.cuda.memory_allocated()
                start_time = time.time()
                
                # 执行编译函数
                result = func(*args, **kwargs)
                
                # 记录编译后状态
                end_time = time.time()
                end_memory = torch.cuda.memory_allocated()
                
                # 更新性能统计
                if func_name not in self.profiles:
                    self.profiles[func_name] = {
                        'call_count': 0,
                        'total_time': 0.0,
                        'avg_time': 0.0,
                        'memory_delta': 0,
                        'peak_memory': 0
                    }
                
                profile = self.profiles[func_name]
                profile['call_count'] += 1
                profile['total_time'] += (end_time - start_time)
                profile['avg_time'] = profile['total_time'] / profile['call_count']
                profile['memory_delta'] = end_memory - start_memory
                profile['peak_memory'] = max(profile['peak_memory'], end_memory)
                
                return result
            
            return wrapper
        return decorator
    
    def get_report(self) -> str:
        """生成性能报告"""
        report = "编译性能报告\n" + "=" * 50 + "\n"
        
        for func_name, profile in self.profiles.items():
            report += f"\n函数: {func_name}\n"
            report += f"  调用次数: {profile['call_count']}\n"
            report += f"  平均时间: {profile['avg_time']:.4f}s\n"
            report += f"  内存变化: {profile['memory_delta'] / 1024**2:.2f} MB\n"
            report += f"  峰值内存: {profile['peak_memory'] / 1024**2:.2f} MB\n"
        
        return report
```

## 📊 性能优化技巧

### 1. 编译模式选择

```python
# 根据场景选择编译模式
compilation_strategies = {
    'development': {
        'mode': 'default',
        'dynamic': True,
        'fullgraph': False
    },
    'production': {
        'mode': 'max-autotune',
        'dynamic': False,
        'fullgraph': True
    },
    'inference': {
        'mode': 'reduce-overhead',
        'dynamic': False,
        'fullgraph': True
    }
}
```

### 2. 内存优化策略

```python
# 内存优化编译
@torch.compile(mode="reduce-overhead")
def memory_efficient_attention(q, k, v):
    # 使用 Flash Attention 风格的分块计算
    chunk_size = 1024
    seq_len = q.size(-2)
    
    output = torch.zeros_like(q)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[..., i:end_i, :]
        
        # 计算注意力分数
        scores = torch.matmul(q_chunk, k.transpose(-2, -1))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 计算输出
        output[..., i:end_i, :] = torch.matmul(attn_weights, v)
    
    return output
```

### 3. 动态形状优化

```python
# 动态形状处理
def create_dynamic_model(max_seq_len: int):
    @torch.compile(dynamic=True)
    def dynamic_forward(input_ids, attention_mask):
        # 获取实际序列长度
        actual_len = attention_mask.sum(dim=1).max().item()
        
        # 裁剪到实际长度
        input_ids = input_ids[:, :actual_len]
        attention_mask = attention_mask[:, :actual_len]
        
        # 模型前向传播
        return model(input_ids, attention_mask)
    
    return dynamic_forward
```

## ⚠️ 注意事项与限制

### 技术限制

1. **编译开销**：首次编译需要额外时间
2. **动态形状**：可能影响优化效果
3. **调试困难**：编译后的代码难以调试
4. **内存使用**：编译可能增加内存使用

### 最佳实践

1. **预热编译**：在正式推理前进行预热
2. **批量处理**：使用固定批大小以获得最佳性能
3. **模式选择**：根据场景选择合适的编译模式
4. **性能监控**：定期监控编译效果

### 调试技巧

```python
# 编译调试配置
import torch._dynamo as dynamo

# 启用详细日志
dynamo.config.log_level = logging.DEBUG
dynamo.config.verbose = True

# 禁用特定优化进行调试
dynamo.config.suppress_errors = False

# 查看编译图
@torch.compile(backend="eager")  # 使用 eager 后端进行调试
def debug_function(x):
    return torch.relu(torch.matmul(x, x.T))
```

## 🔬 进阶学习

### 研究方向

1. **自定义后端开发**：开发专用编译后端
2. **图优化算法**：研究新的图优化技术
3. **硬件适配**：针对特定硬件的优化
4. **动态优化**：运行时自适应优化

### 相关论文

- "TorchInductor: A PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes"
- "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
- "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"

### 开源项目

- **TorchInductor**: PyTorch 官方编译后端
- **TorchScript**: PyTorch 静态图编译
- **ONNX Runtime**: 跨平台推理优化

## 📚 相关资源

### 官方文档

- [PyTorch 2.0 Compilation](https://pytorch.org/docs/stable/torch.compiler.html)
- [TorchInductor Documentation](https://pytorch.org/docs/stable/torch.compiler_inductor.html)

### 技术博客

- [PyTorch 2.0 Performance Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [torch.compile Deep Dive](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

### 视频教程

- [PyTorch 2.0 Compilation Workshop](https://example.com)
- [Advanced torch.compile Techniques](https://example.com)

---

**下一步**：运行 `compile_basics.py` 开始你的 torch.compile 优化之旅！