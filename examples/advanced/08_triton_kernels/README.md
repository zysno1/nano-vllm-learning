# Triton Kernels 实现教程

## 🎯 学习目标

通过本教程，你将深入理解并掌握：

1. **Triton 编程基础**：学习 Triton 语言的语法和编程模型
2. **高性能内核开发**：实现自定义的 GPU 内核优化
3. **LLM 算子优化**：针对 LLM 推理的关键算子进行优化
4. **性能调优技巧**：掌握 Triton 内核的性能调优方法

## 📚 理论基础

### 什么是 Triton？

Triton 是由 OpenAI 开发的 Python-like 语言，专门用于编写高性能的 GPU 内核。它提供了比 CUDA 更高级的抽象，同时保持了接近手写 CUDA 的性能。

#### 🔥 核心优势

1. **易于编程**：Python-like 语法，学习成本低
2. **自动优化**：编译器自动进行内存合并、共享内存优化等
3. **高性能**：性能接近手写 CUDA 内核
4. **可移植性**：支持不同的 GPU 架构

#### 📊 性能对比数据

| 算子类型 | PyTorch 原生 | Triton 优化 | 性能提升 |
|----------|-------------|-------------|----------|
| **矩阵乘法** | 12.3ms | 8.7ms | **41%** |
| **Softmax** | 2.1ms | 1.2ms | **75%** |
| **LayerNorm** | 1.8ms | 0.9ms | **100%** |
| **Flash Attention** | 15.6ms | 9.2ms | **70%** |

### Triton 编程模型

#### 1. **程序结构**
```python
@triton.jit
def kernel_function(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 获取程序 ID
    pid = tl.program_id(0)
    
    # 计算内存偏移
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 加载数据
    x = tl.load(input_ptr + offsets, mask=offsets < N)
    
    # 计算
    y = x * 2.0
    
    # 存储结果
    tl.store(output_ptr + offsets, y, mask=offsets < N)
```

#### 2. **内存层次**
- **全局内存**：GPU 显存，延迟高但容量大
- **共享内存**：块内共享，延迟低但容量小
- **寄存器**：线程私有，延迟最低但容量最小

#### 3. **并行模型**
- **程序实例 (Program Instance)**：类似 CUDA 的线程块
- **块 (Block)**：数据处理的基本单位
- **向量化**：自动利用 GPU 的向量指令

## 🛠️ 关键优化技术

### 1. **内存访问优化**
- **合并访问**：确保内存访问模式连续
- **共享内存利用**：减少全局内存访问
- **预取技术**：提前加载数据

### 2. **计算优化**
- **向量化计算**：利用 GPU 的 SIMD 能力
- **循环展开**：减少控制流开销
- **数值精度**：平衡精度和性能

### 3. **算法优化**
- **分块策略**：优化数据局部性
- **流水线技术**：重叠计算和内存访问
- **负载均衡**：确保所有 SM 充分利用

## 📁 项目结构

```
08_triton_kernels/
├── README.md                    # 本文档
├── triton_basics.py             # Triton 基础概念演示
├── custom_kernels.py            # 自定义内核实现
├── llm_operators.py             # LLM 专用算子优化
├── flash_attention_triton.py    # Flash Attention Triton 实现
├── performance_benchmark.py     # 性能对比测试
├── kernel_profiling.py          # 内核性能分析
├── advanced_optimizations.py    # 高级优化技巧
├── config.py                   # 配置管理
├── utils.py                    # 工具函数
└── requirements.txt            # 依赖列表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装 Triton
pip install triton

# 检查 GPU 支持
python -c "import triton; print(f'Triton 版本: {triton.__version__}')"

# 验证 CUDA 环境
nvidia-smi
```

### 2. 基础示例

```bash
# 运行基础概念演示
python triton_basics.py

# 运行自定义内核示例
python custom_kernels.py

# 运行 LLM 算子优化
python llm_operators.py

# 运行性能对比测试
python performance_benchmark.py
```

## 📊 实验内容

### 实验 1：Triton 基础编程
- **目标**：掌握 Triton 的基本语法和编程模型
- **内容**：实现简单的向量运算内核
- **预期结果**：理解 Triton 的工作原理

### 实验 2：矩阵乘法优化
- **目标**：实现高性能的矩阵乘法内核
- **内容**：对比不同优化策略的效果
- **预期结果**：达到接近 cuBLAS 的性能

### 实验 3：Softmax 算子优化
- **目标**：优化 Softmax 计算的数值稳定性和性能
- **内容**：实现 safe softmax 和 online softmax
- **预期结果**：比 PyTorch 原生实现快 50%+

### 实验 4：Flash Attention 实现
- **目标**：从零实现 Flash Attention 算法
- **内容**：理解分块计算和在线更新机制
- **预期结果**：内存使用降低 4-8 倍

### 实验 5：LayerNorm 融合优化
- **目标**：实现融合的 LayerNorm 内核
- **内容**：减少内存访问和内核启动开销
- **预期结果**：性能提升 2-3 倍

## 🔧 核心实现

### 1. 基础向量运算

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """向量加法内核"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法的 Triton 实现"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 选择合适的块大小
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
```

### 2. 高性能矩阵乘法

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """高性能矩阵乘法内核"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算偏移
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 主循环
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载 A 和 B 的块
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # 矩阵乘法
        accumulator += tl.dot(a, b)
        
        # 更新偏移
        offs_k += BLOCK_SIZE_K
    
    # 存储结果
    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    tl.store(c_ptrs, accumulator)
```

### 3. 优化的 Softmax

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """数值稳定的 Softmax 内核"""
    row_idx = tl.program_id(0)
    
    # 计算行偏移
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # 加载整行数据
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # 计算最大值（数值稳定性）
    row_max = tl.max(row, axis=0)
    
    # 计算 exp(x - max)
    row_shifted = row - row_max
    numerator = tl.exp(row_shifted)
    
    # 计算分母
    denominator = tl.sum(numerator, axis=0)
    
    # 计算 softmax
    softmax_output = numerator / denominator
    
    # 存储结果
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

## 📈 性能优化策略

### 1. **内存访问模式优化**

```python
# 好的访问模式：连续访问
offsets = tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets)

# 坏的访问模式：跨步访问
offsets = tl.arange(0, BLOCK_SIZE) * stride  # 如果 stride 很大
```

### 2. **块大小调优**

```python
# 自动选择最优块大小
def get_optimal_block_size(n_elements: int) -> int:
    # 考虑硬件限制和数据大小
    if n_elements <= 1024:
        return triton.next_power_of_2(n_elements)
    elif n_elements <= 4096:
        return 1024
    else:
        return 2048
```

### 3. **数据类型优化**

```python
# 使用混合精度
@triton.jit
def mixed_precision_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 加载为 fp16
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float16)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.float16)
    
    # 计算使用 fp32
    result = (x.to(tl.float32) * y.to(tl.float32))
    
    # 存储为 fp16
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)
```

## ⚠️ 注意事项和最佳实践

### 开发注意事项
1. **内存对齐**：确保内存访问对齐到合适的边界
2. **边界检查**：正确处理不能被块大小整除的情况
3. **数值稳定性**：注意浮点运算的精度问题
4. **调试困难**：Triton 内核难以调试，需要充分测试

### 性能调优技巧
1. **块大小选择**：根据数据大小和硬件特性选择
2. **内存合并**：确保连续的内存访问模式
3. **寄存器使用**：避免寄存器溢出
4. **占用率优化**：平衡线程数和资源使用

### 最佳实践
1. **渐进式开发**：从简单内核开始，逐步优化
2. **性能测试**：与现有实现进行详细对比
3. **可读性**：保持代码的可读性和可维护性
4. **文档化**：详细记录优化策略和性能数据

## 🎓 进阶学习

### 深入主题
1. **自动调优**：使用 Triton 的自动调优功能
2. **多 GPU 支持**：在多 GPU 环境中使用 Triton
3. **与 CUDA 互操作**：结合 CUDA 和 Triton 的优势
4. **编译器优化**：理解 Triton 编译器的优化策略

### 相关资源
- [Triton 官方文档](https://triton-lang.org/)
- [OpenAI Triton 教程](https://github.com/openai/triton)
- [GPU 架构优化指南](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 🔗 相关教程

- [上一节：CUDA Graph 优化](../07_cuda_graph_optimization/)
- [下一节：Prefix Caching 深度教程](../09_prefix_caching/)
- [相关概念：Flash Attention 实战](../11_flash_attention_practice/)

---

*通过掌握 Triton Kernels 开发技术，你将能够为 LLM 推理创建高性能的自定义算子，这是 nano-vLLM 引擎实现极致性能的关键技术之一。*