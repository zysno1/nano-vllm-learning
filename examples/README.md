# 📚 学习示例

> 基于费曼学习法重新组织的学习示例，从基础认知到实践应用，循序渐进

## 🎯 学习目标

通过这些示例，你将：
- 🔧 掌握 nano-vLLM 的完整使用方法
- 🧠 深入理解核心技术原理
- 💡 具备解决实际问题的能力
- 🚀 能够构建生产级推理服务

---

## 📚 结构化学习路径

### 🌟 基础认知阶段
**目标**: 建立基本概念，掌握核心使用方法

#### 📁 [basic/](basic/) - 基础示例演练
```bash
# 快速体验 (5分钟)
python basic/00-quick-start/hello_vllm.py

# 入门指南 (30分钟)  
cd basic/01-getting-started && python model_loading.py
cd basic/01-getting-started && python basic_inference.py

# 核心技术理解 (45分钟)
cd basic/02-llm-engine && python main.py
cd ../03-paged-attention && python main.py
```

**学习重点**:
- ✅ 环境配置和基本使用
- ✅ 模型加载与配置
- ✅ 理解 PagedAttention 内存优化
- ✅ 掌握 LLM 引擎工作原理
- ✅ 学会基本参数调优

---

### 🚀 实践应用阶段  
**目标**: 通过复杂场景，深化技术理解

#### 📁 [advanced/](advanced/) - 真实场景进阶案例

##### 🏗️ 系统架构与集成
```bash
# 调度器原理 (45分钟)
cd advanced/04-scheduler && python main.py

# 完整推理流程 (60分钟)
cd ../05-complete-inference && python main.py

# 端到端演示 (30分钟)
cd ../06-end-to-end-demo && python main.py
```

##### ⚡ 性能优化技术
```bash
# 性能优化 (90分钟)
cd advanced/performance-optimization && python cuda_graph_basics.py
cd advanced/performance-optimization && python compile_basics.py

# 内核优化 (120分钟)
cd ../kernel-optimization && python triton_basics.py
cd ../kernel-optimization && python flash_attention_basics.py
```

##### 🔧 高级特性
```bash
# 前缀缓存 (45分钟)
cd advanced/09-prefix-caching && python main.py

# 多GPU并行 (60分钟)
cd ../12-multi-gpu-tensor-parallel && python main.py

# 分布式推理 (75分钟)
cd ../13-distributed-inference && python main.py
```

---

## 🧪 学习验证

### 快速自测
1. **基础概念**: 能否解释 PagedAttention 的工作原理？
2. **系统理解**: 如何设计一个高效的调度策略？
3. **实践能力**: 能否独立构建推理服务？

### 实践挑战
```bash
# 性能对比实验
python basic/01-getting-started/batch_inference.py --batch_size 8
python basic/01-getting-started/batch_inference.py --batch_size 16

# 压力测试
cd advanced/06-end-to-end-demo
python test_nano_vllm.py --stress_test --concurrent_users 50
```

---

## 📁 目录结构

```
examples/
├── basic/                         # 🌟 基础认知阶段
│   ├── 00-quick-start/            # 🚀 快速开始示例
│   ├── 01-getting-started/        # 📚 入门指南 (合并基础使用和模型加载)
│   ├── 02-llm-engine/             # ⚙️ LLM引擎核心
│   └── 03-paged-attention/        # 🧠 PagedAttention技术
└── advanced/                      # 🚀 实践应用阶段
    ├── 02-step-by-step-tutorials/  # 📝 分步骤教程
    ├── 03-code-templates/          # 🛠️ 代码模板
    ├── 04-scheduler/               # 📊 智能调度器
    ├── 05-complete-inference/      # 🔄 完整推理流程
    ├── 06-end-to-end-demo/         # 🎯 端到端演示
    ├── 09-prefix-caching/          # 🔄 前缀缓存优化
    ├── 12-multi-gpu-tensor-parallel/ # 🔗 多GPU张量并行
    ├── 13-distributed-inference/   # 🌐 分布式推理
    ├── performance-optimization/   # ⚡ 性能优化技术 (CUDA Graph + Torch Compile)
    └── kernel-optimization/        # 🔧 内核优化技术 (Triton + Flash Attention)
```

## 🚀 快速开始示例 (00-quick-start)

**目标用户：** 完全没有 vLLM 经验的初学者  
**学习时间：** 30-60 分钟  
**前置要求：** Python 基础知识

### 📋 示例列表

| 文件名 | 功能描述 | 难度 | 时间 |
|--------|----------|------|------|
| `environment_check.py` | 环境检查和配置验证 | ⭐ | 5分钟 |
| `hello_vllm.py` | 第一个推理示例 | ⭐ | 10分钟 |
| `basic_concepts.py` | 核心概念演示 | ⭐⭐ | 20分钟 |

### 🎯 学习目标
- 验证环境配置是否正确
- 理解 vLLM 的基本工作流程
- 掌握核心概念：Tokenization、Batching、Sampling

### 💡 使用建议
```bash
cd examples/basic/00-quick-start

# 1. 首先检查环境
python environment_check.py

# 2. 运行第一个示例
python hello_vllm.py

# 3. 学习核心概念
python basic_concepts.py
```

## 📚 入门指南 (01-getting-started)

**目标用户：** 已完成快速开始的学习者  
**学习时间：** 2-3 小时  
**前置要求：** 完成快速开始示例

### 📋 示例列表

| 文件名 | 功能描述 | 难度 | 时间 |
|--------|----------|------|------|
| `model_loading.py` | 模型加载与配置 | ⭐⭐ | 25分钟 |
| `basic_inference.py` | 基础推理操作 | ⭐⭐ | 20分钟 |
| `batch_inference.py` | 批量文本处理 | ⭐⭐⭐ | 30分钟 |
| `streaming_inference.py` | 流式文本生成 | ⭐⭐⭐ | 25分钟 |
| `parameter_tuning.py` | 参数调优实践 | ⭐⭐⭐ | 35分钟 |
| `error_handling.py` | 错误处理最佳实践 | ⭐⭐ | 20分钟 |

### 🎯 学习目标
- 掌握模型加载和配置方法
- 理解批处理的优势和使用场景
- 学会参数调优技巧
- 具备基本的错误处理能力

### 💡 推荐学习顺序
```bash
cd examples/basic/01-getting-started

# 按以下顺序学习，效果最佳
python model_loading.py         # 1. 模型加载
python basic_inference.py       # 2. 基础推理
python batch_inference.py       # 3. 批量处理
python streaming_inference.py   # 4. 流式生成
python parameter_tuning.py      # 5. 参数调优
python error_handling.py        # 6. 错误处理
```

## 🔧 LLM 引擎 (02-llm-engine)

**目标用户：** 希望深入理解 vLLM 引擎机制的学习者  
**学习时间：** 2-3 小时  
**前置要求：** 完成入门指南

### 📋 示例列表

| 文件名 | 功能描述 | 难度 | 时间 |
|--------|----------|------|------|
| `engine_initialization.py` | 引擎初始化详解 | ⭐⭐⭐ | 30分钟 |
| `request_processing.py` | 请求处理机制 | ⭐⭐⭐ | 35分钟 |
| `scheduler_demo.py` | 调度器工作原理 | ⭐⭐⭐⭐ | 40分钟 |
| `memory_management.py` | 内存管理策略 | ⭐⭐⭐⭐ | 45分钟 |

### 🎯 学习目标
- 理解 LLM 引擎的核心架构
- 掌握请求调度和处理机制
- 学会内存管理和优化策略
- 培养引擎级别的调试能力

## 📄 分页注意力 (03-paged-attention)

**目标用户：** 关注内存优化技术的开发者  
**学习时间：** 1.5-2 小时  
**前置要求：** 完成 LLM 引擎学习

### 📋 示例列表

| 文件名 | 功能描述 | 难度 | 时间 |
|--------|----------|------|------|
| `paged_attention_basics.py` | 分页注意力基础 | ⭐⭐⭐ | 25分钟 |
| `memory_efficiency.py` | 内存效率分析 | ⭐⭐⭐ | 30分钟 |
| `block_management.py` | 块管理机制 | ⭐⭐⭐⭐ | 35分钟 |
| `performance_comparison.py` | 性能对比测试 | ⭐⭐⭐ | 30分钟 |

### 🎯 学习目标
- 理解分页注意力的工作原理
- 掌握内存块管理机制
- 学会内存效率优化技巧
- 对比传统注意力机制的优势

## 🛠️ 代码模板 (03-code-templates)

**目标用户：** 需要快速开发应用的开发者  
**学习时间：** 1-2 小时  
**前置要求：** 完成前面的学习阶段

### 📋 模板列表

| 文件名 | 模板用途 | 特性 | 适用场景 |
|--------|----------|------|----------|
| `basic_inference_template.py` | 基础推理模板 | 完整注释、错误处理 | 单次推理应用 |
| `batch_processing_template.py` | 批处理模板 | 性能优化、进度显示 | 大量文本处理 |
# 高级示例 (Advanced Examples)

## 📝 分步骤教程 (02-step-by-step-tutorials)

**目标用户：** 希望深入理解每个步骤的学习者  
**学习时间：** 3-4 小时  
**前置要求：** 完成基础示例学习

### 📋 教程列表

| 文件名 | 教程主题 | 难度 | 时间 |
|--------|----------|------|------|
| `tutorial_01_first_inference.py` | 第一次推理详解 | ⭐⭐ | 45分钟 |
| `tutorial_02_parameter_exploration.py` | 参数探索实践 | ⭐⭐⭐ | 50分钟 |
| `tutorial_03_batch_processing.py` | 批处理深入学习 | ⭐⭐⭐ | 55分钟 |
| `tutorial_04_performance_tuning.py` | 性能调优指南 | ⭐⭐⭐⭐ | 60分钟 |

### 🎯 学习目标
- 深入理解每个操作步骤
- 掌握参数调优的系统方法
- 学会性能分析和优化
- 培养独立解决问题的能力

## 🛠️ 代码模板 (03-code-templates)

**目标用户：** 需要快速开发应用的开发者  
**学习时间：** 1-2 小时  
**前置要求：** 完成前面的学习阶段

### 📋 模板列表

| 文件名 | 模板用途 | 特性 | 适用场景 |
|--------|----------|------|----------|
| `basic_inference_template.py` | 基础推理模板 | 完整注释、错误处理 | 单次推理应用 |
| `batch_processing_template.py` | 批处理模板 | 性能优化、进度显示 | 大量文本处理 |
| `streaming_template.py` | 流式生成模板 | 实时输出、用户交互 | 聊天应用 |
| `api_service_template.py` | API 服务模板 | RESTful API、并发处理 | Web 服务 |

### 🎯 模板特色
- **开箱即用**：可直接运行的完整代码
- **高度可配置**：通过配置文件轻松定制
- **生产就绪**：包含完善的错误处理和日志
- **性能优化**：内置性能监控和优化建议

## 🔧 调度器 (04-scheduler)

**目标用户：** 深入理解调度机制的开发者  
**学习时间：** 2-3 小时  
**前置要求：** 完成基础示例

### 📋 示例列表

| 文件名 | 功能描述 | 难度 | 时间 |
|--------|----------|------|------|
| `scheduler_basics.py` | 调度器基础概念 | ⭐⭐⭐ | 30分钟 |
| `request_scheduling.py` | 请求调度策略 | ⭐⭐⭐⭐ | 45分钟 |
| `priority_management.py` | 优先级管理 | ⭐⭐⭐⭐ | 40分钟 |
| `load_balancing.py` | 负载均衡实现 | ⭐⭐⭐⭐⭐ | 50分钟 |

## 🚀 性能优化 (performance-optimization)

**目标用户：** 关注极致性能的开发者  
**学习时间：** 3-4 小时  
**前置要求：** 完成调度器学习

### 📋 优化技术

| 技术模块 | 功能描述 | 性能提升 | 复杂度 |
|----------|----------|----------|--------|
| **CUDA Graph** | GPU 计算图优化 | 20-40% | ⭐⭐⭐⭐ |
| **Torch Compile** | 编译时优化 | 15-30% | ⭐⭐⭐ |

### 🎯 学习目标
- 掌握 CUDA Graph 优化技术
- 理解 Torch Compile 编译优化
- 学会性能瓶颈分析方法
- 实现生产级性能优化

## ⚡ 内核优化 (kernel-optimization)

**目标用户：** 底层优化专家  
**学习时间：** 4-5 小时  
**前置要求：** 完成性能优化学习

### 📋 优化技术

| 技术模块 | 功能描述 | 性能提升 | 复杂度 |
|----------|----------|----------|--------|
| **Triton Kernels** | 自定义 GPU 内核 | 30-60% | ⭐⭐⭐⭐⭐ |
| **Flash Attention** | 注意力机制优化 | 40-80% | ⭐⭐⭐⭐ |

### 🎯 学习目标
- 掌握 Triton 内核编程
- 理解 Flash Attention 原理
- 学会内存访问模式优化
- 实现自定义高性能内核

## 🎓 学习建议

## 🎓 学习建议

### 📋 推荐学习路径

#### 基础阶段 (1-2 周)
```
00-quick-start → 01-getting-started → 02-llm-engine → 03-paged-attention
```

#### 高级阶段 (2-3 周)
```
02-step-by-step-tutorials → 03-code-templates → 04-scheduler → performance-optimization → kernel-optimization
```

### 💡 学习技巧

#### 1. 动手实践
- 每个示例都要亲自运行
- 尝试修改参数观察效果
- 用自己的数据测试

#### 2. 理解原理
- 阅读代码注释理解每个步骤
- 思考为什么这样实现
- 尝试用不同方法实现相同功能

#### 3. 记录学习
- 记录重要概念和技巧
- 记录遇到的问题和解决方案
- 总结最佳实践

#### 4. 循序渐进
- 不要跳跃式学习
- 确保每个阶段都扎实掌握
- 遇到困难及时查看文档

### 🔧 环境准备

#### 基本要求
```bash
# Python 版本
python >= 3.8

# 内存要求
RAM >= 8GB (推荐 16GB+)

# 存储空间
磁盘空间 >= 10GB
```

#### 依赖安装
```bash
# 安装基础依赖
pip install -r requirements.txt

# 验证安装
cd examples/basic/00-quick-start
python environment_check.py
```

## 🆘 故障排除

### 常见问题

#### 1. 环境问题
```bash
# 检查 Python 版本
python --version

# 检查依赖包
pip list | grep torch

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

#### 2. 内存不足
```python
# 减少批大小
batch_size = 1  # 从默认值减少

# 使用 CPU 模式
device = "cpu"
```

#### 3. 模型加载失败
```python
# 检查模型路径
import os
print(os.path.exists(model_path))

# 使用默认模型
model_name = "gpt2"  # 使用小模型测试
```

### 🔍 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 使用调试模式
```python
# 在代码中添加调试信息
print(f"Debug: {variable_name} = {variable_value}")
```

#### 3. 分步执行
```python
# 将复杂操作分解为多个步骤
# 每个步骤都检查结果
```

## 📚 相关资源

### 📖 文档链接
- [快速开始指南](../docs/00-quick-start/README.md)
- [基础概念](../docs/01-basic-concepts.md)
- [架构设计](../docs/02-architecture.md)

### 🧪 测试用例
- [基础推理测试](../tests/test_basic_inference.py)
- [批处理测试](../tests/test_batch_processing.py)
- [性能测试](../tests/test_performance.py)

### 🔗 外部资源
- [vLLM 官方文档](https://docs.vllm.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch 文档](https://pytorch.org/docs/)

## 🤝 贡献代码

我们欢迎您贡献新的示例代码：

### 📝 贡献指南
1. **代码质量**：遵循项目的代码规范
2. **文档完善**：提供详细的注释和说明
3. **测试验证**：确保代码可以正常运行
4. **实用性**：提供有实际价值的示例

### 🎯 贡献方向
- 新的使用场景示例
- 性能优化技巧
- 错误处理最佳实践
- 集成其他工具的示例

---

**开始您的实践之旅吧！** 记住，最好的学习方法就是动手实践。 🚀