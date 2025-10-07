# NanoVLLM 端到端演示项目

这是一个完整的 NanoVLLM 端到端演示项目，展示了如何构建一个简化版的 vLLM 推理系统。本项目从零开始实现了大语言模型推理系统的核心组件，包括内存管理、请求调度、API服务等。

## 🎯 学习目标

通过这个项目，你将深入学习到：

### 核心技术
1. **模型加载与管理**：Transformers模型的加载、配置和优化
2. **内存管理**：PagedAttention 和 KV Cache 的实现原理与优化
3. **请求调度**：批处理、抢占式调度和资源分配策略
4. **并发处理**：异步请求处理和线程安全设计
5. **性能优化**：内存池、缓存策略和计算优化

### 系统设计
6. **API 设计**：RESTful API 接口设计和最佳实践
7. **监控体系**：性能指标收集、健康检查和系统监控
8. **错误处理**：异常处理、重试机制和故障恢复
9. **配置管理**：多环境配置和参数调优
10. **测试框架**：单元测试、集成测试和性能测试

### 工程实践
11. **代码架构**：模块化设计和依赖注入
12. **日志系统**：结构化日志和调试工具
13. **部署运维**：容器化部署和服务管理
14. **文档规范**：API文档和使用指南

## 📁 项目结构

```
06_end_to_end_demo/
├── README.md                 # 项目说明文档（本文件）
├── nano_vllm.py             # 🔥 核心推理引擎实现
├── config.py                # ⚙️  配置管理和环境适配
├── utils.py                 # 🛠️  工具函数和监控组件
├── demo.py                  # 🎮 交互式演示脚本
├── api_server.py            # 🌐 FastAPI 服务器实现
├── client_examples.py       # 🐍 Python 客户端示例
├── curl_examples.sh         # 📡 Curl 命令行示例
├── test_nano_vllm.py        # 🧪 完整测试套件
└── requirements.txt         # 📦 项目依赖列表
```

## 🚀 核心组件介绍

### 1. NanoVLLM 核心引擎 (`nano_vllm.py`)
- **Block & BlockTable**: KV Cache 内存块管理
- **BlockAllocator**: 内存分配器，支持引用计数和Copy-on-Write
- **PagedAttentionEngine**: PagedAttention 实现，优化内存使用
- **Scheduler**: 请求调度器，支持抢占式调度和批处理
- **MetricsCollector**: 性能指标收集和分析
- **NanoVLLM**: 主引擎类，整合所有组件

### 2. 配置系统 (`config.py`)
- **多环境配置**: DEV、TEST、PROD等预设配置
- **自动适配**: 根据硬件资源自动调整参数
- **参数验证**: 配置参数的合法性检查
- **性能优化**: 针对不同场景的优化配置

### 3. 工具组件 (`utils.py`)
- **Logger**: 彩色日志系统，支持文件轮转
- **MetricsCollector**: 详细的性能指标收集
- **HealthChecker**: 系统健康检查和告警
- **SystemMonitor**: 实时系统资源监控

### 4. API服务 (`api_server.py`)
- **RESTful API**: 标准的HTTP接口设计
- **流式响应**: 支持Server-Sent Events
- **批量处理**: 高效的批量推理接口
- **错误处理**: 完善的异常处理机制

## 📋 详细功能特性

### 🔧 内存管理
- **PagedAttention**: 分页式注意力机制，减少内存碎片
- **KV Cache优化**: 高效的键值缓存管理
- **Copy-on-Write**: 写时复制机制，节省内存
- **内存池**: 预分配内存池，减少动态分配开销
- **引用计数**: 自动内存回收和泄漏检测

### 📊 请求调度
- **批处理**: 动态批处理，提高吞吐量
- **抢占式调度**: 支持高优先级请求抢占
- **公平调度**: 防止请求饥饿，保证公平性
- **资源感知**: 根据内存和计算资源动态调度
- **队列管理**: 等待、运行、交换队列的智能管理

### 🌐 API接口
- **单次生成**: `/v1/generate` - 单个文本生成
- **批量生成**: `/v1/generate/batch` - 批量文本生成
- **流式生成**: `/v1/generate/stream` - 实时流式输出
- **系统指标**: `/v1/metrics` - 性能指标查询
- **健康检查**: `/v1/health` - 系统健康状态
- **模型信息**: `/v1/models` - 可用模型列表

### 📈 监控体系
- **性能指标**: 延迟、吞吐量、错误率等
- **资源监控**: CPU、内存、GPU使用率
- **健康检查**: 系统组件状态检查
- **告警机制**: 异常情况自动告警
- **历史数据**: 指标历史记录和趋势分析

## 🛠️ 安装和配置

### 1. 环境要求
- **Python**: 3.8+ (推荐 3.9+)
- **操作系统**: Linux, macOS, Windows
- **硬件**: 
  - CPU: 4核心以上
  - 内存: 8GB+ (推荐16GB+)
  - GPU: 可选，支持CUDA的NVIDIA GPU

### 2. 安装依赖

```bash
# 克隆项目（如果还没有）
cd nano-vllm-learning/examples/06_end_to_end_demo

# 安装依赖
pip install -r requirements.txt

# 或者使用conda环境
conda create -n nano-vllm python=3.9
conda activate nano-vllm
pip install -r requirements.txt
```

### 3. 配置检查

```bash
# 检查系统配置
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
"
```

## 🚀 快速开始

### 方式一：基础演示

```bash
# 运行基础演示
python demo.py

# 运行特定演示场景
python demo.py --demo basic          # 基础生成
python demo.py --demo batch          # 批量处理
python demo.py --demo benchmark      # 性能测试
python demo.py --demo interactive    # 交互式聊天
```

### 方式二：API服务器

```bash
# 启动API服务器
python api_server.py

# 自定义配置启动
python api_server.py --host 0.0.0.0 --port 8000 --config prod

# 后台运行
nohup python api_server.py > server.log 2>&1 &
```

### 方式三：客户端调用

```bash
# Python客户端示例
python client_examples.py --server http://localhost:8000

# 使用curl命令
./curl_examples.sh

# 运行特定curl示例
./curl_examples.sh basic            # 基础生成
./curl_examples.sh stream           # 流式生成
./curl_examples.sh batch            # 批量处理
```

## 📖 使用指南

### 1. 基础使用

#### 单次文本生成
```python
from nano_vllm import NanoVLLM, GenerationParams

# 创建引擎实例
engine = NanoVLLM.create_instance()

# 生成文本
response = engine.generate(
    prompt="解释一下什么是人工智能",
    params=GenerationParams(
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
)

print(f"生成结果: {response.text}")
print(f"生成时间: {response.generation_time:.2f}s")
```

#### 批量处理
```python
# 批量生成
prompts = [
    "什么是机器学习？",
    "深度学习的原理是什么？",
    "神经网络如何工作？"
]

responses = engine.generate_batch(prompts)
for i, response in enumerate(responses):
    print(f"问题{i+1}: {response.text}")
```

### 2. API服务使用

#### 启动服务
```python
# 使用默认配置
python api_server.py

# 使用生产环境配置
python api_server.py --config prod --workers 4

# 自定义参数
python api_server.py --host 0.0.0.0 --port 8080 --log-level info
```

#### API调用示例

**单次生成**
```bash
curl -X POST "http://localhost:8000/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "解释量子计算的基本原理",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**流式生成**
```bash
curl -X POST "http://localhost:8000/v1/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "写一首关于春天的诗",
    "max_tokens": 100
  }'
```

**批量生成**
```bash
curl -X POST "http://localhost:8000/v1/generate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "什么是区块链？",
      "人工智能的发展历史",
      "云计算的优势"
    ],
    "max_tokens": 150
  }'
```

### 3. 配置管理

#### 使用预设配置
```python
from config import get_config

# 开发环境配置
dev_config = get_config("DEV")

# 生产环境配置
prod_config = get_config("PROD_THROUGHPUT")

# 创建引擎
engine = NanoVLLM.create_instance(config=prod_config)
```

#### 自定义配置
```python
from config import create_custom_config

# 创建自定义配置
custom_config = create_custom_config(
    model_name="gpt2-medium",
    max_model_len=2048,
    block_size=32,
    max_num_seqs=64,
    gpu_memory_utilization=0.8
)

engine = NanoVLLM.create_instance(config=custom_config)
```

### 4. 性能监控

#### 获取系统指标
```python
# 获取性能指标
metrics = engine.get_metrics()
print(f"平均延迟: {metrics.avg_latency:.2f}ms")
print(f"吞吐量: {metrics.throughput:.2f} tokens/s")
print(f"内存使用: {metrics.memory_usage:.1f}%")
```

#### API监控
```bash
# 获取系统指标
curl http://localhost:8000/v1/metrics

# 健康检查
curl http://localhost:8000/v1/health
```

### 5. 测试和调试

#### 运行测试套件
```bash
# 运行所有测试
python test_nano_vllm.py

# 运行特定测试
python -m pytest test_nano_vllm.py::TestNanoVLLM::test_basic_generation -v

# 运行性能测试
python -m pytest test_nano_vllm.py::TestPerformance -v --tb=short
```

#### 调试模式
```python
# 启用调试日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用调试配置
debug_config = get_config("DEV")
debug_config.enable_debug = True
debug_config.log_level = "DEBUG"

engine = NanoVLLM.create_instance(config=debug_config)
```

## ⚡ 性能优化建议

### GPU环境优化
```python
# GPU优化配置
gpu_config = get_config("PROD_THROUGHPUT")
gpu_config.gpu_memory_utilization = 0.85  # 提高GPU内存使用率
gpu_config.max_num_seqs = 128             # 增加并发序列数
gpu_config.enable_chunked_prefill = True  # 启用分块预填充
gpu_config.max_num_batched_tokens = 8192  # 增加批处理token数

engine = NanoVLLM.create_instance(config=gpu_config)
```

### CPU环境优化
```python
# CPU优化配置
cpu_config = get_config("CPU")
cpu_config.max_num_seqs = 32              # 适中的并发数
cpu_config.block_size = 16                # 较小的块大小
cpu_config.enable_prefix_caching = True   # 启用前缀缓存

engine = NanoVLLM.create_instance(config=cpu_config)
```

### 内存优化
```python
# 内存优化配置
memory_config = get_config("MEMORY_OPTIMIZED")
memory_config.swap_space = 8              # 启用交换空间
memory_config.cpu_offload_gb = 4          # CPU卸载内存
memory_config.enable_lora = False         # 禁用LoRA以节省内存

engine = NanoVLLM.create_instance(config=memory_config)
```

## 🔧 常见问题解决

### 1. 内存不足
```bash
# 问题：CUDA out of memory
# 解决方案：
# 1. 减少gpu_memory_utilization
# 2. 降低max_num_seqs
# 3. 使用CPU模式
python demo.py --config CPU
```

### 2. 模型加载失败
```bash
# 问题：模型文件未找到
# 解决方案：
# 1. 检查模型路径
# 2. 下载模型文件
# 3. 使用在线模型
export HF_HOME=/path/to/huggingface/cache
python demo.py --model gpt2
```

### 3. API服务无响应
```bash
# 问题：服务启动失败
# 解决方案：
# 1. 检查端口占用
lsof -i :8000

# 2. 查看日志
tail -f server.log

# 3. 使用不同端口
python api_server.py --port 8080
```

### 4. 性能问题
```python
# 问题：生成速度慢
# 解决方案：
# 1. 启用批处理
config.enable_continuous_batching = True

# 2. 优化调度策略
config.scheduler_policy = "fcfs"  # 先来先服务

# 3. 调整块大小
config.block_size = 32  # 增加块大小
```

## 🚀 扩展建议

### 1. 添加新模型支持
```python
# 在config.py中添加新模型配置
SUPPORTED_MODELS = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium", 
    "gpt2-large": "gpt2-large",
    "your-model": "path/to/your/model"  # 添加自定义模型
}
```

### 2. 实现自定义调度策略
```python
# 继承Scheduler类实现自定义调度
class CustomScheduler(Scheduler):
    def _select_requests_to_schedule(self):
        # 实现自定义调度逻辑
        pass
```

### 3. 添加新的API端点
```python
# 在api_server.py中添加新端点
@app.post("/v1/custom/endpoint")
async def custom_endpoint(request: CustomRequest):
    # 实现自定义功能
    pass
```

### 4. 集成外部服务
```python
# 添加数据库支持
# 添加缓存系统
# 集成监控系统
# 添加认证授权
```

## 📚 学习路径

### 初级阶段（1-2周）
1. **理解基础概念**
   - 阅读 `nano_vllm.py` 核心代码
   - 运行基础演示 `demo.py`
   - 学习配置系统 `config.py`

2. **动手实践**
   - 修改生成参数
   - 尝试不同模型
   - 观察性能指标

### 中级阶段（2-4周）
1. **深入系统架构**
   - 理解内存管理机制
   - 学习请求调度算法
   - 掌握API设计原理

2. **性能优化**
   - 调优配置参数
   - 分析性能瓶颈
   - 实现缓存策略

### 高级阶段（4-8周）
1. **系统扩展**
   - 添加新功能模块
   - 实现分布式部署
   - 集成监控告警

2. **生产部署**
   - 容器化部署
   - 负载均衡配置
   - 故障恢复机制

## 📞 技术支持

### 问题反馈
- 在项目中创建Issue
- 提供详细的错误信息和环境配置
- 包含复现步骤和期望结果

### 贡献代码
- Fork项目并创建分支
- 遵循代码规范和测试要求
- 提交Pull Request并描述改动

### 学习资源
- [Transformers文档](https://huggingface.co/docs/transformers)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [FastAPI文档](https://fastapi.tiangolo.com/)
- [vLLM论文](https://arxiv.org/abs/2309.06180)

---

🎉 **恭喜！** 你已经完成了一个完整的大语言模型推理系统的学习和实践。这个项目涵盖了从底层内存管理到上层API设计的全栈技术，为你深入理解和开发LLM系统奠定了坚实的基础。

继续探索和优化，让你的NanoVLLM系统更加强大和高效！ 🚀

## 🔧 配置选项

系统支持多种配置模式：

- `dev` - 开发模式（默认）
- `test` - 测试模式
- `prod_throughput` - 生产模式（高吞吐量）
- `prod_latency` - 生产模式（低延迟）
- `benchmark` - 基准测试模式

使用环境变量指定配置：

```bash
export NANO_VLLM_CONFIG=prod_throughput
python demo.py
```

## 📊 性能监控

系统内置了完整的性能监控功能：

### 实时指标

- **吞吐量**：tokens/秒，requests/秒
- **延迟**：平均延迟，P95延迟，P99延迟
- **内存使用**：GPU内存，CPU内存，KV Cache使用率
- **队列状态**：等待队列，运行队列，交换队列
- **错误率**：请求成功率，错误类型分布

### 指标查看

```bash
# 查看实时指标
curl http://localhost:8000/metrics

# 查看健康状态
curl http://localhost:8000/health
```

## 🧪 测试

运行完整的测试套件：

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_nano_vllm.py -v

# 运行性能测试
python -m pytest tests/test_performance.py -v
```

## 🔍 调试模式

启用调试模式获取详细日志：

```bash
export NANO_VLLM_DEBUG=true
export NANO_VLLM_LOG_LEVEL=DEBUG
python demo.py
```

调试模式将显示：
- 详细的请求处理流程
- 内存分配和释放过程
- 调度决策过程
- 性能分析数据

## 📈 性能优化建议

### GPU环境

1. **内存优化**
   - 调整 `gpu_memory_utilization` 到 0.9-0.95
   - 启用 `enable_swap` 处理内存不足
   - 使用 `torch.float16` 减少内存使用

2. **吞吐量优化**
   - 启用 `enable_continuous_batching`
   - 增大 `max_batch_size` 和 `max_num_seqs`
   - 启用 `enable_chunked_prefill`

3. **延迟优化**
   - 使用 `scheduler_policy="sjf"`
   - 减小 `engine_loop_interval`
   - 启用 `preemption_mode="recompute"`

### CPU环境

1. **资源配置**
   - 使用 `torch.float32` 保证精度
   - 减小 `max_num_seqs` 和 `max_batch_size`
   - 增加 `num_cpu_blocks`

2. **性能调优**
   - 设置合适的线程数
   - 启用CPU优化选项
   - 使用较小的模型

## 🚨 常见问题

### 内存不足

```
OutOfMemoryError: CUDA out of memory
```

**解决方案**：
1. 减小 `max_num_seqs` 或 `max_batch_size`
2. 降低 `gpu_memory_utilization`
3. 启用内存交换 `enable_swap=True`
4. 使用更小的模型

### 模型加载失败

```
OSError: Can't load tokenizer for 'model_name'
```

**解决方案**：
1. 检查模型名称是否正确
2. 确保网络连接正常
3. 检查Hugging Face访问权限
4. 尝试使用本地模型路径

### API服务器启动失败

```
Address already in use
```

**解决方案**：
1. 更改端口号：`python api_server.py --port 8001`
2. 杀死占用端口的进程
3. 等待端口释放

## 🎯 扩展建议

这个演示项目可以进一步扩展：

1. **功能扩展**
   - 添加流式输出支持
   - 实现多模态输入
   - 添加插件系统

2. **性能优化**
   - 集成Flash Attention
   - 实现模型并行
   - 添加推测解码

3. **部署优化**
   - Docker容器化
   - Kubernetes部署
   - 负载均衡

4. **监控增强**
   - Prometheus集成
   - Grafana仪表板
   - 告警系统

## 📝 学习路径

建议按以下顺序学习：

1. **基础理解**：先运行 `demo.py` 了解整体流程
2. **API使用**：启动服务器，尝试不同的客户端调用
3. **配置调优**：尝试不同配置，观察性能变化
4. **代码分析**：深入阅读 `nano_vllm.py` 理解实现细节
5. **扩展实践**：基于这个框架实现自己的功能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个演示项目！

## 📄 许可证

MIT License - 详见LICENSE文件