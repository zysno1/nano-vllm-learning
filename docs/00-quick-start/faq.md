# ❓ 常见问题解答 (FAQ)

本文档收集了新手在学习 nano-vLLM 过程中最常遇到的问题和解决方案。

## 🚀 入门问题

### Q1: 什么是 vLLM？我为什么要学习它？

**A:** vLLM 是一个高性能的大语言模型推理引擎。学习它的原因：

- 🚀 **性能优势**：比传统方法快 10-20 倍
- 💰 **成本节约**：更高效的资源利用
- 🏭 **生产就绪**：专为生产环境设计
- 📈 **行业趋势**：AI 应用的核心技术

### Q2: 我需要什么基础知识？

**A:** 建议具备以下基础：

**必需基础：**
- Python 编程 (基础语法、函数、类)
- 基本的机器学习概念

**推荐基础：**
- PyTorch 基础使用
- Transformer 架构了解
- Linux 命令行操作

**不需要：**
- 深度学习专家级知识
- CUDA 编程经验
- 分布式系统经验

### Q3: 学习需要多长时间？

**A:** 根据基础不同，学习时间安排：

```
📅 学习时间规划

🟢 快速上手 (1-2天)
├── 环境配置
├── 运行第一个示例  
└── 理解基本概念

🟡 基础掌握 (1-2周)
├── 核心概念理解
├── 基础API使用
├── 简单性能测试
└── 常见问题解决

🔴 深入精通 (1-2个月)
├── 架构深度理解
├── 性能优化技巧
├── 高级功能使用
└── 生产环境部署
```

## 🛠️ 环境配置问题

### Q4: 安装依赖时出现错误怎么办？

**A:** 常见安装问题解决方案：

**问题1：pip 安装超时**
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**问题2：CUDA 版本不匹配**
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**问题3：权限问题**
```bash
# 使用用户安装
pip install --user -r requirements.txt
```

### Q5: 我没有GPU，可以学习吗？

**A:** 完全可以！CPU 也能运行：

```python
# CPU 推理配置
import torch

# 强制使用CPU
device = torch.device("cpu")
model = model.to(device)

# 使用更小的模型
model_name = "distilgpt2"  # 更适合CPU

# 调整批处理大小
batch_size = 1  # CPU推理建议使用小批次
```

**CPU 学习建议：**
- 使用小模型 (GPT-2, DistilGPT-2)
- 减少序列长度
- 专注理论学习
- 使用云GPU进行实验

### Q6: 内存不够怎么办？

**A:** 内存优化策略：

```python
# 1. 使用半精度
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
)

# 2. 减少批处理大小
batch_size = 1

# 3. 使用梯度检查点
model.gradient_checkpointing_enable()

# 4. 清理缓存
torch.cuda.empty_cache()
```

## 🤖 模型使用问题

### Q7: 如何选择合适的模型？

**A:** 模型选择指南：

| 学习阶段 | 推荐模型 | 显存需求 | 特点 |
|----------|----------|----------|------|
| **入门** | GPT-2 | 1-2GB | 快速加载，适合学习 |
| **进阶** | GPT-2 Medium | 2-4GB | 更好效果，合理资源 |
| **实战** | LLaMA-7B | 8-16GB | 生产级性能 |
| **高级** | LLaMA-13B+ | 16GB+ | 最佳效果 |

### Q8: 生成的文本质量不好怎么办？

**A:** 提升文本质量的方法：

```python
# 1. 调整采样参数
outputs = model.generate(
    inputs,
    max_length=100,
    temperature=0.7,      # 降低随机性
    top_p=0.9,           # 核采样
    top_k=50,            # 限制候选词
    repetition_penalty=1.1,  # 避免重复
    do_sample=True
)

# 2. 改进提示词
prompt = "请详细解释人工智能的发展历程："  # 更具体的提示

# 3. 使用更大的模型
model_name = "microsoft/DialoGPT-medium"
```

### Q9: 推理速度太慢怎么办？

**A:** 性能优化技巧：

```python
# 1. 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 2. 批处理
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# 3. 编译优化
model = torch.compile(model)  # PyTorch 2.0+

# 4. 使用KV缓存
past_key_values = None  # 保存之前的计算结果
```

## 📊 性能测试问题

### Q10: 如何测试模型性能？

**A:** 使用我们提供的测试工具：

```bash
# 基础性能测试
python examples/02-performance-testing/benchmark.py

# 延迟测试
python examples/02-performance-testing/latency_test.py

# 吞吐量测试  
python examples/02-performance-testing/throughput_test.py

# 内存分析
python examples/02-performance-testing/memory_profiling.py
```

### Q11: 性能指标怎么理解？

**A:** 关键性能指标解释：

| 指标 | 含义 | 好的范围 | 影响因素 |
|------|------|----------|----------|
| **延迟 (Latency)** | 单次请求响应时间 | <100ms | 模型大小、硬件 |
| **吞吐量 (Throughput)** | 每秒处理请求数 | >10 req/s | 批处理、并行度 |
| **内存使用** | GPU/CPU内存占用 | <80% | 模型大小、批次 |
| **GPU利用率** | GPU计算资源使用率 | >80% | 批处理效率 |

## 🔧 故障排除问题

### Q12: 程序崩溃了怎么办？

**A:** 系统性故障排除：

```bash
# 1. 运行诊断工具
python examples/06-troubleshooting/diagnostic_tools.py

# 2. 检查系统状态
python examples/06-troubleshooting/system_check.py

# 3. 查看详细错误
python your_script.py 2>&1 | tee error.log
```

**常见崩溃原因：**
- 内存不足 → 减少批次大小
- CUDA错误 → 检查GPU驱动
- 模型文件损坏 → 重新下载
- 依赖版本冲突 → 重建环境

### Q13: 如何调试性能问题？

**A:** 性能调试步骤：

```python
# 1. 启用性能分析
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # 你的推理代码
    outputs = model.generate(inputs)

# 2. 查看分析结果
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 🚀 进阶学习问题

### Q14: 学完基础后应该学什么？

**A:** 进阶学习路径：

```
📈 进阶学习路径

🎯 中级目标
├── 掌握批处理优化
├── 理解内存管理
├── 学会性能调优
└── 部署简单应用

🎯 高级目标  
├── 分布式推理
├── 自定义优化
├── 生产环境部署
└── 贡献开源项目
```

### Q15: 如何跟上技术发展？

**A:** 持续学习建议：

**📚 学习资源：**
- [vLLM 官方文档](https://docs.vllm.ai/)
- [Hugging Face 教程](https://huggingface.co/course)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)

**🔄 实践建议：**
- 定期运行新示例
- 参与开源项目
- 关注技术博客
- 加入技术社区

## 🆘 获取更多帮助

### 遇到文档中没有的问题？

1. **查看故障排除工具**
   ```bash
   python examples/06-troubleshooting/diagnostic_tools.py
   ```

2. **运行系统检查**
   ```bash
   python examples/03-setup-scripts/check_system.py
   ```

3. **查看详细日志**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python your_script.py
   ```

4. **社区求助**
   - GitHub Issues
   - Stack Overflow
   - 技术论坛

---

💡 **记住**：学习是一个过程，遇到问题很正常！每个问题都是学习的机会。

🎯 **建议**：先尝试自己解决，实在不行再求助。这样学习效果最好！

[返回快速开始 ←](README.md) | [查看故障排除指南 →](../04-advanced-topics/troubleshooting.md)