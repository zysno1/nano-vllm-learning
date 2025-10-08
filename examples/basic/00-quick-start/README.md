# 🚀 快速开始示例

欢迎来到 nano-vLLM 的快速开始示例！这里包含了最适合初学者的代码示例。

## 📁 文件说明

### 🎯 核心示例

| 文件 | 难度 | 用时 | 说明 |
|------|------|------|------|
| `hello_vllm.py` | ⭐ | 5分钟 | 您的第一个推理示例 |
| `basic_concepts.py` | ⭐⭐ | 10分钟 | 核心概念演示 |
| `step_by_step_tutorial.py` | ⭐⭐ | 15分钟 | 分步骤详细教程 |

### 🛠️ 辅助工具

| 文件 | 用途 | 说明 |
|------|------|------|
| `environment_check.py` | 环境检查 | 验证安装是否正确 |
| `model_downloader.py` | 模型下载 | 预下载常用模型 |
| `simple_benchmark.py` | 性能测试 | 简单的性能评估 |

## 🎮 快速开始

### 1. 运行第一个示例

```bash
# 进入示例目录
cd examples/00-quick-start

# 运行Hello World示例
python hello_vllm.py
```

**期望输出：**
```
🚀 欢迎使用 nano-vLLM 推理引擎!
📦 正在加载模型...
✅ 模型加载成功!
🎉 生成结果: 人工智能的未来是充满无限可能的...
```

### 2. 检查环境配置

```bash
# 验证环境是否正确配置
python environment_check.py
```

### 3. 学习核心概念

```bash
# 运行概念演示
python basic_concepts.py
```

## 📚 学习路径

### 🎯 推荐顺序

1. **环境检查** → `environment_check.py`
2. **第一个示例** → `hello_vllm.py`  
3. **核心概念** → `basic_concepts.py`
4. **详细教程** → `step_by_step_tutorial.py`
5. **性能测试** → `simple_benchmark.py`

### 🎓 学习目标

完成这些示例后，您将能够：

- ✅ 成功运行 vLLM 推理
- ✅ 理解基本概念和术语
- ✅ 掌握基础API使用
- ✅ 进行简单的性能评估
- ✅ 解决常见问题

## 🔧 故障排除

### 常见问题

**Q: 模型下载很慢？**
```bash
# 使用模型下载工具
python model_downloader.py --model gpt2
```

**Q: 内存不够？**
```python
# 在代码中使用CPU模式
device = "cpu"
torch_dtype = torch.float32
```

**Q: 运行出错？**
```bash
# 检查详细错误信息
python hello_vllm.py 2>&1 | tee error.log
```

### 获取帮助

1. 查看 [FAQ文档](../../docs/00-quick-start/faq.md)
2. 运行环境检查工具
3. 查看错误日志
4. 参考故障排除指南

## 🚀 下一步

完成快速开始后，建议继续学习：

### 📖 理论学习
- [基础概念](../../docs/01-basic-concepts.md) - 理解核心原理
- [架构分析](../../docs/02-architecture.md) - 了解系统设计

### 💻 实践练习  
- [基础使用](../01-basic-usage/) - 更多基础示例
- [性能测试](../02-performance-testing/) - 深入性能分析

### 🎯 进阶应用
- [高级示例](../04-advanced-examples/) - 学习高级功能
- [集成应用](../05-integration-examples/) - 构建实际应用

---

🎉 **准备好开始您的 nano-vLLM 学习之旅了吗？从 `hello_vllm.py` 开始吧！**

💡 **提示**：每个示例都包含详细注释，不要害怕阅读代码！