# 📚 基础使用示例

本目录包含 nano-vLLM 的基础使用示例，适合已经完成快速入门的学习者。

## 📁 目录结构

```
01-basic-usage/
├── README.md                    # 本文件
├── simple_inference.py          # 简单推理示例
├── batch_inference.py           # 批量推理示例
├── streaming_inference.py       # 流式推理示例
├── parameter_tuning.py          # 参数调优示例
├── model_comparison.py          # 模型对比示例
└── error_handling.py            # 错误处理示例
```

## 🎯 学习目标

通过本章节的学习，您将掌握：

1. **基础推理技能**
   - 单次推理的完整流程
   - 批量处理多个请求
   - 流式输出的实现

2. **参数调优技巧**
   - 温度、top-k、top-p 参数的使用
   - 不同场景下的参数选择
   - 性能与质量的平衡

3. **实用技能**
   - 模型选择和比较
   - 错误处理和调试
   - 性能监控和优化

## 🚀 快速开始

### 前置条件

确保您已经：
- ✅ 完成了 `examples/00-quick-start/` 中的所有示例
- ✅ 环境检查通过 (`environment_check.py`)
- ✅ 理解了基本概念 (`basic_concepts.py`)

### 推荐学习顺序

1. **简单推理** (`simple_inference.py`)
   ```bash
   python simple_inference.py
   ```
   学习最基础的推理流程

2. **批量推理** (`batch_inference.py`)
   ```bash
   python batch_inference.py
   ```
   掌握批量处理技巧

3. **流式推理** (`streaming_inference.py`)
   ```bash
   python streaming_inference.py
   ```
   实现实时输出效果

4. **参数调优** (`parameter_tuning.py`)
   ```bash
   python parameter_tuning.py
   ```
   学习参数调优技巧

5. **模型对比** (`model_comparison.py`)
   ```bash
   python model_comparison.py
   ```
   比较不同模型的效果

6. **错误处理** (`error_handling.py`)
   ```bash
   python error_handling.py
   ```
   学习调试和错误处理

## 💡 使用技巧

### 性能优化建议

1. **批量处理**
   - 尽可能使用批量推理
   - 合理设置批次大小
   - 注意内存使用情况

2. **参数选择**
   - 创意任务：较高温度 (0.8-1.2)
   - 事实性任务：较低温度 (0.1-0.5)
   - 平衡质量和多样性：top-p=0.9

3. **内存管理**
   - 及时清理不需要的变量
   - 使用 `torch.cuda.empty_cache()`
   - 监控GPU内存使用

### 常见问题解决

1. **内存不足**
   ```python
   # 减少批次大小
   batch_size = 2  # 而不是 8
   
   # 使用半精度
   model = model.half()
   
   # 清理缓存
   torch.cuda.empty_cache()
   ```

2. **推理速度慢**
   ```python
   # 使用GPU
   device = "cuda" if torch.cuda.is_available() else "cpu"
   
   # 启用编译优化
   model = torch.compile(model)
   
   # 使用批量推理
   results = model.generate(batch_inputs)
   ```

3. **输出质量不佳**
   ```python
   # 调整采样参数
   outputs = model.generate(
       inputs,
       temperature=0.7,      # 调整随机性
       top_p=0.9,           # 核采样
       top_k=50,            # 限制候选词
       repetition_penalty=1.1  # 避免重复
   )
   ```

## 📖 相关文档

- [基础概念文档](../../docs/01-basic-concepts/)
- [架构设计文档](../../docs/02-architecture/)
- [常见问题解答](../../docs/00-quick-start/faq.md)

## 🔗 下一步学习

完成本章节后，建议继续学习：

1. **性能测试** (`examples/02-performance-testing/`)
   - 基准测试
   - 性能分析
   - 优化策略

2. **高级用法** (`examples/03-advanced-usage/`)
   - 自定义模型
   - 插件开发
   - 分布式推理

3. **实际项目** (`examples/04-real-projects/`)
   - 聊天机器人
   - 文本生成API
   - 批量处理服务

## ❓ 获取帮助

如果在学习过程中遇到问题：

1. 查看 [FAQ文档](../../docs/00-quick-start/faq.md)
2. 运行 `environment_check.py` 检查环境
3. 查看错误处理示例 (`error_handling.py`)
4. 参考相关文档和代码注释

---

💡 **提示**: 每个示例都包含详细的注释和说明，建议仔细阅读代码中的注释来理解实现细节。