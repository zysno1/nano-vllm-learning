# 🚀 基础示例演练

> 基于费曼学习法的核心实践验证

## 🎯 学习目标

- 🔧 掌握 nano-vLLM 基本使用方法
- 🧠 理解核心技术概念的实际应用
- 💡 建立对 LLM 推理优化的直观认知

---

## 📚 费曼学习路径

### 🧠 第一步：概念理解 (Understand)
**目标**: 建立基本认知，验证环境配置

#### 📁 [00-quick-start/](00-quick-start/) - 快速体验
```bash
# 环境检查
python 00-quick-start/environment_check.py

# 第一个推理示例
python 00-quick-start/hello_vllm.py
```

### 🛠️ 第二步：实践应用 (Practice)
**目标**: 掌握核心技术的实际应用

#### 📁 [01-getting-started/](01-getting-started/) - 模型加载机制
```bash
cd 01-getting-started/ && python main.py
```
**核心学习点**: 模型加载流程、配置参数、内存管理

#### 📁 [02-llm-engine/](02-llm-engine/) - LLM引擎核心
```bash
cd 02-llm-engine/ && python main.py
```
**核心学习点**: 引擎初始化、请求处理、性能监控

#### 📁 [03-paged-attention/](03-paged-attention/) - PagedAttention技术
```bash
cd 03-paged-attention/ && python main.py
```
**核心学习点**: 分页内存管理、内存利用率优化

### 🔍 第三步：回顾简化 (Simplify)
**目标**: 发现理解盲点，简化复杂概念

#### 参数调优实验
```bash
# 对比不同参数配置的性能
python 01-basic-usage/simple_inference.py --max_tokens 100
python 01-basic-usage/simple_inference.py --max_tokens 500
```

#### 快速自测
1. **模型加载**: 能否解释模型加载的主要步骤？
2. **内存管理**: PagedAttention 如何提高内存利用率？
3. **批处理**: 批量推理相比单次推理有什么优势？

### 🎯 第四步：系统整理 (Organize)
**目标**: 系统掌握，准备进阶学习

#### 实践挑战
1. **修改参数**: 调整 `max_tokens`, `temperature` 等参数
2. **性能对比**: 比较不同批次大小的性能差异
3. **错误处理**: 观察错误处理机制

---

## 🚀 下一步学习

完成基础示例后，推荐学习路径：
1. **深入理解** → [核心概念](../../docs/concepts.md) - 重温核心概念
2. **进阶实践** → [高级示例](../advanced/) - 挑战复杂场景
3. **系统架构** → [架构设计](../../docs/architecture.md) - 理解整体设计

---

## 💡 费曼学习建议

1. **理解**: 运行每个示例，观察输出结果
2. **简化**: 用自己的话解释每个技术的作用
3. **实践**: 修改代码参数，验证你的理解
4. **验证**: 向他人解释你学到的概念

> 记住：基础扎实，进阶才能游刃有余！