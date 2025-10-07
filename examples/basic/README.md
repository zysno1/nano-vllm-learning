# 🚀 基础示例演练

> 基于费曼学习法：从简单概念开始，逐步构建完整理解

## 🎯 学习目标

通过这些基础示例，你将：
- 🔧 掌握 nano-vLLM 的基本使用方法
- 🧠 理解核心技术概念的实际应用
- 💡 建立对 LLM 推理优化的直观认知
- 🚀 为进阶学习打下坚实基础

---

## 📚 学习路径

### 第一阶段：快速入门 (5-10分钟)
**目标**: 建立基本认知，验证环境配置

#### 📁 [00-quick-start/](00-quick-start/) - 5分钟快速体验
```bash
# 环境检查
python 00-quick-start/environment_check.py

# 第一个推理示例
python 00-quick-start/hello_vllm.py

# 基础概念演示
python 00-quick-start/basic_concepts.py
```

**学习重点**:
- ✅ 验证环境配置正确
- ✅ 体验 nano-vLLM 的基本功能
- ✅ 理解推理流程的基本概念

---

### 第二阶段：基础使用 (15-20分钟)
**目标**: 掌握基本API使用，理解核心参数

#### 📁 [01-basic-usage/](01-basic-usage/) - 基础API使用
```bash
# 单次推理示例
python 01-basic-usage/simple_inference.py

# 批量推理示例  
python 01-basic-usage/batch_inference.py
```

**学习重点**:
- ✅ 掌握基本推理API
- ✅ 理解批处理的优势
- ✅ 学会调整关键参数

---

### 第三阶段：核心技术理解 (30-45分钟)
**目标**: 深入理解三大核心技术的工作原理

#### 📁 [01_model_loading/](01_model_loading/) - 模型加载机制
```bash
cd 01_model_loading/
python main.py
```

**学习重点**:
- ✅ 理解模型加载流程
- ✅ 掌握配置参数设置
- ✅ 学会内存管理基础

#### 📁 [02_llm_engine/](02_llm_engine/) - LLM引擎核心
```bash
cd 02_llm_engine/
python main.py
```

**学习重点**:
- ✅ 理解引擎初始化过程
- ✅ 掌握请求处理流程
- ✅ 学会性能监控方法

#### 📁 [03_paged_attention/](03_paged_attention/) - PagedAttention技术
```bash
cd 03_paged_attention/
python main.py
```

**学习重点**:
- ✅ 理解分页内存管理
- ✅ 观察内存利用率提升
- ✅ 掌握内存优化策略

---

## 🧪 实践验证

### 性能对比实验
```bash
# 运行所有基础示例，观察性能差异
for dir in 01_model_loading 02_llm_engine 03_paged_attention; do
    echo "=== 运行 $dir ==="
    cd $dir && python main.py && cd ..
done
```

### 参数调优实验
```bash
# 尝试不同的参数配置
python 01-basic-usage/simple_inference.py --max_tokens 100
python 01-basic-usage/simple_inference.py --max_tokens 500
python 01-basic-usage/simple_inference.py --max_tokens 1000
```

---

## 📊 学习检验

### 快速自测
1. **模型加载**: 能否解释模型加载的主要步骤？
2. **内存管理**: PagedAttention 如何提高内存利用率？
3. **批处理**: 批量推理相比单次推理有什么优势？

### 实践挑战
1. **修改参数**: 尝试调整 `max_tokens`, `temperature` 等参数
2. **性能对比**: 比较不同批次大小的性能差异
3. **错误处理**: 故意输入错误参数，观察错误处理机制

---

## 🚀 下一步学习

### 准备进阶学习
完成基础示例后，你应该能够：
- ✅ 独立运行 nano-vLLM 推理任务
- ✅ 理解核心技术的基本原理
- ✅ 调整基本参数优化性能

### 推荐学习路径
1. **深入理解** → [../docs/concepts.md](../../docs/concepts.md) - 重温核心概念
2. **进阶实践** → [../advanced/](../advanced/) - 挑战复杂场景
3. **系统架构** → [../docs/architecture.md](../../docs/architecture.md) - 理解整体设计

---

## 💡 学习建议

### 费曼学习法应用
1. **理解**: 运行每个示例，观察输出结果
2. **简化**: 用自己的话解释每个技术的作用
3. **实践**: 修改代码参数，验证你的理解
4. **验证**: 向同事解释你学到的概念

### 常见问题解决
- **环境问题**: 检查 Python 版本和依赖安装
- **内存不足**: 减少批次大小或模型大小
- **性能问题**: 检查 GPU 配置和驱动版本

### 学习节奏建议
- **第一遍**: 快速浏览，建立整体印象
- **第二遍**: 仔细运行，理解每个步骤
- **第三遍**: 修改参数，深入探索

> 记住：基础扎实，进阶才能游刃有余！每个概念都要真正理解，而不是简单记忆。