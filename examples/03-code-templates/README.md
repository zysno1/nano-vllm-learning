# 📝 代码模板 (Code Templates)

欢迎来到 nano-vLLM 代码模板目录！这里提供了各种经过精心设计的代码模板，帮助您快速开始项目开发。

## 🎯 模板特点

- **📋 完整注释**: 每个模板都包含详细的中文注释
- **🔧 可配置**: 支持灵活的参数配置
- **⚡ 高性能**: 遵循最佳实践，优化性能
- **🛡️ 错误处理**: 包含完善的错误处理机制
- **📊 监控统计**: 内置性能监控和统计功能
- **🧹 资源管理**: 自动资源清理和内存管理

## 📁 模板列表

### 1. 基础推理模板 (`basic_inference_template.py`)
**🎓 适用对象**: 初学者到中级用户  
**⏱️ 预计时间**: 30-45分钟学习  
**🌟 推荐指数**: ⭐⭐⭐⭐⭐

**功能特点**:
- 完整的推理流程封装
- 灵活的配置系统
- 单个和批量推理支持
- 性能统计和监控
- 自动设备选择和优化

**使用场景**:
- 快速原型开发
- 学习推理流程
- 项目基础框架
- API服务开发

**核心组件**:
```python
# 配置类
InferenceConfig - 推理参数配置

# 主要类
NanoVLLMInference - 推理引擎主类

# 核心方法
setup() - 环境设置
generate() - 文本生成
batch_generate() - 批量生成
get_stats() - 性能统计
```

### 2. 批处理模板 (`batch_processing_template.py`) 🚧
**🎓 适用对象**: 中级用户  
**⏱️ 预计时间**: 45-60分钟学习  
**🌟 推荐指数**: ⭐⭐⭐⭐

**功能特点**:
- 高效批处理算法
- 内存优化策略
- 进度跟踪和恢复
- 并行处理支持

### 3. 流式生成模板 (`streaming_template.py`) 🚧
**🎓 适用对象**: 中级到高级用户  
**⏱️ 预计时间**: 60-90分钟学习  
**🌟 推荐指数**: ⭐⭐⭐⭐

**功能特点**:
- 实时流式输出
- WebSocket支持
- 异步处理
- 用户交互优化

### 4. API服务模板 (`api_service_template.py`) 🚧
**🎓 适用对象**: 高级用户  
**⏱️ 预计时间**: 90-120分钟学习  
**🌟 推荐指数**: ⭐⭐⭐⭐⭐

**功能特点**:
- RESTful API设计
- 请求队列管理
- 负载均衡
- 监控和日志

## 🚀 快速开始

### 1. 选择合适的模板
根据您的需求和技能水平选择合适的模板：

```bash
# 初学者推荐
examples/03-code-templates/basic_inference_template.py

# 需要批处理
examples/03-code-templates/batch_processing_template.py

# 需要实时交互
examples/03-code-templates/streaming_template.py

# 构建API服务
examples/03-code-templates/api_service_template.py
```

### 2. 复制模板到您的项目
```bash
# 复制基础模板
cp examples/03-code-templates/basic_inference_template.py your_project/

# 或者复制整个模板目录
cp -r examples/03-code-templates/ your_project/templates/
```

### 3. 根据需要修改配置
```python
# 在模板中找到配置部分，例如：
config = InferenceConfig(
    model_name="your-model-name",    # 🔧 修改模型
    max_new_tokens=100,              # 🔧 修改生成长度
    temperature=0.8,                 # 🔧 修改创造性
    device="cuda:0"                  # 🔧 修改设备
)
```

### 4. 运行和测试
```bash
# 直接运行模板
python basic_inference_template.py

# 或者导入到您的代码中
from basic_inference_template import NanoVLLMInference, InferenceConfig
```

## 🔧 自定义指南

### 修改模型配置
```python
# 在 InferenceConfig 中修改
config = InferenceConfig(
    model_name="gpt2-medium",        # 使用更大的模型
    torch_dtype="float16",           # 使用半精度
    device="cuda:1"                  # 使用特定GPU
)
```

### 调整生成参数
```python
# 更有创意的生成
config.temperature = 1.2
config.top_p = 0.95
config.repetition_penalty = 1.2

# 更保守的生成
config.temperature = 0.3
config.top_p = 0.8
config.do_sample = False
```

### 添加自定义处理
```python
class CustomInference(NanoVLLMInference):
    def _single_inference(self, prompt: str, **kwargs) -> str:
        # 自定义预处理
        prompt = self.preprocess(prompt)
        
        # 调用原始方法
        result = super()._single_inference(prompt, **kwargs)
        
        # 自定义后处理
        result = self.postprocess(result)
        
        return result
    
    def preprocess(self, text: str) -> str:
        # 您的预处理逻辑
        return text.strip().lower()
    
    def postprocess(self, text: str) -> str:
        # 您的后处理逻辑
        return text.strip().capitalize()
```

## 📊 性能优化建议

### 1. 内存优化
```python
# 使用半精度
config.torch_dtype = "float16"

# 启用梯度检查点
config.use_cache = True

# 批处理大小调优
config.batch_size = 4  # 根据GPU内存调整
```

### 2. 速度优化
```python
# 使用编译优化
torch.compile(model)  # PyTorch 2.0+

# 使用TensorRT (如果可用)
# 需要额外配置

# 调整生成参数
config.max_new_tokens = 50  # 减少生成长度
config.do_sample = False    # 使用贪婪搜索
```

### 3. 并发优化
```python
# 使用异步处理
import asyncio

async def async_generate(prompts):
    tasks = [engine.generate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

## 🐛 常见问题解决

### 1. 内存不足
```python
# 解决方案1: 减少批处理大小
config.batch_size = 1

# 解决方案2: 使用CPU
config.device = "cpu"

# 解决方案3: 使用更小的模型
config.model_name = "distilgpt2"
```

### 2. 生成质量问题
```python
# 解决方案1: 调整温度
config.temperature = 0.7  # 平衡创造性和连贯性

# 解决方案2: 调整重复惩罚
config.repetition_penalty = 1.2

# 解决方案3: 使用更好的提示词
prompt = "请写一个关于...的详细故事："
```

### 3. 速度慢
```python
# 解决方案1: 使用GPU
config.device = "cuda"

# 解决方案2: 减少生成长度
config.max_new_tokens = 30

# 解决方案3: 禁用采样
config.do_sample = False
```

## 📚 学习路径

### 初学者路径
1. **基础模板学习** (30分钟)
   - 运行 `basic_inference_template.py`
   - 理解配置参数
   - 尝试不同的输入

2. **参数调优实践** (45分钟)
   - 修改生成参数
   - 观察输出变化
   - 记录最佳配置

3. **自定义开发** (60分钟)
   - 添加自定义处理
   - 集成到项目中
   - 测试和优化

### 进阶路径
1. **批处理优化** (60分钟)
   - 学习批处理模板
   - 实现并行处理
   - 性能基准测试

2. **流式生成** (90分钟)
   - 实现实时输出
   - 用户交互优化
   - WebSocket集成

3. **生产部署** (120分钟)
   - API服务开发
   - 负载测试
   - 监控和日志

## 🔗 相关资源

### 文档链接
- [基础概念](../../docs/01-basic-concepts/) - 理论基础
- [架构设计](../../docs/02-architecture/) - 系统架构
- [代码分析](../../docs/03-code-analysis/) - 源码解析
- [高级主题](../../docs/04-advanced-topics/) - 进阶内容

### 示例代码
- [快速开始](../00-quick-start/) - 入门示例
- [基础用法](../01-basic-usage/) - 基本操作
- [分步教程](../02-step-by-step-tutorials/) - 详细教程
- [测试用例](../04-test-cases/) - 测试示例

### 工具和资源
- [性能测试工具](../../tools/performance/) - 性能分析
- [调试工具](../../tools/debug/) - 问题诊断
- [配置生成器](../../tools/config/) - 配置助手

## 💡 贡献指南

我们欢迎您贡献新的模板或改进现有模板！

### 贡献步骤
1. Fork 项目
2. 创建新的模板文件
3. 添加详细注释和文档
4. 提交 Pull Request

### 模板规范
- 使用中文注释
- 包含完整的错误处理
- 提供使用示例
- 添加性能统计
- 遵循代码风格

## 📞 获取帮助

如果您在使用模板时遇到问题：

1. **查看文档**: 先查看相关文档和FAQ
2. **运行示例**: 尝试运行完整的示例代码
3. **检查配置**: 确认配置参数是否正确
4. **查看日志**: 检查错误日志和调试信息
5. **寻求帮助**: 在项目Issues中提问

---

🎯 **开始您的 nano-vLLM 开发之旅吧！选择一个模板，开始编码！**