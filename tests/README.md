# 🧪 nano-vLLM 测试套件

欢迎使用 nano-vLLM 的测试套件！这个目录包含了完整的测试用例，帮助您验证代码功能、确保质量并学习最佳实践。

## 📁 目录结构

```
tests/
├── README.md                    # 测试套件说明文档
├── test_basic_inference.py      # 基础推理功能测试
├── test_advanced_features.py    # 高级功能测试（计划中）
├── test_performance.py          # 性能基准测试（计划中）
├── test_integration.py          # 集成测试（计划中）
├── conftest.py                  # pytest配置文件（计划中）
└── fixtures/                    # 测试数据和固定装置（计划中）
    ├── sample_texts.json
    ├── test_models/
    └── expected_outputs/
```

## 🎯 测试类型

### 1. 单元测试 (Unit Tests)
- **配置类测试**: 验证 `InferenceConfig` 的各种配置选项
- **推理引擎测试**: 测试 `NanoVLLMInference` 的核心功能
- **工具函数测试**: 验证辅助函数的正确性

### 2. 集成测试 (Integration Tests)
- **端到端推理**: 完整的推理流程测试
- **批量处理**: 批量推理功能验证
- **参数覆盖**: 动态参数配置测试

### 3. 性能测试 (Performance Tests)
- **推理速度**: 单次推理时间测量
- **批量性能**: 批量推理效率测试
- **内存使用**: 内存占用监控

### 4. 错误处理测试 (Error Handling Tests)
- **无效输入**: 异常输入处理验证
- **资源限制**: 内存/GPU限制测试
- **网络问题**: 模型下载失败处理

## 🚀 快速开始

### 环境准备

1. **安装测试依赖**:
```bash
pip install pytest pytest-html pytest-cov psutil
```

2. **安装核心依赖**:
```bash
pip install torch transformers
```

### 运行测试

#### 方式一：直接运行
```bash
# 运行基础推理测试
python test_basic_inference.py

# 快速测试模式
python test_basic_inference.py --quick

# 性能测试模式
python test_basic_inference.py --performance

# 单元测试模式
python test_basic_inference.py --unit
```

#### 方式二：使用 pytest
```bash
# 运行所有测试
pytest -v

# 运行特定测试文件
pytest test_basic_inference.py -v

# 运行特定测试用例
pytest test_basic_inference.py::TestInferenceConfig::test_default_config -v

# 生成HTML报告
pytest test_basic_inference.py --html=report.html --self-contained-html

# 生成覆盖率报告
pytest test_basic_inference.py --cov=examples --cov-report=html
```

## 📊 测试报告解读

### 测试输出示例
```
🧪 开始运行 nano-vLLM 基础推理测试套件
============================================================
🔍 环境检查:
   Python版本: 3.8.10
   PyTorch可用: True
   模板可用: True
   CUDA可用: True
   GPU数量: 1

test_default_config (test_basic_inference.TestInferenceConfig) ... ok
test_custom_config (test_basic_inference.TestInferenceConfig) ... ok
test_initialization (test_basic_inference.TestNanoVLLMInference) ... ok
test_single_inference (test_basic_inference.TestInferenceIntegration) ... ok
推理时间: 2.345秒
test_inference_speed (test_basic_inference.TestPerformance) ... ok

============================================================
📊 测试结果摘要:
   总测试数: 15
   成功: 13
   失败: 1
   错误: 1
   跳过: 3
```

### 结果解释
- **成功**: 测试通过，功能正常
- **失败**: 断言失败，需要检查代码逻辑
- **错误**: 运行时异常，需要修复代码
- **跳过**: 由于环境限制跳过的测试

## 🔧 测试配置

### 环境变量
```bash
# 设置测试模式
export NANO_VLLM_TEST_MODE=true

# 指定测试模型
export NANO_VLLM_TEST_MODEL=gpt2

# 设置设备
export NANO_VLLM_TEST_DEVICE=cpu
```

### pytest 配置文件 (pytest.ini)
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU
    network: marks tests as requiring network access
```

## 🐛 故障排除

### 常见问题

#### 1. 模型下载失败
```
错误: ConnectionError: Failed to download model
解决: 检查网络连接，或使用本地模型路径
```

#### 2. 内存不足
```
错误: RuntimeError: CUDA out of memory
解决: 使用更小的模型或CPU模式
```

#### 3. 依赖缺失
```
错误: ModuleNotFoundError: No module named 'torch'
解决: pip install torch transformers
```

#### 4. 权限问题
```
错误: PermissionError: Cannot write to cache directory
解决: 检查缓存目录权限或设置 HF_HOME 环境变量
```

### 调试技巧

1. **详细输出**:
```bash
pytest -v -s test_basic_inference.py
```

2. **只运行失败的测试**:
```bash
pytest --lf test_basic_inference.py
```

3. **进入调试模式**:
```bash
pytest --pdb test_basic_inference.py
```

4. **查看测试覆盖率**:
```bash
pytest --cov=examples --cov-report=term-missing
```

## 📚 学习资源

### 测试最佳实践
1. **测试命名**: 使用描述性的测试名称
2. **测试隔离**: 每个测试应该独立运行
3. **数据准备**: 使用 setUp 和 tearDown 方法
4. **断言清晰**: 使用具体的断言消息
5. **异常测试**: 验证错误处理逻辑

### 扩展测试

#### 添加新测试用例
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.config = InferenceConfig()
    
    def test_new_functionality(self):
        """测试新功能"""
        # 测试逻辑
        self.assertTrue(True)
    
    def tearDown(self):
        """测试后清理"""
        pass
```

#### 使用 pytest fixtures
```python
import pytest

@pytest.fixture
def inference_engine():
    """创建推理引擎fixture"""
    config = InferenceConfig(model_name="gpt2", device="cpu")
    engine = NanoVLLMInference(config)
    engine.setup()
    yield engine
    engine.cleanup()

def test_with_fixture(inference_engine):
    """使用fixture的测试"""
    result = inference_engine.generate("Hello")
    assert isinstance(result, str)
```

## 🤝 贡献指南

### 添加新测试
1. 在相应的测试文件中添加测试用例
2. 遵循现有的命名约定
3. 添加适当的文档字符串
4. 确保测试可以独立运行

### 测试标准
- **覆盖率**: 新功能应有相应测试
- **性能**: 包含性能基准测试
- **错误处理**: 验证异常情况
- **文档**: 提供清晰的测试说明

## 📞 获取帮助

如果您在运行测试时遇到问题：

1. **查看日志**: 检查详细的错误信息
2. **检查环境**: 确认依赖和配置正确
3. **参考文档**: 查看相关的使用文档
4. **社区支持**: 在项目仓库提交 issue

---

**记住**: 测试不仅是验证代码正确性的工具，也是学习和理解代码的最佳方式。通过阅读和运行测试，您可以更好地理解 nano-vLLM 的工作原理和最佳实践。

祝您测试愉快！🎉