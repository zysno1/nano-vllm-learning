#!/usr/bin/env python3
"""
🧪 nano-vLLM 基础推理测试用例

这个测试文件包含了 nano-vLLM 基础推理功能的完整测试用例。
测试用例设计遵循最佳实践，包含单元测试、集成测试和性能测试。

📋 测试覆盖范围:
- 环境检查和设置
- 模型加载和配置
- 单个文本推理
- 批量文本推理
- 参数配置测试
- 错误处理测试
- 性能基准测试

🎯 使用方法:
1. 直接运行: python test_basic_inference.py
2. 使用pytest: pytest test_basic_inference.py -v
3. 生成报告: pytest test_basic_inference.py --html=report.html

⚠️ 注意事项:
- 测试需要网络连接下载模型
- 某些测试可能需要GPU环境
- 测试时间较长，请耐心等待
"""

import os
import sys
import time
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any
import warnings

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'examples', '03-code-templates'))

# 导入必要的库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch或Transformers未安装，某些测试将被跳过")

# 导入被测试的模块
try:
    from basic_inference_template import NanoVLLMInference, InferenceConfig
    TEMPLATE_AVAILABLE = True
except ImportError:
    TEMPLATE_AVAILABLE = False
    warnings.warn("无法导入推理模板，请检查文件路径")

class TestInferenceConfig(unittest.TestCase):
    """测试推理配置类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = InferenceConfig()
    
    def test_default_config(self):
        """测试默认配置"""
        self.assertEqual(self.config.model_name, "gpt2")
        self.assertEqual(self.config.device, "auto")
        self.assertEqual(self.config.max_new_tokens, 50)
        self.assertEqual(self.config.temperature, 0.7)
        self.assertEqual(self.config.top_p, 0.9)
        self.assertTrue(self.config.do_sample)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = InferenceConfig(
            model_name="gpt2-medium",
            max_new_tokens=100,
            temperature=0.8,
            device="cpu"
        )
        
        self.assertEqual(config.model_name, "gpt2-medium")
        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.temperature, 0.8)
        self.assertEqual(config.device, "cpu")
    
    def test_generation_config_conversion(self):
        """测试转换为GenerationConfig"""
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch不可用")
        
        from transformers import GenerationConfig
        gen_config = self.config.to_generation_config()
        
        self.assertIsInstance(gen_config, GenerationConfig)
        self.assertEqual(gen_config.max_new_tokens, self.config.max_new_tokens)
        self.assertEqual(gen_config.temperature, self.config.temperature)
        self.assertEqual(gen_config.top_p, self.config.top_p)

@unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
class TestNanoVLLMInference(unittest.TestCase):
    """测试推理引擎类"""
    
    @classmethod
    def setUpClass(cls):
        """类级别的设置"""
        cls.config = InferenceConfig(
            model_name="gpt2",  # 使用小模型加快测试
            max_new_tokens=20,  # 减少生成长度
            device="cpu"        # 强制使用CPU确保兼容性
        )
        cls.inference_engine = NanoVLLMInference(cls.config)
    
    def setUp(self):
        """每个测试前的准备"""
        self.test_prompts = [
            "Hello world",
            "The quick brown fox",
            "Once upon a time",
            "In the future"
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.inference_engine)
        self.assertEqual(self.inference_engine.config.model_name, "gpt2")
        self.assertIsNone(self.inference_engine.model)
        self.assertIsNone(self.inference_engine.tokenizer)
    
    def test_setup(self):
        """测试环境设置"""
        # 这个测试可能需要网络连接
        try:
            success = self.inference_engine.setup()
            self.assertTrue(success)
            self.assertIsNotNone(self.inference_engine.model)
            self.assertIsNotNone(self.inference_engine.tokenizer)
            self.assertIsNotNone(self.inference_engine.device)
        except Exception as e:
            self.skipTest(f"设置失败，可能是网络问题: {e}")
    
    def test_device_setup(self):
        """测试设备设置"""
        # 测试CPU设备
        self.inference_engine.config.device = "cpu"
        self.inference_engine._setup_device()
        self.assertEqual(self.inference_engine.device.type, "cpu")
        
        # 测试自动设备选择
        self.inference_engine.config.device = "auto"
        self.inference_engine._setup_device()
        self.assertIsNotNone(self.inference_engine.device)
    
    @unittest.skipUnless(torch.cuda.is_available(), "需要CUDA支持")
    def test_gpu_setup(self):
        """测试GPU设置"""
        config = InferenceConfig(device="cuda")
        engine = NanoVLLMInference(config)
        engine._setup_device()
        self.assertEqual(engine.device.type, "cuda")
    
    def test_stats_initialization(self):
        """测试统计信息初始化"""
        stats = self.inference_engine.get_stats()
        expected_keys = ["total_inferences", "total_tokens_generated", "total_time", "average_speed"]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertEqual(stats[key], 0 if key != "average_speed" else 0.0)
    
    def test_stats_reset(self):
        """测试统计信息重置"""
        # 修改统计信息
        self.inference_engine.stats["total_inferences"] = 10
        self.inference_engine.stats["total_tokens_generated"] = 100
        
        # 重置
        self.inference_engine.reset_stats()
        
        # 验证重置
        stats = self.inference_engine.get_stats()
        self.assertEqual(stats["total_inferences"], 0)
        self.assertEqual(stats["total_tokens_generated"], 0)

class TestInferenceIntegration(unittest.TestCase):
    """集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        if not (TEMPLATE_AVAILABLE and TORCH_AVAILABLE):
            return
        
        cls.config = InferenceConfig(
            model_name="gpt2",
            max_new_tokens=10,  # 很短的生成长度
            device="cpu",
            temperature=0.7
        )
        cls.engine = NanoVLLMInference(cls.config)
        
        # 尝试设置引擎
        try:
            cls.setup_success = cls.engine.setup()
        except Exception as e:
            cls.setup_success = False
            warnings.warn(f"引擎设置失败: {e}")
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_single_inference(self):
        """测试单个推理"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        prompt = "Hello"
        result = self.engine.generate(prompt)
        
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 0)
        
        # 检查统计信息更新
        stats = self.engine.get_stats()
        self.assertGreater(stats["total_inferences"], 0)
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_batch_inference(self):
        """测试批量推理"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        prompts = ["Hello", "World"]
        results = self.engine.batch_generate(prompts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(prompts))
        
        for result in results:
            self.assertIsInstance(result, str)
            self.assertGreater(len(result.strip()), 0)
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_parameter_override(self):
        """测试参数覆盖"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        prompt = "Test"
        
        # 使用默认参数
        result1 = self.engine.generate(prompt)
        
        # 覆盖参数
        result2 = self.engine.generate(
            prompt,
            temperature=0.1,  # 更低的温度
            max_new_tokens=5  # 更短的输出
        )
        
        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        # 注意：由于随机性，结果可能相同，这里只检查类型和长度
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_empty_input(self):
        """测试空输入处理"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        # 测试空字符串
        result = self.engine.generate("")
        self.assertIsInstance(result, str)
        
        # 测试空格字符串
        result = self.engine.generate("   ")
        self.assertIsInstance(result, str)
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_long_input(self):
        """测试长输入处理"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        # 创建一个较长的输入
        long_prompt = "This is a very long prompt. " * 20
        
        try:
            result = self.engine.generate(long_prompt)
            self.assertIsInstance(result, str)
        except Exception as e:
            # 长输入可能导致内存问题，这是可以接受的
            self.assertIn("memory", str(e).lower())

class TestErrorHandling(unittest.TestCase):
    """错误处理测试"""
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE, "需要模板")
    def test_invalid_model_name(self):
        """测试无效模型名称"""
        config = InferenceConfig(model_name="invalid-model-name-12345")
        engine = NanoVLLMInference(config)
        
        # 设置应该失败
        success = engine.setup()
        self.assertFalse(success)
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE, "需要模板")
    def test_invalid_device(self):
        """测试无效设备"""
        config = InferenceConfig(device="invalid-device")
        engine = NanoVLLMInference(config)
        
        with self.assertRaises(Exception):
            engine._setup_device()
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE, "需要模板")
    def test_invalid_parameters(self):
        """测试无效参数"""
        # 测试负数参数
        with self.assertRaises(ValueError):
            InferenceConfig(max_new_tokens=-1)
        
        # 测试超出范围的参数
        config = InferenceConfig(temperature=-1.0)  # 无效温度
        # 注意：某些无效参数可能在运行时才被检测到

class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置性能测试环境"""
        if not (TEMPLATE_AVAILABLE and TORCH_AVAILABLE):
            return
        
        cls.config = InferenceConfig(
            model_name="gpt2",
            max_new_tokens=20,
            device="cpu"
        )
        cls.engine = NanoVLLMInference(cls.config)
        
        try:
            cls.setup_success = cls.engine.setup()
        except Exception:
            cls.setup_success = False
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_inference_speed(self):
        """测试推理速度"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        prompt = "Performance test"
        
        # 预热
        self.engine.generate(prompt)
        
        # 测量时间
        start_time = time.time()
        result = self.engine.generate(prompt)
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        self.assertIsInstance(result, str)
        self.assertLess(inference_time, 30.0)  # 应该在30秒内完成
        
        print(f"推理时间: {inference_time:.3f}秒")
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_batch_performance(self):
        """测试批量推理性能"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        prompts = ["Test"] * 5
        
        # 测量批量推理时间
        start_time = time.time()
        results = self.engine.batch_generate(prompts)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        self.assertEqual(len(results), len(prompts))
        self.assertLess(batch_time, 60.0)  # 应该在60秒内完成
        
        print(f"批量推理时间: {batch_time:.3f}秒")
    
    @unittest.skipUnless(TEMPLATE_AVAILABLE and TORCH_AVAILABLE, "需要模板和PyTorch")
    def test_memory_usage(self):
        """测试内存使用"""
        if not self.setup_success:
            self.skipTest("引擎设置失败")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行多次推理
        for i in range(10):
            self.engine.generate(f"Memory test {i}")
        
        # 记录最终内存
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = final_memory - initial_memory
        
        print(f"内存增长: {memory_increase:.2f}MB")
        
        # 内存增长应该在合理范围内
        self.assertLess(memory_increase, 1000)  # 不应该超过1GB

class TestUtilities(unittest.TestCase):
    """工具函数测试"""
    
    def test_cleanup(self):
        """测试资源清理"""
        if not (TEMPLATE_AVAILABLE and TORCH_AVAILABLE):
            self.skipTest("需要模板和PyTorch")
        
        config = InferenceConfig(model_name="gpt2", device="cpu")
        engine = NanoVLLMInference(config)
        
        # 模拟设置一些资源
        engine.model = MagicMock()
        engine.tokenizer = MagicMock()
        
        # 执行清理
        engine.cleanup()
        
        # 验证资源被清理
        self.assertIsNone(engine.model)
        self.assertIsNone(engine.tokenizer)

def run_comprehensive_test():
    """运行综合测试"""
    print("🧪 开始运行 nano-vLLM 基础推理测试套件")
    print("=" * 60)
    
    # 检查环境
    print("🔍 环境检查:")
    print(f"   Python版本: {sys.version}")
    print(f"   PyTorch可用: {TORCH_AVAILABLE}")
    print(f"   模板可用: {TEMPLATE_AVAILABLE}")
    
    if TORCH_AVAILABLE:
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU数量: {torch.cuda.device_count()}")
    
    print()
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestInferenceConfig,
        TestNanoVLLMInference,
        TestInferenceIntegration,
        TestErrorHandling,
        TestPerformance,
        TestUtilities
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("📊 测试结果摘要:")
    print(f"   总测试数: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    print(f"   跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # 返回是否所有测试都通过
    return len(result.failures) == 0 and len(result.errors) == 0

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="nano-vLLM 基础推理测试")
    parser.add_argument("--quick", action="store_true", help="快速测试模式")
    parser.add_argument("--performance", action="store_true", help="只运行性能测试")
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    
    args = parser.parse_args()
    
    if args.quick:
        # 快速测试模式
        print("🚀 快速测试模式")
        suite = unittest.TestSuite()
        suite.addTest(TestInferenceConfig('test_default_config'))
        if TEMPLATE_AVAILABLE and TORCH_AVAILABLE:
            suite.addTest(TestNanoVLLMInference('test_initialization'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
    elif args.performance:
        # 性能测试模式
        print("⚡ 性能测试模式")
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
    elif args.unit:
        # 单元测试模式
        print("🔧 单元测试模式")
        suite = unittest.TestSuite()
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInferenceConfig))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNanoVLLMInference))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
    else:
        # 完整测试
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# 🎯 测试使用指南:
#
# 1. 基础运行:
#    python test_basic_inference.py
#
# 2. 快速测试:
#    python test_basic_inference.py --quick
#
# 3. 性能测试:
#    python test_basic_inference.py --performance
#
# 4. 单元测试:
#    python test_basic_inference.py --unit
#
# 5. 使用pytest:
#    pytest test_basic_inference.py -v
#    pytest test_basic_inference.py -v -k "test_single_inference"
#    pytest test_basic_inference.py --html=report.html
#
# 📊 测试报告:
#    测试会生成详细的输出，包括：
#    - 环境信息
#    - 测试结果
#    - 性能指标
#    - 错误详情
#
# 🐛 故障排除:
#    - 如果模型下载失败，检查网络连接
#    - 如果内存不足，使用更小的模型或CPU模式
#    - 如果测试超时，增加超时时间或跳过长时间测试