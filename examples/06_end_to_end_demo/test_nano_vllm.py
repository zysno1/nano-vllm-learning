"""
NanoVLLM 测试框架

包含单元测试和集成测试，验证系统各个组件的功能。
"""

import os
import sys
import time
import json
import pytest
import asyncio
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from nano_vllm import (
    NanoVLLM, Block, BlockTable, BlockAllocator, 
    PagedAttentionEngine, Scheduler, MetricsCollector,
    GenerationParams, InferenceRequest, InferenceResponse,
    RequestStatus, BlockStatus, SchedulerPolicy
)
from utils import Logger, HealthChecker, SystemMonitor, MetricsCollector as UtilsMetricsCollector
from config import NanoVLLMConfig, get_config


class TestBlock:
    """Block类的单元测试"""
    
    def test_block_creation(self):
        """测试Block创建"""
        block = Block(block_id=1, block_size=16)
        assert block.block_id == 1
        assert block.block_size == 16
        assert block.status == BlockStatus.FREE
        assert block.ref_count == 0
        assert len(block.token_ids) == 0
    
    def test_block_allocation(self):
        """测试Block分配"""
        block = Block(block_id=1, block_size=16)
        
        # 分配block
        block.allocate()
        assert block.status == BlockStatus.ALLOCATED
        assert block.ref_count == 1
        
        # 增加引用
        block.add_ref()
        assert block.ref_count == 2
        
        # 减少引用
        block.remove_ref()
        assert block.ref_count == 1
        assert block.status == BlockStatus.ALLOCATED
        
        # 释放block
        block.remove_ref()
        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE
    
    def test_block_token_operations(self):
        """测试Block token操作"""
        block = Block(block_id=1, block_size=4)
        
        # 添加tokens
        assert block.append_tokens([1, 2, 3]) == True
        assert block.token_ids == [1, 2, 3]
        assert block.is_full() == False
        
        # 填满block
        assert block.append_tokens([4]) == True
        assert block.is_full() == True
        
        # 尝试添加更多tokens
        assert block.append_tokens([5]) == False
        assert len(block.token_ids) == 4


class TestBlockTable:
    """BlockTable类的单元测试"""
    
    def test_block_table_creation(self):
        """测试BlockTable创建"""
        table = BlockTable(sequence_id="seq_1")
        assert table.sequence_id == "seq_1"
        assert len(table.blocks) == 0
    
    def test_block_table_operations(self):
        """测试BlockTable操作"""
        table = BlockTable(sequence_id="seq_1")
        
        # 添加blocks
        block1 = Block(block_id=1, block_size=4)
        block2 = Block(block_id=2, block_size=4)
        
        table.add_block(block1)
        table.add_block(block2)
        
        assert len(table.blocks) == 2
        assert table.get_total_tokens() == 0
        
        # 添加tokens
        block1.append_tokens([1, 2, 3, 4])
        block2.append_tokens([5, 6])
        
        assert table.get_total_tokens() == 6
        assert table.get_all_tokens() == [1, 2, 3, 4, 5, 6]


class TestBlockAllocator:
    """BlockAllocator类的单元测试"""
    
    def test_allocator_creation(self):
        """测试BlockAllocator创建"""
        allocator = BlockAllocator(num_blocks=10, block_size=16)
        assert allocator.num_blocks == 10
        assert allocator.block_size == 16
        assert len(allocator.free_blocks) == 10
        assert len(allocator.allocated_blocks) == 0
    
    def test_block_allocation_and_free(self):
        """测试block分配和释放"""
        allocator = BlockAllocator(num_blocks=5, block_size=16)
        
        # 分配blocks
        block1 = allocator.allocate_block()
        block2 = allocator.allocate_block()
        
        assert block1 is not None
        assert block2 is not None
        assert len(allocator.free_blocks) == 3
        assert len(allocator.allocated_blocks) == 2
        
        # 释放block
        allocator.free_block(block1)
        assert len(allocator.free_blocks) == 4
        assert len(allocator.allocated_blocks) == 1
        assert block1.status == BlockStatus.FREE
    
    def test_allocator_exhaustion(self):
        """测试allocator耗尽"""
        allocator = BlockAllocator(num_blocks=2, block_size=16)
        
        # 分配所有blocks
        block1 = allocator.allocate_block()
        block2 = allocator.allocate_block()
        block3 = allocator.allocate_block()  # 应该返回None
        
        assert block1 is not None
        assert block2 is not None
        assert block3 is None
        assert allocator.get_free_blocks() == 0


class TestPagedAttentionEngine:
    """PagedAttentionEngine类的单元测试"""
    
    def test_engine_creation(self):
        """测试PagedAttentionEngine创建"""
        engine = PagedAttentionEngine(num_blocks=100, block_size=16)
        assert engine.block_allocator.num_blocks == 100
        assert engine.block_allocator.block_size == 16
        assert len(engine.sequence_tables) == 0
    
    def test_sequence_allocation(self):
        """测试序列分配"""
        engine = PagedAttentionEngine(num_blocks=10, block_size=4)
        
        # 分配序列
        success = engine.allocate_sequence("seq_1", num_tokens=6)
        assert success == True
        assert "seq_1" in engine.sequence_tables
        
        table = engine.sequence_tables["seq_1"]
        assert len(table.blocks) == 2  # 需要2个block来存储6个tokens
    
    def test_sequence_append_tokens(self):
        """测试序列添加tokens"""
        engine = PagedAttentionEngine(num_blocks=10, block_size=4)
        
        # 分配序列并添加tokens
        engine.allocate_sequence("seq_1", num_tokens=3)
        success = engine.append_tokens("seq_1", [1, 2, 3])
        assert success == True
        
        table = engine.sequence_tables["seq_1"]
        assert table.get_all_tokens() == [1, 2, 3]
    
    def test_sequence_free(self):
        """测试序列释放"""
        engine = PagedAttentionEngine(num_blocks=10, block_size=4)
        
        # 分配并释放序列
        engine.allocate_sequence("seq_1", num_tokens=6)
        initial_free_blocks = engine.block_allocator.get_free_blocks()
        
        engine.free_sequence("seq_1")
        assert "seq_1" not in engine.sequence_tables
        assert engine.block_allocator.get_free_blocks() > initial_free_blocks


class TestScheduler:
    """Scheduler类的单元测试"""
    
    def test_scheduler_creation(self):
        """测试Scheduler创建"""
        scheduler = Scheduler(max_num_seqs=4, max_model_len=1024)
        assert scheduler.max_num_seqs == 4
        assert scheduler.max_model_len == 1024
        assert len(scheduler.waiting) == 0
        assert len(scheduler.running) == 0
        assert len(scheduler.swapped) == 0
    
    def test_add_request(self):
        """测试添加请求"""
        scheduler = Scheduler(max_num_seqs=4, max_model_len=1024)
        
        request = InferenceRequest(
            request_id="req_1",
            prompt="Hello world",
            params=GenerationParams()
        )
        
        scheduler.add_request(request)
        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0].request_id == "req_1"
    
    def test_schedule_requests(self):
        """测试请求调度"""
        scheduler = Scheduler(max_num_seqs=2, max_model_len=1024)
        
        # 添加多个请求
        for i in range(3):
            request = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Hello world {i}",
                params=GenerationParams()
            )
            scheduler.add_request(request)
        
        # 模拟内存分配器
        mock_allocator = Mock()
        mock_allocator.can_allocate.return_value = True
        
        # 调度请求
        scheduled = scheduler.schedule(mock_allocator)
        
        # 应该调度前2个请求（max_num_seqs=2）
        assert len(scheduled) <= 2
        assert len(scheduler.running) <= 2
        assert len(scheduler.waiting) >= 1


class TestMetricsCollector:
    """MetricsCollector类的单元测试"""
    
    def test_metrics_creation(self):
        """测试MetricsCollector创建"""
        collector = MetricsCollector()
        assert collector.request_count == 0
        assert collector.error_count == 0
        assert collector.total_tokens == 0
        assert collector.total_latency == 0.0
    
    def test_record_request(self):
        """测试记录请求"""
        collector = MetricsCollector()
        
        # 记录成功请求
        collector.record_request(latency=0.5, tokens=100, success=True)
        assert collector.request_count == 1
        assert collector.error_count == 0
        assert collector.total_tokens == 100
        assert collector.total_latency == 0.5
        
        # 记录失败请求
        collector.record_request(latency=1.0, tokens=0, success=False)
        assert collector.request_count == 2
        assert collector.error_count == 1
        assert collector.total_tokens == 100  # 失败请求不计入tokens
    
    def test_get_metrics(self):
        """测试获取指标"""
        collector = MetricsCollector()
        
        # 记录一些请求
        collector.record_request(0.5, 100, True)
        collector.record_request(1.0, 200, True)
        collector.record_request(0.8, 0, False)
        
        metrics = collector.get_current_metrics()
        assert metrics.request_count == 3
        assert metrics.total_tokens == 300
        assert metrics.error_count == 1
        assert metrics.error_rate == 1/3


class TestNanoVLLM:
    """NanoVLLM类的单元测试"""
    
    @pytest.fixture
    def mock_model_components(self):
        """模拟模型组件"""
        with patch('nano_vllm.AutoTokenizer') as mock_tokenizer_class, \
             patch('nano_vllm.AutoModelForCausalLM') as mock_model_class:
            
            # 模拟tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.decode.return_value = "Generated text"
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # 模拟model
            mock_model = Mock()
            mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7]]
            mock_model_class.from_pretrained.return_value = mock_model
            
            yield mock_tokenizer, mock_model
    
    def test_nano_vllm_creation(self, mock_model_components):
        """测试NanoVLLM创建"""
        config = get_config("dev")
        nano_vllm = NanoVLLM(config)
        
        assert nano_vllm.config == config
        assert nano_vllm.scheduler is not None
        assert nano_vllm.paged_attention is not None
        assert nano_vllm.metrics_collector is not None
    
    def test_model_loading(self, mock_model_components):
        """测试模型加载"""
        config = get_config("dev")
        nano_vllm = NanoVLLM(config)
        
        # 加载模型
        nano_vllm.load_model()
        
        assert nano_vllm.model is not None
        assert nano_vllm.tokenizer is not None
        assert nano_vllm.model_loaded == True
    
    def test_text_generation(self, mock_model_components):
        """测试文本生成"""
        config = get_config("dev")
        nano_vllm = NanoVLLM(config)
        nano_vllm.load_model()
        
        # 生成文本
        response = nano_vllm.generate(
            prompt="Hello world",
            max_tokens=50,
            temperature=0.8
        )
        
        assert isinstance(response, InferenceResponse)
        assert response.generated_text is not None
        assert response.request_id is not None
        assert response.total_tokens > 0


class TestUtils:
    """工具函数的单元测试"""
    
    def test_logger_creation(self):
        """测试Logger创建"""
        from utils import Logger
        
        logger = Logger("TestLogger", "INFO")
        assert logger.logger.name == "TestLogger"
        assert logger.logger.level == 20  # INFO level
    
    def test_health_checker(self):
        """测试HealthChecker"""
        from utils import HealthChecker, Logger
        
        logger = Logger("TestLogger", "INFO")
        health_checker = HealthChecker(logger)
        
        # 注册自定义检查
        health_checker.register_check("test_check", lambda: True)
        
        # 执行健康检查
        health_status = health_checker.check_system_health()
        
        assert isinstance(health_status, dict)
        assert 'healthy' in health_status
        assert 'checks' in health_status
        assert 'test_check' in health_status['checks']
    
    def test_system_monitor(self):
        """测试SystemMonitor"""
        from utils import SystemMonitor, Logger
        
        logger = Logger("TestLogger", "INFO")
        monitor = SystemMonitor(logger)
        
        # 获取系统信息
        system_info = monitor.get_system_info()
        
        assert system_info.cpu_count > 0
        assert system_info.memory_total > 0
        assert system_info.disk_total > 0


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def nano_vllm_instance(self):
        """创建NanoVLLM实例用于集成测试"""
        with patch('nano_vllm.AutoTokenizer') as mock_tokenizer_class, \
             patch('nano_vllm.AutoModelForCausalLM') as mock_model_class:
            
            # 模拟tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.decode.return_value = "This is a generated response."
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            # 模拟model
            mock_model = Mock()
            mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            mock_model_class.from_pretrained.return_value = mock_model
            
            config = get_config("dev")
            nano_vllm = NanoVLLM(config)
            nano_vllm.load_model()
            
            yield nano_vllm
    
    def test_end_to_end_generation(self, nano_vllm_instance):
        """测试端到端文本生成"""
        nano_vllm = nano_vllm_instance
        
        # 单个请求
        response = nano_vllm.generate(
            prompt="What is artificial intelligence?",
            max_tokens=50,
            temperature=0.8
        )
        
        assert response.generated_text is not None
        assert response.total_tokens > 0
        assert response.tokens_per_second > 0
        assert response.request_id is not None
    
    def test_batch_generation(self, nano_vllm_instance):
        """测试批量生成"""
        nano_vllm = nano_vllm_instance
        
        prompts = [
            "What is machine learning?",
            "How does AI work?",
            "Explain neural networks."
        ]
        
        responses = nano_vllm.generate_batch(
            prompts=prompts,
            max_tokens=30,
            temperature=0.7
        )
        
        assert len(responses) == len(prompts)
        for response in responses:
            assert response.generated_text is not None
            assert response.total_tokens > 0
    
    def test_metrics_collection(self, nano_vllm_instance):
        """测试指标收集"""
        nano_vllm = nano_vllm_instance
        
        # 生成一些请求
        for i in range(3):
            nano_vllm.generate(
                prompt=f"Test prompt {i}",
                max_tokens=20,
                temperature=0.8
            )
        
        # 检查指标
        metrics = nano_vllm.get_metrics()
        assert metrics['request_count'] >= 3
        assert metrics['total_tokens'] > 0
        assert metrics['avg_latency'] > 0
    
    def test_health_status(self, nano_vllm_instance):
        """测试健康状态"""
        nano_vllm = nano_vllm_instance
        
        health_status = nano_vllm.get_health()
        
        assert isinstance(health_status, dict)
        assert 'healthy' in health_status
        assert 'model_loaded' in health_status
        assert health_status['model_loaded'] == True


class TestAPIServer:
    """API服务器测试"""
    
    @pytest.fixture
    def api_client(self):
        """创建API客户端用于测试"""
        # 这里假设API服务器在测试时运行在localhost:8000
        base_url = "http://localhost:8000"
        return base_url
    
    def test_health_endpoint(self, api_client):
        """测试健康检查端点"""
        try:
            response = requests.get(f"{api_client}/v1/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                assert 'healthy' in data
            else:
                pytest.skip("API服务器未运行")
        except requests.exceptions.RequestException:
            pytest.skip("API服务器未运行")
    
    def test_generate_endpoint(self, api_client):
        """测试生成端点"""
        try:
            payload = {
                "prompt": "Hello, how are you?",
                "max_tokens": 50,
                "temperature": 0.8
            }
            
            response = requests.post(
                f"{api_client}/v1/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                assert 'generated_text' in data
                assert 'request_id' in data
            else:
                pytest.skip("API服务器未运行或配置错误")
        except requests.exceptions.RequestException:
            pytest.skip("API服务器未运行")


class TestPerformance:
    """性能测试"""
    
    @pytest.fixture
    def nano_vllm_instance(self):
        """创建NanoVLLM实例用于性能测试"""
        with patch('nano_vllm.AutoTokenizer') as mock_tokenizer_class, \
             patch('nano_vllm.AutoModelForCausalLM') as mock_model_class:
            
            # 模拟快速响应
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
            mock_tokenizer.decode.return_value = "Fast generated response."
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            mock_model = Mock()
            mock_model.generate.return_value = [[1, 2, 3, 4, 5, 6, 7, 8]]
            mock_model_class.from_pretrained.return_value = mock_model
            
            config = get_config("dev")
            nano_vllm = NanoVLLM(config)
            nano_vllm.load_model()
            
            yield nano_vllm
    
    def test_generation_latency(self, nano_vllm_instance):
        """测试生成延迟"""
        nano_vllm = nano_vllm_instance
        
        start_time = time.time()
        response = nano_vllm.generate(
            prompt="Test prompt for latency measurement",
            max_tokens=50,
            temperature=0.8
        )
        end_time = time.time()
        
        latency = end_time - start_time
        assert latency < 5.0  # 应该在5秒内完成
        assert response.latency > 0
    
    def test_throughput(self, nano_vllm_instance):
        """测试吞吐量"""
        nano_vllm = nano_vllm_instance
        
        num_requests = 10
        start_time = time.time()
        
        for i in range(num_requests):
            nano_vllm.generate(
                prompt=f"Throughput test {i}",
                max_tokens=20,
                temperature=0.8
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        assert throughput > 0
        print(f"吞吐量: {throughput:.2f} requests/second")
    
    def test_concurrent_requests(self, nano_vllm_instance):
        """测试并发请求"""
        nano_vllm = nano_vllm_instance
        
        def make_request(request_id):
            return nano_vllm.generate(
                prompt=f"Concurrent request {request_id}",
                max_tokens=30,
                temperature=0.8
            )
        
        # 使用线程池进行并发测试
        import concurrent.futures
        
        num_concurrent = 5
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(responses) == num_concurrent
        assert all(r.generated_text is not None for r in responses)
        print(f"并发测试完成: {num_concurrent} 请求在 {total_time:.2f} 秒内完成")


def run_tests():
    """运行所有测试"""
    print("🧪 开始运行 NanoVLLM 测试套件...")
    
    # 运行pytest
    pytest_args = [
        __file__,
        "-v",  # 详细输出
        "--tb=short",  # 简短的traceback
        "--color=yes",  # 彩色输出
    ]
    
    # 如果需要覆盖率报告
    try:
        import pytest_cov
        pytest_args.extend(["--cov=nano_vllm", "--cov=utils", "--cov-report=term-missing"])
    except ImportError:
        print("⚠️  pytest-cov 未安装，跳过覆盖率报告")
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    
    return exit_code


if __name__ == "__main__":
    # 设置测试环境
    os.environ["TESTING"] = "1"
    
    # 运行测试
    exit_code = run_tests()
    sys.exit(exit_code)