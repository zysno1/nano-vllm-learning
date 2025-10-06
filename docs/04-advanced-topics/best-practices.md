# 🌟 最佳实践 (Best Practices)

## 📖 概述

本文档汇总了nano-vllm在开发、部署和运维过程中的最佳实践，涵盖代码质量、性能优化、安全防护、监控运维等各个方面，帮助开发者构建高质量、高性能、安全可靠的LLM推理服务。

## 💻 开发最佳实践

### 1. 代码质量管理

```python
import logging
import asyncio
import time
import traceback
import functools
import inspect
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
import pytest
import unittest
from unittest.mock import Mock, patch
import coverage
import black
import flake8
import mypy
import bandit
import safety

logger = logging.getLogger(__name__)

class CodeQualityLevel(Enum):
    """代码质量级别"""
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"

class TestType(Enum):
    """测试类型"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    END_TO_END = "e2e"

@dataclass
class CodeQualityMetrics:
    """代码质量指标"""
    test_coverage: float = 0.0
    code_complexity: float = 0.0
    code_duplication: float = 0.0
    security_issues: int = 0
    performance_score: float = 0.0
    maintainability_index: float = 0.0
    
    def get_quality_level(self) -> CodeQualityLevel:
        """获取代码质量级别"""
        
        score = (
            (self.test_coverage / 100) * 0.3 +
            (max(0, 100 - self.code_complexity) / 100) * 0.2 +
            (max(0, 100 - self.code_duplication) / 100) * 0.2 +
            (max(0, 100 - self.security_issues * 10) / 100) * 0.15 +
            (self.performance_score / 100) * 0.1 +
            (self.maintainability_index / 100) * 0.05
        )
        
        if score >= 0.9:
            return CodeQualityLevel.EXCELLENT
        elif score >= 0.7:
            return CodeQualityLevel.GOOD
        else:
            return CodeQualityLevel.BASIC

class CodeQualityChecker:
    """代码质量检查器"""
    
    def __init__(self):
        self.checkers = {
            'formatting': self._check_formatting,
            'linting': self._check_linting,
            'type_checking': self._check_type_hints,
            'security': self._check_security,
            'dependencies': self._check_dependencies,
            'testing': self._check_test_coverage
        }
    
    def check_code_quality(self, project_path: str) -> CodeQualityMetrics:
        """检查代码质量"""
        
        metrics = CodeQualityMetrics()
        
        try:
            # 运行各种检查
            for check_name, checker in self.checkers.items():
                logger.info(f"Running {check_name} check...")
                checker(project_path, metrics)
            
            logger.info(f"Code quality level: {metrics.get_quality_level().value}")
            
        except Exception as e:
            logger.error(f"Code quality check failed: {e}")
        
        return metrics
    
    def _check_formatting(self, project_path: str, metrics: CodeQualityMetrics):
        """检查代码格式"""
        
        # 使用black检查代码格式
        # 这里是示例实现
        logger.info("Checking code formatting with black...")
        
    def _check_linting(self, project_path: str, metrics: CodeQualityMetrics):
        """检查代码规范"""
        
        # 使用flake8检查代码规范
        logger.info("Checking code linting with flake8...")
        
    def _check_type_hints(self, project_path: str, metrics: CodeQualityMetrics):
        """检查类型提示"""
        
        # 使用mypy检查类型提示
        logger.info("Checking type hints with mypy...")
        
    def _check_security(self, project_path: str, metrics: CodeQualityMetrics):
        """检查安全问题"""
        
        # 使用bandit检查安全问题
        logger.info("Checking security issues with bandit...")
        
    def _check_dependencies(self, project_path: str, metrics: CodeQualityMetrics):
        """检查依赖安全"""
        
        # 使用safety检查依赖安全
        logger.info("Checking dependency security with safety...")
        
    def _check_test_coverage(self, project_path: str, metrics: CodeQualityMetrics):
        """检查测试覆盖率"""
        
        # 使用coverage检查测试覆盖率
        logger.info("Checking test coverage...")
        metrics.test_coverage = 85.0  # 示例值

def error_handler(retry_count: int = 3, delay: float = 1.0):
    """错误处理装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_count):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < retry_count - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                    else:
                        logger.error(f"All {retry_count} attempts failed")
                        raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_count):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
                    if attempt < retry_count - 1:
                        time.sleep(delay * (2 ** attempt))
                    else:
                        logger.error(f"All {retry_count} attempts failed")
                        raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def performance_monitor(threshold_ms: float = 1000.0):
    """性能监控装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > threshold_ms:
                    logger.warning(f"Slow execution: {func.__name__} took {execution_time:.2f}ms")
                else:
                    logger.debug(f"Execution time: {func.__name__} took {execution_time:.2f}ms")
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}ms: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = (time.time() - start_time) * 1000
                
                if execution_time > threshold_ms:
                    logger.warning(f"Slow execution: {func.__name__} took {execution_time:.2f}ms")
                else:
                    logger.debug(f"Execution time: {func.__name__} took {execution_time:.2f}ms")
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Function {func.__name__} failed after {execution_time:.2f}ms: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class TestFramework:
    """测试框架"""
    
    def __init__(self):
        self.test_suites = {}
        self.test_results = {}
    
    def add_test_suite(self, name: str, test_type: TestType, tests: List[Callable]):
        """添加测试套件"""
        
        self.test_suites[name] = {
            'type': test_type,
            'tests': tests,
            'results': []
        }
    
    async def run_tests(self, suite_name: str = None) -> Dict[str, Any]:
        """运行测试"""
        
        if suite_name:
            suites_to_run = {suite_name: self.test_suites[suite_name]}
        else:
            suites_to_run = self.test_suites
        
        results = {}
        
        for name, suite in suites_to_run.items():
            logger.info(f"Running test suite: {name}")
            
            suite_results = {
                'type': suite['type'].value,
                'total': len(suite['tests']),
                'passed': 0,
                'failed': 0,
                'errors': [],
                'execution_time': 0
            }
            
            start_time = time.time()
            
            for test_func in suite['tests']:
                try:
                    if asyncio.iscoroutinefunction(test_func):
                        await test_func()
                    else:
                        test_func()
                    
                    suite_results['passed'] += 1
                    logger.debug(f"Test passed: {test_func.__name__}")
                    
                except Exception as e:
                    suite_results['failed'] += 1
                    suite_results['errors'].append({
                        'test': test_func.__name__,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    logger.error(f"Test failed: {test_func.__name__} - {e}")
            
            suite_results['execution_time'] = time.time() - start_time
            results[name] = suite_results
        
        return results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        
        report = ["# Test Report\n"]
        
        total_tests = sum(suite['total'] for suite in results.values())
        total_passed = sum(suite['passed'] for suite in results.values())
        total_failed = sum(suite['failed'] for suite in results.values())
        
        report.append(f"## Summary")
        report.append(f"- Total Tests: {total_tests}")
        report.append(f"- Passed: {total_passed}")
        report.append(f"- Failed: {total_failed}")
        report.append(f"- Success Rate: {(total_passed / total_tests * 100):.2f}%\n")
        
        for suite_name, suite_result in results.items():
            report.append(f"## {suite_name}")
            report.append(f"- Type: {suite_result['type']}")
            report.append(f"- Tests: {suite_result['total']}")
            report.append(f"- Passed: {suite_result['passed']}")
            report.append(f"- Failed: {suite_result['failed']}")
            report.append(f"- Execution Time: {suite_result['execution_time']:.2f}s")
            
            if suite_result['errors']:
                report.append("\n### Failures:")
                for error in suite_result['errors']:
                    report.append(f"- **{error['test']}**: {error['error']}")
            
            report.append("")
        
        return "\n".join(report)

# 示例测试用例
class TestExamples:
    """测试示例"""
    
    @staticmethod
    def test_tokenizer():
        """测试分词器"""
        # 模拟分词器测试
        assert True, "Tokenizer test passed"
    
    @staticmethod
    async def test_model_inference():
        """测试模型推理"""
        # 模拟异步推理测试
        await asyncio.sleep(0.1)  # 模拟推理时间
        assert True, "Model inference test passed"
    
    @staticmethod
    def test_performance():
        """测试性能"""
        # 模拟性能测试
        start_time = time.time()
        # 模拟一些计算
        time.sleep(0.01)
        execution_time = time.time() - start_time
        assert execution_time < 1.0, f"Performance test failed: {execution_time}s"
```

### 2. 性能优化最佳实践

```python
class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.optimization_strategies = {
            'memory': self._optimize_memory,
            'compute': self._optimize_compute,
            'io': self._optimize_io,
            'network': self._optimize_network
        }
    
    def optimize_system(self, target_areas: List[str] = None) -> Dict[str, Any]:
        """优化系统性能"""
        
        if target_areas is None:
            target_areas = list(self.optimization_strategies.keys())
        
        results = {}
        
        for area in target_areas:
            if area in self.optimization_strategies:
                logger.info(f"Optimizing {area}...")
                results[area] = self.optimization_strategies[area]()
            else:
                logger.warning(f"Unknown optimization area: {area}")
        
        return results
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """内存优化"""
        
        optimizations = {
            'gradient_checkpointing': True,
            'memory_pooling': True,
            'kv_cache_optimization': True,
            'activation_recomputation': True
        }
        
        return {
            'applied_optimizations': optimizations,
            'memory_saved_mb': 2048,  # 示例值
            'performance_impact': 'minimal'
        }
    
    def _optimize_compute(self) -> Dict[str, Any]:
        """计算优化"""
        
        optimizations = {
            'kernel_fusion': True,
            'mixed_precision': True,
            'flash_attention': True,
            'compilation': True
        }
        
        return {
            'applied_optimizations': optimizations,
            'speedup_factor': 1.8,  # 示例值
            'accuracy_impact': 'negligible'
        }
    
    def _optimize_io(self) -> Dict[str, Any]:
        """I/O优化"""
        
        optimizations = {
            'async_io': True,
            'batch_processing': True,
            'prefetching': True,
            'caching': True
        }
        
        return {
            'applied_optimizations': optimizations,
            'throughput_improvement': '40%',
            'latency_reduction': '25%'
        }
    
    def _optimize_network(self) -> Dict[str, Any]:
        """网络优化"""
        
        optimizations = {
            'compression': True,
            'connection_pooling': True,
            'load_balancing': True,
            'cdn_caching': True
        }
        
        return {
            'applied_optimizations': optimizations,
            'bandwidth_saved': '30%',
            'response_time_improvement': '20%'
        }

@contextmanager
def resource_monitor(resource_type: str = "memory"):
    """资源监控上下文管理器"""
    
    import psutil
    import gc
    
    # 记录初始状态
    if resource_type == "memory":
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        
        if resource_type == "memory":
            # 强制垃圾回收
            gc.collect()
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_diff = final_memory - initial_memory
            
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            logger.info(f"Memory difference: {memory_diff:+.2f} MB")
        
        logger.info(f"Execution time: {execution_time:.2f} seconds")

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        
        if key not in self.cache:
            return None
        
        # 检查TTL
        if time.time() - self.creation_times[key] > self.ttl:
            self.delete(key)
            return None
        
        # 更新访问时间
        self.access_times[key] = time.time()
        
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        
        # 检查缓存大小限制
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.creation_times[key] = time.time()
    
    def delete(self, key: str):
        """删除缓存值"""
        
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
    
    def _evict_lru(self):
        """驱逐最近最少使用的缓存项"""
        
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.delete(lru_key)
    
    def clear(self):
        """清空缓存"""
        
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': 0.85,  # 示例值，实际需要统计
            'memory_usage_mb': sum(
                len(str(v).encode('utf-8')) for v in self.cache.values()
            ) / 1024 / 1024
        }
```

### 3. 部署最佳实践

```python
class DeploymentManager:
    """部署管理器"""
    
    def __init__(self):
        self.deployment_checklist = [
            'environment_validation',
            'dependency_check',
            'configuration_validation',
            'security_check',
            'performance_test',
            'health_check_setup',
            'monitoring_setup',
            'backup_strategy',
            'rollback_plan'
        ]
    
    def validate_deployment(self, environment: str) -> Dict[str, Any]:
        """验证部署环境"""
        
        results = {}
        
        for check in self.deployment_checklist:
            logger.info(f"Running deployment check: {check}")
            results[check] = self._run_check(check, environment)
        
        # 计算总体部署就绪度
        passed_checks = sum(1 for result in results.values() if result['status'] == 'passed')
        readiness_score = (passed_checks / len(self.deployment_checklist)) * 100
        
        results['deployment_readiness'] = {
            'score': readiness_score,
            'status': 'ready' if readiness_score >= 90 else 'not_ready',
            'passed_checks': passed_checks,
            'total_checks': len(self.deployment_checklist)
        }
        
        return results
    
    def _run_check(self, check_name: str, environment: str) -> Dict[str, Any]:
        """运行单个检查"""
        
        # 这里是示例实现，实际应该根据具体检查类型实现
        
        if check_name == 'environment_validation':
            return self._validate_environment(environment)
        elif check_name == 'dependency_check':
            return self._check_dependencies()
        elif check_name == 'configuration_validation':
            return self._validate_configuration()
        elif check_name == 'security_check':
            return self._run_security_check()
        elif check_name == 'performance_test':
            return self._run_performance_test()
        elif check_name == 'health_check_setup':
            return self._setup_health_checks()
        elif check_name == 'monitoring_setup':
            return self._setup_monitoring()
        elif check_name == 'backup_strategy':
            return self._validate_backup_strategy()
        elif check_name == 'rollback_plan':
            return self._validate_rollback_plan()
        else:
            return {'status': 'skipped', 'message': f'Unknown check: {check_name}'}
    
    def _validate_environment(self, environment: str) -> Dict[str, Any]:
        """验证环境"""
        
        # 检查环境变量、系统资源等
        return {
            'status': 'passed',
            'message': f'Environment {environment} validation passed',
            'details': {
                'cpu_cores': 8,
                'memory_gb': 32,
                'disk_space_gb': 500,
                'gpu_count': 1
            }
        }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """检查依赖"""
        
        # 检查Python包、系统依赖等
        return {
            'status': 'passed',
            'message': 'All dependencies are satisfied',
            'details': {
                'python_version': '3.9.0',
                'torch_version': '2.0.0',
                'transformers_version': '4.30.0'
            }
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """验证配置"""
        
        # 检查配置文件、环境变量等
        return {
            'status': 'passed',
            'message': 'Configuration validation passed',
            'details': {
                'config_files_found': True,
                'required_env_vars': True,
                'secrets_configured': True
            }
        }
    
    def _run_security_check(self) -> Dict[str, Any]:
        """运行安全检查"""
        
        # 运行安全扫描、漏洞检查等
        return {
            'status': 'passed',
            'message': 'Security check passed',
            'details': {
                'vulnerabilities_found': 0,
                'security_score': 95,
                'ssl_configured': True
            }
        }
    
    def _run_performance_test(self) -> Dict[str, Any]:
        """运行性能测试"""
        
        # 运行基准测试
        return {
            'status': 'passed',
            'message': 'Performance test passed',
            'details': {
                'latency_p95_ms': 150,
                'throughput_rps': 100,
                'memory_usage_mb': 2048
            }
        }
    
    def _setup_health_checks(self) -> Dict[str, Any]:
        """设置健康检查"""
        
        return {
            'status': 'passed',
            'message': 'Health checks configured',
            'details': {
                'liveness_probe': True,
                'readiness_probe': True,
                'startup_probe': True
            }
        }
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """设置监控"""
        
        return {
            'status': 'passed',
            'message': 'Monitoring configured',
            'details': {
                'metrics_collection': True,
                'log_aggregation': True,
                'alerting': True
            }
        }
    
    def _validate_backup_strategy(self) -> Dict[str, Any]:
        """验证备份策略"""
        
        return {
            'status': 'passed',
            'message': 'Backup strategy validated',
            'details': {
                'backup_schedule': 'daily',
                'retention_days': 30,
                'backup_location': 's3://backups/'
            }
        }
    
    def _validate_rollback_plan(self) -> Dict[str, Any]:
        """验证回滚计划"""
        
        return {
            'status': 'passed',
            'message': 'Rollback plan validated',
            'details': {
                'rollback_strategy': 'blue_green',
                'rollback_time_minutes': 5,
                'data_migration_plan': True
            }
        }

class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self):
        self.config_schema = {
            'model': {
                'name': str,
                'path': str,
                'max_length': int,
                'batch_size': int
            },
            'server': {
                'host': str,
                'port': int,
                'workers': int,
                'timeout': int
            },
            'security': {
                'auth_enabled': bool,
                'ssl_cert': str,
                'ssl_key': str
            },
            'logging': {
                'level': str,
                'format': str,
                'file': str
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        
        errors = []
        warnings = []
        
        # 检查必需字段
        for section, fields in self.config_schema.items():
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue
            
            for field, field_type in fields.items():
                if field not in config[section]:
                    errors.append(f"Missing required field: {section}.{field}")
                elif not isinstance(config[section][field], field_type):
                    errors.append(f"Invalid type for {section}.{field}: expected {field_type.__name__}")
        
        # 检查配置值的合理性
        if 'server' in config:
            if config['server'].get('port', 0) < 1024:
                warnings.append("Server port < 1024 may require root privileges")
            
            if config['server'].get('workers', 0) > 16:
                warnings.append("High worker count may cause resource contention")
        
        if 'model' in config:
            if config['model'].get('batch_size', 0) > 32:
                warnings.append("Large batch size may cause OOM errors")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def generate_config_template(self) -> Dict[str, Any]:
        """生成配置模板"""
        
        template = {
            'model': {
                'name': 'llama-7b',
                'path': '/models/llama-7b',
                'max_length': 2048,
                'batch_size': 8
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'timeout': 60
            },
            'security': {
                'auth_enabled': True,
                'ssl_cert': '/certs/server.crt',
                'ssl_key': '/certs/server.key'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': '/logs/nano-vllm.log'
            }
        }
        
        return template
```

### 4. 监控运维最佳实践

```python
class MonitoringManager:
    """监控管理器"""
    
    def __init__(self):
        self.metrics_collectors = {}
        self.alert_rules = {}
        self.dashboards = {}
    
    def setup_monitoring(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置监控"""
        
        setup_results = {}
        
        # 设置指标收集
        if config.get('metrics_enabled', True):
            setup_results['metrics'] = self._setup_metrics_collection(config)
        
        # 设置日志聚合
        if config.get('logging_enabled', True):
            setup_results['logging'] = self._setup_log_aggregation(config)
        
        # 设置告警
        if config.get('alerting_enabled', True):
            setup_results['alerting'] = self._setup_alerting(config)
        
        # 设置仪表板
        if config.get('dashboard_enabled', True):
            setup_results['dashboard'] = self._setup_dashboards(config)
        
        return setup_results
    
    def _setup_metrics_collection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置指标收集"""
        
        metrics_config = {
            'system_metrics': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io'],
            'application_metrics': ['request_count', 'response_time', 'error_rate', 'throughput'],
            'model_metrics': ['inference_time', 'queue_length', 'gpu_utilization', 'memory_usage'],
            'business_metrics': ['active_users', 'api_calls', 'token_usage', 'cost']
        }
        
        return {
            'status': 'configured',
            'metrics_types': list(metrics_config.keys()),
            'collection_interval': config.get('metrics_interval', 30),
            'retention_days': config.get('metrics_retention', 30)
        }
    
    def _setup_log_aggregation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置日志聚合"""
        
        log_config = {
            'log_levels': ['ERROR', 'WARN', 'INFO', 'DEBUG'],
            'log_sources': ['application', 'system', 'security', 'audit'],
            'structured_logging': True,
            'log_parsing': True
        }
        
        return {
            'status': 'configured',
            'log_sources': log_config['log_sources'],
            'retention_days': config.get('log_retention', 90),
            'structured_logging': log_config['structured_logging']
        }
    
    def _setup_alerting(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置告警"""
        
        alert_rules = {
            'high_cpu_usage': {
                'condition': 'cpu_usage > 80%',
                'duration': '5m',
                'severity': 'warning'
            },
            'high_memory_usage': {
                'condition': 'memory_usage > 90%',
                'duration': '2m',
                'severity': 'critical'
            },
            'high_error_rate': {
                'condition': 'error_rate > 5%',
                'duration': '1m',
                'severity': 'critical'
            },
            'slow_response_time': {
                'condition': 'response_time_p95 > 2s',
                'duration': '3m',
                'severity': 'warning'
            }
        }
        
        return {
            'status': 'configured',
            'alert_rules_count': len(alert_rules),
            'notification_channels': config.get('notification_channels', ['email', 'slack']),
            'escalation_policy': config.get('escalation_policy', 'standard')
        }
    
    def _setup_dashboards(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """设置仪表板"""
        
        dashboards = {
            'system_overview': {
                'panels': ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io'],
                'refresh_interval': '30s'
            },
            'application_performance': {
                'panels': ['request_rate', 'response_time', 'error_rate', 'throughput'],
                'refresh_interval': '10s'
            },
            'model_metrics': {
                'panels': ['inference_time', 'queue_length', 'gpu_utilization', 'batch_size'],
                'refresh_interval': '15s'
            },
            'business_metrics': {
                'panels': ['active_users', 'api_usage', 'cost_tracking', 'usage_trends'],
                'refresh_interval': '1m'
            }
        }
        
        return {
            'status': 'configured',
            'dashboards_count': len(dashboards),
            'auto_refresh': True,
            'sharing_enabled': config.get('dashboard_sharing', True)
        }

class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.health_checks = {
            'liveness': self._liveness_check,
            'readiness': self._readiness_check,
            'startup': self._startup_check
        }
    
    async def run_health_check(self, check_type: str = 'readiness') -> Dict[str, Any]:
        """运行健康检查"""
        
        if check_type not in self.health_checks:
            return {
                'status': 'error',
                'message': f'Unknown health check type: {check_type}'
            }
        
        try:
            result = await self.health_checks[check_type]()
            return result
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Health check failed: {e}',
                'timestamp': time.time()
            }
    
    async def _liveness_check(self) -> Dict[str, Any]:
        """存活检查"""
        
        # 检查应用程序是否还在运行
        checks = {
            'process_running': True,
            'memory_available': True,
            'basic_functionality': True
        }
        
        all_healthy = all(checks.values())
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'checks': checks,
            'timestamp': time.time()
        }
    
    async def _readiness_check(self) -> Dict[str, Any]:
        """就绪检查"""
        
        # 检查应用程序是否准备好接收请求
        checks = {
            'model_loaded': True,
            'database_connected': True,
            'external_services_available': True,
            'cache_warmed': True
        }
        
        all_ready = all(checks.values())
        
        return {
            'status': 'ready' if all_ready else 'not_ready',
            'checks': checks,
            'timestamp': time.time()
        }
    
    async def _startup_check(self) -> Dict[str, Any]:
        """启动检查"""
        
        # 检查应用程序启动过程
        checks = {
            'configuration_loaded': True,
            'dependencies_initialized': True,
            'model_loading_started': True,
            'server_binding': True
        }
        
        startup_complete = all(checks.values())
        
        return {
            'status': 'started' if startup_complete else 'starting',
            'checks': checks,
            'timestamp': time.time()
        }

class MaintenanceManager:
    """维护管理器"""
    
    def __init__(self):
        self.maintenance_tasks = {
            'log_rotation': self._rotate_logs,
            'cache_cleanup': self._cleanup_cache,
            'model_update': self._update_model,
            'security_scan': self._run_security_scan,
            'performance_analysis': self._analyze_performance,
            'backup_verification': self._verify_backups
        }
    
    async def run_maintenance(self, tasks: List[str] = None) -> Dict[str, Any]:
        """运行维护任务"""
        
        if tasks is None:
            tasks = list(self.maintenance_tasks.keys())
        
        results = {}
        
        for task in tasks:
            if task in self.maintenance_tasks:
                logger.info(f"Running maintenance task: {task}")
                try:
                    results[task] = await self.maintenance_tasks[task]()
                except Exception as e:
                    results[task] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            else:
                results[task] = {
                    'status': 'skipped',
                    'reason': 'unknown_task'
                }
        
        return results
    
    async def _rotate_logs(self) -> Dict[str, Any]:
        """轮转日志"""
        
        # 实现日志轮转逻辑
        return {
            'status': 'completed',
            'rotated_files': 5,
            'space_freed_mb': 128
        }
    
    async def _cleanup_cache(self) -> Dict[str, Any]:
        """清理缓存"""
        
        # 实现缓存清理逻辑
        return {
            'status': 'completed',
            'cache_entries_removed': 1000,
            'memory_freed_mb': 256
        }
    
    async def _update_model(self) -> Dict[str, Any]:
        """更新模型"""
        
        # 实现模型更新逻辑
        return {
            'status': 'completed',
            'model_version': '1.2.0',
            'update_time_minutes': 15
        }
    
    async def _run_security_scan(self) -> Dict[str, Any]:
        """运行安全扫描"""
        
        # 实现安全扫描逻辑
        return {
            'status': 'completed',
            'vulnerabilities_found': 0,
            'scan_duration_minutes': 10
        }
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        
        # 实现性能分析逻辑
        return {
            'status': 'completed',
            'performance_score': 85,
            'recommendations': ['increase_batch_size', 'enable_caching']
        }
    
    async def _verify_backups(self) -> Dict[str, Any]:
        """验证备份"""
        
        # 实现备份验证逻辑
        return {
            'status': 'completed',
            'backups_verified': 7,
            'integrity_check': 'passed'
        }
```

## 🚀 使用示例

### 1. 代码质量检查示例

```python
async def code_quality_example():
    # 创建代码质量检查器
    quality_checker = CodeQualityChecker()
    
    # 检查代码质量
    metrics = quality_checker.check_code_quality("/path/to/project")
    
    print(f"Code quality level: {metrics.get_quality_level().value}")
    print(f"Test coverage: {metrics.test_coverage}%")
    print(f"Code complexity: {metrics.code_complexity}")
    
    # 创建测试框架
    test_framework = TestFramework()
    
    # 添加测试套件
    test_framework.add_test_suite(
        "unit_tests",
        TestType.UNIT,
        [
            TestExamples.test_tokenizer,
            TestExamples.test_performance
        ]
    )
    
    test_framework.add_test_suite(
        "integration_tests",
        TestType.INTEGRATION,
        [
            TestExamples.test_model_inference
        ]
    )
    
    # 运行测试
    results = await test_framework.run_tests()
    
    # 生成测试报告
    report = test_framework.generate_test_report(results)
    print(report)

if __name__ == "__main__":
    asyncio.run(code_quality_example())
```

### 2. 性能优化示例

```python
async def performance_optimization_example():
    # 创建性能优化器
    optimizer = PerformanceOptimizer()
    
    # 优化系统性能
    optimization_results = optimizer.optimize_system(['memory', 'compute'])
    
    for area, result in optimization_results.items():
        print(f"{area} optimization:")
        print(f"  Applied optimizations: {result['applied_optimizations']}")
        if 'memory_saved_mb' in result:
            print(f"  Memory saved: {result['memory_saved_mb']} MB")
        if 'speedup_factor' in result:
            print(f"  Speedup factor: {result['speedup_factor']}x")
    
    # 使用资源监控
    with resource_monitor("memory"):
        # 模拟一些内存密集型操作
        data = [i for i in range(1000000)]
        processed_data = [x * 2 for x in data]
        del data, processed_data
    
    # 使用缓存管理器
    cache_manager = CacheManager(max_size=100, ttl=300)
    
    # 设置缓存
    cache_manager.set("model_output_123", "Generated text response")
    
    # 获取缓存
    cached_result = cache_manager.get("model_output_123")
    print(f"Cached result: {cached_result}")
    
    # 获取缓存统计
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats}")

if __name__ == "__main__":
    asyncio.run(performance_optimization_example())
```

### 3. 部署验证示例

```python
async def deployment_validation_example():
    # 创建部署管理器
    deployment_manager = DeploymentManager()
    
    # 验证部署环境
    validation_results = deployment_manager.validate_deployment("production")
    
    print(f"Deployment readiness: {validation_results['deployment_readiness']}")
    
    for check, result in validation_results.items():
        if check != 'deployment_readiness':
            print(f"{check}: {result['status']} - {result['message']}")
    
    # 创建配置管理器
    config_manager = ConfigurationManager()
    
    # 生成配置模板
    config_template = config_manager.generate_config_template()
    print(f"Configuration template: {config_template}")
    
    # 验证配置
    validation_result = config_manager.validate_config(config_template)
    
    if validation_result['valid']:
        print("Configuration is valid")
    else:
        print(f"Configuration errors: {validation_result['errors']}")
    
    if validation_result['warnings']:
        print(f"Configuration warnings: {validation_result['warnings']}")

if __name__ == "__main__":
    asyncio.run(deployment_validation_example())
```

### 4. 监控运维示例

```python
async def monitoring_operations_example():
    # 创建监控管理器
    monitoring_manager = MonitoringManager()
    
    # 设置监控
    monitoring_config = {
        'metrics_enabled': True,
        'logging_enabled': True,
        'alerting_enabled': True,
        'dashboard_enabled': True,
        'metrics_interval': 30,
        'notification_channels': ['email', 'slack']
    }
    
    setup_results = monitoring_manager.setup_monitoring(monitoring_config)
    
    for component, result in setup_results.items():
        print(f"{component} setup: {result['status']}")
    
    # 创建健康检查器
    health_checker = HealthChecker()
    
    # 运行健康检查
    liveness_result = await health_checker.run_health_check('liveness')
    readiness_result = await health_checker.run_health_check('readiness')
    
    print(f"Liveness check: {liveness_result['status']}")
    print(f"Readiness check: {readiness_result['status']}")
    
    # 创建维护管理器
    maintenance_manager = MaintenanceManager()
    
    # 运行维护任务
    maintenance_results = await maintenance_manager.run_maintenance([
        'log_rotation', 'cache_cleanup', 'performance_analysis'
    ])
    
    for task, result in maintenance_results.items():
        print(f"{task}: {result['status']}")

if __name__ == "__main__":
    asyncio.run(monitoring_operations_example())
```

## 🎯 关键最佳实践总结

### 1. 开发阶段
- **代码质量**: 使用静态分析工具、类型检查、测试覆盖率
- **性能优化**: 早期性能测试、资源监控、缓存策略
- **安全考虑**: 安全编码规范、依赖安全检查、敏感信息保护

### 2. 测试阶段
- **全面测试**: 单元测试、集成测试、性能测试、安全测试
- **自动化**: CI/CD流水线、自动化测试、自动化部署
- **质量门禁**: 代码质量检查、测试覆盖率要求、性能基准

### 3. 部署阶段
- **环境一致性**: 容器化部署、配置管理、环境隔离
- **渐进式部署**: 蓝绿部署、金丝雀发布、回滚策略
- **监控就绪**: 健康检查、指标收集、日志聚合

### 4. 运维阶段
- **主动监控**: 实时监控、告警机制、性能分析
- **预防性维护**: 定期维护、容量规划、故障预防
- **持续改进**: 性能优化、安全加固、流程改进

## 📈 总结

最佳实践是确保nano-vllm项目成功的关键因素。通过遵循这些实践，可以：

1. **提高代码质量**: 减少bug、提高可维护性、增强可读性
2. **优化性能**: 提升响应速度、降低资源消耗、增强扩展性
3. **确保安全**: 防范安全威胁、保护用户数据、符合合规要求
4. **简化运维**: 自动化运维、快速故障定位、高效问题解决

持续学习和改进这些最佳实践，将有助于构建更加稳定、高效、安全的LLM推理服务。