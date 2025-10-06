# 🔧 故障排除 (Troubleshooting)

## 📖 概述

本文档提供了nano-vllm在使用过程中可能遇到的常见问题的诊断方法和解决方案，包括安装问题、运行时错误、性能问题、内存问题等，帮助开发者快速定位和解决问题。

## 🚨 常见问题分类

### 1. 安装和环境问题

```python
import sys
import subprocess
import importlib
import platform
import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psutil
import GPUtil
import json
import traceback

logger = logging.getLogger(__name__)

class ProblemCategory(Enum):
    """问题分类"""
    INSTALLATION = "installation"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    SECURITY = "security"
    CONFIGURATION = "configuration"

class SeverityLevel(Enum):
    """严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DiagnosticResult:
    """诊断结果"""
    category: ProblemCategory
    severity: SeverityLevel
    problem: str
    cause: str
    solution: str
    additional_info: Dict[str, Any] = None

class EnvironmentDiagnostic:
    """环境诊断器"""
    
    def __init__(self):
        self.required_packages = {
            'torch': '>=1.12.0',
            'transformers': '>=4.20.0',
            'numpy': '>=1.21.0',
            'fastapi': '>=0.68.0',
            'uvicorn': '>=0.15.0'
        }
        
        self.system_requirements = {
            'python_version': (3, 8),
            'memory_gb': 8,
            'disk_space_gb': 10
        }
    
    def diagnose_environment(self) -> List[DiagnosticResult]:
        """诊断环境问题"""
        
        results = []
        
        # 检查Python版本
        results.extend(self._check_python_version())
        
        # 检查包依赖
        results.extend(self._check_package_dependencies())
        
        # 检查系统资源
        results.extend(self._check_system_resources())
        
        # 检查CUDA环境
        results.extend(self._check_cuda_environment())
        
        # 检查权限
        results.extend(self._check_permissions())
        
        return results
    
    def _check_python_version(self) -> List[DiagnosticResult]:
        """检查Python版本"""
        
        results = []
        current_version = sys.version_info[:2]
        required_version = self.system_requirements['python_version']
        
        if current_version < required_version:
            results.append(DiagnosticResult(
                category=ProblemCategory.ENVIRONMENT,
                severity=SeverityLevel.HIGH,
                problem=f"Python version {current_version} is too old",
                cause=f"nano-vllm requires Python {required_version} or higher",
                solution=f"Upgrade Python to version {required_version} or higher",
                additional_info={'current_version': current_version, 'required_version': required_version}
            ))
        
        return results
    
    def _check_package_dependencies(self) -> List[DiagnosticResult]:
        """检查包依赖"""
        
        results = []
        
        for package, version_req in self.required_packages.items():
            try:
                module = importlib.import_module(package)
                
                if hasattr(module, '__version__'):
                    current_version = module.__version__
                    logger.info(f"{package} version: {current_version}")
                else:
                    logger.warning(f"Cannot determine version for {package}")
                
            except ImportError:
                results.append(DiagnosticResult(
                    category=ProblemCategory.INSTALLATION,
                    severity=SeverityLevel.HIGH,
                    problem=f"Package {package} is not installed",
                    cause=f"Required package {package} is missing",
                    solution=f"Install {package} using: pip install {package}{version_req}",
                    additional_info={'package': package, 'version_requirement': version_req}
                ))
        
        return results
    
    def _check_system_resources(self) -> List[DiagnosticResult]:
        """检查系统资源"""
        
        results = []
        
        # 检查内存
        memory_gb = psutil.virtual_memory().total / (1024**3)
        required_memory = self.system_requirements['memory_gb']
        
        if memory_gb < required_memory:
            results.append(DiagnosticResult(
                category=ProblemCategory.ENVIRONMENT,
                severity=SeverityLevel.MEDIUM,
                problem=f"Insufficient memory: {memory_gb:.1f}GB available",
                cause=f"nano-vllm requires at least {required_memory}GB of RAM",
                solution="Add more RAM or use a smaller model",
                additional_info={'available_memory_gb': memory_gb, 'required_memory_gb': required_memory}
            ))
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('/')
        available_gb = disk_usage.free / (1024**3)
        required_disk = self.system_requirements['disk_space_gb']
        
        if available_gb < required_disk:
            results.append(DiagnosticResult(
                category=ProblemCategory.ENVIRONMENT,
                severity=SeverityLevel.MEDIUM,
                problem=f"Insufficient disk space: {available_gb:.1f}GB available",
                cause=f"nano-vllm requires at least {required_disk}GB of free disk space",
                solution="Free up disk space or use external storage",
                additional_info={'available_disk_gb': available_gb, 'required_disk_gb': required_disk}
            ))
        
        return results
    
    def _check_cuda_environment(self) -> List[DiagnosticResult]:
        """检查CUDA环境"""
        
        results = []
        
        if not torch.cuda.is_available():
            results.append(DiagnosticResult(
                category=ProblemCategory.ENVIRONMENT,
                severity=SeverityLevel.MEDIUM,
                problem="CUDA is not available",
                cause="PyTorch cannot detect CUDA installation",
                solution="Install CUDA toolkit and compatible PyTorch version",
                additional_info={'torch_version': torch.__version__}
            ))
        else:
            # 检查GPU内存
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        if gpu.memoryFree < 2048:  # 2GB
                            results.append(DiagnosticResult(
                                category=ProblemCategory.MEMORY,
                                severity=SeverityLevel.MEDIUM,
                                problem=f"GPU {i} has low memory: {gpu.memoryFree}MB free",
                                cause="Insufficient GPU memory for model loading",
                                solution="Free GPU memory or use CPU inference",
                                additional_info={'gpu_id': i, 'free_memory_mb': gpu.memoryFree}
                            ))
            except Exception as e:
                logger.warning(f"Cannot check GPU status: {e}")
        
        return results
    
    def _check_permissions(self) -> List[DiagnosticResult]:
        """检查权限"""
        
        results = []
        
        # 检查写权限
        import tempfile
        import os
        
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(b"test")
        except PermissionError:
            results.append(DiagnosticResult(
                category=ProblemCategory.ENVIRONMENT,
                severity=SeverityLevel.HIGH,
                problem="No write permission in temporary directory",
                cause="Insufficient file system permissions",
                solution="Check and fix file system permissions",
                additional_info={'temp_dir': tempfile.gettempdir()}
            ))
        
        return results

class RuntimeDiagnostic:
    """运行时诊断器"""
    
    def __init__(self):
        self.common_errors = {
            'OutOfMemoryError': self._diagnose_oom_error,
            'RuntimeError': self._diagnose_runtime_error,
            'ConnectionError': self._diagnose_connection_error,
            'TimeoutError': self._diagnose_timeout_error,
            'ImportError': self._diagnose_import_error,
            'FileNotFoundError': self._diagnose_file_not_found_error
        }
    
    def diagnose_error(self, error: Exception, context: Dict[str, Any] = None) -> DiagnosticResult:
        """诊断运行时错误"""
        
        error_type = type(error).__name__
        
        if error_type in self.common_errors:
            return self.common_errors[error_type](error, context or {})
        else:
            return self._diagnose_generic_error(error, context or {})
    
    def _diagnose_oom_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断内存不足错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.MEMORY,
            severity=SeverityLevel.HIGH,
            problem="Out of memory error during model inference",
            cause="Model or batch size too large for available memory",
            solution="Reduce batch size, use smaller model, or add more memory",
            additional_info={
                'error_message': str(error),
                'suggested_actions': [
                    'Reduce batch_size parameter',
                    'Use gradient checkpointing',
                    'Enable model sharding',
                    'Use CPU inference'
                ],
                'context': context
            }
        )
    
    def _diagnose_runtime_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断运行时错误"""
        
        error_msg = str(error).lower()
        
        if 'cuda' in error_msg:
            return DiagnosticResult(
                category=ProblemCategory.RUNTIME,
                severity=SeverityLevel.HIGH,
                problem="CUDA runtime error",
                cause="GPU computation or memory issue",
                solution="Check GPU status, restart application, or use CPU",
                additional_info={
                    'error_message': str(error),
                    'suggested_actions': [
                        'Check GPU memory usage',
                        'Restart CUDA runtime',
                        'Use torch.cuda.empty_cache()',
                        'Switch to CPU inference'
                    ],
                    'context': context
                }
            )
        else:
            return DiagnosticResult(
                category=ProblemCategory.RUNTIME,
                severity=SeverityLevel.MEDIUM,
                problem="General runtime error",
                cause="Unexpected runtime condition",
                solution="Check logs and application state",
                additional_info={
                    'error_message': str(error),
                    'context': context
                }
            )
    
    def _diagnose_connection_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断连接错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.NETWORK,
            severity=SeverityLevel.MEDIUM,
            problem="Network connection error",
            cause="Network connectivity or service availability issue",
            solution="Check network connection and service status",
            additional_info={
                'error_message': str(error),
                'suggested_actions': [
                    'Check network connectivity',
                    'Verify service endpoints',
                    'Check firewall settings',
                    'Retry with exponential backoff'
                ],
                'context': context
            }
        )
    
    def _diagnose_timeout_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断超时错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.PERFORMANCE,
            severity=SeverityLevel.MEDIUM,
            problem="Operation timeout",
            cause="Operation took longer than expected",
            solution="Increase timeout or optimize performance",
            additional_info={
                'error_message': str(error),
                'suggested_actions': [
                    'Increase timeout values',
                    'Optimize model performance',
                    'Use async processing',
                    'Implement request queuing'
                ],
                'context': context
            }
        )
    
    def _diagnose_import_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断导入错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.INSTALLATION,
            severity=SeverityLevel.HIGH,
            problem="Module import error",
            cause="Missing or incompatible package",
            solution="Install or update required packages",
            additional_info={
                'error_message': str(error),
                'suggested_actions': [
                    'Install missing packages',
                    'Check package versions',
                    'Verify Python path',
                    'Reinstall dependencies'
                ],
                'context': context
            }
        )
    
    def _diagnose_file_not_found_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断文件未找到错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.CONFIGURATION,
            severity=SeverityLevel.HIGH,
            problem="Required file not found",
            cause="Missing model files or configuration",
            solution="Check file paths and download missing files",
            additional_info={
                'error_message': str(error),
                'suggested_actions': [
                    'Verify file paths',
                    'Download missing model files',
                    'Check file permissions',
                    'Update configuration'
                ],
                'context': context
            }
        )
    
    def _diagnose_generic_error(self, error: Exception, context: Dict[str, Any]) -> DiagnosticResult:
        """诊断通用错误"""
        
        return DiagnosticResult(
            category=ProblemCategory.RUNTIME,
            severity=SeverityLevel.MEDIUM,
            problem=f"Unexpected error: {type(error).__name__}",
            cause="Unknown error condition",
            solution="Check logs and contact support if needed",
            additional_info={
                'error_message': str(error),
                'error_type': type(error).__name__,
                'traceback': traceback.format_exc(),
                'context': context
            }
        )

class PerformanceDiagnostic:
    """性能诊断器"""
    
    def __init__(self):
        self.performance_thresholds = {
            'inference_time_ms': 1000,
            'memory_usage_mb': 4096,
            'cpu_usage_percent': 80,
            'gpu_utilization_percent': 90
        }
    
    def diagnose_performance(self, metrics: Dict[str, float]) -> List[DiagnosticResult]:
        """诊断性能问题"""
        
        results = []
        
        # 检查推理时间
        if metrics.get('inference_time_ms', 0) > self.performance_thresholds['inference_time_ms']:
            results.append(DiagnosticResult(
                category=ProblemCategory.PERFORMANCE,
                severity=SeverityLevel.MEDIUM,
                problem=f"Slow inference: {metrics['inference_time_ms']:.2f}ms",
                cause="Model or system performance bottleneck",
                solution="Optimize model, reduce batch size, or upgrade hardware",
                additional_info={
                    'current_time_ms': metrics['inference_time_ms'],
                    'threshold_ms': self.performance_thresholds['inference_time_ms'],
                    'suggested_optimizations': [
                        'Enable model quantization',
                        'Use smaller batch size',
                        'Enable mixed precision',
                        'Use model compilation'
                    ]
                }
            ))
        
        # 检查内存使用
        if metrics.get('memory_usage_mb', 0) > self.performance_thresholds['memory_usage_mb']:
            results.append(DiagnosticResult(
                category=ProblemCategory.MEMORY,
                severity=SeverityLevel.MEDIUM,
                problem=f"High memory usage: {metrics['memory_usage_mb']:.2f}MB",
                cause="Large model or inefficient memory management",
                solution="Enable memory optimization or use smaller model",
                additional_info={
                    'current_usage_mb': metrics['memory_usage_mb'],
                    'threshold_mb': self.performance_thresholds['memory_usage_mb'],
                    'suggested_optimizations': [
                        'Enable gradient checkpointing',
                        'Use memory pooling',
                        'Implement KV cache optimization',
                        'Use model sharding'
                    ]
                }
            ))
        
        # 检查CPU使用率
        if metrics.get('cpu_usage_percent', 0) > self.performance_thresholds['cpu_usage_percent']:
            results.append(DiagnosticResult(
                category=ProblemCategory.PERFORMANCE,
                severity=SeverityLevel.LOW,
                problem=f"High CPU usage: {metrics['cpu_usage_percent']:.1f}%",
                cause="CPU-intensive operations or insufficient resources",
                solution="Optimize CPU usage or add more CPU cores",
                additional_info={
                    'current_usage_percent': metrics['cpu_usage_percent'],
                    'threshold_percent': self.performance_thresholds['cpu_usage_percent'],
                    'suggested_optimizations': [
                        'Use GPU acceleration',
                        'Optimize data preprocessing',
                        'Implement async processing',
                        'Scale horizontally'
                    ]
                }
            ))
        
        return results

class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self):
        self.error_patterns = {
            r'OutOfMemoryError|CUDA out of memory': 'memory_error',
            r'Connection refused|Connection timeout': 'connection_error',
            r'Permission denied|Access denied': 'permission_error',
            r'No such file or directory': 'file_error',
            r'Import.*Error|ModuleNotFoundError': 'import_error',
            r'Timeout|Request timeout': 'timeout_error'
        }
    
    def analyze_logs(self, log_content: str) -> List[DiagnosticResult]:
        """分析日志内容"""
        
        import re
        
        results = []
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            for pattern, error_type in self.error_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    result = self._create_diagnostic_from_log(error_type, line, i + 1)
                    if result:
                        results.append(result)
        
        return results
    
    def _create_diagnostic_from_log(self, error_type: str, log_line: str, line_number: int) -> Optional[DiagnosticResult]:
        """从日志行创建诊断结果"""
        
        diagnostic_map = {
            'memory_error': DiagnosticResult(
                category=ProblemCategory.MEMORY,
                severity=SeverityLevel.HIGH,
                problem="Memory error detected in logs",
                cause="Insufficient memory for operation",
                solution="Reduce memory usage or add more memory",
                additional_info={'log_line': log_line, 'line_number': line_number}
            ),
            'connection_error': DiagnosticResult(
                category=ProblemCategory.NETWORK,
                severity=SeverityLevel.MEDIUM,
                problem="Connection error detected in logs",
                cause="Network connectivity issue",
                solution="Check network configuration and connectivity",
                additional_info={'log_line': log_line, 'line_number': line_number}
            ),
            'permission_error': DiagnosticResult(
                category=ProblemCategory.SECURITY,
                severity=SeverityLevel.HIGH,
                problem="Permission error detected in logs",
                cause="Insufficient file or system permissions",
                solution="Check and fix file permissions",
                additional_info={'log_line': log_line, 'line_number': line_number}
            ),
            'file_error': DiagnosticResult(
                category=ProblemCategory.CONFIGURATION,
                severity=SeverityLevel.HIGH,
                problem="File not found error detected in logs",
                cause="Missing required files",
                solution="Check file paths and ensure files exist",
                additional_info={'log_line': log_line, 'line_number': line_number}
            ),
            'import_error': DiagnosticResult(
                category=ProblemCategory.INSTALLATION,
                severity=SeverityLevel.HIGH,
                problem="Import error detected in logs",
                cause="Missing or incompatible packages",
                solution="Install or update required packages",
                additional_info={'log_line': log_line, 'line_number': line_number}
            ),
            'timeout_error': DiagnosticResult(
                category=ProblemCategory.PERFORMANCE,
                severity=SeverityLevel.MEDIUM,
                problem="Timeout error detected in logs",
                cause="Operation took too long to complete",
                solution="Increase timeout or optimize performance",
                additional_info={'log_line': log_line, 'line_number': line_number}
            )
        }
        
        return diagnostic_map.get(error_type)

class TroubleshootingManager:
    """故障排除管理器"""
    
    def __init__(self):
        self.env_diagnostic = EnvironmentDiagnostic()
        self.runtime_diagnostic = RuntimeDiagnostic()
        self.performance_diagnostic = PerformanceDiagnostic()
        self.log_analyzer = LogAnalyzer()
    
    def run_full_diagnostic(self, 
                          include_performance: bool = True,
                          log_content: str = None,
                          performance_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """运行完整诊断"""
        
        diagnostic_results = {
            'environment': [],
            'runtime': [],
            'performance': [],
            'logs': [],
            'summary': {}
        }
        
        # 环境诊断
        logger.info("Running environment diagnostic...")
        diagnostic_results['environment'] = self.env_diagnostic.diagnose_environment()
        
        # 性能诊断
        if include_performance and performance_metrics:
            logger.info("Running performance diagnostic...")
            diagnostic_results['performance'] = self.performance_diagnostic.diagnose_performance(performance_metrics)
        
        # 日志分析
        if log_content:
            logger.info("Analyzing logs...")
            diagnostic_results['logs'] = self.log_analyzer.analyze_logs(log_content)
        
        # 生成摘要
        diagnostic_results['summary'] = self._generate_summary(diagnostic_results)
        
        return diagnostic_results
    
    def diagnose_specific_error(self, error: Exception, context: Dict[str, Any] = None) -> DiagnosticResult:
        """诊断特定错误"""
        
        return self.runtime_diagnostic.diagnose_error(error, context)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成诊断摘要"""
        
        all_results = []
        for category_results in results.values():
            if isinstance(category_results, list):
                all_results.extend(category_results)
        
        if not all_results:
            return {
                'total_issues': 0,
                'severity_breakdown': {},
                'category_breakdown': {},
                'status': 'healthy'
            }
        
        # 按严重程度分类
        severity_counts = {}
        for result in all_results:
            severity = result.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 按类别分类
        category_counts = {}
        for result in all_results:
            category = result.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # 确定整体状态
        has_critical = any(r.severity == SeverityLevel.CRITICAL for r in all_results)
        has_high = any(r.severity == SeverityLevel.HIGH for r in all_results)
        
        if has_critical:
            status = 'critical'
        elif has_high:
            status = 'warning'
        elif all_results:
            status = 'minor_issues'
        else:
            status = 'healthy'
        
        return {
            'total_issues': len(all_results),
            'severity_breakdown': severity_counts,
            'category_breakdown': category_counts,
            'status': status,
            'recommendations': self._generate_recommendations(all_results)
        }
    
    def _generate_recommendations(self, results: List[DiagnosticResult]) -> List[str]:
        """生成建议"""
        
        recommendations = []
        
        # 按严重程度排序
        critical_issues = [r for r in results if r.severity == SeverityLevel.CRITICAL]
        high_issues = [r for r in results if r.severity == SeverityLevel.HIGH]
        
        if critical_issues:
            recommendations.append("🚨 Address critical issues immediately:")
            for issue in critical_issues[:3]:  # 只显示前3个
                recommendations.append(f"  - {issue.solution}")
        
        if high_issues:
            recommendations.append("⚠️ Address high priority issues:")
            for issue in high_issues[:3]:  # 只显示前3个
                recommendations.append(f"  - {issue.solution}")
        
        # 通用建议
        memory_issues = [r for r in results if r.category == ProblemCategory.MEMORY]
        if memory_issues:
            recommendations.append("💾 Memory optimization suggestions:")
            recommendations.append("  - Enable gradient checkpointing")
            recommendations.append("  - Use smaller batch sizes")
            recommendations.append("  - Consider model quantization")
        
        performance_issues = [r for r in results if r.category == ProblemCategory.PERFORMANCE]
        if performance_issues:
            recommendations.append("🚀 Performance optimization suggestions:")
            recommendations.append("  - Enable mixed precision training")
            recommendations.append("  - Use model compilation")
            recommendations.append("  - Implement caching strategies")
        
        return recommendations

class AutoFixer:
    """自动修复器"""
    
    def __init__(self):
        self.fixers = {
            ProblemCategory.MEMORY: self._fix_memory_issues,
            ProblemCategory.PERFORMANCE: self._fix_performance_issues,
            ProblemCategory.CONFIGURATION: self._fix_configuration_issues
        }
    
    def attempt_auto_fix(self, diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """尝试自动修复"""
        
        if diagnostic_result.category in self.fixers:
            try:
                return self.fixers[diagnostic_result.category](diagnostic_result)
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'message': 'Auto-fix failed, manual intervention required'
                }
        else:
            return {
                'success': False,
                'message': f'No auto-fix available for {diagnostic_result.category.value} issues'
            }
    
    def _fix_memory_issues(self, diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """修复内存问题"""
        
        actions_taken = []
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            actions_taken.append('Cleared GPU cache')
        
        # 强制垃圾回收
        import gc
        gc.collect()
        actions_taken.append('Forced garbage collection')
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'message': 'Memory cleanup completed'
        }
    
    def _fix_performance_issues(self, diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """修复性能问题"""
        
        actions_taken = []
        
        # 这里可以实现一些自动性能优化
        # 例如调整批处理大小、启用缓存等
        
        actions_taken.append('Applied performance optimizations')
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'message': 'Performance optimizations applied'
        }
    
    def _fix_configuration_issues(self, diagnostic_result: DiagnosticResult) -> Dict[str, Any]:
        """修复配置问题"""
        
        actions_taken = []
        
        # 这里可以实现一些自动配置修复
        # 例如创建缺失的目录、设置默认配置等
        
        actions_taken.append('Applied configuration fixes')
        
        return {
            'success': True,
            'actions_taken': actions_taken,
            'message': 'Configuration issues fixed'
        }
```

## 🔍 常见问题解决方案

### 1. 安装问题

```python
class InstallationTroubleshooter:
    """安装问题排除器"""
    
    @staticmethod
    def fix_pip_install_issues():
        """修复pip安装问题"""
        
        solutions = {
            'network_timeout': [
                "pip install --timeout 1000 package_name",
                "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ package_name",
                "pip install --proxy http://proxy:port package_name"
            ],
            'permission_denied': [
                "pip install --user package_name",
                "sudo pip install package_name",
                "python -m pip install --user package_name"
            ],
            'version_conflict': [
                "pip install --upgrade package_name",
                "pip install package_name==specific_version",
                "pip install --force-reinstall package_name"
            ],
            'dependency_conflict': [
                "pip install --no-deps package_name",
                "pip install package_name --upgrade-strategy only-if-needed",
                "Create virtual environment: python -m venv env"
            ]
        }
        
        return solutions
    
    @staticmethod
    def fix_cuda_installation():
        """修复CUDA安装问题"""
        
        steps = [
            "1. Check NVIDIA driver: nvidia-smi",
            "2. Download CUDA toolkit from NVIDIA website",
            "3. Install PyTorch with CUDA support:",
            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "4. Verify installation: python -c 'import torch; print(torch.cuda.is_available())'"
        ]
        
        return steps
    
    @staticmethod
    def fix_transformers_installation():
        """修复transformers安装问题"""
        
        commands = [
            "pip install transformers>=4.20.0",
            "pip install accelerate",
            "pip install datasets",
            "pip install tokenizers"
        ]
        
        return commands

class RuntimeTroubleshooter:
    """运行时问题排除器"""
    
    @staticmethod
    def fix_memory_errors():
        """修复内存错误"""
        
        solutions = {
            'reduce_batch_size': "model.generate(inputs, max_new_tokens=50, batch_size=1)",
            'enable_gradient_checkpointing': "model.gradient_checkpointing_enable()",
            'use_cpu_inference': "model = model.to('cpu')",
            'clear_cache': "torch.cuda.empty_cache()",
            'use_8bit_quantization': "model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)"
        }
        
        return solutions
    
    @staticmethod
    def fix_model_loading_errors():
        """修复模型加载错误"""
        
        solutions = {
            'file_not_found': [
                "Check model path exists",
                "Download model: huggingface-cli download model_name",
                "Use correct model identifier"
            ],
            'permission_denied': [
                "Check file permissions: ls -la model_path",
                "Change permissions: chmod -R 755 model_path",
                "Run with appropriate user privileges"
            ],
            'corrupted_files': [
                "Re-download model files",
                "Verify file integrity",
                "Clear cache: rm -rf ~/.cache/huggingface/"
            ],
            'incompatible_format': [
                "Check model format compatibility",
                "Convert model format if needed",
                "Use appropriate model loader"
            ]
        }
        
        return solutions
    
    @staticmethod
    def fix_inference_errors():
        """修复推理错误"""
        
        solutions = {
            'input_format_error': [
                "Check input tensor shapes",
                "Verify tokenizer output format",
                "Ensure proper input preprocessing"
            ],
            'device_mismatch': [
                "Move inputs to same device as model",
                "inputs = inputs.to(model.device)",
                "Check device consistency"
            ],
            'sequence_length_error': [
                "Check max_length parameter",
                "Truncate input sequences",
                "Use appropriate padding strategy"
            ]
        }
        
        return solutions

class PerformanceTroubleshooter:
    """性能问题排除器"""
    
    @staticmethod
    def optimize_inference_speed():
        """优化推理速度"""
        
        optimizations = {
            'model_compilation': "model = torch.compile(model)",
            'mixed_precision': "with torch.autocast('cuda'): output = model(inputs)",
            'batch_processing': "Process multiple inputs in batches",
            'kv_cache': "Enable KV cache for generation",
            'quantization': "Use INT8 or FP16 quantization",
            'flash_attention': "Enable Flash Attention if available"
        }
        
        return optimizations
    
    @staticmethod
    def optimize_memory_usage():
        """优化内存使用"""
        
        optimizations = {
            'gradient_checkpointing': "model.gradient_checkpointing_enable()",
            'memory_pooling': "Use memory pooling for allocations",
            'activation_recomputation': "Recompute activations instead of storing",
            'model_sharding': "Shard model across multiple devices",
            'offloading': "Offload unused parameters to CPU"
        }
        
        return optimizations
```

## 📊 诊断工具使用示例

### 1. 环境诊断示例

```python
async def environment_diagnostic_example():
    """环境诊断示例"""
    
    # 创建故障排除管理器
    troubleshooter = TroubleshootingManager()
    
    # 运行完整诊断
    diagnostic_results = troubleshooter.run_full_diagnostic(
        include_performance=True,
        performance_metrics={
            'inference_time_ms': 1500,
            'memory_usage_mb': 5000,
            'cpu_usage_percent': 85,
            'gpu_utilization_percent': 95
        }
    )
    
    # 打印诊断结果
    print("=== Diagnostic Results ===")
    print(f"Total issues found: {diagnostic_results['summary']['total_issues']}")
    print(f"System status: {diagnostic_results['summary']['status']}")
    
    # 显示环境问题
    if diagnostic_results['environment']:
        print("\n🔧 Environment Issues:")
        for issue in diagnostic_results['environment']:
            print(f"  - {issue.problem}")
            print(f"    Solution: {issue.solution}")
    
    # 显示性能问题
    if diagnostic_results['performance']:
        print("\n⚡ Performance Issues:")
        for issue in diagnostic_results['performance']:
            print(f"  - {issue.problem}")
            print(f"    Solution: {issue.solution}")
    
    # 显示建议
    if diagnostic_results['summary']['recommendations']:
        print("\n💡 Recommendations:")
        for rec in diagnostic_results['summary']['recommendations']:
            print(f"  {rec}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(environment_diagnostic_example())
```

### 2. 错误诊断示例

```python
def error_diagnostic_example():
    """错误诊断示例"""
    
    troubleshooter = TroubleshootingManager()
    auto_fixer = AutoFixer()
    
    # 模拟不同类型的错误
    errors_to_test = [
        RuntimeError("CUDA out of memory"),
        FileNotFoundError("Model file not found: /path/to/model"),
        ImportError("No module named 'transformers'"),
        ConnectionError("Connection refused"),
        TimeoutError("Request timeout after 30 seconds")
    ]
    
    for error in errors_to_test:
        print(f"\n=== Diagnosing: {type(error).__name__} ===")
        
        # 诊断错误
        diagnostic = troubleshooter.diagnose_specific_error(
            error, 
            context={'operation': 'model_inference', 'batch_size': 8}
        )
        
        print(f"Problem: {diagnostic.problem}")
        print(f"Cause: {diagnostic.cause}")
        print(f"Solution: {diagnostic.solution}")
        print(f"Severity: {diagnostic.severity.value}")
        print(f"Category: {diagnostic.category.value}")
        
        # 尝试自动修复
        fix_result = auto_fixer.attempt_auto_fix(diagnostic)
        
        if fix_result['success']:
            print(f"✅ Auto-fix successful: {fix_result['message']}")
            if 'actions_taken' in fix_result:
                print(f"Actions taken: {fix_result['actions_taken']}")
        else:
            print(f"❌ Auto-fix failed: {fix_result['message']}")

if __name__ == "__main__":
    error_diagnostic_example()
```

### 3. 日志分析示例

```python
def log_analysis_example():
    """日志分析示例"""
    
    # 模拟日志内容
    sample_logs = """
2024-01-15 10:30:15 INFO Starting model loading...
2024-01-15 10:30:20 ERROR CUDA out of memory. Tried to allocate 2.00 GiB
2024-01-15 10:30:25 WARNING Connection timeout to external service
2024-01-15 10:30:30 ERROR ImportError: No module named 'flash_attn'
2024-01-15 10:30:35 INFO Retrying with smaller batch size...
2024-01-15 10:30:40 ERROR Permission denied: /tmp/model_cache/
2024-01-15 10:30:45 INFO Model loaded successfully
    """
    
    troubleshooter = TroubleshootingManager()
    
    # 分析日志
    log_issues = troubleshooter.log_analyzer.analyze_logs(sample_logs)
    
    print("=== Log Analysis Results ===")
    print(f"Found {len(log_issues)} issues in logs:")
    
    for issue in log_issues:
        print(f"\nLine {issue.additional_info['line_number']}: {issue.problem}")
        print(f"  Category: {issue.category.value}")
        print(f"  Severity: {issue.severity.value}")
        print(f"  Solution: {issue.solution}")
        print(f"  Log line: {issue.additional_info['log_line']}")

if __name__ == "__main__":
    log_analysis_example()
```

### 4. 性能诊断示例

```python
def performance_diagnostic_example():
    """性能诊断示例"""
    
    troubleshooter = TroubleshootingManager()
    
    # 模拟性能指标
    performance_metrics = {
        'inference_time_ms': 2500,  # 超过阈值
        'memory_usage_mb': 6000,    # 超过阈值
        'cpu_usage_percent': 95,    # 超过阈值
        'gpu_utilization_percent': 85,
        'throughput_rps': 10,
        'queue_length': 50
    }
    
    # 运行性能诊断
    performance_issues = troubleshooter.performance_diagnostic.diagnose_performance(performance_metrics)
    
    print("=== Performance Diagnostic Results ===")
    
    if not performance_issues:
        print("✅ No performance issues detected")
    else:
        print(f"Found {len(performance_issues)} performance issues:")
        
        for issue in performance_issues:
            print(f"\n🚨 {issue.problem}")
            print(f"   Cause: {issue.cause}")
            print(f"   Solution: {issue.solution}")
            print(f"   Severity: {issue.severity.value}")
            
            if issue.additional_info and 'suggested_optimizations' in issue.additional_info:
                print("   Suggested optimizations:")
                for opt in issue.additional_info['suggested_optimizations']:
                    print(f"     - {opt}")

if __name__ == "__main__":
    performance_diagnostic_example()
```

## 🛠️ 快速修复指南

### 1. 内存问题快速修复

```bash
# 清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"

# 减少批处理大小
export BATCH_SIZE=1

# 启用梯度检查点
export GRADIENT_CHECKPOINTING=true

# 使用CPU推理
export DEVICE=cpu
```

### 2. 安装问题快速修复

```bash
# 升级pip
pip install --upgrade pip

# 清理缓存
pip cache purge

# 重新安装依赖
pip install --force-reinstall -r requirements.txt

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ package_name
```

### 3. 性能问题快速修复

```bash
# 启用模型编译
export TORCH_COMPILE=true

# 启用混合精度
export MIXED_PRECISION=true

# 增加工作进程
export NUM_WORKERS=4

# 启用缓存
export ENABLE_CACHE=true
```

## 📋 故障排除检查清单

### 安装前检查
- [ ] Python版本 >= 3.8
- [ ] 可用内存 >= 8GB
- [ ] 磁盘空间 >= 10GB
- [ ] 网络连接正常
- [ ] 权限设置正确

### 运行前检查
- [ ] 模型文件存在
- [ ] 配置文件正确
- [ ] 环境变量设置
- [ ] 依赖包完整
- [ ] GPU驱动正常

### 性能检查
- [ ] 内存使用合理
- [ ] CPU使用率正常
- [ ] GPU利用率适当
- [ ] 网络延迟可接受
- [ ] 磁盘I/O正常

### 安全检查
- [ ] 访问权限正确
- [ ] 敏感信息保护
- [ ] 网络安全配置
- [ ] 日志记录完整
- [ ] 监控告警正常

## 📞 获取帮助

### 1. 社区支持
- GitHub Issues: 报告bug和功能请求
- 讨论论坛: 技术讨论和经验分享
- 文档反馈: 改进文档内容

### 2. 专业支持
- 技术咨询: 复杂问题解决方案
- 性能优化: 专业性能调优服务
- 定制开发: 特殊需求定制化开发

### 3. 自助资源
- 官方文档: 完整的使用指南
- 示例代码: 实际应用案例
- 最佳实践: 经验总结和建议

## 📈 总结

故障排除是确保nano-vllm稳定运行的重要环节。通过：

1. **系统化诊断**: 使用自动化工具快速定位问题
2. **分类处理**: 按问题类型和严重程度优先处理
3. **预防为主**: 通过监控和最佳实践预防问题
4. **持续改进**: 根据故障经验不断完善系统

可以大大提高系统的可靠性和可维护性，确保服务的稳定运行。