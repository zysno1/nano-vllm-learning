"""
NanoVLLM 工具函数模块

提供指标收集、日志工具、健康检查、性能监控等实用功能。
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import GPUtil


@dataclass
class SystemInfo:
    """系统信息数据类"""
    cpu_count: int
    cpu_usage: float
    memory_total: float  # GB
    memory_used: float   # GB
    memory_usage: float  # %
    disk_total: float    # GB
    disk_used: float     # GB
    disk_usage: float    # %
    gpu_count: int
    gpu_info: List[Dict[str, Any]]
    python_version: str
    platform: str
    timestamp: str


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    request_count: int = 0
    total_tokens: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    timestamp: str = ""


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        # 格式化时间
        record.asctime = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        return super().format(record)


class Logger:
    """增强的日志工具类"""
    
    def __init__(self, name: str = "NanoVLLM", level: str = "INFO", 
                 log_file: Optional[str] = None, max_file_size: int = 10):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            level: 日志级别
            log_file: 日志文件路径
            max_file_size: 最大文件大小(MB)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_file_size * 1024 * 1024,
                backupCount=5
            )
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(self._format_message(message, kwargs))
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(self._format_message(message, kwargs))
    
    def exception(self, message: str, **kwargs):
        """异常日志（包含堆栈跟踪）"""
        self.logger.exception(self._format_message(message, kwargs))
    
    def _format_message(self, message: str, kwargs: Dict) -> str:
        """格式化日志消息"""
        if kwargs:
            extra_info = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {extra_info}"
        return message


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化指标收集器
        
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.metrics_history: List[Dict] = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # 计数器
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        
        # 最近请求记录
        self.recent_requests: List[Dict] = []
    
    def record_request(self, latency: float, tokens: int, success: bool = True, 
                      request_id: Optional[str] = None, **kwargs):
        """
        记录请求指标
        
        Args:
            latency: 请求延迟(秒)
            tokens: 生成的token数量
            success: 是否成功
            request_id: 请求ID
            **kwargs: 其他指标
        """
        with self.lock:
            timestamp = time.time()
            
            # 更新计数器
            self.request_count += 1
            if success:
                self.total_tokens += tokens
                self.total_latency += latency
            else:
                self.error_count += 1
            
            # 记录详细信息
            record = {
                'timestamp': timestamp,
                'request_id': request_id,
                'latency': latency,
                'tokens': tokens,
                'success': success,
                'tokens_per_second': tokens / latency if latency > 0 else 0,
                **kwargs
            }
            
            self.recent_requests.append(record)
            
            # 保持窗口大小
            if len(self.recent_requests) > self.window_size:
                self.recent_requests.pop(0)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # 计算平均值
            avg_latency = self.total_latency / max(self.request_count - self.error_count, 1)
            tokens_per_second = self.total_tokens / max(self.total_latency, 0.001)
            requests_per_second = self.request_count / max(uptime, 0.001)
            error_rate = self.error_count / max(self.request_count, 1)
            
            # 获取系统资源使用情况
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = self._get_gpu_usage()
            
            return PerformanceMetrics(
                request_count=self.request_count,
                total_tokens=self.total_tokens,
                total_latency=self.total_latency,
                avg_latency=avg_latency,
                tokens_per_second=tokens_per_second,
                requests_per_second=requests_per_second,
                error_count=self.error_count,
                error_rate=error_rate,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                timestamp=datetime.fromtimestamp(current_time).isoformat()
            )
    
    def get_recent_metrics(self, seconds: int = 60) -> Dict[str, Any]:
        """获取最近N秒的指标"""
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - seconds
            
            recent_records = [
                r for r in self.recent_requests 
                if r['timestamp'] >= cutoff_time
            ]
            
            if not recent_records:
                return {
                    'period_seconds': seconds,
                    'request_count': 0,
                    'avg_latency': 0,
                    'tokens_per_second': 0,
                    'error_rate': 0
                }
            
            total_requests = len(recent_records)
            successful_requests = [r for r in recent_records if r['success']]
            
            avg_latency = sum(r['latency'] for r in successful_requests) / max(len(successful_requests), 1)
            total_tokens = sum(r['tokens'] for r in successful_requests)
            total_time = sum(r['latency'] for r in successful_requests)
            tokens_per_second = total_tokens / max(total_time, 0.001)
            error_rate = (total_requests - len(successful_requests)) / total_requests
            
            return {
                'period_seconds': seconds,
                'request_count': total_requests,
                'successful_requests': len(successful_requests),
                'avg_latency': avg_latency,
                'tokens_per_second': tokens_per_second,
                'error_rate': error_rate,
                'total_tokens': total_tokens
            }
    
    def _get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return sum(gpu.load for gpu in gpus) / len(gpus) * 100
        except:
            pass
        return 0.0
    
    def reset(self):
        """重置指标"""
        with self.lock:
            self.request_count = 0
            self.error_count = 0
            self.total_tokens = 0
            self.total_latency = 0.0
            self.recent_requests.clear()
            self.start_time = time.time()


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        初始化健康检查器
        
        Args:
            logger: 日志器实例
        """
        self.logger = logger or Logger("HealthChecker")
        self.checks: Dict[str, Callable] = {}
        self.thresholds = {
            'memory_usage': 90.0,  # %
            'disk_usage': 90.0,    # %
            'gpu_memory': 95.0,    # %
            'error_rate': 10.0,    # %
            'avg_latency': 10.0,   # seconds
        }
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """
        注册健康检查函数
        
        Args:
            name: 检查名称
            check_func: 检查函数，返回True表示健康
        """
        self.checks[name] = check_func
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        health_status = {
            'healthy': True,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 系统资源检查
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 内存检查
            memory_healthy = memory.percent < self.thresholds['memory_usage']
            health_status['checks']['memory'] = {
                'healthy': memory_healthy,
                'usage_percent': memory.percent,
                'threshold': self.thresholds['memory_usage']
            }
            if not memory_healthy:
                health_status['warnings'].append(f"内存使用率过高: {memory.percent:.1f}%")
            
            # 磁盘检查
            disk_healthy = disk.percent < self.thresholds['disk_usage']
            health_status['checks']['disk'] = {
                'healthy': disk_healthy,
                'usage_percent': disk.percent,
                'threshold': self.thresholds['disk_usage']
            }
            if not disk_healthy:
                health_status['warnings'].append(f"磁盘使用率过高: {disk.percent:.1f}%")
            
            # GPU检查
            gpu_healthy = True
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        gpu_memory_usage = gpu.memoryUtil * 100
                        gpu_check_healthy = gpu_memory_usage < self.thresholds['gpu_memory']
                        health_status['checks'][f'gpu_{i}'] = {
                            'healthy': gpu_check_healthy,
                            'memory_usage_percent': gpu_memory_usage,
                            'load_percent': gpu.load * 100,
                            'temperature': gpu.temperature
                        }
                        if not gpu_check_healthy:
                            health_status['warnings'].append(
                                f"GPU {i} 内存使用率过高: {gpu_memory_usage:.1f}%"
                            )
                            gpu_healthy = False
            except Exception as e:
                health_status['checks']['gpu'] = {
                    'healthy': False,
                    'error': str(e)
                }
                gpu_healthy = False
            
            # 自定义检查
            for name, check_func in self.checks.items():
                try:
                    result = check_func()
                    health_status['checks'][name] = {'healthy': result}
                    if not result:
                        health_status['warnings'].append(f"自定义检查失败: {name}")
                except Exception as e:
                    health_status['checks'][name] = {
                        'healthy': False,
                        'error': str(e)
                    }
                    health_status['errors'].append(f"检查 {name} 时出错: {str(e)}")
            
            # 总体健康状态
            health_status['healthy'] = (
                memory_healthy and 
                disk_healthy and 
                gpu_healthy and 
                all(check.get('healthy', False) for check in health_status['checks'].values())
            )
            
        except Exception as e:
            health_status['healthy'] = False
            health_status['errors'].append(f"健康检查时出错: {str(e)}")
            self.logger.exception("健康检查失败", error=str(e))
        
        return health_status
    
    def check_model_health(self, model_instance: Any) -> Dict[str, Any]:
        """检查模型健康状态"""
        model_health = {
            'healthy': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        try:
            # 检查模型是否已加载
            if hasattr(model_instance, 'model') and model_instance.model is not None:
                model_health['checks']['model_loaded'] = {'healthy': True}
            else:
                model_health['checks']['model_loaded'] = {'healthy': False}
                model_health['errors'].append("模型未加载")
                model_health['healthy'] = False
            
            # 检查tokenizer
            if hasattr(model_instance, 'tokenizer') and model_instance.tokenizer is not None:
                model_health['checks']['tokenizer_loaded'] = {'healthy': True}
            else:
                model_health['checks']['tokenizer_loaded'] = {'healthy': False}
                model_health['errors'].append("Tokenizer未加载")
                model_health['healthy'] = False
            
            # 检查设备状态
            if hasattr(model_instance, 'device'):
                device_str = str(model_instance.device)
                model_health['checks']['device'] = {
                    'healthy': True,
                    'device': device_str
                }
                
                # 如果是GPU设备，检查GPU状态
                if 'cuda' in device_str:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_id = int(device_str.split(':')[-1]) if ':' in device_str else 0
                            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
                            
                            model_health['checks']['gpu_memory'] = {
                                'healthy': True,
                                'allocated_gb': memory_allocated,
                                'reserved_gb': memory_reserved
                            }
                        else:
                            model_health['checks']['gpu_memory'] = {'healthy': False}
                            model_health['warnings'].append("CUDA不可用")
                    except Exception as e:
                        model_health['checks']['gpu_memory'] = {
                            'healthy': False,
                            'error': str(e)
                        }
            
        except Exception as e:
            model_health['healthy'] = False
            model_health['errors'].append(f"模型健康检查时出错: {str(e)}")
            self.logger.exception("模型健康检查失败", error=str(e))
        
        return model_health


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, logger: Optional[Logger] = None, 
                 metrics_collector: Optional[MetricsCollector] = None):
        """
        初始化系统监控器
        
        Args:
            logger: 日志器实例
            metrics_collector: 指标收集器实例
        """
        self.logger = logger or Logger("SystemMonitor")
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 30  # 秒
    
    def get_system_info(self) -> SystemInfo:
        """获取系统信息"""
        try:
            # CPU信息
            cpu_count = psutil.cpu_count()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024**3)  # GB
            memory_used = memory.used / (1024**3)    # GB
            memory_usage = memory.percent
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            disk_total = disk.total / (1024**3)  # GB
            disk_used = disk.used / (1024**3)    # GB
            disk_usage = disk.percent
            
            # GPU信息
            gpu_count = 0
            gpu_info = []
            try:
                gpus = GPUtil.getGPUs()
                gpu_count = len(gpus)
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_usage': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
            except:
                pass
            
            # 系统信息
            python_version = sys.version
            platform = sys.platform
            
            return SystemInfo(
                cpu_count=cpu_count,
                cpu_usage=cpu_usage,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_usage=memory_usage,
                disk_total=disk_total,
                disk_used=disk_used,
                disk_usage=disk_usage,
                gpu_count=gpu_count,
                gpu_info=gpu_info,
                python_version=python_version,
                platform=platform,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.exception("获取系统信息失败", error=str(e))
            raise
    
    def start_monitoring(self, interval: int = 30):
        """
        启动系统监控
        
        Args:
            interval: 监控间隔(秒)
        """
        if self.monitoring:
            self.logger.warning("系统监控已在运行")
            return
        
        self.monitor_interval = interval
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("系统监控已启动", interval=interval)
    
    def stop_monitoring(self):
        """停止系统监控"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统信息
                system_info = self.get_system_info()
                
                # 记录关键指标
                self.logger.debug(
                    "系统监控",
                    cpu_usage=f"{system_info.cpu_usage:.1f}%",
                    memory_usage=f"{system_info.memory_usage:.1f}%",
                    disk_usage=f"{system_info.disk_usage:.1f}%",
                    gpu_count=system_info.gpu_count
                )
                
                # 检查警告阈值
                if system_info.memory_usage > 90:
                    self.logger.warning("内存使用率过高", usage=f"{system_info.memory_usage:.1f}%")
                
                if system_info.disk_usage > 90:
                    self.logger.warning("磁盘使用率过高", usage=f"{system_info.disk_usage:.1f}%")
                
                for gpu in system_info.gpu_info:
                    if gpu['memory_usage'] > 95:
                        self.logger.warning(
                            "GPU内存使用率过高",
                            gpu_id=gpu['id'],
                            usage=f"{gpu['memory_usage']:.1f}%"
                        )
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.exception("监控循环出错", error=str(e))
                time.sleep(5)  # 出错时短暂等待


def create_logger(name: str = "NanoVLLM", level: str = "INFO", 
                 log_dir: Optional[str] = None) -> Logger:
    """
    创建日志器的便捷函数
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_dir: 日志目录
    
    Returns:
        Logger实例
    """
    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = log_dir_path / f"{name.lower()}.log"
    
    return Logger(name, level, str(log_file) if log_file else None)


def format_bytes(bytes_value: Union[int, float]) -> str:
    """格式化字节数为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """格式化持续时间为人类可读格式"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    return numerator / denominator if denominator != 0 else default


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().isoformat()


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.json':
            return json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


def save_config_to_file(config: Dict[str, Any], config_path: str):
    """保存配置到文件"""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2, ensure_ascii=False)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            import yaml
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")


# 示例用法
if __name__ == "__main__":
    # 创建日志器
    logger = create_logger("TestLogger", "DEBUG")
    
    # 创建指标收集器
    metrics = MetricsCollector()
    
    # 创建健康检查器
    health_checker = HealthChecker(logger)
    
    # 创建系统监控器
    monitor = SystemMonitor(logger, metrics)
    
    # 演示功能
    logger.info("开始演示工具函数")
    
    # 记录一些测试指标
    metrics.record_request(0.5, 100, True, "test_001")
    metrics.record_request(0.8, 150, True, "test_002")
    metrics.record_request(1.2, 80, False, "test_003")
    
    # 获取当前指标
    current_metrics = metrics.get_current_metrics()
    logger.info("当前性能指标", **asdict(current_metrics))
    
    # 获取系统信息
    system_info = monitor.get_system_info()
    logger.info("系统信息", **asdict(system_info))
    
    # 健康检查
    health_status = health_checker.check_system_health()
    logger.info("系统健康状态", healthy=health_status['healthy'])
    
    logger.info("工具函数演示完成")