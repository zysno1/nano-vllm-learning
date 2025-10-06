# ⚡ 性能调优 (Performance Tuning)

## 📖 概述

性能调优是nano-vllm部署和优化的关键环节，涉及计算优化、内存管理、并发控制、系统配置等多个方面。本文档提供全面的性能调优指南，帮助开发者最大化推理性能。

## 🏗️ 性能调优架构

### 1. 性能分析器

```python
import torch
import time
import psutil
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
import matplotlib.pyplot as plt
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    GPU_UTILIZATION = "gpu_utilization"
    CPU_UTILIZATION = "cpu_utilization"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    # 测试参数
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048])
    num_iterations: int = 100
    warmup_iterations: int = 10
    
    # 模型参数
    model_name: str = "test_model"
    dtype: str = "float16"
    device: str = "cuda"
    
    # 优化参数
    enable_flash_attention: bool = True
    enable_kv_cache: bool = True
    enable_tensor_parallel: bool = False
    tensor_parallel_size: int = 1

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.active_timers = {}
        self.monitoring_enabled = True
        
        # 系统监控
        self.system_monitor = SystemMonitor()
        self.gpu_monitor = GPUMonitor() if torch.cuda.is_available() else None
        
        # 性能历史
        self.performance_history = deque(maxlen=1000)
        
        # 分析结果
        self.analysis_results = {}
    
    @contextmanager
    def profile(self, operation_name: str):
        """性能分析上下文管理器"""
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            
            # 记录指标
            latency = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.record_metric(
                MetricType.LATENCY,
                f"{operation_name}_latency",
                latency,
                "seconds"
            )
            
            if memory_delta != 0:
                self.record_metric(
                    MetricType.MEMORY,
                    f"{operation_name}_memory_delta",
                    memory_delta,
                    "bytes"
                )
    
    def record_metric(
        self,
        metric_type: MetricType,
        name: str,
        value: float,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """记录性能指标"""
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.metrics[metric_type].append(metric)
        
        # 添加到历史记录
        self.performance_history.append({
            'type': metric_type.value,
            'name': name,
            'value': value,
            'unit': unit,
            'timestamp': metric.timestamp
        })
    
    def benchmark_inference(self, model, tokenizer, test_prompts: List[str]) -> Dict[str, Any]:
        """基准测试推理性能"""
        
        results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:
                test_key = f"batch_{batch_size}_seq_{seq_length}"
                
                # 准备测试数据
                test_inputs = self._prepare_test_inputs(
                    test_prompts[:batch_size],
                    tokenizer,
                    seq_length
                )
                
                # 预热
                self._warmup_model(model, test_inputs)
                
                # 基准测试
                benchmark_results = self._run_benchmark(
                    model,
                    test_inputs,
                    test_key
                )
                
                results[test_key] = benchmark_results
        
        return results
    
    def _prepare_test_inputs(
        self,
        prompts: List[str],
        tokenizer,
        target_length: int
    ) -> Dict[str, torch.Tensor]:
        """准备测试输入"""
        
        # 扩展prompts到目标长度
        extended_prompts = []
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)
            if len(tokens) < target_length:
                # 重复prompt直到达到目标长度
                repeat_count = (target_length // len(tokens)) + 1
                extended_tokens = (tokens * repeat_count)[:target_length]
            else:
                extended_tokens = tokens[:target_length]
            
            extended_prompts.append(tokenizer.decode(extended_tokens))
        
        # 编码为张量
        encoded = tokenizer(
            extended_prompts,
            padding=True,
            truncation=True,
            max_length=target_length,
            return_tensors="pt"
        )
        
        return {k: v.to(self.config.device) for k, v in encoded.items()}
    
    def _warmup_model(self, model, test_inputs: Dict[str, torch.Tensor]):
        """模型预热"""
        
        logger.info("Warming up model...")
        
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                try:
                    _ = model(**test_inputs)
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def _run_benchmark(
        self,
        model,
        test_inputs: Dict[str, torch.Tensor],
        test_key: str
    ) -> Dict[str, Any]:
        """运行基准测试"""
        
        logger.info(f"Running benchmark: {test_key}")
        
        latencies = []
        throughputs = []
        memory_usages = []
        
        batch_size = test_inputs['input_ids'].shape[0]
        seq_length = test_inputs['input_ids'].shape[1]
        
        for iteration in range(self.config.num_iterations):
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 记录开始状态
            start_time = time.perf_counter()
            start_memory = self._get_gpu_memory_usage()
            
            # 执行推理
            with torch.no_grad():
                try:
                    outputs = model(**test_inputs)
                    
                    # 同步GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # 记录结束状态
                    end_time = time.perf_counter()
                    end_memory = self._get_gpu_memory_usage()
                    
                    # 计算指标
                    latency = end_time - start_time
                    throughput = batch_size / latency  # samples/second
                    memory_usage = end_memory - start_memory
                    
                    latencies.append(latency)
                    throughputs.append(throughput)
                    memory_usages.append(memory_usage)
                    
                except Exception as e:
                    logger.error(f"Benchmark iteration {iteration} failed: {e}")
                    continue
        
        # 计算统计信息
        if latencies:
            results = {
                'batch_size': batch_size,
                'sequence_length': seq_length,
                'iterations': len(latencies),
                'latency': {
                    'mean': np.mean(latencies),
                    'std': np.std(latencies),
                    'min': np.min(latencies),
                    'max': np.max(latencies),
                    'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                },
                'throughput': {
                    'mean': np.mean(throughputs),
                    'std': np.std(throughputs),
                    'min': np.min(throughputs),
                    'max': np.max(throughputs),
                },
                'memory_usage': {
                    'mean': np.mean(memory_usages),
                    'std': np.std(memory_usages),
                    'peak': np.max(memory_usages),
                }
            }
            
            # 记录到性能指标
            self.record_metric(
                MetricType.LATENCY,
                f"{test_key}_latency_mean",
                results['latency']['mean'],
                "seconds"
            )
            
            self.record_metric(
                MetricType.THROUGHPUT,
                f"{test_key}_throughput_mean",
                results['throughput']['mean'],
                "samples/second"
            )
            
            return results
        
        return {}
    
    def _get_memory_usage(self) -> int:
        """获取内存使用量"""
        
        process = psutil.Process()
        return process.memory_info().rss
    
    def _get_gpu_memory_usage(self) -> int:
        """获取GPU内存使用量"""
        
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        return 0
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能数据"""
        
        analysis = {
            'summary': self._generate_summary(),
            'bottlenecks': self._identify_bottlenecks(),
            'recommendations': self._generate_recommendations(),
            'trends': self._analyze_trends(),
        }
        
        self.analysis_results = analysis
        return analysis
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成性能摘要"""
        
        summary = {}
        
        for metric_type, metrics in self.metrics.items():
            if metrics:
                values = [m.value for m in metrics]
                summary[metric_type.value] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
        
        return summary
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        
        bottlenecks = []
        
        # 检查延迟瓶颈
        latency_metrics = self.metrics.get(MetricType.LATENCY, [])
        if latency_metrics:
            high_latency_threshold = np.percentile([m.value for m in latency_metrics], 95)
            high_latency_operations = [
                m for m in latency_metrics
                if m.value > high_latency_threshold
            ]
            
            if high_latency_operations:
                bottlenecks.append({
                    'type': 'high_latency',
                    'description': f'Found {len(high_latency_operations)} operations with high latency',
                    'threshold': high_latency_threshold,
                    'operations': [m.name for m in high_latency_operations[:5]]  # Top 5
                })
        
        # 检查内存瓶颈
        memory_metrics = self.metrics.get(MetricType.MEMORY, [])
        if memory_metrics:
            high_memory_threshold = np.percentile([m.value for m in memory_metrics], 90)
            high_memory_operations = [
                m for m in memory_metrics
                if m.value > high_memory_threshold
            ]
            
            if high_memory_operations:
                bottlenecks.append({
                    'type': 'high_memory',
                    'description': f'Found {len(high_memory_operations)} operations with high memory usage',
                    'threshold': high_memory_threshold,
                    'operations': [m.name for m in high_memory_operations[:5]]
                })
        
        # 检查GPU利用率
        if self.gpu_monitor:
            gpu_utilization = self.gpu_monitor.get_average_utilization()
            if gpu_utilization < 0.7:  # 70%
                bottlenecks.append({
                    'type': 'low_gpu_utilization',
                    'description': f'GPU utilization is low: {gpu_utilization:.2%}',
                    'current_utilization': gpu_utilization,
                    'target_utilization': 0.8
                })
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        
        recommendations = []
        
        # 基于瓶颈生成建议
        bottlenecks = self._identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'high_latency':
                recommendations.append({
                    'category': 'latency_optimization',
                    'priority': 'high',
                    'title': 'Optimize High-Latency Operations',
                    'description': 'Consider using Flash Attention, tensor parallelism, or model quantization',
                    'actions': [
                        'Enable Flash Attention if not already enabled',
                        'Consider tensor parallelism for large models',
                        'Evaluate model quantization (FP16/INT8)',
                        'Optimize batch size and sequence length'
                    ]
                })
            
            elif bottleneck['type'] == 'high_memory':
                recommendations.append({
                    'category': 'memory_optimization',
                    'priority': 'high',
                    'title': 'Reduce Memory Usage',
                    'description': 'Implement memory optimization techniques',
                    'actions': [
                        'Enable gradient checkpointing',
                        'Use KV cache compression',
                        'Implement memory pooling',
                        'Consider CPU offloading for large sequences'
                    ]
                })
            
            elif bottleneck['type'] == 'low_gpu_utilization':
                recommendations.append({
                    'category': 'utilization_optimization',
                    'priority': 'medium',
                    'title': 'Improve GPU Utilization',
                    'description': 'Increase GPU utilization for better performance',
                    'actions': [
                        'Increase batch size if memory allows',
                        'Use dynamic batching',
                        'Optimize kernel fusion',
                        'Consider mixed precision training'
                    ]
                })
        
        # 通用建议
        recommendations.extend([
            {
                'category': 'general_optimization',
                'priority': 'medium',
                'title': 'General Performance Optimizations',
                'description': 'Apply general optimization techniques',
                'actions': [
                    'Profile code to identify hotspots',
                    'Use appropriate data types (FP16 vs FP32)',
                    'Optimize data loading and preprocessing',
                    'Consider model compilation (TorchScript/ONNX)'
                ]
            }
        ])
        
        return recommendations
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        
        trends = {}
        
        # 分析最近的性能历史
        recent_history = list(self.performance_history)[-100:]  # 最近100个记录
        
        if len(recent_history) > 10:
            # 按指标类型分组
            by_type = defaultdict(list)
            for record in recent_history:
                by_type[record['type']].append(record)
            
            for metric_type, records in by_type.items():
                if len(records) > 5:
                    values = [r['value'] for r in records]
                    timestamps = [r['timestamp'] for r in records]
                    
                    # 计算趋势
                    if len(values) > 1:
                        # 简单线性趋势
                        x = np.arange(len(values))
                        slope = np.polyfit(x, values, 1)[0]
                        
                        trends[metric_type] = {
                            'slope': slope,
                            'direction': 'improving' if slope < 0 and metric_type == 'latency' else 
                                        'improving' if slope > 0 and metric_type == 'throughput' else
                                        'degrading' if slope > 0 and metric_type == 'latency' else
                                        'degrading' if slope < 0 and metric_type == 'throughput' else
                                        'stable',
                            'recent_mean': np.mean(values[-5:]),
                            'overall_mean': np.mean(values),
                        }
        
        return trends
    
    def generate_report(self, output_path: str = "performance_report.json"):
        """生成性能报告"""
        
        report = {
            'config': {
                'batch_sizes': self.config.batch_sizes,
                'sequence_lengths': self.config.sequence_lengths,
                'num_iterations': self.config.num_iterations,
                'model_name': self.config.model_name,
                'device': self.config.device,
            },
            'metrics': {
                metric_type.value: [
                    {
                        'name': m.name,
                        'value': m.value,
                        'unit': m.unit,
                        'timestamp': m.timestamp,
                        'metadata': m.metadata
                    }
                    for m in metrics
                ]
                for metric_type, metrics in self.metrics.items()
            },
            'analysis': self.analysis_results,
            'timestamp': time.time(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")
        
        return report
    
    def visualize_performance(self, output_dir: str = "performance_plots"):
        """可视化性能数据"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 延迟分布图
        self._plot_latency_distribution(output_dir)
        
        # 吞吐量对比图
        self._plot_throughput_comparison(output_dir)
        
        # 内存使用趋势图
        self._plot_memory_trends(output_dir)
        
        # 性能热力图
        self._plot_performance_heatmap(output_dir)
    
    def _plot_latency_distribution(self, output_dir: str):
        """绘制延迟分布图"""
        
        latency_metrics = self.metrics.get(MetricType.LATENCY, [])
        if not latency_metrics:
            return
        
        latencies = [m.value * 1000 for m in latency_metrics]  # 转换为毫秒
        
        plt.figure(figsize=(10, 6))
        plt.hist(latencies, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')
        plt.title('Latency Distribution')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        plt.axvline(mean_latency, color='red', linestyle='--', label=f'Mean: {mean_latency:.2f}ms')
        plt.axvline(p95_latency, color='orange', linestyle='--', label=f'P95: {p95_latency:.2f}ms')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, output_dir: str):
        """绘制吞吐量对比图"""
        
        throughput_metrics = self.metrics.get(MetricType.THROUGHPUT, [])
        if not throughput_metrics:
            return
        
        # 按批次大小分组
        by_batch_size = defaultdict(list)
        for metric in throughput_metrics:
            # 从名称中提取批次大小
            if 'batch_' in metric.name:
                batch_size = int(metric.name.split('batch_')[1].split('_')[0])
                by_batch_size[batch_size].append(metric.value)
        
        if by_batch_size:
            batch_sizes = sorted(by_batch_size.keys())
            throughputs = [np.mean(by_batch_size[bs]) for bs in batch_sizes]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(batch_sizes)), throughputs, alpha=0.7)
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/second)')
            plt.title('Throughput vs Batch Size')
            plt.xticks(range(len(batch_sizes)), batch_sizes)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, throughput in enumerate(throughputs):
                plt.text(i, throughput + max(throughputs) * 0.01, f'{throughput:.1f}', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_memory_trends(self, output_dir: str):
        """绘制内存使用趋势图"""
        
        memory_metrics = self.metrics.get(MetricType.MEMORY, [])
        if not memory_metrics:
            return
        
        timestamps = [m.timestamp for m in memory_metrics]
        memory_values = [m.value / (1024**2) for m in memory_metrics]  # 转换为MB
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_values, marker='o', markersize=3, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Over Time')
        plt.grid(True, alpha=0.3)
        
        # 格式化时间轴
        import matplotlib.dates as mdates
        from datetime import datetime
        
        datetime_timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/memory_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmap(self, output_dir: str):
        """绘制性能热力图"""
        
        # 收集批次大小和序列长度的性能数据
        performance_matrix = {}
        
        for metric in self.metrics.get(MetricType.LATENCY, []):
            if 'batch_' in metric.name and 'seq_' in metric.name:
                parts = metric.name.split('_')
                batch_size = int(parts[1])
                seq_length = int(parts[3])
                
                if batch_size not in performance_matrix:
                    performance_matrix[batch_size] = {}
                performance_matrix[batch_size][seq_length] = metric.value * 1000  # 转换为毫秒
        
        if performance_matrix:
            batch_sizes = sorted(performance_matrix.keys())
            seq_lengths = sorted(set().union(*[d.keys() for d in performance_matrix.values()]))
            
            # 创建矩阵
            matrix = np.zeros((len(batch_sizes), len(seq_lengths)))
            for i, bs in enumerate(batch_sizes):
                for j, sl in enumerate(seq_lengths):
                    if sl in performance_matrix[bs]:
                        matrix[i, j] = performance_matrix[bs][sl]
                    else:
                        matrix[i, j] = np.nan
            
            plt.figure(figsize=(12, 8))
            im = plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
            
            plt.xlabel('Sequence Length')
            plt.ylabel('Batch Size')
            plt.title('Latency Heatmap (ms)')
            
            plt.xticks(range(len(seq_lengths)), seq_lengths)
            plt.yticks(range(len(batch_sizes)), batch_sizes)
            
            # 添加颜色条
            cbar = plt.colorbar(im)
            cbar.set_label('Latency (ms)')
            
            # 添加数值标签
            for i in range(len(batch_sizes)):
                for j in range(len(seq_lengths)):
                    if not np.isnan(matrix[i, j]):
                        plt.text(j, i, f'{matrix[i, j]:.1f}', 
                                ha='center', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring_data = defaultdict(list)
        self.monitoring_active = False
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # CPU使用率
                    cpu_percent = self.process.cpu_percent()
                    self.monitoring_data['cpu_percent'].append({
                        'value': cpu_percent,
                        'timestamp': time.time()
                    })
                    
                    # 内存使用
                    memory_info = self.process.memory_info()
                    self.monitoring_data['memory_rss'].append({
                        'value': memory_info.rss,
                        'timestamp': time.time()
                    })
                    
                    # 系统负载
                    load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
                    self.monitoring_data['load_avg'].append({
                        'value': load_avg,
                        'timestamp': time.time()
                    })
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
    
    def get_average_cpu_usage(self) -> float:
        """获取平均CPU使用率"""
        
        cpu_data = self.monitoring_data.get('cpu_percent', [])
        if cpu_data:
            return np.mean([d['value'] for d in cpu_data])
        return 0.0
    
    def get_peak_memory_usage(self) -> int:
        """获取峰值内存使用"""
        
        memory_data = self.monitoring_data.get('memory_rss', [])
        if memory_data:
            return max(d['value'] for d in memory_data)
        return 0

class GPUMonitor:
    """GPU监控器"""
    
    def __init__(self):
        self.monitoring_data = defaultdict(list)
        self.monitoring_active = False
    
    def start_monitoring(self, interval: float = 1.0):
        """开始GPU监控"""
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, GPU monitoring disabled")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # GPU内存使用
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_reserved = torch.cuda.memory_reserved()
                    
                    self.monitoring_data['memory_allocated'].append({
                        'value': memory_allocated,
                        'timestamp': time.time()
                    })
                    
                    self.monitoring_data['memory_reserved'].append({
                        'value': memory_reserved,
                        'timestamp': time.time()
                    })
                    
                    # GPU利用率 (需要nvidia-ml-py)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        self.monitoring_data['gpu_utilization'].append({
                            'value': utilization.gpu,
                            'timestamp': time.time()
                        })
                        
                    except ImportError:
                        pass  # pynvml not available
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"GPU monitoring error: {e}")
                    time.sleep(interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """停止GPU监控"""
        self.monitoring_active = False
    
    def get_average_utilization(self) -> float:
        """获取平均GPU利用率"""
        
        utilization_data = self.monitoring_data.get('gpu_utilization', [])
        if utilization_data:
            return np.mean([d['value'] for d in utilization_data]) / 100.0
        return 0.0
    
    def get_peak_memory_usage(self) -> int:
        """获取峰值GPU内存使用"""
        
        memory_data = self.monitoring_data.get('memory_allocated', [])
        if memory_data:
            return max(d['value'] for d in memory_data)
        return 0
```

### 2. 自动调优器

```python
class AutoTuner:
    """自动调优器"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.tuning_history = []
        self.best_config = None
        self.best_performance = float('inf')
        
        # 调优参数空间
        self.parameter_space = {
            'batch_size': [1, 2, 4, 8, 16, 32],
            'max_sequence_length': [128, 256, 512, 1024, 2048],
            'dtype': ['float32', 'float16', 'bfloat16'],
            'enable_flash_attention': [True, False],
            'enable_kv_cache': [True, False],
            'kv_cache_size': [128, 256, 512, 1024],  # MB
            'tensor_parallel_size': [1, 2, 4, 8],
            'pipeline_parallel_size': [1, 2, 4],
        }
        
        # 调优策略
        self.tuning_strategy = 'bayesian'  # 'grid', 'random', 'bayesian'
        self.max_trials = 50
        self.optimization_target = 'latency'  # 'latency', 'throughput', 'memory'
    
    def auto_tune(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """自动调优"""
        
        logger.info("Starting auto-tuning process...")
        
        constraints = constraints or {}
        
        if self.tuning_strategy == 'grid':
            return self._grid_search_tuning(model, tokenizer, test_prompts, constraints)
        elif self.tuning_strategy == 'random':
            return self._random_search_tuning(model, tokenizer, test_prompts, constraints)
        elif self.tuning_strategy == 'bayesian':
            return self._bayesian_optimization_tuning(model, tokenizer, test_prompts, constraints)
        else:
            raise ValueError(f"Unknown tuning strategy: {self.tuning_strategy}")
    
    def _grid_search_tuning(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """网格搜索调优"""
        
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations(constraints)
        
        logger.info(f"Grid search: testing {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # 应用参数配置
                self._apply_configuration(model, params)
                
                # 运行基准测试
                results = self._run_tuning_benchmark(model, tokenizer, test_prompts, params)
                
                # 评估性能
                performance_score = self._evaluate_performance(results, params)
                
                # 记录结果
                self.tuning_history.append({
                    'params': params,
                    'results': results,
                    'performance_score': performance_score,
                    'timestamp': time.time()
                })
                
                # 更新最佳配置
                if performance_score < self.best_performance:
                    self.best_performance = performance_score
                    self.best_config = params.copy()
                    logger.info(f"New best configuration found: {params}, score: {performance_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to test configuration {params}: {e}")
                continue
        
        return {
            'best_config': self.best_config,
            'best_performance': self.best_performance,
            'tuning_history': self.tuning_history,
            'total_trials': len(self.tuning_history)
        }
    
    def _random_search_tuning(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """随机搜索调优"""
        
        logger.info(f"Random search: testing {self.max_trials} random combinations")
        
        for trial in range(self.max_trials):
            # 随机生成参数组合
            params = self._generate_random_configuration(constraints)
            
            logger.info(f"Trial {trial+1}/{self.max_trials}: {params}")
            
            try:
                # 应用参数配置
                self._apply_configuration(model, params)
                
                # 运行基准测试
                results = self._run_tuning_benchmark(model, tokenizer, test_prompts, params)
                
                # 评估性能
                performance_score = self._evaluate_performance(results, params)
                
                # 记录结果
                self.tuning_history.append({
                    'params': params,
                    'results': results,
                    'performance_score': performance_score,
                    'timestamp': time.time()
                })
                
                # 更新最佳配置
                if performance_score < self.best_performance:
                    self.best_performance = performance_score
                    self.best_config = params.copy()
                    logger.info(f"New best configuration found: {params}, score: {performance_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to test configuration {params}: {e}")
                continue
        
        return {
            'best_config': self.best_config,
            'best_performance': self.best_performance,
            'tuning_history': self.tuning_history,
            'total_trials': len(self.tuning_history)
        }
    
    def _bayesian_optimization_tuning(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """贝叶斯优化调优"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Categorical, Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.warning("scikit-optimize not available, falling back to random search")
            return self._random_search_tuning(model, tokenizer, test_prompts, constraints)
        
        # 定义搜索空间
        dimensions = []
        param_names = []
        
        for param_name, param_values in self.parameter_space.items():
            if param_name in constraints:
                continue  # 跳过约束参数
            
            param_names.append(param_name)
            
            if isinstance(param_values[0], int):
                dimensions.append(Integer(min(param_values), max(param_values), name=param_name))
            elif isinstance(param_values[0], float):
                dimensions.append(Real(min(param_values), max(param_values), name=param_name))
            else:
                dimensions.append(Categorical(param_values, name=param_name))
        
        @use_named_args(dimensions)
        def objective(**params):
            # 添加约束参数
            full_params = {**constraints, **params}
            
            try:
                # 应用参数配置
                self._apply_configuration(model, full_params)
                
                # 运行基准测试
                results = self._run_tuning_benchmark(model, tokenizer, test_prompts, full_params)
                
                # 评估性能
                performance_score = self._evaluate_performance(results, full_params)
                
                # 记录结果
                self.tuning_history.append({
                    'params': full_params,
                    'results': results,
                    'performance_score': performance_score,
                    'timestamp': time.time()
                })
                
                # 更新最佳配置
                if performance_score < self.best_performance:
                    self.best_performance = performance_score
                    self.best_config = full_params.copy()
                
                return performance_score
                
            except Exception as e:
                logger.error(f"Objective function failed for params {params}: {e}")
                return float('inf')
        
        # 运行贝叶斯优化
        logger.info(f"Bayesian optimization: running {self.max_trials} trials")
        
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.max_trials,
            n_initial_points=min(10, self.max_trials // 2),
            random_state=42
        )
        
        return {
            'best_config': self.best_config,
            'best_performance': self.best_performance,
            'tuning_history': self.tuning_history,
            'total_trials': len(self.tuning_history),
            'optimization_result': result
        }
    
    def _generate_parameter_combinations(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成参数组合"""
        
        import itertools
        
        # 过滤约束参数
        available_params = {
            k: v for k, v in self.parameter_space.items()
            if k not in constraints
        }
        
        # 生成所有组合
        param_names = list(available_params.keys())
        param_values = list(available_params.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            params.update(constraints)  # 添加约束参数
            combinations.append(params)
        
        return combinations
    
    def _generate_random_configuration(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """生成随机配置"""
        
        import random
        
        params = constraints.copy()
        
        for param_name, param_values in self.parameter_space.items():
            if param_name not in constraints:
                params[param_name] = random.choice(param_values)
        
        return params
    
    def _apply_configuration(self, model, params: Dict[str, Any]):
        """应用配置参数"""
        
        # 这里需要根据具体模型实现配置应用逻辑
        # 例如：
        # - 设置数据类型
        # - 启用/禁用优化特性
        # - 配置并行参数
        
        if 'dtype' in params:
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            target_dtype = dtype_map.get(params['dtype'], torch.float16)
            
            # 转换模型数据类型
            if hasattr(model, 'to'):
                model.to(dtype=target_dtype)
        
        # 其他配置应用逻辑...
    
    def _run_tuning_benchmark(
        self,
        model,
        tokenizer,
        test_prompts: List[str],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """运行调优基准测试"""
        
        # 使用参数中的批次大小和序列长度
        batch_size = params.get('batch_size', 1)
        max_seq_length = params.get('max_sequence_length', 512)
        
        # 准备测试输入
        test_inputs = self.profiler._prepare_test_inputs(
            test_prompts[:batch_size],
            tokenizer,
            max_seq_length
        )
        
        # 运行基准测试
        results = self.profiler._run_benchmark(
            model,
            test_inputs,
            f"tuning_batch_{batch_size}_seq_{max_seq_length}"
        )
        
        return results
    
    def _evaluate_performance(self, results: Dict[str, Any], params: Dict[str, Any]) -> float:
        """评估性能得分"""
        
        if not results:
            return float('inf')
        
        # 根据优化目标计算得分
        if self.optimization_target == 'latency':
            # 优化延迟（越小越好）
            latency = results.get('latency', {}).get('mean', float('inf'))
            return latency
        
        elif self.optimization_target == 'throughput':
            # 优化吞吐量（越大越好）
            throughput = results.get('throughput', {}).get('mean', 0)
            return -throughput  # 负值，因为优化器最小化目标
        
        elif self.optimization_target == 'memory':
            # 优化内存使用（越小越好）
            memory = results.get('memory_usage', {}).get('mean', float('inf'))
            return memory
        
        else:
            # 综合得分
            latency = results.get('latency', {}).get('mean', float('inf'))
            throughput = results.get('throughput', {}).get('mean', 0)
            memory = results.get('memory_usage', {}).get('mean', float('inf'))
            
            # 归一化并加权
            latency_score = latency / 1.0  # 假设1秒为基准
            throughput_score = 100.0 / max(throughput, 1)  # 假设100 samples/s为基准
            memory_score = memory / (1024**3)  # 假设1GB为基准
            
            return 0.4 * latency_score + 0.4 * throughput_score + 0.2 * memory_score
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """获取调优摘要"""
        
        if not self.tuning_history:
            return {}
        
        # 性能改进
        initial_score = self.tuning_history[0]['performance_score']
        improvement = (initial_score - self.best_performance) / initial_score * 100
        
        # 参数重要性分析
        param_importance = self._analyze_parameter_importance()
        
        return {
            'total_trials': len(self.tuning_history),
            'best_performance': self.best_performance,
            'performance_improvement': f"{improvement:.2f}%",
            'best_config': self.best_config,
            'parameter_importance': param_importance,
            'tuning_strategy': self.tuning_strategy,
        }
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """分析参数重要性"""
        
        if len(self.tuning_history) < 10:
            return {}
        
        # 简单的相关性分析
        param_importance = {}
        
        # 收集所有参数值和性能得分
        all_params = defaultdict(list)
        all_scores = []
        
        for record in self.tuning_history:
            params = record['params']
            score = record['performance_score']
            
            if score != float('inf'):
                all_scores.append(score)
                for param_name, param_value in params.items():
                    all_params[param_name].append(param_value)
        
        # 计算每个参数与性能的相关性
        for param_name, param_values in all_params.items():
            if len(set(param_values)) > 1:  # 参数有变化
                try:
                    # 对于数值参数，计算相关系数
                    if all(isinstance(v, (int, float)) for v in param_values):
                        correlation = np.corrcoef(param_values, all_scores)[0, 1]
                        param_importance[param_name] = abs(correlation)
                    else:
                        # 对于分类参数，计算方差解释比例
                        unique_values = list(set(param_values))
                        if len(unique_values) > 1:
                            group_scores = defaultdict(list)
                            for i, value in enumerate(param_values):
                                group_scores[value].append(all_scores[i])
                            
                            # 计算组间方差与总方差的比例
                            total_var = np.var(all_scores)
                            if total_var > 0:
                                group_means = [np.mean(scores) for scores in group_scores.values()]
                                between_var = np.var(group_means)
                                param_importance[param_name] = between_var / total_var
                            else:
                                param_importance[param_name] = 0.0
                        else:
                            param_importance[param_name] = 0.0
                except:
                    param_importance[param_name] = 0.0
            else:
                param_importance[param_name] = 0.0
        
        return param_importance
```

## 🚀 使用示例

### 1. 基本性能分析

```python
def basic_performance_analysis():
    # 配置基准测试
    config = BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8],
        sequence_lengths=[128, 256, 512, 1024],
        num_iterations=50,
        warmup_iterations=5,
        enable_flash_attention=True,
        enable_kv_cache=True
    )
    
    # 创建性能分析器
    profiler = PerformanceProfiler(config)
    
    # 启动系统监控
    profiler.system_monitor.start_monitoring()
    if profiler.gpu_monitor:
        profiler.gpu_monitor.start_monitoring()
    
    try:
        # 加载模型和分词器（示例）
        # model = load_model("your_model")
        # tokenizer = load_tokenizer("your_tokenizer")
        
        # 准备测试数据
        test_prompts = [
            "What is the capital of France?",
            "Explain the concept of machine learning.",
            "Write a short story about a robot.",
            "How does photosynthesis work?",
        ]
        
        # 运行基准测试
        # benchmark_results = profiler.benchmark_inference(model, tokenizer, test_prompts)
        
        # 分析性能
        analysis = profiler.analyze_performance()
        
        # 生成报告
        report = profiler.generate_report("performance_report.json")
        
        # 可视化结果
        profiler.visualize_performance("performance_plots")
        
        print("Performance Analysis Results:")
        print(f"Summary: {analysis['summary']}")
        print(f"Bottlenecks: {analysis['bottlenecks']}")
        print(f"Recommendations: {analysis['recommendations']}")
        
    finally:
        # 停止监控
        profiler.system_monitor.stop_monitoring()
        if profiler.gpu_monitor:
            profiler.gpu_monitor.stop_monitoring()

if __name__ == "__main__":
    basic_performance_analysis()
```

### 2. 自动调优示例

```python
def auto_tuning_example():
    # 配置基准测试
    config = BenchmarkConfig(
        batch_sizes=[1, 2, 4, 8, 16],
        sequence_lengths=[128, 256, 512, 1024],
        num_iterations=20,
        warmup_iterations=3
    )
    
    # 创建性能分析器和自动调优器
    profiler = PerformanceProfiler(config)
    tuner = AutoTuner(profiler)
    
    # 配置调优参数
    tuner.tuning_strategy = 'bayesian'
    tuner.max_trials = 30
    tuner.optimization_target = 'latency'
    
    try:
        # 加载模型和分词器
        # model = load_model("your_model")
        # tokenizer = load_tokenizer("your_tokenizer")
        
        # 准备测试数据
        test_prompts = [
            "Translate the following text to French: Hello, how are you?",
            "Summarize the main points of quantum computing.",
            "Generate a creative story about time travel.",
        ]
        
        # 设置约束条件
        constraints = {
            'batch_size': 4,  # 固定批次大小
            'tensor_parallel_size': 1,  # 不使用张量并行
        }
        
        # 运行自动调优
        # tuning_results = tuner.auto_tune(model, tokenizer, test_prompts, constraints)
        
        # 获取调优摘要
        summary = tuner.get_tuning_summary()
        
        print("Auto-Tuning Results:")
        print(f"Best Configuration: {tuning_results['best_config']}")
        print(f"Best Performance: {tuning_results['best_performance']:.4f}")
        print(f"Total Trials: {tuning_results['total_trials']}")
        print(f"Performance Improvement: {summary['performance_improvement']}")
        print(f"Parameter Importance: {summary['parameter_importance']}")
        
        # 应用最佳配置
        if tuning_results['best_config']:
            tuner._apply_configuration(model, tuning_results['best_config'])
            print("Best configuration applied to model")
        
    except Exception as e:
        logger.error(f"Auto-tuning failed: {e}")

if __name__ == "__main__":
    auto_tuning_example()
```

### 3. 实时性能监控

```python
def real_time_monitoring():
    # 创建监控器
    system_monitor = SystemMonitor()
    gpu_monitor = GPUMonitor()
    
    # 启动监控
    system_monitor.start_monitoring(interval=0.5)
    gpu_monitor.start_monitoring(interval=0.5)
    
    try:
        # 模拟推理工作负载
        print("Starting inference workload simulation...")
        
        for i in range(100):
            # 模拟推理操作
            if torch.cuda.is_available():
                # 创建一些GPU张量进行计算
                a = torch.randn(1000, 1000, device='cuda')
                b = torch.randn(1000, 1000, device='cuda')
                c = torch.matmul(a, b)
                
                # 模拟内存分配和释放
                temp_tensors = []
                for j in range(10):
                    temp_tensors.append(torch.randn(100, 100, device='cuda'))
                
                del temp_tensors
                torch.cuda.empty_cache()
            
            time.sleep(0.1)
            
            # 每10次迭代打印监控信息
            if i % 10 == 0:
                cpu_usage = system_monitor.get_average_cpu_usage()
                memory_usage = system_monitor.get_peak_memory_usage()
                
                print(f"Iteration {i}: CPU: {cpu_usage:.1f}%, Memory: {memory_usage/(1024**2):.1f}MB")
                
                if gpu_monitor.monitoring_active:
                    gpu_utilization = gpu_monitor.get_average_utilization()
                    gpu_memory = gpu_monitor.get_peak_memory_usage()
                    print(f"  GPU: {gpu_utilization:.1%}, GPU Memory: {gpu_memory/(1024**2):.1f}MB")
    
    finally:
        # 停止监控
        system_monitor.stop_monitoring()
        gpu_monitor.stop_monitoring()
        
        print("\nFinal Statistics:")
        print(f"Average CPU Usage: {system_monitor.get_average_cpu_usage():.1f}%")
        print(f"Peak Memory Usage: {system_monitor.get_peak_memory_usage()/(1024**2):.1f}MB")
        
        if gpu_monitor.monitoring_active:
            print(f"Average GPU Utilization: {gpu_monitor.get_average_utilization():.1%}")
            print(f"Peak GPU Memory: {gpu_monitor.get_peak_memory_usage()/(1024**2):.1f}MB")

if __name__ == "__main__":
    real_time_monitoring()
```

## 🎯 最佳实践

### 1. 性能分析策略
- **全面基准测试**: 测试不同批次大小和序列长度组合
- **多轮测试**: 进行多轮测试以获得稳定结果
- **预热模型**: 在正式测试前进行充分预热
- **监控系统资源**: 同时监控CPU、GPU、内存使用情况

### 2. 调优方法选择
- **网格搜索**: 适用于参数空间较小的情况
- **随机搜索**: 适用于高维参数空间
- **贝叶斯优化**: 适用于昂贵的评估函数
- **多目标优化**: 同时优化延迟、吞吐量和内存使用

### 3. 性能优化技巧
- **批处理优化**: 找到最优的批次大小
- **内存管理**: 合理配置内存池和缓存
- **并行策略**: 根据硬件选择合适的并行方案
- **数据类型**: 使用混合精度训练减少内存占用

### 4. 监控和告警
- **实时监控**: 持续跟踪关键性能指标
- **异常检测**: 及时发现性能异常
- **趋势分析**: 分析性能变化趋势
- **自动告警**: 设置性能阈值告警

## 📈 总结

性能调优是一个持续的过程，需要综合考虑计算效率、内存使用、系统资源等多个方面。通过系统化的性能分析、自动调优和实时监控，可以显著提升nano-vllm的推理性能。

关键要点：
1. **系统化分析**: 建立完整的性能分析框架
2. **自动化调优**: 使用智能算法自动寻找最优配置
3. **实时监控**: 持续跟踪系统性能状态
4. **持续优化**: 根据监控结果持续改进性能

通过合理应用这些性能调优技术，可以在各种硬件环境下获得最佳的推理性能。