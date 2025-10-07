"""
分布式推理基础概念演示

本模块演示分布式推理的核心概念，包括：
1. 推理服务架构设计
2. 请求路由机制
3. 性能监控基础
4. 负载均衡策略
"""

import asyncio
import time
import random
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import uuid
from collections import defaultdict, deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestType(Enum):
    """请求类型枚举"""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"

class NodeStatus(Enum):
    """节点状态枚举"""
    HEALTHY = "healthy"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"

@dataclass
class InferenceRequest:
    """推理请求数据类"""
    request_id: str
    request_type: RequestType
    input_text: str
    max_tokens: int = 100
    temperature: float = 0.7
    timestamp: float = 0.0
    priority: int = 1  # 1-5, 5为最高优先级
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class InferenceResponse:
    """推理响应数据类"""
    request_id: str
    response_text: str
    processing_time: float
    node_id: str
    success: bool = True
    error_message: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class NodeMetrics:
    """节点性能指标"""
    node_id: str
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    queue_length: int
    active_requests: int
    total_requests: int
    avg_response_time: float
    error_rate: float
    status: NodeStatus
    last_update: float = 0.0
    
    def __post_init__(self):
        if self.last_update == 0.0:
            self.last_update = time.time()

@dataclass
class PerformanceStats:
    """性能统计数据"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

class InferenceNode:
    """推理节点模拟器"""
    
    def __init__(self, node_id: str, capacity: int = 10, processing_time_range: Tuple[float, float] = (0.1, 2.0)):
        self.node_id = node_id
        self.capacity = capacity
        self.processing_time_range = processing_time_range
        self.request_queue = Queue(maxsize=capacity * 2)
        self.active_requests = 0
        self.total_requests = 0
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.is_running = False
        self.worker_thread = None
        
        # 模拟资源使用
        self.base_cpu_usage = random.uniform(0.1, 0.3)
        self.base_memory_usage = random.uniform(0.2, 0.4)
        self.base_gpu_usage = random.uniform(0.1, 0.5)
    
    def start(self):
        """启动节点"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_requests, daemon=True)
        self.worker_thread.start()
        logger.info(f"节点 {self.node_id} 已启动")
    
    def stop(self):
        """停止节点"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        logger.info(f"节点 {self.node_id} 已停止")
    
    def submit_request(self, request: InferenceRequest) -> bool:
        """提交请求到节点"""
        try:
            self.request_queue.put_nowait(request)
            return True
        except:
            return False
    
    def _process_requests(self):
        """处理请求的工作线程"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                self._process_single_request(request)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"节点 {self.node_id} 处理请求时出错: {e}")
    
    def _process_single_request(self, request: InferenceRequest):
        """处理单个请求"""
        start_time = time.time()
        self.active_requests += 1
        self.total_requests += 1
        
        try:
            # 模拟推理处理时间
            processing_time = random.uniform(*self.processing_time_range)
            
            # 根据请求类型调整处理时间
            if request.request_type == RequestType.TEXT_GENERATION:
                processing_time *= (request.max_tokens / 100)
            elif request.request_type == RequestType.EMBEDDING:
                processing_time *= 0.5
            
            time.sleep(processing_time)
            
            # 模拟偶发错误
            if random.random() < 0.02:  # 2% 错误率
                raise Exception("模拟推理错误")
            
            # 生成响应
            response_text = f"Generated response for request {request.request_id}"
            actual_time = time.time() - start_time
            
            response = InferenceResponse(
                request_id=request.request_id,
                response_text=response_text,
                processing_time=actual_time,
                node_id=self.node_id,
                success=True
            )
            
            self.response_times.append(actual_time)
            
        except Exception as e:
            self.error_count += 1
            actual_time = time.time() - start_time
            
            response = InferenceResponse(
                request_id=request.request_id,
                response_text="",
                processing_time=actual_time,
                node_id=self.node_id,
                success=False,
                error_message=str(e)
            )
        
        finally:
            self.active_requests -= 1
    
    def get_metrics(self) -> NodeMetrics:
        """获取节点性能指标"""
        queue_length = self.request_queue.qsize()
        
        # 计算动态资源使用率
        load_factor = min(1.0, (self.active_requests + queue_length) / self.capacity)
        cpu_usage = min(0.95, self.base_cpu_usage + load_factor * 0.6)
        memory_usage = min(0.95, self.base_memory_usage + load_factor * 0.4)
        gpu_usage = min(0.95, self.base_gpu_usage + load_factor * 0.4)
        
        # 计算平均响应时间
        avg_response_time = np.mean(self.response_times) if self.response_times else 0.0
        
        # 计算错误率
        error_rate = self.error_count / max(1, self.total_requests)
        
        # 确定节点状态
        if not self.is_running:
            status = NodeStatus.OFFLINE
        elif cpu_usage > 0.9 or memory_usage > 0.9 or queue_length > self.capacity:
            status = NodeStatus.OVERLOADED
        elif cpu_usage > 0.7 or memory_usage > 0.7 or queue_length > self.capacity * 0.7:
            status = NodeStatus.BUSY
        else:
            status = NodeStatus.HEALTHY
        
        return NodeMetrics(
            node_id=self.node_id,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            queue_length=queue_length,
            active_requests=self.active_requests,
            total_requests=self.total_requests,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            status=status
        )

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.nodes: List[InferenceNode] = []
        self.current_index = 0
        self.node_weights = {}
        self.request_history = deque(maxlen=1000)
    
    def add_node(self, node: InferenceNode):
        """添加节点"""
        self.nodes.append(node)
        self.node_weights[node.node_id] = 1.0
        logger.info(f"添加节点: {node.node_id}")
    
    def remove_node(self, node_id: str):
        """移除节点"""
        self.nodes = [node for node in self.nodes if node.node_id != node_id]
        if node_id in self.node_weights:
            del self.node_weights[node_id]
        logger.info(f"移除节点: {node_id}")
    
    def select_node(self, request: InferenceRequest) -> Optional[InferenceNode]:
        """选择最优节点"""
        available_nodes = [node for node in self.nodes if node.is_running]
        
        if not available_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.strategy == "least_connections":
            return self._least_connections_selection(available_nodes)
        elif self.strategy == "weighted_round_robin":
            return self._weighted_round_robin_selection(available_nodes)
        elif self.strategy == "response_time":
            return self._response_time_selection(available_nodes)
        else:
            return available_nodes[0]
    
    def _round_robin_selection(self, nodes: List[InferenceNode]) -> InferenceNode:
        """轮询选择"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node
    
    def _least_connections_selection(self, nodes: List[InferenceNode]) -> InferenceNode:
        """最少连接数选择"""
        return min(nodes, key=lambda n: n.active_requests + n.request_queue.qsize())
    
    def _weighted_round_robin_selection(self, nodes: List[InferenceNode]) -> InferenceNode:
        """加权轮询选择"""
        # 根据节点性能动态调整权重
        for node in nodes:
            metrics = node.get_metrics()
            # 权重与负载成反比
            load = (metrics.cpu_usage + metrics.memory_usage + metrics.gpu_usage) / 3
            self.node_weights[node.node_id] = max(0.1, 1.0 - load)
        
        # 加权选择
        weights = [self.node_weights.get(node.node_id, 1.0) for node in nodes]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return nodes[0]
        
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return nodes[i]
        
        return nodes[-1]
    
    def _response_time_selection(self, nodes: List[InferenceNode]) -> InferenceNode:
        """响应时间优化选择"""
        return min(nodes, key=lambda n: n.get_metrics().avg_response_time)

class DistributedInferenceSystem:
    """分布式推理系统"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.performance_stats = PerformanceStats()
        self.is_running = False
        self.dispatcher_thread = None
        self.monitor_thread = None
        self.request_latencies = []
    
    def start(self):
        """启动系统"""
        self.is_running = True
        self.performance_stats.start_time = time.time()
        
        # 启动所有节点
        for node in self.load_balancer.nodes:
            node.start()
        
        # 启动请求分发器
        self.dispatcher_thread = threading.Thread(target=self._dispatch_requests, daemon=True)
        self.dispatcher_thread.start()
        
        # 启动监控器
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
        
        logger.info("分布式推理系统已启动")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        self.performance_stats.end_time = time.time()
        
        # 停止所有节点
        for node in self.load_balancer.nodes:
            node.stop()
        
        # 等待线程结束
        if self.dispatcher_thread:
            self.dispatcher_thread.join(timeout=1.0)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("分布式推理系统已停止")
    
    def submit_request(self, request: InferenceRequest) -> bool:
        """提交推理请求"""
        try:
            self.request_queue.put_nowait(request)
            return True
        except:
            return False
    
    def _dispatch_requests(self):
        """请求分发器"""
        while self.is_running:
            try:
                request = self.request_queue.get(timeout=0.1)
                self._process_request(request)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"分发请求时出错: {e}")
    
    def _process_request(self, request: InferenceRequest):
        """处理单个请求"""
        start_time = time.time()
        
        # 选择节点
        node = self.load_balancer.select_node(request)
        
        if node is None:
            logger.warning(f"没有可用节点处理请求 {request.request_id}")
            self.performance_stats.failed_requests += 1
            return
        
        # 提交到节点
        success = node.submit_request(request)
        
        if success:
            self.performance_stats.total_requests += 1
            # 记录延迟（这里简化处理，实际应该等待响应）
            latency = time.time() - start_time
            self.request_latencies.append(latency)
        else:
            logger.warning(f"节点 {node.node_id} 队列已满，请求 {request.request_id} 被拒绝")
            self.performance_stats.failed_requests += 1
    
    def _monitor_system(self):
        """系统监控器"""
        while self.is_running:
            try:
                # 收集所有节点指标
                all_metrics = []
                for node in self.load_balancer.nodes:
                    metrics = node.get_metrics()
                    all_metrics.append(metrics)
                
                # 更新系统统计
                self._update_performance_stats(all_metrics)
                
                time.sleep(1.0)  # 每秒监控一次
                
            except Exception as e:
                logger.error(f"监控系统时出错: {e}")
    
    def _update_performance_stats(self, node_metrics: List[NodeMetrics]):
        """更新性能统计"""
        if not self.request_latencies:
            return
        
        # 计算延迟统计
        latencies = np.array(self.request_latencies)
        self.performance_stats.avg_latency = np.mean(latencies)
        self.performance_stats.p95_latency = np.percentile(latencies, 95)
        self.performance_stats.p99_latency = np.percentile(latencies, 99)
        
        # 计算吞吐量
        elapsed_time = time.time() - self.performance_stats.start_time
        if elapsed_time > 0:
            self.performance_stats.throughput = self.performance_stats.total_requests / elapsed_time
        
        # 计算错误率
        total_requests = self.performance_stats.total_requests + self.performance_stats.failed_requests
        if total_requests > 0:
            self.performance_stats.error_rate = self.performance_stats.failed_requests / total_requests
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        node_metrics = []
        for node in self.load_balancer.nodes:
            metrics = node.get_metrics()
            node_metrics.append(asdict(metrics))
        
        return {
            "performance_stats": asdict(self.performance_stats),
            "node_metrics": node_metrics,
            "system_health": self._calculate_system_health(node_metrics)
        }
    
    def _calculate_system_health(self, node_metrics: List[Dict]) -> str:
        """计算系统健康状态"""
        if not node_metrics:
            return "CRITICAL"
        
        healthy_nodes = sum(1 for m in node_metrics if m["status"] == "healthy")
        total_nodes = len(node_metrics)
        
        health_ratio = healthy_nodes / total_nodes
        
        if health_ratio >= 0.8:
            return "HEALTHY"
        elif health_ratio >= 0.5:
            return "WARNING"
        else:
            return "CRITICAL"

class RequestGenerator:
    """请求生成器"""
    
    def __init__(self):
        self.request_templates = {
            RequestType.TEXT_GENERATION: [
                "请生成一个关于人工智能的故事",
                "写一首关于春天的诗",
                "解释什么是机器学习",
                "描述一下未来的城市"
            ],
            RequestType.EMBEDDING: [
                "计算这段文本的向量表示",
                "生成句子嵌入",
                "文本相似度计算"
            ],
            RequestType.CLASSIFICATION: [
                "这是正面还是负面评价？",
                "分类这篇文章的主题",
                "判断情感倾向"
            ],
            RequestType.TRANSLATION: [
                "将这段中文翻译成英文",
                "英译中翻译",
                "多语言翻译"
            ]
        }
    
    def generate_request(self, request_type: Optional[RequestType] = None) -> InferenceRequest:
        """生成随机请求"""
        if request_type is None:
            request_type = random.choice(list(RequestType))
        
        templates = self.request_templates[request_type]
        input_text = random.choice(templates)
        
        return InferenceRequest(
            request_id=str(uuid.uuid4()),
            request_type=request_type,
            input_text=input_text,
            max_tokens=random.randint(50, 200),
            temperature=random.uniform(0.1, 1.0),
            priority=random.randint(1, 5)
        )
    
    def generate_batch_requests(self, count: int, request_type: Optional[RequestType] = None) -> List[InferenceRequest]:
        """生成批量请求"""
        return [self.generate_request(request_type) for _ in range(count)]

class DistributedInferenceBenchmark:
    """分布式推理基准测试"""
    
    def __init__(self):
        self.results = {}
        self.request_generator = RequestGenerator()
    
    def run_load_balancing_comparison(self, num_requests: int = 1000, num_nodes: int = 4):
        """运行负载均衡策略对比"""
        strategies = ["round_robin", "least_connections", "weighted_round_robin", "response_time"]
        
        print("🔄 运行负载均衡策略对比...")
        
        for strategy in strategies:
            print(f"\n测试策略: {strategy}")
            
            # 创建负载均衡器和节点
            load_balancer = LoadBalancer(strategy=strategy)
            
            # 添加不同性能的节点
            for i in range(num_nodes):
                # 模拟不同性能的节点
                if i == 0:  # 高性能节点
                    processing_time = (0.05, 0.2)
                    capacity = 20
                elif i == 1:  # 中等性能节点
                    processing_time = (0.1, 0.5)
                    capacity = 15
                else:  # 普通性能节点
                    processing_time = (0.2, 1.0)
                    capacity = 10
                
                node = InferenceNode(
                    node_id=f"node_{i}_{strategy}",
                    capacity=capacity,
                    processing_time_range=processing_time
                )
                load_balancer.add_node(node)
            
            # 创建分布式系统
            system = DistributedInferenceSystem(load_balancer)
            system.start()
            
            # 生成并提交请求
            requests = self.request_generator.generate_batch_requests(num_requests)
            
            start_time = time.time()
            for request in requests:
                system.submit_request(request)
            
            # 等待处理完成
            time.sleep(5.0)
            
            # 收集结果
            status = system.get_system_status()
            processing_time = time.time() - start_time
            
            self.results[strategy] = {
                "strategy": strategy,
                "processing_time": processing_time,
                "performance_stats": status["performance_stats"],
                "node_metrics": status["node_metrics"],
                "system_health": status["system_health"]
            }
            
            system.stop()
            
            print(f"  处理时间: {processing_time:.2f}s")
            print(f"  吞吐量: {status['performance_stats']['throughput']:.2f} req/s")
            print(f"  平均延迟: {status['performance_stats']['avg_latency']*1000:.2f}ms")
            print(f"  错误率: {status['performance_stats']['error_rate']*100:.2f}%")
            print(f"  系统健康: {status['system_health']}")
    
    def run_scalability_test(self, max_nodes: int = 8, requests_per_node: int = 200):
        """运行可扩展性测试"""
        print("\n📈 运行可扩展性测试...")
        
        scalability_results = {}
        
        for num_nodes in range(1, max_nodes + 1):
            print(f"\n测试节点数: {num_nodes}")
            
            # 创建负载均衡器
            load_balancer = LoadBalancer(strategy="least_connections")
            
            # 添加节点
            for i in range(num_nodes):
                node = InferenceNode(
                    node_id=f"scale_node_{i}",
                    capacity=15,
                    processing_time_range=(0.1, 0.5)
                )
                load_balancer.add_node(node)
            
            # 创建系统
            system = DistributedInferenceSystem(load_balancer)
            system.start()
            
            # 生成请求
            num_requests = num_nodes * requests_per_node
            requests = self.request_generator.generate_batch_requests(num_requests)
            
            start_time = time.time()
            for request in requests:
                system.submit_request(request)
            
            # 等待处理
            time.sleep(3.0)
            
            # 收集结果
            status = system.get_system_status()
            processing_time = time.time() - start_time
            
            scalability_results[num_nodes] = {
                "num_nodes": num_nodes,
                "num_requests": num_requests,
                "processing_time": processing_time,
                "throughput": status["performance_stats"]["throughput"],
                "avg_latency": status["performance_stats"]["avg_latency"],
                "system_health": status["system_health"]
            }
            
            system.stop()
            
            print(f"  请求数: {num_requests}")
            print(f"  吞吐量: {status['performance_stats']['throughput']:.2f} req/s")
            print(f"  平均延迟: {status['performance_stats']['avg_latency']*1000:.2f}ms")
        
        self.results["scalability"] = scalability_results
    
    def print_results(self):
        """打印测试结果"""
        print("\n" + "="*80)
        print("📊 分布式推理基准测试结果")
        print("="*80)
        
        # 负载均衡策略对比
        if any(key in self.results for key in ["round_robin", "least_connections", "weighted_round_robin", "response_time"]):
            print("\n🔄 负载均衡策略对比:")
            print("-" * 60)
            print(f"{'策略':<20} {'吞吐量(req/s)':<15} {'延迟(ms)':<12} {'错误率(%)':<10} {'健康状态':<10}")
            print("-" * 60)
            
            for strategy in ["round_robin", "least_connections", "weighted_round_robin", "response_time"]:
                if strategy in self.results:
                    result = self.results[strategy]
                    stats = result["performance_stats"]
                    print(f"{strategy:<20} {stats['throughput']:<15.2f} {stats['avg_latency']*1000:<12.2f} "
                          f"{stats['error_rate']*100:<10.2f} {result['system_health']:<10}")
        
        # 可扩展性测试结果
        if "scalability" in self.results:
            print("\n📈 可扩展性测试结果:")
            print("-" * 60)
            print(f"{'节点数':<8} {'请求数':<10} {'吞吐量(req/s)':<15} {'延迟(ms)':<12} {'健康状态':<10}")
            print("-" * 60)
            
            for num_nodes, result in self.results["scalability"].items():
                print(f"{num_nodes:<8} {result['num_requests']:<10} {result['throughput']:<15.2f} "
                      f"{result['avg_latency']*1000:<12.2f} {result['system_health']:<10}")
    
    def visualize_results(self):
        """可视化测试结果"""
        plt.style.use('seaborn-v0_8')
        
        # 负载均衡策略对比图
        strategies = ["round_robin", "least_connections", "weighted_round_robin", "response_time"]
        strategy_results = {k: v for k, v in self.results.items() if k in strategies}
        
        if strategy_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 吞吐量对比
            throughputs = [result["performance_stats"]["throughput"] for result in strategy_results.values()]
            strategy_names = [s.replace("_", " ").title() for s in strategy_results.keys()]
            
            bars1 = ax1.bar(strategy_names, throughputs, color='skyblue', alpha=0.8)
            ax1.set_title('负载均衡策略 - 吞吐量对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('吞吐量 (requests/second)')
            ax1.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars1, throughputs):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # 延迟对比
            latencies = [result["performance_stats"]["avg_latency"] * 1000 for result in strategy_results.values()]
            
            bars2 = ax2.bar(strategy_names, latencies, color='lightcoral', alpha=0.8)
            ax2.set_title('负载均衡策略 - 平均延迟对比', fontsize=14, fontweight='bold')
            ax2.set_ylabel('平均延迟 (milliseconds)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars2, latencies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # 错误率对比
            error_rates = [result["performance_stats"]["error_rate"] * 100 for result in strategy_results.values()]
            
            bars3 = ax3.bar(strategy_names, error_rates, color='orange', alpha=0.8)
            ax3.set_title('负载均衡策略 - 错误率对比', fontsize=14, fontweight='bold')
            ax3.set_ylabel('错误率 (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars3, error_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # 节点负载分布（以第一个策略为例）
            first_strategy = list(strategy_results.keys())[0]
            node_metrics = strategy_results[first_strategy]["node_metrics"]
            
            node_ids = [m["node_id"].split("_")[1] for m in node_metrics]
            cpu_usage = [m["cpu_usage"] * 100 for m in node_metrics]
            memory_usage = [m["memory_usage"] * 100 for m in node_metrics]
            
            x = np.arange(len(node_ids))
            width = 0.35
            
            ax4.bar(x - width/2, cpu_usage, width, label='CPU使用率', alpha=0.8)
            ax4.bar(x + width/2, memory_usage, width, label='内存使用率', alpha=0.8)
            
            ax4.set_title(f'节点资源使用率 ({first_strategy.replace("_", " ").title()})', fontsize=14, fontweight='bold')
            ax4.set_ylabel('使用率 (%)')
            ax4.set_xlabel('节点ID')
            ax4.set_xticks(x)
            ax4.set_xticklabels(node_ids)
            ax4.legend()
            
            plt.tight_layout()
            plt.savefig('outputs/load_balancing_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 可扩展性测试图
        if "scalability" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            scalability_data = self.results["scalability"]
            node_counts = list(scalability_data.keys())
            throughputs = [data["throughput"] for data in scalability_data.values()]
            latencies = [data["avg_latency"] * 1000 for data in scalability_data.values()]
            
            # 吞吐量随节点数变化
            ax1.plot(node_counts, throughputs, 'o-', linewidth=2, markersize=8, color='blue')
            ax1.set_title('可扩展性测试 - 吞吐量', fontsize=14, fontweight='bold')
            ax1.set_xlabel('节点数量')
            ax1.set_ylabel('吞吐量 (requests/second)')
            ax1.grid(True, alpha=0.3)
            
            # 添加数值标签
            for x, y in zip(node_counts, throughputs):
                ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            # 延迟随节点数变化
            ax2.plot(node_counts, latencies, 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_title('可扩展性测试 - 平均延迟', fontsize=14, fontweight='bold')
            ax2.set_xlabel('节点数量')
            ax2.set_ylabel('平均延迟 (milliseconds)')
            ax2.grid(True, alpha=0.3)
            
            # 添加数值标签
            for x, y in zip(node_counts, latencies):
                ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig('outputs/scalability_test.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """主函数"""
    print("🚀 分布式推理基础概念演示")
    print("="*60)
    
    # 创建输出目录
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 创建基准测试实例
    benchmark = DistributedInferenceBenchmark()
    
    # 运行负载均衡策略对比
    benchmark.run_load_balancing_comparison(num_requests=500, num_nodes=3)
    
    # 运行可扩展性测试
    benchmark.run_scalability_test(max_nodes=6, requests_per_node=100)
    
    # 打印结果
    benchmark.print_results()
    
    # 可视化结果
    benchmark.visualize_results()
    
    print("\n✅ 分布式推理基础概念演示完成！")
    print("📁 结果已保存到 outputs/ 目录")

if __name__ == "__main__":
    main()