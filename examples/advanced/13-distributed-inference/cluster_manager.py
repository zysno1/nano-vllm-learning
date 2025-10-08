"""
集群管理器实现

本模块实现分布式推理集群的管理功能，包括：
1. 服务发现和注册
2. 健康检查机制
3. 节点状态管理
4. 自动故障恢复
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import socket
import requests
from collections import defaultdict, deque
import uuid
import hashlib
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    """服务类型枚举"""
    INFERENCE_NODE = "inference_node"
    LOAD_BALANCER = "load_balancer"
    MONITOR = "monitor"
    GATEWAY = "gateway"

class NodeState(Enum):
    """节点状态枚举"""
    REGISTERING = "registering"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

@dataclass
class ServiceInfo:
    """服务信息"""
    service_id: str
    service_type: ServiceType
    host: str
    port: int
    metadata: Dict[str, Any]
    health_check_url: str
    registration_time: float
    last_heartbeat: float
    state: NodeState = NodeState.REGISTERING
    
    def __post_init__(self):
        if self.registration_time == 0:
            self.registration_time = time.time()
        if self.last_heartbeat == 0:
            self.last_heartbeat = time.time()

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_id: str
    is_healthy: bool
    response_time: float
    error_message: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

@dataclass
class ClusterStats:
    """集群统计信息"""
    total_nodes: int = 0
    healthy_nodes: int = 0
    unhealthy_nodes: int = 0
    offline_nodes: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    cluster_load: float = 0.0
    uptime: float = 0.0

class ServiceRegistry:
    """服务注册中心"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.service_types: Dict[ServiceType, Set[str]] = defaultdict(set)
        self.lock = threading.RLock()
        self.change_listeners: List[Callable] = []
    
    def register_service(self, service_info: ServiceInfo) -> bool:
        """注册服务"""
        with self.lock:
            try:
                self.services[service_info.service_id] = service_info
                self.service_types[service_info.service_type].add(service_info.service_id)
                
                logger.info(f"服务注册成功: {service_info.service_id} "
                           f"({service_info.service_type.value}) "
                           f"at {service_info.host}:{service_info.port}")
                
                self._notify_listeners("register", service_info)
                return True
                
            except Exception as e:
                logger.error(f"服务注册失败: {e}")
                return False
    
    def unregister_service(self, service_id: str) -> bool:
        """注销服务"""
        with self.lock:
            try:
                if service_id in self.services:
                    service_info = self.services[service_id]
                    del self.services[service_id]
                    self.service_types[service_info.service_type].discard(service_id)
                    
                    logger.info(f"服务注销成功: {service_id}")
                    self._notify_listeners("unregister", service_info)
                    return True
                else:
                    logger.warning(f"尝试注销不存在的服务: {service_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"服务注销失败: {e}")
                return False
    
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """获取服务信息"""
        with self.lock:
            return self.services.get(service_id)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[ServiceInfo]:
        """根据类型获取服务列表"""
        with self.lock:
            service_ids = self.service_types.get(service_type, set())
            return [self.services[sid] for sid in service_ids if sid in self.services]
    
    def get_healthy_services(self, service_type: Optional[ServiceType] = None) -> List[ServiceInfo]:
        """获取健康的服务列表"""
        with self.lock:
            if service_type:
                services = self.get_services_by_type(service_type)
            else:
                services = list(self.services.values())
            
            return [s for s in services if s.state == NodeState.HEALTHY]
    
    def update_service_state(self, service_id: str, state: NodeState) -> bool:
        """更新服务状态"""
        with self.lock:
            if service_id in self.services:
                old_state = self.services[service_id].state
                self.services[service_id].state = state
                
                if old_state != state:
                    logger.info(f"服务状态更新: {service_id} {old_state.value} -> {state.value}")
                    self._notify_listeners("state_change", self.services[service_id])
                
                return True
            return False
    
    def update_heartbeat(self, service_id: str) -> bool:
        """更新心跳时间"""
        with self.lock:
            if service_id in self.services:
                self.services[service_id].last_heartbeat = time.time()
                return True
            return False
    
    def add_change_listener(self, listener: Callable):
        """添加变更监听器"""
        self.change_listeners.append(listener)
    
    def _notify_listeners(self, event_type: str, service_info: ServiceInfo):
        """通知监听器"""
        for listener in self.change_listeners:
            try:
                listener(event_type, service_info)
            except Exception as e:
                logger.error(f"通知监听器失败: {e}")

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, registry: ServiceRegistry, check_interval: float = 10.0, timeout: float = 5.0):
        self.registry = registry
        self.check_interval = check_interval
        self.timeout = timeout
        self.is_running = False
        self.check_thread = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    
    def start(self):
        """启动健康检查"""
        self.is_running = True
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()
        logger.info("健康检查器已启动")
    
    def stop(self):
        """停止健康检查"""
        self.is_running = False
        if self.check_thread:
            self.check_thread.join(timeout=1.0)
        self.executor.shutdown(wait=False)
        logger.info("健康检查器已停止")
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                services = list(self.registry.services.values())
                
                # 并发执行健康检查
                futures = []
                for service in services:
                    if service.state != NodeState.OFFLINE:
                        future = self.executor.submit(self._check_service_health, service)
                        futures.append(future)
                
                # 收集结果
                for future in futures:
                    try:
                        result = future.result(timeout=self.timeout + 1)
                        self._process_health_result(result)
                    except Exception as e:
                        logger.error(f"健康检查异常: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                time.sleep(1.0)
    
    def _check_service_health(self, service: ServiceInfo) -> HealthCheckResult:
        """检查单个服务健康状态"""
        start_time = time.time()
        
        try:
            # HTTP健康检查
            response = requests.get(
                service.health_check_url,
                timeout=self.timeout,
                headers={'User-Agent': 'ClusterManager/1.0'}
            )
            
            response_time = time.time() - start_time
            is_healthy = response.status_code == 200
            
            return HealthCheckResult(
                service_id=service.service_id,
                is_healthy=is_healthy,
                response_time=response_time,
                error_message="" if is_healthy else f"HTTP {response.status_code}"
            )
            
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                service_id=service.service_id,
                is_healthy=False,
                response_time=self.timeout,
                error_message="请求超时"
            )
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                service_id=service.service_id,
                is_healthy=False,
                response_time=time.time() - start_time,
                error_message="连接失败"
            )
        except Exception as e:
            return HealthCheckResult(
                service_id=service.service_id,
                is_healthy=False,
                response_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _process_health_result(self, result: HealthCheckResult):
        """处理健康检查结果"""
        service = self.registry.get_service(result.service_id)
        if not service:
            return
        
        # 记录健康检查历史
        self.health_history[result.service_id].append(result)
        
        # 更新心跳时间
        if result.is_healthy:
            self.registry.update_heartbeat(result.service_id)
        
        # 根据健康检查结果和历史记录决定状态
        new_state = self._determine_service_state(result.service_id, result)
        
        if new_state != service.state:
            self.registry.update_service_state(result.service_id, new_state)
    
    def _determine_service_state(self, service_id: str, current_result: HealthCheckResult) -> NodeState:
        """根据健康检查历史确定服务状态"""
        history = self.health_history[service_id]
        
        if len(history) < 2:
            return NodeState.HEALTHY if current_result.is_healthy else NodeState.UNHEALTHY
        
        # 计算最近几次检查的成功率
        recent_checks = list(history)[-5:]  # 最近5次检查
        success_rate = sum(1 for r in recent_checks if r.is_healthy) / len(recent_checks)
        
        if success_rate >= 0.8:
            return NodeState.HEALTHY
        elif success_rate >= 0.4:
            return NodeState.UNHEALTHY
        else:
            return NodeState.OFFLINE

class LoadBalancingStrategy:
    """负载均衡策略接口"""
    
    def select_service(self, services: List[ServiceInfo], request_info: Dict = None) -> Optional[ServiceInfo]:
        """选择服务"""
        raise NotImplementedError

class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询策略"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select_service(self, services: List[ServiceInfo], request_info: Dict = None) -> Optional[ServiceInfo]:
        if not services:
            return None
        
        with self.lock:
            service = services[self.current_index % len(services)]
            self.current_index += 1
            return service

class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """加权轮询策略"""
    
    def __init__(self):
        self.current_weights = {}
        self.lock = threading.Lock()
    
    def select_service(self, services: List[ServiceInfo], request_info: Dict = None) -> Optional[ServiceInfo]:
        if not services:
            return None
        
        with self.lock:
            # 获取权重（基于CPU和内存使用率）
            best_service = None
            best_weight = -1
            
            for service in services:
                # 从metadata中获取负载信息
                cpu_usage = service.metadata.get('cpu_usage', 0.5)
                memory_usage = service.metadata.get('memory_usage', 0.5)
                
                # 计算权重（负载越低权重越高）
                weight = 1.0 - (cpu_usage + memory_usage) / 2
                
                if weight > best_weight:
                    best_weight = weight
                    best_service = service
            
            return best_service

class ClusterManager:
    """集群管理器"""
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.registry = ServiceRegistry()
        self.health_checker = HealthChecker(self.registry)
        self.load_balancer = WeightedRoundRobinStrategy()
        self.is_running = False
        self.stats = ClusterStats()
        self.start_time = time.time()
        
        # 监控数据
        self.request_count = 0
        self.response_times = deque(maxlen=1000)
        
        # 注册变更监听器
        self.registry.add_change_listener(self._on_service_change)
    
    def start(self):
        """启动集群管理器"""
        self.is_running = True
        self.start_time = time.time()
        
        # 启动健康检查器
        self.health_checker.start()
        
        # 启动统计更新线程
        stats_thread = threading.Thread(target=self._update_stats_loop, daemon=True)
        stats_thread.start()
        
        logger.info(f"集群管理器已启动: {self.host}:{self.port}")
    
    def stop(self):
        """停止集群管理器"""
        self.is_running = False
        self.health_checker.stop()
        logger.info("集群管理器已停止")
    
    def register_service(self, service_info: ServiceInfo) -> bool:
        """注册服务"""
        return self.registry.register_service(service_info)
    
    def unregister_service(self, service_id: str) -> bool:
        """注销服务"""
        return self.registry.unregister_service(service_id)
    
    def get_available_services(self, service_type: ServiceType) -> List[ServiceInfo]:
        """获取可用服务列表"""
        return self.registry.get_healthy_services(service_type)
    
    def select_service(self, service_type: ServiceType, request_info: Dict = None) -> Optional[ServiceInfo]:
        """选择服务"""
        available_services = self.get_available_services(service_type)
        return self.load_balancer.select_service(available_services, request_info)
    
    def record_request(self, response_time: float):
        """记录请求统计"""
        self.request_count += 1
        self.response_times.append(response_time)
    
    def get_cluster_stats(self) -> ClusterStats:
        """获取集群统计信息"""
        return self.stats
    
    def get_service_list(self) -> List[Dict]:
        """获取服务列表"""
        services = []
        for service in self.registry.services.values():
            service_dict = asdict(service)
            service_dict['service_type'] = service.service_type.value
            service_dict['state'] = service.state.value
            services.append(service_dict)
        return services
    
    def _on_service_change(self, event_type: str, service_info: ServiceInfo):
        """服务变更回调"""
        logger.info(f"服务变更事件: {event_type} - {service_info.service_id}")
        
        # 可以在这里添加自定义逻辑，如通知其他组件、记录日志等
        if event_type == "register":
            logger.info(f"新服务上线: {service_info.service_id}")
        elif event_type == "unregister":
            logger.info(f"服务下线: {service_info.service_id}")
        elif event_type == "state_change":
            logger.info(f"服务状态变更: {service_info.service_id} -> {service_info.state.value}")
    
    def _update_stats_loop(self):
        """更新统计信息循环"""
        while self.is_running:
            try:
                self._update_cluster_stats()
                time.sleep(5.0)  # 每5秒更新一次
            except Exception as e:
                logger.error(f"更新统计信息失败: {e}")
                time.sleep(1.0)
    
    def _update_cluster_stats(self):
        """更新集群统计信息"""
        services = list(self.registry.services.values())
        
        # 统计节点数量
        self.stats.total_nodes = len(services)
        self.stats.healthy_nodes = sum(1 for s in services if s.state == NodeState.HEALTHY)
        self.stats.unhealthy_nodes = sum(1 for s in services if s.state == NodeState.UNHEALTHY)
        self.stats.offline_nodes = sum(1 for s in services if s.state == NodeState.OFFLINE)
        
        # 统计请求信息
        self.stats.total_requests = self.request_count
        
        if self.response_times:
            self.stats.avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # 计算集群负载
        if services:
            total_load = 0
            for service in services:
                cpu_usage = service.metadata.get('cpu_usage', 0)
                memory_usage = service.metadata.get('memory_usage', 0)
                total_load += (cpu_usage + memory_usage) / 2
            
            self.stats.cluster_load = total_load / len(services)
        
        # 计算运行时间
        self.stats.uptime = time.time() - self.start_time

class ServiceClient:
    """服务客户端"""
    
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager
    
    def make_request(self, service_type: ServiceType, request_data: Dict) -> Dict:
        """发起请求"""
        start_time = time.time()
        
        try:
            # 选择服务
            service = self.cluster_manager.select_service(service_type, request_data)
            
            if not service:
                return {
                    "success": False,
                    "error": "没有可用的服务",
                    "response_time": time.time() - start_time
                }
            
            # 模拟请求处理
            processing_time = 0.1 + (hash(str(request_data)) % 100) / 1000  # 0.1-0.2秒
            time.sleep(processing_time)
            
            response_time = time.time() - start_time
            
            # 记录请求统计
            self.cluster_manager.record_request(response_time)
            
            return {
                "success": True,
                "service_id": service.service_id,
                "response_data": f"Response from {service.service_id}",
                "response_time": response_time
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time
            }

def create_mock_service(service_id: str, service_type: ServiceType, port: int) -> ServiceInfo:
    """创建模拟服务"""
    return ServiceInfo(
        service_id=service_id,
        service_type=service_type,
        host="localhost",
        port=port,
        metadata={
            "cpu_usage": 0.3 + (hash(service_id) % 40) / 100,  # 0.3-0.7
            "memory_usage": 0.2 + (hash(service_id) % 50) / 100,  # 0.2-0.7
            "gpu_usage": 0.1 + (hash(service_id) % 60) / 100,  # 0.1-0.7
            "model_name": f"model_{service_id}",
            "version": "1.0.0"
        },
        health_check_url=f"http://localhost:{port}/health"
    )

def simulate_cluster_operations():
    """模拟集群操作"""
    print("🚀 启动集群管理器演示")
    print("="*60)
    
    # 创建集群管理器
    cluster_manager = ClusterManager()
    cluster_manager.start()
    
    try:
        # 注册多个推理服务
        services = []
        for i in range(5):
            service = create_mock_service(
                service_id=f"inference_node_{i}",
                service_type=ServiceType.INFERENCE_NODE,
                port=8001 + i
            )
            services.append(service)
            cluster_manager.register_service(service)
            time.sleep(0.1)
        
        print(f"✅ 已注册 {len(services)} 个推理服务")
        
        # 创建客户端
        client = ServiceClient(cluster_manager)
        
        # 模拟请求
        print("\n📊 开始模拟请求...")
        for i in range(20):
            request_data = {"text": f"请求 {i}", "max_tokens": 100}
            response = client.make_request(ServiceType.INFERENCE_NODE, request_data)
            
            if response["success"]:
                print(f"请求 {i}: 成功 - 服务 {response['service_id']} - "
                      f"响应时间 {response['response_time']*1000:.1f}ms")
            else:
                print(f"请求 {i}: 失败 - {response['error']}")
            
            time.sleep(0.1)
        
        # 模拟服务故障
        print("\n⚠️  模拟服务故障...")
        failed_service = services[0]
        cluster_manager.registry.update_service_state(failed_service.service_id, NodeState.OFFLINE)
        
        # 继续发送请求
        print("📊 故障后继续请求...")
        for i in range(10):
            request_data = {"text": f"故障后请求 {i}", "max_tokens": 100}
            response = client.make_request(ServiceType.INFERENCE_NODE, request_data)
            
            if response["success"]:
                print(f"请求 {i}: 成功 - 服务 {response['service_id']}")
            else:
                print(f"请求 {i}: 失败 - {response['error']}")
            
            time.sleep(0.1)
        
        # 显示集群状态
        print("\n📈 集群统计信息:")
        stats = cluster_manager.get_cluster_stats()
        print(f"  总节点数: {stats.total_nodes}")
        print(f"  健康节点: {stats.healthy_nodes}")
        print(f"  不健康节点: {stats.unhealthy_nodes}")
        print(f"  离线节点: {stats.offline_nodes}")
        print(f"  总请求数: {stats.total_requests}")
        print(f"  平均响应时间: {stats.avg_response_time*1000:.1f}ms")
        print(f"  集群负载: {stats.cluster_load*100:.1f}%")
        print(f"  运行时间: {stats.uptime:.1f}s")
        
        # 显示服务列表
        print("\n📋 服务列表:")
        services_list = cluster_manager.get_service_list()
        for service in services_list:
            print(f"  {service['service_id']}: {service['state']} "
                  f"(CPU: {service['metadata']['cpu_usage']*100:.1f}%, "
                  f"内存: {service['metadata']['memory_usage']*100:.1f}%)")
        
        # 恢复故障服务
        print("\n🔄 恢复故障服务...")
        cluster_manager.registry.update_service_state(failed_service.service_id, NodeState.HEALTHY)
        
        time.sleep(2)
        
    finally:
        cluster_manager.stop()
    
    print("\n✅ 集群管理器演示完成！")

def main():
    """主函数"""
    simulate_cluster_operations()

if __name__ == "__main__":
    main()