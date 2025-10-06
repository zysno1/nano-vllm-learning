# 🚀 部署策略 (Deployment Strategies)

## 📖 概述

部署策略是将nano-vllm模型从开发环境迁移到生产环境的关键环节。本文档详细介绍了多种部署方案，包括单机部署、分布式部署、云原生部署等，以及相应的配置、监控和优化策略。

## 🏗️ 部署架构

### 1. 部署管理器

```python
import torch
import torch.distributed as dist
import asyncio
import aiohttp
import docker
import kubernetes
import yaml
import json
import os
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

class DeploymentType(Enum):
    """部署类型"""
    SINGLE_NODE = "single_node"
    MULTI_NODE = "multi_node"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    SERVERLESS = "serverless"
    EDGE = "edge"

class ServiceType(Enum):
    """服务类型"""
    HTTP_API = "http_api"
    GRPC = "grpc"
    WEBSOCKET = "websocket"
    BATCH_PROCESSING = "batch_processing"
    STREAMING = "streaming"

class ScalingStrategy(Enum):
    """扩缩容策略"""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_REQUEST = "auto_request"
    AUTO_CUSTOM = "auto_custom"

@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_name: str
    deployment_type: DeploymentType
    service_type: ServiceType
    
    # 资源配置
    cpu_cores: int = 4
    memory_gb: int = 16
    gpu_count: int = 1
    gpu_memory_gb: int = 24
    
    # 网络配置
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 1000
    
    # 模型配置
    model_path: str = ""
    model_name: str = ""
    batch_size: int = 32
    max_sequence_length: int = 2048
    
    # 扩缩容配置
    scaling_strategy: ScalingStrategy = ScalingStrategy.MANUAL
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    
    # 健康检查配置
    health_check_path: str = "/health"
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    
    # 自定义配置
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'deployment_name': self.deployment_name,
            'deployment_type': self.deployment_type.value,
            'service_type': self.service_type.value,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'gpu_count': self.gpu_count,
            'gpu_memory_gb': self.gpu_memory_gb,
            'host': self.host,
            'port': self.port,
            'max_connections': self.max_connections,
            'model_path': self.model_path,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_sequence_length': self.max_sequence_length,
            'scaling_strategy': self.scaling_strategy.value,
            'min_replicas': self.min_replicas,
            'max_replicas': self.max_replicas,
            'target_cpu_utilization': self.target_cpu_utilization,
            'target_memory_utilization': self.target_memory_utilization,
            'health_check_path': self.health_check_path,
            'health_check_interval': self.health_check_interval,
            'health_check_timeout': self.health_check_timeout,
            'log_level': self.log_level,
            'log_format': self.log_format,
            'custom_config': self.custom_config,
        }

class BaseDeploymentStrategy:
    """基础部署策略"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = None
        self.status = "stopped"
        self.metrics = {}
        
    async def deploy(self) -> str:
        """部署服务"""
        raise NotImplementedError
    
    async def undeploy(self) -> bool:
        """停止部署"""
        raise NotImplementedError
    
    async def scale(self, replicas: int) -> bool:
        """扩缩容"""
        raise NotImplementedError
    
    async def update(self, new_config: DeploymentConfig) -> bool:
        """更新配置"""
        raise NotImplementedError
    
    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        raise NotImplementedError
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """健康检查"""
        raise NotImplementedError

class SingleNodeDeployment(BaseDeploymentStrategy):
    """单机部署策略"""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.process = None
        self.server_thread = None
        
    async def deploy(self) -> str:
        """部署单机服务"""
        
        logger.info(f"Deploying single node service: {self.config.deployment_name}")
        
        try:
            # 验证资源
            await self._validate_resources()
            
            # 准备环境
            await self._prepare_environment()
            
            # 启动服务
            await self._start_service()
            
            # 等待服务就绪
            await self._wait_for_ready()
            
            self.deployment_id = f"single-{self.config.deployment_name}-{int(time.time())}"
            self.status = "running"
            
            logger.info(f"Single node deployment completed: {self.deployment_id}")
            
            return self.deployment_id
            
        except Exception as e:
            logger.error(f"Single node deployment failed: {e}")
            await self.undeploy()
            raise
    
    async def _validate_resources(self):
        """验证资源"""
        
        # 检查CPU
        available_cpu = psutil.cpu_count()
        if self.config.cpu_cores > available_cpu:
            raise ValueError(f"Insufficient CPU: need {self.config.cpu_cores}, available {available_cpu}")
        
        # 检查内存
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if self.config.memory_gb > available_memory:
            raise ValueError(f"Insufficient memory: need {self.config.memory_gb}GB, available {available_memory:.1f}GB")
        
        # 检查GPU
        if self.config.gpu_count > 0:
            if not torch.cuda.is_available():
                raise ValueError("CUDA not available but GPU required")
            
            available_gpu = torch.cuda.device_count()
            if self.config.gpu_count > available_gpu:
                raise ValueError(f"Insufficient GPU: need {self.config.gpu_count}, available {available_gpu}")
    
    async def _prepare_environment(self):
        """准备环境"""
        
        # 设置环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(self.config.gpu_count))
        os.environ['OMP_NUM_THREADS'] = str(self.config.cpu_cores)
        
        # 创建工作目录
        work_dir = Path(f"/tmp/nano-vllm-{self.config.deployment_name}")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成配置文件
        config_path = work_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    async def _start_service(self):
        """启动服务"""
        
        if self.config.service_type == ServiceType.HTTP_API:
            await self._start_http_service()
        elif self.config.service_type == ServiceType.GRPC:
            await self._start_grpc_service()
        elif self.config.service_type == ServiceType.WEBSOCKET:
            await self._start_websocket_service()
        else:
            raise ValueError(f"Unsupported service type: {self.config.service_type}")
    
    async def _start_http_service(self):
        """启动HTTP服务"""
        
        from aiohttp import web, web_runner
        
        app = web.Application()
        
        # 添加路由
        app.router.add_post('/generate', self._handle_generate)
        app.router.add_get('/health', self._handle_health)
        app.router.add_get('/metrics', self._handle_metrics)
        
        # 启动服务器
        runner = web_runner.AppRunner(app)
        await runner.setup()
        
        site = web_runner.TCPSite(
            runner,
            self.config.host,
            self.config.port
        )
        
        await site.start()
        
        logger.info(f"HTTP service started on {self.config.host}:{self.config.port}")
    
    async def _handle_generate(self, request):
        """处理生成请求"""
        
        try:
            data = await request.json()
            
            # 模拟生成逻辑
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 100)
            
            # 这里应该调用实际的模型推理
            response = {
                'text': f"Generated response for: {prompt}",
                'tokens': max_tokens,
                'model': self.config.model_name
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def _handle_health(self, request):
        """处理健康检查"""
        
        health_status = {
            'status': self.status,
            'deployment_id': self.deployment_id,
            'timestamp': time.time(),
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
        
        return web.json_response(health_status)
    
    async def _handle_metrics(self, request):
        """处理指标请求"""
        
        metrics = await self.get_metrics()
        return web.json_response(metrics)
    
    async def _wait_for_ready(self):
        """等待服务就绪"""
        
        max_wait = 60  # 最大等待60秒
        wait_interval = 1
        
        for _ in range(max_wait):
            try:
                if await self.health_check():
                    return
            except:
                pass
            
            await asyncio.sleep(wait_interval)
        
        raise TimeoutError("Service failed to become ready")
    
    async def undeploy(self) -> bool:
        """停止单机部署"""
        
        logger.info(f"Stopping single node deployment: {self.deployment_id}")
        
        try:
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=30)
            
            if self.server_thread:
                # 停止服务器线程的逻辑
                pass
            
            self.status = "stopped"
            
            logger.info("Single node deployment stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop deployment: {e}")
            return False
    
    async def scale(self, replicas: int) -> bool:
        """单机部署不支持扩缩容"""
        
        logger.warning("Single node deployment does not support scaling")
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        
        return {
            'deployment_id': self.deployment_id,
            'status': self.status,
            'config': self.config.to_dict(),
            'metrics': await self.get_metrics()
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        
        # 获取系统指标
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = {
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'timestamp': time.time()
        }
        
        # 获取GPU指标
        if torch.cuda.is_available():
            gpu_metrics = []
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_used = torch.cuda.memory_allocated(i) / (1024**3)
                
                gpu_metrics.append({
                    'gpu_id': i,
                    'gpu_utilization': torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0,
                    'gpu_memory_total': gpu_memory,
                    'gpu_memory_used': gpu_used,
                    'gpu_memory_utilization': (gpu_used / gpu_memory) * 100
                })
            
            metrics['gpu_metrics'] = gpu_metrics
        
        return metrics
    
    async def health_check(self) -> bool:
        """健康检查"""
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{self.config.host}:{self.config.port}{self.config.health_check_path}"
                
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class KubernetesDeployment(BaseDeploymentStrategy):
    """Kubernetes部署策略"""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.k8s_client = None
        self.namespace = config.custom_config.get('namespace', 'default')
        
    async def deploy(self) -> str:
        """部署到Kubernetes"""
        
        logger.info(f"Deploying to Kubernetes: {self.config.deployment_name}")
        
        try:
            # 初始化Kubernetes客户端
            await self._init_k8s_client()
            
            # 创建部署资源
            await self._create_deployment()
            
            # 创建服务资源
            await self._create_service()
            
            # 创建HPA（如果启用自动扩缩容）
            if self.config.scaling_strategy != ScalingStrategy.MANUAL:
                await self._create_hpa()
            
            # 等待部署就绪
            await self._wait_for_deployment_ready()
            
            self.deployment_id = f"k8s-{self.config.deployment_name}"
            self.status = "running"
            
            logger.info(f"Kubernetes deployment completed: {self.deployment_id}")
            
            return self.deployment_id
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            await self.undeploy()
            raise
    
    async def _init_k8s_client(self):
        """初始化Kubernetes客户端"""
        
        try:
            from kubernetes import client, config
            
            # 尝试加载集群内配置
            try:
                config.load_incluster_config()
            except:
                # 加载本地配置
                config.load_kube_config()
            
            self.k8s_client = client.ApiClient()
            
        except ImportError:
            raise ImportError("kubernetes library not installed")
    
    async def _create_deployment(self):
        """创建Kubernetes Deployment"""
        
        from kubernetes import client
        
        # 构建容器规格
        container = client.V1Container(
            name=self.config.deployment_name,
            image=self.config.custom_config.get('image', 'nano-vllm:latest'),
            ports=[client.V1ContainerPort(container_port=self.config.port)],
            resources=client.V1ResourceRequirements(
                requests={
                    'cpu': f"{self.config.cpu_cores}",
                    'memory': f"{self.config.memory_gb}Gi"
                },
                limits={
                    'cpu': f"{self.config.cpu_cores}",
                    'memory': f"{self.config.memory_gb}Gi"
                }
            ),
            env=[
                client.V1EnvVar(name="MODEL_PATH", value=self.config.model_path),
                client.V1EnvVar(name="MODEL_NAME", value=self.config.model_name),
                client.V1EnvVar(name="BATCH_SIZE", value=str(self.config.batch_size)),
                client.V1EnvVar(name="MAX_SEQUENCE_LENGTH", value=str(self.config.max_sequence_length)),
            ],
            liveness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=self.config.health_check_path,
                    port=self.config.port
                ),
                initial_delay_seconds=30,
                period_seconds=self.config.health_check_interval
            ),
            readiness_probe=client.V1Probe(
                http_get=client.V1HTTPGetAction(
                    path=self.config.health_check_path,
                    port=self.config.port
                ),
                initial_delay_seconds=10,
                period_seconds=5
            )
        )
        
        # 添加GPU资源（如果需要）
        if self.config.gpu_count > 0:
            if container.resources.requests is None:
                container.resources.requests = {}
            if container.resources.limits is None:
                container.resources.limits = {}
            
            container.resources.requests['nvidia.com/gpu'] = str(self.config.gpu_count)
            container.resources.limits['nvidia.com/gpu'] = str(self.config.gpu_count)
        
        # 构建Pod规格
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Always"
        )
        
        # 构建Deployment规格
        deployment_spec = client.V1DeploymentSpec(
            replicas=self.config.min_replicas,
            selector=client.V1LabelSelector(
                match_labels={'app': self.config.deployment_name}
            ),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={'app': self.config.deployment_name}
                ),
                spec=pod_spec
            )
        )
        
        # 创建Deployment对象
        deployment = client.V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=client.V1ObjectMeta(
                name=self.config.deployment_name,
                namespace=self.namespace
            ),
            spec=deployment_spec
        )
        
        # 应用Deployment
        apps_v1 = client.AppsV1Api(self.k8s_client)
        apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment
        )
        
        logger.info(f"Kubernetes deployment created: {self.config.deployment_name}")
    
    async def _create_service(self):
        """创建Kubernetes Service"""
        
        from kubernetes import client
        
        # 构建Service规格
        service_spec = client.V1ServiceSpec(
            selector={'app': self.config.deployment_name},
            ports=[client.V1ServicePort(
                port=self.config.port,
                target_port=self.config.port,
                protocol="TCP"
            )],
            type="ClusterIP"
        )
        
        # 创建Service对象
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=f"{self.config.deployment_name}-service",
                namespace=self.namespace
            ),
            spec=service_spec
        )
        
        # 应用Service
        core_v1 = client.CoreV1Api(self.k8s_client)
        core_v1.create_namespaced_service(
            namespace=self.namespace,
            body=service
        )
        
        logger.info(f"Kubernetes service created: {self.config.deployment_name}-service")
    
    async def _create_hpa(self):
        """创建Horizontal Pod Autoscaler"""
        
        from kubernetes import client
        
        # 构建HPA规格
        hpa_spec = client.V2HorizontalPodAutoscalerSpec(
            scale_target_ref=client.V2CrossVersionObjectReference(
                api_version="apps/v1",
                kind="Deployment",
                name=self.config.deployment_name
            ),
            min_replicas=self.config.min_replicas,
            max_replicas=self.config.max_replicas,
            metrics=[
                client.V2MetricSpec(
                    type="Resource",
                    resource=client.V2ResourceMetricSource(
                        name="cpu",
                        target=client.V2MetricTarget(
                            type="Utilization",
                            average_utilization=int(self.config.target_cpu_utilization)
                        )
                    )
                )
            ]
        )
        
        # 创建HPA对象
        hpa = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(
                name=f"{self.config.deployment_name}-hpa",
                namespace=self.namespace
            ),
            spec=hpa_spec
        )
        
        # 应用HPA
        autoscaling_v2 = client.AutoscalingV2Api(self.k8s_client)
        autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
            namespace=self.namespace,
            body=hpa
        )
        
        logger.info(f"HPA created: {self.config.deployment_name}-hpa")
    
    async def _wait_for_deployment_ready(self):
        """等待部署就绪"""
        
        from kubernetes import client
        
        apps_v1 = client.AppsV1Api(self.k8s_client)
        
        max_wait = 300  # 最大等待5分钟
        wait_interval = 10
        
        for _ in range(max_wait // wait_interval):
            try:
                deployment = apps_v1.read_namespaced_deployment(
                    name=self.config.deployment_name,
                    namespace=self.namespace
                )
                
                if (deployment.status.ready_replicas and 
                    deployment.status.ready_replicas >= self.config.min_replicas):
                    logger.info("Kubernetes deployment is ready")
                    return
                
            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")
            
            await asyncio.sleep(wait_interval)
        
        raise TimeoutError("Kubernetes deployment failed to become ready")
    
    async def undeploy(self) -> bool:
        """停止Kubernetes部署"""
        
        logger.info(f"Stopping Kubernetes deployment: {self.config.deployment_name}")
        
        try:
            from kubernetes import client
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            core_v1 = client.CoreV1Api(self.k8s_client)
            autoscaling_v2 = client.AutoscalingV2Api(self.k8s_client)
            
            # 删除HPA
            try:
                autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{self.config.deployment_name}-hpa",
                    namespace=self.namespace
                )
            except:
                pass
            
            # 删除Service
            try:
                core_v1.delete_namespaced_service(
                    name=f"{self.config.deployment_name}-service",
                    namespace=self.namespace
                )
            except:
                pass
            
            # 删除Deployment
            try:
                apps_v1.delete_namespaced_deployment(
                    name=self.config.deployment_name,
                    namespace=self.namespace
                )
            except:
                pass
            
            self.status = "stopped"
            
            logger.info("Kubernetes deployment stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Kubernetes deployment: {e}")
            return False
    
    async def scale(self, replicas: int) -> bool:
        """扩缩容"""
        
        try:
            from kubernetes import client
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            # 更新副本数
            deployment = apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.namespace
            )
            
            deployment.spec.replicas = replicas
            
            apps_v1.patch_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Scaled deployment to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        
        try:
            from kubernetes import client
            
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            deployment = apps_v1.read_namespaced_deployment(
                name=self.config.deployment_name,
                namespace=self.namespace
            )
            
            return {
                'deployment_id': self.deployment_id,
                'status': self.status,
                'replicas': deployment.spec.replicas,
                'ready_replicas': deployment.status.ready_replicas or 0,
                'available_replicas': deployment.status.available_replicas or 0,
                'config': self.config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {'error': str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        
        # 这里应该集成Prometheus或其他监控系统
        # 简化实现
        return {
            'timestamp': time.time(),
            'deployment_name': self.config.deployment_name,
            'namespace': self.namespace
        }
    
    async def health_check(self) -> bool:
        """健康检查"""
        
        try:
            status = await self.get_status()
            ready_replicas = status.get('ready_replicas', 0)
            return ready_replicas > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class DockerDeployment(BaseDeploymentStrategy):
    """Docker部署策略"""
    
    def __init__(self, config: DeploymentConfig):
        super().__init__(config)
        self.docker_client = None
        self.container = None
        
    async def deploy(self) -> str:
        """部署Docker容器"""
        
        logger.info(f"Deploying Docker container: {self.config.deployment_name}")
        
        try:
            # 初始化Docker客户端
            self._init_docker_client()
            
            # 构建或拉取镜像
            await self._prepare_image()
            
            # 启动容器
            await self._start_container()
            
            # 等待容器就绪
            await self._wait_for_container_ready()
            
            self.deployment_id = f"docker-{self.container.id[:12]}"
            self.status = "running"
            
            logger.info(f"Docker deployment completed: {self.deployment_id}")
            
            return self.deployment_id
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            await self.undeploy()
            raise
    
    def _init_docker_client(self):
        """初始化Docker客户端"""
        
        try:
            import docker
            self.docker_client = docker.from_env()
            
        except ImportError:
            raise ImportError("docker library not installed")
    
    async def _prepare_image(self):
        """准备Docker镜像"""
        
        image_name = self.config.custom_config.get('image', 'nano-vllm:latest')
        
        try:
            # 尝试拉取镜像
            self.docker_client.images.pull(image_name)
            logger.info(f"Pulled Docker image: {image_name}")
            
        except Exception as e:
            logger.warning(f"Failed to pull image {image_name}: {e}")
            
            # 如果拉取失败，检查本地是否存在
            try:
                self.docker_client.images.get(image_name)
                logger.info(f"Using local Docker image: {image_name}")
                
            except:
                raise ValueError(f"Docker image not found: {image_name}")
    
    async def _start_container(self):
        """启动容器"""
        
        image_name = self.config.custom_config.get('image', 'nano-vllm:latest')
        
        # 构建环境变量
        environment = {
            'MODEL_PATH': self.config.model_path,
            'MODEL_NAME': self.config.model_name,
            'BATCH_SIZE': str(self.config.batch_size),
            'MAX_SEQUENCE_LENGTH': str(self.config.max_sequence_length),
        }
        
        # 构建端口映射
        ports = {f"{self.config.port}/tcp": self.config.port}
        
        # 构建资源限制
        mem_limit = f"{self.config.memory_gb}g"
        cpu_count = self.config.cpu_cores
        
        # GPU支持
        device_requests = []
        if self.config.gpu_count > 0:
            device_requests = [
                docker.types.DeviceRequest(
                    count=self.config.gpu_count,
                    capabilities=[['gpu']]
                )
            ]
        
        # 启动容器
        self.container = self.docker_client.containers.run(
            image_name,
            name=self.config.deployment_name,
            environment=environment,
            ports=ports,
            mem_limit=mem_limit,
            cpu_count=cpu_count,
            device_requests=device_requests,
            detach=True,
            restart_policy={"Name": "unless-stopped"}
        )
        
        logger.info(f"Docker container started: {self.container.id[:12]}")
    
    async def _wait_for_container_ready(self):
        """等待容器就绪"""
        
        max_wait = 120  # 最大等待2分钟
        wait_interval = 5
        
        for _ in range(max_wait // wait_interval):
            try:
                self.container.reload()
                
                if self.container.status == 'running':
                    # 检查健康状态
                    if await self.health_check():
                        return
                
            except Exception as e:
                logger.warning(f"Error checking container status: {e}")
            
            await asyncio.sleep(wait_interval)
        
        raise TimeoutError("Docker container failed to become ready")
    
    async def undeploy(self) -> bool:
        """停止Docker容器"""
        
        logger.info(f"Stopping Docker container: {self.deployment_id}")
        
        try:
            if self.container:
                self.container.stop(timeout=30)
                self.container.remove()
            
            self.status = "stopped"
            
            logger.info("Docker container stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Docker container: {e}")
            return False
    
    async def scale(self, replicas: int) -> bool:
        """Docker单容器不支持扩缩容"""
        
        logger.warning("Docker single container deployment does not support scaling")
        return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        
        if not self.container:
            return {'status': 'not_deployed'}
        
        try:
            self.container.reload()
            
            return {
                'deployment_id': self.deployment_id,
                'container_id': self.container.id,
                'status': self.container.status,
                'config': self.config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return {'error': str(e)}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        
        if not self.container:
            return {}
        
        try:
            stats = self.container.stats(stream=False)
            
            # 解析CPU使用率
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100.0
            
            # 解析内存使用率
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'cpu_utilization': cpu_percent,
                'memory_utilization': memory_percent,
                'memory_used_bytes': memory_usage,
                'memory_limit_bytes': memory_limit,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get container metrics: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """健康检查"""
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:{self.config.port}{self.config.health_check_path}"
                
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class DeploymentManager:
    """部署管理器"""
    
    def __init__(self):
        self.deployments = {}
        self.strategies = {
            DeploymentType.SINGLE_NODE: SingleNodeDeployment,
            DeploymentType.KUBERNETES: KubernetesDeployment,
            DeploymentType.DOCKER_SWARM: DockerDeployment,  # 简化为Docker
        }
        
    async def deploy(self, config: DeploymentConfig) -> str:
        """部署服务"""
        
        logger.info(f"Starting deployment: {config.deployment_name}")
        
        # 获取部署策略
        strategy_class = self.strategies.get(config.deployment_type)
        if not strategy_class:
            raise ValueError(f"Unsupported deployment type: {config.deployment_type}")
        
        # 创建部署策略实例
        strategy = strategy_class(config)
        
        # 执行部署
        deployment_id = await strategy.deploy()
        
        # 保存部署实例
        self.deployments[deployment_id] = strategy
        
        logger.info(f"Deployment completed: {deployment_id}")
        
        return deployment_id
    
    async def undeploy(self, deployment_id: str) -> bool:
        """停止部署"""
        
        strategy = self.deployments.get(deployment_id)
        if not strategy:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        success = await strategy.undeploy()
        
        if success:
            del self.deployments[deployment_id]
        
        return success
    
    async def scale(self, deployment_id: str, replicas: int) -> bool:
        """扩缩容"""
        
        strategy = self.deployments.get(deployment_id)
        if not strategy:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        return await strategy.scale(replicas)
    
    async def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """获取部署状态"""
        
        strategy = self.deployments.get(deployment_id)
        if not strategy:
            return {'error': f'Deployment not found: {deployment_id}'}
        
        return await strategy.get_status()
    
    async def get_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """获取部署指标"""
        
        strategy = self.deployments.get(deployment_id)
        if not strategy:
            return {'error': f'Deployment not found: {deployment_id}'}
        
        return await strategy.get_metrics()
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """列出所有部署"""
        
        deployments = []
        
        for deployment_id, strategy in self.deployments.items():
            status = await strategy.get_status()
            deployments.append(status)
        
        return deployments
    
    async def health_check_all(self) -> Dict[str, bool]:
        """检查所有部署的健康状态"""
        
        results = {}
        
        for deployment_id, strategy in self.deployments.items():
            try:
                results[deployment_id] = await strategy.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {deployment_id}: {e}")
                results[deployment_id] = False
        
        return results
    
    def register_strategy(self, deployment_type: DeploymentType, strategy_class):
        """注册自定义部署策略"""
        
        self.strategies[deployment_type] = strategy_class
        logger.info(f"Registered deployment strategy: {deployment_type.value}")
```

## 🚀 使用示例

### 1. 单机部署

```python
async def single_node_deployment_example():
    # 创建部署管理器
    manager = DeploymentManager()
    
    # 配置单机部署
    config = DeploymentConfig(
        deployment_name="nano-vllm-single",
        deployment_type=DeploymentType.SINGLE_NODE,
        service_type=ServiceType.HTTP_API,
        cpu_cores=8,
        memory_gb=32,
        gpu_count=1,
        gpu_memory_gb=24,
        host="0.0.0.0",
        port=8000,
        model_path="/models/llama-7b",
        model_name="llama-7b",
        batch_size=16,
        max_sequence_length=2048
    )
    
    try:
        # 执行部署
        deployment_id = await manager.deploy(config)
        print(f"Deployment successful: {deployment_id}")
        
        # 等待服务就绪
        await asyncio.sleep(10)
        
        # 检查状态
        status = await manager.get_status(deployment_id)
        print(f"Deployment status: {status}")
        
        # 获取指标
        metrics = await manager.get_metrics(deployment_id)
        print(f"Deployment metrics: {metrics}")
        
        # 健康检查
        health = await manager.health_check_all()
        print(f"Health check: {health}")
        
        # 模拟运行一段时间
        await asyncio.sleep(60)
        
    finally:
        # 停止部署
        await manager.undeploy(deployment_id)
        print("Deployment stopped")

if __name__ == "__main__":
    asyncio.run(single_node_deployment_example())
```

### 2. Kubernetes部署

```python
async def kubernetes_deployment_example():
    # 创建部署管理器
    manager = DeploymentManager()
    
    # 配置Kubernetes部署
    config = DeploymentConfig(
        deployment_name="nano-vllm-k8s",
        deployment_type=DeploymentType.KUBERNETES,
        service_type=ServiceType.HTTP_API,
        cpu_cores=4,
        memory_gb=16,
        gpu_count=1,
        model_path="/models/llama-7b",
        model_name="llama-7b",
        scaling_strategy=ScalingStrategy.AUTO_CPU,
        min_replicas=2,
        max_replicas=10,
        target_cpu_utilization=70.0,
        custom_config={
            'namespace': 'nano-vllm',
            'image': 'nano-vllm:v1.0.0'
        }
    )
    
    try:
        # 执行部署
        deployment_id = await manager.deploy(config)
        print(f"Kubernetes deployment successful: {deployment_id}")
        
        # 等待部署就绪
        await asyncio.sleep(30)
        
        # 检查状态
        status = await manager.get_status(deployment_id)
        print(f"Deployment status: {status}")
        
        # 手动扩容
        await manager.scale(deployment_id, 5)
        print("Scaled to 5 replicas")
        
        # 等待扩容完成
        await asyncio.sleep(60)
        
        # 再次检查状态
        status = await manager.get_status(deployment_id)
        print(f"Updated status: {status}")
        
    finally:
        # 停止部署
        await manager.undeploy(deployment_id)
        print("Kubernetes deployment stopped")

if __name__ == "__main__":
    asyncio.run(kubernetes_deployment_example())
```

### 3. Docker部署

```python
async def docker_deployment_example():
    # 创建部署管理器
    manager = DeploymentManager()
    
    # 配置Docker部署
    config = DeploymentConfig(
        deployment_name="nano-vllm-docker",
        deployment_type=DeploymentType.DOCKER_SWARM,  # 使用Docker策略
        service_type=ServiceType.HTTP_API,
        cpu_cores=4,
        memory_gb=8,
        gpu_count=1,
        port=8080,
        model_path="/app/models/llama-7b",
        model_name="llama-7b",
        custom_config={
            'image': 'nano-vllm:latest'
        }
    )
    
    try:
        # 执行部署
        deployment_id = await manager.deploy(config)
        print(f"Docker deployment successful: {deployment_id}")
        
        # 等待容器就绪
        await asyncio.sleep(20)
        
        # 检查状态
        status = await manager.get_status(deployment_id)
        print(f"Container status: {status}")
        
        # 获取容器指标
        metrics = await manager.get_metrics(deployment_id)
        print(f"Container metrics: {metrics}")
        
        # 持续监控
        for i in range(5):
            await asyncio.sleep(10)
            health = await manager.health_check_all()
            print(f"Health check {i+1}: {health}")
        
    finally:
        # 停止部署
        await manager.undeploy(deployment_id)
        print("Docker deployment stopped")

if __name__ == "__main__":
    asyncio.run(docker_deployment_example())
```

## 🎯 最佳实践

### 1. 部署策略选择
- **单机部署**: 适用于开发测试、小规模应用
- **Kubernetes**: 适用于生产环境、需要自动扩缩容
- **Docker**: 适用于容器化部署、CI/CD集成
- **边缘部署**: 适用于边缘计算、低延迟需求

### 2. 资源配置优化
- **CPU配置**: 根据模型大小和并发需求配置
- **内存配置**: 预留足够内存用于模型加载和推理
- **GPU配置**: 合理分配GPU资源，避免资源浪费
- **存储配置**: 使用高速存储提升模型加载速度

### 3. 监控和告警
- **健康检查**: 定期检查服务健康状态
- **性能监控**: 监控CPU、内存、GPU使用率
- **业务监控**: 监控请求量、响应时间、错误率
- **告警机制**: 设置合理的告警阈值和通知方式

### 4. 安全配置
- **网络安全**: 配置防火墙、VPN、SSL/TLS
- **访问控制**: 实施身份认证和授权机制
- **数据安全**: 加密敏感数据和通信
- **审计日志**: 记录关键操作和访问日志

## 📈 总结

部署策略是nano-vllm从开发到生产的关键环节。通过合理选择部署方案和配置优化，可以确保服务的稳定性、可扩展性和高性能。

关键要点：
1. **多样化部署**: 支持单机、Kubernetes、Docker等多种部署方式
2. **自动化管理**: 提供统一的部署管理接口和自动化工具
3. **弹性扩缩容**: 支持手动和自动扩缩容策略
4. **全面监控**: 集成健康检查、性能监控和告警机制

通过遵循最佳实践和持续优化，可以构建稳定可靠的nano-vllm生产环境。