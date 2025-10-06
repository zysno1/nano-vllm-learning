# 接口设计

## 🎯 接口设计原则

nano-vllm 的接口设计遵循以下核心原则：
- **简洁性**：API 简单易用，降低学习成本
- **一致性**：接口风格统一，命名规范清晰
- **扩展性**：支持未来功能扩展和定制
- **性能优化**：接口设计考虑性能影响
- **向后兼容**：保持API稳定性

## 🌐 HTTP API 接口

### RESTful API 设计

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
import uuid

app = FastAPI(
    title="nano-vllm API",
    description="高性能大语言模型推理服务",
    version="1.0.0"
)

# 请求模型定义
class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str = Field(..., description="输入提示文本", min_length=1)
    max_tokens: int = Field(512, description="最大生成token数", ge=1, le=4096)
    temperature: float = Field(0.7, description="采样温度", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="核采样参数", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-K采样参数", ge=1, le=100)
    repetition_penalty: float = Field(1.0, description="重复惩罚", ge=0.0, le=2.0)
    stop_sequences: Optional[List[str]] = Field(None, description="停止序列")
    stream: bool = Field(False, description="是否流式返回")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "解释什么是人工智能",
                "max_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
        }

class GenerationResponse(BaseModel):
    """生成响应模型"""
    id: str = Field(..., description="请求ID")
    text: str = Field(..., description="生成的文本")
    finish_reason: str = Field(..., description="完成原因")
    usage: Dict[str, int] = Field(..., description="使用统计")
    model: str = Field(..., description="使用的模型")
    created: int = Field(..., description="创建时间戳")

class StreamChunk(BaseModel):
    """流式响应块"""
    id: str = Field(..., description="请求ID")
    delta: str = Field(..., description="增量文本")
    finish_reason: Optional[str] = Field(None, description="完成原因")
    usage: Optional[Dict[str, int]] = Field(None, description="使用统计")

class BatchRequest(BaseModel):
    """批量请求模型"""
    requests: List[GenerationRequest] = Field(..., description="请求列表")
    batch_id: Optional[str] = Field(None, description="批次ID")

class BatchResponse(BaseModel):
    """批量响应模型"""
    batch_id: str = Field(..., description="批次ID")
    responses: List[GenerationResponse] = Field(..., description="响应列表")
    total_time: float = Field(..., description="总处理时间")

# API 端点实现
@app.post("/v1/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    生成文本
    
    生成基于输入提示的文本内容，支持多种采样策略和参数配置。
    """
    try:
        # 生成请求ID
        request_id = str(uuid.uuid4())
        
        # 参数验证
        await validate_generation_request(request)
        
        # 创建推理请求
        inference_request = InferenceRequest(
            id=request_id,
            prompt=request.prompt,
            generation_config=GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                stop_sequences=request.stop_sequences or []
            ),
            user_id=request.user_id
        )
        
        # 执行推理
        result = await inference_engine.generate(inference_request)
        
        # 构造响应
        response = GenerationResponse(
            id=request_id,
            text=result.generated_text,
            finish_reason=result.finish_reason,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens
            },
            model=result.model_name,
            created=int(result.created_time)
        )
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate_text: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/v1/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """
    流式生成文本
    
    以服务器发送事件(SSE)的形式流式返回生成的文本。
    """
    if not request.stream:
        request.stream = True
    
    async def generate_stream():
        try:
            request_id = str(uuid.uuid4())
            
            # 创建流式推理请求
            inference_request = InferenceRequest(
                id=request_id,
                prompt=request.prompt,
                generation_config=GenerationConfig(
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    stop_sequences=request.stop_sequences or []
                ),
                stream=True,
                user_id=request.user_id
            )
            
            # 流式生成
            async for chunk in inference_engine.generate_stream(inference_request):
                stream_chunk = StreamChunk(
                    id=request_id,
                    delta=chunk.delta_text,
                    finish_reason=chunk.finish_reason,
                    usage=chunk.usage if chunk.is_final else None
                )
                
                yield f"data: {stream_chunk.json()}\n\n"
            
            # 发送结束标记
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "inference_error"
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/v1/batch/generate", response_model=BatchResponse)
async def batch_generate(request: BatchRequest):
    """
    批量生成文本
    
    同时处理多个生成请求，提高吞吐量。
    """
    try:
        batch_id = request.batch_id or str(uuid.uuid4())
        start_time = time.time()
        
        # 创建批量推理请求
        inference_requests = []
        for i, gen_request in enumerate(request.requests):
            inference_request = InferenceRequest(
                id=f"{batch_id}_{i}",
                prompt=gen_request.prompt,
                generation_config=GenerationConfig(
                    max_tokens=gen_request.max_tokens,
                    temperature=gen_request.temperature,
                    top_p=gen_request.top_p,
                    top_k=gen_request.top_k,
                    repetition_penalty=gen_request.repetition_penalty,
                    stop_sequences=gen_request.stop_sequences or []
                ),
                user_id=gen_request.user_id
            )
            inference_requests.append(inference_request)
        
        # 执行批量推理
        results = await inference_engine.batch_generate(inference_requests)
        
        # 构造响应
        responses = []
        for result in results:
            response = GenerationResponse(
                id=result.request_id,
                text=result.generated_text,
                finish_reason=result.finish_reason,
                usage={
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens
                },
                model=result.model_name,
                created=int(result.created_time)
            )
            responses.append(response)
        
        total_time = time.time() - start_time
        
        return BatchResponse(
            batch_id=batch_id,
            responses=responses,
            total_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch_generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 模型管理接口
@app.get("/v1/models")
async def list_models():
    """获取可用模型列表"""
    models = await model_manager.list_available_models()
    return {
        "data": [
            {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "max_context_length": model.max_context_length,
                "created": model.created_time
            }
            for model in models
        ]
    }

@app.post("/v1/models/{model_id}/load")
async def load_model(model_id: str, background_tasks: BackgroundTasks):
    """加载模型"""
    try:
        # 异步加载模型
        background_tasks.add_task(model_manager.load_model, model_id)
        
        return {
            "message": f"Model {model_id} loading started",
            "status": "loading"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/models/{model_id}")
async def unload_model(model_id: str):
    """卸载模型"""
    try:
        await model_manager.unload_model(model_id)
        return {
            "message": f"Model {model_id} unloaded successfully",
            "status": "unloaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 系统状态接口
@app.get("/v1/status")
async def get_system_status():
    """获取系统状态"""
    status = await system_monitor.get_status()
    return {
        "status": "healthy" if status.is_healthy else "unhealthy",
        "version": "1.0.0",
        "uptime": status.uptime,
        "gpu_memory": {
            "used": status.gpu_memory_used,
            "total": status.gpu_memory_total,
            "utilization": status.gpu_memory_utilization
        },
        "active_requests": status.active_requests,
        "total_requests": status.total_requests,
        "average_latency": status.average_latency
    }

@app.get("/v1/metrics")
async def get_metrics():
    """获取详细指标"""
    metrics = await metrics_collector.get_all_metrics()
    return metrics

# 请求验证
async def validate_generation_request(request: GenerationRequest):
    """验证生成请求"""
    # 检查提示长度
    if len(request.prompt) > MAX_PROMPT_LENGTH:
        raise ValidationError(f"Prompt too long: {len(request.prompt)} > {MAX_PROMPT_LENGTH}")
    
    # 检查参数范围
    if request.temperature < 0 or request.temperature > 2:
        raise ValidationError(f"Invalid temperature: {request.temperature}")
    
    if request.top_p < 0 or request.top_p > 1:
        raise ValidationError(f"Invalid top_p: {request.top_p}")
    
    # 检查用户权限
    if request.user_id:
        await validate_user_permissions(request.user_id)

async def validate_user_permissions(user_id: str):
    """验证用户权限"""
    # 实现用户权限验证逻辑
    pass
```

### WebSocket 接口

```python
from fastapi import WebSocket, WebSocketDisconnect
import json

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            "connected_at": time.time(),
            "requests_count": 0
        }
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.connection_metadata[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        """发送消息给指定客户端"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: dict):
        """广播消息给所有客户端"""
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")

websocket_manager = WebSocketManager()

@app.websocket("/v1/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket端点"""
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理不同类型的消息
            await handle_websocket_message(client_id, message)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect(client_id)

async def handle_websocket_message(client_id: str, message: dict):
    """处理WebSocket消息"""
    message_type = message.get("type")
    
    if message_type == "generate":
        await handle_websocket_generate(client_id, message)
    elif message_type == "cancel":
        await handle_websocket_cancel(client_id, message)
    elif message_type == "ping":
        await websocket_manager.send_message(client_id, {"type": "pong"})
    else:
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })

async def handle_websocket_generate(client_id: str, message: dict):
    """处理WebSocket生成请求"""
    try:
        request_id = message.get("request_id", str(uuid.uuid4()))
        
        # 创建推理请求
        inference_request = InferenceRequest(
            id=request_id,
            prompt=message["prompt"],
            generation_config=GenerationConfig(**message.get("config", {})),
            stream=True
        )
        
        # 发送开始消息
        await websocket_manager.send_message(client_id, {
            "type": "generation_started",
            "request_id": request_id
        })
        
        # 流式生成
        async for chunk in inference_engine.generate_stream(inference_request):
            await websocket_manager.send_message(client_id, {
                "type": "generation_chunk",
                "request_id": request_id,
                "delta": chunk.delta_text,
                "finish_reason": chunk.finish_reason
            })
            
            if chunk.is_final:
                break
        
        # 发送完成消息
        await websocket_manager.send_message(client_id, {
            "type": "generation_completed",
            "request_id": request_id
        })
        
    except Exception as e:
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "request_id": message.get("request_id"),
            "message": str(e)
        })
```

## 🐍 Python SDK

### 客户端SDK设计

```python
import asyncio
import aiohttp
import json
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

class FinishReason(Enum):
    """完成原因枚举"""
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"

@dataclass
class GenerationConfig:
    """生成配置"""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

@dataclass
class GenerationResult:
    """生成结果"""
    id: str
    text: str
    finish_reason: FinishReason
    usage: Dict[str, int]
    model: str
    created: int

@dataclass
class StreamChunk:
    """流式响应块"""
    id: str
    delta: str
    finish_reason: Optional[FinishReason] = None
    usage: Optional[Dict[str, int]] = None

class NanoVLLMClient:
    """nano-vllm Python客户端"""
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 timeout: int = 300):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """确保会话存在"""
        if not self.session:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
    
    async def generate(self, 
                      prompt: str,
                      config: Optional[GenerationConfig] = None,
                      user_id: Optional[str] = None) -> GenerationResult:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            config: 生成配置
            user_id: 用户ID
            
        Returns:
            生成结果
        """
        await self._ensure_session()
        
        if config is None:
            config = GenerationConfig()
        
        request_data = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "stop_sequences": config.stop_sequences,
            "stream": False
        }
        
        if user_id:
            request_data["user_id"] = user_id
        
        async with self.session.post(
            f"{self.base_url}/v1/generate",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API request failed: {response.status} - {error_text}")
            
            result_data = await response.json()
            
            return GenerationResult(
                id=result_data["id"],
                text=result_data["text"],
                finish_reason=FinishReason(result_data["finish_reason"]),
                usage=result_data["usage"],
                model=result_data["model"],
                created=result_data["created"]
            )
    
    async def generate_stream(self,
                            prompt: str,
                            config: Optional[GenerationConfig] = None,
                            user_id: Optional[str] = None) -> AsyncGenerator[StreamChunk, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入提示
            config: 生成配置
            user_id: 用户ID
            
        Yields:
            流式响应块
        """
        await self._ensure_session()
        
        if config is None:
            config = GenerationConfig()
        
        config.stream = True
        
        request_data = {
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "stop_sequences": config.stop_sequences,
            "stream": True
        }
        
        if user_id:
            request_data["user_id"] = user_id
        
        async with self.session.post(
            f"{self.base_url}/v1/generate/stream",
            json=request_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Stream request failed: {response.status} - {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    data = line[6:]  # 移除 'data: ' 前缀
                    
                    if data == '[DONE]':
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        
                        if 'error' in chunk_data:
                            raise Exception(chunk_data['error']['message'])
                        
                        yield StreamChunk(
                            id=chunk_data["id"],
                            delta=chunk_data["delta"],
                            finish_reason=FinishReason(chunk_data["finish_reason"]) if chunk_data.get("finish_reason") else None,
                            usage=chunk_data.get("usage")
                        )
                        
                    except json.JSONDecodeError:
                        continue  # 跳过无效的JSON行
    
    async def batch_generate(self,
                           prompts: List[str],
                           config: Optional[GenerationConfig] = None,
                           user_id: Optional[str] = None) -> List[GenerationResult]:
        """
        批量生成文本
        
        Args:
            prompts: 提示列表
            config: 生成配置
            user_id: 用户ID
            
        Returns:
            生成结果列表
        """
        await self._ensure_session()
        
        if config is None:
            config = GenerationConfig()
        
        requests = []
        for prompt in prompts:
            request_data = {
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "stop_sequences": config.stop_sequences,
                "stream": False
            }
            
            if user_id:
                request_data["user_id"] = user_id
            
            requests.append(request_data)
        
        batch_data = {"requests": requests}
        
        async with self.session.post(
            f"{self.base_url}/v1/batch/generate",
            json=batch_data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Batch request failed: {response.status} - {error_text}")
            
            result_data = await response.json()
            
            results = []
            for response_data in result_data["responses"]:
                result = GenerationResult(
                    id=response_data["id"],
                    text=response_data["text"],
                    finish_reason=FinishReason(response_data["finish_reason"]),
                    usage=response_data["usage"],
                    model=response_data["model"],
                    created=response_data["created"]
                )
                results.append(result)
            
            return results
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        await self._ensure_session()
        
        async with self.session.get(f"{self.base_url}/v1/models") as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"List models failed: {response.status} - {error_text}")
            
            result = await response.json()
            return result["data"]
    
    async def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        await self._ensure_session()
        
        async with self.session.get(f"{self.base_url}/v1/status") as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Get status failed: {response.status} - {error_text}")
            
            return await response.json()

# 使用示例
async def example_usage():
    """使用示例"""
    async with NanoVLLMClient("http://localhost:8000") as client:
        # 单次生成
        result = await client.generate(
            prompt="解释什么是人工智能",
            config=GenerationConfig(
                max_tokens=256,
                temperature=0.7
            )
        )
        print(f"生成结果: {result.text}")
        
        # 流式生成
        print("流式生成:")
        async for chunk in client.generate_stream(
            prompt="写一首关于春天的诗",
            config=GenerationConfig(max_tokens=200)
        ):
            print(chunk.delta, end='', flush=True)
        print()
        
        # 批量生成
        prompts = [
            "什么是机器学习？",
            "解释深度学习的原理",
            "人工智能的应用领域有哪些？"
        ]
        
        results = await client.batch_generate(prompts)
        for i, result in enumerate(results):
            print(f"问题 {i+1}: {result.text[:100]}...")

# 同步包装器
class NanoVLLMSyncClient:
    """同步客户端包装器"""
    
    def __init__(self, *args, **kwargs):
        self.async_client = NanoVLLMClient(*args, **kwargs)
        self.loop = None
    
    def _run_async(self, coro):
        """运行异步协程"""
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
        return self.loop.run_until_complete(coro)
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None, 
                user_id: Optional[str] = None) -> GenerationResult:
        """同步生成文本"""
        async def _generate():
            async with self.async_client as client:
                return await client.generate(prompt, config, user_id)
        
        return self._run_async(_generate())
    
    def batch_generate(self, prompts: List[str], 
                      config: Optional[GenerationConfig] = None,
                      user_id: Optional[str] = None) -> List[GenerationResult]:
        """同步批量生成"""
        async def _batch_generate():
            async with self.async_client as client:
                return await client.batch_generate(prompts, config, user_id)
        
        return self._run_async(_batch_generate())
    
    def list_models(self) -> List[Dict[str, Any]]:
        """同步获取模型列表"""
        async def _list_models():
            async with self.async_client as client:
                return await client.list_models()
        
        return self._run_async(_list_models())
```

## 🔌 插件接口

### 插件系统设计

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import importlib
import inspect

class Plugin(ABC):
    """插件基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.version = getattr(self, 'VERSION', '1.0.0')
        self.enabled = True
    
    @abstractmethod
    async def initialize(self):
        """初始化插件"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """清理插件资源"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'config': self.config
        }

class PreprocessorPlugin(Plugin):
    """预处理器插件"""
    
    @abstractmethod
    async def preprocess(self, text: str, metadata: Dict[str, Any]) -> str:
        """预处理文本"""
        pass

class PostprocessorPlugin(Plugin):
    """后处理器插件"""
    
    @abstractmethod
    async def postprocess(self, text: str, metadata: Dict[str, Any]) -> str:
        """后处理文本"""
        pass

class SamplerPlugin(Plugin):
    """采样器插件"""
    
    @abstractmethod
    async def sample(self, logits: torch.Tensor, 
                    generation_state: Dict[str, Any]) -> torch.Tensor:
        """自定义采样"""
        pass

class MetricsPlugin(Plugin):
    """指标收集插件"""
    
    @abstractmethod
    async def collect_metrics(self, request_data: Dict[str, Any], 
                            response_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集指标"""
        pass

class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_hooks: Dict[str, List[Plugin]] = {
            'preprocess': [],
            'postprocess': [],
            'sample': [],
            'metrics': []
        }
    
    async def load_plugin(self, plugin_path: str, config: Dict[str, Any]):
        """加载插件"""
        try:
            # 动态导入插件模块
            module = importlib.import_module(plugin_path)
            
            # 查找插件类
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, Plugin) and 
                    obj != Plugin):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                raise ValueError(f"No plugin class found in {plugin_path}")
            
            # 实例化插件
            plugin_class = plugin_classes[0]  # 使用第一个找到的插件类
            plugin = plugin_class(config)
            
            # 初始化插件
            await plugin.initialize()
            
            # 注册插件
            self.plugins[plugin.name] = plugin
            
            # 根据插件类型注册到相应的钩子
            if isinstance(plugin, PreprocessorPlugin):
                self.plugin_hooks['preprocess'].append(plugin)
            if isinstance(plugin, PostprocessorPlugin):
                self.plugin_hooks['postprocess'].append(plugin)
            if isinstance(plugin, SamplerPlugin):
                self.plugin_hooks['sample'].append(plugin)
            if isinstance(plugin, MetricsPlugin):
                self.plugin_hooks['metrics'].append(plugin)
            
            logger.info(f"Plugin {plugin.name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            raise
    
    async def unload_plugin(self, plugin_name: str):
        """卸载插件"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        plugin = self.plugins[plugin_name]
        
        # 清理插件资源
        await plugin.cleanup()
        
        # 从钩子中移除
        for hook_list in self.plugin_hooks.values():
            if plugin in hook_list:
                hook_list.remove(plugin)
        
        # 从插件字典中移除
        del self.plugins[plugin_name]
        
        logger.info(f"Plugin {plugin_name} unloaded successfully")
    
    async def execute_preprocess_hooks(self, text: str, 
                                     metadata: Dict[str, Any]) -> str:
        """执行预处理钩子"""
        result = text
        for plugin in self.plugin_hooks['preprocess']:
            if plugin.enabled:
                result = await plugin.preprocess(result, metadata)
        return result
    
    async def execute_postprocess_hooks(self, text: str,
                                      metadata: Dict[str, Any]) -> str:
        """执行后处理钩子"""
        result = text
        for plugin in self.plugin_hooks['postprocess']:
            if plugin.enabled:
                result = await plugin.postprocess(result, metadata)
        return result
    
    async def execute_sample_hooks(self, logits: torch.Tensor,
                                 generation_state: Dict[str, Any]) -> torch.Tensor:
        """执行采样钩子"""
        result = logits
        for plugin in self.plugin_hooks['sample']:
            if plugin.enabled:
                result = await plugin.sample(result, generation_state)
        return result
    
    async def execute_metrics_hooks(self, request_data: Dict[str, Any],
                                  response_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行指标收集钩子"""
        metrics = {}
        for plugin in self.plugin_hooks['metrics']:
            if plugin.enabled:
                plugin_metrics = await plugin.collect_metrics(request_data, response_data)
                metrics[plugin.name] = plugin_metrics
        return metrics
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有插件"""
        return [plugin.get_info() for plugin in self.plugins.values()]
    
    def enable_plugin(self, plugin_name: str):
        """启用插件"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
    
    def disable_plugin(self, plugin_name: str):
        """禁用插件"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False

# 示例插件实现
class TextNormalizationPlugin(PreprocessorPlugin):
    """文本标准化插件"""
    
    VERSION = "1.0.0"
    
    async def initialize(self):
        """初始化插件"""
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.remove_extra_spaces = self.config.get('remove_extra_spaces', True)
        logger.info("TextNormalizationPlugin initialized")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("TextNormalizationPlugin cleaned up")
    
    async def preprocess(self, text: str, metadata: Dict[str, Any]) -> str:
        """预处理文本"""
        result = text
        
        if self.normalize_unicode:
            import unicodedata
            result = unicodedata.normalize('NFKC', result)
        
        if self.remove_extra_spaces:
            import re
            result = re.sub(r'\s+', ' ', result).strip()
        
        return result

class ProfanityFilterPlugin(PostprocessorPlugin):
    """敏感词过滤插件"""
    
    VERSION = "1.0.0"
    
    async def initialize(self):
        """初始化插件"""
        self.bad_words = set(self.config.get('bad_words', []))
        self.replacement = self.config.get('replacement', '***')
        logger.info("ProfanityFilterPlugin initialized")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("ProfanityFilterPlugin cleaned up")
    
    async def postprocess(self, text: str, metadata: Dict[str, Any]) -> str:
        """后处理文本"""
        result = text
        for bad_word in self.bad_words:
            result = result.replace(bad_word, self.replacement)
        return result

class CustomMetricsPlugin(MetricsPlugin):
    """自定义指标插件"""
    
    VERSION = "1.0.0"
    
    async def initialize(self):
        """初始化插件"""
        self.metrics_storage = {}
        logger.info("CustomMetricsPlugin initialized")
    
    async def cleanup(self):
        """清理资源"""
        logger.info("CustomMetricsPlugin cleaned up")
    
    async def collect_metrics(self, request_data: Dict[str, Any],
                            response_data: Dict[str, Any]) -> Dict[str, Any]:
        """收集自定义指标"""
        metrics = {
            'request_length': len(request_data.get('prompt', '')),
            'response_length': len(response_data.get('text', '')),
            'processing_time': response_data.get('processing_time', 0),
            'tokens_per_second': response_data.get('tokens_per_second', 0)
        }
        
        return metrics
```

## 🔧 配置接口

### 配置管理系统

```python
from pydantic import BaseSettings, Field
from typing import Optional, List, Dict, Any
import yaml
import json
import os

class ServerConfig(BaseSettings):
    """服务器配置"""
    host: str = Field("0.0.0.0", description="服务器主机")
    port: int = Field(8000, description="服务器端口")
    workers: int = Field(1, description="工作进程数")
    max_concurrent_requests: int = Field(100, description="最大并发请求数")
    request_timeout: int = Field(300, description="请求超时时间(秒)")
    
    class Config:
        env_prefix = "NANO_VLLM_SERVER_"

class ModelConfig(BaseSettings):
    """模型配置"""
    model_path: str = Field(..., description="模型路径")
    model_name: str = Field("default", description="模型名称")
    max_context_length: int = Field(4096, description="最大上下文长度")
    tensor_parallel_size: int = Field(1, description="张量并行大小")
    dtype: str = Field("float16", description="数据类型")
    quantization: Optional[str] = Field(None, description="量化方法")
    
    class Config:
        env_prefix = "NANO_VLLM_MODEL_"

class GenerationConfig(BaseSettings):
    """生成配置"""
    max_tokens: int = Field(512, description="最大生成token数")
    temperature: float = Field(0.7, description="采样温度")
    top_p: float = Field(0.9, description="核采样参数")
    top_k: int = Field(50, description="Top-K采样参数")
    repetition_penalty: float = Field(1.0, description="重复惩罚")
    stop_sequences: List[str] = Field([], description="停止序列")
    
    class Config:
        env_prefix = "NANO_VLLM_GENERATION_"

class MemoryConfig(BaseSettings):
    """内存配置"""
    gpu_memory_utilization: float = Field(0.9, description="GPU内存利用率")
    cpu_memory_gb: Optional[int] = Field(None, description="CPU内存限制(GB)")
    kv_cache_dtype: str = Field("auto", description="KV Cache数据类型")
    enable_prefix_caching: bool = Field(False, description="启用前缀缓存")
    
    class Config:
        env_prefix = "NANO_VLLM_MEMORY_"

class LoggingConfig(BaseSettings):
    """日志配置"""
    level: str = Field("INFO", description="日志级别")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")
    file: Optional[str] = Field(None, description="日志文件路径")
    max_file_size: str = Field("100MB", description="日志文件最大大小")
    backup_count: int = Field(5, description="日志文件备份数量")
    
    class Config:
        env_prefix = "NANO_VLLM_LOGGING_"

class NanoVLLMConfig(BaseSettings):
    """nano-vllm 主配置"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # 插件配置
    plugins: Dict[str, Dict[str, Any]] = Field({}, description="插件配置")
    
    # 安全配置
    api_key: Optional[str] = Field(None, description="API密钥")
    cors_origins: List[str] = Field(["*"], description="CORS允许的源")
    rate_limit: Dict[str, int] = Field({"requests_per_minute": 60}, description="速率限制")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config: Optional[NanoVLLMConfig] = None
        self.watchers = []
    
    def load_config(self, config_path: Optional[str] = None) -> NanoVLLMConfig:
        """加载配置"""
        if config_path:
            self.config_path = config_path
        
        if self.config_path and os.path.exists(self.config_path):
            # 从文件加载配置
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            # 合并环境变量
            self.config = NanoVLLMConfig(**config_data)
        else:
            # 仅从环境变量加载
            self.config = NanoVLLMConfig()
        
        return self.config
    
    def save_config(self, config_path: Optional[str] = None):
        """保存配置"""
        if not self.config:
            raise ValueError("No config to save")
        
        save_path = config_path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")
        
        config_dict = self.config.dict()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif save_path.endswith('.json'):
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {save_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        if not self.config:
            self.load_config()
        
        # 深度更新配置
        self._deep_update(self.config.dict(), updates)
        
        # 重新创建配置对象
        self.config = NanoVLLMConfig(**self.config.dict())
        
        # 通知观察者
        self._notify_watchers()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def watch_config(self, callback):
        """监听配置变化"""
        self.watchers.append(callback)
    
    def _notify_watchers(self):
        """通知配置观察者"""
        for callback in self.watchers:
            try:
                callback(self.config)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if not self.config:
            errors.append("No config loaded")
            return errors
        
        # 验证模型路径
        if not os.path.exists(self.config.model.model_path):
            errors.append(f"Model path does not exist: {self.config.model.model_path}")
        
        # 验证端口范围
        if not (1 <= self.config.server.port <= 65535):
            errors.append(f"Invalid port number: {self.config.server.port}")
        
        # 验证内存配置
        if not (0.1 <= self.config.memory.gpu_memory_utilization <= 1.0):
            errors.append(f"Invalid GPU memory utilization: {self.config.memory.gpu_memory_utilization}")
        
        # 验证生成参数
        if not (0.0 <= self.config.generation.temperature <= 2.0):
            errors.append(f"Invalid temperature: {self.config.generation.temperature}")
        
        if not (0.0 <= self.config.generation.top_p <= 1.0):
            errors.append(f"Invalid top_p: {self.config.generation.top_p}")
        
        return errors

# 配置文件示例
EXAMPLE_CONFIG_YAML = """
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  max_concurrent_requests: 100
  request_timeout: 300

model:
  model_path: "/path/to/model"
  model_name: "llama-7b"
  max_context_length: 4096
  tensor_parallel_size: 1
  dtype: "float16"
  quantization: null

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.0
  stop_sequences: []

memory:
  gpu_memory_utilization: 0.9
  cpu_memory_gb: null
  kv_cache_dtype: "auto"
  enable_prefix_caching: false

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null
  max_file_size: "100MB"
  backup_count: 5

plugins:
  text_normalization:
    enabled: true
    normalize_unicode: true
    remove_extra_spaces: true
  
  profanity_filter:
    enabled: false
    bad_words: ["badword1", "badword2"]
    replacement: "***"

api_key: null
cors_origins: ["*"]
rate_limit:
  requests_per_minute: 60
"""

# 使用示例
def example_config_usage():
    """配置使用示例"""
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 加载配置
    config = config_manager.load_config("config.yaml")
    
    # 验证配置
    errors = config_manager.validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # 使用配置
    print(f"Server will run on {config.server.host}:{config.server.port}")
    print(f"Model: {config.model.model_name} at {config.model.model_path}")
    
    # 更新配置
    config_manager.update_config({
        "server": {"port": 8080},
        "generation": {"temperature": 0.8}
    })
    
    # 保存配置
    config_manager.save_config("updated_config.yaml")
```

---

*通过精心设计的接口系统，nano-vllm 提供了灵活、易用、可扩展的API，支持多种使用场景和集成需求。无论是简单的文本生成还是复杂的批量处理，都能通过统一的接口实现。*