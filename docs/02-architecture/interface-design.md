# 接口设计

## 🎯 接口设计原则

nano-vllm 的接口设计遵循以下核心原则：

<!-- 
设计思想：接口设计原则
1. 简洁性原则：
   - 减少认知负担：API设计应该直观易懂，用户无需深入了解内部实现
   - 最小化参数：只暴露必要的配置参数，隐藏复杂的内部细节
   - 合理默认值：为所有参数提供合理的默认值，降低使用门槛
   - 一致的命名：使用统一的命名规范，避免歧义和混淆

2. 一致性原则：
   - 接口风格统一：所有API端点遵循相同的设计模式和响应格式
   - 错误处理一致：统一的错误码和错误信息格式
   - 数据格式统一：请求和响应使用一致的数据结构和字段命名
   - 版本管理：通过版本号保证接口的向后兼容性

3. 扩展性原则：
   - 模块化设计：接口设计支持功能模块的独立扩展
   - 插件机制：支持第三方插件和自定义扩展
   - 配置灵活性：提供丰富的配置选项满足不同场景需求
   - 协议无关：支持多种通信协议（HTTP、gRPC、WebSocket等）

4. 性能优化原则：
   - 批处理支持：支持批量请求处理，提高吞吐量
   - 流式处理：支持流式响应，降低首字节延迟
   - 缓存机制：合理使用缓存减少重复计算
   - 异步处理：采用异步I/O提高并发性能

5. 向后兼容原则：
   - 版本控制：通过API版本号管理接口变更
   - 渐进式升级：新功能以可选方式添加，不破坏现有功能
   - 废弃策略：提供明确的废弃时间表和迁移指南
   - 文档维护：保持API文档的准确性和完整性
-->

- **简洁性**：API 简单易用，降低学习成本
- **一致性**：接口风格统一，命名规范清晰
- **扩展性**：支持未来功能扩展和定制
- **性能优化**：接口设计考虑性能影响
- **向后兼容**：保持API稳定性

## 🌐 HTTP API 接口

### RESTful API 设计

<!-- 
设计思想：RESTful API架构
1. 资源导向设计：
   - 将API设计为对资源的操作，而不是对动作的调用
   - 使用HTTP动词（GET、POST、PUT、DELETE）表示操作类型
   - URL路径表示资源层次结构，清晰直观
   - 状态码语义化，便于客户端处理

2. FastAPI框架选择：
   - 自动文档生成：基于类型注解自动生成OpenAPI文档
   - 类型安全：编译时类型检查，减少运行时错误
   - 高性能：基于Starlette和Pydantic，性能优异
   - 异步支持：原生支持异步处理，适合高并发场景

3. 数据模型设计：
   - Pydantic模型：提供数据验证、序列化和文档生成
   - 字段约束：通过Field定义参数范围和描述信息
   - 示例数据：提供清晰的使用示例，降低学习成本
   - 嵌套模型：支持复杂数据结构的层次化组织
-->

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import asyncio
import uuid

# FastAPI应用初始化
# 设计思想：应用配置集中化
# - title/description：提供清晰的API标识和说明
# - version：支持API版本管理和兼容性控制
# - 元数据集中：便于API文档生成和服务发现
app = FastAPI(
    title="nano-vllm API",
    description="高性能大语言模型推理服务",
    version="1.0.0"
)

# 请求模型定义
class GenerationRequest(BaseModel):
    """
    生成请求模型
    
    设计思想：
    1. 参数完整性：涵盖文本生成的所有关键参数
    2. 类型安全：使用强类型定义，避免参数错误
    3. 参数验证：通过Field约束确保参数合法性
    4. 文档友好：详细的字段描述便于API文档生成
    5. 示例驱动：提供完整的使用示例
    """
    # 核心参数：输入文本
    # 设计思想：必需参数，最小长度约束防止空输入
    prompt: str = Field(..., description="输入提示文本", min_length=1)
    
    # 生成控制参数
    # 设计思想：合理的默认值和范围约束，平衡质量和性能
    max_tokens: int = Field(512, description="最大生成token数", ge=1, le=4096)
    temperature: float = Field(0.7, description="采样温度", ge=0.0, le=2.0)
    top_p: float = Field(0.9, description="核采样参数", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Top-K采样参数", ge=1, le=100)
    repetition_penalty: float = Field(1.0, description="重复惩罚", ge=0.0, le=2.0)
    
    # 可选控制参数
    # 设计思想：可选参数提供高级功能，不影响基础使用
    stop_sequences: Optional[List[str]] = Field(None, description="停止序列")
    stream: bool = Field(False, description="是否流式返回")
    user_id: Optional[str] = Field(None, description="用户ID")
    
    # 配置示例
    # 设计思想：提供典型使用场景的参数配置示例
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
    """
    生成响应模型
    
    设计思想：
    1. 完整信息：包含生成结果和元数据信息
    2. 标准化：遵循OpenAI API响应格式，便于集成
    3. 可追踪：包含请求ID支持请求追踪和调试
    4. 统计信息：提供使用统计便于计费和监控
    """
    id: str = Field(..., description="请求ID")
    text: str = Field(..., description="生成的文本")
    finish_reason: str = Field(..., description="完成原因")
    usage: Dict[str, int] = Field(..., description="使用统计")
    model: str = Field(..., description="使用的模型")
    created: int = Field(..., description="创建时间戳")

class StreamChunk(BaseModel):
    """
    流式响应块
    
    设计思想：
    1. 增量传输：只传输新增内容，减少网络开销
    2. 状态同步：通过finish_reason同步生成状态
    3. 统计延迟：usage信息在最后一个chunk中提供
    4. 兼容性：与非流式响应保持字段一致性
    """
    id: str = Field(..., description="请求ID")
    delta: str = Field(..., description="增量文本")
    finish_reason: Optional[str] = Field(None, description="完成原因")
    usage: Optional[Dict[str, int]] = Field(None, description="使用统计")

class BatchRequest(BaseModel):
    """
    批量请求模型
    
    设计思想：
    1. 批处理优化：支持多个请求批量处理，提高吞吐量
    2. 请求复用：复用GenerationRequest模型，保持一致性
    3. 批次管理：支持批次ID便于批量操作管理
    4. 灵活性：支持不同参数的请求混合批处理
    """
    requests: List[GenerationRequest] = Field(..., description="请求列表")
    batch_id: Optional[str] = Field(None, description="批次ID")

class BatchResponse(BaseModel):
    """
    批量响应模型
    
    设计思想：
    1. 结果对应：响应列表与请求列表一一对应
    2. 批次标识：包含批次ID便于结果关联
    3. 性能统计：提供总处理时间便于性能分析
    4. 错误处理：支持部分成功的批处理结果
    """
    batch_id: str = Field(..., description="批次ID")
    responses: List[GenerationResponse] = Field(..., description="响应列表")
    total_time: float = Field(..., description="总处理时间")

# API 端点实现
@app.post("/v1/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """
    生成文本
    
    生成基于输入提示的文本内容，支持多种采样策略和参数配置。
    
    设计思想：
    1. 异步处理：使用async/await支持高并发请求
    2. 参数验证：多层次验证确保请求参数合法性
    3. 错误处理：统一的异常处理和错误响应格式
    4. 请求追踪：生成唯一ID支持请求生命周期追踪
    5. 配置转换：将API参数转换为内部推理配置
    """
    try:
        # 生成请求ID
        # 设计思想：唯一标识符便于请求追踪和调试
        request_id = str(uuid.uuid4())
        
        # 参数验证
        # 设计思想：多层验证确保参数合法性和业务逻辑正确性
        await validate_generation_request(request)
        
        # 创建推理请求
        # 设计思想：将WebSocket消息转换为内部推理请求格式
        inference_request = InferenceRequest(
            id=request_id,
            prompt=message["prompt"],
            generation_config=GenerationConfig(**message.get("config", {})),
            stream=True  # WebSocket默认启用流式模式
        )
        
        # 发送开始消息
        # 设计思想：通知客户端生成任务已开始，便于状态跟踪
        await websocket_manager.send_message(client_id, {
            "type": "generation_started",
            "request_id": request_id
        })
        
        # 流式生成
        # 设计思想：实时传输生成结果，提供即时反馈
        async for chunk in inference_engine.generate_stream(inference_request):
            # 发送生成片段
            # 设计思想：每个chunk包含增量文本和状态信息
            await websocket_manager.send_message(client_id, {
                "type": "generation_chunk",
                "request_id": request_id,
                "delta": chunk.delta_text,
                "finish_reason": chunk.finish_reason
            })
            
            # 检查是否完成
            # 设计思想：根据finish_reason判断生成是否结束
            if chunk.is_final:
                break
        
        # 发送完成消息
        # 设计思想：明确通知客户端生成任务已完成
        await websocket_manager.send_message(client_id, {
            "type": "generation_completed",
            "request_id": request_id
        })
        
    except Exception as e:
        # 异常处理
        # 设计思想：将异常信息传递给客户端，便于错误处理
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "request_id": message.get("request_id"),
            "message": str(e)
        })

async def handle_websocket_cancel(client_id: str, message: dict):
    """
    处理WebSocket取消请求
    
    设计思想：
    1. 请求取消：支持客户端主动取消正在进行的生成任务
    2. 资源释放：及时释放被取消任务占用的计算资源
    3. 状态同步：通知客户端取消操作的结果
    4. 错误处理：处理取消操作可能出现的异常
    """
    try:
        # 获取要取消的请求ID
        # 设计思想：通过请求ID精确定位要取消的任务
        request_id = message.get("request_id")
        if not request_id:
            # 请求ID缺失处理
            # 设计思想：返回明确的错误信息便于客户端调试
            await websocket_manager.send_message(client_id, {
                "type": "error",
                "message": "Missing request_id for cancel operation"
            })
            return
        
        # 执行取消操作
        # 设计思想：委托给推理引擎处理具体的取消逻辑
        success = await inference_engine.cancel_request(request_id)
        
        if success:
            # 取消成功
            # 设计思想：通知客户端取消操作已成功执行
            await websocket_manager.send_message(client_id, {
                "type": "generation_cancelled",
                "request_id": request_id
            })
        else:
            # 取消失败（可能请求已完成或不存在）
            # 设计思想：提供具体的失败原因便于客户端处理
            await websocket_manager.send_message(client_id, {
                "type": "error",
                "request_id": request_id,
                "message": "Failed to cancel request (may already be completed)"
            })
            
    except Exception as e:
        # 异常处理
        # 设计思想：记录异常并通知客户端取消操作失败
        logger.error(f"Error cancelling request: {e}")
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "request_id": message.get("request_id"),
            "message": f"Cancel operation failed: {str(e)}"
        })

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
    """
    完成原因枚举
    
    设计思想：
    1. 状态标准化：统一定义生成完成的各种原因
    2. 类型安全：使用枚举避免字符串拼写错误
    3. 扩展性：便于添加新的完成原因类型
    """
    STOP = "stop"      # 正常停止（遇到停止序列）
    LENGTH = "length"  # 达到最大长度限制
    ERROR = "error"    # 生成过程中出现错误

@dataclass
class GenerationConfig:
    """
    生成配置
    
    设计思想：
    1. 参数封装：将所有生成参数封装在一个类中
    2. 默认值：提供合理的默认参数值
    3. 类型提示：明确每个参数的类型
    4. 可选参数：支持部分参数的自定义配置
    """
    max_tokens: int = 512                           # 最大生成token数
    temperature: float = 0.7                        # 温度参数，控制随机性
    top_p: float = 0.9                             # 核采样参数
    top_k: int = 50                                # Top-K采样参数
    repetition_penalty: float = 1.0                # 重复惩罚系数
    stop_sequences: Optional[List[str]] = None     # 停止序列
    stream: bool = False                           # 是否启用流式输出

@dataclass
class GenerationResult:
    """
    生成结果
    
    设计思想：
    1. 完整信息：包含生成结果和元数据信息
    2. 标准化：遵循OpenAI API响应格式，便于集成
    3. 可追踪：包含请求ID支持请求追踪和调试
    4. 统计信息：提供使用统计便于计费和监控
    """
    id: str
    text: str
    finish_reason: FinishReason
    usage: Dict[str, int]
    model: str
    created: int

@dataclass
class StreamChunk:
    """
    流式响应块
    
    设计思想：
    1. 增量传输：只传输新增内容，减少网络开销
    2. 状态同步：通过finish_reason同步生成状态
    3. 统计延迟：usage信息在最后一个chunk中提供
    4. 兼容性：与非流式响应保持字段一致性
    """
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

class ServerConfig(BaseSettings):
    """
    服务器配置
    
    设计思想：
    1. 网络配置：定义服务器的网络监听参数
    2. 并发控制：控制服务器的并发处理能力
    3. 超时管理：设置合理的请求超时时间
    4. 环境变量：支持通过环境变量覆盖配置
    """
    host: str = Field("0.0.0.0", description="服务器主机")                    # 监听所有网络接口
    port: int = Field(8000, description="服务器端口")                         # 默认HTTP端口
    workers: int = Field(1, description="工作进程数")                         # 单进程模式，避免模型重复加载
    max_concurrent_requests: int = Field(100, description="最大并发请求数")    # 并发限制，防止资源耗尽
    request_timeout: int = Field(300, description="请求超时时间(秒)")          # 5分钟超时，适合长文本生成
    
    class Config:
        # 环境变量前缀，支持配置覆盖
        # 设计思想：通过环境变量实现配置的灵活性和安全性
        env_prefix = "NANO_VLLM_SERVER_"

class ModelConfig(BaseSettings):
    """
    模型配置
    
    设计思想：
    1. 模型定位：指定模型文件的位置和标识
    2. 上下文管理：控制模型的上下文窗口大小
    3. 并行配置：支持多GPU并行推理
    4. 精度控制：支持不同的数据类型和量化方法
    """
    model_path: str = Field(..., description="模型路径")                      # 必需参数，模型文件路径
    model_name: str = Field("default", description="模型名称")                # 模型标识符
    max_context_length: int = Field(4096, description="最大上下文长度")        # 上下文窗口大小
    tensor_parallel_size: int = Field(1, description="张量并行大小")           # 多GPU并行度
    dtype: str = Field("float16", description="数据类型")                     # 半精度浮点，平衡精度和性能
    quantization: Optional[str] = Field(None, description="量化方法")          # 可选的模型量化
    
    class Config:
        # 模型相关环境变量前缀
        # 设计思想：模型配置通常需要根据部署环境调整
        env_prefix = "NANO_VLLM_MODEL_"

class GenerationConfig(BaseSettings):
    """
    生成配置
    
    设计思想：
    1. 长度控制：限制生成文本的最大长度
    2. 随机性控制：通过temperature控制生成的随机性
    3. 采样策略：支持多种采样方法（top-p, top-k）
    4. 质量优化：通过重复惩罚提高生成质量
    5. 停止条件：支持自定义停止序列
    """
    max_tokens: int = Field(512, description="最大生成token数")               # 生成长度限制
    temperature: float = Field(0.7, description="采样温度")                   # 控制随机性，0.7为平衡值
    top_p: float = Field(0.9, description="核采样参数")                       # 核采样阈值
    top_k: int = Field(50, description="Top-K采样参数")                       # Top-K采样候选数
    repetition_penalty: float = Field(1.0, description="重复惩罚")            # 重复惩罚系数
    stop_sequences: List[str] = Field([], description="停止序列")             # 自定义停止条件
    
    class Config:
        # 生成参数环境变量前缀
        # 设计思想：生成参数可能需要根据应用场景调整
        env_prefix = "NANO_VLLM_GENERATION_"

class MemoryConfig(BaseSettings):
    """
    内存配置
    
    设计思想：
    1. GPU内存管理：控制GPU内存的使用率
    2. CPU内存限制：防止CPU内存溢出
    3. 缓存优化：配置KV缓存的数据类型
    4. 性能优化：启用前缀缓存提高效率
    """
    gpu_memory_utilization: float = Field(0.9, description="GPU内存利用率")   # 90%利用率，留出缓冲空间
    cpu_memory_gb: Optional[int] = Field(None, description="CPU内存限制(GB)")  # 可选的CPU内存限制
    kv_cache_dtype: str = Field("auto", description="KV Cache数据类型")       # 自动选择最优数据类型
    enable_prefix_caching: bool = Field(False, description="启用前缀缓存")     # 前缀缓存优化
    
    class Config:
        # 内存相关环境变量前缀
        # 设计思想：内存配置通常需要根据硬件环境调整
        env_prefix = "NANO_VLLM_MEMORY_"

class LoggingConfig(BaseSettings):
    """
    日志配置
    
    设计思想：
    1. 级别控制：支持不同的日志级别
    2. 格式标准化：统一的日志格式便于解析
    3. 文件管理：支持日志文件轮转和大小限制
    4. 运维友好：便于生产环境的日志管理
    """
    level: str = Field("INFO", description="日志级别")                        # 默认INFO级别
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="日志格式")  # 标准格式
    file: Optional[str] = Field(None, description="日志文件路径")              # 可选的文件输出
    max_file_size: str = Field("100MB", description="日志文件最大大小")        # 文件大小限制
    backup_count: int = Field(5, description="日志文件备份数量")               # 轮转备份数量
    
    class Config:
        # 日志相关环境变量前缀
        # 设计思想：日志配置需要根据运维需求调整
        env_prefix = "NANO_VLLM_LOGGING_"

class NanoVLLMConfig(BaseSettings):
    """
    nano-vllm 主配置
    
    设计思想：
    1. 配置聚合：将所有子配置聚合到一个主配置类中
    2. 分层管理：按功能模块分组配置项
    3. 默认工厂：使用工厂模式创建默认配置实例
    4. 类型安全：通过Pydantic确保配置类型正确性
    """
    server: ServerConfig = Field(default_factory=ServerConfig)               # 服务器配置
    model: ModelConfig = Field(default_factory=ModelConfig)                  # 模型配置
    generation: GenerationConfig = Field(default_factory=GenerationConfig)   # 生成配置
    memory: MemoryConfig = Field(default_factory=MemoryConfig)               # 内存配置
    logging: LoggingConfig = Field(default_factory=LoggingConfig)            # 日志配置
    
    # 插件配置
    # 设计思想：插件系统的配置管理，支持动态启用/禁用插件
    plugins: Dict[str, Dict[str, Any]] = Field({}, description="插件配置")
    
    # 安全配置
    # 设计思想：API安全和访问控制配置
    api_key: Optional[str] = Field(None, description="API密钥")               # 可选的API密钥认证
    cors_origins: List[str] = Field(["*"], description="CORS允许的源")        # 跨域资源共享配置
    rate_limit: Dict[str, int] = Field({"requests_per_minute": 60}, description="速率限制")  # 请求频率限制
    
    class Config:
        # 环境变量配置文件
        # 设计思想：支持从.env文件加载环境变量，便于开发和部署
        env_file = ".env"
        env_file_encoding = "utf-8"

class ConfigManager:
    """
    配置管理器
    
    设计思想：
    1. 配置生命周期：管理配置的加载、更新、保存和验证
    2. 多格式支持：支持YAML和JSON配置文件格式
    3. 环境变量集成：自动合并环境变量配置
    4. 配置监听：支持配置变化的观察者模式
    5. 深度更新：支持嵌套配置的部分更新
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        设计思想：
        1. 路径管理：记录配置文件路径
        2. 状态初始化：初始化配置对象和观察者列表
        3. 延迟加载：配置对象在需要时才加载
        """
        self.config_path = config_path                                        # 配置文件路径
        self.config: Optional[NanoVLLMConfig] = None                         # 配置对象
        self.watchers = []                                                   # 配置变化观察者列表
    
    def load_config(self, config_path: Optional[str] = None) -> NanoVLLMConfig:
        """
        加载配置
        
        设计思想：
        1. 路径灵活性：支持动态指定配置文件路径
        2. 格式自动识别：根据文件扩展名自动选择解析器
        3. 环境变量优先：环境变量可以覆盖文件配置
        4. 默认配置：文件不存在时使用默认配置
        """
        if config_path:
            self.config_path = config_path
        
        if self.config_path and os.path.exists(self.config_path):
            # 从文件加载配置
            # 设计思想：支持多种配置文件格式，提高灵活性
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)                          # YAML格式解析
                elif self.config_path.endswith('.json'):
                    config_data = json.load(f)                               # JSON格式解析
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
            
            # 合并环境变量
            # 设计思想：环境变量优先级高于文件配置，便于部署时覆盖
            self.config = NanoVLLMConfig(**config_data)
        else:
            # 仅从环境变量加载
            # 设计思想：支持纯环境变量配置，适合容器化部署
            self.config = NanoVLLMConfig()
        
        return self.config
    
    def save_config(self, config_path: Optional[str] = None):
        """
        保存配置
        
        设计思想：
        1. 路径灵活性：支持保存到不同路径
        2. 格式保持：根据文件扩展名选择保存格式
        3. 编码统一：使用UTF-8编码确保国际化支持
        4. 格式化输出：保存时格式化配置便于阅读
        """
        if not self.config:
            raise ValueError("No config to save")
        
        save_path = config_path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")
        
        # 转换为字典格式
        # 设计思想：使用Pydantic的dict()方法确保数据完整性
        config_dict = self.config.dict()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                # YAML格式保存，便于人工编辑
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            elif save_path.endswith('.json'):
                # JSON格式保存，便于程序处理
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {save_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        设计思想：
        1. 部分更新：支持只更新部分配置项
        2. 深度合并：支持嵌套配置的更新
        3. 类型安全：重新创建配置对象确保类型正确
        4. 变化通知：更新后通知所有观察者
        """
        if not self.config:
            self.load_config()
        
        # 深度更新配置
        # 设计思想：递归更新嵌套字典，保持配置结构完整
        self._deep_update(self.config.dict(), updates)
        
        # 重新创建配置对象
        # 设计思想：通过重新创建确保Pydantic验证生效
        self.config = NanoVLLMConfig(**self.config.dict())
        
        # 通知观察者
        # 设计思想：观察者模式，支持配置变化的响应式处理
        self._notify_watchers()
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """
        深度更新字典
        
        设计思想：
        1. 递归更新：处理嵌套字典的更新
        2. 类型检查：确保只对字典类型进行递归
        3. 覆盖策略：新值覆盖旧值
        4. 结构保持：保持原有字典结构
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                # 递归更新嵌套字典
                self._deep_update(base_dict[key], value)
            else:
                # 直接覆盖值
                base_dict[key] = value
    
    def watch_config(self, callback):
        """
        监听配置变化
        
        设计思想：
        1. 观察者模式：支持多个观察者监听配置变化
        2. 回调机制：配置变化时自动调用回调函数
        3. 解耦设计：配置管理与业务逻辑解耦
        4. 扩展性：便于添加新的配置变化处理逻辑
        """
        self.watchers.append(callback)
    
    def _notify_watchers(self):
        """
        通知配置观察者
        
        设计思想：
        1. 批量通知：一次性通知所有观察者
        2. 异常隔离：单个观察者异常不影响其他观察者
        3. 错误日志：记录观察者处理异常
        4. 继续执行：异常不中断通知流程
        """
        for callback in self.watchers:
            try:
                callback(self.config)
            except Exception as e:
                logger.error(f"Error in config watcher: {e}")
    
    def validate_config(self) -> List[str]:
        """
        验证配置
        
        设计思想：
        1. 全面验证：检查配置的各个方面
        2. 错误收集：收集所有验证错误而不是遇到第一个就停止
        3. 业务规则：除了类型检查还包含业务逻辑验证
        4. 友好提示：提供具体的错误信息便于修复
        """
        errors = []
        
        if not self.config:
            errors.append("No config loaded")
            return errors
        
        # 验证模型路径
        # 设计思想：确保模型文件存在，避免运行时错误
        if not os.path.exists(self.config.model.model_path):
            errors.append(f"Model path does not exist: {self.config.model.model_path}")
        
        # 验证端口范围
        # 设计思想：确保端口号在有效范围内
        if not (1 <= self.config.server.port <= 65535):
            errors.append(f"Invalid port number: {self.config.server.port}")
        
        # 验证内存配置
        # 设计思想：确保GPU内存利用率在合理范围内
        if not (0.1 <= self.config.memory.gpu_memory_utilization <= 1.0):
            errors.append(f"Invalid GPU memory utilization: {self.config.memory.gpu_memory_utilization}")
        
        # 验证生成参数
        # 设计思想：确保生成参数在有效范围内
        if not (0.0 <= self.config.generation.temperature <= 2.0):
            errors.append(f"Invalid temperature: {self.config.generation.temperature}")
        
        if not (0.0 <= self.config.generation.top_p <= 1.0):
            errors.append(f"Invalid top_p: {self.config.generation.top_p}")
        
        return errors

# 配置文件示例
# 设计思想：提供完整的配置示例，便于用户理解和使用
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
    """
    配置使用示例
    
    设计思想：
    1. 完整流程：展示配置管理的完整使用流程
    2. 错误处理：展示如何处理配置验证错误
    3. 动态更新：展示如何动态更新配置
    4. 实用性：提供实际可用的代码示例
    """
    # 创建配置管理器
    # 设计思想：统一的配置管理入口
    config_manager = ConfigManager()
    
    # 加载配置
    # 设计思想：从文件或环境变量加载配置
    config = config_manager.load_config("config.yaml")
    
    # 验证配置
    # 设计思想：启动前验证配置有效性
    errors = config_manager.validate_config()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # 使用配置
    # 设计思想：配置加载后即可使用
    print(f"Server will run on {config.server.host}:{config.server.port}")
    print(f"Model: {config.model.model_name} at {config.model.model_path}")
    
    # 更新配置
    # 设计思想：支持运行时配置更新
    config_manager.update_config({
        "server": {"port": 8080},
        "generation": {"temperature": 0.8}
    })
    
    # 保存配置
    # 设计思想：配置更新后可以持久化保存
    config_manager.save_config("updated_config.yaml")
```

---

*通过精心设计的接口系统，nano-vllm 提供了灵活、易用、可扩展的API，支持多种使用场景和集成需求。无论是简单的文本生成还是复杂的批量处理，都能通过统一的接口实现。*