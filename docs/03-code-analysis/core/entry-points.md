# 入口点分析

## 🚪 系统入口概览

nano-vllm 提供了多种入口点来满足不同的使用场景，从简单的API服务到复杂的分布式部署。理解这些入口点是掌握系统架构的第一步。

## 📋 主要入口点

### 1. HTTP API 服务器

```python
#!/usr/bin/env python3
"""
nano-vllm HTTP API 服务器入口点
提供 OpenAI 兼容的 API 接口

🎯 设计思想：
- 采用 FastAPI 框架提供高性能异步 HTTP 服务
- 实现 OpenAI API 兼容接口，便于现有应用迁移
- 支持流式和非流式两种响应模式
- 集成完整的监控、日志和错误处理机制
"""

import asyncio          # 异步编程支持，处理并发请求
import argparse         # 命令行参数解析
import logging          # 日志记录系统
import signal           # 信号处理，优雅关闭服务
import sys              # 系统相关功能
from typing import Optional, Dict, Any  # 类型注解支持
from pathlib import Path                 # 路径操作工具

# FastAPI 相关导入 - 现代高性能 Web 框架
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn          # ASGI 服务器，生产级部署

# nano-vllm 核心组件导入
from nano_vllm.engine import InferenceEngine      # 推理引擎核心
from nano_vllm.config import ServerConfig, ModelConfig  # 配置管理
from nano_vllm.api.models import (                # API 数据模型
    CompletionRequest, 
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse
)
from nano_vllm.utils.logging import setup_logging    # 日志配置工具
from nano_vllm.utils.metrics import MetricsCollector # 性能指标收集

logger = logging.getLogger(__name__)

class NanoVLLMServer:
    """
    nano-vllm HTTP API 服务器
    
    🏗️ 架构设计：
    - 单例模式管理推理引擎，避免重复初始化
    - 中间件模式处理横切关注点（CORS、日志、指标）
    - 异步处理模式支持高并发请求
    - 优雅关闭机制确保资源正确释放
    """
    
    def __init__(self, config: ServerConfig):
        """
        初始化服务器实例
        
        Args:
            config: 服务器配置对象，包含所有运行参数
        
        🔧 初始化流程：
        1. 保存配置信息
        2. 创建 FastAPI 应用实例
        3. 初始化推理引擎占位符
        4. 创建指标收集器
        5. 设置中间件和路由
        6. 配置信号处理器
        """
        self.config = config
        
        # 创建 FastAPI 应用，配置基本信息
        self.app = FastAPI(
            title="nano-vllm API",                    # API 标题
            description="High-performance LLM inference server",  # 描述
            version="0.1.0"                          # 版本号
        )
        
        # 推理引擎延迟初始化，避免构造函数中的重操作
        self.engine: Optional[InferenceEngine] = None
        
        # 指标收集器，用于监控系统性能
        self.metrics = MetricsCollector()
        
        # 设置应用组件
        self._setup_middleware()      # 配置中间件
        self._setup_routes()         # 注册API路由
        self._setup_signal_handlers() # 配置信号处理
    
    def _setup_middleware(self):
        """
        设置中间件
        
        🛡️ 中间件作用：
        - CORS：处理跨域请求，支持前端应用调用
        - 请求日志：记录所有HTTP请求的详细信息
        - 性能指标：收集请求处理时间和状态码统计
        """
        
        # CORS 中间件 - 处理跨域资源共享
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,  # 允许的源域名
            allow_credentials=True,                  # 允许携带认证信息
            allow_methods=["*"],                     # 允许所有HTTP方法
            allow_headers=["*"],                     # 允许所有请求头
        )
        
        # 请求日志和指标中间件
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            """
            HTTP请求中间件
            
            📊 功能：
            1. 记录请求开始时间和基本信息
            2. 调用下一个处理器处理请求
            3. 计算处理时间并记录响应信息
            4. 更新性能指标统计
            
            🔍 监控指标：
            - 请求方法和路径
            - 响应状态码
            - 处理耗时
            - 请求频率统计
            """
            import time  # 导入时间模块用于计时
            start_time = time.time()  # 记录请求开始时间
            
            # 记录请求开始 - 便于调试和监控
            logger.info(f"Request: {request.method} {request.url}")
            
            # 处理请求 - 调用实际的路由处理函数
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录请求完成信息
            logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")
            
            # 更新性能指标 - 用于监控和分析
            self.metrics.record_request(
                method=request.method,              # HTTP方法
                endpoint=str(request.url.path),     # 请求路径
                status_code=response.status_code,   # 响应状态码
                duration=process_time               # 处理时长
            )
            
            return response
    
    def _setup_routes(self):
        """
        设置API路由
        
        🛣️ 路由设计：
        - RESTful API 设计原则
        - OpenAI API 兼容接口
        - 健康检查和监控端点
        - 流式和非流式响应支持
        """
        
        @self.app.get("/health")
        async def health_check():
            """
            健康检查端点
            
            🏥 检查项目：
            1. 推理引擎是否已初始化
            2. 推理引擎运行状态
            3. 系统性能指标摘要
            
            返回：
            - status: 服务状态
            - engine: 引擎状态详情
            - metrics: 性能指标摘要
            """
            # 检查推理引擎是否已初始化
            if self.engine is None:
                raise HTTPException(
                    status_code=503,  # 服务不可用
                    detail="Engine not initialized"
                )
            
            # 获取引擎状态信息
            engine_status = await self.engine.get_status()
            
            return {
                "status": "healthy",                    # 服务状态
                "engine": engine_status,                # 引擎详细状态
                "metrics": self.metrics.get_summary()   # 性能指标摘要
            }
        
        @self.app.get("/models")
        async def list_models():
            """
            列出可用模型端点
            
            📋 功能：
            - 返回当前加载的模型列表
            - 兼容 OpenAI API 格式
            - 提供模型基本信息
            """
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            # 获取可用模型列表
            models = await self.engine.list_models()
            return {"data": models}  # OpenAI API 兼容格式
        
        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """
            文本补全端点
            
            🎯 核心功能：
            1. 接收文本补全请求
            2. 调用推理引擎生成文本
            3. 支持流式和非流式响应
            4. 返回 OpenAI 兼容格式
            
            Args:
                request: 补全请求对象，包含提示文本和生成参数
            
            Returns:
                CompletionResponse: 补全结果或流式响应
            """
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                # 判断是否为流式请求
                if request.stream:
                    # 返回流式响应 - 实时生成文本
                    return StreamingResponse(
                        self._stream_completion(request),  # 流式生成器
                        media_type="text/plain"           # 内容类型
                    )
                else:
                    # 非流式响应 - 一次性返回完整结果
                    result = await self.engine.generate(
                        prompt=request.prompt,              # 输入提示
                        max_tokens=request.max_tokens,      # 最大生成长度
                        temperature=request.temperature,    # 随机性控制
                        top_p=request.top_p,               # 核采样参数
                        stop=request.stop                  # 停止词列表
                    )
                    
                    # 构造 OpenAI 兼容响应格式
                    import uuid
                    response = CompletionResponse(
                        id=f"cmpl-{uuid.uuid4()}",         # 唯一请求ID
                        choices=[{                         # 生成选择列表
                            "text": result.text,           # 生成的文本
                            "index": 0,                    # 选择索引
                            "finish_reason": result.finish_reason  # 结束原因
                        }],
                        usage={                            # Token使用统计
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens
                        }
                    )
                    
                    return response
                    
            except Exception as e:
                # 错误处理和日志记录
                logger.error(f"Completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """
            聊天补全端点
            
            💬 聊天功能：
            1. 接收多轮对话消息
            2. 转换为模型可理解的提示格式
            3. 生成助手回复
            4. 支持流式对话体验
            
            Args:
                request: 聊天补全请求，包含消息历史和参数
            
            Returns:
                ChatCompletionResponse: 聊天回复或流式响应
            """
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                # 将聊天消息转换为提示文本
                # 这是关键步骤：多轮对话 -> 单一提示文本
                prompt = self._messages_to_prompt(request.messages)
                
                if request.stream:
                    # 流式聊天响应
                    return StreamingResponse(
                        self._stream_chat_completion(request, prompt),
                        media_type="text/plain"
                    )
                else:
                    # 非流式聊天响应
                    result = await self.engine.generate(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop
                    )
                    
                    # 构造聊天响应格式
                    import uuid
                    response = ChatCompletionResponse(
                        id=f"chatcmpl-{uuid.uuid4()}",
                        choices=[{
                            "message": {                   # 助手消息
                                "role": "assistant",       # 角色标识
                                "content": result.text     # 回复内容
                            },
                            "index": 0,
                            "finish_reason": result.finish_reason
                        }],
                        usage={
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens
                        }
                    )
                    
                    return response
                    
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """
            系统指标端点
            
            📊 监控数据：
            1. 推理引擎性能指标
            2. HTTP服务器统计信息
            3. 系统资源使用情况
            4. 实时时间戳
            
            用途：
            - 性能监控和告警
            - 容量规划分析
            - 故障诊断支持
            """
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            # 收集各组件指标
            engine_metrics = await self.engine.get_metrics()    # 引擎指标
            server_metrics = self.metrics.get_detailed_stats()  # 服务器指标
            
            return {
                "engine": engine_metrics,      # 推理引擎指标
                "server": server_metrics,      # HTTP服务器指标
                "timestamp": time.time()       # 数据时间戳
            }
    
    async def _stream_completion(self, request: CompletionRequest):
        """
        流式文本补全生成器
        
        🌊 流式处理优势：
        1. 降低首字延迟 - 用户更快看到响应
        2. 改善用户体验 - 实时显示生成过程
        3. 减少内存占用 - 逐块处理而非全量缓存
        4. 支持长文本生成 - 避免超时问题
        
        Args:
            request: 补全请求参数
        
        Yields:
            str: SSE格式的数据块
        """
        import json
        
        # 异步生成文本流
        async for chunk in self.engine.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        ):
            # 按照 Server-Sent Events (SSE) 格式输出
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest, prompt: str):
        """
        流式聊天补全生成器
        
        💬 聊天流式特点：
        1. 保持对话上下文连续性
        2. 实时显示助手思考过程
        3. 支持长对话不中断
        4. 兼容前端聊天界面
        
        Args:
            request: 聊天请求参数
            prompt: 转换后的提示文本
        
        Yields:
            str: SSE格式的聊天数据块
        """
        import json
        import uuid
        
        # 异步生成聊天回复流
        async for chunk in self.engine.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        ):
            # 构造聊天流式响应格式
            chat_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "choices": [{
                    "delta": {"content": chunk.get("text", "")},  # 增量内容
                    "index": 0
                }]
            }
            yield f"data: {json.dumps(chat_chunk)}\n\n"
        
        # 发送聊天结束标记
        yield "data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: list) -> str:
        """
        将聊天消息转换为提示文本
        
        🔄 转换逻辑：
        1. 遍历消息历史
        2. 根据角色添加前缀标识
        3. 组合成连续的对话文本
        4. 添加助手回复提示
        
        Args:
            messages: 聊天消息列表
        
        Returns:
            str: 格式化的提示文本
        
        💡 设计考虑：
        - 保持对话上下文完整性
        - 明确角色区分便于模型理解
        - 支持系统消息设定角色行为
        - 兼容不同的对话模板格式
        """
        prompt_parts = []
        
        # 遍历消息历史，构建对话文本
        for message in messages:
            role = message.get("role", "user")      # 获取消息角色
            content = message.get("content", "")    # 获取消息内容
            
            # 根据角色添加相应前缀
            if role == "system":
                prompt_parts.append(f"System: {content}")      # 系统设定
            elif role == "user":
                prompt_parts.append(f"User: {content}")        # 用户输入
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")   # 助手回复
        
        # 添加助手回复提示，引导模型生成
        prompt_parts.append("Assistant:")
        
        # 用换行符连接所有部分
        return "\n".join(prompt_parts)
    
    def _setup_signal_handlers(self):
        """
        设置信号处理器
        
        🛡️ 优雅关闭机制：
        1. 捕获系统终止信号
        2. 执行清理操作
        3. 释放系统资源
        4. 确保数据完整性
        
        支持的信号：
        - SIGINT: Ctrl+C 中断信号
        - SIGTERM: 系统终止信号
        """
        def signal_handler(signum, frame):
            """
            信号处理函数
            
            Args:
                signum: 信号编号
                frame: 当前执行帧
            """
            logger.info(f"Received signal {signum}, shutting down...")
            # 创建异步任务执行关闭流程
            asyncio.create_task(self.shutdown())
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    async def startup(self):
        """
        启动服务器
        
        🚀 启动流程：
        1. 记录启动日志
        2. 创建模型配置
        3. 初始化推理引擎
        4. 等待引擎就绪
        5. 记录启动完成
        
        ⚠️ 注意事项：
        - 推理引擎初始化可能耗时较长
        - 需要足够的GPU内存和计算资源
        - 模型加载失败会导致服务无法启动
        """
        logger.info("Starting nano-vllm server...")
        
        # 创建模型配置对象
        model_config = ModelConfig(
            model_path=self.config.model_path,                      # 模型路径
            tensor_parallel_size=self.config.tensor_parallel_size,  # 张量并行大小
            max_model_len=self.config.max_model_len,               # 最大模型长度
            gpu_memory_utilization=self.config.gpu_memory_utilization  # GPU内存利用率
        )
        
        # 初始化推理引擎
        self.engine = InferenceEngine(model_config)
        await self.engine.initialize()  # 异步初始化，可能耗时较长
        
        logger.info("nano-vllm server started successfully")
    
    async def shutdown(self):
        """
        关闭服务器
        
        🛑 关闭流程：
        1. 记录关闭开始日志
        2. 停止接受新请求
        3. 等待现有请求完成
        4. 关闭推理引擎
        5. 释放系统资源
        6. 记录关闭完成
        """
        logger.info("Shutting down nano-vllm server...")
        
        # 关闭推理引擎
        if self.engine:
            await self.engine.shutdown()
        
        logger.info("nano-vllm server shutdown complete")

def parse_args():
    """
    解析命令行参数
    
    📝 参数分类：
    1. 服务器配置：主机、端口、工作进程数
    2. 模型配置：模型路径、并行设置、内存配置
    3. 日志配置：日志级别、日志文件路径
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    
    💡 设计原则：
    - 提供合理的默认值
    - 支持生产环境配置
    - 参数验证和错误提示
    - 帮助信息清晰易懂
    """
    parser = argparse.ArgumentParser(
        description="nano-vllm HTTP API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python server.py --model /path/to/model --port 8000
  python server.py --model /path/to/model --tensor-parallel-size 2 --gpu-memory-utilization 0.8
        """
    )
    
    # 服务器配置参数组
    server_group = parser.add_argument_group('服务器配置')
    server_group.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="服务器监听地址 (默认: 0.0.0.0)"
    )
    server_group.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="服务器监听端口 (默认: 8000)"
    )
    server_group.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="工作进程数量 (默认: 1)"
    )
    
    # 模型配置参数组
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="模型路径或HuggingFace模型名称 (必需)"
    )
    model_group.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        default=1, 
        help="张量并行大小，用于多GPU推理 (默认: 1)"
    )
    model_group.add_argument(
        "--max-model-len", 
        type=int, 
        default=2048, 
        help="模型最大序列长度 (默认: 2048)"
    )
    model_group.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        default=0.9, 
        help="GPU内存利用率 (0.0-1.0, 默认: 0.9)"
    )
    
    # 日志配置参数组
    log_group = parser.add_argument_group('日志配置')
    log_group.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    log_group.add_argument(
        "--log-file", 
        type=str, 
        help="日志文件路径 (可选，默认输出到控制台)"
    )
    
    return parser.parse_args()

async def main():
    """
    主函数 - 应用程序入口点
    
    🎯 主要职责：
    1. 解析命令行参数
    2. 配置日志系统
    3. 创建服务器实例
    4. 启动HTTP服务
    5. 处理异常和清理
    
    🔄 执行流程：
    1. 参数解析和验证
    2. 日志系统初始化
    3. 服务器配置创建
    4. 推理引擎启动
    5. HTTP服务运行
    6. 优雅关闭处理
    """
    try:
        # 1. 解析命令行参数
        args = parse_args()
        
        # 2. 设置日志系统
        setup_logging(
            level=args.log_level,    # 日志级别
            log_file=args.log_file   # 日志文件（可选）
        )
        
        # 3. 创建服务器配置
        config = ServerConfig(
            host=args.host,                                    # 监听地址
            port=args.port,                                    # 监听端口
            model_path=args.model,                             # 模型路径
            tensor_parallel_size=args.tensor_parallel_size,    # 张量并行
            max_model_len=args.max_model_len,                 # 最大长度
            gpu_memory_utilization=args.gpu_memory_utilization # GPU内存利用率
        )
        
        # 4. 创建并启动服务器
        server = NanoVLLMServer(config)
        await server.startup()  # 异步启动，初始化推理引擎
        
        # 5. 配置并运行HTTP服务器
        uvicorn_config = uvicorn.Config(
            app=server.app,                    # FastAPI应用实例
            host=config.host,                  # 监听地址
            port=config.port,                  # 监听端口
            workers=args.workers,              # 工作进程数
            log_level=args.log_level.lower()   # 日志级别
        )
        
        # 创建并运行服务器实例
        server_instance = uvicorn.Server(uvicorn_config)
        await server_instance.serve()  # 阻塞运行，直到收到关闭信号
        
    except KeyboardInterrupt:
        # 处理用户中断 (Ctrl+C)
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        # 处理其他异常
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        # 确保资源清理
        logger.info("Server shutdown complete")

# 程序入口点
if __name__ == "__main__":
    """
    脚本直接执行时的入口点
    
    🚀 启动方式：
    python server.py --model /path/to/model --port 8000
    
    ⚡ 异步运行：
    使用 asyncio.run() 启动异步主函数
    这是 Python 3.7+ 推荐的异步程序启动方式
    """
    asyncio.run(main())