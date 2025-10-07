#!/usr/bin/env python3
"""
NanoVLLM API 服务器

提供 RESTful API 接口来调用 nano-vLLM 推理系统。
支持单个请求、批量请求、流式响应等功能。
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Web框架
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    
    # NanoVLLM组件
    from nano_vllm import NanoVLLM, GenerationParams, InferenceResponse
    from config import get_config, validate_config
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装依赖: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# ================================
# API 数据模型
# ================================

class GenerationRequest(BaseModel):
    """生成请求模型"""
    prompt: str = Field(..., description="输入提示文本")
    max_tokens: int = Field(100, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p采样参数")
    top_k: int = Field(50, ge=1, le=100, description="Top-k采样参数")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="重复惩罚")
    stream: bool = Field(False, description="是否流式返回")
    request_id: Optional[str] = Field(None, description="请求ID")

class BatchGenerationRequest(BaseModel):
    """批量生成请求模型"""
    prompts: List[str] = Field(..., description="输入提示文本列表")
    max_tokens: int = Field(100, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="温度参数")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p采样参数")
    top_k: int = Field(50, ge=1, le=100, description="Top-k采样参数")
    repetition_penalty: float = Field(1.0, ge=0.0, le=2.0, description="重复惩罚")

class GenerationResponse(BaseModel):
    """生成响应模型"""
    request_id: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    finish_reason: str
    timestamp: str

class BatchGenerationResponse(BaseModel):
    """批量生成响应模型"""
    results: List[GenerationResponse]
    total_time: float
    total_tokens: int
    average_tokens_per_second: float

class SystemMetricsResponse(BaseModel):
    """系统指标响应模型"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    tokens_per_second: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_memory_utilization: float
    kv_cache_usage: float
    waiting_requests: int
    running_requests: int
    swapped_requests: int

class HealthResponse(BaseModel):
    """健康状态响应模型"""
    healthy: bool
    uptime: float
    model_loaded: bool
    engine_running: bool
    issues: List[str] = []

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None

# ================================
# 全局变量
# ================================

nano_vllm: Optional[NanoVLLM] = None
server_start_time = time.time()
request_counter = 0

# ================================
# 应用生命周期管理
# ================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global nano_vllm
    
    # 启动时初始化
    print("🚀 启动 NanoVLLM API 服务器...")
    
    try:
        # 从环境变量或默认值获取配置
        config_name = os.getenv("NANO_VLLM_CONFIG", "dev")
        model_name = os.getenv("NANO_VLLM_MODEL", "microsoft/DialoGPT-small")
        
        print(f"📋 加载配置: {config_name}")
        config = get_config(config_name)
        
        if not validate_config(config):
            raise ValueError("配置验证失败")
        
        # 转换配置
        config_dict = {
            "device": config.device,
            "torch_dtype": config.torch_dtype,
            "max_num_seqs": config.max_num_seqs,
            "max_batch_size": config.max_batch_size,
            "block_size": config.block_size,
            "num_gpu_blocks": config.num_gpu_blocks,
            "scheduler_policy": config.scheduler_policy,
            "memory_threshold": config.memory_threshold,
            "engine_loop_interval": config.engine_loop_interval,
        }
        
        # 初始化系统
        nano_vllm = NanoVLLM(config_dict)
        nano_vllm.__enter__()
        
        # 加载模型
        print(f"📦 加载模型: {model_name}")
        if not nano_vllm.load_model(model_name):
            raise ValueError("模型加载失败")
        
        print("✅ NanoVLLM API 服务器启动成功！")
        
        yield
        
    except Exception as e:
        print(f"❌ 服务器启动失败: {e}")
        raise
    
    finally:
        # 关闭时清理
        print("🔄 关闭 NanoVLLM API 服务器...")
        if nano_vllm:
            try:
                nano_vllm.__exit__(None, None, None)
            except Exception as e:
                print(f"⚠️ 清理时出错: {e}")
        print("👋 NanoVLLM API 服务器已关闭")

# ================================
# FastAPI 应用
# ================================

app = FastAPI(
    title="NanoVLLM API",
    description="轻量级 vLLM 推理系统 API 接口",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# 工具函数
# ================================

def generate_request_id() -> str:
    """生成请求ID"""
    global request_counter
    request_counter += 1
    return f"req_{int(time.time())}_{request_counter}"

def create_error_response(error: str, message: str, request_id: Optional[str] = None) -> ErrorResponse:
    """创建错误响应"""
    return ErrorResponse(
        error=error,
        message=message,
        timestamp=datetime.now().isoformat(),
        request_id=request_id
    )

def convert_inference_response(response: InferenceResponse, request_id: str) -> GenerationResponse:
    """转换推理响应"""
    return GenerationResponse(
        request_id=request_id,
        generated_text=response.generated_text,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        generation_time=response.generation_time,
        tokens_per_second=response.tokens_per_second,
        finish_reason=response.finish_reason,
        timestamp=datetime.now().isoformat()
    )

# ================================
# API 端点
# ================================

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "NanoVLLM API 服务器",
        "version": "1.0.0",
        "status": "running",
        "uptime": time.time() - server_start_time
    }

@app.post("/v1/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """单个文本生成"""
    if not nano_vllm:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    request_id = request.request_id or generate_request_id()
    
    try:
        # 创建生成参数
        params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        # 执行生成
        response = nano_vllm.generate(
            prompt=request.prompt,
            params=params
        )
        
        return convert_inference_response(response, request_id)
        
    except Exception as e:
        logging.error(f"生成失败 {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("generation_failed", str(e), request_id).dict()
        )

@app.post("/v1/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """流式文本生成"""
    if not nano_vllm:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    request_id = request.request_id or generate_request_id()
    
    async def generate_stream():
        try:
            # 创建生成参数
            params = GenerationParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty
            )
            
            # 模拟流式生成（实际实现需要修改nano_vllm支持流式）
            response = nano_vllm.generate(
                prompt=request.prompt,
                params=params
            )
            
            # 分块发送响应
            words = response.generated_text.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "request_id": request_id,
                    "chunk": word + " ",
                    "is_final": i == len(words) - 1,
                    "timestamp": datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.1)  # 模拟延迟
            
            # 发送结束标记
            final_data = {
                "request_id": request_id,
                "chunk": "",
                "is_final": True,
                "total_tokens": response.total_tokens,
                "generation_time": response.generation_time,
                "tokens_per_second": response.tokens_per_second,
                "finish_reason": response.finish_reason,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
        except Exception as e:
            error_data = {
                "request_id": request_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.post("/v1/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    """批量文本生成"""
    if not nano_vllm:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    if len(request.prompts) > 10:  # 限制批量大小
        raise HTTPException(status_code=400, detail="批量大小不能超过10")
    
    try:
        start_time = time.time()
        results = []
        total_tokens = 0
        
        # 创建生成参数
        params = GenerationParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        # 批量处理
        for i, prompt in enumerate(request.prompts):
            request_id = generate_request_id()
            
            response = nano_vllm.generate(
                prompt=prompt,
                params=params
            )
            
            result = convert_inference_response(response, request_id)
            results.append(result)
            total_tokens += response.total_tokens
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return BatchGenerationResponse(
            results=results,
            total_time=total_time,
            total_tokens=total_tokens,
            average_tokens_per_second=total_tokens / total_time if total_time > 0 else 0
        )
        
    except Exception as e:
        logging.error(f"批量生成失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("batch_generation_failed", str(e)).dict()
        )

@app.get("/v1/metrics", response_model=SystemMetricsResponse)
async def get_metrics():
    """获取系统指标"""
    if not nano_vllm:
        raise HTTPException(status_code=503, detail="服务不可用")
    
    try:
        metrics = nano_vllm.get_metrics()
        
        return SystemMetricsResponse(
            total_requests=metrics.total_requests,
            successful_requests=metrics.successful_requests,
            failed_requests=metrics.failed_requests,
            requests_per_second=metrics.requests_per_second,
            tokens_per_second=metrics.tokens_per_second,
            avg_latency=metrics.avg_latency,
            p95_latency=metrics.p95_latency,
            p99_latency=metrics.p99_latency,
            gpu_memory_used=metrics.gpu_memory_used,
            gpu_memory_total=metrics.gpu_memory_total,
            gpu_memory_utilization=metrics.gpu_memory_utilization,
            kv_cache_usage=metrics.kv_cache_usage,
            waiting_requests=metrics.waiting_requests,
            running_requests=metrics.running_requests,
            swapped_requests=metrics.swapped_requests
        )
        
    except Exception as e:
        logging.error(f"获取指标失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("metrics_failed", str(e)).dict()
        )

@app.get("/v1/health", response_model=HealthResponse)
async def get_health():
    """健康检查"""
    if not nano_vllm:
        return HealthResponse(
            healthy=False,
            uptime=time.time() - server_start_time,
            model_loaded=False,
            engine_running=False,
            issues=["NanoVLLM 未初始化"]
        )
    
    try:
        health = nano_vllm.get_health_status()
        
        return HealthResponse(
            healthy=health.get('healthy', False),
            uptime=time.time() - server_start_time,
            model_loaded=health.get('model_loaded', False),
            engine_running=health.get('engine_running', False),
            issues=health.get('issues', [])
        )
        
    except Exception as e:
        logging.error(f"健康检查失败: {e}")
        return HealthResponse(
            healthy=False,
            uptime=time.time() - server_start_time,
            model_loaded=False,
            engine_running=False,
            issues=[f"健康检查错误: {str(e)}"]
        )

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "models": [
            {
                "id": "microsoft/DialoGPT-small",
                "name": "DialoGPT Small",
                "description": "小型对话生成模型"
            },
            {
                "id": "microsoft/DialoGPT-medium", 
                "name": "DialoGPT Medium",
                "description": "中型对话生成模型"
            },
            {
                "id": "gpt2",
                "name": "GPT-2",
                "description": "OpenAI GPT-2 模型"
            }
        ]
    }

# ================================
# 错误处理
# ================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error="http_error",
            message=exc.detail,
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error="internal_error",
            message="内部服务器错误"
        ).dict()
    )

# ================================
# 主函数
# ================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NanoVLLM API 服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--log-level", default="info", help="日志级别")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print(f"🚀 启动 NanoVLLM API 服务器")
    print(f"   📍 地址: http://{args.host}:{args.port}")
    print(f"   📋 API文档: http://{args.host}:{args.port}/docs")
    print(f"   🔧 工作进程: {args.workers}")
    print(f"   📊 日志级别: {args.log_level}")
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=args.reload
    )

if __name__ == "__main__":
    main()