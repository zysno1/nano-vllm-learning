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
"""

import asyncio
import argparse
import logging
import signal
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from nano_vllm.engine import InferenceEngine
from nano_vllm.config import ServerConfig, ModelConfig
from nano_vllm.api.models import (
    CompletionRequest, 
    CompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse
)
from nano_vllm.utils.logging import setup_logging
from nano_vllm.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class NanoVLLMServer:
    """nano-vllm HTTP API 服务器"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(
            title="nano-vllm API",
            description="High-performance LLM inference server",
            version="0.1.0"
        )
        self.engine: Optional[InferenceEngine] = None
        self.metrics = MetricsCollector()
        
        # 设置中间件
        self._setup_middleware()
        
        # 注册路由
        self._setup_routes()
        
        # 设置信号处理
        self._setup_signal_handlers()
    
    def _setup_middleware(self):
        """设置中间件"""
        # CORS 中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 请求日志中间件
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # 记录请求开始
            logger.info(f"Request: {request.method} {request.url}")
            
            # 处理请求
            response = await call_next(request)
            
            # 记录请求完成
            process_time = time.time() - start_time
            logger.info(f"Response: {response.status_code} ({process_time:.3f}s)")
            
            # 更新指标
            self.metrics.record_request(
                method=request.method,
                endpoint=str(request.url.path),
                status_code=response.status_code,
                duration=process_time
            )
            
            return response
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            engine_status = await self.engine.get_status()
            return {
                "status": "healthy",
                "engine": engine_status,
                "metrics": self.metrics.get_summary()
            }
        
        @self.app.get("/models")
        async def list_models():
            """列出可用模型"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            models = await self.engine.list_models()
            return {"data": models}
        
        @self.app.post("/v1/completions")
        async def create_completion(request: CompletionRequest):
            """文本补全"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                if request.stream:
                    return StreamingResponse(
                        self._stream_completion(request),
                        media_type="text/plain"
                    )
                else:
                    result = await self.engine.generate(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop
                    )
                    
                    response = CompletionResponse(
                        id=f"cmpl-{uuid.uuid4()}",
                        choices=[{
                            "text": result.text,
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
                logger.error(f"Completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            """聊天补全"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            try:
                # 将聊天消息转换为提示文本
                prompt = self._messages_to_prompt(request.messages)
                
                if request.stream:
                    return StreamingResponse(
                        self._stream_chat_completion(request, prompt),
                        media_type="text/plain"
                    )
                else:
                    result = await self.engine.generate(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stop=request.stop
                    )
                    
                    response = ChatCompletionResponse(
                        id=f"chatcmpl-{uuid.uuid4()}",
                        choices=[{
                            "message": {
                                "role": "assistant",
                                "content": result.text
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
            """获取系统指标"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            engine_metrics = await self.engine.get_metrics()
            server_metrics = self.metrics.get_detailed_stats()
            
            return {
                "engine": engine_metrics,
                "server": server_metrics,
                "timestamp": time.time()
            }
    
    async def _stream_completion(self, request: CompletionRequest):
        """流式文本补全"""
        async for chunk in self.engine.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    async def _stream_chat_completion(self, request: ChatCompletionRequest, prompt: str):
        """流式聊天补全"""
        async for chunk in self.engine.generate_stream(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        ):
            chat_chunk = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "choices": [{
                    "delta": {"content": chunk.get("text", "")},
                    "index": 0
                }]
            }
            yield f"data: {json.dumps(chat_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    def _messages_to_prompt(self, messages: list) -> str:
        """将聊天消息转换为提示文本"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def startup(self):
        """启动服务器"""
        logger.info("Starting nano-vllm server...")
        
        # 初始化推理引擎
        model_config = ModelConfig(
            model_path=self.config.model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization
        )
        
        self.engine = InferenceEngine(model_config)
        await self.engine.initialize()
        
        logger.info("nano-vllm server started successfully")
    
    async def shutdown(self):
        """关闭服务器"""
        logger.info("Shutting down nano-vllm server...")
        
        if self.engine:
            await self.engine.shutdown()
        
        logger.info("nano-vllm server shutdown complete")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="nano-vllm HTTP API Server")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    # 模型配置
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # 日志配置
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--log-file", type=str, help="Log file path")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    # 创建服务器配置
    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 创建并启动服务器
    server = NanoVLLMServer(config)
    await server.startup()
    
    # 运行服务器
    uvicorn_config = uvicorn.Config(
        app=server.app,
        host=config.host,
        port=config.port,
        workers=args.workers,
        log_level=args.log_level.lower()
    )
    
    server_instance = uvicorn.Server(uvicorn_config)
    await server_instance.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 命令行接口

```python
#!/usr/bin/env python3
"""
nano-vllm 命令行接口
提供交互式文本生成功能
"""

import asyncio
import argparse
import sys
from typing import Optional, List
import readline  # 启用命令行编辑功能

from nano_vllm.engine import InferenceEngine
from nano_vllm.config import ModelConfig
from nano_vllm.utils.logging import setup_logging

class InteractiveCLI:
    """交互式命令行界面"""
    
    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.conversation_history = []
        self.commands = {
            '/help': self.show_help,
            '/clear': self.clear_history,
            '/history': self.show_history,
            '/stats': self.show_stats,
            '/quit': self.quit,
            '/exit': self.quit
        }
    
    async def run(self):
        """运行交互式界面"""
        print("🚀 Welcome to nano-vllm Interactive CLI!")
        print("Type '/help' for available commands or start typing your prompt.")
        print("Press Ctrl+C or type '/quit' to exit.\n")
        
        try:
            while True:
                try:
                    # 获取用户输入
                    prompt = input(">>> ").strip()
                    
                    if not prompt:
                        continue
                    
                    # 处理命令
                    if prompt.startswith('/'):
                        await self.handle_command(prompt)
                    else:
                        # 生成响应
                        await self.generate_response(prompt)
                
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        finally:
            await self.engine.shutdown()
    
    async def handle_command(self, command: str):
        """处理命令"""
        cmd = command.split()[0].lower()
        
        if cmd in self.commands:
            await self.commands[cmd]()
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type '/help' for available commands.")
    
    async def generate_response(self, prompt: str):
        """生成响应"""
        print("🤖 Generating response...")
        
        try:
            # 添加到对话历史
            self.conversation_history.append({"role": "user", "content": prompt})
            
            # 构建完整提示
            full_prompt = self.build_prompt()
            
            # 生成响应
            result = await self.engine.generate(
                prompt=full_prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "Human:"]
            )
            
            response = result.text.strip()
            
            # 添加到对话历史
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # 显示响应
            print(f"\n🤖 Assistant: {response}\n")
            
            # 显示统计信息
            print(f"📊 Tokens - Prompt: {result.prompt_tokens}, "
                  f"Completion: {result.completion_tokens}, "
                  f"Total: {result.total_tokens}")
            print(f"⏱️  Generation time: {result.generation_time:.2f}s")
            print()
        
        except Exception as e:
            print(f"❌ Generation error: {e}\n")
    
    def build_prompt(self) -> str:
        """构建完整提示"""
        prompt_parts = []
        
        for message in self.conversation_history[-10:]:  # 保留最近10轮对话
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def show_help(self):
        """显示帮助信息"""
        help_text = """
📖 Available Commands:
  /help     - Show this help message
  /clear    - Clear conversation history
  /history  - Show conversation history
  /stats    - Show engine statistics
  /quit     - Exit the CLI
  /exit     - Exit the CLI

💡 Tips:
  - Type your prompt and press Enter to generate a response
  - Use Ctrl+C to interrupt generation
  - The CLI maintains conversation context automatically
        """
        print(help_text)
    
    async def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        print("✅ Conversation history cleared.")
    
    async def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("📝 No conversation history.")
            return
        
        print("📝 Conversation History:")
        print("-" * 50)
        
        for i, message in enumerate(self.conversation_history, 1):
            role = message["role"].title()
            content = message["content"]
            print(f"{i}. {role}: {content}")
        
        print("-" * 50)
    
    async def show_stats(self):
        """显示引擎统计"""
        try:
            stats = await self.engine.get_metrics()
            
            print("📊 Engine Statistics:")
            print("-" * 30)
            print(f"Model: {stats.get('model_name', 'Unknown')}")
            print(f"Total requests: {stats.get('total_requests', 0)}")
            print(f"Average latency: {stats.get('avg_latency', 0):.2f}s")
            print(f"GPU memory usage: {stats.get('gpu_memory_usage', 0):.1f}%")
            print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
            print("-" * 30)
        
        except Exception as e:
            print(f"❌ Failed to get statistics: {e}")
    
    async def quit(self):
        """退出CLI"""
        print("👋 Goodbye!")
        sys.exit(0)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="nano-vllm Interactive CLI")
    
    # 模型配置
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # 生成配置
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    
    # 日志配置
    parser.add_argument("--log-level", type=str, default="WARNING", help="Log level")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    
    # 创建模型配置
    config = ModelConfig(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 初始化推理引擎
    print("🔄 Initializing nano-vllm engine...")
    engine = InferenceEngine(config)
    await engine.initialize()
    print("✅ Engine initialized successfully!\n")
    
    # 启动交互式CLI
    cli = InteractiveCLI(engine)
    await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 批处理脚本

```python
#!/usr/bin/env python3
"""
nano-vllm 批处理脚本
用于批量处理文本生成任务
"""

import asyncio
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import time

from nano_vllm.engine import InferenceEngine
from nano_vllm.config import ModelConfig
from nano_vllm.utils.logging import setup_logging

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, engine: InferenceEngine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config
        self.results = []
    
    async def process_file(self, input_file: Path, output_file: Path):
        """处理文件"""
        print(f"📂 Processing file: {input_file}")
        
        # 读取输入数据
        prompts = self.load_prompts(input_file)
        print(f"📝 Loaded {len(prompts)} prompts")
        
        # 批量处理
        start_time = time.time()
        
        if self.config.get('concurrent', False):
            results = await self.process_concurrent(prompts)
        else:
            results = await self.process_sequential(prompts)
        
        end_time = time.time()
        
        # 保存结果
        self.save_results(results, output_file)
        
        # 显示统计
        total_time = end_time - start_time
        avg_time = total_time / len(results) if results else 0
        
        print(f"✅ Processing complete!")
        print(f"📊 Statistics:")
        print(f"   - Total prompts: {len(prompts)}")
        print(f"   - Successful: {len(results)}")
        print(f"   - Total time: {total_time:.2f}s")
        print(f"   - Average time per prompt: {avg_time:.2f}s")
        print(f"   - Output saved to: {output_file}")
    
    def load_prompts(self, input_file: Path) -> List[Dict[str, Any]]:
        """加载提示数据"""
        prompts = []
        
        if input_file.suffix.lower() == '.json':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    prompts = [data]
        
        elif input_file.suffix.lower() == '.csv':
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                prompts = list(reader)
        
        elif input_file.suffix.lower() == '.txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                prompts = [{'prompt': line.strip()} for line in lines if line.strip()]
        
        else:
            raise ValueError(f"Unsupported file format: {input_file.suffix}")
        
        # 标准化提示格式
        normalized_prompts = []
        for i, prompt_data in enumerate(prompts):
            if isinstance(prompt_data, str):
                normalized_prompts.append({
                    'id': i,
                    'prompt': prompt_data,
                    'max_tokens': self.config.get('max_tokens', 512),
                    'temperature': self.config.get('temperature', 0.7),
                    'top_p': self.config.get('top_p', 0.9)
                })
            elif isinstance(prompt_data, dict):
                normalized_prompts.append({
                    'id': prompt_data.get('id', i),
                    'prompt': prompt_data.get('prompt', ''),
                    'max_tokens': prompt_data.get('max_tokens', self.config.get('max_tokens', 512)),
                    'temperature': prompt_data.get('temperature', self.config.get('temperature', 0.7)),
                    'top_p': prompt_data.get('top_p', self.config.get('top_p', 0.9))
                })
        
        return normalized_prompts
    
    async def process_sequential(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """顺序处理"""
        results = []
        
        for i, prompt_data in enumerate(prompts):
            print(f"🔄 Processing {i+1}/{len(prompts)}: {prompt_data['prompt'][:50]}...")
            
            try:
                result = await self.engine.generate(
                    prompt=prompt_data['prompt'],
                    max_tokens=prompt_data['max_tokens'],
                    temperature=prompt_data['temperature'],
                    top_p=prompt_data['top_p']
                )
                
                results.append({
                    'id': prompt_data['id'],
                    'prompt': prompt_data['prompt'],
                    'response': result.text,
                    'prompt_tokens': result.prompt_tokens,
                    'completion_tokens': result.completion_tokens,
                    'total_tokens': result.total_tokens,
                    'generation_time': result.generation_time,
                    'finish_reason': result.finish_reason,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"❌ Error processing prompt {i+1}: {e}")
                results.append({
                    'id': prompt_data['id'],
                    'prompt': prompt_data['prompt'],
                    'response': '',
                    'error': str(e),
                    'status': 'error'
                })
        
        return results
    
    async def process_concurrent(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并发处理"""
        semaphore = asyncio.Semaphore(self.config.get('max_concurrent', 5))
        
        async def process_single(prompt_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.engine.generate(
                        prompt=prompt_data['prompt'],
                        max_tokens=prompt_data['max_tokens'],
                        temperature=prompt_data['temperature'],
                        top_p=prompt_data['top_p']
                    )
                    
                    return {
                        'id': prompt_data['id'],
                        'prompt': prompt_data['prompt'],
                        'response': result.text,
                        'prompt_tokens': result.prompt_tokens,
                        'completion_tokens': result.completion_tokens,
                        'total_tokens': result.total_tokens,
                        'generation_time': result.generation_time,
                        'finish_reason': result.finish_reason,
                        'status': 'success'
                    }
                    
                except Exception as e:
                    return {
                        'id': prompt_data['id'],
                        'prompt': prompt_data['prompt'],
                        'response': '',
                        'error': str(e),
                        'status': 'error'
                    }
        
        # 并发执行
        tasks = [process_single(prompt_data) for prompt_data in prompts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: Path):
        """保存结果"""
        if output_file.suffix.lower() == '.json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        elif output_file.suffix.lower() == '.csv':
            if results:
                fieldnames = results[0].keys()
                with open(output_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
        
        else:
            # 默认保存为JSON
            output_file = output_file.with_suffix('.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="nano-vllm Batch Processor")
    
    # 输入输出
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    # 模型配置
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    
    # 生成配置
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    
    # 处理配置
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent processing")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent requests")
    
    # 日志配置
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    return parser.parse_args()

async def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    
    # 验证输入文件
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    # 创建输出目录
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建模型配置
    model_config = ModelConfig(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # 创建处理配置
    process_config = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'concurrent': args.concurrent,
        'max_concurrent': args.max_concurrent
    }
    
    # 初始化推理引擎
    print("🔄 Initializing nano-vllm engine...")
    engine = InferenceEngine(model_config)
    await engine.initialize()
    print("✅ Engine initialized successfully!")
    
    try:
        # 创建批处理器并处理文件
        processor = BatchProcessor(engine, process_config)
        await processor.process_file(input_file, output_file)
    
    finally:
        # 清理资源
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 入口点配置

### 配置文件示例

```yaml
# nano-vllm 配置文件
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  cors_origins: ["*"]

model:
  model_path: "/path/to/model"
  tensor_parallel_size: 1
  max_model_len: 2048
  gpu_memory_utilization: 0.9
  dtype: "float16"

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.0

logging:
  level: "INFO"
  file: "/var/log/nano-vllm.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

metrics:
  enabled: true
  port: 9090
  path: "/metrics"
```

## 📊 入口点对比

| 入口点 | 适用场景 | 优势 | 限制 |
|--------|----------|------|------|
| HTTP API | 生产环境、微服务 | 标准化接口、易集成 | 网络开销 |
| CLI | 开发测试、交互使用 | 简单直观、快速验证 | 单用户 |
| 批处理 | 大规模处理、离线任务 | 高效批处理、资源优化 | 非实时 |

## 🚀 启动示例

```bash
# HTTP API 服务器
python -m nano_vllm.api.server --model /path/to/model --port 8000

# 交互式CLI
python -m nano_vllm.cli --model /path/to/model

# 批处理
python -m nano_vllm.batch --model /path/to/model --input prompts.json --output results.json
```

---

*通过理解这些入口点，你可以根据不同的使用场景选择最适合的方式来使用 nano-vllm，从而最大化系统的效率和便利性。*