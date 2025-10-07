#!/usr/bin/env python3
"""
NanoVLLM API 客户端示例

展示如何使用 Python 调用 NanoVLLM API 服务器的各种功能。
包括单个请求、批量请求、流式请求、系统监控等。
"""

import json
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    import aiohttp
    import sseclient  # pip install sseclient-py
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请安装依赖: pip install requests aiohttp sseclient-py")
    exit(1)

# ================================
# 客户端类
# ================================

class NanoVLLMClient:
    """NanoVLLM API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'NanoVLLM-Client/1.0'
        })
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """单个文本生成"""
        url = f"{self.base_url}/v1/generate"
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "request_id": kwargs.get("request_id")
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_stream(self, prompt: str, **kwargs):
        """流式文本生成"""
        url = f"{self.base_url}/v1/generate/stream"
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "stream": True,
            "request_id": kwargs.get("request_id")
        }
        
        response = self.session.post(url, json=data, stream=True)
        response.raise_for_status()
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data:
                yield json.loads(event.data)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """批量文本生成"""
        url = f"{self.base_url}/v1/generate/batch"
        
        data = {
            "prompts": prompts,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0)
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        url = f"{self.base_url}/v1/metrics"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_health(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}/v1/health"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """列出可用模型"""
        url = f"{self.base_url}/v1/models"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """关闭会话"""
        self.session.close()

# ================================
# 异步客户端类
# ================================

class AsyncNanoVLLMClient:
    """异步 NanoVLLM API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'NanoVLLM-AsyncClient/1.0'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """异步单个文本生成"""
        url = f"{self.base_url}/v1/generate"
        
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "request_id": kwargs.get("request_id")
        }
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> Dict[str, Any]:
        """异步批量文本生成"""
        url = f"{self.base_url}/v1/generate/batch"
        
        data = {
            "prompts": prompts,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.8),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0)
        }
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """异步获取系统指标"""
        url = f"{self.base_url}/v1/metrics"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_health(self) -> Dict[str, Any]:
        """异步健康检查"""
        url = f"{self.base_url}/v1/health"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

# ================================
# 演示函数
# ================================

def demo_basic_generation(client: NanoVLLMClient):
    """基础生成演示"""
    print("\n" + "="*50)
    print("🎯 基础文本生成演示")
    print("="*50)
    
    test_prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "Tell me a joke.",
        "Explain quantum computing.",
        "Write a short poem."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 测试 {i}/5: {prompt}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = client.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.8
            )
            end_time = time.time()
            
            print(f"✅ 生成结果:")
            print(f"   📄 文本: {response['generated_text']}")
            print(f"   📊 统计: {response['prompt_tokens']} + {response['completion_tokens']} = {response['total_tokens']} tokens")
            print(f"   ⏱️  时间: {end_time - start_time:.2f}秒")
            print(f"   🚀 速度: {response['tokens_per_second']:.1f} tokens/s")
            print(f"   🏁 结束原因: {response['finish_reason']}")
            
        except Exception as e:
            print(f"❌ 错误: {e}")

def demo_stream_generation(client: NanoVLLMClient):
    """流式生成演示"""
    print("\n" + "="*50)
    print("🌊 流式文本生成演示")
    print("="*50)
    
    prompt = "Write a creative story about a robot learning to paint"
    print(f"📝 提示: {prompt}")
    print("🔄 流式生成中...")
    print("-" * 40)
    
    try:
        full_text = ""
        chunk_count = 0
        
        for chunk in client.generate_stream(
            prompt=prompt,
            max_tokens=100,
            temperature=0.9
        ):
            if 'chunk' in chunk and chunk['chunk']:
                print(chunk['chunk'], end='', flush=True)
                full_text += chunk['chunk']
                chunk_count += 1
            
            if chunk.get('is_final'):
                print(f"\n\n✅ 流式生成完成!")
                print(f"   📊 总块数: {chunk_count}")
                if 'total_tokens' in chunk:
                    print(f"   📄 总tokens: {chunk['total_tokens']}")
                    print(f"   ⏱️  时间: {chunk.get('generation_time', 0):.2f}秒")
                    print(f"   🚀 速度: {chunk.get('tokens_per_second', 0):.1f} tokens/s")
                break
                
    except Exception as e:
        print(f"\n❌ 流式生成错误: {e}")

def demo_batch_generation(client: NanoVLLMClient):
    """批量生成演示"""
    print("\n" + "="*50)
    print("📦 批量文本生成演示")
    print("="*50)
    
    batch_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks.",
        "What is deep learning?",
        "Describe natural language processing."
    ]
    
    print(f"📋 批量处理 {len(batch_prompts)} 个请求...")
    
    try:
        start_time = time.time()
        response = client.generate_batch(
            prompts=batch_prompts,
            max_tokens=40,
            temperature=0.7
        )
        end_time = time.time()
        
        print(f"\n📊 批量生成结果:")
        print(f"   📝 请求数量: {len(response['results'])}")
        print(f"   📄 总tokens: {response['total_tokens']}")
        print(f"   ⏱️  总时间: {response['total_time']:.2f}秒")
        print(f"   🚀 平均速度: {response['average_tokens_per_second']:.1f} tokens/s")
        print(f"   📈 客户端时间: {end_time - start_time:.2f}秒")
        
        print(f"\n📋 部分生成结果:")
        for i, result in enumerate(response['results'][:3]):
            print(f"   {i+1}. {result['generated_text'][:60]}...")
            
    except Exception as e:
        print(f"❌ 批量生成错误: {e}")

def demo_concurrent_requests(client: NanoVLLMClient):
    """并发请求演示"""
    print("\n" + "="*50)
    print("🔀 并发请求演示")
    print("="*50)
    
    concurrent_prompts = [
        "What is the capital of France?",
        "How do computers work?",
        "Explain photosynthesis.",
        "What is quantum mechanics?",
        "Describe the solar system.",
        "How does the internet work?"
    ]
    
    print(f"🚀 启动 {len(concurrent_prompts)} 个并发请求...")
    
    def make_request(prompt_data):
        idx, prompt = prompt_data
        try:
            start_time = time.time()
            response = client.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.8
            )
            end_time = time.time()
            
            return {
                "index": idx,
                "prompt": prompt,
                "response": response,
                "time": end_time - start_time,
                "success": True
            }
        except Exception as e:
            return {
                "index": idx,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    try:
        concurrent_start = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(make_request, (i, prompt)) 
                      for i, prompt in enumerate(concurrent_prompts)]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"   ✅ 完成请求 {result['index']+1}: {result['time']:.2f}s")
                else:
                    print(f"   ❌ 请求 {result['index']+1} 失败: {result['error']}")
        
        concurrent_end = time.time()
        concurrent_time = concurrent_end - concurrent_start
        
        # 统计成功的请求
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            total_tokens = sum(r['response']['total_tokens'] for r in successful_results)
            avg_latency = sum(r['time'] for r in successful_results) / len(successful_results)
            throughput = total_tokens / concurrent_time
            
            print(f"\n📊 并发处理结果:")
            print(f"   📝 总请求数: {len(results)}")
            print(f"   ✅ 成功请求: {len(successful_results)}")
            print(f"   ❌ 失败请求: {len(results) - len(successful_results)}")
            print(f"   ⏱️  总时间: {concurrent_time:.2f}秒")
            print(f"   📈 平均延迟: {avg_latency:.2f}秒")
            print(f"   📄 总tokens: {total_tokens}")
            print(f"   🚀 整体吞吐量: {throughput:.1f} tokens/s")
        
    except Exception as e:
        print(f"❌ 并发请求错误: {e}")

def demo_system_monitoring(client: NanoVLLMClient):
    """系统监控演示"""
    print("\n" + "="*50)
    print("📊 系统监控演示")
    print("="*50)
    
    try:
        # 健康检查
        print("🏥 健康状态检查:")
        health = client.get_health()
        
        status_icon = "✅" if health['healthy'] else "❌"
        print(f"   {status_icon} 整体状态: {'健康' if health['healthy'] else '异常'}")
        print(f"   ⏱️  运行时间: {health['uptime']:.1f}秒")
        print(f"   🧠 模型状态: {'已加载' if health['model_loaded'] else '未加载'}")
        print(f"   🔄 引擎状态: {'运行中' if health['engine_running'] else '已停止'}")
        
        if health.get('issues'):
            print(f"   ⚠️  发现问题:")
            for issue in health['issues']:
                print(f"      - {issue}")
        
        # 系统指标
        print(f"\n📈 系统性能指标:")
        metrics = client.get_metrics()
        
        print(f"   🎯 请求统计:")
        print(f"      总请求数: {metrics['total_requests']}")
        print(f"      成功请求: {metrics['successful_requests']}")
        print(f"      失败请求: {metrics['failed_requests']}")
        if metrics['total_requests'] > 0:
            success_rate = metrics['successful_requests'] / metrics['total_requests'] * 100
            print(f"      成功率: {success_rate:.1f}%")
        
        print(f"   ⚡ 性能指标:")
        print(f"      请求/秒: {metrics['requests_per_second']:.2f}")
        print(f"      tokens/秒: {metrics['tokens_per_second']:.2f}")
        print(f"      平均延迟: {metrics['avg_latency']:.3f}秒")
        print(f"      P95延迟: {metrics['p95_latency']:.3f}秒")
        print(f"      P99延迟: {metrics['p99_latency']:.3f}秒")
        
        print(f"   💾 内存指标:")
        print(f"      GPU内存使用: {metrics['gpu_memory_used']:.2f}GB / {metrics['gpu_memory_total']:.2f}GB")
        print(f"      GPU内存利用率: {metrics['gpu_memory_utilization']:.1%}")
        print(f"      KV Cache使用率: {metrics['kv_cache_usage']:.1%}")
        
        print(f"   📋 队列状态:")
        print(f"      等待请求: {metrics['waiting_requests']}")
        print(f"      运行请求: {metrics['running_requests']}")
        print(f"      交换请求: {metrics['swapped_requests']}")
        
    except Exception as e:
        print(f"❌ 监控获取错误: {e}")

def demo_model_listing(client: NanoVLLMClient):
    """模型列表演示"""
    print("\n" + "="*50)
    print("📋 可用模型列表")
    print("="*50)
    
    try:
        models = client.list_models()
        
        print(f"🤖 可用模型 ({len(models['models'])} 个):")
        for model in models['models']:
            print(f"   📦 {model['name']}")
            print(f"      ID: {model['id']}")
            print(f"      描述: {model['description']}")
            print()
            
    except Exception as e:
        print(f"❌ 获取模型列表错误: {e}")

async def demo_async_requests():
    """异步请求演示"""
    print("\n" + "="*50)
    print("⚡ 异步请求演示")
    print("="*50)
    
    async_prompts = [
        "What is async programming?",
        "How does asyncio work?",
        "Explain coroutines.",
        "What are async/await keywords?",
        "Describe event loops."
    ]
    
    try:
        async with AsyncNanoVLLMClient() as client:
            print(f"🚀 启动 {len(async_prompts)} 个异步请求...")
            
            start_time = time.time()
            
            # 并发执行所有请求
            tasks = [
                client.generate(
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0.8
                )
                for prompt in async_prompts
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 处理结果
            successful_responses = []
            failed_count = 0
            
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"   ❌ 请求 {i+1} 失败: {response}")
                    failed_count += 1
                else:
                    successful_responses.append(response)
                    print(f"   ✅ 请求 {i+1} 完成: {response['tokens_per_second']:.1f} tokens/s")
            
            if successful_responses:
                total_tokens = sum(r['total_tokens'] for r in successful_responses)
                avg_speed = sum(r['tokens_per_second'] for r in successful_responses) / len(successful_responses)
                
                print(f"\n📊 异步请求结果:")
                print(f"   📝 总请求数: {len(responses)}")
                print(f"   ✅ 成功请求: {len(successful_responses)}")
                print(f"   ❌ 失败请求: {failed_count}")
                print(f"   ⏱️  总时间: {total_time:.2f}秒")
                print(f"   📄 总tokens: {total_tokens}")
                print(f"   🚀 平均速度: {avg_speed:.1f} tokens/s")
                print(f"   📈 整体吞吐量: {total_tokens / total_time:.1f} tokens/s")
            
    except Exception as e:
        print(f"❌ 异步请求错误: {e}")

def run_interactive_client(client: NanoVLLMClient):
    """交互式客户端"""
    print("\n" + "="*50)
    print("💬 交互式客户端")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'stream' 切换流式模式")
    print("输入 'metrics' 查看系统指标")
    print("输入 'health' 查看健康状态")
    print("-" * 50)
    
    stream_mode = False
    
    while True:
        try:
            user_input = input(f"\n👤 您{'(流式)' if stream_mode else ''}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if user_input.lower() == 'stream':
                stream_mode = not stream_mode
                print(f"🔄 {'启用' if stream_mode else '禁用'}流式模式")
                continue
            
            if user_input.lower() == 'metrics':
                demo_system_monitoring(client)
                continue
            
            if user_input.lower() == 'health':
                health = client.get_health()
                status = "✅ 健康" if health['healthy'] else "❌ 异常"
                print(f"🏥 系统状态: {status}")
                continue
            
            if not user_input:
                continue
            
            print("🤖 AI: ", end="", flush=True)
            
            if stream_mode:
                # 流式响应
                for chunk in client.generate_stream(
                    prompt=user_input,
                    max_tokens=100,
                    temperature=0.8
                ):
                    if 'chunk' in chunk and chunk['chunk']:
                        print(chunk['chunk'], end='', flush=True)
                    
                    if chunk.get('is_final'):
                        if 'tokens_per_second' in chunk:
                            print(f"\n    (🚀 {chunk['tokens_per_second']:.1f} tokens/s)")
                        break
            else:
                # 普通响应
                start_time = time.time()
                response = client.generate(
                    prompt=user_input,
                    max_tokens=100,
                    temperature=0.8
                )
                end_time = time.time()
                
                print(response['generated_text'])
                print(f"    (⏱️ {end_time-start_time:.2f}s, 🚀 {response['tokens_per_second']:.1f} tokens/s)")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")

# ================================
# 主函数
# ================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NanoVLLM API 客户端示例")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器地址")
    parser.add_argument("--demo", choices=[
        "all", "basic", "stream", "batch", "concurrent", 
        "async", "monitoring", "models", "interactive"
    ], default="all", help="演示类型")
    parser.add_argument("--timeout", type=int, default=30, help="请求超时时间")
    
    args = parser.parse_args()
    
    print("🚀 NanoVLLM API 客户端示例")
    print("=" * 60)
    print(f"📍 服务器地址: {args.url}")
    
    # 创建客户端
    client = NanoVLLMClient(args.url)
    client.session.timeout = args.timeout
    
    try:
        # 检查服务器连接
        print("🔍 检查服务器连接...")
        health = client.get_health()
        if health['healthy']:
            print("✅ 服务器连接正常")
        else:
            print("⚠️ 服务器状态异常，但可以继续演示")
        
        # 运行演示
        demos = {
            "basic": lambda: demo_basic_generation(client),
            "stream": lambda: demo_stream_generation(client),
            "batch": lambda: demo_batch_generation(client),
            "concurrent": lambda: demo_concurrent_requests(client),
            "async": lambda: asyncio.run(demo_async_requests()),
            "monitoring": lambda: demo_system_monitoring(client),
            "models": lambda: demo_model_listing(client),
            "interactive": lambda: run_interactive_client(client),
        }
        
        if args.demo == "all":
            # 运行所有演示（除了交互式）
            for name, func in demos.items():
                if name != "interactive":
                    try:
                        print(f"\n🎯 运行演示: {name}")
                        func()
                    except Exception as e:
                        print(f"❌ 演示 {name} 失败: {e}")
            
            # 询问是否进入交互模式
            try:
                choice = input("\n🤔 是否进入交互式客户端？(y/N): ").strip().lower()
                if choice in ['y', 'yes']:
                    run_interactive_client(client)
            except KeyboardInterrupt:
                pass
        
        elif args.demo in demos:
            demos[args.demo]()
        
        else:
            print(f"❌ 未知演示类型: {args.demo}")
    
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 客户端错误: {e}")
    
    finally:
        client.close()
    
    print("\n🎉 客户端演示完成！")

if __name__ == "__main__":
    main()