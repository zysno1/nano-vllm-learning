#!/usr/bin/env python3
"""
NanoVLLM 完整演示脚本

这个脚本展示了如何使用完整的 nano-vLLM 系统进行文本生成推理。
包含单个请求、批量请求、性能测试等多种演示场景。
"""

import os
import sys
import time
import asyncio
import argparse
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from nano_vllm import NanoVLLM, GenerationParams, InferenceResponse
    from config import get_config, print_config, validate_config
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保所有依赖文件都在当前目录中")
    sys.exit(1)

# ================================
# 演示场景
# ================================

class DemoScenarios:
    """演示场景集合"""
    
    @staticmethod
    def basic_generation_demo(nano_vllm: NanoVLLM):
        """基础文本生成演示"""
        print("\n" + "="*50)
        print("🎯 基础文本生成演示")
        print("="*50)
        
        test_prompts = [
            "Hello, how are you today?",
            "What is artificial intelligence?",
            "Explain the concept of machine learning.",
            "Tell me a short story about a robot.",
            "How does natural language processing work?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 测试 {i}/5: {prompt}")
            print("-" * 40)
            
            start_time = time.time()
            
            response = nano_vllm.generate(
                prompt=prompt,
                params=GenerationParams(
                    max_tokens=50,
                    temperature=0.8,
                    top_p=0.9
                )
            )
            
            end_time = time.time()
            
            print(f"✅ 生成结果:")
            print(f"   📄 文本: {response.generated_text}")
            print(f"   📊 统计: {response.prompt_tokens} + {response.completion_tokens} = {response.total_tokens} tokens")
            print(f"   ⏱️  时间: {end_time - start_time:.2f}秒")
            print(f"   🚀 速度: {response.tokens_per_second:.1f} tokens/s")
            print(f"   🏁 结束原因: {response.finish_reason}")
    
    @staticmethod
    def batch_processing_demo(nano_vllm: NanoVLLM):
        """批量处理演示"""
        print("\n" + "="*50)
        print("📦 批量处理演示")
        print("="*50)
        
        batch_prompts = [
            "Describe the benefits of renewable energy.",
            "What are the main challenges in space exploration?",
            "Explain quantum computing in simple terms.",
            "How do neural networks learn?",
            "What is the future of autonomous vehicles?",
            "Describe the impact of social media on society.",
            "What are the principles of sustainable development?",
            "How does blockchain technology work?"
        ]
        
        print(f"📋 处理 {len(batch_prompts)} 个请求...")
        
        batch_start = time.time()
        responses = []
        
        # 顺序处理
        print("\n🔄 顺序处理模式:")
        for i, prompt in enumerate(batch_prompts, 1):
            print(f"   处理请求 {i}/{len(batch_prompts)}: {prompt[:30]}...")
            
            response = nano_vllm.generate(
                prompt=prompt,
                params=GenerationParams(
                    max_tokens=40,
                    temperature=0.7
                )
            )
            responses.append(response)
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        
        # 统计结果
        total_tokens = sum(r.total_tokens for r in responses)
        avg_latency = sum(r.generation_time for r in responses) / len(responses)
        throughput = total_tokens / batch_time
        
        print(f"\n📊 批量处理结果:")
        print(f"   📝 请求数量: {len(responses)}")
        print(f"   📄 总token数: {total_tokens}")
        print(f"   ⏱️  总时间: {batch_time:.2f}秒")
        print(f"   📈 平均延迟: {avg_latency:.2f}秒")
        print(f"   🚀 整体吞吐量: {throughput:.1f} tokens/s")
        
        # 显示部分结果
        print(f"\n📋 部分生成结果:")
        for i, response in enumerate(responses[:3]):
            print(f"   {i+1}. {response.generated_text[:60]}...")
    
    @staticmethod
    def parameter_comparison_demo(nano_vllm: NanoVLLM):
        """参数对比演示"""
        print("\n" + "="*50)
        print("⚙️ 生成参数对比演示")
        print("="*50)
        
        test_prompt = "Write a creative story about a time traveler who"
        
        parameter_sets = [
            {"name": "保守生成", "params": GenerationParams(max_tokens=60, temperature=0.3, top_p=0.8)},
            {"name": "平衡生成", "params": GenerationParams(max_tokens=60, temperature=0.7, top_p=0.9)},
            {"name": "创意生成", "params": GenerationParams(max_tokens=60, temperature=1.0, top_p=0.95)},
            {"name": "随机生成", "params": GenerationParams(max_tokens=60, temperature=1.2, top_p=1.0)},
        ]
        
        print(f"📝 测试提示: {test_prompt}")
        print("-" * 50)
        
        for param_set in parameter_sets:
            print(f"\n🎛️  {param_set['name']}:")
            print(f"   参数: temp={param_set['params'].temperature}, top_p={param_set['params'].top_p}")
            
            response = nano_vllm.generate(
                prompt=test_prompt,
                params=param_set['params']
            )
            
            print(f"   结果: {response.generated_text}")
            print(f"   速度: {response.tokens_per_second:.1f} tokens/s")
    
    @staticmethod
    def performance_benchmark(nano_vllm: NanoVLLM):
        """性能基准测试"""
        print("\n" + "="*50)
        print("🏃‍♂️ 性能基准测试")
        print("="*50)
        
        # 不同长度的提示测试
        test_cases = [
            {"name": "短提示", "prompt": "Hello", "max_tokens": 20},
            {"name": "中等提示", "prompt": "Explain the concept of artificial intelligence and its applications in modern technology", "max_tokens": 50},
            {"name": "长提示", "prompt": "Write a detailed analysis of the impact of machine learning on various industries including healthcare, finance, transportation, and education. Discuss both the benefits and challenges.", "max_tokens": 100},
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n🧪 测试: {test_case['name']}")
            print(f"   提示长度: {len(test_case['prompt'])} 字符")
            print(f"   目标生成: {test_case['max_tokens']} tokens")
            
            # 多次运行取平均
            run_times = []
            token_speeds = []
            
            for run in range(3):
                print(f"   运行 {run+1}/3...", end=" ")
                
                start_time = time.time()
                response = nano_vllm.generate(
                    prompt=test_case['prompt'],
                    params=GenerationParams(
                        max_tokens=test_case['max_tokens'],
                        temperature=0.7
                    )
                )
                end_time = time.time()
                
                run_time = end_time - start_time
                run_times.append(run_time)
                token_speeds.append(response.tokens_per_second)
                
                print(f"{run_time:.2f}s, {response.tokens_per_second:.1f} tokens/s")
            
            # 计算平均值
            avg_time = sum(run_times) / len(run_times)
            avg_speed = sum(token_speeds) / len(token_speeds)
            
            results.append({
                "name": test_case['name'],
                "avg_time": avg_time,
                "avg_speed": avg_speed,
                "prompt_length": len(test_case['prompt']),
                "max_tokens": test_case['max_tokens']
            })
            
            print(f"   📊 平均时间: {avg_time:.2f}秒")
            print(f"   📊 平均速度: {avg_speed:.1f} tokens/s")
        
        # 性能总结
        print(f"\n📈 性能基准测试总结:")
        print("-" * 40)
        for result in results:
            print(f"{result['name']:12} | {result['avg_time']:6.2f}s | {result['avg_speed']:6.1f} tokens/s")
    
    @staticmethod
    def concurrent_requests_demo(nano_vllm: NanoVLLM):
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
            "How does the internet work?",
            "What is DNA?",
            "Explain climate change."
        ]
        
        print(f"🚀 启动 {len(concurrent_prompts)} 个并发请求...")
        
        def process_request(prompt_data):
            idx, prompt = prompt_data
            print(f"   🔄 开始处理请求 {idx+1}: {prompt[:30]}...")
            
            start_time = time.time()
            response = nano_vllm.generate(
                prompt=prompt,
                params=GenerationParams(max_tokens=30, temperature=0.8)
            )
            end_time = time.time()
            
            return {
                "index": idx,
                "prompt": prompt,
                "response": response,
                "time": end_time - start_time
            }
        
        # 使用线程池并发处理
        concurrent_start = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_request, (i, prompt)) 
                      for i, prompt in enumerate(concurrent_prompts)]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"   ✅ 完成请求 {result['index']+1}: {result['time']:.2f}s")
        
        concurrent_end = time.time()
        concurrent_time = concurrent_end - concurrent_start
        
        # 排序结果
        results.sort(key=lambda x: x['index'])
        
        # 统计
        total_tokens = sum(r['response'].total_tokens for r in results)
        avg_latency = sum(r['time'] for r in results) / len(results)
        throughput = total_tokens / concurrent_time
        
        print(f"\n📊 并发处理结果:")
        print(f"   📝 请求数量: {len(results)}")
        print(f"   ⏱️  总时间: {concurrent_time:.2f}秒")
        print(f"   📈 平均延迟: {avg_latency:.2f}秒")
        print(f"   📄 总token数: {total_tokens}")
        print(f"   🚀 整体吞吐量: {throughput:.1f} tokens/s")
    
    @staticmethod
    def system_monitoring_demo(nano_vllm: NanoVLLM):
        """系统监控演示"""
        print("\n" + "="*50)
        print("📊 系统监控演示")
        print("="*50)
        
        # 执行一些请求来生成指标
        print("🔄 执行测试请求以生成监控数据...")
        
        test_prompts = [
            "Monitor test 1: System status check",
            "Monitor test 2: Performance evaluation",
            "Monitor test 3: Resource utilization",
            "Monitor test 4: Health assessment",
            "Monitor test 5: Metrics collection"
        ]
        
        for prompt in test_prompts:
            nano_vllm.generate(
                prompt=prompt,
                params=GenerationParams(max_tokens=20, temperature=0.7)
            )
        
        # 获取系统指标
        print("\n📈 系统性能指标:")
        metrics = nano_vllm.get_metrics()
        
        print(f"   🎯 请求统计:")
        print(f"      总请求数: {metrics.total_requests}")
        print(f"      成功请求: {metrics.successful_requests}")
        print(f"      失败请求: {metrics.failed_requests}")
        print(f"      成功率: {(metrics.successful_requests/metrics.total_requests*100):.1f}%")
        
        print(f"   ⚡ 性能指标:")
        print(f"      请求/秒: {metrics.requests_per_second:.2f}")
        print(f"      tokens/秒: {metrics.tokens_per_second:.2f}")
        print(f"      平均延迟: {metrics.avg_latency:.3f}秒")
        print(f"      P95延迟: {metrics.p95_latency:.3f}秒")
        print(f"      P99延迟: {metrics.p99_latency:.3f}秒")
        
        print(f"   💾 内存指标:")
        print(f"      GPU内存使用: {metrics.gpu_memory_used:.2f}GB / {metrics.gpu_memory_total:.2f}GB")
        print(f"      GPU内存利用率: {metrics.gpu_memory_utilization:.1%}")
        print(f"      KV Cache使用率: {metrics.kv_cache_usage:.1%}")
        
        print(f"   📋 队列状态:")
        print(f"      等待请求: {metrics.waiting_requests}")
        print(f"      运行请求: {metrics.running_requests}")
        print(f"      交换请求: {metrics.swapped_requests}")
        
        # 健康状态检查
        print("\n🏥 系统健康状态:")
        health = nano_vllm.get_health_status()
        
        status_icon = "✅" if health['healthy'] else "❌"
        print(f"   {status_icon} 整体状态: {'健康' if health['healthy'] else '异常'}")
        
        if health.get('issues'):
            print(f"   ⚠️  发现问题:")
            for issue in health['issues']:
                print(f"      - {issue}")
        
        print(f"   ⏱️  运行时间: {health.get('uptime', 0):.1f}秒")
        print(f"   🧠 模型状态: {'已加载' if health.get('model_loaded') else '未加载'}")
        print(f"   🔄 引擎状态: {'运行中' if health.get('engine_running') else '已停止'}")

# ================================
# 主演示函数
# ================================

def run_interactive_demo(nano_vllm: NanoVLLM):
    """交互式演示"""
    print("\n" + "="*50)
    print("💬 交互式对话演示")
    print("="*50)
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not user_input:
                continue
            
            print("🤖 AI: ", end="", flush=True)
            
            start_time = time.time()
            response = nano_vllm.generate(
                prompt=user_input,
                params=GenerationParams(
                    max_tokens=100,
                    temperature=0.8,
                    top_p=0.9
                )
            )
            end_time = time.time()
            
            print(response.generated_text)
            print(f"    (⏱️ {end_time-start_time:.2f}s, 🚀 {response.tokens_per_second:.1f} tokens/s)")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="NanoVLLM 完整演示")
    parser.add_argument("--config", default="dev", help="配置名称 (dev/test/prod_throughput/prod_latency)")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", help="模型名称")
    parser.add_argument("--scenario", choices=[
        "all", "basic", "batch", "params", "benchmark", 
        "concurrent", "monitoring", "interactive"
    ], default="all", help="演示场景")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    print("🚀 NanoVLLM 完整演示系统")
    print("=" * 60)
    
    # 加载配置
    print(f"📋 加载配置: {args.config}")
    config = get_config(args.config)
    
    if args.verbose:
        print_config(config)
    
    if not validate_config(config):
        print("❌ 配置验证失败")
        return
    
    # 转换配置为字典
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
    
    # 创建并运行系统
    print(f"\n🔧 初始化NanoVLLM系统...")
    
    try:
        with NanoVLLM(config_dict) as nano_vllm:
            # 加载模型
            print(f"📦 加载模型: {args.model}")
            if not nano_vllm.load_model(args.model):
                print("❌ 模型加载失败")
                return
            
            print("✅ 系统初始化完成！")
            
            # 运行演示场景
            scenarios = {
                "basic": DemoScenarios.basic_generation_demo,
                "batch": DemoScenarios.batch_processing_demo,
                "params": DemoScenarios.parameter_comparison_demo,
                "benchmark": DemoScenarios.performance_benchmark,
                "concurrent": DemoScenarios.concurrent_requests_demo,
                "monitoring": DemoScenarios.system_monitoring_demo,
                "interactive": run_interactive_demo,
            }
            
            if args.scenario == "all":
                # 运行所有演示（除了交互式）
                for name, func in scenarios.items():
                    if name != "interactive":
                        try:
                            func(nano_vllm)
                        except Exception as e:
                            print(f"❌ 演示 {name} 失败: {e}")
                
                # 询问是否进入交互模式
                try:
                    choice = input("\n🤔 是否进入交互式对话模式？(y/N): ").strip().lower()
                    if choice in ['y', 'yes']:
                        run_interactive_demo(nano_vllm)
                except KeyboardInterrupt:
                    pass
            
            elif args.scenario in scenarios:
                scenarios[args.scenario](nano_vllm)
            
            else:
                print(f"❌ 未知演示场景: {args.scenario}")
            
            # 最终系统状态
            print("\n" + "="*50)
            print("📊 最终系统状态")
            print("="*50)
            
            metrics = nano_vllm.get_metrics()
            print(f"总处理请求: {metrics.total_requests}")
            print(f"平均吞吐量: {metrics.tokens_per_second:.1f} tokens/s")
            print(f"系统成功率: {(metrics.successful_requests/max(metrics.total_requests,1)*100):.1f}%")
            
            health = nano_vllm.get_health_status()
            print(f"系统健康度: {'✅ 健康' if health['healthy'] else '❌ 异常'}")
            
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\n🎉 演示完成！感谢使用 NanoVLLM！")

if __name__ == "__main__":
    main()