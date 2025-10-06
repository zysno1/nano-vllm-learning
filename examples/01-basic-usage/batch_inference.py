#!/usr/bin/env python3
"""
🚀 nano-vLLM 批量推理示例

这个示例展示如何使用 nano-vLLM 进行批量文本推理，
包括批量处理、性能优化和结果分析。

📋 学习目标:
- 理解批量推理的优势
- 掌握批量处理的最佳实践
- 学习性能优化技巧
- 了解结果分析方法

🎯 适用场景:
- 大量文本需要处理
- 提高推理效率
- 批量内容生成
- 性能基准测试
"""

import os
import sys
import time
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 导入必要的库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    print("✅ 成功导入 PyTorch 和 Transformers")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请安装必要的依赖: pip install torch transformers")
    sys.exit(1)

class BatchInferenceEngine:
    """批量推理引擎"""
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        """
        初始化批量推理引擎
        
        Args:
            model_name: 模型名称
            device: 计算设备
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.stats = {
            "total_batches": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "batch_times": [],
            "throughput_history": []
        }
        
        print(f"🔧 初始化批量推理引擎")
        print(f"   模型: {self.model_name}")
        print(f"   设备: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎮 自动选择 GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("💻 使用 CPU 进行推理")
        
        return torch.device(device)
    
    def setup(self) -> bool:
        """设置模型和分词器"""
        try:
            print(f"📥 加载模型: {self.model_name}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            print("✅ 模型加载成功")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            batch_size: 批处理大小
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            do_sample: 是否使用采样
            show_progress: 是否显示进度
        
        Returns:
            生成的文本列表
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 setup()")
        
        print(f"🚀 开始批量推理")
        print(f"   输入数量: {len(prompts)}")
        print(f"   批处理大小: {batch_size}")
        print(f"   最大生成长度: {max_new_tokens}")
        
        # 准备生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        start_time = time.time()
        
        # 分批处理
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            current_batch = batch_idx // batch_size + 1
            
            if show_progress:
                print(f"📦 处理批次 {current_batch}/{total_batches} (大小: {len(batch_prompts)})")
            
            batch_start_time = time.time()
            
            # 批量推理
            batch_results = self._process_batch(batch_prompts, generation_config)
            results.extend(batch_results)
            
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            
            # 更新统计信息
            self.stats["batch_times"].append(batch_time)
            throughput = len(batch_prompts) / batch_time
            self.stats["throughput_history"].append(throughput)
            
            if show_progress:
                print(f"   ⏱️  批次时间: {batch_time:.2f}秒")
                print(f"   🚄 吞吐量: {throughput:.2f} 文本/秒")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 更新总体统计
        self.stats["total_batches"] += total_batches
        self.stats["total_texts"] += len(prompts)
        self.stats["total_time"] += total_time
        
        # 计算总token数（近似）
        total_tokens = sum(len(self.tokenizer.encode(result)) for result in results)
        self.stats["total_tokens"] += total_tokens
        
        print(f"✅ 批量推理完成")
        print(f"   总时间: {total_time:.2f}秒")
        print(f"   平均吞吐量: {len(prompts) / total_time:.2f} 文本/秒")
        print(f"   生成token数: {total_tokens}")
        
        return results
    
    def _process_batch(self, prompts: List[str], generation_config: GenerationConfig) -> List[str]:
        """处理单个批次"""
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解码输出
            results = []
            for i, output in enumerate(outputs):
                # 只保留新生成的部分
                input_length = inputs["input_ids"][i].shape[0]
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # 组合完整结果
                full_result = prompts[i] + generated_text
                results.append(full_result)
            
            return results
            
        except Exception as e:
            print(f"❌ 批次处理失败: {e}")
            # 返回原始输入作为fallback
            return prompts
    
    def parallel_generate(
        self,
        prompts: List[str],
        max_workers: int = 2,
        batch_size: int = 2,
        **kwargs
    ) -> List[str]:
        """
        并行批量生成（实验性功能）
        
        Args:
            prompts: 输入提示列表
            max_workers: 最大工作线程数
            batch_size: 每个线程的批处理大小
            **kwargs: 其他生成参数
        
        Returns:
            生成的文本列表
        """
        print(f"🔄 并行批量推理 (工作线程: {max_workers})")
        
        # 将prompts分组
        chunks = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        results = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_index = {
                executor.submit(self._process_batch, chunk, GenerationConfig(**kwargs)): i
                for i, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    chunk_results = future.result()
                    results[index] = chunk_results
                    print(f"✅ 完成块 {index + 1}/{len(chunks)}")
                except Exception as e:
                    print(f"❌ 块 {index + 1} 处理失败: {e}")
                    results[index] = chunks[index]  # fallback
        
        # 展平结果
        flat_results = []
        for chunk_results in results:
            if chunk_results:
                flat_results.extend(chunk_results)
        
        return flat_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if not self.stats["batch_times"]:
            return {"message": "暂无统计数据"}
        
        return {
            "总批次数": self.stats["total_batches"],
            "总文本数": self.stats["total_texts"],
            "总token数": self.stats["total_tokens"],
            "总时间": f"{self.stats['total_time']:.2f}秒",
            "平均批次时间": f"{statistics.mean(self.stats['batch_times']):.2f}秒",
            "最快批次时间": f"{min(self.stats['batch_times']):.2f}秒",
            "最慢批次时间": f"{max(self.stats['batch_times']):.2f}秒",
            "平均吞吐量": f"{statistics.mean(self.stats['throughput_history']):.2f} 文本/秒",
            "峰值吞吐量": f"{max(self.stats['throughput_history']):.2f} 文本/秒",
            "总体吞吐量": f"{self.stats['total_texts'] / self.stats['total_time']:.2f} 文本/秒" if self.stats['total_time'] > 0 else "N/A"
        }
    
    def save_results(self, prompts: List[str], results: List[str], filename: str = "batch_results.json"):
        """保存批量推理结果"""
        data = {
            "metadata": {
                "model_name": self.model_name,
                "device": str(self.device),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_prompts": len(prompts),
                "performance_stats": self.get_performance_stats()
            },
            "results": [
                {
                    "index": i,
                    "prompt": prompt,
                    "generated": result,
                    "prompt_length": len(prompt),
                    "result_length": len(result)
                }
                for i, (prompt, result) in enumerate(zip(prompts, results))
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到: {filename}")
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 资源清理完成")

def create_sample_prompts() -> List[str]:
    """创建示例提示"""
    return [
        "人工智能的未来发展趋势是",
        "深度学习在自然语言处理中的应用包括",
        "机器学习模型的训练过程需要考虑",
        "大语言模型的优势在于",
        "神经网络的基本原理是",
        "自然语言生成技术可以用于",
        "计算机视觉与自然语言处理的结合能够",
        "强化学习在实际应用中的挑战包括",
        "数据预处理在机器学习中的重要性体现在",
        "模型评估的常用指标有",
        "过拟合问题的解决方法包括",
        "迁移学习的核心思想是",
        "注意力机制在Transformer中的作用是",
        "生成对抗网络的训练难点在于",
        "联邦学习的优势和挑战分别是"
    ]

def demonstrate_basic_batch():
    """演示基础批量推理"""
    print("=" * 60)
    print("🎯 基础批量推理演示")
    print("=" * 60)
    
    # 创建推理引擎
    engine = BatchInferenceEngine(model_name="gpt2", device="cpu")
    
    if not engine.setup():
        print("❌ 引擎设置失败")
        return
    
    # 准备测试数据
    prompts = create_sample_prompts()[:8]  # 使用前8个提示
    
    print(f"\n📝 输入提示预览:")
    for i, prompt in enumerate(prompts[:3]):
        print(f"   {i+1}. {prompt}")
    print(f"   ... 共 {len(prompts)} 个提示")
    
    # 执行批量推理
    results = engine.batch_generate(
        prompts=prompts,
        batch_size=4,
        max_new_tokens=30,
        temperature=0.7,
        show_progress=True
    )
    
    # 显示结果
    print(f"\n📊 生成结果预览:")
    for i, (prompt, result) in enumerate(zip(prompts[:3], results[:3])):
        print(f"\n   提示 {i+1}: {prompt}")
        print(f"   生成: {result[len(prompt):].strip()}")
    
    # 显示性能统计
    print(f"\n📈 性能统计:")
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 保存结果
    engine.save_results(prompts, results, "basic_batch_results.json")
    
    engine.cleanup()

def demonstrate_batch_size_comparison():
    """演示不同批处理大小的性能对比"""
    print("=" * 60)
    print("⚡ 批处理大小性能对比")
    print("=" * 60)
    
    engine = BatchInferenceEngine(model_name="gpt2", device="cpu")
    
    if not engine.setup():
        print("❌ 引擎设置失败")
        return
    
    prompts = create_sample_prompts()[:12]  # 使用12个提示
    batch_sizes = [1, 2, 4, 6]
    
    results_comparison = {}
    
    for batch_size in batch_sizes:
        print(f"\n🔄 测试批处理大小: {batch_size}")
        
        # 重置统计
        engine.stats = {
            "total_batches": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "batch_times": [],
            "throughput_history": []
        }
        
        start_time = time.time()
        results = engine.batch_generate(
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=20,
            show_progress=False
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(prompts) / total_time
        
        results_comparison[batch_size] = {
            "总时间": f"{total_time:.2f}秒",
            "吞吐量": f"{throughput:.2f} 文本/秒",
            "平均批次时间": f"{statistics.mean(engine.stats['batch_times']):.2f}秒"
        }
    
    # 显示对比结果
    print(f"\n📊 性能对比结果:")
    print(f"{'批处理大小':<10} {'总时间':<12} {'吞吐量':<15} {'平均批次时间':<15}")
    print("-" * 55)
    
    for batch_size, stats in results_comparison.items():
        print(f"{batch_size:<10} {stats['总时间']:<12} {stats['吞吐量']:<15} {stats['平均批次时间']:<15}")
    
    # 找出最佳批处理大小
    best_batch_size = max(results_comparison.keys(), 
                         key=lambda x: float(results_comparison[x]['吞吐量'].split()[0]))
    
    print(f"\n🏆 最佳批处理大小: {best_batch_size} (吞吐量: {results_comparison[best_batch_size]['吞吐量']})")
    
    engine.cleanup()

def demonstrate_advanced_features():
    """演示高级功能"""
    print("=" * 60)
    print("🚀 高级批量推理功能演示")
    print("=" * 60)
    
    engine = BatchInferenceEngine(model_name="gpt2", device="cpu")
    
    if not engine.setup():
        print("❌ 引擎设置失败")
        return
    
    prompts = create_sample_prompts()[:6]
    
    # 1. 不同参数设置对比
    print("\n🎛️  参数设置对比:")
    
    configs = [
        {"temperature": 0.3, "top_p": 0.8, "name": "保守生成"},
        {"temperature": 0.7, "top_p": 0.9, "name": "平衡生成"},
        {"temperature": 1.0, "top_p": 0.95, "name": "创意生成"}
    ]
    
    for config in configs:
        print(f"\n   📋 {config['name']} (温度: {config['temperature']}, top_p: {config['top_p']})")
        
        results = engine.batch_generate(
            prompts=prompts[:3],
            batch_size=2,
            max_new_tokens=25,
            temperature=config['temperature'],
            top_p=config['top_p'],
            show_progress=False
        )
        
        for i, (prompt, result) in enumerate(zip(prompts[:3], results)):
            generated = result[len(prompt):].strip()
            print(f"      {i+1}. {generated[:50]}...")
    
    # 2. 错误处理演示
    print(f"\n🛡️  错误处理演示:")
    
    # 测试空输入
    empty_results = engine.batch_generate(
        prompts=["", "   ", "测试"],
        batch_size=2,
        max_new_tokens=10,
        show_progress=False
    )
    print(f"   空输入处理: 成功处理 {len(empty_results)} 个结果")
    
    # 测试超长输入
    long_prompt = "这是一个很长的输入提示。" * 50
    long_results = engine.batch_generate(
        prompts=[long_prompt],
        batch_size=1,
        max_new_tokens=10,
        show_progress=False
    )
    print(f"   长输入处理: 成功处理 {len(long_results)} 个结果")
    
    engine.cleanup()

def interactive_batch_demo():
    """交互式批量推理演示"""
    print("=" * 60)
    print("🎮 交互式批量推理演示")
    print("=" * 60)
    
    engine = BatchInferenceEngine(model_name="gpt2", device="cpu")
    
    if not engine.setup():
        print("❌ 引擎设置失败")
        return
    
    while True:
        print(f"\n📝 请选择操作:")
        print("1. 使用预设提示")
        print("2. 输入自定义提示")
        print("3. 查看性能统计")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            # 使用预设提示
            prompts = create_sample_prompts()[:5]
            print(f"\n使用预设提示 (共 {len(prompts)} 个)")
            
        elif choice == "2":
            # 输入自定义提示
            print(f"\n请输入提示 (每行一个，空行结束):")
            prompts = []
            while True:
                prompt = input().strip()
                if not prompt:
                    break
                prompts.append(prompt)
            
            if not prompts:
                print("❌ 未输入任何提示")
                continue
                
        elif choice == "3":
            # 查看统计
            stats = engine.get_performance_stats()
            print(f"\n📊 当前性能统计:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            continue
            
        elif choice == "4":
            break
            
        else:
            print("❌ 无效选择")
            continue
        
        # 获取参数
        try:
            batch_size = int(input(f"批处理大小 (默认 2): ") or "2")
            max_tokens = int(input(f"最大生成长度 (默认 30): ") or "30")
            temperature = float(input(f"温度参数 (默认 0.7): ") or "0.7")
        except ValueError:
            print("❌ 参数格式错误，使用默认值")
            batch_size, max_tokens, temperature = 2, 30, 0.7
        
        # 执行推理
        print(f"\n🚀 开始推理...")
        results = engine.batch_generate(
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        # 显示结果
        print(f"\n📊 生成结果:")
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            generated = result[len(prompt):].strip()
            print(f"\n   {i+1}. 提示: {prompt}")
            print(f"      生成: {generated}")
        
        # 询问是否保存
        save_choice = input(f"\n💾 是否保存结果? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("文件名 (默认 interactive_results.json): ").strip() or "interactive_results.json"
            engine.save_results(prompts, results, filename)
    
    engine.cleanup()
    print("👋 感谢使用！")

def main():
    """主函数"""
    print("🎯 nano-vLLM 批量推理示例")
    print("=" * 60)
    
    demos = [
        ("基础批量推理", demonstrate_basic_batch),
        ("批处理大小对比", demonstrate_batch_size_comparison),
        ("高级功能演示", demonstrate_advanced_features),
        ("交互式演示", interactive_batch_demo)
    ]
    
    print("请选择演示:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. 运行所有演示")
    
    try:
        choice = int(input("\n请输入选择 (0-4): "))
        
        if choice == 0:
            # 运行所有演示
            for name, demo_func in demos[:-1]:  # 排除交互式演示
                print(f"\n{'='*20} {name} {'='*20}")
                demo_func()
                input("\n按回车键继续...")
        elif 1 <= choice <= len(demos):
            demos[choice - 1][1]()
        else:
            print("❌ 无效选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")

if __name__ == "__main__":
    main()

# 🎯 学习要点总结:
#
# 1. 批量推理优势:
#    - 提高GPU利用率
#    - 减少模型加载开销
#    - 提升整体吞吐量
#
# 2. 关键参数:
#    - batch_size: 影响内存使用和速度
#    - padding: 确保批次内序列长度一致
#    - 生成参数: 控制输出质量和多样性
#
# 3. 性能优化:
#    - 选择合适的批处理大小
#    - 使用适当的数据类型 (float16)
#    - 合理设置最大长度
#
# 4. 最佳实践:
#    - 监控内存使用
#    - 处理异常情况
#    - 保存和分析结果
#    - 定期清理资源
#
# 🚀 下一步学习:
#    - 尝试不同的批处理大小
#    - 测试更大的模型
#    - 实现流式批量推理
#    - 添加更多性能指标