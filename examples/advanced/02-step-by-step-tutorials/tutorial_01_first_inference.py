#!/usr/bin/env python3
"""
🎯 教程1: 第一次推理

这是您的第一个完整的 nano-vLLM 推理教程！
我们将一步步引导您完成从环境检查到成功生成文本的全过程。

📋 学习目标:
- 理解推理的完整流程
- 掌握基本的模型操作
- 学会分析生成结果
- 建立调试思维

⏱️ 预计时间: 15-20分钟
🎓 难度等级: 初级 ⭐
"""

import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional

class FirstInferenceTutorial:
    """第一次推理教程"""
    
    def __init__(self):
        self.step_count = 0
        self.total_steps = 6
        self.completed_steps = []
        self.tutorial_data = {}
        
    def print_header(self):
        """打印教程标题"""
        print("=" * 60)
        print("🎯 nano-vLLM 教程1: 第一次推理")
        print("=" * 60)
        print("欢迎来到您的第一个 vLLM 推理教程！")
        print("我们将一步步完成您的第一次文本生成。")
        print()
        print("📋 教程步骤:")
        steps = [
            "环境检查和准备",
            "选择和加载模型", 
            "准备输入文本",
            "配置生成参数",
            "执行推理生成",
            "分析结果和总结"
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step}")
        
        print(f"\n⏱️  预计完成时间: 15-20分钟")
        print(f"🎓 难度等级: 初级 ⭐")
        
    def print_progress(self):
        """打印进度条"""
        progress = len(self.completed_steps) / self.total_steps
        filled = int(progress * 20)
        bar = "█" * filled + "░" * (20 - filled)
        percentage = int(progress * 100)
        
        print(f"\n🎯 教程进度: [{bar}] {percentage}% ({len(self.completed_steps)}/{self.total_steps}步骤完成)")
        
    def wait_for_continue(self, message: str = "按回车键继续..."):
        """等待用户确认继续"""
        input(f"\n💡 {message}")
        
    def step_1_environment_check(self) -> bool:
        """步骤1: 环境检查和准备"""
        self.step_count = 1
        print(f"\n{'='*20} 步骤 {self.step_count}: 环境检查和准备 {'='*20}")
        
        print("🔍 正在检查您的学习环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        print(f"   Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or python_version.minor < 8:
            print("   ❌ Python版本过低，需要3.8+")
            return False
        else:
            print("   ✅ Python版本符合要求")
        
        # 检查必要的包
        required_packages = ["torch", "transformers"]
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package} 已安装")
            except ImportError:
                print(f"   ❌ {package} 未安装")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n❌ 缺少必要的包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        # 检查设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   🖥️  计算设备: {device}")
        
        if device.type == "cuda":
            gpu_count = torch.cuda.device_count()
            print(f"   🎮 GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_properties(i).name
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"      GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("   💡 将使用CPU模式（速度较慢但功能完整）")
        
        self.tutorial_data["device"] = device
        self.tutorial_data["python_version"] = python_version
        
        print("\n✅ 环境检查完成！您的环境已准备就绪。")
        
        self.completed_steps.append(1)
        self.print_progress()
        
        self.wait_for_continue("环境检查完成，准备进入下一步")
        return True
    
    def step_2_model_selection(self) -> bool:
        """步骤2: 选择和加载模型"""
        self.step_count = 2
        print(f"\n{'='*20} 步骤 {self.step_count}: 选择和加载模型 {'='*20}")
        
        print("🤖 现在我们来选择和加载一个语言模型...")
        
        # 模型选择
        print("\n📚 可用的模型选项:")
        models = [
            {
                "name": "gpt2",
                "description": "GPT-2 (小型，快速，适合学习)",
                "size": "~500MB",
                "recommended": True
            },
            {
                "name": "gpt2-medium", 
                "description": "GPT-2 Medium (中型，质量更好)",
                "size": "~1.5GB",
                "recommended": False
            },
            {
                "name": "distilgpt2",
                "description": "DistilGPT-2 (更小更快)",
                "size": "~300MB", 
                "recommended": False
            }
        ]
        
        for i, model in enumerate(models, 1):
            marker = "🌟 推荐" if model["recommended"] else "  "
            print(f"   {i}. {model['name']} - {model['description']}")
            print(f"      大小: {model['size']} {marker}")
        
        print(f"\n💡 对于第一次学习，我们推荐使用 'gpt2' 模型")
        print(f"   - 下载速度快")
        print(f"   - 内存占用少") 
        print(f"   - 功能完整")
        
        # 使用推荐模型
        model_name = "gpt2"
        print(f"\n🔄 正在加载模型: {model_name}")
        
        try:
            # 加载分词器
            print("   📝 加载分词器...")
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            tokenizer_time = time.time() - start_time
            print(f"   ✅ 分词器加载完成 ({tokenizer_time:.2f}秒)")
            
            # 加载模型
            print("   🧠 加载模型...")
            start_time = time.time()
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.tutorial_data["device"].type == "cuda" else torch.float32
            )
            model = model.to(self.tutorial_data["device"])
            model.eval()
            
            model_time = time.time() - start_time
            print(f"   ✅ 模型加载完成 ({model_time:.2f}秒)")
            
            # 显示模型信息
            print(f"\n📊 模型信息:")
            print(f"   模型名称: {model_name}")
            print(f"   参数数量: {model.num_parameters():,}")
            print(f"   词汇表大小: {len(tokenizer):,}")
            print(f"   设备: {self.tutorial_data['device']}")
            
            if self.tutorial_data["device"].type == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"   GPU内存使用: {memory_used:.2f} GB")
            
            self.tutorial_data["model"] = model
            self.tutorial_data["tokenizer"] = tokenizer
            self.tutorial_data["model_name"] = model_name
            
            print(f"\n✅ 模型加载成功！现在可以进行文本生成了。")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("💡 可能的解决方案:")
            print("   1. 检查网络连接")
            print("   2. 尝试重新运行")
            print("   3. 使用更小的模型")
            return False
        
        self.completed_steps.append(2)
        self.print_progress()
        
        self.wait_for_continue("模型加载完成，准备准备输入文本")
        return True
    
    def step_3_prepare_input(self) -> bool:
        """步骤3: 准备输入文本"""
        self.step_count = 3
        print(f"\n{'='*20} 步骤 {self.step_count}: 准备输入文本 {'='*20}")
        
        print("📝 现在我们来准备输入文本（也叫做'提示词'或'prompt'）...")
        
        print(f"\n💡 什么是提示词？")
        print(f"   提示词是您给AI模型的输入文本，模型会基于这个文本继续生成内容。")
        print(f"   就像给作家一个开头，让他继续写故事一样。")
        
        # 提供示例提示词
        print(f"\n📚 示例提示词:")
        example_prompts = [
            "从前有一只聪明的小猫",
            "人工智能的未来发展",
            "今天是美好的一天",
            "在遥远的星球上",
            "科技改变了我们的生活"
        ]
        
        for i, prompt in enumerate(example_prompts, 1):
            print(f"   {i}. \"{prompt}\"")
        
        # 让用户选择或输入
        print(f"\n🎯 您可以:")
        print(f"   1. 选择上面的示例之一")
        print(f"   2. 输入您自己的提示词")
        
        while True:
            choice = input("\n请输入选项 (1-5) 或直接输入您的提示词: ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= 5:
                selected_prompt = example_prompts[int(choice) - 1]
                print(f"✅ 您选择了: \"{selected_prompt}\"")
                break
            elif choice:
                selected_prompt = choice
                print(f"✅ 您输入了: \"{selected_prompt}\"")
                break
            else:
                print("❌ 请输入有效的选项或提示词")
        
        # 分析提示词
        print(f"\n🔍 分析您的提示词:")
        tokenizer = self.tutorial_data["tokenizer"]
        
        # 分词
        tokens = tokenizer.tokenize(selected_prompt)
        token_ids = tokenizer.encode(selected_prompt)
        
        print(f"   原文: \"{selected_prompt}\"")
        print(f"   Token数量: {len(tokens)}")
        print(f"   Tokens: {tokens}")
        print(f"   Token IDs: {token_ids}")
        
        # 编码为模型输入
        inputs = tokenizer.encode(selected_prompt, return_tensors="pt")
        inputs = inputs.to(self.tutorial_data["device"])
        
        print(f"   输入张量形状: {inputs.shape}")
        print(f"   输入设备: {inputs.device}")
        
        self.tutorial_data["prompt"] = selected_prompt
        self.tutorial_data["inputs"] = inputs
        self.tutorial_data["input_length"] = len(token_ids)
        
        print(f"\n✅ 输入文本准备完成！")
        
        self.completed_steps.append(3)
        self.print_progress()
        
        self.wait_for_continue("输入准备完成，接下来配置生成参数")
        return True
    
    def step_4_configure_parameters(self) -> bool:
        """步骤4: 配置生成参数"""
        self.step_count = 4
        print(f"\n{'='*20} 步骤 {self.step_count}: 配置生成参数 {'='*20}")
        
        print("⚙️ 现在我们来配置文本生成的参数...")
        
        print(f"\n📚 重要参数说明:")
        
        parameters = [
            {
                "name": "max_length",
                "description": "生成文本的最大长度",
                "example": "50 (生成约50个词)",
                "tip": "太短可能不完整，太长可能偏题"
            },
            {
                "name": "temperature", 
                "description": "控制随机性 (0.1-2.0)",
                "example": "0.7 (平衡创造性和连贯性)",
                "tip": "低值更保守，高值更有创意"
            },
            {
                "name": "top_p",
                "description": "核采样参数 (0.1-1.0)", 
                "example": "0.9 (考虑90%概率的词汇)",
                "tip": "控制词汇选择的多样性"
            },
            {
                "name": "do_sample",
                "description": "是否使用随机采样",
                "example": "True (启用随机性)",
                "tip": "False会总是选择最可能的词"
            }
        ]
        
        for param in parameters:
            print(f"\n   🔧 {param['name']}:")
            print(f"      说明: {param['description']}")
            print(f"      示例: {param['example']}")
            print(f"      提示: {param['tip']}")
        
        # 为初学者推荐参数
        print(f"\n🌟 初学者推荐参数:")
        recommended_params = {
            "max_length": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1
        }
        
        for param, value in recommended_params.items():
            print(f"   {param}: {value}")
        
        print(f"\n💡 这些参数适合大多数场景，能产生既有创意又连贯的文本。")
        
        # 让用户选择是否使用推荐参数
        while True:
            choice = input("\n使用推荐参数？(y/n): ").strip().lower()
            if choice in ['y', 'yes', '是', '']:
                params = recommended_params.copy()
                print("✅ 使用推荐参数")
                break
            elif choice in ['n', 'no', '否']:
                print("🔧 自定义参数模式（高级用户）")
                params = self._get_custom_parameters()
                break
            else:
                print("请输入 y 或 n")
        
        # 计算实际的max_length
        input_length = self.tutorial_data["input_length"]
        total_max_length = input_length + params["max_length"]
        params["actual_max_length"] = total_max_length
        
        print(f"\n📊 最终参数配置:")
        for param, value in params.items():
            print(f"   {param}: {value}")
        
        print(f"\n💡 解释:")
        print(f"   输入长度: {input_length} tokens")
        print(f"   生成长度: {params['max_length']} tokens")
        print(f"   总长度: {total_max_length} tokens")
        
        self.tutorial_data["generation_params"] = params
        
        print(f"\n✅ 参数配置完成！")
        
        self.completed_steps.append(4)
        self.print_progress()
        
        self.wait_for_continue("参数配置完成，准备执行推理")
        return True
    
    def _get_custom_parameters(self) -> Dict:
        """获取自定义参数（高级用户）"""
        params = {}
        
        # 简化的自定义参数输入
        try:
            params["max_length"] = int(input("生成长度 (20-100): ") or "50")
            params["temperature"] = float(input("温度 (0.1-2.0): ") or "0.7")
            params["top_p"] = float(input("top_p (0.1-1.0): ") or "0.9")
            params["top_k"] = int(input("top_k (10-100): ") or "50")
            params["do_sample"] = True
            params["repetition_penalty"] = 1.1
        except ValueError:
            print("❌ 输入格式错误，使用默认值")
            params = {
                "max_length": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.1
            }
        
        return params
    
    def step_5_execute_inference(self) -> bool:
        """步骤5: 执行推理生成"""
        self.step_count = 5
        print(f"\n{'='*20} 步骤 {self.step_count}: 执行推理生成 {'='*20}")
        
        print("🚀 激动人心的时刻到了！我们要生成您的第一个AI文本...")
        
        # 准备数据
        model = self.tutorial_data["model"]
        tokenizer = self.tutorial_data["tokenizer"]
        inputs = self.tutorial_data["inputs"]
        params = self.tutorial_data["generation_params"]
        prompt = self.tutorial_data["prompt"]
        
        print(f"\n📋 推理信息:")
        print(f"   输入提示: \"{prompt}\"")
        print(f"   模型: {self.tutorial_data['model_name']}")
        print(f"   设备: {self.tutorial_data['device']}")
        print(f"   生成长度: {params['max_length']} tokens")
        
        print(f"\n⏳ 正在生成文本...")
        
        try:
            # 记录开始时间
            start_time = time.time()
            
            # 执行推理
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=params["actual_max_length"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    do_sample=params["do_sample"],
                    repetition_penalty=params["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 解码结果
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(prompt):].strip()
            
            # 计算统计信息
            input_tokens = len(inputs[0])
            output_tokens = len(outputs[0])
            generated_tokens = output_tokens - input_tokens
            tokens_per_second = generated_tokens / inference_time
            
            print(f"\n🎉 生成完成！")
            
            print(f"\n📊 生成结果:")
            print(f"   完整文本: \"{generated_text}\"")
            print(f"   新生成部分: \"{new_text}\"")
            
            print(f"\n📈 性能统计:")
            print(f"   推理时间: {inference_time:.3f}秒")
            print(f"   输入tokens: {input_tokens}")
            print(f"   生成tokens: {generated_tokens}")
            print(f"   生成速度: {tokens_per_second:.1f} tokens/秒")
            
            # 保存结果
            self.tutorial_data["generated_text"] = generated_text
            self.tutorial_data["new_text"] = new_text
            self.tutorial_data["inference_time"] = inference_time
            self.tutorial_data["tokens_per_second"] = tokens_per_second
            self.tutorial_data["generated_tokens"] = generated_tokens
            
            print(f"\n✅ 恭喜！您已经成功完成了第一次AI文本生成！")
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            print("💡 可能的原因:")
            print("   1. 内存不足")
            print("   2. 参数设置错误")
            print("   3. 模型加载问题")
            return False
        
        self.completed_steps.append(5)
        self.print_progress()
        
        self.wait_for_continue("推理完成，最后我们来分析结果")
        return True
    
    def step_6_analyze_results(self) -> bool:
        """步骤6: 分析结果和总结"""
        self.step_count = 6
        print(f"\n{'='*20} 步骤 {self.step_count}: 分析结果和总结 {'='*20}")
        
        print("🔍 让我们来分析您的第一次推理结果...")
        
        # 获取结果数据
        prompt = self.tutorial_data["prompt"]
        generated_text = self.tutorial_data["generated_text"]
        new_text = self.tutorial_data["new_text"]
        inference_time = self.tutorial_data["inference_time"]
        tokens_per_second = self.tutorial_data["tokens_per_second"]
        generated_tokens = self.tutorial_data["generated_tokens"]
        
        print(f"\n📝 文本质量分析:")
        
        # 基本质量检查
        quality_checks = []
        
        # 检查是否有新内容生成
        if len(new_text.strip()) > 0:
            quality_checks.append(("✅", "成功生成了新内容"))
        else:
            quality_checks.append(("❌", "没有生成新内容"))
        
        # 检查文本长度
        if len(new_text.split()) >= 5:
            quality_checks.append(("✅", "生成了足够长度的文本"))
        else:
            quality_checks.append(("⚠️", "生成的文本较短"))
        
        # 检查是否有重复
        words = new_text.split()
        if len(words) == len(set(words)):
            quality_checks.append(("✅", "没有明显的词汇重复"))
        else:
            quality_checks.append(("⚠️", "存在一些词汇重复"))
        
        # 检查是否与输入相关
        if any(word in new_text.lower() for word in prompt.lower().split()):
            quality_checks.append(("✅", "生成内容与输入相关"))
        else:
            quality_checks.append(("ℹ️", "生成了新的主题内容"))
        
        for status, message in quality_checks:
            print(f"   {status} {message}")
        
        print(f"\n⚡ 性能分析:")
        
        # 性能评估
        if tokens_per_second > 10:
            speed_rating = "🚀 很快"
        elif tokens_per_second > 5:
            speed_rating = "✅ 正常"
        else:
            speed_rating = "🐌 较慢"
        
        print(f"   生成速度: {tokens_per_second:.1f} tokens/秒 ({speed_rating})")
        print(f"   推理时间: {inference_time:.3f}秒")
        print(f"   生成效率: {generated_tokens/inference_time:.1f} tokens/秒")
        
        if self.tutorial_data["device"].type == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"   内存使用: {memory_used:.2f} GB")
        
        print(f"\n🎓 学习成果:")
        achievements = [
            "✅ 成功加载和使用语言模型",
            "✅ 理解了提示词的概念和作用",
            "✅ 掌握了基本的生成参数",
            "✅ 完成了完整的推理流程",
            "✅ 学会了分析生成结果"
        ]
        
        for achievement in achievements:
            print(f"   {achievement}")
        
        print(f"\n💡 改进建议:")
        suggestions = []
        
        if tokens_per_second < 5:
            suggestions.append("考虑使用GPU加速推理")
        
        if len(new_text.split()) < 10:
            suggestions.append("尝试增加max_length参数")
        
        if "重复" in [check[1] for check in quality_checks if "重复" in check[1]]:
            suggestions.append("调整repetition_penalty参数")
        
        suggestions.extend([
            "尝试不同的temperature值看效果",
            "测试不同类型的提示词",
            "学习更高级的参数调优技巧"
        ])
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        
        print(f"\n🚀 下一步学习建议:")
        next_steps = [
            "完成教程2: 参数探索 (tutorial_02_parameter_exploration.py)",
            "尝试examples/01-basic-usage/中的更多示例",
            "阅读docs/01-basic-concepts.md了解理论知识",
            "实验不同的模型和参数组合"
        ]
        
        for i, step in enumerate(next_steps, 1):
            print(f"   {i}. {step}")
        
        self.completed_steps.append(6)
        self.print_progress()
        
        print(f"\n🎉 恭喜！您已经完成了第一个推理教程！")
        print(f"您现在已经掌握了nano-vLLM的基础使用方法。")
        
        return True
    
    def cleanup(self):
        """清理资源"""
        print(f"\n🧹 清理资源...")
        
        if "model" in self.tutorial_data:
            del self.tutorial_data["model"]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✅ 资源清理完成")
    
    def run_tutorial(self):
        """运行完整教程"""
        try:
            # 打印标题
            self.print_header()
            
            # 等待开始
            self.wait_for_continue("准备好开始您的第一次推理之旅了吗？")
            
            # 执行各个步骤
            steps = [
                self.step_1_environment_check,
                self.step_2_model_selection,
                self.step_3_prepare_input,
                self.step_4_configure_parameters,
                self.step_5_execute_inference,
                self.step_6_analyze_results
            ]
            
            for step_func in steps:
                if not step_func():
                    print(f"❌ 步骤 {self.step_count} 失败，教程中断")
                    return False
            
            # 教程完成
            print(f"\n" + "=" * 60)
            print(f"🎉 教程1完成！您已经成功掌握了基础推理技能！")
            print(f"=" * 60)
            
            return True
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  教程被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 教程执行出错: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """主函数"""
    tutorial = FirstInferenceTutorial()
    tutorial.run_tutorial()

if __name__ == "__main__":
    main()