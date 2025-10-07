#!/usr/bin/env python3
"""
🚀 简单推理示例

本示例演示如何使用 nano-vLLM 进行最基础的文本推理。
这是学习 vLLM 推理的第一步，涵盖：

1. 模型加载和初始化
2. 文本输入处理
3. 推理参数设置
4. 结果输出和解析
5. 资源清理

适合：刚完成快速入门的学习者
前置：完成 examples/00-quick-start/ 中的示例
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import gc
from typing import Optional, Dict, Any, List

class SimpleInferenceEngine:
    """简单推理引擎"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        初始化推理引擎
        
        Args:
            model_name: 模型名称，默认使用 gpt2
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        加载模型和分词器
        
        Returns:
            bool: 加载是否成功
        """
        print(f"🔄 正在加载模型: {self.model_name}")
        
        try:
            # 1. 检测设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 使用设备: {self.device}")
            
            # 2. 加载分词器
            print("   📝 加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 设置pad_token（如果没有的话）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("   🔧 设置pad_token为eos_token")
            
            # 3. 加载模型
            print("   🧠 加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            # 如果是CPU，手动移动到设备
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            # 4. 设置为评估模式
            self.model.eval()
            
            self.is_loaded = True
            print(f"✅ 模型加载成功！")
            
            # 显示模型信息
            self._print_model_info()
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _print_model_info(self):
        """打印模型信息"""
        if not self.is_loaded:
            return
            
        print(f"\n📊 模型信息:")
        print(f"   模型名称: {self.model_name}")
        print(f"   参数数量: {self.model.num_parameters():,}")
        print(f"   词汇表大小: {len(self.tokenizer)}")
        print(f"   最大序列长度: {self.tokenizer.model_max_length}")
        
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"   GPU内存使用: {memory_allocated:.2f} GB")
    
    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_length: 最大生成长度
            temperature: 温度参数 (0.1-2.0)
            top_p: 核采样参数 (0.1-1.0)
            top_k: top-k采样参数
            do_sample: 是否使用采样
            num_return_sequences: 返回序列数量
            
        Returns:
            List[str]: 生成的文本列表
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        print(f"\n🎯 开始推理...")
        print(f"   输入: '{prompt}'")
        print(f"   参数: max_len={max_length}, temp={temperature}, top_p={top_p}")
        
        try:
            # 1. 编码输入
            start_time = time.time()
            
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512  # 限制输入长度
            ).to(self.device)
            
            input_length = inputs.shape[1]
            print(f"   输入Token数: {input_length}")
            
            # 2. 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=input_length + max_length,  # 总长度 = 输入 + 生成
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # 避免重复
                    length_penalty=1.0       # 长度惩罚
                )
            
            # 3. 解码结果
            results = []
            for i, output in enumerate(outputs):
                # 只取新生成的部分
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )
                
                # 完整文本（包含输入）
                full_text = self.tokenizer.decode(output, skip_special_tokens=True)
                
                results.append(generated_text.strip())
                
                print(f"\n   结果 {i+1}:")
                print(f"   完整输出: '{full_text}'")
                print(f"   新生成: '{generated_text.strip()}'")
                print(f"   生成Token数: {len(generated_tokens)}")
            
            # 4. 计算性能指标
            end_time = time.time()
            inference_time = end_time - start_time
            total_tokens = sum(len(output) - input_length for output in outputs)
            tokens_per_second = total_tokens / inference_time
            
            print(f"\n⚡ 性能指标:")
            print(f"   推理时间: {inference_time:.3f}秒")
            print(f"   生成速度: {tokens_per_second:.1f} tokens/秒")
            
            return results
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return []
    
    def interactive_chat(self):
        """交互式聊天模式"""
        if not self.is_loaded:
            print("❌ 模型未加载，请先调用 load_model()")
            return
        
        print("\n💬 进入交互式聊天模式")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清屏")
        print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if user_input.lower() == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                if not user_input:
                    continue
                
                # 开始生成过程，展示推理状态
                print("🤖 开始文本生成...")
                responses = self.generate_text(
                    user_input,
                    max_length=50,
                    temperature=0.8,
                    top_p=0.9
                )
                
                if responses:
                    print(f"🤖 AI: {responses[0]}")
                else:
                    print("🤖 AI: 抱歉，我无法生成回复。")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
    
    def cleanup(self):
        """清理资源"""
        print("\n🧹 清理资源...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        gc.collect()
        
        self.is_loaded = False
        print("✅ 资源清理完成")

def demonstrate_basic_inference():
    """演示基础推理功能"""
    print("🚀 nano-vLLM 简单推理演示")
    print("=" * 50)
    
    # 创建推理引擎
    engine = SimpleInferenceEngine("gpt2")
    
    try:
        # 1. 加载模型
        if not engine.load_model():
            return
        
        # 2. 演示不同类型的推理
        test_cases = [
            {
                "name": "创意写作",
                "prompt": "在一个遥远的星球上",
                "params": {"temperature": 1.0, "max_length": 80}
            },
            {
                "name": "技术解释",
                "prompt": "人工智能是",
                "params": {"temperature": 0.3, "max_length": 60}
            },
            {
                "name": "故事续写",
                "prompt": "从前有一只聪明的小猫",
                "params": {"temperature": 0.8, "max_length": 100}
            },
            {
                "name": "问答对话",
                "prompt": "什么是机器学习？",
                "params": {"temperature": 0.5, "max_length": 80}
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n{'='*20} 测试 {i}: {case['name']} {'='*20}")
            
            results = engine.generate_text(
                case["prompt"],
                **case["params"]
            )
            
            if results:
                print(f"✅ 生成成功")
            else:
                print(f"❌ 生成失败")
        
        # 3. 参数对比演示
        print(f"\n{'='*20} 参数对比演示 {'='*20}")
        prompt = "科技的发展"
        
        param_sets = [
            {"name": "保守模式", "temperature": 0.1, "top_p": 0.5},
            {"name": "平衡模式", "temperature": 0.7, "top_p": 0.9},
            {"name": "创意模式", "temperature": 1.2, "top_p": 0.95}
        ]
        
        for param_set in param_sets:
            print(f"\n🎛️  {param_set['name']}:")
            results = engine.generate_text(
                prompt,
                max_length=50,
                temperature=param_set["temperature"],
                top_p=param_set["top_p"]
            )
        
        # 4. 交互式聊天（可选）
        print(f"\n{'='*20} 交互式聊天 {'='*20}")
        user_choice = input("是否进入交互式聊天模式？(y/n): ").strip().lower()
        
        if user_choice in ['y', 'yes', '是']:
            engine.interactive_chat()
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
    
    finally:
        # 清理资源
        engine.cleanup()

def print_learning_summary():
    """打印学习总结"""
    print("\n" + "=" * 60)
    print("🎓 简单推理学习总结")
    print("=" * 60)
    
    key_points = [
        "模型加载和初始化流程",
        "基础推理参数的使用",
        "文本生成和解码过程",
        "性能监控和资源管理",
        "错误处理和调试技巧"
    ]
    
    print("\n📚 您已经学会了:")
    for i, point in enumerate(key_points, 1):
        print(f"   {i}. {point}")
    
    print(f"\n🎯 核心概念:")
    concepts = [
        ("Temperature", "控制输出的随机性，越高越随机"),
        ("Top-p", "核采样，控制候选词的范围"),
        ("Top-k", "限制每步考虑的词汇数量"),
        ("Max Length", "控制生成文本的最大长度"),
        ("Repetition Penalty", "避免重复内容的惩罚机制")
    ]
    
    for concept, description in concepts:
        print(f"   • {concept}: {description}")
    
    print(f"\n🚀 下一步学习建议:")
    next_steps = [
        "学习批量推理 (batch_inference.py)",
        "掌握流式输出 (streaming_inference.py)",
        "深入参数调优 (parameter_tuning.py)",
        "阅读架构文档了解原理"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")
    
    print(f"\n💡 实用技巧:")
    tips = [
        "根据任务类型调整temperature",
        "使用GPU可以显著提升速度",
        "及时清理资源避免内存泄漏",
        "监控性能指标优化体验"
    ]
    
    for tip in tips:
        print(f"   • {tip}")

def main():
    """主函数"""
    try:
        # 运行演示
        demonstrate_basic_inference()
        
        # 打印学习总结
        print_learning_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        print("💡 请检查环境配置或查看错误处理示例")

if __name__ == "__main__":
    main()