#!/usr/bin/env python3
"""
🚀 nano-vLLM 快速入门示例

这是您的第一个 vLLM 推理示例！
本示例将带您体验：
1. 模型加载
2. 文本生成
3. 结果输出

适合完全的初学者，包含详细的注释和错误处理。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys
from pathlib import Path

def print_welcome():
    """打印欢迎信息"""
    print("=" * 60)
    print("🚀 欢迎使用 nano-vLLM 推理引擎!")
    print("=" * 60)
    print("📚 这是您的第一个推理示例")
    print("🎯 目标：学习基本的文本生成")
    print("⏱️  预计用时：2-3分钟")
    print("-" * 60)

def check_environment():
    """检查运行环境"""
    print("🔍 正在检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"   Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    print(f"   PyTorch版本: {torch.__version__}")
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"   CUDA可用: ✅ (设备数量: {torch.cuda.device_count()})")
        print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
    else:
        print("   CUDA可用: ❌ (将使用CPU)")
    
    print("✅ 环境检查完成!\n")

def load_model_with_progress():
    """加载模型并显示进度"""
    print("📦 正在加载模型...")
    print("   模型: GPT-2 (小型模型，适合学习)")
    print("   大小: ~500MB")
    print("   用途: 文本生成演示")
    
    model_name = "gpt2"
    
    try:
        start_time = time.time()
        
        # 加载分词器
        print("   🔤 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设置pad_token（重要！）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   ⚙️  配置分词器...")
        
        # 加载模型
        print("   🧠 加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # 移动到GPU（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功! (耗时: {load_time:.2f}秒)")
        print(f"   设备: {device}")
        print(f"   参数量: ~117M")
        
        return tokenizer, model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 建议检查网络连接或使用本地模型")
        return None, None, None

def generate_text_demo(tokenizer, model, device):
    """文本生成演示"""
    print("\n" + "=" * 60)
    print("🎮 开始文本生成演示")
    print("=" * 60)
    
    # 准备多个示例提示
    prompts = [
        "人工智能的未来是",
        "在一个遥远的星球上",
        "今天是美好的一天，因为",
        "科技改变生活的方式包括"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 示例 {i}/4")
        print(f"输入提示: '{prompt}'")
        
        try:
            # 编码输入
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            print("🤔 模型推理中，请稍候...")
            start_time = time.time()
            
            # 生成文本
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,  # 原长度 + 30个新token
                    num_return_sequences=1,
                    temperature=0.8,  # 适中的创造性
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # 避免重复
                )
            
            generation_time = time.time() - start_time
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 只显示新生成的部分
            new_text = generated_text[len(prompt):].strip()
            
            print(f"🎉 生成结果:")
            print(f"   完整文本: {prompt}{new_text}")
            print(f"   生成时间: {generation_time:.2f}秒")
            print(f"   生成长度: {len(outputs[0]) - len(inputs[0])} tokens")
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
        
        # 添加分隔线
        if i < len(prompts):
            print("-" * 40)

def interactive_demo(tokenizer, model, device):
    """交互式演示"""
    print("\n" + "=" * 60)
    print("🎯 交互式文本生成")
    print("=" * 60)
    print("现在轮到您了！输入任何文本，AI将为您续写。")
    print("💡 提示：输入 'quit' 或 'exit' 退出")
    print("-" * 60)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n📝 请输入您的提示 (或 'quit' 退出): ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用！再见！")
                break
            
            if not user_input:
                print("⚠️  请输入一些文本")
                continue
            
            print(f"🤔 AI正在为 '{user_input}' 续写...")
            
            # 编码和生成
            inputs = tokenizer.encode(user_input, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 40,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(user_input):].strip()
            
            print(f"🎉 AI续写: {user_input}{new_text}")
            
        except KeyboardInterrupt:
            print("\n👋 检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"❌ 生成出错: {e}")

def print_learning_summary():
    """打印学习总结"""
    print("\n" + "=" * 60)
    print("🎓 学习总结")
    print("=" * 60)
    print("恭喜！您已经成功完成了第一个 vLLM 推理示例！")
    print("\n📚 您学到了什么：")
    print("   ✅ 如何加载预训练模型")
    print("   ✅ 如何进行文本生成")
    print("   ✅ 如何调整生成参数")
    print("   ✅ 如何处理输入输出")
    
    print("\n🚀 下一步建议：")
    print("   1. 尝试不同的提示词")
    print("   2. 调整生成参数 (temperature, max_length)")
    print("   3. 学习批量推理: examples/01-basic-usage/batch_inference.py")
    print("   4. 了解性能测试: examples/02-performance-testing/")
    
    print("\n📖 推荐阅读：")
    print("   • docs/01-basic-concepts/ - 理解核心概念")
    print("   • docs/02-architecture/ - 学习系统架构")
    print("   • examples/01-basic-usage/ - 更多基础示例")
    
    print("\n💡 记住：")
    print("   • 学习是一个渐进的过程")
    print("   • 多动手实践，深入理解原理")
    print("   • 遇到问题查看 docs/00-quick-start/faq.md")
    print("=" * 60)

def main():
    """主函数"""
    try:
        # 1. 欢迎信息
        print_welcome()
        
        # 2. 环境检查
        check_environment()
        
        # 3. 加载模型
        tokenizer, model, device = load_model_with_progress()
        
        if tokenizer is None or model is None:
            print("❌ 无法继续，请检查环境配置")
            return
        
        # 4. 演示文本生成
        generate_text_demo(tokenizer, model, device)
        
        # 5. 交互式演示
        try:
            interactive_demo(tokenizer, model, device)
        except KeyboardInterrupt:
            print("\n⏭️  跳过交互式演示")
        
        # 6. 学习总结
        print_learning_summary()
        
    except Exception as e:
        print(f"❌ 程序出现错误: {e}")
        print("💡 请查看 docs/00-quick-start/faq.md 获取帮助")
    
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 已清理GPU内存")

if __name__ == "__main__":
    main()