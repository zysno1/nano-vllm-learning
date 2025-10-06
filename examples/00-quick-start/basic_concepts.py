#!/usr/bin/env python3
"""
📚 nano-vLLM 核心概念演示

本示例通过实际代码演示 vLLM 的核心概念：
1. Token 和 Tokenization (分词)
2. 批处理 (Batching)
3. 采样策略 (Sampling Strategies)
4. KV Cache 概念
5. 内存管理

适合已经运行过 hello_vllm.py 的学习者。
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import numpy as np
from typing import List, Dict, Any

class ConceptDemonstrator:
    """核心概念演示器"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def setup(self):
        """初始化模型"""
        print("🔧 初始化演示环境...")
        
        model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"✅ 环境初始化完成 (设备: {self.device})")

def demonstrate_tokenization():
    """演示 Token 和分词概念"""
    print("\n" + "=" * 60)
    print("📝 概念1: Token 和 Tokenization (分词)")
    print("=" * 60)
    
    print("🎯 什么是Token？")
    print("   Token是文本的最小处理单位，可以是字符、词或子词")
    
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 演示不同文本的分词
    examples = [
        "Hello World!",
        "人工智能",
        "artificial intelligence",
        "I'm learning vLLM today.",
        "这是一个很长的句子，用来演示分词的效果。"
    ]
    
    print("\n📊 分词演示:")
    for text in examples:
        # 分词
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text)
        
        print(f"\n原文: '{text}'")
        print(f"Token数量: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # 演示解码
        decoded = tokenizer.decode(token_ids)
        print(f"解码结果: '{decoded}'")
    
    print("\n💡 关键概念:")
    print("   • 不同语言的分词效果不同")
    print("   • 英文通常按子词分割")
    print("   • 中文可能按字符分割")
    print("   • Token数量影响计算成本")

def demonstrate_batching(demonstrator: ConceptDemonstrator):
    """演示批处理概念"""
    print("\n" + "=" * 60)
    print("🚀 概念2: 批处理 (Batching)")
    print("=" * 60)
    
    print("🎯 什么是批处理？")
    print("   同时处理多个请求，提高GPU利用率和吞吐量")
    
    # 准备多个输入
    prompts = [
        "今天天气",
        "人工智能的发展",
        "编程语言中",
        "在遥远的未来"
    ]
    
    print(f"\n📊 批处理演示 (批次大小: {len(prompts)}):")
    
    # 单个处理 vs 批处理对比
    print("\n🐌 单个处理模式:")
    start_time = time.time()
    
    single_results = []
    for i, prompt in enumerate(prompts):
        inputs = demonstrator.tokenizer.encode(prompt, return_tensors="pt").to(demonstrator.device)
        
        with torch.no_grad():
            outputs = demonstrator.model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=demonstrator.tokenizer.pad_token_id
            )
        
        result = demonstrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        single_results.append(result)
        print(f"   请求{i+1}: {result}")
    
    single_time = time.time() - start_time
    
    print(f"\n🚀 批处理模式:")
    start_time = time.time()
    
    # 批处理
    batch_inputs = demonstrator.tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,
        truncation=True
    ).to(demonstrator.device)
    
    with torch.no_grad():
        batch_outputs = demonstrator.model.generate(
            batch_inputs.input_ids,
            attention_mask=batch_inputs.attention_mask,
            max_length=batch_inputs.input_ids.shape[1] + 10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=demonstrator.tokenizer.pad_token_id
        )
    
    batch_results = []
    for i, output in enumerate(batch_outputs):
        result = demonstrator.tokenizer.decode(output, skip_special_tokens=True)
        batch_results.append(result)
        print(f"   请求{i+1}: {result}")
    
    batch_time = time.time() - start_time
    
    print(f"\n⚡ 性能对比:")
    print(f"   单个处理耗时: {single_time:.3f}秒")
    print(f"   批处理耗时: {batch_time:.3f}秒")
    print(f"   加速比: {single_time/batch_time:.2f}x")
    
    print(f"\n💡 批处理优势:")
    print(f"   • 更高的GPU利用率")
    print(f"   • 更好的吞吐量")
    print(f"   • 更低的平均延迟")

def demonstrate_sampling_strategies(demonstrator: ConceptDemonstrator):
    """演示采样策略"""
    print("\n" + "=" * 60)
    print("🎲 概念3: 采样策略 (Sampling Strategies)")
    print("=" * 60)
    
    print("🎯 什么是采样策略？")
    print("   控制模型如何选择下一个Token的方法")
    
    prompt = "人工智能的未来"
    inputs = demonstrator.tokenizer.encode(prompt, return_tensors="pt").to(demonstrator.device)
    
    # 不同采样策略
    strategies = [
        {
            "name": "贪婪搜索 (Greedy)",
            "params": {"do_sample": False},
            "description": "总是选择概率最高的Token"
        },
        {
            "name": "随机采样 (Random)",
            "params": {"do_sample": True, "temperature": 1.0},
            "description": "按概率随机选择Token"
        },
        {
            "name": "低温采样 (Low Temperature)",
            "params": {"do_sample": True, "temperature": 0.3},
            "description": "更保守，更确定的输出"
        },
        {
            "name": "高温采样 (High Temperature)",
            "params": {"do_sample": True, "temperature": 1.5},
            "description": "更随机，更有创造性"
        },
        {
            "name": "Top-k采样",
            "params": {"do_sample": True, "top_k": 10, "temperature": 0.8},
            "description": "只从概率最高的k个Token中选择"
        },
        {
            "name": "Top-p采样 (核采样)",
            "params": {"do_sample": True, "top_p": 0.9, "temperature": 0.8},
            "description": "从累积概率达到p的Token集合中选择"
        }
    ]
    
    print(f"\n📊 采样策略对比 (提示: '{prompt}'):")
    
    for strategy in strategies:
        print(f"\n🎲 {strategy['name']}:")
        print(f"   说明: {strategy['description']}")
        
        try:
            with torch.no_grad():
                outputs = demonstrator.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 15,
                    pad_token_id=demonstrator.tokenizer.pad_token_id,
                    **strategy['params']
                )
            
            result = demonstrator.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = result[len(prompt):].strip()
            print(f"   结果: {prompt}{new_text}")
            
        except Exception as e:
            print(f"   错误: {e}")
    
    print(f"\n💡 采样策略选择建议:")
    print(f"   • 贪婪搜索: 需要确定性输出时")
    print(f"   • 低温采样: 需要相对保守的创作")
    print(f"   • 高温采样: 需要更多创造性")
    print(f"   • Top-k/Top-p: 平衡质量和多样性")

def demonstrate_kv_cache_concept():
    """演示KV Cache概念"""
    print("\n" + "=" * 60)
    print("🧠 概念4: KV Cache (键值缓存)")
    print("=" * 60)
    
    print("🎯 什么是KV Cache？")
    print("   缓存注意力机制中的键(Key)和值(Value)，避免重复计算")
    
    print("\n📊 KV Cache工作原理:")
    print("   1. 第一次生成: 计算所有Token的K,V")
    print("   2. 后续生成: 只计算新Token的K,V，复用之前的")
    print("   3. 内存换时间: 用更多内存换取更快速度")
    
    # 模拟KV Cache的效果
    print("\n🔍 模拟演示:")
    
    sequence_lengths = [10, 20, 50, 100]
    
    for seq_len in sequence_lengths:
        # 模拟计算量
        without_cache = seq_len * seq_len  # 每次都要重新计算所有
        with_cache = seq_len + (seq_len - 1)  # 第一次全计算，后续只计算新的
        
        savings = (without_cache - with_cache) / without_cache * 100
        
        print(f"   序列长度 {seq_len}:")
        print(f"     无缓存计算量: {without_cache}")
        print(f"     有缓存计算量: {with_cache}")
        print(f"     节省计算: {savings:.1f}%")
    
    print(f"\n💡 KV Cache的影响:")
    print(f"   ✅ 优势: 显著减少计算量，提升生成速度")
    print(f"   ❌ 代价: 需要更多GPU内存存储缓存")
    print(f"   🎯 适用: 长序列生成，对话系统")

def demonstrate_memory_management():
    """演示内存管理概念"""
    print("\n" + "=" * 60)
    print("💾 概念5: 内存管理")
    print("=" * 60)
    
    print("🎯 为什么内存管理重要？")
    print("   GPU内存有限，需要高效管理以支持更大批次和更长序列")
    
    if torch.cuda.is_available():
        # 显示GPU内存信息
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**3
        cached_memory = torch.cuda.memory_reserved(device) / 1024**3
        
        print(f"\n📊 当前GPU内存状态:")
        print(f"   总内存: {total_memory:.2f} GB")
        print(f"   已分配: {allocated_memory:.2f} GB ({allocated_memory/total_memory*100:.1f}%)")
        print(f"   已缓存: {cached_memory:.2f} GB ({cached_memory/total_memory*100:.1f}%)")
        print(f"   可用内存: {total_memory - cached_memory:.2f} GB")
        
        # 演示内存清理
        print(f"\n🧹 内存清理演示:")
        print(f"   清理前缓存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        torch.cuda.empty_cache()
        
        print(f"   清理后缓存: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
    else:
        print(f"\n📊 CPU内存管理:")
        print(f"   CPU模式下主要关注系统内存使用")
        print(f"   可以通过减少批次大小来控制内存")
    
    print(f"\n💡 内存优化策略:")
    print(f"   • 使用半精度 (float16) 减少内存")
    print(f"   • 合理设置批次大小")
    print(f"   • 及时清理不需要的缓存")
    print(f"   • 使用梯度检查点技术")

def print_summary():
    """打印学习总结"""
    print("\n" + "=" * 60)
    print("🎓 核心概念学习总结")
    print("=" * 60)
    
    concepts = [
        {
            "name": "Token化",
            "key_points": ["文本的最小处理单位", "影响计算成本", "不同语言效果不同"]
        },
        {
            "name": "批处理", 
            "key_points": ["同时处理多个请求", "提高GPU利用率", "显著提升吞吐量"]
        },
        {
            "name": "采样策略",
            "key_points": ["控制输出的随机性", "平衡质量和创造性", "根据场景选择"]
        },
        {
            "name": "KV Cache",
            "key_points": ["缓存注意力计算", "内存换时间", "适合长序列生成"]
        },
        {
            "name": "内存管理",
            "key_points": ["GPU内存有限", "需要高效利用", "多种优化策略"]
        }
    ]
    
    for i, concept in enumerate(concepts, 1):
        print(f"\n{i}. {concept['name']}:")
        for point in concept['key_points']:
            print(f"   • {point}")
    
    print(f"\n🚀 下一步学习建议:")
    print(f"   1. 运行 step_by_step_tutorial.py 进行深入学习")
    print(f"   2. 阅读 docs/01-basic-concepts/ 理论文档")
    print(f"   3. 尝试 examples/01-basic-usage/ 中的更多示例")
    print(f"   4. 进行性能测试: examples/02-performance-testing/")
    
    print(f"\n💡 记住:")
    print(f"   • 理论和实践相结合")
    print(f"   • 多动手实验不同参数")
    print(f"   • 关注性能和内存使用")

def main():
    """主函数"""
    print("📚 nano-vLLM 核心概念演示")
    print("本演示将通过实际代码展示 vLLM 的核心概念")
    
    try:
        # 1. Token化演示
        demonstrate_tokenization()
        
        # 2. 初始化演示器
        demonstrator = ConceptDemonstrator()
        demonstrator.setup()
        
        # 3. 批处理演示
        demonstrate_batching(demonstrator)
        
        # 4. 采样策略演示
        demonstrate_sampling_strategies(demonstrator)
        
        # 5. KV Cache概念
        demonstrate_kv_cache_concept()
        
        # 6. 内存管理
        demonstrate_memory_management()
        
        # 7. 学习总结
        print_summary()
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("💡 请检查环境配置或查看FAQ")
    
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()