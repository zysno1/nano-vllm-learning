#!/usr/bin/env python3
"""
🎯 nano-vLLM 基础推理模板

这是一个完整的推理代码模板，包含详细的注释和最佳实践。
您可以基于这个模板快速开始您的项目。

📋 模板特点:
- 完整的错误处理
- 详细的代码注释
- 可配置的参数
- 性能监控
- 资源管理

🎓 使用方法:
1. 复制这个模板到您的项目
2. 根据需要修改配置参数
3. 替换示例输入文本
4. 运行并观察结果

⚠️ 注意事项:
- 确保已安装必要的依赖包
- 根据您的硬件调整参数
- 注意内存使用情况
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

# 导入必要的库
try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        GenerationConfig
    )
except ImportError as e:
    print(f"❌ 缺少必要的依赖包: {e}")
    print("请运行: pip install torch transformers")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """推理配置类
    
    这个类包含了所有推理相关的配置参数。
    您可以根据需要修改这些参数。
    """
    
    # 模型配置
    model_name: str = "gpt2"  # 🔧 修改这里来使用不同的模型
    device: str = "auto"      # "auto", "cpu", "cuda", "cuda:0" 等
    torch_dtype: str = "auto" # "auto", "float16", "float32", "bfloat16"
    
    # 生成参数
    max_new_tokens: int = 50        # 🔧 生成的最大token数量
    temperature: float = 0.7        # 🔧 控制随机性 (0.1-2.0)
    top_p: float = 0.9             # 🔧 核采样参数 (0.1-1.0)
    top_k: int = 50                # 🔧 top-k采样参数
    do_sample: bool = True         # 🔧 是否使用采样
    repetition_penalty: float = 1.1 # 🔧 重复惩罚 (1.0-2.0)
    
    # 性能配置
    batch_size: int = 1            # 🔧 批处理大小
    use_cache: bool = True         # 🔧 是否使用KV缓存
    
    # 输出配置
    return_full_text: bool = False  # 🔧 是否返回完整文本
    clean_up_tokenization_spaces: bool = True
    
    def to_generation_config(self) -> GenerationConfig:
        """转换为GenerationConfig对象"""
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            use_cache=self.use_cache
        )

class NanoVLLMInference:
    """nano-vLLM 推理引擎
    
    这是主要的推理类，封装了所有推理相关的功能。
    """
    
    def __init__(self, config: InferenceConfig):
        """初始化推理引擎
        
        Args:
            config: 推理配置对象
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None
        
        # 性能统计
        self.stats = {
            "total_inferences": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "average_speed": 0.0
        }
        
        logger.info("🚀 初始化 nano-vLLM 推理引擎")
        
    def setup(self) -> bool:
        """设置推理环境
        
        Returns:
            bool: 设置是否成功
        """
        try:
            # 1. 设置设备
            self._setup_device()
            
            # 2. 加载分词器
            self._load_tokenizer()
            
            # 3. 加载模型
            self._load_model()
            
            # 4. 配置生成参数
            self._setup_generation_config()
            
            logger.info("✅ 推理引擎设置完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 推理引擎设置失败: {e}")
            return False
    
    def _setup_device(self):
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"🎮 使用GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device("cpu")
                logger.info("💻 使用CPU")
        else:
            self.device = torch.device(self.config.device)
            logger.info(f"🖥️  使用指定设备: {self.device}")
    
    def _load_tokenizer(self):
        """加载分词器"""
        logger.info(f"📝 加载分词器: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True  # 🔧 如果使用自定义模型，可能需要这个参数
        )
        
        # 设置pad_token（如果没有的话）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("🔧 设置pad_token为eos_token")
        
        logger.info(f"✅ 分词器加载完成，词汇表大小: {len(self.tokenizer):,}")
    
    def _load_model(self):
        """加载模型"""
        logger.info(f"🧠 加载模型: {self.config.model_name}")
        
        # 确定数据类型
        if self.config.torch_dtype == "auto":
            if self.device.type == "cuda":
                torch_dtype = torch.float16  # GPU使用float16节省内存
            else:
                torch_dtype = torch.float32  # CPU使用float32
        else:
            torch_dtype = getattr(torch, self.config.torch_dtype)
        
        logger.info(f"🔧 使用数据类型: {torch_dtype}")
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,  # 🔧 如果使用自定义模型，可能需要这个参数
            low_cpu_mem_usage=True   # 🔧 减少CPU内存使用
        )
        
        # 如果没有使用device_map，手动移动到设备
        if self.device.type != "cuda" or "device_map" not in locals():
            self.model = self.model.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 显示模型信息
        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ 模型加载完成")
        logger.info(f"📊 参数数量: {param_count:,}")
        logger.info(f"🖥️  模型设备: {next(self.model.parameters()).device}")
        
        # 显示内存使用情况
        if self.device.type == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"💾 GPU内存使用: {memory_used:.2f}GB / {memory_total:.2f}GB")
    
    def _setup_generation_config(self):
        """设置生成配置"""
        self.generation_config = self.config.to_generation_config()
        
        # 设置特殊token
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        
        logger.info("⚙️ 生成配置设置完成")
        logger.info(f"   max_new_tokens: {self.generation_config.max_new_tokens}")
        logger.info(f"   temperature: {self.generation_config.temperature}")
        logger.info(f"   top_p: {self.generation_config.top_p}")
        logger.info(f"   do_sample: {self.generation_config.do_sample}")
    
    def generate(self, 
                prompt: Union[str, List[str]], 
                **kwargs) -> Union[str, List[str]]:
        """生成文本
        
        Args:
            prompt: 输入提示词，可以是单个字符串或字符串列表
            **kwargs: 额外的生成参数，会覆盖默认配置
            
        Returns:
            生成的文本，格式与输入对应
        """
        
        # 🔧 您可以在这里添加自定义的预处理逻辑
        
        # 处理输入
        is_single = isinstance(prompt, str)
        if is_single:
            prompts = [prompt]
        else:
            prompts = prompt
        
        logger.info(f"🚀 开始生成，输入数量: {len(prompts)}")
        
        results = []
        total_start_time = time.time()
        
        try:
            # 批处理生成（这里简化为逐个处理）
            for i, single_prompt in enumerate(prompts):
                logger.info(f"📝 处理第 {i+1}/{len(prompts)} 个输入")
                
                # 单个推理
                result = self._single_inference(single_prompt, **kwargs)
                results.append(result)
                
                # 🔧 您可以在这里添加进度回调
                
        except Exception as e:
            logger.error(f"❌ 生成过程出错: {e}")
            raise
        
        # 更新统计信息
        total_time = time.time() - total_start_time
        self.stats["total_inferences"] += len(prompts)
        self.stats["total_time"] += total_time
        
        logger.info(f"✅ 生成完成，总耗时: {total_time:.3f}秒")
        
        # 返回结果
        if is_single:
            return results[0]
        else:
            return results
    
    def _single_inference(self, prompt: str, **kwargs) -> str:
        """单个推理
        
        Args:
            prompt: 输入提示词
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本
        """
        start_time = time.time()
        
        # 1. 编码输入
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors="pt",
            add_special_tokens=True
        )
        inputs = inputs.to(self.device)
        
        input_length = inputs.shape[1]
        logger.debug(f"📊 输入长度: {input_length} tokens")
        
        # 2. 准备生成配置
        generation_config = GenerationConfig(**self.generation_config.to_dict())
        
        # 应用额外参数
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)
                logger.debug(f"🔧 覆盖参数 {key}: {value}")
        
        # 3. 执行生成
        with torch.no_grad():  # 🔧 禁用梯度计算以节省内存
            outputs = self.model.generate(
                inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,  # 🔧 返回详细信息
                output_scores=False  # 🔧 不返回分数以节省内存
            )
        
        # 4. 解码输出
        generated_sequence = outputs.sequences[0]
        
        if self.config.return_full_text:
            # 返回完整文本
            generated_text = self.tokenizer.decode(
                generated_sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=self.config.clean_up_tokenization_spaces
            )
        else:
            # 只返回新生成的部分
            new_tokens = generated_sequence[input_length:]
            generated_text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=self.config.clean_up_tokenization_spaces
            )
        
        # 5. 计算统计信息
        end_time = time.time()
        inference_time = end_time - start_time
        generated_tokens = len(generated_sequence) - input_length
        tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0
        
        # 更新统计
        self.stats["total_tokens_generated"] += generated_tokens
        if self.stats["total_time"] > 0:
            self.stats["average_speed"] = self.stats["total_tokens_generated"] / self.stats["total_time"]
        
        logger.debug(f"⚡ 推理时间: {inference_time:.3f}秒")
        logger.debug(f"📊 生成tokens: {generated_tokens}")
        logger.debug(f"🚀 生成速度: {tokens_per_second:.1f} tokens/秒")
        
        # 🔧 您可以在这里添加自定义的后处理逻辑
        
        return generated_text.strip()
    
    def batch_generate(self, 
                      prompts: List[str], 
                      batch_size: Optional[int] = None,
                      **kwargs) -> List[str]:
        """批量生成文本
        
        Args:
            prompts: 输入提示词列表
            batch_size: 批处理大小，None表示使用配置中的值
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本列表
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        logger.info(f"🔄 批量生成，总数: {len(prompts)}, 批大小: {batch_size}")
        
        results = []
        
        # 分批处理
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            logger.info(f"📦 处理批次 {i//batch_size + 1}, 大小: {len(batch_prompts)}")
            
            # 🔧 这里可以实现真正的批处理推理
            # 目前简化为逐个处理
            batch_results = []
            for prompt in batch_prompts:
                result = self._single_inference(prompt, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            统计信息字典
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            "total_inferences": 0,
            "total_tokens_generated": 0,
            "total_time": 0.0,
            "average_speed": 0.0
        }
        logger.info("📊 统计信息已重置")
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理资源...")
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ 资源清理完成")

def main():
    """主函数 - 使用示例
    
    🔧 您可以修改这个函数来测试不同的配置和输入
    """
    
    # 1. 创建配置
    config = InferenceConfig(
        model_name="gpt2",           # 🔧 修改模型名称
        max_new_tokens=50,           # 🔧 修改生成长度
        temperature=0.7,             # 🔧 修改温度
        top_p=0.9,                   # 🔧 修改top_p
        do_sample=True               # 🔧 修改采样方式
    )
    
    # 2. 创建推理引擎
    inference_engine = NanoVLLMInference(config)
    
    try:
        # 3. 设置引擎
        if not inference_engine.setup():
            logger.error("❌ 推理引擎设置失败")
            return
        
        # 4. 准备测试输入
        test_prompts = [
            "从前有一只聪明的小猫",           # 🔧 修改测试提示词
            "人工智能的未来发展",
            "今天是美好的一天",
            "在遥远的星球上"
        ]
        
        print("\n" + "="*60)
        print("🎯 nano-vLLM 推理模板演示")
        print("="*60)
        
        # 5. 单个推理示例
        print(f"\n📝 单个推理示例:")
        single_prompt = test_prompts[0]
        print(f"输入: \"{single_prompt}\"")
        
        result = inference_engine.generate(single_prompt)
        print(f"输出: \"{result}\"")
        
        # 6. 批量推理示例
        print(f"\n📦 批量推理示例:")
        batch_results = inference_engine.batch_generate(test_prompts[:2])
        
        for i, (prompt, result) in enumerate(zip(test_prompts[:2], batch_results)):
            print(f"输入 {i+1}: \"{prompt}\"")
            print(f"输出 {i+1}: \"{result}\"")
            print()
        
        # 7. 参数覆盖示例
        print(f"🔧 参数覆盖示例:")
        creative_result = inference_engine.generate(
            "写一个创意故事:",
            temperature=1.2,      # 🔧 更高的创造性
            max_new_tokens=80     # 🔧 更长的输出
        )
        print(f"高创造性输出: \"{creative_result}\"")
        
        # 8. 显示统计信息
        print(f"\n📊 性能统计:")
        stats = inference_engine.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n✅ 演示完成！")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断")
    except Exception as e:
        logger.error(f"❌ 演示过程出错: {e}")
        raise
    finally:
        # 9. 清理资源
        inference_engine.cleanup()

if __name__ == "__main__":
    main()

# 🎯 使用提示:
# 
# 1. 基础使用:
#    python basic_inference_template.py
# 
# 2. 自定义配置:
#    修改 main() 函数中的 InferenceConfig 参数
# 
# 3. 集成到您的项目:
#    from basic_inference_template import NanoVLLMInference, InferenceConfig
#    
#    config = InferenceConfig(model_name="your-model")
#    engine = NanoVLLMInference(config)
#    engine.setup()
#    result = engine.generate("your prompt")
# 
# 4. 高级用法:
#    - 实现自定义的预处理和后处理
#    - 添加更多的生成参数
#    - 实现真正的批处理推理
#    - 添加缓存机制
#    - 集成到Web服务中
# 
# 📚 相关文档:
#    - docs/01-basic-concepts/ - 基础概念
#    - docs/02-architecture/ - 架构设计
#    - examples/01-basic-usage/ - 基础用法示例
#    - examples/02-step-by-step-tutorials/ - 分步教程