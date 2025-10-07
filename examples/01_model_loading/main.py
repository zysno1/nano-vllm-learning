#!/usr/bin/env python3
"""
第一步：模型加载与基础配置

这个脚本演示了如何从 Hugging Face 加载大语言模型，
并展示各种配置参数对模型加载和性能的影响。

学习目标：
1. 理解模型加载的完整流程
2. 掌握配置参数的含义和作用
3. 学会监控和优化内存使用
4. 处理常见的加载问题
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Tuple, Optional
import psutil
import gc

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class ModelLoader:
    """模型加载器类，封装模型加载的完整流程"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        初始化模型加载器
        
        Args:
            model_path: 模型路径（本地路径或 HuggingFace 模型名）
            device: 设备类型（auto/cuda/cpu）
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self, 
                   torch_dtype: torch.dtype = torch.float16,
                   use_cache: bool = True,
                   low_cpu_mem_usage: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载模型和分词器
        
        Args:
            torch_dtype: 模型数据类型，影响内存使用和推理速度
            use_cache: 是否启用KV缓存，影响推理效率
            low_cpu_mem_usage: 是否使用低CPU内存模式
            
        Returns:
            (model, tokenizer): 加载好的模型和分词器
        """
        print(f"\n🚀 开始加载模型：{self.model_path}")
        print("=" * 50)
        
        # 记录开始时间和内存状态
        start_time = time.time()
        self._print_memory_status("加载前")
        
        try:
            # 步骤1：加载分词器
            print("\n📝 步骤 1/4：加载分词器")
            self.tokenizer = self._load_tokenizer()
            
            # 步骤2：加载配置
            print("\n⚙️  步骤 2/4：加载模型配置")
            self.config = self._load_config(torch_dtype, use_cache)
            
            # 步骤3：确定设备映射
            print("\n🎯 步骤 3/4：确定设备映射")
            self.device_map = self._get_device_map()
            print(f"   📍 设备映射：{self.device_map}")
            
            # 步骤4：加载模型权重
            print("\n🏋️  步骤 4/4：加载模型权重")
            self.model = self._load_model_weights(torch_dtype, low_cpu_mem_usage)
            
            # 验证和后处理
            print("\n🔍 验证模型配置...")
            self._validate_config()
            
            print("\n🔧 模型后处理...")
            self._post_process_model()
            
            # 显示最终结果
            total_time = time.time() - start_time
            print(f"\n✅ 模型加载完成！总耗时：{total_time:.2f}秒")
            
            self._print_memory_status("加载后")
            self._print_model_info()
            
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"\n❌ 模型加载失败：{str(e)}")
            self._cleanup_on_error()
            raise
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,  # 信任远程代码（某些模型需要）
                use_fast=True,           # 使用快速分词器
            )
            
            # 设置特殊 token（如果没有的话）
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            print(f"   ✅ 分词器加载成功，词汇表大小：{len(tokenizer):,}")
            return tokenizer
            
        except Exception as e:
            raise RuntimeError(f"分词器加载失败：{str(e)}")
    
    def _load_config(self, torch_dtype: torch.dtype, use_cache: bool) -> AutoConfig:
        """加载并配置模型配置"""
        try:
            config = AutoConfig.from_pretrained(self.model_path)
            
            # 设置推理优化参数
            config.torch_dtype = torch_dtype
            config.use_cache = use_cache
            
            print(f"   ✅ 模型配置加载成功")
            print(f"      - 模型类型：{config.model_type}")
            print(f"      - 隐藏层维度：{config.hidden_size}")
            print(f"      - 层数：{config.num_hidden_layers}")
            print(f"      - 注意力头数：{config.num_attention_heads}")
            print(f"      - 词汇表大小：{config.vocab_size:,}")
            
            return config
            
        except Exception as e:
            raise RuntimeError(f"模型配置加载失败：{str(e)}")
    
    def _validate_config(self):
        """验证模型配置的合理性"""
        print("🔍 验证模型配置...")
        
        # 检查基础参数
        assert self.config.vocab_size > 0, "词汇表大小必须大于 0"
        assert self.config.hidden_size > 0, "隐藏层维度必须大于 0"
        assert self.config.num_hidden_layers > 0, "层数必须大于 0"
        
        # 检查注意力头配置
        if hasattr(self.config, 'num_attention_heads'):
            assert self.config.hidden_size % self.config.num_attention_heads == 0, \
                "隐藏层维度必须能被注意力头数整除"
        
        # 检查分词器兼容性
        if len(self.tokenizer) != self.config.vocab_size:
            print(f"⚠️  警告：分词器词汇表大小 ({len(self.tokenizer):,}) "
                  f"与模型配置不匹配 ({self.config.vocab_size:,})")
        
        print("   ✅ 配置验证通过")
    
    def _load_model_weights(self, torch_dtype: torch.dtype, low_cpu_mem_usage: bool) -> AutoModelForCausalLM:
        """加载模型权重"""
        print("🔄 开始加载模型权重...")
        print(f"   📊 目标数据类型：{torch_dtype}")
        print(f"   💾 低内存模式：{low_cpu_mem_usage}")
        
        start_time = time.time()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=torch_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True,
            )
            
            load_time = time.time() - start_time
            print(f"   ✅ 模型权重加载成功，耗时：{load_time:.2f}秒")
            
            # 显示模型加载详情
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   📈 模型统计信息：")
            print(f"      - 总参数量：{param_count:,}")
            print(f"      - 可训练参数：{trainable_params:,}")
            print(f"      - 模型大小：{param_count * 2 / 1024**3:.2f} GB (FP16)")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"模型权重加载失败：{str(e)}")
    
    def _get_device_map(self) -> str:
        """确定设备映射策略"""
        if self.device == "auto":
            if torch.cuda.is_available():
                return "auto"  # 自动分配到可用的 GPU
            else:
                return "cpu"
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                print("⚠️  警告：CUDA 不可用，回退到 CPU")
                return "cpu"
            return "cuda"
        else:
            return self.device
    
    def _post_process_model(self):
        """模型后处理"""
        # 设置为评估模式
        self.model.eval()
        
        # 禁用梯度计算（推理时不需要）
        for param in self.model.parameters():
            param.requires_grad = False
        
        print("   ✅ 模型后处理完成")
    
    def _print_memory_status(self, stage: str):
        """打印内存使用状态"""
        print(f"📊 {stage}内存状态：")
        
        # CPU 内存
        cpu_memory = psutil.virtual_memory()
        print(f"   CPU 内存：{cpu_memory.used / 1024**3:.2f} GB / {cpu_memory.total / 1024**3:.2f} GB "
              f"({cpu_memory.percent:.1f}%)")
        
        # GPU 内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"   GPU {i} 内存：{allocated:.2f} GB 已分配, {reserved:.2f} GB 已预留, "
                      f"{total:.2f} GB 总计 ({allocated/total*100:.1f}%)")
    
    def _print_model_info(self):
        """打印模型详细信息"""
        print("\n📋 模型详细信息：")
        
        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   总参数量：{total_params:,}")
        print(f"   可训练参数：{trainable_params:,}")
        print(f"   模型大小：{total_params * 2 / 1024**3:.2f} GB (FP16)")
        
        # 模型结构信息
        print(f"   数据类型：{self.config.torch_dtype}")
        print(f"   使用缓存：{self.config.use_cache}")
        
        # 设备信息
        if hasattr(self.model, 'device'):
            print(f"   设备：{self.model.device}")
        
    def _cleanup_on_error(self):
        """错误时清理资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("🧹 已清理资源")


def test_model_loading():
    """测试模型加载功能"""
    print("🧪 开始模型加载测试...\n")
    
    # 使用较小的模型进行测试（避免内存不足）
    model_path = "microsoft/DialoGPT-small"  # 约 117M 参数
    
    try:
        # 创建模型加载器
        loader = ModelLoader(model_path, device="auto")
        
        # 加载模型
        model, tokenizer = loader.load_model(
            torch_dtype=torch.float16,
            use_cache=True,
            low_cpu_mem_usage=True
        )
        
        # 简单推理测试
        print("\n🔬 进行简单推理测试...")
        test_text = "Hello, how are you?"
        inputs = tokenizer.encode(test_text, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   输入：{test_text}")
        print(f"   输出：{response}")
        
        print("\n✅ 模型加载测试成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        raise


def demonstrate_different_configs():
    """演示不同配置的影响"""
    print("\n🔬 演示不同配置的影响...\n")
    
    model_path = "microsoft/DialoGPT-small"
    
    configs = [
        {"torch_dtype": torch.float32, "name": "FP32"},
        {"torch_dtype": torch.float16, "name": "FP16"},
    ]
    
    for config in configs:
        print(f"📊 测试配置：{config['name']}")
        try:
            loader = ModelLoader(model_path, device="auto")
            start_time = time.time()
            
            model, tokenizer = loader.load_model(
                torch_dtype=config["torch_dtype"],
                use_cache=True,
                low_cpu_mem_usage=True
            )
            
            load_time = time.time() - start_time
            print(f"   加载时间：{load_time:.2f} 秒\n")
            
            # 清理内存
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"   ❌ 配置 {config['name']} 测试失败：{str(e)}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 第一步：模型加载与基础配置")
    print("=" * 60)
    
    # 检查环境
    print(f"🔍 环境检查：")
    print(f"   Python 版本：{sys.version}")
    print(f"   PyTorch 版本：{torch.__version__}")
    print(f"   CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本：{torch.version.cuda}")
        print(f"   GPU 数量：{torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}：{torch.cuda.get_device_name(i)}")
    print()
    
    try:
        # 基础模型加载测试
        test_model_loading()
        
        # 不同配置对比测试
        demonstrate_different_configs()
        
        print("\n🎉 所有测试完成！")
        print("\n📚 学习要点总结：")
        print("   1. 模型加载包含分词器、配置、权重三个主要步骤")
        print("   2. 不同数据类型对内存和加载时间有显著影响")
        print("   3. 内存监控对于大模型部署至关重要")
        print("   4. 错误处理和资源清理是生产环境的必要考虑")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序")
    except Exception as e:
        print(f"\n💥 程序异常：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 资源清理完成")