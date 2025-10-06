#!/usr/bin/env python3
"""
🔧 nano-vLLM 环境检查工具

在开始学习之前，使用此脚本检查您的环境是否正确配置。
这个工具会检查：
1. Python版本
2. 必要的依赖包
3. GPU/CUDA支持
4. 内存状况
5. 模型下载能力

运行方式: python environment_check.py
"""

import sys
import subprocess
import importlib
import platform
import psutil
import os
from typing import List, Tuple, Dict, Any

class EnvironmentChecker:
    """环境检查器"""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        print("🐍 检查Python版本...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        print(f"   当前版本: Python {version_str}")
        
        if version.major == 3 and version.minor >= 8:
            print("   ✅ Python版本符合要求 (>= 3.8)")
            self.results.append(("Python版本", "✅ 通过", version_str))
            return True
        else:
            print("   ❌ Python版本过低，建议使用Python 3.8+")
            self.errors.append("Python版本过低")
            self.results.append(("Python版本", "❌ 失败", version_str))
            return False
    
    def check_required_packages(self) -> bool:
        """检查必要的Python包"""
        print("\n📦 检查必要的Python包...")
        
        required_packages = [
            ("torch", "PyTorch深度学习框架"),
            ("transformers", "Hugging Face Transformers"),
            ("numpy", "数值计算库"),
            ("psutil", "系统信息库")
        ]
        
        optional_packages = [
            ("accelerate", "模型加速库"),
            ("datasets", "数据集处理库"),
            ("tokenizers", "快速分词器"),
            ("safetensors", "安全张量存储")
        ]
        
        all_good = True
        
        # 检查必要包
        print("   必要包:")
        for package, description in required_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', '未知版本')
                print(f"   ✅ {package} ({version}) - {description}")
                self.results.append((f"包: {package}", "✅ 已安装", version))
            except ImportError:
                print(f"   ❌ {package} - {description} (未安装)")
                self.errors.append(f"缺少必要包: {package}")
                self.results.append((f"包: {package}", "❌ 未安装", "N/A"))
                all_good = False
        
        # 检查可选包
        print("\n   可选包:")
        for package, description in optional_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', '未知版本')
                print(f"   ✅ {package} ({version}) - {description}")
                self.results.append((f"可选包: {package}", "✅ 已安装", version))
            except ImportError:
                print(f"   ⚠️  {package} - {description} (未安装，建议安装)")
                self.warnings.append(f"建议安装可选包: {package}")
                self.results.append((f"可选包: {package}", "⚠️ 未安装", "N/A"))
        
        return all_good
    
    def check_gpu_support(self) -> Dict[str, Any]:
        """检查GPU和CUDA支持"""
        print("\n🎮 检查GPU和CUDA支持...")
        
        gpu_info = {
            "has_cuda": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_memory": []
        }
        
        try:
            import torch
            
            # 检查CUDA支持
            if torch.cuda.is_available():
                gpu_info["has_cuda"] = True
                gpu_info["cuda_version"] = torch.version.cuda
                gpu_info["gpu_count"] = torch.cuda.device_count()
                
                print(f"   ✅ CUDA可用 (版本: {gpu_info['cuda_version']})")
                print(f"   🎮 检测到 {gpu_info['gpu_count']} 个GPU:")
                
                for i in range(gpu_info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    gpu_name = props.name
                    gpu_memory = props.total_memory / 1024**3  # GB
                    
                    gpu_info["gpu_names"].append(gpu_name)
                    gpu_info["gpu_memory"].append(gpu_memory)
                    
                    print(f"     GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                self.results.append(("CUDA支持", "✅ 可用", f"{gpu_info['gpu_count']}个GPU"))
                
            else:
                print("   ⚠️  CUDA不可用，将使用CPU模式")
                print("   💡 CPU模式下推理速度较慢，建议使用GPU")
                self.warnings.append("CUDA不可用，性能可能受限")
                self.results.append(("CUDA支持", "⚠️ 不可用", "CPU模式"))
                
        except ImportError:
            print("   ❌ PyTorch未安装，无法检查CUDA支持")
            self.errors.append("PyTorch未安装")
            self.results.append(("CUDA支持", "❌ 无法检查", "PyTorch未安装"))
        
        return gpu_info
    
    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        print("\n💾 检查系统资源...")
        
        # 内存信息
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / 1024**3
        available_memory_gb = memory.available / 1024**3
        memory_usage_percent = memory.percent
        
        print(f"   💾 系统内存:")
        print(f"     总内存: {total_memory_gb:.1f} GB")
        print(f"     可用内存: {available_memory_gb:.1f} GB")
        print(f"     使用率: {memory_usage_percent:.1f}%")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"   🖥️  CPU信息:")
        print(f"     CPU核心数: {cpu_count}")
        print(f"     CPU使用率: {cpu_percent:.1f}%")
        
        # 磁盘空间
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / 1024**3
        
        print(f"   💿 磁盘空间:")
        print(f"     可用空间: {disk_free_gb:.1f} GB")
        
        # 评估资源充足性
        resource_status = "✅ 充足"
        if total_memory_gb < 8:
            resource_status = "⚠️ 内存可能不足"
            self.warnings.append("系统内存较少，可能影响性能")
        elif total_memory_gb < 4:
            resource_status = "❌ 内存严重不足"
            self.errors.append("系统内存过少")
        
        if disk_free_gb < 5:
            self.warnings.append("磁盘空间不足，可能影响模型下载")
        
        self.results.append(("系统内存", resource_status, f"{total_memory_gb:.1f} GB"))
        self.results.append(("CPU核心", "ℹ️ 信息", f"{cpu_count}核"))
        self.results.append(("磁盘空间", "ℹ️ 信息", f"{disk_free_gb:.1f} GB可用"))
        
        return {
            "total_memory_gb": total_memory_gb,
            "available_memory_gb": available_memory_gb,
            "cpu_count": cpu_count,
            "disk_free_gb": disk_free_gb
        }
    
    def check_network_and_model_access(self) -> bool:
        """检查网络连接和模型访问"""
        print("\n🌐 检查网络连接和模型访问...")
        
        try:
            # 尝试导入transformers并测试模型下载
            from transformers import AutoTokenizer
            
            print("   🔍 测试模型下载能力...")
            
            # 尝试加载一个小模型的tokenizer
            test_model = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(test_model)
            
            print(f"   ✅ 成功访问Hugging Face Hub")
            print(f"   ✅ 成功下载测试模型: {test_model}")
            
            self.results.append(("网络连接", "✅ 正常", "可访问HF Hub"))
            return True
            
        except Exception as e:
            print(f"   ❌ 网络或模型访问失败: {e}")
            print("   💡 可能的解决方案:")
            print("     1. 检查网络连接")
            print("     2. 配置代理设置")
            print("     3. 使用本地模型")
            
            self.errors.append("无法访问Hugging Face Hub")
            self.results.append(("网络连接", "❌ 失败", str(e)))
            return False
    
    def check_installation_commands(self):
        """提供安装命令建议"""
        print("\n📋 安装命令建议:")
        
        if self.errors:
            print("   🔧 修复错误的命令:")
            
            if any("Python版本" in error for error in self.errors):
                print("     # 升级Python (建议使用conda或pyenv)")
                print("     conda install python=3.9")
                print("     # 或")
                print("     pyenv install 3.9.0")
            
            if any("torch" in error for error in self.errors):
                print("     # 安装PyTorch")
                print("     pip install torch torchvision torchaudio")
                print("     # 或GPU版本:")
                print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
            if any("transformers" in error for error in self.errors):
                print("     # 安装Transformers")
                print("     pip install transformers")
            
            if any("numpy" in error for error in self.errors):
                print("     # 安装NumPy")
                print("     pip install numpy")
        
        if self.warnings:
            print("\n   💡 改善性能的可选安装:")
            
            if any("accelerate" in warning for warning in self.warnings):
                print("     pip install accelerate")
            
            if any("CUDA" in warning for warning in self.warnings):
                print("     # 安装CUDA版本的PyTorch")
                print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    def print_summary(self):
        """打印检查总结"""
        print("\n" + "=" * 60)
        print("📊 环境检查总结")
        print("=" * 60)
        
        # 打印详细结果
        print("\n📋 详细检查结果:")
        for item, status, detail in self.results:
            print(f"   {item:<20} {status:<12} {detail}")
        
        # 总体状态
        print(f"\n🎯 总体状态:")
        if not self.errors:
            if not self.warnings:
                print("   🎉 完美！您的环境已完全配置好")
                print("   🚀 可以开始学习nano-vLLM了")
            else:
                print("   ✅ 良好！环境基本配置正确")
                print("   💡 有一些可以改善的地方")
        else:
            print("   ⚠️  需要修复一些问题才能正常使用")
        
        # 错误和警告
        if self.errors:
            print(f"\n❌ 需要修复的错误 ({len(self.errors)}个):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings:
            print(f"\n⚠️  建议改善的项目 ({len(self.warnings)}个):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # 下一步建议
        print(f"\n🚀 下一步建议:")
        if not self.errors:
            print("   1. 运行 hello_vllm.py 进行第一次推理")
            print("   2. 学习 basic_concepts.py 了解核心概念")
            print("   3. 查看 docs/00-quick-start/README.md")
        else:
            print("   1. 根据上面的命令修复环境问题")
            print("   2. 重新运行此检查脚本")
            print("   3. 如有问题，查看 docs/00-quick-start/faq.md")

def main():
    """主函数"""
    print("🔧 nano-vLLM 环境检查工具")
    print("=" * 50)
    print("正在检查您的学习环境...")
    
    checker = EnvironmentChecker()
    
    try:
        # 执行各项检查
        checker.check_python_version()
        checker.check_required_packages()
        gpu_info = checker.check_gpu_support()
        system_info = checker.check_system_resources()
        checker.check_network_and_model_access()
        
        # 提供安装建议
        checker.check_installation_commands()
        
        # 打印总结
        checker.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  检查被用户中断")
    except Exception as e:
        print(f"\n❌ 检查过程中出现意外错误: {e}")
        print("💡 请报告此问题或查看FAQ")

if __name__ == "__main__":
    main()