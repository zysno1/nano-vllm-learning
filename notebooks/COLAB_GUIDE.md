# 🌟 Google Colab 使用指南

本指南将帮助你在 Google Colab 中顺利运行 nano-vLLM 的 Jupyter Notebooks，享受免费的 GPU 资源和云端计算环境。

## 🚀 快速开始

### 1️⃣ 打开 Colab
访问 [Google Colab](https://colab.research.google.com/) 并登录你的 Google 账户。

### 2️⃣ 导入 Notebook
有三种方式导入我们的 Notebooks：

#### 方式一：直接从 GitHub 导入
1. 在 Colab 首页点击 "GitHub" 标签
2. 输入仓库地址：`your-username/nano-vllm-learning`
3. 选择你想要的 Notebook 文件

#### 方式二：通过链接打开
点击下面的链接直接在 Colab 中打开：

- 🔗 [00_环境设置和快速开始](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/00_环境设置和快速开始.ipynb)
- 🔗 [01_基础推理和批处理](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/01_基础推理和批处理.ipynb)
- 🔗 [02_高级调度和内存管理](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/02_高级调度和内存管理.ipynb)
- 🔗 [03_完整推理流程和端到端系统](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/03_完整推理流程和端到端系统.ipynb)

#### 方式三：上传本地文件
1. 下载 Notebook 文件到本地
2. 在 Colab 中点击 "上传" 标签
3. 选择并上传文件

## ⚙️ 环境配置

### 🔥 启用 GPU 加速
1. 点击菜单栏 `运行时` → `更改运行时类型`
2. 在 "硬件加速器" 中选择 `GPU`
3. 点击 `保存`

**GPU 类型说明**：
- **T4**: 免费用户可用，16GB 显存，适合大部分实验
- **V100**: Colab Pro 用户可用，32GB 显存，性能更强
- **A100**: Colab Pro+ 用户可用，40GB 显存，顶级性能

### 💾 检查资源限制
运行以下代码检查当前可用资源：

```python
# 检查 GPU
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 检查 RAM
import psutil
ram_gb = psutil.virtual_memory().total / 1024**3
print(f"可用 RAM: {ram_gb:.1f} GB")

# 检查磁盘空间
import shutil
disk_usage = shutil.disk_usage("/")
free_gb = disk_usage.free / 1024**3
print(f"可用磁盘空间: {free_gb:.1f} GB")
```

## 📦 依赖安装

### 🔄 自动安装
每个 Notebook 的第一个代码单元格都包含自动环境检测和依赖安装：

```python
# 检测 Colab 环境
try:
    import google.colab
    IN_COLAB = True
    print("🔍 检测到 Google Colab 环境")
except ImportError:
    IN_COLAB = False
    print("🔍 检测到本地环境")

# 自动安装依赖
if IN_COLAB:
    !pip install torch torchvision torchaudio
    !pip install transformers accelerate
    # ... 其他依赖
```

### 🛠️ 手动安装
如果自动安装失败，可以手动运行：

```python
# 基础依赖
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate
!pip install matplotlib seaborn numpy pandas tqdm
!pip install psutil jupyter-widgets

# 验证安装
import torch
import transformers
print("✅ 依赖安装成功")
```

## 💡 使用技巧

### 🎯 优化性能

#### 1. 使用混合精度
```python
# 启用自动混合精度
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
```

#### 2. 清理内存
```python
# 定期清理 GPU 内存
import gc
import torch

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# 在需要时调用
cleanup_memory()
```

#### 3. 监控资源使用
```python
# 实时监控 GPU 内存
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU 内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")

print_gpu_memory()
```

### 📁 文件管理

#### 上传文件
```python
from google.colab import files

# 上传文件
uploaded = files.upload()

# 查看上传的文件
import os
print("上传的文件:", list(uploaded.keys()))
```

#### 下载文件
```python
from google.colab import files

# 下载文件
files.download('output.txt')
```

#### 挂载 Google Drive
```python
from google.colab import drive

# 挂载 Google Drive
drive.mount('/content/drive')

# 访问文件
with open('/content/drive/My Drive/data.txt', 'r') as f:
    content = f.read()
```

### 🔄 版本控制

#### 保存到 GitHub
1. 在 Colab 中修改 Notebook
2. 点击 `文件` → `在 GitHub 中保存副本`
3. 选择仓库和分支
4. 添加提交信息并保存

#### 保存到 Google Drive
1. 点击 `文件` → `保存副本到云端硬盘`
2. 文件将自动保存到你的 Google Drive

## ⚠️ 注意事项

### 🕐 会话限制
- **免费用户**: 12小时连续使用限制
- **Colab Pro**: 24小时连续使用限制
- **空闲断开**: 90分钟无操作自动断开

### 💾 存储限制
- **临时存储**: 会话结束后丢失
- **持久存储**: 需要保存到 Google Drive 或下载

### 🔒 资源限制
- **GPU 配额**: 每日有使用限制
- **RAM 限制**: 标准版 12.7GB，Pro 版 25.5GB
- **磁盘空间**: 约 100GB 临时存储

## 🐛 常见问题

### ❓ GPU 不可用
**问题**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查运行时类型是否设置为 GPU
2. 重启运行时：`运行时` → `重启运行时`
3. 检查 GPU 配额是否用完

### ❓ 内存不足
**问题**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 减小批处理大小
batch_size = 8  # 从 32 减少到 8

# 使用梯度检查点
torch.utils.checkpoint.checkpoint_sequential()

# 清理内存
cleanup_memory()
```

### ❓ 依赖安装失败
**问题**: 包安装失败或版本冲突

**解决方案**:
```python
# 重启运行时后重新安装
!pip install --upgrade pip
!pip install --force-reinstall package_name

# 指定版本
!pip install torch==1.12.0
```

### ❓ 文件找不到
**问题**: `FileNotFoundError`

**解决方案**:
```python
# 检查当前目录
import os
print("当前目录:", os.getcwd())
print("文件列表:", os.listdir('.'))

# 克隆项目
!git clone https://github.com/your-username/nano-vllm-learning.git
%cd nano-vllm-learning
```

## 🎓 最佳实践

### 1. 📋 代码组织
- 将长代码分解为多个单元格
- 添加清晰的注释和说明
- 使用 Markdown 单元格添加文档

### 2. 🔄 定期保存
- 经常保存工作进度
- 使用版本控制跟踪更改
- 备份重要结果

### 3. 📊 结果可视化
```python
# 使用交互式图表
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

### 4. 🚀 性能优化
- 使用向量化操作
- 避免不必要的数据复制
- 合理使用缓存

## 🆘 获取帮助

### 📚 官方资源
- [Colab 官方文档](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

### 💬 社区支持
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-colaboratory)
- [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [GitHub Issues](https://github.com/your-username/nano-vllm-learning/issues)

### 🔧 调试技巧
```python
# 启用详细错误信息
import traceback
import sys

def detailed_error():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)

# 在 try-except 中使用
try:
    # 你的代码
    pass
except Exception as e:
    print(f"错误: {e}")
    detailed_error()
```

## 🎉 开始使用

现在你已经掌握了在 Colab 中使用 nano-vLLM Notebooks 的所有技巧！选择一个 Notebook 开始你的学习之旅：

1. 🚀 **初学者**: [00_环境设置和快速开始](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/00_环境设置和快速开始.ipynb)
2. 🔥 **进阶者**: [02_高级调度和内存管理](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/02_高级调度和内存管理.ipynb)
3. 🚀 **专家**: [03_完整推理流程和端到端系统](https://colab.research.google.com/github/your-username/nano-vllm-learning/blob/main/notebooks/03_完整推理流程和端到端系统.ipynb)

**Happy Coding in the Cloud! ☁️✨**

---

*记住：Colab 是一个强大的工具，但也要合理使用资源。尊重使用限制，与社区分享你的发现！*