# 第一步：模型加载与基础配置 🔧

欢迎来到 nano-vLLM 学习教程的第一步！在这个章节中，我们将学习如何从 Hugging Face 加载大语言模型，并理解各种配置参数的作用。

## 🎯 学习目标

通过本章节的学习，你将：
- 理解大语言模型的加载过程和内存分布
- 掌握模型配置参数的含义和调优方法
- 学会处理模型加载过程中的常见问题
- 建立对推理引擎初始化流程的基础认知

## 📚 理论背景

### 大语言模型的存储结构

大语言模型通常以以下格式存储：

```
model_directory/
├── config.json          # 模型配置文件
├── tokenizer.json       # 分词器配置
├── tokenizer_config.json
├── pytorch_model.bin    # 模型权重文件（PyTorch格式）
├── model.safetensors    # 模型权重文件（SafeTensors格式）
└── special_tokens_map.json
```

### 模型加载的关键步骤

1. **配置解析**：读取 `config.json`，获取模型架构信息
2. **权重加载**：从权重文件中加载模型参数
3. **内存分配**：在 GPU/CPU 上为模型分配内存空间
4. **模型初始化**：构建模型结构，加载预训练权重
5. **优化设置**：应用推理优化（如半精度、融合算子等）

## 🔍 核心概念详解

### 1. 模型配置参数

#### 基础架构参数
```python
{
    "vocab_size": 32000,           # 词汇表大小
    "hidden_size": 4096,           # 隐藏层维度
    "intermediate_size": 11008,    # FFN 中间层维度
    "num_hidden_layers": 32,       # Transformer 层数
    "num_attention_heads": 32,     # 注意力头数
    "max_position_embeddings": 2048, # 最大序列长度
    "rms_norm_eps": 1e-6,         # RMS 归一化的 epsilon
    "rope_theta": 10000.0,        # RoPE 位置编码的 theta 参数
}
```

#### 推理优化参数
```python
{
    "torch_dtype": "float16",      # 数据类型（float32/float16/bfloat16）
    "use_cache": True,            # 是否使用 KV Cache
    "pad_token_id": 0,            # 填充 token ID
    "bos_token_id": 1,            # 开始 token ID
    "eos_token_id": 2,            # 结束 token ID
}
```

### 2. 内存管理策略

#### GPU 内存分配
```python
# 模型权重内存计算
model_memory = num_parameters * bytes_per_parameter
# 例如：7B 模型，float16 精度
# 7 * 10^9 * 2 bytes = 14 GB

# KV Cache 内存计算  
kv_cache_memory = batch_size * seq_len * hidden_size * num_layers * 2 * bytes_per_element
# 例如：batch=8, seq_len=2048, hidden=4096, layers=32, float16
# 8 * 2048 * 4096 * 32 * 2 * 2 bytes = 8.6 GB
```

#### 内存优化技术
- **模型分片**：将大模型分割到多个 GPU 上
- **梯度检查点**：牺牲计算换取内存
- **混合精度**：使用 float16 减少内存占用
- **动态形状**：根据实际序列长度分配内存

### 3. 分词器（Tokenizer）

#### 分词过程
```python
# 文本 → Token IDs
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
# [1, 15043, 29892, 920, 526, 366, 29973]

# Token IDs → 文本
decoded = tokenizer.decode(tokens)
# "Hello, how are you?"
```

#### 特殊 Token 处理
- **BOS (Beginning of Sequence)**：序列开始标记
- **EOS (End of Sequence)**：序列结束标记  
- **PAD (Padding)**：填充标记，用于批处理
- **UNK (Unknown)**：未知词标记

## 💻 代码实现解析

### 核心加载流程

```python
def load_model(model_path: str, device: str = "cuda"):
    """
    加载大语言模型的完整流程
    
    Args:
        model_path: 模型路径（本地路径或 HuggingFace 模型名）
        device: 设备类型（cuda/cpu）
    
    Returns:
        model: 加载完成的模型
        tokenizer: 对应的分词器
    """
    
    # 1. 加载分词器
    print("🔄 正在加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 加载模型配置
    print("🔄 正在加载模型配置...")
    config = AutoConfig.from_pretrained(model_path)
    
    # 3. 设置推理优化参数
    config.torch_dtype = torch.float16  # 使用半精度
    config.use_cache = True             # 启用 KV Cache
    
    # 4. 加载模型权重
    print("🔄 正在加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动分配设备
        low_cpu_mem_usage=True,  # 减少 CPU 内存使用
    )
    
    # 5. 模型优化设置
    model.eval()  # 设置为评估模式
    
    print("✅ 模型加载完成！")
    return model, tokenizer
```

### 内存监控工具

```python
def monitor_gpu_memory():
    """监控 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        print(f"📊 GPU 内存使用情况：")
        print(f"   已分配：{allocated:.2f} GB")
        print(f"   已预留：{reserved:.2f} GB")
        print(f"   利用率：{allocated/reserved*100:.1f}%")
```

### 配置验证工具

```python
def validate_model_config(config):
    """验证模型配置的合理性"""
    
    # 检查基础参数
    assert config.vocab_size > 0, "词汇表大小必须大于 0"
    assert config.hidden_size > 0, "隐藏层维度必须大于 0"
    assert config.num_hidden_layers > 0, "层数必须大于 0"
    
    # 检查注意力头配置
    assert config.hidden_size % config.num_attention_heads == 0, \
        "隐藏层维度必须能被注意力头数整除"
    
    # 检查数据类型
    supported_dtypes = ["float32", "float16", "bfloat16"]
    assert str(config.torch_dtype).split('.')[-1] in supported_dtypes, \
        f"不支持的数据类型：{config.torch_dtype}"
    
    print("✅ 模型配置验证通过！")
```

## 🚀 实践练习

### 练习 1：基础模型加载

运行以下代码，观察模型加载过程：

```bash
cd examples/01_model_loading
python main.py
```

**观察要点**：
- 加载过程中的内存变化
- 不同精度设置的影响
- 配置参数的作用

### 练习 2：内存优化实验

尝试修改以下参数，观察内存使用的变化：

```python
# 在 main.py 中修改这些参数
torch_dtype = torch.float32  # 改为 float32
device_map = "cpu"           # 改为 CPU 加载
low_cpu_mem_usage = False    # 关闭内存优化
```

### 练习 3：错误处理实验

故意使用错误的模型路径或配置，观察错误信息：

```python
# 测试错误情况
model_path = "non-existent-model"  # 不存在的模型
device = "cuda:10"                 # 不存在的 GPU
```

## 🔧 常见问题与解决方案

### 问题 1：CUDA 内存不足

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**：
```python
# 1. 使用更小的模型
model_path = "microsoft/DialoGPT-small"  # 而不是 large

# 2. 使用 CPU 加载
device_map = "cpu"

# 3. 使用模型分片
device_map = "auto"  # 自动分配到多个 GPU

# 4. 清理 GPU 缓存
torch.cuda.empty_cache()
```

### 问题 2：模型加载速度慢

**优化方案**：
```python
# 1. 使用本地缓存
from transformers import AutoModel
model = AutoModel.from_pretrained(
    model_path,
    cache_dir="./model_cache",  # 指定缓存目录
    local_files_only=True,      # 只使用本地文件
)

# 2. 使用 SafeTensors 格式
# SafeTensors 格式加载更快，更安全

# 3. 预下载模型
# 提前下载模型到本地，避免网络延迟
```

### 问题 3：分词器不匹配

**检查方法**：
```python
# 验证分词器和模型的兼容性
def check_tokenizer_model_compatibility(tokenizer, model):
    vocab_size_tokenizer = len(tokenizer)
    vocab_size_model = model.config.vocab_size
    
    if vocab_size_tokenizer != vocab_size_model:
        print(f"⚠️  警告：分词器词汇表大小 ({vocab_size_tokenizer}) "
              f"与模型配置不匹配 ({vocab_size_model})")
    else:
        print("✅ 分词器与模型兼容")
```

## 📊 性能基准测试

### 加载时间对比

| 模型大小 | 精度 | 设备 | 加载时间 | 内存占用 |
|---------|------|------|----------|----------|
| 1.3B    | FP16 | GPU  | 15s      | 2.6GB    |
| 1.3B    | FP32 | GPU  | 18s      | 5.2GB    |
| 7B      | FP16 | GPU  | 45s      | 14GB     |
| 7B      | FP16 | CPU  | 120s     | 14GB     |

### 内存使用分析

```python
def analyze_memory_usage(model):
    """分析模型内存使用情况"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数统计：")
    print(f"   总参数量：{total_params:,}")
    print(f"   可训练参数：{trainable_params:,}")
    print(f"   参数占用内存：{total_params * 2 / 1024**3:.2f} GB (FP16)")
```

## 🎓 学习检查点

在继续下一步之前，请确保你理解了以下概念：

### 基础概念检查
- [ ] 模型配置文件包含哪些关键信息？
- [ ] 不同数据类型（FP32/FP16）对内存和性能的影响？
- [ ] 分词器的作用和工作原理？
- [ ] KV Cache 是什么，为什么重要？

### 实践技能检查
- [ ] 能够成功加载一个大语言模型？
- [ ] 会监控和分析 GPU 内存使用情况？
- [ ] 能够处理常见的加载错误？
- [ ] 理解不同配置参数的作用？

### 思考题
1. **内存优化**：如果你的 GPU 内存不足以加载完整模型，有哪些解决方案？
2. **精度选择**：什么情况下应该选择 FP32 而不是 FP16？
3. **缓存策略**：如何设计一个高效的模型缓存系统？
4. **错误恢复**：如何实现模型加载失败后的自动重试机制？

## 🔗 相关资源

### 官方文档
- [Transformers 模型加载指南](https://huggingface.co/docs/transformers/model_doc/auto)
- [PyTorch 内存管理](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

### 推荐阅读
- [大语言模型的内存优化技术](https://arxiv.org/abs/2205.05198)
- [SafeTensors：更安全的模型存储格式](https://github.com/huggingface/safetensors)

---

**恭喜！** 🎉 你已经掌握了模型加载的基础知识。

现在让我们进入下一步：[核心引擎的初始化](../02_llm_engine/)，学习如何构建 LLM 推理引擎！