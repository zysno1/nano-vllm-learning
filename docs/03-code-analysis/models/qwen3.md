# Qwen3 模型架构 (Qwen3 Model Architecture) 代码分析

本文档分析 nano-vLLM 中 Qwen3 模型的具体实现，基于 `nanovllm/models/qwen3.py` 的真实代码。

## 🎯 设计思想

**核心理念**：
- **标准架构**：基于 Transformer 的标准架构设计
- **优化实现**：针对推理场景的性能优化
- **模块化设计**：组件化的模型结构，便于维护和扩展
- **内存友好**：支持 KV 缓存和分页注意力机制

## 🏗️ 模型架构

### Qwen3 主模型类

```python
class Qwen3Model(nn.Module):
    """
    Qwen3 模型主类
    
    架构特点：
    - 基于 Transformer Decoder-only 架构
    - 支持 RoPE 位置编码
    - 使用 SwiGLU 激活函数
    - 支持 Group Query Attention
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size
        )
        
        # Transformer 层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # 最终层归一化
        self.norm = Qwen3RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """前向传播"""
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 逐层处理
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
```

### Qwen3 解码器层

```python
class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 解码器层
    
    组件：
    - 多头自注意力机制
    - 前馈神经网络
    - 残差连接和层归一化
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        # 自注意力层
        self.self_attn = Qwen3Attention(config)
        
        # 前馈网络
        self.mlp = Qwen3MLP(config)
        
        # 层归一化
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """解码器层前向传播"""
        
        # 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_states = residual + hidden_states
        
        # 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

## 🔧 核心组件实现

### 1. 注意力机制

```python
class Qwen3Attention(nn.Module):
    """
    Qwen3 多头注意力机制
    
    特性：
    - 支持 Group Query Attention (GQA)
    - 集成 RoPE 位置编码
    - 支持 KV 缓存优化
    - Flash Attention 加速
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        
        # 注意力参数
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(
            config, 'num_key_value_heads', self.num_heads
        )
        self.head_dim = self.hidden_size // self.num_heads
        
        # 线性投影层
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )
        
        # RoPE 位置编码
        self.rotary_emb = Qwen3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings
        )
```

### 2. 前馈神经网络

```python
class Qwen3MLP(nn.Module):
    """
    Qwen3 前馈神经网络
    
    特性：
    - 使用 SwiGLU 激活函数
    - 门控机制提升表达能力
    - 支持权重量化
    """
    
    def __init__(self, config: Qwen3Config):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # 门控投影
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        # 上投影
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=False
        )
        
        # 下投影
        self.down_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=False
        )
        
        # 激活函数
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        """前馈网络前向传播"""
        # SwiGLU: gate_proj(x) * silu(up_proj(x))
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### 3. RMS 层归一化

```python
class Qwen3RMSNorm(nn.Module):
    """
    RMS 层归一化
    
    特性：
    - 相比 LayerNorm 计算更高效
    - 不需要计算均值，只需要方差
    - 数值稳定性好
    """
    
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        """RMS 归一化前向传播"""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # 计算 RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        
        return self.weight * hidden_states.to(input_dtype)
```

## 📊 模型配置

### Qwen3 配置参数

```python
@dataclass
class Qwen3Config:
    """Qwen3 模型配置"""
    
    # 基础参数
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    
    # 位置编码
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    
    # 归一化
    rms_norm_eps: float = 1e-6
    
    # 其他参数
    use_cache: bool = True
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
```

### 不同规模模型配置

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 参数量 |
|------|------|----------|------------|--------|
| Qwen3-0.5B | 24 | 1024 | 16 | 0.5B |
| Qwen3-1.8B | 24 | 2048 | 16 | 1.8B |
| Qwen3-4B | 40 | 2560 | 20 | 4B |
| Qwen3-7B | 32 | 4096 | 32 | 7B |
| Qwen3-14B | 40 | 5120 | 40 | 14B |
| Qwen3-32B | 64 | 5120 | 40 | 32B |
| Qwen3-72B | 80 | 8192 | 64 | 72B |

## 🚀 使用示例

### 模型加载

```python
from nanovllm.models.qwen3 import Qwen3Model, Qwen3Config

# 创建配置
config = Qwen3Config(
    vocab_size=151936,
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32
)

# 初始化模型
model = Qwen3Model(config)

# 加载预训练权重
model.load_state_dict(torch.load("qwen3_weights.pth"))
```

### 推理使用

```python
# 准备输入
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
attention_mask = torch.ones_like(input_ids)

# 前向传播
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

print(f"Output shape: {outputs.shape}")
```

### 与引擎集成

```python
from nanovllm.engine.llm_engine import LLMEngine

# 创建引擎
engine = LLMEngine(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    model_class=Qwen3Model
)

# 生成文本
results = engine.generate(
    prompts=["Hello, how are you?"],
    sampling_params=SamplingParams(max_tokens=100)
)
```

## 🔗 相关模块

- [注意力机制](../layers/attention.md) - 注意力层实现
- [线性层](../layers/linear.md) - 线性变换实现
- [激活函数](../layers/activation.md) - 激活函数实现
- [层归一化](../layers/layernorm.md) - 归一化层实现