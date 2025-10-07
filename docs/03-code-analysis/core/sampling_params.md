# 采样参数 (Sampling Parameters) 代码分析

本文档分析 nano-vLLM 的采样参数配置实现，基于 `nanovllm/sampling_params.py` 的真实代码。

## 🎯 设计思想

**核心理念**：
- **灵活控制**：提供丰富的采样策略控制参数
- **默认优化**：提供经过优化的默认参数值
- **类型安全**：使用类型注解确保参数正确性
- **兼容性**：与主流 LLM 框架的参数保持兼容

## 🏗️ 采样参数架构

### SamplingParams 类

```python
@dataclass
class SamplingParams:
    """
    文本生成采样参数配置
    
    设计特点：
    - 涵盖所有主要采样策略
    - 提供合理的默认值
    - 支持参数验证
    - 易于序列化和反序列化
    """
    
    # 基础参数
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    
    # Top 采样参数
    top_p: float = 1.0
    top_k: int = -1
    
    # 长度控制
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    
    # 停止条件
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    
    # 输出控制
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    
    # 高级参数
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    
    # 输出格式
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    
    # Logprobs
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    detokenize: bool = True
```

## 🔧 核心采样策略

### 1. 温度采样 (Temperature Sampling)

**功能**：控制生成文本的随机性
- **参数**：`temperature` (0.0 - 2.0)
- **效果**：
  - `temperature = 0.0`：贪婪解码，确定性输出
  - `temperature = 1.0`：标准随机采样
  - `temperature > 1.0`：增加随机性，更多样化
  - `temperature < 1.0`：减少随机性，更保守

```python
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """应用温度缩放"""
    if temperature == 0.0:
        return logits  # 贪婪解码
    return logits / temperature
```

### 2. Top-p 采样 (Nucleus Sampling)

**功能**：基于累积概率的采样策略
- **参数**：`top_p` (0.0 - 1.0)
- **原理**：选择累积概率达到 p 的最小词汇集合
- **优势**：动态调整候选词数量，适应不同上下文

```python
def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """应用 Top-p 采样"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 移除不符合条件的词汇
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    return logits
```

### 3. Top-k 采样

**功能**：基于固定数量的采样策略
- **参数**：`top_k` (正整数或 -1)
- **原理**：只考虑概率最高的 k 个词汇
- **特点**：固定候选词数量，计算简单

```python
def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """应用 Top-k 采样"""
    if top_k <= 0:
        return logits
    
    # 找到第 k 大的值
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    min_value = values[..., -1].unsqueeze(-1)
    
    # 将小于第 k 大值的位置设为负无穷
    logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)
    
    return logits
```

### 4. 重复惩罚 (Repetition Penalty)

**功能**：减少重复生成的内容
- **参数**：`repetition_penalty` (> 0.0)
- **原理**：对已生成的词汇降低概率
- **效果**：`> 1.0` 惩罚重复，`< 1.0` 鼓励重复

```python
def apply_repetition_penalty(logits: torch.Tensor, 
                           input_ids: torch.Tensor, 
                           penalty: float) -> torch.Tensor:
    """应用重复惩罚"""
    if penalty == 1.0:
        return logits
    
    # 对已出现的词汇应用惩罚
    for token_id in input_ids.unique():
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty
    
    return logits
```

## 📊 参数配置指南

### 常用配置组合

#### 1. 创意写作模式
```python
creative_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    max_tokens=200
)
```

#### 2. 精确问答模式
```python
precise_params = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=100
)
```

#### 3. 代码生成模式
```python
code_params = SamplingParams(
    temperature=0.2,
    top_p=0.95,
    stop=["```", "\n\n"],
    max_tokens=500
)
```

#### 4. 对话模式
```python
chat_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    stop=["Human:", "Assistant:"],
    max_tokens=150
)
```

### 参数调优建议

| 场景 | Temperature | Top-p | Top-k | Repetition Penalty |
|------|-------------|-------|-------|-------------------|
| 事实问答 | 0.1-0.3 | 0.9-0.95 | 40-100 | 1.0-1.05 |
| 创意写作 | 0.7-1.0 | 0.8-0.9 | 50-100 | 1.1-1.2 |
| 代码生成 | 0.1-0.3 | 0.95 | -1 | 1.0-1.05 |
| 翻译任务 | 0.3-0.5 | 0.9 | 50 | 1.0 |
| 摘要生成 | 0.3-0.7 | 0.9 | 50 | 1.05-1.1 |

## 🚀 使用示例

### 基础使用

```python
from nanovllm.sampling_params import SamplingParams

# 创建采样参数
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
    stop=[".", "!", "?"]
)

# 使用参数生成文本
results = llm.generate(prompts, sampling_params=params)
```

### 批量不同参数

```python
# 为不同 prompt 使用不同参数
prompts = ["Write a story", "Solve this math problem"]
params_list = [
    SamplingParams(temperature=0.8, max_tokens=200),  # 创意写作
    SamplingParams(temperature=0.1, max_tokens=50)    # 数学问题
]

results = llm.generate(prompts, sampling_params=params_list)
```

### 动态参数调整

```python
# 根据内容类型动态调整参数
def get_sampling_params(content_type: str) -> SamplingParams:
    if content_type == "creative":
        return SamplingParams(temperature=0.8, top_p=0.9)
    elif content_type == "factual":
        return SamplingParams(temperature=0.2, top_p=0.95)
    else:
        return SamplingParams()  # 默认参数

params = get_sampling_params("creative")
results = llm.generate(prompts, sampling_params=params)
```

## 🔗 相关模块

- [LLM 接口](llm.md) - 使用采样参数进行生成
- [推理引擎](../engine/llm_engine.md) - 采样参数的应用
- [采样器](../layers/sampler.md) - 底层采样实现