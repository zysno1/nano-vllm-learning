# 神经网络层 (Neural Network Layers) 代码分析

本目录包含 nano-vLLM 神经网络层实现的代码分析文档。

## 📁 目录结构

- `attention.py` - 注意力机制层分析
- `linear.py` - 线性变换层分析
- `layernorm.py` - 层归一化分析
- `activation.py` - 激活函数层分析
- `embed_head.py` - 嵌入和输出头分析
- `rotary_embedding.py` - 旋转位置编码分析
- `sampler.py` - 采样器实现分析

## 🎯 网络层组件概述

### 注意力机制
- **文件**: `nanovllm/layers/attention.py`
- **功能**: 多头注意力机制实现
- **关键特性**: Flash Attention、KV 缓存、分页注意力

### 线性变换层
- **文件**: `nanovllm/layers/linear.py`
- **功能**: 线性变换和矩阵乘法
- **关键特性**: 权重量化、并行计算、内存优化

### 层归一化
- **文件**: `nanovllm/layers/layernorm.py`
- **功能**: 层归一化实现
- **关键特性**: RMS Norm、数值稳定性、高效计算

### 激活函数
- **文件**: `nanovllm/layers/activation.py`
- **功能**: 各种激活函数实现
- **关键特性**: SwiGLU、GELU、ReLU 变体

### 嵌入和输出头
- **文件**: `nanovllm/layers/embed_head.py`
- **功能**: 词嵌入和语言模型头
- **关键特性**: 词汇表映射、权重共享、输出投影

### 旋转位置编码
- **文件**: `nanovllm/layers/rotary_embedding.py`
- **功能**: RoPE 位置编码实现
- **关键特性**: 相对位置、旋转矩阵、长度外推

### 采样器
- **文件**: `nanovllm/layers/sampler.py`
- **功能**: 文本生成采样策略
- **关键特性**: Top-k/Top-p、温度采样、束搜索

## 🔗 相关模块

- [引擎模块](../engine/) - 推理引擎实现
- [模型模块](../models/) - 完整模型架构
- [工具模块](../utils/) - 辅助计算函数