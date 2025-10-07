# 推理引擎 (Inference Engine) 代码分析

本目录包含 nano-vLLM 推理引擎核心组件的代码分析文档。

## 📁 目录结构

- `llm_engine.py` - LLM 推理引擎主类分析
- `scheduler.py` - 请求调度器分析
- `block_manager.py` - 内存块管理器分析
- `model_runner.py` - 模型运行器分析
- `sequence.py` - 序列管理分析

## 🎯 引擎组件概述

### LLM 引擎
- **文件**: `nanovllm/engine/llm_engine.py`
- **功能**: 推理引擎主控制器
- **关键特性**: 请求管理、调度协调、资源分配

### 请求调度器
- **文件**: `nanovllm/engine/scheduler.py`
- **功能**: 智能请求调度和批处理
- **关键特性**: 内存感知调度、抢占机制、批量优化

### 内存块管理器
- **文件**: `nanovllm/engine/block_manager.py`
- **功能**: KV 缓存内存管理
- **关键特性**: 分页内存、前缀缓存、动态分配

### 模型运行器
- **文件**: `nanovllm/engine/model_runner.py`
- **功能**: 模型推理执行器
- **关键特性**: 批量推理、内存优化、并行处理

### 序列管理
- **文件**: `nanovllm/engine/sequence.py`
- **功能**: 生成序列状态管理
- **关键特性**: 序列状态跟踪、令牌管理、完成检测

## 🔗 相关模块

- [核心模块](../core/) - 核心接口和配置
- [层模块](../layers/) - 神经网络层实现
- [模型模块](../models/) - 具体模型架构