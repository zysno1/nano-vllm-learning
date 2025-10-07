#!/usr/bin/env python3
"""
批处理演示脚本
展示 Continuous Batching 与传统静态批处理的区别

基于费曼学习法：通过对比实验理解核心概念
"""

import time
import asyncio
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue, Empty
import json

class RequestStatus(Enum):
    """请求状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class InferenceRequest:
    """推理请求数据类"""
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float = 0.7
    created_at: float = None
    status: RequestStatus = RequestStatus.PENDING
    generated_tokens: List[str] = None
    completion_time: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.generated_tokens is None:
            self.generated_tokens = []

class StaticBatchProcessor:
    """传统静态批处理器"""
    
    def __init__(self, batch_size: int = 4, max_wait_time: float = 2.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.processing_stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_wait_time': 0,
            'total_processing_time': 0,
            'completed_requests': []
        }
    
    def add_request(self, request: InferenceRequest):
        """添加请求到队列"""
        request.status = RequestStatus.PENDING
        self.pending_requests.append(request)
        self.processing_stats['total_requests'] += 1
        print(f"📥 静态批处理器收到请求 {request.request_id}")
    
    def process_batch(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        """处理一个批次的请求"""
        print(f"🔄 开始处理批次，包含 {len(batch)} 个请求")
        
        # 更新状态
        for req in batch:
            req.status = RequestStatus.PROCESSING
        
        # 模拟批处理：等待最慢的请求完成
        max_tokens = max(req.max_tokens for req in batch)
        processing_time = max_tokens * 0.01  # 每个token 10ms
        
        print(f"⏳ 批处理需要等待最长序列完成 ({max_tokens} tokens, {processing_time:.2f}s)")
        time.sleep(min(processing_time, 1.0))  # 限制最大等待时间
        
        # 所有请求同时完成
        completion_time = time.time()
        results = []
        
        for req in batch:
            # 生成模拟tokens
            for i in range(req.max_tokens):
                req.generated_tokens.append(f"token_{i}")
            
            req.status = RequestStatus.COMPLETED
            req.completion_time = completion_time
            
            # 计算延迟
            latency = completion_time - req.created_at
            
            results.append(req)
            self.processing_stats['completed_requests'].append({
                'request_id': req.request_id,
                'latency': latency,
                'tokens': len(req.generated_tokens),
                'wait_time': completion_time - req.created_at - processing_time
            })
            
            print(f"✅ 请求 {req.request_id} 完成，延迟: {latency:.2f}s")
        
        return results
    
    def run_processing_loop(self):
        """运行处理循环"""
        print("🚀 启动静态批处理器")
        
        while True:
            if len(self.pending_requests) == 0:
                time.sleep(0.1)
                continue
            
            # 等待批次填满或超时
            batch_start_time = time.time()
            
            while (len(self.pending_requests) < self.batch_size and 
                   time.time() - batch_start_time < self.max_wait_time):
                time.sleep(0.1)
            
            if len(self.pending_requests) == 0:
                continue
            
            # 创建批次
            batch_size = min(self.batch_size, len(self.pending_requests))
            current_batch = self.pending_requests[:batch_size]
            self.pending_requests = self.pending_requests[batch_size:]
            
            wait_time = time.time() - batch_start_time
            self.processing_stats['total_wait_time'] += wait_time
            self.processing_stats['total_batches'] += 1
            
            print(f"📦 创建批次 {self.processing_stats['total_batches']}，等待时间: {wait_time:.2f}s")
            
            # 处理批次
            start_time = time.time()
            self.process_batch(current_batch)
            processing_time = time.time() - start_time
            self.processing_stats['total_processing_time'] += processing_time

class ContinuousBatchProcessor:
    """Continuous Batching 处理器"""
    
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.active_requests = {}  # request_id -> request
        self.request_queue = Queue()
        self.processing_stats = {
            'total_requests': 0,
            'total_steps': 0,
            'completed_requests': []
        }
        self.running = False
    
    def add_request(self, request: InferenceRequest):
        """添加请求到队列"""
        request.status = RequestStatus.PENDING
        self.request_queue.put(request)
        self.processing_stats['total_requests'] += 1
        print(f"📥 Continuous Batching 收到请求 {request.request_id}")
    
    def process_step(self):
        """处理一个推理步骤"""
        # 添加新请求到活跃批次
        while (len(self.active_requests) < self.max_batch_size and 
               not self.request_queue.empty()):
            try:
                request = self.request_queue.get_nowait()
                request.status = RequestStatus.PROCESSING
                self.active_requests[request.request_id] = request
                print(f"🔄 请求 {request.request_id} 加入活跃批次")
            except Empty:
                break
        
        if not self.active_requests:
            return
        
        print(f"⚡ 处理步骤 {self.processing_stats['total_steps'] + 1}，"
              f"活跃请求: {len(self.active_requests)}")
        
        # 模拟一步推理
        step_time = 0.05  # 每步50ms
        time.sleep(step_time)
        
        # 更新所有活跃请求
        completed_requests = []
        
        for request_id, request in list(self.active_requests.items()):
            # 生成一个token
            token_index = len(request.generated_tokens)
            request.generated_tokens.append(f"token_{token_index}")
            
            # 检查是否完成
            if len(request.generated_tokens) >= request.max_tokens:
                request.status = RequestStatus.COMPLETED
                request.completion_time = time.time()
                
                # 计算延迟
                latency = request.completion_time - request.created_at
                
                completed_requests.append(request)
                self.processing_stats['completed_requests'].append({
                    'request_id': request.request_id,
                    'latency': latency,
                    'tokens': len(request.generated_tokens)
                })
                
                print(f"✅ 请求 {request.request_id} 完成，延迟: {latency:.2f}s")
        
        # 移除完成的请求
        for request in completed_requests:
            del self.active_requests[request.request_id]
        
        self.processing_stats['total_steps'] += 1
    
    def run_processing_loop(self):
        """运行处理循环"""
        print("🚀 启动 Continuous Batching 处理器")
        self.running = True
        
        while self.running:
            self.process_step()
            
            # 如果没有活跃请求且队列为空，短暂休息
            if not self.active_requests and self.request_queue.empty():
                time.sleep(0.1)
    
    def stop(self):
        """停止处理器"""
        self.running = False

def generate_test_requests(num_requests: int = 10) -> List[InferenceRequest]:
    """生成测试请求"""
    requests = []
    
    for i in range(num_requests):
        request = InferenceRequest(
            request_id=f"req_{i:03d}",
            prompt=f"Generate a story about {random.choice(['dragons', 'space', 'magic', 'robots', 'ocean'])}",
            max_tokens=random.randint(20, 100),
            temperature=0.7
        )
        requests.append(request)
    
    return requests

def run_static_batch_demo():
    """运行静态批处理演示"""
    print("\n" + "="*60)
    print("🔸 静态批处理演示")
    print("="*60)
    
    processor = StaticBatchProcessor(batch_size=4, max_wait_time=1.0)
    
    # 在单独线程中运行处理器
    processing_thread = threading.Thread(target=processor.run_processing_loop, daemon=True)
    processing_thread.start()
    
    # 生成测试请求
    requests = generate_test_requests(10)
    
    # 模拟请求到达（有时间间隔）
    for i, request in enumerate(requests):
        processor.add_request(request)
        
        # 模拟请求间隔
        if i < len(requests) - 1:
            time.sleep(random.uniform(0.1, 0.5))
    
    # 等待所有请求完成
    while len(processor.processing_stats['completed_requests']) < len(requests):
        time.sleep(0.1)
    
    return processor.processing_stats

def run_continuous_batch_demo():
    """运行 Continuous Batching 演示"""
    print("\n" + "="*60)
    print("🔹 Continuous Batching 演示")
    print("="*60)
    
    processor = ContinuousBatchProcessor(max_batch_size=8)
    
    # 在单独线程中运行处理器
    processing_thread = threading.Thread(target=processor.run_processing_loop, daemon=True)
    processing_thread.start()
    
    # 生成测试请求
    requests = generate_test_requests(10)
    
    # 模拟请求到达（有时间间隔）
    for i, request in enumerate(requests):
        processor.add_request(request)
        
        # 模拟请求间隔
        if i < len(requests) - 1:
            time.sleep(random.uniform(0.1, 0.5))
    
    # 等待所有请求完成
    while len(processor.processing_stats['completed_requests']) < len(requests):
        time.sleep(0.1)
    
    processor.stop()
    return processor.processing_stats

def analyze_performance(static_stats: Dict, continuous_stats: Dict):
    """分析性能对比"""
    print("\n" + "="*60)
    print("📊 性能分析对比")
    print("="*60)
    
    # 计算平均延迟
    static_latencies = [req['latency'] for req in static_stats['completed_requests']]
    continuous_latencies = [req['latency'] for req in continuous_stats['completed_requests']]
    
    static_avg_latency = sum(static_latencies) / len(static_latencies)
    continuous_avg_latency = sum(continuous_latencies) / len(continuous_latencies)
    
    # 计算吞吐量
    static_total_time = static_stats['total_processing_time'] + static_stats['total_wait_time']
    continuous_total_time = max(continuous_latencies) if continuous_latencies else 0
    
    static_throughput = len(static_stats['completed_requests']) / static_total_time if static_total_time > 0 else 0
    continuous_throughput = len(continuous_stats['completed_requests']) / continuous_total_time if continuous_total_time > 0 else 0
    
    # 打印对比结果
    print(f"{'指标':<20} {'静态批处理':<15} {'Continuous Batching':<20} {'改进':<10}")
    print("-" * 70)
    
    latency_improvement = (static_avg_latency - continuous_avg_latency) / static_avg_latency * 100
    throughput_improvement = (continuous_throughput - static_throughput) / static_throughput * 100 if static_throughput > 0 else 0
    
    print(f"{'平均延迟 (s)':<20} {static_avg_latency:.3f:<15} {continuous_avg_latency:.3f:<20} {latency_improvement:+.1f}%")
    print(f"{'吞吐量 (req/s)':<20} {static_throughput:.2f:<15} {continuous_throughput:.2f:<20} {throughput_improvement:+.1f}%")
    print(f"{'总批次数':<20} {static_stats['total_batches']:<15} {'N/A':<20} {'N/A'}")
    print(f"{'总处理步骤':<20} {'N/A':<15} {continuous_stats['total_steps']:<20} {'N/A'}")
    
    # 延迟分布分析
    print(f"\n📈 延迟分布分析:")
    print(f"静态批处理:")
    print(f"  最小延迟: {min(static_latencies):.3f}s")
    print(f"  最大延迟: {max(static_latencies):.3f}s")
    print(f"  延迟标准差: {(sum((x - static_avg_latency)**2 for x in static_latencies) / len(static_latencies))**0.5:.3f}s")
    
    print(f"Continuous Batching:")
    print(f"  最小延迟: {min(continuous_latencies):.3f}s")
    print(f"  最大延迟: {max(continuous_latencies):.3f}s")
    print(f"  延迟标准差: {(sum((x - continuous_avg_latency)**2 for x in continuous_latencies) / len(continuous_latencies))**0.5:.3f}s")

def demonstrate_batch_dynamics():
    """演示批处理动态特性"""
    print("\n" + "="*60)
    print("🔄 批处理动态特性演示")
    print("="*60)
    
    print("💡 关键差异分析:")
    print("\n1. 静态批处理特点:")
    print("   ✓ 等待批次填满或超时")
    print("   ✓ 所有请求同时开始和结束")
    print("   ✓ 短请求需要等待长请求完成")
    print("   ✗ 可能存在等待时间浪费")
    print("   ✗ 批次大小固定，灵活性差")
    
    print("\n2. Continuous Batching 特点:")
    print("   ✓ 请求立即开始处理")
    print("   ✓ 完成的请求立即返回")
    print("   ✓ 新请求可以动态加入")
    print("   ✓ 更好的资源利用率")
    print("   ✓ 更低的平均延迟")
    
    print("\n3. 适用场景:")
    print("   静态批处理: 离线批量处理，对延迟不敏感")
    print("   Continuous Batching: 在线服务，实时交互应用")

def save_results_to_file(static_stats: Dict, continuous_stats: Dict):
    """保存结果到文件"""
    results = {
        'timestamp': time.time(),
        'static_batch': static_stats,
        'continuous_batch': continuous_stats
    }
    
    with open('batch_processing_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到 batch_processing_results.json")

def main():
    """主函数"""
    print("🎯 批处理方法对比演示")
    print("基于费曼学习法：通过实验理解 Continuous Batching")
    print("=" * 80)
    
    try:
        # 运行静态批处理演示
        static_stats = run_static_batch_demo()
        
        # 短暂休息
        time.sleep(1)
        
        # 运行 Continuous Batching 演示
        continuous_stats = run_continuous_batch_demo()
        
        # 性能分析
        analyze_performance(static_stats, continuous_stats)
        
        # 动态特性演示
        demonstrate_batch_dynamics()
        
        # 保存结果
        save_results_to_file(static_stats, continuous_stats)
        
        print("\n🎉 批处理演示完成！")
        print("\n💡 关键理解:")
        print("1. Continuous Batching 显著降低平均延迟")
        print("2. 动态批次管理提高资源利用率")
        print("3. 适合实时交互式应用场景")
        print("4. 是现代 LLM 推理服务的核心技术")
        
        print("\n📚 深入学习建议:")
        print("1. 理解调度策略 → examples/advanced/04_scheduler/")
        print("2. 学习完整推理流程 → examples/advanced/05_complete_inference/")
        print("3. 实践端到端系统 → examples/advanced/06_end_to_end_demo/")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("💡 请查看 docs/faq.md 获取帮助")

if __name__ == "__main__":
    main()