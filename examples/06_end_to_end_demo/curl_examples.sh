#!/bin/bash

# NanoVLLM API Curl 示例脚本
# 展示如何使用 curl 命令调用 NanoVLLM API 的各种功能

# 默认配置
API_URL="http://localhost:8000"
CONTENT_TYPE="Content-Type: application/json"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印分隔线
print_separator() {
    echo -e "${BLUE}=================================================${NC}"
}

# 打印标题
print_title() {
    echo -e "${CYAN}$1${NC}"
}

# 打印成功信息
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 打印错误信息
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 打印警告信息
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 检查服务器状态
check_server() {
    print_title "🔍 检查服务器状态"
    
    response=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "$API_URL/v1/health")
    http_code="${response: -3}"
    
    if [ "$http_code" -eq 200 ]; then
        print_success "服务器连接正常"
        echo "健康状态响应:"
        cat /tmp/health_response.json | jq '.' 2>/dev/null || cat /tmp/health_response.json
        echo
        return 0
    else
        print_error "服务器连接失败 (HTTP $http_code)"
        return 1
    fi
}

# 1. 基础文本生成
demo_basic_generation() {
    print_separator
    print_title "🎯 基础文本生成演示"
    print_separator
    
    echo "📝 发送单个生成请求..."
    
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Hello, how are you today?",
            "max_tokens": 50,
            "temperature": 0.8,
            "top_p": 0.9,
            "request_id": "demo_basic_001"
        }' \
        -s | jq '.' 2>/dev/null || echo "JSON解析失败，原始响应:"
    
    echo
    print_success "基础生成演示完成"
}

# 2. 批量文本生成
demo_batch_generation() {
    print_separator
    print_title "📦 批量文本生成演示"
    print_separator
    
    echo "📝 发送批量生成请求..."
    
    curl -X POST "$API_URL/v1/generate/batch" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompts": [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "Explain neural networks.",
                "What is deep learning?"
            ],
            "max_tokens": 40,
            "temperature": 0.7
        }' \
        -s | jq '.' 2>/dev/null || echo "JSON解析失败，原始响应:"
    
    echo
    print_success "批量生成演示完成"
}

# 3. 流式文本生成
demo_stream_generation() {
    print_separator
    print_title "🌊 流式文本生成演示"
    print_separator
    
    echo "📝 发送流式生成请求..."
    echo "🔄 流式响应:"
    echo "----------------------------------------"
    
    curl -X POST "$API_URL/v1/generate/stream" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Write a creative story about a robot learning to paint",
            "max_tokens": 80,
            "temperature": 0.9,
            "stream": true,
            "request_id": "demo_stream_001"
        }' \
        -N -s | while IFS= read -r line; do
            if [[ $line == data:* ]]; then
                # 提取JSON数据
                json_data="${line#data: }"
                if [ "$json_data" != "" ]; then
                    # 尝试解析JSON并提取chunk
                    chunk=$(echo "$json_data" | jq -r '.chunk // empty' 2>/dev/null)
                    is_final=$(echo "$json_data" | jq -r '.is_final // false' 2>/dev/null)
                    
                    if [ "$chunk" != "" ] && [ "$chunk" != "null" ]; then
                        printf "%s" "$chunk"
                    fi
                    
                    if [ "$is_final" = "true" ]; then
                        echo
                        echo "----------------------------------------"
                        print_success "流式生成完成"
                        break
                    fi
                fi
            fi
        done
    
    echo
}

# 4. 系统指标查询
demo_system_metrics() {
    print_separator
    print_title "📊 系统指标查询演示"
    print_separator
    
    echo "📈 获取系统性能指标..."
    
    curl -X GET "$API_URL/v1/metrics" \
        -H "$CONTENT_TYPE" \
        -s | jq '.' 2>/dev/null || echo "JSON解析失败，原始响应:"
    
    echo
    print_success "系统指标查询完成"
}

# 5. 健康状态检查
demo_health_check() {
    print_separator
    print_title "🏥 健康状态检查演示"
    print_separator
    
    echo "🔍 检查系统健康状态..."
    
    response=$(curl -X GET "$API_URL/v1/health" -H "$CONTENT_TYPE" -s)
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
    
    # 检查健康状态
    healthy=$(echo "$response" | jq -r '.healthy // false' 2>/dev/null)
    if [ "$healthy" = "true" ]; then
        print_success "系统状态健康"
    else
        print_warning "系统状态异常"
    fi
    
    echo
}

# 6. 模型列表查询
demo_model_listing() {
    print_separator
    print_title "📋 模型列表查询演示"
    print_separator
    
    echo "🤖 获取可用模型列表..."
    
    curl -X GET "$API_URL/v1/models" \
        -H "$CONTENT_TYPE" \
        -s | jq '.' 2>/dev/null || echo "JSON解析失败，原始响应:"
    
    echo
    print_success "模型列表查询完成"
}

# 7. 参数对比测试
demo_parameter_comparison() {
    print_separator
    print_title "⚙️ 生成参数对比演示"
    print_separator
    
    prompt="Write a creative story about a time traveler who"
    
    echo "📝 测试提示: $prompt"
    echo
    
    # 保守生成
    echo "🎛️  保守生成 (temperature=0.3, top_p=0.8):"
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": 60,
            \"temperature\": 0.3,
            \"top_p\": 0.8,
            \"request_id\": \"demo_conservative\"
        }" \
        -s | jq -r '.generated_text // "生成失败"' 2>/dev/null
    echo
    
    # 平衡生成
    echo "🎛️  平衡生成 (temperature=0.7, top_p=0.9):"
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": 60,
            \"temperature\": 0.7,
            \"top_p\": 0.9,
            \"request_id\": \"demo_balanced\"
        }" \
        -s | jq -r '.generated_text // "生成失败"' 2>/dev/null
    echo
    
    # 创意生成
    echo "🎛️  创意生成 (temperature=1.0, top_p=0.95):"
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": 60,
            \"temperature\": 1.0,
            \"top_p\": 0.95,
            \"request_id\": \"demo_creative\"
        }" \
        -s | jq -r '.generated_text // "生成失败"' 2>/dev/null
    echo
    
    print_success "参数对比演示完成"
}

# 8. 错误处理测试
demo_error_handling() {
    print_separator
    print_title "🚨 错误处理测试演示"
    print_separator
    
    echo "🧪 测试无效参数..."
    
    # 测试无效的temperature值
    response=$(curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Test prompt",
            "max_tokens": 50,
            "temperature": 5.0,
            "request_id": "demo_error_001"
        }' \
        -s -w "%{http_code}")
    
    http_code="${response: -3}"
    response_body="${response%???}"
    
    echo "HTTP状态码: $http_code"
    echo "响应内容:"
    echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    echo
    
    # 测试空提示
    echo "🧪 测试空提示..."
    
    response=$(curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "",
            "max_tokens": 50,
            "request_id": "demo_error_002"
        }' \
        -s -w "%{http_code}")
    
    http_code="${response: -3}"
    response_body="${response%???}"
    
    echo "HTTP状态码: $http_code"
    echo "响应内容:"
    echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
    echo
    
    print_success "错误处理测试完成"
}

# 9. 性能基准测试
demo_performance_benchmark() {
    print_separator
    print_title "🏃‍♂️ 性能基准测试演示"
    print_separator
    
    echo "⏱️  测试不同长度提示的响应时间..."
    
    # 短提示测试
    echo "📏 短提示测试:"
    start_time=$(date +%s.%N)
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Hello",
            "max_tokens": 20,
            "temperature": 0.7,
            "request_id": "benchmark_short"
        }' \
        -s | jq -r '.tokens_per_second // "N/A"' | xargs -I {} echo "   速度: {} tokens/s"
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    echo "   客户端时间: ${duration}s"
    echo
    
    # 中等提示测试
    echo "📏 中等提示测试:"
    start_time=$(date +%s.%N)
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Explain the concept of artificial intelligence and its applications in modern technology",
            "max_tokens": 50,
            "temperature": 0.7,
            "request_id": "benchmark_medium"
        }' \
        -s | jq -r '.tokens_per_second // "N/A"' | xargs -I {} echo "   速度: {} tokens/s"
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    echo "   客户端时间: ${duration}s"
    echo
    
    # 长提示测试
    echo "📏 长提示测试:"
    start_time=$(date +%s.%N)
    curl -X POST "$API_URL/v1/generate" \
        -H "$CONTENT_TYPE" \
        -d '{
            "prompt": "Write a detailed analysis of the impact of machine learning on various industries including healthcare, finance, transportation, and education. Discuss both the benefits and challenges.",
            "max_tokens": 100,
            "temperature": 0.7,
            "request_id": "benchmark_long"
        }' \
        -s | jq -r '.tokens_per_second // "N/A"' | xargs -I {} echo "   速度: {} tokens/s"
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
    echo "   客户端时间: ${duration}s"
    echo
    
    print_success "性能基准测试完成"
}

# 10. 并发请求测试
demo_concurrent_requests() {
    print_separator
    print_title "🔀 并发请求测试演示"
    print_separator
    
    echo "🚀 启动并发请求测试..."
    
    # 创建临时目录存储结果
    temp_dir="/tmp/nano_vllm_concurrent_$$"
    mkdir -p "$temp_dir"
    
    # 并发请求函数
    make_concurrent_request() {
        local id=$1
        local prompt=$2
        local start_time=$(date +%s.%N)
        
        local response=$(curl -X POST "$API_URL/v1/generate" \
            -H "$CONTENT_TYPE" \
            -d "{
                \"prompt\": \"$prompt\",
                \"max_tokens\": 30,
                \"temperature\": 0.8,
                \"request_id\": \"concurrent_$id\"
            }" \
            -s)
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
        
        echo "$id|$duration|$response" > "$temp_dir/result_$id.txt"
        echo "   ✅ 请求 $id 完成: ${duration}s"
    }
    
    # 启动并发请求
    prompts=(
        "What is the capital of France?"
        "How do computers work?"
        "Explain photosynthesis."
        "What is quantum mechanics?"
        "Describe the solar system."
        "How does the internet work?"
    )
    
    concurrent_start=$(date +%s.%N)
    
    for i in "${!prompts[@]}"; do
        make_concurrent_request $((i+1)) "${prompts[$i]}" &
    done
    
    # 等待所有请求完成
    wait
    
    concurrent_end=$(date +%s.%N)
    total_time=$(echo "$concurrent_end - $concurrent_start" | bc -l 2>/dev/null || echo "0")
    
    # 统计结果
    echo
    echo "📊 并发请求结果:"
    echo "   📝 总请求数: ${#prompts[@]}"
    echo "   ⏱️  总时间: ${total_time}s"
    
    # 计算成功率和平均延迟
    successful=0
    total_latency=0
    total_tokens=0
    
    for result_file in "$temp_dir"/result_*.txt; do
        if [ -f "$result_file" ]; then
            IFS='|' read -r id duration response < "$result_file"
            
            # 检查响应是否成功
            if echo "$response" | jq -e '.generated_text' >/dev/null 2>&1; then
                successful=$((successful + 1))
                total_latency=$(echo "$total_latency + $duration" | bc -l 2>/dev/null || echo "$total_latency")
                
                tokens=$(echo "$response" | jq -r '.total_tokens // 0' 2>/dev/null)
                total_tokens=$((total_tokens + tokens))
            fi
        fi
    done
    
    if [ $successful -gt 0 ]; then
        avg_latency=$(echo "scale=2; $total_latency / $successful" | bc -l 2>/dev/null || echo "0")
        throughput=$(echo "scale=1; $total_tokens / $total_time" | bc -l 2>/dev/null || echo "0")
        
        echo "   ✅ 成功请求: $successful"
        echo "   📈 平均延迟: ${avg_latency}s"
        echo "   📄 总tokens: $total_tokens"
        echo "   🚀 整体吞吐量: ${throughput} tokens/s"
    fi
    
    # 清理临时文件
    rm -rf "$temp_dir"
    
    print_success "并发请求测试完成"
}

# 交互式模式
interactive_mode() {
    print_separator
    print_title "💬 交互式模式"
    print_separator
    
    echo "输入 'quit' 或 'exit' 退出"
    echo "输入 'metrics' 查看系统指标"
    echo "输入 'health' 查看健康状态"
    echo "----------------------------------------"
    
    while true; do
        echo -n "👤 您: "
        read -r user_input
        
        case "$user_input" in
            "quit"|"exit"|"q")
                echo "👋 再见！"
                break
                ;;
            "metrics")
                curl -X GET "$API_URL/v1/metrics" -H "$CONTENT_TYPE" -s | jq '.' 2>/dev/null
                ;;
            "health")
                curl -X GET "$API_URL/v1/health" -H "$CONTENT_TYPE" -s | jq '.' 2>/dev/null
                ;;
            "")
                continue
                ;;
            *)
                echo -n "🤖 AI: "
                curl -X POST "$API_URL/v1/generate" \
                    -H "$CONTENT_TYPE" \
                    -d "{
                        \"prompt\": \"$user_input\",
                        \"max_tokens\": 100,
                        \"temperature\": 0.8,
                        \"request_id\": \"interactive_$(date +%s)\"
                    }" \
                    -s | jq -r '.generated_text // "生成失败"' 2>/dev/null
                echo
                ;;
        esac
    done
}

# 显示帮助信息
show_help() {
    echo "NanoVLLM API Curl 示例脚本"
    echo
    echo "用法: $0 [选项] [演示类型]"
    echo
    echo "选项:"
    echo "  -u, --url URL        API服务器地址 (默认: http://localhost:8000)"
    echo "  -h, --help          显示帮助信息"
    echo
    echo "演示类型:"
    echo "  all                 运行所有演示"
    echo "  basic               基础文本生成"
    echo "  batch               批量文本生成"
    echo "  stream              流式文本生成"
    echo "  metrics             系统指标查询"
    echo "  health              健康状态检查"
    echo "  models              模型列表查询"
    echo "  params              参数对比测试"
    echo "  errors              错误处理测试"
    echo "  benchmark           性能基准测试"
    echo "  concurrent          并发请求测试"
    echo "  interactive         交互式模式"
    echo
    echo "示例:"
    echo "  $0                  运行所有演示"
    echo "  $0 basic            只运行基础生成演示"
    echo "  $0 -u http://localhost:8080 health"
    echo "  $0 interactive      进入交互式模式"
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--url)
                API_URL="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                DEMO_TYPE="$1"
                shift
                ;;
        esac
    done
    
    # 默认演示类型
    DEMO_TYPE="${DEMO_TYPE:-all}"
    
    echo -e "${PURPLE}🚀 NanoVLLM API Curl 示例脚本${NC}"
    print_separator
    echo -e "${CYAN}📍 服务器地址: $API_URL${NC}"
    echo -e "${CYAN}🎯 演示类型: $DEMO_TYPE${NC}"
    print_separator
    
    # 检查依赖
    if ! command -v curl &> /dev/null; then
        print_error "curl 命令未找到，请先安装 curl"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_warning "jq 命令未找到，JSON输出可能不够美观"
    fi
    
    # 检查服务器连接
    if ! check_server; then
        print_error "无法连接到服务器，请确保服务器正在运行"
        exit 1
    fi
    
    # 运行演示
    case "$DEMO_TYPE" in
        "all")
            demo_basic_generation
            demo_batch_generation
            demo_stream_generation
            demo_system_metrics
            demo_health_check
            demo_model_listing
            demo_parameter_comparison
            demo_error_handling
            demo_performance_benchmark
            demo_concurrent_requests
            
            echo
            echo -n "🤔 是否进入交互式模式？(y/N): "
            read -r choice
            if [[ "$choice" =~ ^[Yy]$ ]]; then
                interactive_mode
            fi
            ;;
        "basic")
            demo_basic_generation
            ;;
        "batch")
            demo_batch_generation
            ;;
        "stream")
            demo_stream_generation
            ;;
        "metrics")
            demo_system_metrics
            ;;
        "health")
            demo_health_check
            ;;
        "models")
            demo_model_listing
            ;;
        "params")
            demo_parameter_comparison
            ;;
        "errors")
            demo_error_handling
            ;;
        "benchmark")
            demo_performance_benchmark
            ;;
        "concurrent")
            demo_concurrent_requests
            ;;
        "interactive")
            interactive_mode
            ;;
        *)
            print_error "未知演示类型: $DEMO_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    print_separator
    print_success "演示完成！感谢使用 NanoVLLM API！"
}

# 运行主函数
main "$@"