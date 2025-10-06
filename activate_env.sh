#!/bin/bash

# nano-vllm 虚拟环境激活脚本
# 使用方法: source activate_env.sh

echo "🚀 激活 nano-vllm Python 虚拟环境..."

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行以下命令创建："
    echo "   python3 -m venv venv"
    return 1
fi

# 激活虚拟环境
source venv/bin/activate

# 验证激活状态
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ 虚拟环境已成功激活！"
    echo "📍 虚拟环境路径: $VIRTUAL_ENV"
    echo "🐍 Python 版本: $(python --version)"
    echo "📦 pip 版本: $(pip --version)"
    echo ""
    echo "💡 提示："
    echo "   - 要退出虚拟环境，请运行: deactivate"
    echo "   - 要运行学习环境，请切换到学习环境目录："
    echo "     cd nano-vllm-learning/examples/00-progressive-learning/level_09_production/learning-environment"
    echo "     python main_app.py"
else
    echo "❌ 虚拟环境激活失败！"
    return 1
fi