#!/usr/bin/env bash
# 个人知识库启动脚本
# 使用方式: ./run.sh <command> [args]

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# 激活虚拟环境（如果存在）
if [ -d venv ]; then
    source venv/bin/activate
fi

# 国内用户可设置 HuggingFace 镜像
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
fi

python main.py "$@"
