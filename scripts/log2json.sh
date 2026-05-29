#!/usr/bin/env bash
# 训练日志转 JSON
# 用法: bash scripts/log2json.sh [LOG_FILES...]
#
# 参数:
#   logs       .log 文件路径 (默认 experiments/*/*.log)
#   --out_dir  JSON 输出目录 (默认与 .log 同目录)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.common.log2json
