#!/usr/bin/env bash
# 跨实验 metrics 对比报告
# 用法: bash scripts/compare.sh [JSON_FILES...]
#
# 参数:
#   json_files  一个或多个 metrics.json 路径
#   --out_dir   报告输出目录 (默认 report)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.common.compare experiments/*/metrics.json
