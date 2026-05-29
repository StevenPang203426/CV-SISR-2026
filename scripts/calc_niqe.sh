#!/usr/bin/env bash
# 无参考质量指标 (NIQE/BRISQUE)
# 用法: bash scripts/calc_niqe.sh
#
# 参数:
#   --input     单个图像目录
#   --compare   对比模式：多子目录父目录
#   --metrics   指标列表 (默认 niqe brisque)
#   --device    cpu | cuda (默认 cpu)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.blind_sr.calc_niqe \
    --compare experiments/blind_sr/eval_results/exp3 \
    --metrics niqe brisque
