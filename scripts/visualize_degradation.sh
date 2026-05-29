#!/usr/bin/env bash
# 退化可视化 — 展示不同退化管道效果
# 用法: bash scripts/visualize_degradation.sh
#
# 参数:
#   --image       HR 图像路径         (必填)
#   --sf          下采样倍数 (默认 4)
#   --seeds       BSRGAN 随机样本数 (默认 8)
#   --output-dir  输出目录
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.blind_sr.visualize_degradation \
    --image data/datasets/Set5/baby.png \
    --sf 4
