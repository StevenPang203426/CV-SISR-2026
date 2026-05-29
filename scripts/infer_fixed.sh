#!/usr/bin/env bash
# 固定倍率 SR 推理 — 对任意图像超分
# 用法: bash scripts/infer_fixed.sh
#
# 参数:
#   --ckpt          模型权重路径        (必填)
#   --input         输入图像/目录       (必填)
#   --model         srcnn|fsrcnn|espcn|edsr|imdn (必填)
#   --scale         放大倍率 (默认 2)
#   --output        输出路径
#   --tile_size     分块大小 (默认 256)
#   --tile_overlap  块重叠 (默认 16)
set -euo pipefail
cd "$(dirname "$0")/.."
python infer.py \
    --ckpt  experiments/edsr_x2/best.pt \
    --input demo/DemoLRPhoto/cat3.jpg \
    --model edsr \
    --scale 2
