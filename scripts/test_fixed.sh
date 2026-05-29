#!/usr/bin/env bash
# 固定倍率 SR 测试 — 输出 PSNR/SSIM/FLOPs/FPS
# 用法: bash scripts/test_fixed.sh
#
# 参数:
#   --ckpt        模型权重路径          (必填)
#   --test_dir    测试集目录            (必填)
#   --model       srcnn|fsrcnn|espcn|edsr|imdn (必填)
#   --scale       放大倍率 (默认 2)
#   --save_images 保存 SR 图像
#   --out_dir     输出目录
#   --json        自定义 metrics.json 路径
set -euo pipefail
cd "$(dirname "$0")/.."
python test.py \
    --ckpt  experiments/edsr_x2/best.pt \
    --test_dir data/datasets/Set5 \
    --model edsr \
    --scale 2 \
    --save_images
