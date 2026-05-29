#!/usr/bin/env bash
# LIIF 任意倍率测试 — 支持单图/多图/多倍率
# 用法: bash scripts/test_liif.sh
#
# 参数:
#   --ckpt         模型权重           (必填)
#   --scale        单一倍率 (默认 4.0)
#   --scales       多倍率列表 (如 2 3 4 6)
#   --input        单张图推理
#   --test_dir     测试目录（计算 PSNR）
#   --output       输出目录 (默认 experiments/liif_edsr/test)
#   --max_images   最多测试图片数
#   --batch_q      分块查询大小 (默认 30000)
#   --save_images  保存 SR 图像
#   --n_feats      编码器通道 (默认 64)
#   --n_resblocks  ResBlock 数量 (默认 16)
#   --hidden_dim   MLP 隐藏层 (默认 256)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.liif.test \
    --ckpt experiments/liif_edsr/best.pt \
    --test_dir data/datasets/Set5 \
    --scales 2 3 4 6 \
    --save_images
