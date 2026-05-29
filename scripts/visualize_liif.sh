#!/usr/bin/env bash
# LIIF 多倍率对比可视化面板
# 用法: bash scripts/visualize_liif.sh
#
# 参数:
#   --ckpt         模型权重           (必填)
#   --input        HR 原图            (必填)
#   --scales       倍率列表 (默认 1.5 2 3.5 4 6)
#   --output       输出路径 (默认 experiments/liif_edsr/test/comparison.png)
#   --batch_q      分块查询大小 (默认 30000)
#   --n_feats      编码器通道 (默认 64)
#   --n_resblocks  ResBlock 数量 (默认 16)
#   --hidden_dim   MLP 隐藏层 (默认 256)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.liif.visualize \
    --ckpt experiments/liif_edsr/best.pt \
    --input data/datasets/Set5/baby.png \
    --scales 1.5 2 3.5 4 6
