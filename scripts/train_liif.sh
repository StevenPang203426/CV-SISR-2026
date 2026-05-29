#!/usr/bin/env bash
# LIIF 任意倍率 SR 训练
# 用法: bash scripts/train_liif.sh [CONFIG]
#
# 配置文件参数 (YAML):
#   train_dir:    data/DIV2K_train_HR
#   val_dir:      data/datasets/Set5
#   scale_min:    1.0
#   scale_max:    4.0
#   sample_q:     2304
#   inp_size:     48
#   batch_size:   16
#   epochs:       200
#   lr:           1e-4
#   n_feats:      64
#   n_resblocks:  16
#   hidden_dim:   256
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/liif_edsr_x1-4.yaml}"
echo "[train_liif] config=$CONFIG"
python -m src.liif.train --config "$CONFIG"
