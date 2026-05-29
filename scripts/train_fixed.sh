#!/usr/bin/env bash
# 固定倍率 SR 训练
# 用法: bash scripts/train_fixed.sh [CONFIG]
#
# 参数说明 (通过 YAML 配置):
#   model:      srcnn | fsrcnn | espcn | edsr | imdn
#   scale:      2 | 3 | 4
#   train_dir:  data/DIV2K_train_HR
#   val_dir:    data/datasets/Set5
#   batch_size: 16
#   patch_size: 48 (SRCNN 33)
#   epochs:     200
#   lr:         1e-3 (SRCNN 1e-4)
#   opt:        Adam | SGD
#   crit:       L1 | MSE
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/edsr_x2.yaml}"
echo "[train_fixed] config=$CONFIG"
python train.py --config "$CONFIG"
