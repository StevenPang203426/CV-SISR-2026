#!/usr/bin/env bash
# 从项目根目录运行: bash scripts/infer.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python infer.py \
  --ckpt experiments/espcn_x4/best.pt \
  --input demo/original/10.png \
  --output experiments/espcn_x4/infer \
  --model espcn \
  --scale 4
