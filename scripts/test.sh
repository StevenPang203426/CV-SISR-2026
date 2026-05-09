#!/usr/bin/env bash
# 从项目根目录运行: bash scripts/test.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python test.py \
  --ckpt experiments/edsr_x2/best.pt \
  --test_dir demo/original \
  --model edsr \
  --scale 2 \
  --save_images \
  --out_dir experiments/edsr_x2/test/original
