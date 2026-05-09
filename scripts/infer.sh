#!/usr/bin/env bash
# 从项目根目录运行: bash scripts/infer.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python infer.py \
  --ckpt experiments/edsr_x2/best.pt \
  --input demo/DemoLRPhoto/cat3.jpg \
  --output experiments/edsr_x2/infer \
  --model edsr \
  --scale 2
