#!/usr/bin/env bash
# Blind SR 统一评估 — 实验 1/2/3
# 用法: bash scripts/eval_blind.sh [EXPERIMENT]
#
# 实验说明:
#   1  Bicubic SR vs Blind SR 在多退化下的对比
#   2  退化因素消融实验
#   3  PSNR vs GAN 感知质量 (RealSRSet)
#
# 参数:
#   --experiment   实验编号 1|2|3     (必填)
#   --device       cuda | cpu (默认 auto)
#   --num-images   测试图像数量 (默认 20, 实验1)
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${1:-1}"
echo "[eval_blind] experiment=$EXP"
python -m src.blind_sr.eval --experiment "$EXP"
