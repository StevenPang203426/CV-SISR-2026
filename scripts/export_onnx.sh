#!/usr/bin/env bash
# 导出 ONNX 模型（轻量 + RRDB pretrained）
# 用法: bash scripts/export_onnx.sh [OPTIONS]
#
# 轻量模型 (espcn/fsrcnn):
#   --model   espcn | fsrcnn
#   --scale   2 | 3 | 4
#   --ckpt    checkpoint 路径
#   --output  输出 .onnx 路径 (可选)
#
# RRDB pretrained 模型 (无需 --scale/--ckpt):
#   --model   bsrnet | bsrgan | realesrgan_x4plus | realesrnet_x4plus | esrgan
#   --output  输出 .onnx 路径 (可选，默认 web/models/<model>_x4.onnx)
#
# 通用参数:
#   --opset   ONNX opset 版本 (默认 18)
#   --all     导出所有可用模型（轻量 + RRDB）
#
# 示例:
#   bash scripts/export_onnx.sh --all                      # 导出全部
#   bash scripts/export_onnx.sh --model bsrnet             # 仅导出 BSRNet
#   bash scripts/export_onnx.sh --model fsrcnn --scale 4 \
#       --ckpt experiments/fsrcnn_x4_pixelshuffle/best.pt   # 导出 FSRCNN x4
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.fixed_sr.export_onnx "${@:---all}"
