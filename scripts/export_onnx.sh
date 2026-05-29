#!/usr/bin/env bash
# 导出 ONNX 模型
# 用法: bash scripts/export_onnx.sh
#
# 参数:
#   --model   espcn | fsrcnn
#   --scale   2 | 3 | 4
#   --ckpt    checkpoint 路径
#   --output  输出 .onnx 路径
#   --opset   ONNX opset 版本 (默认 18)
#   --all     导出所有可用轻量模型
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.fixed_sr.export_onnx --all
