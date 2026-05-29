#!/usr/bin/env bash
# 验证 ONNX 模型输出一致性
# 用法: bash scripts/verify_onnx.sh
#
# 参数:
#   --model  espcn | fsrcnn
#   --scale  2 | 3 | 4
#   --ckpt   PyTorch checkpoint
#   --onnx   ONNX 文件路径
#   --all    验证所有
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.fixed_sr.verify_onnx --all
