#!/usr/bin/env bash
# 下载预训练模型权重
# 用法: bash scripts/download_pretrained.sh
#
# 参数:
#   --models  指定模型 (默认全部): BSRGAN BSRNet ESRGAN RealESRGAN_x4plus RealESRNet_x4plus
#   --proxy   HTTP 代理 (如 http://127.0.0.1:7890)
#   --force   强制重新下载
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.common.download_pretrained
