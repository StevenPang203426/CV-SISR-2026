"""
将 PyTorch 超分辨率模型导出为 ONNX 格式，供 Web 前端 (ONNX Runtime Web) 使用。

用法
----
    # 导出单个模型
    python scripts/export_onnx.py --model espcn --scale 4 \
        --ckpt experiments/espcn_x4/best.pt \
        --output web/models/espcn_x4.onnx

    python scripts/export_onnx.py --model fsrcnn --scale 4 \
        --ckpt experiments/fsrcnn_x4/best.pt \
        --output web/models/fsrcnn_x4.onnx
    # 批量导出所有可用的轻量模型
    python scripts/export_onnx.py --all
"""

import argparse
import os
import sys

import torch

# 确保项目根目录在 sys.path 中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_model
from core.checkpoint import load_checkpoint


def export_one(model_name: str, scale: int, ckpt_path: str, output_path: str,
               opset: int = 11, input_h: int = 64, input_w: int = 64):
    """导出单个模型为 ONNX。"""
    # 1. 构建模型并加载权重
    model = build_model(model_name, scale=scale, in_channels=3)
    load_checkpoint(model, ckpt_path, device='cpu')
    model.eval()

    # 2. 创建 dummy 输入
    dummy = torch.randn(1, 3, input_h, input_w)

    # 3. 导出
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
        },
    )

    # 4. 打印模型大小
    size_kb = os.path.getsize(output_path) / 1024
    print(f'[OK] Exported: {output_path}  ({size_kb:.1f} KB)')
    return output_path


# 轻量模型列表（适合浏览器）
WEB_MODELS = ['espcn', 'fsrcnn']
SCALES = [2, 3, 4]


def export_all(opset: int):
    """批量导出所有可用的轻量模型。"""
    exported, skipped = [], []
    for model_name in WEB_MODELS:
        for scale in SCALES:
            ckpt = os.path.join(ROOT, f'experiments/{model_name}_x{scale}/best.pt')
            if os.path.exists(ckpt):
                out = os.path.join(ROOT, f'web/models/{model_name}_x{scale}.onnx')
                export_one(model_name, scale, ckpt, out, opset)
                exported.append(out)
            else:
                print(f'[SKIP] {ckpt} not found')
                skipped.append(ckpt)
    print(f'\nDone. Exported {len(exported)}, skipped {len(skipped)}.')


def main():
    p = argparse.ArgumentParser(description='Export PyTorch SR models to ONNX')
    p.add_argument('--model', type=str, help='Model name (espcn / fsrcnn)')
    p.add_argument('--scale', type=int, help='Super-resolution scale factor')
    p.add_argument('--ckpt', type=str, help='Checkpoint path (.pt)')
    p.add_argument('--output', type=str, help='Output .onnx path')
    p.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11)')
    p.add_argument('--all', action='store_true', help='Export all available lightweight models')
    args = p.parse_args()

    if args.all:
        export_all(args.opset)
    else:
        if not all([args.model, args.scale, args.ckpt, args.output]):
            p.error('--model, --scale, --ckpt, --output are required when not using --all')
        export_one(args.model, args.scale, args.ckpt, args.output, args.opset)


if __name__ == '__main__':
    main()
