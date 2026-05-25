"""
验证 ONNX 模型与 PyTorch 原始模型的输出一致性。

用法
----
    # 验证单个模型
    python scripts/verify_onnx.py --model espcn --scale 4 \
        --ckpt experiments/espcn_x4/best.pt \
        --onnx web/models/espcn_x4.onnx

    # 批量验证所有已导出的模型
    python scripts/verify_onnx.py --all
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import build_model
from core.checkpoint import load_checkpoint


def verify(model_name: str, scale: int, ckpt_path: str, onnx_path: str,
           input_h: int = 64, input_w: int = 64):
    """比较 PyTorch 与 ONNX Runtime 的推理结果。"""
    import onnxruntime as ort

    # PyTorch 推理
    model = build_model(model_name, scale=scale, in_channels=3)
    load_checkpoint(model, ckpt_path, device='cpu')
    model.eval()
    dummy = torch.randn(1, 3, input_h, input_w)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # ONNX Runtime 推理
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = session.run(None, {'input': dummy.numpy()})[0]

    # 比较
    max_diff = np.abs(pt_out - ort_out).max()
    mean_diff = np.abs(pt_out - ort_out).mean()
    status = 'PASS' if max_diff < 1e-5 else 'FAIL'

    print(f'[{status}] {model_name}_x{scale}:  '
          f'max_diff={max_diff:.8f}  mean_diff={mean_diff:.8f}  '
          f'output_shape={pt_out.shape}')
    return max_diff < 1e-5


WEB_MODELS = ['espcn', 'fsrcnn']
SCALES = [2, 3, 4]


def verify_all():
    """批量验证所有已导出的 ONNX 模型。"""
    passed, failed, skipped = 0, 0, 0
    for model_name in WEB_MODELS:
        for scale in SCALES:
            ckpt = os.path.join(ROOT, f'experiments/{model_name}_x{scale}/best.pt')
            onnx = os.path.join(ROOT, f'web/models/{model_name}_x{scale}.onnx')
            if os.path.exists(ckpt) and os.path.exists(onnx):
                ok = verify(model_name, scale, ckpt, onnx)
                if ok:
                    passed += 1
                else:
                    failed += 1
            else:
                skipped += 1
    print(f'\nResults: {passed} passed, {failed} failed, {skipped} skipped.')


def main():
    p = argparse.ArgumentParser(description='Verify ONNX model output consistency')
    p.add_argument('--model', type=str)
    p.add_argument('--scale', type=int)
    p.add_argument('--ckpt', type=str)
    p.add_argument('--onnx', type=str)
    p.add_argument('--all', action='store_true')
    args = p.parse_args()

    if args.all:
        verify_all()
    else:
        if not all([args.model, args.scale, args.ckpt, args.onnx]):
            p.error('--model, --scale, --ckpt, --onnx are required when not using --all')
        verify(args.model, args.scale, args.ckpt, args.onnx)


if __name__ == '__main__':
    main()
