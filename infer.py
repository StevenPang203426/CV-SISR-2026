"""
推理入口脚本
=============

对任意输入图像执行超分辨率推理。

用法::

    python infer.py --ckpt output/edsr_x2/best.pt \\
                    --input demo/DemoLRPhoto/cat3.jpg \\
                    --model edsr --scale 2
"""

import argparse
import torch

from models import build_model, REGISTRY
from core.checkpoint import load_checkpoint
from core.inferencer import Inferencer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='模型检查点路径')
    p.add_argument('--input', required=True, help='输入图像路径或目录')
    p.add_argument('--output', default=None, help='输出路径')
    p.add_argument('--model', required=True, choices=REGISTRY.keys())
    p.add_argument('--scale', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型并加载权重
    model = build_model(args.model, scale=args.scale, in_channels=3)
    load_checkpoint(model, args.ckpt, device='cpu')
    model = model.to(device).eval()

    # 推理
    inferencer = Inferencer(model, device, args.scale)
    inferencer.run(args.input, args.output)


if __name__ == '__main__':
    main()
