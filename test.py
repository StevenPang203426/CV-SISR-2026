"""
测试入口脚本
=============

在基准数据集上评估模型，输出 PSNR/SSIM/参数量/FLOPs/FPS 报告。

用法::

    python test.py --ckpt output/edsr_x2/best.pt \\
                   --test_dir data/datasets/Set5 \\
                   --model edsr --scale 2 --save_images
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import SRDataset
from models import build_model, REGISTRY
from core.checkpoint import load_checkpoint
from core.evaluator import Evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='模型检查点路径')
    p.add_argument('--test_dir', required=True, help='测试集目录')
    p.add_argument('--model', required=True, choices=REGISTRY.keys())
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--out_dir', default=None, help='SR/LR 输出和 metrics.json 的目录')
    p.add_argument('--json', default=None, help='自定义 metrics json 路径')
    args = p.parse_args()

    dataset_name = os.path.basename(os.path.abspath(args.test_dir))
    out_dir = Path(args.out_dir) if args.out_dir else None
    json_path = Path(args.json) if args.json else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型并加载权重
    model = build_model(args.model, scale=args.scale, in_channels=3)
    load_checkpoint(model, args.ckpt, device='cpu')
    model = model.to(device).eval()

    # 数据集
    test_set = SRDataset(args.model, args.test_dir, scale=args.scale, is_train=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 评估
    evaluator = Evaluator(model, device, args.scale)
    evaluator.test(
        loader, dataset_name, args.model,
        out_dir=out_dir,
        save_images=args.save_images,
        json_path=json_path,
    )


if __name__ == '__main__':
    main()
