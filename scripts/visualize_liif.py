"""
LIIF 多倍率可视化对比面板
==========================

生成一张对比图：原图 + Bicubic + LIIF 在多个倍率下的效果。

用法::

    python scripts/visualize_liif.py \
        --ckpt experiments/liif_edsr/best.pt \
        --input demo/original/10.png \
        --scales 1.5 2 3.5 4 6 \
        --output experiments/liif_edsr/test/comparison.png
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

from models.liif_model import build_liif_model
from scripts.test_liif import infer_liif, tensor_to_pil


def make_panel(lr_img, sr_liif, sr_bicubic, scale, target_h):
    """为单个倍率创建一列：Bicubic | LIIF。"""
    # 统一高度
    for img in [sr_bicubic, sr_liif]:
        pass  # 保持原始分辨率

    # 拼接 Bicubic（上）和 LIIF（下）
    w = max(sr_bicubic.width, sr_liif.width)
    h = sr_bicubic.height + sr_liif.height + 30  # 30px label space

    col = Image.new('RGB', (w, h), (255, 255, 255))
    col.paste(sr_bicubic, (0, 0))
    col.paste(sr_liif, (0, sr_bicubic.height + 30))

    # 添加标签
    draw = ImageDraw.Draw(col)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    label_y = sr_bicubic.height + 5
    draw.text((5, label_y), f"x{scale} Bicubic ↑ | LIIF ↓", fill=(0, 0, 0), font=font)

    return col


def create_comparison(model, input_path, scales, device, output_path, batch_q=30000):
    """生成多倍率对比面板。"""
    to_tensor = ToTensor()

    hr_img = Image.open(input_path).convert('RGB')
    hr_w, hr_h = hr_img.size

    panels = []
    panel_info = []

    for scale in scales:
        # 生成 LR
        lr_h, lr_w = max(1, round(hr_h / scale)), max(1, round(hr_w / scale))
        lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)

        # Bicubic 上采样（baseline）
        sr_bicubic = lr_img.resize((hr_w, hr_h), Image.BICUBIC)

        # LIIF 推理
        lr_tensor = to_tensor(lr_img).unsqueeze(0)
        sr_tensor = infer_liif(model, lr_tensor, scale, device, batch_q)
        sr_liif = tensor_to_pil(sr_tensor)

        # 统一尺寸到 HR
        if sr_liif.size != (hr_w, hr_h):
            sr_liif = sr_liif.resize((hr_w, hr_h), Image.BICUBIC)

        panels.append((sr_bicubic, sr_liif))
        panel_info.append(f'x{scale}')

    # 布局: HR原图 | x1.5 | x2 | x3.5 | x4 | x6
    # 每列: 上方 Bicubic，下方 LIIF
    label_h = 35
    gap = 4
    col_w = hr_w
    n_cols = 1 + len(scales)  # HR + each scale
    total_w = n_cols * col_w + (n_cols - 1) * gap
    total_h = label_h + hr_h * 2 + gap  # Bicubic row + LIIF row

    canvas = Image.new('RGB', (total_w, total_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    x_offset = 0

    # 第一列：HR Ground Truth
    draw.text((x_offset + 5, 5), "HR Ground Truth", fill=(0, 0, 0), font=font)
    canvas.paste(hr_img, (x_offset, label_h))
    canvas.paste(hr_img, (x_offset, label_h + hr_h + gap))
    x_offset += col_w + gap

    # 每个倍率一列
    for i, scale in enumerate(scales):
        sr_bic, sr_liif = panels[i]

        # 标签
        draw.text((x_offset + 5, 5), f"x{scale}", fill=(0, 0, 0), font=font)
        draw.text((x_offset + 5, label_h + hr_h - 20), "Bicubic", fill=(200, 50, 50), font=font_small)
        draw.text((x_offset + 5, label_h + hr_h + gap + hr_h - 20), "LIIF", fill=(50, 50, 200), font=font_small)

        canvas.paste(sr_bic, (x_offset, label_h))
        canvas.paste(sr_liif, (x_offset, label_h + hr_h + gap))
        x_offset += col_w + gap

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    canvas.save(output_path, quality=95)
    print(f'Comparison panel saved: {output_path} ({total_w}x{total_h})')


def main():
    p = argparse.ArgumentParser(description='LIIF Multi-Scale Comparison Visualization')
    p.add_argument('--ckpt', type=str, required=True, help='LIIF checkpoint')
    p.add_argument('--input', type=str, required=True, help='输入图片（HR 原图）')
    p.add_argument('--scales', type=float, nargs='+', default=[1.5, 2, 3.5, 4, 6],
                   help='对比倍率列表')
    p.add_argument('--output', type=str, default='experiments/liif_edsr/test/comparison.png',
                   help='输出路径')
    p.add_argument('--batch_q', type=int, default=30000)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--n_feats', type=int, default=64)
    p.add_argument('--n_resblocks', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = build_liif_model(
        encoder_args={'n_feats': args.n_feats, 'n_resblocks': args.n_resblocks},
        liif_args={'hidden_dim': args.hidden_dim},
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    create_comparison(model, args.input, args.scales, device, args.output, args.batch_q)


if __name__ == '__main__':
    main()
