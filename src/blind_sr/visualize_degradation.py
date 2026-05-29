"""
退化管道可视化脚本 — 对应 BLIND_SR_PLAN 实验 4。

对同一张 HR 图，生成多种退化效果的对比面板图。

用法
----
    python scripts/visualize_degradation.py --image data/DIV2K_valid_HR/0801.png
    python scripts/visualize_degradation.py --image data/DIV2K_valid_HR/0801.png --seeds 8
"""

import argparse
import os
import sys
import random as py_random

import cv2
import numpy as np


from src.blind_sr.degradation import (
    degradation_bsrgan, degradation_bicubic, degradation_custom,
    modcrop, uint2single, single2uint
)


def visualize_degradation_comparison(hr_img, sf=4, output_path='degradation_comparison.png'):
    """
    生成不同退化方式的对比面板。

    展示:
      行1: Bicubic / 仅模糊 / 仅噪声 / 仅JPEG
      行2: 模糊+噪声 / 完整退化(弱) / 完整退化(强) / BSRGAN随机
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hr = modcrop(hr_img, sf)
    h, w = hr.shape[:2]

    # 为了展示方便，裁剪中心区域
    crop_h, crop_w = min(h, 512), min(w, 512)
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    hr_crop = hr[top:top + crop_h, left:left + crop_w]

    conditions = [
        ('Bicubic',        dict(blur_sigma=0,   noise_level=0,  jpeg_quality=0)),
        ('Blur (σ=2.0)',   dict(blur_sigma=2.0, noise_level=0,  jpeg_quality=0)),
        ('Noise (σ=20)',   dict(blur_sigma=0,   noise_level=20, jpeg_quality=0)),
        ('JPEG (q=30)',    dict(blur_sigma=0,   noise_level=0,  jpeg_quality=30)),
        ('Blur+Noise',     dict(blur_sigma=1.5, noise_level=15, jpeg_quality=0)),
        ('Mild',           dict(blur_sigma=0.8, noise_level=5,  jpeg_quality=70)),
        ('Heavy',          dict(blur_sigma=2.5, noise_level=25, jpeg_quality=25)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # HR 参考（左上角）
    axes[0, 0].imshow(hr_crop)
    axes[0, 0].set_title('HR (Ground Truth)', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')

    # 各种退化
    positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    for (r, c), (name, params) in zip(positions, conditions):
        lr = degradation_custom(hr_crop, sf=sf, **params)
        # 放大显示（nearest neighbor，保留像素感）
        lr_up = cv2.resize(lr, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        axes[r, c].imshow(lr_up.clip(0, 1))
        axes[r, c].set_title(f'LR: {name}', fontsize=9)
        axes[r, c].axis('off')

    plt.suptitle(f'Degradation Comparison (×{sf})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 对比面板已保存: {output_path}')


def visualize_bsrgan_diversity(hr_img, sf=4, n_samples=8, output_path='bsrgan_diversity.png'):
    """
    展示 BSRGAN 退化管道的多样性：同一 HR 图，不同随机种子生成不同退化结果。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hr = modcrop(hr_img, sf)
    h, w = hr.shape[:2]
    patch_size = 72

    # 确保图片够大
    if h < patch_size * sf or w < patch_size * sf:
        print(f'[WARN] 图片太小 ({h}x{w}), 减小 patch_size')
        patch_size = min(h, w) // sf

    cols = min(n_samples, 4)
    rows = (n_samples + cols - 1) // cols + 1  # +1 for HR row

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # 第一行放 HR
    for c in range(cols):
        if c == 0:
            # 显示 HR 中心裁剪
            ch, cw = patch_size * sf, patch_size * sf
            top, left = (h - ch) // 2, (w - cw) // 2
            hr_patch = hr[top:top + ch, left:left + cw]
            axes[0, c].imshow(hr_patch)
            axes[0, c].set_title('HR patch', fontsize=10, fontweight='bold')
        else:
            axes[0, c].axis('off')
        axes[0, c].axis('off')

    # 后续行放不同退化结果
    for i in range(n_samples):
        r, c = 1 + i // cols, i % cols
        seed = i * 42 + 7
        py_random.seed(seed)
        np.random.seed(seed)

        try:
            lr, hq = degradation_bsrgan(hr, sf=sf, lq_patchsize=patch_size)
            # 放大 LR
            lr_up = cv2.resize(lr, (patch_size * sf, patch_size * sf),
                               interpolation=cv2.INTER_NEAREST)
            axes[r, c].imshow(lr_up.clip(0, 1))
            axes[r, c].set_title(f'BSRGAN seed={seed}', fontsize=8)
        except Exception as e:
            axes[r, c].text(0.5, 0.5, str(e), ha='center', va='center',
                           fontsize=7, transform=axes[r, c].transAxes)
        axes[r, c].axis('off')

    # 隐藏多余子图
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c > 0:
                axes[r, c].axis('off')

    plt.suptitle(f'BSRGAN Degradation Diversity (×{sf})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 多样性面板已保存: {output_path}')


def main():
    p = argparse.ArgumentParser(description='退化管道可视化')
    p.add_argument('--image', type=str, required=True, help='HR 图像路径')
    p.add_argument('--sf', type=int, default=4, help='下采样倍数')
    p.add_argument('--seeds', type=int, default=8, help='BSRGAN 随机退化样本数')
    p.add_argument('--output-dir', type=str,
                   default=os.path.join(ROOT, 'experiments/blind_sr/visualizations'))
    args = p.parse_args()

    img = np.array(Image.open(args.image).convert('RGB'))
    hr = uint2single(img)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 退化因素对比
    visualize_degradation_comparison(
        hr, sf=args.sf,
        output_path=os.path.join(args.output_dir, f'{basename}_comparison.png')
    )

    # 2. BSRGAN 多样性
    visualize_bsrgan_diversity(
        hr, sf=args.sf, n_samples=args.seeds,
        output_path=os.path.join(args.output_dir, f'{basename}_bsrgan_diversity.png')
    )


if __name__ == '__main__':
    from PIL import Image
    main()
