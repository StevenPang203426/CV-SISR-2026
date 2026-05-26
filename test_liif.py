"""
LIIF 测试入口 — 任意倍率超分辨率
==================================

支持整数和分数倍率测试。

用法::

    # 单张图任意倍率推理
    python test_liif.py --ckpt experiments/liif_edsr/best.pt --scale 3.5 \\
                        --input demo/original/10.png --output results/

    # 在目录上评估 PSNR（需要 HR ground truth）
    python test_liif.py --ckpt experiments/liif_edsr/best.pt --scale 4 \\
                        --test_dir data/DIV2K_valid_HR --output results/

    # 多倍率批量评估
    python test_liif.py --ckpt experiments/liif_edsr/best.pt \\
                        --scales 2 4 6 1.5 3.5 \\
                        --test_dir data/DIV2K_valid_HR --output results/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.liif_model import build_liif_model
from data.dataset_liif import make_coord, make_cell
from core.checkpoint import load_checkpoint


# ============================================================
# 推理核心
# ============================================================

def infer_liif(model, lr_tensor, scale, device, batch_q=30000):
    """
    用 LIIF 模型以任意倍率放大一张图。

    Parameters
    ----------
    model : LIIFModel
    lr_tensor : Tensor (1, 3, H, W)
    scale : float
    device : torch.device
    batch_q : int
        分块查询大小，防 OOM。

    Returns
    -------
    sr : Tensor (1, 3, out_H, out_W)
    """
    model.eval()
    lr_tensor = lr_tensor.to(device)
    _, _, h, w = lr_tensor.shape
    out_h = round(h * scale)
    out_w = round(w * scale)

    # 生成 HR 坐标
    coord = make_coord((out_h, out_w)).unsqueeze(0).to(device)  # (1, N, 2)
    cell = make_cell((out_h, out_w)).unsqueeze(0).to(device)    # (1, N, 2)

    with torch.no_grad():
        feat = model.encoder(lr_tensor)

        n = coord.shape[1]
        pred_parts = []
        for i in range(0, n, batch_q):
            coord_chunk = coord[:, i:i + batch_q, :]
            cell_chunk = cell[:, i:i + batch_q, :]
            pred = model.liif.query(feat, coord_chunk, cell_chunk)
            pred_parts.append(pred.cpu())

        pred_rgb = torch.cat(pred_parts, dim=1)  # (1, N, 3)

    sr = pred_rgb.view(1, out_h, out_w, 3).permute(0, 3, 1, 2)  # (1, 3, H, W)
    return sr.clamp(0, 1)


def tensor_to_pil(tensor):
    """(1, 3, H, W) float tensor → PIL Image"""
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def calc_psnr(sr, hr):
    """计算 PSNR (Y channel)。"""
    sr_np = sr.squeeze(0).permute(1, 2, 0).numpy()
    hr_np = hr.squeeze(0).permute(1, 2, 0).numpy()

    # 转 Y channel
    sr_y = 16 + 65.481 * sr_np[:, :, 0] + 128.553 * sr_np[:, :, 1] + 24.966 * sr_np[:, :, 2]
    hr_y = 16 + 65.481 * hr_np[:, :, 0] + 128.553 * hr_np[:, :, 1] + 24.966 * hr_np[:, :, 2]

    # Shave border (scale pixels)
    shave = max(round(max(sr_np.shape[0], sr_np.shape[1]) / sr_np.shape[0]), 2)
    sr_y = sr_y[shave:-shave, shave:-shave]
    hr_y = hr_y[shave:-shave, shave:-shave]

    mse = np.mean((sr_y - hr_y) ** 2)
    if mse == 0:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)


# ============================================================
# 单张图推理
# ============================================================

def run_single(model, args, device):
    """对单张图做任意倍率推理并保存。"""
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()

    img = Image.open(args.input).convert('RGB')
    lr_tensor = to_tensor(img).unsqueeze(0)

    print(f'Input: {args.input} ({img.size[0]}x{img.size[1]})')
    print(f'Scale: x{args.scale}')

    t0 = time.time()
    sr = infer_liif(model, lr_tensor, args.scale, device, args.batch_q)
    dt = time.time() - t0

    sr_img = tensor_to_pil(sr)
    out_h, out_w = sr_img.size[1], sr_img.size[0]
    print(f'Output: {out_w}x{out_h}, time={dt:.2f}s')

    os.makedirs(args.output, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f'{basename}_x{args.scale}.png')
    sr_img.save(out_path)
    print(f'Saved: {out_path}')


# ============================================================
# 目录评估（计算 PSNR）
# ============================================================

def run_eval(model, args, device):
    """在测试目录上评估 PSNR，支持多倍率。"""
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()

    scales = args.scales if args.scales else [args.scale]
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    files = sorted([
        f for f in os.listdir(args.test_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not files:
        print(f'[ERROR] No images in {args.test_dir}')
        return

    max_images = args.max_images if args.max_images else len(files)
    files = files[:max_images]
    print(f'Testing {len(files)} images at scales: {scales}')

    os.makedirs(args.output, exist_ok=True)
    results = {}

    for scale in scales:
        psnrs = []
        scale_dir = os.path.join(args.output, f'x{scale}')
        os.makedirs(scale_dir, exist_ok=True)

        for fname in tqdm(files, desc=f'x{scale}'):
            fpath = os.path.join(args.test_dir, fname)
            hr_img = Image.open(fpath).convert('RGB')
            hr_tensor = to_tensor(hr_img).unsqueeze(0)

            # 生成 LR
            _, _, h, w = hr_tensor.shape
            lr_h, lr_w = round(h / scale), round(w / scale)
            lr_h, lr_w = max(1, lr_h), max(1, lr_w)
            lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
            lr_tensor = to_tensor(lr_img).unsqueeze(0)

            # 推理
            sr = infer_liif(model, lr_tensor, scale, device, args.batch_q)

            # 确保尺寸匹配（处理 round 误差）
            sr_h, sr_w = sr.shape[2], sr.shape[3]
            if sr_h != h or sr_w != w:
                sr = F.interpolate(sr, size=(h, w), mode='bicubic',
                                   align_corners=False).clamp(0, 1)

            psnr = calc_psnr(sr, hr_tensor)
            psnrs.append(psnr)

            # 保存 SR 图像
            if args.save_images:
                sr_img = tensor_to_pil(sr)
                basename = os.path.splitext(fname)[0]
                sr_img.save(os.path.join(scale_dir, f'{basename}_x{scale}.png'))

        avg_psnr = np.mean(psnrs)
        results[f'x{scale}'] = {
            'avg_psnr': round(float(avg_psnr), 2),
            'num_images': len(psnrs),
        }
        print(f'  x{scale}: PSNR = {avg_psnr:.2f} dB ({len(psnrs)} images)')

    # 汇总
    print(f'\n{"=" * 40}')
    print(f'{"Scale":<10} {"PSNR (dB)":<12} {"Images":<8}')
    print('-' * 30)
    for k, v in results.items():
        print(f'{k:<10} {v["avg_psnr"]:<12.2f} {v["num_images"]:<8}')

    # 保存 JSON
    json_path = os.path.join(args.output, 'liif_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved: {json_path}')


# ============================================================
# 入口
# ============================================================

def main():
    p = argparse.ArgumentParser(description='LIIF Arbitrary-Scale SR Testing')
    p.add_argument('--ckpt', type=str, required=True, help='模型 checkpoint 路径')
    p.add_argument('--scale', type=float, default=4.0, help='放大倍率')
    p.add_argument('--scales', type=float, nargs='+', default=None,
                   help='多倍率评估 (e.g., 2 4 6 1.5 3.5)')
    p.add_argument('--input', type=str, default=None, help='单张图推理')
    p.add_argument('--test_dir', type=str, default=None, help='测试目录（计算 PSNR）')
    p.add_argument('--output', type=str, default='results/liif', help='输出目录')
    p.add_argument('--max_images', type=int, default=None, help='最多测试图片数')
    p.add_argument('--batch_q', type=int, default=30000, help='分块查询大小')
    p.add_argument('--save_images', action='store_true', help='保存 SR 图像')
    p.add_argument('--device', type=str, default=None)

    # 模型参数（需和训练时一致）
    p.add_argument('--n_feats', type=int, default=64)
    p.add_argument('--n_resblocks', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=256)

    args = p.parse_args()

    if not args.input and not args.test_dir:
        p.error('需要指定 --input（单张图）或 --test_dir（目录评估）')

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = build_liif_model(
        encoder_args={'n_feats': args.n_feats, 'n_resblocks': args.n_resblocks},
        liif_args={'hidden_dim': args.hidden_dim},
    ).to(device)

    # 加载权重（兼容 core/checkpoint 的 'model' key）
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded checkpoint: {args.ckpt}')

    if args.input:
        run_single(model, args, device)
    else:
        run_eval(model, args, device)


if __name__ == '__main__':
    main()
