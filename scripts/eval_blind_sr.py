"""
Blind SR 统一评估脚本 — 对应 BLIND_SR_PLAN 实验 1-3。

功能
----
- 加载多个预训练模型（RRDB 系列 + 我们的 bicubic SR 模型）
- 在多种退化条件下生成 LR 图像
- 逐一推理并计算 PSNR / SSIM
- 输出结果表格 (JSON + 终端打印)
- 保存可视化对比图

用法
----
    # 实验 1：Bicubic SR vs Blind SR 在多退化下的对比
    python scripts/eval_blind_sr.py --experiment 1

    # 实验 2：不同退化因素的消融
    python scripts/eval_blind_sr.py --experiment 2

    # 实验 3：PSNR vs GAN 感知质量（在 RealSRSet 上，无参考指标）
    python scripts/eval_blind_sr.py --experiment 3

    # 指定 GPU
    python scripts/eval_blind_sr.py --experiment 1 --device cuda:0
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """让 json.dump 支持 numpy 数值类型。"""
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.rrdbnet import RRDBNet
from models import build_model
from core.checkpoint import load_checkpoint
from data.degradation import (
    DegradationPipeline, degradation_bicubic, degradation_custom, modcrop,
    uint2single, single2uint
)


# ============================================================
# 模型加载
# ============================================================

PRETRAINED_MODELS = {
    'RealESRGAN_x4plus': {'path': 'pretrained/RealESRGAN_x4plus.pth', 'type': 'rrdb'},
    'RealESRNet_x4plus': {'path': 'pretrained/RealESRNet_x4plus.pth', 'type': 'rrdb'},
    'BSRGAN':            {'path': 'pretrained/BSRGAN.pth',            'type': 'rrdb'},
    'BSRNet':            {'path': 'pretrained/BSRNet.pth',            'type': 'rrdb'},
    'ESRGAN':            {'path': 'pretrained/ESRGAN.pth',            'type': 'rrdb'},
}

OUR_MODELS = {
    'EDSR_x4':  {'path': 'experiments/edsr_x4/best.pt',  'name': 'edsr',  'scale': 4},
    'ESPCN_x4': {'path': 'experiments/espcn_x4/best.pt', 'name': 'espcn', 'scale': 4},
    'SRCNN_x4': {'path': 'experiments/srcnn_x4/best.pt', 'name': 'srcnn', 'scale': 4},
}


def load_rrdb_model(name, device):
    """加载 RRDB 预训练模型。"""
    info = PRETRAINED_MODELS[name]
    path = os.path.join(ROOT, info['path'])
    if not os.path.exists(path):
        print(f'[SKIP] {path} 不存在，请先运行 scripts/download_pretrained.py')
        return None
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    model.load_pretrained(path)
    model.eval().to(device)
    return model


def load_our_model(name, device):
    """加载我们训练的 bicubic SR 模型。"""
    info = OUR_MODELS[name]
    path = os.path.join(ROOT, info['path'])
    if not os.path.exists(path):
        print(f'[SKIP] {path} 不存在')
        return None
    model = build_model(info['name'], scale=info['scale'], in_channels=3)
    load_checkpoint(model, path, device=str(device))
    model.eval().to(device)
    return model


# ============================================================
# 推理工具
# ============================================================

def infer_rrdb(model, lr_img, device):
    """RRDB 模型推理。lr_img: HxWxC numpy [0,1]"""
    tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)


def infer_our(model, lr_img, device):
    """我们的 SR 模型推理。lr_img: HxWxC numpy [0,1]"""
    tensor = torch.from_numpy(lr_img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(tensor)
    return output.squeeze(0).cpu().numpy().transpose(1, 2, 0).clip(0, 1)


def compute_metrics(sr, hr):
    """计算 PSNR 和 SSIM。sr, hr: HxWxC numpy [0,1]"""
    # 确保尺寸一致
    h = min(sr.shape[0], hr.shape[0])
    w = min(sr.shape[1], hr.shape[1])
    sr, hr = sr[:h, :w], hr[:h, :w]

    psnr = calc_psnr(hr, sr, data_range=1.0)
    ssim = calc_ssim(hr, sr, data_range=1.0, channel_axis=2)
    return psnr, ssim


def load_hr_images(img_dir, max_images=20):
    """加载 HR 图像列表。"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:max_images]
    images = []
    for f in files:
        img = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
        images.append((f, uint2single(img)))
    return images


def save_comparison(images_dict, output_path, title=''):
    """保存多张图像的水平对比图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, images_dict.items()):
        ax.imshow(img.clip(0, 1))
        ax.set_title(name, fontsize=8)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 实验 1：Bicubic SR vs Blind SR
# ============================================================

def run_experiment_1(args):
    """Bicubic SR 在真实退化下的崩溃。"""
    print('\n' + '=' * 60)
    print('实验 1：Bicubic SR vs Blind SR 在多退化下的对比')
    print('=' * 60)

    device = torch.device(args.device)
    sf = 4
    out_dir = os.path.join(ROOT, 'experiments/blind_sr/eval_results/exp1')
    os.makedirs(out_dir, exist_ok=True)

    # 加载模型
    models = {}
    for name in ['EDSR_x4', 'ESPCN_x4']:
        m = load_our_model(name, device)
        if m:
            models[name] = ('our', m)
    for name in ['RealESRGAN_x4plus', 'BSRGAN']:
        m = load_rrdb_model(name, device)
        if m:
            models[name] = ('rrdb', m)

    if not models:
        print('没有可用的模型，请先下载预训练权重或训练 bicubic SR 模型')
        return

    # 加载测试图像
    hr_dir = os.path.join(ROOT, 'data/DIV2K_valid_HR')
    if not os.path.isdir(hr_dir):
        # 尝试其他可能的路径
        for alt in ['data/DIV2K/DIV2K_valid_HR', 'data/datasets/DIV2K_valid_HR']:
            alt_path = os.path.join(ROOT, alt)
            if os.path.isdir(alt_path):
                hr_dir = alt_path
                break
    if not os.path.isdir(hr_dir):
        print(f'[ERROR] 找不到验证集目录: {hr_dir}')
        print('请确保 DIV2K 验证集放在 data/DIV2K_valid_HR/')
        return

    images = load_hr_images(hr_dir, max_images=args.num_images)
    print(f'加载了 {len(images)} 张测试图像')

    # 退化条件
    degradations = {
        'Bicubic': lambda img: degradation_bicubic(img, sf=sf),
        'BSRGAN-mild': lambda img: degradation_custom(
            img, sf=sf, blur_sigma=1.0, noise_level=5, jpeg_quality=70),
        'BSRGAN-heavy': lambda img: degradation_custom(
            img, sf=sf, blur_sigma=2.5, noise_level=20, jpeg_quality=30),
    }

    # 结果收集
    results = defaultdict(lambda: defaultdict(list))

    for img_name, hr_img in tqdm(images, desc='评估中'):
        hr = modcrop(hr_img, sf)

        for deg_name, deg_fn in degradations.items():
            lr = deg_fn(hr_img)

            for model_name, (model_type, model) in models.items():
                if model_type == 'rrdb':
                    sr = infer_rrdb(model, lr, device)
                else:
                    sr = infer_our(model, lr, device)

                psnr, ssim = compute_metrics(sr, hr)
                results[deg_name][model_name].append({'psnr': psnr, 'ssim': ssim})

        # 保存第一张图的对比
        if img_name == images[0][0]:
            for deg_name, deg_fn in degradations.items():
                lr = deg_fn(hr_img)
                comparison = {'LR (bicubic upscale)': cv2.resize(
                    lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)}
                for model_name, (model_type, model) in models.items():
                    sr = infer_rrdb(model, lr, device) if model_type == 'rrdb' else infer_our(model, lr, device)
                    comparison[model_name] = sr
                comparison['HR (Ground Truth)'] = hr
                save_comparison(
                    comparison,
                    os.path.join(out_dir, f'compare_{deg_name}_{img_name}.png'),
                    title=f'{deg_name} degradation'
                )

    # 汇总并打印
    summary = {}
    print(f'\n{"退化条件":<16}', end='')
    for model_name in models:
        print(f'{model_name:<22}', end='')
    print()
    print('-' * (16 + 22 * len(models)))

    for deg_name in degradations:
        print(f'{deg_name:<16}', end='')
        summary[deg_name] = {}
        for model_name in models:
            vals = results[deg_name][model_name]
            avg_psnr = np.mean([v['psnr'] for v in vals])
            avg_ssim = np.mean([v['ssim'] for v in vals])
            print(f'{avg_psnr:.2f}/{avg_ssim:.4f}       ', end='')
            summary[deg_name][model_name] = {
                'psnr': round(avg_psnr, 2), 'ssim': round(avg_ssim, 4)
            }
        print()

    # 保存 JSON
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f'\n结果已保存: {json_path}')


# ============================================================
# 实验 2：退化因素消融
# ============================================================

def run_experiment_2(args):
    """不同退化类型对 SR 质量的影响。"""
    print('\n' + '=' * 60)
    print('实验 2：退化因素消融实验')
    print('=' * 60)

    device = torch.device(args.device)
    sf = 4
    out_dir = os.path.join(ROOT, 'experiments/blind_sr/eval_results/exp2')
    os.makedirs(out_dir, exist_ok=True)

    # 加载模型
    models = {}
    m = load_our_model('EDSR_x4', device)
    if m:
        models['EDSR_x4'] = ('our', m)
    m = load_rrdb_model('RealESRGAN_x4plus', device)
    if m:
        models['RealESRGAN'] = ('rrdb', m)

    if not models:
        print('需要 EDSR_x4 和 RealESRGAN_x4plus')
        return

    hr_dir = os.path.join(ROOT, 'data/DIV2K_valid_HR')
    if not os.path.isdir(hr_dir):
        for alt in ['data/DIV2K/DIV2K_valid_HR', 'data/datasets/DIV2K_valid_HR']:
            if os.path.isdir(os.path.join(ROOT, alt)):
                hr_dir = os.path.join(ROOT, alt)
                break
    if not os.path.isdir(hr_dir):
        print(f'[ERROR] 找不到验证集: {hr_dir}')
        return

    images = load_hr_images(hr_dir, max_images=10)

    # 退化组合
    conditions = {
        'D0: Bicubic':          dict(blur_sigma=0,   noise_level=0,  jpeg_quality=0),
        'D1: +Blur':            dict(blur_sigma=1.5, noise_level=0,  jpeg_quality=0),
        'D2: +Noise':           dict(blur_sigma=0,   noise_level=15, jpeg_quality=0),
        'D3: +JPEG':            dict(blur_sigma=0,   noise_level=0,  jpeg_quality=40),
        'D4: Blur+Noise':       dict(blur_sigma=1.5, noise_level=15, jpeg_quality=0),
        'D5: Blur+Noise+JPEG':  dict(blur_sigma=1.5, noise_level=15, jpeg_quality=40),
    }

    results = defaultdict(lambda: defaultdict(list))

    for img_name, hr_img in tqdm(images, desc='消融实验'):
        hr = modcrop(hr_img, sf)
        for cond_name, params in conditions.items():
            lr = degradation_custom(hr_img, sf=sf, **params)
            for model_name, (model_type, model) in models.items():
                sr = infer_rrdb(model, lr, device) if model_type == 'rrdb' else infer_our(model, lr, device)
                psnr, ssim = compute_metrics(sr, hr)
                results[cond_name][model_name].append(psnr)

    # 打印表格
    print(f'\n{"退化条件":<24}', end='')
    for mn in models:
        print(f'{mn:<18}', end='')
    print()
    print('-' * (24 + 18 * len(models)))

    summary = {}
    for cond_name in conditions:
        print(f'{cond_name:<24}', end='')
        summary[cond_name] = {}
        for mn in models:
            avg = np.mean(results[cond_name][mn])
            print(f'{avg:.2f} dB          ', end='')
            summary[cond_name][mn] = round(avg, 2)
        print()

    # 保存
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    # 画折线图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(len(conditions)))
        labels = [k.split(': ')[1] for k in conditions]
        for mn in models:
            vals = [np.mean(results[cn][mn]) for cn in conditions]
            ax.plot(x, vals, 'o-', label=mn, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Degradation Ablation: PSNR under Different Conditions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'ablation_chart.png'), dpi=150)
        plt.close()
        print(f'折线图已保存: {out_dir}/ablation_chart.png')
    except ImportError:
        pass

    print(f'结果已保存: {json_path}')


# ============================================================
# 实验 3：PSNR vs GAN 感知质量
# ============================================================

def run_experiment_3(args):
    """PSNR 训练 vs GAN 训练的感知质量对比（无参考指标）。"""
    print('\n' + '=' * 60)
    print('实验 3：PSNR vs GAN 感知质量对比')
    print('=' * 60)

    device = torch.device(args.device)
    out_dir = os.path.join(ROOT, 'experiments/blind_sr/eval_results/exp3')
    os.makedirs(out_dir, exist_ok=True)

    # PSNR vs GAN 对比模型
    model_pairs = {
        'BSRNet':            'pretrained/BSRNet.pth',
        'BSRGAN':            'pretrained/BSRGAN.pth',
        'RealESRNet_x4plus': 'pretrained/RealESRNet_x4plus.pth',
        'RealESRGAN_x4plus': 'pretrained/RealESRGAN_x4plus.pth',
    }

    models = {}
    for name, path in model_pairs.items():
        full_path = os.path.join(ROOT, path)
        if os.path.exists(full_path):
            m = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
            m.load_pretrained(full_path)
            m.eval().to(device)
            models[name] = m
        else:
            print(f'[SKIP] {full_path} 不存在')

    if not models:
        print('需要至少一个预训练模型')
        return

    # 测试图像：RealSRSet（真实退化，无 GT）
    test_dirs = [
        os.path.join(ROOT, 'data/RealSRSet'),
        os.path.join(ROOT, 'features/BSRGAN/testsets/RealSRSet'),  # 旧路径兼容
    ]
    test_dir = None
    for td in test_dirs:
        if os.path.isdir(td):
            test_dir = td
            break

    if test_dir is None:
        print('[ERROR] 找不到 RealSRSet 测试集')
        print('请确保 data/RealSRSet/ 存在')
        return

    images = load_hr_images(test_dir, max_images=20)
    print(f'加载了 {len(images)} 张真实退化图像（无 GT）')

    # 推理并保存结果
    for img_name, lr_img in tqdm(images[:5], desc='实验3 推理'):
        comparison = {'Input (LR)': lr_img}
        for model_name, model in models.items():
            sr = infer_rrdb(model, lr_img, device)
            comparison[model_name] = sr

            # 单独保存 SR 结果
            sr_dir = os.path.join(out_dir, model_name)
            os.makedirs(sr_dir, exist_ok=True)
            Image.fromarray(single2uint(sr)).save(
                os.path.join(sr_dir, img_name))

        save_comparison(
            comparison,
            os.path.join(out_dir, f'compare_{img_name}'),
            title=f'PSNR vs GAN: {img_name}'
        )

    print(f'\n对比图已保存: {out_dir}/')
    print('提示: 运行 scripts/calc_niqe.py 计算无参考质量指标 (NIQE/BRISQUE)')


# ============================================================
# 入口
# ============================================================

def main():
    p = argparse.ArgumentParser(description='Blind SR 统一评估脚本')
    p.add_argument('--experiment', type=int, required=True, choices=[1, 2, 3],
                   help='实验编号: 1=Bicubic崩溃, 2=退化消融, 3=PSNR vs GAN')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-images', type=int, default=20, help='测试图像数量 (实验1)')
    args = p.parse_args()

    if args.experiment == 1:
        run_experiment_1(args)
    elif args.experiment == 2:
        run_experiment_2(args)
    elif args.experiment == 3:
        run_experiment_3(args)


if __name__ == '__main__':
    import cv2  # 确保 cv2 在顶层可用
    main()
