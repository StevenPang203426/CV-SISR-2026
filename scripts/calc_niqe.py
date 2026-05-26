"""
无参考图像质量评估脚本 — 计算 NIQE / BRISQUE。

用于实验 3：真实退化图像没有 HR Ground Truth，只能用无参考指标。

依赖
----
    pip install pyiqa

用法
----
    # 评估单个目录中所有图像
    python scripts/calc_niqe.py --input experiments/blind_sr/eval_results/exp3/BSRGAN/

    # 对比多个目录（一次性评估所有模型的输出）
    python scripts/calc_niqe.py --compare experiments/blind_sr/eval_results/exp3/

    # 指定指标
    python scripts/calc_niqe.py --input path/to/images --metrics niqe brisque
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_pyiqa():
    """检查 pyiqa 是否安装。"""
    try:
        import pyiqa
        return True
    except ImportError:
        print('[ERROR] pyiqa 未安装。请运行:')
        print('  uv add pyiqa')
        print('  # 或 pip install pyiqa')
        return False


def evaluate_directory(img_dir, metrics=('niqe', 'brisque'), device='cpu'):
    """
    计算一个目录中所有图像的无参考质量指标。

    Returns
    -------
    results : dict[str, dict[str, float]]
        {filename: {metric: value}}
    averages : dict[str, float]
        {metric: average_value}
    """
    import pyiqa
    import torch

    # 创建指标计算器
    calculators = {}
    for m in metrics:
        try:
            calculators[m] = pyiqa.create_metric(m, device=torch.device(device))
        except Exception as e:
            print(f'[WARN] 无法创建 {m} 指标: {e}')

    if not calculators:
        return {}, {}

    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = sorted([
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])

    if not files:
        print(f'[WARN] {img_dir} 中没有找到图像文件')
        return {}, {}

    results = {}
    sums = defaultdict(float)

    for fname in tqdm(files, desc=os.path.basename(img_dir)):
        fpath = os.path.join(img_dir, fname)
        scores = {}
        for metric_name, calc in calculators.items():
            try:
                score = calc(fpath).item()
                scores[metric_name] = round(score, 4)
                sums[metric_name] += score
            except Exception as e:
                scores[metric_name] = None
                print(f'  [WARN] {fname} {metric_name}: {e}')
        results[fname] = scores

    n = len(files)
    averages = {m: round(sums[m] / n, 4) for m in calculators if sums[m] > 0}

    return results, averages


def run_single(args):
    """评估单个目录。"""
    if not check_pyiqa():
        return

    results, averages = evaluate_directory(
        args.input, metrics=args.metrics, device=args.device)

    if not results:
        return

    print(f'\n--- {args.input} ---')
    print(f'图像数: {len(results)}')
    for m, v in averages.items():
        lower_better = m in ('niqe', 'brisque')
        arrow = '↓' if lower_better else '↑'
        print(f'  {m}: {v:.4f} ({arrow} lower is better)' if lower_better
              else f'  {m}: {v:.4f} ({arrow} higher is better)')

    # 保存
    out = {'averages': averages, 'per_image': results}
    json_path = os.path.join(args.input, 'niqe_scores.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'结果已保存: {json_path}')


def run_compare(args):
    """对比多个子目录（每个子目录是一个模型的输出）。"""
    if not check_pyiqa():
        return

    base_dir = args.compare
    subdirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    if not subdirs:
        print(f'[ERROR] {base_dir} 下没有子目录')
        return

    print(f'找到 {len(subdirs)} 个模型目录: {subdirs}\n')

    all_results = {}
    for subdir in subdirs:
        full_path = os.path.join(base_dir, subdir)
        _, averages = evaluate_directory(
            full_path, metrics=args.metrics, device=args.device)
        all_results[subdir] = averages

    # 汇总表格
    print(f'\n{"=" * 50}')
    print(f'{"模型":<25}', end='')
    for m in args.metrics:
        print(f'{m:<12}', end='')
    print()
    print('-' * (25 + 12 * len(args.metrics)))

    for model_name, avgs in all_results.items():
        print(f'{model_name:<25}', end='')
        for m in args.metrics:
            v = avgs.get(m, 'N/A')
            if isinstance(v, float):
                print(f'{v:<12.4f}', end='')
            else:
                print(f'{v:<12}', end='')
        print()

    # 保存
    json_path = os.path.join(base_dir, 'niqe_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n对比结果已保存: {json_path}')


def main():
    p = argparse.ArgumentParser(description='无参考图像质量评估 (NIQE/BRISQUE)')
    p.add_argument('--input', type=str, help='单个图像目录')
    p.add_argument('--compare', type=str, help='对比模式：包含多个子目录的父目录')
    p.add_argument('--metrics', nargs='+', default=['niqe', 'brisque'],
                   help='指标列表 (默认: niqe brisque)')
    p.add_argument('--device', type=str, default='cpu',
                   help='计算设备 (cpu/cuda)')
    args = p.parse_args()

    if not args.input and not args.compare:
        p.error('需要指定 --input 或 --compare')

    if args.compare:
        run_compare(args)
    elif args.input:
        run_single(args)


if __name__ == '__main__':
    main()
