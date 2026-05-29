"""
log → json 格式转换脚本
========================

将 experiments/*/*.log 训练日志解析为结构化的 JSON 文件，
包含超参数、逐 epoch 指标、最佳指标和训练摘要。

用法::

    # 转换所有日志
    python -m src.common.log2json

    # 转换指定日志
    python -m src.common.log2json experiments/edsr_x2/edsr_x2.log

    # 自定义输出目录
    python -m src.common.log2json --out_dir report/
"""

import os
import re
import json
import glob
import argparse


def parse_log(log_path: str) -> dict:
    """
    解析单个训练日志文件，返回结构化字典。

    Returns
    -------
    dict
        {
            "log_file": str,
            "config": { "model", "scale", "epochs", "batch_size", ... },
            "epochs": [ { "epoch", "train_loss", "train_psnr", "val_loss", "val_psnr", "time" }, ... ],
            "best_psnr": float,
            "summary": { "total_epochs", "best_epoch", "best_val_psnr", "final_train_loss", ... }
        }
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ---- 解析超参数头部 ----
    config = {}
    # 映射: 日志中的标签 → JSON 键名
    config_map = {
        'Model':          ('model',        str),
        'Scale':          ('scale',        lambda s: int(s.replace('x', ''))),
        'Epochs':         ('epochs',       int),
        'Batch Size':     ('batch_size',   int),
        'Patch Size':     ('patch_size',   int),
        'LR':             ('lr',           str),
        'Optimizer':      ('optimizer',    str),
        'Criterion':      ('criterion',    str),
        'Wandb Project':  ('wandb_project', str),
    }
    for line in lines:
        line = line.strip()
        for label, (key, cast) in config_map.items():
            if line.startswith(f'{label}:'):
                value = line.split(':', 1)[1].strip()
                try:
                    config[key] = cast(value)
                except (ValueError, TypeError):
                    config[key] = value
                break

    # ---- 解析 epoch 行 ----
    epoch_pattern = re.compile(
        r'\[Epoch\s+(\d+)\]\s+'
        r'train_loss=([\d.]+)\s*\|\s*'
        r'train_psnr=([\d.]+)\s*dB\s*\|\s*'
        r'val_loss=([\d.]+)\s*\|\s*'
        r'val_psnr=([\d.]+)\s*dB\s*\|\s*'
        r'time=([\d.]+)s'
    )
    epochs = []
    for line in lines:
        m = epoch_pattern.search(line)
        if m:
            epochs.append({
                'epoch':      int(m.group(1)),
                'train_loss': float(m.group(2)),
                'train_psnr': float(m.group(3)),
                'val_loss':   float(m.group(4)),
                'val_psnr':   float(m.group(5)),
                'time':       float(m.group(6)),
            })

    # ---- 解析 Best PSNR ----
    best_psnr = None
    for line in lines:
        m = re.search(r'Best PSNR:\s*([\d.]+)\s*dB', line)
        if m:
            best_psnr = float(m.group(1))
            break

    # ---- 生成摘要 ----
    summary = {}
    if epochs:
        best_epoch_data = max(epochs, key=lambda e: e['val_psnr'])
        total_time = sum(e['time'] for e in epochs)
        summary = {
            'total_epochs':     len(epochs),
            'best_epoch':       best_epoch_data['epoch'],
            'best_val_psnr':    best_epoch_data['val_psnr'],
            'best_val_loss':    best_epoch_data['val_loss'],
            'final_train_loss': epochs[-1]['train_loss'],
            'final_train_psnr': epochs[-1]['train_psnr'],
            'final_val_loss':   epochs[-1]['val_loss'],
            'final_val_psnr':   epochs[-1]['val_psnr'],
            'total_time_sec':   round(total_time, 1),
            'avg_epoch_sec':    round(total_time / len(epochs), 1),
        }

    return {
        'log_file': log_path,
        'config':    config,
        'best_psnr': best_psnr,
        'summary':   summary,
        'epochs':    epochs,
    }


def main():
    p = argparse.ArgumentParser(description='将训练日志 (.log) 转换为 JSON')
    p.add_argument('logs', nargs='*', default=None,
                   help='要转换的 .log 文件路径（默认: experiments/*/*.log）')
    p.add_argument('--out_dir', default=None,
                   help='JSON 输出目录（默认: 与 .log 同目录）')
    args = p.parse_args()

    # 收集日志文件
    if args.logs:
        log_files = args.logs
    else:
        log_files = sorted(glob.glob('experiments/*/*.log'))

    if not log_files:
        print('未找到任何 .log 文件。')
        return

    print(f'找到 {len(log_files)} 个日志文件\n')

    for log_path in log_files:
        result = parse_log(log_path)

        # 确定输出路径
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(log_path))[0]
            json_path = os.path.join(args.out_dir, f'{base}.json')
        else:
            json_path = os.path.splitext(log_path)[0] + '.json'

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        model = result['config'].get('model', '?')
        scale = result['config'].get('scale', '?')
        n = len(result['epochs'])
        best = result['best_psnr'] or result['summary'].get('best_val_psnr', '?')
        print(f'  [{model}_x{scale}] {n} epochs, best PSNR={best} dB → {json_path}')

    print(f'\n转换完成，共 {len(log_files)} 个文件。')


if __name__ == '__main__':
    main()
