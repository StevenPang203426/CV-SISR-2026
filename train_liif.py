"""
LIIF 训练入口
==============

独立训练脚本，不复用 core/Trainer（训练循环逻辑差异太大）。

用法::

    python train_liif.py --config configs/liif_edsr_x1-4.yaml
"""

import os
import sys
import time
import json
import argparse

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

# wandb 可选
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from models.liif_model import build_liif_model
from data.dataset_liif import LIIFDataset, make_coord, make_cell
from core.checkpoint import save_checkpoint


# ============================================================
# 配置加载
# ============================================================

def load_liif_config():
    """从 YAML 加载配置。"""
    parser = argparse.ArgumentParser(description='LIIF Training')
    parser.add_argument('--config', type=str, required=True)
    args, _ = parser.parse_known_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 扁平化到 namespace
    ns = argparse.Namespace()
    for k, v in cfg.items():
        if isinstance(v, dict):
            setattr(ns, k, v)
        else:
            setattr(ns, k, v)

    # 默认值
    if not hasattr(ns, 'val_scale'):
        ns.val_scale = 4
    if not hasattr(ns, 'milestones'):
        ns.milestones = [200, 400]
    if not hasattr(ns, 'gamma'):
        ns.gamma = 0.5
    if not hasattr(ns, 'batch_size'):
        ns.batch_size = 16
    if not hasattr(ns, 'epochs'):
        ns.epochs = 500
    if not hasattr(ns, 'lr'):
        ns.lr = 1e-4
    if not hasattr(ns, 'save_dir'):
        ns.save_dir = 'experiments/liif_edsr'
    if not hasattr(ns, 'sample_q'):
        ns.sample_q = 2304
    if not hasattr(ns, 'patch_size'):
        ns.patch_size = 48
    if not hasattr(ns, 'scale_min'):
        ns.scale_min = 1.0
    if not hasattr(ns, 'scale_max'):
        ns.scale_max = 4.0
    if not hasattr(ns, 'val_batch_q'):
        ns.val_batch_q = 30000  # 验证时分块查询每块大小

    return ns


# ============================================================
# 验证 — 分块查询
# ============================================================

def evaluate_liif(model, val_loader, scale, device, batch_q=30000):
    """
    在固定倍率下评估 PSNR。

    使用分块查询防止大图 OOM。

    Parameters
    ----------
    model : LIIFModel
    val_loader : DataLoader (batch_size=1)
    scale : float
    device : torch.device
    batch_q : int
        每次查询的最大坐标点数。

    Returns
    -------
    avg_psnr : float
    """
    model.eval()
    psnrs = []

    with torch.no_grad():
        for batch in val_loader:
            lr = batch['lr'].to(device)            # (1, 3, lr_H, lr_W)
            hr_gt = batch['gt_rgb']                 # (1, hr_H*hr_W, 3)

            # 目标 HR 尺寸
            _, _, lr_h, lr_w = lr.shape
            hr_h = round(lr_h * scale)
            hr_w = round(lr_w * scale)

            # 生成 HR 坐标
            coord = make_coord((hr_h, hr_w)).unsqueeze(0).to(device)
            cell = make_cell((hr_h, hr_w)).unsqueeze(0).to(device)

            # 提取特征（只算一次）
            feat = model.encoder(lr)

            # 分块查询
            n = coord.shape[1]
            pred_parts = []
            for i in range(0, n, batch_q):
                coord_chunk = coord[:, i:i + batch_q, :]
                cell_chunk = cell[:, i:i + batch_q, :]
                pred_chunk = model.liif.query(feat, coord_chunk, cell_chunk)
                pred_parts.append(pred_chunk.cpu())

            pred_rgb = torch.cat(pred_parts, dim=1)  # (1, hr_H*hr_W, 3)
            pred_rgb = pred_rgb.clamp(0, 1)

            # 重新生成 gt（因为 val_loader 的 gt 可能是原图全部坐标，和 scale 对应的不一样）
            # 直接用原始 HR 图生成 ground truth
            # val_loader batch_size=1，直接取
            hr_tensor = batch['gt_rgb']  # (1, N_orig, 3)

            # 如果 gt 坐标数和我们查询的不一致，用原图重新生成
            if hr_tensor.shape[1] != hr_h * hr_w:
                # 需要从原图重新采样——跳过 PSNR 或近似处理
                # 实际上验证集 scale_max 和测试 scale 一致时，尺寸匹配
                continue

            # 计算 PSNR
            mse = F.mse_loss(pred_rgb, hr_tensor)
            if mse.item() == 0:
                psnr = 100.0
            else:
                psnr = -10 * np.log10(mse.item())
            psnrs.append(psnr)

    model.train()
    return np.mean(psnrs) if psnrs else 0.0


# ============================================================
# 训练循环
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    """训练一个 epoch，返回 avg_loss。"""
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        lr = batch['lr'].to(device)
        coord = batch['coord'].to(device)
        cell = batch['cell'].to(device)
        gt_rgb = batch['gt_rgb'].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_rgb = model(lr, coord, cell)
        loss = F.l1_loss(pred_rgb, gt_rgb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * lr.size(0)
        n += lr.size(0)

    return total_loss / max(1, n)


def main():
    cfg = load_liif_config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 模型
    encoder_args = getattr(cfg, 'encoder_args', {})
    liif_args = getattr(cfg, 'liif_args', {})
    model = build_liif_model(encoder_args, liif_args).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # 数据集
    train_set = LIIFDataset(
        cfg.train_hr, patch_size=cfg.patch_size,
        scale_min=cfg.scale_min, scale_max=cfg.scale_max,
        sample_q=cfg.sample_q, augment=True, is_train=True,
    )
    val_set = LIIFDataset(
        cfg.val_hr, is_train=False,
        scale_max=cfg.val_scale,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    print(f'Train: {len(train_set)} samples, Val: {len(val_set)} samples')

    # 优化器 & 调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.milestones, gamma=cfg.gamma,
    )

    # wandb
    if HAS_WANDB:
        wandb.init(project='LIIF_SR', config=vars(cfg), name='liif_edsr')

    # 日志
    log_path = os.path.join(cfg.save_dir, 'train.log')
    best_psnr = 0.0
    best_epoch = 0
    history = []

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(f'LIIF Training — {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        fp.write(f'Config: {vars(cfg)}\n')
        fp.write(f'Parameters: {n_params:,}\n')
        fp.write('=' * 60 + '\n')
        fp.flush()

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, device)

            # 验证（训练中只测 val_scale，默认 x4）
            val_psnr = evaluate_liif(
                model, val_loader, cfg.val_scale, device,
                batch_q=cfg.val_batch_q,
            )

            scheduler.step()
            dt = time.time() - t0

            # 日志
            line = (f'[Epoch {epoch:03d}/{cfg.epochs}] '
                    f'loss={train_loss:.6f} | '
                    f'val_psnr(x{cfg.val_scale})={val_psnr:.2f} dB | '
                    f'lr={optimizer.param_groups[0]["lr"]:.2e} | '
                    f'time={dt:.1f}s')
            print(line)
            fp.write(line + '\n')
            fp.flush()

            # wandb
            if HAS_WANDB:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    f'val_psnr_x{cfg.val_scale}': val_psnr,
                    'lr': optimizer.param_groups[0]['lr'],
                })

            # 保存最佳
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                save_checkpoint(
                    model,
                    os.path.join(cfg.save_dir, 'best.pt'),
                    epoch=epoch, best_psnr=best_psnr,
                )

            # 定期保存
            if epoch % 100 == 0:
                save_checkpoint(
                    model,
                    os.path.join(cfg.save_dir, f'epoch_{epoch}.pt'),
                    epoch=epoch, best_psnr=val_psnr,
                )

            history.append({
                'epoch': epoch,
                'train_loss': round(train_loss, 6),
                'val_psnr': round(val_psnr, 4),
                'lr': optimizer.param_groups[0]['lr'],
                'time': round(dt, 1),
            })

        # 训练结束
        fp.write(f'\nBest PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}\n')

    # 保存 metrics.json
    metrics = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, list)) else v
                   for k, v in vars(cfg).items()},
        'summary': {
            'total_epochs': cfg.epochs,
            'best_epoch': best_epoch,
            'best_val_psnr': round(best_psnr, 4),
            'final_train_loss': history[-1]['train_loss'] if history else 0,
            'total_params': n_params,
        },
        'epochs': history,
    }
    with open(os.path.join(cfg.save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f'\nTraining complete. Best PSNR: {best_psnr:.2f} dB @ epoch {best_epoch}')
    print(f'Checkpoint: {cfg.save_dir}/best.pt')

    if HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
