"""
训练入口脚本
=============

薄包装器：解析配置 → 构建模型/优化器/数据集 → 委托给 core.Trainer 执行。

用法::

    python train.py --config configs/edsr_x2.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import load_config
from core.trainer import Trainer
from data.dataset import SRDataset
from models import build_model


def _build_optimizer(model, args):
    """
    根据配置构建优化器。

    SRCNN 使用分组学习率（reconstruction 层 lr × 0.1）。
    """
    lr = float(args.lr)

    if args.model == 'srcnn':
        param_groups = [
            {'params': model.features.parameters()},
            {'params': model.map.parameters()},
            {'params': model.reconstruction.parameters(), 'lr': lr * 0.1},
        ]
    else:
        param_groups = model.parameters()

    optimizers = {
        'SGD':  lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=1e-4),
        'Adam': lambda p: torch.optim.Adam(p, lr=lr, betas=(0.9, 0.999)),
    }
    return optimizers[args.opt](param_groups)


def main():
    args = load_config()

    # 设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    # 数据集
    train_set = SRDataset(
        args.model, args.train_dir,
        scale=args.scale, patch_size=args.patch_size,
        augment=True, is_train=True,
    )
    val_set = SRDataset(
        args.model, args.val_dir,
        scale=args.scale, is_train=False,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    # 模型（通过统一注册表构建）
    model = build_model(args.model, scale=args.scale, in_channels=3).to(device)

    # 优化器 & 损失 & 调度器
    optimizer = _build_optimizer(model, args)
    criterion = {'L1': nn.L1Loss(), 'MSE': nn.MSELoss()}[args.crit]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6,
    )

    # 训练
    trainer = Trainer(model, optimizer, criterion, device, args.scale, scheduler)
    trainer.fit(train_loader, val_loader, args)


if __name__ == '__main__':
    main()
