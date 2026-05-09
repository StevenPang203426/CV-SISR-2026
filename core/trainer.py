"""
训练模块
=========

封装完整的训练循环，包括前向/反向传播、验证、日志记录、
学习率调度、最佳模型保存、指标曲线绘制。
"""

import os
import time

import torch
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from utils.metrics import psnr_y
from .checkpoint import save_checkpoint
from .evaluator import Evaluator, _to_uint8


def _log_line(fp, text: str):
    """打印到控制台并写入日志文件。"""
    print(text)
    fp.write(text + '\n')
    fp.flush()


def _plot_metrics(model_label: str, train_loss, train_psnr,
                  val_loss, val_psnr, save_dir: str):
    """绘制训练/验证的 Loss 和 PSNR 曲线并保存为 PNG。"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title(f'{model_label} Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(train_psnr, label='Train PSNR', color='green')
    plt.plot(val_psnr, label='Validation PSNR', color='red')
    plt.title(f'{model_label} PSNR Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics.png'))
    plt.close()


class Trainer:
    """
    训练器：管理完整的训练生命周期。

    Parameters
    ----------
    model : nn.Module
        待训练模型。
    optimizer : torch.optim.Optimizer
        优化器。
    criterion : nn.Module
        损失函数。
    device : torch.device
        训练设备。
    scale : int
        超分辨率倍数。
    scheduler : optional
        学习率调度器。
    """

    def __init__(self, model, optimizer, criterion, device, scale, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scale = scale
        self.scheduler = scheduler
        self.evaluator = Evaluator(model, device, scale)

    def _train_one_epoch(self, loader: DataLoader):
        """执行一个 epoch 的训练，返回 (avg_loss, avg_psnr)。"""
        self.model.train()
        total_loss = 0.0
        total_psnr = 0.0
        n = 0

        for batch in loader:
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            batch_size = lr.size(0)

            self.optimizer.zero_grad(set_to_none=True)
            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            n += batch_size

            # 计算训练 PSNR
            for i in range(batch_size):
                total_psnr += psnr_y(
                    Image.fromarray(_to_uint8(sr[i].detach())),
                    Image.fromarray(_to_uint8(hr[i])),
                    shave=self.scale,
                )

        avg_loss = total_loss / max(1, n)
        avg_psnr = total_psnr / max(1, n)
        return avg_loss, avg_psnr

    def fit(self, train_loader, val_loader, args):
        """
        执行完整训练流程。

        Parameters
        ----------
        train_loader : DataLoader
            训练集。
        val_loader : DataLoader
            验证集。
        args : Namespace
            包含 epochs, save_dir, model, scale 等属性的配置对象。
        """
        os.makedirs(args.save_dir, exist_ok=True)
        log_path = os.path.join(args.save_dir, f"{args.model}_x{args.scale}.log")
        model_label = f"{args.model.upper()}_x{args.scale}"

        # 初始化 wandb
        project_name = getattr(args, 'project_name', f"{args.model.upper()}_X{args.scale}")
        wandb.init(project=project_name, config=vars(args), name=f"{args.model}_x{args.scale}")

        # 历史记录
        train_loss_hist, train_psnr_hist = [], []
        val_loss_hist, val_psnr_hist = [], []
        best = 0.0

        with open(log_path, 'w', encoding='utf-8') as fp:
            # 记录超参数
            _log_line(fp, "====================")
            _log_line(fp, f"Model: {args.model.upper()}")
            _log_line(fp, f"Scale: x{args.scale}")
            _log_line(fp, f"Epochs: {args.epochs}")
            _log_line(fp, f"Batch Size: {args.batch_size}")
            _log_line(fp, f"Patch Size: {args.patch_size}")
            _log_line(fp, f"LR: {args.lr}")
            _log_line(fp, f"Optimizer: {args.opt}")
            _log_line(fp, f"Criterion: {args.crit}")
            _log_line(fp, f"Wandb Project: {project_name}")
            _log_line(fp, "====================")

            for epoch in range(1, args.epochs + 1):
                t0 = time.time()

                train_loss, train_psnr = self._train_one_epoch(train_loader)
                val_loss, val_psnr = self.evaluator.evaluate(val_loader, self.criterion)

                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step(val_psnr)

                # 记录历史
                train_loss_hist.append(train_loss)
                train_psnr_hist.append(train_psnr)
                val_loss_hist.append(val_loss)
                val_psnr_hist.append(val_psnr)

                dt = time.time() - t0
                _log_line(
                    fp,
                    f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
                    f"train_psnr={train_psnr:.2f} dB | val_loss={val_loss:.4f} | "
                    f"val_psnr={val_psnr:.2f} dB | time={dt:.1f}s"
                )

                # wandb 日志
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_psnr': train_psnr,
                    'val_loss': val_loss,
                    'val_psnr': val_psnr,
                    'lr': self.optimizer.param_groups[0]['lr'],
                })

                # 保存最佳模型
                if val_psnr > best:
                    best = val_psnr
                    save_checkpoint(
                        self.model,
                        os.path.join(args.save_dir, 'best.pt'),
                        epoch=epoch, best_psnr=best,
                    )

            _log_line(fp, f"\nBest PSNR: {best:.2f} dB")
            _plot_metrics(
                model_label,
                train_loss_hist, train_psnr_hist,
                val_loss_hist, val_psnr_hist,
                args.save_dir,
            )
            _log_line(fp, f"Training complete. Metrics plot saved to {args.save_dir}/metrics.png")
            wandb.finish()
