"""
评估模块
=========

提供在验证集/测试集上评估模型性能的统一接口。
"""

import time
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from utils.metrics import psnr_y, ssim_y
from utils.profile import count_params_m, profile_flops_g


def _to_uint8(tensor):
    """将 [0,1] 范围的张量转换为 uint8 numpy 数组。"""
    return (tensor.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')


class Evaluator:
    """
    模型评估器。

    支持两种模式：
    - **快速评估**：``evaluate()`` 仅计算平均 loss 和 PSNR(Y)，用于训练中的验证。
    - **完整测试**：``test()`` 计算 PSNR/SSIM/参数量/FLOPs/FPS，可选保存 SR 图像和 JSON 报告。
    """

    def __init__(self, model: nn.Module, device: torch.device, scale: int):
        self.model = model
        self.device = device
        self.scale = scale

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, criterion: nn.Module):
        """
        在验证集上快速评估，返回 (avg_loss, avg_psnr)。

        用于训练循环中每个 epoch 的验证步骤。
        """
        self.model.eval()
        losses, scores = [], []

        for batch in loader:
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            sr = self.model(lr)

            loss = criterion(sr, hr)
            losses.append(loss.item())

            psnr = psnr_y(
                Image.fromarray(_to_uint8(sr[0])),
                Image.fromarray(_to_uint8(hr[0])),
                shave=self.scale,
            )
            scores.append(psnr)

        self.model.train()
        avg_loss = sum(losses) / max(len(losses), 1)
        avg_psnr = sum(scores) / max(len(scores), 1)
        return avg_loss, avg_psnr

    @torch.no_grad()
    def test(self, loader: DataLoader, dataset_name: str,
             model_name: str, out_dir: Path = None, save_images: bool = False,
             json_path: Path = None):
        """
        完整测试：计算 PSNR/SSIM/参数量/FLOPs/FPS，输出 JSON 报告。

        Parameters
        ----------
        loader : DataLoader
            测试集 DataLoader（batch_size=1）。
        dataset_name : str
            数据集名称（用于报告）。
        model_name : str
            模型名称（用于报告）。
        out_dir : Path, optional
            SR 图像输出目录。
        save_images : bool
            是否保存 SR/LR 图像。
        json_path : Path, optional
            JSON 报告输出路径。

        Returns
        -------
        dict
            包含所有测试指标的汇总字典。
        """
        self.model.eval()
        if out_dir is None:
            out_dir = Path('experiments') / f"{model_name}_x{self.scale}" / 'test' / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 计算模型参数量和 FLOPs
        example = next(iter(loader))
        inp_size = (1,) + tuple(example['lr'].shape[1:])
        params = count_params_m(self.model)
        flops_g = profile_flops_g(self.model, inp_size)

        psnrs, ssims, image_metrics = [], [], []
        n, total_time = 0, 0.0

        for i, batch in enumerate(loader, start=1):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)

            t0 = time.time()
            sr = self.model(lr)
            total_time += time.time() - t0
            n += 1

            sr_img = Image.fromarray(_to_uint8(sr[0]))
            hr_img = Image.fromarray(_to_uint8(hr[0]))
            lr_img = Image.fromarray(_to_uint8(lr[0]))

            psnr = psnr_y(sr_img, hr_img, shave=self.scale)
            ssim = ssim_y(sr_img, hr_img, shave=self.scale)
            psnrs.append(psnr)
            ssims.append(ssim)
            image_metrics.append({'image_index': i, 'psnr_y': psnr, 'ssim_y': ssim})

            if save_images:
                sr_img.save(out_dir / f'{i:02d}_SR.png')
                lr_img.save(out_dir / f'{i:02d}_LR.png')

        summary = {
            'model': model_name,
            'scale': self.scale,
            'dataset': dataset_name,
            'num_images': n,
            'psnr_y': sum(psnrs) / max(len(psnrs), 1),
            'ssim_y': sum(ssims) / max(len(ssims), 1),
            'params': params,
            'flops_G': round(flops_g, 4) if flops_g >= 0 else None,
            'fps': round(n / max(total_time, 1e-9), 3),
            'image_metrics': image_metrics,
        }

        # 输出到控制台
        print(
            f"Avg PSNR(Y): {summary['psnr_y']:.2f} dB | "
            f"SSIM(Y): {summary['ssim_y']:.4f} | "
            f"Params: {summary['params']} | "
            f"FLOPs: {summary['flops_G']}G | "
            f"FPS: {summary['fps']:.2f}"
        )

        # 保存 JSON 报告
        if json_path is None:
            json_path = out_dir / 'metrics.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to {json_path}")

        return summary
