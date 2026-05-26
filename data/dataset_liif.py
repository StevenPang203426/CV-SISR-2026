"""
LIIF 训练数据集 — 随机倍率 + 坐标采样
=======================================

与固定倍率 SRDataset 的关键区别:
1. 每次随机采样一个倍率 s ∈ [scale_min, scale_max]
2. 输出的不是 HR 图像，而是坐标点 + 对应的 ground truth RGB

用法::

    train_set = LIIFDataset('data/DIV2K_train_HR', patch_size=48,
                            scale_min=1.0, scale_max=4.0, sample_q=2304)
    val_set   = LIIFDataset('data/DIV2K_valid_HR', is_train=False,
                            scale_max=4.0)
"""

import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def make_coord(shape, device='cpu'):
    """
    生成 [-1, 1] 范围内的坐标网格。

    Parameters
    ----------
    shape : tuple (H, W)

    Returns
    -------
    Tensor (H*W, 2) — 每行 (row_coord, col_coord)
    """
    H, W = shape
    coord_h = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
    coord_w = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid = torch.stack(
        torch.meshgrid(coord_h, coord_w, indexing='ij'), dim=-1
    )
    return grid.view(-1, 2)  # (H*W, 2)


def make_cell(shape, device='cpu'):
    """
    计算每个坐标点的 cell 大小。

    Parameters
    ----------
    shape : tuple (H, W)

    Returns
    -------
    Tensor (H*W, 2) — 每行 (2/H, 2/W)
    """
    H, W = shape
    cell = torch.ones(H * W, 2, device=device)
    cell[:, 0] *= 2 / H
    cell[:, 1] *= 2 / W
    return cell


class LIIFDataset(Dataset):
    """
    LIIF 训练/验证数据集。

    Parameters
    ----------
    hr_dir : str
        HR 图像目录。
    patch_size : int
        HR patch 大小（训练时裁剪），默认 48。
    scale_min, scale_max : float
        训练时随机采样的倍率范围。
    sample_q : int
        每张图随机采样的查询坐标数（训练时不查全部像素），默认 2304。
    augment : bool
        是否数据增强，默认 True。
    is_train : bool
        训练模式 vs 验证模式。
    """

    def __init__(self, hr_dir, patch_size=48,
                 scale_min=1.0, scale_max=4.0,
                 sample_q=2304, augment=True, is_train=True):
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.hr_paths = sorted([
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if f.lower().endswith(exts)
        ])
        if not self.hr_paths:
            raise FileNotFoundError(f'No images found in {hr_dir}')

        self.patch_size = patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.augment = augment and is_train
        self.is_train = is_train
        self.to_tensor = T.ToTensor()

    def __len__(self):
        # 训练时每张图用 4 次（不同裁剪/倍率），增加多样性
        return len(self.hr_paths) * (4 if self.is_train else 1)

    def __getitem__(self, idx):
        img_idx = idx // 4 if self.is_train else idx
        hr = Image.open(self.hr_paths[img_idx]).convert('RGB')

        if self.is_train:
            # 随机采样倍率
            scale = random.uniform(self.scale_min, self.scale_max)

            w, h = hr.size
            ps = self.patch_size

            # 确保图片够大
            if w < ps or h < ps:
                hr = hr.resize((max(ps, w), max(ps, h)), Image.BICUBIC)
                w, h = hr.size

            # 随机裁剪 HR patch
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            hr = hr.crop((x, y, x + ps, y + ps))

            # 数据增强
            if self.augment:
                if random.random() < 0.5:
                    hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
                rotation = random.choice([0, 90, 180, 270])
                if rotation:
                    hr = hr.rotate(rotation, expand=True)

            # 生成 LR（bicubic 下采样）
            hr_h, hr_w = hr.size[1], hr.size[0]
            lr_h = round(hr_h / scale)
            lr_w = round(hr_w / scale)
            # 确保 LR 至少 1×1
            lr_h, lr_w = max(1, lr_h), max(1, lr_w)
            lr = hr.resize((lr_w, lr_h), Image.BICUBIC)

        else:
            # 验证时用固定倍率
            scale = self.scale_max
            w, h = hr.size
            lr_w = round(w / scale)
            lr_h = round(h / scale)
            lr_w, lr_h = max(1, lr_w), max(1, lr_h)
            lr = hr.resize((lr_w, lr_h), Image.BICUBIC)

        hr_tensor = self.to_tensor(hr)     # (3, hr_H, hr_W)
        lr_tensor = self.to_tensor(lr)     # (3, lr_H, lr_W)

        # 生成 HR 坐标网格和 cell
        hr_shape = (hr_tensor.shape[1], hr_tensor.shape[2])
        coord = make_coord(hr_shape)          # (hr_H*hr_W, 2)
        cell = make_cell(hr_shape)            # (hr_H*hr_W, 2)

        # 将 HR 图像展平为 RGB ground truth
        gt_rgb = hr_tensor.permute(1, 2, 0).reshape(-1, 3)  # (hr_H*hr_W, 3)

        if self.is_train and self.sample_q is not None:
            # 随机采样 sample_q 个坐标（训练时不用全部像素）
            n_total = coord.shape[0]
            if n_total > self.sample_q:
                indices = torch.randperm(n_total)[:self.sample_q]
                coord = coord[indices]
                cell = cell[indices]
                gt_rgb = gt_rgb[indices]

        return {
            'lr': lr_tensor,
            'coord': coord,
            'cell': cell,
            'gt_rgb': gt_rgb,
        }
