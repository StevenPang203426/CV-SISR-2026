"""
LIIF 解码器 — Local Implicit Image Function
=============================================

核心创新：把图像看作连续函数，通过 MLP 对任意坐标查询 RGB 值。
配合 Encoder 提取的特征网格使用。

参考: Chen et al., "Learning Continuous Image Representation
       with Local Implicit Image Function", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIIF(nn.Module):
    """
    Local Implicit Image Function 解码器。

    接收 Encoder 输出的特征网格，对任意连续坐标查询 RGB 值。

    Parameters
    ----------
    feat_dim : int
        Encoder 输出的特征通道数（如 EDSR 的 64）。
    hidden_dim : int
        MLP 隐藏层维度，默认 256。
    local_ensemble : bool
        是否使用局部集成（4 邻域加权），默认 True。
    feat_unfold : bool
        是否使用 Feature Unfolding（3×3 展开），默认 True。
    cell_decode : bool
        是否将 cell 大小编码输入 MLP，默认 True。
    """

    def __init__(self, feat_dim=64, hidden_dim=256,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        # 计算 MLP 输入维度
        imnet_in_dim = feat_dim
        if feat_unfold:
            imnet_in_dim *= 9          # 3×3 邻域展开
        imnet_in_dim += 2              # 相对坐标 (Δx, Δy)
        if cell_decode:
            imnet_in_dim += 2          # cell 大小 (cell_h, cell_w)

        # 5 层 MLP (256 hidden)
        self.imnet = nn.Sequential(
            nn.Linear(imnet_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),   # 输出 RGB
        )

    def query(self, feat, coord, cell=None):
        """
        对特征网格在指定坐标处查询 RGB 值。

        Parameters
        ----------
        feat : Tensor (B, C, H, W)
            Encoder 输出的特征网格。
        coord : Tensor (B, N, 2)
            查询坐标，值域 [-1, 1]，N 为查询点数。
        cell : Tensor (B, N, 2), optional
            每个查询点的 cell 大小。

        Returns
        -------
        Tensor (B, N, 3)
            每个查询坐标的 RGB 预测值。
        """
        B, C, H, W = feat.shape

        # Feature Unfolding: 3×3 展开
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1) \
                    .view(B, C * 9, H, W)       # (B, 9C, H, W)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst = [0], [0]
            eps_shift = 0

        # 特征网格的格点间距
        rx = 2 / feat.shape[-2] / 2    # 半格距 x
        ry = 2 / feat.shape[-1] / 2    # 半格距 y

        # 生成特征网格的坐标图
        feat_coord = _make_coord_grid(feat.shape[-2:], device=feat.device,
                                       flatten=False)  # (H, W, 2)
        feat_coord = feat_coord.permute(2, 0, 1) \
                               .unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                # 偏移坐标
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # grid_sample 查询最近特征
                # coord 格式: (B, N, 2) → (B, 1, N, 2) for grid_sample
                # grid_sample 期望的坐标顺序是 (x, y) 即 (W, H)，需要 flip
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)  # (B, N, feat_C)

                # 相对坐标
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)  # (B, N, 2)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # 拼接 MLP 输入
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                pred = self.imnet(inp)          # (B, N, 3)
                preds.append(pred)

                # 计算面积权重（用于局部集成加权）
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        # 加权平均
        if self.local_ensemble:
            total_area = torch.stack(areas).sum(dim=0)
            ret = 0
            for pred, area in zip(preds, areas):
                ret = ret + pred * (area / total_area).unsqueeze(-1)
            return ret
        else:
            return preds[0]


def _make_coord_grid(shape, device='cpu', flatten=True):
    """
    生成 [-1, 1] 范围内的坐标网格。

    Parameters
    ----------
    shape : tuple (H, W)
    device : str or torch.device
    flatten : bool
        True → (H*W, 2), False → (H, W, 2)

    Returns
    -------
    Tensor
    """
    H, W = shape
    coord_h = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=device)
    coord_w = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=device)
    grid = torch.stack(
        torch.meshgrid(coord_h, coord_w, indexing='ij'), dim=-1
    )  # (H, W, 2)
    if flatten:
        grid = grid.view(-1, 2)
    return grid


def make_cell(shape, device='cpu', flatten=True):
    """
    计算每个坐标点的 cell 大小。

    Parameters
    ----------
    shape : tuple (H, W)
    device : str or torch.device
    flatten : bool

    Returns
    -------
    Tensor — same shape as coord grid, each entry is (2/H, 2/W)
    """
    H, W = shape
    cell = torch.ones(H, W, 2, device=device)
    cell[:, :, 0] *= 2 / H
    cell[:, :, 1] *= 2 / W
    if flatten:
        cell = cell.view(-1, 2)
    return cell
