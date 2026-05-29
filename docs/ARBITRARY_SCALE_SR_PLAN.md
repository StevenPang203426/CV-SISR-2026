# 任意尺度超分辨率（Arbitrary-Scale SR）实施方案

> 引入隐式神经表示（LIIF），让单一模型支持 3.14x、5.5x 等无级连续放大，
> 突破传统模型只能固定 2x/3x/4x 的限制。

---

## 一、背景：为什么需要任意尺度

### 1.1 现有模型的局限

当前项目的 5 个模型（SRCNN / FSRCNN / ESPCN / EDSR / IMDN）都是**固定倍率**的：

```
训练 x2 模型 → 只能放大 2 倍
训练 x3 模型 → 只能放大 3 倍
训练 x4 模型 → 只能放大 4 倍
```

想放大 3.5 倍？要么训练一个新模型，要么先 x4 再缩回 3.5x（质量损失）。

### 1.2 LIIF 的核心思想

LIIF（Local Implicit Image Function，CVPR 2021 Oral）把图像看作一个**连续函数**：

```
传统 SR:  LR 图像 → 固定上采样网络 → 固定倍率的 HR 图像（离散像素网格）

LIIF SR:  LR 图像 → Encoder 提取特征网格 → 对任意连续坐标 (x,y) 查询 MLP
          → MLP 返回该坐标的 RGB 值 → 想要多大的图就查多少个坐标点
```

**关键洞察**：上采样倍率不再硬编码在网络结构中，而是由你**查询多少个坐标点**决定。
查询 128×128 个点就是 x2，查询 200×200 个点就是 x3.14，查询 320×320 个点就是 x5——同一个模型、同一组权重。

---

## 二、LIIF 架构详解

### 2.1 整体流程

```
LR 图像 (H×W×3)
    │
    ▼
┌──────────────┐
│   Encoder    │  ← EDSR-baseline（去掉上采样头）
│  提取特征网格 │     输出: H×W×C 的特征图（和 LR 同分辨率）
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              LIIF 解码器（核心创新）                    │
│                                                      │
│  对于目标 HR 图的每个像素位置 (x_q, y_q)：             │
│                                                      │
│  1. 坐标映射：把 HR 像素坐标映射到 [-1, 1] 连续空间    │
│  2. 特征查询：在 LR 特征网格上找最近的 4 个特征向量     │
│  3. 计算相对坐标：当前查询点相对于 4 个特征点的偏移      │
│  4. Cell Decoding：把目标像素的"面积"也编码进去          │
│  5. MLP 预测：将 [特征, 相对坐标, cell 大小] 输入 MLP   │
│     → 输出该坐标的 RGB 值                              │
│  6. 双线性加权：4 个邻近特征的预测结果加权平均           │
│                                                      │
└──────────────────────────────────────────────────────┘
       │
       ▼
HR 图像 (H*s × W*s × 3)    ← s 可以是任意正实数！
```

### 2.2 各组件解释

| 组件 | 作用 | 细节 |
|------|------|------|
| **Encoder** | 从 LR 图像提取深度特征 | LIIF 原文用 EDSR-baseline（16 ResBlock, 64 通道），去掉上采样头（Upsampler + 最后的 Conv），只保留 head + body，输出 H×W×64 的特征图 |
| **Feature Unfolding** | 扩展每个特征点的感受野 | 把每个位置的 3×3 邻域特征拼接起来：C → 9C。让 MLP 看到更大的局部上下文 |
| **坐标编码** | 告诉 MLP "你在哪" | 相对坐标 (Δx, Δy)，即查询点相对于最近特征点的偏移量 |
| **Cell Decoding** | 告诉 MLP "目标像素多大" | 编码目标分辨率下每个像素的 cell 大小 (cell_h, cell_w)。放大 2x 时 cell 大，放大 8x 时 cell 小，MLP 据此调整预测策略 |
| **MLP** | 预测 RGB | 5 层全连接网络 (256 hidden)：输入 [feature(9×64) + coord(2) + cell(2)] → 输出 RGB(3) |

### 2.3 训练策略

LIIF 训练时**每个 batch 随机采样不同的倍率**，这是它能泛化到任意倍率的关键：

```
for each training step:
    1. 从 DIV2K 取一张 HR 图，随机裁剪 patch
    2. 随机采样一个倍率 s ∈ [1, 4]（均匀分布）
    3. 将 HR patch 下采样 s 倍得到 LR
    4. 在 HR patch 上随机采样 N 个坐标点作为查询
    5. 模型预测这些坐标的 RGB 值
    6. 与 HR patch 的 ground truth 比较，计算 L1 Loss
```

---

## 三、集成到本项目的方案

### 3.1 新增文件结构

```
SISR-Team8/
├── models/
│   ├── __init__.py            ← 不修改（LIIF 不进 REGISTRY）
│   ├── edsr.py                ← 修改：新增 EDSREncoder 类
│   ├── liif.py                ← 新增：LIIF 解码器 + MLP
│   ├── liif_model.py          ← 新增：Encoder + LIIF 组合模型 + build_liif_model()
│   └── ...
│
├── data/
│   ├── dataset.py             ← 现有（固定倍率）
│   └── dataset_liif.py        ← 新增：随机倍率 + 坐标采样的 Dataset
│
├── configs/
│   └── liif_edsr_x1-4.yaml    ← 新增：LIIF 训练配置
│
└── scripts/
    ├── train_liif.py           ← 新增：LIIF 训练入口（独立脚本）
    ├── test_liif.py            ← 新增：LIIF 测试入口（任意倍率推理 + 多倍率评估）
    └── visualize_liif.py       ← 新增：多倍率对比可视化面板
```

### 3.2 为什么单独写 `train_liif.py` 而不复用 `train.py`

LIIF 的训练 pipeline 和固定倍率 SR 有本质区别：

| | 固定倍率 SR（现有 train.py） | LIIF（新 train_liif.py） |
|---|---|---|
| 数据集输出 | `{'lr': tensor, 'hr': tensor}` | `{'lr': tensor, 'coord': tensor, 'cell': tensor, 'gt_rgb': tensor}` |
| Loss 计算 | `criterion(model(lr), hr)` 像素对像素 | `criterion(model(lr, coord, cell), gt_rgb)` 坐标对 RGB |
| 倍率 | 固定，由模型结构决定 | 每 batch 随机采样 |
| 模型输入 | `model(lr)` | `model(lr, coord, cell)` |
| 验证 | 对整张图推理 | 对整张图的所有坐标推理（需要分块） |

差异太大，硬塞进现有 Trainer 会让代码变复杂。单独写更清晰。

---

## 四、核心代码设计

### 4.1 拆分 EDSR Encoder

从现有 `models/edsr.py` 中拆出 Encoder 部分（去掉上采样头）：

```python
# models/edsr.py —— 新增 EDSREncoder 类

class EDSREncoder(nn.Module):
    """EDSR 的特征提取部分（无上采样头），作为 LIIF 的 Encoder。"""
    
    def __init__(self, in_channels=3, n_feats=64, n_resblocks=16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)]
        )
        # 注意：没有 tail（Upsampler），输出和输入同分辨率
    
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return x + res  # 输出: (B, n_feats, H, W)
```

原有的 `EDSR` 类保持不变，不影响已有功能。

### 4.2 LIIF 解码器

```python
# models/liif.py

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
        
        # 5 层 MLP
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
            # 局部集成：对每个查询坐标，找 4 个最近的特征格点
            # 通过微小偏移生成 4 组坐标，分别查询后加权平均
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst = [0], [0]
            eps_shift = 0
        
        # 特征网格的格点间距
        rx = 2 / feat.shape[-2] / 2    # 半格距 x
        ry = 2 / feat.shape[-1] / 2    # 半格距 y
        
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
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False
                )[:, :, 0, :].permute(0, 2, 1)  # (B, N, C)
                
                # 相对坐标
                q_coord = F.grid_sample(
                    self._make_coord_grid(feat),
                    coord_.flip(-1).unsqueeze(1),
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
        total_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            # 面积越大的邻居，权重越小（反比）
            ret = 0
            for pred, area in zip(preds, areas):
                ret = ret + pred * (area / total_area).unsqueeze(-1)
            return ret
        else:
            return preds[0]
    
    @staticmethod
    def _make_coord_grid(feat):
        """生成特征网格的坐标图，值域 [-1, 1]。"""
        B, C, H, W = feat.shape
        # 生成 (H, W, 2) 的坐标网格
        coord_h = torch.linspace(-1 + 1/H, 1 - 1/H, H, device=feat.device)
        coord_w = torch.linspace(-1 + 1/W, 1 - 1/W, W, device=feat.device)
        grid = torch.stack(torch.meshgrid(coord_h, coord_w, indexing='ij'), dim=-1)
        # (1, 2, H, W)
        return grid.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
```

### 4.3 组合模型

```python
# models/liif_model.py

import torch
import torch.nn as nn


class LIIFModel(nn.Module):
    """
    LIIF 完整模型 = Encoder + LIIF 解码器。
    
    Parameters
    ----------
    encoder : nn.Module
        特征提取器，输出 (B, C, H, W) 特征图。
    liif : nn.Module
        LIIF 解码器，接收特征和坐标，返回 RGB。
    """
    
    def __init__(self, encoder, liif):
        super().__init__()
        self.encoder = encoder
        self.liif = liif
    
    def forward(self, lr, coord, cell=None):
        """
        Parameters
        ----------
        lr : Tensor (B, 3, H, W)
            低分辨率输入图像。
        coord : Tensor (B, N, 2)
            查询坐标。
        cell : Tensor (B, N, 2), optional
            cell 大小。
            
        Returns
        -------
        Tensor (B, N, 3)
        """
        feat = self.encoder(lr)             # (B, C, H, W)
        pred = self.liif.query(feat, coord, cell)  # (B, N, 3)
        return pred
```

### 4.4 LIIF 专用 Dataset

```python
# data/dataset_liif.py

"""
LIIF 训练数据集：随机倍率 + 坐标采样。

与固定倍率 SRDataset 的关键区别：
1. 每次随机采样一个倍率 s ∈ [scale_min, scale_max]
2. 输出的不是 HR 图像，而是坐标点 + 对应的 ground truth RGB
"""

import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def make_coord(shape, flatten=True):
    """生成 [-1, 1] 范围内的坐标网格。"""
    H, W = shape
    coord_h = torch.linspace(-1 + 1/H, 1 - 1/H, H)
    coord_w = torch.linspace(-1 + 1/W, 1 - 1/W, W)
    grid = torch.stack(torch.meshgrid(coord_h, coord_w, indexing='ij'), dim=-1)
    if flatten:
        grid = grid.view(-1, 2)   # (H*W, 2)
    return grid


def make_cell(coord, shape):
    """计算每个坐标点的 cell 大小。"""
    H, W = shape
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / H
    cell[:, 1] *= 2 / W
    return cell


class LIIFDataset(Dataset):
    """
    Parameters
    ----------
    hr_dir : str
        HR 图像目录。
    patch_size : int
        HR patch 大小（训练时裁剪）。
    scale_min, scale_max : float
        训练时随机采样的倍率范围。
    sample_q : int
        每张图随机采样的查询坐标数（训练时不查全部像素，太慢）。
    augment : bool
        是否数据增强。
    is_train : bool
        训练模式 vs 验证模式。
    """
    
    def __init__(self, hr_dir, patch_size=96, 
                 scale_min=1.0, scale_max=4.0,
                 sample_q=2304, augment=True, is_train=True):
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.hr_paths = sorted([
            os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
            if f.lower().endswith(exts)
        ])
        self.patch_size = patch_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.augment = augment
        self.is_train = is_train
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.hr_paths) * (4 if self.is_train else 1)
    
    def __getitem__(self, idx):
        img_idx = idx // 4 if self.is_train else idx
        hr = Image.open(self.hr_paths[img_idx]).convert('RGB')
        
        if self.is_train:
            # 随机采样倍率
            scale = random.uniform(self.scale_min, self.scale_max)
            
            # 确保 HR patch 能被 scale 整除（LR 尺寸为整数）
            w, h = hr.size
            ps = self.patch_size
            
            # 确保图片够大
            if w < ps or h < ps:
                hr = hr.resize((max(ps, w), max(ps, h)), Image.BICUBIC)
                w, h = hr.size
            
            # 随机裁剪
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
            lr = hr.resize((lr_w, lr_h), Image.BICUBIC)
            
        else:
            # 验证时用固定倍率（外部指定）
            scale = self.scale_max
            w, h = hr.size
            lr_w = round(w / scale)
            lr_h = round(h / scale)
            lr = hr.resize((lr_w, lr_h), Image.BICUBIC)
            hr_w, hr_h = w, h
        
        hr_tensor = self.to_tensor(hr)     # (3, hr_H, hr_W)
        lr_tensor = self.to_tensor(lr)     # (3, lr_H, lr_W)
        
        # 生成 HR 坐标网格
        coord = make_coord((hr_tensor.shape[1], hr_tensor.shape[2]))  # (hr_H*hr_W, 2)
        cell = make_cell(coord, (hr_tensor.shape[1], hr_tensor.shape[2]))
        
        # 将 HR 图像展平为 RGB ground truth
        gt_rgb = hr_tensor.permute(1, 2, 0).reshape(-1, 3)  # (hr_H*hr_W, 3)
        
        if self.is_train and self.sample_q is not None:
            # 随机采样 sample_q 个坐标（训练时不用全部像素）
            indices = torch.randperm(coord.shape[0])[:self.sample_q]
            coord = coord[indices]
            cell = cell[indices]
            gt_rgb = gt_rgb[indices]
        
        return {
            'lr': lr_tensor,
            'coord': coord,
            'cell': cell,
            'gt_rgb': gt_rgb,
        }
```

### 4.5 训练配置

```yaml
# configs/liif_edsr_x1-4.yaml

# ---- 模型 ----
model: liif
encoder: edsr
encoder_args:
  n_feats: 64
  n_resblocks: 16

liif_args:
  feat_dim: 64
  hidden_dim: 256
  local_ensemble: true
  feat_unfold: true
  cell_decode: true

# ---- 数据 ----
train_hr: data/DIV2K_train_HR
val_hr: data/DIV2K_valid_HR
patch_size: 48                  # HR patch 大小（LIIF 原文用 48）
scale_min: 1.0
scale_max: 4.0
sample_q: 2304                  # 每张图采样 48×48=2304 个坐标
val_scale: 4                    # 训练中验证只测 x4

# ---- 训练 ----
epochs: 500
batch_size: 16
lr: 0.0001
opt: adam
crit: l1                        # LIIF 用 L1 Loss
scheduler: multi_step           # MultiStepLR
milestones: [200, 400]
gamma: 0.5

# ---- 输出 ----
save_dir: experiments/liif_edsr
```

---

## 五、训练与测试流程

### 5.1 训练入口 `train_liif.py`（伪代码）

```python
def main():
    args = load_config()
    
    # 1. 构建模型
    encoder = EDSREncoder(**args.encoder_args)
    liif = LIIF(**args.liif_args)
    model = LIIFModel(encoder, liif).to(device)
    
    # 2. 构建数据集（注意：不是 SRDataset，是 LIIFDataset）
    train_set = LIIFDataset(args.train_hr, patch_size=args.patch_size,
                            scale_min=args.scale_min, scale_max=args.scale_max,
                            sample_q=args.sample_q)
    val_set = LIIFDataset(args.val_hr, is_train=False, scale_max=4.0)
    
    # 3. 训练循环
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            lr = batch['lr'].to(device)
            coord = batch['coord'].to(device)
            cell = batch['cell'].to(device)
            gt_rgb = batch['gt_rgb'].to(device)
            
            pred_rgb = model(lr, coord, cell)   # (B, N, 3)
            loss = F.l1_loss(pred_rgb, gt_rgb)
            
            loss.backward()
            optimizer.step()
        
        # 4. 验证（在固定倍率 x2, x3, x4 上分别评估 PSNR）
        for test_scale in [2, 3, 4]:
            psnr = evaluate_liif(model, val_set, test_scale)
            print(f'Epoch {epoch}, x{test_scale} PSNR: {psnr:.2f} dB')
```

### 5.2 测试入口 `test_liif.py`

```python
def test_arbitrary_scale(model, lr_image, scale):
    """
    用训练好的 LIIF 模型，以任意倍率放大一张图片。
    
    比如 scale=3.14：
    - LR: 100×100
    - 生成 314×314 的坐标网格
    - 查询每个坐标的 RGB
    - 得到 314×314 的 SR 图像
    """
    model.eval()
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
    
    _, _, h, w = lr_tensor.shape
    out_h, out_w = round(h * scale), round(w * scale)
    
    coord = make_coord((out_h, out_w)).unsqueeze(0).to(device)
    cell = make_cell(coord[0], (out_h, out_w)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(lr_tensor, coord, cell)    # (1, out_h*out_w, 3)
    
    sr = pred.view(1, out_h, out_w, 3).permute(0, 3, 1, 2)  # (1, 3, out_h, out_w)
    return to_pil_image(sr.squeeze(0).clamp(0, 1))
```

**用法示例**：

```bash
# 训练
python scripts/train_liif.py --config configs/liif_edsr_x1-4.yaml

# 固定倍率测试（对比现有模型）
python scripts/test_liif.py --ckpt experiments/liif_edsr/best.pt --scale 4 \
    --test_dir data/DIV2K/DIV2K_valid_HR

# 任意倍率测试（LIIF 的独特能力）
python scripts/test_liif.py --ckpt experiments/liif_edsr/best.pt --scale 3.14 \
    --input demo/original/10.png

# 多倍率批量评估
python scripts/test_liif.py --ckpt experiments/liif_edsr/best.pt \
    --scales 1.5 2 3.5 4 6 --test_dir data/DIV2K/DIV2K_valid_HR --save_images

# 多倍率对比可视化面板（HR | Bicubic | LIIF 并排对比）
python scripts/visualize_liif.py --ckpt experiments/liif_edsr/best.pt \
    --input demo/original/10.png --scales 1.5 2 3.5 4 6
```

---

## 六、实施计划

### EDSR Encoder + LIIF

| # | 任务 | 预计时间 | 产出 |
|---|------|----------|------|
| 1 | 在 `edsr.py` 中新增 `EDSREncoder` 类 | 10 min | 修改后的 edsr.py |
| 2 | 编写 `models/liif.py`（LIIF 解码器 + MLP） | 1 小时 | 核心解码器 |
| 3 | 编写 `models/liif_model.py`（Encoder + LIIF 组合 + 工厂函数） | 20 min | 完整模型 |
| 4 | 编写 `data/dataset_liif.py`（随机倍率 + 坐标采样） | 1 小时 | LIIF 数据集 |
| 5 | 编写 `train_liif.py`（独立训练入口） | 1 小时 | 训练入口 |
| 6 | 编写 `configs/liif_edsr_x1-4.yaml` | 10 min | 训练配置 |
| 7 | 在云服务器上训练（DIV2K, 500 epoch, ~5-8h） | 5-8 小时 | `experiments/liif_edsr/best.pt` |
| 8 | 编写 `test_liif.py`，在 x2/x4/x6/x1.5/x3.5 上评估 | 30 min | 测试入口 |

> **阶段 2（轻量 Encoder 替换）暂不实施**，待阶段 1 跑通后再决定。

---

## 七、关键注意事项

### 7.1 坐标系统

LIIF 使用 **[-1, 1] 归一化坐标**，和 PyTorch 的 `grid_sample` 一致：

```
(-1, -1) ─────────── (-1, 1)
    │                    │
    │    图像区域         │
    │                    │
 (1, -1) ─────────── (1, 1)
```

坐标不是像素索引，而是连续的浮点数。这是支持任意倍率的基础。

### 7.2 训练时的内存管理

LIIF 训练时**不对整张 HR 图做推理**（太大了），而是随机采样 `sample_q` 个坐标点。LIIF 原文用 `sample_q = 2304`（= 48×48），即每张图只查 2304 个点。这让 batch_size 可以设得较大（16 或 32）。

验证时需要查全部像素，采用**分块查询**策略：把坐标分成若干 batch（默认每次 30000 个点），循环推理后拼接。这对 DIV2K 验证集（100 张 2K 图）尤为重要——x4 放大后单张图的坐标数可达 800 万。

### 7.3 与现有项目的兼容性

| 组件 | 影响 |
|------|------|
| `models/__init__.py` | **不修改**。LIIF 不进 REGISTRY，使用独立的 `build_liif_model()` 工厂函数 |
| `core/trainer.py` | **不复用、不修改**。LIIF 训练循环逻辑不同，独立写 `train_liif.py` |
| `data/dataset.py` | **不影响**。LIIF 用独立的 `dataset_liif.py` |
| `models/edsr.py` | 新增 `EDSREncoder` 类，现有 `EDSR` 类零改动 |
| 现有 5 个模型 | **完全不受影响**。LIIF 是新增功能 |

### 7.4 预期性能参考

LIIF 原文在 DIV2K 训练、Set5 测试的 PSNR（EDSR-baseline Encoder）：

| 倍率 | LIIF (EDSR-baseline) | 传统固定倍率 EDSR |
|------|---------------------|------------------|
| x2 | ~34.7 dB | ~34.6 dB |
| x3 | ~30.8 dB | ~30.9 dB |
| x4 | ~28.9 dB | ~28.9 dB |
| x6 | ~26.5 dB | 无法测试（没训过 x6 模型） |
| x12 | ~23.7 dB | 无法测试 |

在整数倍率上 LIIF 和专用模型持平，但 LIIF 用**一个模型**覆盖了所有倍率。

---

## 八、设计决策记录

> 以下决策通过逐项讨论确定（2026-05-26）。

| # | 决策点 | 结论 | 理由 |
|---|--------|------|------|
| 1 | EDSREncoder 设计 | 独立类，复用 ResBlock，不继承 EDSR | EDSR 零改动，已有 checkpoint 不受影响 |
| 2 | 训练脚本 | 独立 `train_liif.py`，不复用 core/Trainer | 训练循环差异太大（坐标查询 vs 像素对像素），硬塞进 Trainer 有回归风险 |
| 3 | 模型注册 | 不进 REGISTRY，用 `build_liif_model()` | LIIF 构建需两步（Encoder + Decoder），和现有单类工厂模式不兼容 |
| 4 | 训练数据 | DIV2K（800 张） | T91 只有 91 张，对 5 层 256 维 MLP 来说数据量不足 |
| 5 | GPU / 训练时长 | 32G 显存，5-8 小时 | 单卡 32G 跑 500 epoch 约 4-6 小时 |
| 6 | 阶段范围 | 只做 EDSR Encoder | ESPCN 轻量替换待阶段 1 跑通后再决定 |
| 7 | ONNX / Web 部署 | 不纳入本次 | grid_sample / 动态 coord 对 ONNX 不友好，先跑通训练 |
| 8 | 验证集 | DIV2K 验证集（100 张），分块查询防 OOM | 比 Set5（5 张）更有统计意义 |
| 9 | 训练中验证倍率 | 只测 x4 | 控制每 epoch 验证时间 |
| 10 | 训练后测试倍率 | x2、x4、x6 + x1.5、x3.5 | 整数 + 分数倍率完整展示 LIIF 能力 |
| 11 | MLP 规格 | 5 层 256 维（沿用原文） | 经过充分验证的配置，改动收益不明确 |
| 12 | 学习率调度 | MultiStepLR，milestone [200, 400]，gamma=0.5 | 和 LIIF 原文一致，减少不确定性 |

---

## 九、实施进度

> 最后更新：2026-05-29

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | EDSREncoder 类 | 已完成 | `models/edsr.py` 中新增独立类 |
| 2 | LIIF 解码器 | 已完成 | `models/liif.py`，5 层 256 维 MLP |
| 3 | LIIFModel 组合 | 已完成 | `models/liif_model.py` + `build_liif_model()` |
| 4 | LIIF Dataset | 已完成 | `data/dataset_liif.py`，固定 LR 尺寸训练策略 |
| 5 | 训练入口 | 已完成 | `scripts/train_liif.py`，独立脚本 |
| 6 | 训练配置 | 已完成 | `configs/liif_edsr_x1-4.yaml` |
| 7 | 测试入口 | 已完成 | `scripts/test_liif.py`，支持 `--scales` 多倍率 |
| 8 | 可视化脚本 | 已完成 | `scripts/visualize_liif.py`，多倍率对比面板 |
| 9 | 云端训练 | 待执行 | 需在 GPU 服务器上运行 |

### 关键修复记录

- **DataLoader collate 错误**：随机倍率导致不同样本 LR 尺寸不同，无法 stack。修复：LR 固定为 `patch_size`，HR 计算为 `round(patch_size × scale)`。
- **PyTorch 2.6 weights_only 错误**：默认改为 `True`，checkpoint 含 numpy 类型时报错。修复：`torch.load(..., weights_only=False)`，影响 `test_liif.py`、`core/checkpoint.py`、`models/rrdbnet.py`。
- **脚本路径迁移**：`train_liif.py` 和 `test_liif.py` 从项目根目录移至 `scripts/`，添加 ROOT sys.path 处理。

---

## 十、参考资源

| 资源 | 链接 | 用途 |
|------|------|------|
| LIIF 论文 (CVPR 2021) | https://arxiv.org/abs/2012.09161 | 理论基础 |
| LIIF 官方代码 | https://github.com/yinboc/liif | 参考实现（主要参考 `models/liif.py`） |
| LIIF Project Page | https://yinboc.github.io/liif/ | 可视化演示 |
| A-LIIF（改进版） | https://arxiv.org/abs/2208.04318 | 进阶优化 |
| UltraSR | https://github.com/SHI-Labs/UltraSR-Arbitrary-Scale-Super-Resolution | 另一种实现 |
| Meta-SR (CVPR 2019) | https://github.com/XuecaiHu/Meta-SR-Pytorch | 早期任意倍率方案 |
| 任意尺度 SR 合集 | https://github.com/Weepingchestnut/Arbitrary-Scale-SR | 论文+代码汇总 |
