"""
RRDB (Residual in Residual Dense Block) 网络

移植自 BSRGAN / Real-ESRGAN，用于加载以下预训练模型：
  - RealESRGAN_x4plus / RealESRNet_x4plus
  - BSRGAN / BSRNet
  - ESRGAN

所有模型共享同一架构：RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)

参考:
  - Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image SR", ICCV 2021
  - Wang et al., "Real-ESRGAN: Training Real-World Blind SR with Pure Synthetic Data", ICCVW 2021
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """5-layer Residual Dense Block (growth channel = gc)."""

    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Residual scaling (0.1) 避免训练初期不稳定
        self._init_weights()

    def _init_weights(self):
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in')
            conv.weight.data *= 0.1
            if conv.bias is not None:
                conv.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block — 3 个 ResidualDenseBlock 级联。"""

    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB 网络 (ESRGAN / RealESRGAN / BSRGAN 共用架构)。

    Parameters
    ----------
    in_nc : int
        输入通道数 (默认 3)
    out_nc : int
        输出通道数 (默认 3)
    nf : int
        特征通道数 (默认 64)
    nb : int
        RRDB 块数量 (默认 23)
    gc : int
        Dense Block growth channel (默认 32)
    scale : int
        上采样倍数 (2 或 4，默认 4)
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4,
                 # 兼容原始参数名 sf
                 sf=None, in_channels=None):
        super().__init__()

        # 兼容不同调用方式
        if sf is not None:
            scale = sf
        if in_channels is not None:
            in_nc = in_channels

        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(
            *[RRDB(nf=nf, gc=gc) for _ in range(nb)]
        )
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # Upsampling: nearest interpolation + conv (×2 per stage)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.scale == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.body(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(
            F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale == 4:
            fea = self.lrelu(self.upconv2(
                F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return out

    def load_pretrained(self, path, strict=True):
        """加载预训练权重，自动处理 key 名称差异。"""
        state = torch.load(path, map_location='cpu')

        # 有些权重文件是直接的 state_dict，有些包在 dict 里
        if isinstance(state, dict) and 'params_ema' in state:
            state = state['params_ema']
        elif isinstance(state, dict) and 'params' in state:
            state = state['params']

        # 处理 key 名称映射：原始 BSRGAN 用 RRDB_trunk.X，我们用 body.X
        # 以及 HRconv → hr_conv
        mapped = {}
        for k, v in state.items():
            new_k = k
            # --- 主干网络 ---
            new_k = new_k.replace('RRDB_trunk.', 'body.')
            # --- 尾部卷积（不同权重文件有不同命名风格）---
            new_k = new_k.replace('conv_body.', 'trunk_conv.')
            new_k = new_k.replace('conv_up1.', 'upconv1.')
            new_k = new_k.replace('conv_up2.', 'upconv2.')
            new_k = new_k.replace('conv_hr.', 'hr_conv.')
            new_k = new_k.replace('HRconv.', 'hr_conv.')
            # --- Dense Block ---
            new_k = new_k.replace('.RDB1.', '.rdb1.')
            new_k = new_k.replace('.RDB2.', '.rdb2.')
            new_k = new_k.replace('.RDB3.', '.rdb3.')
            mapped[new_k] = v

        self.load_state_dict(mapped, strict=strict)
        return self
