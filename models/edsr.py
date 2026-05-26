import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.body(x) * 0.1

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats):
        m = []
        if scale in [2, 3]:
            m += [
                nn.Conv2d(n_feats, n_feats * (scale**2), 3, 1, 1),
                nn.PixelShuffle(scale)
            ]
        elif scale == 4:
            m += [
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            ]
        else:
            raise ValueError("scale must be 2/3/4")
        super().__init__(*m)

class EDSREncoder(nn.Module):
    """EDSR 特征提取器（无上采样头），用作 LIIF 的 Encoder。

    与 EDSR 共用 ResBlock，结构为 head + body，无 tail。
    输出和输入同分辨率: (B, n_feats, H, W)。

    Parameters
    ----------
    in_channels : int
        输入通道数 (默认 3)。
    n_feats : int
        特征通道数 (默认 64)。
    n_resblocks : int
        ResBlock 数量 (默认 16)。
    """

    def __init__(self, in_channels=3, n_feats=64, n_resblocks=16):
        super().__init__()
        self.out_dim = n_feats
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)]
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return x + res  # (B, n_feats, H, W)


class EDSR(nn.Module):
    def __init__(self, scale=2, in_channels=3, n_feats=64, n_resblocks=16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_resblocks)]
        )
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        return self.tail(x)
