"""
LIIF 完整模型 = Encoder + LIIF 解码器
======================================

通过 ``build_liif_model()`` 工厂函数构建，
不走 models/__init__.py 的 REGISTRY（构建逻辑和固定倍率模型本质不同）。
"""

import torch.nn as nn

from .edsr import EDSREncoder
from .liif import LIIF


class LIIFModel(nn.Module):
    """
    LIIF 完整模型。

    Parameters
    ----------
    encoder : nn.Module
        特征提取器，输出 (B, C, H, W) 特征图。
    liif : LIIF
        LIIF 解码器。
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
            查询坐标，值域 [-1, 1]。
        cell : Tensor (B, N, 2), optional
            每个查询点的 cell 大小。

        Returns
        -------
        Tensor (B, N, 3)
            预测的 RGB 值。
        """
        feat = self.encoder(lr)
        return self.liif.query(feat, coord, cell)


def build_liif_model(encoder_args=None, liif_args=None):
    """
    构建 LIIF 模型的工厂函数。

    Parameters
    ----------
    encoder_args : dict, optional
        EDSREncoder 参数，默认 {n_feats: 64, n_resblocks: 16}。
    liif_args : dict, optional
        LIIF 参数，默认 {feat_dim: 64, hidden_dim: 256}。

    Returns
    -------
    LIIFModel
    """
    enc_kw = dict(n_feats=64, n_resblocks=16)
    if encoder_args:
        enc_kw.update(encoder_args)

    liif_kw = dict(feat_dim=enc_kw['n_feats'], hidden_dim=256,
                   local_ensemble=True, feat_unfold=True, cell_decode=True)
    if liif_args:
        liif_kw.update(liif_args)

    # 确保 feat_dim 和 encoder 输出一致
    liif_kw['feat_dim'] = enc_kw['n_feats']

    encoder = EDSREncoder(**enc_kw)
    liif = LIIF(**liif_kw)
    return LIIFModel(encoder, liif)
