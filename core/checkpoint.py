"""
检查点管理模块
===============

统一的模型保存与加载逻辑，支持保存额外元数据（epoch、指标等）。
"""

import os
import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, path: str, **extra):
    """
    保存模型检查点。

    Parameters
    ----------
    model : nn.Module
        要保存的模型。
    path : str
        保存路径。
    **extra
        附加到检查点字典的额外信息（如 epoch、best_psnr 等）。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {'model': model.state_dict()}
    state.update(extra)
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str, device='cpu'):
    """
    从检查点加载模型权重。

    Parameters
    ----------
    model : nn.Module
        目标模型（结构需与检查点匹配）。
    path : str
        检查点文件路径。
    device : str or torch.device
        加载到的设备。

    Returns
    -------
    dict
        完整的检查点字典（包含 'model' 键和其他元数据）。
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    return state
