"""
推理模块
=========

对任意输入图像执行超分辨率推理，支持单张图像或目录批量处理。
"""

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from models.srcnn import SRCNN
from utils.img import imresize_bicubic


def _collect_images(input_path: str) -> list[str]:
    """收集输入路径下的所有图像文件。"""
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
               and f.lower().endswith(exts)
        ])
        return files
    raise FileNotFoundError(f'Input path not found: {input_path}')


class Inferencer:
    """
    推理器：对输入图像执行超分辨率。

    Parameters
    ----------
    model : torch.nn.Module
        已加载权重的模型。
    device : torch.device
        推理设备。
    scale : int
        超分辨率倍数。
    """

    def __init__(self, model, device, scale):
        self.model = model
        self.device = device
        self.scale = scale
        self.to_tensor = T.ToTensor()
        self.to_img = T.ToPILImage()

    @torch.no_grad()
    def run(self, input_path: str, output_path: str = None):
        """
        对单张或多张图像执行超分辨率推理并保存结果。

        Parameters
        ----------
        input_path : str
            输入图像路径或目录。
        output_path : str, optional
            输出路径。若为 None，默认保存到 ``experiments/<model>_x<scale>/infer/``。
        """
        images = _collect_images(input_path)
        single_input = len(images) == 1 and os.path.isfile(input_path)

        if output_path is None:
            out_base = Path('experiments') / 'infer'
        else:
            out_base = Path(output_path)

        output_is_file = bool(out_base.suffix)
        if output_is_file and not single_input:
            raise ValueError('--output points to a file, but --input is a directory.')

        if output_is_file:
            out_base.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_base.mkdir(parents=True, exist_ok=True)

        is_srcnn = isinstance(self.model, SRCNN)

        for pth in images:
            img = Image.open(pth).convert('RGB')

            if is_srcnn:
                x = self.to_tensor(
                    imresize_bicubic(img, self.scale, down=False)
                ).unsqueeze(0).to(self.device)
            else:
                x = self.to_tensor(img).unsqueeze(0).to(self.device)

            sr = self.model(x)

            if output_is_file:
                save_path = out_base
            else:
                stem = os.path.basename(pth).rsplit('.', 1)[0]
                save_path = out_base / f'{stem}_x{self.scale}.png'

            self.to_img(sr.squeeze(0).clamp(0, 1).cpu()).save(save_path)
            print(f'Saved: {save_path}')
