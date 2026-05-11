"""
推理模块
=========

对任意输入图像执行超分辨率推理，支持单张图像或目录批量处理。
当图像过大时自动启用分块推理（tiling），避免显存溢出。
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

    当输入图像的像素总数超过 ``tile_threshold`` 时，自动切换为
    分块推理模式，将图像拆分为重叠的小块分别处理后拼接，避免 OOM。

    Parameters
    ----------
    model : torch.nn.Module
        已加载权重的模型。
    device : torch.device
        推理设备。
    scale : int
        超分辨率倍数。
    tile_size : int
        分块推理时每块的边长（像素），默认 256。
    tile_overlap : int
        相邻块之间的重叠像素数，默认 16，用于消除拼接缝隙。
    tile_threshold : int
        触发分块推理的像素总数阈值，默认 512×512。
        图像像素数 > 该值时自动启用 tiling。
    """

    def __init__(self, model, device, scale,
                 tile_size=256, tile_overlap=16, tile_threshold=512 * 512):
        self.model = model
        self.device = device
        self.scale = scale
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_threshold = tile_threshold
        self.to_tensor = T.ToTensor()
        self.to_img = T.ToPILImage()

    def _forward_tiled(self, x: torch.Tensor) -> torch.Tensor:
        """
        分块推理：将输入张量切为重叠小块，逐块送入模型，拼接输出。

        在重叠区域使用线性混合，消除块边界伪影。
        """
        _, c, h, w = x.shape
        sf = self.scale
        ts = self.tile_size
        ol = self.tile_overlap

        # 输出张量（在 CPU 上分配，节省显存）
        out = torch.zeros(1, c, h * sf, w * sf)
        weight = torch.zeros(1, 1, h * sf, w * sf)

        # 遍历所有块
        y = 0
        while y < h:
            x0 = 0
            # 确保最后一块不越界
            y_end = min(y + ts, h)
            if y_end == h and y_end - y < ts:
                y = max(0, h - ts)
                y_end = h
            while x0 < w:
                x_end = min(x0 + ts, w)
                if x_end == w and x_end - x0 < ts:
                    x0 = max(0, w - ts)
                    x_end = w

                # 取出当前块并推理
                tile_in = x[:, :, y:y_end, x0:x_end].to(self.device)
                tile_out = self.model(tile_in).cpu()

                # 映射到输出坐标
                oy, ox = y * sf, x0 * sf
                oh, ow = tile_out.shape[2], tile_out.shape[3]

                out[:, :, oy:oy + oh, ox:ox + ow] += tile_out
                weight[:, :, oy:oy + oh, ox:ox + ow] += 1.0

                if x_end == w:
                    break
                x0 += ts - ol

            if y_end == h:
                break
            y += ts - ol

        # 加权平均（重叠区域取均值）
        out /= weight.clamp(min=1.0)
        return out

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
                ).unsqueeze(0)
            else:
                x = self.to_tensor(img).unsqueeze(0)

            _, _, h, w = x.shape
            if h * w > self.tile_threshold:
                print(f'  Image {h}x{w} exceeds threshold, using tiled inference '
                      f'(tile={self.tile_size}, overlap={self.tile_overlap})')
                sr = self._forward_tiled(x)
            else:
                sr = self.model(x.to(self.device)).cpu()

            if output_is_file:
                save_path = out_base
            else:
                stem = os.path.basename(pth).rsplit('.', 1)[0]
                save_path = out_base / f'{stem}_x{self.scale}.png'

            self.to_img(sr.squeeze(0).clamp(0, 1)).save(save_path)
            print(f'Saved: {save_path}')
