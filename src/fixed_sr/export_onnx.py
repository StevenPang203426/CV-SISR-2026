"""
将 PyTorch 超分辨率模型导出为 ONNX 格式，供 Web 前端 (ONNX Runtime Web) 使用。

用法
----
    # 导出单个模型
    python -m src.fixed_sr.export_onnx --model espcn --scale 4 \
        --ckpt experiments/espcn_x4/best.pt \
        --output web/models/espcn_x4.onnx

    python -m src.fixed_sr.export_onnx --model fsrcnn --scale 4 \
        --ckpt experiments/fsrcnn_x4/best.pt \
        --output web/models/fsrcnn_x4.onnx
    # 批量导出所有可用的轻量模型
    python -m src.fixed_sr.export_onnx --all
"""

import argparse
import os
import sys

import torch

from src.fixed_sr.models import build_model
from src.fixed_sr.models.fsrcnn import FSRCNN
from src.common.checkpoint import load_checkpoint
from src.blind_sr.rrdbnet import RRDBNet


def _load_fsrcnn_pixelshuffle(model, ckpt_path, scale):
    """Load a deconv-trained FSRCNN checkpoint into a pixelshuffle model.

    Copies all shared layers directly.  For the upsample layer, converts
    ConvTranspose2d weights → Conv2d + PixelShuffle weights.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # Old checkpoints use 'deconv.*', new ones use 'upsample.*'
    has_deconv_keys = any(k.startswith('deconv') for k in state)

    ps_state = model.state_dict()

    for k, v in state.items():
        # Remap old key names
        mapped_k = k.replace('deconv.', 'upsample.') if has_deconv_keys else k

        if mapped_k.startswith('upsample'):
            continue  # handle separately below
        if mapped_k in ps_state:
            ps_state[mapped_k] = v

    # Convert upsample weights
    if has_deconv_keys:
        deconv_w = state['deconv.weight']  # [in_ch, out_ch, kH, kW]
        deconv_b = state['deconv.bias']    # [out_ch]
    else:
        deconv_w = state['upsample.weight']
        deconv_b = state['upsample.bias']

    in_ch, C, kH, kW = deconv_w.shape
    s = scale

    # ConvTranspose2d: [in_ch, C, kH, kW]  →  Conv2d: [C*s*s, in_ch, kH, kW]
    # Flip kernel (ConvTranspose uses cross-correlation with flipped kernel)
    flipped = deconv_w.flip(2, 3)      # [in_ch, C, kH, kW]
    transposed = flipped.permute(1, 0, 2, 3)  # [C, in_ch, kH, kW]

    # Expand for PixelShuffle: each output channel c maps to s*s sub-pixel channels
    conv_w = torch.zeros(C * s * s, in_ch, kH, kW)
    conv_b = torch.zeros(C * s * s)
    for c in range(C):
        for r in range(s):
            for q in range(s):
                idx = c * s * s + r * s + q
                conv_w[idx] = transposed[c]
                conv_b[idx] = deconv_b[c] / (s * s)

    ps_state['upsample.0.weight'] = conv_w
    ps_state['upsample.0.bias'] = conv_b

    model.load_state_dict(ps_state)
    print(f'[INFO] Loaded deconv checkpoint and converted to pixelshuffle')


def export_one(model_name: str, scale: int, ckpt_path: str, output_path: str,
               opset: int = 18, input_h: int = 64, input_w: int = 64):
    """导出单个模型为 ONNX。"""
    # 1. 构建模型并加载权重
    #    FSRCNN: 使用 pixelshuffle 模式（避免 ConvTranspose2d，ORT Web 不支持）
    extra_kwargs = {}
    if model_name == 'fsrcnn':
        extra_kwargs['upsample_mode'] = 'pixelshuffle'
        print(f'[INFO] FSRCNN: using pixelshuffle mode for Web compatibility')

    model = build_model(model_name, scale=scale, in_channels=3, **extra_kwargs)

    # FSRCNN: 判断 checkpoint 是否已经是 pixelshuffle 格式
    if model_name == 'fsrcnn':
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        if 'upsample.0.weight' in state:
            # checkpoint 本身就是 pixelshuffle 格式，直接加载
            load_checkpoint(model, ckpt_path, device='cpu')
            print(f'[INFO] Loaded pixelshuffle checkpoint directly')
        else:
            # deconv checkpoint → 转换为 pixelshuffle 权重
            _load_fsrcnn_pixelshuffle(model, ckpt_path, scale)
    else:
        load_checkpoint(model, ckpt_path, device='cpu')

    model.eval()

    # 2. 创建 dummy 输入
    dummy = torch.randn(1, 3, input_h, input_w)

    # 3. 导出（兼容 PyTorch 2.x dynamo 导出器）
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # PyTorch >=2.6 默认使用 dynamo 导出器，需要用 dynamo=False
    # 回退到 TorchScript 导出器以兼容 dynamic_axes + 低 opset
    export_kwargs = dict(
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
        },
    )

    # PyTorch >=2.6 支持 dynamo=False 参数
    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version >= (2, 6):
        export_kwargs['dynamo'] = False

    torch.onnx.export(model, dummy, output_path, **export_kwargs)

    # 4. 打印模型大小
    size_kb = os.path.getsize(output_path) / 1024
    print(f'[OK] Exported: {output_path}  ({size_kb:.1f} KB)')
    return output_path


# ============================================================
# RRDB 导出（BSRNet / BSRGAN / RealESRGAN 等）
# ============================================================

# pretrained 模型注册表
RRDB_MODELS = {
    'bsrnet':            {'path': 'pretrained/BSRNet.pth',            'scale': 4},
    'bsrgan':            {'path': 'pretrained/BSRGAN.pth',            'scale': 4},
    'realesrgan_x4plus': {'path': 'pretrained/RealESRGAN_x4plus.pth', 'scale': 4},
    'realesrnet_x4plus': {'path': 'pretrained/RealESRNet_x4plus.pth', 'scale': 4},
    'esrgan':            {'path': 'pretrained/ESRGAN.pth',            'scale': 4},
}


def export_rrdb(model_name: str, output_path: str,
                opset: int = 18, input_h: int = 64, input_w: int = 64):
    """导出 RRDB 预训练模型为 ONNX。"""
    info = RRDB_MODELS[model_name]
    ckpt_path = info['path']
    scale = info['scale']

    if not os.path.exists(ckpt_path):
        print(f'[SKIP] {ckpt_path} not found')
        return None

    # 构建并加载权重
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=scale)
    model.load_pretrained(ckpt_path)
    model.eval()
    print(f'[INFO] Loaded RRDB model: {model_name} from {ckpt_path}')

    # dummy 输入
    dummy = torch.randn(1, 3, input_h, input_w)

    # 导出
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    export_kwargs = dict(
        opset_version=opset,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
        },
    )

    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version >= (2, 6):
        export_kwargs['dynamo'] = False

    torch.onnx.export(model, dummy, output_path, **export_kwargs)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'[OK] Exported: {output_path}  ({size_mb:.1f} MB)')
    return output_path


# ============================================================
# 批量导出
# ============================================================

# 轻量模型列表（适合浏览器）
WEB_MODELS = ['espcn', 'fsrcnn']
SCALES = [2, 3, 4]


def export_all(opset: int):
    """批量导出所有可用模型（轻量 + RRDB pretrained）。"""
    exported, skipped = [], []

    # 轻量模型
    for model_name in WEB_MODELS:
        for scale in SCALES:
            ckpt = f'experiments/{model_name}_x{scale}/best.pt'
            if os.path.exists(ckpt):
                out = f'web/models/{model_name}_x{scale}.onnx'
                export_one(model_name, scale, ckpt, out, opset)
                exported.append(out)
            else:
                print(f'[SKIP] {ckpt} not found')
                skipped.append(ckpt)

    # RRDB pretrained 模型
    for name, info in RRDB_MODELS.items():
        if os.path.exists(info['path']):
            out = f'web/models/{name}_x{info["scale"]}.onnx'
            result = export_rrdb(name, out, opset)
            if result:
                exported.append(out)
            else:
                skipped.append(info['path'])
        else:
            print(f'[SKIP] {info["path"]} not found')
            skipped.append(info['path'])

    print(f'\nDone. Exported {len(exported)}, skipped {len(skipped)}.')


def main():
    p = argparse.ArgumentParser(description='Export PyTorch SR models to ONNX')
    p.add_argument('--model', type=str,
                   help='Model name: espcn/fsrcnn (轻量) 或 bsrnet/bsrgan/realesrgan_x4plus/realesrnet_x4plus/esrgan (RRDB)')
    p.add_argument('--scale', type=int, help='Super-resolution scale factor (轻量模型需要)')
    p.add_argument('--ckpt', type=str, help='Checkpoint path (轻量模型需要)')
    p.add_argument('--output', type=str, help='Output .onnx path')
    p.add_argument('--opset', type=int, default=18, help='ONNX opset version (default: 18)')
    p.add_argument('--all', action='store_true', help='Export all available models (轻量 + RRDB)')
    args = p.parse_args()

    if args.all:
        export_all(args.opset)
    elif args.model and args.model in RRDB_MODELS:
        # RRDB pretrained 模型：不需要 --scale/--ckpt
        info = RRDB_MODELS[args.model]
        output = args.output or f'web/models/{args.model}_x{info["scale"]}.onnx'
        export_rrdb(args.model, output, args.opset)
    else:
        if not all([args.model, args.scale, args.ckpt]):
            p.error('轻量模型需要 --model, --scale, --ckpt (--output 可选)')
        output = args.output or f'web/models/{args.model}_x{args.scale}.onnx'
        export_one(args.model, args.scale, args.ckpt, output, args.opset)


if __name__ == '__main__':
    main()
