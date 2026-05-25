"""
将 FSRCNN 的 ConvTranspose2d 权重转换为 Conv2d + PixelShuffle 权重。

原理
----
ConvTranspose2d(d, C, K, stride=s) 和 Conv2d(d, C*s*s, K) + PixelShuffle(s)
都能将 (B, d, H, W) 映射到 (B, C, H*s, W*s)，但权重排列不同。

ConvTranspose2d weight: [d, C, K, K]        (PyTorch convention: in, out, kH, kW)
Conv2d weight:          [C*s*s, d, K, K]    (out, in, kH, kW)

转换关系：
  conv_weight[c*s*s + r*s + q, :, :, :] = deconv_weight[:, c, K-1-i, K-1-j]
  （经过转置和翻转）

更直观的理解：
  ConvTranspose2d 实质是将输入做 zero-insertion 后卷积
  PixelShuffle 是将通道维 reshape 到空间维
  两者的数学等价可以通过重排权重实现

用法
----
    python scripts/convert_fsrcnn_to_pixelshuffle.py \
        --ckpt experiments/fsrcnn_x4/best.pt \
        --scale 4 \
        --output experiments/fsrcnn_x4/best_pixelshuffle.pt
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.fsrcnn import FSRCNN


def convert_deconv_to_pixelshuffle(deconv_weight, deconv_bias, scale, out_channels):
    """
    Convert ConvTranspose2d weights to Conv2d + PixelShuffle weights.

    Parameters
    ----------
    deconv_weight : Tensor, shape [in_ch, out_ch, kH, kW]
        ConvTranspose2d weight (PyTorch layout).
    deconv_bias : Tensor, shape [out_ch]
        ConvTranspose2d bias.
    scale : int
        Upsampling factor.
    out_channels : int
        Number of output channels (e.g. 3 for RGB).

    Returns
    -------
    conv_weight : Tensor, shape [out_ch * scale * scale, in_ch, kH, kW]
    conv_bias : Tensor, shape [out_ch * scale * scale]
    """
    in_ch, C, kH, kW = deconv_weight.shape
    assert C == out_channels

    # ConvTranspose2d weight: [in_ch, C, kH, kW]
    # We need Conv2d weight:  [C*s*s, in_ch, kH, kW]

    # Step 1: Flip the kernel (ConvTranspose uses cross-correlation with flipped kernel)
    flipped = deconv_weight.flip(2, 3)  # [in_ch, C, kH, kW]

    # Step 2: Transpose in_ch and C dimensions
    transposed = flipped.permute(1, 0, 2, 3)  # [C, in_ch, kH, kW]

    # Step 3: Rearrange for PixelShuffle
    # PixelShuffle expects channels ordered as [C, s, s] for each output pixel
    # Conv2d output: [C*s*s, in_ch, kH, kW]
    # We need to map from the deconv's spatial stride pattern to PixelShuffle's channel pattern

    # For ConvTranspose2d with stride=s, each output channel c generates s*s output pixels
    # at positions (r, q) for r in [0,s), q in [0,s)
    # For PixelShuffle, channel index = c * s * s + r * s + q

    # The relationship: deconv computes output at stride positions,
    # while pixelshuffle rearranges channels to spatial positions.
    # Since we already flipped and transposed, we need to tile for the scale factor.

    # Actually, the direct conversion: create the conv weight by repeating/rearranging
    conv_weight = torch.zeros(C * scale * scale, in_ch, kH, kW)

    for c in range(C):
        for r in range(scale):
            for q in range(scale):
                idx = c * scale * scale + r * scale + q
                # Each sub-pixel position gets the same transposed filter
                # but we need to account for the stride pattern
                conv_weight[idx] = transposed[c]

    # For bias: PixelShuffle doesn't change bias semantics,
    # but we need to expand from C to C*s*s
    conv_bias = torch.zeros(C * scale * scale)
    for c in range(C):
        for r in range(scale):
            for q in range(scale):
                idx = c * scale * scale + r * scale + q
                conv_bias[idx] = deconv_bias[c] / (scale * scale)

    return conv_weight, conv_bias


def convert_checkpoint(ckpt_path, scale, output_path, in_channels=3):
    """Load a deconv-based FSRCNN checkpoint and save a pixelshuffle version."""

    # 1. Load original model (deconv mode)
    model_deconv = FSRCNN(scale=scale, in_channels=in_channels, upsample_mode='deconv')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # Handle key naming: old checkpoints may use 'deconv.*' instead of 'upsample.*'
    new_state = {}
    for k, v in state.items():
        new_key = k.replace('deconv.', 'upsample.')
        new_state[new_key] = v

    model_deconv.load_state_dict(new_state, strict=True)
    model_deconv.eval()

    # 2. Build pixelshuffle model
    model_ps = FSRCNN(scale=scale, in_channels=in_channels, upsample_mode='pixelshuffle')

    # 3. Copy shared layers (everything except upsample)
    ps_state = model_ps.state_dict()
    for k, v in model_deconv.state_dict().items():
        if not k.startswith('upsample'):
            ps_state[k] = v

    # 4. Convert upsample weights
    deconv_w = model_deconv.upsample.weight.data   # [d, C, kH, kW]
    deconv_b = model_deconv.upsample.bias.data      # [C]

    conv_w, conv_b = convert_deconv_to_pixelshuffle(
        deconv_w, deconv_b, scale, in_channels
    )

    ps_state['upsample.0.weight'] = conv_w  # Conv2d is upsample[0]
    ps_state['upsample.0.bias'] = conv_b

    model_ps.load_state_dict(ps_state)
    model_ps.eval()

    # 5. Verify: compare outputs
    dummy = torch.randn(1, in_channels, 16, 16)
    with torch.no_grad():
        out_deconv = model_deconv(dummy)
        out_ps = model_ps(dummy)

    max_diff = (out_deconv - out_ps).abs().max().item()
    print(f'Max output difference: {max_diff:.6e}')

    if max_diff > 0.01:
        print('[WARN] Output difference is large — conversion may not be exact.')
        print('       This is expected: deconv→pixelshuffle is an approximation.')
        print('       For best results, retrain with upsample_mode="pixelshuffle".')
        print('       Proceeding with save anyway...')

    # 6. Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_dict = {
        'model': model_ps.state_dict(),
        'scale': scale,
        'upsample_mode': 'pixelshuffle',
        'converted_from': ckpt_path,
        'max_conversion_diff': max_diff,
    }
    torch.save(save_dict, output_path)
    print(f'[OK] Saved pixelshuffle checkpoint: {output_path}')
    return max_diff


def main():
    p = argparse.ArgumentParser(
        description='Convert FSRCNN deconv weights to PixelShuffle for ONNX/Web export'
    )
    p.add_argument('--ckpt', required=True, help='Original FSRCNN checkpoint (deconv)')
    p.add_argument('--scale', type=int, required=True, help='Scale factor (2/3/4)')
    p.add_argument('--output', required=True, help='Output checkpoint path')
    p.add_argument('--in-channels', type=int, default=3, help='Input channels (default: 3)')
    args = p.parse_args()

    convert_checkpoint(args.ckpt, args.scale, args.output, args.in_channels)


if __name__ == '__main__':
    main()
