import os
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.edsr import EDSR
from models.imdn import IMDN
from utils.img import imresize_bicubic

MODELS = {'srcnn': SRCNN, 'fsrcnn': FSRCNN, 'espcn': ESPCN, 'edsr': EDSR, 'imdn': IMDN}


def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
        return sorted([f for f in files if os.path.isfile(f)])
    raise FileNotFoundError(f'Input path not found: {input_path}')


def resolve_output_base(raw_output, model, scale):
    if not raw_output:
        return Path('output') / f'{model}_x{scale}' / 'infer'
    return Path(raw_output)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--input', required=True, help='Path to a single image or an image directory')
    p.add_argument('--output', default=None, help='Output directory, or output file path when input is a single image')
    p.add_argument('--model', required=True, choices=MODELS.keys())
    p.add_argument('--scale', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = {
        'srcnn': SRCNN(),
        'fsrcnn': FSRCNN(scale=args.scale),
        'espcn': ESPCN(scale=args.scale),
        'edsr': EDSR(scale=args.scale),
        'imdn': IMDN(scale=args.scale)
    }[args.model]
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
    model = model.to(device).eval()

    images = collect_images(args.input)
    single_input = len(images) == 1 and os.path.isfile(args.input)

    output_base = resolve_output_base(args.output, args.model, args.scale)
    output_is_file = bool(output_base.suffix)
    if output_is_file and not single_input:
        raise ValueError('--output points to a file, but --input is a directory. Please pass an output directory.')

    if output_is_file:
        output_base.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_base.mkdir(parents=True, exist_ok=True)

    to_tensor, to_img = T.ToTensor(), T.ToPILImage()

    for pth in images:
        if not pth.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            continue

        img = Image.open(pth).convert('RGB')
        lr = img
        x = (
            to_tensor(imresize_bicubic(lr, args.scale, down=False)).unsqueeze(0).to(device)
            if isinstance(model, SRCNN)
            else to_tensor(lr).unsqueeze(0).to(device)
        )
        sr = model(x)

        if output_is_file:
            save_path = output_base
        else:
            stem = os.path.basename(pth).rsplit('.', 1)[0]
            save_path = output_base / f'{stem}_x{args.scale}.png'

        to_img(sr.squeeze(0).clamp(0, 1).cpu()).save(save_path)
        print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
