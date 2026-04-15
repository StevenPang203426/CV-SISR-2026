import os
import argparse
import time
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from data.dataset import SRDataset
from utils.metrics import psnr_y, ssim_y
from utils.profile import count_params_m, profile_flops_g
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.edsr import EDSR
from models.imdn import IMDN

MODELS = {'srcnn': SRCNN, 'fsrcnn': FSRCNN, 'espcn': ESPCN, 'edsr': EDSR, 'imdn': IMDN}

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--model', required=True, choices=MODELS.keys())
    p.add_argument('--scale', type=int, default=2)
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--out_dir', default=None, help='Directory for SR/LR outputs and metrics.json')
    p.add_argument('--json', default=None, help='Optional custom path for metrics json file')
    args = p.parse_args()

    dataset_name = os.path.basename(os.path.abspath(args.test_dir))
    out_dir = Path(args.out_dir) if args.out_dir else Path('output') / f"{args.model}_x{args.scale}" / 'test' / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_set = SRDataset(args.model, args.test_dir, scale=args.scale, is_train=False)
    loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = {
        'srcnn': SRCNN(in_channels=3),
        'fsrcnn': FSRCNN(in_channels=3, scale=args.scale),
        'espcn': ESPCN(in_channels=3, scale=args.scale),
        'edsr': EDSR(in_channels=3, scale=args.scale),
        'imdn': IMDN(in_channels=3, scale=args.scale)
    }[args.model]
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state['model'])
    model = model.to(device).eval()

    example = next(iter(loader))
    lr = example['lr']
    inp_size = (1,) + (lr.shape[1:])
    params = count_params_m(model)
    flops_g = profile_flops_g(model, inp_size)

    psnrs, ssims, n, t = [], [], 0, 0.0
    image_metrics = []

    for i, b in enumerate(loader, start=1):
        lr, hr = b['lr'].to(device), b['hr'].to(device)
        st = time.time()
        sr = model(lr)
        t += time.time() - st
        n += 1
        to_uint8 = lambda x: (x.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
        sr_img, hr_img = Image.fromarray(to_uint8(sr[0])), Image.fromarray(to_uint8(hr[0]))
        lr_img = Image.fromarray(to_uint8(lr[0]))
        psnr = psnr_y(sr_img, hr_img, shave=args.scale)
        psnrs.append(psnr)
        ssim = ssim_y(sr_img, hr_img, shave=args.scale)
        ssims.append(ssim)
        image_metrics.append({
            "image_index": i,
            "psnr_y": psnr,
            "ssim_y": ssim
        })
        if args.save_images:
            sr_img.save(out_dir / f'{i:02d}_SR.png')
            lr_img.save(out_dir / f'{i:02d}_LR.png')

    summary = {
        "model": args.model,
        "scale": args.scale,
        "dataset": dataset_name,
        "num_images": n,
        "psnr_y": sum(psnrs) / len(psnrs),
        "ssim_y": sum(ssims) / len(ssims),
        "params": params,
        "flops_G": round(flops_g, 4) if flops_g >= 0 else None,
        "fps": round(n / max(t, 1e-9), 3),
        "image_metrics": image_metrics
    }
    print(
        f"Avg PSNR(Y): {summary['psnr_y']:.2f} dB | SSIM(Y): {summary['ssim_y']} | "
        f"Params: {summary['params']} | FLOPs: {summary['flops_G']}G | FPS: {summary['fps']:.2f}"
    )

    json_path = Path(args.json) if args.json else out_dir / 'metrics.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics saved to {json_path}")


if __name__ == '__main__':
    main()
