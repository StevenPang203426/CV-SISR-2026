"""
下载 Blind SR 实验所需的预训练模型到 pretrained/ 目录。

用法
----
    # 下载全部 5 个模型
    python -m src.common.download_pretrained

    # 只下载指定模型
    python -m src.common.download_pretrained --models RealESRGAN_x4plus BSRGAN

    # 使用代理
    python -m src.common.download_pretrained --proxy http://127.0.0.1:7890
"""

import argparse
import os
import sys
import hashlib
from urllib.request import urlretrieve, Request, urlopen
from urllib.error import URLError

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'pretrained')

# ---- 模型信息 ----
MODELS = {
    'RealESRGAN_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        'desc': 'Real-ESRGAN GAN版（通用真实图像修复 x4）',
        'size_mb': 66.9,
    },
    'RealESRNet_x4plus': {
        'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
        'desc': 'Real-ESRGAN PSNR版（L1 loss，无 GAN）',
        'size_mb': 66.9,
    },
    'BSRGAN': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth',
        'desc': 'BSRGAN GAN版（随机洗牌退化 x4）',
        'size_mb': 66.9,
    },
    'BSRNet': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth',
        'desc': 'BSRNet PSNR版（L1 loss，无 GAN）',
        'size_mb': 66.9,
    },
    'ESRGAN': {
        'url': 'https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth',
        'desc': 'ESRGAN 原版（仅 bicubic 退化 + GAN）',
        'size_mb': 66.9,
    },
}


def download_with_progress(url, dest, proxy=None):
    """带进度条的下载。"""
    if proxy:
        import urllib.request
        proxy_handler = urllib.request.ProxyHandler({
            'http': proxy, 'https': proxy
        })
        opener = urllib.request.build_opener(proxy_handler)
        urllib.request.install_opener(opener)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            mb_done = count * block_size / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            sys.stdout.write(f'\r  下载中: {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)')
        else:
            mb_done = count * block_size / 1024 / 1024
            sys.stdout.write(f'\r  下载中: {mb_done:.1f} MB')
        sys.stdout.flush()

    try:
        urlretrieve(url, dest, reporthook=reporthook)
        print()  # newline after progress
        return True
    except (URLError, OSError) as e:
        print(f'\n  下载失败: {e}')
        return False


def main():
    p = argparse.ArgumentParser(description='下载 Blind SR 预训练模型')
    p.add_argument('--models', nargs='+', default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help='要下载的模型名称（默认全部）')
    p.add_argument('--proxy', type=str, default=None,
                   help='HTTP 代理地址（如 http://127.0.0.1:7890）')
    p.add_argument('--force', action='store_true',
                   help='强制重新下载已存在的文件')
    args = p.parse_args()

    os.makedirs(PRETRAINED_DIR, exist_ok=True)

    total = len(args.models)
    success, skip, fail = 0, 0, 0

    print(f'下载目录: {PRETRAINED_DIR}')
    print(f'计划下载: {total} 个模型\n')

    for i, name in enumerate(args.models, 1):
        info = MODELS[name]
        dest = os.path.join(PRETRAINED_DIR, f'{name}.pth')

        print(f'[{i}/{total}] {name}')
        print(f'  {info["desc"]}')
        print(f'  URL: {info["url"]}')

        if os.path.exists(dest) and not args.force:
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f'  已存在 ({size_mb:.1f} MB)，跳过。用 --force 强制重新下载\n')
            skip += 1
            continue

        ok = download_with_progress(info['url'], dest, proxy=args.proxy)
        if ok and os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f'  完成: {dest} ({size_mb:.1f} MB)\n')
            success += 1
        else:
            fail += 1
            print(f'  失败！请手动下载:\n  {info["url"]}\n')

    print(f'--- 汇总 ---')
    print(f'成功: {success}  跳过: {skip}  失败: {fail}')

    if fail > 0:
        print(f'\n提示: 如果 GitHub 下载慢，可以：')
        print(f'  1. 使用代理: python -m src.common.download_pretrained --proxy http://127.0.0.1:7890')
        print(f'  2. 手动下载后放到 {PRETRAINED_DIR}/')
        print(f'  3. 使用镜像站（如 ghproxy.com）：')
        for name in args.models:
            url = MODELS[name]['url']
            print(f'     wget https://ghproxy.com/{url} -O pretrained/{name}.pth')


if __name__ == '__main__':
    main()
