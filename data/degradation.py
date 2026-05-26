"""
退化管道 — 移植自 BSRGAN 并封装为可配置的接口。

核心来源: features/BSRGAN/utils/utils_blindsr.py (Zhang et al., ICCV 2021)
本模块移除了对 BSRGAN 内部 utils 的依赖，改为自包含实现。

用法
----
    from data.degradation import DegradationPipeline

    pipeline = DegradationPipeline(sf=4, mode='bsrgan')
    lr, hr = pipeline(hr_image)   # hr_image: HxWxC numpy, [0,1]
"""

import random

import cv2
import numpy as np
from scipy import ndimage
from scipy.linalg import orth
import scipy.stats as ss


# ============================================================
# 基础工具函数
# ============================================================

def uint2single(img):
    """uint8 [0,255] → float32 [0,1]"""
    return np.float32(img / 255.)


def single2uint(img):
    """float32 [0,1] → uint8 [0,255]"""
    return np.uint8(np.clip(img * 255., 0, 255).round())


def modcrop(img, sf):
    """裁剪图像使尺寸能被 sf 整除。"""
    h, w = img.shape[:2]
    return img[:h - h % sf, :w - w % sf, ...]


# ============================================================
# 模糊核生成
# ============================================================

def gm_blur_kernel(mean, cov, size=15):
    """生成多元高斯模糊核。"""
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy, cx = y - center + 1, x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    return k / np.sum(k)


def anisotropic_gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """生成各向异性高斯模糊核。"""
    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]),
        np.array([1., 0.])
    )
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)


def fspecial_gaussian(hsize, sigma):
    """MATLAB-style Gaussian kernel。"""
    siz = [(hsize - 1.0) / 2.0, (hsize - 1.0) / 2.0]
    x, y = np.meshgrid(
        np.arange(-siz[1], siz[1] + 1),
        np.arange(-siz[0], siz[0] + 1)
    )
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def shift_pixel(x, sf, upper_left=True):
    """对模糊核做像素偏移（用于对齐下采样网格）。"""
    from scipy.interpolate import interp2d
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    x1 = xv + shift if upper_left else xv - shift
    y1 = yv + shift if upper_left else yv - shift
    x1, y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    elif x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)
    return x


# ============================================================
# 单步退化操作
# ============================================================

def add_blur(img, sf=4):
    """添加随机模糊（各向异性高斯或各向同性高斯）。"""
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf
    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_gaussian(
            ksize=2 * random.randint(2, 11) + 3,
            theta=random.random() * np.pi, l1=l1, l2=l2
        )
    else:
        k = fspecial_gaussian(
            2 * random.randint(2, 11) + 3,
            wd * random.random()
        )
    img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_gaussian_noise(img, noise_level1=2, noise_level2=25):
    """添加高斯噪声（彩色 / 灰度 / 相关噪声，随机选择）。"""
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:
        img += np.random.normal(0, noise_level / 255., img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += np.random.normal(0, noise_level / 255., (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal(
            [0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]
        ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_jpeg_noise(img, quality_range=(30, 95)):
    """添加 JPEG 压缩噪声。"""
    qf = random.randint(*quality_range)
    img_bgr = cv2.cvtColor(single2uint(img), cv2.COLOR_RGB2BGR)
    _, encimg = cv2.imencode('.jpg', img_bgr,
                             [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    img_bgr = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(uint2single(img_bgr), cv2.COLOR_BGR2RGB)


def add_poisson_noise(img):
    """添加泊松噪声。"""
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10 ** (2 * random.random() + 2.0)
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        gray = np.clip((gray * 255.).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(gray * vals).astype(np.float32) / vals - gray
        img += noise_gray[:, :, np.newaxis]
    return np.clip(img, 0.0, 1.0)


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    """添加散斑噪声。"""
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img * np.random.normal(0, noise_level / 255., img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img * np.random.normal(0, noise_level / 255., (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img * np.random.multivariate_normal(
            [0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]
        ).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_resize(img, sf=4):
    """随机 resize（上 / 下 / 不变）。"""
    rnum = np.random.rand()
    if rnum > 0.8:
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])),
                     interpolation=random.choice([1, 2, 3]))
    return np.clip(img, 0.0, 1.0)


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    """从 LR-HR 对中随机裁剪 patch。"""
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h - lq_patchsize)
    rnd_w = random.randint(0, w - lq_patchsize)
    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]
    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf,
            rnd_w_H:rnd_w_H + lq_patchsize * sf, :]
    return lq, hq


# ============================================================
# 退化管道
# ============================================================

def degradation_bsrgan(img, sf=4, lq_patchsize=72):
    """
    BSRGAN 退化管道：7 种退化操作随机洗牌后依次执行。

    Parameters
    ----------
    img : ndarray, HxWxC, float32, [0,1]
        HR 图像，尺寸需 >= lq_patchsize*sf
    sf : int
        下采样倍数
    lq_patchsize : int
        输出 LR patch 大小

    Returns
    -------
    (lq, hq) : tuple of ndarray
        lq: lq_patchsize x lq_patchsize x C, [0,1]
        hq: lq_patchsize*sf x lq_patchsize*sf x C, [0,1]
    """
    jpeg_prob, scale2_prob = 0.9, 0.25
    sf_ori = sf

    img = modcrop(img.copy(), sf)
    h, w = img.shape[:2]
    if h < lq_patchsize * sf or w < lq_patchsize * sf:
        raise ValueError(f'Image size ({h}x{w}) too small for patch {lq_patchsize}x{sf}')

    hq = img.copy()

    # 25% 概率先做一次 x2 下采样
    if sf == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                             interpolation=random.choice([1, 2, 3]))
        else:
            # bicubic resize
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                             interpolation=cv2.INTER_CUBIC)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    # 7 种退化操作随机排列
    shuffle_order = random.sample(range(7), 7)
    # 约束: downsample3 (idx=3) 必须在 downsample2 (idx=2) 之后
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_blur(img, sf=sf)
        elif i == 2:
            # downsample2
            a, b = img.shape[1], img.shape[0]
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                img = cv2.resize(img, (int(img.shape[1] / sf1), int(img.shape[0] / sf1)),
                                 interpolation=random.choice([1, 2, 3]))
            else:
                k = fspecial_gaussian(25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()
                img = ndimage.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]
            img = np.clip(img, 0.0, 1.0)
        elif i == 3:
            # downsample3 — 最终下采样到目标尺寸
            img = cv2.resize(img, (int(a / sf), int(b / sf)),
                             interpolation=random.choice([1, 2, 3]))
            img = np.clip(img, 0.0, 1.0)
        elif i == 4:
            img = add_gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 5:
            if random.random() < jpeg_prob:
                img = add_jpeg_noise(img)
        elif i == 6:
            pass  # ISP model placeholder (需要预训练 ISP 模型，跳过)

    # 最终 JPEG 压缩
    img = add_jpeg_noise(img)

    # 随机裁剪
    img, hq = random_crop(img, hq, sf_ori, lq_patchsize)
    return img, hq


def degradation_bicubic(img, sf=4):
    """纯 Bicubic 下采样（基线退化）。

    Parameters
    ----------
    img : ndarray, HxWxC, [0,1]
    sf : int

    Returns
    -------
    lr : ndarray, H/sf x W/sf x C, [0,1]
    """
    img = modcrop(img, sf)
    h, w = img.shape[:2]
    lr = cv2.resize(img, (w // sf, h // sf), interpolation=cv2.INTER_CUBIC)
    return np.clip(lr, 0.0, 1.0)


def degradation_custom(img, sf=4, blur_sigma=0, noise_level=0, jpeg_quality=0):
    """自定义退化：可单独控制每种退化因素（用于消融实验）。

    Parameters
    ----------
    img : ndarray, HxWxC, [0,1]
    sf : int
    blur_sigma : float, 0 = 不加模糊
    noise_level : int, 0 = 不加噪声，否则为噪声 sigma (1-50)
    jpeg_quality : int, 0 = 不加 JPEG, 否则为质量因子 (1-100)

    Returns
    -------
    lr : ndarray
    """
    img = modcrop(img.copy(), sf)

    # 模糊
    if blur_sigma > 0:
        ksize = int(np.ceil(blur_sigma * 6)) | 1  # 确保为奇数
        ksize = max(ksize, 3)
        k = fspecial_gaussian(ksize, blur_sigma)
        img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    # 下采样
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // sf, h // sf), interpolation=cv2.INTER_CUBIC)

    # 噪声
    if noise_level > 0:
        img += np.random.normal(0, noise_level / 255., img.shape).astype(np.float32)

    # JPEG
    if 0 < jpeg_quality < 100:
        img_bgr = cv2.cvtColor(single2uint(np.clip(img, 0, 1)), cv2.COLOR_RGB2BGR)
        _, encimg = cv2.imencode('.jpg', img_bgr,
                                 [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        img_bgr = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(uint2single(img_bgr), cv2.COLOR_BGR2RGB)

    return np.clip(img, 0.0, 1.0)


# ============================================================
# 高层封装
# ============================================================

class DegradationPipeline:
    """可配置的退化管道，支持多种模式。

    Parameters
    ----------
    sf : int
        下采样倍数 (2 或 4)
    mode : str
        'bicubic'  — 仅 bicubic 下采样（基线）
        'bsrgan'   — BSRGAN 随机洗牌退化
        'custom'   — 自定义退化参数
    lq_patchsize : int
        输出 LR patch 大小 (仅 bsrgan 模式)
    blur_sigma : float
        模糊核 sigma (仅 custom 模式)
    noise_level : int
        噪声强度 (仅 custom 模式)
    jpeg_quality : int
        JPEG 质量因子 (仅 custom 模式)
    """

    def __init__(self, sf=4, mode='bsrgan', lq_patchsize=72,
                 blur_sigma=0, noise_level=0, jpeg_quality=0):
        self.sf = sf
        self.mode = mode
        self.lq_patchsize = lq_patchsize
        self.blur_sigma = blur_sigma
        self.noise_level = noise_level
        self.jpeg_quality = jpeg_quality

    def __call__(self, hr_img):
        """
        Parameters
        ----------
        hr_img : ndarray, HxWxC, float32, [0,1]

        Returns
        -------
        (lr, hr) : tuple
            bsrgan 模式返回裁剪后的 (lr_patch, hr_patch)
            bicubic/custom 模式返回 (lr_full, hr_cropped)
        """
        if self.mode == 'bsrgan':
            return degradation_bsrgan(hr_img, sf=self.sf,
                                      lq_patchsize=self.lq_patchsize)
        elif self.mode == 'bicubic':
            hr = modcrop(hr_img.copy(), self.sf)
            lr = degradation_bicubic(hr_img, sf=self.sf)
            return lr, hr
        elif self.mode == 'custom':
            hr = modcrop(hr_img.copy(), self.sf)
            lr = degradation_custom(hr_img, sf=self.sf,
                                    blur_sigma=self.blur_sigma,
                                    noise_level=self.noise_level,
                                    jpeg_quality=self.jpeg_quality)
            return lr, hr
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __repr__(self):
        return (f"DegradationPipeline(sf={self.sf}, mode='{self.mode}', "
                f"lq_patchsize={self.lq_patchsize})")
