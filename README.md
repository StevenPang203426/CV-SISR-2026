# 单图像超分辨率系统 (SISR)

从零搭建的端到端图像超分辨率系统，覆盖模型训练、实验管理、ONNX 导出到浏览器端实时推理的完整链路。

```
用户上传图片 → 浏览器 ONNX Runtime Web (WASM/WebGPU) → 实时超分辨率 → 对比展示
                         ↑
         PyTorch 训练 → ONNX 导出 → 模型文件 (.onnx)
```

---

## 项目概述

本项目是深度学习课程的团队项目，围绕图像超分辨率（Super-Resolution）展开。项目不以追求 SOTA 为目标，侧重工程实践，采用三域分离架构组织代码：

- **固定倍率 SR**：5 种经典模型（SRCNN / FSRCNN / ESPCN / EDSR / IMDN），x2/x3/x4 三档倍率
- **任意倍率 SR**：基于 LIIF（Local Implicit Image Function），单一模型覆盖任意正实数倍率（x1~x6+）
- **盲超分 SR**：引入 RealESRGAN / BSRGAN 预训练模型，在真实退化条件下与 bicubic SR 模型做对比实验
- **Web 部署**：单文件 HTML 前端，ONNX Runtime Web 浏览器端推理，零服务器依赖

---

## 实验结果

### 固定倍率 Baseline（验证集 PSNR）

训练数据：DIV2K（800 张），统一设置 500 epoch、batch_size=16、patch_size=96、Adam + MSE。

| Scale | SRCNN | ESPCN | EDSR |
|-------|------:|------:|-----:|
| x2    | 32.89 dB | 32.57 dB | 34.22 dB |
| x3    | 29.48 dB | 29.20 dB | 30.54 dB |
| x4    | 27.67 dB | 27.48 dB | 28.56 dB |

EDSR 在三个尺度上均取得最优验证结果，验证了更深残差结构在细节恢复上的优势。

### LIIF 任意倍率

单一模型覆盖 x1.5、x2、x3.14、x4、x6 等任意正实数倍率。采用 EDSREncoder（64 通道、16 ResBlock）提取特征，5 层 256 维 MLP 对连续坐标查询 RGB 值。训练时随机采样 [1, 4] 倍率。

### Blind SR 实验

针对真实世界退化（模糊、噪声、JPEG 压缩等非理想 bicubic 下采样），设计了三组对比实验：
1. Bicubic SR vs Blind SR 在多退化条件下的崩溃对比
2. 不同退化因素（模糊/噪声/JPEG）的消融实验
3. PSNR 训练（BSRNet）vs GAN 训练（BSRGAN）的感知质量对比

---

## 项目结构

```
SISR-Team8/
├── src/                        # 核心代码（三域分离）
│   ├── common/                 # 公共模块：checkpoint、metrics、img 工具
│   ├── fixed_sr/               # 固定倍率 SR
│   │   ├── models/             #   SRCNN / FSRCNN / ESPCN / EDSR / IMDN
│   │   ├── config.py           #   YAML 配置解析
│   │   ├── trainer.py          #   训练引擎
│   │   ├── evaluator.py        #   评估引擎（PSNR/SSIM/FLOPs/FPS）
│   │   ├── inferencer.py       #   推理引擎（分块推理）
│   │   └── export_onnx.py      #   ONNX 导出（含 RRDB pretrained）
│   ├── liif/                   # 任意倍率 SR
│   │   ├── encoder.py          #   EDSREncoder（独立副本，零跨域依赖）
│   │   ├── liif.py             #   LIIF 解码器 + MLP
│   │   ├── liif_model.py       #   Encoder + LIIF 组合
│   │   ├── dataset.py          #   随机倍率 + 坐标采样
│   │   ├── train.py / test.py  #   训练/测试入口
│   │   └── visualize.py        #   多倍率对比可视化
│   └── blind_sr/               # 盲超分 SR
│       ├── rrdbnet.py          #   RRDB 网络（加载 5 种预训练权重）
│       ├── degradation.py      #   退化管道（移植自 BSRGAN）
│       └── eval.py             #   统一评估脚本（实验 1/2/3）
├── configs/                    # YAML 训练配置
├── scripts/                    # Shell 脚本（带默认参数模板）
├── web/                        # 浏览器端推理前端
│   ├── index.html              #   单文件应用（ESPCN/FSRCNN/BSRNet）
│   └── models/                 #   导出的 .onnx 模型文件
├── train.py / test.py / infer.py   # 根入口脚本（固定倍率 SR）
├── pretrained/                 # RRDB 预训练权重（BSRNet/BSRGAN/RealESRGAN 等）
└── docs/                       # 项目文档
```

---

## 快速开始

### 环境安装

```bash
# 推荐使用 uv（清华源镜像）
uv sync

# 或 pip
pip install -e .
```

### 固定倍率训练

```bash
# 单个模型
bash scripts/train_fixed.sh configs/edsr_x4.yaml

# 批量（tmux 多窗口）
bash scripts/train.sh 2 3 4
```

### 固定倍率测试

```bash
python test.py \
    --ckpt experiments/edsr_x2/best.pt \
    --test_dir data/datasets/Set5 \
    --model edsr --scale 2 --save_images
```

### LIIF 训练与测试

```bash
# 训练
bash scripts/train_liif.sh configs/liif_edsr_x1-4.yaml

# 多倍率测试
python -m src.liif.test \
    --ckpt experiments/liif_edsr/best.pt \
    --test_dir data/datasets/Set5 \
    --scales 2 3 4 6 --save_images

# 可视化对比面板
python -m src.liif.visualize \
    --ckpt experiments/liif_edsr/best.pt \
    --input data/datasets/Set5/baby.png \
    --scales 1.5 2 3.5 4 6
```

### Blind SR 评估

```bash
# 下载预训练权重
bash scripts/download_pretrained.sh

# 运行实验（1=退化对比, 2=因素消融, 3=PSNR vs GAN）
bash scripts/eval_blind.sh 1
```

### ONNX 导出与 Web 部署

```bash
# 导出所有可用模型（轻量 + RRDB）
bash scripts/export_onnx.sh --all

# 仅导出 BSRNet
python -m src.fixed_sr.export_onnx --model bsrnet

# 启动 Web 前端
cd web && python -m http.server 8080
# 浏览器访问 http://localhost:8080
```

---

## 数据集

训练和评估使用以下数据集，LR 图像由 bicubic 下采样动态生成，无需手动准备：

```
data/
├── DIV2K/
│   ├── DIV2K_train_HR/       # 训练集（800 张 2K 图像）
│   └── DIV2K_valid_HR/       # 验证集
├── datasets/
│   ├── Set5/                 # 经典测试集
│   ├── Set14/
│   ├── BSD100/
│   └── Urban100/
└── RealSRSet/                # Blind SR 真实退化测试集（无 GT）
```

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.x, TorchVision, CUDA |
| 模型部署 | ONNX, ONNX Runtime Web (WASM/WebGPU) |
| 包管理 | uv + pyproject.toml |
| 实验管理 | WandB, YAML 配置驱动 |
| 前端 | 原生 HTML/CSS/JS, ONNX Runtime Web CDN |
| 评估指标 | PSNR, SSIM, NIQE, BRISQUE (pyiqa) |
| 退化建模 | OpenCV, SciPy（BSRGAN 退化管道移植） |

---

## 关键工程决策

**三域分离架构**：`src/fixed_sr/`、`src/liif/`、`src/blind_sr/` 三个域零交叉依赖，公共模块（checkpoint、metrics）抽取到 `src/common/`。LIIF 的 EDSREncoder 是独立副本，不依赖 fixed_sr 域。

**FSRCNN 双模式上采样**：原始 ConvTranspose2d 在 ORT Web WASM 后端不支持，设计了 `upsample_mode` 参数支持 `deconv` 和 `pixelshuffle` 两种模式。Web 部署使用 PixelShuffle 版本从头训练，避免有损权重转换。

**RRDB 模型导出**：扩展 `export_onnx.py` 统一支持轻量模型和 RRDB pretrained 模型导出，自动检测 checkpoint 格式（deconv/pixelshuffle）按需处理。Web 端 BSRNet 强制 WASM 后端（64MB 模型在 WebGPU 下易 OOM）。

**模型注册表 + 工厂函数**：`build_model(name, **kwargs)` 自省参数签名自动过滤，新增模型只需写一个文件 + 注册一行。

---

## 参考文献

- **SRCNN**: Dong et al., "Learning a Deep Convolutional Network for Image Super-Resolution," ECCV 2014. [Paper](https://arxiv.org/abs/1501.00092)
- **FSRCNN**: Dong et al., "Accelerating the Super-Resolution Convolutional Neural Network," ECCV 2016. [Paper](https://arxiv.org/abs/1608.00367)
- **ESPCN**: Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," CVPR 2016. [Paper](https://arxiv.org/abs/1609.05158)
- **EDSR**: Lim et al., "Enhanced Deep Residual Networks for Single Image Super-Resolution," CVPR Workshops 2017. [Paper](https://arxiv.org/abs/1707.02921)
- **IMDN**: Hui et al., "Lightweight Image Super-Resolution with Information Multi-Distillation Network," ACM MM 2019. [Paper](https://arxiv.org/abs/1909.11856)
- **LIIF**: Chen et al., "Learning Continuous Image Representation with Local Implicit Image Function," CVPR 2021. [Paper](https://arxiv.org/abs/2012.09161)
- **BSRGAN**: Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution," ICCV 2021. [Paper](https://arxiv.org/abs/2103.14006)
- **Real-ESRGAN**: Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCVW 2021. [Paper](https://arxiv.org/abs/2107.10833)
