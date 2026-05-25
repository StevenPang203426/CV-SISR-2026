# 真实世界盲超分 (Blind SR) 实验方案

> 目标：理解并实践 Blind SR 的核心思想——复杂退化建模 + GAN 训练，
> 通过系统的对比实验展示"为什么 Bicubic SR 在真实场景下失效"以及"Blind SR 如何解决这个问题"。

---

## 一、定位：我们到底该做什么？

### 1.1 问题分析

| 选项 | 是否可行 | 原因 |
|------|----------|------|
| **从头训练 Real-ESRGAN** | 不现实 | 原作用 DIV2K + Flickr2K + OST 共 ~3500 张 HR 图，配合在线退化管道，8 卡 V100 训练数天。我们只有 DIV2K 且单卡 |
| **用 DIV2K 微调** | 意义不大 | DIV2K 已经是 Real-ESRGAN / BSRGAN 的训练集子集，在同一数据上微调不会学到新知识 |
| **纯复现（对齐原论文指标）** | 无法做到 | 无法获得完全相同的训练数据、随机种子、训练步数，差异不可控 |
| **对比实验 + 退化管道分析 + 轻量微调** | **推荐** | 有深度、有工作量、展示理解能力、可在单卡 16GB 上完成 |

### 1.2 核心定位

我们的工作分三个层次，由浅到深：

```
第一层：评估层（必做）
  用预训练模型在多种退化条件下做系统对比 → 量化 Bicubic SR 的局限性

第二层：理解层（必做）
  移植 BSRGAN 退化管道到我们的项目 → 可视化不同退化组合的效果

第三层：实践层（加分）
  基于退化管道 + 预训练模型，在自定义小数据集上做轻量微调
```

---

## 二、预训练模型选型

### 2.1 我们需要的模型

| 模型 | 来源 | 用途 | 参数量 | 下载 |
|------|------|------|--------|------|
| **RealESRGAN_x4plus** | Real-ESRGAN | 通用真实图像修复（x4） | ~16.7M (RRDB) | [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| **RealESRNet_x4plus** | Real-ESRGAN | PSNR 导向版本（无 GAN，纯 L1 训练） | ~16.7M | [GitHub Release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth) |
| **BSRGAN** | BSRGAN | BSRGAN 退化模型训练的盲 SR（x4） | ~16.7M (RRDB) | [KAIR Release](https://github.com/cszn/KAIR/releases/tag/v1.0) |
| **BSRNet** | BSRGAN | PSNR 导向版本（无 GAN） | ~16.7M | [KAIR Release](https://github.com/cszn/KAIR/releases/tag/v1.0) |
| **ESRGAN** | ESRGAN (原版) | 经典 GAN SR，仅针对 bicubic 退化 | ~16.7M | [KAIR Release](https://github.com/cszn/KAIR/releases/tag/v1.0) |

> **为什么选这些？**
> - RealESRGAN vs ESRGAN：展示"blind 退化建模"的提升
> - RealESRGAN vs RealESRNet：展示"GAN 训练 vs PSNR 训练"的感知差异
> - BSRGAN vs RealESRGAN：展示不同退化管道设计的效果差异
> - BSRNet vs BSRGAN：BSRGAN 项目内部的 PSNR vs GAN 对比
> - 所有 Blind SR vs 我们已有的 EDSR/ESPCN/SRCNN：展示 bicubic SR 在真实退化下的崩溃

### 2.2 它们共用同一个网络结构

所有模型都基于 **RRDB (Residual in Residual Dense Block)**，即 `features/BSRGAN/models/network_rrdbnet.py` 中的：

```python
RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
```

- `nf=64`：特征通道数
- `nb=23`：RRDB 块数量
- `gc=32`：Dense Block 的 growth channel
- `sf=4`：上采样倍数（PixelShuffle）

它们的区别仅在于**训练数据的退化方式**和**损失函数**：

```
ESRGAN    = RRDB + bicubic 退化 + GAN loss
RealESRNet = RRDB + Real-ESRGAN 二阶退化 + L1 loss (PSNR 导向)
RealESRGAN = RRDB + Real-ESRGAN 二阶退化 + GAN loss (感知导向)
BSRNet    = RRDB + BSRGAN 随机洗牌退化 + L1 loss
BSRGAN    = RRDB + BSRGAN 随机洗牌退化 + GAN loss
```

### 2.3 模型存放位置

```
SISR-Team8/
└── pretrained/                   ← 新建，存放所有预训练权重
    ├── RealESRGAN_x4plus.pth
    ├── RealESRNet_x4plus.pth
    ├── BSRGAN.pth
    ├── BSRNet.pth
    └── ESRGAN.pth
```

---

## 三、退化管道（核心贡献之一）

### 3.1 为什么要自己实现退化管道？

这是本任务最核心的技术点。Bicubic 降采样是"理想化"的退化：

```
HR → Bicubic ↓ → LR   （干净、单一、不真实）
```

真实世界的退化远比这复杂：

```
HR → 模糊(多种核) → 降采样(多种方法) → 噪声(高斯+泊松) → JPEG压缩 → LR
     ↑ 这些步骤的顺序还是随机打乱的
```

### 3.2 BSRGAN 退化管道核心流程

我们已有 `features/BSRGAN/utils/utils_blindsr.py` 中的 `degradation_bsrgan()` 函数。
它的核心逻辑是将 7 种退化操作**随机洗牌**后依次执行：

```
退化操作池 = [模糊1, 模糊2, 降采样2, 降采样3, 高斯噪声, JPEG压缩, 相机ISP噪声]

执行顺序 = random_shuffle(退化操作池)
          但约束：降采样3 必须在降采样2 之后

对于 x4：有 25% 概率先做一次 x2 降采样，然后剩余退化在 x2 尺度上操作
最终：再追加一次 JPEG 压缩
```

### 3.3 Real-ESRGAN 的二阶退化（对比）

Real-ESRGAN 用了更复杂的"二阶退化"：

```
HR → [模糊 → 降采样 → 噪声 → JPEG]₁ → [模糊 → 降采样 → 噪声 → JPEG]₂ → sinc filter → LR
                第一阶退化                          第二阶退化                   振铃伪影
```

每一阶内部的参数（模糊核类型、噪声强度、JPEG 质量）都随机采样。

### 3.4 我们的实现策略

**不重新造轮子**，而是**移植 + 适配 + 可视化**：

```python
# 新建 data/degradation.py
# 核心：从 BSRGAN 的 utils_blindsr.py 提取退化函数，
#       适配到我们项目的接口（PIL Image → 退化 → PIL Image）

class DegradationPipeline:
    """可配置的退化管道，支持 BSRGAN 风格和自定义参数"""

    def __init__(self, sf=4, mode='bsrgan'):
        """
        mode:
          'bicubic'  - 仅 bicubic 降采样（基线）
          'bsrgan'   - BSRGAN 随机洗牌退化
          'custom'   - 自定义退化参数（用于消融实验）
        """

    def __call__(self, hr_img):
        """输入 HR PIL Image，返回 (LR, HR) pair"""
```

```python
# 新建 scripts/visualize_degradation.py
# 对同一张 HR 图，生成不同退化方式的 LR 对比图：
#   - Bicubic only
#   - BSRGAN (弱退化)
#   - BSRGAN (中等退化)
#   - BSRGAN (强退化)
#   - 仅模糊 + 降采样（无噪声）
#   - 仅噪声 + JPEG（无模糊）
# 输出一张多图对比的可视化面板
```

---

## 四、实验设计（5 组实验）

### 实验 1：Bicubic SR 模型在真实退化下的崩溃（核心实验）

**目的**：量化证明"用 bicubic 退化训练的模型，遇到真实退化就废了"。

**方法**：
1. 从 DIV2K 验证集（`data/DIV2K_valid_HR/`）取 20 张图
2. 对每张图生成 3 种退化的 LR：
   - **Bicubic**：标准 bicubic ×4 降采样
   - **BSRGAN-mild**：轻度 BSRGAN 退化（低噪声、弱模糊）
   - **BSRGAN-heavy**：重度 BSRGAN 退化（高噪声、强模糊、重 JPEG）
3. 用以下模型分别推理：
   - 我们训练的 EDSR_x4、ESPCN_x4、SRCNN_x4（bicubic SR）
   - RealESRGAN_x4plus（blind SR，GAN）
   - BSRGAN（blind SR，GAN）
4. 计算 PSNR / SSIM

**预期结论**：
- 在 Bicubic LR 上：EDSR ≈ RealESRGAN（都能处理）
- 在 BSRGAN-heavy LR 上：EDSR 严重退化，RealESRGAN / BSRGAN 保持稳定

**输出**：表格 + 可视化对比图

### 实验 2：不同退化类型的影响（消融实验）

**目的**：分析不同退化因素（模糊、噪声、JPEG）各自对 SR 质量的影响。

**方法**：
1. 取 10 张 DIV2K 验证图
2. 固定 x4 降采样，逐个加入退化因素：
   - `D0`：仅 bicubic 降采样（干净基线）
   - `D1`：+ 高斯模糊 (sigma=1.5)
   - `D2`：+ 高斯噪声 (sigma=15)
   - `D3`：+ JPEG 压缩 (quality=40)
   - `D4`：模糊 + 噪声
   - `D5`：模糊 + 噪声 + JPEG（完整退化）
   - `D6`：BSRGAN 随机退化（随机顺序）
3. 在每种退化下用 EDSR_x4 和 RealESRGAN_x4plus 推理
4. 计算 PSNR / SSIM，画折线图

**预期结论**：
- 模糊对 bicubic SR 杀伤力最大（分布偏移）
- JPEG 压缩引入块效应，blind SR 能处理
- 退化组合后效果叠加，bicubic SR 完全崩溃

### 实验 3：PSNR 训练 vs GAN 训练（感知质量分析）

**目的**：展示 L1 loss（PSNR 导向）和 GAN loss（感知导向）的视觉差异。

**方法**：
1. 在 BSRGAN 自带的 `testsets/RealSRSet/` (20 张真实退化图) 上推理
2. 对比 4 个模型：
   - BSRNet（PSNR 导向）
   - BSRGAN（GAN 导向）
   - RealESRNet_x4plus（PSNR 导向）
   - RealESRGAN_x4plus（GAN 导向）
3. 真实图像**没有 HR Ground Truth**，所以**不算 PSNR**，改用：
   - **NIQE**（无参考感知质量指标，越低越好）
   - **BRISQUE**（无参考质量指标）
   - 视觉对比（zoom-in 细节裁剪）

**预期结论**：
- PSNR 导向的模型输出更平滑，细节偏模糊
- GAN 导向的模型输出更锐利，有更多纹理（但偶尔引入伪影）
- 这就是为什么论文要分两阶段训练：先 PSNR 预训练 → 再 GAN 微调

### 实验 4：退化管道可视化与分析

**目的**：直观展示退化管道的工作原理，作为技术理解的佐证。

**方法**：
1. 选 3 张代表性 HR 图（人脸、建筑/纹理、自然风景）
2. 对每张图用 `degradation_bsrgan()` 生成 8 次退化结果（不同随机种子）
3. 可视化：
   - 退化前后的图像对比
   - 每次退化使用的模糊核可视化
   - 退化参数统计（噪声强度、JPEG 质量、模糊 sigma 的分布直方图）

**输出**：一张大面板图（类似论文 Figure），展示退化的多样性

### 实验 5（可选加分）：轻量微调

**目的**：展示动手能力，用自定义数据微调 RealESRNet。

**方法**：
1. **数据**：从网上收集 20-30 张真实老照片 / 手机低光照片（非 DIV2K）
   - 或者用 DIV2K 的 HR 图通过 BSRGAN 退化管道生成 LR-HR 对
   - 这不算"用 DIV2K 微调"——因为退化方式是新的，模型能学到新的退化-重建映射
2. **微调策略**（参考 Real-ESRGAN README 的 finetune 方案）：
   - 加载 RealESRNet_x4plus 预训练权重
   - 冻结前 15 个 RRDB 块，只训练后 8 个 + upsampler
   - 小学习率 `1e-5`，训练 2000-5000 iterations
   - L1 + Perceptual loss（VGG feature matching）
   - 单卡 16GB 可行（batch_size=4, patch=128）
3. **对比**：微调前后在自己的测试集上的效果

> **关键点**：即使用 DIV2K HR + BSRGAN 退化生成训练对，这也是有意义的工作——
> 因为你的退化参数可以定制（比如偏重 JPEG 压缩或偏重模糊），
> 相当于"针对特定退化场景的定制化微调"。

---

## 五、我们的项目与 Real-ESRGAN / BSRGAN 的关系

```
                    参考项目（只读，不修改）
                    ┌──────────────────────────────────────────┐
                    │  features/BSRGAN/                         │
                    │    └─ utils/utils_blindsr.py  ← 退化管道源码│
                    │    └─ models/network_rrdbnet.py ← RRDB网络 │
                    │    └─ testsets/RealSRSet/  ← 真实退化测试图  │
                    │                                           │
                    │  (Real-ESRGAN 没有 clone 到本地，            │
                    │   只需下载预训练权重)                        │
                    └──────────────────────────────────────────┘
                              │
                              │ 提取 / 移植 / 参考
                              ▼
              我们的代码（新增/修改）
              ┌──────────────────────────────────────────┐
              │  data/                                     │
              │    └─ degradation.py       ← 移植退化管道   │
              │    └─ blind_sr_dataset.py  ← 新数据集类     │
              │                                           │
              │  models/                                   │
              │    └─ rrdbnet.py           ← 移植 RRDB 网络 │
              │    └─ __init__.py          ← 注册 rrdb      │
              │                                           │
              │  scripts/                                  │
              │    └─ eval_blind_sr.py     ← 对比评估脚本   │
              │    └─ visualize_degradation.py ← 退化可视化 │
              │    └─ finetune_realesrnet.py   ← 微调脚本   │
              │                                           │
              │  pretrained/                               │
              │    └─ *.pth               ← 预训练权重      │
              │                                           │
              │  experiments/blind_sr/                      │
              │    └─ eval_results/        ← 实验结果       │
              │    └─ visualizations/      ← 可视化输出     │
              │    └─ finetune/            ← 微调检查点     │
              └──────────────────────────────────────────┘
```

---

## 六、新增文件清单

| 文件 | 作用 | 优先级 |
|------|------|--------|
| `models/rrdbnet.py` | 从 BSRGAN 移植 RRDB 网络，适配我们的 `build_model()` 注册表 | 必须 |
| `data/degradation.py` | 退化管道，从 BSRGAN 移植核心函数并封装为可配置的 class | 必须 |
| `data/blind_sr_dataset.py` | 在线退化 Dataset：每次取样时随机退化 HR → LR | 实验5需要 |
| `scripts/eval_blind_sr.py` | 统一评估脚本：加载多个预训练模型，在多种退化下推理并计算指标 | 必须 |
| `scripts/visualize_degradation.py` | 退化管道可视化脚本 | 必须 |
| `scripts/finetune_realesrnet.py` | 微调脚本（部分冻结 + 小 lr） | 可选 |
| `scripts/calc_niqe.py` | 无参考图像质量评估（NIQE / BRISQUE） | 实验3需要 |
| `pretrained/` | 预训练模型存放目录 | 必须 |
| `experiments/blind_sr/` | 实验结果输出目录 | 必须 |

---

## 七、实施路线图

### 第 1 天：基础设施

| 步骤 | 工作 | 产出 |
|------|------|------|
| 1.1 | 下载 5 个预训练模型到 `pretrained/` | .pth 文件 |
| 1.2 | 移植 `models/rrdbnet.py`，注册到 `build_model()` | 能加载所有预训练权重 |
| 1.3 | 编写统一推理脚本 `scripts/eval_blind_sr.py` | 能用任意模型推理任意图片 |
| 1.4 | 验证：用 BSRGAN.pth 推理 `testsets/RealSRSet/` 的图片，对比原仓库输出 | 确认移植正确 |

### 第 2 天：退化管道

| 步骤 | 工作 | 产出 |
|------|------|------|
| 2.1 | 移植 `data/degradation.py`，处理依赖问题 | 可配置的退化管道 |
| 2.2 | 编写 `scripts/visualize_degradation.py` | 退化可视化面板图 |
| 2.3 | 运行实验 4：退化管道可视化 | 多样退化结果图 |

### 第 3 天：核心实验

| 步骤 | 工作 | 产出 |
|------|------|------|
| 3.1 | 运行实验 1：Bicubic SR vs Blind SR 在多退化下的对比 | PSNR/SSIM 表格 + 对比图 |
| 3.2 | 运行实验 2：不同退化类型的消融 | 折线图 |
| 3.3 | 运行实验 3：PSNR vs GAN 感知质量对比 | NIQE/BRISQUE 表格 + zoom-in 对比 |

### 第 4 天（可选）：微调

| 步骤 | 工作 | 产出 |
|------|------|------|
| 4.1 | 编写 `data/blind_sr_dataset.py` 和 `scripts/finetune_realesrnet.py` | 训练代码 |
| 4.2 | 用 DIV2K HR + BSRGAN 退化生成训练对，微调 RealESRNet | 微调权重 |
| 4.3 | 对比微调前后效果 | 对比表格 |

---

## 八、关于"用 DIV2K 是否有意义"的澄清

这是你最核心的疑虑，直接回答：

**对比实验完全可以用 DIV2K**。原因：

1. **实验 1-2 的测试数据**：我们用 DIV2K 验证集的 HR 图作为 Ground Truth，自己生成退化 LR。这和训练数据无关——我们不是在训练，只是在评估。预训练模型是否见过这些图不影响实验结论，因为我们要展示的是"退化方式的影响"，而非"模型泛化能力"。

2. **实验 3 的测试数据**：我们直接用 BSRGAN 自带的 `testsets/RealSRSet/`（20 张真实退化图），不需要 DIV2K。

3. **实验 5 的微调数据**：虽然 HR 来自 DIV2K，但退化方式是我们自定义的。模型学到的是"新退化 → 重建"的映射，这是有意义的增量训练。如果要更有说服力，也可以从网上收集 20-30 张非 DIV2K 的高清图。

**唯一不能做的是**：声称"我们在 DIV2K 上从头训练了 Real-ESRGAN 并达到了 SOTA"——这没有意义，因为原作已经这么做过了。

---

## 九、评估指标

| 指标 | 类型 | 用途 | 库 |
|------|------|------|-----|
| **PSNR** | 有参考 | 衡量像素级重建精度 | `skimage.metrics.peak_signal_noise_ratio` |
| **SSIM** | 有参考 | 衡量结构相似性 | `skimage.metrics.structural_similarity` |
| **LPIPS** | 有参考（学习型） | 衡量感知相似度（更符合人眼） | `pip install lpips` |
| **NIQE** | 无参考 | 自然图像质量评估（越低越好） | `pyiqa` 库 |
| **BRISQUE** | 无参考 | 盲图像质量评估（越低越好） | `pyiqa` 库 |

> 有参考指标（PSNR/SSIM/LPIPS）用于实验 1、2（有 HR ground truth）
> 无参考指标（NIQE/BRISQUE）用于实验 3（真实退化图无 ground truth）

---

## 十、预期成果

完成后的交付物：

1. **退化管道模块** (`data/degradation.py`)：可配置、可复用的退化管道
2. **RRDB 模型集成** (`models/rrdbnet.py`)：统一注册到项目框架
3. **5 组实验结果**：
   - 实验 1：Bicubic SR 崩溃的量化证据（PSNR 表格 + 可视化）
   - 实验 2：退化因素消融（折线图）
   - 实验 3：PSNR vs GAN 感知对比（NIQE 表格 + zoom-in 对比图）
   - 实验 4：退化管道可视化面板
   - 实验 5（可选）：微调前后对比
4. **实验报告 / 展示材料**：汇总所有实验的分析和结论

---

## 附录 A：关键依赖

```bash
# 核心（已有）
torch, torchvision, Pillow

# 退化管道
opencv-python    # cv2，BSRGAN 退化管道依赖
scipy            # 模糊核生成

# 评估指标
scikit-image     # PSNR, SSIM
lpips            # 学习型感知指标（可选）
pyiqa            # NIQE, BRISQUE（可选）

# 可视化
matplotlib       # 对比图生成
```

## 附录 B：预训练模型下载命令

```bash
mkdir -p pretrained && cd pretrained

# Real-ESRGAN 系列
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth

# BSRGAN 系列（从 KAIR releases 下载）
# 手动下载: https://github.com/cszn/KAIR/releases/tag/v1.0
# 或从腾讯微云: https://share.weiyun.com/5qO32s3
# 需要: BSRGAN.pth, BSRNet.pth, ESRGAN.pth
```

## 附录 C：RealESRGAN_x4plus 与 BSRGAN 的退化策略差异

| 维度 | BSRGAN | Real-ESRGAN |
|------|--------|-------------|
| 退化阶数 | 一阶（7 步随机洗牌） | 二阶（每阶 4 步，两次叠加） |
| 模糊核 | 各向异性高斯 | 各向异性高斯 + generalized Gaussian + plateau |
| 降采样 | bicubic / bilinear / nearest 随机 | 同左 |
| 噪声 | 高斯 + 相机 ISP | 高斯 + 泊松 |
| JPEG | 标准 JPEG | 标准 JPEG |
| 振铃伪影 (sinc filter) | 无 | 有，第二阶末尾 |
| 退化步骤顺序 | 随机洗牌 | 固定（模糊→降采样→噪声→JPEG，但每阶独立） |
| 论文 | Zhang et al., ICCV 2021 | Wang et al., ICCVW 2021 |
