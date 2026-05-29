# 项目介绍：单图像超分辨率系统 (SISR)

## 一句话概括

从零搭建的端到端图像超分辨率系统，覆盖模型训练、实验管理、ONNX 导出到浏览器端实时推理的完整链路。

---

## 项目背景

图像超分辨率（Super-Resolution）是计算机视觉中的经典问题：将低分辨率图像恢复为高分辨率。本项目作为深度学习课程的团队项目，不以追求 SOTA 为目标，而是侧重工程实践——如何用规范的软件工程方法组织一个 AI 项目。

---

## 技术架构

```
用户上传图片 → 浏览器 ONNX Runtime Web (WASM) → 实时超分辨率 → 对比展示
                         ↑
PyTorch 训练 → ONNX 导出 → 模型文件 (.onnx)
```

项目分为三层：

- **训练层**：PyTorch 2.11，支持 5 种经典固定倍率模型（SRCNN / FSRCNN / ESPCN / EDSR / IMDN）+ LIIF 任意倍率模型，YAML 配置驱动，WandB 实验追踪
- **导出层**：PyTorch → ONNX 自动导出，处理了 ConvTranspose2d 算子兼容性、PyTorch Dynamo 导出器适配等实际工程问题
- **部署层**：单文件 HTML 前端，ONNX Runtime Web 推理，支持 WASM 和 WebGPU 后端，零服务器依赖

---

## 我负责的核心工作

**1. 项目架构设计与重构**

将原始的"一个大文件"重构为分层架构：

```
入口脚本 (train.py, test.py)
    ↓
核心引擎 (core/trainer, evaluator, inferencer)
    ↓
模型定义 (models/) + 数据加载 (data/)
    ↓
工具函数 (utils/)
```

通过模型注册表 + 工厂函数实现松耦合，新增模型只需写一个文件 + 注册一行代码。

**2. Web 端实时推理**

解决了"训练好的模型怎么让非技术人员使用"的问题：

- 设计 PyTorch → ONNX 导出流水线，处理了 PixelShuffle 算子分解、动态输入尺寸等问题
- 适配 PyTorch 2.x Dynamo 导出器的 breaking change（opset 版本、导出器选择）
- 解决 FSRCNN 的 ConvTranspose2d 在 WASM 后端不支持的问题：设计了双模式架构 + 权重转换方案
- 前端使用 ONNX Runtime Web，实现 HWC↔CHW 张量转换、模型缓存、WebGPU 自动检测

**3. 任意尺度超分辨率（LIIF）**

引入 LIIF（Local Implicit Image Function）实现任意倍率超分辨率：单一模型覆盖 x1.5、x2、x3.14、x4、x6 等任意正实数倍率。采用 EDSREncoder 提取特征 + 5 层 MLP 对连续坐标查询 RGB，训练时随机采样 [1, 4] 倍率。独立的训练/测试/可视化脚本，分块查询防 OOM。

**4. Blind SR 实验方案设计**

针对真实世界退化（非理想双三次下采样）的超分方向，设计了基于 RealESRGAN / BSRGAN 的对比实验方案，包括退化管道分析、PSNR 训练 vs GAN 训练的感知质量对比等。

**5. 工程规范建设**

- Git 多机协作工作流（本地 ↔ GitHub ↔ 云服务器双向同步）
- 环境同步策略（uv 包管理 + 清华源镜像 + CUDA 版本隔离）
- Vibe Coding Review 指南（8 维度代码审查 + 跨平台部署踩坑案例）

---

## 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习 | PyTorch 2.11, TorchVision, CUDA |
| 模型部署 | ONNX, ONNX Runtime Web (WASM/WebGPU) |
| 包管理 | uv (清华源镜像), pyproject.toml |
| 实验管理 | WandB, YAML 配置 |
| 前端 | 原生 HTML/CSS/JS, ONNX Runtime Web CDN |
| 工程工具 | Git, shell scripts, Python AST 检查 |
| 数据 | DIV2K, Pillow, scikit-image |
| 评估指标 | PSNR, SSIM, NIQE, BRISQUE (pyiqa) |
| 退化建模 | OpenCV, SciPy (BSRGAN 退化管道移植) |

---

## 关键数据

- 模型规模：ESPCN 约 25KB (ONNX)，EDSR 约 5.8MB
- 训练数据：DIV2K（800 张 2K 分辨率图像）
- 最佳 PSNR (x4)：EDSR 28.56 dB，SRCNN 27.67 dB，ESPCN 27.48 dB
- LIIF 支持任意倍率（x1~x6+），单模型覆盖所有尺度
- 浏览器推理延迟：512×512 输入约 200-500ms (WASM)

---

## 面试可能追问的点

**Q: 为什么选 ONNX 而不是 TorchScript 部署？**

ONNX 是跨平台标准格式，ORT Web 可以跑在浏览器里（WASM/WebGPU），不需要后端服务器。TorchScript 只能在有 LibTorch 的环境跑，浏览器做不到。

**Q: ConvTranspose2d 不兼容是怎么解决的？**

FSRCNN 用 ConvTranspose2d 做上采样，但 ORT Web WASM 不支持这个算子。我的方案是给模型加了 `upsample_mode` 参数，支持 `'deconv'` 和 `'pixelshuffle'` 两种模式。最初尝试权重转换脚本（deconv→pixelshuffle），但发现转换是有损的（sub-pixel 位置映射不精确导致色偏）。最终方案：直接用 `upsample_mode='pixelshuffle'` 从头训练（`configs/fsrcnn_x4_pixelshuffle.yaml`），训练完直接 ONNX 导出，无需任何权重转换。

**Q: 项目中遇到的最大工程挑战是什么？**

跨环境一致性。本地 Windows + 云服务器 Linux，PyTorch 版本不同导致 ONNX 导出行为不同（Dynamo vs TorchScript 导出器），CUDA 版本不同导致依赖冲突。我的解决方案是：精简 requirements.txt 只列主依赖、PyTorch/CUDA 各环境单独装、导出脚本自动检测版本并选择兼容路径。

**Q: 如何保证 AI 辅助编码的代码质量？**

制定了 8 维度 Review 框架（架构合规、耦合度、文件粒度、DRY、变更范围、命名、错误处理、依赖管理），每次 AI 生成代码后做 5 分钟快速 Review，合并前做 30 分钟深度 Review。同时维护了跨平台部署的踩坑案例库，避免重复踩坑。
