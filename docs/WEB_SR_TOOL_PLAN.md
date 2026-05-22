# Web 端纯前端超分辨率工具 —— 实施方案

> 将训练好的轻量级超分模型（ESPCN / FSRCNN）转为 ONNX 格式，
> 开发一个纯前端运行的"图片无损放大"网页小工具。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     浏览器（纯前端）                       │
│                                                         │
│  ┌──────────┐   ┌───────────┐   ┌────────────────────┐  │
│  │  用户上传  │──▶│  前处理    │──▶│  ONNX Runtime Web  │  │
│  │  图片      │   │  Canvas→   │   │  (WASM / WebGPU)   │  │
│  │  拖拽/选择 │   │  Tensor    │   │  加载 .onnx 模型   │  │
│  └──────────┘   └───────────┘   └────────┬───────────┘  │
│                                          │              │
│  ┌──────────┐   ┌───────────┐            │              │
│  │  下载/预览 │◀──│  后处理    │◀───────────┘              │
│  │  SR 图片  │   │  Tensor→   │                          │
│  │           │   │  Canvas    │                          │
│  └──────────┘   └───────────┘                           │
└─────────────────────────────────────────────────────────┘
```

**技术选型**：

| 层 | 选择 | 理由 |
|---|---|---|
| 前端框架 | 纯 HTML/JS + Canvas | 零依赖，一个 HTML 文件即可运行，适合课程演示 |
| 推理引擎 | ONNX Runtime Web (WASM) | 兼容所有现代浏览器，无需特殊开关；WebGPU 可选加速 |
| 模型格式 | ONNX (opset 11+) | PyTorch 原生支持导出，ORT Web 直接加载 |
| 主力模型 | ESPCN（首选）/ FSRCNN（备选） | 参数少、推理快，适合浏览器环境 |

---

## 二、模型准备：PyTorch → ONNX

### 2.1 为什么选 ESPCN 作为首选

| 模型 | 参数量 | 关键算子 | ONNX 兼容性 |
|------|--------|----------|-------------|
| **ESPCN** | ~20K | Conv2d + PixelShuffle | PixelShuffle 导出为 Reshape+Transpose，ORT 完美支持 |
| **FSRCNN** | ~13K | Conv2d + ConvTranspose2d + PReLU | ConvTranspose2d 的 `output_padding` 在某些 opset 下需注意 |
| SRCNN | ~57K | Conv2d（但需要先 bicubic 上采样输入） | 前处理依赖 `imresize_bicubic`，前端实现较麻烦 |
| EDSR/IMDN | 大 | ResBlock 堆叠 | 参数太大，浏览器加载慢、推理慢 |

ESPCN 结构最简单（3 层 Conv + PixelShuffle），参数最少，所有算子 ONNX/ORT 原生支持。

### 2.2 导出脚本：`scripts/export_onnx.py`

```python
"""
将 PyTorch 模型导出为 ONNX 格式。

用法:
    python scripts/export_onnx.py --model espcn --scale 4 \
        --ckpt experiments/espcn_x4/best.pt \
        --output web/models/espcn_x4.onnx

    # 批量导出所有可用模型
    python scripts/export_onnx.py --all
"""

import argparse
import os
import torch
from models import build_model
from core.checkpoint import load_checkpoint


def export_one(model_name: str, scale: int, ckpt_path: str, output_path: str,
               opset: int = 11, input_size: tuple = (1, 3, 64, 64)):
    """导出单个模型为 ONNX。"""
    # 1. 构建模型并加载权重
    model = build_model(model_name, scale=scale, in_channels=3)
    load_checkpoint(model, ckpt_path, device='cpu')
    model.eval()

    # 2. 创建 dummy 输入
    dummy = torch.randn(*input_size)

    # 3. 导出
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=['input'],        # LR 图像
        output_names=['output'],      # SR 图像
        dynamic_axes={                 # ⬅️ 关键：支持任意分辨率输入
            'input':  {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'},
        },
    )
    # 4. 打印模型大小
    size_kb = os.path.getsize(output_path) / 1024
    print(f'Exported: {output_path} ({size_kb:.1f} KB)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, help='模型名称')
    p.add_argument('--scale', type=int, help='超分倍数')
    p.add_argument('--ckpt', type=str, help='检查点路径')
    p.add_argument('--output', type=str, help='输出 .onnx 路径')
    p.add_argument('--opset', type=int, default=11)
    p.add_argument('--all', action='store_true', help='导出所有可用模型')
    args = p.parse_args()

    if args.all:
        for model in ['espcn', 'fsrcnn']:           # 只导出轻量模型
            for scale in [2, 3, 4]:
                ckpt = f'experiments/{model}_x{scale}/best.pt'
                if os.path.exists(ckpt):
                    out = f'web/models/{model}_x{scale}.onnx'
                    export_one(model, scale, ckpt, out, args.opset)
                else:
                    print(f'[SKIP] {ckpt} not found')
    else:
        export_one(args.model, args.scale, args.ckpt, args.output, args.opset)


if __name__ == '__main__':
    main()
```

### 2.3 导出注意事项

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **动态尺寸** | 用户上传的图片大小不固定 | `dynamic_axes` 让 height/width 维度动态 |
| **PixelShuffle** | ONNX 无原生 PixelShuffle 算子 | PyTorch 导出时自动拆解为 Reshape + Transpose，ORT 支持 |
| **ConvTranspose2d** | FSRCNN 用了反卷积 | opset ≥ 11 时 `output_padding` 正确导出 |
| **PReLU** | FSRCNN 激活函数 | ONNX 原生支持 PReLU |
| **SRCNN 不导出** | 需要前端实现 bicubic 上采样 | 前端不好实现精确的 bicubic，跳过 |

### 2.4 验证导出的模型

```python
"""scripts/verify_onnx.py —— 验证 ONNX 模型与 PyTorch 输出一致"""
import numpy as np
import onnxruntime as ort
import torch
from models import build_model
from core.checkpoint import load_checkpoint

def verify(model_name, scale, ckpt_path, onnx_path):
    # PyTorch 推理
    model = build_model(model_name, scale=scale, in_channels=3)
    load_checkpoint(model, ckpt_path, device='cpu')
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        pt_out = model(dummy).numpy()

    # ONNX Runtime 推理
    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {'input': dummy.numpy()})[0]

    # 对比
    max_diff = np.abs(pt_out - ort_out).max()
    print(f'{model_name}_x{scale}: max_diff = {max_diff:.8f}', 
          '✅' if max_diff < 1e-5 else '❌')
```

---

## 三、Web 前端实现

### 3.1 目录结构

```
web/                            ← 新建目录，放在项目根下
├── index.html                  ← 唯一的 HTML 文件（包含 CSS + JS）
├── models/                     ← ONNX 模型文件
│   ├── espcn_x2.onnx           ← ~80 KB
│   ├── espcn_x3.onnx
│   ├── espcn_x4.onnx
│   ├── fsrcnn_x2.onnx          ← ~52 KB（如果有训练好的权重）
│   ├── fsrcnn_x3.onnx
│   └── fsrcnn_x4.onnx
└── README.md                   ← 使用说明
```

### 3.2 核心流程（伪代码）

```
1. 用户选择图片 → <input type="file"> 或拖拽
2. 图片 → Canvas → ImageData → Float32Array（归一化到 [0,1]）
3. 创建 ORT InferenceSession，加载 .onnx 模型
4. 构造输入 Tensor: shape = [1, 3, H, W]（CHW 格式，注意 Canvas 是 HWC）
5. session.run() → 输出 Tensor: shape = [1, 3, H*scale, W*scale]
6. 输出 Tensor → Canvas → 预览 & 下载
```

### 3.3 关键代码片段

#### 加载 ONNX Runtime Web（从 CDN）

```html
<!-- 从 CDN 引入，无需 npm -->
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```

#### HWC ↔ CHW 转换（最容易踩坑的地方）

```javascript
/**
 * Canvas ImageData (RGBA, HWC) → Float32Array (RGB, CHW, [0,1])
 */
function imageDataToTensor(imageData, width, height) {
    const { data } = imageData;  // Uint8ClampedArray, RGBA 交替排列
    const float32 = new Float32Array(3 * height * width);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcIdx = (y * width + x) * 4;   // RGBA
            const r = data[srcIdx]     / 255.0;
            const g = data[srcIdx + 1] / 255.0;
            const b = data[srcIdx + 2] / 255.0;
            
            // CHW 排列: [R 通道全部, G 通道全部, B 通道全部]
            float32[0 * height * width + y * width + x] = r;
            float32[1 * height * width + y * width + x] = g;
            float32[2 * height * width + y * width + x] = b;
        }
    }
    return new ort.Tensor('float32', float32, [1, 3, height, width]);
}

/**
 * 输出 Tensor (RGB, CHW, [0,1]) → Canvas ImageData (RGBA, HWC)
 */
function tensorToImageData(tensor, width, height) {
    const float32 = tensor.data;
    const rgba = new Uint8ClampedArray(4 * height * width);
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dstIdx = (y * width + x) * 4;
            rgba[dstIdx]     = Math.round(clamp(float32[0 * height * width + y * width + x]) * 255);
            rgba[dstIdx + 1] = Math.round(clamp(float32[1 * height * width + y * width + x]) * 255);
            rgba[dstIdx + 2] = Math.round(clamp(float32[2 * height * width + y * width + x]) * 255);
            rgba[dstIdx + 3] = 255;  // Alpha
        }
    }
    return new ImageData(rgba, width, height);
}

function clamp(v) { return Math.max(0, Math.min(1, v)); }
```

#### 推理主流程

```javascript
async function runSuperResolution(imageElement, modelName, scale) {
    // 1. 图像 → Canvas → Tensor
    const canvas = document.createElement('canvas');
    canvas.width = imageElement.naturalWidth;
    canvas.height = imageElement.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageElement, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const inputTensor = imageDataToTensor(imageData, canvas.width, canvas.height);

    // 2. 加载模型 & 推理
    const session = await ort.InferenceSession.create(
        `models/${modelName}_x${scale}.onnx`,
        { executionProviders: ['wasm'] }   // 或 ['webgpu', 'wasm'] 自动回退
    );
    const results = await session.run({ input: inputTensor });
    const outputTensor = results.output;

    // 3. 输出 → Canvas → 显示
    const outW = canvas.width * scale;
    const outH = canvas.height * scale;
    const outImageData = tensorToImageData(outputTensor, outW, outH);
    
    const outCanvas = document.getElementById('output-canvas');
    outCanvas.width = outW;
    outCanvas.height = outH;
    outCanvas.getContext('2d').putImageData(outImageData, 0, 0);
}
```

### 3.4 UI 设计方案

```
┌──────────────────────────────────────────────────────────────┐
│  🔍 SISR — 图片超分辨率放大工具                    [GitHub]   │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ 设置 ─────────────────────────────────────────────────┐  │
│  │  模型: [ESPCN ▼]     放大倍数: [●2x  ○3x  ○4x]       │  │
│  │  后端: [WASM ▼]  (WebGPU 可选，需浏览器支持)           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────── 拖拽上传区域 ─────────────┐                  │
│  │                                        │                  │
│  │     📁 点击选择图片或拖拽到此处         │                  │
│  │        支持 PNG / JPG / WebP            │                  │
│  │        建议尺寸 ≤ 512×512              │                  │
│  │                                        │                  │
│  └────────────────────────────────────────┘                  │
│                                                              │
│           [ 🚀 开始超分辨率 ]                                │
│                                                              │
│  ┌──── 原图 ─────────┐  ┌──── 超分结果 ─────────┐           │
│  │                    │  │                       │           │
│  │    (LR 预览)       │  │    (SR 预览)          │           │
│  │    128 × 128       │  │    512 × 512          │           │
│  │                    │  │                       │           │
│  └────────────────────┘  └───────────────────────┘           │
│                                                              │
│  📊 推理耗时: 1.23s | 模型大小: 79 KB | PSNR: 32.4 dB       │
│                                                              │
│           [ 💾 下载 SR 图片 ]                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 四、性能优化与边界处理

### 4.1 输入尺寸限制

浏览器内存有限，大图会导致推理极慢甚至卡死。需要在前端做限制：

```javascript
const MAX_PIXELS = 512 * 512;  // 约 26 万像素

function checkImageSize(img) {
    const pixels = img.naturalWidth * img.naturalHeight;
    if (pixels > MAX_PIXELS) {
        // 自动缩小到限制内
        const ratio = Math.sqrt(MAX_PIXELS / pixels);
        const newW = Math.floor(img.naturalWidth * ratio);
        const newH = Math.floor(img.naturalHeight * ratio);
        alert(`图片过大 (${img.naturalWidth}×${img.naturalHeight})，`
            + `已自动缩小到 ${newW}×${newH}`);
        return { width: newW, height: newH, resized: true };
    }
    return { width: img.naturalWidth, height: img.naturalHeight, resized: false };
}
```

### 4.2 分块推理（Tiling）—— 可选进阶

如果想支持更大的图片，可以在前端也实现分块推理（和 Python 的 `Inferencer._forward_tiled` 逻辑相同）：

```javascript
async function tiledInference(session, inputTensor, scale, tileSize = 128, overlap = 8) {
    const [_, C, H, W] = inputTensor.dims;
    const outH = H * scale, outW = W * scale;
    const output = new Float32Array(C * outH * outW);
    const weight = new Float32Array(outH * outW);

    for (let y = 0; y < H; y += tileSize - overlap) {
        for (let x = 0; x < W; x += tileSize - overlap) {
            const yEnd = Math.min(y + tileSize, H);
            const xEnd = Math.min(x + tileSize, W);
            const yStart = yEnd === H ? Math.max(0, H - tileSize) : y;
            const xStart = xEnd === W ? Math.max(0, W - tileSize) : x;

            // 切块 → 推理 → 累加回大图
            const tile = extractTile(inputTensor, yStart, xStart, yEnd, xEnd);
            const result = await session.run({ input: tile });
            accumulateTile(output, weight, result.output, yStart * scale, xStart * scale, scale);
        }
    }
    // 加权平均
    for (let i = 0; i < output.length; i++) {
        output[i] /= Math.max(weight[i % (outH * outW)], 1);
    }
    return new ort.Tensor('float32', output, [1, C, outH, outW]);
}
```

### 4.3 WebWorker（防止 UI 卡死）

推理计算密集，应放在 Web Worker 中执行：

```javascript
// main.js
const worker = new Worker('sr-worker.js');
worker.postMessage({ imageData, model: 'espcn', scale: 4 });
worker.onmessage = (e) => {
    displayResult(e.data.outputImageData);
};

// sr-worker.js
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
self.onmessage = async (e) => {
    const { imageData, model, scale } = e.data;
    // ... 推理逻辑 ...
    self.postMessage({ outputImageData });
};
```

> **注意**：如果使用 WebWorker，就不能在 Worker 内使用 Canvas API。
> 需要在主线程做 ImageData ↔ Tensor 转换，只把 Float32Array 传给 Worker。
> 或者使用 OffscreenCanvas（部分浏览器支持）。

### 4.4 WebGPU 加速（可选）

```javascript
// 优先使用 WebGPU，不支持则回退到 WASM
const session = await ort.InferenceSession.create('models/espcn_x4.onnx', {
    executionProviders: ['webgpu', 'wasm'],
});
```

WebGPU 在 Chrome 113+ 默认可用，推理速度比 WASM 快 3-10 倍。

---

## 五、实施步骤清单

### 阶段 1：模型导出与验证（半天）

| # | 任务 | 产出 |
|---|------|------|
| 1.1 | 编写 `scripts/export_onnx.py` | 导出脚本 |
| 1.2 | 导出 ESPCN x2/x3/x4 为 ONNX | `web/models/espcn_x{2,3,4}.onnx` |
| 1.3 | （可选）导出 FSRCNN x2/x3/x4 | `web/models/fsrcnn_x{2,3,4}.onnx` |
| 1.4 | 编写 `scripts/verify_onnx.py`，验证输出一致性 | 确认 max_diff < 1e-5 |
| 1.5 | 记录每个 ONNX 模型的文件大小 | 预期 ESPCN ~80KB，FSRCNN ~52KB |

### 阶段 2：前端核心功能（1 天）

| # | 任务 | 产出 |
|---|------|------|
| 2.1 | 创建 `web/index.html` 基础 UI | HTML + CSS 布局 |
| 2.2 | 实现图片上传（拖拽 + 点击选择） | 文件读取 → Image → Canvas |
| 2.3 | 实现 HWC ↔ CHW 转换 | `imageDataToTensor()` / `tensorToImageData()` |
| 2.4 | 集成 ONNX Runtime Web | 从 CDN 加载 ORT，创建 Session |
| 2.5 | 实现推理主流程 | 上传 → 推理 → 显示 SR 结果 |
| 2.6 | 实现下载功能 | Canvas → Blob → download link |

### 阶段 3：体验优化（半天）

| # | 任务 | 产出 |
|---|------|------|
| 3.1 | 输入尺寸检查与自动缩放 | 大图预警/自动 resize |
| 3.2 | 加载进度条 + 推理耗时显示 | UX 反馈 |
| 3.3 | 模型 & 倍数选择器 | 下拉框切换 |
| 3.4 | LR/SR 并排对比展示 | 滑块或左右对比 |
| 3.5 | 样例图片（一键 demo） | 内置几张测试图 |

### 阶段 4（可选）：进阶功能

| # | 任务 | 说明 |
|---|------|------|
| 4.1 | 分块推理（Tiling） | 支持更大图片 |
| 4.2 | WebWorker | 推理时 UI 不卡顿 |
| 4.3 | WebGPU 后端 | 加速推理 |
| 4.4 | PSNR 计算（如果有 HR 原图） | 可选的质量评估 |

---

## 六、可能遇到的坑与解决方案

| 坑 | 表现 | 解决 |
|----|------|------|
| **HWC/CHW 搞反** | 输出图像颜色异常（偏色/花屏） | 仔细检查 `imageDataToTensor` 的索引：Canvas 是 RGBA-HWC，模型要 RGB-CHW |
| **归一化不一致** | 输出全黑或全白 | PyTorch 训练时用 `ToTensor()`（[0,255] → [0,1]），前端也必须除以 255 |
| **ONNX 动态尺寸失败** | 报错 shape mismatch | 确保 `torch.onnx.export` 时设置了 `dynamic_axes` |
| **PixelShuffle 导出问题** | 老版本 PyTorch 导出的 Reshape 参数不对 | 用 PyTorch ≥ 1.10，opset ≥ 11 |
| **ConvTranspose2d output_padding** | FSRCNN 在某些尺寸下输出大小不对 | 导出时用固定 opset 11+；或在前端做 crop |
| **CORS 跨域** | 本地 file:// 打开时无法加载 .onnx | 用 `python -m http.server 8080` 本地起个服务器 |
| **浏览器内存不足** | 大图推理时页面崩溃 | 限制最大输入尺寸；实现 tiling |
| **首次加载慢** | WASM 初始化 + 模型下载 | 模型本身很小（<100KB）；ORT WASM ~2MB 会被缓存 |

---

## 七、部署方式

### 方案 A：GitHub Pages（推荐，零成本）

```bash
# 项目根目录
git add web/
git commit -m "feat: add web-based SR tool"
git push origin main

# GitHub 仓库 → Settings → Pages → Source: main → /web
# 访问: https://stevenpang203426.github.io/CV-SISR-2026/
```

### 方案 B：本地运行

```bash
# 任意 HTTP 服务器
cd web/
python -m http.server 8080
# 浏览器打开 http://localhost:8080
```

> **注意**：不能直接双击 `index.html` 打开（file:// 协议下 fetch 会被 CORS 拦截），
> 必须通过 HTTP 服务器访问。

---

## 八、参考资源

| 资源 | 链接 | 用途 |
|------|------|------|
| ONNX Runtime Web 官方文档 | https://onnxruntime.ai/docs/get-started/with-javascript/web.html | API 参考 |
| ORT Web 示例仓库 | https://github.com/microsoft/onnxruntime-inference-examples | 代码参考 |
| WebNN EP 教程 | https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html | WebNN 配置 |
| PyTorch ONNX 导出指南 | https://pytorch.org/docs/stable/onnx.html | 导出细节 |
| WebNN Showcase | https://webnn.io/en/showcase | 在线 Demo 参考 |
| ORT Mobile SR 教程 | https://onnxruntime.ai/docs/tutorials/mobile/superres.html | 移动端扩展参考 |

---

## 九、预期效果

- **模型文件极小**：ESPCN x4 约 80 KB，FSRCNN x4 约 52 KB，几乎瞬间加载
- **推理速度**：512×512 输入在普通笔记本 WASM 下约 1-3 秒，WebGPU 约 0.3-1 秒
- **兼容性**：Chrome / Edge / Firefox / Safari 均支持 WASM 后端
- **零后端依赖**：所有计算在浏览器本地完成，不传输用户图片，隐私安全
