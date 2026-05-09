# SISR-Team8 重构记录与经验指南

> 本文档记录了项目重构的完整过程：发现了哪些问题、如何解决、为什么这么改，以及未来遇到类似情况时的通用处理方法。

---

## 一、重构前的项目结构

```
SISR-Team8/
├── train.py            ← 268 行，承担 6 种职责
├── test.py             ← 独立的模型实例化逻辑
├── infer.py            ← 又一套独立的模型实例化逻辑
├── train.sh / test.sh / infer.sh
├── configs/*.yaml
├── models/
│   └── srcnn.py, fsrcnn.py, espcn.py, edsr.py, imdn.py  （无 __init__.py）
├── data/
│   ├── dataset.py
│   ├── Set5/, Set14/, T91/      ← 代码和数据混在一起
│   └── （无 __init__.py）
├── utils/
│   ├── img.py, metrics.py, plot.py, profile.py
│   └── （无 __init__.py）
├── temp/
│   ├── compare.py       ← 有用的工具被埋在临时目录
│   ├── kill.py, plot.py  ← 临时脚本
├── output/              ← 训练产物直接进了 git
├── wandb/               ← 日志直接进了 git
└── （无 .gitignore）
```

---

## 二、发现的问题与解决方案

### 问题 ①：MODELS 字典三处重复

**现象**

`train.py`、`test.py`、`infer.py` 各自定义了一遍模型注册表，且实例化参数不一致：

```python
# train.py — 传了所有参数
'edsr': EDSR(in_channels=3, scale=args.scale, n_feats=64, n_resblocks=16)

# test.py — 只传了部分参数
'edsr': EDSR(in_channels=3, scale=args.scale)

# infer.py — 连 in_channels 都没传
'edsr': EDSR(scale=args.scale)
```

**风险**：新增模型需要改三个文件，改漏一个就出 bug。参数不一致可能导致训练和推理用的模型结构不同。

**解决方案**

在 `models/__init__.py` 中建立统一的注册表和 `build_model()` 工厂函数：

```python
REGISTRY = {
    'srcnn': SRCNN, 'fsrcnn': FSRCNN,
    'espcn': ESPCN, 'edsr': EDSR, 'imdn': IMDN,
}

def build_model(name, **kwargs):
    cls = REGISTRY[name]
    # 自省过滤：只保留目标构造函数接受的参数
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
```

**关键设计**：`inspect` 自省过滤使得调用者可以统一传 `scale=2, in_channels=3`，SRCNN 不接受 `scale` 参数也不会报错——它会被静默忽略。

**通用经验**

> **注册表模式（Registry Pattern）**：当多个入口点需要根据名称创建同一类对象时，永远在一个地方集中注册。判断标准是：如果你在两个以上的文件里看到同一个字典映射，就该提取。
>
> 具体做法：在包的 `__init__.py` 里维护注册表 + 工厂函数，外部只 `from models import build_model`。

---

### 问题 ②：train.py 职责过重（God Object）

**现象**

一个 268 行的文件承担了：配置解析、模型构建、优化器构建、训练循环、验证评估、日志记录、曲线绘制、模型保存 —— 共 6-8 种不同职责。

**风险**：难以单独测试某一环节，修改绘图逻辑可能误触训练逻辑，新人难以理解整体流程。

**解决方案**

按职责拆分到 `core/` 包：

| 文件 | 职责 | 行数 |
|------|------|------|
| `core/config.py` | YAML 配置加载 | ~30 |
| `core/checkpoint.py` | 模型保存/加载 | ~40 |
| `core/evaluator.py` | 验证和完整测试 | ~120 |
| `core/trainer.py` | 训练循环 | ~130 |
| `core/inferencer.py` | 推理逻辑 | ~70 |

重构后的 `train.py` 仅 ~60 行，只负责"组装"：

```python
args = load_config()
model = build_model(args.model, scale=args.scale, in_channels=3).to(device)
optimizer = _build_optimizer(model, args)
trainer = Trainer(model, optimizer, criterion, device, args.scale, scheduler)
trainer.fit(train_loader, val_loader, args)
```

**通用经验**

> **单一职责原则（SRP）的实操判断**：如果一个文件/类需要因为多个不相关的原因被修改（比如"改绘图样式"和"改训练逻辑"是不相关的原因），就说明它承担了过多职责。
>
> 拆分策略：
> 1. 列出文件中所有"做了什么"
> 2. 把这些"做了什么"按相关性分组
> 3. 每组提取为一个独立模块
> 4. 原文件退化为"组装器"——只负责把各模块连接起来

---

### 问题 ③：代码与数据混在同一目录

**现象**

`data/` 目录下既有 `dataset.py` 源代码，又有 Set5/Set14/T91 几百张图片。

**风险**：`git status` 一片绿、仓库体积膨胀、clone 速度变慢、代码审查时噪音大。

**解决方案**

```
data/
├── __init__.py
├── dataset.py          ← 代码留在这里
└── datasets/           ← 数据文件移到子目录
    ├── Set5/
    ├── Set14/
    └── T91/
```

同时在 `.gitignore` 中排除 `data/datasets/`。

**通用经验**

> **代码与数据分离**：代码是"逻辑"，数据是"资源"，两者的变更频率和管理方式完全不同。
>
> 判断标准：如果一个目录下既有 `.py` 文件又有大量非代码文件（图片、CSV、模型权重等），就应该分离。
>
> 通用做法：
> - 代码放在包目录下（有 `__init__.py`）
> - 数据放在子目录或外部目录，通过配置文件引用路径
> - 大文件用 `.gitignore` 排除，或用 Git LFS 管理

---

### 问题 ④：缺少 `__init__.py`

**现象**

`models/`、`data/`、`utils/` 都没有 `__init__.py`。

**风险**：虽然 Python 3 支持隐式命名空间包（不需要 `__init__.py` 也能导入），但这不是标准做法。缺少 `__init__.py` 意味着无法控制包的公开 API，也无法在导入时做初始化。

**解决方案**

为每个包添加 `__init__.py`，明确导出接口：

```python
# utils/__init__.py
from .img import imresize_bicubic, rgb2y
from .metrics import psnr_y, ssim_y
from .profile import count_params_m, profile_flops_g

__all__ = ['imresize_bicubic', 'rgb2y', 'psnr_y', 'ssim_y', ...]
```

**通用经验**

> **始终为 Python 包添加 `__init__.py`**，即使它是空的。这明确声明"这是一个包"，而且可以在 `__init__.py` 中：
> - 定义 `__all__` 控制 `from pkg import *` 的行为
> - 提供便捷导入路径（`from utils import psnr_y` 而不是 `from utils.metrics import psnr_y`）
> - 做包级别的初始化

---

### 问题 ⑤：缺少 `.gitignore`

**现象**

`__pycache__/`、`wandb/`、`output/`（含 `.pt` 权重文件）、`uv.lock` 全部被 git 跟踪。

**风险**：仓库体积膨胀（模型权重可达数百 MB）、`git diff` 充满噪音、协作时冲突频繁。

**解决方案**

创建 `.gitignore`，排除：
- `__pycache__/`、`*.pyc` — Python 编译缓存
- `wandb/` — 实验跟踪日志
- `experiments/`、`output/`、`*.pt` — 训练产物
- `data/datasets/` — 大体积数据集
- `.venv/` — 虚拟环境
- `uv.lock` — 锁文件

**通用经验**

> **项目第一天就创建 `.gitignore`**。每种项目类型（Python、Node、Java……）都有标准的 gitignore 模板，GitHub 提供了 [gitignore 模板集合](https://github.com/github/gitignore)。
>
> 黄金法则：**生成的文件不进仓库**。判断方法：如果一个文件可以通过运行某个命令重新生成（编译、训练、安装依赖），就不应该被 git 跟踪。

---

### 问题 ⑥：临时脚本（temp/）与正式代码混在一起

**现象**

`temp/` 目录下有三个脚本：
- `compare.py` — 有用的多模型对比报告生成器
- `kill.py` — 杀 GPU 进程的运维工具
- `plot.py` — 独立绘图脚本，与 `utils/plot.py` 重名

**风险**：有价值的工具被埋在"临时"目录里，新人不知道它们存在；命名冲突造成混淆。

**解决方案**

- `compare.py` → 提升到 `scripts/compare.py`（正式化）
- `kill.py` — 保留在 `temp/`（纯运维工具）
- Shell 脚本 → 统一移到 `scripts/` 目录

**通用经验**

> **定期审计临时目录**：如果一个"临时"文件存活超过一周且被多次使用，它就不再是临时的——要么正式化（移入正式目录、添加文档），要么删除。
>
> 命名规范：工具脚本统一放在 `scripts/` 或 `tools/`，不要用 `temp/`、`misc/`、`stuff/` 这类模糊命名。

---

## 三、重构后的项目结构

```
SISR-Team8/
├── configs/                    # 配置层：超参数集中管理
│   ├── edsr_x2.yaml ... imdn_x4.yaml
│
├── core/                       # 核心引擎层（新增）
│   ├── __init__.py
│   ├── config.py               # YAML 配置加载
│   ├── checkpoint.py           # 模型保存/加载
│   ├── trainer.py              # 训练循环
│   ├── evaluator.py            # 验证 & 完整测试
│   └── inferencer.py           # 推理逻辑
│
├── data/                       # 数据层
│   ├── __init__.py
│   ├── dataset.py              # SRDataset 类
│   └── datasets/               # 数据集文件（.gitignore）
│       ├── Set5/ · Set14/ · T91/
│
├── models/                     # 模型层：统一注册表
│   ├── __init__.py             # REGISTRY + build_model()
│   ├── srcnn.py · fsrcnn.py · espcn.py · edsr.py · imdn.py
│
├── utils/                      # 工具层
│   ├── __init__.py
│   ├── img.py · metrics.py · profile.py · plot.py
│
├── scripts/                    # Shell/工具脚本
│   ├── train.sh · test.sh · infer.sh · compare.py
│
├── demo/                       # 演示素材
├── experiments/                # 实验输出（.gitignore）
├── train.py                    # 入口：训练（薄包装器）
├── test.py                     # 入口：测试（薄包装器）
├── infer.py                    # 入口：推理（薄包装器）
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 四、依赖关系（自底向上）

```
utils/          ← 零依赖，纯工具函数
  ↑
models/         ← 只依赖 PyTorch
  ↑
data/           ← 依赖 utils.img
  ↑
core/           ← 依赖 models, data, utils
  ↑
train/test/infer.py  ← 依赖 core, models, data（薄入口）
```

关键规则：**依赖只能向下**，不能出现循环依赖。

---

## 五、通用经验清单（Checklist）

以下是从这次重构中提炼的通用规则，适用于任何中小型深度学习项目：

### 项目初始化时

- [ ] 第一天就创建 `.gitignore`
- [ ] 所有 Python 包目录都有 `__init__.py`
- [ ] 代码目录和数据目录物理分离
- [ ] 入口脚本（train/test/infer）保持精简，业务逻辑放在库模块中

### 开发过程中

- [ ] 同一逻辑不在两个以上的地方出现（DRY）——发现重复立即提取
- [ ] 单个文件超过 200 行时考虑拆分
- [ ] "临时"文件存活超过一周就正式化或删除
- [ ] 生成的文件（权重、日志、缓存）不进 git

### 重构时

- [ ] 先画依赖关系图，再动手改代码
- [ ] 从最底层（零依赖）的模块开始改，逐层向上
- [ ] 每改完一层，立即验证 import 链是否完整
- [ ] 保持 git 历史可追溯——先 commit 当前状态，再开始重构

### 新增模型时

重构后只需两步：
1. 在 `models/` 下创建新模型文件（如 `rcan.py`）
2. 在 `models/__init__.py` 的 `REGISTRY` 中添加一行

不需要修改 `train.py`、`test.py`、`infer.py` 中的任何代码。

---

## 六、改动文件清单

| 操作 | 文件 |
|------|------|
| 新增 | `.gitignore` |
| 新增 | `models/__init__.py` |
| 新增 | `data/__init__.py` |
| 新增 | `utils/__init__.py` |
| 新增 | `core/__init__.py`, `core/config.py`, `core/checkpoint.py`, `core/trainer.py`, `core/evaluator.py`, `core/inferencer.py` |
| 新增 | `scripts/train.sh`, `scripts/test.sh`, `scripts/infer.sh`, `scripts/compare.py` |
| 重写 | `train.py`（268行 → ~60行） |
| 重写 | `test.py`（107行 → ~50行） |
| 重写 | `infer.py`（94行 → ~35行） |
| 移动 | `data/Set5,Set14,T91` → `data/datasets/` |
| 修改 | `configs/*.yaml`（`output/` → `experiments/`） |
