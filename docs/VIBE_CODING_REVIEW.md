# Vibe Coding Review 指南

> 你让 AI 写了代码，跑通了，commit 了。然后呢？
> 本文档定义一套 Review 维度和 checklist，帮助你在 AI 辅助编码（Vibe Coding）后
> 系统性地审查代码质量，避免"能跑就行"变成"能跑就烂"。

---

## 一、为什么 Vibe Coding 更需要 Review？

传统开发中，程序员自己写的代码至少过了一遍脑子。Vibe Coding 的风险在于：

| 风险 | 表现 |
|------|------|
| **理解断层** | AI 生成的代码你不完全理解，出 bug 时不知道从哪查 |
| **过度工程** | AI 倾向于生成"完整"的方案，可能引入你不需要的抽象层 |
| **冗余修复** | 让 AI 修 bug 时，它可能同时改了不相关的地方，引入隐性变更 |
| **风格漂移** | 多次对话后，AI 生成的代码风格、命名习惯、架构决策可能前后不一致 |
| **依赖膨胀** | AI 可能随手引入一个库来解决本可以几行代码搞定的问题 |
| **上下文丢失** | AI 没有全局视角，可能在 A 文件里重复实现了 B 文件已有的功能 |

Review 不是对 AI 的不信任，而是**你作为项目所有者的质量把控**。

---

## 二、Review 的 8 个维度

### 维度 1：架构合规 (Architectural Conformance)

**检查什么**：新代码是否遵守项目已建立的分层结构和依赖方向。

```
项目约定的依赖方向（只能向下依赖，不能反向）：

  入口脚本 (train.py, test.py, infer.py)
      ↓
  核心引擎 (core/)
      ↓
  模型定义 (models/) + 数据加载 (data/)
      ↓
  工具函数 (utils/)
```

**检查项**：

```
[ ] 新文件放对了层（模型逻辑不该出现在 scripts/ 里）
[ ] 依赖方向正确（utils/ 不应该 import core/ 或 models/）
[ ] 入口脚本保持轻薄（只做参数解析 + 调用核心模块，不写业务逻辑）
[ ] 没有跨层直接调用（scripts/ 直接调 models/ 的内部方法而绕过 core/）
```

**反面例子**：
```python
# scripts/eval_blind_sr.py 里直接写了推理循环和指标计算
# 问题：这些逻辑应该在 core/evaluator.py 或 core/inferencer.py 中
for img in images:
    tensor = to_tensor(img)
    output = model(tensor)         # 应该调用 inferencer.run()
    psnr = calc_psnr(output, gt)   # 应该调用 evaluator.evaluate()
```

**正面做法**：
```python
# scripts/eval_blind_sr.py 应该只做编排
inferencer = Inferencer(model, device, scale)
evaluator = Evaluator(metrics=['psnr', 'ssim'])
for img_path in images:
    sr = inferencer.run(img_path)
    scores = evaluator.evaluate(sr, gt)
```

---

### 维度 2：模块耦合度 (Coupling)

**检查什么**：模块之间是否通过清晰的接口交互，还是互相纠缠。

**检查项**：

```
[ ] 模块之间通过公共接口通信，不访问私有成员（_前缀方法）
[ ] 修改一个文件时，不需要同时改另外 3 个文件才能跑通
[ ] 没有循环依赖（A import B，B 又 import A）
[ ] 配置/参数通过注入传递，而非在模块内部硬编码
```

**耦合度自测**：如果你要替换掉 `models/espcn.py`，需要改几个文件？
- 只改 `models/__init__.py` 的注册表 → 松耦合（正确）
- 还要改 `data/dataset.py`、`core/trainer.py`、`scripts/xxx.py` → 紧耦合（有问题）

---

### 维度 3：文件粒度 (File Granularity)

**检查什么**：单个文件是否过长，是否包含了应该拆分的独立职责。

**经验阈值**：

| 文件类型 | 建议上限 | 超过后考虑拆分 |
|----------|----------|---------------|
| 模型定义 (`models/*.py`) | 150 行 | 把子模块（如 ResBlock）拆到 `models/blocks.py` |
| 核心引擎 (`core/*.py`) | 300 行 | 按职责拆分（如 trainer 和 lr_scheduler 分开） |
| 工具函数 (`utils/*.py`) | 200 行 | 按功能域拆分（图像处理 vs 指标计算 vs IO） |
| 脚本 (`scripts/*.py`) | 150 行 | 把可复用逻辑提到 core/ 或 utils/ |
| 入口脚本 (`train.py` 等) | 80 行 | 如果超过，说明业务逻辑泄漏到了入口层 |

**检查项**：

```
[ ] 单文件不超过 300 行（硬上限），理想在 150 行以内
[ ] 一个文件只做一件事（单一职责）
[ ] 如果一个函数超过 50 行，考虑拆分子函数
[ ] 如果一个类有超过 10 个方法，考虑是否承担了过多职责
```

---

### 维度 4：重复与复用 (DRY — Don't Repeat Yourself)

**检查什么**：是否存在复制粘贴的逻辑，是否有已存在的工具函数被重新实现。

**检查项**：

```
[ ] 搜索项目中是否已有类似功能（grep 关键函数名）
[ ] AI 新生成的工具函数是否和 utils/ 中已有的功能重叠
[ ] 多个 scripts/ 脚本中是否有重复的模型加载 / 图像处理代码
[ ] 常量（路径、阈值、默认参数）是否集中定义，还是到处硬编码
```

**高频重复场景**（AI 特别容易犯）：

```python
# AI 在 scripts/eval_blind_sr.py 里又写了一遍图像加载和 tensor 转换
img = Image.open(path).convert('RGB')
tensor = T.ToTensor()(img).unsqueeze(0).to(device)

# 但 core/inferencer.py 里已经有完整的实现了！
# 正确做法：直接调用 inferencer.run(path)
```

**检查命令**：
```bash
# 快速查找可能的重复代码
grep -rn "Image.open" --include="*.py" | grep -v features/ | grep -v __pycache__
grep -rn "ToTensor" --include="*.py" | grep -v features/ | grep -v __pycache__
grep -rn "torch.load" --include="*.py" | grep -v features/ | grep -v __pycache__
```

---

### 维度 5：变更范围 (Change Scope)

**检查什么**：这次 commit 是否只改了它该改的东西，有没有"顺便"改了不相关的地方。

这是 Vibe Coding 最独特的风险点。你让 AI "修一个 bug"，它可能同时：
- 重命名了不相关的变量
- 调整了不相关函数的参数顺序
- 添加了你没要求的"优化"
- 修改了配置文件的默认值

**检查项**：

```
[ ] git diff 只包含与本次任务相关的变更
[ ] 没有不相关的格式调整（空行增删、import 重排序）
[ ] 没有不相关的重构（变量重命名、函数签名变更）
[ ] 没有默认参数被悄悄改了
[ ] 没有 .gitignore / 配置文件被意外修改
```

**检查命令**：
```bash
# 提交前一定要看完整 diff
git diff --stat                  # 哪些文件改了？有没有意外的文件
git diff                         # 逐行看改动
git diff --name-only HEAD~1      # 上一次提交改了哪些文件
```

> **黄金法则**：如果你让 AI 做 task A，但 diff 里出现了和 task A 无关的文件变更，
> 那要么拆成两个 commit，要么 revert 掉无关的改动。

---

### 维度 6：命名与语义 (Naming & Semantics)

**检查什么**：变量名、函数名、文件名是否清晰传达意图，是否和项目现有风格一致。

**项目命名约定**：

| 对象 | 风格 | 示例 |
|------|------|------|
| 文件名 | snake_case | `blind_sr_dataset.py`, `export_onnx.py` |
| 类名 | PascalCase | `SRDataset`, `Inferencer`, `ESPCN` |
| 函数/方法 | snake_case | `build_model()`, `load_checkpoint()` |
| 常量 | UPPER_SNAKE | `MAX_PIXELS`, `WEB_MODELS` |
| 私有方法 | _前缀 | `_forward_tiled()`, `_train_one_epoch()` |
| 模型注册名 | 小写 | `'espcn'`, `'edsr'`, `'rrdb'` |

**检查项**：

```
[ ] 新代码的命名风格与项目现有风格一致
[ ] 函数名是动词短语（do_something），类名是名词（SomeThing）
[ ] 避免泛化命名（data, info, result, tmp, handle, process）
[ ] 缩写要统一（sr 不要有时写 super_res 有时写 sr 有时写 SR）
[ ] AI 生成的注释是否准确（AI 有时会写看似合理但实际错误的注释）
```

---

### 维度 7：错误处理与鲁棒性 (Error Handling)

**检查什么**：代码是否处理了边界情况和异常，还是只考虑了 happy path。

**检查项**：

```
[ ] 文件 I/O 有异常处理（文件不存在、权限不足、格式损坏）
[ ] 输入验证存在（图像尺寸为 0？scale 为负数？空路径？）
[ ] GPU 相关操作有 fallback（CUDA OOM 时的处理、CPU fallback）
[ ] 不会静默吞掉异常（bare except: pass 是禁忌）
[ ] 关键操作有日志输出（模型加载、推理开始/结束、异常信息）
```

**反面例子**（AI 经常生成的）：
```python
try:
    result = model(input)
except:          # bare except — 什么异常都吃掉，出了问题根本不知道
    pass
```

---

### 维度 8：依赖管理 (Dependency Hygiene)

**检查什么**：新代码是否引入了不必要的外部依赖。

**检查项**：

```
[ ] 新 import 的库是否真的需要？能否用标准库实现？
[ ] 引入的新库和已有的库是否功能重叠？（比如已有 Pillow 又引入 cv2 做同一件事）
[ ] 新库是否活跃维护？最近 commit 是什么时候？
[ ] requirements.txt 是否同步更新？
[ ] 新库的体积是否合理？（别为了一个小功能引入一个 500MB 的库）
```

---

## 三、Review 流程：实践操作

### 3.1 每次 AI 生成代码后的快速 Review（5 分钟）

```bash
# Step 1: 看变更范围
git diff --stat

# Step 2: 逐文件看 diff，问自己这三个问题：
#   1. 这个改动是我要求的吗？（变更范围）
#   2. 这段逻辑我能理解吗？（理解断层）
#   3. 这个功能项目里已经有了吗？（重复复用）
git diff

# Step 3: 检查依赖变更
git diff -- requirements.txt
git diff -- "*.py" | grep "^+import\|^+from" | sort -u

# Step 4: 快速语法检查
python -m py_compile path/to/new_file.py
```

### 3.2 分支合并前的深度 Review（30 分钟）

```bash
# Step 1: 生成完整变更报告
git log main..feature/blind-sr --oneline    # 这个分支做了哪些 commit
git diff main..feature/blind-sr --stat      # 改了哪些文件

# Step 2: 架构检查 — 依赖方向
# 检查 utils/ 里有没有 import core/ 或 models/（不应该有）
grep -rn "from core\|import core\|from models\|import models" utils/ --include="*.py"

# 检查 models/ 里有没有 import core/（不应该有）
grep -rn "from core\|import core" models/ --include="*.py"

# Step 3: 重复代码检查
# 看看有没有多处重复的模式
grep -rn "Image.open" --include="*.py" | grep -v features/ | grep -v __pycache__
grep -rn "torch.load" --include="*.py" | grep -v features/ | grep -v __pycache__

# Step 4: 文件粒度检查
find . -name "*.py" -not -path "./features/*" -not -path "./__pycache__/*" \
  -not -path "./wandb/*" -exec wc -l {} + | sort -rn | head -20

# Step 5: 命名一致性 — 快速扫描
grep -rn "class " --include="*.py" | grep -v features/ | grep -v __pycache__
grep -rn "^def " --include="*.py" | grep -v features/ | grep -v __pycache__
```

### 3.3 Review 记录模板

每次做完 Review，在 commit message 或 PR description 中记录：

```
Review: feature/blind-sr

[架构合规]     OK — 新文件均在正确的层
[模块耦合]     OK — 通过 build_model() 注册表交互
[文件粒度]     WARN — data/degradation.py 280 行，接近上限，暂时可接受
[重复复用]     FIX — scripts/eval_blind_sr.py 中的推理循环改为调用 Inferencer
[变更范围]     FIX — 移除了 AI 顺便添加的 .gitignore 变更
[命名语义]     OK — 命名风格一致
[错误处理]     WARN — degradation.py 的 JPEG 压缩缺少质量参数校验，已添加
[依赖管理]     OK — 新增 onnxruntime，已更新 requirements.txt
```

### 3.4 Review 实录：Blind SR 构建（2026-05-25）

本次一次性生成 6 个文件（约 800 行代码），以下是 Review 结果：

```
Review: Blind SR 基础设施 + 退化管道 + 实验脚本

[架构合规]     OK
  models/rrdbnet.py       → models/ 层 ✓
  data/degradation.py     → data/ 层 ✓
  scripts/eval_blind_sr.py, visualize_degradation.py, calc_niqe.py, download_pretrained.py → scripts/ 层 ✓
  eval_blind_sr.py 调用 models + data，不跨层 ✓

[模块耦合]     OK
  RRDBNet 通过 models/__init__.py 注册，'rrdb' 和 'rrdbnet' 两个名称均可
  degradation.py 完全自包含，不依赖 features/BSRGAN/ 的任何模块
  eval_blind_sr.py 同时使用 build_model() 和 RRDBNet.load_pretrained()
    → 因为预训练模型有特殊权重映射，合理的例外

[文件粒度]     REVIEW
  data/degradation.py     ~300 行 — 达到上限，但退化函数高度内聚，暂不拆分
  scripts/eval_blind_sr.py ~310 行 — 3 个实验函数各 ~80 行 + 工具函数，结构清晰
  models/rrdbnet.py       ~150 行 — 合理
  其余脚本均 <150 行 ✓

[重复复用]     WARN
  eval_blind_sr.py 中的 infer_rrdb() 和 infer_our() 逻辑几乎相同
    → 未来可统一为 infer(model, lr, device) 一个函数
  load_hr_images() 和 uint2single 在 eval 和 visualize 中各有调用
    → 来自 data/degradation.py 的导出，属于正常复用

[变更范围]     OK
  .gitignore 只新增了 pretrained/*.pth 一行
  models/__init__.py 只新增了 import 和注册表条目
  无不相关的修改

[命名语义]     OK
  函数名用 snake_case: degradation_bsrgan, add_gaussian_noise ✓
  类名用 PascalCase: DegradationPipeline, RRDBNet ✓
  BSRGAN 原始命名（add_Gaussian_noise）已修正为 add_gaussian_noise

[错误处理]     OK
  download_pretrained.py: URLError 捕获 + 手动下载提示 ✓
  eval_blind_sr.py: 模型/数据缺失时 [SKIP] 不中断 ✓
  degradation.py: 图像过小时抛出 ValueError ✓

[依赖管理]     ACTION NEEDED
  新增 opencv-python, pyiqa — 需同步到 pyproject.toml
  → uv add opencv-python pyiqa
```

---

## 四、Review 维度速查卡片

```
┌────────────────────────────────────────────────────────────────┐
│              Vibe Coding Review — 8 维度速查                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 架构合规    文件在对的层？依赖方向正确？                       │
│  2. 模块耦合    改一个文件要连带改几个？接口清晰？                  │
│  3. 文件粒度    单文件 < 300 行？单函数 < 50 行？单一职责？        │
│  4. 重复复用    项目里已有类似功能吗？有没有复制粘贴？              │
│  5. 变更范围    diff 里有没有你没要求的改动？                      │
│  6. 命名语义    命名清晰、风格统一、注释准确？                     │
│  7. 错误处理    边界情况处理了？异常不会被吞掉？                    │
│  8. 依赖管理    新库真的需要？requirements.txt 同步了？            │
│                                                                │
│  快捷操作：                                                     │
│  git diff --stat         → 变更范围一览                         │
│  git diff                → 逐行审查                             │
│  wc -l *.py | sort -rn   → 文件粒度检查                         │
│  grep "^+import" diff    → 新依赖检查                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 五、术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| 架构合规 | Architectural Conformance | 代码结构是否符合项目约定的分层和依赖规则 |
| 模块耦合 | Coupling | 模块之间的依赖紧密程度；松耦合是目标 |
| 文件粒度 | File Granularity | 单个文件的大小和职责范围是否合理 |
| 重复复用 | DRY (Don't Repeat Yourself) | 避免重复代码，提取共用逻辑 |
| 变更范围 | Change Scope / Blast Radius | 一次修改影响的文件和功能范围 |
| 命名语义 | Naming & Semantics | 标识符命名是否清晰传达意图 |
| 错误处理 | Error Handling / Robustness | 对异常情况的防御能力 |
| 依赖管理 | Dependency Hygiene | 外部依赖的必要性和版本管理 |
| 冗余修复 | Regression Fix / Shotgun Surgery | AI 修 bug 时"顺便"改了不相关的地方 |
| 环境漂移 | Environment Drift | 不同机器上的运行环境逐渐不一致 |
| 理解断层 | Comprehension Gap | 对 AI 生成的代码缺乏足够理解 |
| 风格漂移 | Style Drift | 多次 AI 交互后代码风格逐渐不一致 |
| 依赖膨胀 | Dependency Bloat | 引入了不必要的或过重的外部库 |
| 单一职责 | Single Responsibility Principle | 一个模块/函数只做一件事 |
| 冒烟测试 | Smoke Test | 最基本的"能不能跑起来"的快速测试 |
| 松耦合 | Loose Coupling | 模块间通过接口交互，互不依赖实现细节 |
