# 本地 → GitHub → 云服务器：Git 工作流指南

> 适用场景：本地开发（Claude Code），GitHub 做中转，云服务器（GPU）跑训练/推理。
> 本文档回答三个核心问题，并给出可直接复用的操作流程。

---

## 一、你的架构全貌

先看清楚三台机器各自的角色：

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   本地电脑     │  push   │    GitHub     │  pull   │   云服务器    │
│  (开发环境)    │ ──────▶ │  (中央仓库)   │ ◀────── │  (训练环境)   │
│              │         │              │         │              │
│ • Claude Code│  pull   │ • 你的个人仓库 │  push   │ • GPU 训练    │
│ • 写代码      │ ◀────── │ • 所有分支     │ ──────▶ │ • 跑实验      │
│ • 主力开发    │         │ • 唯一的真相源 │         │ • Hotfix      │
└──────────────┘         └──────────────┘         └──────────────┘
        │                                                 │
        │  remote: origin = 你的 GitHub 仓库               │  remote: origin = 同一个 GitHub 仓库
        │  remote: real-esrgan = 别人的仓库（参考用）        │  不需要别人的仓库
        │  remote: bsrgan = 别人的仓库（参考用）            │
        └─────────────────────────────────────────────────┘
```

**核心原则**：
- **GitHub 是唯一的中转站**，本地和云服务器都只跟你自己的 GitHub 仓库交互。
- **本地是主力开发端**，大部分代码在本地写。
- **云服务器是训练端，也可以 push**——但只推送训练产出（日志、轻量配置改动、hotfix），不做大规模开发。

---

## 二、问题一：云服务器怎么拉取 feature 分支？

### 答案：直接 fetch + checkout，不需要手动"新建分支"

很多人以为要在云服务器上先 `git branch xxx`，再 `git pull`。不需要。Git 的 `checkout` 命令在检出一个本地不存在但远程存在的分支时，会**自动创建追踪分支**。

### 操作流程

**前提**：云服务器已经 clone 过你的仓库。

```bash
# ── 场景 A：第一次在云服务器上拉取某个 feature 分支 ──

# 1. 先拉取远程最新的分支信息（不会修改任何本地文件）
git fetch origin

# 2. 直接 checkout 远程分支，Git 自动创建本地追踪分支
git checkout feature/blind-sr
#   等价于: git checkout -b feature/blind-sr origin/feature/blind-sr
#   Git 会输出: Branch 'feature/blind-sr' set up to track 'origin/feature/blind-sr'.

# 就这样，你现在在 feature/blind-sr 上了，代码和本地推送的一样。
```

```bash
# ── 场景 B：分支已经在云服务器上存在，本地有新提交要同步 ──

# 1. 切到目标分支
git checkout feature/blind-sr

# 2. 拉取最新代码
git pull origin feature/blind-sr

# 或者更简洁（因为已经设置了追踪关系）：
git pull
```

```bash
# ── 场景 C：一次性同步所有分支信息 ──

# fetch --all 拉取所有远程分支的元数据
git fetch --all

# 查看所有远程分支
git branch -r

# 想切哪个就 checkout 哪个
git checkout feature/web-sr-tool
git checkout feature/liif
```

### 关键认知

| 误区 | 正确理解 |
|------|----------|
| "要先 `git branch xxx` 再 `git pull`" | 不需要。`git checkout xxx` 会自动从远程创建追踪分支 |
| "要保持 git 树完全一致" | 不需要。云服务器只 checkout 你需要跑的分支就行 |
| "每次都要在 main 上 pull 再 merge" | 不需要。直接 `git checkout 分支名` + `git pull` |

> **Tips：`git fetch` vs `git pull` 的区别**
>
> `git fetch` 只下载远程的提交和分支信息到本地的 `.git/objects`，不改动你的工作目录。
> 就像"查看邮件列表但不打开任何一封"。
>
> `git pull` = `git fetch` + `git merge`，不但下载还会把远程的变更合并到你当前分支。
> 就像"查看邮件并自动回复了"。
>
> 在云服务器上推荐先 `git fetch`，看看有什么变更，再决定 checkout 哪个分支。

---

## 三、问题二：云服务器需要添加别人的远程仓库吗？

### 答案：不需要。只添加你自己的 GitHub 仓库。

理解这个问题的关键在于：**别人仓库的代码已经通过你本地的操作，被合并/放进了你的分支，然后 push 到了你的 GitHub 仓库**。

### 数据流向图

```
别人的仓库                   你的 GitHub 仓库                云服务器
(Real-ESRGAN)               (你的 origin)
     │                            │                           │
     │  本地 git fetch             │                           │
     ├────────────┐               │                           │
     │            ▼               │                           │
     │      你的本地电脑            │                           │
     │     git read-tree          │                           │
     │     或 merge 进分支         │                           │
     │            │               │                           │
     │            │  git push     │                           │
     │            └──────────────▶│                           │
     │                            │  git pull                  │
     │                            ├──────────────────────────▶│
     │                            │                           │
     │     别人的代码已经在        │  云服务器只需要            │
     │     你的分支里了            │  跟你的 GitHub 交互        │
     └────────────────────────────┴───────────────────────────┘
```

### 云服务器的 remote 设置

```bash
# 云服务器上只需要一个 remote
git remote -v
# origin  https://github.com/你的用户名/CV-SISR-2026.git (fetch)
# origin  https://github.com/你的用户名/CV-SISR-2026.git (push)

# 不需要：
# git remote add real-esrgan https://github.com/xinntao/Real-ESRGAN.git    ← 不需要
# git remote add bsrgan https://github.com/cszn/BSRGAN.git                ← 不需要
```

### 为什么不需要？

因为你在本地已经完成了"整合"工作：

1. 你在本地 `git remote add real-esrgan ...` 并 `git fetch real-esrgan`
2. 你用 `git read-tree --prefix=features/Real-ESRGAN/` 把代码放进了你的分支
3. 你 `git push origin feature/xxx` 到你的 GitHub

到这一步，**别人的代码已经成为了你仓库的一部分**。云服务器只需要从你的仓库 pull 就行了。

### 唯一需要添加别人 remote 的场景

如果别人的仓库频繁更新，你需要在云服务器上也直接 fetch 别人的最新代码——但在你的场景中，Real-ESRGAN 和 BSRGAN 都是稳定的参考仓库，不会频繁变动，所以完全不需要。

---

## 四、问题三：这和多人协作有什么区别？

### 你的模式 vs 多人协作

```
你的模式（单人 + 多机器）               多人协作模式
─────────────────────────              ─────────────────────────
• 只有你一个人写代码                    • 多人同时写代码
• 多台机器（本地 + 云服务器）            • 每人一台机器
• 云服务器只读（pull，不 push）          • 每人都可能 push
• 不会有合并冲突                        • 经常出现合并冲突
• 不需要 Pull Request / Code Review    • 通常需要 PR + Review
• 分支策略简单                          • 需要约定分支命名规范
```

### 详细对比

| 维度 | 你的场景（单人多机） | 多人协作 |
|------|-------------------|---------:|
| **remote 数量** | 本地多个（origin + 别人仓库），云服务器只有 origin | 每人只有 origin（指向共享仓库），有时加 upstream（fork 模式） |
| **push 权限** | 只有本地 push | 每人都能 push（或通过 PR） |
| **冲突可能** | 几乎为零（只有你在写） | 常见，需要 merge/rebase 解决 |
| **分支保护** | 不需要 | main/master 通常锁定，只能通过 PR 合并 |
| **Code Review** | 不需要 | 必须，通过 GitHub PR 页面 |
| **CI/CD** | 不需要 | 通常有 GitHub Actions 自动测试 |
| **同步频率** | 你决定什么时候 push/pull | 需要频繁 pull 避免分支分叉太远 |

### 如果将来变成多人协作怎么办？

你当前的流程只需要做两个调整：

```
1. 云服务器不再只是"只读"
   → 如果队友也需要在云服务器上改代码并 push，需要设置 SSH key
   → 建议：依然保持"云服务器只 pull"，改动都在本地做

2. 引入 Pull Request 流程
   → 每人在自己的 feature 分支开发
   → 完成后提 PR 到 main
   → 至少一个人 review 后再 merge
   → GitHub 上操作，不影响你的本地/云服务器流程
```

---

## 五、完整操作手册

### 5.1 初始设置（只做一次）

#### 本地电脑

```bash
# 你的个人仓库作为 origin（应该已经设好了）
git remote add origin https://github.com/你的用户名/CV-SISR-2026.git

# 别人的仓库作为参考源（只在本地需要）
git remote add real-esrgan https://github.com/xinntao/Real-ESRGAN.git
git remote add bsrgan https://github.com/cszn/BSRGAN.git
```

#### 云服务器

```bash
# 方式一：首次 clone（推荐用 SSH，速度更快且不用每次输密码）
git clone git@github.com:你的用户名/CV-SISR-2026.git
cd CV-SISR-2026

# 方式二：如果用 HTTPS
git clone https://github.com/你的用户名/CV-SISR-2026.git
cd CV-SISR-2026

# 确认 remote 只有一个
git remote -v
# origin  git@github.com:你的用户名/CV-SISR-2026.git (fetch)
# origin  git@github.com:你的用户名/CV-SISR-2026.git (push)
```

### 5.2 日常开发流程

```
本地开发                          GitHub                         云服务器
────────                         ──────                         ────────
1. 写代码
   git add .
   git commit -m "feat: ..."
   
2. 推送到 GitHub
   git push origin feature/xxx
                                 3. 分支已更新
                                                                4. 拉取并切换
                                                                   git fetch origin
                                                                   git checkout feature/xxx
                                                                   git pull
                                                                
                                                                5. 跑训练/实验
                                                                   python train.py ...
```

### 5.3 常用命令速查

```bash
# ═══ 在云服务器上 ═══

# 查看所有远程分支
git fetch origin && git branch -r

# 切换到某个 feature 分支（首次会自动创建追踪）
git checkout feature/blind-sr

# 同步最新代码（已在目标分支上）
git pull

# 查看当前在哪个分支
git branch

# 查看本地所有分支
git branch -a

# 不小心改了云服务器上的文件，想恢复到远程版本
git checkout -- .
# 或者更彻底：
git reset --hard origin/feature/blind-sr
```

### 5.4 分支管理建议

```bash
# 本地的分支命名约定
main                        # 主干，稳定版本
feature/web-sr-tool         # Web 超分工具
feature/blind-sr            # Blind SR 实验
feature/liif                # LIIF 任意尺度超分

# 在云服务器上，你不需要所有分支
# 只 checkout 你当前需要跑的那个
git checkout feature/blind-sr
```

---

## 六、云服务器上的 Push 策略

### 6.1 什么该 push，什么不该 push

云服务器会产出很多文件，但不是所有文件都应该 push 到 GitHub：

| 产出类型 | 是否 push | 原因 |
|----------|----------|------|
| Bug hotfix（修改 .py 脚本） | **push** | 代码变更必须版本控制 |
| 训练配置调整（.yaml / .py 中的超参） | **push** | 记录实验配置，便于复现 |
| 训练日志（.log / metrics.json） | **push** | 轻量文本文件，有记录价值 |
| 轻量实验结果（对比图、表格 .csv） | **push** | 轻量文件，有展示价值 |
| 模型权重（.pt / .pth） | **不 push** | 太大（几十 MB ~ 几 GB），用 .gitignore 排除 |
| 数据集 | **不 push** | 太大，且通常已有独立来源 |
| 大量推理输出图片 | **不 push** | 太大，挑几张代表性的 push 即可 |
| wandb 本地缓存 | **不 push** | 已同步到 wandb 云端，本地缓存无需提交 |

### 6.2 .gitignore 策略（保护 push 安全）

确保你的 `.gitignore` 覆盖了所有大文件：

```bash
# .gitignore 中应该有这些规则
experiments/*/best.pt
experiments/*/latest.pt
experiments/*/test/
experiments/*/infer/
pretrained/*.pth
data/DIV2K*/
data/*.tar
data/*.zip
wandb/
__pycache__/
*.pyc
```

> **Tips：push 前检查有没有大文件混进去**
>
> ```bash
> # 列出暂存区中超过 1MB 的文件
> git diff --cached --name-only | xargs -I{} sh -c 'size=$(stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null); if [ "$size" -gt 1048576 ]; then echo "$1 ($(($size/1024))KB)"; fi' _ {}
>
> # 更简单的方式：看 git status 中有没有 .pt / .pth / 图片目录
> git status
> ```

### 6.3 云服务器 Push 的日常操作

```bash
# ═══ 场景 A：在云服务器上修了一个 bug ═══

# 1. 确认在正确分支
git branch
# * feature/blind-sr

# 2. 查看改了什么
git diff
git status

# 3. 只 add 你改的文件（不要 git add .，避免误提交大文件）
git add scripts/eval_blind_sr.py
git commit -m "fix: 修复评估脚本中的路径错误"

# 4. push 到 GitHub
git push origin feature/blind-sr

# 5. ⚠️ 重要：回到本地后记得 pull
# （在本地电脑上）
git checkout feature/blind-sr
git pull origin feature/blind-sr
```

```bash
# ═══ 场景 B：训练完成，提交日志和配置 ═══

# 1. 只添加轻量文件
git add experiments/blind_sr/train.log
git add experiments/blind_sr/metrics.json
git add configs/blind_sr_finetune.yaml

# 2. 不要添加权重文件（确认 .gitignore 生效）
git status    # 确认 best.pt 等文件显示为 "Untracked" 或被忽略

# 3. commit + push
git commit -m "results: blind-sr 训练完成，PSNR=28.5dB"
git push origin feature/blind-sr
```

```bash
# ═══ 场景 C：调整了超参，想记录下来 ═══

git add configs/blind_sr.yaml train.py    # 只添加配置和脚本变更
git commit -m "tune: 调整 lr=1e-5, batch_size=8"
git push origin feature/blind-sr
```

### 6.4 双向同步的冲突处理

当本地和云服务器都有 push 时，拉取时可能出现冲突：

```bash
# ═══ 云服务器上 pull 时遇到冲突 ═══

git pull origin feature/blind-sr
# CONFLICT (content): Merge conflict in scripts/eval_blind_sr.py

# 方案一：如果云服务器上的改动不重要，直接用远程版本
git checkout --theirs scripts/eval_blind_sr.py
git add scripts/eval_blind_sr.py
git commit -m "merge: 采用本地版本的 eval 脚本"

# 方案二：如果云服务器上的改动重要，保留云服务器版本
git checkout --ours scripts/eval_blind_sr.py
git add scripts/eval_blind_sr.py
git commit -m "merge: 保留云服务器版本的 eval 脚本"

# 方案三：手动合并（打开文件，解决 <<<< ==== >>>> 标记）
vim scripts/eval_blind_sr.py
git add scripts/eval_blind_sr.py
git commit -m "merge: 手动合并 eval 脚本冲突"
```

```bash
# ═══ 本地 pull 时遇到冲突（同理） ═══

git pull origin feature/blind-sr
# 如果有冲突，同上三种方案
```

### 6.5 避免冲突的最佳实践

冲突的根源是"两边同时改了同一个文件的同一个区域"。遵循以下规则可以大幅减少冲突：

```
规则 1：分工明确
  本地 → 写代码、改架构、加新功能
  云服务器 → 调参数、修 hotfix、提交训练产出

规则 2：Push 前先 Pull
  无论在哪台机器上，push 前先 pull，确保基于最新代码改动
  git pull origin feature/blind-sr    # 先拉
  # ... 改代码 ...
  git add & commit & push             # 再推

规则 3：不要两边同时改同一个文件
  如果本地正在大改 train.py，云服务器上就不要动 train.py
  云服务器只改云服务器特有的东西（路径配置、超参、小 bug）

规则 4：commit 粒度要小
  不要攒一堆改动一次性 commit
  每个小改动单独 commit，冲突时更容易定位和解决
```

---

## 七、常见问题

### Q: 本地有多个 remote，push 时怎么确保推到自己的仓库？

```bash
# 显式指定 remote 名称
git push origin feature/blind-sr     # ← 推到你的仓库
# 而不是
git push real-esrgan feature/xxx     # ← 这会推到别人仓库（而且你也没权限）

# 查看每个 remote 指向哪里
git remote -v
```

### Q: 想在云服务器上看别人仓库的某个文件/分支怎么办？

**不要在云服务器上添加别人的 remote。** 两个替代方案：

```bash
# 方案一：在本地整合好再 push（推荐）
# 本地操作：
git fetch real-esrgan
git checkout -b temp/check-realesrgan real-esrgan/master
# 看完后删掉或合并需要的部分到你的分支

# 方案二：直接在云服务器上临时下载（不走 git）
wget https://raw.githubusercontent.com/xinntao/Real-ESRGAN/master/README.md
```

### Q: 不小心把大文件 push 到了 GitHub 怎么办？

```bash
# 如果还没被别人 pull（只有你一个人用），可以重写历史
# ⚠️ 危险操作，确保你理解后果

# 方案一：如果是最近一次 commit 里的
git rm --cached experiments/blind_sr/best.pt
echo "experiments/*/best.pt" >> .gitignore
git commit --amend -m "fix: 移除误提交的权重文件"
git push origin feature/blind-sr --force

# 方案二：如果是更早的 commit，用 git filter-branch 或 BFG Repo-Cleaner
# 参考: https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage
```

### Q: 云服务器上改了代码但还不想 commit，想先 pull 最新代码？

```bash
# 用 stash 暂存你的改动
git stash                           # 暂存未提交的改动
git pull origin feature/blind-sr    # 拉取最新代码
git stash pop                       # 恢复你的改动

# 如果 pop 时有冲突，手动解决
```

---

## 七、跨分支同步文件

你在某个分支上修改了文件，想把这些改动同步到其他分支。根据改动是否已经 commit，有不同的做法。

### 7.1 改动还没 commit（工作区有未提交的修改）

这是最简单的情况——未提交的改动不属于任何分支，切换分支时 Git 会尝试带过去。

```bash
# 当前在 feature/blind-sr，修改了 utils/img.py 和 data/dataset.py，还没 commit

# 方案一：直接切分支（如果没有冲突，改动会自动带过去）
git checkout feature/web-sr-tool
# 现在 feature/web-sr-tool 上也能看到那些修改，直接 commit 即可
git add utils/img.py data/dataset.py
git commit -m "fix: 修复图像处理工具函数"

# 方案二：如果切分支时报冲突，用 stash 暂存
git stash                           # 暂存所有未提交的改动
git checkout feature/web-sr-tool    # 切到目标分支
git stash pop                       # 恢复暂存的改动
# 如果 pop 时有冲突，手动解决后 git add + commit
```

> **Tips：`git stash` 是一个"临时抽屉"**
>
> `git stash` 把你的未提交改动存到一个栈里，工作区变干净。
> `git stash pop` 把最近一次 stash 的内容取出来应用到当前分支。
> `git stash list` 可以查看栈里有多少暂存。
> `git stash drop` 丢弃最近一次暂存。
>
> 这是跨分支搬运未提交改动最安全的方式。

### 7.2 改动已经 commit，想把某几个 commit 搬到其他分支

#### 场景 A：搬运整个 commit → `git cherry-pick`

```bash
# 当前在 feature/blind-sr，刚提交了一个修复
git log --oneline -3
# a1b2c3d fix: 修复 degradation.py 的噪声范围
# e4f5g6h feat: 添加退化管道
# i7j8k9l initial commit

# 想把 a1b2c3d 这个修复也应用到 feature/web-sr-tool
git checkout feature/web-sr-tool
git cherry-pick a1b2c3d

# 完成！这个 commit 的所有改动（包括 commit message）都搬过来了
# 注意：会生成一个新的 commit hash，但内容一样
```

```bash
# 搬运多个连续的 commit
git cherry-pick a1b2c3d e4f5g6h      # 列出多个 hash

# 搬运一个范围（左开右闭：不包含起点，包含终点）
git cherry-pick e4f5g6h..a1b2c3d     # 从 e4f5g6h 之后到 a1b2c3d
```

```bash
# cherry-pick 时遇到冲突
git cherry-pick a1b2c3d
# CONFLICT (content): Merge conflict in utils/img.py
# 手动解决冲突后：
git add utils/img.py
git cherry-pick --continue

# 或者放弃这次 cherry-pick
git cherry-pick --abort
```

#### 场景 B：只搬运特定文件（不搬整个 commit）→ `git checkout -- 文件`

```bash
# 想把 feature/blind-sr 分支上的某个文件直接覆盖到当前分支
git checkout feature/web-sr-tool           # 先切到目标分支
git checkout feature/blind-sr -- utils/img.py data/dataset.py
# 这会直接把那两个文件替换为 feature/blind-sr 上的版本
# 改动已经在暂存区，直接 commit
git commit -m "sync: 从 blind-sr 同步工具函数"
```

> **注意**：这是"覆盖"，不是"合并"。目标分支上这些文件原来的内容会被完全替换。

#### 场景 C：把一个分支的所有改动合并过来 → `git merge`

```bash
# 想把 feature/blind-sr 的所有改动合并到 main
git checkout main
git merge feature/blind-sr
# 如果有冲突，解决后 git add + git commit
```

### 7.3 场景决策表

| 你想做什么 | 改动状态 | 用什么命令 |
|-----------|---------|-----------|
| 把未保存的改动带到另一个分支 | 未 commit | `git stash` → 切分支 → `git stash pop` |
| 把某个 commit 复制到另一个分支 | 已 commit | `git cherry-pick <hash>` |
| 把某个分支上的特定文件拿过来 | 已 commit | `git checkout <分支> -- <文件路径>` |
| 把整个分支的改动合并过来 | 已 commit | `git merge <分支>` |
| 把多个 commit 整理后搬过来 | 已 commit | `git cherry-pick <hash1> <hash2> ...` |

### 7.4 实际例子：你的项目中最常见的场景

```bash
# ═══ 场景：你在 main 上修了一个公共工具函数 bug，想同步到所有 feature 分支 ═══

# 1. 在 main 上修复并提交
git checkout main
# ... 修改 utils/img.py ...
git add utils/img.py
git commit -m "fix: 修复 imresize_bicubic 边界处理"
# 记下 commit hash，假设是 abc1234

# 2. 同步到每个 feature 分支
git checkout feature/blind-sr
git cherry-pick abc1234

git checkout feature/web-sr-tool
git cherry-pick abc1234

git checkout feature/liif
git cherry-pick abc1234

# 3. 全部 push
git push origin main feature/blind-sr feature/web-sr-tool feature/liif
```

```bash
# ═══ 场景：你在 feature/blind-sr 上写了 data/degradation.py，
#           其他分支也需要这个文件 ═══

git checkout feature/web-sr-tool
git checkout feature/blind-sr -- data/degradation.py
git commit -m "sync: 引入退化管道模块"

git checkout feature/liif
git checkout feature/blind-sr -- data/degradation.py
git commit -m "sync: 引入退化管道模块"
```

```bash
# ═══ 场景：feature/blind-sr 开发完成，合并回 main ═══

git checkout main
git merge feature/blind-sr
git push origin main

# 合并后可以删除 feature 分支（可选）
git branch -d feature/blind-sr              # 删本地
git push origin --delete feature/blind-sr   # 删远程
```

---

## 八、环境同步

代码同步了但环境没同步，pull 下来照样跑不了。环境同步和代码同步同等重要。

### 8.1 问题：环境漂移 (Environment Drift)

| 症状 | 原因 |
|------|------|
| 本地能跑，云服务器报 `ModuleNotFoundError` | 本地新装了库，没同步 pyproject.toml |
| 云服务器跑出的结果和本地不一样 | PyTorch / CUDA 版本不一致 |
| `uv sync` 报冲突 | pyproject.toml 里的版本约束和云服务器上 CUDA 版本不兼容 |

### 8.2 环境同步流程

本项目使用 **uv** 管理依赖（依赖声明在 `pyproject.toml` 中，而非 requirements.txt）。

#### 8.2.1 永久配置清华源镜像（每台机器只需一次）

```bash
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

配置后所有 `uv add` / `uv pip install` 命令都会自动走清华源，无需每次手动指定 `--default-index`。

#### 8.2.2 日常依赖管理

```bash
# ═══ 安装新依赖 ═══

# uv add 会自动更新 pyproject.toml 和 uv.lock
uv add opencv-python-headless
uv add pyiqa

# 提交依赖变更
git add pyproject.toml uv.lock
git commit -m "deps: add opencv-python-headless for degradation pipeline"
```

```bash
# ═══ 云服务器：每次 pull 后同步依赖 ═══

git pull origin feature/blind-sr

# 检查依赖是否有变更
git diff HEAD~1 -- pyproject.toml

# 如果有变更，同步安装
uv sync
```

### 8.3 pyproject.toml vs requirements.txt

本项目用 `pyproject.toml` 声明依赖，`uv.lock` 锁定精确版本。不再需要手动维护 requirements.txt。

| | requirements.txt + pip | pyproject.toml + uv |
|--|--|--|
| 添加依赖 | 手动编辑文件 | `uv add xxx` 自动写入 |
| 版本锁定 | `pip freeze` 手动导出 | `uv.lock` 自动生成 |
| 安装依赖 | `pip install -r requirements.txt` | `uv sync` |
| CUDA 隔离 | 需手动排除 nvidia-* | pyproject.toml 只列主依赖，PyTorch/CUDA 单独装 |

> **注意**：如果项目根目录仍存在旧的 `requirements.txt`，它仅作为参考，实际以 `pyproject.toml` 为准。

### 8.4 CUDA / PyTorch 版本不一致怎么办

本地和云服务器的 CUDA 版本经常不一样（比如本地 CUDA 12.4、云服务器 CUDA 11.8）。
核心原则：**pyproject.toml 里不要写 CUDA 相关的包**（`nvidia-*`、`cuda-*`、`triton`）。

```bash
# 云服务器上单独安装 PyTorch（根据 CUDA 版本选）
# 查看云服务器 CUDA 版本
nvidia-smi

# 根据 CUDA 版本安装对应的 PyTorch
# CUDA 11.8:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 然后同步其他依赖
uv sync
```

---

## 九、Checklist：每次操作前/后的检查清单

### 9.1 本地 Push 前检查

```
[ ] 分支检查
    [ ] git branch — 确认在正确的分支上
    [ ] git status — 没有遗漏的未暂存文件
    [ ] git diff --staged — 确认暂存区内容是你想提交的
    [ ] git log --oneline -3 — 确认最近的 commit message 有意义

[ ] 代码检查
    [ ] 新增的 .py 文件有没有语法错误（python -c "import ast; ast.parse(open('xxx.py').read())"）
    [ ] 没有把密钥、token、绝对路径硬编码进代码
    [ ] .gitignore 是否覆盖了不该提交的文件（checkpoints、数据集、__pycache__）

[ ] 环境检查
    [ ] 新装了依赖？→ uv add xxx，提交 pyproject.toml 和 uv.lock
    [ ] 改了配置文件格式？→ 确认云服务器上的脚本还兼容

[ ] 数据检查
    [ ] 没有把大文件（.pt 权重、图片数据集）提交到 git
    [ ] experiments/ 下的输出没有误提交（检查 .gitignore）
```

### 9.2 云服务器 Pull 后检查

```
[ ] 分支检查
    [ ] git branch — 确认在正确的分支上
    [ ] git log --oneline -3 — 确认和本地 push 的一致

[ ] 环境检查
    [ ] git diff HEAD~1 -- pyproject.toml — 检查依赖是否有变更
    [ ] 有变更 → uv sync
    [ ] python -c "import torch; print(torch.__version__, torch.cuda.is_available())" — CUDA 正常

[ ] 运行检查
    [ ] 快速冒烟测试：python train.py --help 或 python test.py --help 不报错
    [ ] 如果改了数据路径 / 配置文件，确认云服务器上路径存在
```

### 9.3 分支合并前检查

```
[ ] 目标分支是否最新
    [ ] git checkout main && git pull origin main

[ ] 功能分支是否完成
    [ ] 所有 TODO / FIXME 已处理或记录
    [ ] 新功能有对应的文档或 README 说明

[ ] 合并测试
    [ ] git merge --no-commit --no-ff feature/xxx — 先试合并看看有没有冲突
    [ ] 有冲突 → 解决后测试
    [ ] 没冲突 → git merge --abort，然后正式 merge

[ ] 清理
    [ ] 合并后的分支是否还需要？不需要 → 删除本地和远程分支
```

### 9.4 新建 Feature 分支前检查

```
[ ] 起点检查
    [ ] 从哪个分支 checkout？（通常从 main）
    [ ] main 是否是最新的？→ git pull origin main

[ ] 命名规范
    [ ] feature/功能名   — 新功能
    [ ] fix/问题描述     — Bug 修复
    [ ] docs/文档名      — 文档变更
    [ ] experiment/实验名 — 实验分支（可能不合并回 main）
```

### 9.5 云服务器 Push 前检查

```
[ ] 分支检查
    [ ] git branch — 确认在正确的分支上
    [ ] git pull origin <branch> — 先拉最新，避免冲突

[ ] 文件检查
    [ ] git status — 只有你打算提交的文件
    [ ] git diff --stat — 确认改动文件列表合理
    [ ] 不要 git add . — 逐文件添加，避免误提交大文件
    [ ] git diff --staged --stat — 暂存区没有 .pt / .pth / 数据集文件

[ ] 大文件检查（最重要！）
    [ ] 权重文件（*.pt, *.pth, *.onnx）没有被 add
    [ ] 数据集、图片结果没有被 add
    [ ] .gitignore 已覆盖上述文件类型
    [ ] find . -path ./.git -prune -o -size +10M -print — 确认没有大文件待提交

[ ] Commit 消息
    [ ] 消息清晰说明了改动内容
    [ ] 前缀规范：fix: / results: / tune: / config:

[ ] Push
    [ ] git push origin <branch>
    [ ] 回到本地后记得 git pull origin <branch>
```

---

## 十、一张图总结

```
┌──────────────────────────────────────────────────────────────────┐
│                         你的工作流全貌                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  本地（主要开发）                                                 │
│  ┌──────────────────────────────────────────────┐               │
│  │  remote: origin ──────▶ 你的 GitHub           │               │
│  │  remote: real-esrgan ──▶ 别人仓库（只读参考）   │               │
│  │  remote: bsrgan ──────▶ 别人仓库（只读参考）    │               │
│  │                                               │               │
│  │  工作：写代码 → commit → push origin           │               │
│  └──────────────────────────────────────────────┘               │
│                          │                  ▲                    │
│                git push  │                  │  git pull          │
│                          ▼                  │                    │
│  GitHub（双向中转站）                                             │
│  ┌──────────────────────────────────────────────┐               │
│  │  你的个人仓库                                  │               │
│  │  包含所有分支 + 已整合的别人代码                │               │
│  └──────────────────────────────────────────────┘               │
│                          │                  ▲                    │
│                git pull  │                  │  git push（轻量）   │
│                          ▼                  │                    │
│  云服务器（训练 + 轻量 push）                                     │
│  ┌──────────────────────────────────────────────┐               │
│  │  remote: origin ──────▶ 你的 GitHub（唯一）    │               │
│  │                                               │               │
│  │  工作：pull → 跑实验 → 有限 push               │               │
│  │  可 push：hotfix、训练日志、配置、超参调整       │               │
│  │  不 push：权重 .pt/.pth、数据集、大量图片        │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  规则：                                                           │
│  ✓ 本地：可以有多个 remote（你的 + 别人的），主要开发在本地          │
│  ✓ GitHub：双向中转站，本地和云服务器都可 push / pull              │
│  ✓ 云服务器：只有一个 remote（你的 GitHub），双向同步              │
│  ✓ 云服务器可 push：hotfix、训练日志、配置调整、metrics            │
│  ✗ 云服务器不 push：权重文件、数据集、大量图片结果                  │
│  ✗ 云服务器：不需要添加别人的 remote                              │
│  ⚠ 核心原则：push 前先 pull，避免两边同时改同一个文件              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
