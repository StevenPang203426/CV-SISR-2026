# Git 多仓库融合指南

> 如何把多个 GitHub 仓库合并到一个仓库中，保留各自的完整 Git 历史。

---

## 本次实际场景

| 角色 | 仓库 | 说明 |
|------|------|------|
| **主仓库 (main)** | `StevenPang203426/CV-SISR-2026` | 已完成的主干功能 |
| **Feature 1** | `xinntao/Real-ESRGAN` | 真实场景超分辨率 |
| **Feature 2** | `cszn/BSRGAN` | 盲超分辨率 |

**融合策略**：独立子目录 —— 主仓库文件保持不动，每个 feature 仓库放入独立文件夹。

**融合后的目录结构**：

```
CV-SISR-2026/              ← 主仓库根目录（所有原文件保持原位）
├── configs/
├── core/
├── data/
├── models/
├── utils/
├── train.py
├── test.py
├── infer.py
├── ...                    ← 主仓库原有文件不动
│
├── features/              ← 新建目录，存放外部仓库
│   ├── Real-ESRGAN/       ← feature1 的全部文件
│   │   ├── realesrgan/
│   │   ├── inference_realesrgan.py
│   │   └── ...
│   │
│   └── BSRGAN/            ← feature2 的全部文件
│       ├── main_test_bsrgan.py
│       ├── models/
│       └── ...
│
├── .gitignore
└── README.md
```

---

## 方案一：命令行操作（完整步骤）

### 第 0 步：在正式操作前创建备份分支

```bash
# 进入主仓库本地目录
cd CV-SISR-2026

# 备份当前状态，以防万一可以回退
git checkout main
git checkout -b backup/before-merge
git checkout main
```

### 第 1 步：添加 feature 仓库为远程源

```bash
# 添加 Real-ESRGAN 为远程源（命名为 real-esrgan）
git remote add real-esrgan https://github.com/xinntao/Real-ESRGAN.git

# 添加 BSRGAN 为远程源（命名为 bsrgan）
git remote add bsrgan https://github.com/cszn/BSRGAN.git
```

此时 `git remote -v` 应该显示三个远程源：

```
origin        https://github.com/StevenPang203426/CV-SISR-2026.git (fetch)
real-esrgan   https://github.com/xinntao/Real-ESRGAN.git (fetch)
bsrgan        https://github.com/cszn/BSRGAN.git (fetch)
```

### 第 2 步：拉取 feature 仓库的历史

```bash
# 拉取（只下载历史，不会修改你的工作区）
git fetch real-esrgan
git fetch bsrgan
```

> **💡 Tips：为什么用 `fetch` 而不是 `pull`？**
>
> **`git fetch` 的原理**：fetch 只做一件事 —— 把远程仓库的提交历史、分支引用下载到本地的 `.git/objects` 数据库中，并更新远程跟踪分支（如 `real-esrgan/master`）。它**不会修改你的工作区文件，也不会改动任何本地分支**，是一个纯粹的"只读"操作。
>
> **`git pull` = `fetch` + `merge`**：pull 会在 fetch 之后自动把远程分支合并到你**当前所在的本地分支**。在多仓库融合场景中这是危险的 —— 如果你当前在 `main` 分支上执行 `git pull real-esrgan master`，它会直接把 Real-ESRGAN 的文件混入你的主分支根目录，而且立刻产生一个合并提交，很难撤销。
>
> **在融合场景中用 fetch 的好处**：
> - **安全**：fetch 之后你可以先检查远程分支的内容（`git log real-esrgan/master`），确认无误再手动 merge
> - **可控**：你可以选择在哪个分支上、以什么方式（`--no-commit`、`--squash`）进行合并
> - **可回退**：如果 merge 出了问题，因为是分步操作，回退更容易
>
> **一句话总结**：fetch 是"先看菜单"，pull 是"直接上菜" —— 融合外部仓库时，一定要先看再决定怎么合。

### 第 3 步：为 Real-ESRGAN 创建 feature 分支并合并到子目录

```bash
# 基于 main 创建 feature 分支
git checkout -b feature/real-esrgan main

# 直接把 Real-ESRGAN 的文件树读入子目录 —— 不经过根目录，不需要 git mv
git read-tree --prefix=features/Real-ESRGAN/ -u real-esrgan/master

# 提交
git commit -m "feat: integrate Real-ESRGAN into features/Real-ESRGAN/"
```

> **💡 Tips：为什么不 merge 到根目录再 `git mv`？**
>
> 如果用 `git merge real-esrgan/master`，feature 仓库的几百个文件会全部涌入根目录，和主仓库的文件混在一起。然后你得想办法分辨哪些是新来的、哪些是原有的，再一个个 `git mv` 进子目录 —— 文件少还好，文件一多就是噩梦。
>
> **`git read-tree --prefix=<dir>/ -u <branch>`** 的做法完全不同：它直接在 Git 的暂存区（index）层面，把目标分支的整棵文件树**带着路径前缀**写入，然后 `-u` 参数把暂存区的内容更新到工作区。结果就是 feature 仓库的文件**从一开始就在子目录里**，根目录干干净净，根本不需要移动任何东西。
>
> **对比**：
> | | merge → git mv | read-tree --prefix |
> |---|---|---|
> | 文件落在哪 | 先落在根目录，手动移 | 直接落在子目录 |
> | 操作步骤 | merge → 识别新文件 → 逐个/批量 mv → commit | read-tree → commit |
> | 出错风险 | 高（可能误移主仓库文件） | 低（路径由 prefix 决定） |
> | 保留 feature 历史 | 是（`--allow-unrelated-histories`） | 否（只读入一个快照） |
>
> **唯一取舍**：`read-tree` 不会在 `git log` 中保留 feature 仓库的提交历史（和 `subtree --squash` 效果一样）。如果你需要完整历史，可以先 `git merge --allow-unrelated-histories --no-commit`，再用 `git read-tree` 覆盖暂存区来修正路径：
>
> ```bash
> git merge real-esrgan/master --allow-unrelated-histories --no-commit
> git read-tree HEAD                              # 恢复主仓库的暂存区
> git read-tree --prefix=features/Real-ESRGAN/ -u real-esrgan/master  # 把 feature 读入子目录
> git commit -m "feat: integrate Real-ESRGAN (with full history)"
> ```
>
> 这样既保留了两棵历史树的合并关系，又不需要手动移动任何文件。

### 第 4 步：为 BSRGAN 重复相同操作

```bash
git checkout -b feature/bsrgan main

# 同样直接读入子目录
git read-tree --prefix=features/BSRGAN/ -u bsrgan/main

git commit -m "feat: integrate BSRGAN into features/BSRGAN/"
```

### 第 5 步：推送到远程

```bash
# 推送 feature 分支
git push origin feature/real-esrgan
git push origin feature/bsrgan

# 如果要合并到 main
git checkout main
git merge feature/real-esrgan
git merge feature/bsrgan
git push origin main
```

### 第 6 步（可选）：清理远程源

融合完成后，feature 仓库的远程源已经没用了，可以删除：

```bash
git remote remove real-esrgan
git remote remove bsrgan
```

---

## 方案二：更简洁的替代方法 —— git subtree

`git subtree` 可以一步完成"拉取 + 放入子目录"，不需要手动 `git mv`。

```bash
cd CV-SISR-2026
git checkout main

# 一条命令：拉取 Real-ESRGAN 并放入 features/Real-ESRGAN/
git subtree add --prefix=features/Real-ESRGAN \
    https://github.com/xinntao/Real-ESRGAN.git master --squash

# 一条命令：拉取 BSRGAN 并放入 features/BSRGAN/
git subtree add --prefix=features/BSRGAN \
    https://github.com/cszn/BSRGAN.git main --squash
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `--prefix=features/Real-ESRGAN` | 放入哪个子目录 |
| `https://...` | feature 仓库地址 |
| `master` 或 `main` | feature 仓库的分支名 |
| `--squash` | 把 feature 仓库的历史压缩为一次提交（推荐，保持历史干净） |

如果要保留 feature 的完整历史，去掉 `--squash` 即可。

**创建 feature 分支版本**（不直接合入 main）：

```bash
# 先在 feature 分支上操作
git checkout -b feature/real-esrgan main
git subtree add --prefix=features/Real-ESRGAN \
    https://github.com/xinntao/Real-ESRGAN.git master --squash
git push origin feature/real-esrgan

# BSRGAN 同理
git checkout -b feature/bsrgan main
git subtree add --prefix=features/BSRGAN \
    https://github.com/cszn/BSRGAN.git main --squash
git push origin feature/bsrgan
```

---

## 方案对比

| | 方案一：remote + merge + mv | 方案二：git subtree |
|---|---|---|
| **操作复杂度** | 较高（需手动移文件） | 低（一条命令） |
| **保留完整历史** | 是（默认） | 可选（`--squash` 或不加） |
| **后续从上游拉更新** | 需要重新 fetch + merge | `git subtree pull` 一条命令 |
| **适合场景** | 需要精细控制合并过程 | 快速融合，推荐大多数情况 |

**推荐**：绝大多数场景直接用 **方案二 git subtree**，简洁不易出错。

---

## 在 VSCode 图形界面中操作

VSCode 的内置 Git 面板不直接支持 `subtree` 和 `remote add`，但可以通过以下方式操作：

### 使用 VSCode 内置终端

1. 按 `` Ctrl+` `` 打开 VSCode 集成终端
2. 执行上述命令行操作（方案一或方案二均可）
3. 操作完成后，左侧 Source Control 面板会自动刷新显示变更

### 使用 GitLens 扩展（推荐安装）

1. 安装扩展：`Extensions` → 搜索 `GitLens` → Install
2. 左侧栏出现 GitLens 面板，可以可视化查看 remotes、分支图
3. `Remotes` 区域右键 → `Add Remote` → 输入名称和 URL
4. 但 subtree 操作仍需在终端中执行

### 使用 Git Graph 扩展（可视化分支图）

1. 安装扩展：`Extensions` → 搜索 `Git Graph` → Install
2. 点击底部状态栏的 `Git Graph` 按钮
3. 合并后可以直观看到各仓库的历史如何汇入

### VSCode 中的典型工作流

```
1. Ctrl+`          → 打开终端
2. git subtree add  → 执行融合命令
3. Ctrl+Shift+G     → 打开 Source Control 面板，查看变更
4. 在 Source Control 面板中 Stage → Commit → Push
5. 打开 Git Graph 扩展查看分支拓扑
```

---

## 通用经验：以后遇到类似情况怎么办

### 场景判断：选择哪种工具

```
我想把外部仓库的代码放进我的项目
       │
       ├── 只是引用，不修改源码 ──→ git submodule
       │                          （保持独立仓库，链接引用）
       │
       ├── 要融入项目，可能修改 ──→ git subtree     ★ 最常用
       │                          （代码复制进来，可选保留历史）
       │
       └── 要完全合并历史 ────────→ remote + merge
                                   （两棵历史树嫁接在一起）
```

### 决策清单

1. **确定主仓库**：哪个仓库是"底座"？它的 main 分支不动。

2. **确定融合结构**：
   - **独立子目录**（推荐）：互不干扰，路径清晰，适合代码差异大的情况
   - **文件混合**：适合两个仓库结构高度相似、要深度整合的情况
   - **根目录并列**：适合只是把多个项目放在同一个 mono-repo

3. **确定历史策略**：
   - **squash**（推荐）：压缩为一次提交，历史干净
   - **保留完整历史**：需要追溯 feature 仓库的变更时使用

4. **是否需要持续同步上游**：
   - 需要 → 用 `git subtree`（支持 `pull`/`push`）
   - 不需要，一次性融合 → 任何方案都行

### 操作模板

```bash
# ============================================================
# 通用模板：将外部仓库 REPO_URL 融合到当前仓库的 TARGET_DIR
# ============================================================

# 变量（替换为实际值）
REPO_URL="https://github.com/someone/some-repo.git"
BRANCH="main"                    # 外部仓库的分支名
TARGET_DIR="features/some-repo"  # 放入的子目录
FEATURE_BRANCH="feature/some-repo"

# 步骤 1：创建 feature 分支
git checkout -b $FEATURE_BRANCH main

# 步骤 2：subtree add
git subtree add --prefix=$TARGET_DIR $REPO_URL $BRANCH --squash

# 步骤 3：推送
git push origin $FEATURE_BRANCH

# 步骤 4（可选）：合并到 main
git checkout main
git merge $FEATURE_BRANCH
git push origin main
```

### 后续从上游拉取更新

```bash
# 更新 Real-ESRGAN 到最新版本
git subtree pull --prefix=features/Real-ESRGAN \
    https://github.com/xinntao/Real-ESRGAN.git master --squash
```

### 常见报错与解决

| 报错 | 原因 | 解决 |
|------|------|------|
| `fatal: refusing to merge unrelated histories` | 两个仓库没有共同祖先 | 加 `--allow-unrelated-histories` |
| `working tree has modifications` | 有未提交的修改 | 先 `git stash` 或 `git commit` |
| `prefix 'xxx' already exists` | 目标目录已存在 | 换个目录名，或先删除再重试 |
| `can't squash-merge: 'xxx' was never added` | 首次 subtree 没用 add | 用 `git subtree add` 而不是 `pull` |
| 合并后发现文件冲突 | 两个仓库有同名文件（如 README.md） | 手动解决冲突，优先保留主仓库版本 |

### 已经 merge 了怎么撤销？

根据你的情况选择对应方案：

**情况 1：merge 后还没有 commit（用了 `--no-commit`，或 merge 正在冲突中）**

```bash
# 直接中止合并，回到 merge 之前的状态
git merge --abort
```

**情况 2：merge 后已经 commit，但还没有 push**

```bash
# 查看 merge 前的那次提交（找到 merge 提交的上一个）
git log --oneline -5

# 方法 A：直接回退到 merge 前的提交（推荐，干净利落）
git reset --hard HEAD~1
# 如果 merge 产生了多个提交，用具体的 commit hash：
# git reset --hard <merge前的commit-hash>

# 方法 B：如果你不确定回退几步，用 reflog 找到 merge 前的状态
git reflog
# 找到 merge 之前的那条记录，比如 HEAD@{2}
git reset --hard HEAD@{2}
```

**情况 3：merge 后已经 push 到远程（别人可能已经拉取了）**

```bash
# 用 revert 生成一个"反向提交"来撤销 merge（不会改写历史）
# -m 1 表示保留主分支（第一个 parent）的内容
git revert -m 1 <merge-commit-hash>
git push origin main
```

> **💡 Tips：`reset` vs `revert` 怎么选？**
>
> | | `git reset --hard` | `git revert -m 1` |
> |---|---|---|
> | 原理 | 把分支指针直接移回去，抹掉 merge 提交 | 创建一个新提交，内容是 merge 的反操作 |
> | 历史 | 被回退的提交从分支历史中消失 | 历史完整保留，多了一个 revert 提交 |
> | 适用 | **没 push** —— 只影响本地，随便用 | **已 push** —— 不改写公共历史，安全 |
> | 风险 | 如果已 push 后 reset，需要 `force push`，会影响其他协作者 | 无风险，但后续想重新合并该分支需要额外处理 |
>
> **一句话**：没推过用 `reset`，推过了用 `revert`。

**情况 4：用了 `git subtree add` 想撤销**

`subtree add` 本质上也是创建提交，撤销方式和上面一样：

```bash
# 没 push：直接回退（subtree add 通常生成 2 个提交）
git reset --hard HEAD~2

# 已 push：用 revert
git log --oneline -5    # 找到 subtree add 产生的提交
git revert <commit-hash>
git push origin main
```

**情况 5：我在第 0 步创建了备份分支**

如果你按照指南的第 0 步创建过 `backup/before-merge`，最简单：

```bash
# 直接用备份分支覆盖 main（这就是备份的意义）
git checkout main
git reset --hard backup/before-merge
```

---

## 本次操作的推荐命令（复制即用）

```bash
cd CV-SISR-2026

# ---- Real-ESRGAN ----
git checkout -b feature/real-esrgan main
git subtree add --prefix=features/Real-ESRGAN \
    https://github.com/xinntao/Real-ESRGAN.git master --squash
git push origin feature/real-esrgan

# ---- BSRGAN ----
git checkout -b feature/bsrgan main
git subtree add --prefix=features/BSRGAN \
    https://github.com/cszn/BSRGAN.git main --squash
git push origin feature/bsrgan

# ---- 合并到 main ----
git checkout main
git merge feature/real-esrgan
git merge feature/bsrgan
git push origin main
```

> **注意**：执行前请确认 Real-ESRGAN 的默认分支是 `master` 还是 `main`，BSRGAN 同理。可通过访问 GitHub 页面查看。
