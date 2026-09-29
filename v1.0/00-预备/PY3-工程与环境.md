# 预备 PY3 · 工程与环境

> **对应 AlphaProof 工程**：`nanoproof/nanoproof/pretrain.py`、`sft.py` 的命令行入口与 `common.py` 的随机种子/日志初始化；`v1-result/reproduction/code/src/cpu_runtime` 与 `gpu_runtime` 中“同一份代码、不同后端/设备”的安装差异。
> **配套代码**：`v1.0/00-预备/code/py3_engineering.py`（Tag：`F-pre-py3`）
> **配套 notebook**：`v1.0/notebooks/00-预备/N20_工程环境与工具链.ipynb`（Tag：`N-20`）
> **预计学时**：3 学时
> **前置知识**：PY1、PY2 的模块与异常基础；本章是 02–08 各篇代码与全部 notebook 的环境底座。
> **【文档｜DOC-PY3】**（doccode = `PY3`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 1. 解释器与依赖管理

Python 没有“编译期”，依赖在**导入时**从文件系统解析，因此“用哪个解释器、装在哪套库目录”直接决定程序行为。这一节把这件事工程化。

**【定义 PY3.1.1｜D-PY3.1.1】（虚拟环境与解释器隔离）**

**虚拟环境（virtual environment）**是一个目录，内含指向某个基础解释器的可执行文件副本（或符号链接）以及一套**私有**的 `site-packages` 库目录。激活后，`python`、`pip` 只在该目录内解析与安装，不污染系统级解释器。

与 C/C++ 的对照：

- 基础解释器 ≈ 系统编译器（如 `/usr/bin/gcc`）；
- 虚拟环境 ≈ 一份独立的 `--sysroot` + 私有 `-I/-L` 前缀，保证头文件（库）版本互不串味；
- 激活脚本（`activate`）≈ 修改 `PATH`/`LD_LIBRARY_PATH` 的 `setenv` 脚本，只影响当前 shell 会话；
- `python -m venv .venv` 创建的 `.venv/` 才是真正的隔离边界，`requirements.txt` 只是其中一份“链接清单”。

关键推论：**同一项目一律在固定虚拟环境内运行**；`which python` 与 `python -c "import sys; print(sys.executable)"` 必须指向该环境，否则“在我机器上能跑”无从复现。

**【注 PY3.1.2｜R-PY3.1.2】（pip、依赖解析与 requirements.txt）**

`pip` 是 Python 的包安装器，从索引（默认 PyPI）解析**依赖图**并解出满足约束的一组版本，再下载 wheel/sdist 安装。

- wheel（`.whl`）≈ 预编译库（`.so`/`.a` 打包），无需本地编译；
- sdist（`.tar.gz`）≈ 源码包，安装时本地构建，可能需要编译器与系统头文件；
- `requirements.txt` ≈ 一份精确的依赖/版本清单；`pip install -r requirements.txt` 等价于“按清单拉取并链接”。

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

工程约定：固定**顶层直接依赖**的版本区间，提交 `requirements.txt`；需要逐位复现时用 `pip freeze > requirements.lock` 锁定全部传递依赖（见 §7）。**不要**用 `sudo pip install` 往系统解释器装库。

**【注 PY3.1.3｜R-PY3.1.3】（conda：跨语言包与环境管理）**

`conda` 用“环境（environment）”同时管理 Python 与非 Python 的二进制依赖（CUDA 运行时、BLAS、编译器等），因此能装 `pytorch-cuda` 这类带原生组件的包。与 `venv+pip` 的对照：

- `conda create -n <env> python=3.11` ≈ `python -m venv`，但连解释器版本一起固定；
- `conda install` 可装非 Python 依赖，`pip` 只装 Python 包；
- 混用时风险在于**两套依赖解析器各管一半**：建议在同一环境内先 `conda install` 原生组件，再 `pip install` 纯 Python 包，并记录 `conda env export`。

本教材的 CPU 路径用 `venv+pip` 即可；GPU 集群若已提供 conda/容器，遵循宿主约定，不擅自改底层 CUDA。

**【注 PY3.1.4｜R-PY3.1.4】（CUDA 版与 ROCm 版 PyTorch 的安装差异）**

PyTorch 的 wheel 按**计算后端**分渠道发布，包名相同（`torch`）但内容不同：

- NVIDIA CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu121`（示例，`cuXXX` 需与本机驱动兼容）；
- AMD ROCm 版：`pip install torch --index-url https://download.pytorch.org/whl/rocm6.1`（示例）；
- 纯 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cpu`。

运行期用 `torch.version.cuda`、`torch.version.hip`、`torch.cuda.is_available()` 判定实际后端。与 C/C++ 对照：这相当于同一 API 头文件链接不同的底层运行时库；**驱动版本、wheel 后端、设备可见性**三者必须自洽，否则出现“能 import 但 `is_available()` 为假”或算子报错。AlphaProof 的 CPU/GPU 双版本（`F-SRC6` 对应文档）正是靠这一点区分。

**【代码 PY3.1.5｜Cd-PY3.1.5】（环境自检片段）**

```python
import sys
print(sys.executable)          # 当前解释器（应指向 .venv/）
try:
    import torch
    print(torch.__version__, torch.version.cuda, torch.version.hip)
    print("cuda:", torch.cuda.is_available())
except ImportError:
    print("torch 未安装")
```

## 2. 项目结构

**【注 PY3.2.1｜R-PY3.2.1】（包布局与 import 解析）**

`import` 的解析规则：**包的根目录必须有 `__init__.py`**（常规包），解释器沿 `sys.path` 逐项查找；`sys.path[0]` 通常是“脚本所在目录”或“`python -m` 时的当前目录”。

- 包（package）≈ 一个目录 + 一个头文件（`__init__.py`）声明它的公共接口；
- 模块（module）≈ 一个编译单元（`.py` 文件）；
- 顶层导入 ≈ `#include <...>`（从 `sys.path` 找），相对导入 ≈ 包内 `#include "..."`。

本仓采用“脚本 + 同目录模块”的扁平布局（如 `01-基础/code/from_scratch/model.py` 里 `from configs import GPTConfig`），因此运行脚本时必须让该目录成为 `sys.path[0]`：直接 `python train.py`（脚本所在目录入路径）或 `cd` 到目录后用 `python -m <module>`。

**【注 PY3.2.2｜R-PY3.2.2】（pyproject.toml：声明“可安装”与“可运行”）**

`pyproject.toml` 是现代 Python 项目的**单一元数据文件**，声明构建后端、项目元信息、依赖与命令入口：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "pytorch-training-notes"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = ["numpy", "torch"]

[project.scripts]
py3-eng = "py3_engineering:main"
```

- `[project]` ≈ 编译产物的元信息 + 依赖清单；
- `[project.scripts]` ≈ 安装后在 `bin/` 里生成一个**入口命令**，等价于 C 项目 `make install` 安装出的可执行文件；
- `pip install -e .`（editable）≈ 建一个指向源码目录的“软链接安装”，改代码立即生效，适合开发。

**可安装**（有 `pyproject.toml` + 包目录）便于被 `pip` 管理与复用；**可运行**（单文件 + `if __name__ == "__main__"`）便于教学与直接 `python xxx.py`。本教材代码以后者为主，遇到 `02–08` 的 `with_api` 侧才需要前者。

**【注 PY3.2.3｜R-PY3.2.3】（本仓目录职责）**

- `v1.0/00-预备/code/*.py`：预备篇教学代码，单文件、可直接运行（本文件对应 `py3_engineering.py`）；
- `v1.0/<章>/code/from_scratch/`：纯 PyTorch 实现；`.../with_api/`：生态库实现；
- `v1.0/notebooks/<章>/N*.ipynb`：可运行小实验；
- `v1.0/tools/*.sh`：云端流水线脚本。

## 3. 命令行与配置：argparse、环境变量与 logging

**【注 PY3.3.1｜R-PY3.3.1】（argparse：把配置从源码里挪出来）**

`argparse` 负责解析命令行、生成帮助、做类型转换与校验，替代手写 `sys.argv` 解析（对照 C 的 `getopt`/`getopt_long`）：

- **位置参数**（`parser.add_argument("ckpt")`）≈ 必需实参；
- **可选参数**（`parser.add_argument("--lr", type=float, default=3e-4)`）≈ 带默认值的开关；
- **子命令**（`add_subparsers()`，如 `train`/`eval`）≈ `busybox` 式多入口；
- `type=`/`choices=`/`required=` 在解析阶段即失败并给出用法，比运行到一半才崩更早暴露错误。

**参数优先级**（推荐自低到高）：源码默认值 < 配置文件 < 环境变量 < 命令行。命令行最高，便于临时覆盖；生产任务用配置文件固定，CI/集群用环境变量注入。

**【注 PY3.3.2｜R-PY3.3.2】（环境变量与 os.environ）**

环境变量是**进程外注入的字符串配置**，适合放机器相关、易变或不宜入库的信息（路径、设备、token）。对照 C：≈ `getenv()` 读取的运行时配置，或编译期的 `-D` 宏，但可在不重编译的情况下改变行为。常用：

- `PATH`：决定 `python`/`pip` 解析到哪个环境；
- `CUDA_VISIBLE_DEVICES`：限制可见 GPU（对照设备白名单），多卡时按序号暴露；
- `PYTHONHASHSEED`：固定哈希随机化，影响依赖哈希顺序的行为（见 §7）；
- `PYTHONPATH`：追加 `sys.path` 搜索路径。

**安全约束**：token/密码只经环境变量或密钥文件注入，**不得**写进源码、命令行参数或日志（命令行参数可见于进程列表）。

**【注 PY3.3.3｜R-PY3.3.3】（logging：替代 print/printf 的调试）**

`logging` 用**级别 + 通道（handler）**组织日志，替代散落的 `print`（对照 C 的 `printf` 宏 + `syslog`）：

- 级别：`DEBUG < INFO < WARNING < ERROR < CRITICAL`，`setLevel` 控制**输出阈值**而不改调用点；
- `logger = logging.getLogger(__name__)`：按模块名分层，可独立开关；
- handler：`StreamHandler`（终端）、`FileHandler`（落盘）、`RotatingFileHandler`（滚动）；formatter 统一加时间戳/模块名/级别；
- `logger.info(...)` 惰性拼接（`%s` 占位），避免“即使不输出也构造字符串”的开销。

工程价值：训练脚本用 `logger.info("step=%d loss=%.4f", step, loss)` 记录曲线；库代码只管记录，**不在 import 时调用 `basicConfig`**，把配置权留给应用入口。

**【代码 PY3.3.4｜Cd-PY3.3.4】（argparse + logging 最小骨架）**

配套代码 `F-pre-py3` 给出可运行骨架：`parse_args()` 定义 `--seed/--steps/--log-level`，`configure_logging()` 统一格式，`seed_everything()` 固定随机源，`summarize()` 是纯函数便于测试，`main()` 串起全流程。核心片段：

```python
parser = argparse.ArgumentParser(description="PY3 工程骨架")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
```

## 4. 调试与测试

**【注 PY3.4.1｜R-PY3.4.1】（pdb 与 breakpoint()，对照 gdb）**

`pdb` 是标准库的**交互式调试器**，对照 C 的 `gdb`：

- `breakpoint()`（Python 3.7+，等价 `import pdb; pdb.set_trace()`）在源码处设断点；
- `n`（next，步过）、`s`（step，步入）、`c`（continue，继续）、`l`（list，看源码）、`p expr`/`pp`（打印）、`q`（退出）；
- 断点位置写成 `file.py:line`，可 `break` 动态添加；`import pdb; pdb.pm()` 在异常后进入**事后调试**（post-mortem），查看崩溃现场栈。

与 C 的差别：Python 无编译，`p` 可直接对任意对象求值（如 `p list(x.shape)`），但解释执行下 `n` 一次仍可能跑很多 C 级循环；调试数值问题更常用“断言 + 小张量检查”而非逐行单步。

**【注 PY3.4.2｜R-PY3.4.2】（assert：契约检查与它的开关）**

`assert cond, "msg"` 声明**不变量**，失败抛 `AssertionError`。对照 C 的 `assert()`：

- 相同点：都是“开发期契约”，失败即中止；
- 不同点：Python 的 `assert` 在 `python -O` 优化模式下**被整体移除**，故**不能**用于校验外部输入或承担业务逻辑；需要恒定生效的检查要用 `if not cond: raise ValueError(...)`。

Tensor 场景常用 `assert x.shape == (B, T, C)`、`assert torch.isfinite(loss)` 作为廉价护栏。

**【注 PY3.4.3｜R-PY3.4.3】（pytest：发现、断言、fixture、参数化）**

`pytest` 是事实标准测试框架，对照 C++ 的单元测试框架（CppUTest/Catch2）：

- **发现**：`pytest` 递归收集 `test_*.py` / `*_test.py` 中 `test_*` 函数；
- **断言**：直接用裸 `assert`，失败时框架**重写**表达式并打印中间值；
- **fixture**：`@pytest.fixture` 提供测试前置/后置资源（临时目录、构造好的对象），对照 `setUp/tearDown`，但可按需注入、作用域可调；
- **参数化**：`@pytest.mark.parametrize` 用一组参数生成多条用例，等价于手写循环但报错定位到单个参数组合。

```python
import pytest

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seed_determinism(seed):
    assert draw(seed) == draw(seed)
```

调用方式：`python -m pytest -q`（把当前目录加入 `sys.path`，比裸 `pytest` 更稳）。

**【注 PY3.4.4｜R-PY3.4.4】（属性测试简介）**

**属性测试（property-based testing）**不枚举样例，而是描述**对任意输入都应成立的性质**（如“排序后单调不减”“逆运算还原”“mean 在 min/max 之间”），由框架（`hypothesis`）自动生成并收缩反例。对照 C：类似带随机输入的模糊测试（fuzzing），但以“性质”为断言。它擅长发现边界（空序列、极值、浮点 NaN）；数值代码中常用“与朴素实现结果一致”作为性质。

## 5. Jupyter / Colab 与本仓 notebook 惯例

**【注 PY3.5.1｜R-PY3.5.1】（cell 语义、内核与执行顺序）**

Jupyter notebook（`.ipynb`）是 JSON 文档，由若干 **cell** 组成：

- code cell：交给**内核（kernel）**执行，内核是一个长期存活的 Python 进程，**变量跨 cell 共享**；
- markdown cell：只渲染文档；
- **执行顺序按你点的先后，而非从上到下**：序号 `In [n]` 记录真实顺序。这正是 notebook 最著名的坑——从上到下重跑可能得到不同结果。

工程纪律：交付/提交前用 `Restart Kernel and Run All` 或命令行 `jupyter nbconvert --to notebook --execute --inplace <nb.ipynb>` 从干净内核重跑，确保结果可复现（见 §7）。

**【注 PY3.5.2｜R-PY3.5.2】（魔法命令与 shell 转义）**

- `%` 开头的**行魔法**（line magic）作用于单行，如 `%timeit`、`%matplotlib inline`；
- `%%` 开头的**单元魔法**（cell magic）作用于整个 cell，如 `%%time`、`%%capture`；
- `!` 前缀执行**系统 shell 命令**，如 `!pip install -q torch`、`!nvidia-smi`；`{var}` 可在其中插值，`!` 后接 `&` 会异步执行。
- Colab 中安装依赖惯用 `!pip install ...`；非交互环境（脚本/CI）请回到 §1 的 `pip install -r`。

注意：`!pip` 不是合法 Python 语法，仅在 notebook 内核里被特殊解释。本仓 notebook 的 code cell 保持纯 Python（便于 `py_compile`/`_validate_nb` 校验），把 `!pip` 示例放在 markdown 说明中。

**【注 PY3.5.3｜R-PY3.5.3】（内核算力与 GPU 检测）**

Colab/云环境按需分配 GPU。标准检测片段：

```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print(torch.version.cuda, torch.version.hip)
```

显存与设备可用性随会话变化，故 notebook 的 **CPU 路径必须能独立跑通**，GPU 段落用 `if torch.cuda.is_available():` 兜底跳过（本仓 N01–N17 均遵循此约定）。

**【注 PY3.5.4｜R-PY3.5.4】（与本仓 notebook 惯例对接）**

本仓 notebook 遵守以下约定（见《00-风格与编号规范》§5）：

- 首个 markdown cell 以 `# 【notebook｜N-xx】（标题）` 开头，并给出对应理论文档/代码引用；
- 每个 code cell 首行写 `# 【N-xx.cK】`（K 为该 notebook 内 code cell 序号，从 1 起）；
- 主要小节 markdown cell 可标 `**【N-xx.mJ】**`；
- 提交的 notebook 保留可复现的输出；JSON 用 `ensure_ascii=False` 读写。

本章 notebook 为 `N-20`（`00-预备/N20_工程环境与工具链.ipynb`），其“最小训练循环”一类可运行片段与第 1 章样板【代码 1.6.2】同构；损失口径沿用交叉熵/负对数似然【定义 1.2.4】。

## 6. git 基础与协作

**【注 PY3.6.1｜R-PY3.6.1】（提交、分支与合并）**

`git` 是分布式版本控制：

- **工作区 → 暂存区 → 仓库**：`git add <file>` 把改动放入暂存区，`git commit -m "..."` 生成一次快照（commit）。对照 C：`add` 是“选择要进这次构建的源文件”，`commit` 是一次不可变快照；
- **分支（branch）**是指向某次提交的可移动指针；`git switch -c <name>` 新建分支，提交只前进该分支；
- **合并（merge）**把另一分支的提交并入当前分支；无冲突时生成合并提交，有冲突时手动解决后 `git add` 标记已解决；
- `git log --oneline`、`git diff`、`git status` 是查看历史与差异的常用命令。

工程纪律：**只 `git add` 自己负责的文件，严禁 `git add -A`**；**逐文件提交**、提交信息说明“改了什么/为什么”。发现误改先 `git diff` 定位，不要盲目回滚。

**【注 PY3.6.2｜R-PY3.6.2】（.gitignore 与不入库的产物）**

`.gitignore` 列出**不应进入版本库**的路径，对照 C 项目里“忽略 `*.o`/`build/`”。Python/Jupyter 场景至少忽略：

```text
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/  venv/
.ipynb_checkpoints/
```

大数据集、模型权重、含密钥的 `.env` 同样不得入库（密钥见 §3 的安全约束）。

**【注 PY3.6.3｜R-PY3.6.3】（worktree：一个仓库、多个工作目录）**

`git worktree` 允许**同一个仓库同时检出多个分支到不同目录**，共享同一份 `.git` 对象库。

- 对照 C：不必“复制整棵树再分别改”，而是多个 `build/` 目录共享同一源码仓库；
- 用途：多名作者并行开发互不干扰；一个人同时处理主线与热修；
- 基本命令：

```bash
git worktree add ../repo-feature feature       # 新目录 + 新分支
git worktree list                              # 列出所有工作树
git worktree remove ../repo-feature            # 清理
```

**与本仓 `team-worktree` 工作流的对照**：本团队用封装脚本为每名作者建独立 worktree + 分支 `tm/<team>/<name>`，作者只在自己的目录里工作、只提交自己的文件，合并与推送由 lead 统一执行。因此作者**不调用**任何自动合并操作，也**不**改动他人文件；这与 §8 的可复现/可审计目标一致。

**【注 PY3.6.4｜R-PY3.6.4】（冲突的处理原则）**

冲突发生在同一文件的同一区域被两条历史线修改时。处理流程：`git status` 找出冲突文件 → 打开文件定位 `<<<<<<<`/`=======`/`>>>>>>>` 标记 → 人工取舍保留正确版本 → `git add <file>` 标记已解决 → 完成 merge/rebase。**原则**：不静默丢弃他人改动，不猜测语义；拿不准时先 `git diff` 全貌再决定。

## 7. 可复现性

**【定义 PY3.7.1｜D-PY3.7.1】（可复现性的三要素）**

一次训练/实验“可复现（reproducible）”要求固定三件事：

1. **代码版本**：确切的 commit（记录 `git rev-parse HEAD`）；
2. **环境版本**：解释器版本、框架与依赖版本、计算后端（CUDA/ROCm/CPU）；
3. **随机性与非确定性来源**：随机种子、算子实现、并行归约顺序。

三者任一漂移，结果就可能不同；**“可复现”通常指统计意义一致，追求逐位一致需额外付出**（见下）。

**【注 PY3.7.2｜R-PY3.7.2】（随机种子与确定性开关）**

Python 各随机源相互独立，需分别播种：

```python
import random, numpy as np, torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- 对照 C：`random.seed` ≈ `srand(seed)`，`torch.manual_seed` 还要覆盖设备侧生成器；
- 追求确定性时设 `torch.use_deterministic_algorithms(True)`、为 `DataLoader` 设 `worker_init_fn` 与 `generator`；GPU 上部分原子算子仍可能非确定，需权衡性能；
- 数值核对（如梯度检查）应使用 `float64` 并避开不可微点，详见【注 1.5.2】；正确性以数学损失口径【定义 1.2.4】为准。

**【注 PY3.7.3｜R-PY3.7.3】（记录环境：把现场写进产物）**

每次实验结束记录一份环境快照，便于事后定位：

```bash
python -c "import sys, platform; print(sys.version, platform.platform())"
python -m pip freeze > requirements.lock      # 精确版本
git rev-parse HEAD                            # 代码版本
```

配合 `--seed`、`--log-level` 等命令行参数（§3）与日志（§3.3.3），可使“从命令 + 清单”重建整次运行。训练脚本应在启动时把上述信息写进日志头。

## 8. 来源与许可

**【注 PY3.8.1｜R-PY3.8.1】（外部来源清单）**

本章材料以官方文档为主，**仅链接引用并改写要点，未整段复制**（NC/SA 类只链接）。逐条登记如下（lead 汇总入 `tags/external.tsv`，关系 `adapted_from`）：

| 来源 | URL | 许可 | 搬运范围 |
| --- | --- | --- | --- |
| Python 官方文档（venv / argparse / logging / pdb / random / subprocess） | `https://docs.python.org/3/` | PSF-2.0 | 要点改写与最小示例 |
| pip 文档 | `https://pip.pypa.io/en/stable/` | MIT | 命令与要点 |
| Python Packaging User Guide（包布局 / pyproject.toml） | `https://packaging.python.org/` | 见站点仓库 LICENSE（仅链接） | 结构说明 |
| PyTorch 安装指南 | `https://pytorch.org/get-started/locally/` | BSD-3-Clause | 安装命令改写 |
| pytest 文档 | `https://docs.pytest.org/` | MIT | 用法要点 |
| hypothesis 文档 | `https://hypothesis.readthedocs.io/` | MPL-2.0 | 概念简介 |
| Jupyter 文档 | `https://docs.jupyter.org/` | BSD-3-Clause | 概念说明 |
| Google Colab FAQ | `https://research.google.com/colaboratory/faq.html` | Google 服务条款（仅链接） | GPU/环境说明 |
| git 官方文档（worktree 等） | `https://git-scm.com/docs` | GPL-2.0 | 命令与概念 |
| Pro Git 书 | `https://git-scm.com/book/` | CC BY-NC-SA 3.0（仅链接） | 概念参照 |

## 9. 自测

**【练习 PY3.9.1｜Ex-PY3.9.1】（虚拟环境与依赖隔离）**

在一个空目录用 `python -m venv .venv` 建环境，激活后打印 `sys.executable`，再从系统解释器打印一次并对比；用 `pip install` 安装一个纯 Python 小包，确认系统解释器 `import` 不到它。说明为什么 `sudo pip install` 会破坏这个隔离。

**【练习 PY3.9.2｜Ex-PY3.9.2】（参数与日志）**

给 `F-pre-py3` 增加 `--repeat`（重复次数）并接入 `logging.debug`，使 `--log-level DEBUG` 时能看到每步明细、`INFO` 时只看到汇总。要求：默认行为不变，且 `python py3_engineering.py --help` 能列出新参数（对照 `PY3.3.1`）。

**【练习 PY3.9.3｜Ex-PY3.9.3】（测试与复现）**

为 `F-pre-py3` 的纯函数写 3 条 `pytest` 用例：同种子两次结果一致、不同种子（高概率）不同、对空输入给出明确异常；用 `jupyter nbconvert --execute --inplace` 从干净内核重跑 `N-20` 并确认输出稳定（对照 §7 与【注 PY3.5.1】）。
