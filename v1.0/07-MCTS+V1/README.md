# 07 · MCTS + V1

> **【文档｜DOC-RMMCTS】**（doccode = `RMMCTS`）｜编号与 Tag 规范见《00-风格与编号规范》。

本目录讲清楚 AlphaZero 式搜索如何驱动 Lean 证明，以及 V1 训练闭环。

## 文件

**【注 RMMCTS.1.1｜R-RMMCTS.1.1】（文件）**

本文对应代码 Tag：`F-b07-mcts`、`F-b07-policy_value_eval`、`F-b07-self_play`、`F-b07-run_demo`；配套 notebook：`N-15`、`N-16`。

| 文件 | 内容 |
|---|---|
| [`01-MCTS原理与V1闭环.md`](01-MCTS原理与V1闭环.md) | OR/AND 节点、focus 子目标、PUCT、γ 折扣备份、Q 变换、progressive sampling、Dirichlet 旁注、与 Lean `reap` 引擎和 `v1_run.py`/`online_ttt.py` 的对应 |
| [`code/mcts.py`](code/mcts.py) | 极简 PUCT / OR-AND（含价值头占位接口与 toy 环境，纯 Python 直接跑） |
| [`code/v1_tiny/policy_value_eval.py`](code/v1_tiny/policy_value_eval.py) | tiny 策略 + 价值网络（提供 tactic 先验与剩余步数） |
| [`code/v1_tiny/self_play.py`](code/v1_tiny/self_play.py) | MCTS 自博弈，收集 `(state, tactic, value_target)` |
| [`code/v1_tiny/run_demo.py`](code/v1_tiny/run_demo.py) | 合成可验证任务上的 V1 闭环端到端 demo |
| notebooks `N15`、`N16` | 可运行实验 |

## 快速开始

**【注 RMMCTS.2.1｜R-RMMCTS.2.1】（快速开始）**

```bash
python3 code/mcts.py                 # 纯 Python，无需依赖
python3 code/v1_tiny/run_demo.py     # 需要 torch（CPU 即可）
```

## 与 AlphaProof 的对照

**【注 RMMCTS.3.1｜R-RMMCTS.3.1】（与 AlphaProof 的对照）**

- Lean 搜索内核：`reap-upstream/Tactic/TreeSearch.lean`（`computePUCTScores:312` / `shouldProgressiveSample:333` / `backpropValueTowardsMin:386`）。
- 逐轮手算：`/home/a/文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`。
- 编排：`app/v1_run.py`；在线训练：`v1-result/reproduction/code/src/cpu_runtime/online_ttt.py`。

## 思考题

**【注 RMMCTS.4.1｜R-RMMCTS.4.1】（思考题）**

见 `01-MCTS原理与V1闭环.md` §9。
