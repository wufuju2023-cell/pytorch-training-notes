# datasets · v1.0 数据集说明

本目录用 [`prepare_tiny.py`](prepare_tiny.py) 把各数据集各取一个**极小切片**（可离线、
网络不可用时合成兜底），本文件记录每个数据集的 HF id / 本地路径 / 规模 / schema / 许可证。

> **教学只用切片**：本仓库不分发完整语料（Nemotron-CC-Math 约 20B token、Lean-GitHub
> 约 65M token、LeanTree 约 250MB…），也不含模型权重。默认每个数据集最多取 `--max-rows`
> 条，或用合成数据替代。

## 快速开始

```bash
cd datasets
python3 prepare_tiny.py --out ./_tiny --max-rows 200             # 有网：优先本地/HF
python3 prepare_tiny.py --out ./_tiny --max-rows 50 --offline    # 离线：本地/合成
```

产物：`local_corpus.txt`、`minif2f_subset.jsonl`、`fatem-100.jsonl`、`holdout-30.jsonl`、
`state_tactic_pairs.jsonl`、`leantree.jsonl`、`value_head_features.md`、`manifest.json`。

## 1. HF 数据集

| 名称 | HF id | 规模 | schema | 许可证 | 获取 |
|---|---|---|---|---|---|
| LeanTree | `ufal/leantree` | ~260k transitions（`leantree_mathlib.jsonl` ~250MB） | `proof_state, next_tactic, trajectory` | Apache-2.0 | `datasets.load_dataset("ufal/leantree")` |
| state_tactic_pairs | `FrenzyMath/state_tactic_pairs` | ~50k (state, tactic) | `state, tactic` | 见 HF 页 | `datasets.load_dataset` |
| Lean-Workbook | `internlm/Lean-Workbook` | 见 HF 页 | `lean_workbook.json` | 见 HF 页 | nanoproof 使用 |
| NuminaMath-LEAN | `AI-MO/NuminaMath-LEAN` | 见 HF 页 | parquet | 见 HF 页 | 预训练/增强 |
| DeepSeek-Prover-V1 | `deepseek-ai/DeepSeek-Prover-V1` | 见 HF 页 | `dataset.jsonl` | 见 HF 页 | nanoproof RL |
| Nemotron-CC-Math | `nvidia/Nemotron-CC-Math-v1` | ~20B token | 数学网页文本 | 见 HF 页（**GATED**） | 接受条款后下载 |
| Lean-Github | `internlm/Lean-Github` | ~65M token | Lean 源码 | 见 HF 页 | midtrain（重建 `leangithubraw.parquet` ~142MB） |
| NTP-Mathlib-Instruct | `l3lab/ntp-mathlib-instruct-context-fullproof` | 见 HF 页 | instruct/context | 见 HF 页 | 备选 |

`prepare_tiny.py` 对上述数据集只取前 N 条；不可用时用同 schema 的合成数据替代，并在
`manifest.json` 标注 `degraded`。

## 2. 本地路径（my-new-linux）

| 数据 | 路径 | 规模 | schema | 说明 |
|---|---|---|---|---|
| eval sets | `/mnt/gloway/projects/reap-new-update-model/v1-spec/v1-1-training-example/runs/eval_sets/{fatem-100,holdout-30}.jsonl` | 100 行 / 30 行 | `{"id","formal_statement"}`（含 `sorry`） | 评测用 |
| LeanTree 备份 | `/mnt/gloway/projects/9-27-hsy-modelscope-备份/mnt-workspace/new_value_head/data/leantree_mathlib.jsonl` | 250MB | 同 ufal/leantree | 值头/迁移 |
| 值头数据集 | `.../new_value_head/dataset/{train,validation,test}.jsonl` | 118MB / 11.8MB / 11.8MB | 见下 | 价值头离线训练 |
| 值头特征分片 | `.../new_value_head/features-79efd240/{train,validation,test}/features-*.pt` | 每片 ~14.96MB / 2000 行 | hidden 特征 | 大文件，教学不复制 |
| mathlib4 | `/mnt/gloway/projects/lean-corpus/mathlib4` | 4.7G | `.lean` 源码 | 预训练/midtrain 语料 |
| 容器 SFT 数据 | `.../full_modelscope/inference_v1/data/full_canonical_v4_val400_v2/{train,validation}.jsonl` | 4888 / 400 行 | 见下 | Qwen3-1.7B LoRA 训练 |

### 容器 SFT target schema

```json
{"host_metadata": {...}, "materialized_dsl_thinking": "...", "ordinal": 0,
 "proof_gap": "...", "public_proof_state": "...", "schema": "...",
 "target": {"calls": ["...", "..."]}}
```

对应 `train_full_supervised.py`：prompt = 指令 + `<proof_gap><dsl_thinking><public_proof_state>`，
答案 = `<answer>{"calls":[...]}</answer>`，prompt 的 label 全为 -100。

### 值头特征 schema

- `hidden`：冻结策略最后一层 hidden（fp16），shape `(N, H)`；
- `value_target` / `proof_depth`：目标值，`-min(d,64)/64`；
- `mask`：有效行掩码。

`prepare_tiny.py` 只写 `value_head_features.md` 说明，不复制大 tensor。

## 3. 评测集

| 名称 | 来源 | 规模 | 说明 |
|---|---|---|---|
| fatem-100 | 本地 eval_sets | 100 | 评测用，**不参与训练** |
| holdout-30 | 本地 eval_sets | 30 | 评测用，**不参与训练** |
| miniF2F | `google-deepmind/miniF2F`（GitHub） | valid 256 / test 244 | 标准 benchmark |
| ProofNet | `Kripner/ProofNet`（GitHub）`data/proofnet.jsonl` | 见仓库 | 备选 benchmark |

## 4. 许可证要点

- `ufal/leantree`：Apache-2.0（见 HF 页）；`mathlib4`：Apache-2.0。
- `FrenzyMath/state_tactic_pairs`、`internlm/*`、`deepseek-ai/*`、`l3lab/*`：以 HF 页为准。
- `nvidia/Nemotron-CC-Math-v1`：GATED，需在 HF 页面接受条款；本仓库不分发。

## 5. "教学只用切片" 的做法

1. `prepare_tiny.py --max-rows N`：每个数据集最多取前 N 条；
2. 优先本地复制（fatem/holdout/leantree），避免重复下载；
3. 无网络/无本地时**合成**同 schema 数据，保证 notebook 一定能跑；
4. 大文件（值头特征 `.pt`、20B token 语料、模型权重）只说明、不复制；
5. `manifest.json` 记录每个产物的来源、行数、schema、是否降级，便于审阅。
- miniF2F / ProofNet：见各自仓库许可。
- 本仓库所有产物为脚本 + 小切片说明，**不含**完整受版权保护语料。

