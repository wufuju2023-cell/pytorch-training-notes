# 05 · GPU 版本 E：value-head 与 nanoproof 移植

> 源码位置：
> - `a@my-new-linux:/mnt/gloway/projects/reap-new-update-model-value-head/app/`（value head 版 app：`value_head.py` 与 01 的相同；`policy_server.py` 略有增强）
> - `.../reap-new-update-model-value-head/nanoproof/`（nanoproof 的 value-head fork：核心 `nanoproof/*.py` 与 04 篇逐字一致，额外含 `PORTING_REAP_VALUE_HEAD.md`、`pseudocode.py`、`scripts/`、`tests/`、`web/`、`data/rl/`）
> - `.../reap-agentic-v1-1-sync/code-79efd240/scripts/{prepare_leantree.py,extract_real7_features.py,train_value_head.py,assess_value_head.py,preflight_ms.py}`
> 角色：把 nanoproof 的**价值训练信号与数据语义**（成功轨迹的负剩余证明深度 / 64 个离散桶）移植到 REAP 的 **REAL-Prover 共享 backbone** 上；生产用 64-bin 分类头（与 03 篇 `categorical_search_backend.py` 对应），app 里另有一套连续标量头（与 01 篇对应）。
> 教学链接：`../04-价值头/`、`../05-LoRA/`（app 的 TTT 联合更新）、`../06-RL-RLVR/`、`../07-MCTS+V1/`。

## 1. 移植映射（`PORTING_REAP_VALUE_HEAD.md`）

| nanoproof 概念 | REAP 实现 | 备注 |
|---|---|---|
| solved tree 的 proof-depth target | `proof_depth_to_target`（归一化到 `[-1,0]`） | `app/value_head.py:155` |
| terminal/rollout return | `value_target`/`return`，MSE/Huber | `app/train_value_head.py:60` |
| value 推理 | `POST /value` 或 `value/v1/chat/completions` | `app/policy_server.py:657` |
| value 损失权重 | `--value-coefficient` / `value_coefficient` | `app/policy_server.py:49` |
| checkpoint | 独立 `value_head.pt`（不含 backbone） | schema `reap.value-head.v1` |

关键差异：nanoproof 的价值是 `<|value|>` 后的 64 个离散 bin token；REAP 的 app 版是共享 backbone 后的连续标量 `Linear(H,256)→SiLU→Linear(256,1)→Tanh`（H 动态读取，不再硬编码 4096）。移植的是**训练信号与数据语义**，不是 token-bin checkpoint。

## 2. `app/`：连续标量价值头（与 01 篇同源）

- `value_head.py`：`VALUE_HEAD_SCHEMA="reap.value-head.v1"`；`ValueHead`（`Linear(H,256)→SiLU→Linear(256,1)→Tanh`）；`proof_depth_to_target(depth,max_depth=64) = -min(depth,64)/64`；`discounted_returns`；原子 `save_value_head`/`load_value_head`（支持 legacy 裸 state_dict）；`ValueHeadTrainer`（MSE/Huber + 有限性校验）。
- `policy_server.py`（略强于 01）：端点 `GET /health`、`POST /v1/chat/completions`、`/value`、`/value/v1/chat/completions`、`/value/train`、`/ttt_step`、`/adapter/snapshot`、`/adapter/restore`；`Engine.ttt_step` 做 policy（REINFORCE）+ KL + value（MSE）联合更新；`value_output_mode ∈ {scalar, distance}`，`distance` 映射到 `[1,max_distance]` 以适配 nanoproof/verified-collector 客户端（客户端会在 Lean 侧再取负一次）。
- `train_value_head.py`：冻结 backbone、离线训 head，接受 `value_target`（含 `target_kind=nanoproof` 的负剩余深度，如 `value_target=-8`）或 `proof_depth` 或 `states/rewards` 轨迹（discounted return）；默认**不把旧 `value.score` 当标签**，除非 `--allow-score`。
- `VALUE_HEAD.md`：数值语义、HTTP 输出模式、离线训练与启动、在线 TTT item 字段（`value_target`/`proof_depth`/`next_value`/`next_prompt` → 一步 TD `r+γV(s')`，裁剪 `[-1,1]`）、以及“仓库无公开可用已训 head”的限制说明。

## 3. nanoproof fork（`nanoproof/`）

- **核心文件与 04 篇逐字相同**（`model.py/optim.py/sft.py/rl.py/search.py/experience_collection.py` 等无差异）；本 fork 增加：
  - `PORTING_REAP_VALUE_HEAD.md`（移植说明）。
  - `pseudocode.py`（937 行）：**AlphaProof 论文的 RL / autoformalization / variant generation 伪代码**（JAX）。关键点：
    - `Config`（`:67`）：MCTS `pb_c_base=3200, pb_c_init=0.001, value_discount=0.99, prior_temperature=200, no_legal_actions_value=-40`，`num_value_bins=64`（`:97`），`value_weight=0.001`；Matchmaker 参数（`:108-113`）。
    - `compute_value_target`（`:187`）：OR 取 `-1+child`、AND 取 `min(child)`——04 篇 `experience_collection.compute_value_target` 的论文原型。
    - `run_mcts`/`ucb_score`（`:510`/`:557`）、`backprop_value_towards_min`（`:629`）、`backpropagate`（`:640`）。
    - `Matchmaker`（`:353`）：per-theorem 成功/失败统计与权重（`mm_*`）。
    - `auto_formalize_problem`（`:764`）与 `sample_variants`（`:889`）：自然语言→Lean 形式化与变体生成的完整流程（含去重、语法/反例/循环一致性检查）。
  - `data/rl/{deepseek_prover,leanworkbook,numinamath}.py`、`scripts/`（含 `prover_eval.py`/`policy_eval.py`/`bench_inference.py`）、`tests/`、`web/`（React 监控面板）。

## 4. `real7_scripts/`：3584→256→64 价值头数据管线与训练

### 4.1 `prepare_leantree.py`（271 行）——根隔离的确定性切分
- schema `new_value_head.leantree-row.v1`；固定源 revision/size/sha256（`:14-16`）。
- Prompt 模板（`:17-19`）：`User: … Here're some theorems that may be helpful:\n\nSTATE:\n{state}\nTACTIC:\n\nAssistant:`。
- `state_to_text`（`:44`）：把 LeanTree 的 goals/hypotheses 渲染成 `name : type` + `⊢ goal`。
- `root_split`（`:80`）：按 `sha256(seed\0root_id)` 的前 16 位把 root 分到 80/10/10（train/validation/test）——**按 proof root 隔离**防止同树泄漏。
- **`value_class`（`:85`）**：`clipped = min(proof_depth, 64)`，`value_class = clipped - 1`（0..63）。
- `iter_nodes`（`:92`）：递归/扁平两种 LeanTree 结构都支持，产出 `sample_id/family_id/root_id/state/prompt/proof_depth/value_class/tactic`。
- `choose_rows`（`:168`）：按 `state_sha256` 分组去重，交叉 split 的 state 丢弃，取**已知成功深度最小**者（乐观最短距离上界），稳定排序后取前 N。
- `main`（`:229`）：校验源 size/sha256 → 输出 `{train,validation,test}.jsonl` + `manifest.json`（含 split_policy、depth_histogram、prompt_sha256）。

### 4.2 `extract_real7_features.py`（300 行）——冻结 backbone 特征缓存
- `model_fingerprint`（`:28`）：REAL7B（`FrenzyMath/REAL-Prover`，revision `fe76f68d…`，`reap-model-lock.json`，4 个 safetensors 共 `15_231_271_864` 字节，339 tensors）复算指纹。
- 加载 backbone（`:284`），`padding_side="left"`（`:283`），对每行算 **最后一个有效 token 的 hidden**（`last_hidden_state[:, -1, :]`，`:182`）并存为 fp16。
- Shard schema `new_value_head.feature-shard.v2`（`:9`）：含 `features [N,3584]`、`labels`、`depths`、`sample_ids/root_ids/family_ids/state_sha256s` 与全部 provenance SHA。
- `validate_shard`（`:143`）做**断点续传校验**（已存在且 provenance 一致则跳过）。
- 训练分片边界 `TRAIN_BOUNDARIES=(20000,40000,60000,80000)`（`:13`）；每到一个边界写 `boundary-{n}.json`；支持 `--deadline-epoch/--stop-file` 在边界安全停机（`:211-227`）。
- `exclusive_run_lock`（`:83`）用 `fcntl.flock` 保证单 worker 独占 GPU/output。

### 4.3 `train_value_head.py`（239 行）——离线训 64-bin 头
- `load_features`（`:15`）：按 manifest 的 shard 顺序拼接并校验 gap/hash/shape/dtype/`labels+1==depths`。
- **头结构（`:161`）：`Linear(3584,256)→SiLU→Linear(256,64)`**，参数量校验 `934,208`（`:162`）；分类 CE（`:179`）。
- 指标 `metrics`（`:57`）：NLL、accuracy、`expected_clipped_depth_mae`、`within_two`、`argmax_within_two_bins`、`pearson_expected_vs_clipped_depth`，并按深度桶（1-4/5-8/9-16/17-32/33-64）分组（`:72-83`）。
- `within_root_order_accuracy`（`:87`）：同一 proof tree 内状态对是否按已知深度正确排序（关键价值排序指标）。
- 训练：30 epoch、早停 patience 5、AdamW lr 1e-3 wd 0.01（`:130-136`），确定性算法（`:143`），验证用 NLL 选最佳（`:187`）。
- **产物（`:213-223`）**：`value-head.pt`（role `pretrained`）+ `value-head-initial.pt`（role `matched-random-initial`），schema `new_value_head.categorical-head.v2`，含 hidden_size=3584/classes=64/decode/online_target/initial_head_sha256/feature fingerprint/dataset manifest sha/optimizer，另写 `report.json` 与 `COMPLETE`。

### 4.4 `assess_value_head.py`（57 行）——进入搜索前的离线门禁
- `assess`（`:9`）：要求验收集 NLL 与 expected-MAE 相对 matched random head 改善、`pearson_expected_vs_clipped_depth > 0`、`within_root_order` 在 ≥100 对上准确率 > 0.5（`:26-31`）；全部通过才 `eligible_for_matched_search_ablation`；明确“离线信号不等于证明成功”。

### 4.5 其它
- `preflight_ms.py`、`download_leantree.sh`、`setup_lean427.sh`（环境准备）。

## 5. 特征分片 schema（速查）

```
feature-shard.v2:
  schema_version, source_sha256, dataset_manifest_sha256, model_fingerprint_sha256,
  hidden_size=3584,
  sample_ids[], root_ids[], family_ids[], state_sha256s[],   # 身份（防泄漏/去重）
  features float16 [N,3584], labels int64 [N] (0..63), depths int64 [N] (1..64)
feature-manifest.v2:
  source, source_sha256, model_fingerprint, hidden_size, dataset_manifest_sha256,
  dtype=float16, rows, rows_per_shard, shards[{path,rows,sha256,resumed}]
```

## 6. 张量/语义小结

| 项 | app（连续） | real7（64-bin 分类） |
|---|---|---|
| 头 | `H→256→1→Tanh` | `3584→256→64`（934,208 参数） |
| 输出 | `V∈[-1,1]` | 64 类 logits；`E[distance]=Σ softmax·[1..64]` |
| 标签 | normalized discounted return / `-min(depth,64)/64` | `value_class=min(depth,64)-1` |
| 损失 | MSE（或 Huber） | 交叉熵 |
| 部署 | `value_head.pt`（`reap.value-head.v1`） | `value-head.pt`（`new_value_head.categorical-head.v2`，role pretrained/R64） |

## 7. 与其它版本差异

- 与 **04 nanoproof**：本版把 nanoproof 的价值信号移植到 HF REAL-Prover；核心 `nanoproof/` 代码与 04 相同，新增移植文档与 AlphaProof 伪代码。
- 与 **01 app**：01 与 05 的 app 共享 `value_head.py`（连续标量头）；05 补充了 64-bin 生产版管线与 `distance` 输出模式。
- 与 **03 gpu_runtime**：03 的 `categorical_search_backend.py` 直接消费本版 `new_value_head.categorical-head.v2/v3` artifact（见 `_validate_artifact`）。

## 8. 想改造应先动哪里

- **换头结构/维度**：`real7_scripts/train_value_head.py:161`（64 类）或 `app/value_head.py:43`（标量）；参数数量校验在 `:162`。
- **改标签语义**：`prepare_leantree.py:85`（`value_class`）、`app/value_head.py:155`（`proof_depth_to_target`）。
- **改切分/防泄漏**：`prepare_leantree.py:80`（root_split）与 `:191-200`（cross-split drop）。
- **改特征**：`extract_real7_features.py:182`（取 hidden 的位置）与 `:28`（fingerprint）。
- **改门禁**：`assess_value_head.py:26-31`。
- **迁移/长训**：`PORTING_REAP_VALUE_HEAD.md`。

## 9. 对应教学篇

- `../04-价值头/`（3584→256→64、校准、within-root 排序）、`../05-LoRA/`（app 的 TTT 联合更新）、`../06-RL-RLVR/`（value target=负剩余深度）、`../02-预训练/`（数据切分/provenance）、`../08-V1-1/`（artifact 接入 gpu_runtime）。
