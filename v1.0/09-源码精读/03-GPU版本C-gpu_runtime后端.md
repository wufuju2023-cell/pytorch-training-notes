# 03 · GPU 版本 C：`gpu_runtime` 后端

> 源码根：`/mnt/gloway/projects/reap-agentic-v1-1-sync/categorical-runtime-fullv3-refresh-r8b/gpu_runtime/`（与 `/mnt/gloway/projects/reap-new-update-model/v1-result/reproduction/code/src/gpu_runtime/` 同族；含全部 backend/objective 的完整 32 文件版）。CPU 容器补丁在 `.../v1-result/reproduction/code/src/containers/cpu/patches/0001..0004`。
> 角色：V1-1 的**生产化 GPU 运行时**。一个常驻 HTTP 服务，用 `GpuActor` 串行化所有可变 GPU 操作；每个「定理/学习」是一个**会话（session）**，各自持有命名 LoRA adapter + value head + 独立 optimizer/RNG；`learn` 一次事件 = 一次事务性 optimizer step，失败回滚、隔离（quarantine）。支持多种 `backend`：toy / real / real-search / real-search-categorical / qwen35-search / verified-replay / mixed-replay。
> 教学链接：`../08-V1-1/`、`../06-RL-RLVR/`、`../04-价值头/`（64-bin 分类头）、`../05-LoRA/`。
> **【文档｜DOC-SRC3】**（doccode = `SRC3`）｜编号与 Tag 规范见《00-风格与编号规范》。

## 0. 文件树与职责

**【注 SRC3.0.1｜R-SRC3.0.1】（文件树与职责）**

```
gpu_runtime/
├── __init__.py                  # 导出 Backend/GpuActor/GpuRuntime/SessionStore/SnapshotStore/RealProverBackend 等
├── errors.py                    # 领域异常（GpuRuntimeError / DuplicateSessionError / VersionConflictError …）
├── identifiers.py               # validate_identifier（^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$）+ contained_path 防目录逃逸
├── schemas.py                   # ChatRequest / parse_chat_request / chat_response / value_chat_response（OpenAI 兼容）
├── backend.py                   # Backend Protocol + BackendLearnResult
├── actor.py                     # GpuActor：单 worker FIFO，串行化全部 GPU 操作 + 指标
├── session_store.py             # SessionState / SessionStore（线程安全逻辑层）
├── snapshot_store.py            # 快照（session+backend state）落盘
├── experience_store.py          # 已完成定理经验的不可变发布/加载
├── learner_release_store.py     # learner checkpoint/release 内容寻址存储（CAS）+ CAS 原语
├── runtime.py                   # GpuRuntime：编排 create/delete/retire/policy/value/learn/snapshot/restore/experience
├── server.py                    # HTTP 路由（/sessions/<id>/policy/v1/chat/completions 等）+ build_backend
├── real_backend.py              # RealProverBackend：REAL-Prover + 命名 LoRA + 标量 value head（无 torch 导入时延迟加载）
├── search_backend.py            # RealSearchBackend：visit/backup 目标 + sigmoid 头 + 可选 KL guard
├── categorical_search_backend.py# CategoricalSearchBackend：64-bin 分类价值头（P64/R64 artifact）
├── qwen35_backend.py            # Qwen35SearchBackend：Qwen3.5 thinking context + 严格 action 格式
├── verified_backend.py          # VerifiedReplayBackend：成功轨迹 replay（分类头，负 longest-action）
├── success_finalize_backend.py  # learn_success()：同会话终止 CE/折扣价值
├── mixed_backend.py             # MixedReplayBackend：9 replay + 1 Mathlib SFT
├── toy_backend.py               # 内置最小后端（CPU/测试）
├── search_objective.py          # visit/backup 纯契约：value_to_distance / distance_to_value / visit 分布 / 目标
├── verified_objective.py        # replay 纯契约：content-addressed rows → 准备样本
├── mixed_objective.py           # 9/1 采样器 + 纯契约
├── success_finalize_objective.py# 终止目标契约 + prepare_success_event
├── learner.py                   # LearnerCoordinator：持久 learner（每步 checkpoint + writer lease）
├── release_head.py              # 发布头：把 committed checkpoint 原子发布为 model release
├── mixed_learner_store.py / continual_mixed_store.py / continual_mixed_learner.py
├── mixed_learner.py / frozen_base_manifest.py / lean_action_format.py / identifiers.py
├── learner_release_store.py / experience_store.py / snapshot_store.py
└── (CPU patches) 0001-training-endpoints-and-value.patch … 0004-selection-value-refresh.patch
```

## 1. 基础设施

### 1.1 错误与标识符

**【代码 SRC3.1.1｜Cd-SRC3.1.1】（错误与标识符）**
- `errors.py`：`GpuRuntimeError` 基类；`InvalidIdentifierError`（继承 ValueError）、`DuplicateSessionError`、`SessionNotFoundError`、`VersionConflictError`（stale policy_version）、`EventConflictError`（event_id 复用不同 payload）、`SnapshotIntegrityError`、`SnapshotNotFoundError`。
- `identifiers.py:11-19`：`validate_identifier` 限制标识符为 64 字符内的安全串；`contained_path`（`:22-29`）把路径限制在 root 内，防逃逸。

### 1.2 协议与 schemas

**【代码 SRC3.1.2｜Cd-SRC3.1.2】（协议与 schemas）**
- `backend.py:19-48`：`Backend` Protocol，方法 `create_session/delete_session/policy/value/learn/export_session/import_session/experience_contract/experience_weights/initialize_from_experience`；`BackendLearnResult`（`adapter_metadata/value_metadata/optimizer_metadata/detail`）。
  - `policy` 返回 `(候选文本, 逐 token logprob)`；`value` 返回 Reap 期望的 JSON 内 score；`learn` 一次幂等保护的更新。
- `schemas.py`：`ChatRequest`（model/messages/n/temperature/max_tokens/logprobs，`prompt` 属性拼接 content，`:21-23`）；`parse_chat_request`（`:26-66`，校验 role/范围 n∈[1,64]、max_tokens∈[1,8192]）；`chat_response`（`:69-95`，含 token_logprobs 与 `policy_version`）；`value_chat_response`（`:98-102`，`content = {"score": score}`——Reap 侧再取负）。

### 1.3 `GpuActor`：并发方案（`actor.py`）

**【代码 SRC3.1.3｜Cd-SRC3.1.3】（`GpuActor`：并发方案）**
- 后台单线程消费 `queue.Queue`（`:41-42`），`submit(fn)`（`:77-93`）投递并阻塞等待 done（`:90`），返回结果或抛出原始异常（`:91-92`）。
- 指标（`:95-121`）：submitted/started/completed/failed/active/max_active/queued/max_queued、queue_wait、execution 时长；`schema_version="reap.gpu-actor.metrics.v1"`。
- `close()`（`:123-131`）投递 None 哨兵并 join。**所有可变 GPU 操作都经 actor**，HTTP/会话层可并发。

### 1.4 `SessionStore`（`session_store.py`）

**【代码 SRC3.1.4｜Cd-SRC3.1.4】（`SessionStore`）**
- `SessionState`（`:16-89`）：session_id/role(`theorem|learner|actor`)/theorem_id/lineage/completed/policy_version/adapter_metadata/value_metadata/optimizer_metadata/reference_metadata/`buffer_metadata`（events、pending_event_ids、consumed_event_ids）/`event_receipts`/created_at。
- `snapshot()`（`:34-53`）序列化为 `reap.gpu.session.v1`；`from_snapshot()`（`:55-89`）校验 schema/role/version（learner 不得带 theorem/completed；actor 必须有 theorem+release lineage）。
- `SessionStore`（`:92-153`）：RWMutex 风格的 `_guard` + 每会话 RLock；`locked()`（`:133-144`）上下文管理器、`replace_locked()`（`:146-153`）。

### 1.5 `GpuRuntime`：编排核心（`runtime.py`）

**【代码 SRC3.1.5｜Cd-SRC3.1.5】（`GpuRuntime`：编排核心）**
- 构造（`:64-113`）：持有 backend/actor/sessions/snapshots/experiences/learner_releases；`max_resident_sessions` 限制常驻会话；保留 `_resident_session_ids`/`_retired_session_ids`/`_unusable_sessions`（隔离）。
- `_invalidate_snapshot_reuse`（`:115-120`）：任何后端操作前使快照复用 witness 失效并递增 revision。
- **`create_session`**（`:138-236`）：校验 role 与 lineage（theorem/learner/actor 各自约束）；支持从 `experience_id`（旧经验）或 `model_release_sha256`（learner release）初始化；先占用 residency 名额再调后端，失败则清理或隔离。
- **`learn`**（`:425-497`）——一次事务：
  1. 幂等：同 event_id 同 digest 返回缓存响应（`applied=False, idempotent=True`，`:445-451`）；不同 digest 抛 `EventConflictError`。
  2. 版本：`state.policy_version != expected` → `VersionConflictError`（`:452-455`）。
  3. **快照回滚点**：`old_session=state.snapshot()`、`old_backend=backend.export_session()`（`:456-459`）。
  4. 标 `pending` → `backend.learn` → JSON 序列化校验（`allow_nan=False`）→ `policy_version += 1`、写 receipt、标 `consumed`（`:460-491`）。
  5. 任何异常 → `_rollback`（`:492-494`，`:128-136`），rollback 失败则 quarantine。
- `snapshot(for_experience=True)`（`:499-530`）要求 theorem 身份、version≥1、无 pending，并附 `experience_contract`；`publish_experience`（`:532-548`）；`restore`（`:550-577`）同会话回滚；`retire_session`（`:252-324`）写不可变 retirement 记录（prepared/released）并占 tombstone；`retirement_receipt`（`:326-392`）。
- `policy`/`value`（`:394-423`）：经 actor + 会话锁，返回 `chat_response`/`value_chat_response`，带当前 `policy_version`。

### 1.6 `server.py`

**【代码 SRC3.1.6｜Cd-SRC3.1.6】（`server.py` 路由）**
- 路由 `/sessions/<id>/<action>`（正则 `^/sessions/([^/]+)(?:/(.*))?` 整串匹配，`:32`）；`do_POST`（`:161-215`）：创建会话、`policy/v1/chat/completions`、`value/v1/chat/completions`、`learn/v1`、`snapshot/v1`、`experience/v1`、`restore/v1`、`retire/v1`。
- `do_GET`（`:128-159`）：`/initialization-contract`、`/health`（含 actor 指标）。
- `build_backend`（`:291-369`）按 `--backend` 选类；参数含 `--gamma`（search 必需）、`--verified-dataset-root/--verified-max-distance`、`--categorical-value-artifact*`、`--thinking-budget`、`--policy-scoring`、`--max-post-update-kl`、`--learner-profile continual-mixed-v3` 等；严格互斥校验（默认 backend=`real`）。

## 2. 后端实现

### 2.1 `real_backend.py`（`RealProverBackend`，L67）—— 默认 TTT 后端

**【代码 SRC3.2.1｜Cd-SRC3.2.1】（`RealProverBackend`：默认 TTT 后端）**
- 常量：`TARGET_MODULES`（7 投影，L24-27）、`MAX_SCORING_CANDIDATES/SEQUENCE_TOKENS/DEFERRED_SCALARS`（L51-53）。
- `PolicyScoringConfig`（L30-48）：`tokenwise`（默认）/`candidate_chunks`/`tokenwise_deferred`。
- 构造（L70-135）：延迟 import torch/peft/transformers；`expected_hidden_size=3584`；LoRA `r=16, alpha=32, dropout=0, init_lora_weights=True`（L117-127，零 B ⇒ 等价 base）；`get_peft_model(base, cfg, adapter_name="__bootstrap__")`。
- 每会话状态 `_RealSession`（L56-64）：value_head/optimizer/steps/examples/rng_seed/cpu+device RNG。
- `create_session`（L229-269）：会话 id 不能含 `.`/`__bootstrap__`；为每会话 `add_adapter` 并建 **value head `Linear(3584,256)→SiLU→Linear(256,1)→Tanh`**（L245-250，fp32）与 AdamW（params: adapter lr=1e-4、head lr=3e-4，L252-257）；初始化返回 base-equivalent 声明。
- 采样评分（`_score_generated_tokens` L279、`_score_generated_candidates` L402、`_score_generated_candidates_deferred` L336）：用 `log_softmax` 重算 `log P(token|prefix)`，`logits_to_keep` 限内存。
- `policy`（L473-530）：`generate` 采样，修剪首个 EOS；按 scoring 模式算 token logprobs。
- `value`（L532-541）：`hidden_states[-1][:,-1,:].float()` → value_head → float（**标量 V ∈ (-1,1)**）。
- **`learn`（L569-658）**：`reward`∈[-1,1] 且正奖励需 `terminal_verified`；构造 `input_ids=[prompt|target]`，`labels` prompt 段 -100；三种损失：
  - `policy_loss = reward * token_nll`（L601-602）
  - `kl = kl_div(reference‖current)/target_len`（L607-613）
  - `value_loss = mse(V(hidden_prompt_last), reward)`（L614-617）
  - `total = policy_loss + kl_beta*kl + value_coefficient*value_loss`（L618）
  默认 `kl_beta=0.02, value_coefficient=0.5, lr=1e-4, value_lr=3e-4, max_grad_norm=1.0`（L78-82）。全部有限性校验 + `clip_grad_norm_`。
- `experience_contract`（L713-742）：对**已加载冻结权重**做 sha256（分块，跳过 lora_），绑 config/tokenizer；`experience_weights`/`initialize_from_experience`（L750-794）只搬运 adapter/value，禁用 optimizer/RNG/counters。

### 2.2 `search_backend.py`（`RealSearchBackend`，L37）

**【代码 SRC3.2.2｜Cd-SRC3.2.2】（`RealSearchBackend`）**
- 继承 RealProverBackend，构造带 `gamma`（严格 0<γ<1）、`value_floor`、`max_sequence_tokens`、`max_candidates`、可选 `max_post_update_kl`、`success_dataset_root`（L42-66）。
- **value head 换成 sigmoid**（`create_session` L92-99）：`value_head[-1]=Sigmoid()`，参数与 RNG 不变。
- `value`（L101-103）：`value_to_distance(sigmoid_prob, gamma, floor)` → **距离域**。
- `learn`（L276-402）：
  - 非终局 backup 目标 `search_visit_backup`：candidate 有 `visits`、`raw_logprob`、`behavior_version`；按 visit 分布加权。
  - 逐候选：`nll = -Σ log P(target tokens)`；`kl = visit-weighted prefix KL(current‖frozen base)`；`term = weight*(nll + kl_beta*kl)`，`term.backward()`（L307-339）。
  - 单次 **prompt-only value MSE**（`_search_value_loss` L105-121）。
  - `total = policy_total + kl_beta*kl_total + value_coefficient*value_loss`（L349）。
  - **可选 KL guard**（L356-375）：optimizer.step 后测 `_measure_post_update_kl`（L232-274），超限抛 `KLGuardExceeded` → runtime 回滚。
  - 详细 `parameter_diffs`（`_fingerprint_changes` L215-230，session_fingerprints L203）。

### 2.3 `categorical_search_backend.py`（`CategoricalSearchBackend`，L127）—— 64-bin 分类头

**【代码 SRC3.2.3｜Cd-SRC3.2.3】（`CategoricalSearchBackend`：64-bin 分类头）**
- 常量（L32-50）：`SUPPORT_MAX=64`、`HEAD_KIND="linear-3584-silu-256-linear-64"`、`MODEL_REVISION=fe76f68d…`、`MODEL_WEIGHT_BYTES=15_231_271_864`、`TRAIN_BOUNDARIES={20k,40k,60k,80k}`、artifact schema v2/v3。
- `real7_feature_fingerprint`（L66-93）：对 REAL7B 目录四权重 + config/tokenizer 做 sha256，绑定 `reap-model-lock.json`（revision、339 tensors）。
- **`distance_two_hot`（L96-124）**：把实数距离投影到相邻 bin 的 two-hot（bin 64 饱和 ≥64），返回 `weights[64]`。
- 构造（L132-152）：校验 artifact 文件 SHA256、role(`pretrained|matched-random-initial`)、feature fingerprint、frozen base manifest；`_validate_artifact`（L154-221）校验 schema/`classes==64`/`hidden_size==3584`/optimizer 元数据/`state_dict` 形状 `0.weight(256,3584),2.weight(64,256)` 与初始头 sha。
- 会话（L244-271）：**head 换成 `Linear(3584,256)→SiLU→Linear(256,64)`**，加载 P64/R64 artifact 状态，新建 AdamW（v0、空状态）。
- `value`（L273-286）：`softmax(logits) · support(1..64)` 求**期望距离**，clip 到 [1,64]。
- `_categorical_loss`（L288-305）：two-hot 目标 + 交叉熵（`-(target*log_softmax).sum`），返回期望距离审计。
- `_search_value_loss`/`_success_value_loss`（L307-313）分别用 `example["value_trace"]["distance"]` 与 `-row["return"]`。

### 2.4 `qwen35_backend.py`（`Qwen35SearchBackend`，L89）

**【代码 SRC3.2.4｜Cd-SRC3.2.4】（`Qwen35SearchBackend`）**
- 常量（L26-68）：`THINKING_MODE="shared-state-context-v1"`、Reap prompt 包裹、`LANGUAGE_TARGETS`（含 Qwen3.5 的 in/out_proj）。要求 `Qwen3_5ForConditionalGeneration` 且 text hidden=4096（L125-138）。
- **核心：每个 state 只 think 一次**（`_ensure_context` L228-268）：在 `_session_rng` 事务内 `_generate_thought_tokens`（L212-222，temperature 0.6/top_p .95/top_k 20，≤thinking_budget），必须以 `</think>` 自然收尾或正好用满 budget；思考 prefix 冻结（`prefix_ids`），旧状态更新后保持（L159）。
- `_tokenize_prompt`（L270-276）只查已记录上下文，learn 时绝不新造 behavior。
- `policy`（L330-371）/严格 action 模式 `_policy_strict_actions`（L401-448）：`normalize_lean_action` 归一 + canonical action 重编码评分，记录 `ACTION_TRACE_SCHEMA` 审计。
- `learn`（L464-487）：校验候选 action/version/logprob 与已记录 policy generation 一致，再调用父类 learn。

### 2.5 `verified_backend.py`（`VerifiedReplayBackend`，L21）

**【代码 SRC3.2.5｜Cd-SRC3.2.5】（`VerifiedReplayBackend`）**
- 独立 profile（不继承 search 的 learn）：head = `Linear(hidden,256)→SiLU→Linear(256,max_distance)`（L83-84），**分类头**；`value()` 返回期望距离（L95-108）。
- `_config`（L58-74）：记录 tokenization（tactic 单独 tokenize + 恰好一个 EOS）、`head=linear-silu-linear-categorical`、`value_loss=categorical_cross_entropy_exact_integer_class`、`policy_loss=joint_tactic_plus_one_EOS_negative_log_prob`、support 1..max_distance。
- **`_learn_prepared`（L221-316）**：每样本 prompt+tactic（+EOS），逐样本：
  - `nll`（tactic+EOS 联合 NLL）
  - `kl`（prefix KL current vs frozen base）
  - `value_loss = cross_entropy(logits, value_class)`，其中 `value_class = -return-1`（`verified_objective.py:76`）
  - `loss = (nll + kl_beta*kl + value_coefficient*value_loss) * sample_weight`
  - 可选 KL guard；`extract_checkpoint_weights`（L136-180）从 learner checkpoint 提取 adapter/value（形状校验）。
- `_load_dataset`（L200-204）走 `cpu_runtime.verified_trajectory.load_verified_dataset`。

### 2.6 `success_finalize_backend.py`（`learn_success`，L12）

**【代码 SRC3.2.6｜Cd-SRC3.2.6】（`learn_success`：同会话终止目标）**
- 同会话终止目标：`prepare_success_event`（job）。逐行 policy `nll + kl_beta*kl`、value 用 `_success_value_loss`（可分类/标量），权重 `sample_weight`；可选 KL guard（L94-107）。返回带 `online_update_consumed_by_later_generation=False`。

### 2.7 `mixed_backend.py`（`MixedReplayBackend`，L20）

**【代码 SRC3.2.7｜Cd-SRC3.2.7】（`MixedReplayBackend`）**
- 继承 VerifiedReplayBackend；`_config`（L35-45）加入 mixture `{replay:9, mathlib_sft:1}`、`sample_weight=1/10`、源 profile。
- `_load_mathlib_dataset`（L47-51）走 `cpu_runtime.mathlib_trajectory.load_mathlib_dataset`；`_source_loss_totals/_source_loss_detail`（L58-70）分源统计损失。

## 3. 目标契约（objective）

**【代码 SRC3.3.1｜Cd-SRC3.3.1】（目标契约 objective）**

- **`search_objective.py`**：`OBJECTIVE_KIND="search_visit_backup"`；`value_to_distance`（L53）/`distance_to_value`（L62）；`visit_distribution`（L71）；`weighted_joint_nll`（L81，参考式 `-Σ_a w_a Σ_token log p`）；`prepare_search_event`（L97）从 OR/AND 节点 `value_sum/visits` 求 `distance=-mean`，`value_target=γ^(distance-1)` 并 floor。
- **`verified_objective.py`**：`OBJECTIVE_KIND="verified_success_replay"`；`prepare_verified_event`（L33）只接受 `{dataset_sha256,row}` 引用，`return∈[-max_distance,-1]`，`value_class=-value-1`。
- **`success_finalize_objective.py`**：`OBJECTIVE_KIND="verified_success_discounted_v1"`；`CONTRACT`（L18-26）；`prepare_success_event`（L29）校验 course acceptance pin、dataset/replay envelope 一致。
- **`mixed_objective.py`**：`OBJECTIVE_KIND="verified_replay_mathlib_sft"`；`SOURCE_COUNTS={replay:9, mathlib_sft:1}`、`BATCH_SIZE=10`、`SAMPLE_WEIGHT=0.1`；`make_mixed_sampler`（L179）/`next_mixed_batch`（L215）纯函数采样（独立游标、无全局 RNG）。

## 4. Learner 与发布头

**【代码 SRC3.4.1｜Cd-SRC3.4.1】（Learner 与发布头）**

- **`learner.py`**：`_WriterLease`（L57-88）用 OS 文件锁（fcntl/msvcrt）保证单 coordinator；`describe_dataset`（L91-104）绑定 dataset 与 lineage；`publish_checkpoint`（L107-129）；`LearnerCoordinator`（L132-378）：
  - `_start_run`（L159-187）持久创建 run/control/journal 并建 learner 会话；
  - `train_next`（L259-296）：`_prepare_next` 采样 9/1 批 → 写 intent → `runtime.learn` → 写 data-receipt → `runtime.snapshot` → `store.create_checkpoint` → 写 commit；**任何异常 `blocked=True`，绝不重试不确定步**；
  - `restore`（L318-370）从 checkpoint 恢复；`publish`（L298-316）。
- **`release_head.py`**：`read_training_commits`（L23）校验 intent/commit 链；`publication_state`（L78）状态机；`enable`（L136）/`publish`（L152）把 committed checkpoint 原子发布为 release（`published-<step>.json` + `model_release_sha256`）；`reconcile_published`（L172）离线对账。

## 5. 关键超参汇总（各 backend）

**【例 SRC3.5.1｜E-SRC3.5.1】（关键超参汇总）**

| backend | hidden | head | policy | kl | value | 备注 |
|---|---|---|---|---|---|---|
| real | 3584 | 3584-256-1-tanh | `reward·NLL` | `kl_div/l` β=0.02 | MSE `γ≡0.99` | 默认 |
| real-search | 3584 | 3584-256-1-**sigmoid** | visit 加权 joint NLL | prefix KL，β 可调 | MSE `γ^(d-1)` | 距离域 |
| real-search-categorical | 3584 | 3584-256-**64** | 同 search | 同 search | **two-hot CE** | 64-bin |
| qwen35-search | 4096 | 同 search | 同 search（严格 action） | 同 search | 距离域 | thinking |
| verified-replay | 3584 | 3584-256-**D** | tactic+EOS 联合 NLL | prefix KL | **CE 整数类** `-return-1` | 离线 replay |
| mixed-replay | 3584 | 同 verified | 9/1 联合 | 同 verified | 同 verified | 复现比例 |
| success-finalize | 3584 | 同 search/分类 | 均匀 action CE+EOS | prefix KL | 折扣价值 | 终止一次 |

共享超参（`real_backend.py:78-82`）：`learning_rate=1e-4`、`value_learning_rate=3e-4`、`kl_beta=0.02`、`value_coefficient=0.5`、`max_grad_norm=1.0`；LoRA `r=16, α=32`。

## 6. 并发与 checkpoint

**【注 SRC3.6.1｜R-SRC3.6.1】（并发与 checkpoint）**

- **并发**：`GpuActor` 单线程串行所有后端访问（`actor.py`），HTTP 层多线程但都 `actor.submit`；每会话另有 `SessionStore.locked` 锁。`learn` 是**全或无事务**：先快照 session+backend，异常 `_rollback`，回滚失败会话进入 `_unusable_sessions`（隔离，不再服务）。
- **checkpoint**：
  - 会话级快照 `SnapshotStore`（session state + backend base64 torch-save），`schema reap.gpu.session.v1` + backend 各自 schema；
  - learner 内容寻址 checkpoint/release（`learner_release_store.py` CAS）；
  - 退休（retire）写不可变 `prepared.json`/`released.json`，`tombstone_scope=current_runtime`。

## 7. CPU 容器补丁（`containers/cpu/patches/0001..0004`）

**【代码 SRC3.7.1｜Cd-SRC3.7.1】（CPU 容器补丁 0001–0004）**

这些 patch 把 GPU runtime 的协议接到 Lean 训练观察者：
- **0001**（`Generator.lean`/`PremiseSelection/API.lean`）：端点支持环境变量覆盖 `REAP_POLICY_ENDPOINT/REAP_VALUE_ENDPOINT/REAP_PS_ENDPOINT`。
- **0002**（`TreeSearch.lean`）：新增 `MCTSObserver` 与 `observe?`，在 selection/generation/eval/backup/checkpoint 处发 JSON 事件（不改搜索状态）。
- **0003**（`Generator.lean`）：`REAP_OBSERVER_PATH` 存在时对 value 响应**严格**——无 choices 或 3 次重试失败即 `throwError`，不返回 `-1000.0` 哨兵。
- **0004**（`TreeSearch.lean` + `Observer.lean` + `RolloutSink.lean`）：新增 `SelectionValueRefresh`（仅选择用的版本化价值刷新，不重算历史 Q），`selectionValue?` 参与 PUCT，`policyVersionRef?` 共享给 observer。

## 8. 与其它版本差异

**【注 SRC3.8.1｜R-SRC3.8.1】（与其它版本差异）**

- 与 **01 V1 app**：app 是单模型全局 RLock 的极简在线版；gpu_runtime 是**多会话隔离 + 事务回滚 + 多目标 backend + 内容寻址持久化**的生产版。
- 与 **04 nanoproof**：nanoproof 自研栈端到端训练；gpu_runtime 复用 HF REAL-Prover/Qwen3.5 + PEFT，走「服务 + session」模式。
- 与 **06 CPU V1**：cpu_runtime 是控制面/数据面（Lean 观察者、collector、matchmaker）；gpu_runtime 是模型面。

## 9. 想改造应先动哪里

**【注 SRC3.9.1｜R-SRC3.9.1】（想改造应先动哪里）**

- **换 head/损失**：`real_backend.py` 的 head（L245）与 `learn`（L569）；search/categorical/verified 各自 override `create_session/_search_value_loss/_success_value_loss`。
- **64-bin 支持范围**：`categorical_search_backend.py:46 SUPPORT_MAX` 与 `distance_two_hot`（L96）。
- **并发粒度**：`actor.py`（单 worker）；若要并行需改 actor/多进程。
- **会话生命周期/持久化**：`runtime.py`（create/learn/rollback/retire）与 `snapshot_store.py`/`learner_release_store.py`。
- **目标语义**：`search_objective.py`/`verified_objective.py`/`soon`。
- **Lean 侧接入**：patches 0001–0004；端点环境变量与 observer JSON schema。

## 10. 对应教学篇

**【注 SRC3.10.1｜R-SRC3.10.1】（对应教学篇）**

- `../08-V1-1/`（生产化运行时/多会话）、`../06-RL-RLVR/`（visit 加权/REINFORCE/KL）、`../04-价值头/`（64-bin 分类与校准）、`../05-LoRA/`（命名 adapter）。
