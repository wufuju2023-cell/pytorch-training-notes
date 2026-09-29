# 02 · GPU 版本 B：容器 Qwen3-1.7B 4bit QLoRA SFT

> 源码位置：`a@my-new-linux:/mnt/gloway/projects/9-27-hsy-modelscope-备份/mnt-workspace/`（ModelScope / A10 云容器备份）。
> 注意（实测路径）：`train_full_supervised.py`、`supervisor.sh`、`a10_autotune/` **不在** `full_modelscope/src/final*`（那里为空），而是在备份根的 zip 内：
> - `_full_supervisor_local_build_20260925_FINAL_3VAL_ADAPTER.zip` → `train_full_supervised.py`（855 行）/ `supervisor.sh` / `test_dataset_safety.py`
> - `_full_supervisor_local_build_20260925_FINAL.zip` → 早期版本（约 570 行）
> - `_full_a10_autotune_20260925_bundle_v3.zip` → `a10_autotune/a10_autotune.py`（478 行）
> `inference_v1/**` 在 `full_modelscope/inference_v1/`。
> 角色：V1 主线里的**有监督长训**分支——把 Full caller 的 `{"calls":[…]}` 多步动作序列 SFT 到 Qwen3-1.7B。
> 教学篇：`../03-SFT/`（SFT/数据格式）、`../05-LoRA/`（LoRA/QLoRA/4bit）、`../01-基础/`（AMP/梯度检查点/优化器）。
> **【文档｜DOC-SRC2】**（doccode = `SRC2`）｜编号与 Tag 规范见《00-风格与编号规范》。

## 0. 文件树与职责

**【注 SRC2.0.1｜R-SRC2.0.1】（文件树与职责）**

```
备份根（zip 解出后）:
├── train_full_supervised.py   # 终版：target 保留的 SFT、断点续训、早停、自动扩轮、smoke 门禁（855 行）
├── supervisor.sh              # 编排：token 审计 → 逐候选 autotune → 选配置 → 全量训练（269 行）
├── test_dataset_safety.py      # 单测：截断/target 保护/门禁签名/早停/延长逻辑（84 行）
├── start_nohup.sh / requirements-colab.txt / README.md / REVIEW.md / COLAB_ONE_CELL.md
├── a10_autotune/a10_autotune.py  # 单卡 A10 吞吐调优器（478 行）
├── a10_autotune/run_priority_probe.py / README.md
└── full_modelscope/inference_v1/
    ├── generate_pass1.py / parse_pass1.py / smoke_generate.py / parse_smoke.py / pass48.py
    └── data/
        ├── train_full_qwen3_1p7b_lora.py   # 更早的简化训练脚本（271 行）
        ├── prepare_smoke.py                # calls 规范化 + smoke 子集构造（165 行）
        ├── build_val400.py                 # 构造 canonical v4 val400 切分（106 行）
        └── audit_contract.py               # 数据集契约审计（72 行）
```

## 1. 数据格式：`{"calls":[...]}`

**【注 SRC2.1.1｜R-SRC2.1.1】（数据格式 `{"calls":[...]}`）**

每条样本一行 JSON（`validation.jsonl` / `train.jsonl`），关键字段：

```
ordinal, proof_gap, materialized_dsl_thinking, public_proof_state,
target: { "calls": [ {"action_id"|"macro_id"|"action"|"macro": ...,
                      "typed_params": {...}}, ... ] }
```

组织成 prompt / target 两段（`train_full_supervised.py:125-140`）：

```
You are a structured Full caller. Read the proof gap, materialized proof plan,
and current public proof state. Emit only a JSON object with a calls array.
Each call must contain action_id or macro_id and typed_params.
<proof_gap>…</proof_gap>
<dsl_thinking>…</dsl_thinking>
<public_proof_state>{compact json}</public_proof_state>
<answer>{"calls":[...]}</answer>
```

- `public_proof_state` 用 `json.dumps(..., sort_keys=True, separators=(",",":"))` 紧凑化（`:127`）。
- 监督 target = `<answer>` 之后的紧凑 JSON（`:138-140`）。
- **契约校验** `validate_prepared_row`（`:23-51`）：`target.calls` 必须是 list；每个 call 必须有动作名与 `typed_params` dict；空 `typed_params` 仅允许 `or_intro/assert_open/assert_close`（`:20, 39`）；禁止遗留 executor 字段 `output/outputs`（`:41`）；递归禁止裸 host 引用 `H\d+`（`:19, 46`）。
- **规范化**（`inference_v1/data/prepare_smoke.py:37-89`）`canonicalize_calls`：剥离 executor 拥有字段，把 host 句柄 `H<N>` 按首次出现顺序改写为 `H_SLOT_<i>`（`:54-63`），统计写入 manifest。

## 2. `train_full_supervised.py`（终版 855 行）

### 2.1 断点与门禁基础设施

**【代码 SRC2.2.1｜Cd-SRC2.2.1】（断点与门禁基础设施）**
- `jsonl_fingerprint`（`:61-71`）：对文件算 sha256 + 行数 + 字节数；**刻意不含挂载路径**，便于 Colab 换挂载点续训。
- `config_signature`（`:74-86`）：model/precision_mode/max_seq_len/min_prompt_tokens/batch_size/gradient_accumulation/gradient_checkpointing/attention_implementation + train/validation 指纹。
- `verify_smoke_receipt`（`:89-94`）：full 跑必须携带与当前签名完全一致的 `SMOKE_PASS` 收据。
- `atomic_json`（`:54-58`）临时文件 + `os.replace`；`append_jsonl`（`:225-230`）`flush + os.fsync`。

### 2.2 `FullDataset`：target 保留的因果 LM 数据集

**【代码 SRC2.2.2｜Cd-SRC2.2.2】（`FullDataset`：target 保留的因果 LM 数据集）**
- `prompt`（`:125-136`）/`target`（`:138-140`）见 §1；`_raw_ids`（`:142-145`）prompt 带特殊 token、target 不带。
- **`_fit`（`:147-158`）**：`prompt_budget = max_seq_len - len(target_ids)`；若 `< min_prompt_tokens` 则报错；否则超长时**只截断 prompt 尾部**（`prompt_ids[:prompt_budget]`），**绝不截断 target**（注释 `:156-157`）。
- `_compute_token_stats`（`:160-189`）：统计 `target_too_long_rows`、`zero_supervised_rows` 等；构造时任一非零即抛错（`:118-123`，fail-closed）。
- `__getitem__`（`:194-202`）：`ids = prompt_ids + target_ids`；`labels = [-100]*len(prompt_ids) + target_ids`（**token mask：只对 target 计算损失**），断言长度与监督 token 数。
- `collate`（`:205-215`）：右侧 pad；`input_ids` 用 pad_id、`labels` 用 `-100`、`attention_mask` 0/1。形状 `[B,T]`，`T ≤ max_seq_len`（默认 6144）。

### 2.3 检查点与最佳 adapter

**【代码 SRC2.2.3｜Cd-SRC2.2.3】（检查点与最佳 adapter）**
- `save_checkpoint(...)`（`:253-299`）：目录名 `checkpoint-<kind>-step-<global_step>-epoch-<e>-batch-<b>`；先写 `.tmp`（`save_pretrained` + `trainer_state.pt` 含 optimizer/scheduler/scaler/state 与 `torch_rng/cuda_rng/python_rng`），原子 `os.replace`，写 `latest.json`（`:298`）；同游标已存在则保留旧目录（`:291-296`）。
- `save_best_adapter`（`:302-320`）：不可变 `best_val_adapters/best-step-<g>-eval-<i>` + 原子提升 `best_val.json`。
- `evaluate`（`:323-342`）：`eval()+no_grad+autocast`，按样本数加权平均验证 loss。

### 2.4 主流程 `main`（`:349-851`）

**【代码 SRC2.2.4｜Cd-SRC2.2.4】（主流程 `main`）**
关键参数：`--model` 默认 `Qwen/Qwen3-1.7B-Base`（`:351`）；`--precision-mode {4bit,fp16,bf16}`（`:356`）；`--epochs 5`（`:357`）；`--max-seq-len 6144`（`:358`）；`--min-prompt-tokens 256`（`:359`）；`--batch-size 1`、`--gradient-accumulation 8`（`:360-361`）；`--learning-rate 1e-4`（`:362`）；`--evals-per-epoch 3`（`:365`）；`--expected-validation-rows 400`（`:367`）；`--gradient-checkpointing on`（`:374`）；`--attention-implementation sdpa`（`:375`）；`--auto-extend-epochs 2`、`--min-relative-improvement 0.005`、`--early-stopping-patience 3`（`:370-372`）。

1. **dry-run / token-audit-only**（`:410-443`）：只校验数据契约与 token 统计，不加载 CUDA。
2. **4bit QLoRA**（`:464-474`）：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)`；随后 `prepare_model_for_kbit_training(base_model)`（`:476-477`）。
3. **梯度检查点**（`:478-480`）：`gradient_checkpointing_enable()` + `enable_input_require_grads()`；`use_cache=False`（`:481`）。
4. **LoRA**（`:490-497`）：`r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"`，7 投影层；续训 `PeftModel.from_pretrained(..., is_trainable=True)`（`:483-488`）。
5. 优化器 AdamW 只对可训参数（`:519`）；线性 warmup = 5% 总步（`:526-527`）；`GradScaler(enabled=dtype==fp16)`（`:528`）。
6. **自动扩轮**（`:520-527`）：`max_epochs = epochs + auto_extend_epochs`；`scheduler_total_steps = steps_per_epoch*max_epochs`。
7. **续训**（`:581-593`）：校验 signature/training_contract 后恢复 optimizer/scheduler/scaler/RNG。
8. **训练循环**（`:673-791`）：每 epoch 带 seed 的 generator shuffle（`:676-685`）；`skip_batches` 中途续训（`:686,694-696`）；梯度累积 `loss = raw_loss / window_size`，窗口满才 `scaler.step/update/zero_grad/scheduler.step`（`:706-717`）；记录 loss/lr/tokens/tokens_per_second/显存（`:722-735`）。
9. **评估时机** `epoch_evaluation_steps`（`:233-238`）：每 epoch 均匀 3 个点（611 步 → {204,408,611}，单测 `test_dataset_safety.py:73-75`）；或 `--eval-every` 固定间隔（`:741-745`）。
10. **最优与早停** `run_evaluation`（`:606-671`）：相对最优提升 ≥ `min_relative_improvement` 才刷新 `best_validation_loss` 并 `save_best_adapter`；否则 `evals_without_improvement++`；full 且过最少 epoch 且连续 patience 次无改进则早停（`:626-639`）。
11. **自动延长**（`:767-784`）：名义轮末用半 epoch 均值 `has_significant_downward_trend`（`:241-250`）决定是否再训 `extension_epochs`。
12. 收尾补最终评估与 `final` checkpoint（`:793-808`），写 `summary.json`（`:819-848`，status ∈ `TRAIN_COMPLETE/TRAIN_COMPLETE_EXTENDED/TRAIN_EARLY_STOPPED/FULL_STOPPED_AT_LIMIT`），smoke 额外写 `smoke_receipt.json`（`:849-850`）。

## 3. `supervisor.sh`（269 行）——编排与 autotune

**【代码 SRC2.3.1｜Cd-SRC2.3.1】（`supervisor.sh`：编排与 autotune）**

- 必需 `TRAIN_JSONL/VALID_JSONL/RUN_ROOT`（`:5-7`）；默认 `MODEL=Qwen/Qwen3-1.7B-Base`（`:9`），三档上下文 6144/4096/4096（`:10-12`），profiles `1:8:on 2:4:on 4:2:on 8:1:on 1:8:off 2:4:off 4:2:off 8:1:off`（`:24`），最小 headroom 768（`:25`），`RESUME=auto`（`:34`）。
- `heartbeat`（`:40-60`）原子写心跳；`run_with_heartbeat`（`:69-95`）后台跑阶段命令并写 `stage.pid/stage.log`。
- 顺序：preflight dry-run（`:102-105`）→ token 审计 6144/4096（必须 `zero_supervised_rows=0` 且 `target_too_long_rows=0`，`:109-115`）→ `try_candidate`（`:124-194`，在 headroom 门槛内选 tokens/s 最高，`:163-186`）→ 回退 fp16/6144 → fp16/4096 → 4bit/4096（`:197-207`）→ 写 `selection/selected.json`（`:217-236`）→ `FULL_DIR` 按精度/上下文隔离（`:240-243`）→ `RESUME=auto` 读 `latest.json`（`:245-255`）→ full 训练（`:257-269`，必须带 smoke 收据）。

## 4. `a10_autotune/a10_autotune.py`（478 行）——单卡 A10 吞吐调优

**【代码 SRC2.4.1｜Cd-SRC2.4.1】（`a10_autotune.py`：单卡 A10 吞吐调优）**

- `prepare_samples`（`:55-129`）：tokenize 一次，取 p10..p90 代表 + **最长样本**（峰值内存压测），写 `benchmark_samples.pt` + `sample_manifest.json`；不安全行 `SystemExit`（`:83-84`）。
- `GpuSampler`（`:132-172`）：后台每 0.25s 采样 `nvidia-smi`（util/mem.used/mem.free/power）。
- `run_one_config`（`:188-318`）：复原初始可训参数（`:209-212`）→ 最长样本 forward+backward 压测（`:227-241`）→ warmup（`:243-252`）→ `grad_accum` micro-step 计时（`:264-278`）→ `headroom = min(torch, nvidia, stress)`（`:283-288`）→ OOM 返回 `status=OOM`（`:311-318`）。
- `worker`（`:321-372`）：`checkpointing∈{on,off}` × `microbatch∈{1,2,4,8}`（`grad_accum=8/mb`）；同族 OOM 后更大 microbatch 标 `SKIPPED_AFTER_OOM`。
- `controller`（`:375-446`）：`fp16`(+`bf16`) × `{sdpa,eager}` 逐族子进程；在 `headroom≥min_headroom_mb` 中选最高 `nonpad_tokens_s`（`:408-421`），写 `autotune_summary.json` + `selected.env`（`:432-445`）。

## 5. `inference_v1/**` 与数据脚本

**【代码 SRC2.5.1｜Cd-SRC2.5.1】（`inference_v1/**` 与数据脚本）**

- `generate_pass1.py`（29 行）：`merged_eval11` 对 val400 逐题贪心生成（`do_sample=False, max_new_tokens=1024`），写 `pass1_generation.jsonl` + 进度（`:18-28`）。
- `parse_pass1.py`（35 行）：抽取首个 JSON，校验 `calls`/forbidden（`:7-19`）；`action`→`action_id`（`:20-24`）；统计 `schema_valid/call_count_match/action_seq_match/exact_target_calls` 写 `pass1_metrics.json`（`:27-35`）。
- `smoke_generate.py`（26 行）/`parse_smoke.py`（26 行）：5 行冒烟生成与解析（先剥 `<answer>/</answer>`，`:8-9`）。
- `pass48.py`（41 行）：48 采样，batch=20、left padding、`temperature=0.6, top_p=0.95, top_k=20`、seed `42+(i//batch)*1000+s`（`:33-37`）；按 `(row_index,sample_index)` 断点续跑（`:13-18,30-32`）。
- `data/train_full_qwen3_1p7b_lora.py`（271 行，早期版）：始终 4bit QLoRA + fp16、`max_seq_len 4096`、线性 warmup、`save/every`、`eval/every`、强制保留 `{20,40}`（`:232-234`）；比终版少了 target 审计、签名门禁、best adapter、自动扩轮。
- `data/prepare_smoke.py`（165 行）：`canonicalize_calls` 去 executor 字段、H→`H_SLOT_i`、空参数白名单（`:37-89`）；确定性抽子集不重叠（`:103-116`）；写 manifest（sha256）。
- `data/build_val400.py`（106 行）：从 combined train 抽 200 + 原 val 200 = 400；重叠用 fallback 替换（`:37-50`）；校验 `remaining=4888, moved=200, original_val=200`（`:57`）。
- `data/audit_contract.py`（72 行）：统计 questions/calls/空参数/executor 字段/裸 H/action_counts（`:25-55`）。

## 6. 张量形状与训练配置

**【例 SRC2.6.1｜E-SRC2.6.1】（张量形状与训练配置）**

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen3-1.7B-Base` |
| 量化 | fp16 LoRA（默认）或 NF4 4bit QLoRA（回退） |
| max_seq_len | 6144（主）/4096（安全）/4096（QLoRA） |
| 序列 | `input_ids [B,T]`，`labels` prompt 段 = -100 |
| 损失 | 因果 LM 交叉熵，仅监督 target token |
| AMP | `autocast(dtype)` + `GradScaler(fp16)` |
| 显存优化 | 梯度检查点 + `enable_input_require_grads` |
| LoRA | r=16, alpha=32, dropout=0.05, 7 投影 |
| 默认 | 5 epoch, lr 1e-4, batch 1 × accum 8, 每 epoch 3 次验证；0.5% 相对改进才刷新 best；patience 3；可自动+2 epoch |

## 7. 与其它版本差异

**【注 SRC2.7.1｜R-SRC2.7.1】（与其它版本差异）**

- 与 **01 V1 app**：app 是 0 长训 + 在线 RTTT（零初始化 LoRA）；本版是离线 QLoRA SFT，产出可合并 adapter。
- 与 **04 nanoproof SFT**：nanoproof 在自研 GPT 上从零训练；本版在 HF Qwen3-1.7B 上做参数高效微调。
- 与 **05 value-head/nanoproof 移植**：那套是 3584→256→64 的价值头；本版训练的是 policy 的 calls 生成能力。

## 8. 想改造应先动哪里

**【注 SRC2.8.1｜R-SRC2.8.1】（想改造应先动哪里）**

- **换基座/上下文**：`train_full_supervised.py:351/358`、`supervisor.sh:9-13`。
- **数据格式/target 语义**：`FullDataset.prompt/target`（`:125-140`）与 `validate_prepared_row`（`:23-51`）；规范化在 `inference_v1/data/prepare_smoke.py:37-89`。
- **QLoRA/精度**：`BitsAndBytesConfig`（`:469-474`）、`prepare_model_for_kbit_training`（`:476-477`）。
- **AMP/显存**：梯度检查点（`:478-481`）、`GradScaler`（`:528`）。
- **早停/扩轮**：`:241-250` 与 `:626-639,767-784`；`--min-relative-improvement/--early-stopping-*/--auto-extend-epochs`。
- **autotune 维度**：`a10_autotune.py:350-353`、`:386-388`、`:408-421`。

## 9. 对应教学篇

**【注 SRC2.9.1｜R-SRC2.9.1】（对应教学篇）**

- SFT/数据格式与 token mask：`../03-SFT/`。
- LoRA 与 QLoRA（4bit/nf4/double quant、`prepare_model_for_kbit_training`）：`../05-LoRA/`。
- AMP/梯度缩放/梯度检查点/优化器：`../01-基础/`。
