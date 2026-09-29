# 04 · GPU 版本 D：nanoproof 全流程（教学参照）

> 源码根：`a@my-new-linux:/mnt/gloway/projects/reap-new-update-model/nanoproof/`
> 角色：**本系列最重要的教学参照**——唯一完整实现「自研 tokenizer + 自研 Transformer + MuonAdamW + 预训练 → midtrain → SFT → MCTS+RL → 评测」的全栈。其它版本（01 HF+LoRA 在线、02 Qwen QLoRA、03 gpu_runtime、05 value-head 移植）都由此派生或与之对应。
> 教学篇：`../01-基础/`、`../02-预训练/`、`../03-SFT/`、`../04-价值头/`、`../06-RL-RLVR/`、`../07-MCTS+V1/`。
> **【文档｜DOC-SRC4】**（doccode = `SRC4`）｜编号与 Tag 规范见《00-风格与编号规范》。

## 0. 文件树与职责

**【注 SRC4.0.1｜R-SRC4.0.1】（文件树与职责）**

```
nanoproof/nanoproof/
├── model.py        516  自研 GPT：RoPE/GQA/relu²/滑窗/smear/backout/per-layer lambdas
├── optim.py        503  MuonAdamW / DistMuonAdamW（Polar Express 正交化 + NorMuon）
├── common.py      1073  全局配置 GLOBAL_CONFIG、分布式初始化、日志/指标、LR schedule、时间线、证明构造
├── tokenizer.py    268  GPT-2 BPE + Lean/数学特殊 token（含 64 个值桶 <|bin_i|>）
├── engine.py       531  KV cache（FA3 布局）+ 批量生成引擎
├── flash_attention.py 287  FA3 封装与 SDPA 回退
├── fp8.py          160  FP8 训练转换
├── checkpoints.py  306  保存/加载 checkpoint 与模型
├── loss_eval.py     75  bits-per-byte（bpb）评估
├── cli.py         2480  日志/错误上报/WebMonitor 监控面板
├── pretrain.py     712  预训练（scaling law 自动定 batch/LR/wd）
├── midtrain.py     399  中训（Lean GitHub raw 语料续训，更简）
├── sft.py          537  监督微调（policy + value 混合加权）
├── rl.py          1093  RL 主循环（MCTS 采集 + replay/负例训练）
├── search.py       896  MCTS（PUCT/AND-OR/backprop/value target）
├── experience_collection.py 923  Matchmaker、Replay/NegativeBuffer、经验抽取
├── prover.py       903  Prover/ProverWorker（actor 池、collect/evaluate）
├── inference.py   1121  TacticModel/Blocking/Remote/Balancer/推理服务
└── data/                各阶段数据管线（nemotron/leangithubraw/leantree/bench）
scripts/run_train.py     端到端流水线（pretrain→midtrain→sft→prover_eval→rl）
```

## 1. `model.py`（516 行）——自研 Transformer

**【代码 SRC4.1.1｜Cd-SRC4.1.1】（`model.py`：自研 Transformer）**

- `NetworkConfig`（`:32`）：`sequence_len/vocab_size/n_layer/n_head/n_kv_head/n_embd/window_pattern`。
- `norm`（`:45`）= `F.rms_norm`（无可学参数）；`Linear`（`:49`）forward 时把权重 cast 到输入 dtype（**显式 dtype 管理替代 autocast**）。
- `apply_rotary_emb`（`:58`）：把最后一维对半做旋转。
- `CausalSelfAttention`（`:67`）：`c_q/c_k/c_v/c_proj`；GQA（`n_kv_head ≤ n_head`）；RoPE 后 **QK norm + ×1.2 锐化**（`:93-96`）；训练用 `flash_attn_func(causal=True, window_size=...)`（`:101`），推理用 `flash_attn_with_kvcache`（`:107`）。
- `MLP`（`:127`）：`c_fc → relu(x)² → c_proj`（relu² 激活）。
- `Block`（`:140`）：pre-norm 残差 `x + attn(norm x)`、`x + mlp(norm x)`。
- `Transformer`（`:152`）：
  - `wte`（词表 pad 到 64 的倍数）+ 每层 Block + `lm_head`（**未 tie 权重**）；可学标量 `resid_lambdas`（初始 ~1.15 递减）、`x0_lambdas`（~0.20 递减）。
  - `smear_gate`/`smear_lambda`（`:183-184`）：把上一 token 嵌入按 sigmoid 门控混入（`:438-457`）。
  - `backout_lambda`（`:186`）：减去中间层残差（`:462-471`）。
  - `init_weights`（`:194-225`）：embedding std=0.8、lm_head std=0.001、注意力/MLP 矩阵 uniform ±s（`s=√3·n_embd^-0.5`）、投影置零、RoPE 预算。
  - `_compute_window_sizes`（`:241`）：按 `window_pattern`（L/S）逐层决定滑窗，短窗 `ceil(seq/4/128)*128`；**最后一层恒为全程**。
  - `estimate_flops`/`num_scaling_params`（`:263`/`:289`）：scaling law 用。
  - `setup_optimizer`（`:313`）：5 个 AdamW 组（lm_head/embedding/resid/x0/smear）+ Muon 矩阵组（按 shape 分组）；AdamW LR 按 `(d_model/768)^-0.5` 缩放；DDP 用 `DistMuonAdamW`。
  - `forward`（`:405`）：RoPE 位置（支持 kv_cache per-row）；embedding → norm → smear；逐层 `x = resid[i]*x + x0[i]*x0 + block(...)`，记录中层 `x_backout`，末尾 `x - backout*x_backout`；norm → lm_head → **softcap 15**（`15*tanh(logits/15)`）；有 targets 时交叉熵（`ignore_index=-1`，支持 reduction）。

## 2. `optim.py`（503 行）——MuonAdamW

**【代码 SRC4.2.1｜Cd-SRC4.2.1】（`optim.py`：MuonAdamW）**

- `adamw_step_fused`（`:22`，`@torch.compile(fullgraph)`）：解耦权重衰减 → 一/二阶矩 → bias correction → 更新。
- `polar_express_coeffs`（`:74`）；`muon_step_fused`（`:83`）：Nesterov 动量 → **Polar Express 正交化**（`:105-119`，tall/wide 两分支）→ **NorMuon 方差归一**（`:121-134`）→ **cautious weight decay**（`:136-140`）。
- `MuonAdamW`（`:147`）：单卡；按 `kind` 分 adamw/muon；muon 组把同 shape 参数 stack 后一次更新（`:205-248`）。
- `DistMuonAdamW`（`:265`）：reduce_scatter → 本地计算 → all_gather 三阶段异步通信。
- `optimizer_to_cpu`/`optimizer_to_gpu`（`:466`/`:491`）：训练与推理之间搬运优化器状态省显存。

## 3. `common.py`（1073 行）

**【代码 SRC4.3.1｜Cd-SRC4.3.1】（`common.py`：配置/分布式/日志/证明工具）**

- `GLOBAL_CONFIG`（`:71`）：含 `num_value_bins`、`max_seq_len`、`tactic_max_len` 等。
- `get_lr_multiplier`（`:86`）：warmup/warmdown 线性调度。
- 分布式：`is_ddp_initialized`（`:288`）、`get_dist_info`（`:297`）、`compute_init/cleanup`（`:333`/`:377`）、`active_barrier`（`:908`）。
- 日志/监控：`create_metrics_logger`（`:583`，wandb）、`MetricsLogger`（`:484`）、`TimelineRecorder`（`:890`）。
- 证明工具：`linearize_proof`（`:948`）、`construct_proof_source`（`:1016`）、`theorem_to_example`（`:1046`）。

## 4. `tokenizer.py`（268 行）

**【代码 SRC4.4.1｜Cd-SRC4.4.1】（`tokenizer.py`：特殊 token 与 64 值桶）**

- `SPECIAL_TOKENS`（`:14-139`）：`<|pad|>`、`<|tactic|>`、`<|value|>`、**64 个 `<|bin_01|>…<|bin_64|>`**（`GLOBAL_CONFIG.num_value_bins`）+ 大量 Mathlib 高频数学符号。
- `value_to_token_ids`（`:233`）/`token_ids_to_value`（`:240`）：价值 1..64 ↔ 单个 bin token。
- `HuggingFaceTokenizer`（`:142`）：encode/decode 包装；`get_bos/eos_token_id` 均为 `<|endoftext|>`。

## 5. `engine.py`（531 行）——推理引擎

**【代码 SRC4.5.1｜Cd-SRC4.5.1】（`engine.py`：KV cache 推理引擎）**

- `KVCache`（`:23`）：`(n_layers,B,T,H,D)` 预分配（FA3 布局）、`cache_seqlens` per-row、`prev_embedding`（smear）。
- 采样：`sample_next_token`（`:85`）、`sample_next_token_limited_replacement`（`:104`，Gumbel-Top-k + 每 token 出现次数上限，用于首 token 去重）。
- `Engine.generate`（`:177`）：变长批量单次 prefill（右 pad BOS）→ K/V 复制 `num_samples` 份到 decode cache → 逐 token 解码；支持 min/max tokens、logits、token_logprobs；OOM dump；finally 显式释放 KV cache。
- `generate_batch`（`:424`）：非流式包装。

## 6. 训练四阶段

### 6.1 `pretrain.py`（712 行）

**【代码 SRC4.6.1｜Cd-SRC4.6.1】（`pretrain.py`：scaling law 预训练）**
- `build_model_meta`（`:242`）：meta device 建结构后 `to_empty+init_weights`。
- FP8 训练（`:286-309`）与评估时 `disable_fp8`（`:313`）。
- **Scaling law / muP**（`:355-432`）：`num_scaling_params` → `target_tokens = ratio*scaling_params`；参考 d12 模型的 `B_REF=2^19`；按 Power-Lines `B ∝ D^0.383` 自动定 batch；按 batch/data 缩放 LR 与 weight decay。
- 优化器（`:436`）；fp16 用 `GradScaler`（`:449`）。
- 训练循环（`:542-702`）：评估 bpb、梯度累积、Muon 动量调度 `get_muon_momentum`（`:496`，0.85→0.97→warmdown 0.90）、cosine weight decay（`:510`）、日志/GC。

### 6.2 `midtrain.py`（399 行）

**【代码 SRC4.6.2｜Cd-SRC4.6.2】（`midtrain.py`：Lean 语料续训）**
同构更简：数据用 `leangithubraw_batches`（`:41`），`total_batch_size=491520`（`:95`），`init_lr_frac=0.8`，只存最终 checkpoint（`:299`）。

### 6.3 `sft.py`（537 行）

**【代码 SRC4.6.3｜Cd-SRC4.6.3】（`sft.py`：policy + value 监督微调）**
- 数据：`leantree_transitions(split="train")`（`:251`），可选 `leantree.augmentations`（ShuffleGoalsAndHypotheses / RandomRename，`:243`）。
- 模型：`load_model(args.model_path)`（`:200`）或从零建（`:201-221`）。
- 优化器 + `init_lr_frac=0.8`（`:262-273`）。
- **训练循环**（`:285-518`）：`loss_reduction="none"` 得 per-token loss（`:455-458`）；含 `<|value|>` 的样本乘 `--value-weight`（默认 0.01）（`:460-471`）；按 `token_mask` 求加权平均。
- 评估（`:292-346`）：val loss + `eval_tactic_accuracy`/`eval_critic_errors`，打印 64 类 confusion matrix（`:338`）。
- 采样示例验证（`:349-407`）与 checkpoint（`:409-440`、`:520-532`）。

### 6.4 `rl.py`（1093 行）★

**【代码 SRC4.6.4｜Cd-SRC4.6.4】（`rl.py`：MCTS 采集 + replay 训练）**
- 超参（`:84-276`）：`--num-sampled-tactics 6`、`--max-gen-tokens 24`、`--collect-transitions 100`、`--replay-buffer-window-size 250000`、`--fraction-sft 0.1`、`--negative-buffer-window-size 250000`、`--negative-fraction 0`、`--unlikelihood-weight 0.5`、`--value-weight 0.01`、`--num-updates-per-step 2`、`--eval-every/--save-every`。
- 恢复：`_resolve_resume`（`:308`）从 log 目录推 model 目录并校验 optim shard；`_log_resume_arg_diff`（`:369`）对比历史 args。
- 组件：`TacticModel`/`BlockingTacticModel`（`:441-453`）、`setup_distributed_inference`（`:505`）、`ProverWorker`（`:507`）、`ReplayBuffer/NegativeBuffer/Matchmaker`（`:483-502`）。
- **`train_generator`**（`:619-651`）：按 `fraction_sft` 从 Mathlib 抽、按 `negative_fraction` 抽负例、否则从 replay buffer 抽 `(state,tactic,value_target)`；`proof_depth = -value_target`。
- **损失**（`:978-1003`）：正样本按 token 均值并给 value 样本乘 `value_weight`；负样本 `unlikelihood = -log(1-exp(-CE))` 逐序列均值；`loss = positive + unlikelihood_weight*negative`。
- **循环**（`:722-1093`）：eval（`:746-876`）→ collect（`:878-913`，rank0 采集，`extend_and_sync` 广播）→ train（`:931-1033`，`optimizer_to_gpu/cpu`、暂停/恢复推理、`active_barrier`）→ 保存 `step_*/`。

## 7. `search.py`（896 行）——MCTS

**【代码 SRC4.7.1｜Cd-SRC4.7.1】（`search.py`：MCTS）**

- `SearchConfig`（`:47`）：`pb_c_base=200, pb_c_init=0.001, value_discount=0.98, prior_temperature=200, no_legal_actions_value=-5, c_and=64, unvisited_value_penalty=16, ps_c=0.1, ps_alpha=0.6, verify_timeout=5000`。
- `Node`（`:85`）：parent/action/prior/state/reward/to_play/is_solved/visit_count/evaluations/value_sum/children/value_target/id；`calculate_solved`（`:134`，OR=any、AND=all）；serialize/deserialize（`:183`/`:212`）。
- `run_mcts`（`:475`）：选择（`select_child`+`progressive_sample`）→ `model.sample_tactic`（`:535`，`value = -value` 转 MCTS 尺度）→ `expand_node`（立即在 Lean 试算）→ `backpropagate`（`:567`）。
- `ucb_score`（`:616`）：`pb_c` 随父访问次数、AND 乘 `c_and`；未访问子节点取 `parent.value - unvisited_value_penalty`；`value_score = γ^(-1-value)`，AND 反转并惩罚已解子节点（`-1e9`）。
- `expand_node`（`:658`）：去重 + `try_apply_tactic`，多子目标时建 AND 节点并补 focus 子节点（`:739-751`）。
- `backpropagate`（`:861`）/`backprop_value_towards_min`（`:890`）：AND 取子节点最小（最难分支）。
- `close_leaves_with_grind`（`:758`）：`--disable-solvers` 时用 grind 闭合叶节点。

## 8. `experience_collection.py`（923 行）

**【代码 SRC4.8.1｜Cd-SRC4.8.1】（`experience_collection.py`：经验抽取）**

- **`compute_value_target`（`:771`）**：求解树上 OR 取 `-1+max(children)`、AND 取 `min(children)`、terminal=0 → 每个节点的价值回归目标（负剩余证明深度）。
- `extract_transitions`（`:867`）/`_extract_transitions_recursive`（`:889`）：沿 solved OR 路径抽 `(context, tactic, value_target)`；`filter_grind` 跳过 solver tactic。
- `prune_redundant_nodes`（`:796`）：删除冗余 OR 中间节点（缩短证明）。
- `ReplayBuffer`（`:410`，FIFO + `extend_and_sync` 广播）、`NegativeBuffer`（`:457`，失败 tactic 供 unlikelihood）、`CollectedExperience`（`:523`，record_attempt/tactic/train_samples + save JSONL）、`CollectExperienceHolder`（`:719`，rotate 防跨步丢记录）、`Matchmaker`（`:289`，按历史分配搜索预算）。

## 9. `prover.py` / `inference.py`

**【代码 SRC4.9.1｜Cd-SRC4.9.1】（`prover.py` / `inference.py`）**

- `Prover`（`prover.py:109`）：与 Lean 服务交互、`LeanPoolTimeoutError`/`ProofInitError`。
- `ProverWorker`（`:253`）：actor 池并行跑搜索、`collect`/`evaluate`、pause/resume。
- `TacticModel`（`inference.py:61`）：本地模型采样 tactic+value；`BlockingTacticModel`（`:292`）加批预算/超时；`RemoteTacticModel`（`:697`）；`InferenceBalancer`（`:844`）；`setup_distributed_inference`（`:1035`）。

## 10. 流水线与数据

**【注 SRC4.10.1｜R-SRC4.10.1】（流水线与数据）**

- `scripts/run_train.py`（231 行）：`ALL_STAGES=[pretrain,midtrain,sft,rl]`（`:40`）；逐阶段子进程 + 解析 `Log directory`/`Model directory`（`:76-81`）；SFT 后自动跑 `prover_eval.py --datasets minif2f`（`:214-223`）；`find_latest_checkpoint`（`:43`）链式传模型。
- `data/pretrain/nemotron_dataloader.py`（`nemotron_batches`）、`data/midtrain/leangithubraw.py`、`data/sft/leantree.py`（`leantree_transitions`）+ `leantree_dataloader.py`（`sft_data_generator`/`rl_data_generator`）、`data/bench/{minif2f,proofnet}.py`。

## 11. 张量形状与训练配置

**【例 SRC4.11.1｜E-SRC4.11.1】（张量形状与训练配置）**

| 阶段 | 数据 | 目标 | 关键点 |
|---|---|---|---|
| 预训练 | nemotron token 流 `[B,T]` | 下一 token CE | scaling law 自动 batch/LR/wd |
| midtrain | Lean GitHub raw | 下一 token CE | 更短、`init_lr_frac=0.8` |
| SFT | leantree `(state,tactic)` | CE（value 样本 ×`value_weight`） | token mask；critic 64 类 |
| RL | MCTS 树抽取的 `(state,tactic,value_target)` | 正 CE + 负 unlikelihood | replay/negative buffer；MuonAdamW |
| 推理 | KV cache | — | FA3 / SDPA |

## 12. 与其它版本差异

**【注 SRC4.12.1｜R-SRC4.12.1】（与其它版本差异）**

- 与 **01/03**：nanoproof 自研模型与优化器；01/03 复用 HF REAL-Prover + PEFT（01 在线 RTTT，03 生产 gpu_runtime）。
- 与 **02**：02 是 HF Qwen3-1.7B QLoRA；nanoproof 的 SFT 是自研栈的一部分。
- 与 **05**：05 把价值头移植到 REAL-Prover（`3584→256→64`）。

## 13. 想改造应先动哪里

**【注 SRC4.13.1｜R-SRC4.13.1】（想改造应先动哪里）**

- **模型结构**：`model.py`（`CausalSelfAttention:67`、`MLP:127`、`_compute_window_sizes:241`、`forward:405`）。
- **优化器/超参**：`optim.py`（`muon_step_fused:83`）与 `model.setup_optimizer:313`。
- **分词/值域**：`tokenizer.py`（`SPECIAL_TOKENS:14`、`num_value_bins`）。
- **预训练预算**：`pretrain.py:355-432`。
- **SFT 损失权重**：`sft.py:460-471`。
- **RL 目标/数据**：`rl.py:978-1003`、`experience_collection.py:771`。
- **搜索**：`search.py`（`ucb_score:616`、`expand_node:658`、`backpropagate:861`）。

## 14. 对应教学篇

**【注 SRC4.14.1｜R-SRC4.14.1】（对应教学篇）**

- `../01-基础/`（RoPE/GQA/relu²/RMSNorm/初始化/Optimizer/LR schedule）、`../02-预训练/`（pretrain/midtrain、bpb、scaling law）、`../03-SFT/`（token mask、value 加权、critic 混淆矩阵）、`../04-价值头/`（64-bin value target）、`../06-RL-RLVR/`（unlikelihood、RL 循环）、`../07-MCTS+V1/`（PUCT/AND-OR/backprop）。
- 数学与流程：`文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`。
