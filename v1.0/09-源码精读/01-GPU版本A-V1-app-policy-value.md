# 01 · GPU 版本 A：V1 `app/`（policy + value head + RTTT）

> 源码根：`/mnt/gloway/projects/reap-new-update-model/app/`
> 角色：V1-1 主线实现。**0 长训**：直接加载 REAL-Prover（BF16, cuda:0）+ 一个零初始化 LoRA adapter，线上按需做 RTTT（Real-Time Test-Time Training）一步更新；同时挂一个独立可训练的标量价值头。
> 上游规格：`v1-spec/00-overview.md`、`v1-spec/01-policy-value.md`、`v1-spec/02-mcts-verifier.md`；笔记 `文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`。
> 教学链接：`../05-LoRA/`（LoRA 注入）、`../04-价值头/`（标量 critic）、`../06-RL-RLVR/`（REINFORCE+KL）、`../07-MCTS+V1/`（闭环）。

## 0. 文件树与职责

```
app/
├── policy_server.py        # 【核心】HTTP 服务：policy 采样 + value head + RTTT 训练  737 行
├── value_head.py           # 纯 PyTorch 标量价值头：结构/存取/训练器/目标变换      376 行
├── train_value_head.py     # 离线（预计算 hidden）价值头训练脚本                  250 行
├── train_sft.py            # [DEPRECATED] SFT 骨架，仅存档                         32 行
├── rttt_demo.py            # 真实调用 /v1/chat/completions → /ttt_step 循环演示     57 行
├── mock_policy_server.py   # CPU mock：无 GPU 时的协议端到端替身                    64 行
├── v1_run.py               # BatchSolver 编排器：生成 Lean 文件 → 容器 lake 执行    98 行
├── v1_sink.py              # RolloutSink：node_visited/task_done/rttt_update JSONL  71 行
├── VALUE_HEAD.md / archive.sh / restore.sh / smoke.sh / requirements.lock
└── value_train.example.jsonl
```

数据流总览（在线闭环）：

```
Lean 状态 --(v1_run.py 注入 set_option reap.policy_endpoint/value_endpoint)--> Reap 的 reapMCTS
   │  HTTP
   ├─► POST /v1/chat/completions ─► Engine.generate ─► n 条 tactic 候选(+logprob)
   ├─► POST /value|/value/chat/completions ─► Engine.value ─► score=-V(s)（或 distance 模式）
   └─► 收集 (prompt,target,r,logprob_old) ─► POST /ttt_step ─► Engine.ttt_step
                                          └─ 更新 LoRA(policy) + ValueHead + AdamW
   └─► v1_sink.Sink.node_visited/task_done/rttt_update 落 JSONL
```

## 1. `value_head.py`（先读它，policy_server 依赖它）

### 1.1 `ValueHead` — 标量 critic（L26–63）
- 结构：`Linear(hidden_size, hidden_dim) → SiLU → Linear(hidden_dim, 1) → Tanh`（L43–48），默认 `hidden_dim=256`。
- `forward`（L50–63）：接受 `[B,H]` 或 `[B,T,H]`（后者取最后位置 L53–54）；强制 `hidden_states.float()` 后再过 MLP（L63），保证 backbone 跑 bf16 时头部仍 fp32。
- 输出形状 `[B]`，值域 `(-1,1)`。对应教学 `../04-价值头/`。
- 代码里 `hidden_size` 对 REAL-Prover 为 **3584**（Qwen2.5-7B 系），实际值由 `model_hidden_size()` 推断。

### 1.2 辅助函数
- `model_hidden_size(model)`（L66–81）：依次找 `hidden_size/n_embd/d_model/n_embed`，再退回 `embedding_dim`。
- `last_token_hidden(output, attention_mask)`（L84–116）：从 `last_hidden_state` 或 `hidden_states[-1]` 取 `[B,T,H]`；用 attention_mask 求**最后一个非 pad 位置**（L109–116），兼容左右 padding。
- `finite_scalar`（L119–132）/ `clamp_target`（L135–138，夹到 `[-1,1]`）。
- `discounted_returns(rewards, gamma)`（L141–152）：从后往前 `running = r_i + γ·running`，并 clip 到 `[-1,1]`。
- `proof_depth_to_target(depth, max_depth=64)`（L155–170）：`-min(depth,max_depth)/max_depth`。**剩余深度 0 = 最优 = 0.0**，深度越大越负。这是把 nanoproof replay 的“负剩余深度”转成有界 target 的桥。

### 1.3 持久化（原子写）
- `VALUE_HEAD_SCHEMA = "reap.value-head.v1"`（L23）。
- `save_value_head(...)`（L185–237）：payload 含 `schema_version/hidden_size/hidden_dim/head(CPU state)/optimizer_steps/examples_seen/metadata`，可选 `optimizer`；写临时文件后 `os.replace` 原子替换（L220–229）。
- `load_value_head(...)`（L240–295）：校验 schema 与 `hidden_size/hidden_dim`，支持**裸 state_dict 的 legacy 迁移**（L260–262，返回 `legacy=True`）；可选加载 optimizer 并把状态搬到 head 设备（L282–285）。
- checkpoint 只存小头，**不动 policy backbone**，避免频繁 RTTT 覆盖大模型（L196–199 注释）。

### 1.4 `ValueHeadTrainer`（L298–376）
- 只吃**预计算的 hidden**（`[B, hidden_size]`），做 `mse` 或 `huber`（`smooth_l1_loss`，L337–340）。
- 全过程有限性校验：loss（L341）、grad（L345）、grad_norm（L350）、参数（L353）；`clip_grad_norm_` 保护（L347）。
- `update` 返回 loss/prediction_mean/target_mean/grad_norm/steps（L357–364）；`checkpoint` 落盘（L366–375）。

## 2. `train_value_head.py` — 离线价值头训练

- 数据抽取 `iter_examples(path, max_depth, allow_score, include_binary_outcomes)`（L91–136）：
  - 若记录含**并行数组** `states[]`+`rewards[]`，用 `discounted_returns` 展开成多个 `(state,target)`（L112–120）。
  - 否则走 `_prompt()`（L52–57，`prompt/state/state_pp`）取 prompt，`_explicit_target()`（L60–88）取 target。优先级：`target_kind∈{proof_depth,depth}` → `{negative_proof_depth,nanoproof}` → `value_target/return/td_target/target_value` → 顶层 `proof_depth` → 嵌套 `value.target`。
  - `auto` 模式下 `raw < -1.0` 判定为“负深度”并转 `proof_depth_to_target`（L77–78）。
  - **默认绝不使用 `value.score`**（那是旧/随机头产生的 -V），仅 `--allow-score` 时启用（L126–132）；`--include-binary-outcomes` 用 `was_solved` 造 ±1（L133–134）。
- `main`（L160–246）：冻结 backbone（`requires_grad_(False)`，L193–194），tokenizer 右 padding（L188），`--max-sequence-tokens` 超长直接报错而非静默截断（L213–218）；每 batch `output_hidden_states=True` → `last_token_hidden` → `trainer.update`（L219–222）；最后 `trainer.checkpoint(output, metadata=...)`（L236–244）。
- 张量形状：`input_ids [B,T]` → `hidden_states[-1] [B,T,3584]` → `last_token_hidden [B,3584]` → head `[B]`；target `[B]` ∈ `[-1,1]`。

## 3. `policy_server.py` — 服务核心

### 3.1 常量与设计
- `TARGET_MODULES`（L45）：LoRA 打在注意力和 MLP 全部线性层 `q/k/v/o/gate/up/down_proj`。
- `BETA_KL = 0.05`（L46）：KL 防忘系数（v1-spec 02）。
- 超参默认（L47–52）：value_lr 3e-4、policy_lr 1e-4、value_coefficient 0.5、gamma 0.99、max_grad_norm 1.0、max_distance 64；`MAX_TTT_ITEMS = 16`（L53）。

### 3.2 `class Engine`（L55）
`__init__`（L64–152）：
1. 参数校验（L81–92）：gamma∈[0,1]、lr>0、coeff≥0、max_grad_norm>0、`value_output_mode∈{scalar,distance}`、`max_distance` 正整数。
2. `self._update_lock = threading.RLock()`（L105）——**并发保护**，所有会改参数的入口（`value/raw_value/train_value/ttt_step`）都进出这把锁。
3. tokenizer（L106–108，pad=eos），backbone `AutoModelForCausalLM(..., torch_dtype=torch.bfloat16, device_map=self.device)`（L109–111）。
4. **LoRA 注入**（L112–116）：`r=16, lora_alpha=32, lora_dropout=0.02, bias="none", init_lora_weights=True`。注释（L115）说明：零 B 初始化 ⇒ adapter 增量 ΔW=0 ⇒ **初始等价 base**（这就是“0 长训”）。教学见 `../05-LoRA/`。
5. `self.hidden_size = model_hidden_size(self.model)`（L118），`ValueHead(hidden_size, hidden_dim).to(device, dtype=torch.float32)`（L120–122）。
6. 两个优化器：
   - `param_groups` = [LoRA 参数 lr=policy_lr, vhead 参数 lr=value_lr] → `self.opt = AdamW(param_groups)`（L124–132），RTTT 联合更新用。
   - `self.value_opt = AdamW(vhead.parameters(), lr=value_lr)`（L131），**仅头**，用于可移植 value_head.pt 的步数/动量同步。
7. 可选加载 value head（L133–151）：校验 checkpoint 的 `value_output_mode` 与 server 一致，并恢复 `optimizer_steps/examples_seen`；路径不存在则 `FileNotFoundError`。

其它方法：
- `_encode`（L154–161）：`tok(prompt, return_tensors="pt")`。
- `_item_prompt`（L163–176）：从 item 的 `prompt/state/state_pp` 取非空字符串。
- `_hidden_from_encoded`（L178–199）：`output_hidden_states=True, use_cache=False`，需要梯度时直接前向，否则 `no_grad` 并保留/还原训练态。
- `_raw_value`（L201–209）：取 last hidden → `vhead` → 有限性校验 → python float。
- `generate`（L211–232）：校验 n∈[1,64]；`do_sample=True, temperature, num_return_sequences=n, max_new_tokens`；只取生成段并 `.split("\n")[0]`（L229）；注意返回的 `logprob_avg` 目前恒为 `0.0`（L230，采样未开 `output_scores`）。
- `value`（L234–245）/`_score_from_raw`（L247–258）：
  - `scalar` 模式返回 `-V`（项目原契约，L258）；
  - `distance` 模式返回 `1 + (max_distance-1)(1-V)/2 ∈ [1,max_distance]`（L254–257），给 nanoproof / verified-collector 用，客户端再取负恰好一次。
  - raw 先夹到 `[-1,1]`（L251）。
- `raw_value`（L260–264）：返回 `V∈[-1,1]`。
- `_target_for_item`（L266–311）：RTTT item → 有界 target。优先 `value_target/return/[td_target/target_value]`；`proof_depth` 走 `proof_depth_to_target`；`reward/r` + `next_value`（或 `next_prompt` 现算）做一步 TD：`r + γ·V(s')`（L308–311）。
- `_value_optimizer`（L313–320）、`_sync_value_optimizer_state`（L322–335，把 `self.opt` 中 vhead 的动量镜像到 `value_opt`）、`_save_value_head`（L337–354，写 metadata 含 hidden_size/coeff/gamma/mode/max_distance）。

### 3.3 训练循环

**`train_value(items, epochs)`（L356–414）— 纯头更新：**
```
对每个 item: target=_target_for_item; hidden=_hidden_from_encoded(no_grad).detach()
features=[B,3584]; target_tensor=[B]
pred = vhead(features)            # [B]
loss = MSE(pred, target)          # 有界 [-1,1] 回归
(coeff * loss).backward(); clip_grad_norm_; value_opt.step()
```
返回 loss/examples/steps/optimizer_steps + value_prediction_mean/value_target_mean/value_grad_norm（L398–414）。

**`ttt_step(items)`（L416–507）— 联合 policy+value 一步：**
逐 item（batch ≤16）：
- 拼接 `input_ids = cat(prompt_ids, target_ids)`（L440），`labels` 只对 target 段计 loss（prompt 段置 -100，L444–445），注意力 mask 同样拼接（L442–443）。
- `out = model(...)`；`nll = out.loss`；`logp = -nll`（L452–453）。
- `logp_old` from item（默认 -12.0，L456）；`reward = r/reward`（L459）。
- **policy 损失** `policy_loss = -reward * (logp - logp_old)`（L462，REINFORCE 重要性式）。
- **KL 项** `kl = BETA_KL * (logp - logp_old)^2`（L463）。
- **value 损失**：取 prompt 段最后位置 `out.hidden_states[-1][:, prompt_position, :].float()`（L468–469），`prediction=vhead(value_hidden)`，`MSE(prediction, value_target)`（L470–472）。
- 聚合：`total = (policy_total + kl_total + value_coefficient*value_total) / len(batch)`（L476–477）。
- `total.backward()` → `clip_grad_norm_(params, max_grad_norm)` → `self.opt.step()`（L480–487）→ `_sync_value_optimizer_state()`（L488）→ 计数/落盘。
- 返回 loss/policy_loss/kl/value_loss/value_updates/grad_norm/steps/optimizer_steps（L495–506）。

### 3.4 快照 / 回滚
- `snapshot(name)`（L509–527）：schema `reap.policy-server.snapshot.v2`，存 `model + value_head + optimizer + value_optimizer + steps` 到 `output_dir/adapter_<name>.pt`。
- `restore(name)`（L529–562）：校验 schema/hidden_size/output_mode/max_distance 后恢复；兼容旧式裸 LoRA state_dict（L554–557）。

### 3.5 HTTP 层 `class H`（L566）
- `_prompt`（L575–596）：`prompt/state` 或 OpenAI `messages` 拼接。
- `_value_chat`（L598–612）：**OpenAI chat 包装**，`assistant.content = json({"score": float})`，Lean `OpenAIClient` 解析后取负。
- `_policy_chat`（L614–633）：同时给旧 `text` 字段与 OpenAI `message` 字段。
- `do_GET /health`（L636–653）：返回 device/hidden_size/value_head 元信息（kind=`linear-silu-linear-tanh`、loaded、optimizer_steps…）、`adapter_loaded="zero-init lora (0 long-train)"`。
- `do_POST` 路由（L657–685），**重点端点**：
  - `…/value/chat/completions` 或 `…/value/v1/chat/completions` → `_value_chat(ENGINE.value(prompt))`（L662–665）。
  - `…/chat/completions` → `ENGINE.generate`（L666–671）。
  - `…/value` → `{"score": ENGINE.value(prompt)}`（L672–673）。
  - `…/value/train`（L674–675）。
  - `…/ttt_step` 或 `…/ttt/step` → `ENGINE.ttt_step(items)`（L676–677）。
  - `…/adapter/snapshot|/snapshot`、`…/adapter/restore|/restore`（L678–681）。
  - 任何异常 → 500 `{"error": str(e)}`（L684–685）。
- `main`（L687–734）：参数含 `--base --port(8760) --lora-r --device --value-head --value-hidden-dim --value-lr --policy-lr --value-coefficient --gamma --max-grad-norm --value-output-mode --max-distance --output-dir`；若未显式给 `--value-head` 但 `output_dir/value_head.pt` 存在则自动用（L708–711）；即使新头也在首次更新后落盘（L730–731）；`ThreadingHTTPServer(("0.0.0.0", port), H)`（L734）。

## 4. `v1_run.py` — BatchSolver 编排器

- `make_lean(task, policy, value, ps, import_extra)`（L21–45）：拼出
  ```
  import Reap / import Reap.Tactic.Syntax / 任务 imports
  set_option reap.policy_endpoint "http://127.0.0.1:8760"
  set_option reap.value_endpoint  "http://127.0.0.1:8760/value"
  set_option reap.ps_endpoint     "http://127.0.0.1:8760"
  theorem <name> : <stmt> := by
    reapMCTS
  #eval IO.println "%%TASK_<id>_DONE%%"
  ```
  `reapMCTS` 是 Lean 侧的 MCTS 入口，端点在运行时被注入（L27–30）。
- `run_one`（L48–72）：`podman run --rm --network host -v <out>:/batch:ro <image> bash -lc "cp … && cd /workspace/reap && lake env lean /batch/<id>.lean | tail -20"`；判定成功 = 输出含 `%%TASK_<id>_DONE%%` 且无 error（L62–64）；`subprocess.TimeoutExpired`（per_task_timeout，默认 300s）记 TIMEOUT。
- `main`（L74–96）：读取 `--batch` JSONL；checkpoint 目录 `out/state/<id>.done`；`--continue` 时跳过已 done 的题（L79–82）；统计 solved/failed。

## 5. `v1_sink.py` — RolloutSink JSONL 样本面

- `ALLOWED_KINDS = {node_visited, task_done, rttt_update}`（L7）；`VERDICT_CLASSES`（L8–9）。
- `Sink.node_visited`（L16–27）：记录树节点 visited，含 `tree_hash/node_idx/parent_idx/depth/state_pp/state_key/goal_count/tactic/verdict.kernel_check/policy.logprob_avg/value.score/was_solved`。
- `Sink.task_done`（L29–33）/`Sink.rttt_update`（L35–37）。
- `state_key(state_pp)=sha256:<hex>`（L48–49）。
- CLI `--validate <file>`（L51–68）统计 valid/broken。

## 6. `rttt_demo.py` / `mock_policy_server.py` / `train_sft.py`

- `rttt_demo.py`：先 `/health`（L31），循环 `POST /v1/chat/completions`（n=2）取首个候选，构造 item（`target/r/value_target/done/logprob_old`，L40–47），攒够 `--k`（默认 8）后 `POST /ttt_step`，指标追加到 `rttt_metrics.jsonl`（L48–52）。演示用 `reward = 1.0 if i%3==0 else -0.5`（L39），生产应换成 Lean verifier 的 discounted return。
- `mock_policy_server.py`：CPU 替身，`/v1/chat/completions` 随机返回 tactic + `logprob_avg∈[-18,-2]`（L42–54）；`/value` 随机 `[0,1]`（L55–56）；`/value/chat/completions` 用 **sha256 前 4 字节确定 score**（L34–35，便于协议测试）。
- `train_sft.py`：**DEPRECATED（2026-08-26）**，仅有 argparse + 打印行数（L10–29），注释指向 v1-spec/01 的显存预算；正式 SFT 实现见 02 篇（容器 QLoRA）。

## 7. 张量/数据流小结

| 环节 | 形状 | 说明 |
|---|---|---|
| prompt token | `[B,T]`（在线多为 B=1） | tokenizer 输出 |
| 拼接训练序列 | `[B, T_p+T_t]` | labels prompt 段 -100 |
| backbone hidden | `[B, T, 3584]` | `output_hidden_states` |
| last hidden | `[B, 3584]` | `last_token_hidden` |
| value 预测 | `[B]` ∈ (-1,1) | `ValueHead` |
| 生成候选 | n × text | `generate` |
| RTTT item | dict | prompt/target/r/logprob_old/value_target |

## 8. 与其它版本差异

- 与 **03 gpu_runtime**：这里是“单进程 HTTP + PEFT LoRA + 标量头”的极简在线实现；gpu_runtime 是“多 backend、release head、64-bin 分类头、learner/actor 分离”的生产化实现。
- 与 **04 nanoproof**：nanoproof 是**从零预训练**到 RL 的完整研究栈（自定义 tokenizer/Muon/FlashAttn）；本版直接复用 HF REAL-Prover + LoRA。
- 与 **05 value-head/nanoproof 移植**：本版 value 头是 `3584→256→1`（tanh），05 篇是 `3584→256→64`（64-bin 分类）并带 real7 特征管线。

## 9. 想改造应先动哪里

- **换基座/换 hidden 维**：改 `policy_server.py` L109–116（`--base`、LoRA target_modules）与 `--value-hidden-dim`；`model_hidden_size` 自动适配。
- **改 RL 目标**：`ttt_step` L462（policy）、L463（KL）、L466–475（value）；超参 `BETA_KL` L46、`value_coefficient`。
- **接入真实 Lean 奖励**：把 `rttt_demo` 的假 reward 换成 verifier；或让 `_target_for_item`（L266–311）吃 `proof_depth`/TD。
- **闭环协议**：直接改 `v1_run.py` 的 `set_option reap.*_endpoint`（L27–30）。
- **样本面 schema**：`v1_sink.py` L7–9 的 kind/verdict 集合。
- **并发/吞吐**：`ThreadingHTTPServer` + 全局 `RLock`（L105）意味着训练串行；要并发需拆 Engine 或多进程。

## 10. 对应关系

- 教学篇：`../05-LoRA/`（LoRA/`init_lora_weights`）、`../04-价值头/`（`3584→256→1`、MSE/校准）、`../06-RL-RLVR/`（REINFORCE+KL 形式）、`../07-MCTS+V1/`（`reapMCTS` + endpoint 注入）。
- 数学文档：`文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`。
- 规格：`v1-spec/00-overview.md`、`01-policy-value.md`、`02-mcts-verifier.md`。
