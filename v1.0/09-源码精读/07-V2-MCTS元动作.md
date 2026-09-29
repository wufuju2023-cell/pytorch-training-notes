# 07 · V2：MCTS 元动作（塔 / Eff 通道）

> 源码位置（实测勘误）：任务给出的 `/mnt/gloway/projects/reap-mcts-lean-v2-code-1/v2/*` 实际位于
> `a@my-new-linux:/mnt/gloway/projects/reap-new-update-model-value-head/reap-mcts-lean-v2-code-1/v2/`
> （另有镜像路径 `.../reap-new-update-model-value-head/lean-v2/v2/`）。
> 角色：V2 把 V1“每条边 = 一个 Lean tactic”的搜索，升级为“每条边 = 一个**元动作（meta action）**”——元动作可以执行效应（Eff 通道）、新增引理并送 kernel gate、从观察序列中挖掘规律、打补丁/填洞。核心新概念：**塔（Tower）L + gate**、**Eff 通道白名单**、**元动作空间的 PUCT**。
> 教学链接：`../07-MCTS+V1/`、`../08-V1-1/`、`../06-RL-RLVR/`。

## 0. 文件树与职责

```
v2/
├── mcts_loop.py    117  ★ V2 MCTS 主循环：policy 采样 → 元动作执行(eff/adddecl/mine/patch/fillhole) → PUCT → 塔增长
├── policy_client.py 50  ★ 策略客户端（OpenAI 兼容端点 + mock 回退），返回“元动作”文本
├── tower.py         50  ★ 塔：库 L + gate 回调；register 仅当 gate ok；depth/height 单调
├── gate_lean.py     36  ★ 真实 kernel 门（reap-lean 容器执行 Lean 验证引理条目）
├── eff_registry.py  55  ★ Eff 通道白名单（deterministic / existential；existential 默认禁用）
├── mine.py         117  ★ 规律挖掘器：多项式插值/线性递推（F_k 安全类）+ 未证命题（F_c）
├── runner.py        91  V2 最小 harness（元动作循环 + 塔 + Eff + 样本输出）
├── smoke_v2.py / smoke_v2_full.py / smoke_emergent.py  冒烟脚本
└── __init__.py
```

## 1. `mcts_loop.py`（117 行）——V2 主循环 ★

- PUCT 常量 `C_BASE=3200.0, C_INIT=1.0`（`:16`）；`Node`（`:20`）：`state/prior/n_visits/value_sum/children`。
- `c_init_for(N)`（`:28`）：`c(N)=C_INIT + log((N+C_BASE+1)/C_BASE)`。
- `V2MCTS`（`:32`）：构造带 `goal/policy/tower/gate_mode/num_samples/seed/series`；默认 `series` 是立方和序列前 10 项（`:44`，多项式安全类原料）。
- `_select`（`:46`）：`Q = value_sum/visits`，`U = c(N)·prior·√N/(n+1)`，取 `Q+U` 最大子动作。
- `_evaluate`（`:58`）：按元动作前缀分派：
  - `effect:<verifier>`（`:60`）：构造 `EffSpec` 查白名单，成功 +0.1 / 失败 -0.2（`:63-64`）。
  - `adddecl:<type>~<body>`（`:65`）：构造 `TowerEntry`，`gate_mode=="lean"` 时过 `gate_lean`；`tower.register` 成功 +0.5 / 拒绝 -0.3（`:74-75`）。
  - `mine:series`（`:76`）：`detect(series)`；若判定为安全类 `F_k` 则过 gate 并注册（+0.5/-0.2），否则返回未证命题（+0.05，无 gate）。
  - `patch:`/`fillhole:`（`:85-88`）：占位 0.0；未知 -0.5。
- `run(steps=16)`（`:91`）：每步从根下行 `_select` ≤4 层 → 用 `policy.sample(prompt, n=ns)` 采样元动作（`prior=exp(logp)`）→ `_evaluate` 得 reward → 更新节点 visits/value_sum → 记录 `{depth, action, r, tower size, height, note, ts}`；返回 `tower_size/tower_height/nodes/log`。**塔高/塔大小是 V2 的核心产物指标**。

## 2. `policy_client.py`（50 行）——元动作策略

- `MOCK_ACTIONS`（`:8`）：`effect:arith-check`、`effect:sqsum-check`、`adddecl:1 + 1 = 2~decide`、`mine:series`、`patch:∀→∃`、`fillhole:h0` 等。
- `PolicyClient`（`:19`）：`_probe`（`:26`）探测 `/health` 决定 `v2`(mock-cpu) / `gpu` / `mock-local`。
- `sample()`（`:35`）：mock-local 随机返回元动作 + logprob；否则 POST `/v1/chat/completions`，解析 `(text, logprob_avg)`；异常回退到 `effect:arith-check`（`:49-50`）。**与 01/03 的 policy 端点协议一致**，只是动作语义从 Lean tactic 变成元动作。

## 3. `tower.py`（50 行）与 `gate_lean.py`（36 行）——塔与 kernel 门 ★

- `TowerEntry`（`:12`）：`name/body/deps/type`；`type` 是 gate 要验证的目标语句（标准库命题，默认 `1 + 1 = 2`）。
- `Tower`（`:20`）：`register(e, gate_ok)`（`:23`）**仅当 gate ok 才入 L**；`depth(e)`（`:29`）= 已注册依赖数；`height()`（`:32`）= 全部条目最大 depth；`save/load`（`:35/:41`）JSON 持久化。
- `gate_lean`（`gate_lean.py:17`）：把 `theorem <name> : <type> := by <body>` 写入临时 `.lean`，`podman run` 进 `reap-lean` 镜像执行 `lean`；无 `error`/`unsolved` 即 kernel 合法（`:33`）。镜像 `ghcr.io/wufuju2023-cell/reap-lean:4.28.0-rc1`，超时 120s（`:12-13`）。**gate 是外部 kernel 校验（回调），塔的抽象深度因此可被信任。**

## 4. `eff_registry.py`（55 行）——Eff 通道白名单 ★

- `EffClass`（`:10`）：`DETERMINISTIC` / `EXISTENTIAL`。
- `EffSpec`（`:16`）：`name/in_vals/verifier/klass`；`EffObs`（`:28`）：`value/ok/trace`。
- `_REGISTRY`（`:35`）：白名单效应（code-verified）——`sqsum-check = Σv²`、`arith-check = Σv`。
- `lookup`（`:41`）：**existential 默认禁用**（“禁止把实验噪声当证据”，`:42-43`）；未知 verifier 拒绝；纯函数执行，异常返回 ok=False。
- `public_verifiers`（`:54`）：列出可用 verifier。

## 5. `mine.py`（117 行）——规律挖掘器 ★

- `Candidate`（`:14`）：`kind(poly/recurrence/identity/open)/coeffs/stmt/cls(F_k/F_c)/evidence/score`。
- `finite_diff_const`（`:27`）：等距整数序列的 k 阶有限差是否恒定（判断 ≤k 阶多项式）。
- `fit_polynomial`（`:38）：尝试 ≤6 阶多项式 → `Candidate("poly", F_k)`。
- `fit_linear_recurrence`（`:50`）：2 阶线性递推的 Cramer 求解 + 其余点验证 → `Candidate("recurrence", F_k)`。
- `refute_search`（`:82`）：非安全类 `F_c` 在证据窗口后 m 个点做反例搜索。
- `classify`（`:96）/`score`（`:100）：安全类 `F_k` 直接给 score=1（交由 gate 终裁）；`F_c` 用 `ρ·(1-反例)`。
- `detect`（`:111`）：主入口——先试 F_k（poly/recur），失败降级为带 score 的 open `F_c` 候选。**所有计算为 DeterministicE 语义（纯函数）**。

## 6. `runner.py`（91 行）——最小 harness

- `V2State`（`:15`）：`goal/tower/obs_history/depth_budget`。
- `V2Harness`（`:27`）：`step`（`:39`）执行元动作（effect/adddecl/fillhole/patch），写与 `v1_sink` 兼容的样本（`kind: effect/tower/tower_reject`，`:57-59`）；`_sink`（`:34`）追加 JSONL。
- `main`（`:71`）：用一组元动作序列跑一遍，打印塔大小/高度与条目。

## 7. 与 V1 的差异

| 维度 | V1（03/06 篇） | V2（本篇） |
|---|---|---|
| 动作空间 | Lean tactic（字符串） | **元动作**：`effect/adddecl/mine/patch/fillhole` |
| 树结构 | OR/AND（子目标） | 单一 OR 式元动作链 + 塔（Tower） |
| 新增概念 | — | **Eff 通道**（确定性效应白名单）、**Tower**（kernel-gated 库）、**Mine**（规律挖掘） |
| 价值 | search_visit_backup（visit/value_sum） | 元动作即时 reward（+0.5/-0.3/±0.1/±0.2 等启发式） |
| PUCT | `pb_c_base=200, pb_c_init=0.001`（search.py） | `C_BASE=3200, C_INIT=1.0`（接近 AlphaProof 论文 `pb_c_base=3200`） |
| gate | Lean 直接执行 tactic | `gate_lean` 用 `podman run lean` 做 kernel 校验，只放行通过的条目进塔 |
| 产物 | proof_script / 轨迹 | `tower_size/tower_height` + obs_history + 元动作 log |

## 8. 想改造应先动哪里

- **元动作集合/奖励**：`mcts_loop.py:_evaluate`（`:58`）与 `policy_client.py:MOCK_ACTIONS`（`:8`）。
- **PUCT 超参**：`mcts_loop.py:C_BASE/C_INIT`（`:16`）与 `c_init_for`（`:28`）。
- **塔门控**：`tower.py:register`（`:23`）与 `gate_lean.py:gate_lean`（`:17`）。
- **Eff 白名单**：`eff_registry.py:_REGISTRY`（`:35`）与 `lookup`（`:41`）。
- **挖掘器策略**：`mine.py:fit_polynomial`（`:38`）/`fit_linear_recurrence`（`:50`）/`score`（`:100`）。

## 9. 对应教学篇

- `../07-MCTS+V1/`（PUCT / 搜索闭环）、`../08-V1-1/`（sink/经验、policy 端点）、`../06-RL-RLVR/`（奖励/价值）。
- 与 04 篇 `search.py`/`experience_collection.py` 对照：V2 把 AND/OR 与价值 backup 换成“元动作 + 塔 + Eff”。
