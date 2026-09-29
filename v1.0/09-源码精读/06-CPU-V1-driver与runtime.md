# 06 · CPU 版本：V1 driver 与 runtime

> 源码位置：
> - CPU 运行时：`a@my-new-linux:/mnt/gloway/projects/reap-new-update-model/v1-result/reproduction/code/src/cpu_runtime/`（28 个 `.py`，约 6.4k 行）。
> - CPU 侧源码包：`.../reap-new-update-model/new-v1-gather-source-code-cpu/`（`python-driver/{v1_run.py,v1_sink.py,mock_policy_server.py}`、`reap-training/`、`reap-upstream/`、`lakefile.toml`、`lean-toolchain`）。
> - Lean 侧补丁：`.../v1-result/reproduction/code/src/containers/cpu/patches/0001..0004`（见 03 篇 §11）。
> 角色：V1 的 **CPU 控制面/数据面**——把“为每个定理开一个隔离的 Lean 进程 → 搜索中途按 observer 屏障做 TTT 更新 → 把成功经验切成可复用数据集”整条链路做成**可断点、可续传、不确定不重试**的系统。GPU 模型由 03 篇的 `GpuRuntime` 提供，本目录用 `GpuHttpClient` 调用。
> 教学链接：`../07-MCTS+V1/`、`../08-V1-1/`、`../06-RL-RLVR/`。

## 0. 文件树与职责

```
cpu_runtime/
├── batch_solver.py      254  并发、隔离地跑多个 Lean/Reap 会话（每会话独立目录 + 端点 env）
├── online_batch.py      608  有界在线 V1 批次：durable resume、证据重放校验、retirement
├── online_ttt.py        579  observer 驱动的 in-search TTT 协调器（checkpoint ACK 屏障 + learn）
├── segmented_ttt.py     552  分段 TTT 控制器（每段从根重搜，更新只在树之间发生）
├── http_clients.py      119  GpuHttpClient（标准库 urllib）：create/snapshot/learn/retire/restore
├── transport_budget.py   44  传输层嵌套超时预算（fetch<extension<daemon<daemon_http<cli）
├── matchmaker.py        503  有限 Matchmaker（plan_next/reserve/record_result，有限状态机）
├── collector_batch.py   366  有界 collector 执行器（search 2 槽 + verify 1 槽）
├── verified_collector.py 346 已验证证据收集器
├── collector_verify.py  264  验证回调
├── verified_dataset_store.py 261 内容寻址的 verified 数据集读写
├── verified_trajectory.py 377    strict verified trajectory 加载（17 文件 bundle）
├── mathlib_trajectory.py 281     human Mathlib SFT 加载（22 文件 bundle）
├── release/retire 相关：released_attempts.py 63 / publish_experience.py 40 / experience_policy.py 327
├── curriculum_admission.py 128 / target_curriculum.py 271 / target_variants.py 116
├── closed_problem.py 153 / normalize_rollout.py 191 / replica_collector.py 291
├── reap_prompt.py 33 / mock_services.py 62 / reap_act_runner.py 118
├── run_ttt.py 8 / run_toy_ttt.py 69 / __init__.py 2
gather/python-driver/   v1_run.py（BatchSolver）、v1_sink.py（RolloutSink）、mock_policy_server.py
gather/reap-training/ + reap-upstream/    Lean 侧 Reap（配合补丁 0001–0004）
```

## 1. `batch_solver.py`（254 行）：并发隔离的 Lean 会话

- `SessionSpec`（L27）与 `parse_spec`（L55）：session_id（正则 `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}` 整串匹配）/theorem_file/policy_base_url/value_base_url/ps_endpoint；`load_manifest`（L68）读 JSONL 并拒绝重复。
- `run_session`（L149）：为每个会话建 `output_root/<session_id>/`，注入环境变量 `REAP_SESSION_ID/REAP_SESSION_DIR/REAP_POLICY_ENDPOINT/REAP_VALUE_ENDPOINT/REAP_PS_ENDPOINT`（L163–171），执行 `lake env lean <theorem>`，写 stdout/stderr 与 `session.json`。
- `run_process`（L103）：新进程组（POSIX `start_new_session` / Windows `CREATE_NEW_PROCESS_GROUP`），超时用 `_stop_process_tree`（L88）**只杀本会话进程树**（POSIX `killpg SIGTERM`，Windows `taskkill /T /F`），返回 `(124, True, …)`。
- `run_all`（L201）：`asyncio.Semaphore(concurrency)` 限流；`write_summary`（L217）写 `summary.jsonl`；`main`（L234）返回 0 当全部 solved。

## 2. `online_batch.py`（608 行）：有界在线批次 ★

- `PROFILE="online-search-visit-backup-v1"`（L29）；`BatchBlocked`（L36）表示“需人工介入、不得重试原会话”。
- 原子发布 `publish`（L87，fsync → `os.replace`，默认拒绝覆盖）+ `batch_lock`（L101，fcntl/msvcrt 独占，第二写者即 Blocked）+ `no_links`（L53，拒绝 symlink/junction）。
- `load_manifest`（L136）：每行 `session_id`+`theorem_file`；可选 experience 三件套必须同时给（L148–160）；theorem 必须位于 project 内（L164–169）且记录 sha256。
- **`completed_record`（L204）**：对 completed 会话做**全量证据重放校验**——intent 与 batch identity 绑定、session.json 的 profile/tree_id/gamma/max_updates/端点、online-result.json 的 status/returncode/root_verified/optimizer_updates==policy_version、result.json 的 `reap.training.result.v1`、proof_script（solved）或 exhausted 终态（未解）、`learn-*.request.json` 与 `learn-*.receipt.json` 一一配对且版本连续、create-receipt policy_version=0、snapshot before/after，并对带 experience_policy 的会话调用 `validate_policy`/`validate_execution_contract`（L289–300）。任一不符即 `BatchBlocked`。
- `retirement_intent`（L330）与 `retire_completed_record`（L384）：提交 retirement 前先写持久 intent（dispatch 边界），收到 `released` 收据才推进；`validate_retirement_receipt`（L340）校验 v1/v2（含 reuse 模式）。
- **`run_batch`（L399）**：参数校验（concurrency∈[1,32]、gamma、URL origin 无凭据）→ 构造 `identity`（manifest sha + entries + config + 实现文件 sha，L435）并与 `batch-config.json` 绑定 → 分类 pending/recovered/skipped → `ThreadPoolExecutor(concurrency)` 派发（派发前先写 `.batch-intents/<sid>.json`，L530）→ 完成后 `completed_record` 校验再原子追加到 `solutions.jsonl`/`terminal-unsolved.jsonl`（`append` L482 保持前缀不变）→ 失败即 `manual_intervention_required`。返回 `reap.online-batch.result.v1`（L563）+ 调度指标。

## 3. `online_ttt.py`（579 行）：in-search TTT 协调 ★

- 常量（L25–30）：`OBJECTIVE="search_visit_backup"`、`VALUE_SEMANTICS`、`MAX_LEARN_BODY_BYTES=8192`、`EXPERIENCE_RESETS`。
- `validate_initialization`（L44）/`validate_created`（L90）：校验 create receipt 的 theorem/experience lineage、fresh optimizer/version/buffer/event 状态、objective/gamma/value_semantics。
- `validate_learn_receipt`（L102）：event/version/applied/idempotent、objective/optimizer_steps、5 个 finiteness 门（`finite_loss/gradients/parameters/optimizer_state/base_parameters_frozen`）、loss/policy_loss/kl/value_loss/grad_norm/value_target 有限、`search_trace` 与事件一致、prompt sha、**`parameter_diffs` 各组确实改变**（L123–128）。
- `validate_learn_body`（L131）：候选 ≤64，编码 body ≤8192 字节（浏览器传输契约，**绝不截断**）。
- `JsonlTail`（L159）：只读**完整换行**的 observer 记录，保留尾半行。
- **`OnlineCoordinator`（L178）**：
  - `_validate`（L202）：observer schema `reap.training.observer.v1`、session/tree 匹配、`sequence` 严格递增、`policy_version` 一致。
  - `_target`（L213）：从观察到的 `backed_up_nodes` 中找未解 OR 节点，要求 `numVisit>0` 且 `valueSum/numVisit <= -1`（非终局价值目标，L229）；每个 tactic 边必须有 generation provenance（`edge_sources`，L237）、同一目标只来自一个 prompt（L248）；已消费的 stats 不再训练（L251）。构造事件 `{kind:"search_visit_backup", event_id, policy_version, prompt, gamma, candidates[tactic/visits/raw_logprob/behavior_version], backup{value_sum,visits,kind:"OR",valid:true}, reward:0, terminal_verified:false}`（L253–259）。
  - `accept`（L265）：处理 `generation`（登记，version>0 时计入 `post_update_generations`）、`eval`（须匹配 generation；`created` 边登记 provenance；merged/eval_rejected/ancestor_rejected 允许）、`backup`（记 backed_up_nodes）、`checkpoint`（L301：写 intent → `self.learn` → 校验 receipt → 写 receipt → version+1 → 写 ACK；异常则 ACK status=error 并抛，**绝不重发**）、`selection_value_refresh`（L334）。
  - `_accept_selection_refresh`（L334）：校验 scope=`selection-only-not-training-Q`、版本/step、OR 值 ≤0、AND 值=未解子节点最小，更新后再继续搜索。
  - `finish`（L372）：若仍有 pending refresh 则报错。
- `run_online`（L395）：写 session/create-receipt/snapshot-before，设置 `REAP_OBSERVER_PATH/REAP_CHECKPOINT_DIR/REAP_CHECKPOINT_TIMEOUT_SECONDS/REAP_SELECTION_VALUE_REFRESH`（L469–476），启动 Lean 并循环 `tail.read → coordinator.accept`；结束写 report（`passed_execution`/`solved_without_online_update`/…）与 snapshot-after（仅在无 error 时，L522–525）。

## 4. `segmented_ttt.py`（552 行）：分段 TTT 状态机

- 契约 dataclass：`RolloutEvent`（L38）、`ActRequest`（L50）、`ActResult`（L60）、`TrainingEvent`（L68）、`LearnRequest`（L85）、`LearnResult`（L95）、`SessionOutcome`（L102）；`ActRunner`/`LearnClient` 协议（L113/L117）。
- **每段 ACT 从定理根重新搜索**，一棵 MCTS 树绝不会含不同 policy version 的节点（模块 docstring）。
- `_training_reward`（L146）：非终局的 verifier/tactic 失败记 -1，其余（含基础设施 timeout）记 0；`build_training_events`（L159）：`terminal_verified` 必须配 `root_verified` verdict。
- `SegmentedTTTController.run`（L253）：循环 `max_segments`，每段后构造 events 调用 LEARN，并要求 **acknowledged event_ids 集合恰好等于提交集合**（L487）且 **policy_version 严格增大**（L500），否则抛 `EventAcknowledgementError`/`PolicyVersionError`；只在两棵树之间更新。
- `JsonlJournal`（L196）：每事件 `flush`。

## 5. 传输与工具

- `http_clients.py`（119 行）：`GpuHttpClient`（L15）——`_request`（L22，JSON、`allow_nan=False`）；`create_session`（L40，支持 theorem/experience/release/role/contract pin）、`snapshot`（L62，可 `for_experience`）、`publish_experience`（L68）、`restore`（L73）、`delete_session`（L76）、`retire_session`（L79，**只提交一次**）、`retirement_receipt`（L89）、`learn`（L93，逐事件提交并校验 event_id 与 policy_version 递增）。
- `transport_budget.py`（44 行）：`TransportBudget`（L11）——`fetch=20 < extension=115 < daemon=120 < daemon_http=130 < cli=150`（L25 强制内到外递增），另有 `lock=60, worker=600, recovery=60`；派生 `transport_call`（L29）、`bridge`（L32）、`client`（L36）、`barrier`（L40）。`DEFAULT_BUDGET`（L44）被 `online_batch/online_ttt` 使用。
- `matchmaker.py`（503 行）：有限 Matchmaker——`MatchmakerBlocked`、`canonical_bytes/content_sha256`、`_publish/_read/_fields/_sha/_int`、`_Lease`（目录租约）、`plan_next`/`reserve`/`record_result`/`_validate_result`/`status`（带 `run_sha256`）。
- `collector_batch.py`（366 行）：**有界回调执行器**——`run_collector_batch`（L153）要求 `search_workers∈[1,2]`、`validation_workers==1`、`max_pending_validation∈[1,2]`（L172–174）；`execute_search`（L223）/`execute_verification`（L241）把每次 mutation 只提交一次；任何异常经 `diagnostic`（L204）转 `unknown`（记录 phase/exception 类型、`mutation_retry_allowed=False`），并停止派发；已完成尝试 `audit_completed`（L113）逐文件核对后跳过。
- `verified_collector.py`（346）/`collector_verify.py`（264）/`replica_collector.py`（291）：证据收集与独立验证。
- `verified_dataset_store.py`（261）/`verified_trajectory.py`（377）/`mathlib_trajectory.py`（281）：内容寻址的 verified bundle（`safe_directory`/`read_bundle`）与“生成 17 文件 / Mathlib 22 文件”两套加载器，供 03 篇 backend 使用。
- `experience_policy.py`（327）：经验初始化策略（fresh/chain/bank catalog 与 selection），`validate_policy`/`validate_execution_contract`。
- `released_attempts.py`（63）：`ReleasedAttemptProvider`（已发布 attempt 的 selection sidecar）。
- `curriculum_admission.py`（128）/`target_curriculum.py`（271）/`target_variants.py`（116）/`closed_problem.py`（153）/`normalize_rollout.py`（191）/`reap_prompt.py`（33）/`mock_services.py`（62）/`reap_act_runner.py`（118）：课程、目标变体、rollout 归一化、prompt、mock、ACT 适配。

## 6. CPU 容器源码包（`new-v1-gather-source-code-cpu`）

- `python-driver/v1_run.py`：BatchSolver 编排器（`make_lean` 生成 `import Reap` + `set_option reap.*_endpoint` + `theorem … := by reapMCTS` + `#eval IO.println "%%TASK_…_DONE%%"`，`podman run --network host` 执行，`state/<id>.done` 断点）。
- `python-driver/v1_sink.py`：RolloutSink（`node_visited/task_done/rttt_update` + verdict 白名单，`state_key=sha256`）。
- `python-driver/mock_policy_server.py`：CPU mock。
- `reap-training/`、`reap-upstream/`：Lean 侧 Reap 源码；补丁 `0001–0004`（见 03 篇 §11）：环境变量端点、`MCTSObserver`/`observe?`/`reportCheckpoint?`、严格 value 错误（禁 -1000 哨兵）、`SelectionValueRefresh`（选择专用价值刷新）。

## 7. 并发 / 隔离 / 断点（核心不变式）

- **并发**：批次用 `ThreadPoolExecutor(concurrency)`（`online_batch`），会话用 `asyncio.Semaphore`（`batch_solver`）；collector 用 2 search + 1 verify 槽；GPU 侧所有可变操作仍由 03 篇的单一 `GpuActor` 串行（本目录只是客户端）。
- **隔离**：每会话独立输出目录与端点；`student output` 通过独立 `session_id` 绑定独立 LoRA/value/optimizer（03 篇）。
- **断点**：`state/*.done`（driver）、`.batch-intents/<sid>.json`（派发意图先落盘）、`checkpoints/learn-*.request.json`/`learn-*.receipt.json`（LEARN 意图→ACK）、`snapshot-before/after`；重启时已完成会话被**重放校验并跳过**。
- **不重试**：`publish` 默认拒绝覆盖；不确定结果抛 `BatchBlocked`/`unknown` 并停止新派发（人工介入）。

## 8. 与其它版本差异

- 与 **01 V1 app**：01 的 `v1_run.py`/`v1_sink.py` 是极简 BatchSolver/JSONL；本目录把同样的闭环工程化为“多会话 + observer 屏障 + 断点续传 + 证据链审计”。
- 与 **03 gpu_runtime**：03 是 GPU 侧会话/事务/发布；本目录是其 CPU 侧 driver（通过 `GpuHttpClient` 与 `/sessions/...` 端点对接，超时改用 `transport_budget` 的 FIFO 预算）。
- 与 **04 nanoproof**：nanoproof 自带 MCTS/Prover；本目录把搜索交给 Lean 侧的 `reapMCTS`，CPU 只做编排与经验抽取。

## 9. 想改造应先动哪里

- **并发/隔离**：`batch_solver.py:run_session`（L149）、`online_batch.py:run_batch`（L399）。
- **in-search 更新语义**：`online_ttt.py:_target`（L213）/`accept`（L265）/`validate_learn_receipt`（L102）。
- **分段更新**：`segmented_ttt.py:SegmentedTTTController.run`（L253）。
- **证据/断点**：`online_batch.py:publish`（L87）、`completed_record`（L204）。
- **超时预算**：`transport_budget.py:TransportBudget`（L11）。
- **采集/验证**：`collector_batch.py:run_collector_batch`（L153）、`matchmaker.py`。
- **Lean 侧事件**：`containers/cpu/patches/0002`、`0004`。

## 10. 对应教学篇

- `../07-MCTS+V1/`（MCTS 闭环、`reapMCTS`）、`../08-V1-1/`（会话/事务/发布/经验）、`../06-RL-RLVR/`（搜索中途 TTT、replay/负例）、`../04-价值头/`（value 端点）。
- 既有流程笔记：`文档/mcts-theory-learning/MCTS算法流程/V1-MCTS完整运行流程与算例.md`。
