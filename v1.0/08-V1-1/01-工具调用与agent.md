# 08 · 工具调用与 Agent（V1-1 证据环）

> 前两章模型只能“自己采样”或“在证明树里搜索”。V1-1 再进一步：让 agent **主动调用工具**收集证据，
> 把树外信息以贝叶斯方式注入先验，并与 GPU runtime、opencode 打通。
>
> 配套代码：[`code/tool_agent.py`](code/tool_agent.py)（`F-b08-tool_agent`）、[`code/opencode_bridge.md`](code/opencode_bridge.md)（doccode `BRIDGE`）。
> 实验：notebook `N17_tool_calling_agent`（`N-17`）。
>
> **【文档｜DOC-V11】**（doccode = `V11`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 0. 三个关键词

**【注 V11.1.1｜R-V11.1.1】（三个关键词）**

1. **agent 循环**：模型输出“思考 + 工具调用”，系统执行工具并把结果回灌，循环直到给出最终答案。
2. **工具调用协议**：OpenAI 原生 `tool_calls`，以及 v1-1 的 agentic DSL `{"calls":[...]}`。
3. **证据环（evidence ring）**：把工具产出变成 `Evidence`，用 $\phi=\tanh(\sum w)$ 重加权策略先验。

---

## 1. agent 循环

**【代码 V11.2.1｜Cd-V11.2.1】（agent 循环伪代码）**

最小循环（对应 `code/tool_agent.py`）：

```
messages = [system, user]
while steps < max_steps:
    out = policy(messages)                     # LLM 生成
    calls = parse_tool_calls(out)              # 解析 tool_calls / {"calls":[...]}
    if not calls:
        return out                             # 无调用 -> 最终答案
    for c in calls:
        result = run_tool(c["name"], c["args"])   # 本地/远程执行
        messages.append(tool_message(c, result))  # 结果回灌
return last_out
```

**【注 V11.2.2｜R-V11.2.2】（循环要点）**

要点：**结构化动作**（函数名+参数）、**证据回灌**（工具结果进入下一轮上下文）、
**终止条件**（无调用 / 达 `max_steps` / 显式 final）、**可验证**（效果最终由环境判定）。

---

## 2. 工具调用协议

### 2.1 OpenAI tool-call（工业标准）

**【定义 V11.3.1｜D-V11.3.1】（OpenAI tool-call 协议）**

请求里声明 schema：

```json
{"tools": [{"type": "function", "function": {
  "name": "lean_search", "description": "Search Mathlib for lemmas",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]}
```

模型回复 `{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
"function": {"name": "lean_search", "arguments": "{\"query\": \"sq_connected\"}"}}]}`，
系统执行后以 `{"role": "tool", "tool_call_id": "call_1", "content": "..."}` 回灌。

### 2.2 agentic DSL `{"calls":[...]}`（v1-1 / 容器数据格式）

**【定义 V11.3.2｜D-V11.3.2】（agentic DSL `{"calls":[...]}`）**

```json
{"calls": [
  {"name": "leanSearch", "args": {"query": "sq_connected"}},
  {"name": "fileSearch", "args": {"path": "Mathlib/Topology/Basic.lean"}}
]}
```

**【注 V11.3.3｜R-V11.3.3】（DSL 解析规则与 `EvidenceKind`）**

解析（`tool_agent.py`）：截取第一个平衡的 `{...}`；校验 `calls` 是列表、每项含 `name/args`；
依次执行并把结果包成 `{"name","ok","result"}` 回灌；解析失败或无 `calls` 键则当作最终答案。

`Agentic.lean` 的 `EvidenceKind` = 工具类别：`linkSearch` / `leanSearch` / `fileSearch` / `toolCall` / `manual`。

---

## 3. 证据注入与贝叶斯重加权（`Agentic.lean`）

**【定义 V11.4.1｜D-V11.4.1】（证据与审计结构）**

- `Evidence {id, kind, weight, payload, sourceDesc}`，`weight ∈ [0,1]`（0=无关，1=决定性）；
- `RingEvent {atStep, kind, evidenceId, payloadHash, phi}`：审计事件；
- `RingLog {session, events}`：会话级审计日志；
- `V11Contract {maxEvidenceQueries=24, temperature=100, decayAfterSteps=8}`。

**【定义 V11.4.2｜D-V11.4.2】（组合律与贝叶斯重加权）**

组合律：

$$
\ell = \sum_k w_k, \qquad \phi = \tanh(\ell) \in [0,1].
$$

`tanh` 饱和避免单条“决定性”证据无限主导。对应贝叶斯重加权：

$$
\pi'(t \mid s, R) \;\propto\; \pi_\theta(t \mid s)\cdot P(R \mid t, s),
$$

其中树内先验 $\pi_\theta(t\mid s)$ 由策略网络的 softmax 给出（【定义 1.2.1】）。

**【代码 V11.4.3｜Cd-V11.4.3】（`build_prompt` 实现）**

工程实现（`driver/evidence_loop.py`）不真改概率，而是把证据拼进 prompt 的 prior 段：

```python
def build_prompt(goal, related):
    ps = "\n".join(f"lemma {r}" for r in related if r)
    return "State:\n" + goal + "\n\nRelated theorems:\n" + (ps or "(none)")
```

**【注 V11.4.4｜R-V11.4.4】（证据 A/B 度量）**

`run_problem_with_evidence` 做 A/B：无证据 value0 vs 注入证据 value1，用 `delta=value1-value0` 度量证据是否帮助。

---

## 4. GPU runtime：actor / session / learner 与 RTTT hooks

### 4.1 HTTP 服务器（`gpu/gpu_runtime/server.py`）

**【代码 V11.5.1｜Cd-V11.5.1】（HTTP 服务器路由表）**

| 方法 | 路径 | 作用 |
|---|---|---|
| POST | `/sessions/{id}` | 建会话（`theorem_id`, `role`, `model_release_sha256`…） |
| POST | `/sessions/{id}/policy/v1/chat/completions` | 策略采样（OpenAI chat） |
| POST | `/sessions/{id}/value/v1/chat/completions` | 价值查询 |
| POST | `/sessions/{id}/learn/v1` | 一步更新（带 `expected_policy_version`） |
| POST | `/sessions/{id}/snapshot/v1` | 快照 |
| GET | `/sessions/{id}/retire/v1` | 回收会话 |
| GET | `/health` | 健康 + GPU actor 指标 |

### 4.2 单工作线程 actor（`actor.py`）

**【定义 V11.5.2｜D-V11.5.2】（单工作线程 actor）**

`GpuActor` 是单线程 FIFO：所有可变 GPU 操作经 `submit(fn)` 排队串行执行，避免并发 HTTP 同时改模型；
记录 `submitted/started/completed/failed/active/queued` 与队列等待/执行耗时。

### 4.3 会话生命周期（`runtime.py` `GpuRuntime`）

**【定义 V11.5.3｜D-V11.5.3】（会话生命周期与乐观并发）**

- `create_session(session_id, role=...)`：`theorem` / `learner` / `actor`；
- **乐观并发**：`learn(session_id, expected_policy_version, event)` 版本不匹配即拒绝（`:425`）；
- 快照/回滚：失败 `import_session` 回滚；回滚失败则进**隔离区**（`_unusable_sessions`）不再服务；
- `max_resident_sessions` 限制常驻会话数。

### 4.4 持久 learner（`learner.py` `LearnerCoordinator`）

**【定义 V11.5.4｜D-V11.5.4】（持久 learner 与 release）**

带**持久化 journal + 写者锁**的验证回放 learner：每个 committed step 得到不可变 checkpoint，
发布为 content-addressed release（`publish_checkpoint` / `LearnerReleaseStore`）；“durable intent”，未知结果不重试。

### 4.5 RTTT hooks

**【定义 V11.5.5｜D-V11.5.5】（RTTT hooks）**

Lean 搜索写 `observer.jsonl` → `OnlineCoordinator.accept` 聚合成 learn event
→ `POST /sessions/{id}/learn/v1` → 新策略服务后续展开（`post_update_generations` 证明闭环闭合）；
每步写 checkpoint，`validate_learn_receipt` 校验有限性与冻结基座标志。

---

## 5. 与 opencode 的对接

`driver/v1_transports.py` 三档 transport（env `V11_TRANSPORT`）：

**【定义 V11.6.1｜D-V11.6.1】（三档 transport）**

| 模式 | 说明 |
|---|---|
| `mock` | 本地 `mock_policy_server`（smoke 默认，零云端负载） |
| `local-opencode` | 本地 opencode 证据环（`OpencodeEvidenceBackend`，默认固定证据 stub） |
| `remote` | 用户侧/云端 GPU `HttpPolicyServer` |

**【注 V11.6.2｜R-V11.6.2】（RuntimeTransport 与证据生产者）**

`RuntimeTransport`（`driver/runtime_transport.py`）实现真实 session-based 协议：
`session(pid)` 建会话 → `policy(prompt, n)` / `value(prompt)` → `retire(sid)` 回收。
`OpencodeEvidenceBackend.evidence_snapshot` 是证据生产者骨架：真正接线时通过 opencode CLI/run
子进程产出 `linkSearch` / `leanSearch` / `fileSearch` 证据，交给 evidence loop 注入。

**【注 V11.6.3｜R-V11.6.3】（树内采样 vs 树外证据）**

> 树内采样 = GPU policy（RULE-0）；agent 证据环 = 树外证据；$\phi=\tanh(\sum w)$ 贝叶斯重加权。Lean 侧零 HTTP。

---

## 6. V1-1 与容器 agentic DSL 数据格式

**【代码 V11.7.1｜Cd-V11.7.1】（DSL 样例）**

```json
{
  "state": "State:\n⊢ a^2 + 1 = b^2 + 1 → a = b ∨ a = -b",
  "evidence": [
    {"id": "e0", "kind": "leanSearch", "weight": 0.25,
     "payload": "lemma sq_eq_sq_iff {a b : ℝ} : a^2 = b^2 ↔ a = b ∨ a = -b",
     "sourceDesc": "opencode:lean-search"}
  ],
  "calls": [
    {"name": "leanSearch", "args": {"query": "sq_eq_sq_iff"}},
    {"name": "fileSearch", "args": {"path": "Mathlib/Algebra/Ring/Basic.lean"}}
  ],
  "phi": 0.2449,
  "tactic": "linarith [sq_nonneg (a - b)]"
}
```

**【注 V11.7.2｜R-V11.7.2】（字段说明）**

`state` 当前目标；`evidence` 证据（带权重）；`calls` 本轮工具调用（DSL）；`phi` 聚合强度；`tactic` 最终动作。

---

## 7. 本目录代码与实验

**【注 V11.8.1｜R-V11.8.1】（本目录代码与实验）**

- `code/tool_agent.py`：最小 tool-calling agent——本地工具 + `{"calls":[...]}` 解析 + 循环回灌；纯 Python。
- `code/opencode_bridge.md`：把 opencode 当“证据生产者”接入（transport / 工具映射 / 落盘）。
- notebook `N17_tool_calling_agent`：agent 循环 + 证据注入 A/B。

---

## 8. 参考

**【注 V11.9.1｜R-V11.9.1】（参考与源码索引）**

- `v1-1-agentic-tool/cpulean/Reap/Agentic.lean`（Evidence / RingLog / reweightBayes / V11Contract）
- `v1-1-agentic-tool/driver/{evidence_loop.py,v1_contract.py,v1_transports.py,runtime_transport.py}`
- `v1-1-agentic-tool/gpu/gpu_runtime/{server.py,actor.py,runtime.py,learner.py}`
- `v1-1-agentic-tool/README.md`
