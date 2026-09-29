# opencode 桥接说明（V1-1 证据生产者）

> **【文档｜DOC-BRIDGE】**（doccode = `BRIDGE`）｜编号与 Tag 规范见《00-风格与编号规范》。

本文件说明如何把 **opencode**（或任意浏览器/搜索/文件工具链）接到 V1-1 的 agent 上，
使工具产出变成 `Evidence` 注入先验——对应 `driver/v1_transports.py` 的 `local-opencode` 模式与
`driver/evidence_loop.py` 的 A/B 对照。

## 1. 角色分工

**【注 BRIDGE.1.1｜R-BRIDGE.1.1】（角色分工）**

```
        ┌───────────────── evidence loop ─────────────────┐
        │  problem -> 工具查询 -> Evidence -> 注入 prompt   │
        └───────▲───────────────────────────────┬─────────┘
                │ evidence_snapshot             │ policy/value
        opencode / 浏览器 / 搜索              GPU runtime (session)
```

- **opencode**：证据生产者。执行 `linkSearch`（网页/仓库链接）、`leanSearch`（Lean 库检索）、
  `fileSearch`（仓库文件导航）等，返回原始载荷。
- **evidence loop**：把载荷包成 `Evidence{id,kind,weight,payload,sourceDesc}`，算 $\phi=\tanh(\sum w)$。
- **GPU runtime**：接收注入证据后的 prompt，返回候选 tactic（policy）与 value。

## 2. 最小接入（伪代码，对本目录 `tool_agent.py` 同样适用）

**【代码 BRIDGE.2.1｜Cd-BRIDGE.2.1】（最小接入伪代码）**

```python
from v1_contract import Evidence, evidence_composite

def evidence_snapshot(problem_id, step):
    hits = opencode_search(problem_id)          # 真实 opencode CLI/run / HTTP
    return [Evidence(id=f"{problem_id}-{i}", kind=h["kind"], weight=h["weight"],
                     payload=h["payload"], sourceDesc=h["source"]).to_json()
            for i, h in enumerate(hits)]

evs = [Evidence(**e) for e in evidence_snapshot(pid, step=0)]
phi = evidence_composite(evs)                    # tanh(sum w)
hints = [e.payload for e in evs if e.kind == "leanSearch"]
prompt = build_prompt(goal, hints)               # 证据进 prompt 的 prior 段
cands = transport.policy(prompt, n=4)
```

## 3. 三种 transport 的切换（`V11_TRANSPORT`）

**【例 BRIDGE.3.1｜E-BRIDGE.3.1】（三种 transport 的切换）**

| 值 | 用法 | 适用 |
|---|---|---|
| `mock` | 本地 mock policy server，固定证据 | smoke / CI，零云端负载 |
| `local-opencode` | `OpencodeEvidenceBackend` 真调 opencode 工具 | 本地开发 |
| `remote` | 云端 GPU `HttpPolicyServer`（session-based） | 真训练 |

## 4. 工具映射（建议）

**【例 BRIDGE.4.1｜E-BRIDGE.4.1】（工具映射建议）**

| EvidenceKind | 工具 | 产出 |
|---|---|---|
| `linkSearch` | opencode 的 web/搜索工具 | URL + 摘要 |
| `leanSearch` | Mathlib/Lean 检索（PreMiseSelection 风格） | `lemma ...` 语句 |
| `fileSearch` | 仓库文件读 | 代码片段 |
| `toolCall` | 执行/验证（`lake env lean`、算术检查） | 通过/失败 + 输出 |

## 5. 注意

**【注 BRIDGE.5.1｜R-BRIDGE.5.1】（接入注意事项）**

- **可验证优先**：证据只是先验，最终动作仍须 Lean 验证；`toolCall` 的 `ok` 才是硬信号。
- **权重校准**：`weight` 决定 `phi`；建议先固定（如 0.25）跑通，再按命中率校准。
- **审计**：每条证据写 `RingEvent{atStep,kind,evidenceId,payloadHash,phi}`，可回放。
- **成本护栏**：`V11Contract.maxEvidenceQueries=24`；超出即停。
