# 08 · V1-1（工具调用与 Agent）

> **【文档｜DOC-RM08C】**（doccode = `RM08C`）｜编号与 Tag 规范见《00-风格与编号规范》。

本目录讲清楚 agent 循环、工具调用协议、证据环与 GPU runtime 的对接。

## 文件

**【注 RM08C.1.1｜R-RM08C.1.1】（文件）**

本文对应代码 Tag：`F-b08-tool_agent`；配套 notebook：`N-17`。

| 文件 | 内容 |
|---|---|
| [`01-工具调用与agent.md`](01-工具调用与agent.md) | agent 循环、OpenAI tool-call / agentic DSL `{"calls":[...]}`、`Evidence`/`RingLog`/$\phi=\tanh(\sum w)$、driver 闭环、GPU runtime（actor/session/learner）、RTTT hooks、opencode transport、V1-1/容器数据格式 |
| [`code/tool_agent.py`](code/tool_agent.py) | 最小 tool-calling agent（本地模拟工具 + `{"calls":[...]}` 解析），纯 Python |
| [`code/opencode_bridge.md`](code/opencode_bridge.md) | 把 opencode 当证据生产者接入的说明 |
| notebook `N17_tool_calling_agent` | 可运行 agent 实验 |

## 快速开始

**【注 RM08C.2.1｜R-RM08C.2.1】（快速开始）**

```bash
python3 code/tool_agent.py     # 无需依赖，秒级；打印工具轨迹 + phi
```

## 与 AlphaProof 的对照

**【注 RM08C.3.1｜R-RM08C.3.1】（与 AlphaProof 的对照）**

- Lean 数据/组合律：`v1-1-agentic-tool/cpulean/Reap/Agentic.lean`。
- Python transport：`driver/{v1_contract.py,v1_transports.py,runtime_transport.py,evidence_loop.py}`。
- GPU 服务：`gpu/gpu_runtime/{server.py,actor.py,runtime.py,learner.py}`。

## 思考题

**【注 RM08C.4.1｜R-RM08C.4.1】（思考题）**

见 `01-工具调用与agent.md` §8。
