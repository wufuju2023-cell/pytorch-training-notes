# 08 · V1-1（工具调用与 Agent）

本目录讲清楚 agent 循环、工具调用协议、证据环与 GPU runtime 的对接。

## 文件

| 文件 | 内容 |
|---|---|
| [`01-工具调用与agent.md`](01-工具调用与agent.md) | agent 循环、OpenAI tool-call / agentic DSL `{"calls":[...]}`、`Evidence`/`RingLog`/$\phi=\tanh(\sum w)$、driver 闭环、GPU runtime（actor/session/learner）、RTTT hooks、opencode transport、V1-1/容器数据格式 |
| [`code/tool_agent.py`](code/tool_agent.py) | 最小 tool-calling agent（本地模拟工具 + `{"calls":[...]}` 解析），纯 Python |
| [`code/opencode_bridge.md`](code/opencode_bridge.md) | 把 opencode 当证据生产者接入的说明 |
| notebook `N17_tool_calling_agent` | 可运行 agent 实验 |

## 快速开始

```bash
python3 code/tool_agent.py     # 无需依赖，秒级；打印工具轨迹 + phi
```

## 与 AlphaProof 的对照

- Lean 数据/组合律：`v1-1-agentic-tool/cpulean/Reap/Agentic.lean`。
- Python transport：`driver/{v1_contract.py,v1_transports.py,runtime_transport.py,evidence_loop.py}`。
- GPU 服务：`gpu/gpu_runtime/{server.py,actor.py,runtime.py,learner.py}`。

## 思考题

见 `01-工具调用与agent.md` §8。
