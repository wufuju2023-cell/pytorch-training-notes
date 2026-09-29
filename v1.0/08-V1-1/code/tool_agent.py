# 【源代码｜F-b08-tool_agent】v1.0/08-V1-1/code/tool_agent.py — 最小 tool-calling agent
# 相关文档：《08-V1-1/01-工具调用与agent.md》
"""最小 tool-calling agent（本地模拟工具 + `{"calls":[...]}` 解析）。

对应《08 · 工具调用与 Agent》§1–§2：
    agent 循环 = 生成 -> 解析工具调用 -> 执行 -> 结果回灌 -> 直到最终答案。

本文件不依赖任何 LLM：`ScriptedPolicy` 演示"模型输出"，也可以传入任意
`policy(messages) -> str`（真实模型时把它的输出接进来即可）。
支持两种协议：
    * OpenAI 风格 `{"tool_calls": [{"function": {"name":..., "arguments": "..."}}]}`
    * v1-1 agentic DSL `{"calls": [{"name":..., "args": {...}}]}`
并提供 `evidence_composite`（phi = tanh(sum w)），与 `Agentic.lean` 同公式。

运行：
    python3 tool_agent.py
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 0. 证据（与 Agentic.lean 对应）
# ---------------------------------------------------------------------------
# 【F-b08-tool_agent.Evidence｜类】证据项（内容与权重）
@dataclass
class Evidence:
    id: str
    kind: str          # linkSearch | leanSearch | fileSearch | toolCall | manual
    weight: float
    payload: str
    sourceDesc: str = ""


# 【F-b08-tool_agent.evidence_composite｜函数】phi = tanh(sum w) 饱和压缩
def evidence_composite(evs) -> float:
    """phi = tanh(sum w)，饱和压缩到 [0,1]。"""
    return math.tanh(sum(e.weight for e in evs))


# ---------------------------------------------------------------------------
# 1. 本地模拟工具
# ---------------------------------------------------------------------------
LEAN_DB = {
    "sq_connected": "lemma sq_connected {a b : ℝ} : a^2 = b^2 ↔ a = b ∨ a = -b",
    "add_comm": "lemma add_comm (a b : ℕ) : a + b = b + a",
    "mul_zero": "lemma mul_zero (a : ℕ) : a * 0 = 0",
}
FILE_DB = {
    "Mathlib/Algebra/Ring/Basic.lean": "import Mathlib.Algebra.Ring.Basic\n-- ring lemmas ...",
    "Mathlib/Topology/Basic.lean": "import Mathlib.Topology.Basic\n-- topology ...",
}


# 【F-b08-tool_agent.run_tool｜函数】执行本地模拟工具
def run_tool(name: str, args: dict, trace: list | None = None):
    if name == "leanSearch":
        q = str(args.get("query", ""))
        hits = [v for k, v in LEAN_DB.items() if k in q or q in k]
        result = {"results": hits or [v for v in LEAN_DB.values()][:1]}
    elif name == "fileSearch":
        path = str(args.get("path", ""))
        result = {"content": FILE_DB.get(path, "(file not found)")}
    elif name == "calculator":
        expr = str(args.get("expr", "0"))
        try:
            result = {"value": eval(re.sub(r"[^0-9+\-*/(). ]", "", expr), {"__builtins__": {}})}
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
    elif name == "verifyTactic":
        state, tactic = str(args.get("state", "")), str(args.get("tactic", ""))
        result = {"ok": bool(tactic.strip()), "state": state, "tactic": tactic}
    else:
        result = {"error": f"unknown tool {name}"}
    if trace is not None:
        trace.append({"tool": name, "args": args, "result": result})
    return result


# ---------------------------------------------------------------------------
# 2. 工具调用解析（两种协议）
# ---------------------------------------------------------------------------
# 【F-b08-tool_agent._extract_json｜函数】截取第一个平衡的 { ... }
def _extract_json(text: str):
    """从文本里截取第一个平衡的 { ... }。"""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# 【F-b08-tool_agent.parse_tool_calls｜函数】解析工具调用列表
def parse_tool_calls(text: str) -> list[dict]:
    """返回 [{"name":..., "args": {...}}]；解析不到则返回 []。"""
    obj = _extract_json(text)
    if not isinstance(obj, dict):
        return []
    calls = []
    # OpenAI 风格
    for tc in obj.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": fn.get("name", ""), "args": args})
    # v1-1 agentic DSL
    for c in obj.get("calls", []) or []:
        if isinstance(c, dict) and "name" in c:
            calls.append({"name": c["name"], "args": c.get("args", {})})
    return [c for c in calls if c["name"]]


# ---------------------------------------------------------------------------
# 3. agent 循环
# ---------------------------------------------------------------------------
# 【F-b08-tool_agent.tool_message｜函数】构造工具结果消息
def tool_message(call: dict, result) -> dict:
    return {"role": "tool", "name": call["name"], "content": json.dumps(result, ensure_ascii=False)}


# 【F-b08-tool_agent.ScriptedPolicy｜类】演示用脚本化模型
class ScriptedPolicy:
    """演示用“模型”：第 1 步检索，第 2 步读文件，第 3 步给答案，之后结束。"""

    def __call__(self, messages) -> str:
        step = sum(1 for m in messages if m["role"] == "tool")
        if step == 0:
            return '{"calls": [{"name": "leanSearch", "args": {"query": "sq_connected"}}]}'
        if step == 1:
            return '{"calls": [{"name": "fileSearch", "args": {"path": "Mathlib/Algebra/Ring/Basic.lean"}}]}'
        return "final: use `exact sq_connected` then `linarith`."


# 【F-b08-tool_agent.agent_loop｜函数】生成→解析→执行→回灌循环
def agent_loop(policy, task: str, max_steps: int = 6):
    messages = [
        {"role": "system", "content": "You are a Lean proof agent. Emit {\"calls\":[...]} to use tools."},
        {"role": "user", "content": task},
    ]
    evidence: list[Evidence] = []
    trace: list[dict] = []
    for step in range(max_steps):
        out = policy(messages)
        calls = parse_tool_calls(out)
        messages.append({"role": "assistant", "content": out})
        if not calls:
            return out, evidence, trace
        for c in calls:
            result = run_tool(c["name"], c["args"], trace)
            messages.append(tool_message(c, result))
            payload = json.dumps(result, ensure_ascii=False)
            evidence.append(Evidence(
                id=f"e{len(evidence)}", kind=c["name"], weight=0.25,
                payload=payload[:200], sourceDesc="local-tool"))
    return messages[-1].get("content", ""), evidence, trace


# 【F-b08-tool_agent.main｜函数】演示入口
def main():
    task = "Prove: a^2 + 1 = b^2 + 1 -> a = b or a = -b"
    answer, evidence, trace = agent_loop(ScriptedPolicy(), task)
    print("task:", task)
    for t in trace:
        print(f"  tool {t['tool']}({t['args']}) -> {str(t['result'])[:80]}")
    print("final answer:", answer)
    print("evidence:", [(e.kind, e.weight) for e in evidence])
    print("phi = tanh(sum w) =", round(evidence_composite(evidence), 4))


if __name__ == "__main__":
    main()
