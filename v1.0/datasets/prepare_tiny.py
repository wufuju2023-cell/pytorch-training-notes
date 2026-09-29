#!/usr/bin/env python3
# 【源代码｜F-ds-prepare_tiny】v1.0/datasets/prepare_tiny.py — 教学用小切片数据准备器
# 相关文档：《datasets/README.md》
"""教学用小切片数据准备器（可运行、可离线、网络不可用时优雅降级为造数据）。

产物（写入 ``--out``，默认 ``./_tiny``）：

* ``local_corpus.txt``          自造"math/code/web"混合小语料（预训练用）
* ``minif2f.jsonl``             miniF2F 子集（id, formal_statement）
* ``fatem-100.jsonl``           eval set 切片复制（或合成）
* ``holdout-30.jsonl``          eval set 切片复制（或合成）
* ``state_tactic_pairs.jsonl``  (state, tactic) 前 N 条（或合成）
* ``leantree.jsonl``            LeanTree 前 N 条（或合成）
* ``value_head_features.md``    价值头特征分片说明（只给 schema，不复制大 tensor）
* ``manifest.json``             每个产物的来源/行数/schema/license

获取优先级：本地参考路径 -> HuggingFace datasets 前 N 条 -> 合成。
参考路径：
* eval sets：``/mnt/gloway/projects/reap-new-update-model/v1-spec/v1-1-training-example/runs/eval_sets/``
* 备份：``/mnt/gloway/projects/9-27-hsy-modelscope-备份/mnt-workspace/new_value_head/``
* mathlib4：``/mnt/gloway/projects/lean-corpus/mathlib4``

用法::

    python3 prepare_tiny.py --out ./_tiny --max-rows 200
    python3 prepare_tiny.py --out ./_tiny --offline        # 只用本地/合成
"""

from __future__ import annotations

import argparse
import json
import os
import random

EVAL_SETS_DEFAULT = ("/mnt/gloway/projects/reap-new-update-model/v1-spec/"
                     "v1-1-training-example/runs/eval_sets")
BACKUP_DATA = ("/mnt/gloway/projects/9-27-hsy-modelscope-备份/mnt-workspace/"
               "new_value_head/data/leantree_mathlib.jsonl")
MATHLIB4 = "/mnt/gloway/projects/lean-corpus/mathlib4"

_STATES = [
    ("⊢ a + b = b + a", "rw [Nat.add_comm]"),
    ("⊢ a * b = b * a", "rw [Nat.mul_comm]"),
    ("h : a + b = b + a ⊢ a + b = b + a", "exact h"),
    ("⊢ (0 : Nat) + n = n", "simp"),
    ("⊢ (a + b) + c = a + (b + c)", "rw [Nat.add_assoc]"),
    ("⊢ n * 1 = n", "simp"),
    ("h : P ∧ Q ⊢ P", "exact h.1"),
    ("⊢ Nat.Prime 2", "norm_num"),
]

_CORPUS = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_add, Nat.add_succ, ih]
theorem mul_comm (a b : Nat) : a * b = b * a := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_mul, Nat.mul_succ, ih]
theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_add, ih]
example (n : Nat) : 0 + n = n := by simp
example (n : Nat) : n * 1 = n := by simp
theorem map_map (f : α → β) (g : β → γ) (xs : List α) :
    (xs.map f).map g = xs.map (g ∘ f) := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [ih]
import Mathlib
lemma succ_add (a b : Nat) : (a + 1) + b = (a + b) + 1 := by
  rw [Nat.add_assoc, Nat.add_comm 1 b, ← Nat.add_assoc]
""".strip()


# 【F-ds-prepare_tiny.log｜函数】带前缀日志
def log(*a) -> None:
    print(*a, flush=True)


# 【F-ds-prepare_tiny.ensure_dir｜函数】确保目录存在
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# 【F-ds-prepare_tiny.atomic_write_jsonl｜函数】原子写入 jsonl
def atomic_write_jsonl(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# 【F-ds-prepare_tiny.atomic_write_text｜函数】原子写入文本
def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# 【F-ds-prepare_tiny.head_jsonl｜函数】取 jsonl 前 N 行
def head_jsonl(path: str, n: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= n:
                break
    return rows


# 【F-ds-prepare_tiny.synth_pairs｜函数】合成 (state,tactic) 样本
def synth_pairs(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [{"state": s, "tactic": t} for s, t in (rng.choice(_STATES) for _ in range(n))]


# 【F-ds-prepare_tiny.synth_statements｜函数】合成形式化陈述
def synth_statements(n: int, prefix: str, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.choice("abcdefghn"), rng.choice("abcdefghn")
        out.append({"id": f"{prefix}-{i}",
                    "formal_statement": f"import Mathlib\n\nexample ({a} {b} : Nat) : "
                                        f"{a} + {b} = {b} + {a} := by sorry"})
    return out


# 【F-ds-prepare_tiny.load_local｜函数】读取本地数据切片
def load_local(path: str, n: int, kind: str) -> list[dict] | None:
    if not path or not os.path.exists(path):
        return None
    try:
        raw = head_jsonl(path, n) if path.endswith(".jsonl") else None
    except Exception as e:  # noqa: BLE001
        log(f"[本地] 读取 {path} 失败：{e}")
        return None
    if not raw:
        return None
    out = []
    for r in raw:
        if kind == "state_tactic":
            state = r.get("state") or r.get("proof_state") or r.get("public_proof_state")
            tactic = r.get("tactic") or r.get("next_tactic")
            if not tactic and isinstance(r.get("trajectory"), list) and r["trajectory"]:
                tactic = r["trajectory"][0]
            if state and tactic:
                out.append({"state": str(state), "tactic": str(tactic)})
        elif kind == "leantree":
            state = r.get("state") or r.get("proof_state")
            tactic = r.get("tactic") or r.get("next_tactic")
            traj = r.get("trajectory") if isinstance(r.get("trajectory"), list) else []
            if state and (tactic or traj):
                out.append({"state": str(state), "tactic": str(tactic or (traj[0] if traj else "")),
                            "trajectory": traj})
        elif kind == "statements":
            stmt = r.get("formal_statement") or r.get("statement") or r.get("goal")
            if stmt:
                out.append({"id": str(r.get("id", len(out))), "formal_statement": str(stmt)})
    return out or None


# 【F-ds-prepare_tiny.load_hf｜函数】从 HF 拉取数据切片
def load_hf(hf_id: str, n: int, offline: bool) -> list[dict] | None:
    if offline:
        log(f"[HF] offline，跳过 {hf_id}")
        return None
    try:
        from datasets import load_dataset
    except ImportError:
        log("[HF] 未安装 datasets，跳过")
        return None
    try:
        ds = load_dataset(hf_id, split="train", streaming=True)
        rows = []
        for r in ds:
            rows.append(dict(r))
            if len(rows) >= n:
                break
        log(f"[HF] {hf_id} 取到 {len(rows)} 条")
        return rows or None
    except Exception as e:  # noqa: BLE001
        log(f"[HF] {hf_id} 不可用（{type(e).__name__}: {e}），降级")
        return None


# 【F-ds-prepare_tiny.normalize_pairs｜函数】归一化 (state,tactic) 字段
def normalize_pairs(rows: list[dict], n: int) -> list[dict]:
    out = []
    for r in rows[:n]:
        state = r.get("state") or r.get("proof_state") or r.get("public_proof_state")
        tactic = r.get("tactic") or r.get("next_tactic")
        if not tactic and isinstance(r.get("trajectory"), list) and r["trajectory"]:
            tactic = r["trajectory"][0]
        if state and tactic:
            out.append({"state": str(state), "tactic": str(tactic)})
    return out


# 【F-ds-prepare_tiny.prepare_corpus｜函数】准备预训练小语料
def prepare_corpus(out: str) -> dict:
    text = _CORPUS
    source = "synthetic"
    extra = []
    if os.path.isdir(MATHLIB4):
        for root, _dirs, files in os.walk(MATHLIB4):
            for fn in files:
                if fn.endswith(".lean"):
                    try:
                        with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                            extra.append(f.read(4000))
                    except Exception:  # noqa: BLE001
                        pass
                    break
            if len(extra) >= 5:
                break
    if extra:
        text = "\n\n".join(extra) + "\n\n" + _CORPUS
        source = "mathlib4 切片 + 合成"
    atomic_write_text(os.path.join(out, "local_corpus.txt"), text)
    return {"file": "local_corpus.txt", "source": source, "rows": len(text),
            "schema": "plain text (chars)", "license": "mathlib4: Apache-2.0"}


# 【F-ds-prepare_tiny.prepare_statements｜函数】准备 miniF2F 子集
def prepare_statements(out: str, n: int, offline: bool, eval_sets: str) -> list[dict]:
    files = []
    for name in ("fatem-100.jsonl", "holdout-30.jsonl"):
        path = os.path.join(eval_sets, name)
        rows = load_local(path, n, "statements")
        if rows:
            atomic_write_jsonl(os.path.join(out, name), rows)
            files.append({"file": name, "source": f"local:{path}", "rows": len(rows),
                          "schema": ["id", "formal_statement"], "license": "见上游仓库",
                          "note": "eval set 切片，仅评测不训练"})
        else:
            rows = synth_statements(min(n, 30), name.split(".")[0].split("-")[0])
            atomic_write_jsonl(os.path.join(out, name), rows)
            files.append({"file": name, "source": "synthetic", "rows": len(rows),
                          "schema": ["id", "formal_statement"], "license": "CC0",
                          "note": "本地 eval set 不可用，已合成"})
    rows = (load_hf("AI-MO/NuminaMath-LEAN", min(n, 100), offline)
            or synth_statements(min(n, 40), "minif2f"))
    norm = []
    for r in rows:
        stmt = r.get("formal_statement") or r.get("problem") or r.get("statement") or r.get("goal")
        if stmt:
            norm.append({"id": str(r.get("id", len(norm))), "formal_statement": str(stmt)})
    if not norm:
        norm = synth_statements(min(n, 40), "minif2f")
    atomic_write_jsonl(os.path.join(out, "minif2f.jsonl"), norm)
    files.append({"file": "minif2f.jsonl", "source": "HF/local/synthetic", "rows": len(norm),
                  "schema": ["id", "formal_statement"], "license": "见上游",
                  "note": "miniF2F valid/test 子集切片"})
    return files


# 【F-ds-prepare_tiny.prepare_state_tactic｜函数】抽取 state_tactic_pairs 前 N 条
def prepare_state_tactic(out: str, n: int, offline: bool) -> dict:
    rows = (load_local(BACKUP_DATA.replace("leantree_mathlib.jsonl", "dataset/train.jsonl"),
                       n, "state_tactic")
            or load_hf("FrenzyMath/state_tactic_pairs", n, offline))
    if rows:
        pairs = normalize_pairs(rows, n)
        source = "local/HF"
    else:
        pairs, source = synth_pairs(n), "synthetic"
    if not pairs:
        pairs, source = synth_pairs(n), "synthetic"
    atomic_write_jsonl(os.path.join(out, "state_tactic_pairs.jsonl"), pairs)
    return {"file": "state_tactic_pairs.jsonl", "source": source, "rows": len(pairs),
            "schema": ["state", "tactic"], "license": "FrenzyMath/state_tactic_pairs 见 HF 页",
            "note": "教学只用切片；完整约 5 万对"}


# 【F-ds-prepare_tiny.prepare_leantree｜函数】抽取 LeanTree 前 N 条
def prepare_leantree(out: str, n: int, offline: bool) -> dict:
    rows = load_local(BACKUP_DATA, n, "leantree") or load_hf("ufal/leantree", n, offline)
    if rows:
        pairs = normalize_pairs(rows, n)
        source = "local/HF"
    else:
        pairs, source = synth_pairs(n), "synthetic"
    if not pairs:
        pairs, source = synth_pairs(n), "synthetic"
    atomic_write_jsonl(os.path.join(out, "leantree.jsonl"),
                       [{"state": p["state"], "tactic": p["tactic"], "trajectory": [p["tactic"]]}
                        for p in pairs])
    return {"file": "leantree.jsonl", "source": source, "rows": len(pairs),
            "schema": ["state", "tactic", "trajectory"], "license": "ufal/leantree: Apache-2.0",
            "note": "教学只用前 N 条（完整约 26 万转移）"}


# 【F-ds-prepare_tiny.write_value_head_note｜函数】写价值头特征分片说明
def write_value_head_note(out: str) -> dict:
    text = (
        "# 价值头特征分片说明\n\n"
        "真实分片位于备份：\n"
        "`/mnt/gloway/projects/9-27-hsy-modelscope-备份/mnt-workspace/new_value_head/features-79efd240/`\n\n"
        "- `train/` 40 分片、`validation/` 4、`test/` 4；\n"
        "- 每个 `features-*.pt` 约 14.96MB / 2000 行，内容为冻结策略最后一层 hidden 特征；\n"
        "- 价值头训练脚本按 batch 读出特征，配合 `value_target` 做回归/分类。\n\n"
        "**教学不复制大文件**：本脚本只生成说明。需要时按 `04-价值头/` 的代码结构，用合成特征\n"
        "（随机张量 + proof_depth 目标）跑通流程。\n"
    )
    atomic_write_text(os.path.join(out, "value_head_features.md"), text)
    return {"file": "value_head_features.md", "source": "说明文档", "rows": 0,
            "schema": "hidden features 分片", "license": "N/A", "note": "大 tensor 不在仓库内"}


# 【F-ds-prepare_tiny.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v1.0 教学用小切片数据准备器")
    p.add_argument("--out", default="./_tiny")
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--eval-sets", default=EVAL_SETS_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# 【F-ds-prepare_tiny.main｜函数】准备入口
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out = ensure_dir(args.out)
    manifest = {"output_dir": os.path.abspath(out), "max_rows": args.max_rows,
                "offline": args.offline, "files": []}
    manifest["files"].append(prepare_corpus(out))
    manifest["files"].extend(prepare_statements(out, args.max_rows, args.offline, args.eval_sets))
    manifest["files"].append(prepare_state_tactic(out, args.max_rows, args.offline))
    manifest["files"].append(prepare_leantree(out, args.max_rows, args.offline))
    manifest["files"].append(write_value_head_note(out))
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(f"\n[完成] -> {out}")
    for it in manifest["files"]:
        log(f"  - {it['file']:32s} rows={it['rows']:>5}  source={it['source']}")


if __name__ == "__main__":
    main()
