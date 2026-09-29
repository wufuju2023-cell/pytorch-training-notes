#!/usr/bin/env python3
# 【源代码｜F-b04-train_value_head】v1.0/04-价值头/code/from_scratch/train_value_head.py — 从零训练价值头（scalar/64-bin）
# 相关文档：《04-价值头/01-价值头原理.md》
"""从零训练价值头：scalar（MSE/Huber）与 64-bin（two-hot 交叉熵）。

与 ``app/train_value_head.py`` 的区别：这里不加载真实 backbone，而是读取
缓存好的 hidden-state 特征分片（每行 ``{"feature":[...],"proof_depth":d}`` 或
``{"hidden":[...],"value_target":v}``），没有数据时回退到合成特征，CPU 秒级跑完。

示例：
    python train_value_head.py --mode both --steps 300
    python train_value_head.py --mode categorical --data feat.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from value_head import (
    CategoricalValueHead,
    ScalarValueHead,
    expected_distance,
    proof_depth_to_target,
    regression_reliability,
    synthetic_features,
    two_hot_cross_entropy,
)


# 【F-b04-train_value_head.load_shard｜函数】读缓存特征分片
def load_shard(path: str, max_rows: int):
    """读缓存特征分片，返回 (features[B,H], depth[B], hidden_size)。"""
    feats, depths = [], []
    hidden_size = None
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vec = rec.get("feature", rec.get("hidden"))
            if vec is None:
                continue
            hidden_size = len(vec)
            feats.append(vec)
            if "proof_depth" in rec:
                depths.append(float(rec["proof_depth"]))
            elif "value_target" in rec:
                depths.append(-float(rec["value_target"]))
            else:
                raise ValueError("record needs proof_depth or value_target")
            if max_rows and len(feats) >= max_rows:
                break
    if not feats:
        raise SystemExit(f"no usable rows in {path}")
    return (torch.tensor(feats, dtype=torch.float32),
            torch.tensor(depths, dtype=torch.float32), hidden_size)


# 【F-b04-train_value_head.make_data｜函数】无数据时合成特征
def make_data(args):
    if args.data:
        return load_shard(args.data, args.max_rows)
    return synthetic_features(args.n, args.hidden_size, seed=args.seed)


# 【F-b04-train_value_head.train_scalar｜函数】训练 MSE/Huber 回归头
def train_scalar(x, depths, args):
    y = torch.tensor([proof_depth_to_target(d, args.max_depth) for d in depths.tolist()])
    head = ScalarValueHead(x.shape[1], args.hidden_dim)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss() if args.loss == "mse" else torch.nn.SmoothL1Loss()
    n = x.shape[0]
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (min(args.batch_size, n),))
        pred = head(x[idx])
        loss = loss_fn(pred, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            with torch.no_grad():
                allp = head(x)
                rel = regression_reliability(allp, y, n_bins=10)
                print(json.dumps({
                    "head": "scalar", "step": step, "loss": round(float(loss), 5),
                    "mae": round(float((allp - y).abs().mean()), 4),
                    "ece": round(rel["ece"], 4),
                }, ensure_ascii=False), flush=True)
    return head, y


# 【F-b04-train_value_head.train_categorical｜函数】训练 64-bin two-hot 分类头
def train_categorical(x, depths, args):
    head = CategoricalValueHead(x.shape[1], args.hidden_dim, args.num_bins)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    n = x.shape[0]
    dist = torch.clamp(depths, min=1.0)   # 终止 d=0 记为 1
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (min(args.batch_size, n),))
        logits = head(x[idx])
        loss = two_hot_cross_entropy(logits, dist[idx], args.num_bins)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            with torch.no_grad():
                pred = expected_distance(head(x), args.num_bins)
                mae = (pred - dist).abs().mean()
                print(json.dumps({
                    "head": "categorical", "step": step,
                    "loss": round(float(loss), 5),
                    "mean_expected_distance": round(float(pred.mean()), 3),
                    "mae_distance": round(float(mae), 4),
                }, ensure_ascii=False), flush=True)
    return head, dist


# 【F-b04-train_value_head.parse_args｜函数】解析命令行参数
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("scalar", "categorical", "both"), default="both")
    p.add_argument("--data", default=None, help="feature shard JSONL (可选)")
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--hidden-size", type=int, default=3584)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-bins", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=("mse", "huber"), default="huber")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="保存 head.state_dict 的路径前缀")
    return p.parse_args()


# 【F-b04-train_value_head.main｜函数】训练入口
def main() -> int:
    args = parse_args()
    x, depths, hidden_size = make_data(args)
    if hidden_size and hidden_size != args.hidden_size:
        print(json.dumps({"note": f"hidden_size={hidden_size} (覆盖参数)"}))
        args.hidden_size = hidden_size
    print(json.dumps({
        "rows": int(x.shape[0]), "hidden_size": args.hidden_size,
        "depth_mean": round(float(depths.float().mean()), 3),
    }, ensure_ascii=False), flush=True)

    if args.mode in ("scalar", "both"):
        head, _ = train_scalar(x, depths, args)
        if args.out:
            torch.save(head.state_dict(), args.out + ".scalar.pt")
    if args.mode in ("categorical", "both"):
        head, _ = train_categorical(x, depths, args)
        if args.out:
            torch.save(head.state_dict(), args.out + ".cat.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
