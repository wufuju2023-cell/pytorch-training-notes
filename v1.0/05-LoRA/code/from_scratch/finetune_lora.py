#!/usr/bin/env python3
# 【源代码｜F-b05-finetune_lora】v1.0/05-LoRA/code/from_scratch/finetune_lora.py — 冻结/全参/LoRA 三种微调对比
# 相关文档：《05-LoRA/01-LoRA原理.md》
"""对比"冻结 / 全参 / LoRA"三种微调：参数量、显存与效果。

任务：合成一个小型序列复制任务（把输入 token 的逆序作为标签），
在有 GPU 时报告峰值显存；CPU 也能跑（显存列显示为 None）。

对照源码：``app/policy_server.py`` 只更新 LoRA adapter；全参微调对应直接
对 ``model.parameters()`` 求梯度。

用法：
    python finetune_lora.py --mode all --steps 100
    python finetune_lora.py --mode lora --r 8 --alpha 16
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn
from torch.nn import functional as F

from lora import (TARGET_MODULES, TinyGPT, count_parameters, inject_lora,
                  lora_parameters)


# 【F-b05-finetune_lora.make_data｜函数】合成序列逆序复制任务
def make_data(n: int, seq_len: int, vocab: int, seed: int = 0):
    """合成任务：标签是输入序列的逆序（要求模型学会重排）。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (n, seq_len), generator=g)
    y = torch.flip(x, dims=[1])
    return x, y


# 【F-b05-finetune_lora.train｜函数】按模式训练并统计参数量/显存
def train(model, x, y, steps: int, lr: float, mode: str, device) -> dict:
    params = lora_parameters(model) if mode == "lora" else \
        [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last = float("nan")
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, x.shape[0], (32,))
        xb, yb = x[idx].to(device), y[idx].to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if params:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        last = float(loss)
    peak = (torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else None)
    return {"mode": mode, "final_loss": round(last, 4),
            "seconds": round(time.time() - t0, 2),
            "peak_mem_MB": round(peak, 1) if peak else None}


# 【F-b05-finetune_lora.main｜函数】对比实验入口
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("frozen", "full", "lora", "all"), default="all")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    x, y = make_data(args.n, args.seq_len, args.vocab)
    base_state = TinyGPT(vocab_size=args.vocab, max_len=args.seq_len).state_dict()

    modes = ["frozen", "full", "lora"] if args.mode == "all" else [args.mode]
    for mode in modes:
        torch.manual_seed(0)
        model = TinyGPT(vocab_size=args.vocab, max_len=args.seq_len).to(device)
        if mode == "lora":
            inject_lora(model, TARGET_MODULES, r=args.r, alpha=args.alpha)
            for n_, p in model.named_parameters():
                p.requires_grad_("lora_" in n_)
        elif mode == "frozen":
            for p in model.parameters():
                p.requires_grad_(False)

        total = count_parameters(model, trainable_only=False)
        trainable = count_parameters(model, trainable_only=True)
        result = train(model, x, y, args.steps, args.lr, mode, device)
        result.update({"total_params": total, "trainable_params": trainable,
                       "trainable_pct": round(100 * trainable / total, 3)})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
