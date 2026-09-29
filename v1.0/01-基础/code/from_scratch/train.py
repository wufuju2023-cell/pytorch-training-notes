# 【源代码｜F-b01-train】v1.0/01-基础/code/from_scratch/train.py — tiny GPT 训练脚本（AMP/累积/断点续训）
# 相关文档：《01-基础/05-训练循环与数据管线.md》
"""tiny GPT 训练脚本（纯 PyTorch）：AMP / 梯度累积 / warmup+cosine / 断点续训。

用法
----
CPU 快速跑通（micro，几十秒）：:

    python3 train.py --config micro --max-iters 200 --batch-size 32

GPU 上开 AMP + 梯度累积（等效大 batch）：:

    python3 train.py --config tiny --device cuda --amp --batch-size 16 --grad-accum 4

从 checkpoint 续训：:

    python3 train.py --resume out/micro/last.pt

对照 AlphaProof / nanoproof
--------------------------
* 学习率形状 ``get_lr_multiplier``（linear warmup + flat + warmdown）见
  ``nanoproof/nanoproof/common.py:86``；本脚本用等价的 warmup + cosine。
* 训练循环、梯度累积、checkpoint 思想对应 ``nanoproof/nanoproof/pretrain.py``。
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext

import torch

from configs import get_config
from data import CharDataset, get_batch, prepare
from model import GPT


# 【F-b01-train.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny GPT 训练")
    p.add_argument("--config", default="micro", choices=["micro", "tiny"])
    p.add_argument("--text-file", default=None, help="本地语料路径（缺省用内置/下载）")
    p.add_argument("--download", action="store_true", help="尝试下载 tiny-shakespeare")
    p.add_argument("--block-size", type=int, default=None, help="缺省取 config.sequence_len")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--max-iters", type=int, default=300)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true", help="用 bf16/fp16 自动混合精度（CUDA 才有意义）")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="out")
    p.add_argument("--resume", default=None, help="checkpoint .pt 路径")
    p.add_argument("--num-threads", type=int, default=0)
    return p.parse_args()


# 【F-b01-train.resolve_device｜函数】选择运行设备
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# 【F-b01-train.lr_at｜函数】warmup + cosine 学习率
def lr_at(step: int, args: argparse.Namespace) -> float:
    """warmup + cosine 退火到 ``min_lr``。"""
    if step < args.warmup_iters:
        return args.lr * (step + 1) / max(1, args.warmup_iters)
    if step >= args.max_iters:
        return args.min_lr
    progress = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + coeff * (args.lr - args.min_lr)


# 【F-b01-train.estimate_loss｜函数】估计 train/val loss
@torch.no_grad()
def estimate_loss(model: GPT, dataset: CharDataset, args, device, ctx) -> float:
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for i in range(args.eval_iters):
        x, y = get_batch(dataset, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


# 【F-b01-train.main｜函数】训练主循环
def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    cfg = get_config(args.config)
    if args.block_size:
        cfg.sequence_len = args.block_size

    # ---- 数据 ----
    tokenizer, train_ds, val_ds, train_ids, val_ids = prepare(
        text_path=args.text_file,
        download=args.download,
        tokenizer_path=os.path.join(args.out_dir, args.config, "tokenizer.json"),
        block_size=cfg.sequence_len,
    )
    cfg.vocab_size = tokenizer.vocab_size
    print(f"[数据] vocab={tokenizer.vocab_size} train={len(train_ids)} val={len(val_ids)} block={cfg.sequence_len}")

    # ---- 模型 ----
    model = GPT(cfg).to(device)
    raw_model = model
    print(f"[模型] 参数 {model.num_params()/1e6:.3f}M（{sum(p.numel() for p in model.parameters())/1e6:.3f}M 含共享）")

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type=device.type,
    )

    # ---- AMP ----
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    )
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    print(f"[AMP] enabled={use_amp} dtype={amp_dtype if use_amp else 'fp32'}")

    # ---- 断点续训 ----
    start_iter, best_val = 0, float("inf")
    ckpt_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[续训] 从 {args.resume} 的第 {start_iter} 步继续")

    tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))

    # ---- 训练循环 ----
    model.train()
    t0 = time.time()
    tokens_seen = 0
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_ds, args.batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if it % args.log_interval == 0 or it == args.max_iters - 1:
            dt = time.time() - t0
            print(
                f"iter {it:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                f"{tokens_seen / max(dt,1e-9):8.0f} tok/s"
            )

        if (it + 1) % args.eval_interval == 0 or it == args.max_iters - 1:
            val = estimate_loss(model, val_ds, args, device, amp_ctx)
            print(f"  [eval] iter {it} val_loss {val:.4f}")
            if val < best_val:
                best_val = val
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "config": cfg.to_dict(),
                        "tokenizer": tokenizer.itos,
                        "iter": it,
                        "best_val": best_val,
                    },
                    os.path.join(ckpt_dir, "best.pt"),
                )

        # 每步滚动保存 last.pt，便于随时续训
        if (it + 1) % args.eval_interval == 0 or it == args.max_iters - 1:
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                    "config": cfg.to_dict(),
                    "tokenizer": tokenizer.itos,
                    "iter": it,
                    "best_val": best_val,
                },
                os.path.join(ckpt_dir, "last.pt"),
            )

    print(f"[完成] 用时 {time.time()-t0:.1f}s，best val_loss={best_val:.4f}，checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
