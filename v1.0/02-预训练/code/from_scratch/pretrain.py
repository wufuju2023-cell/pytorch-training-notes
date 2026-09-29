# 【源代码｜F-b02-pretrain】v1.0/02-预训练/code/from_scratch/pretrain.py — 从零预训练 next-token 语言模型
# 相关文档：《02-预训练/01-预训练原理.md》
r"""从零预训练：在 tiny 语料上做 next-token 语言建模（纯 PyTorch 教学版）。

复用 found-code 在 ``01-基础/code/from_scratch`` 给出的 tinyGPT 接口
（``GPT`` / ``get_config`` / ``CharTokenizer`` / ``get_batch`` / ``CharDataset``），
把它挂到 ``sys.path`` 保证自身独立可跑（须与 ``01-基础/`` 同一棵 v1.0 目录树）。

要点
----
自回归语言模型按链式法则分解 $p_\theta(x)=\\prod_t p_\\theta(x_t\\mid x_{<t})$，
最小化负对数似然（= 与真实分布的交叉熵）。教师强制：预测第 t 个 token 时喂真实前缀。
相对 ``01-基础/train.py`` 多讲：多域配比与课程、target-flops 预算、
DDP/混合精度/梯度累积、以及 midtrain（继续预训练）。

用法::

    python3 pretrain.py --config micro --max-iters 300 --batch-size 32 --domains "math:1"
    torchrun --nproc_per_node=2 pretrain.py --config tiny --device cuda --amp --param-data-ratio 20
    python3 pretrain.py --config tiny --target-flops 1e15 --domains math:1,code:1
    python3 pretrain.py --config micro --resume out_pretrain/micro/last.pt --domains code:1

对照 AlphaProof / nanoproof
--------------------------
* nanoproof/pretrain.py：Nemotron-CC-Math(~20B tok)，DDP + MuonAdamW + target_flops。
* nanoproof/midtrain.py：从 ckpt 继续在 Lean-GitHub(~65M tok) midtrain。
* Muon 优化器见 nanoproof/optim.py；教学版统一用 AdamW。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext


# 【F-b02-pretrain._bootstrap_tiny_gpt｜函数】把 01-基础/code/from_scratch 加入 sys.path
def _bootstrap_tiny_gpt() -> str:
    """把 01-基础/code/from_scratch 加进 sys.path（相对本文件 ../../../）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(here, "..", "..", "..", "01-基础", "code", "from_scratch"))
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
    return candidate


_TINY_DIR = _bootstrap_tiny_gpt()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from configs import get_config  # noqa: E402
from data import CharDataset, CharTokenizer, get_batch, split_train_val  # noqa: E402
from model import GPT  # noqa: E402

# ---------------------------------------------------------------------------
# 1. 多域语料：配比 + 课程
# ---------------------------------------------------------------------------
DOMAINS: dict[str, str] = {
    "web": """
Mathematical language modeling trains a model to predict the next token in a
sequence of equations and explanations. The cross entropy objective rewards
assigning high probability to the observed continuation. A lower perplexity
means the model is less surprised by held-out mathematics. We mix web and code
so that syntactic fluency transfers to formal proof generation.
""".strip(),
    "math": """
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
""".strip(),
    "code": """
import Mathlib
open Function Set

namespace Nat

lemma succ_add (a b : Nat) : (a + 1) + b = (a + b) + 1 := by
  rw [Nat.add_assoc, Nat.add_comm 1 b, ← Nat.add_assoc]

lemma mul_succ (a b : Nat) : a * (b + 1) = a * b + a := by
  rw [Nat.mul_add, Nat.mul_one]

end Nat

theorem List.append_nil (xs : List α) : xs ++ [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [ih]
""".strip(),
}


# 【F-b02-pretrain.parse_domains｜函数】解析 "math:3,code:1" 域配比
def parse_domains(spec: str) -> dict[str, int]:
    """把 "math:3,code:1,web:1" 解析成 {"math":3,"code":1,"web":1}。"""
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        if name not in DOMAINS:
            raise ValueError(f"未知域 {name!r}，可选：{list(DOMAINS)}")
        out[name] = int(w) if w else 1
    if not out:
        raise ValueError("--domains 不能为空")
    return out


# 【F-b02-pretrain.build_corpus｜函数】按配比重复拼接各域文本
def build_corpus(domains: dict[str, int], reps: int = 8) -> str:
    """按配比重复拼接各域文本（字符级占比近似配比）。"""
    total_w = sum(domains.values())
    chunks: list[str] = []
    for name, w in domains.items():
        n = max(1, round(reps * w / total_w))
        chunks.extend([DOMAINS[name]] * n)
    return "\n\n".join(chunks)


# 【F-b02-pretrain.domain_schedule｜函数】课程式配比线性插值
def domain_schedule(step: int, total: int, warm: dict[str, int], final: dict[str, int]) -> dict[str, int]:
    """课程：把两套配比按训练进度线性插值（web/code -> math）。"""
    t = min(1.0, step / max(1, total - 1))
    names = set(warm) | set(final)
    out: dict[str, int] = {}
    for name in names:
        w0, w1 = warm.get(name, 0), final.get(name, 0)
        out[name] = max(1, round((1 - t) * w0 + t * w1))
    return out


# ---------------------------------------------------------------------------
# 2. 算力 / 数据预算
# ---------------------------------------------------------------------------
# 【F-b02-pretrain.tokens_from_target_flops｜函数】由 C≈6ND 反解 token 数
def tokens_from_target_flops(target_flops: float, n_params: int) -> float:
    """由 C≈6ND 反解 token 数 D = C / (6N)。"""
    return target_flops / (6.0 * max(1, n_params))


# 【F-b02-pretrain.planned_tokens｜函数】token 预算优先级
def planned_tokens(args: argparse.Namespace, n_params: int) -> float:
    """token 预算优先级：--total-tokens > --target-flops > --param-data-ratio。"""
    if args.total_tokens > 0:
        return args.total_tokens
    if args.target_flops > 0:
        return tokens_from_target_flops(args.target_flops, n_params)
    return args.param_data_ratio * n_params


# ---------------------------------------------------------------------------
# 3. 分布式 & 精度
# ---------------------------------------------------------------------------
# 【F-b02-pretrain.is_dist｜函数】是否处于 torch.distributed
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


# 【F-b02-pretrain.is_master｜函数】是否主进程
def is_master() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


# 【F-b02-pretrain.log｜函数】只在主进程打印
def log(*a) -> None:
    if is_master():
        print(*a, flush=True)


# 【F-b02-pretrain.setup_dist｜函数】初始化 DDP
def setup_dist() -> tuple[int, int, torch.device]:
    if not is_dist():
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return 0, 1, torch.device(dev)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


# 【F-b02-pretrain.build_amp｜函数】构建混合精度上下文
def build_amp(use_amp: bool, device: torch.device):
    enabled = use_amp and device.type == "cuda"
    dtype = torch.bfloat16 if (enabled and torch.cuda.is_bf16_supported()) else torch.float16
    ctx = torch.autocast(device_type=device.type, dtype=dtype) if enabled else nullcontext()
    scaler = torch.amp.GradScaler(enabled=enabled and dtype == torch.float16)
    return ctx, scaler


# 【F-b02-pretrain.lr_at｜函数】warmup + cosine 学习率
def lr_at(step: int, args: argparse.Namespace, total: int) -> float:
    if step < args.warmup_iters:
        return args.lr * (step + 1) / max(1, args.warmup_iters)
    prog = min(1.0, max(0.0, (step - args.warmup_iters) / max(1, total - args.warmup_iters)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return args.min_lr + coeff * (args.lr - args.min_lr)


# 【F-b02-pretrain.estimate_loss｜函数】估计 train/val loss
@torch.no_grad()
def estimate_loss(model, dataset, args, device, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(dataset, args.batch_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
# 【F-b02-pretrain.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny 语料预训练（教学版）")
    p.add_argument("--config", default="micro", choices=["micro", "tiny"])
    p.add_argument("--domains", default="math:3,code:1,web:1")
    p.add_argument("--final-domains", default=None, help="课程终点配比；缺省不做课程")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--max-iters", type=int, default=400)
    p.add_argument("--total-tokens", type=float, default=0.0)
    p.add_argument("--target-flops", type=float, default=0.0, help="算力预算 C，按 C≈6ND 反解")
    p.add_argument("--param-data-ratio", type=float, default=0.0, help="token 预算 = ratio*参数量")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="out_pretrain")
    p.add_argument("--resume", default=None, help="midtrain：从预训练 ckpt 继续")
    p.add_argument("--num-threads", type=int, default=0)
    return p.parse_args()


# 【F-b02-pretrain.main｜函数】预训练主循环
def main() -> None:
    args = parse_args()
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)

    rank, world, device = setup_dist()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    warm_spec = parse_domains(args.domains)
    final_spec = parse_domains(args.final_domains) if args.final_domains else warm_spec

    text = build_corpus(warm_spec, reps=args.reps)
    tokenizer = CharTokenizer.from_text(text)
    ids = torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, val_frac=0.1)

    cfg = get_config(args.config)
    cfg.vocab_size = tokenizer.vocab_size
    train_ds = CharDataset(train_ids, cfg.sequence_len)
    val_ds = CharDataset(val_ids, cfg.sequence_len)

    model = GPT(cfg).to(device)
    n_params = model.num_params()
    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    planned = planned_tokens(args, n_params)
    tokens_per_step = args.batch_size * cfg.sequence_len * args.grad_accum * world
    iters_from_budget = int(planned // max(1, tokens_per_step)) if planned > 0 else 0
    total_iters = max(args.max_iters, iters_from_budget)
    log(f"[预算] params={n_params/1e6:.3f}M 计划tokens={planned:,.0f} "
        f"(target_flops={args.target_flops:g}, ratio={args.param_data_ratio:g}) "
        f"每步tokens={tokens_per_step} -> iters={total_iters}")
    log(f"[数据] vocab={tokenizer.vocab_size} train={len(train_ids)} val={len(val_ids)} "
        f"domains={warm_spec} -> {final_spec}")
    log(f"[模型] 参数 {n_params/1e6:.3f}M world={world} device={device}")

    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr,
        betas=(0.9, 0.95), device_type=device.type)
    amp_ctx, scaler = build_amp(args.amp, device)

    start_iter, best_val = 0, float("inf")
    ckpt_dir = os.path.join(args.out_dir, args.config)
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", -1) + 1
        best_val = ckpt.get("best_val", float("inf"))
        log(f"[midtrain] 从 {args.resume} 的第 {start_iter} 步继续，切换到 {final_spec}")

    if is_master():
        tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))

    model.train()
    t0 = time.time()
    tokens_seen = 0
    for it in range(start_iter, total_iters):
        lr = lr_at(it, args, total_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_ds, args.batch_size, device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += loss.item()
            tokens_seen += x.numel()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if it % args.log_interval == 0 or it == total_iters - 1:
            dt = max(time.time() - t0, 1e-9)
            cur = domain_schedule(it, total_iters, warm_spec, final_spec)
            log(f"iter {it:5d} | loss {accum:.4f} | lr {lr:.2e} | "
                f"{tokens_seen/dt:8.0f} tok/s | domains {cur}")

        if (it + 1) % args.eval_interval == 0 or it == total_iters - 1:
            val = estimate_loss(model, val_ds, args, device, amp_ctx)
            log(f"  [eval] iter {it} val_loss {val:.4f} ppl {math.exp(min(val, 20)):.2f}")
            if is_master() and val < best_val:
                best_val = val
                torch.save({"model": raw_model.state_dict(), "config": cfg.to_dict(),
                            "iter": it, "best_val": best_val},
                           os.path.join(ckpt_dir, "best.pt"))
            if is_master():
                torch.save({"model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict() if scaler.is_enabled() else {},
                            "config": cfg.to_dict(), "iter": it, "best_val": best_val},
                           os.path.join(ckpt_dir, "last.pt"))

    log(f"[完成] {time.time()-t0:.1f}s best_val={best_val:.4f} -> {ckpt_dir}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
