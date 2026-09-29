# 【源代码｜F-b03-sft】v1.0/03-SFT/code/from_scratch/sft.py — token-masked (state→tactic) 监督微调
# 相关文档：《03-SFT/01-SFT原理.md》
r"""从零实现 SFT：在 tiny 模型上做 (proof state -> tactic) 的 token-masked 监督微调。

本脚本复用 ``01-基础/code/from_scratch`` 的 tinyGPT（``GPT`` / ``get_config`` /
``CharTokenizer``），但**独立闭环**：自带状态-策略样本构造、token 级 label mask、
只在答案 token 上算交叉熵、留出集评估与生成。

核心思想：只在答案上算 loss
-------------------------
一个训练样本形如::

    input  : STATE:\n<goal>\nTACTIC:\n<策略文本><eos>
    labels : [-100 ... -100]              <策略 token ids>

``-100`` 是 ``torch.nn.functional.cross_entropy(..., ignore_index=-100)`` 的默认
忽略值，也是 HF 生态的约定。交叉熵只在 ``labels != -100`` 的位置求和：

$$
\mathcal{L}_{\text{SFT}}(\theta)
=-\frac{1}{\sum_t m_t}\sum_{t} m_t \log p_\theta(y_t\mid x_{<t})
$$

其中 $m_t\in\{0,1\}$ 是 mask（prompt 位置为 0，答案位置为 1）。这样模型**不会**
去拟合"如何复述题面"这种无信息的 token，只会被推着提高生成正确策略的概率。

对照 AlphaProof / nanoproof
---------------------------
* ``app/train_sft.py``（DEPRECATED 骨架）：本意就是 ``state_tactic_pairs`` 上的
  token-masked LoRA SFT，与本文件的 mask 逻辑一致。
* 容器 ``train_full_supervised.py``（Qwen3-1.7B QLoRA，target JSON ``{"calls":[...]}``）：
  prompt 位置同样整段 -100，只在 ``<answer>{"calls":...}</answer>`` 上算 loss。
* ``nanoproof/nanoproof/sft.py``：LeanTree (~260k transitions) token-masked CE，
  value 行按 ``--value-weight`` 加权；本文件是它的字符级最小复刻。

用法
----
::

    python3 sft.py --max-iters 400 --batch-size 16            # CPU 冒烟
    python3 sft.py --states-file pairs.jsonl --max-iters 800  # 真实 (state,tactic) 切片
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import torch


# 【F-b03-sft._bootstrap_tiny_gpt｜函数】把 01-基础/code/from_scratch 加入 sys.path
def _bootstrap_tiny_gpt() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(here, "..", "..", "..", "01-基础", "code", "from_scratch"))
    if cand not in sys.path:
        sys.path.insert(0, cand)


_bootstrap_tiny_gpt()

from configs import get_config  # noqa: E402
from data import CharTokenizer  # noqa: E402
from model import GPT  # noqa: E402

IGNORE_INDEX = -100
PROMPT_PREFIX = "STATE:\n"
PROMPT_MIDDLE = "\nTACTIC:\n"


# ---------------------------------------------------------------------------
# 1. 状态-策略样本
# ---------------------------------------------------------------------------
# 【F-b03-sft.synthetic_pairs｜函数】造 (state, tactic) 样本
def synthetic_pairs(n: int = 240, seed: int = 0) -> list[dict]:
    """造 ``n`` 条 (state, tactic) 样本，状态与策略一一对应，模型可真正学会。"""
    rng = random.Random(seed)
    templates = [
        ("⊢ {a} + {b} = {b} + {a}", "Nat.add_comm"),
        ("⊢ {a} * {b} = {b} * {a}", "Nat.mul_comm"),
        ("⊢ 0 + {a} = {a}", "Nat.zero_add"),
        ("⊢ {a} * 1 = {a}", "Nat.mul_one"),
        ("⊢ ({a} + {b}) + {c} = {a} + ({b} + {c})", "Nat.add_assoc"),
    ]
    pairs: list[dict] = []
    for i in range(n):
        tmpl, tactic = rng.choice(templates)
        a, b, c = rng.choice("abcxyz"), rng.choice("abcxyz"), rng.choice("abcxyz")
        state = tmpl.format(a=a, b=b, c=c)
        pairs.append({"state": state, "tactic": tactic})
    return pairs


# 【F-b03-sft.load_pairs｜函数】读取 (state, tactic) jsonl
def load_pairs(path: str, limit: int = 2000) -> list[dict]:
    """读取 (state, tactic) jsonl，兼容几种常见字段名；坏行跳过。"""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = row.get("state") or row.get("goal") or row.get("public_proof_state") or row.get("prompt")
            tactic = row.get("tactic") or row.get("proof_step") or row.get("next_tactic") or row.get("completion")
            if state and tactic:
                out.append({"state": str(state), "tactic": str(tactic)})
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------------------
# 2. token 化 + label mask
# ---------------------------------------------------------------------------
# 【F-b03-sft.encode_pair｜函数】返回 (input_ids, labels)，prompt 段置 -100
def encode_pair(tok: CharTokenizer, state: str, tactic: str, max_len: int):
    """返回 ``(input_ids, labels)``：prompt 段 labels 全 -100，答案段为真实 id。"""
    prompt = PROMPT_PREFIX + state.strip() + PROMPT_MIDDLE
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(tactic.strip()) + [tok.eos_id]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([IGNORE_INDEX] * len(p_ids) + a_ids)[:max_len]
    return ids, labels


# 【F-b03-sft.SFTDataset｜类】state→tactic 的 token-masked 数据集
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[dict], tok: CharTokenizer, max_len: int):
        self.rows = [encode_pair(tok, p["state"], p["tactic"], max_len) for p in pairs]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        ids, labels = self.rows[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# 【F-b03-sft.make_collate｜函数】右填充并置 pad label=-100
def make_collate(pad_id: int):
    """右填充到 batch 内最大长度；pad 位置 label=-100（不参与 loss）。"""

    def collate(batch):
        maxlen = max(x.size(0) for x, _ in batch)
        xs, ys = [], []
        for x, y in batch:
            n = maxlen - x.size(0)
            xs.append(torch.cat([x, torch.full((n,), pad_id, dtype=torch.long)]))
            ys.append(torch.cat([y, torch.full((n,), IGNORE_INDEX, dtype=torch.long)]))
        return torch.stack(xs), torch.stack(ys)

    return collate


# 【F-b03-sft.answer_token_accuracy｜函数】只在答案 token 上统计 top-1 命中率
def answer_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """只在 labels != -100 的位置统计 teacher-forcing 的 top-1 命中率。"""
    mask = labels.ne(IGNORE_INDEX)
    pred = logits.argmax(dim=-1)
    correct = (pred.eq(labels) & mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


# ---------------------------------------------------------------------------
# 3. 训练 / 评估
# ---------------------------------------------------------------------------
# 【F-b03-sft.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny 模型 token-masked SFT（教学版）")
    p.add_argument("--config", default="micro", choices=["micro", "tiny"])
    p.add_argument("--states-file", default=None, help="(state,tactic) jsonl；缺省用合成样本")
    p.add_argument("--num-samples", type=int, default=240)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--max-len", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=20)
    p.add_argument("--max-iters", type=int, default=400)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="out_sft")
    p.add_argument("--resume", default=None)
    return p.parse_args()


# 【F-b03-sft.resolve_device｜函数】选择运行设备
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# 【F-b03-sft.lr_at｜函数】warmup + cosine 学习率
def lr_at(step: int, args) -> float:
    if step < args.warmup_iters:
        return args.lr * (step + 1) / max(1, args.warmup_iters)
    if step >= args.max_iters:
        return args.min_lr
    prog = (step - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * prog)) * (args.lr - args.min_lr)


# 【F-b03-sft.evaluate｜函数】留出集评估与生成
@torch.no_grad()
def evaluate(model, ds, collate, device, tok, amp_ctx, n_show: int = 3) -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            logits, loss = model(x, y)
        loss_sum += loss.item() * x.size(0)
        c, t = answer_token_accuracy(logits, y)
        correct += c
        total += t
    shows = []
    for i in range(min(n_show, len(ds))):
        ids, labels = ds[i]
        n_prompt = int((labels == IGNORE_INDEX).sum().item())
        prompt_ids = ids[: max(1, n_prompt)].unsqueeze(0).to(device)
        out = model.generate(prompt_ids, max_new_tokens=24, temperature=0.0)
        shows.append(tok.decode(out[0].tolist()[n_prompt:]))
    model.train()
    return {"loss": loss_sum / max(1, len(ds)),
            "tok_acc": correct / max(1, total), "shows": shows}


# 【F-b03-sft.main｜函数】SFT 训练主循环
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    pairs = load_pairs(args.states_file) if args.states_file else synthetic_pairs(args.num_samples, args.seed)
    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    vocab_text = "".join(p["state"] + p["tactic"] for p in pairs) + PROMPT_PREFIX + PROMPT_MIDDLE
    tok = CharTokenizer.from_text(vocab_text)

    train_ds = SFTDataset(train_pairs, tok, args.max_len)
    val_ds = SFTDataset(val_pairs, tok, args.max_len)
    collate = make_collate(tok.pad_id)

    cfg = get_config(args.config)
    cfg.vocab_size = tok.vocab_size
    cfg.sequence_len = max(cfg.sequence_len, args.max_len)
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    print(f"[数据] train={len(train_ds)} val={len(val_ds)} vocab={tok.vocab_size} max_len={args.max_len}")
    print(f"[模型] {args.config} 参数 {n_params/1e6:.3f}M")
    print(f"[mask] prompt 位置 label={IGNORE_INDEX}，仅答案 token 参与 loss")

    optimizer = model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=args.lr,
                                           betas=(0.9, 0.95), device_type=device.type)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_iter, best = 0, float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_iter = ckpt.get("iter", -1) + 1
        best = ckpt.get("best_val", float("inf"))
        print(f"[续训] {args.resume} @ {start_iter}")

    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = lr_at(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum = 0.0
        for _ in range(args.grad_accum):
            idx = torch.randint(len(train_ds), (min(args.batch_size, len(train_ds)),))
            x, y = collate([train_ds[int(i)] for i in idx])
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum += loss.item()
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
            print(f"iter {it:5d} | sft_loss {accum:.4f} | lr {lr:.2e}")

        if (it + 1) % args.eval_interval == 0 or it == args.max_iters - 1:
            m = evaluate(model, val_ds, collate, device, tok, amp_ctx)
            print(f"  [eval] iter {it} val_loss {m['loss']:.4f} answer_tok_acc {m['tok_acc']*100:.1f}%")
            if m["loss"] < best:
                best = m["loss"]
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(),
                            "tokenizer": tok.itos, "iter": it, "best_val": best},
                           os.path.join(args.out_dir, "best.pt"))

    print(f"[完成] {time.time()-t0:.1f}s best_val={best:.4f}")
    m = evaluate(model, val_ds, collate, device, tok, amp_ctx)
    for i, s in enumerate(m["shows"]):
        print(f"[生成 {i}] {s!r}")


if __name__ == "__main__":
    main()
