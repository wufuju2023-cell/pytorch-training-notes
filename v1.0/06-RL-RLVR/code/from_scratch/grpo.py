"""最小 GRPO 教学实现（纯 PyTorch，CPU 可跑）。

任务：单位数加法 mod 10。输入 "a+b="，模型需生成答案数字。
奖励 = 可验证的精确匹配（exact-match），即 RLVR 的最简形式。

实现要点（对照 06-RL-RLVR/01-RL与RLVR原理.md）：
* 组采样：同一 prompt 采样 G 个 completion；
* 组内相对优势：A_i = (r_i - mean(r)) / (std(r) + eps)；
* PPO-style clip 代理目标 + KL 正则（k2）；
* 参考策略冻结，防止灾难性遗忘。

运行：
    python grpo.py --steps 300 --G 8 --batch 16
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 0. 字符级词表：数字 + "+" + "="
# ---------------------------------------------------------------------------
CHARS = "0123456789+="
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB = len(CHARS)
BLOCK = 8  # "a+b=" 最多 4 个字符，答案 1 个


def encode(text: str) -> torch.Tensor:
    return torch.tensor([STOI[c] for c in text], dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. 极小 Transformer LM（与 01 章 tinyGPT 同构，缩小版）
# ---------------------------------------------------------------------------
@dataclass
class Config:
    vocab: int = VOCAB
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    block: int = BLOCK
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x):
        h = self.ln1(x)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.block, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))  # (B, T, vocab)

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n_tokens: int, temperature: float = 1.0):
        """自回归采样 n_tokens 个 token，返回 (B, n_tokens) 与每步 logprob。"""
        self.eval()
        out, logps = [], []
        idx = prompt
        for _ in range(n_tokens):
            logits = self.forward(idx)[:, -1, :] / max(temperature, 1e-6)
            logp = F.log_softmax(logits, dim=-1)
            tok = torch.multinomial(logp.exp(), 1)
            out.append(tok)
            logps.append(logp.gather(-1, tok))
            idx = torch.cat([idx, tok], dim=1)
        return torch.cat(out, dim=1), torch.cat(logps, dim=1)

    def seq_logprob(self, prompt: torch.Tensor, completion: torch.Tensor):
        """completion 每个 token 在给定 [prompt|completion] 下的 logprob。"""
        idx = torch.cat([prompt, completion], dim=1)
        logits = self.forward(idx)
        logp = F.log_softmax(logits, dim=-1)
        p = prompt.size(1)
        return logp[:, p - 1 : p - 1 + completion.size(1), :].gather(
            -1, completion.unsqueeze(-1)
        ).squeeze(-1)


# ---------------------------------------------------------------------------
# 2. 可验证奖励（RLVR）：精确匹配
# ---------------------------------------------------------------------------
def make_prompts(batch: int, generator: torch.Generator):
    a = torch.randint(0, 10, (batch,), generator=generator)
    b = torch.randint(0, 10, (batch,), generator=generator)
    prompts = [f"{int(x)}+{int(y)}=" for x, y in zip(a, b)]
    answers = [(int(x) + int(y)) % 10 for x, y in zip(a, b)]
    return prompts, answers


def verify(prompt: str, completion: str, answer: int) -> float:
    """确定性验证器：只看第一个生成字符。"""
    if not completion:
        return 0.0
    return 1.0 if completion[0] == str(answer) else 0.0


# ---------------------------------------------------------------------------
# 3. GRPO
# ---------------------------------------------------------------------------
class GRPOTrainer:
    def __init__(self, policy: TinyLM, ref: TinyLM, lr: float = 3e-3,
                 clip: float = 0.2, beta_kl: float = 0.05, eps_std: float = 1e-4):
        self.policy = policy
        self.ref = ref
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr)
        self.clip = clip
        self.beta_kl = beta_kl
        self.eps_std = eps_std

    def step(self, prompts, answers, G: int, temperature: float = 1.0):
        B = len(prompts)
        prompt_ids = torch.stack([encode(p) for p in prompts])
        # --- 组采样：每个 prompt 复制 G 次 ---
        rep = prompt_ids.repeat_interleave(G, dim=0)
        comp, _ = self.policy.sample(rep, n_tokens=1, temperature=temperature)
        # --- 可验证奖励 ---
        rewards = torch.zeros(B * G)
        for i in range(B * G):
            text = "".join(ITOS[int(t)] for t in comp[i])
            rewards[i] = verify(prompts[i // G], text, answers[i // G])
        rewards = rewards.view(B, G)
        # --- 组内相对优势 ---
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        adv = (rewards - mean) / (std + self.eps_std)  # (B, G)
        adv_flat = adv.view(-1)

        # --- 新/旧策略 logprob ---
        logp = self.policy.seq_logprob(rep, comp).sum(dim=1)  # (B*G,)
        with torch.no_grad():
            logp_old = self.ref.seq_logprob(rep, comp).sum(dim=1)
        ratio = torch.exp(logp - logp_old)
        unclipped = ratio * adv_flat
        clipped = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_flat
        pg_loss = -torch.min(unclipped, clipped).mean()
        # --- KL(k2) 正则 ---
        kl = self.beta_kl * ((logp - logp_old) ** 2).mean()
        loss = pg_loss + kl

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        stats = {
            "loss": float(loss.detach()),
            "pg_loss": float(pg_loss.detach()),
            "kl": float(kl.detach()),
            "reward": float(rewards.mean()),
            "pass@1": float((rewards[:, 0] > 0).float().mean()),
            "adv_abs": float(adv_flat.abs().mean()),
        }
        return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cfg = Config()
    policy = TinyLM(cfg)
    ref = TinyLM(cfg)
    ref.load_state_dict(policy.state_dict())
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    trainer = GRPOTrainer(policy, ref, lr=args.lr)
    for step in range(1, args.steps + 1):
        prompts, answers = make_prompts(args.batch, gen)
        stats = trainer.step(prompts, answers, args.G)
        if step % 20 == 0 or step == 1:
            print(
                f"step {step:4d} | loss {stats['loss']:.4f} | "
                f"pg {stats['pg_loss']:.4f} | kl {stats['kl']:.4f} | "
                f"reward {stats['reward']:.3f} | pass@1 {stats['pass@1']:.3f}"
            )

    prompts, answers = make_prompts(16, gen)
    with torch.no_grad():
        comp, _ = policy.sample(torch.stack([encode(p) for p in prompts]), 1, 0.5)
    print("\n抽样验证：")
    for p, a, c in zip(prompts, answers, comp):
        text = "".join(ITOS[int(t)] for t in c)
        print(f"  {p}{text}  (期望 {a}, {'OK' if text[0] == str(a) else 'X'})")


if __name__ == "__main__":
    main()
