"""纯 PyTorch 手写 tiny GPT（教学向，单文件可读）。

目标：把“从字符到下一个字符”的最小可训练语言模型拆开讲清楚，每个组件
都对应后面的理论篇与 AlphaProof 源码。整模型 <= 20M 参数，CPU 也能跑。

结构（自底向上）
--------------
1. ``RMSNorm``            —— 无偏置、无可学习参数的均方根归一化。
2. ``precompute_rope``    —— RoPE 的 cos/sin 频率表（无参数位置编码）。
3. ``apply_rope``         —— 把 q/k 的相邻两半做二维旋转。
4. ``repeat_kv``          —— GQA：把 KV 头复制到 query 头数。
5. ``CausalSelfAttention``—— QKV 投影 + RoPE + 因果注意力 + 输出投影。
6. ``SwiGLU``             —— gated MLP（对应 nanoproof 的 relu^2 MLP 的变体）。
7. ``Block``              —— Pre-LN + 残差。
8. ``GPT``                —— embedding + blocks + 权重共享 + loss + 采样。

对照 AlphaProof / nanoproof：

* ``nanoproof/nanoproof/model.py:32``  ``NetworkConfig``
* ``nanoproof/nanoproof/model.py:58``  ``apply_rotary_emb``
* ``nanoproof/nanoproof/model.py:67``  ``CausalSelfAttention``（GQA + QK norm）
* ``nanoproof/nanoproof/model.py:127`` ``MLP``
* ``nanoproof/nanoproof/model.py:140`` ``Block``
* ``nanoproof/nanoproof/model.py:405`` ``forward`` / ``generate``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import GPTConfig


# ---------------------------------------------------------------------------
# 0. 基础归一化
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """均方根归一化（Zhang & Sennrich, 2019）。

    ``RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight``

    与 LayerNorm 的区别：不减均值、无偏置；nanoproof 甚至不用可学习 ``weight``
    （``model.py:45`` 直接调 ``F.rms_norm(x, (x.size(-1),))``）。这里保留
    ``weight`` 方便消融“可学习缩放是否重要”。
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dtype)
        if self.weight is not None:
            x = x * self.weight
        return x


# ---------------------------------------------------------------------------
# 1. RoPE：旋转位置编码
# ---------------------------------------------------------------------------
def precompute_rope(
    seq_len: int, head_dim: int, base: float = 10000.0, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 cos/sin 频率表。

    返回形状 ``(seq_len, head_dim // 2)``（每对维度一个角度），便于广播到
    ``(B, H, T, D/2)``。对应 nanoproof ``_precompute_rotary_embeddings``
    （``model.py:227``），只是那边为了 FA3 把布局存成 ``(1, T, 1, D/2)``。
    """
    assert head_dim % 2 == 0, "RoPE 需要偶数 head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # (T, D/2)
    return freqs.cos().to(device), freqs.sin().to(device)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 ``x`` 的最后两半维做二维旋转。

    ``x``: ``(B, H, T, D)``；``cos``/``sin``: ``(T, D/2)``（会广播成
    ``(1, 1, T, D/2)``）。这是 LLaMA / nanoproof 的“分半”实现：

    ``y1 = x1*cos + x2*sin``，``y2 = -x1*sin + x2*cos``
    """
    assert x.dim() == 4, "apply_rope 期望 (B, H, T, D)"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


# ---------------------------------------------------------------------------
# 2. GQA 辅助
# ---------------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV 头沿 head 维复制 ``n_rep`` 次（GQA / MQA）。

    ``x``: ``(B, n_kv_head, T, D)`` -> ``(B, n_kv_head*n_rep, T, D)``。
    """
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


# ---------------------------------------------------------------------------
# 3. 因果自注意力（含 GQA、RoPE、可选的 QK-norm）
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, use_qk_norm: bool = True):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.use_qk_norm = use_qk_norm

        # 无偏置投影（nanoproof 全线 bias=False）
        self.c_q = nn.Linear(cfg.n_embd, cfg.q_proj_dim, bias=False)
        self.c_k = nn.Linear(cfg.n_embd, cfg.kv_proj_dim, bias=False)
        self.c_v = nn.Linear(cfg.n_embd, cfg.kv_proj_dim, bias=False)
        self.c_proj = nn.Linear(cfg.q_proj_dim, cfg.n_embd, bias=False)

        # QK-norm：对每个头做 RMSNorm，稳定 attention logits（nanoproof model.py:94）
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # PyTorch 原生 SDPA：内部走 flash / mem-efficient / math 后端
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# 4. SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """``down(silu(gate(x)) * up(x))``，中间维度对齐 LLaMA。

    nanoproof 用的是 ``relu(x)^2``（``model.py:135``），二者都是 gated/门控家族；
    在 N05 notebook 里可以一行开关互换。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = cfg.multiple_of * ((hidden + cfg.multiple_of - 1) // cfg.multiple_of)
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# 5. Transformer block（Pre-LN + 残差）
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, use_qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(cfg, use_qk_norm=use_qk_norm)
        self.norm2 = RMSNorm(cfg.n_embd, elementwise_affine=False)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 6. 完整 GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = None
        if not cfg.use_rope:  # 消融：退回可学习绝对位置嵌入
            self.pos_emb = nn.Embedding(cfg.sequence_len, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, elementwise_affine=False)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.use_rope:
            cos, sin = precompute_rope(cfg.sequence_len, cfg.head_dim, cfg.rope_base)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 残差投影除以 sqrt(2*n_layer)（GPT-2 技巧），训练更稳
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.sequence_len, f"序列长度 {T} 超过配置 {self.cfg.sequence_len}"
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        if self.cfg.use_rope:
            cos, sin = self.cos[:T], self.sin[:T]
        else:
            cos = sin = None
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """自回归采样；``temperature<=0`` 时退化为贪心。

        支持 top-k（保留概率最高的 k 个）与 top-p / nucleus（累计概率 <= p）。
        参考 nanoproof ``generate``（``model.py:492``，只实现了 temperature+top-k）。
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.sequence_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cum = probs.cumsum(dim=-1)
                    remove = cum - probs > top_p
                    sorted_logits[remove] = -float("inf")
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == 0:  # 约定 0 为 <eos>，仅作教学演示
                break
        if was_training:
            self.train()
        return idx

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: tuple[float, float],
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """标准 nanoGPT 参数分组：矩阵权重 decay，其余（norm/embedding）不 decay。"""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and "emb" not in name:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs = {"fused": True} if device_type == "cuda" else {}
        try:
            return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **kwargs)
        except (TypeError, RuntimeError):
            return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


def build_model(name: str = "micro") -> GPT:
    """便捷工厂：``build_model("micro"|"tiny")``。"""
    from configs import get_config

    return GPT(get_config(name))


if __name__ == "__main__":
    from configs import MICRO, TINY

    for cfg in (MICRO, TINY):
        model = GPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        logits, loss = model(x, x)
        print(
            f"cfg(n_layer={cfg.n_layer}, n_embd={cfg.n_embd}, n_head={cfg.n_head}, "
            f"n_kv_head={cfg.n_kv_head}) -> 实测参数 {model.num_params()/1e6:.3f}M, "
            f"logits {tuple(logits.shape)}, loss {loss.item():.3f}"
        )
