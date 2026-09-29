"""从零实现 LoRA 并注入极简 GPT（tinyGPT）。

对照 app/policy_server.py:112 与 gpu_runtime/real_backend.py:24。
h = W0 x + (alpha/r) * B(A x)，B=0 初始化 => 初始等价 base。
只依赖 PyTorch，CPU 可运行。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden)
        self.up_proj = nn.Linear(d_model, hidden)
        self.down_proj = nn.Linear(hidden, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 128,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = 64,
                 mlp_ratio: int = 2) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return self.base


def inject_lora(model: nn.Module, target_names=TARGET_MODULES, r: int = 16,
                alpha: int = 32, dropout: float = 0.0) -> nn.Module:
    names = tuple(target_names)
    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and full.split(".")[-1] in names:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
    return model


def lora_parameters(model: nn.Module):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def mark_only_lora_trainable(model: nn.Module) -> int:
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    return count_parameters(model, trainable_only=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyGPT()
    total = count_parameters(model, trainable_only=False)
    inject_lora(model, r=16, alpha=32)
    n_lora = mark_only_lora_trainable(model)
    x = torch.randint(0, 128, (2, 16))
    logits = model(x)
    print("total params  :", total)
    print("lora trainable:", n_lora, f"({100 * n_lora / total:.2f}%)")
    print("logits shape  :", tuple(logits.shape))
