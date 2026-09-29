#!/usr/bin/env python3
# 【源代码｜F-b04-value_head_api】v1.0/04-价值头/code/with_api/value_head_api.py — HF hidden states + 小 MLP 价值头
# 相关文档：《04-价值头/01-价值头原理.md》
"""with_api 版价值头：HuggingFace hidden states + 小 MLP 头。

对照 app/train_value_head.py（冻结 backbone、取最后 token hidden、只训 MLP）
与 gpu_runtime/real_backend.py（LoRA 训练时把 value head 一起挂进优化器）。

要点:
  * 基座用 AutoModelForCausalLM 加载并冻结;
  * last_token_hidden 支持左/右 padding, 取最后一个非 pad 位置;
  * 头始终 fp32（即使基座 bf16）;
  * --use-lora 时基座冻结, 只加 LoRA adapter, 价值头与 adapter 一起训练。

用法:
    python value_head_api.py --model sshleifer/tiny-gpt2 --steps 20
    python value_head_api.py --model sshleifer/tiny-gpt2 --use-lora
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# 【F-b04-value_head_api.ValueHead｜类】Linear→SiLU→Linear 价值头
class ValueHead(nn.Module):
    """Linear(H,hd)->SiLU->Linear(hd,out)。out=1 用 Tanh, out=64 是分类头。"""

    def __init__(self, hidden_size: int, hidden_dim: int = 256,
                 num_bins: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        out_dim = num_bins if num_bins else 1
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        if not num_bins:
            self.mlp.append(nn.Tanh())

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.mlp(hidden_states.float()).squeeze(-1)


# 【F-b04-value_head_api.last_token_hidden｜函数】取最后一个非 pad token 的 hidden
def last_token_hidden(output, attention_mask: Tensor | None = None) -> Tensor:
    """从 HF 输出取最后一个非 padding token 的 hidden state。"""
    hidden = getattr(output, "hidden_states", None)
    hidden = hidden[-1] if hidden else output.last_hidden_state
    if attention_mask is None:
        return hidden[:, -1, :]
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    positions = positions.unsqueeze(0).expand(hidden.shape[0], -1)
    valid = attention_mask.to(torch.bool)
    last = torch.where(valid, positions, torch.zeros_like(positions)).amax(dim=1)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, last, :]


# 【F-b04-value_head_api.build_backbone｜函数】加载并冻结基座（可选注入 LoRA）
def build_backbone(model_name: str, use_lora: bool, lora_r: int = 16,
                   lora_alpha: int = 32, lora_dropout: float = 0.02):
    """加载并冻结基座; 可选注入 LoRA。返回 (model, tokenizer, hidden_size)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(False)
    hidden_size = int(
        getattr(base.config, "hidden_size", 0)
        or getattr(base.config, "n_embd", 0)
        or getattr(base.config, "d_model", 0)
    )
    if hidden_size <= 0:
        raise ValueError("cannot infer hidden size from model config")

    if use_lora:
        from peft import LoraConfig, get_peft_model
        # LLaMA 风格与 GPT-2 风格命名不同；按配置自动选择
        targets = (["c_attn", "c_proj"] if hasattr(base.config, "n_embd")
                   else ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"])
        lora = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=targets,
            init_lora_weights=True,
        )
        base = get_peft_model(base, lora)
    return base, tok, hidden_size


# 【F-b04-value_head_api.main｜函数】训练入口
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--num-bins", type=int, default=0)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, hidden_size = build_backbone(args.model, args.use_lora)
    model.to(device)
    head = ValueHead(hidden_size, args.hidden_dim, args.num_bins).to(device)
    trainable = [p for p in head.parameters() if p.requires_grad]
    if args.use_lora:
        trainable += [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    prompts = args.prompts or [
        "The goal is to prove that for all n, n + 0 = n.",
        "We need to show the following inequality holds.",
        "Consider the sequence defined by the recurrence.",
        "By induction on the structure of the term, we get",
    ]
    depths = torch.tensor(
        [1.0 + (len(p) % 12) for p in prompts], dtype=torch.float32, device=device)

    for step in range(1, args.steps + 1):
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, return_dict=True)
            hidden = last_token_hidden(out, enc.get("attention_mask")).detach()
        logits = head(hidden)
        if args.num_bins:
            target = torch.clamp(depths, max=float(args.num_bins)).floor().long()
            loss = F.cross_entropy(logits, target)
        else:
            target = -torch.clamp(depths, max=64.0) / 64.0
            loss = F.mse_loss(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(json.dumps({
                "step": step, "loss": round(float(loss), 5),
                "hidden_size": hidden_size,
                "trainable_params": int(sum(p.numel() for p in trainable)),
            }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
