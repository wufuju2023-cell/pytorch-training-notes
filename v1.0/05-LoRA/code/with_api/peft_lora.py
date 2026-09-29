#!/usr/bin/env python3
# 【源代码｜F-b05-peft_lora】v1.0/05-LoRA/code/with_api/peft_lora.py — HF+peft 的低秩微调
# 相关文档：《05-LoRA/01-LoRA原理.md》
"""PEFT 版 LoRA：用 HuggingFace + peft 在真实模型上做低秩微调。

对照 app/policy_server.py:112（r=16, alpha=32, dropout=0.02, 全部 attn+mlp,
init_lora_weights=True）。用很小的基座演示完整流程：
  1. 加载基座并打印参数；
  2. get_peft_model 注入 LoRA，打印可训练参数量与占比；
  3. 做几步小任务微调（合成文本 next-token），展示 loss 下降；
  4. 可选保存 adapter。

依赖：torch、transformers、peft。CPU 可跑 sshleifer/tiny-gpt2。
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


# 【F-b05-peft_lora.build_peft_model｜函数】加载基座并注入 peft LoRA
def build_peft_model(model_name: str, r: int, alpha: int, dropout: float,
                     target_modules, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        init_lora_weights=True,   # B=0 => 初始等价 base
    )
    model = get_peft_model(base, config).to(device)
    return model, tok


# 【F-b05-peft_lora.main｜函数】微调入口
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", default=None)
    ap.add_argument("--target-modules", nargs="*", default=["c_attn", "c_proj"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = build_peft_model(args.model, args.r, args.alpha, args.dropout,
                                  args.target_modules, device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({
        "total_params": total, "trainable_params": trainable,
        "trainable_ratio": round(trainable / total, 6),
        "target_modules": args.target_modules,
        "alpha_over_r": round(args.alpha / args.r, 3),
    }, ensure_ascii=False), flush=True)

    if trainable == 0:
        print(json.dumps({"error": "no trainable params; check target_modules"}))
        return 1

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    texts = ["theorem add_comm : a + b = b + a",
             "lemma zero_add : 0 + n = n",
             "example : 2 + 2 = 4",
             "theorem and_comm : P -> Q -> Q and P"]
    model.train()
    for step in range(1, args.steps + 1):
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        out = model(**enc, labels=enc["input_ids"])
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(json.dumps({"step": step, "loss": round(float(loss), 5)}),
                  flush=True)

    if args.save:
        model.save_pretrained(args.save)
        print(json.dumps({"saved_adapter": args.save}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
