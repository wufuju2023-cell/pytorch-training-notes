#!/usr/bin/env python3
# 【源代码｜F-b05-qlora_bnb】v1.0/05-LoRA/code/with_api/qlora_bnb.py — 4bit nf4 冻结基座 + LoRA（QLoRA）
# 相关文档：《05-LoRA/01-LoRA原理.md》
"""QLoRA：4-bit (nf4) 冻结基座 + LoRA，用 bitsandbytes 省显存。

对照课程 05-LoRA 原理 QLoRA 一节：
  * nf4 4-bit 量化 + double quant + paged optimizer；
  * 反量化后前向，梯度只流向 LoRA adapter。

依赖：torch、transformers、peft、bitsandbytes（GPU 推荐）。
无 bitsandbytes 时脚本会给出提示并退出，不伪造结果。
用法：
    python qlora_bnb.py --model <local-7b> --r 16 --alpha 32 --steps 10
"""

from __future__ import annotations

import argparse
import json

import torch


# 【F-b05-qlora_bnb.build_qlora｜函数】4bit 量化基座 + LoRA 配置
def build_qlora(model_name: str, r: int, alpha: int, dropout: float,
                target_modules):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # 正态等概率 16 级
        bnb_4bit_use_double_quant=True,     # scale 再量化
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    return get_peft_model(model, config), tok


# 【F-b05-qlora_bnb.main｜函数】微调入口
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--target-modules", nargs="*",
                    default=["c_attn", "c_proj"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"skip": "CUDA/ROCm GPU not available; QLoRA 4-bit 需要 GPU"},
                         ensure_ascii=False))
        return 0
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        print(json.dumps({"skip": f"bitsandbytes unavailable: {exc}"},
                         ensure_ascii=False))
        return 0

    model, tok = build_qlora(args.model, args.r, args.alpha, args.dropout,
                             args.target_modules)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({
        "total_params": total, "trainable_params": trainable,
        "trainable_ratio": round(trainable / total, 6),
        "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }, ensure_ascii=False), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    texts = ["theorem t : P -> P", "lemma l : a = a"]
    for step in range(1, args.steps + 1):
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(json.dumps({"step": step, "loss": round(float(out.loss), 5)}),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
