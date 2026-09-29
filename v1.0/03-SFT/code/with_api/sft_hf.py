r"""SFT（with_api 版）：HuggingFace transformers + peft LoRA，带 chat template。

与 from_scratch/sft.py 做同一件事（state -> tactic 的 token-masked SFT），但用
工业栈：``apply_chat_template`` 构造 prompt、``peft`` 只训练 LoRA 适配器、``Trainer``
托管训练循环。这是 AlphaProof 容器 ``train_full_supervised.py``（Qwen3-1.7B QLoRA）
的最小同构版本。

关键点
------
* **chat template**：把 (state, tactic) 变成 messages ``[{"role":"user",...},
  {"role":"assistant",...}]``，再 ``apply_chat_template`` 得到带特殊标记的字符串。
  这保证推理时的输入格式与训练一致。
* **label mask**：先 tokenize prompt，长度记为 L；整段 tokenize 后把前 L 个 label
  置 -100，只在 assistant 回复（tactic）上算交叉熵。
* **LoRA**：冻结基座，只训练低秩增量 $\Delta W = BA$（$B\in\mathbb{R}^{d\times r}$，
  $A\in\mathbb{R}^{r\times k}$），显存与存储都小。

用法
----
::

    pip install "transformers>=4.46" "peft>=0.13" "datasets>=3.0" accelerate
    python3 sft_hf.py --max-steps 30 --batch-size 2            # CPU 也能跑
    python3 sft_hf.py --model-name Qwen/Qwen2.5-0.5B-Instruct --max-steps 100 --amp
    python3 sft_hf.py --states-file /path/to/state_tactic.jsonl

对照 AlphaProof / nanoproof
--------------------------
* 容器 train_full_supervised.py：Qwen3-1.7B QLoRA（4bit nf4、r=16/alpha=32、
  dropout=0.05，targets q/k/v/o/gate/up/down），prompt 全 -100，答案 =
  ``<answer>{"calls":[...]}</answer>``。
* app/train_sft.py（DEPRECATED）：LoRA r=16 + HF Trainer。
* nanoproof/sft.py：LeanTree token-masked CE（DDP）。
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


# ---------------------------------------------------------------------------
# 1. 数据：state -> tactic
# ---------------------------------------------------------------------------
_TEMPLATES = [
    ("⊢ a + b = b + a", "rw [Nat.add_comm]"),
    ("⊢ a * b = b * a", "rw [Nat.mul_comm]"),
    ("h : a + b = b + a ⊢ a + b = b + a", "exact h"),
    ("⊢ (0 : Nat) + n = n", "simp"),
    ("⊢ (a + b) + c = a + (b + c)", "rw [Nat.add_assoc]"),
    ("⊢ n * 1 = n", "simp"),
    ("⊢ ([] : List α) ++ xs = xs", "simp"),
    ("h : P ∧ Q ⊢ P", "exact h.1"),
]


def synthetic_pairs(n: int = 160, seed: int = 0) -> list[dict]:
    import random

    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st, tac = rng.choice(_TEMPLATES)
        out.append({"state": st, "tactic": tac})
    return out


def load_pairs(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = row.get("state") or row.get("proof_state") or row.get("public_proof_state")
            tactic = row.get("tactic") or row.get("next_tactic")
            if state and tactic:
                out.append({"state": str(state), "tactic": str(tactic)})
    return out


class SFTDataset(Dataset):
    """按 chat template 构造 prompt，只对 assistant 回复计 loss。"""

    def __init__(self, pairs, tokenizer, max_len: int = 256):
        self.rows = []
        for p in pairs:
            messages = [
                {"role": "user", "content": "Given the Lean proof state, output the next tactic.\n\n"
                                            "STATE:\n" + p["state"]},
                {"role": "assistant", "content": p["tactic"]},
            ]
            if getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False,
                                                            add_generation_prompt=True)
            else:  # 无 chat template 的模型：退回手工模板
                text = f"STATE:\n{p['state']}\nTACTIC:\n{p['tactic']}{tokenizer.eos_token}"
                prompt_text = f"STATE:\n{p['state']}\nTACTIC:\n"
            enc = tokenizer(text, truncation=True, max_length=max_len)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            labels = list(enc["input_ids"])
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100          # prompt 位置屏蔽
            self.rows.append({"input_ids": enc["input_ids"], "labels": labels,
                              "attention_mask": enc["attention_mask"]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF + peft LoRA SFT（教学版）")
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2",
                   help="基座；有 chat template 的模型（如 Qwen/Qwen2.5-0.5B-Instruct）效果更贴近真实")
    p.add_argument("--states-file", default=None)
    p.add_argument("--num-samples", type=int, default=160)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out-dir", default="out_sft_lora")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:  # noqa: BLE001
        raise SystemExit("需要 peft：pip install 'peft>=0.13'") from e

    pairs = load_pairs(args.states_file) if args.states_file else synthetic_pairs(args.num_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    # 推断 LoRA 的目标模块名（不同架构不同）
    target_modules = None
    for cand in (["q_proj", "v_proj"], ["c_attn"], ["query", "value"],
                 ["q_proj", "k_proj", "v_proj", "o_proj"]):
        if any(name.endswith(cand[0]) for name, _ in model.named_modules()):
            target_modules = cand
            break
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM", target_modules=target_modules)
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] r={args.lora_r} alpha={args.lora_alpha} targets={target_modules} "
          f"可训练 {trainable/1e6:.4f}M / 总 {total/1e6:.3f}M ({100*trainable/total:.2f}%)")
    print(f"[chat template] {'有' if getattr(tokenizer, 'chat_template', None) else '无（退回手工模板）'}")

    train_ds = SFTDataset(pairs, tokenizer, args.max_len)
    targs = TrainingArguments(
        output_dir=args.out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=max(1, args.max_steps),
        save_total_limit=1,
        bf16=args.amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.amp and torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=default_data_collator)
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # 生成一条看效果
    model.eval()
    sample = pairs[0]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Given the Lean proof state, output the next tactic.\n\n"
                                          "STATE:\n" + sample["state"]}],
            tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"STATE:\n{sample['state']}\nTACTIC:\n"
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    print("[样本 state]", repr(sample["state"]))
    print("[gold tactic]", repr(sample["tactic"]))
    print("[生成]", repr(tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
    print(f"[完成] LoRA 适配器保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
