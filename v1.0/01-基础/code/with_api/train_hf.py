# 【源代码｜F-b01-train_hf】v1.0/01-基础/code/with_api/train_hf.py — 用 HF Trainer/accelerate 训练同一小 GPT
# 相关文档：《01-基础/05-训练循环与数据管线.md》
"""用 HuggingFace 现成 API 训练同一个小 GPT（对照 from_scratch）。

两条路径
--------
* 默认：``transformers.Trainer`` + ``GPT2LMHeadModel``，几十行搞定训练/日志/保存。
* ``--use-accelerate``：手写 ``accelerate.Accelerator`` 训练循环（更接近生产代码）。

与 ``from_scratch/train.py`` 逐项对照见 ``README.md`` 的对照表。

用法
----
    # 字符级（无额外依赖，最快）
    python3 train_hf.py --tokenizer char --config micro --max-steps 200

    # BPE（需 tokenizers/transformers/datasets）
    python3 train_hf.py --tokenizer bpe --vocab-size 512 --max-steps 500

    # 用 accelerate 手写循环
    python3 train_hf.py --tokenizer char --use-accelerate --max-steps 200

    # 对照 nanoproof：训练完成后用 pipeline 采样
    python3 -c "from transformers import pipeline; g=pipeline('text-generation', model='out/hf-micro'); print(g('theorem ', max_new_tokens=60))"
"""

from __future__ import annotations

import argparse
import os

# 注意：重库（torch/transformers/datasets/accelerate）都在函数内 import，
# 这样即使本地没装也能通过 ``python3 -m py_compile`` 语法检查。


DEMO_TEXT = (
    "theorem add_comm (a b : Nat) : a + b = b + a := by\n"
    "  induction a with\n"
    "  | zero => simp\n"
    "  | succ a ih => rw [Nat.succ_add, Nat.add_succ, ih]\n"
) * 60


# 【F-b01-train_hf.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF GPT2 训练 tiny 语言模型")
    p.add_argument("--config", default="micro", choices=["micro", "tiny"])
    p.add_argument("--tokenizer", default="char", choices=["char", "bpe"])
    p.add_argument("--vocab-size", type=int, default=512, help="BPE 词表大小")
    p.add_argument("--text-file", default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--use-accelerate", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


# 【F-b01-train_hf.get_text｜函数】取训练文本
def get_text(args) -> str:
    if args.text_file and os.path.exists(args.text_file):
        return open(args.text_file, encoding="utf-8").read()
    return DEMO_TEXT


# 【F-b01-train_hf.build_tokenizer｜函数】返回 (tokenizer, vocab_size)
def build_tokenizer(args, text):
    """返回 (hf_tokenizer, vocab_size)。char 走本地轻量实现，bpe 走 tokenizers。"""
    if args.tokenizer == "bpe":
        import tokenizer_api

        tok = tokenizer_api.train_bpe(
            text, vocab_size=args.vocab_size, out_dir=os.path.join(args.out_dir, "tokenizer")
        )
    else:
        tok = tokenizer_api_build_char(text)
    return tok, tok.vocab_size


# 【F-b01-train_hf.tokenizer_api_build_char｜函数】借用 tokenizer_api 的字符级实现
def tokenizer_api_build_char(text):
    import tokenizer_api

    return tokenizer_api.build_char_tokenizer(text)


# 【F-b01-train_hf.build_datasets｜函数】长文本切定长 block 并造 labels
def build_datasets(tok, text, block_size):
    """把长文本切成定长 block，交给 ``DataCollatorForLanguageModeling`` 造 labels。"""
    from datasets import Dataset

    ids = tok.encode(text)
    n_blocks = (len(ids) - 1) // block_size
    if n_blocks <= 0:
        raise ValueError("语料太短，切不出一个 block")
    blocks = [ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    n_val = max(1, int(0.1 * n_blocks))
    ds = Dataset.from_dict({"input_ids": blocks[: n_blocks - n_val]})
    val = Dataset.from_dict({"input_ids": blocks[n_blocks - n_val :]})
    return ds, val


# 【F-b01-train_hf.build_model｜函数】构建 GPT2LMHeadModel
def build_model(vocab_size, block_size, n_layer, n_head, n_embd):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=1,
    )
    return GPT2LMHeadModel(cfg)


# 【F-b01-train_hf.train_with_trainer｜函数】用 Trainer 托管训练
def train_with_trainer(model, tokenizer, train_ds, val_ds, args):
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    collator = DataCollatorForLanguageModeling(tokenizer=None, mlm=False)
    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=max(50, args.max_steps // 4),
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    return trainer


# 【F-b01-train_hf.train_with_accelerate｜函数】用 accelerate 手写训练循环
def train_with_accelerate(model, tok, train_ds, val_ds, args, device):
    """手写 accelerate 循环：等价于 from_scratch/train.py 的 AMP+累积+cosine。"""
    import math

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from transformers import DataCollatorForLanguageModeling

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16" if args.bf16 else "no",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=None, mlm=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    for step in range(args.max_steps):
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out.loss)
                optimizer.step()
                optimizer.zero_grad()
            break  # 一个 DataLoader 批次当作一次优化步（教学从简）
        if step % 10 == 0:
            print(f"step {step} loss {out.loss.item():.4f}")

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    os.makedirs(args.out_dir, exist_ok=True)
    unwrapped.save_pretrained(args.out_dir)
    return unwrapped


# 【F-b01-train_hf.main｜函数】训练入口
def main() -> None:
    args = parse_args()
    import torch

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)

    from configs import get_config

    cfg = get_config(args.config)
    block_size = args.block_size or cfg.sequence_len
    args.out_dir = args.out_dir or os.path.join("out", f"hf-{args.config}")

    text = get_text(args)
    tok, vocab_size = build_tokenizer(args, text)
    train_ds, val_ds = build_datasets(tok, text, block_size)
    model = build_model(vocab_size, block_size, cfg.n_layer, cfg.n_head, cfg.n_embd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[HF] vocab={vocab_size} blocks={len(train_ds)} 参数={n_params/1e6:.3f}M block={block_size}")

    if args.use_accelerate:
        train_with_accelerate(model, tok, train_ds, val_ds, args, None)
    else:
        train_with_trainer(model, tok, train_ds, val_ds, args)
    print(f"[完成] 模型已保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
