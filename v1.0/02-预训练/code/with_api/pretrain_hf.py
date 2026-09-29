# 【源代码｜F-b02-pretrain_hf】v1.0/02-预训练/code/with_api/pretrain_hf.py — HF datasets+Trainer 预训练版
# 相关文档：《02-预训练/01-预训练原理.md》
r"""预训练（with_api 版）：用 HuggingFace ``datasets`` + ``transformers.Trainer``。

与 ``from_scratch/pretrain.py`` 做同一件事（next-token 语言建模），但全部交给
HF 生态：自动 batch、学习率调度、混合精度、日志、checkpoint、断点续训。
适合快速对比"自己写循环"和"框架托管"的差异。

特点
----
* **小基座**：默认用 ``GPT2Config`` 从零初始化一个 ~10–20M 参数的 GPT-2
  （``--use-pretrained`` 时才加载 ``sshleifer/tiny-gpt2`` 权重）。
* **小语料**：默认用内置多域合成语料；也可 ``--hf-dataset`` 指向真实数据集
  （例如 ``internlm/Lean-Github`` 的极小切片）。
* **小步数**：``--max-steps 60`` 即可看到 loss 下降，Colab CPU/GPU 都能跑。

用法
----
::

    pip install "transformers>=4.46" "datasets>=3.0" accelerate torch
    python3 pretrain_hf.py --max-steps 60 --config tiny
    python3 pretrain_hf.py --use-pretrained --model-name sshleifer/tiny-gpt2 --max-steps 30

对照 AlphaProof / nanoproof
--------------------------
* nanoproof 自己实现了 DDP + MuonAdamW 训练循环（``pretrain.py``）；本文件的
  ``Trainer`` 用 AdamW + ``get_linear_schedule_with_warmup`` 提供等价能力。
* 真实预训练语料（Nemotron-CC-Math ~20B tok / Lean-GitHub ~65M tok）在正文
  01-预训练原理.md 与 datasets/README.md 里说明。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

# 与 from_scratch/pretrain.py 同源的合成多域语料（这里保持字符级可读性）
CORPUS = """
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
import Mathlib
open Function Set
lemma succ_add (a b : Nat) : (a + 1) + b = (a + b) + 1 := by
  rw [Nat.add_assoc, Nat.add_comm 1 b, ← Nat.add_assoc]
the next token prediction objective assigns probability to observed text.
perplexity is the exponential of the mean cross entropy over tokens.
""".strip()


# 【F-b02-pretrain_hf.Cfg｜类】预训练配置容器
@dataclass
class Cfg:
    config: str = "tiny"
    model_name: str = "sshleifer/tiny-gpt2"
    use_pretrained: bool = False
    tokenizer_name: str = "gpt2"
    hf_dataset: str | None = None
    hf_split: str = "train"
    hf_text_field: str = "text"
    max_rows: int = 2000
    block_size: int = 128
    max_steps: int = 60
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 3e-3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    seed: int = 1337
    out_dir: str = "out_pretrain_hf"


# 两档小基座（参数量 ~5M / ~19M，均 ≤ 20M）
SIZES = {
    "micro": dict(n_layer=4, n_head=4, n_embd=128, n_positions=256),
    "tiny": dict(n_layer=6, n_head=8, n_embd=256, n_positions=512),
}


# 【F-b02-pretrain_hf.parse_args｜函数】解析命令行参数
def parse_args() -> Cfg:
    p = argparse.ArgumentParser(description="HF Trainer 预训练（教学版）")
    p.add_argument("--config", default="tiny", choices=list(SIZES))
    p.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    p.add_argument("--use-pretrained", action="store_true",
                   help="加载 model-name 的权重；缺省从零初始化 GPT2Config")
    p.add_argument("--tokenizer-name", default="gpt2")
    p.add_argument("--hf-dataset", default=None, help="可选：真实 HF 数据集 id")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-text-field", default="text")
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="out_pretrain_hf")
    a = p.parse_args()
    return Cfg(**vars(a))


# 【F-b02-pretrain_hf.collect_texts｜函数】收集多域文本
def collect_texts(cfg: Cfg) -> list[str]:
    if cfg.hf_dataset:
        # 真实数据集（教学只用切片），失败时回退内置语料
        try:
            ds = Dataset.load_dataset(cfg.hf_dataset, split=cfg.hf_split)  # type: ignore[attr-defined]
            ds = ds.select(range(min(cfg.max_rows, len(ds))))
            field = cfg.hf_text_field
            if field not in ds.column_names:  # 自动挑第一个字符串列
                field = next(c for c in ds.column_names if isinstance(ds[0][c], str))
            return [str(t) for t in ds[field] if t]
        except Exception as e:  # noqa: BLE001
            print(f"[数据] 加载 {cfg.hf_dataset} 失败（{e}），回退内置语料")
    # 内置语料：重复到够 max_rows 条（教学近似；真实是流式 tokenizer）
    n = max(cfg.max_rows // 40, 32)
    return [CORPUS for _ in range(n)]


# 【F-b02-pretrain_hf.build_tokenizer｜函数】构建并返回分词器
def build_tokenizer(cfg: Cfg):
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# 【F-b02-pretrain_hf.build_model｜函数】构建小 GPT-2 基座
def build_model(cfg: Cfg, vocab_size: int, tok) -> GPT2LMHeadModel:
    if cfg.use_pretrained:
        model = GPT2LMHeadModel.from_pretrained(cfg.model_name)
        model.resize_token_embeddings(len(tok))
        return model
    hparams = SIZES[cfg.config]
    conf = GPT2Config(
        vocab_size=vocab_size,
        n_layer=hparams["n_layer"],
        n_head=hparams["n_head"],
        n_embd=hparams["n_embd"],
        n_positions=hparams["n_positions"],
        n_ctx=hparams["n_positions"],
        bos_token_id=tok.bos_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return GPT2LMHeadModel(conf)


# 【F-b02-pretrain_hf.main｜函数】预训练入口
def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    tok = build_tokenizer(cfg)
    texts = collect_texts(cfg)

    def tokenize(batch):
        return tok(batch["text"])

    ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.filter(lambda e: len(e["input_ids"]) >= 8)

    # 拼接 + 切 block：把整条语料连成 token 流，按 block_size 切
    def group(batch):
        concat = sum(batch["input_ids"], [])
        n = len(concat) // cfg.block_size * cfg.block_size
        chunks = [concat[i:i + cfg.block_size] for i in range(0, n, cfg.block_size)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    def group_texts(batch):
        concat = []
        for ids in batch["input_ids"]:
            concat += ids
        n = len(concat) // cfg.block_size * cfg.block_size
        chunks = [concat[i:i + cfg.block_size] for i in range(0, n, cfg.block_size)]
        return {"input_ids": chunks, "labels": [c[:] for c in chunks]}

    ds = ds.map(group_texts, batched=True)
    model = build_model(cfg, len(tok), tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] {cfg.config} 参数量 {n_params/1e6:.3f}M  样本块 {len(ds)}  词表 {len(tok)}")

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=cfg.max_steps,           # 结束时存一次
        save_total_limit=1,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        report_to=[],
        seed=cfg.seed,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    # 生成一小段文本，直观对比 from_scratch 版
    model.eval()
    prompt = "theorem "
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=60, do_sample=True, temperature=0.8,
                             top_k=40, pad_token_id=tok.pad_token_id)
    print("[生成]\n" + tok.decode(out[0], skip_special_tokens=True))
    print(f"[完成] checkpoint -> {cfg.out_dir}")


if __name__ == "__main__":
    main()
