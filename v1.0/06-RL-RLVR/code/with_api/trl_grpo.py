"""trl GRPO + PEFT/LoRA 版（with_api）——在极小基座上做 RLVR。

与 from_scratch/grpo.py 的区别：这里把“策略、组采样、clip、KL、参考模型”全部
交给 HuggingFace `trl.GRPOTrainer`，我们只提供：
  * 数据集：prompt（"a+b="），
  * 奖励函数：可验证的精确匹配（RLVR），
  * LoRA 配置：只训低秩适配器（对应第 05 章）。

Colab / CPU 可跑（用 tiny 模型）；GPU 更快。
    pip install "trl>=0.9" transformers peft datasets accelerate
    python trl_grpo.py --model sshleifer/tiny-gpt2 --steps 30
"""

from __future__ import annotations

import argparse
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def build_dataset(n: int = 512):
    rng = random.Random(0)
    rows = []
    for _ in range(n):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        rows.append({
            "prompt": f"{a}+{b}=",
            "answer": str((a + b) % 10),
        })
    return Dataset.from_list(rows)


def extract_digit(text: str):
    m = re.search(r"(\d)", text)
    return m.group(1) if m else None


def reward_fn(completions, answer, **kwargs):
    """可验证奖励：completion 中第一个数字是否等于正确答案。"""
    rewards = []
    for completion, ans in zip(completions, answer):
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        pred = extract_digit(text)
        rewards.append(1.0 if pred == ans else 0.0)
    return rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="grpo_out")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    target = ["c_attn"] if "gpt2" in args.model else ["q_proj", "v_proj"]
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=target, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        num_generations=args.num_generations,
        max_prompt_length=64,
        max_completion_length=8,
        learning_rate=args.lr,
        logging_steps=5,
        max_steps=args.steps,
        beta=0.05,      # KL 系数，对应 /ttt_step 的 BETA_KL=0.05
        epsilon=0.2,    # PPO/GRPO clip
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=build_dataset(),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[done] saved to {args.out}")


if __name__ == "__main__":
    main()
