# 【源代码｜F-b01-generate】v1.0/01-基础/code/from_scratch/generate.py — 从 checkpoint 采样文本（temperature/top-k/top-p）
"""从 checkpoint 加载 tiny GPT 并采样文本（temperature / top-k / top-p）。

用法
----
    python3 generate.py --ckpt out/micro/last.pt --prompt "theorem " --max-new-tokens 120
    python3 generate.py --ckpt out/micro/last.pt --temperature 0.0     # 贪心
    python3 generate.py --ckpt out/micro/last.pt --temperature 0.8 --top-k 40 --top-p 0.9

对照 AlphaProof：``nanoproof/nanoproof/model.py:492`` ``Transformer.generate``。
"""

from __future__ import annotations

import argparse

import torch

from configs import GPTConfig
from data import CharTokenizer
from model import GPT


# 【F-b01-generate.load_checkpoint｜函数】加载 checkpoint 中的模型与分词器
def load_checkpoint(path: str, device) -> tuple[GPT, CharTokenizer, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    itos = {int(k): v for k, v in ckpt["tokenizer"].items()}
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.pad_token, tok.eos_token = "<pad>", "<eos>"
    tok.itos = itos
    tok.stoi = {c: i for i, c in itos.items()}
    tok.pad_id = tok.stoi["<pad>"]
    tok.eos_id = tok.stoi["<eos>"]
    return model, tok, ckpt


# 【F-b01-generate.parse_args｜函数】解析命令行参数
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tiny GPT 文本生成")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径（last.pt/best.pt）")
    p.add_argument("--prompt", default="theorem ")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8, help="<=0 表示贪心")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=None, help="nucleus 采样阈值")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


# 【F-b01-generate.main｜函数】采样并打印生成文本
def main() -> None:
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    model, tok, ckpt = load_checkpoint(args.ckpt, device)
    print(f"[ckpt] step={ckpt.get('iter')} best_val={ckpt.get('best_val')}")
    print(f"[cfg] {ckpt['config']}")
    print(f"[模型] 参数 {model.num_params()/1e6:.3f}M\n")

    g = torch.Generator(device=device.type)
    g.manual_seed(args.seed)
    if args.prompt:
        ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
        print(f"--- PROMPT ---\n{args.prompt}\n--- SAMPLES ---")
    else:
        ids = torch.zeros((1, 1), dtype=torch.long, device=device)  # 从 <pad> 起
    for i in range(args.num_samples):
        out = model.generate(
            ids.clone(),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            generator=g,
        )
        print(f"[sample {i}]\n{tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
