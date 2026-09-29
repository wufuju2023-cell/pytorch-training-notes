"""用 HuggingFace 生态做分词：``tokenizers`` 训练小 BPE + ``PreTrainedTokenizerFast`` 封装。

与 from_scratch 的字符级 ``CharTokenizer`` 对照：
* from_scratch：字符表 = sorted(set(text))，无合并规则，词表 ~ 几十。
* 本文件：byte-level BPE，从 ``vocab_size=512`` 起训练合并规则，能压更短序列，
  与 ``nanoproof/nanoproof/tokenizer.py``（GPT-2 BPE + Lean 专用 token）一脉相承。

依赖（Colab 已自带，本地按需装）：``tokenizers``、``transformers``。

用法
----
    python3 tokenizer_api.py --corpus data/input.txt --vocab-size 512 --out out/tokenizer
    python3 tokenizer_api.py --demo            # 不依赖外部语料，用内置示例文本
"""

from __future__ import annotations

import argparse
import os


def train_bpe(
    text: str,
    vocab_size: int = 512,
    out_dir: str = "out/tokenizer",
    special_tokens: list[str] | None = None,
):
    """训练 byte-level BPE 并保存到 ``out_dir``，返回 ``PreTrainedTokenizerFast``。"""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)
    special_tokens = special_tokens or ["<pad>", "<eos>", "<bos>", "<unk>"]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator([text], vocab_size=vocab_size, special_tokens=special_tokens)
    tok.save_model(out_dir)

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(out_dir, "tokenizer.json"),
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tok.save_pretrained(out_dir)
    return hf_tok


def build_char_tokenizer(text: str):
    """无 ``tokenizers`` 依赖的字符级 fallback，接口尽量贴近 HF tokenizer。

    返回一个轻量对象：``encode`` / ``decode`` / ``vocab_size`` / ``__call__``，
    并带 ``get_vocab``，方便 ``train_hf.py`` 统一切换。
    """
    itos = ["<pad>", "<eos>", "<unk>"] + sorted(set(text))
    stoi = {c: i for i, c in enumerate(itos)}

    class _CharTok:
        pad_token_id = stoi["<pad>"]
        eos_token_id = stoi["<eos>"]
        unk_token_id = stoi["<unk>"]
        vocab_size = len(itos)

        def encode(self, s: str) -> list[int]:
            return [stoi.get(c, self.unk_token_id) for c in s]

        def decode(self, ids) -> str:
            skip = {self.pad_token_id, self.eos_token_id, self.unk_token_id}
            return "".join(itos[int(i)] for i in ids if int(i) not in skip)

        def get_vocab(self) -> dict:
            return dict(stoi)

    return _CharTok()


DEMO_TEXT = (
    "theorem add_comm (a b : Nat) : a + b = b + a := by induction a with "
    "| zero => simp | succ a ih => rw [Nat.succ_add, Nat.add_succ, ih]"
) * 40


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HF BPE 分词器")
    p.add_argument("--corpus", default=None)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--out", default="out/tokenizer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = DEMO_TEXT if args.demo or not args.corpus else open(args.corpus, encoding="utf-8").read()
    try:
        tok = train_bpe(text, args.vocab_size, args.out)
        sample = "theorem add_comm (a b : Nat)"
        ids = tok.encode(sample)
        print(f"BPE vocab_size={tok.vocab_size}")
        print(f"encode({sample!r}) = {ids}")
        print(f"decode -> {tok.decode(ids)!r}")
        print(f"保存到 {args.out}")
    except ImportError as e:
        tok = build_char_tokenizer(text)
        print(f"缺少 tokenizers/transformers（{e}），回退字符级：vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()
