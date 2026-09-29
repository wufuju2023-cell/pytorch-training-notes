# 【源代码｜F-b01-data】v1.0/01-基础/code/from_scratch/data.py — 字符级数据管线与 block 采样
# 相关文档：《01-基础/05-训练循环与数据管线.md》
"""字符级数据管线 + block 采样（纯 PyTorch，无第三方依赖）。

职责
----
1. ``CharTokenizer``：从语料里建字符表，``encode`` / ``decode``。
2. ``load_text``：优先读本地文件；``--download`` 时可拉 tiny-shakespeare；
   否则回退到内置的“数学证明风格”小语料（离线可用）。
3. ``CharDataset``：把长文本切成一维 token 流，按 block 随机采样 ``(x, y)``
   其中 ``y = x`` 右移一位（next-token prediction）。
4. ``get_batch``：从训练/验证集取一个 batch，支持 ``pin_memory``。

对应 AlphaProof
--------------
* 数据字段（state_max_len / tactic_max_len / max_seq_len）见
  ``nanoproof/nanoproof/common.py:70`` ``GlobalConfig``。
* 预训练里的 block 采样思想见 ``nanoproof/nanoproof/pretrain.py``。
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# 内置离线语料：一段“像数学证明”的合成文本，够字符级模型学出结构
# ---------------------------------------------------------------------------
BUILTIN_CORPUS = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_add, Nat.add_succ, ih]
theorem add_assoc (a b c : Nat) : (a + b) + c = a + (b + c) := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_add, ih]
theorem mul_comm (a b : Nat) : a * b = b * a := by
  induction a with
  | zero => simp
  | succ a ih => rw [Nat.succ_mul, Nat.mul_succ, ih]
theorem mul_assoc (a b c : Nat) : (a * b) * c = a * (b * c) := by
  induction a with
  | zero => simp
  | succ a ih => simp [Nat.succ_mul, ih]
theorem zero_add (n : Nat) : 0 + n = n := by simp
theorem add_zero (n : Nat) : n + 0 = n := by simp
theorem one_mul (n : Nat) : 1 * n = n := by simp
theorem mul_one (n : Nat) : n * 1 = n := by simp
theorem list_append_nil (xs : List α) : xs ++ [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [ih]
theorem map_map (f : α → β) (g : β → γ) (xs : List α) :
    (xs.map f).map g = xs.map (g ∘ f) := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [ih]
example (f : α → β) : Function.Injective f ↔ ∀ a b, f a = f b → a = b := by
  constructor
  · intro h a b hab; exact h hab
  · intro h a b hab; exact h hab
""".strip()


# ---------------------------------------------------------------------------
# 字符级 tokenizer
# ---------------------------------------------------------------------------
# 【F-b01-data.CharTokenizer｜类】字符级分词器（stoi/itos + BOS/EOS 约定）
class CharTokenizer:
    """极简字符级分词器：``stoi`` / ``itos`` 两张表 + BOS/EOS 约定。"""

    def __init__(self, chars: list[str], pad_token: str = "<pad>", eos_token: str = "<eos>"):
        self.pad_token = pad_token
        self.eos_token = eos_token
        itos = [pad_token, eos_token] + sorted(set(chars))
        self.itos = {i: c for i, c in enumerate(itos)}
        self.stoi = {c: i for i, c in self.itos.items()}
        self.pad_id = self.stoi[pad_token]
        self.eos_id = self.stoi[eos_token]

    @classmethod
    def from_text(cls, text: str, **kwargs) -> "CharTokenizer":
        return cls(list(text), **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ids = [self.stoi[c] for c in text if c in self.stoi]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.eos_id}
        return "".join(
            self.itos[int(i)] for i in ids if not (skip_special and int(i) in specials)
        )

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, pad_token: str = "<pad>", eos_token: str = "<eos>") -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        itos = {int(k): v for k, v in data["itos"].items()}
        chars = [itos[i] for i in sorted(itos) if itos[i] not in (pad_token, eos_token)]
        return cls(chars, pad_token=pad_token, eos_token=eos_token)


# ---------------------------------------------------------------------------
# 语料加载
# ---------------------------------------------------------------------------
# 【F-b01-data.load_text｜函数】取语料文本（本地/下载/内置回退）
def load_text(path: str | None = None, download: bool = False) -> str:
    """取语料文本。

    * ``path`` 存在 -> 直接读；
    * 否则若 ``download`` -> 拉 tiny-shakespeare（Karpathy）；
    * 否则 -> 内置 ``BUILTIN_CORPUS``。
    """
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    if download:
        url = (
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:  # noqa: S310
                return resp.read().decode("utf-8")
        except Exception as e:  # 网络不通就回退
            print(f"[data] 下载失败（{e}），回退内置语料")
    return BUILTIN_CORPUS


# ---------------------------------------------------------------------------
# Dataset / block 采样
# ---------------------------------------------------------------------------
# 【F-b01-data.Batch｜类】一个训练 batch 的容器
@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


# 【F-b01-data.CharDataset｜类】按 block_size 切块的 next-token 数据集
class CharDataset:
    """把 1D token 流按 ``block_size`` 切块的 next-token 数据集。

    ``__getitem__(i)`` 返回第 i 个 block 的 ``(x, y)``；``y`` 是 ``x`` 右移一位。
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, i: int) -> Batch:
        chunk = self.data[i : i + self.block_size + 1]
        return Batch(x=chunk[:-1], y=chunk[1:])


# 【F-b01-data.split_train_val｜函数】按比例划分训练/验证集
def split_train_val(ids: torch.Tensor, val_frac: float = 0.1):
    n = len(ids)
    n_val = max(1, int(n * val_frac))
    return ids[:-n_val], ids[-n_val:]


# 【F-b01-data.get_batch｜函数】随机抽一个 (x, y) batch
def get_batch(
    dataset: CharDataset,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机抽一个 batch（有放回）。返回 ``x, y``，形状 ``(B, T)``。"""
    n = len(dataset)
    if n <= 0:
        raise ValueError("语料太短，一个 block 都切不出来；请用更大的语料")
    ix = torch.randint(n, (batch_size,), generator=generator)
    xs = torch.stack([dataset[i].x for i in ix])
    ys = torch.stack([dataset[i].y for i in ix])
    return xs.to(device), ys.to(device)


# 【F-b01-data.prepare｜函数】文本→tokenizer→train/val 张量一站式准备
def prepare(
    text_path: str | None = None,
    download: bool = False,
    tokenizer_path: str | None = None,
    block_size: int = 128,
    val_frac: float = 0.1,
):
    """一站式准备：文本 -> tokenizer -> train/val tensor。

    返回 ``(tokenizer, train_data, val_data, train_ids, val_ids)``。
    """
    text = load_text(text_path, download=download)
    tokenizer = (
        CharTokenizer.load(tokenizer_path)
        if tokenizer_path and os.path.exists(tokenizer_path)
        else CharTokenizer.from_text(text)
    )
    ids = torch.tensor(tokenizer.encode(text, add_eos=False), dtype=torch.long)
    train_ids, val_ids = split_train_val(ids, val_frac=val_frac)
    train_data = CharDataset(train_ids, block_size)
    val_data = CharDataset(val_ids, block_size)
    return tokenizer, train_data, val_data, train_ids, val_ids


if __name__ == "__main__":
    tok, train_ds, val_ds, train_ids, val_ids = prepare(block_size=64)
    print(f"vocab_size={tok.vocab_size}, train_tokens={len(train_ids)}, val_tokens={len(val_ids)}")
    x, y = get_batch(train_ds, batch_size=2)
    print("x[0][:20] =", tok.decode(x[0][:20].tolist()))
    print("y[0][:20] =", tok.decode(y[0][:20].tolist()))
