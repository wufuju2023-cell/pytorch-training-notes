"""tiny GPT 的可复现配置（纯 PyTorch 版）。

设计原则
--------
* 字段命名与 nanoproof 的 ``NetworkConfig`` 对齐（见
  ``nanoproof/nanoproof/model.py:32``）：
  ``sequence_len / vocab_size / n_layer / n_head / n_kv_head / n_embd``。
  另外补上教学需要的 ``head_dim / dropout / use_rope`` 等派生字段。
* 只提供两档：``micro``（约 0.3M 参数，CPU 秒级）与 ``tiny``（约 3M 参数，
  CPU 分钟级 / 单卡 GPU 秒级）。都远小于 20M 上限。
* 配置里所有数字都是可以直接复现的，参数量估算见 ``README.md`` 与
  ``tiny/README.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class GPTConfig:
    """tiny GPT 超参数。

    与 nanoproof ``NetworkConfig`` 的对应关系：

    ==================  ==========================  ====================
    本字段              nanoproof 字段              说明
    ==================  ==========================  ====================
    sequence_len        sequence_len                上下文长度 T
    vocab_size          vocab_size                  词表大小
    n_layer             n_layer                     Transformer block 数
    n_head              n_head                      query 头数
    n_kv_head           n_kv_head                   KV 头数（GQA，<n_head）
    n_embd              n_embd                      隐藏维度 C
    ==================  ==========================  ====================

    额外教学字段（nanoproof 把它们硬编码在别处）：

    * ``head_dim``：``n_embd // n_head``。这里允许显式给出，方便做
      ``head_dim != n_embd // n_head`` 的消融（经典 Transformer 允许）。
    * ``multiple_of``：SwiGLU 中间维度向上取整到该倍数（对齐 LLaMA）。
    * ``use_rope``：False 时退回可学习绝对位置嵌入，用于 N05 消融。
    * ``tie_weights``：embedding 与 lm_head 权重共享（nanoproof 为 untied，
      这里默认 tied，教学上更能看到参数量的变化）。
    """

    sequence_len: int = 128
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 128
    head_dim: int | None = None
    multiple_of: int = 32
    dropout: float = 0.0
    use_rope: bool = True
    tie_weights: bool = True
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.n_embd % self.n_head != 0:
                raise ValueError("n_embd 必须能被 n_head 整除（或显式给出 head_dim）")
            self.head_dim = self.n_embd // self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("GQA 要求 n_head % n_kv_head == 0")
        if self.n_kv_head > self.n_head:
            raise ValueError("GQA 要求 n_kv_head <= n_head")

    @property
    def q_proj_dim(self) -> int:
        return self.n_head * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    def estimate_params(self) -> dict[str, int]:
        """逐项估算参数量（不含位置嵌入，RoPE 无参数）。"""
        emb = self.vocab_size * self.n_embd
        lm_head = 0 if self.tie_weights else self.vocab_size * self.n_embd
        # 每个 block：q/k/v/o + SwiGLU(gate, up, down) + 2 个 RMSNorm 权重
        inter = _swiglu_hidden(self.n_embd, self.multiple_of)
        per_block = (
            self.n_embd * self.q_proj_dim
            + self.n_embd * self.kv_proj_dim * 2
            + self.q_proj_dim * self.n_embd
            + self.n_embd * inter * 3
            + 2 * self.n_embd
        )
        blocks = per_block * self.n_layer
        pos = 0 if self.use_rope else self.sequence_len * self.n_embd
        final_norm = self.n_embd
        total = emb + lm_head + blocks + pos + final_norm
        return {
            "embedding": emb,
            "lm_head": lm_head,
            "blocks": blocks,
            "pos_embed": pos,
            "final_norm": final_norm,
            "total": total,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def _swiglu_hidden(n_embd: int, multiple_of: int) -> int:
    """SwiGLU 中间维度：8/3 * C 后向上取整到 multiple_of（对齐 LLaMA）。"""
    hidden = int(8 * n_embd / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


# ---------------------------------------------------------------------------
# 两档预设
# ---------------------------------------------------------------------------

MICRO = GPTConfig(
    sequence_len=64,
    vocab_size=64,          # 便于字符级玩具语料
    n_layer=2,
    n_head=4,
    n_kv_head=2,            # GQA: 4 query heads share 2 kv heads
    n_embd=64,
    multiple_of=32,
    dropout=0.0,
)

TINY = GPTConfig(
    sequence_len=128,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=2,
    n_embd=128,
    multiple_of=32,
    dropout=0.0,
)

# 兼容简短别名
configs: dict[str, GPTConfig] = {"micro": MICRO, "tiny": TINY}


def get_config(name: str) -> GPTConfig:
    """按名字取配置，返回**深拷贝**，避免训练脚本改坏全局预设。"""
    import copy

    key = name.lower()
    if key not in configs:
        raise KeyError(f"未知配置 {name!r}，可选：{list(configs)}")
    return copy.deepcopy(configs[key])


if __name__ == "__main__":
    for name, cfg in configs.items():
        p = cfg.estimate_params()
        print(f"[{name}] {cfg}")
        print(f"    预估参数量：{p['total'] / 1e6:.3f}M  {p}")
