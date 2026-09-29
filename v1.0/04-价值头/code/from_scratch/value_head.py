"""从零实现价值头（scalar 回归 + 64-bin 分类）与校准工具。

对应理论：``04-价值头/01-价值头原理.md``
对照源码：``reap-new-update-model/app/value_head.py``、
``v1-1-agentic-tool/gpu/gpu_runtime/categorical_search_backend.py``。

本文件只依赖 PyTorch，CPU 即可运行；也提供合成/缓存特征的加载接口。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# 1. 语义工具：折扣回报 / proof-depth / two-hot
# ---------------------------------------------------------------------------
def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> list[float]:
    """后向计算折扣回报 ``G_t = r_t + gamma * G_{t+1}``，裁剪到 [-1, 1]。"""
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = max(-1.0, min(1.0, running))
    return out


def proof_depth_to_target(depth: float, max_depth: int = 64) -> float:
    """剩余证明深度 -> 标量目标 ``-min(depth, max_depth) / max_depth``。

    终止 ``depth = 0`` 给 0（最好），``depth >= max_depth`` 饱和到 -1。
    """
    if depth < 0:
        raise ValueError("proof_depth must be non-negative")
    return -min(float(depth), float(max_depth)) / float(max_depth)


def distance_two_hot(distance: float, support_max: int = 64) -> Tensor:
    """把（可小数的）距离投影到相邻两桶，返回长度 ``support_max`` 的软标签。

    桶 ``support_max`` 饱和：``distance >= support_max`` 时概率质量全给最后一桶。
    """
    if not math.isfinite(distance) or distance < 1:
        raise ValueError("distance must be finite and >= 1")
    clipped = min(float(distance), float(support_max))
    lower, upper = math.floor(clipped), math.ceil(clipped)
    upper_w = clipped - lower
    lower_w = 1.0 - upper_w
    weights = torch.zeros(support_max, dtype=torch.float32)
    weights[lower - 1] += lower_w
    if upper_w:
        weights[upper - 1] += upper_w
    return weights


def two_hot_ce_loss(logits: Tensor, distance: float, support_max: int = 64) -> Tensor:
    """soft-target 交叉熵损失（target 为 two-hot 分布）。"""
    target = distance_two_hot(distance, support_max).to(logits.device)
    log_prob = F.log_softmax(logits, dim=-1)
    return -(target * log_prob).sum(-1).mean()


def expected_distance(logits: Tensor) -> Tensor:
    """分类头解码：``E[k] = sum_k k * softmax(logits)_k``，k = 1..num_bins。"""
    probs = F.softmax(logits, dim=-1)
    support = torch.arange(1, logits.shape[-1] + 1, device=logits.device,
                           dtype=probs.dtype)
    return (probs * support).sum(-1)


# ---------------------------------------------------------------------------
# 2. 两种价值头（对照 app/value_head.py:35 与 categorical_search_backend.py:41）
# ---------------------------------------------------------------------------
class ScalarValueHead(nn.Module):
    """Linear(H,256) -> SiLU -> Linear(256,1) -> Tanh，输出落在 [-1,1]。"""

    def __init__(self, hidden_size: int = 3584, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim == 3:            # [B, T, H] -> 取最后 token
            hidden_states = hidden_states[:, -1, :]
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be [B,H] or [B,T,H]")
        # 头保持 fp32，即使 backbone 跑 bf16（app/value_head.py:63）
        return self.mlp(hidden_states.float()).squeeze(-1)


class CategoricalValueHead(nn.Module):
    """Linear(H,256) -> SiLU -> Linear(256,num_bins)，输出 num_bins 个 bin logits。

    对应 gpu_runtime/categorical_search_backend.py 的
    ``linear-3584-silu-256-linear-64`` 与 nanoproof 的 ``<|bin_XX|>`` 头。
    """

    def __init__(self, hidden_size: int = 3584, hidden_dim: int = 256,
                 num_bins: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states[:, -1, :]
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must be [B,H] or [B,T,H]")
        return self.mlp(hidden_states.float())


def two_hot_cross_entropy(logits: Tensor, distances, support_max: int = 64) -> Tensor:
    """对一批距离做 two-hot 软标签，返回平均交叉熵。"""
    if isinstance(distances, Tensor):
        distances = distances.detach().reshape(-1).tolist()
    losses = [two_hot_ce_loss(logits[i], float(d), support_max)
              for i, d in enumerate(distances)]
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 3. 校准（ECE / 可靠性图 / 温度）
# ---------------------------------------------------------------------------
def regression_reliability(pred: Tensor, target: Tensor, n_bins: int = 10):
    """回归价值头的可靠性统计，返回 dict(centers, empirical, counts, ece)。"""
    p = pred.detach().reshape(-1).float()
    y = target.detach().reshape(-1).float()
    lo, hi = float(p.min()), float(p.max())
    if hi - lo < 1e-9:
        hi = lo + 1e-6
    edges = torch.linspace(lo, hi, n_bins + 1)
    centers, empirical, counts, ece = [], [], [], 0.0
    n = max(1, p.numel())
    for b in range(n_bins):
        if b == n_bins - 1:
            mask = (p >= edges[b]) & (p <= edges[b + 1])
        else:
            mask = (p >= edges[b]) & (p < edges[b + 1])
        c = int(mask.sum())
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        counts.append(c)
        if c == 0:
            empirical.append(float("nan"))
            continue
        emp = float(y[mask].mean())
        empirical.append(emp)
        ece += (c / n) * abs(centers[-1] - emp)
    return {"centers": centers, "empirical": empirical,
            "counts": counts, "ece": ece}


def fit_temperature(logits: Tensor, labels: Tensor, steps: int = 200) -> float:
    """学一个标量温度 T，使 softmax(logits/T) 的 NLL 最小（不改排序）。"""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------------------
# 4. 特征数据：真实分片 / 合成
# ---------------------------------------------------------------------------
def synthetic_features(n: int, hidden_size: int = 3584, seed: int = 0,
                       max_depth: int = 64):
    """合成 (features[B,H], depth[B])。前 8 维决定剩余深度，其余为噪声。"""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, hidden_size, generator=g)
    signal = torch.randn(n, 8, generator=g)
    depth = torch.clamp((signal.norm(dim=1) * 8.0).round(), 0, max_depth)
    z[:, :8] = signal
    return z, depth.long()
