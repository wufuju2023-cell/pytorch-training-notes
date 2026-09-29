# 【源代码｜F-b07-self_play】v1.0/07-MCTS+V1/code/v1_tiny/self_play.py — 自博弈搜索并生成训练样本
# 相关文档：《07-MCTS+V1/01-MCTS原理与V1闭环.md》
"""自博弈：用 MCTS(+tiny 策略/价值网络) 解合成题，并把搜索树变成训练样本。

对照 V1 闭环：搜索产生“访问计数 = 策略目标”，用可验证的剩余步数 = 价值目标，
定期训练网络（策略网络采样 → PUCT → 展开 → 回传 → 更新），下一轮搜索用更新后的网络。
"""

from __future__ import annotations

import random
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from mcts import MCTS, Config
from policy_value_eval import PolicyValueNet, remaining_steps, K


# 【F-b07-self_play.make_problem｜函数】合成一个可验证问题
def make_problem(rng: random.Random, k: int = K):
    hyp = frozenset(i for i in range(k) if rng.random() < 0.7)
    goal = frozenset(i for i in range(k) if rng.random() < 0.5)
    if not goal:
        goal = frozenset([rng.randrange(k)])
    return hyp, goal


# 【F-b07-self_play.expand_fn｜函数】MCTS 展开函数
def expand_fn(state, action):
    hyp, remaining = state
    if action not in remaining or action not in hyp:
        return None
    new = frozenset(remaining - {action})
    return [(hyp, new)] if new else []


# 【F-b07-self_play.search｜函数】对单题跑 MCTS 搜索
def search(net: PolicyValueNet, hyp, goal, sims: int = 32):
    from policy_value_eval import make_evaluator

    cfg = Config(c_init=0.1, c_base=10.0, ps_c=1.0, max_depth=16)
    mcts = MCTS(make_evaluator(net), expand_fn, cfg)
    return mcts.run((hyp, goal), sims)


# 【F-b07-self_play.collect_from_tree｜函数】把搜索树转成训练样本
def collect_from_tree(root) -> list:
    samples, stack = [], [root]
    while stack:
        node = stack.pop()
        if node.children:
            total = sum(c.visit_count for c in node.children.values())
            if total > 0 and isinstance(node.state, tuple):
                dist = {a: c.visit_count / total for a, c in node.children.items()}
                hyp, remaining = node.state
                samples.append({
                    "state": (hyp, remaining),
                    "dist": dist,
                    "value": float(remaining_steps(hyp, remaining)),
                })
            stack.extend(node.children.values())
    return samples


# 【F-b07-self_play.train_step｜函数】用搜索样本更新网络
def train_step(net: PolicyValueNet, opt, samples: list):
    if not samples:
        return 0.0, 0.0
    X = torch.stack([net.encode(*s["state"]) for s in samples])
    logits, v = net(X)
    logp = F.log_softmax(logits, dim=-1)
    terms = []
    for i, s in enumerate(samples):
        acts = list(s["dist"].keys())
        tgt = torch.tensor([s["dist"][a] for a in acts], dtype=torch.float32)
        terms.append(-(tgt * logp[i, acts]).sum())
    policy_loss = torch.stack(terms).mean()
    value_target = torch.tensor([s["value"] for s in samples], dtype=torch.float32)
    value_loss = F.mse_loss(v, value_target)
    loss = policy_loss + value_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(policy_loss), float(value_loss)
