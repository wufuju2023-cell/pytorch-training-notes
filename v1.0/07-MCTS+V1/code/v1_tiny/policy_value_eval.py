# 【源代码｜F-b07-policy_value_eval】v1.0/07-MCTS+V1/code/v1_tiny/policy_value_eval.py — tiny 策略/价值网络（MCTS 占位）
# 相关文档：《07-MCTS+V1/01-MCTS原理与V1闭环.md》
"""tiny 策略/价值网络——V1 闭环 demo 的“策略网络占位”。

合成任务（可验证）：有 K 个原子命题。每个问题给定
  * ``hyp``：可证明的原子集合（相当于上下文里的假设）；
  * ``goal``：需要证明的原子合取。
动作 ``i`` 表示尝试关闭原子 ``i``：仅当 ``i ∈ hyp ∩ remaining`` 时成功，成功则该原子从 ``remaining`` 移除；
``remaining`` 变空即证明完成（可被确定性验证器判定）。

网络输入 = one-hot(hyp) ⊕ one-hot(remaining) ⊕ [solved]；输出：
  * policy logits（K 个动作的先验）；
  * value：预测“还需几步”（剩余可证原子数），对应 MCTS 的 depth。

这与 V1 里“LLM 输出 tactic 先验 + 价值头输出剩余步数”同构，只是规模极小、CPU 可训。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

K = 6  # 原子数


# 【F-b07-policy_value_eval.PolicyValueNet｜类】tiny 策略/价值网络
class PolicyValueNet(nn.Module):
    def __init__(self, k: int = K, hidden: int = 128):
        super().__init__()
        self.k = k
        self.body = nn.Sequential(
            nn.Linear(3 * k, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, k)
        self.value_head = nn.Linear(hidden, 1)

    def encode(self, hyp, remaining, device="cpu") -> torch.Tensor:
        h = torch.zeros(self.k, device=device)
        if hyp:
            h[list(hyp)] = 1.0
        r = torch.zeros(self.k, device=device)
        if remaining:
            r[list(remaining)] = 1.0
        solved = torch.tensor([1.0 if not remaining else 0.0], device=device)
        return torch.cat([h, r, solved])

    def forward(self, x: torch.Tensor):
        z = self.body(x)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def priors(self, hyp, remaining, valid_actions):
        """返回 ({action: prob}, value) ；value = 预估剩余步数。"""
        logits, v = self(self.encode(hyp, remaining).unsqueeze(0))
        logits = logits[0].clone()
        mask = torch.full_like(logits, -1e9)
        for a in valid_actions:
            mask[a] = 0.0
        logits = logits + mask
        p = F.softmax(logits, dim=-1)
        return {a: float(p[a]) for a in valid_actions}, float(v.item())


# 【F-b07-policy_value_eval.encode_state｜函数】状态→one-hot 输入向量
def encode_state(net: PolicyValueNet, state) -> torch.Tensor:
    hyp, remaining = state
    return net.encode(hyp, remaining)


# 【F-b07-policy_value_eval.remaining_steps｜函数】可验证的剩余步数
def remaining_steps(hyp, remaining) -> int:
    """可验证的剩余步数：remaining 中可证的原子数；若有不可证原子则给 K 惩罚。"""
    if any(i not in hyp for i in remaining):
        return K
    return len(remaining)


# 【F-b07-policy_value_eval.make_evaluator｜函数】state→(priors, depth) 评估器
def make_evaluator(net: PolicyValueNet):
    """给 mcts.MCTS 用的 evaluator：state -> (priors, depth)。"""
    def evaluator(state):
        hyp, remaining = state
        if not remaining:
            return {}, 0.0
        valid = [i for i in remaining if i in hyp]
        if not valid:
            return {}, float(K)          # 死胡同
        priors, depth = net.priors(hyp, remaining, valid)
        return priors, max(depth, 0.0)
    return evaluator


if __name__ == "__main__":
    net = PolicyValueNet()
    hyp, remaining = frozenset({0, 2, 4}), frozenset({0, 2})
    priors, v = net.priors(hyp, remaining, [0, 2])
    print("priors =", {k: round(p, 3) for k, p in priors.items()}, "| value =", round(v, 3))
