# 【源代码｜F-b07-mcts】v1.0/07-MCTS+V1/code/mcts.py — 极简 PUCT/OR-AND MCTS
# 相关文档：《07-MCTS+V1/01-MCTS原理与V1闭环.md》
"""极简 PUCT / OR-AND MCTS（教学实现，纯 Python，无依赖）。

对应《07 · MCTS 原理与 V1 闭环》：
* OR 节点：目标是析取（tactic 多选一），取 any；
* AND 节点：目标是合取（多个子目标），取 all，并挂 focus 孩子专攻子目标；
* 选择：score = Q + c(N) * p_a/sum(p) * sqrt(N)/(n+1)；
* 价值变换：valueScore = gamma ** (-1 - v)，v = child.value - stepCost；
* 回传：OR 原样 + reward；AND 取未解孩子的最小值。

evaluator 是“策略/价值网络占位接口”：
    priors, value = evaluator(state)
其中 priors 是 {action: prob}，value 是“预估剩余步数” d（>=0），
内部转成 v = -d。（notebook 里用真实 tiny 模型替换 evaluator。）

toy 环境模拟流程文档 §4 的 P ∧ Q 例子：见模块底部 __main__。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OR, AND = "OR", "AND"


# 【F-b07-mcts.Config｜类】MCTS 配置
@dataclass
class Config:
    c_init: float = 0.001
    c_base: float = 3200.0
    gamma: float = 0.99
    unvisited_score: float = 0.0   # V1 用 0；论文用 V(parent) - penalty
    c_and: float = 1.0
    ps_c: float = 0.01             # progressive sampling C
    ps_alpha: float = 0.6
    max_depth: int = 64


# 【F-b07-mcts.Node｜类】搜索树节点（OR/AND）
@dataclass
class Node:
    state: object
    to_play: str = OR
    parent: "Node | None" = None
    action: object = None
    prior: float = 0.0
    reward: float = 0.0            # tactic 边 = -1，focus 边 = 0
    visit_count: int = 0
    value_sum: float = 0.0
    evaluations: int = 0
    is_solved: bool = False
    terminal: bool = False
    children: dict = field(default_factory=dict)

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def prior_sum(self) -> float:
        return sum(c.prior for c in self.children.values()) or 1.0

    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.terminal


# 【F-b07-mcts.MCTS｜类】通用 PUCT 搜索
class MCTS:
    """通用 PUCT。expand_fn(state, action) -> 子状态列表（[] 表示关闭目标，None 表示非法）。"""

    def __init__(self, evaluator, expand_fn, config: Config | None = None):
        self.evaluator = evaluator
        self.expand_fn = expand_fn
        self.cfg = config or Config()

    # -- 选择 --------------------------------------------------------------
    def _value_score(self, parent: Node, child: Node) -> float:
        if child.visit_count > 0:
            v = child.reward + child.value()
        else:
            v = self.cfg.unvisited_score
        score = self.cfg.gamma ** (-1.0 - v)
        if parent.to_play == AND:
            score = 1.0 - score
            if child.is_solved:
                score = -1e9
        return score

    def _ucb(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + self.cfg.c_base + 1) / self.cfg.c_base) \
            + self.cfg.c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        if parent.to_play == AND:
            pb_c *= self.cfg.c_and
        prior_score = pb_c * child.prior / parent.prior_sum()
        return self._value_score(parent, child) + prior_score

    def _select(self, node: Node) -> Node:
        return max(node.children.values(), key=lambda c: self._ucb(node, c))

    # -- 展开 --------------------------------------------------------------
    def _expand(self, node: Node) -> float:
        node.evaluations += 1
        priors, depth = self.evaluator(node.state)
        node.terminal = depth <= 0 and not priors
        for action, p in priors.items():
            children = self.expand_fn(node.state, action)
            if children is None:          # 非法 tactic
                continue
            if len(children) == 0:        # tactic 关闭目标 -> 终止
                child = Node(state=(), action=action, parent=node, prior=p,
                             reward=-1.0, to_play=OR, terminal=True, is_solved=True)
                node.children[action] = child
                node.is_solved = True
            elif len(children) == 1:
                node.children[action] = Node(state=children[0], action=action,
                                             parent=node, prior=p, reward=-1.0, to_play=OR)
            else:                         # AND 节点 + focus 孩子
                and_node = Node(state=tuple(children), action=action, parent=node,
                                prior=p, reward=-1.0, to_play=AND)
                for i, st in enumerate(children):
                    and_node.children[i] = Node(state=st, action=i, parent=and_node,
                                                prior=1.0 / len(children), reward=0.0, to_play=OR)
                node.children[action] = and_node
        # 更新 solved
        if node.to_play == OR:
            node.is_solved = node.is_solved or any(c.is_solved for c in node.children.values())
        else:
            node.is_solved = bool(node.children) and all(c.is_solved for c in node.children.values())
        if not node.children:
            node.terminal = True
        return -float(depth)              # 价值尺度：v = -剩余步数

    # -- 回传 --------------------------------------------------------------
    def _min_unsolved(self, node: Node) -> float:
        val = 1.0
        for c in node.children.values():
            if not c.is_solved and c.visit_count > 0:
                val = min(val, c.value())
        return val

    def _backprop(self, path: list[Node], value: float) -> None:
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.value_sum += value
            node.visit_count += 1
            if node.terminal:
                node.is_solved = True
            elif node.to_play == AND:
                node.is_solved = bool(node.children) and all(
                    c.is_solved for c in node.children.values())
            else:
                node.is_solved = any(c.is_solved for c in node.children.values())
            if i == 0:
                break
            parent = path[i - 1]
            if parent.to_play == AND:
                value = self._min_unsolved(parent)
            else:
                value = node.reward + value

    def _progressive(self, node: Node) -> bool:
        return (node.to_play == OR and node.evaluations <= self.cfg.ps_c * node.visit_count ** self.cfg.ps_alpha)

    # -- 主循环 ------------------------------------------------------------
    def run(self, root_state, n_sims: int = 64) -> Node:
        root = Node(state=root_state, to_play=OR)
        for _ in range(n_sims):
            node, path = root, [root]
            while node.expanded and node.children and not self._progressive(node):
                node = self._select(node)
                path.append(node)
                if len(path) > self.cfg.max_depth:
                    break
            value = self._expand(node)
            self._backprop(path, value)
            if root.is_solved:
                break
        return root

    @staticmethod
    def best_action(root: Node):
        if not root.children:
            return None
        return max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]


# ---------------------------------------------------------------------------
# toy 环境：复刻流程文档 §4 的 P ∧ Q（constructor 分裂 -> 分别 exact hP / exact hQ）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TACTICS = {
        "P&Q": {"constructor": ["P", "Q"]},
        "P": {"exact hP": []},
        "Q": {"exact hQ": []},
    }
    PRIORS = {
        "P&Q": {"constructor": 0.5, "assumption": 0.3, "left": 0.2},
        "P": {"exact hP": 0.7, "simp": 0.3},
        "Q": {"exact hQ": 0.8, "simp": 0.2},
    }

    def evaluator(state):
        return PRIORS.get(state, {}), {"P&Q": 2.0, "P": 1.0, "Q": 1.0}.get(state, 0.0)

    def expand_fn(state, action):
        return TACTICS.get(state, {}).get(action, None)

    root = MCTS(evaluator, expand_fn).run("P&Q", n_sims=32)
    print("solved =", root.is_solved, "| 访问数 =", root.visit_count)
    for a, c in root.children.items():
        print(f"  {a:12s} visits={c.visit_count:3d} value={c.value():+.3f} solved={c.is_solved}")
