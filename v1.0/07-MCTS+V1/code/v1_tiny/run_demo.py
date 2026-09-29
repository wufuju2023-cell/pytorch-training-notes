# 【源代码｜F-b07-run_demo】v1.0/07-MCTS+V1/code/v1_tiny/run_demo.py — V1 闭环 tiny demo（训练→搜索→轨迹）
# 相关文档：《07-MCTS+V1/01-MCTS原理与V1闭环.md》
"""V1 闭环 tiny demo：训练策略/价值 -> MCTS 搜索 -> 打印闭环轨迹。

运行（CPU 即可，分钟级以内）：
    python3 run_demo.py --iters 5 --sims 24
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from policy_value_eval import PolicyValueNet, remaining_steps, K
from self_play import make_problem, search, collect_from_tree, train_step


# 【F-b07-run_demo.evaluate｜函数】评估当前网络解出率
def evaluate(net, rng, n=40, sims=24):
    solved = 0
    for _ in range(n):
        hyp, goal = make_problem(rng)
        root = search(net, hyp, goal, sims)
        solved += int(root.is_solved)
    return solved / n


# 【F-b07-run_demo.best_path｜函数】从搜索树取最优路径
def best_path(root):
    path = []
    node = root
    while node.children:
        a, node = max(node.children.items(), key=lambda kv: kv[1].visit_count)
        path.append((a, node.visit_count))
        if node.is_solved:
            break
    return path


# 【F-b07-run_demo.main｜函数】闭环 demo 主循环
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--problems", type=int, default=24)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    net = PolicyValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    for it in range(1, args.iters + 1):
        samples = []
        for _ in range(args.problems):
            hyp, goal = make_problem(rng)
            root = search(net, hyp, goal, args.sims)
            samples += collect_from_tree(root)
        pl, vl = train_step(net, opt, samples)
        rate = evaluate(net, rng, n=32, sims=args.sims)
        print(f"iter {it} | samples {len(samples):4d} | policy_loss {pl:.3f} | "
              f"value_loss {vl:.3f} | solve_rate {rate:.2%}")

    hyp, goal = make_problem(rng)
    root = search(net, hyp, goal, args.sims)
    print(f"\n示例: hyp={sorted(hyp)} goal={sorted(goal)} solved={root.is_solved} "
          f"remaining_steps={remaining_steps(hyp, goal)}")
    for a, n in best_path(root):
        print(f"  tactic {a} (visits={n})")


if __name__ == "__main__":
    main()
