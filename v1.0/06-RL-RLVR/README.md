# 06 · 后训练 RL / RLVR

本目录讲清楚“用奖励训练证明策略”的完整链条。

## 文件

| 文件 | 内容 |
|---|---|
| [`01-RL与RLVR原理.md`](01-RL与RLVR原理.md) | MDP/策略梯度/REINFORCE/优势、PPO clip、GRPO 组内优势、RLVR、KL、on-policy 稳定性、RTTT/TTRL；对照 `/ttt_step`、`gpu_runtime learn()`、`nanoproof/rl.py` |
| [`code/from_scratch/grpo.py`](code/from_scratch/grpo.py) | 纯 PyTorch 最小 GRPO（组采样 + 相对优势 + clip + KL），任务=可验证加法 |
| [`code/with_api/trl_grpo.py`](code/with_api/trl_grpo.py) | `trl.GRPOTrainer` + PEFT/LoRA 版 RLVR |
| notebooks `N13_GRPO最小实现`、`N14_RLVR可验证奖励` | 可运行实验 |

## 一句话路线

预训练 → SFT（03）→ **RL/RLVR 用可验证奖励强化策略** → 价值头（04）为搜索提供 $V(s)$ → MCTS（07）把策略+价值变成搜索。

## 与 AlphaProof 的对照

- 在线 RTTT 一步：`app/policy_server.py:416` `ttt_step`，损失
  $`-r(\log p-\log p_{\text{old}}) + \beta_{\mathrm{KL}}(\log p-\log p_{\text{old}})^2 + c_v\,\mathrm{MSE}`$，
  $\beta_{\mathrm{KL}}=0.05$。
- 各 GPU 后端：`gpu_runtime/verified_backend.py:216`、`qwen35_backend.py:464`、`search_backend.py:276`、`real_backend.py:569`、`toy_backend.py:74` 的 `learn()`；
  混合目标 `mixed_objective.py`（SFT + replay + 64-bin 价值分类）。
- 离线回放 RL：`nanoproof/nanoproof/rl.py:983-1003`，正样本 token-mean CE + 负样本 unlikelihood，
  `loss = positive_loss + unlikelihood_weight * negative_loss`。

## 运行

```bash
python3 code/from_scratch/grpo.py --steps 300
```

要求：`torch`（CPU 即可）。CPU 上数百步、秒级到分钟级。
`code/with_api/trl_grpo.py` 需要 `trl transformers peft datasets`，默认用 tiny 模型，Colab 可跑。

## 思考题

见 `01-RL与RLVR原理.md` §12。
