# 04-价值头：理论 + 两版代码

价值头（value head）是挂在策略 backbone 最后一个 hidden state 上的小 MLP，估计
"当前证明状态还差多少步、有多大价值"。本章从 RL 价值函数推导到 AlphaProof 的
scalar / 64-bin 两种实现。

- 理论篇：[`01-价值头原理.md`](01-价值头原理.md)
- 代码：
  - `code/from_scratch/` —— 纯 PyTorch
  - `code/with_api/` —— transformers +（可选）peft
- Notebook：`../notebooks/04-价值头/N11_价值头与校准.ipynb`

## 目录

```
04-价值头/
├─ 01-价值头原理.md
├─ README.md
└─ code/
   ├─ from_scratch/
   │  ├─ value_head.py           # 语义工具 + scalar/categorical 头 + 校准
   │  └─ train_value_head.py     # 训练（合成或缓存特征，MSE/Huber + two-hot CE）
   └─ with_api/
      └─ value_head_api.py       # HF hidden states + 小 MLP 头（可选 LoRA 冻结基座）
```

## 两版对照

| 维度 | `from_scratch` | `with_api` |
| --- | --- | --- |
| 特征来源 | 合成 3584 维 / 缓存分片 | `AutoModelForCausalLM` 的 `hidden_states[-1]` |
| 头 | 手写 `ScalarValueHead` / `CategoricalValueHead` | 手写 `ValueHead`（可挂 LoRA） |
| 损失 | MSE/Huber、two-hot 交叉熵 | MSE / cross-entropy |
| 依赖 | 仅 PyTorch | transformers（+peft 可选） |
| 运行 | CPU 秒级 | CPU 可跑 tiny-gpt2 |

> `with_api` 的额外价值：演示"基座冻结 + LoRA adapter + 价值头"如何共存在一个
> 优化器里，对应 `app/policy_server.py:108-116` 与 `gpu_runtime/real_backend.py:245`。

## 数据结构

`train_value_head.py` 支持两种输入：

1. **缓存特征分片**（JSONL，200 行/分片也可）：

   ```json
   {"feature": [0.1, -0.2, 0, 0], "proof_depth": 7}
   {"hidden":  [0.1, -0.2, 0, 0], "value_target": -0.11}
   ```

   `proof_depth` 是非负剩余步数；`value_target` 是 nanoproof 风格负深度，两者等价。
2. **合成特征**：`value_head.synthetic_features(n, hidden_size)`，前 8 维决定深度。

标签映射（对应 `app/value_head.py:155`）：

$$
y = -\frac{\min(d,\,64)}{64}.
$$

分类头内部再用 `distance_two_hot(d)` 把距离 $d$ 投到 64 桶。

## 指标与校准

- **scalar**：MSE/Huber、MAE、可靠性图 + ECE（`regression_reliability`）。
- **categorical**：two-hot 交叉熵、期望距离 MAE、温度缩放（`fit_temperature`）。
- Notebook N11 会画 reliability diagram、ECE，并画 `depth -> value` 曲线。

## 运行

```bash
cd code/from_scratch
python3 train_value_head.py --mode both --steps 300
python3 train_value_head.py --mode scalar --loss mse --n 4000
python3 train_value_head.py --mode categorical --data /path/to/feat.jsonl

cd ../with_api
python3 value_head_api.py --model sshleifer/tiny-gpt2 --steps 20
python3 value_head_api.py --model sshleifer/tiny-gpt2 --use-lora --num-bins 64
```

## 未决点 / 注意

- 本地未安装 torch，代码只做了 `python3 -m py_compile` 语法校验；数值与 notebook
  执行由 lead 在云 GPU（torch 2.12+rocm7.2）统一跑。
- 备份的特征目录 `fullv3-train-features/` 实测为空，代码已回退到合成特征；如需真实
  训练，把真实分片路径传给 `--data`。
- `with_api` 里 LoRA 的 `target_modules` 随 backbone 命名不同（gpt2 用 `c_attn`，
  LLaMA 用 `q_proj` 等），生产应像 `real_backend.py:TARGET_MODULES` 一样显式指定。
