# 03 · SFT（Supervised Fine-Tuning）

目标：把预训练语言模型对齐到"给定 Lean 证明状态，输出下一步 tactic"（容器版则是"给定
证明缺口，输出结构化调用 JSON"）。理论见 [01-SFT原理.md](01-SFT原理.md)，实验见
`../notebooks/03-SFT/N10_状态到tactic_SFT.ipynb`。

## 两版实现对照

| 维度 | `code/from_scratch/sft.py` | `code/with_api/sft_hf.py` |
|---|---|---|
| 模型 | 复用 `01-基础` 的 tinyGPT | HF 基座（默认 `sshleifer/tiny-gpt2`）+ peft LoRA |
| 数据 | 合成 (state,tactic) + 可选 `--data` jsonl | 同 + `datasets` / chat template |
| label mask | 手写 `encode_pair`，prompt 全 `-100` | 先 tokenize prompt 得长度 L，前 L 置 `-100` |
| 训练循环 | 手写：grad-accum / clip / warmup+cosine | `Trainer` 托管 |
| 微调方式 | 全参（tiny 模型） | LoRA（`--lora-r`），可选 4bit QLoRA |
| 评估 | answer token 准确率 + 贪心 exact-match | `Trainer` loss + 生成样例 |
| 适用 | 看清 mask 与 loss 的每个位置 | 贴近 AlphaProof 容器配方 |

## from_scratch 版运行

```bash
cd 03-SFT/code/from_scratch
python3 -m py_compile sft.py
python3 sft.py --config micro --steps 400 --batch-size 16          # CPU 冒烟
python3 sft.py --config tiny --device cuda --amp --steps 3000
python3 sft.py --data /path/to/state_tactic.jsonl --max-samples 2000
```

> 依赖：仅 `torch`；`sys.path` 复用 `../../../01-基础/code/from_scratch` 的 tinyGPT。

## with_api 版运行

```bash
cd 03-SFT/code/with_api
pip install "transformers>=4.46" "peft>=0.13" "datasets>=3.0" accelerate
python3 sft_hf.py --max-steps 30 --batch-size 2                    # CPU 也能跑
python3 sft_hf.py --model-name Qwen/Qwen2.5-0.5B-Instruct --max-steps 100 --amp
python3 sft_hf.py --states-file /path/to/state_tactic.jsonl --lora-r 16
```

## 预期行为

* **loss**：`sft_loss` 从 ~2–4 降到 < 1；因为只对答案 token 计损失，数值比预训练低。
* **answer token 准确率**：内置模板数据上，`micro` 模型几百步内可达 0.8 以上。
* **exact-match**：字符级小模型 + 贪心解码，简单状态可达 0.5 以上；真实 Lean 状态上会低
  很多，这正说明需要更大基座 + 更多数据 + RL。
* **过拟合**：数据少时训练 loss 迅速趋 0，但生成质量不再提升。

## 与 AlphaProof 的对应

* `app/train_sft.py`（DEPRECATED）：LoRA r=16 + HF `Trainer`，思路即 with_api 版。
* 容器 `train_full_supervised.py`：Qwen3-1.7B QLoRA；prompt（指令 +
  `<proof_gap>/<dsl_thinking>/<public_proof_state>`）全 `-100`，只对
  `<answer>{"calls":[...]}</answer>` 计损失。
* `nanoproof/nanoproof/sft.py`：LeanTree（约 260k transitions）token-masked CE。
* 数据获取见 [`../datasets/README.md`](../datasets/README.md)。
