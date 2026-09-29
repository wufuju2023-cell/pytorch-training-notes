# 05-LoRA · 代码与实验说明

配套理论：[`01-LoRA原理.md`](01-LoRA原理.md)。本目录给出 **两版** LoRA 实现：
`from_scratch/`（手写 LoRALinear + 注入 tinyGPT）与 `with_api/`（PEFT + 4-bit
QLoRA），外加 notebook `notebooks/05-LoRA/N12_LoRA从零与PEFT.ipynb`。

## 目录

```
05-LoRA/
├─ 01-LoRA原理.md
├─ README.md
├─ code/
│  ├─ from_scratch/
│  │  ├─ lora.py             # LoRALinear + TinyGPT + inject_lora + 参数统计
│  │  └─ finetune_lora.py    # 冻结 vs 全参 vs LoRA：参数/显存/效果
│  └─ with_api/
│     ├─ peft_lora.py        # HF + peft 注入 LoRA
│     └─ qlora_bnb.py        # 4-bit nf4 + double quant + LoRA
```

## 两版对照

| 维度 | from_scratch | with_api |
| --- | --- | --- |
| LoRA 层 | 手写 `LoRALinear`（A/B/缩放/dropout/merge） | `peft.LoraConfig` + `get_peft_model` |
| 目标模型 | 自建 `TinyGPT`（~1-2M） | HF 基座（tiny-gpt2 等） |
| 量化 | 无 | `bitsandbytes` nf4 + double quant（QLoRA） |
| 关注指标 | 参数量、显存峰值、冻结/全参/LoRA 对比 | 可训练占比、adapter 保存、4bit 显存 |
| 关键对应 | `real_backend.py:24,117-128` | `policy_server.py:112` |

## 接口与数据结构

`lora.py`：

- `LoRALinear(base, r, alpha, dropout)`：`forward` 为
  `base(x) + (alpha/r) * dropout(x) @ Aᵀ @ Bᵀ`；`merge()` 把增量并回 base。
- `TinyGPT(...)`：命名与 LLaMA 对齐（`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`）。
- `inject_lora(model, target_names, r, alpha, dropout)`。
- `count_parameters(model, trainable_only)`、`lora_parameters(model)`。

`finetune_lora.py`：合成任务（逆序复制），三组对照，输出 JSON：
`mode / total_params / trainable_params / trainable_pct / final_loss / peak_mem_MB / seconds`。

## 指标 / 对照

- **参数量**：LoRA 通常占总参数 0.1%–1%（`lora.py` 自检会打印占比）。
- **显存**：`torch.cuda.max_memory_allocated`，对比 full vs LoRA vs QLoRA。
- **效果**：相同步数下的 loss 曲线（notebook 中画）。
- **merge 正确性**：`LoRALinear.merge()` 前后输出应一致（误差 < 1e-5）。

## 运行

```bash
# from_scratch
python code/from_scratch/lora.py                       # 自检：参数占比 + 形状
python code/from_scratch/finetune_lora.py --mode all --steps 200

# with_api
python code/with_api/peft_lora.py --model sshleifer/tiny-gpt2 --steps 20
python code/with_api/qlora_bnb.py  --model <llama-7b> --steps 20   # 需 GPU + bnb
```

## 与 AlphaProof 的对应

- `app/policy_server.py:112-116`：`r=16, alpha=32, dropout=0.02, bias=none`，
  `target_modules` 为全部 attn+mlp，`init_lora_weights=True`；
- `gpu_runtime/real_backend.py:24` 的 `TARGET_MODULES` 与两版代码默认目标一致；
- `real_backend.py:242` 的 per-session adapter、`:623` 的"adapter+value head 共训"
  对应"在线持续学习时冻结基座"的场景。

## 未决点

- 本地未装 torch，`from_scratch` 只做 `py_compile`；数值/显存结果由 lead 在云
  GPU 执行 notebook 时产生。
- QLoRA 依赖 bitsandbytes + GPU，脚本在无 GPU 时给出明确提示而非报错。
