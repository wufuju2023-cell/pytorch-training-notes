# PyTorch 和模型训练源码学习 · v1.0

目标：从零（数学 + 代码）彻底读懂 AlphaProof 的一整套 Python 训练/服务脚本，并能自己改。

## 怎么用

1. 先读 [`00-学习路线图.md`](00-学习路线图.md)：全流程路线、阶段验收、与 AlphaProof 的映射。
2. 查 [`00-资料清单.md`](00-资料清单.md)：外部教程/仓库/数据集/既有数学文档索引。
3. 按阶段学：每个主题都是 **理论 MD + 两版代码（from_scratch / with_api）+ notebook 小实验**。
4. 每个 notebook 都在 Colab 可跑（CPU 优先），并对应到 AlphaProof 的具体文件与函数。

## 目录结构

```
v1.0/
├─ 00-学习路线图.md / 00-资料清单.md / README.md
├─ 01-基础/            神经网络·反传·优化器·Attention/Transformer·分词器·训练循环·初始化调度
├─ 02-预训练/          语言建模目标 + tiny 预训练复刻（nanoproof/pretrain.py 对照）
├─ 03-SFT/             证明步骤指令微调（state → tactic，label mask）
├─ 04-价值头/          scalar / 64-bin 价值头 + 校准（REAL-Prover value head 对照）
├─ 05-LoRA/            LoRA / QLoRA 从零 + PEFT（policy_server 注入对照）
├─ 06-RL-RLVR/         REINFORCE/PPO/GRPO/RLVR + RTTT（/ttt_step 对照）
├─ 07-MCTS+V1/         极简 MCTS(PUCT, OR/AND) + V1 闭环 tiny 复刻
├─ 08-V1-1/            tool-calling agent + opencode 桥接 + GPU runtime
├─ 09-源码精读/        按 GPU 版本逐文件精读全部 AlphaProof 脚本
├─ notebooks/          可运行小实验（N01–N17，按阶段分目录）
├─ datasets/           Lean 数据集小切片准备脚本与说明
├─ tiny/               极小模型配置（参数量/显存/耗时）
└─ 资源/               关键外部资料的镜像（可选）
```

## 约定

- 代码两版：`code/from_scratch/`（纯 PyTorch，教学向）与 `code/with_api/`（transformers / peft / trl / accelerate）。
- 规模：from_scratch 用 1–20M 参数 tiny 模型；with_api 用小基座 + LoRA / 4bit QLoRA。
- 文档中文，公式遵循 KaTeX/GFM 规范（见 skill `md-latex-rule`）。
