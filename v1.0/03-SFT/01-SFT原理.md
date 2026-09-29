# 01 · SFT 原理：指令微调与标签掩码

监督微调（SFT）把预训练的语言模型变成"会执行任务的助手"。在 AlphaProof 里 SFT 的任务
具体化为：**给定 Lean 证明状态（state），输出下一步 tactic**；容器里的 Full caller 则是
给定证明缺口，输出结构化调用 JSON。

> **【文档｜DOC-SFT】**（doccode = `SFT`）｜编号与 Tag 规范见《00-风格与编号规范》。
>
> **前置（预备篇）**：见【定义 PY3.1.1】（虚拟环境）、【定义 PY6.3.3】（`peft` LoRA）。

对照源码：

* `app/train_sft.py`（DEPRECATED 骨架）：LoRA r=16 + HF `Trainer`，在
  `state_tactic_pairs` 上做 token-masked SFT；
* 容器 `train_full_supervised.py`：**Qwen3-1.7B QLoRA**（4bit nf4 + LoRA），输入为
  指令 + `<proof_gap>/<dsl_thinking>/<public_proof_state>`，输出为
  `<answer>{"calls":[...]}</answer>`，prompt 的 label 全为 `-100`；
* `nanoproof/nanoproof/sft.py`：LeanTree（约 260k 转移）上的 token-masked CE。

配套代码：[`code/from_scratch/sft.py`](code/from_scratch/sft.py)（`F-b03-sft`；tiny 模型、手写 mask）
与 [`code/with_api/sft_hf.py`](code/with_api/sft_hf.py)（`F-b03-sft_hf`；HF + peft LoRA + chat template）。配套 notebook：`N-10`。

## 1. 指令微调的目标

**【定义 SFT.1.1｜D-SFT.1.1】（SFT 的条件似然目标）**

预训练优化 $p_\theta(x_t\mid x_{<t})$，对"下一段文本"无差别建模。SFT 只关心"给定指令
或状态 $s$，生成答案 $a$"这一条件：

$$
\mathcal{L}_{\text{SFT}}(\theta)
=-\mathbb{E}_{(s,a)\sim\mathcal{D}}\sum_{t=1}^{|a|}
\log p_\theta(a_t\mid s,a_{<t})
$$

关键：答案 $a$ 的条件里包含指令 $s$，但 $s$ 本身的 token 不计入损失。于是模型被推着
提高"预测答案"的概率，而不会浪费容量去复述指令。

## 2. 数据：(state, tactic) 与证明步骤

**【定义 SFT.2.1｜D-SFT.2.1】（(state, tactic) 数据字段）**

一条 Lean SFT 样本来自一次搜索/证明过程的一个转移（transition）：

| 字段 | 含义 | AlphaProof 对应 |
|---|---|---|
| state | 当前证明状态（目标 + 局部假设） | `public_proof_state` |
| tactic | 这一步实际用出的 tactic | 搜索树的边 |
| thinking | （可选）模型生成的思考 | `dsl_thinking` / `<proof_gap>` |
| answer | 结构化答案 | `{"calls":[...]}` |

`FrenzyMath/state_tactic_pairs`（约 5 万对）与 `ufal/leantree`（约 26 万转移）就是这类
数据；`nanoproof/sft.py` 直接吃 LeanTree。

## 3. 拼接与 label mask（-100）

**【定义 SFT.3.1｜D-SFT.3.1】（拼接格式）**

把 prompt 和答案拼成一个序列：

```
STATE:
<goal>
TACTIC:
<tactic><eos>
```

**【定义 SFT.3.2｜D-SFT.3.2】（标签掩码与掩码交叉熵）**

构造 `labels`：prompt 位置全设 `-100`，答案位置保留真实 token id：

```
input_ids : [t1 t2 ... tk | a1 a2 ... am <eos>]
labels    : [-100 ... -100 | a1 a2 ... am <eos>]
```

`torch.nn.functional.cross_entropy(..., ignore_index=-100)` 会跳过 `-100` 的位置（交叉熵见【定义 1.2.4】，掩码语义见【注 1.7.2】），
于是损失只落在答案 token 上：

$$
\mathcal{L}_{\text{SFT}}(\theta)
=-\frac{1}{\sum_t m_t}\sum_t m_t \log p_\theta(y_t\mid y_{<t}),
\qquad m_t=\mathbb{1}[\text{label}_t\ne -100]
$$

**【注 SFT.3.3｜R-SFT.3.3】（为什么要 mask prompt）**

**为什么要 mask prompt**：

1. 否则模型会花大量容量学"复述状态"，梯度被稀释；
2. prompt 的状态在不同样本间高度重复，计入会让模型偏向预测高频状态片段；
3. 推理时只关心答案，mask 让训练目标与推理目标一致。

**【注 SFT.3.4｜R-SFT.3.4】（右填充）**

**右填充**：causal LM 里右填充安全——填充在真实 token 之后，不影响前面的自注意力；
填充位置的 label 也设 `-100`，不参与损失。见 `code/from_scratch/sft.py` 的 `collate`。

## 4. chat template

**【注 SFT.4.1｜R-SFT.4.1】（chat template 一致性）**

真实模型（Qwen、InternLM 等）都预定义了一段对话模板，SFT 必须与推理时使用的模板一致，
否则特殊 token（`<|im_start|>`、`<|im_end|>` 等）错位，模型行为会崩。用 HF 的
`tokenizer.apply_chat_template(messages, tokenize=False)` 生成字符串，再统一 tokenize。

```python
messages = [
  {"role": "system", "content": "You are a Lean 4 tactic generator."},
  {"role": "user", "content": "Proof state:\n⊢ a + b = b + a"},
  {"role": "assistant", "content": "rw [Nat.add_comm]"},
]
```

**【注 SFT.4.2｜R-SFT.4.2】（掩码定位）**

**掩码定位**：先单独 tokenize "user 部分 + generation prompt" 得到长度 $L$，再把完整
对话 tokenize 后把前 $L$ 个 label 置 `-100`。`code/with_api/sft_hf.py` 的 `SFTDataset`
就是这么做的。若基座没有 chat template（如 `gpt2`），退回手工
`### State: ... ### Tactic: ...` 模板。

## 5. LoRA SFT 与全参 SFT

**【注 SFT.5.1｜R-SFT.5.1】（全参 SFT）**

**全参 SFT**：更新全部 $N$ 个参数。表达力最强，但显存/存储开销大，小数据上更易灾难性
遗忘。

**【定义 SFT.5.2｜D-SFT.5.2】（LoRA 增量参数化）**

**LoRA**（Low-Rank Adaptation，完整推导见《05-LoRA原理》【定义 LORA.2.2】）：冻结基座 $W_0$，只学低秩增量

$$
W=W_0+\Delta W=W_0+\frac{\alpha}{r}BA,\qquad B\in\mathbb{R}^{d\times r},\;A\in\mathbb{R}^{r\times k}
$$

$A$ 高斯初始化、$B=0$ 初始化，于是训练开始时 $\Delta W=0$、模型等价基座。可训练参数从
$N$ 降到约 $r(d+k)$，常减少百倍。$\alpha/r$ 控制增量尺度。

**【注 SFT.5.3｜R-SFT.5.3】（三种微调方式对比）**

| | 全参 SFT | LoRA SFT | QLoRA |
|---|---|---|---|
| 可训练参数 | $N$ | 约 $r(d+k)$ | 同 LoRA |
| 基座精度 | bf16 | bf16 | 4bit nf4（double quant） |
| 显存 | 高 | 中 | 低 |
| AlphaProof 用例 | 少见 | `app/train_sft.py`（r=16, alpha=32） | 容器 `train_full_supervised.py` |

**【注 SFT.5.4｜R-SFT.5.4】（容器 QLoRA 超参）**

容器 QLoRA（量化细节见《05-LoRA原理》【定义 LORA.5.1】）目标模块 q/k/v/o/gate/up/down、dropout 0.05、开梯度检查点、`max-seq 6144`、
`batch1 + grad-accum8`、warmup 5%、epochs5（自动延长 <=2）、early-stop patience 3。

## 6. 过拟合与防遗忘

**【注 SFT.6.1｜R-SFT.6.1】（过拟合、防遗忘与评估）**

- **过拟合信号**：训练 loss 持续降到很低，但验证 loss / 下游可证明率抬头。
- **防过拟合**：早停、dropout、weight decay、LoRA（参数少更稳）、更多数据。
- **防遗忘**：混入预训练数据 replay（域配比见《02-预训练原理》【定义 PT.6.1】）；用小学习率（1e-4 到 2e-4）；只训 LoRA；
  可加 KL 正则约束与基座的距离。
- **评估**：不要只看 loss，要看生成的 tactic 是否合法、是否推进证明。

## 7. 与 AlphaProof 源码对照

**【注 SFT.7.1｜R-SFT.7.1】（与 AlphaProof 源码对照）**

| 概念 | AlphaProof 位置 | 本目录位置 |
|---|---|---|
| token 掩码 SFT | `train_full_supervised.py`（prompt 全 -100） | `sft.py` `encode_pair` / `sft_hf.py` |
| LoRA 配置 | `app/train_sft.py`（r=16, alpha=32） | `sft_hf.py` `LoraConfig` |
| QLoRA / 4bit | 容器（nf4 double quant） | 正文说明；`sft_hf.py` 可加 bnb |
| chat template | 容器指令 + `<proof_gap>/<dsl_thinking>/<public_proof_state>` | `sft_hf.py` `apply_chat_template` |
| 目标 JSON | `<answer>{"calls":[...]}</answer>` | tactic 文本（教学简化） |
| 数据 | LeanTree ~260k transitions（nanoproof `sft.py`） | 合成 (state,tactic) + `--data` jsonl |

## 8. 常见坑与小结

**【注 SFT.8.1｜R-SFT.8.1】（常见坑）**

1. **prompt 没 mask**：loss 不降或模型学会抄题；检查 `-100` 是否只落在 prompt 段。
2. **模板不一致**：训练手拼、推理 `apply_chat_template`，分布错位，生成崩坏。
3. **eos 忘记**：答案末尾没有 eos，模型学不会"停"。
4. **长度截断**：`max_len` 把答案截掉，等于制造噪声监督。
5. **学习率过高**：SFT 常用 1e-4~2e-4，全参更小。

**【注 SFT.8.2｜R-SFT.8.2】（小结）**

小结：SFT 在答案 token 上最小化交叉熵、prompt 用 `-100` 屏蔽；chat template 保证训练/
推理一致；LoRA/QLoRA 省显存且抗过拟合。下一步见 `06-RL-RLVR/`（《01-RL与RLVR原理》）。

**【练习 SFT.8.3｜Ex-SFT.8.3】（动手：mask 与 LoRA 秩）**

练习：跑 `code/from_scratch/sft.py --steps 400` 看 `answer_token_acc` 与 `exact_match`；
改 `code/with_api/sft_hf.py --lora-r 4/8/16` 比较可训练参数与收敛；临时不置 `-100`
对比 loss 曲线，直观感受 mask 的作用。
