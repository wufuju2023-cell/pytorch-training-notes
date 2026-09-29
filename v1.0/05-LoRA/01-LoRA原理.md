# 05-LoRA原理：低秩适配、缩放与 QLoRA

> 本文件是 v1.0《PyTorch 和模型训练源码学习》第 05 章理论篇。
> 目标：从低秩分解推导 LoRA，解释 A/B 初始化、$\alpha/r$ 缩放、目标模块选择、
> QLoRA（nf4 / double quant / paged）、DoRA，并与全参微调对比。
> 对照源码：
> - `reap-new-update-model/app/policy_server.py:108-116`（r16, α32, dropout0.02, all attn+mlp, `init_lora_weights=True`）
> - `v1-1-agentic-tool/gpu/gpu_runtime/real_backend.py:68-128`（r16, α32, dropout0, `TARGET_MODULES`, per-session adapter）
> - 既有数学文档 `/home/a/文档/LoRA微调数学理论参考资料`（02/03/04/05/06 篇）
> 代码伴侣：`code/from_scratch/lora.py`（`F-b05-lora`）、`finetune_lora.py`（`F-b05-finetune_lora`）、`code/with_api/peft_lora.py`（`F-b05-peft_lora`）、`qlora_bnb.py`（`F-b05-qlora_bnb`）。配套 notebook：`N-12`。
> **【文档｜DOC-LORA】**（doccode = `LORA`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 0. 一页速览

**【注 LORA.1.1｜R-LORA.1.1】（LoRA 速览）**

| 概念 | 要点 |
| --- | --- |
| 参数化 | $W = W_0 + \Delta W = W_0 + \tfrac{\alpha}{r}BA$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$ |
| 初始化 | $A\sim$ kaiming/正态，$B=0$ $\Rightarrow$ 初始 $\Delta W=0$（等价 base） |
| 缩放 | $\tfrac{\alpha}{r}$，$\alpha$ 常取 $2r$（如 r16→α32） |
| 参数量 | $r(d+k)$，远小于 $dk$ |
| 目标模块 | REAP 用全部 attn+mlp：`q,k,v,o,gate,up,down`；bias=none |
| 推理 | 可 merge 回 $W$，零额外延迟 |
| QLoRA | base 量化到 nf4 + double quant + paged optimizer，LoRA 仍 bf16 |
| DoRA | 分解为幅度 $m$ 与方向 $\mathbf{v}/\|\mathbf{v}\|$，分别训 |
| 数学文档 | `LoRA微调数学理论参考资料/02-LoRA的数学结构.md` 等 |

---

## 1. 低秩分解推导

### 1.1 动机

**【注 LORA.2.1｜R-LORA.2.1】（动机：全参微调的存储代价）**

全参微调把权重更新记为 $\Delta W$，对每个 $d\times k$ 的线性层要新存 $dk$ 个参数。
对 7B 模型，$\sum dk$ 是数十亿参数，优化器状态（Adam 一阶/二阶矩）还要再乘 2–4 倍。应用场景多为 SFT（见《03-SFT原理》【定义 SFT.5.2】），参数化定义见下文【定义 LORA.2.2】。

**【定义 LORA.2.2｜D-LORA.2.2】（低秩更新参数化）**

LoRA 的核心假设：**微调带来的任务相关更新是低秩的**（intrinsic dimension 小）。
于是令

$$
\Delta W = \frac{\alpha}{r}\,B A,
\qquad A\in\mathbb{R}^{r\times k},\quad B\in\mathbb{R}^{d\times r},\quad r \ll \min(d,k).
$$

前向变成

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}B(Ax).
$$

注意 $B(Ax)$ 的计算顺序让参数量只有 $r(d+k)$，而不是 $dk$。

### 1.2 参数量与压缩比

**【命题 LORA.2.3｜P-LORA.2.3】（参数量与压缩比）**

$$
\frac{r(d+k)}{dk} = r\left(\frac1d+\frac1k\right).
$$

以 $d=k=4096, r=16$ 为例：$16\times 8192 / 4096^2 \approx 0.78\%$。对 7B 模型全体
目标模块，LoRA 参数通常在 0.1%–1%。

### 1.3 秩与容量

**【注 LORA.2.4｜R-LORA.2.4】（秩与容量）**

$\mathrm{rank}(\Delta W) \le r$。$r$ 控制"可学方向"的维数：

- $r$ 太小 → 欠拟合任务（容量不足）；
- $r$ 太大 → 参数量上升、过拟合风险、失去 PEFT 意义；
- 经验：$r=8/16/32$ 覆盖多数任务，难任务/大域迁移可到 64–128。

严格地，$\Delta W$ 所在的秩-$r$ 流形不是欧氏空间，而是 Grassmann/Stiefel 型流形；
A/B 分解存在**尺度冗余**（$A\to cA,\ B\to B/c$ 不变），这既带来非唯一性，也解释了
为什么缩放 $\alpha/r$ 重要（固定参数化尺度，稳定优化）。推导细节见
`LoRA微调数学理论参考资料/02-LoRA的数学结构.md`、`04-LoRA变体的统一理论.md`。


---

## 2. 初始化与缩放 $\alpha/r$

### 2.1 为什么 B 初始化为零

**【命题 LORA.3.1｜P-LORA.3.1】（零初始化 ⇒ 初始等价基座）**

若 $B=0$，则 $\Delta W = \frac{\alpha}{r}BA = 0$，微调**从等价基座开始**，训练不会在
第一步就破坏预训练知识。PEFT 用 `init_lora_weights=True` 表达这件事，
`init_lora_weights=False` 则 A、B 都随机（只适合从零训练/调试）：

**【代码 LORA.3.2｜Cd-LORA.3.2】（`LoraConfig`）**

```python
# app/policy_server.py:112-115
lora = LoraConfig(r=lora_r, lora_alpha=32, lora_dropout=0.02,
                  target_modules=TARGET_MODULES, bias="none",
                  init_lora_weights=True)  # 零 B 初始化 => 等价 base
```

**【注 LORA.3.3｜R-LORA.3.3】（零初始化下的梯度流向）**

具体地，PEFT 默认：$A$ 用 kaiming 均匀初始化，$B=0$。于是梯度
$\partial\mathcal{L}/\partial B \ne 0$（因为 $A\ne 0$），第一步 B 开始生长；而
$\partial\mathcal{L}/\partial A = 0$（因为 $B=0$），所以 A 第一步不动——这是有意的
稳定设计。

### 2.2 缩放因子与 $\alpha$ 的含义

**【命题 LORA.3.4｜P-LORA.3.4】（缩放 $s=\alpha/r$ 的作用）**

缩放 $s=\alpha/r$ 出现在 $\Delta W$ 上。它有两个作用：

1. **解耦秩与步长**：改变 $r$ 时若想保持有效更新幅度，应同步调 $\alpha$；固定
   $\alpha/r$ 让不同 $r$ 的学习率敏感性接近。
2. **等价于对 LoRA 分支的"学习率重参数化"**：$\alpha$ 越大更新越猛。

常见设置：$\alpha = 2r$（r16→α32，正是 REAP 的配置）、$\alpha=r$、或固定 $\alpha$。
把 $A$ 视作随 $r$ 变化的高斯矩阵时，$\|BA\|$ 的尺度随 $r$ 增长，$\alpha/r$ 起归一化
作用。理论分析见 `LoRA微调数学理论参考资料/03-LoRA的训练动力学与优化理论.md`。

### 2.3 dropout 与合并

**【注 LORA.3.5｜R-LORA.3.5】（dropout 与权重合并）**

- `lora_dropout` 加在 LoRA 分支的输入上（不是基座），起到正则作用；REAP
  `policy_server` 用 0.02，`real_backend` 用 0。
- 推理时可**合并**：$W \leftarrow W_0 + \frac{\alpha}{r}BA$，之后前向零开销。这也是
  LoRA 相比 adapter 层（串行增加延迟）的关键优势。

---

## 3. 目标模块与秩的选择

LoRA 加在哪些线性层，直接决定容量与收益。REAP / gpu_runtime 的选择是完全一致的 7 个：

**【代码 LORA.4.1｜Cd-LORA.4.1】（`TARGET_MODULES`）**

```python
# gpu_runtime/real_backend.py:24
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj")
```

**【注 LORA.4.2｜R-LORA.4.2】（目标模块选择与容量）**

- `q,k,v,o`：注意力投影；
- `gate,up,down`：MLP（FFN）投影。

实践结论（见数学文档 `05-表达力与容量理论.md`）：

| 配置 | 参数 | 效果 |
| --- | --- | --- |
| 仅 `q,v` | 最少 | 原始论文最省，效果尚可 |
| 全部 attention | 中等 | 通用推荐 |
| attention + MLP | 较多 | 域迁移/大任务更好（REAP 选这个） |
| 加 embedding/lm_head | 很多 | 通常不必要 |

`bias="none"` 表示不训练偏置（几乎不影响效果，省参数）。秩 $r$ 的容量直觉：
LoRA 更新的秩不超过 $r$，$r$ 越大可表达的任务相关方向越多；但当 $r$ 超过任务
intrinsic dimension 后收益迅速饱和，反而增加过拟合与显存。

---

## 4. QLoRA：把基座压到 4-bit

**【定义 LORA.5.1｜D-LORA.5.1】（QLoRA）**

QLoRA = **base 权重 4-bit 量化（冻结） + LoRA（bf16）**。它让 65B 模型能单卡微调。

### 4.1 nf4 量化

**【定义 LORA.5.2｜D-LORA.5.2】（nf4 blockwise 量化）**

对每个 block（通常 64 个权重）做 **blockwise absmax**：

$$
s = \frac{\max|w|}{q_{\max}}, \qquad
\hat w = s \cdot \mathrm{round}\!\left(\frac{w}{s}\cdot q_{\max}\right) / q_{\max}
$$

nf4（4-bit NormalFloat）假设权重近似正态，把 $[-1,1]$ 分成 16 个**等概率**分位点
（而非等间距），因此比 int4 有更小的量化误差。存储时每 block 一个 fp16/bf16 的
scale $s$。

### 4.2 Double Quantization

**【定义 LORA.5.3｜D-LORA.5.3】（double quantization）**

scale 本身也占显存（每 64 权重一个 bf16 = 0.5 bit/param）。double quant 再对
scale 做一次 8-bit 量化，把开销降到约 0.127 bit/param，进一步省显存。

### 4.3 Paged Optimizer

**【注 LORA.5.4｜R-LORA.5.4】（paged optimizer）**

优化器状态用 NVIDIA unified memory 分页，遇到显存尖峰自动换出到 CPU，避免 OOM
（速度有损，但比崩溃好）。

### 4.4 计算流程

**【注 LORA.5.5｜R-LORA.5.5】（QLoRA 计算流程）**

前向时权重**即时反量化**到 bf16 做 matmul，梯度只流向 LoRA 分支（base 冻结）。
因此 QLoRA 的显存 ≈ 4-bit base + 少量 LoRA + 激活。PEFT 里对应：

**【代码 LORA.5.6｜Cd-LORA.5.6】（`BitsAndBytesConfig`）**

```python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True)
```

理论细节见 `LoRA微调数学理论参考资料/06-QLoRA与量化理论.md`。

---

## 5. DoRA（Weight-Decomposed Low-Rank Adaptation）

**【定义 LORA.6.1｜D-LORA.6.1】（DoRA：幅度-方向分解）**

DoRA 把预训练权重分解为**幅度**与**方向**，再只对方向做低秩更新：

$$
W = m \cdot \frac{V}{\|V\|_c}, \qquad V = W_0 + BA
$$

其中 $m\in\mathbb{R}^{d}$ 是可训练幅度向量（每列一个标量），$\|\cdot\|_c$ 是列向
归一化。训练时 $m$ 与 $BA$ 都更新。

**【注 LORA.6.2｜R-LORA.6.2】（DoRA 的特点与统一视角）**

相比 LoRA，DoRA：

- 更接近全参微调的"幅度-方向"更新模式，低 rank 下精度更好；
- 额外参数只有 $d$（幅度向量），开销与 LoRA 同量级；
- 实现稍复杂（多一步列归一化）。

统一视角：LoRA、DoRA、AdaLoRA 等都可看成"在秩约束下对 $\Delta W$ 的不同参数化"（【定义 LORA.2.2】的推广），
见 `LoRA微调数学理论参考资料/04-LoRA变体的统一理论.md`。

---

## 6. 与全参微调对比、秩-容量关系

**【注 LORA.7.1｜R-LORA.7.1】（与全参微调对比）**

| 维度 | 全参微调 | LoRA | QLoRA |
| --- | --- | --- | --- |
| 可训练参数 | 100% | ~0.1–1% | ~0.1–1% |
| 基座显存 | 全精度 | 全精度 | 4-bit |
| 优化器状态 | 数十 GB | 极小 | 极小 |
| 灾难性遗忘 | 高 | 低（基座冻结） | 低 |
| 推理开销 | 无 | 可 merge 归零 | 需反量化或合并 |
| 多任务 | 每任务一份权重 | 每任务一个 adapter | 每任务一个 adapter |
| 上限 | 最高 | 接近全参（秩足够时） | 略低于 LoRA |

**【注 LORA.7.2｜R-LORA.7.2】（秩-容量关系）**

**秩-容量关系**：设任务所需的"固有维度"为 $r^{*}$。

- $r < r^{*}$：欠拟合，增大 rank 明显涨点；
- $r \approx r^{*}$：收益递减；
- $r \gg r^{*}$：接近全参，但开始过拟合/浪费显存。

实践中先用 $r=16$ 试，再按验证集扫描 $r\in\{8,16,32,64\}$；同时 $\alpha=2r$。
数学上的表达力界（给定 rank 可逼近的更新类）见
`LoRA微调数学理论参考资料/05-表达力与容量理论.md`、训练动力学见 `03` 篇。


---

## 7. 源码对照与超参速查

**【注 LORA.8.1｜R-LORA.8.1】（源码对照与超参速查）**

| 概念 | 源码位置 | 取值 |
| --- | --- | --- |
| LoRA 配置（policy_server） | `app/policy_server.py:112-116` | r=16, α=32, dropout=0.02, bias=none, init=True |
| LoRA 配置（gpu real_backend） | `gpu_runtime/real_backend.py:117-128` | r=16, α=32, dropout=0.0, bias=none, init=True |
| 目标模块 | `gpu_runtime/real_backend.py:24` | q,k,v,o,gate,up,down |
| per-session adapter | `real_backend.py:242` (`add_adapter(session_id, ...)`) | 每 session 独立 adapter |
| 冻结基座 + 头共训 | `real_backend.py:623`（adapter + value head 参数） | 一个 AdamW |
| `init_lora_weights=True` 语义 | `real_backend.py:125` 注释 | B=0 ⇒ 等价 base |
| QLoRA | `code/with_api/qlora_bnb.py` | nf4/double quant/paged |
| DoRA | 本文件 §5 | 幅度-方向分解 |

**【注 LORA.8.2｜R-LORA.8.2】（推荐起点）**

推荐起点：

- $r=16$，$\alpha=32$，`lora_dropout=0.0~0.05`，`bias="none"`；
- 目标模块 = 全部 attention + MLP；
- 基座冻结，`init_lora_weights=True`；
- 4-bit（QLoRA）用于显存不够时，`bnb_4bit_compute_dtype=bf16`。

## 8. 与既有数学文档的链接

**【注 LORA.9.1｜R-LORA.9.1】（外部数学文档索引）**

- 低秩结构与 Grassmann 流形：`/home/a/文档/LoRA微调数学理论参考资料/02-LoRA的数学结构.md`
- 训练动力学/尺度冗余：`.../03-LoRA的训练动力学与优化理论.md`
- LoRA 变体统一理论（含 DoRA）：`.../04-LoRA变体的统一理论.md`
- 表达力与容量：`.../05-表达力与容量理论.md`
- 量化误差：`.../06-QLoRA与量化理论.md`
- 超参与缩放律：`.../08-缩放律-超参与理论最佳实践.md`

## 9. 小结

**【注 LORA.10.1｜R-LORA.10.1】（小结）**

1. LoRA 用 $W_0+\frac{\alpha}{r}BA$ 参数化低秩更新，$B=0$ 保证初始等价基座。
2. $\alpha/r$ 是尺度归一化；$r$ 控制容量，$\alpha=2r$ 是稳妥起点。
3. 目标模块选 attention+MLP 容量最大，REAP 正是如此。
4. QLoRA 把冻结基座压到 nf4 + double quant + paged，LoRA 仍高精度。
5. DoRA 在幅度-方向分解上做低秩，低秩时更强。
6. 秩足够时 LoRA 逼近全参，但显存/参数/遗忘都远优。

下一步读代码：`05-LoRA/README.md`、`code/from_scratch/lora.py`、
`code/from_scratch/finetune_lora.py`、`code/with_api/peft_lora.py`、`qlora_bnb.py`，
并跑 `notebooks/05-LoRA/N12_LoRA从零与PEFT.ipynb`。
