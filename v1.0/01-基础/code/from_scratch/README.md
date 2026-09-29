# from_scratch：纯 PyTorch 手写 tiny GPT

不依赖任何第三方模型库，只用 `torch`，把「字符 → 下一个字符」的最小 GPT 拆开写清楚。
整机 ≤ 20M 参数，CPU 可跑；配套理论篇见 `../../../01-基础/*.md`，配套实验见
`../../../notebooks/01-基础/N01–N08.ipynb`。

## 文件

| 文件 | 作用 | 对应 AlphaProof / nanoproof |
|---|---|---|
| `configs.py` | `GPTConfig` + `micro`/`tiny` 两档预设、参数量估算 | `nanoproof/nanoproof/model.py:32` `NetworkConfig` |
| `model.py` | RMSNorm / RoPE / GQA 注意力 / SwiGLU / Pre-LN Block / GPT / 采样 | `model.py:45,58,67,127,140,405,492` |
| `data.py` | 字符级 tokenizer + 语料加载 + block 采样 | `nanoproof/common.py:70` `GlobalConfig`；`pretrain.py` 数据循环 |
| `train.py` | 训练循环：AMP / grad-accum / warmup+cosine / checkpoint / resume | `common.py:86` lr multiplier；`pretrain.py` |
| `generate.py` | 载入 checkpoint，temperature / top-k / top-p 采样 | `model.py:492` `generate` |

## 快速开始（CPU，几分钟内出结果）

```bash
cd 01-基础/code/from_scratch
python3 -m py_compile configs.py model.py data.py train.py generate.py   # 语法自检
python3 configs.py            # 打印两档配置与参数量
python3 model.py              # 构建 micro/tiny，跑一次 forward 看 loss
python3 data.py               # 看 vocab 与一个 batch 的 x/y
python3 train.py --config micro --max-iters 200 --batch-size 16
python3 generate.py --ckpt out/micro/last.pt --prompt "theorem " --max-new-tokens 120
```

有 GPU 时：

```bash
python3 train.py --config tiny --device cuda --amp --max-iters 2000 --batch-size 32 --grad-accum 4
```

断点续训：

```bash
python3 train.py --config micro --resume out/micro/last.pt --max-iters 400
```

## 与理论的对应

**注意力（Attention）。** 单头缩放点积注意力

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

其中 $M$ 是因果掩码（上三角为 $-\infty$）。代码里由
`F.scaled_dot_product_attention(..., is_causal=True)` 实现；GQA 则先把 KV 头
复制到 query 头数（`repeat_kv`，`model.py`）。

**位置编码（RoPE）。** 对 q/k 的最后两半维做二维旋转：

$$
\begin{pmatrix} y_1 \\ y_2 \end{pmatrix}
= \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

频率 $\theta_i = t \cdot b^{-2i/d}$，$b=10000$。见 `precompute_rope` / `apply_rope`。

**SwiGLU。** $\mathrm{SwiGLU}(x)=W_{\text{down}}\big(\mathrm{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\big)$，
中间维度取 $\tfrac{8}{3}C$ 再向上取整。nanoproof 用 $\mathrm{relu}^2$，二者可在 N05 互换。

**Pre-LN 残差。** `x = x + Attn(Norm(x)); x = x + MLP(Norm(x))`，训练比 Post-LN 稳。

**参数共享。** `tie_weights=True` 时 `lm_head.weight = tok_emb.weight`，省一份词表参数。

## 设计取舍（与 nanoproof 的差异，教学简化）

- nanoproof 用自定义 `Linear` + bf16 取代 `autocast`（`model.py:49`），本版用
  标准 `torch.autocast`，更好懂；`--amp` 开关即对应它的 `COMPUTE_DTYPE`。
- nanoproof embedding/lm_head **不共享**（untied），本版默认共享并留开关。
- nanoproof 有滑动窗口、smear、backout、QK sharpening 等工程 trick，本版只保留
  主干 + QK-norm，其余留给 N05 消融。
- 本版 `generate` 无 KV cache（重复前向，教学清晰）；KV cache 的工程收益见 N06。
