# 第 3 章 Attention 与 Transformer

> **对应 AlphaProof 源码**
> - `nanoproof/nanoproof/model.py`：`CausalSelfAttention`、`apply_rotary_emb`、`MLP`、`Block`、`Transformer`、`norm`（RMSNorm）、`_precompute_rotary_embeddings`、`_compute_window_sizes`。
> - `app/policy_server.py`：`AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, device_map=...)`，`use_cache` 与 REAL-Prover 的推理用法。
>
> **配套 notebook**：`notebooks/01-基础/03-Attention与Transformer.ipynb`（`N-03`、`N-05`）
>
> **关联数学文档**
> - [`../../../Transformer数学理论参考资料/01-注意力机制与序列建模基础.md`](../../../Transformer数学理论参考资料/01-注意力机制与序列建模基础.md)
> - [`../../../Transformer数学理论参考资料/02-Transformer架构的完整数学表述.md`](../../../Transformer数学理论参考资料/02-Transformer架构的完整数学表述.md)
> - [`../../../Transformer数学理论参考资料/04-位置编码与外推理论.md`](../../../Transformer数学理论参考资料/04-位置编码与外推理论.md)
> - [`../../../Transformer数学理论参考资料/07-高效注意力的数学理论.md`](../../../Transformer数学理论参考资料/07-高效注意力的数学理论.md)
> - [`../../../残差连接理论参考资料/`](../../../残差连接理论参考资料/)
>
> **预计学时**：6 学时
>
> **前置知识**：第 1-2 章、线性代数（内积、正交）、softmax、三角恒等式（预备篇：见【定义 PY4.7.1】、【注 PY4.4.4】）。
> **【文档｜DOC-3】**（doccode = `3`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 3.1 缩放点积注意力

### 3.1.1 定义

**【定义 3.1.1｜D-3.1.1】（缩放点积注意力）**

注意力是"软检索"：查询向量 $q$ 与键 $k_j$ 的内积衡量相似度，经 softmax（【定义 1.2.1】）归一化后对值 $v_j$ 加权求和。矩阵形式为

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V,
$$

其中 $Q \in \mathbb{R}^{T_q \times d_k}$、$K, V \in \mathbb{R}^{T_k \times d_k}$。

### 3.1.2 为什么除以 $\sqrt{d_k}$

**【注 3.1.2｜R-3.1.2】（缩放因子 $\sqrt{d_k}$ 的方差论证）**

**【证明｜Pf-3.1.2】**

设 $q, k$ 分量独立、零均值、单位方差，则 $q^\top k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$，标准差为 $\sqrt{d_k}$。若不缩放，$d_k$ 增大时 logits 幅度按 $\sqrt{d_k}$ 增长，softmax 进入饱和（近似 one-hot），梯度趋近 0。除以 $\sqrt{d_k}$ 将 logits 方差归一，保证 softmax 梯度处于良好区间。

### 3.1.3 因果掩码

**【定义 3.1.3｜D-3.1.3】（因果掩码）**

自回归要求位置 $t$ 只能看到 $\le t$ 的位置，因此在 softmax 前把 $j > i$ 的 logits 置为 $-\infty$：

$$
s_{ij} \leftarrow s_{ij} - \infty \cdot \mathbf{1}[j > i] .
$$

`model.py` 把因果性交给 Flash Attention，不显式构造掩码：

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

`causal=True` 让融合内核跳过上三角，省去一半计算与 $O(T^2)$ 掩码显存。

---

## 3.2 多头注意力与 GQA

### 3.2.1 MHA

**【定义 3.2.1｜D-3.2.1】（多头注意力 MHA）**

单个注意力只能表达一种"检索模式"。多头把模型维度 $C$ 拆成 $H$ 个子空间，每头独立注意力后拼接投影：

$$
\mathrm{head}_h = \mathrm{Attn}(Q W_h^Q, K W_h^K, V W_h^V), \qquad
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_H) W^O .
$$

`model.py::CausalSelfAttention` 用 `(B, T, H, D)` 布局（Flash Attention 原生），`head_dim = n_embd // n_head`。

### 3.2.2 GQA

**【定义 3.2.2｜D-3.2.2】（分组查询注意力 GQA）**

**分组查询注意力**让多个查询头共享 K/V 头（$n_{\text{kv\_head}} < n_{\text{head}}$），在几乎不损质量的前提下按比例降低推理 KV-cache 与带宽：

```python
assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
```

$n_{\text{kv\_head}} = n_{\text{head}}$ 时退化为 MHA，$=1$ 时为 MQA。

### 3.2.3 QK-Norm 与锐化

**【注 3.2.3｜R-3.2.3】（QK-Norm 与锐化系数）**

`model.py` 在注意力前对 Q、K 做 RMSNorm 并乘 1.2：

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)   # QK norm
q = q * 1.2               # sharper attention
k = k * 1.2
```

QK-Norm 防止 logits 随训练放大，1.2 是经验性锐化系数。

---

## 3.3 位置编码与 RoPE 推导

### 3.3.1 RoPE 的构造

**【定义 3.3.1｜D-3.3.1】（旋转位置编码 RoPE）**

注意力天然对顺序不敏感，必须注入位置。RoPE 把位置信息编码为对 Q/K 成对维度的**旋转**。对 head 维 $d$（偶数），分成 $d/2$ 对，第 $i$ 对的频率为

$$
\theta_i = \mathrm{base}^{-\frac{2i}{d}}, \qquad \mathrm{base} = 100000 .
$$

位置 $t$ 的旋转矩阵分块为

$$
R_t^{(i)} =
\begin{pmatrix}
\cos(t\theta_i) & -\sin(t\theta_i) \\
\sin(t\theta_i) & \cos(t\theta_i)
\end{pmatrix},
$$

整块对角拼接成 $R_t$。对 $q, k$ 施加旋转后，由旋转的性质 $R_t^\top R_s = R_{s-t}$ 得

$$
(R_t q)^\top (R_s k) = q^\top R_{s-t} k ,
$$

即注意力分数只依赖**相对位置** $s - t$。这正是 RoPE 相对位置感知与一定程度外推能力的来源。

### 3.3.2 对照源码

**【注 3.3.2｜R-3.3.2】（RoPE 源码对照与 base 选择）**

`model.py::_precompute_rotary_embeddings` 预计算所有位置的 cos/sin：

```python
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
inv_freq = 1.0 / (base ** (channel_range / head_dim))  # theta_i
t = torch.arange(seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)                        # t * theta_i
cos, sin = freqs.cos(), freqs.sin()
```

`apply_rotary_emb` 把最后一维分成两半做平面旋转：

```python
x1, x2 = x[..., :d], x[..., d:]
y1 = x1 * cos + x2 * sin
y2 = x1 * (-sin) + x2 * cos
```

`base=100000`（而非原版 10000）降低了低频频率，有利于长上下文。

---

## 3.4 FFN / SwiGLU 与残差

### 3.4.1 前馈网络

**【定义 3.4.1｜D-3.4.1】（逐位置前馈网络 FFN）**

每个 block 在注意力后接逐位置 FFN：

$$
\mathrm{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2 .
$$

`nanoproof` 用 $\sigma(z) = \mathrm{relu}(z)^2$（MLP 与 $\mathrm{relu}^2$ 激活见【定义 1.1.3】）、隐藏维 $4C$、无偏置：

```python
x = self.c_fc(x)          # C -> 4C
x = F.relu(x).square()    # relu^2
x = self.c_proj(x)        # 4C -> C
```

### 3.4.2 SwiGLU

**【定义 3.4.2｜D-3.4.2】（门控 FFN：SwiGLU）**

现代 LLM 常用门控 FFN：

$$
\mathrm{SwiGLU}(x) = \left( \mathrm{Swish}(W_1 x) \odot W_3 x \right) W_2, \qquad
\mathrm{Swish}(z) = \frac{z}{1 + e^{-z}} .
$$

它用三个矩阵与逐元素门控，隐藏维常设为约 $\frac{8}{3}C$ 以平衡参数量。`nanoproof` 选的是 relu^2，但范式一致。

### 3.4.3 Pre-LN 与残差

**【定义 3.4.3｜D-3.4.3】（Pre-LN 与 RMSNorm）**

`Block.forward` 是 Pre-LN：

```python
x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
x = x + self.mlp(norm(x))
```

$$
x_{l+1} = x_l + \mathrm{Attn}(\mathrm{RMSNorm}(x_l)), \qquad
x_{l+2} = x_{l+1} + \mathrm{MLP}(\mathrm{RMSNorm}(x_{l+1})) .
$$

Pre-LN 让残差通路为恒等映射，梯度无衰减回传，比 Post-LN 稳定。`norm` 是**无可学习参数**的 RMSNorm：

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{C}\sum_{i=1}^{C} x_i^2 + \epsilon}} .
$$

### 3.4.4 `nanoproof` 的额外技巧

**【注 3.4.4｜R-3.4.4】（nanoproof 的额外工程技巧）**

- **逐层残差标量**：`x = resid_lambdas[i] * x + x0_lambdas[i] * x0`，把初始嵌入按递减系数混入。
- **Smear**：前一 token 嵌入按 `smear_gate` 门控混入当前 token，增强局部连续性。
- **Backout**：缓存中层残差并在最后减去一部分（`backout_lambda`），移除低层特征。
- **Logit softcap**：$z \leftarrow 15 \tanh(z/15)$。
- **Untied weights**：`wte` 与 `lm_head` 不共享权重。
- **词表 padding**：`vocab_size` 向上补到 64 的倍数。
- **滑动窗口**：`window_pattern` 用 "L"/"S" 指定每层窗口，短窗为长窗四分之一并对齐到 128（FA3 tile），最后一层强制全上下文。

---

## 3.5 KV-Cache 与自回归推理

**【定义 3.5.1｜D-3.5.1】（KV-Cache）**

自回归生成每步只新增一个 token，但注意力需要历史 K/V。若每步重算历史，总代价为 $O(T^2)$。**KV-cache** 缓存每层的 $K, V$，新步只算当前 token 的 $q, k, v$ 并追加，使注意力主项降为 $O(T)$，代价是 $O(T)$ 显存。

`model.py` 推理路径用 Flash Attention 的 cache 接口：

```python
k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
y = flash_attn.flash_attn_with_kvcache(
    q, k_cache, v_cache, k=k, v=v,
    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
)
if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
```

**【注 3.5.2｜R-3.5.2】（变长 batch 的 RoPE 位置）**

变长 batch 下每行 cache 长度不同，RoPE 位置必须逐行取：

```python
positions = kv_cache.cache_seqlens.long().unsqueeze(1) + torch.arange(T)
```

**【注 3.5.3｜R-3.5.3】（KV-cache 容量与 GQA）**

KV-cache 大小为 $2 \times n_{\text{layer}} \times H_{kv} \times d_{\text{head}} \times T$，因此 GQA 减少 $H_{kv}$ 直接降低推理显存与带宽。

---

## 3.6 REAL-Prover / `policy_server.py` 的用法

**【注 3.6.1｜R-3.6.1】（REAL-Prover 推理用法）**

`policy_server.py` 不自己实现模型，而是加载 HuggingFace 的 REAL-Prover：

```python
base = AutoModelForCausalLM.from_pretrained(
    base_dir, torch_dtype=torch.bfloat16, device_map=self.device)
base.config.use_cache = True
```

- **BF16** 推理 dtype；**`device_map`** 自动分配层到设备。
- **`use_cache`**：生成时开启以启用 KV-cache；前向/训练可关闭。
- 在 backbone 之上挂 `ValueHead`，读取最后隐状态得到价值 $V_\phi(s)$，供搜索使用。

**【注 3.6.2｜R-3.6.2】（自研实现与 HF 实现的对照）**

`nanoproof/model.py` 是同一结构的高效自研实现（Flash Attention、自定义 `Linear`、无 autocast 显式 dtype），`policy_server.py` 则用 `transformers` 复用已发布权重。

---

## 3.7 小结

**【注 3.7.1｜R-3.7.1】（小结）**

1. 注意力 = softmax 加权软检索；$\sqrt{d_k}$ 缩放防止饱和。
2. 多头在不同子空间并行；GQA 减少 K/V 头以降低推理成本。
3. RoPE 用成对旋转注入相对位置，分数只依赖 $s - t$。
4. Pre-LN + 残差保证深层可训练；FFN 提供逐位置非线性。
5. KV-cache 把自回归推理主项降到 $O(T)$，GQA 决定其显存。

**【练习 3.7.2｜Ex-3.7.2】（RoPE 恒等式与 KV-cache 节省估算）**

**练习**：证明 RoPE 内积恒等式；估算 $C=768$、$H=6$、$H_{kv}=2$、$T=768$ 时 KV-cache 相对 MHA 的节省比例。
