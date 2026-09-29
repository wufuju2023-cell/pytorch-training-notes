# tiny：预置的小模型配置（对齐 nanoproof `NetworkConfig`）

> **【文档｜DOC-TINY】**（doccode = `TINY`）｜编号与 Tag 规范见《00-风格与编号规范》。

两档 JSON 配置，字段与 `nanoproof/nanoproof/model.py:32` 的 `NetworkConfig`
一一对齐，另加教学用的派生字段与资源估算，供 `../01-基础/` 的代码 / notebook 直接读取。

## 配置一览

**【注 TINY.1.1｜R-TINY.1.1】（配置一览）**

本文对应配置 Tag：`F-tiny-config_micro`、`F-tiny-config_tiny`（豁免，不加注释）；Python 版见 `F-b01-configs`。

| | `config_gpt_micro.json` | `config_gpt_tiny.json` |
|---|---|---|
| `sequence_len` | 64 | 128 |
| `vocab_size` | 64（字符级） | 256（字符级/字节级） |
| `n_layer` | 2 | 4 |
| `n_head` | 4 | 4 |
| `n_kv_head`（GQA） | 2 | 2 |
| `n_embd` | 64 | 128 |
| `head_dim` | 16 | 32 |
| `window_pattern` | `"L"` | `"LLSL"` |
| 参数量 | 0.103M | 0.771M |
| 训练显存（fp32 / bf16） | ~20 / ~14 MB | ~60 / ~45 MB |
| 200/500 步耗时（CPU / T4） | 15 s / 2 s | 180 s / 6 s |

两档都 **远小于 20M**，CPU 可跑；`micro` 用于秒级验证链路，`tiny` 用于看
loss 曲线与消融。

## 与 nanoproof `NetworkConfig` 的字段对应

**【注 TINY.2.1｜R-TINY.2.1】（与 nanoproof `NetworkConfig` 的字段对应）**

| nanoproof 字段 | 本 JSON | 含义 |
|---|---|---|
| `sequence_len` | `nanoproof_aligned.sequence_len` | 上下文长度 $T$ |
| `vocab_size` | `nanoproof_aligned.vocab_size` | 词表大小 |
| `n_layer` | `nanoproof_aligned.n_layer` | block 层数 |
| `n_head` | `nanoproof_aligned.n_head` | query 头数 $H_q$ |
| `n_kv_head` | `nanoproof_aligned.n_kv_head` | KV 头数 $H_{kv}$（GQA） |
| `n_embd` | `nanoproof_aligned.n_embd` | 隐藏维 $C$ |
| `window_pattern` | `nanoproof_aligned.window_pattern` | 滑窗模式（`L` 全窗口、`S` 短窗口） |

> nanoproof 的 `window_pattern` 会按层平铺（`L=full, S=quarter`），且**最后一层
> 强制全窗口**（`model.py:241` `_compute_window_sizes`）。本教材在
> `from_scratch/model.py` 里先不实现滑窗，仅保留字段以便后续阶段扩展。

## 参数量估算公式

**【注 TINY.3.1｜R-TINY.3.1】（参数量估算公式）**

对每层（Pre-LN + GQA 注意力 + SwiGLU）：

$$
P_{\text{block}} = C\,H_qd + 2\,C\,H_{kv}d + H_qd\,C + 3\,C\,m + 2C
$$

其中 $d = C/H_q$ 是 head 维，$m$ 是 SwiGLU 中间维：

$$
m = \text{multiple\_of}\cdot\left\lceil \frac{8C/3}{\text{multiple\_of}}\right\rceil
$$

总参数（权重共享时）：

$$
P = V\,C + n_{\text{layer}}P_{\text{block}} + C
$$

`micro`：$P=102{,}720$；`tiny`：$P=771{,}200$，与 JSON 中 `estimate.params_total` 一致。

**前反向 FLOPs/token**（粗估）：

$$
\text{FLOPs}_{\text{tok}} \approx 6\,(P - P_{\text{emb}}) + \sum_{\ell} 12\,H_q\,d\,T
$$

第二项是注意力部分；nanoproof 在 `model.py:263` `estimate_flops` 里就是这么算的
（额外按滑窗修正有效序列长度）。

## 用法

**【注 TINY.4.1｜R-TINY.4.1】（用法）**

```python
import json
cfg = json.load(open("v1.0/tiny/config_gpt_tiny.json"))
aligned = cfg["nanoproof_aligned"]        # 直接喂给 NetworkConfig 等价字段
```

`from_scratch/configs.py` 的 `MICRO` / `TINY` 两个 `GPTConfig` 就是这两份 JSON 的
Python 版（数值一一对应，可直接 `python3 configs.py` 复核）。
