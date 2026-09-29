# with_api：用 HuggingFace 现成 API 训练同一个 tiny GPT

与 `../from_scratch/` 是**同一件事的两种写法**：同语料、同 block 采样、同 lr 形状，
唯一区别是网络和训练循环从手写换成 `transformers` / `datasets` / `accelerate`。
目标是一眼看清「框架替你做了哪些事」。

## 文件

| 文件 | 作用 |
|---|---|
| `train_hf.py` | `GPT2LMHeadModel` + `Trainer`；`--use-accelerate` 切换到手写 accelerate 循环 |
| `tokenizer_api.py` | `tokenizers` 训练小 BPE + `PreTrainedTokenizerFast` 封装；无依赖时回退字符级 |
| `README.md` | 本文，逐模块对照表 |

## 快速开始（Colab / 本地）

```bash
cd 01-基础/code/with_api
python3 tokenizer_api.py --demo --vocab-size 512 --out out/tokenizer
python3 train_hf.py --tokenizer char --max-steps 200 --block-size 128
python3 train_hf.py --tokenizer bpe --tokenizer-dir out/tokenizer --max-steps 200
python3 train_hf.py --use-accelerate --max-steps 200
```

采样：

```python
from transformers import pipeline
gen = pipeline("text-generation", model="out/hf")
print(gen("theorem ", max_new_tokens=60))
```

## 逐模块对照表

| 模块 | from_scratch | with_api | AlphaProof |
|---|---|---|---|
| 配置 | `configs.GPTConfig` | `transformers.GPT2Config` | `nanoproof/model.py:32 NetworkConfig` |
| 归一化 | `RMSNorm` | GPT-2 `LayerNorm` | `model.py:45 F.rms_norm` |
| 位置编码 | `RoPE`（`precompute_rope`/`apply_rope`） | 可学习绝对位置（`wpe`） | `model.py:227` RoPE |
| 注意力 | `CausalSelfAttention`（GQA+RoPE+QK-norm） | `GPT2Attention`（MHA，HF 内部） | `model.py:67` |
| KV 头 | `n_kv_head` + `repeat_kv` | `num_key_value_heads` | `model.py:67` GQA |
| MLP | `SwiGLU` | GPT-2 `gelu` MLP（4×） | `model.py:127` relu² |
| Block | Pre-LN + 残差 | Pre-LN + 残差 | `model.py:140` |
| 权重共享 | `tie_weights` | `tie_word_embeddings=True` | 不共享（untied） |
| Loss | `F.cross_entropy(ignore_index=-100)` | 模型内 `CrossEntropyLoss` | `model.py:481` |
| 采样 | 手写 `generate`（temp/top-k/top-p） | `model.generate(...)` | `model.py:492` |
| 优化器 | `configure_optimizers` 手写分组 | `TrainingArguments` 自动 | `optim.py` MuonAdamW |
| lr 调度 | `get_lr` 手写 warmup+cosine | `lr_scheduler_type="cosine"` | `common.py:86` |
| 梯度累积 | 手写 for 循环累加 | `gradient_accumulation_steps` | — |
| AMP | `torch.autocast` + GradScaler | `bf16`/`fp16` 参数 | `common.py:38 COMPUTE_DTYPE` |
| 断点续训 | 手写 `torch.save` checkpoint | `Trainer` 自动 `checkpoint-*` | `pretrain.py` |
| 分词器 | 字符级 `CharTokenizer` | `tokenizers` BPE + `PreTrainedTokenizerFast` | `nanoproof/tokenizer.py` |

## 理论回顾（两版共同点）

**因果语言模型目标**是最大化每个位置在给定前缀下的对数似然：

$$
\mathcal{L}(\theta)=-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t})
$$

**warmup + cosine** 的学习率（本目录两版都实现）：

$$
\eta_t=\eta_{\min}+\tfrac{1}{2}\big(1+\cos(\pi\cdot\text{progress})\big)(\eta_{\max}-\eta_{\min})
$$

其中 `progress` 在 warmup 之后由 0 线性增到 1。nanoproof 用的是
linear warmup + flat + linear warmdown（`common.py:86`），形状略有不同但目的一致。

**梯度累积**把 $g=\frac{1}{K}\sum_{k=1}^{K}\nabla\ell_k$ 平均后一次更新，等效 batch
放大 $K$ 倍、显存不变。

## 思考题

1. HF 的 `GPT2LMHeadModel` 提供 `labels` 时会自动右移一位；`from_scratch` 里是
   谁做的右移？（提示：`data.get_batch` 的 `y`）
2. 把 `configs.n_kv_head` 从 2 改成 4，手写版的注意力计算量如何变？KV cache 显存呢？
3. 为什么 `--bf16` 不需要 GradScaler，而 fp16 需要？
4. `Trainer` 保存的 checkpoint 目录结构，与手写 `last.pt` 各有什么优缺点？
