# 02 · 预训练（Pre-training）

目标：从随机初始化出发，在（缩小的）数学/Lean 语料上学会"预测下一个 token"，
得到能续写证明、可作为 SFT/RL 起点的语言模型。理论推导见
[01-预训练原理.md](01-预训练原理.md)，可运行实验见
`../notebooks/02-预训练/N09_tiny预训练复刻.ipynb`。

## 两版实现对照

| 维度 | `code/from_scratch/pretrain.py` | `code/with_api/pretrain_hf.py` |
|---|---|---|
| 模型 | 复用 `01-基础` 的 tinyGPT（`GPT`） | `transformers.GPT2LMHeadModel` 小配置 |
| 数据 | 内置多域合成语料 + `CharTokenizer` | `datasets` + `DataCollatorForLanguageModeling` |
| 训练循环 | 手写：AMP / grad-accum / warmup+cosine | `Trainer` 托管 |
| 分布式 | `torchrun` + `DistributedDataParallel` | `accelerate`（`Trainer` 自动） |
| 预算 | `--target-flops` / `--param-data-ratio` | 由 `--max-steps` 指定 |
| 课程 | `--domains` / `--final-domains` 线性插值 | 固定配比 |
| 适用 | 看清每个梯度从哪来 | 快速拿到可用 checkpoint |

两版都只在 next-token 交叉熵上训练，结束时都采样一段文本验证 loss 是否真下降。

## from_scratch 版运行

```bash
cd 02-预训练/code/from_scratch
python3 -m py_compile pretrain.py
python3 pretrain.py --config micro --max-iters 300 --batch-size 32 --domains "math:1"
# GPU 上开混合精度 + 梯度累积：
python3 pretrain.py --config tiny --device cuda --amp --batch-size 32 --grad-accum 4
# 用算力预算反解迭代数（C≈6ND）：
python3 pretrain.py --config tiny --target-flops 1e15 --domains math:1,code:1
# 两卡 DDP：
torchrun --nproc_per_node=2 pretrain.py --config tiny --device cuda --amp --param-data-ratio 20
```

> 依赖：仅 `torch`。`pretrain.py` 通过 `sys.path` 复用
> `../../../01-基础/code/from_scratch` 的 tinyGPT 接口，因此请在完整 v1.0 目录树内
> 运行，或把该目录加入 `PYTHONPATH`。

## with_api 版运行

```bash
cd 02-预训练/code/with_api
pip install "transformers>=4.46" "datasets>=3.0" accelerate torch
python3 pretrain_hf.py --max-steps 60 --block-size 128 --batch-size 8
python3 pretrain_hf.py --model-name sshleifer/tiny-gpt2 --max-steps 30   # 真小基座
```

> 默认完全离线：本地造语料、`GPT2Config` 随机初始化；仅当 `--model-name` 非空时联网下载。

## 预期 loss 曲线

字符级、tiny 模型、内置语料上，交叉熵 $L$ 与困惑度 $\mathrm{PPL}=e^{L}$ 大致为：

| 阶段 | loss 量级 | PPL 量级 | 现象 |
|---|---|---|---|
| 初始化 | $\ln V\approx 4.2$ | $\approx 70$ | 完全随机 |
| 前 50 步 | 2.0 – 3.0 | 7 – 20 | 学会空格/换行/常见 token |
| 200 – 400 步 | 0.3 – 1.0 | 1.3 – 2.7 | 记住模板，能拼 `theorem ... := by` |
| 过拟合 | 小于 0.1 | 趋近 1 | 训练 loss 很低但验证抬头（数据太小） |

判定"训练有效"的最低标准：loss 单调下降且明显低于 $\ln V$。若 loss 不降，先查学习率
（过大震荡 / 过小不动）、batch、以及标签是否被误置为 -100。若验证 loss 先降后升，
说明小语料过拟合——这正是"预训练需要海量语料"的直接证据。

## 与 AlphaProof / nanoproof 的对应

* `nanoproof/nanoproof/pretrain.py`：Nemotron-CC-Math（约 20B token），DDP + MuonAdamW +
  梯度累积，用 `target_flops` 反推迭代数 —— 对应本目录 `--target-flops` / `--param-data-ratio`。
* `nanoproof/nanoproof/midtrain.py`：接着在 Lean-GitHub（约 65M token）代码上 midtrain ——
  对应 `--resume` + 换 `--domains code:1`。
* 优化器 `nanoproof/nanoproof/optim.py` 的 Muon 在 `05-LoRA` 与 RL 篇会再遇到；这里统一
  用 AdamW 降低门槛。

## 数据说明

真实预训练语料（Nemotron-CC-Math、Lean-GitHub 等）的 HF id、规模、许可证见
[`../datasets/README.md`](../datasets/README.md)。本目录脚本默认不下载大数据，只用
内置/合成小语料，保证任何机器上都能跑通并看到 loss 下降。
