# 预备篇 PY6 · PyTorch 进阶与生态

> **对应 AlphaProof / 对接对象**：`v1.0/04-价值头/code/with_api/value_head_api.py`、`v1.0/03-SFT/code/with_api/sft_hf.py`（二者用 HF 生态重写 from_scratch 版本）。
> **配套代码**：`v1.0/00-预备/code/py6_torch_ecosystem.py`（Tag：`F-pre-py6`）。
> **配套 notebook**：`v1.0/notebooks/00-预备/N23_PyTorch进阶与生态.ipynb`（Tag：`N-23`）。
> **预计学时**：3 学时。
> **前置知识**：PY5 的 `Tensor`/autograd/`nn.Module`/训练循环；数学结论沿用第 1 章（如【定义 1.1.3】、【命题 1.2.9】），本篇不重述。
> **【文档｜DOC-PY6】**（doccode = `PY6`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 0. 本篇定位

PY5 建立了「从零写训练循环」的能力；本篇面向两件事：**把常见深水区坑讲清**，以及**能读会用工业生态库**（02–08 章的 `with_api` 代码直接用它们）。全文只讲工程接口与语义，不重复数学推导；需要数学时引用第 1 章已有条目。

## 1. 深水区常见坑

**【定义 PY6.1.1｜D-PY6.1.1】（梯度检查点）**

梯度检查点（gradient checkpointing，也称 activation checkpointing）指：前向只保存少数「边界」激活，反向传播到某段时，从最近的边界重新前向计算该段激活，再求梯度。

- 接口：`torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)`，`fn` 是一段纯前向；
- 存储与计算互换：不检查点时反向需要的激活显存随层数线性增长（见【注 1.3.8】的显存账）；分段检查点可把激活显存从 $O(L)$ 降到 $O(\sqrt{L})$（$L$ 为层数、$\sqrt{L}$ 为分段数），代价是反向时多做一次前向；
- 只对占显存的大段（重复堆叠的 Transformer block）使用；不要包裹已经很小的算子；
- 被包裹的 `fn` 内若含随机性（dropout），需保证重算时 RNG 状态与首次一致（框架默认 `preserve_rng_state=True` 处理）；`use_reentrant=False` 支持 `**kwargs`、非张量输入与嵌套调用，是新代码的推荐用法。

**【注 PY6.1.2｜R-PY6.1.2】（非连续张量与 `view`/`reshape`/`contiguous`）**

- 张量的内存布局由 `stride` 决定；`transpose`/`permute`/`narrow` 只改视图与步长，往往得到**非连续**张量（`x.is_contiguous()` 为 `False`）；
- `view` 要求内存连续、可按下标展平，因此在非连续张量上会抛
  `RuntimeError: view size is not compatible with input tensor's size and stride`；
- `reshape` 语义更宽：能 `view` 就直接 `view`，否则自动等价于 `x.contiguous().view(...)`（多一次拷贝）；
- 需要矩阵乘或 `view` 前，用 `x.contiguous()` 显式拷成连续；但要注意这会在显存里短暂多一份副本，是显存尖峰的常见来源；
- 这与【注 1.4.2】的「张量布局与硬件」相呼应：硬件相关算子通常要求特定布局，布局转换是性能与显存的热点。

**【定义 PY6.1.3｜D-PY6.1.3】（混合精度与 `autocast`）**

混合精度（mixed precision）指：权重保留 fp32 主副本（master weights），部分前向算子用低精度执行。

- `torch.autocast(device_type, dtype)` 是一个**上下文管理器**：在其作用域内，框架按算子白名单自动把输入转成 `dtype`（常见为 `torch.bfloat16` 或 `torch.float16`），输出再按需转回；
- `autocast` 只改算子计算类型，**不改参数的 `dtype`**；优化器更新仍在 fp32 主副本上进行，避免小更新量被低精度吞掉；
- fp16：10 位尾数、动态范围窄，梯度易下溢/上溢；bf16：8 位指数（与 fp32 同动态范围）、尾数少，数值稳但精度略低；
- 训练优先 bf16（Ampere 及更新的 NVIDIA GPU、部分 AMD ROCm 支持）；推理常用 fp16。梯度检查（【注 1.5.2】）仍应在 fp64 下做，与训练精度无关。

**【注 PY6.1.4｜R-PY6.1.4】（`GradScaler` 与溢出处理）**

- fp16 的梯度常有极小值，直接 `backward` 会下溢为 0；`torch.amp.GradScaler` 的做法是：先用放大因子 `s` 缩放损失再反传，使梯度落在可表示区间，更新前再除回 `s`；
- 每次 `scaler.step(optimizer)` 会检查梯度是否出现 inf/nan，出现就跳过该步并调小 `s`，否则逐步放大；
- bf16 动态范围与 fp32 相同，**一般不需要** `GradScaler`；
- 顺序固定：`scaler.scale(loss).backward()` →（需要时 `scaler.unscale_(optimizer)` 后再 `clip_grad_norm_`）→ `scaler.step(optimizer)` → `scaler.update()`；不要在 unscale 后再 unscale 一次。

**【代码 PY6.1.5｜Cd-PY6.1.5】（检查点 + autocast 最小写法）**

```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

seg = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1))

def block(t):
    return seg(t)

x = torch.randn(16, 64, requires_grad=True)
with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16):
    y = checkpoint(block, x, use_reentrant=False)   # 只存边界激活，反向时重算
loss = y.square().mean()
loss.backward()
```

完整可运行版本见 `py6_torch_ecosystem.py --demo checkpoint` 与 `--demo autocast`。

## 2. 性能与显存

**【定义 PY6.2.1｜D-PY6.2.1】（`torch.profiler` 入门）**

- `torch.profiler.profile(activities=[CPU, CUDA])` 采集算子级事件；`prof.key_averages().table(sort_by="cpu_time_total")` 打印耗时排行，或导出 chrome trace 交给 TensorBoard/Perfetto 观察时间线；
- 常用指标：总 CPU 时间、总 CUDA 时间、kernel 数量、调用次数；`self_cuda_time_total` 去掉了子算子时间；
- 用 `schedule(wait=..., warmup=..., active=...)` 只采集若干步，降低开销；
- 典型用途：判断瓶颈是 CPU 数据加载还是 GPU 计算；发现大量小 kernel、频繁 device 同步（`.item()`、`.cpu()`）等反模式。

**【注 PY6.2.2｜R-PY6.2.2】（显存统计）**

- PyTorch 用缓存分配器（caching allocator）管理显存，常用查询：
  - `torch.cuda.memory_allocated()`：当前**张量实际占用**；
  - `torch.cuda.memory_reserved()`：分配器**从驱动保留**的总量（含空闲缓存）；
  - `torch.cuda.max_memory_allocated()`：峰值，训练中每步可用 `reset_peak_memory_stats()` 重置以定位是哪一步涨上去的；
- OOM 时先看 `reserved` 与 `allocated` 的差距：差距大说明是缓存碎片/峰值，而非净占用；激活显存大致随 batch × 序列长度 × 层数增长，检查点（【定义 PY6.1.1】）正是削减激活项。

**【注 PY6.2.3｜R-PY6.2.3】（`empty_cache` 的适用与不适用）**

- `torch.cuda.empty_cache()` 把缓存分配器里**当前空闲**的显存块还给驱动；它**不会**释放仍被张量引用的显存；
- 适用：想在长任务前后观察真实占用、要与其它进程共享同一块 GPU、或确认已 `del` 的张量确实归还；
- 不适用：在训练循环里频繁调用——这会把本可复用的缓存退掉，下一步又得重新申请，反而更慢且易碎片化；
- OOM 的正确优先级是：减小 batch/序列长度 → 用检查点 → 换 bf16 → 必要时 `empty_cache()`，而不是一 OOM 就清缓存。

**【定义 PY6.2.4｜D-PY6.2.4】（`torch.compile` 与固定形状）**

- `torch.compile(model)` 在首次调用时捕获计算图并生成融合后的 kernel（TorchDynamo 抓图 → AOTAutograd → Inductor 后端）；
- 形状固定时只编译一次、之后长期复用，适合重复的训练/推理循环；形状变化会触发**重编译**（recompile），显著变慢，可用 `dynamic=True` 或 `torch._dynamo.mark_dynamic` 提示动态维；
- 首次编译有固定开销（秒级到十秒级），短任务未必划算；
- 可组合：`torch.compile` + `autocast` + `checkpoint` 一起用；但会打断图的操作（`.item()`、依赖张量取值的 Python 控制流）会产生 graph break，削弱收益，写训练代码时应尽量延迟对张量的检查。

**【代码 PY6.2.5｜Cd-PY6.2.5】（profiler 与 compile 用法）**

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
x = torch.randn(256, 64)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    for _ in range(5):
        model(x).sum().backward()
        model.zero_grad(set_to_none=True)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))

compiled = torch.compile(model, backend="eager")   # CPU 用 eager 避免长编译
print(compiled(x).shape)
```

完整版本见 `py6_torch_ecosystem.py --demo profiler|memory|compile`。

## 3. 生态库速览（能读会用，对接 02–08 的 `with_api` 代码）

**【定义 PY6.3.1｜D-PY6.3.1】（`transformers`：`Auto*` 三件套）**

- `AutoConfig.from_pretrained(name)`：读模型超参与结构；`AutoTokenizer.from_pretrained(name)`：词表与特殊 token；`AutoModelForCausalLM.from_pretrained(name)`：因果语言模型权重；
- 常用参数：`torch_dtype=torch.bfloat16`、`low_cpu_mem_usage=True`、`device_map="auto"`；
- 前向返回 `ModelOutput`（`return_dict=True`）：含 `logits`，按需 `output_hidden_states=True` 拿每层隐状态 `hidden_states`（元组，末元素为最后一层），或 `last_hidden_state`；
- 训练时置 `model.config.use_cache = False` 关闭 KV cache，省显存；
- 分词器接口：`tok(texts, return_tensors="pt", padding=True, truncation=True)`；`tok.pad_token`、`tok.padding_side`；`apply_chat_template(...)` 负责把消息列表变成与训练一致的字符串（见 `sft_hf.py`）。

**【注 PY6.3.2｜R-PY6.3.2】（`Trainer` 与 `TrainingArguments`）**

- `TrainingArguments` 汇总训练超参：`output_dir`、`max_steps`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`learning_rate`、`lr_scheduler_type`、`warmup_ratio`、`bf16`/`fp16`、`logging_steps`、`save_steps`、`save_total_limit`、`report_to`、`seed`；
- `Trainer(model, args, train_dataset, data_collator)` 提供 `.train()` / `.save_model()` / `.evaluate()`，内部自动处理混合精度、梯度累积、日志、断点续训、单机多卡；
- 对照 from_scratch：`Trainer` 相当于把 `F-b0x-*` 里手写的「前向-损失-反向-更新」循环打包成黑盒；优点是省事，代价是控制粒度变粗（如在 loss 里做特殊掩码要自定义 `compute_loss`）。

**【定义 PY6.3.3｜D-PY6.3.3】（`peft`：`LoraConfig` / `get_peft_model`）**

- `peft` 用低秩增量 $\Delta W = BA$（$B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$）修改指定线性层，冻结基座、只训小矩阵；
- `LoraConfig(r, lora_alpha, lora_dropout, bias, task_type, target_modules)` → `model = get_peft_model(base, cfg)`；
- `target_modules` 名称随架构而变：GPT-2 用 `["c_attn", "c_proj"]`，LLaMA/Qwen 用 `["q_proj", "k_proj", "v_proj", "o_proj", ...]`；`F-b04-value_head_api.build_backbone` 就是按 `base.config.n_embd` 是否存在自动二选一；
- `model.print_trainable_parameters()` 看可训练比例；推理/合流用 `merge_and_unload()` 把增量并回权重；
- 4bit 量化基座 + LoRA 即 QLoRA（`F-b05-qlora_bnb`），依赖 `bitsandbytes`。

**【注 PY6.3.4｜R-PY6.3.4】（`trl`：`SFTTrainer` / `GRPOTrainer`）**

- `trl` 把常见后训练流程做成 Trainer：`SFTTrainer`（监督微调）、`DPOTrainer`/`PPOTrainer`/`GRPOTrainer`（偏好与 RL）；
- `GRPOTrainer` 需要 prompt 数据集与一个 `reward_funcs` 列表（标量奖励函数），内部完成 rollout、组内归一化优势与策略更新，对应 `F-b06-trl_grpo`；
- 与 from_scratch 的 `F-b06-grpo` 同构：数学与算法相同，`trl` 只是把采样、损失、分布式封装起来；
- 具体 API 随版本变化较大，使用前核对官方文档版本号。

**【定义 PY6.3.5｜D-PY6.3.5】（`accelerate`：`Accelerator`）**

- `Accelerator()` 统一抽象运行设备、混合精度与分布式后端（DDP/FSDP/DeepSpeed）；
- `accelerator.prepare(model, optimizer, dataloader)` 返回适配后的对象；`accelerator.backward(loss)` 取代 `loss.backward()`；`accelerator.gather()` 跨进程收集张量；
- 目标是让同一份训练脚本在 CPU、单卡、多卡之间切换而几乎不改代码；`accelerate config` 生成配置，`accelerate launch` 启动。

**【代码 PY6.3.6｜Cd-PY6.3.6】（生态库最小调用链，可选依赖缺失即跳过）**

```python
try:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, bias="none",
                     task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
except ImportError:
    print("peft 未安装，跳过")

try:
    from accelerate import Accelerator
    print("device =", Accelerator().device)
except ImportError:
    print("accelerate 未安装，跳过")
```

完整版本见 `py6_torch_ecosystem.py --demo ecosystem`。

**【注 PY6.3.7｜R-PY6.3.7】（安装与版本约束）**

- 最小安装：`pip install "transformers>=4.46" "peft>=0.13" "datasets>=3.0" trl accelerate`；QLoRA 另需 `bitsandbytes`（CUDA/ROCm）；
- 生态包彼此强耦合，版本不匹配是常见报错源；建议在 `requirements.txt` 中锁版本，并记录 `python/torch/transformers` 三者的组合；
- 离线或内网：设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，或预先把模型下载到本地缓存再用本地路径加载；
- 本仓环境版本清单见《00-资料清单》§9。

## 4. 与 from_scratch 代码的对照表

**【注 PY6.4.1｜R-PY6.4.1】（同一功能的两种写法）**

from_scratch 用于理解每一步梯度；with_api / 生态用于工程复用。二者做的是同一件事，接口对照如下（`F-*` 均为本仓文件键）：

| 功能 | from_scratch | with_api / 生态 |
| --- | --- | --- |
| 模型定义 | `F-b01-model`（`GPT`/`MLP` 类） | `transformers.AutoModelForCausalLM` |
| 分词器 | `F-b01-tokenizer_api` 的从简实现 | `AutoTokenizer` + `apply_chat_template` |
| 预训练循环 | `F-b01-train`、`F-b02-pretrain` | `F-b02-pretrain_hf`（`Trainer`） |
| SFT | `F-b03-sft`（token-masked CE） | `F-b03-sft_hf`（`Trainer` + `peft` LoRA） |
| 价值头 | `F-b04-value_head`、`F-b04-train_value_head` | `F-b04-value_head_api`（HF hidden + 小 MLP） |
| LoRA / QLoRA | `F-b05-lora`、`F-b05-finetune_lora` | `F-b05-peft_lora`、`F-b05-qlora_bnb` |
| GRPO | `F-b06-grpo` | `F-b06-trl_grpo`（`GRPOTrainer`） |
| 训练循环骨架 | 手写五步（前向/损失/zero_grad/backward/step） | `Trainer.train()` 或 `Accelerator` 封装 |

**【例 PY6.4.2｜E-PY6.4.2】（读 `F-b04-value_head_api` 的调用链并标注对象来源）**

`value_head_api.py` 把「冻结基座 + 小 MLP 价值头」跑通，对象来源可逐层标注：

1. `build_backbone` 内 `AutoTokenizer.from_pretrained` / `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)` —— **来源：`transformers`（HF）**；
2. `for p in base.parameters(): p.requires_grad_(False)` —— **来源：`torch`（autograd 的 `requires_grad`）**；
3. 可选 `LoraConfig(...)` + `get_peft_model(base, lora)` —— **来源：`peft`**；
4. `ValueHead`（`Linear→SiLU→Linear`，无 bins 时接 `Tanh`）与 `last_token_hidden` —— **来源：本仓本地定义**，其数学对应【注 1.7.3】的有界回归头；
5. 训练步：`with torch.no_grad(): model(**enc, output_hidden_states=True, return_dict=True)` → `last_token_hidden(...).detach()` → `head(hidden)` → `F.cross_entropy`/`F.mse_loss` → `loss.backward()` → `clip_grad_norm_` → `opt.step()` —— **来源：`torch`（`nn.functional` + `optim`）**，骨架与第 1 章【代码 1.6.2】一致。

结论：一个「with_api」文件里通常三层来源——HF 负责加载与结构，torch 负责张量与训练，仓库本地负责任务特定的头与数据逻辑。

**【注 PY6.4.3｜R-PY6.4.3】（迁移到 with_api 的注意事项）**

- 数学结论不迁移：损失、梯度公式仍在第 1 章（如【命题 1.2.9】），`with_api` 只是换了执行引擎；
- 易错点：dtype/device 混用、`padding_side` 与取 hidden 的位置不匹配（`last_token_hidden` 已处理 pad）、`Trainer` 默认行为（如自动 `eval`、自动 loss）与预期不符；
- 若需要改 loss（掩码、加权），要么自定义 `Trainer.compute_loss`，要么退回手写循环；
- `use_cache=False` 在训练时是显存优化，推理时必须打开（否则 `generate` 会重算全部前缀）。

## 5. 来源与许可

**【注 PY6.5.1｜R-PY6.5.1】（来源与许可）**

本篇内容为作者自撰的接口讲解与对照表；接口名称、参数与语义依据下述官方文档核对，只做要点改写与链接，未整段复制。官方文档多为 CC-BY / BSD / Apache 许可，示例代码引用遵循其各自许可。

| 材料 | URL | 许可 | 搬运范围 |
| --- | --- | --- | --- |
| PyTorch 文档（checkpoint / AMP / profiler / compile） | `https://pytorch.org/docs/stable/` | BSD-3-Clause | 仅链接与要点改写 |
| `torch.utils.checkpoint` | `https://pytorch.org/docs/stable/checkpoint.html` | BSD-3-Clause | 接口语义 |
| `torch.amp`（autocast / GradScaler） | `https://pytorch.org/docs/stable/amp.html` | BSD-3-Clause | 接口语义 |
| `torch.profiler` | `https://pytorch.org/docs/stable/profiler.html` | BSD-3-Clause | 接口语义 |
| `torch.compile` | `https://pytorch.org/docs/stable/torch.compiler.html` | BSD-3-Clause | 接口语义 |
| Transformers 文档 | `https://huggingface.co/docs/transformers` | Apache-2.0 | 仅链接与要点改写 |
| PEFT 文档 | `https://huggingface.co/docs/peft` | Apache-2.0 | 仅链接与要点改写 |
| TRL 文档 | `https://huggingface.co/docs/trl` | Apache-2.0 | 仅链接与要点改写 |
| Accelerate 文档 | `https://huggingface.co/docs/accelerate` | Apache-2.0 | 仅链接与要点改写 |

（建议 lead 汇总入 `tags/external.tsv` 时使用 `EXT-PY6-01` … `EXT-PY6-09`。）

## 6. 小结与自测

**【注 PY6.6.1｜R-PY6.6.1】（本章小结）**

1. 检查点用计算换显存（【定义 PY6.1.1】）；非连续张量是 `view` 报错的根因（【注 PY6.1.2】）；混合精度靠 `autocast` + fp32 主副本（【定义 PY6.1.3】）。
2. `profiler` 定位瓶颈，显存统计要区分 allocated/reserved，`empty_cache` 只在边界场景用（【定义 PY6.2.1】–【注 PY6.2.3】）。
3. 生态四件套各司其职：`transformers` 加载、`peft` 省参数、`trl` 管后训练、`accelerate` 管分布式（【定义 PY6.3.1】–【定义 PY6.3.5】）。
4. `with_api` 与 from_scratch 是同一算法的两种写法，对照表见【注 PY6.4.1】。

**【例 PY6.6.2｜E-PY6.6.2】（自测：检查点为什么省显存）**

解释梯度检查点省显存的机理与代价，并用 `py6_torch_ecosystem.py --demo checkpoint` 验证「检查点前后向」与「普通前后向」的输出和梯度一致（要求梯度最大差小于 1e-6）。

**【例 PY6.6.3｜E-PY6.6.3】（自测：非连续张量）**

构造一个 `.view(...)` 报错的例子，给出两种修复（`reshape` 与 `contiguous().view`），并说明两种修复在「是否拷贝显存」上的区别。

**【例 PY6.6.4｜E-PY6.6.4】（自测：fp16 与 bf16）**

结合【注 1.5.2】说明：为什么训练用 bf16 时通常不需要 `GradScaler`，而 fp16 需要？`autocast` 是否会改变模型参数的 `dtype`？

**【例 PY6.6.5｜E-PY6.6.5】（自测：读调用链）**

阅读 `F-b04-value_head_api`，画出其调用链并标注每个对象来自 `transformers` / `torch` / 本仓本地；在【注 PY6.4.1】的对照表里找出对应的 from_scratch 实现。

**【例 PY6.6.6｜E-PY6.6.6】（自测：找瓶颈）**

用 `torch.profiler` 对一个小 MLP 的训练循环采样，说出耗时最高的算子；在循环里插入一次 `.item()` 后重新采样，观察同步开销如何体现。
