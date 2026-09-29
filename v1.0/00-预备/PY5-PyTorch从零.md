# 预备篇 PY5 · PyTorch 从零

> **配套代码**：`v1.0/00-预备/code/py5_torch_from_zero.py`（Tag：`F-pre-py5`）
> **配套 notebook**：`v1.0/notebooks/00-预备/N22_PyTorch从零.ipynb`（Tag：`N-22`）
> **对应理论篇**：`01-基础/01-神经网络与反向传播.md`（下文凡引用数学结论以【…】标注，不重复推导）
> **前置知识**：PY4「NumPy 与张量思维」；C/C++ 的数组、指针、结构体与编译期概念（用于对照）；第 1 章已出现的公式会在本文件给出工程语义。
> **预计学时**：4 学时（含手写线性回归与两层 MLP 两个训练闭环）
> **【文档｜DOC-PY5】**（doccode = `PY5`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## PY5.0 阅读约定

本章面向「有 C/C++ 经验、PyTorch 近乎零基础」的读者，目标是**建立可运行的心智模型**，而不是再讲一遍数学。约定如下：

- 本章条目编号为 `PY5.<节>.<序>`，Tag 为 `D-`（定义）、`Cd-`（代码）、`R-`（注）、`A-`（算法）、`Ex-`（练习）等，见《00-风格与编号规范》。
- 涉及数学推导时只给结论与引用。例如「交叉熵是什么」「反向传播递推式」见【定义 1.2.4】与【定理 1.3.4】，本章不再推导。
- 所有代码片段均为 CPU 可运行的最小示例；完整闭环见配套文件 `F-pre-py5` 与 notebook `N-22`。
- 术语首次出现给出英文与 C/C++ 对照。

## PY5.1 Tensor 基础

**【定义 PY5.1.1｜D-PY5.1.1】（张量：带元数据的同构多维数组）**

`torch.Tensor` 是 PyTorch 的基本数据结构：一块**同构**（所有元素同一类型）的内存缓冲区，加上解释它的形状（`shape`）与步长（`stride`）元数据。与 NumPy 的 `ndarray`（见《PY4》）语义基本一致，但额外携带三个元数据：

- `dtype`：元素类型，如 `torch.float32`、`torch.int64`、`torch.bool`；
- `device`：数据所在设备，如 `cpu`、`cuda:0`（MPS 等）；
- `requires_grad`：该张量是否是自动微分的起点（详见 PY5.3）。

C/C++ 对照：把 Tensor 看成「`T* buffer` + `len` + `stride[]` + 类型标签」的结构体。区别是维度、dtype、device 的合法性检查全部发生在**运行时**，编译期不做任何约束；写错只会抛异常，不会编译失败。

**【定义 PY5.1.2｜D-PY5.1.2】（dtype 与 device 的相容规则）**

参与同一运算的所有张量必须 dtype 相容、device 相同，否则运行时抛 `RuntimeError`。默认浮点 dtype 是 `torch.float32`（NumPy 默认是 `float64`），整数索引默认 `torch.int64`。C/C++ 对照：

- dtype 类似 `float*` 与 `double*` 的区别，混用需要显式转换（`tensor.to(torch.float64)`、`tensor.float()`）；
- device 类似两套地址空间（主存 vs 显存），一次运算的两个操作数必须在同一设备；跨设备要显式 `.to(device)`。CPU 与 GPU 张量直接相加是最常见的报错之一（见 PY5.8）。

**【代码 PY5.1.3｜Cd-PY5.1.3】（构造、属性与 NumPy 互操作）**

```python
import numpy as np
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # 按字面量构造，dtype 推断为 float32
b = torch.zeros(2, 3, dtype=torch.float64)          # 显式 dtype
c = torch.arange(6, dtype=torch.int64).reshape(2, 3)
d = torch.randn(2, 3)                                # 标准正态

print(a.shape, a.dtype, a.device)                    # torch.Size([2, 2]) torch.float32 cpu
print(a + 1.0)                                       # 逐元素加；标量自动广播

arr = np.ones((2, 3), dtype=np.float32)
t_from_np = torch.from_numpy(arr)                    # 零拷贝：与 arr 共享内存
t_np = c.numpy()                                     # 仅 CPU、dtype 相容时可零拷贝回 NumPy
t_copy = torch.tensor(arr)                           # 复制一份数据
t_alias = torch.as_tensor(arr)                       # 尽量复用底层内存
print(t_from_np is arr, t_copy.data_ptr() == arr.ctypes.data)
```

**【注 PY5.1.4｜R-PY5.1.4】（拷贝语义：共享、视图与深拷贝）**

PyTorch 与 NumPy 一样区分「视图」与「拷贝」，混淆二者是隐蔽 bug 的来源：

- `torch.from_numpy(arr)` / `tensor.numpy()`：**零拷贝**共享底层缓冲区，改一个另一个同步变化（类似 C 的指针别名）；
- `tensor.clone()`：**深拷贝**，新缓冲区，互不影响；梯度也会被复制（配合 `detach` 使用）；
- `tensor.detach()`：只切断自动微分，**不复制**数据；
- `torch.tensor(arr)`：复制；`torch.as_tensor(arr)`：dtype/device 相容时尽量复用。
- 形状操作 `view`/`reshape`/`transpose`/`permute` 大多返回**视图**（PY5.2），是否共享内存由 `stride` 决定。

**【注 PY5.1.5｜R-PY5.1.5】（requires_grad：进入自动微分世界的开关）**

`requires_grad=True` 的张量会被 autograd 记录；由它参与运算产生的张量自动继承该属性并挂上 `grad_fn`。叶子张量的梯度累加到 `.grad`，细节见 PY5.3。

## PY5.2 形状操作与广播

**【定义 PY5.2.1｜D-PY5.2.1】（view 与 reshape：改变解释方式还是复制数据）**

- `view(*shape)`：在**内存连续**（contiguous）的前提下重新解释同一缓冲区，不复制，返回视图（等价于 C 的指针重解释）；
- `reshape(*shape)`：语义上「尽量 view，不行才复制」。当张量非连续时，`reshape` 会隐式拷贝并返回新张量；
- 二者的输出元素值与顺序一致，差别只在是否与源共享内存。

**【定义 PY5.2.2｜D-PY5.2.2】（transpose/permute 与 contiguous）**

`transpose(d0, d1)` 交换两个维度，`permute(*dims)` 按给定顺序重排所有维度；两者都只改 `stride` 元数据、**返回视图、不搬数据**。因此结果通常**非连续**，此时 `view` 会失败（报 `view size is not compatible...`），需先 `x = x.contiguous()`（或改用 `reshape`）把数据按新顺序压实。矩阵乘、`view` 等算子要求连续布局。

**【定义 PY5.2.3｜D-PY5.2.3】（unsqueeze/squeeze 与广播）**

- `unsqueeze(dim)` 在指定位置插入一个大小 1 的维度（视图）；`squeeze(dim)` 去掉大小为 1 的维度；
- 广播（broadcasting）规则与 PY4 完全一致：**从最右维对齐**，缺失维补 1，大小为 1 的维可扩展到任意大小；不满足则报错。

**【代码 PY5.2.4｜Cd-PY5.2.4】（形状操作与连续性）**

```python
x = torch.arange(12).reshape(3, 4)
print(x.view(4, 3).shape)                # 可以：x 连续
print(x.t().shape, x.t().stride())       # 转置只改 stride
y = x.t().reshape(-1)                    # reshape 容忍非连续：隐式复制
try:
    x.t().view(-1)
except RuntimeError as e:
    print("view 失败：", str(e)[:60])
z = x.t().contiguous().view(-1)          # 先压实再 view
print(z.shape, x.unsqueeze(0).shape, x.unsqueeze(0).squeeze(0).shape)
print((x + torch.ones(4)).shape)         # (3,4)+(4,) 广播 -> (3,4)
```

**【注 PY5.2.5｜R-PY5.2.5】（与 NumPy 的对应表）**

| PyTorch | NumPy | 语义 |
| --- | --- | --- |
| `view` | `reshape`（能视图时） | 重解释，不复制 |
| `reshape` | `reshape` | 尽量视图，否则复制 |
| `transpose`/`permute` | `transpose`/`swapaxes` | 只改 stride，返回视图 |
| `contiguous` | `ascontiguousarray` | 按新布局压实，复制 |
| `unsqueeze`/`squeeze` | `expand_dims`/`squeeze` | 增删大小为 1 的维 |

差异：PyTorch 把「是否连续」显式暴露为 `tensor.is_contiguous()`，并要求某些算子的输入连续；NumPy 多数情形内部自动处理。

## PY5.3 autograd 与计算图

**【定义 PY5.3.1｜D-PY5.3.1】（计算图、叶子与非叶子张量）**

只要参与运算的张量 `requires_grad=True`，PyTorch 就在前向过程中动态搭建一张**有向无环图**：节点是张量，边是产生该张量的运算（挂在 `grad_fn` 上）。自动微分对这张图做**反向模式**的链式法则累加，其数学内容就是【定理 1.3.4】的反向传播递推；实现上等价于依次应用各算子的 VJP（**向量-雅可比积**）。

- **叶子张量**（leaf）：用户直接创建（`torch.tensor(..., requires_grad=True)` 或 `nn.Parameter`）的张量，是梯度的落点；
- **非叶子张量**：由运算产生，它的梯度是中间量，反向时默认**不保留**（可用 `retain_grad()` 强留）。

**【定义 PY5.3.2｜D-PY5.3.2】（requires_grad、grad_fn 与 .grad）**

- `requires_grad`：布尔标记，决定是否记录该张量；
- `grad_fn`：非叶子张量上指向「产生它的函数」的引用，叶子为 `None`；
- `.grad`：叶子张量在 `backward()` 后累积的梯度张量，形状与其数据相同；未反向时为 `None`。

**【定义 PY5.3.3｜D-PY5.3.3】（detach 与 torch.no_grad）**

- `tensor.detach()`：返回与源共享数据、但脱离计算图的新张量（`)` 对照：类似把指针的「可微分」标记去掉）；
- `with torch.no_grad():`：块内所有运算都不构图，用于推理与手动参数更新，可省显存、加速；
- `torch.inference_mode()`：更强的推理模式，允许的原地操作更少但更快。

**【注 PY5.3.4｜R-PY5.3.4】（backward、retain_graph 与梯度累加）**

- `loss.backward()` 从标量 `loss` 出发反向遍历图，把梯度**累加**到各叶子张量的 `.grad`；
- 默认反向后释放图；若想对同一张图多次反向（如多任务），传 `retain_graph=True`；
- 梯度是**累加**而非覆盖（对照 C 的 `+=`）。因此训练循环每步前必须清零：
  `opt.zero_grad(set_to_none=True)`（推荐）或 `opt.zero_grad()`。
- 对非标量调用 `backward()` 必须传 `gradient=` 给出外部梯度。

**【代码 PY5.3.5｜Cd-PY5.3.5】（autograd 最小示例与梯度检查）**

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3 * x            # y = x^2 + 3x
print(y.grad_fn)             # <AddBackward0>
y.backward()
print(x.grad)                # dy/dx = 2x+3 = 7

# 梯度累加：不清零会叠加上一次的结果
x.grad = None
(x * x).backward()
(x * x).backward()
print(x.grad)                # 4 + 4 = 8

# 数值梯度对照：中心差分，误差 O(eps^2)（见【算法 1.5.1】）
f = lambda t: (t * t + 3 * t).sum()
t = torch.tensor([2.0], requires_grad=True)
f(t).backward()
eps = 1e-4
num = (f(t.detach() + eps) - f(t.detach() - eps)) / (2 * eps)
print("解析", t.grad.item(), "数值", num.item())
```

**【注 PY5.3.6｜R-PY5.3.6】（与第 1 章反向传播的对应）**

本章的 `.backward()` 就是把【定理 1.3.4】的递推在计算图上自动执行：线性层、逐元素激活各自贡献一个 VJP（【命题 1.3.2】与【命题 1.3.1】），输出层的误差信号由损失决定（【推论 1.3.6】）。工程上你无需手写递推，但必须理解「谁依赖谁」，否则连不上图（见 PY5.8 的 `detach`/`no_grad` 陷阱）。

## PY5.4 nn.Module：模型的组织单位

**【定义 PY5.4.1｜D-PY5.4.1】（nn.Module：参数注册与子模块）**

`nn.Module` 是模型的基类。把 `nn.Parameter` 或子 `nn.Module` 赋值为实例属性时，PyTorch 会**自动注册**它们：

- 注册的 `nn.Parameter` 出现在 `model.parameters()` 中，可被优化器收集、被 autograd 当作叶子；
- 注册的子模块出现在 `model.children()`/`model.modules()` 中，递归管理其参数；
- `model.to(device)`、`model.train()/eval()`、`model.state_dict()` 都作用于整棵注册树。

C/C++ 对照：`nn.Module` 像一个「自带反射的结构体」，它知道自己包含哪些子对象与参数；`forward` 相当于虚函数，`__call__` 会对 `forward` 加钩子（hook）与状态处理。

必须实现 `forward(self, x)`；调用时写 `model(x)`（不要直接调 `model.forward(x)`，会跳过 hooks）。

**【定义 PY5.4.2｜D-PY5.4.2】（state_dict 与 load_state_dict）**

`state_dict()` 返回「参数名 → 张量」的有序字典（含 buffer，如 BatchNorm 的 running stats）。`load_state_dict(sd)` 按名字把值拷回（默认严格匹配键集合，`strict=False` 可放宽）。它是保存/加载与迁移学习的基础，见 PY5.7。

**【注 PY5.4.3｜R-PY5.4.3】（train/eval 模式）**

`model.train()` 与 `model.eval()` 只切换模块的布尔标志，影响 `Dropout`、`BatchNorm` 等的行为：

- `train()`：Dropout 随机丢弃、BatchNorm 用当前 batch 统计并更新 running stats；
- `eval()`：Dropout 关闭、BatchNorm 用固定 running stats。
- 二者**不影响梯度**；推理时仍应包在 `torch.no_grad()` 中。

**【代码 PY5.4.4｜Cd-PY5.4.4】（两层 MLP，对照【定义 1.1.3】与【代码 1.1.4】）**

```python
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                       # 必须：初始化注册表
        self.fc1 = nn.Linear(in_dim, hidden)     # 子模块，自动注册
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = TwoLayerMLP(4, 16, 3)
print(sum(p.numel() for p in model.parameters()))     # 参数量
print([n for n, _ in model.named_parameters()])        # fc1.weight fc1.bias fc2.weight fc2.bias
```

这与【代码 1.1.4】的 `MLP.forward` 同构：`c_fc` → 激活 → `c_proj`；此处把 `relu²` 换成了 `nn.ReLU`，结构未变。

**【注 PY5.4.5｜R-PY5.4.5】（参数、缓冲区与模块遍历）**

- `model.parameters()` / `named_parameters()`：递归取所有 `nn.Parameter`；
- `model.buffers()` / `named_buffers()`：非参数的状态张量（如 running mean）；
- `model.modules()`：含自身的所有子模块；`model.children()`：直接子模块。
- 注意 `list(model.parameters())` 是**视图式引用**，优化器持有的是同一批张量对象，`.to(device)` 后需重建优化器或确认已同步（见 PY5.7 的顺序陷阱）。

## PY5.5 损失函数与优化器

**【定义 PY5.5.1｜D-PY5.5.1】（nn.CrossEntropyLoss：logits 输入、内部含 log-softmax）**

`nn.CrossEntropyLoss()`（等价 `nn.LogSoftmax` + `nn.NLLLoss`，数值更稳）对标【定义 1.2.4】的交叉熵。约定：

- 输入 `input` 是**未归一化的 logits**，形状 `(N, C)`（或 `(N, C, d1, ...)`）；
- 目标 `target` 是类别索引 `int64`，形状 `(N,)`（或 `(N, d1, ...)`）；
- 不要自己先做 `softmax` 再传入（会重复且数值不稳）；
- `ignore_index=` 可屏蔽某些位置（如 padding，见【注 1.7.2】）；
- 其梯度对 logits 是「预测概率减 one-hot」，见【命题 1.2.9】。

**【定义 PY5.5.2｜D-PY5.5.2】（nn.MSELoss：回归目标，对照【定义 1.2.5】）**

`nn.MSELoss()` 计算均方误差；`reduction='mean'|'sum'|'none'` 控制是否对 batch 与元素求平均。PyTorch 默认 `mean`（对全部元素平均）。注意【定义 1.2.5】带因子 `1/2`，PyTorch 不带，仅差常数因子，不影响最优解。

**【定义 PY5.5.3｜D-PY5.5.3】（优化器：SGD 与 Adam）**

优化器持有「参数列表 + 超参」，在 `step()` 中用已有的 `.grad` 更新参数。

- `torch.optim.SGD(params, lr, momentum=0.0, weight_decay=0.0)`：随机梯度下降，可选动量与权重衰减；
- `torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`：自适应矩估计，默认学习率 `1e-3`；
- 通用 AdamW（解耦权重衰减）是 LLM 训练的默认选择（见第 2 章）。

必须用 `model.parameters()` 构造，且**参数对象要与模型实际使用的张量一致**。

**【注 PY5.5.4｜R-PY5.5.4】（参数组与逐组学习率）**

`optim.SGD([{"params": backbone.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}], lr=1e-3)` 可给不同子模块设不同学习率（微调常用）。构造函数里的 `lr` 是未指定组时的默认值。运行时可用 `scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=...)` 调整（见第 1 章训练工程）。

**【注 PY5.5.5｜R-PY5.5.5】（zero_grad 与 set_to_none）**

`opt.zero_grad()` 遍历参数把 `.grad` 置零；`opt.zero_grad(set_to_none=True)` 置为 `None`，省内存且更快，是推荐写法。忘记清零等价于对累积梯度做更新（见 PY5.8、[R-PY5.3.4]）。

## PY5.6 数据管线：Dataset 与 DataLoader

**【定义 PY5.6.1｜D-PY5.6.1】（Dataset 协议）**

自定义数据集只需实现两个方法（对照 PY2 的协议/魔术方法）：

- `__len__(self) -> int`：样本总数；
- `__getitem__(self, index)`：返回第 `index` 个样本（通常是张量元组，如 `(x, y)`）。

`torch.utils.data.TensorDataset(x, y)` 是最简单的内置实现。映射式数据集应实现上面的协议；流式/迭代式数据集用 `IterableDataset`（实现 `__iter__`）。

**【定义 PY5.6.2｜D-PY5.6.2】（DataLoader：batch/shuffle/collate/num_workers）**

`DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None, drop_last=False, pin_memory=False)` 把 `Dataset` 变成可迭代的 mini-batch 生成器：

- `batch_size`：一个 batch 的样本数；
- `shuffle`：每个 epoch 是否打乱索引（训练为 `True`，验证为 `False`）；用 `torch.Generator` 可固定随机源；
- `collate_fn`：把「样本列表」拼成一个 batch 张量，默认对张量做 `stack`、对序列做填充；序列任务常自定义；
- `num_workers`：子进程数（`0` 表示主进程加载）。多进程可加速 I/O 与预处理，但在 notebook/Windows 下有额外约束；
- `pin_memory=True`：把数据放到锁页内存，配合 GPU 的 `non_blocking` 拷贝加速。

**【代码 PY5.6.3｜Cd-PY5.6.3】（TensorDataset 与自定义 Dataset）**

```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 内置
ds = TensorDataset(torch.randn(100, 4), torch.randint(0, 3, (100,)))
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
xb, yb = next(iter(dl))
print(xb.shape, yb.shape)                      # (16,4) (16,)

# 自定义：__len__ + __getitem__
class ToyDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys
    def __len__(self):
        return self.xs.shape[0]
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]

toy = ToyDataset(torch.randn(10, 4), torch.randn(10, 1))
print(next(iter(DataLoader(toy, batch_size=4)))[0].shape)   # (4,4)
```

**【注 PY5.6.4｜R-PY5.6.4】（与经验风险的对应）**

一个 epoch 遍历 `DataLoader` 得到的每个 mini-batch 就是一次经验风险的 Monte-Carlo 估计：对 batch 内样本求平均即得【定义 1.2.4】中 `L(θ)=1/N Σ ℓ_i` 的 mini-batch 版本（随机梯度）。`shuffle=True` 保证采样近似无偏；`drop_last` 可丢弃不足一个 batch 的尾部（BatchNorm 等要求 batch 大小稳定时使用）。

## PY5.7 训练循环、保存/加载与 AMP

**【算法 PY5.7.1｜A-PY5.7.1】（标准训练循环五步）**

对每个 epoch、每个 mini-batch 循环执行：

1. **前向**：`pred = model(x)`；
2. **算损失**：`loss = criterion(pred, y)`；
3. **清零梯度**：`opt.zero_grad(set_to_none=True)`；
4. **反向**：`loss.backward()`（等价于【定理 1.3.4】的递推）；
5. **更新参数**：`opt.step()`。
   验证阶段：`model.eval()` + `with torch.no_grad():`，只做 1、2 步并累计指标。

顺序要点：`zero_grad` 必须发生在 `backward` 之前（否则累加到上一步的梯度）；`step` 必须在 `backward` 之后。

**【代码 PY5.7.2｜Cd-PY5.7.2】（训练循环与 checkpoint）**

```python
def train_one_epoch(model, loader, criterion, opt, device):
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)                       # 1 前向
        loss = criterion(pred, yb)             # 2 损失
        opt.zero_grad(set_to_none=True)        # 3 清零
        loss.backward()                        # 4 反向
        opt.step()                             # 5 更新
        total += loss.item() * xb.size(0); n += xb.size(0)
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item() * xb.size(0); n += xb.size(0)
    return total / max(n, 1)
```

**【注 PY5.7.3｜R-PY5.7.3】（保存/加载：state_dict、优化器状态与续训）**

- 只存参数：`torch.save(model.state_dict(), path)`，加载 `model.load_state_dict(torch.load(path, map_location="cpu"))`；
- 完整 checkpoint（含优化器、epoch、步数、随机源）才可续训：

```python
ckpt = {"model": model.state_dict(), "opt": opt.state_dict(),
        "epoch": epoch, "step": global_step}
torch.save(ckpt, "ckpt.pt")

state = torch.load("ckpt.pt", map_location="cpu")
model.load_state_dict(state["model"])
opt.load_state_dict(state["opt"])
start_epoch = state["epoch"] + 1
```

- `map_location` 用于跨设备加载（GPU 存的在 CPU 上加载）；
- 用 `torch.save` 存的是 pickle，**只能加载可信文件**（安全提示）；
- 顺序陷阱：先 `model.to(device)`，再用 `model.parameters()` 建优化器；或加载后重建优化器，避免参数对象不一致。

**【注 PY5.7.4｜R-PY5.7.4】（AMP 与 GradScaler 初识）**

混合精度（Automatic Mixed Precision, AMP）用低精度（`float16`/`bfloat16`）做前向/反向以省显存、加速，用 `float32` 做参数更新：

```python
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    loss = criterion(model(xb), yb)
scaler.scale(loss).backward()          # 放大损失，防 fp16 下溢
scaler.step(opt)
scaler.update()
```

- `fp16` 动态范围窄，需 `GradScaler` 缩放损失；`bf16` 范围与 fp32 相同，通常无需 scaler；
- CPU 上无 CUDA autocast（可用 `bfloat16` 的有限支持），本章代码在无 GPU 时自动退化为纯 `float32`；
- AMP 与梯度检查点、`bf16` 细节见 PY6。

**【注 PY5.7.5｜R-PY5.7.5】（与【代码 1.6.2】的对照）**

【代码 1.6.2】是一次性全量数据上的最小训练循环，五步骨架与本算法完全一致；本章的 `F-pre-py5` 把它扩展为「DataLoader 划分 batch + 验证 + checkpoint + 设备/AMP 分支」的工程版本。

## PY5.8 常见错误诊断

**【注 PY5.8.1｜R-PY5.8.1】（in-place on leaf：叶子张量被原地修改）**

对 `requires_grad=True` 的叶子做原地操作（`x += 1`、`x[0] = 1`、`x.add_(...)`）会报 `a leaf Variable that requires grad is being used in an in-place operation`。原因：autograd 依赖前向时的原值，原地改动会破坏它。

- 训练中安全：`with torch.no_grad(): param.sub_(lr * param.grad)`，或直接用 `opt.step()`；
- 一般用非原地写法 `x = x + 1`；
- 非叶子张量在无其他消费者时可原地，但为可读性不推荐。

**【注 PY5.8.2｜R-PY5.8.2】（shape 不匹配）**

典型报错 `size mismatch`（`nn.Linear` 的 `in_features` 与输入最后一维不符）、广播失败、`view` 报 `not compatible`（非连续，PY5.2）。诊断：打印 `x.shape` 与 `layer` 定义，逐层核对；`view` 失败先 `.contiguous()`。

**【注 PY5.8.3｜R-PY5.8.3】（device 混用）**

`Expected all tensors to be on the same device`：模型在 GPU、数据在 CPU（或反向）。统一口径：定义 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`，`model.to(device)`，每个 batch `xb.to(device, non_blocking=True)`。

**【注 PY5.8.4｜R-PY5.8.4】（忘记 zero_grad）**

漏掉 `opt.zero_grad()` 会让梯度在 batch 间不断累加，等效学习率随时间暴涨、loss 迅速发散。反之，**故意**不清零可模拟大 batch（梯度累积，见第 1 章训练工程）：累积 `k` 个 batch 再 `step()`。

**【注 PY5.8.5｜R-PY5.8.5】（eval 未生效与其他陷阱）**

- `model.eval()` 后忘记切回 `model.train()`：下一轮训练 Dropout 被关、BN 统计冻结，指标异常；
- 推理阶段忘记 `torch.no_grad()`：显存暴涨（保存计算图）；
- 用 `with torch.no_grad():` 里更新参数：图被断开，梯度不进入；
- `loss.item()` 会导致同步（GPU 上略慢），累计指标时注意频率；
- `torch.load` 加载不可信文件有代码执行风险。

**【注 PY5.8.6｜R-PY5.8.6】（诊断速查表）**

| 症状 | 常见原因 | 处理 |
| --- | --- | --- |
| leaf in-place 报错 | 对叶子原地改 | `no_grad` 下更新或用 `opt.step()` |
| size mismatch | 形状/维度错 | 打印各层 shape，核对 `in_features` |
| all tensors same device | 设备不一致 | 统一 `device`，batch 显式 `.to()` |
| loss 爆炸/NaN | 忘 `zero_grad`、lr 过大、fp16 溢出 | 清零、降 lr、用 `GradScaler` |
| 验证指标异常 | `eval()`/`train()` 未切换 | 训练前 `train()`，验证前 `eval()` |
| view 失败 | 非连续 | `.contiguous()` 或 `reshape` |

## PY5.9 来源与许可

本章为原创教学整理，仅链接与改写官方文档的公开知识，未整段复制受限制文本。主要参考（均为官方文档/开源许可）：

- PyTorch 官方文档（Tensor、autograd、`nn`、`optim`、`data`、AMP）：<https://pytorch.org/docs/stable/>，BSD-3-Clause（PyTorch 仓库许可）。
- PyTorch Tutorials（*Learn the Basics*、*What is torch.nn really?*）：<https://pytorch.org/tutorials/>，BSD-3-Clause。
- `torch.utils.data` 文档（`Dataset`/`DataLoader`/`collate_fn`）：<https://pytorch.org/docs/stable/data.html>。
- 混合精度（`torch.amp`）文档：<https://pytorch.org/docs/stable/amp.html>。
- 术语与公式引用本仓第 1 章：《01-神经网络与反向传播》（Tag：`DOC-1`）。

搬运范围：仅改写概念性描述与 API 语义，代码示例为本仓自写或改写自官方 tutorial 的最小片段；不搬运任何数据集或模型权重。许可汇总由 lead 登记入 `tags/external.tsv`（`adapted_from` 边）。

## PY5.10 自测

**【练习 PY5.10.1｜Ex-PY5.10.1】（视图与拷贝）**

构造一个 `(3, 4)` 的连续张量，分别用 `view`、`transpose`、`reshape`、`contiguous().view` 得到 `(4, 3)`，用 `data_ptr()` 判断哪些与源共享内存，并解释。

**【练习 PY5.10.2｜Ex-PY5.10.2】（autograd 与梯度累加）**

对 `y = (x**3).sum()`（`x` 为叶子）验证解析梯度 `3x²`；再连续调用两次 `backward()` 而不清零，说明结果并解释原因（对照 [R-PY5.3.4]）。

**【练习 PY5.10.3｜Ex-PY5.10.3】（手写线性回归闭环）**

用 `nn.Linear(1,1)` 与 `nn.MSELoss` + `SGD` 拟合 `y = 3x + 2`（加少量噪声），跑 `F-pre-py5` 的线性回归部分，观察 loss 下降并打印最终 `weight`、`bias`。

**【练习 PY5.10.4｜Ex-PY5.10.4】（两层 MLP 分类闭环）**

用 `TwoLayerMLP` + `CrossEntropyLoss` + `Adam` + `DataLoader` 完成一个二/三分类玩具任务；加验证集、`model.eval()` 与 `no_grad`，并保存/加载 checkpoint 后续训一个 epoch。

**【练习 PY5.10.5｜Ex-PY5.10.5】（制造并修复五个错误）**

依次制造 PY5.8 的五类错误（叶子原地、shape 不匹配、device 混用、忘 `zero_grad`、`eval` 未生效），记录报错/现象，再按速查表修复。

**【练习 PY5.10.6｜Ex-PY5.10.6】（公式与实践对照）**

对照【定义 1.2.4】与【命题 1.2.9】，说明 `nn.CrossEntropyLoss` 的输入为什么必须是 logits；再对照【定理 1.3.4】，指出 `loss.backward()` 对应递推的哪一步。
