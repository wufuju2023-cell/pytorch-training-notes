# PY4 · NumPy 与张量思维

> **【文档｜DOC-PY4】**（doccode = `PY4`）｜编号与 Tag 规范见《00-风格与编号规范》。
> **配套代码**：`v1.0/00-预备/code/py4_numpy.py`（filekey = `F-pre-py4`）。
> **配套 notebook**：`v1.0/notebooks/00-预备/N21_NumPy与张量思维.ipynb`（Tag = `N-21`）。
> **预计学时**：3 学时。
> **前置知识**：C/C++ 的数组、指针算术与内存布局；Python 语法基础（PY1/PY2）。
> **后续衔接**：PY5 的 `Tensor` 在语义上是 ndarray 的 GPU 化扩展，本文件的 shape/dtype/axis/广播/视图结论均直接迁移。

---

本章的目标不是背 API，而是建立**张量思维**：把「嵌套 `for` 循环 + 下标」的 C 式思维，换成「形状对齐 + 整块算子」的数组思维。全章不讲数学推导，只讲**形状、内存与执行语义**；涉及的数学结论一律引用第 1 章既有条目。

## 1. ndarray：形状、类型与内存布局

**【定义 PY4.1.1｜D-PY4.1.1】（ndarray）**

`numpy.ndarray` 是一个「同质定长元素的 N 维数组」对象，由两部分组成：**一段连续（或可由步长描述的）缓冲区**，以及**描述如何解释它的元数据**（`shape`、`dtype`、`strides`、`base`）。对照 C：

- 裸指针 `T*` 只有一个地址，丢失了长度与维度；
- `T arr[3][4]` 把维度编译进类型，维度数固定、不可运行期改变；
- `std::vector<T>` 是一维、可增长的连续缓冲；
- `ndarray` 把「维度 + 类型 + 步长」提升为**运行期数据**，因此可以描述转置、切片、广播等**不复制内存**的新视图。

**【定义 PY4.1.2｜D-PY4.1.2】（shape 与 dtype）**

`shape` 是每维长度的元组，例如 `(B, T, C)`；`dtype` 是元素类型（`int64`/`float32`/`bool_` 等）。`size` 是元素总数（`shape` 之积），`ndim` 是维数，`itemsize` 是单元素字节数，`nbytes = size * itemsize`。

- 元素类型一旦创建即固定；运算会在必要时按**类型提升规则**产生新 dtype（`int64 + float32 -> float64`）。
- 与 C 的对照：`dtype` 兼具「`T`」与「对齐/大小」信息；不像 C 的隐式整型提升那样随意，NumPy 的提升表是显式的、可预测的，这也是调试 shape 错误时最常被忽略的一环。

**【注 PY4.1.3｜R-PY4.1.3】（为什么必须同质定长）**

NumPy 的整块运算（以及后续 PyTorch/GPU）依赖「元素等宽、地址可由下标线性算出」。若允许 `list` 那样混装任意对象，就只能退化为逐元素的 Python 对象操作，失去连续内存与向量化。这与 C 中「数组元素必须同型」是同一约束，只不过 ndarray 把它带到了 Python 层。

**【定义 PY4.1.4｜D-PY4.1.4】（axis 语义）**

`axis` 是**维度的编号**，从 0 起、与 `shape` 的位置一一对应。`axis=0` 是「最外层」即第一个维度，`axis=-1` 是最后一维。归约算子（§5）与 `stack`/`concatenate` 等拼接算子都以 `axis` 指定作用方向。C 的对照：一维数组 `a[i]` 中 `i` 是下标，而这里的 `axis` 是**下标中的某一层被整体消去/合并**的方向，粒度是「整条轴」。

**【定义 PY4.1.5｜D-PY4.1.5】（strides：下标到字节地址的映射）**

`strides` 是每个维度上「下标 + 1 所跨过的字节数」的元组。元素地址由

```text
addr(i0, i1, ..., ik) = base + sum_j strides[j] * i_j
```

给出。C 连续数组（C order）的 strides 形如 `(n1*n2*itemsize, n2*itemsize, itemsize)`。**下标到地址是线性映射**，这正是 C 里 `p + i*stride` 指针算术的一般化；区别在于 ndarray 的 stride 可以任意（甚至为 0，见广播）。

**【注 PY4.1.6｜R-PY4.1.6】（视图 vs 拷贝：对照指针别名与 memcpy）**

- **视图（view）**：新 ndarray 与旧 ndarray 共享同一缓冲区（`base` 指向原数组），只换元数据。切片、`reshape`（在可能时）、`transpose`/`.T`、`ravel`（在可能时）都返回视图。对照 C：`T* q = p + k;`——两个指针别名同一块内存，写一个会改另一个。
- **拷贝（copy）**：新数组拥有独立缓冲区，`base is None`。花式索引、布尔索引、`np.array(a)`、`a.copy()`、`astype(...)` 都产生拷贝。对照 C：`memcpy` 出一份新副本。
- 判据：`np.shares_memory(a, b)` 或 `b.base is a`（或 `b.base is not None`）。写库/训练代码时，**「这个操作返回视图还是拷贝」直接决定后面改数据会不会污染原张量**，是最容易踩的坑之一。

**【代码 PY4.1.7｜Cd-PY4.1.7】（检查形状、步长与共享）**

```python
import numpy as np

a = np.arange(12, dtype=np.float32).reshape(3, 4)   # C 连续
b = a.T                                              # 视图：只换 strides
c = a[1:, ::2]                                       # 视图：切片
d = a[[0, 2]]                                        # 花式索引：拷贝

print(a.shape, a.dtype, a.strides)                   # (3,4) float32 (16,4)
print(b.shape, b.strides, np.shares_memory(a, b))    # (4,3) (4,16) True
print(c.base is a, d.base is None)                   # True True
a[0, 0] = 99
print(b[-1, 0])                                      # 99：视图看到同一内存
```

## 2. 索引、切片与赋值

**【定义 PY4.2.1｜D-PY4.2.1】（基本索引与切片）**

`a[i, j]`、`a[i:j:k]` 是**基本索引**：用整数、`slice`、`Ellipsis`（`...`）、`None`（`np.newaxis`）组合，结果是**视图**。省略号表示「其余维度全要」；`None` 会**新增**一个长度为 1 的轴（用于对齐广播，见 §3）。对照 C：`a[i]` 取一个元素或（数组的数组）一行，而 `a[..., None]` 这种「插入轴」的写法在 C 中没有对应物。

**【注 PY4.2.2｜R-PY4.2.2】（负索引与步长切片的代价）**

负索引从末尾计数（`a[-1]` 是最后一行）；步长 `k` 可为负（`a[::-1]` 反转）。这些同样是**视图**，不复制数据，只把 `offset` 与 `stride` 改成合适的值。但与 C 连续访问相比，步长不为 1 的切片会破坏访存局部性（cache 不友好），在热路径里应先 `np.ascontiguousarray` 再计算。

**【定义 PY4.2.3｜D-PY4.2.3】（花式索引：整数数组索引）**

用一个整数 ndarray（或列表）作为下标，如 `a[[0, 2]]`、`a[np.ix_(rows, cols)]`，称为**花式索引**。它按提供的下标**逐项取值并组装成新数组**，因此结果是**拷贝**，形状由下标数组的广播形状决定。对照 C：相当于 `for` 循环做 `out[k] = src[idx[k]]` 的 gather。

**【定义 PY4.2.4｜D-PY4.2.4】（布尔掩码）**

用与数组形状兼容的布尔数组 `mask` 作下标，`a[mask]` 返回所有 `True` 位置元素组成的**一维拷贝**。它等价于花式索引（下标的整数化），但可读性更好，常用于「选出一批样本/过滤 padding」。训练中 `ignore_index` 的掩码语义可对照【注 1.7.2】。

**【注 PY4.2.5｜R-PY4.2.5】（赋值语义：原地写入目标）**

`a[idx] = value` 把 `value` **写入 `a` 的目标位置**（原地），不改变 `a` 的形状，也不重建数组：

- 基本索引发起的赋值写回原缓冲区（会同时影响指向它的视图）；
- 花式/布尔索引发起的赋值是 scatter 语义（把值分散写入），右侧按广播形状展开；
- 与 C 的对照：这类似 `p[i] = v;`（原地）与「生成一个新数组」的区别，而**Python 列表**的 `a[i] = v` 与 `a[:] = v` 都同样原地，但 NumPy 更强调「左值是视图/掩码时写回谁」。

**【代码 PY4.2.6｜Cd-PY4.2.6】（三种索引的返回类型）**

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
print(a[1, 2], a[1:3, ::2])          # 基本索引：标量 / 视图
print(a[[0, 2]].shape)               # 花式索引：(2, 4) 拷贝
print(a[a % 2 == 0])                 # 布尔掩码：一维拷贝
v = a[0]; v[:] = -1                  # 视图原地写回 -> a[0] 全变 -1
a[a < 0] = 0                         # 掩码赋值：原地清零
```

**【注 PY4.2.7｜R-PY4.2.7】（对照 C 指针算术）**

- `a[i]` 基本索引 ≈ `*(p + i)`，返回视图而非值（标量下标才落成值）；
- `a[i:j]` ≈ 用 `(p + i, 长度)` 构造的一个「胖指针」，不拷贝；
- 花式/布尔索引 ≈ 显式 gather/scatter 循环，返回新缓冲区；
- 因此「拿一行去改」在本章是改原数据，而在某些语言里是拷贝——**这是 NumPy 里最常见的语义误判**。

## 3. 广播：严格规则与反例

**【定义 PY4.3.1｜D-PY4.3.1】（广播严格规则）**

两个形状参与逐元素运算时，NumPy 按下列规则对齐（**从右往左**逐维比较）：

1. **右对齐**：把 `shape` 元组的右端对齐，短的一方在左边补 1；
2. **1 可扩**：某维若一方为 1、另一方为 `m`，则该维按 `m` 扩展（逻辑上复制 `m` 次）；
3. **缺失维补 1**：短形状左边缺失的维度视为 1（第 1 条的重述）；
4. **否则报错**：某维两侧都不为 1 且不相等时，抛 `ValueError`。

广播结果是各维取「非 1 的那一方」的逐点最大值（再与补 1 合并）。

**【例 PY4.3.2｜E-PY4.3.2】（合法广播示例）**

```text
(3, 4) op (4,)      -> (3, 4)      # 右对齐，(4,) 视为 (1, 4) 再扩到 (3, 4)
(3, 1) op (1, 4)    -> (3, 4)      # 两个 1 各自扩展
(5, 3, 1) op (3, 4) -> (5, 3, 4)   # 右侧对齐后中间维相等、末维由 1 扩
()     op (3, 4)    -> (3, 4)      # 标量视为全 1
```

**【例 PY4.3.3｜E-PY4.3.3】（反例：不同时为 1 或相等则报错）**

```text
(3, 4) op (3,)      -> ValueError   # 右对齐：4 vs 3 冲突
(2, 3) op (4, 3, 2)-> ValueError    # 3 vs 2 冲突（注意是逐维比较，不是整体）
```

诊断口诀：**把两形状右对齐后逐列读；只有出现「1 与 任意」或「相等」才算合法**。训练中最常见的 bug 是「想把 `(C,)` 加到 `(B, T, C)`」，却写成了某个 `(T,)` 或 `(B,)`。

**【注 PY4.3.4｜R-PY4.3.4】（广播是零拷贝的 stride=0 视图）**

被扩展的维度并不真的复制内存：NumPy 把该维的 `stride` 记为 0，使「沿该维移动下标不改变地址」。对照 C：等价于让循环变量不参与地址计算，`addr = base`。因此广播本身几乎免费，**真正付费的是随后读入计算**；`np.broadcast_to` 可显式得到这种视图。

**【代码 PY4.3.5｜Cd-PY4.3.5】（广播实验与形状诊断）**

```python
import numpy as np

x = np.zeros((3, 4))
print((x + np.arange(4)).shape)          # (3, 4)
print((x + np.arange(3)[:, None]).shape) # (3, 4)：None 插轴 -> (3,1)

y = np.broadcast_to(np.arange(4), (3, 4))
print(y.strides, np.shares_memory(x, y)) # (0, 8) / 视具体 dtype；视图
try:
    x + np.arange(3)
except ValueError as e:
    print("broadcast error:", str(e)[:60])
```

**【注 PY4.3.6｜R-PY4.3.6】（与 PyTorch 广播一致）**

PyTorch `Tensor` 采用同一套广播规则（右对齐、1 可扩、缺失补 1），因此本章的结论可无损迁移到 PY5；`None` 插轴在 PyTorch 中写作 `unsqueeze` 或 `[..., None]`。与【注 1.4.2】的「batch 维放在最外层」配合，就能写出 `(B, T, C)` 与 `(C,)`、`(T, 1)` 等的对齐写法。

## 4. 向量化：从循环到矩阵乘与 `einsum`

**【注 PY4.4.1｜R-PY4.4.1】（为什么必须向量化）**

Python 解释器逐条执行字节码，纯 Python 双层循环的每次迭代都有对象创建与类型分派开销；而 NumPy 的整块算子在**编译好的 C/Fortran/BLAS 内核**里跑，还可利用 SIMD 与多线程。对照 C：这相当于把「手写逐元素循环」换成「调用高度的库函数」——但前提是你用**形状**而不是**下标**来表达计算。向量化的本质是「把循环藏进算子」，代码更短、通常也快一到两个数量级。

**【算法 PY4.4.2｜A-PY4.4.2】（朴素两层循环 -> 矩阵乘）**

以「`X @ W.T + b`」为例，把 C 式三层循环改写成一次 `matmul`：

```text
输入: X: (B, D), W: (M, D), b: (M,)
朴素: for i in range(B):
          for j in range(M):
              s = 0.0
              for k in range(D):
                  s += X[i, k] * W[j, k]
              Z[i, j] = s + b[j]
向量化: Z = X @ W.T + b          # (B,D)@(D,M) -> (B,M)，再广播加 b
```

三个量对应：`matmul` 把**内层 k 循环**压进 BLAS；`+ b` 把**j 循环**变成广播；剩下 i 维交给批量 matmul。训练里的全连接、注意力打分、投影全部是这一形状。

**【例 PY4.4.3｜E-PY4.4.3】（朴素实现与向量化基准）**

同一计算分别用三重 Python 循环与 `X @ W.T + b` 实现，固定随机种子并测量耗时，可稳定观察到后者快约两个数量级（具体倍数与 BLAS 线程、矩阵大小有关）。可运行代码与自测见【代码 PY4.4.5】及配套 `F-pre-py4`。

**【注 PY4.4.4｜R-PY4.4.4】（`einsum` 初识）**

`np.einsum("bd,md->bm", X, W)` 用**爱因斯坦求和记号**显式写出「哪些下标参与求和、哪些保留」。优点是把「批量矩阵乘、逐元素乘后求和、转置」等统一成一句可读的指标式，适合验证形状推导；缺点是不如专用算子（`@`）易被优化。读法：重复出现的下标若只出现在一侧则求和（或按需保留），输出下标决定结果维度顺序。

**【代码 PY4.4.5｜Cd-PY4.4.5】（三种等价写法）**

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((4, 5)); W = rng.standard_normal((3, 5)); b = rng.standard_normal(3)

Z1 = X @ W.T + b                       # 首选：BLAS 矩阵乘 + 广播
Z2 = np.einsum("bd,md->bm", X, W) + b  # 指标式
Z3 = ((X[:, None, :] - 0) * W[None, :, :]).sum(-1) + b   # 显式广播后归约（不推荐、仅对照）
print(np.allclose(Z1, Z2), np.allclose(Z1, Z3))
```

**【注 PY4.4.6｜R-PY4.4.6】（与 batch 反向公式的衔接）**

向量化后的 `Z = X @ W.T + b` 与【命题 1.4.1】的 batch 反向公式是同一套形状语言的产物：前向把 `(B, D)` 与 `(D, M)` 收缩成 `(B, M)`，反向把上游 `(B, M)` 与 `(B, D)` 收缩回 `(M, D)`。理解「哪个轴被收缩、哪个轴被保留」，就同时理解了前向与反向的形状。

## 5. 归约与轴

**【定义 PY4.5.1｜D-PY4.5.1】（归约算子）**

归约把某条（或全部）轴「消去」为一个值：`sum`、`mean`、`max`/`min`、`prod`、`std`/`var`、`argmax`/`argmin`、`any`/`all`。`axis=None`（默认）对全部元素归约、返回标量；`axis=k` 只消去第 `k` 维。对照 C：相当于在外层循环里维护一个累加器，但整条轴的循环被算子接管。

**【注 PY4.5.2｜R-PY4.5.2】（`axis` 与 `keepdims`）**

- `x.sum(axis=0)` 对 `(B, C)` 得到 `(C,)`；`axis=1` 得到 `(B,)`；`axis=(0, 1)` 得到标量。
- `keepdims=True` 会保留被消去的轴为长度 1，如 `(B, 1)`。它的意义在于**让结果仍能与原张量广播**：`x - x.mean(axis=1, keepdims=True)` 是合法的中心化，而 `x - x.mean(axis=1)` 会因 `(B,C)` 与 `(B,)` 右对齐冲突而报错（见【例 PY4.3.3】）。
- 口诀：**要拿归约结果去修正原张量，几乎总该带 `keepdims=True`**。

**【例 PY4.5.3｜E-PY4.5.3】（softmax 的向量化实现）**

softmax（定义见【定义 1.2.1】）在 NumPy 中就是「沿最后一轴归约 + 广播相减/相除」，数值稳定写法用最大值平移（对照【注 1.2.13】的 logsumexp 技巧）：

```python
def softmax_np(z, axis=-1):
    m = z.max(axis=axis, keepdims=True)      # 数值稳定：先减最大值
    e = np.exp(z - m)                        # 广播相减
    return e / e.sum(axis=axis, keepdims=True)
```

**【代码 PY4.5.4｜Cd-PY4.5.4】（归约与 keepdims 实验）**

```python
import numpy as np

x = np.arange(12, dtype=np.float64).reshape(3, 4)
print(x.sum(axis=0).shape, x.sum(axis=1).shape, x.sum().shape)      # (4,) (3,) ()
print(x.max(axis=1, keepdims=True).shape)                           # (3,1)
print(np.allclose(x - x.mean(axis=1, keepdims=True),
                  x - x.mean(axis=1)[:, None]))                     # 两种等价写法
```

**【注 PY4.5.5｜R-PY4.5.5】（归约 + 广播是「规范化」的标准范式）**

几乎所有归一化与损失都写成同一个三步式：**沿某轴归约 -> `keepdims` -> 广播回原形状**。LayerNorm、softmax、交叉熵的 log-sum-exp、批内均值中心化都是这一范式；把它记成形状模板，就不必逐次推导。

## 6. 随机数

**【注 PY4.6.1｜R-PY4.6.1】（全局状态 vs `Generator`）**

旧式 `np.random.*`（如 `np.random.randn`）操作一个**全局随机状态**，难以复现与并行；现代推荐用 `rng = np.random.default_rng(seed)` 得到一个独立的 `Generator`，再调用 `rng.standard_normal(...)`、`rng.integers(...)` 等方法。对照 C：全局状态像 `rand()` 的进程级隐藏状态，`Generator` 像把自己持有的 `state` 传给显式的随机函数——**状态可见、可保存、可并行**。

**【定义 PY4.6.2｜D-PY4.6.2】（种子与可复现性）**

给定相同种子与相同调用序列，`Generator` 产生相同结果。可复现性要求：(1) 全流程只用显式 `Generator` 而非全局函数；(2) 固定所有随机源（数据打乱、初始化、dropout）；(3) 记录库版本，因为不同版本对分布抽样方式的实现可能变化。`rng = np.random.default_rng(0)` 是本章与 PY5/PY6 的统一起点。

**【注 PY4.6.3｜R-PY4.6.3】（并行随机数：`SeedSequence`）**

多进程/多线程并行时，**不要**各进程用相同种子（会得到相同流）。用 `np.random.SeedSequence(seed).spawn(n)` 生成 `n` 个互相独立、可复现的子种子序列，每个 worker 用其一构造自己的 `Generator`。这与 C 中「每线程独立 RNG 状态以避免数据竞争」是同一工程需求。

**【注 PY4.6.4｜R-PY4.6.4】（与 PyTorch 随机源的对应）**

PyTorch 的全局随机源是 `torch.manual_seed`，并有 `torch.Generator` 可显式传参（如 `DataLoader(generator=...)`）；CPU 上的分布抽样与 NumPy 并非字节级一致，故**混用 NumPy 与 PyTorch 做数据增强时，两边都要固定**。数据管线通常用 `Generator` 生成可复现的 index 序列传给 `DataLoader`，细节见 PY5。

**【代码 PY4.6.5｜Cd-PY4.6.5】（Generator 与 spawn）**

```python
import numpy as np

rng = np.random.default_rng(0)
print(rng.standard_normal(3)[:2], rng.integers(0, 10, size=3))

same = np.random.default_rng(0)
print(np.allclose(rng.standard_normal(3), np.random.default_rng(0).standard_normal(3)))
```

## 7. 线性代数：`matmul` 与 `linalg`

**【定义 PY4.7.1｜D-PY4.7.1】（`matmul` / `@` 语义）**

`A @ B` 与 `np.matmul(A, B)` 对**最后两维**做矩阵乘：`(..., m, k) @ (..., k, n) -> (..., m, n)`，前面的维度按广播对齐，是**批量矩阵乘**。一维参与时按「左侧当行向量、右侧当列向量」处理；`np.dot` 是历史接口，二维时与 `matmul` 相同，但高维行为不同，**新代码统一用 `@`**。对照 C：`matmul` 是 BLAS `gemm` 的调度入口，批量维即 `gemm` 的批量循环。

**【注 PY4.7.2｜R-PY4.7.2】（批量 matmul 与广播组合）**

注意力打分是典型例子：`Q @ K.transpose(0, 2, 1)` 给出 `(B, T, T)`，其中 `B` 是广播/批量维、`(T, C) @ (C, T) -> (T, T)`。这也是【注 1.4.2】所说的「batch 折叠进最后两维线性代数」的具体形状。写代码时先写清楚最后两维的 `(m, k) @ (k, n)`，再看前面维是否广播即可。

**【定义 PY4.7.3｜D-PY4.7.3】（`np.linalg` 常用函数）**

`np.linalg` 提供：`det`（行列式）、`inv`（逆）、`solve`（解线性方程组，优先于显式求逆）、`lstsq`（最小二乘）、`norm`（范数，可指定 `axis`/`ord`）、`svd`、`qr`、`cholesky`、`eig`/`eigh`、`matrix_rank`、`pinv`。工程原则：**能用 `solve`/`lstsq` 就不要先求 `inv`**（数值稳定性与速度都更好），除非确实需要逆本身。

**【注 PY4.7.4｜R-PY4.7.4】（数值稳定与 dtype 提示）**

- 线性代数默认走 LAPACK，输入常被提升到 `float64`（或单精度对应例程）；用 `float32` 可省内存但精度低，梯度检查等场景应显式 `float64`（对照【注 1.5.2】）。
- 归一化前先 `ascontiguousarray` 或 `contiguous`，避免奇异 `strides` 拖慢 `gemm`。
- 形状错误（最后两维不满足 `(m,k)@(k,n)`）会在运行期抛 `ValueError`，且报错信息给出「(a,b) vs (c,d)」的对齐结果，是最有用的调试线索。

**【注 PY4.7.5｜R-PY4.7.5】（为 PY5 铺垫）**

PY5 的 `torch.Tensor` 把本章所有算子搬到 GPU：`@` 变为 `torch.matmul`（批量、广播规则一致），`np.linalg` 对应 `torch.linalg`，dtype/device 语义一一对应（只是多了 `device` 与自动求导）。因此**把本章的「形状 + 轴 + 广播」练熟，等于提前掌握了 PyTorch 报错信息里 90% 的形状问题的读法**。

## 8. 来源与许可

**【注 PY4.8.1｜R-PY4.8.1】（来源与许可）**

本章为原创编写，概念与 API 名称参考 NumPy 官方文档；未整段复制受版权保护的原文，属于概念对照与自撰示例：

| 材料 | URL | 许可 | 搬运范围 |
| --- | --- | --- | --- |
| NumPy User Guide（Array basics / Broadcasting / Copies and views） | https://numpy.org/doc/stable/user/ | BSD-3-Clause（NumPy） | 仅概念与 API 名称对照，未复制正文 |
| NumPy `random.Generator` 参考 | https://numpy.org/doc/stable/reference/random/generator.html | BSD-3-Clause（NumPy） | 仅 API 名称与语义说明 |
| NumPy `linalg` 参考 | https://numpy.org/doc/stable/reference/routines.linalg.html | BSD-3-Clause（NumPy） | 仅函数清单与用法原则 |
| PyTorch Reproducibility Notes | https://pytorch.org/docs/stable/notes/randomness.html | BSD-3-Clause（PyTorch） | 仅随机源对应关系说明 |

外部材料登记（`tags/external.tsv` 的 `EXT-*` 与 `adapted_from` 边）由 lead 汇总；本小节的 URL 与许可信息供其入表。

## 9. 自测

**【练习 PY4.9.1｜Ex-PY4.9.1】（形状推导）**

不运行代码，写出下列结果的 `shape`，再用 NumPy 验证：`(3, 4) + (4,)`；`(5, 1, 4) + (3, 4)`；`(2, 3) @ (3, 4)`；`(8, 2, 3) @ (3, 5)`；`a[mask]`（`mask` 与 `a` 同形）；`x.mean(axis=1, keepdims=True)`。

**【练习 PY4.9.2｜Ex-PY4.9.2】（视图还是拷贝）**

对 `a = np.arange(24).reshape(2, 3, 4)`，判断并验证：`a[0]`、`a[:, 1]`、`a[[0, 1]]`、`a.T`、`a.reshape(6, 4)`、`a[..., None]`、`a.copy()` 各是视图还是拷贝；指出修改哪些会影响 `a`。

**【练习 PY4.9.3｜Ex-PY4.9.3】（把朴素实现向量化并基准）**

给出一段用三层 Python 循环计算 `X @ W.T + b` 的朴素实现，改写为 `@` + 广播；用 `np.random.default_rng(0)` 造数据、`time.perf_counter` 计时并核对 `np.allclose`。要求报告两版耗时比值，并用「哪个轴被压缩进 BLAS、哪个轴交给广播」解释差异（对应【算法 PY4.4.2】与【代码 PY4.4.5】）。
