# 【源代码｜F-pre-py4】v1.0/00-预备/code/py4_numpy.py — NumPy 与张量思维教学代码
"""预备篇 PY4「NumPy 与张量思维」配套代码。

对应理论篇：``v1.0/00-预备/PY4-NumPy与张量思维.md``
（条目：shape/dtype/strides、视图 vs 拷贝、花式/布尔索引、广播严格规则、
向量化与 matmul、归约与 keepdims、Generator 随机数、linalg 提示）。

只依赖 ``numpy``，全部为 CPU 路径；运行方式::

    python3 py4_numpy.py            # 运行自测 + 小基准

自测使用 ``assert``，任一失败即抛 ``AssertionError``。
"""
from __future__ import annotations

import time

import numpy as np


# 【F-pre-py4.describe_array｜函数】打印 ndarray 的形状/类型/步长元数据
def describe_array(a: np.ndarray) -> dict:
    """返回 ndarray 的关键元数据（对照【定义 PY4.1.2】/【定义 PY4.1.5】）。"""
    return {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "strides": a.strides,
        "ndim": a.ndim,
        "size": a.size,
        "itemsize": a.itemsize,
        "nbytes": a.nbytes,
        "c_contiguous": a.flags["C_CONTIGUOUS"],
    }


# 【F-pre-py4.classify_indexing｜函数】判断常见索引结果是视图还是拷贝
def classify_indexing() -> dict[str, str]:
    """对照【注 PY4.1.6】与【注 PY4.2.7】：返回各索引操作的「视图/拷贝」标签。"""
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = {
        "a[0]": "视图",
        "a[:, 1]": "视图",
        "a.T": "视图",
        "a[..., None]": "视图",
        "a[[0, 1]]": "拷贝",          # 花式索引 -> 拷贝
        "a[a > 5]": "拷贝",           # 布尔掩码 -> 拷贝
        "a.copy()": "拷贝",
        "a.reshape(6, 4)": "视图",    # 连续数组的 reshape 可返回视图
        "a.astype(np.float64)": "拷贝",
        "a[1:]": "视图",
    }
    # 运行期自证：base 是否为 None 区分拷贝与视图
    checks = {
        "a[0]": a[0], "a[:, 1]": a[:, 1], "a.T": a.T, "a[..., None]": a[..., None],
        "a[[0, 1]]": a[[0, 1]], "a[a > 5]": a[a > 5], "a.copy()": a.copy(),
        "a.reshape(6, 4)": a.reshape(6, 4), "a.astype(np.float64)": a.astype(np.float64),
        "a[1:]": a[1:],
    }
    for name, obj in checks.items():
        got = "拷贝" if obj.base is None else "视图"
        assert got == result[name], f"{name}: 期望 {result[name]}，实际 {got}"
    return result


# 【F-pre-py4.broadcast_shape｜函数】按严格规则推导广播结果形状（不符则 None）
def broadcast_shape(shape_a: tuple, shape_b: tuple):
    """实现【定义 PY4.3.1】的严格规则：右对齐、1 可扩、缺失补 1；冲突返回 None。"""
    na, nb = len(shape_a), len(shape_b)
    n = max(na, nb)
    out = []
    for k in range(n):
        da = shape_a[na - 1 - k] if k < na else 1
        db = shape_b[nb - 1 - k] if k < nb else 1
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            return None
    return tuple(reversed(out))


# 【F-pre-py4.matmul_naive｜函数】朴素三重循环实现 X @ W.T + b（对照【算法 PY4.4.2】）
def matmul_naive(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """纯 Python 三重循环；不推荐用于生产，仅作向量化的对照。"""
    B, D = X.shape
    M = W.shape[0]
    assert W.shape[1] == D and b.shape == (M,)
    Z = np.empty((B, M), dtype=np.float64)
    X64 = X.astype(np.float64)
    W64 = W.astype(np.float64)
    for i in range(B):
        for j in range(M):
            s = 0.0
            for k in range(D):
                s += X64[i, k] * W64[j, k]
            Z[i, j] = s + b[j]
    return Z


# 【F-pre-py4.matmul_vectorized｜函数】用 @ 与广播实现同一计算
def matmul_vectorized(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """X @ W.T + b：内层 k 交给 BLAS，j 循环交给广播（【算法 PY4.4.2】/【代码 PY4.4.5】）。"""
    return X @ W.T + b


# 【F-pre-py4.softmax_np｜函数】数值稳定的 NumPy softmax（对照【例 PY4.5.3】）
def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """沿 axis 做 softmax；先减最大值以防指数上溢（对照【注 1.2.13】）。"""
    m = z.max(axis=axis, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=axis, keepdims=True)


# 【F-pre-py4.rng_demo｜函数】Generator / 种子 / SeedSequence.spawn 演示
def rng_demo() -> dict:
    """演示【定义 PY4.6.2】与【注 PY4.6.3】：显式 Generator、可复现、并行子流。"""
    rng = np.random.default_rng(0)
    first = rng.standard_normal(3)
    again = np.random.default_rng(0).standard_normal(3)
    # 并行：从同一 SeedSequence 派生互相独立的子流
    children = np.random.SeedSequence(1234).spawn(2)
    child_vals = [np.random.default_rng(c).integers(0, 100, size=2) for c in children]
    return {
        "reproducible": bool(np.allclose(first, again)),
        "spawn_independent": bool(not np.array_equal(child_vals[0], child_vals[1])),
    }


# 【F-pre-py4.benchmark_vectorization｜函数】基准：朴素循环 vs 向量化
def benchmark_vectorization(B: int = 32, M: int = 32, D: int = 64, repeat: int = 5) -> dict:
    """返回两版耗时与加速比；数据由固定种子生成（【例 PY4.4.3】/【练习 PY4.9.3】）。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((B, D))
    W = rng.standard_normal((M, D))
    b = rng.standard_normal(M)

    ref = matmul_vectorized(X, W, b)

    naive = matmul_naive(X, W, b)
    vec = matmul_vectorized(X, W, b)
    t0 = time.perf_counter()
    for _ in range(repeat):
        naive = matmul_naive(X, W, b)
    t_naive = (time.perf_counter() - t0) / repeat

    t0 = time.perf_counter()
    for _ in range(repeat):
        vec = matmul_vectorized(X, W, b)
    t_vec = (time.perf_counter() - t0) / repeat

    assert np.allclose(naive, ref, atol=1e-8) and np.allclose(vec, ref)
    return {
        "shapes": {"X": X.shape, "W": W.shape, "b": b.shape, "Z": ref.shape},
        "naive_seconds": t_naive,
        "vectorized_seconds": t_vec,
        "speedup": (t_naive / t_vec) if t_vec > 0 else float("inf"),
    }


# 【F-pre-py4.selftest｜函数】PY4 全章自测（assert 失败即报错）
def selftest() -> None:
    """覆盖形状/步长、视图-拷贝、广播、向量化、归约、随机数、linalg。"""
    # 1. 元数据与 strides（【定义 PY4.1.2】/【定义 PY4.1.5】）
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    meta = describe_array(a)
    assert meta["shape"] == (3, 4) and meta["dtype"] == "float32"
    assert meta["strides"] == (16, 4) and meta["c_contiguous"]

    # 2. 视图 vs 拷贝（【注 PY4.1.6】/【注 PY4.2.7】）
    classify_indexing()

    # 3. 广播严格规则（【定义 PY4.3.1】/【例 PY4.3.2】/【例 PY4.3.3】）
    assert broadcast_shape((3, 4), (4,)) == (3, 4)
    assert broadcast_shape((3, 1), (1, 4)) == (3, 4)
    assert broadcast_shape((5, 3, 1), (3, 4)) == (5, 3, 4)
    assert broadcast_shape((3, 4), (3,)) is None      # 反例
    assert broadcast_shape((2, 3), (4, 3, 2)) is None  # 反例

    # 4. 向量化与 matmul（【算法 PY4.4.2】/【代码 PY4.4.5】）
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 5)); W = rng.standard_normal((3, 5)); b = rng.standard_normal(3)
    assert np.allclose(matmul_naive(X, W, b), X @ W.T + b)
    assert np.allclose(np.einsum("bd,md->bm", X, W) + b, X @ W.T + b)

    # 5. 归约与 keepdims（【注 PY4.5.2】/【代码 PY4.5.4】）
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert x.sum(axis=0).shape == (4,) and x.sum(axis=1).shape == (3,)
    assert x.max(axis=1, keepdims=True).shape == (3, 1)
    assert np.allclose(x - x.mean(axis=1, keepdims=True), x - x.mean(axis=1)[:, None])

    # 6. softmax（【例 PY4.5.3】）
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax_np(z)
    assert np.allclose(p.sum(axis=-1), 1.0) and p[1, 0] == p[1, 1] == p[1, 2]

    # 7. 随机数（【定义 PY4.6.2】/【注 PY4.6.3】）
    demo = rng_demo()
    assert demo["reproducible"] and demo["spawn_independent"]

    # 8. linalg：solve 优先于显式求逆（【定义 PY4.7.3】/【注 PY4.7.4】）
    A = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    rhs = rng.standard_normal(3)
    sol = np.linalg.solve(A, rhs)
    assert np.allclose(A @ sol, rhs)


# 【F-pre-py4.__main__｜入口】脚本入口：自测 + 小基准
if __name__ == "__main__":
    selftest()
    print("[selftest] OK")
    info = benchmark_vectorization()
    print(f"[benchmark] shapes={info['shapes']}")
    print(f"[benchmark] naive={info['naive_seconds']*1e3:.3f} ms  "
          f"vectorized={info['vectorized_seconds']*1e6:.3f} us  "
          f"speedup={info['speedup']:.1f}x")
    print("[broadcast]", broadcast_shape((5, 3, 1), (3, 4)))
    print("[indexing]", classify_indexing())
