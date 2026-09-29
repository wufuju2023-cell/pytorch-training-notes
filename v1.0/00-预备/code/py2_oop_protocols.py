# 【源代码｜F-pre-py2】v1.0/00-预备/code/py2_oop_protocols.py — Python 语言 II 教学代码：类/协议/异常/生成器/装饰器
# 相关文档：《00-预备/PY2-Python语言II.md》（doccode = PY2）；配套 notebook：N-19
"""Python 语言 II 配套教学代码（纯标准库，CPU 可跑）。

覆盖大纲 §2 的 PY2 主题：

* 类与实例、类属性 vs 实例属性、继承与 MRO；
* 数据模型/魔术方法（``__repr__``/``__len__``/``__getitem__``/``__eq__``/``__hash__``/``__enter__``/``__exit__``）；
* ``property``、描述符概念、``__slots__``；
* ``dataclass``、``typing``、``enum``；
* 异常模型（``try/except/else/finally``、``raise ... from``）；
* 上下文管理器与 ``contextlib``；
* 装饰器与 ``functools``；
* 迭代器与生成器、``itertools``。

运行：``python3 py2_oop_protocols.py``（无第三方依赖）。
"""

from __future__ import annotations

import functools
import itertools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Protocol, Sequence, runtime_checkable


# ---------------------------------------------------------------------------
# 1. 类与实例：类属性 / 实例属性 / 运算符 / repr
# ---------------------------------------------------------------------------
# 【F-pre-py2.Vec2｜类】二维向量：类属性 + 实例属性 + 运算符重载 + __repr__
class Vec2:
    """最小二维向量，示范类属性与实例属性、运算符重载、``__repr__``。"""

    dim = 2                      # 类属性：所有实例共享（对照 C++ static 成员）
    __slots__ = ("x", "y")       # 去掉 __dict__，属性只能落在槽位（见 §3）

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y    # 实例属性：每个实例各自一份

    def __repr__(self) -> str:
        return f"Vec2({self.x!r}, {self.y!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec2):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)


# ---------------------------------------------------------------------------
# 2. 数据模型：容器协议 __len__ / __getitem__ / __iter__
# ---------------------------------------------------------------------------
# 【F-pre-py2.Buffer｜类】只实现容器协议即可被 len()/索引/for 使用
class Buffer:
    """极简可迭代容器：只需 ``__len__``/``__getitem__``/``__iter__``。"""

    def __init__(self, data: Sequence[int]) -> None:
        self._data = list(data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> int:
        return self._data[i]     # slice 情形也可在此显式处理

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)  # 委托给 list 的迭代器

    def __repr__(self) -> str:
        return f"Buffer({self._data!r})"


# ---------------------------------------------------------------------------
# 3. 属性、描述符思想与 __slots__ 内存收益
# ---------------------------------------------------------------------------
# 【F-pre-py2.Counter｜类】property 读只读派生值 + 描述符式校验 setter
class Counter:
    """示范 ``property``：把方法伪装成属性，并在 setter 中做校验。"""

    def __init__(self, value: int = 0) -> None:
        self._value = value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new: int) -> None:
        if new < 0:
            raise ValueError("value 必须非负")
        self._value = new

    @property
    def doubled(self) -> int:
        return self._value * 2   # 只读派生属性：无 setter


# 【F-pre-py2.memory_footprint｜函数】粗略比较有/无 __slots__ 的实例内存
def memory_footprint() -> tuple[int, int]:
    """返回 ``(普通类实例, __slots__ 实例)`` 的 ``sys.getsizeof``（字节）。"""

    class WithDict:
        def __init__(self) -> None:
            self.a = 0
            self.b = 0

    class WithSlots:
        __slots__ = ("a", "b")

        def __init__(self) -> None:
            self.a = 0
            self.b = 0

    import sys

    with_dict = WithDict()
    with_slots = WithSlots()
    # 普通实例的内存 = 对象本身 + 私有 __dict__；__slots__ 实例没有 __dict__
    return sys.getsizeof(with_dict) + sys.getsizeof(with_dict.__dict__), sys.getsizeof(with_slots)


# ---------------------------------------------------------------------------
# 4. dataclass / typing / enum
# ---------------------------------------------------------------------------
# 【F-pre-py2.Batch｜类】dataclass：自动 __init__/__repr__/__eq__（对照 data.py 的 Batch）
@dataclass
class Batch:
    x: Sequence[int]
    y: Sequence[int]
    meta: dict = field(default_factory=dict)   # 可变默认值必须用 default_factory


# 【F-pre-py2.Color｜类】Enum：具名常量，成员是单例
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


# 【F-pre-py2.TokenizerLike｜类】Protocol：结构化子类型（duck typing 的类型化）
@runtime_checkable
class TokenizerLike(Protocol):
    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        ...

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        ...


# 【F-pre-py2.SimpleTokenizer｜类】满足 TokenizerLike 协议的最小实现（不继承）
class SimpleTokenizer:
    """结构化子类型：不继承 ``TokenizerLike`` 也满足该协议。"""

    def __init__(self, chars: str) -> None:
        self.itos = {i: c for i, c in enumerate(sorted(set(chars)))}
        self.stoi = {c: i for i, c in self.itos.items()}

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 5. 异常模型：try/except/else/finally、raise ... from、自定义异常
# ---------------------------------------------------------------------------
# 【F-pre-py2.TokenizerError｜类】库专用异常基类，便于调用方精确捕获
class TokenizerError(Exception):
    """本模块分词相关错误的基类。"""


# 【F-pre-py2.parse_ids｜函数】示范 else/finally 与 raise ... from 的语义
def parse_ids(text: str) -> list[int]:
    """把 ``"1,2,3"`` 解析为 ``[1, 2, 3]``；展示异常链与 else/finally。"""

    log: list[str] = []
    try:
        parts = text.split(",")
        ids = [int(p) for p in parts]
    except ValueError as exc:
        raise TokenizerError(f"非法 token 序列: {text!r}") from exc  # 显式异常链
    else:
        log.append("解析成功")          # 仅当 try 未抛异常时执行
        return ids
    finally:
        log.append("清理")              # 无论成功失败都执行


# 【F-pre-py2.encode_safe｜函数】EAFP 风格：直接查表，失败再处理
def encode_safe(stoi: dict[str, int], text: str) -> list[int]:
    """EAFP：先尝试 ``stoi[ch]``，未知字符抛 ``KeyError`` 则跳过。"""

    out: list[int] = []
    for ch in text:
        try:
            out.append(stoi[ch])
        except KeyError:
            continue
    return out


# ---------------------------------------------------------------------------
# 6. 上下文管理器：__enter__/__exit__、@contextmanager、原子写
# ---------------------------------------------------------------------------
# 【F-pre-py2.Timer｜类】类式上下文管理器：测量代码块耗时
class Timer:
    """类式上下文管理器：``with Timer() as t: ...`` 后读 ``t.elapsed``。"""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self.start
        return False                 # 返回 False：不吞异常，继续向上传播


# 【F-pre-py2.pushed｜函数】@contextmanager：用生成器写上下文管理器
@contextmanager
def pushed(stack: list, item) -> Iterator:
    """把 ``item`` 压栈，退出时保证弹出（RAII 风格的清理）。"""

    stack.append(item)
    try:
        yield item
    finally:
        stack.pop()


# 【F-pre-py2.atomic_write｜函数】临时文件 + os.replace 的原子写
def atomic_write(path: str, text: str) -> None:
    """先写 ``path + '.tmp'``，再 ``os.replace`` 原子改名。"""

    import os

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)            # 同文件系统内原子替换


# ---------------------------------------------------------------------------
# 7. 装饰器与高阶函数
# ---------------------------------------------------------------------------
# 【F-pre-py2.retry｜函数】带参装饰器：三层结构，functools.wraps 保元数据
def retry(times: int):
    """``@retry(3)``：失败时重试 ``times`` 次，最后一次仍失败则抛出。"""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for k in range(times):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError:
                    if k == times - 1:
                        raise
            return None

        return wrapper

    return deco


# 【F-pre-py2.fib｜函数】functools.lru_cache 记忆化
@functools.lru_cache(maxsize=None)
def fib(n: int) -> int:
    """记忆化斐波那契；``fib.cache_info()`` 可观察命中率。"""

    return n if n < 2 else fib(n - 1) + fib(n - 2)


# 【F-pre-py2.sort_by_abs｜函数】高阶函数：key= 回调
def sort_by_abs(values: Sequence[int]) -> list[int]:
    """``sorted(..., key=abs)``：key 是接受元素、返回排序键的回调。"""

    return sorted(values, key=abs)


# ---------------------------------------------------------------------------
# 8. 迭代器与生成器、itertools
# ---------------------------------------------------------------------------
# 【F-pre-py2.windows｜函数】生成器：滑动窗口，惰性求值
def windows(it, n: int) -> Iterator[tuple]:
    """惰性滑动窗口；只保留 ``n`` 个元素，内存 O(n)。"""

    buf: list = []
    for item in it:
        buf.append(item)
        if len(buf) == n:
            yield tuple(buf)
            buf.pop(0)


# 【F-pre-py2.take｜函数】itertools.islice：从无限迭代器取前 n 个
def take(it, n: int) -> list:
    """取前 ``n`` 项，配合 ``itertools.count`` 即为有限采样。"""

    return list(itertools.islice(it, n))


# 【F-pre-py2.feature_pipeline｜函数】生成器流水线：过滤 + 变换，无中间 list
def feature_pipeline(values: Sequence[int]) -> Iterator[int]:
    """惰性流水线：剔除负数后平方。"""

    return (v * v for v in values if v >= 0)


# ---------------------------------------------------------------------------
# 演示入口：__name__ == "__main__" 守卫
# ---------------------------------------------------------------------------
# 【F-pre-py2.main｜函数】运行全部演示
def main() -> None:
    """依次演示各节机制；输出与 notebook N-19 对应。"""

    a, b = Vec2(1, 2), Vec2(3, 4)
    print("Vec2:", a, "+", b, "=", a + b, "| a == Vec2(1,2):", a == Vec2(1, 2))
    print("Trait:", Vec2.dim, "| Buffer:", list(Buffer([10, 20, 30])), "len =", len(Buffer([10, 20, 30])))

    c = Counter(5)
    c.value = 7
    print("Counter:", c.value, c.doubled)
    print("memory (with __dict__ vs __slots__):", memory_footprint())

    batch = Batch(x=[1, 2], y=[2, 3])
    print("Batch:", batch)

    tok = SimpleTokenizer("abc")
    print("Protocol isinstance:", isinstance(tok, TokenizerLike), "| decode:", tok.decode([0, 1, 2]))

    print("parse_ids:", parse_ids("1,2,3"))
    try:
        parse_ids("1,x,3")
    except TokenizerError as exc:
        print("链式异常:", type(exc).__name__, "cause =", type(exc.__cause__).__name__)

    print("encode_safe:", encode_safe({"a": 0, "b": 1}, "axb"))

    with Timer() as t:
        sum(range(100000))
    print(f"Timer: {t.elapsed:.3e}s")
    stack: list = []
    with pushed(stack, "token") as item:
        print("pushed:", item, "| in stack:", stack)
    print("after with, stack:", stack)

    print("retry 包裹后 __name__:", retry(3)(lambda: None).__name__)
    print("fib(10):", fib(10), "| cache:", fib.cache_info().currsize)
    print("sort_by_abs:", sort_by_abs([-3, 1, 2, -5]))

    print("windows:", list(windows(range(5), 3)))
    print("take(count):", take(itertools.count(1), 5))
    print("pipeline:", list(feature_pipeline([-2, -1, 0, 1, 2, 3])))


if __name__ == "__main__":
    main()
