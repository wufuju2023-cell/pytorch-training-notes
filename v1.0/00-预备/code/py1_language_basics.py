# 【源代码｜F-pre-py1】v1.0/00-预备/code/py1_language_basics.py — Python 语言 I 教学代码
# 相关文档：《00-预备/PY1-Python语言I.md》【文档｜DOC-PY1】
# 对应 notebook：v1.0/notebooks/00-预备/N18_Python语言I.ipynb（Tag：N-18）
"""Python 语言 I 的可运行教学示例（CPU 即可运行，仅用标准库）。

覆盖 PY1 章的主干主题：
- 名字绑定与对象模型（绑定、身份、可变性、共享陷阱）；
- 内置类型与切片语义（拷贝 vs 视图、负索引、步长）；
- 控制流（真值、for-else、海象、match）；
- 函数（可变默认值陷阱、*args/**kwargs、只关键字、闭包与 nonlocal）；
- 推导式、生成器表达式与迭代协议初识。

执行：``python3 py1_language_basics.py``。
所有示例均为确定性输出，便于对照 notebook。
"""

from __future__ import annotations

import dis
import sys


# 【F-pre-py1.show_binding_semantics｜函数】绑定/身份/可变性的可观测行为
def show_binding_semantics() -> dict[str, object]:
    """演示“名字是绑定”与“可变对象共享”的行为。

    返回关键观测值，便于 notebook/测试断言。
    """
    a = [1, 2, 3]
    b = a                       # b 与 a 指向同一对象
    b.append(4)                 # 原地修改对 a 可见
    shared = a is b             # True：同一对象

    c = a[:]                    # 切片 = 浅拷贝（新 list）
    c.append(5)
    independent = c is not a and a == [1, 2, 3, 4]

    s = "abc"
    t = s
    t = t + "d"                 # 不可变对象：这是重绑定，不是原地修改
    str_unchanged = s == "abc"

    return {
        "a": list(a),
        "b": list(b),
        "c": list(c),
        "shared": shared,
        "independent": independent,
        "str_unchanged": str_unchanged,
        "s": s,
        "t": t,
    }


# 【F-pre-py1.slice_semantics｜函数】切片：左闭右开、负索引、步长、越界截断
def slice_semantics() -> dict[str, object]:
    """演示内置序列的切片规则（内置序列切片是拷贝，不是视图）。"""
    xs = [0, 1, 2, 3, 4, 5]
    out = {
        "xs[1:4]": xs[1:4],
        "xs[::-1]": xs[::-1],
        "xs[-1]": xs[-1],
        "xs[10:20]": xs[10:20],
        "xs[::2]": xs[::2],
        "xs[4:1:-1]": xs[4:1:-1],
    }
    # 拷贝语义：切片结果与原序列互不影响
    y = xs[:]
    y.append(99)
    out["slice_is_copy"] = xs[-1] != 99 and y[-1] == 99
    return out


# 【F-pre-py1.mutable_default_trap｜函数】可变默认值陷阱（错误示范）
def mutable_default_trap(x: int, acc: list[int] = []) -> list[int]:
    """错误示范：``acc`` 在 def 求值时创建一次，被所有调用共享。"""
    acc.append(x)
    return acc


# 【F-pre-py1.fixed_default｜函数】用 None 哨兵修正可变默认值陷阱
def fixed_default(x: int, acc: list[int] | None = None) -> list[int]:
    """正确写法：默认值用不可变哨兵 ``None``，在函数体内构造新列表。"""
    if acc is None:
        acc = []
    acc.append(x)
    return acc


# 【F-pre-py1.tag｜函数】签名演示：*args、只关键字参数、**kwargs
def tag(name: str, *items: str, sep: str = "-", **meta: object) -> str:
    """拼装标签：``*items`` 收集位置实参，``sep`` 只关键字，``**meta`` 收集关键字。"""
    body = sep.join([name, *items])
    if meta:
        body += ":" + ",".join(f"{k}={v}" for k, v in meta.items())
    return body


# 【F-pre-py1.make_counter｜函数】闭包与 nonlocal：返回带状态的计数器
def make_counter(start: int = 0):
    """返回一个闭包；每次调用自增并返回当前值。

    内层 ``tick`` 需要重新绑定外层变量 ``n``，故必须声明 ``nonlocal n``。
    """
    n = start

    def tick() -> int:
        nonlocal n
        n += 1
        return n

    return tick


# 【F-pre-py1.classify｜函数】match 结构化模式匹配 + 守卫
def classify(value: object) -> str:
    """用结构化模式匹配给值分类。"""
    match value:
        case 0:
            return "zero"
        case int() as n if n < 0:
            return "neg"
        case [_, _, _]:
            return "triple"
        case {"kind": k}:
            return f"mapping:{k}"
        case _:
            return "other"


# 【F-pre-py1.comprehension_vs_generator｜函数】推导式 vs 生成器表达式
def comprehension_vs_generator(n: int = 10) -> dict[str, object]:
    """对比列表推导式与生成器表达式的求值与内存行为。"""
    squares_list = [x * x for x in range(n) if x % 2 == 0]

    gen = (x * x for x in range(n) if x % 2 == 0)
    first = next(gen)
    rest_sum = sum(gen)

    it = iter([10, 20, 30])
    return {
        "squares_list": squares_list,
        "first": first,
        "rest_sum": rest_sum,
        "iter_first": next(it),
        "list_size": sys.getsizeof(squares_list),
    }


# 【F-pre-py1.show_bytecode｜函数】用 dis 观察字节码
def show_bytecode() -> list[str]:
    """返回 ``add`` 函数的字节码指令名列表（不打印，便于教学）。"""
    def add(a: int, b: int) -> int:
        return a + b

    return [instr.opname for instr in dis.get_instructions(add)]


# 【F-pre-py1.main｜函数】运行全部示例并打印结果
def main() -> None:
    print("== 绑定与对象模型 ==")
    for k, v in show_binding_semantics().items():
        print(f"  {k} = {v!r}")

    print("== 切片语义 ==")
    for k, v in slice_semantics().items():
        print(f"  {k} = {v!r}")

    print("== 可变默认值陷阱 ==")
    print("  bad 三次:", mutable_default_trap(1), mutable_default_trap(2), mutable_default_trap(3))
    print("  fixed 三次:", fixed_default(1), fixed_default(2), fixed_default(3))

    print("== 参数形态 ==")
    print("  tag:", tag("a", "b", "c", sep="/"))
    print("  tag:", tag("a", x=1, y=2))

    print("== 闭包 ==")
    c = make_counter()
    print("  counter:", c(), c(), c())

    print("== match ==")
    print("  classify:", [classify(v) for v in (0, -3, [1, 2, 3], {"kind": "k"}, "s")])

    print("== 推导式与生成器 ==")
    for k, v in comprehension_vs_generator().items():
        print(f"  {k} = {v!r}")

    print("== 字节码（add 的 opname）==")
    print("  ", show_bytecode())


if __name__ == "__main__":
    main()
