# PY1 · Python 语言 I

> **对应 notebook**：`v1.0/notebooks/00-预备/N18_Python语言I.ipynb`（Tag：`N-18`）
> **对应代码**：`v1.0/00-预备/code/py1_language_basics.py`（Tag：`F-pre-py1`）
> **预计学时**：3 学时
> **前置知识**：具备 C/C++ 编程经验；对 Python 无任何假设。
> **【文档｜DOC-PY1】**（doccode = `PY1`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 1. 解释器与执行模型

**【定义 PY1.1.1｜D-PY1.1.1】（CPython 的执行模型）**

Python 是语言规范，CPython 是其参考实现。执行一个 `.py` 文件分为三步：解析源码为抽象语法树并编译为字节码（bytecode）；把字节码交给 CPython 的求值循环（evaluation loop）逐条解释执行；运行期对象由解释器自行分配与回收。

与 C/C++ 的对照：

- C/C++ 在运行前由编译器产出机器码、由链接器合成可执行文件，之后 CPU 直接执行机器码；
- Python 的“编译”只到字节码，字节码是面向 CPython 虚拟机的指令，由求值循环（一个 C 写的大 `switch`/分派循环）解释；
- 因此 Python 没有独立的“链接期”：名字解析在运行期完成，函数/模块引用是运行期对象，而非编译期地址。代价是运行速度较低，收益是无需编译-链接即可直接运行与动态改写。

**【定义 PY1.1.2｜D-PY1.1.2】（三种入口：REPL、脚本、模块）**

同一个 `.py` 文件可经三种入口进入解释器：

- REPL（read-eval-print loop）：交互式逐行求值，`python` 无参数启动；适合探索，退出即丢弃状态；
- 脚本入口：`python foo.py`，`__name__` 为 `"__main__"`，文件被当作主程序执行；
- 模块入口：`python -m pkg.foo` 或 `import foo`，`__name__` 为模块的限定名，文件被当作可导入模块执行。

控制某个文件“作为脚本运行”与“被导入”行为差异的惯用法：

```python
if __name__ == "__main__":
    main()
```

C/C++ 对照：C 程序的入口固定为 `main`，链接器决定谁调用它；Python 没有强制入口函数，是否执行 `main()` 由 `__name__` 判断，这是“可执行脚本”与“可复用模块”统一于一个文件的机制。

**【注 PY1.1.3｜R-PY1.1.3】（`__pycache__` 与 `.pyc`）**

导入模块时，CPython 会把编译好的字节码缓存到源文件同目录的 `__pycache__/`，文件名为 `<module>.<tag>.pyc`，`tag` 编码解释器版本与优化级别。若源文件的修改时间与大小与缓存记录一致则直接加载 `.pyc`，跳过重新编译。要点：

- 这只缓存字节码，不缓存机器码，也不消除运行期解释开销；
- 以脚本入口（`__main__`）运行的文件不写 `__pycache__`；
- 清缓存只影响下次启动的编译成本，不影响语义；`.pyc` 可安全删除。

对照 C/C++：相当于把目标文件（`.o`）的缓存行为内置进了解释器，但发生在导入时而非构建时。

**【代码 PY1.1.4｜Cd-PY1.1.4】（用 `dis` 观察字节码）**

```python
import dis

def add(a, b):
    return a + b

dis.dis(add)
# 输出为若干行字节码指令，形如：
#   LOAD_FAST  a
#   LOAD_FAST  b
#   BINARY_OP  +
#   RETURN_VALUE
```

`LOAD_FAST` 按下标取局部变量槽，`BINARY_OP` 执行分派到 `int.__add__` 的通用加法。这解释了为何 Python 的整数加法比 C 的 `a+b` 慢：一次机器指令被展开为取槽、类型分派、结果装箱若干步。

**【例 PY1.1.5｜E-PY1.1.5】（三种入口的可观测差异）**

- `python -c "print(__name__)"` 输出 `__main__`；
- 文件 `m.py` 内容为 `print(__name__)`，`python m.py` 输出 `__main__`，而 `python -c "import m"` 输出 `m`；
- `python -m m` 等价于导入并作为 `__main__` 运行，同时把 `m` 所在目录加入 `sys.path`。

---

## 2. 名字绑定与对象模型

**【定义 PY1.2.1｜D-PY1.2.1】（名字是绑定，不是盒子）**

Python 的变量在语义上是“名字到对象的绑定”（name binding），不是一段可写入的内存槽。执行 `x = obj` 的含义是把名字 `x` 关联到对象 `obj`；名字本身不存储值，也没有类型。类型属于对象，不属于名字。

C/C++ 对照：C 的 `int x = 3;` 声明了一个有固定地址和类型的内存盒子，并把 3 拷进去；Python 的 `x = 3` 则是让名字指向一个整数对象。同一名字先后可指向不同类型的对象，这在 C 中需要 `void*` 或联合体才能表达。

**【命题 PY1.2.2｜P-PY1.2.2】（赋值即重绑定，参数传递为引用传递）**

`a = b` 使名字 `a` 与名字 `b` 指向同一对象，不复制对象；此后对 `a` 重新赋值不影响 `b` 的绑定。函数调用时，实参对象被绑定到形参名，因此：

- 传入不可变对象时，函数内的“修改”（如 `x = x + 1`）只是重绑定局部名字，调用方不可见；
- 传入可变对象并原地修改（如 `lst.append(...)`）则调用方可见。

C/C++ 对照：这类似“按指针传递但指针本身按值传递”——能被调函数改的是所指对象的内容，而不是调用方指针的指向。

**【定义 PY1.2.3｜D-PY1.2.3】（对象身份与相等）**

每个对象有唯一身份（identity），在生命周期内不变；`id(obj)` 返回其身份（CPython 中即内存地址），`a is b` 判定 `a`、`b` 是否为同一对象。`a == b` 判定值相等，默认由 `type(a).__eq__` 定义，可被类型重载。要点：

- `is` 比较身份，`==` 比较值；对小整数与短字符串，CPython 会缓存/驻留，使 `is` 偶然为真，故**判等一律用 `==`**，`is` 仅用于 `None`、`True`、`False` 等单例；
- `is not` 与 `not (a is b)` 语义相同，但前者更易读。

**【注 PY1.2.4｜R-PY1.2.4】（引用计数与 GC 概览）**

CPython 以引用计数为主：每个对象记录指向它的引用数，计数归零即刻回收。引用计数无法处理循环引用，故辅以分代垃圾回收器周期性检测并回收环。实践含义：

- 对象生命周期由引用决定，`del name` 只是解除一个绑定，未必销毁对象；
- 存在循环引用时销毁时刻不确定；需要确定性释放时使用上下文管理器（`with`，见 PY2）；
- `sys.getrefcount(obj)` 可观察引用数（其自身会临时 +1）。

**【定义 PY1.2.5｜D-PY1.2.5】（可变与不可变对象）**

不可变对象创建后值不变：`int`、`float`、`complex`、`bool`、`str`、`bytes`、`tuple`、`frozenset`、`None`。可变对象可取原地修改：`list`、`dict`、`set`、大多数用户自定义实例。要点：

- 不可变对象可安全共享与作为字典键；可变对象不可哈希（`list`、`dict`、`set` 不能作键）；
- `tuple` 不可变指其元素绑定不可变，若元素本身可变（如 `tuple` 内嵌 `list`）则该元素内容仍可变；
- 默认参数、缓存、共享引用是可变对象最常见的陷阱来源（见【注 PY1.5.2】）。

**【代码 PY1.2.6｜Cd-PY1.2.6】（绑定与共享的可观测行为）**

```python
a = [1, 2, 3]
b = a                 # b 与 a 同一对象
b.append(4)
assert a == [1, 2, 3, 4]        # 通过 b 的原地修改对 a 可见

c = a[:]              # 切片产生浅拷贝（新 list）
c.append(5)
assert a == [1, 2, 3, 4]        # a 不受 c 影响
assert c is not a

s = "abc"
t = s
t = t + "d"           # 重绑定 t，不改变 s
assert s == "abc"
```

---

## 3. 内置类型

**【定义 PY1.3.1｜D-PY1.3.1】（标量类型）**

数值与布尔标量：`int`（任意精度整数）、`float`（IEEE-754 双精度二进制浮点）、`complex`（复数，形如 `3+4j`）、`bool`（`True`/`False`，是 `int` 的子类，即 `True == 1`、`False == 0`）。字面量可用下划线分组（`1_000_000`）、进制前缀（`0x`、`0o`、`0b`）。算术运算符 `/` 恒返回 `float`（真除法），`//` 为向下取整除法，`%` 结果符号随除数，`**` 为幂。

**【注 PY1.3.2｜R-PY1.3.2】（`int` 与 `float` 对照 C/C++）**

- `int` 无固定位宽、无溢出回绕，可表示任意大整数，代价是运算不是单条机器指令；C 的 `int` 是定长且有符号溢出是未定义行为；
- `float` 即 C 的 `double`，遵循 IEEE-754：存在 `inf`、`nan`；比较用 `math.isnan`、`math.isinf`；浮点相等不可靠，应比较容差；
- `bool` 参与算术会被当作 0/1，但不要依赖此写业务逻辑。

**【定义 PY1.3.3｜D-PY1.3.3】（`str` 与 `bytes`）**

`str` 是 Unicode 码点的不可变序列，语义上是“文本”；`bytes` 是 0..255 的不可变字节序列，语义上是“二进制”。二者不可隐式互转，必须显式编码/解码：

```python
text = "汉字"                  # str
raw = text.encode("utf-8")     # bytes
back = raw.decode("utf-8")     # str
```

要点：`len(str)` 是码点数而非字节数；文件与网络按字节读写，文本需指定编码；不确定编码时优先 UTF-8。对照 C/C++：`str` 与 `bytes` 的边界就是“文本 vs 字节串”的边界，在 C 中由程序员自行约定，Python 用类型强制区分；`bytes` 类似不可变的 `unsigned char[]`，`str` 是带 Unicode 语义的不可变序列。

**【定义 PY1.3.4｜D-PY1.3.4】（`None`）**

`None` 是单例，表示“无值/缺省”，判等一律用 `is None`。函数无显式 `return` 时返回 `None`。对照 C/C++：类似 `NULL`/`nullptr`，但 `None` 是真正的对象、有类型 `NoneType`，且不是任何数值类型，不会与 0 混淆。

**【定义 PY1.3.5｜D-PY1.3.5】（序列容器 `list` 与 `tuple`）**

`list` 是可变有序序列，`tuple` 是不可变有序序列；二者都支持索引、切片、`len`、成员测试、拼接 `+`、重复 `*`、`in`。字面量：`[1, 2]`、`(1, 2)`；单元素元组须写 `(1,)`。列表有原地方法（`append`、`extend`、`insert`、`sort`、`reverse`、`pop`），元组没有。要点：

- 序列保持插入序，索引从 0 起；
- `list` 的 `append`/`pop` 均摊 O(1)，头部插入/删除为 O(n)；
- 在 `for` 中修改列表长度会导致跳过或重复，遍历副本 `for x in list(lst)` 或构造新列表。

**【定义 PY1.3.6｜D-PY1.3.6】（映射与集合 `dict`/`set`/`frozenset`）**

`dict` 是键到值的哈希映射，键必须可哈希且唯一；`set` 是可变的无重复集合；`frozenset` 是不可变集合、可作字典键或集合元素。要点：

- 自 Python 3.7 起 `dict` 保持插入序（语言规范保证）；`set` 不保证顺序；
- 哈希约束：可哈希对象在一次生命周期内哈希值不变，且相等对象哈希相等；因此 `list`/`dict`/`set` 不可作键；
- `dict` 的 `get`、`setdefault`、`defaultdict` 用于缺省值；`in` 对 `dict` 判键、对 `set` 判成员，均摊 O(1)；
- 用户自定义对象默认按身份哈希，除非实现 `__hash__` 与 `__eq__`（见 PY2）。

**【注 PY1.3.7｜R-PY1.3.7】（切片语义：步长、负索引、拷贝 vs 视图）**

记序列 `s`，切片写作 `s[start:stop:step]`。规则：

- 界域为左闭右开 `[start, stop)`；`start`/`stop` 缺省为两端；`step` 缺省为 1；
- 负索引从末尾计（`-1` 为最后一个）；`step` 为负时从右向左，且缺省 `start`/`stop` 互换；
- 越界不报错，按边界截断（与 C 的指针越界不同）；
- `list`/`tuple`/`str` 的切片是**拷贝**（新对象），`s[:]` 即浅拷贝；`dict` 无切片；
- `numpy` 数组与 PyTorch 张量的切片是**视图**（共享底层存储），与 Python 内置序列相反，必须牢记（见 PY4/PY5）。

**【代码 PY1.3.8｜Cd-PY1.3.8】（类型与切片速览）**

```python
xs = [0, 1, 2, 3, 4, 5]
assert xs[1:4] == [1, 2, 3]
assert xs[::-1] == [5, 4, 3, 2, 1]     # 逆序拷贝
assert xs[-1] == 5
assert xs[10:20] == []                 # 越界不报错

d = {"a": 1, "b": 2}
d.setdefault("a", 99)
assert d["a"] == 1                     # 已存在则不覆盖

assert len("汉字".encode("utf-8")) == 6
assert "汉字".encode("utf-8").decode("utf-8") == "汉字"
```

---

## 4. 控制流

**【定义 PY1.4.1｜D-PY1.4.1】（真值测试规则）**

任意对象在布尔语境（`if`/`while` 条件、`and`/`or`/`not`、`bool()`）中被转为真值。假值集合为：`False`、`None`、数值零（`0`、`0.0`、`0j`）、空序列/空映射/空集合（`""`、`b""`、`[]`、`()`、`{}`、`set()`）。其余对象为真，除非其类型定义了 `__bool__`/`__len__`（见 PY2）。对照 C/C++：C 只有数值零为假，指针非空为真；Python 把“空容器”也定为假，故 `if lst:` 是惯用的非空判断。

**【定义 PY1.4.2｜D-PY1.4.2】（`if/elif/else`）**

条件分支以缩进（推荐 4 空格）划分代码块，无花括号、无 `switch` 回退。`elif` 是 `else if` 的缩写，自上而下取第一个真分支。三目表达式写作 `a if cond else b`。

**【定义 PY1.4.3｜D-PY1.4.3】（`while` 与 `for`）**

`while cond:` 条件为真时循环；`for target in iterable:` 对**可迭代对象**逐项绑定 `target`，没有 C 风格的 `for(init; cond; step)`。需要计数时配合 `range(stop)`、`range(start, stop)`、`range(start, stop, step)`，或 `enumerate(seq)` 同时取下标与元素，或 `zip(a, b)` 并行遍历。`break` 立即退出最近一层循环，`continue` 跳过本次剩余语句。

**【注 PY1.4.4｜R-PY1.4.4】（`for-else` 与 `while-else`）**

`for`/`while` 可跟 `else` 子句：仅当循环因条件为假而正常结束（即未被 `break` 打断）时执行。它表达“未找到则…”的语义：

```python
for x in xs:
    if x == target:
        break
else:
    print("not found")
```

注意与 C 中“用找到标志位控制后续处理”的写法对照：`for-else` 把标志位内置为语言结构。

**【注 PY1.4.5｜R-PY1.4.5】（海象运算符 `:=`）**

赋值表达式 `name := expr` 在表达式内部写入名字并返回该值，用于避免“先计算再判断”的重复或额外变量：

```python
while (line := f.readline()):
    process(line.rstrip("\n"))
```

要点：`:=` 写入的是当前作用域的名字（在推导式中有特殊约束）；不要为省一行而牺牲可读性。

**【定义 PY1.4.6｜D-PY1.4.6】（结构化模式匹配 `match`）**

`match` 按模式匹配结构并绑定变量，自上而下取第一个匹配分支，`case _:` 为通配。支持字面量、序列模式、映射模式、类模式、`|` 或模式、守卫 `case ... if cond:`。要点：

- `match` 是模式解构与分派，不是 C 的整数 `switch` 跳转表；
- 形如 `case name:` 的模式会**绑定**而永不失败，须用 `case _:` 作默认；
- 常量匹配要求点号限定（如 `case Color.RED:`），裸名一律视为绑定。

**【代码 PY1.4.7｜Cd-PY1.4.7】（控制流速览）**

```python
xs = [1, 2, 3]
assert bool(xs) and not bool([])
squares = [x * x for x in range(5) if x % 2 == 0]
for i, x in enumerate("ab"):
    assert (i, x) in {(0, "a"), (1, "b")}

def classify(x):
    match x:
        case 0:
            return "zero"
        case int() as n if n < 0:
            return "neg"
        case [_, _, _]:
            return "triple"
        case _:
            return "other"

assert classify(0) == "zero"
assert classify([1, 2, 3]) == "triple"
```

---

## 5. 函数

**【定义 PY1.5.1｜D-PY1.5.1】（`def`、位置参数与关键字参数）**

`def f(a, b): ...` 定义函数；调用可写 `f(1, 2)`（按位置）或 `f(b=2, a=1)`（按关键字）。形式参数按顺序分为位置参数、默认值参数、`*args`、只关键字参数、`**kwargs`。调用时先匹配位置、再按关键字，重复绑定报 `TypeError`。

**【注 PY1.5.2｜R-PY1.5.2】（默认参数与可变默认值陷阱）**

默认值在 `def` 执行时求值一次，并在多次调用间共享。因此可变默认值是常见错误：

```python
def bad(x, acc=[]):      # acc 在所有调用间共享
    acc.append(x)
    return acc

def good(x, acc=None):
    if acc is None:
        acc = []
    acc.append(x)
    return acc
```

规则：默认值只用不可变对象；表示“缺省”时用 `None` 哨兵再在函数体内构造。

**【定义 PY1.5.3｜D-PY1.5.3】（可变参数与只关键字参数）**

- `*args`：收集多出的位置实参为元组；调用时 `f(*seq)` 可展开序列为位置实参；
- `**kwargs`：收集多出的关键字实参为字典；调用时 `f(**mapping)` 可展开映射为关键字实参；
- 只关键字参数：写在 `*args` 之后（或单独的 `*` 之后），只能以关键字传入，如 `def f(a, *, b): ...` 中 `b` 必须写 `b=...`；`/` 之前的参数只能按位置传入，如 `def f(a, /, b): ...`。

对照 C/C++：C 的可变参数（`...`）无类型信息、需 `va_list` 手动解析；Python 的 `*args/**kwargs` 是有类型的对象，常用于转发与包装。

**【定义 PY1.5.4｜D-PY1.5.4】（LEGB 作用域规则）**

名字解析按 Local → Enclosing → Global → Builtins 顺序查找：

- Local：当前函数的局部绑定（含形参）；
- Enclosing：外层函数的局部作用域（用于闭包）；
- Global：当前模块的全局作用域；
- Builtins：`len`、`print` 等内置名字。

任何在函数内被**赋值**的名字都是该函数的局部名（`global`/`nonlocal` 可改变这一判定）。对照 C/C++：Python 无块级作用域，`if`/`for` 内绑定的名字在函数范围内可见。

**【定义 PY1.5.5｜D-PY1.5.5】（闭包与 `nonlocal`）**

内层函数引用外层函数的局部变量即构成闭包，外层变量被捕获并随内层函数存活。若内层需要**重新绑定**（而非仅读取）外层变量，须声明 `nonlocal`；若需修改模块级名字，须声明 `global`。要点：

- 仅仅读取外层变量不需要 `nonlocal`；
- 循环中创建的多个闭包共享同一变量时，常需默认参数或 `functools.partial` 固定当前值；
- 闭包捕获的是名字绑定而非值快照。

**【定义 PY1.5.6｜D-PY1.5.6】（函数是一等对象；`lambda`）**

函数是对象：可赋给名字、作为实参、作为返回值、存入容器；`f.__name__`、`f.__doc__`、`f.__annotations__` 为其属性。`lambda 参数: 表达式` 创建匿名函数，体内只能是一个表达式。对照 C/C++：等价于函数指针/`std::function`，但可直接携带被捕获的闭包环境。

本仓代码大量使用函数对象：`nn.Module.forward` 是被框架按名字调用的方法，模型实例本身可像函数一样被调用（见【代码 1.1.4】、【定义 1.1.3】）。

**【代码 PY1.5.7｜Cd-PY1.5.7】（参数与闭包速览）**

```python
def tag(name, *items, sep="-", **meta):
    return sep.join([name, *items]) + (":" + ",".join(meta) if meta else "")

assert tag("a", "b", "c", sep="/") == "a/b/c"
assert tag("a", x=1) == "a:x"

def make_counter():
    n = 0
    def tick():
        nonlocal n
        n += 1
        return n
    return tick

c = make_counter()
assert (c(), c(), c()) == (1, 2, 3)
```

---

## 6. 推导式、生成器表达式与迭代协议初识

**【定义 PY1.6.1｜D-PY1.6.1】（推导式）**

推导式以表达式形式构造容器：列表 `[expr for x in it if cond]`、集合 `{expr for x in it}`、字典 `{k: v for k, v in it}`。可嵌套 `for` 且可带多个 `if`。作用域要点：推导式有自己的作用域，循环变量不外泄（与 Python 2 不同）。对照 C/C++：是“映射 + 过滤”模式的紧凑语法，通常比等价显式循环更快，但嵌套过深或副作用过多时应改回显式循环以保可读性。

**【定义 PY1.6.2｜D-PY1.6.2】（生成器表达式与惰性求值）**

`(expr for x in it)` 是生成器表达式：返回一个惰性迭代器，按需产生元素，不预先构造整个容器，内存占用与长度无关。要点：生成器一次性耗尽，不能回退；适合流式处理大序列；若立即需要全部结果且需复用，用列表推导式。

**【定义 PY1.6.3｜D-PY1.6.3】（迭代协议初识：可迭代对象与迭代器）**

可迭代对象（iterable）实现 `__iter__`，返回迭代器；迭代器（iterator）实现 `__next__`，耗尽时抛 `StopIteration`（两者通常也实现 `__iter__` 返回自身）。内置函数 `iter(it)` 取迭代器、`next(it[, default])` 取下一项；`for` 循环、解包、`in`、`sum`、`max` 等统一建立在 `iter()/next()` 之上。对照 C/C++：迭代器类似 `begin()/end()` 的泛化，但以异常 `StopIteration` 表示越界，且可迭代对象与迭代器是两类不同协议。

**【注 PY1.6.4｜R-PY1.6.4】（与 PY2 的衔接）**

生成器函数（含 `yield`）、自定义 `__iter__`/`__next__`、`itertools` 常用件、上下文管理器与装饰器在《PY2-Python语言II》展开；本仓 `data.py` 的数据管线使用生成器惰性产出样本，其用法将在 PY2 精读。

**【代码 PY1.6.5｜Cd-PY1.6.5】（推导式与迭代协议速览）**

```python
xs = [1, 2, 3, 4]
assert [x * x for x in xs if x % 2 == 0] == [4, 16]
assert {x: x * x for x in xs} == {1: 1, 2: 4, 3: 9, 4: 16}

gen = (x * x for x in xs)
assert next(gen) == 1
assert list(gen) == [4, 9, 16]      # 剩余部分

it = iter(xs)
assert next(it) == 1
```

---

## 7. 来源与许可

本章内容在编写时参考下列官方材料；下列材料均为「参考与术语对照」，未整段搬运受限制文本。

| 材料 | URL | 许可 | 搬运范围 |
| --- | --- | --- | --- |
| The Python Tutorial（Python 官方教程） | https://docs.python.org/3/tutorial/ | PSF License Version 2（Python Software Foundation） | 参考概念组织与术语；示例代码为本章自行重写 |
| The Python Language Reference（语言参考） | https://docs.python.org/3/reference/ | PSF License Version 2 | 参考数据模型、执行模型、作用域与真值规则的权威表述 |
| Python Standard Library · Built-in Types | https://docs.python.org/3/library/stdtypes.html | PSF License Version 2 | 参考内置类型与切片语义的接口定义 |
| `dis` — Disassembler for Python bytecode | https://docs.python.org/3/library/dis.html | PSF License Version 2 | 参考字节码指令名称与反汇编用法 |

说明：

- Python 官方文档随 CPython 源码分发，适用 PSF License Version 2；本章仅作概念参照并按自己的语言重述，未复制大段原文；
- 全部示例代码与 notebook 为本项目原创，可随本仓许可分发；
- 若后续需整段引用官方文本，须另行核对许可并登记（非 NC/SA 类方可搬运；NC/SA 类仅链接）。

---

## 8. 自测

**【练习 PY1.8.1｜Ex-PY1.8.1】（执行入口）**

说明 `if __name__ == "__main__":` 在“脚本运行”“被导入”“`python -m`”三种情形下的取值与作用，并解释为何它使同一文件既可执行又可导入。

**【练习 PY1.8.2｜Ex-PY1.8.2】（绑定与身份）**

预测并验证下列表达式的值：`a = [1, 2]; b = a; b += [3]` 后 `a is b`、`a == b` 各为何？若把 `b += [3]` 换成 `b = b + [3]`，结果如何变化？解释 `+=` 对可变对象是原地修改、对不可变对象是重绑定。

**【练习 PY1.8.3｜Ex-PY1.8.3】（切片与拷贝）**

给定 `m = [[0] * 2] * 2`，写出 `m[0][0] = 1` 之后的 `m`，解释与 `[[0] * 2 for _ in range(2)]` 的差异，并说明“浅拷贝 vs 深拷贝”。

**【练习 PY1.8.4｜Ex-PY1.8.4】（哈希与集合）**

解释为何 `{1, 2}` 合法而 `{[1], [2]}` 非法；把 `[1, 2]` 的元素投入 `set` 时正确的做法是什么？说明相等与哈希必须保持一致（相等对象哈希相等）。

**【练习 PY1.8.5｜Ex-PY1.8.5】（控制流）**

用 `for-else` 改写“在列表中查找目标，未找到则报告”的代码，并与“标志位 + `if`”的版本对照，说明各自可读性。

**【练习 PY1.8.6｜Ex-PY1.8.6】（默认参数陷阱）**

写出 `def f(x, acc=[])` 连续三次调用的返回值，解释原因，并给出用 `None` 哨兵修正后的版本。

**【练习 PY1.8.7｜Ex-PY1.8.7】（闭包与 `nonlocal`）**

实现 `make_accumulator(start)`，返回一个函数，每次调用在内部累计并返回当前累计值。解释为何必须使用 `nonlocal`，以及去掉它会得到什么错误。

**【练习 PY1.8.8｜Ex-PY1.8.8】（推导式与生成器）**

分别用列表推导式与生成器表达式计算 `1..10^6` 中偶数的平方和，用 `sys.getsizeof`（或分步求和观察内存峰值）说明二者内存行为差异，并解释何时必须用生成器。

---

*本章对应代码：`v1.0/00-预备/code/py1_language_basics.py`；notebook：`v1.0/notebooks/00-预备/N18_Python语言I.ipynb`。*
