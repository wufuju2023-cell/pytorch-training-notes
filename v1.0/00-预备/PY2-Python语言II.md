# PY2 · Python 语言 II

> **对应预备篇大纲**：《00-预备/00-大纲.md》§1（章节表 PY2 行）与 §2「PY2 · Python 语言 II」条目清单（Tag：`DOC-PREM`）。
> **配套代码**：`00-预备/code/py2_oop_protocols.py`（Tag：【源代码｜F-pre-py2】）。
> **配套 notebook**：`notebooks/00-预备/N19_Python面向对象与协议.ipynb`（Tag：`N-19`）。
> **对接本仓代码**：`01-基础/code/from_scratch/data.py`（`F-b01-data`）与 `model.py`（`F-b01-model`）——本章的目标是让读者能直接读懂这两个文件。
> **预计学时**：4 学时。
> **前置知识**：PY1（名字绑定与对象模型、内置类型、函数与作用域）；语言对照以 C/C++ 为参照。
> **【文档｜DOC-PY2】**（doccode = `PY2`）｜编号与 Tag 规范见《00-风格与编号规范》。

---

## 1 类与实例

面向对象是本章的主线。C/C++ 读者最需要先修正的直觉是：Python 的“成员”不是编译期固定的布局，而是运行时可增删的字典项；方法也只是存放在类字典里的普通函数对象。

**【定义 PY2.1.1｜D-PY2.1.1】（类、实例与 `self`）**

`class` 语句本身是可执行语句：解释器执行类体，得到一个类对象（`type` 的实例），再把类名绑定到该对象。由类创建实例的过程为“先分配、后初始化”：

- `C(...)` 先调用 `C.__new__(C, ...)` 分配一个实例，再调用 `C.__init__(instance, ...)` 完成初始化；
- 方法定义在类体内，其第一个参数由调用方**显式**传入实例，约定名为 `self`；调用 `x.f(a)` 等价于 `C.f(x, a)`；
- `self` 没有语法特权，它只是普通参数；之所以能写成 `x.f(a)`，是因为属性查找把函数变成了“绑定方法”（见【定义 PY2.3.2】）。

对照 C++：C++ 的成员函数隐含 `this`，Python 不隐藏；C++ 非 `virtual` 函数按静态类型分派，而 Python 的**所有**方法都按实例运行时类型查找，效果等价于“全部 `virtual`”。此外 Python 的类属性默认公开，没有 `private`/`protected` 的编译期强制（私有约定见 §3 与命名约定）。

**【定义 PY2.1.2｜D-PY2.1.2】（`__init__` 与构造过程）**

`__init__` 不是“构造函数”本身，它只负责给已存在的实例写入初始状态；真正创建对象的是 `__new__`（通常继承 `object.__new__`，无需自定义）。要点：

- `__init__` 的返回值必须是 `None`，否则抛 `TypeError`；
- `__new__` 是静态方法，第一个参数是类 `cls`；需要自定义不可变对象或单例时才重写；
- 常见写法 `self.x = x` 就是把名字 `x` 绑定到实例字典。

对照 C++：`__new__` 近似“分配 + 构造”，`__init__` 近似构造函数的函数体。区别在于 Python 把两步拆开且可分别覆写；同时没有析构函数的确定性调用，资源释放依靠引用计数回收或显式清理（`with`，见 §6）。

**【命题 PY2.1.3｜P-PY2.1.3】（属性查找顺序：实例字典优先于类字典）**

读取 `obj.attr` 的默认顺序是：

1. 在实例的 `__dict__` 中查找；
2. 若未命中，沿实例类型的 MRO（【定义 PY2.1.4】）在各级类字典中查找，并把找到的函数“绑定”为方法；
3. 特殊方法除外——隐式调用（如 `len(obj)`）走**类型**上的查找，不走实例字典。

赋值 `obj.attr = v` 只写实例字典，不修改类；删除同理。由此得到一条重要陷阱：**可变的类属性被所有实例共享**。例如把 `tricks = []` 定义在类体里，各实例 `self.tricks.append(...)` 操作的是同一个列表。

对照 C++：类属性近似 `static` 成员，类方法近似普通成员函数（但后者按实例分派）。C++ 的成员变量不存在“共享一个 vector”的歧义，因为每个对象各有存储；Python 里若非要用类级可变数据，必须显式改为实例属性，或用 dataclass 的 `field(default_factory=...)`（【定义 PY2.4.1】）。

**【定义 PY2.1.4｜D-PY2.1.4】（继承与 MRO）**

继承写法为 `class D(B1, B2): ...`。属性解析沿**方法解析顺序**（MRO）进行，MRO 由 C3 线性化算法给出，可读作 `D.__mro__` 或调用 `D.mro()`。C3 的性质：

- 保持每个类的直接基类从左到右的相对顺序；
- 每个类在 MRO 中只出现一次（菱形继承下共同祖先只出现一次）；
- 单调：给类加基类不会改变其祖先原有的优先关系。

若查遍 MRO 仍未命中，则回退到实例的 `__getattr__`（若定义了）并最终抛 `AttributeError`。

对照 C++：多继承在 C++ 中需要虚基类处理菱形，且名字查找是按作用域逐分支进行；Python 用单一的 MRO 线性表取代逐分支查找，因此“访问哪个基类成员”总是由一条确定的序列决定，而不是编译期的支配分析。

**【注 PY2.1.5｜R-PY2.1.5】（`super()` 与协作式多继承）**

零参 `super()` 依赖编译器注入的 `__class__` 单元与当前实例，返回 MRO 中“当前类的下一个类”上的代理，因此它绑定的是**调用点的位置**而非某个固定基类。协作式写法要求每层 `__init__` 都调用 `super().__init__(**kwargs)` 并把剩余关键字参数继续向后传，从而让链上每个类都能初始化自己的部分。

对照 C++：`Base::method()` 是直接、静态地指名某个基类；`super().method()` 是“按 MRO 的下一站”，在单继承时与 `Base.method(self)` 结果相同，在多继承中则不等价。

**【例 PY2.1.6｜E-PY2.1.6】（剖析 `CharTokenizer`：类属性、类方法与只读属性）**

`data.py` 中的 `CharTokenizer`（代码 `F-b01-data.CharTokenizer`）是本章机制的集中示例：

- `__init__` 把 `itos`、`stoi`、`pad_id`、`eos_id` 写进实例字典（【定义 PY2.1.2】）；
- `from_text` 是 `@classmethod`，用 `cls(...)` 而非硬编码类名，便于子类复用；
- `vocab_size` 是 `@property`，返回值 `len(self.itos)`，无 setter（见【定义 PY2.3.1】）；
- `save`/`load` 用临时文件加 `os.replace` 做原子替换（见【例 PY2.6.3】）。

它同时示范了【命题 PY2.1.3】的“实例字典优先”与【定义 PY2.3.2】的描述符机制。

**【代码 PY2.1.7｜Cd-PY2.1.7】（最小 `Vec2`：类属性、实例属性、运算符与 `__repr__`）**

```python
class Vec2:
    dim = 2                      # 类属性：所有实例共享
    __slots__ = ("x", "y")       # 去掉 __dict__，见 §3

    def __init__(self, x, y):
        self.x, self.y = x, y    # 实例属性：每个实例各自一份

    def __repr__(self):
        return f"Vec2({self.x!r}, {self.y!r})"

    def __add__(self, other):    # 运算符重载，对照 C++ operator+
        return Vec2(self.x + other.x, self.y + other.y)
```

完整可运行版本见配套代码 `F-pre-py2`（`Vec2` 类）。

## 2 数据模型与魔术方法

Python 的“数据模型”规定：语言的每条语法（`len`、索引、算术、迭代、`with`）在遇到用户类型时，会去调用约定名字的特殊方法。理解这一层，才能把“Python 的操作符”与“C++ 的运算符重载”对应起来。

**【定义 PY2.2.1｜D-PY2.2.1】（数据模型与特殊方法总览）**

特殊方法（dunder）由解释器在特定语法处自动调用：

| 语法 | 调用的特殊方法 | 语义 |
| --- | --- | --- |
| `len(x)` | `x.__len__()` | 返回非负整数，否者抛 `TypeError` |
| `x[k]` | `x.__getitem__(k)` | 下标访问；`k` 也可能是 `slice` 对象 |
| `k in x` | `x.__contains__(k)` | 缺省时回退为遍历 `__iter__` |
| `x.f(a)` | `type(x).f(x, a)` | 绑定方法调用 |
| `a + b` | `a.__add__(b)`，必要时再试 `b.__radd__(a)` | 算术；两侧都失败抛 `TypeError` |
| `with x as v` | `x.__enter__()` / `x.__exit__(...)` | 上下文管理（§6） |
| `iter(x)` | `x.__iter__()` / `x.__next__()` | 迭代协议（§8） |

显式写 `x.__len__()` 虽可运行，但应使用内建 `len(x)`，因为内建会做类型检查并启用上述回退规则。对照 C++：特殊方法相当于把运算符与标准接口（`size`、`begin/end`、`operator==`）统一到一组“可被语言调用的名字”上；但绑定是按对象类型在运行时查找的。

**【命题 PY2.2.2｜P-PY2.2.2】（`__repr__` 与 `__str__` 的分工）**

- `repr(x)`（`__repr__`）面向开发者，目标是**无歧义**；理想情况下 `eval(repr(x))` 能重建对象；
- `str(x)`（`__str__`）面向用户，默认回退到 `__repr__`；
- 在 f-string 中，`f"{x}"` 用 `__str__`，`f"{x!r}"` 用 `__repr__`；容器（list/tuple/dict）打印元素时用的是元素的 `__repr__`。

因此调试时看到的列表内容由元素的 `__repr__` 决定；为自定义类型补一个信息充分的 `__repr__` 是最高性价比的调试投资。

**【命题 PY2.2.3｜P-PY2.2.3】（容器协议：`__len__`、`__getitem__`、`__contains__`）**

- 实现 `__len__` 后可用 `len(x)` 与真值测试（空容器为假）；
- 实现 `__getitem__` 后即可用 `x[k]`、`for ... in x`（缺 `__iter__` 时解释器用整数索引迭代直到 `IndexError`）以及切片（`k` 为 `slice`，需自行处理 `start/stop/step`）；
- `k in x` 优先用 `__contains__`，否则回退为遍历；
- 惯例：越界抛 `IndexError`，键缺失抛 `KeyError`。

对照 C++：`__len__` 近似 `size()`，`__getitem__` 近似 `operator[]`，但 `in`、负索引、切片是语言层语法糖，由这些钩子统一支撑。

**【命题 PY2.2.4｜P-PY2.2.4】（`__eq__` 与 `__hash__` 的契约）**

默认识别与哈希都是基于对象身份：`__eq__` 等价于 `is`，`__hash__` 基于 `id`。一旦自定义 `__eq__`，Python 会把 `__hash__` 置为 `None`，使实例不可哈希，除非同时显式定义 `__hash__`。必须遵守的契约是：

- `a == b` 为真则必须有 `hash(a) == hash(b)`；
- 相等对象在 `set`/`dict` 中应被视为同一键；因此哈希必须基于与相等判定相同的字段；
- 可变对象（值会变）不应作为字典键或集合元素。

对照 C++：近似 `operator==` 与 `std::hash` 特化的配套；差别在于 Python 用“定义 `__eq__` 即失效 `__hash__`”这一硬机制强制你正视契约。

**【注 PY2.2.5｜R-PY2.2.5】（运算符重载一览与 C++ 对照）**

| Python | 语义 | C++ 对照 |
| --- | --- | --- |
| `__add__`/`__radd__` | `a + b`，右操作数回退 | `operator+`（含成员/非成员） |
| `__lt__`/`__le__`/`__gt__`/`__ge__` | 比较；`functools.total_ordering` 可补全 | `operator<` 等 |
| `__call__` | 让实例可调用 | `operator()` |
| `__enter__`/`__exit__` | `with` 语句 | 构造函数 + 析构函数中的资源管理 |
| `__iter__`/`__next__` | 迭代 | `begin()/end()`、`operator++`、`operator*` |

`a + b` 的分派是：先 `a.__add__(b)`；若返回 `NotImplemented` 且两操作数类型不同，再试 `b.__radd__(a)`；仍失败则抛 `TypeError`。这一点与 C++ 的“重载解析 + 隐式转换”不同：Python 的回退是显式的双向尝试，不做隐式数值提升。

**【例 PY2.2.6｜E-PY2.2.6】（`CharDataset`：只靠 `__len__` + `__getitem__` 就成为数据集）**

`data.py` 的 `CharDataset`（代码 `F-b01-data.CharDataset`）只实现了 `__len__` 与 `__getitem__`，因此可直接被 `len()`、`for`、索引访问，并满足后续 `DataLoader` 所依赖的“映射式数据集”协议；`__getitem__` 返回的是 `Batch` dataclass（【命题 PY2.4.2】）。这说明容器/数据集是**结构化接口**，靠实现特殊方法获得能力，而非继承某个基类。

**【代码 PY2.2.7｜Cd-PY2.2.7】（一个支持 `len`/索引/迭代的小容器）**

```python
class Buffer:
    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __iter__(self):
        return iter(self._data)     # 委托给 list 的迭代器

    def __repr__(self):
        return f"Buffer({self._data!r})"
```

## 3 属性、描述符与内存

属性（attribute）的读写可以被自定义逻辑接管：`property` 提供最常用的入口，描述符协议提供底层机制，`__slots__` 改变实例的内存布局。

**【定义 PY2.3.1｜D-PY2.3.1】（`property`：把方法伪装成属性）**

`property(fget, fset, fdel)` 或装饰器写法 `@property` / `@x.setter` / `@x.deleter` 把一个访问器对象绑定到类属性名上。此后读取 `obj.x` 触发 `fget`，赋值 `obj.x = v` 触发 `fset`，删除触发 `fdel`。用途：

- 在不改变调用方语法（仍是 `obj.x`）的前提下加入校验、惰性计算或缓存；
- 提供只读派生属性（不定义 setter 即可，赋值抛 `AttributeError`）；
- 向后兼容：公开字段改为 property 后调用方无需改动。

对照 C++：等价于手写 `getX()/setX()`，但调用处以属性语法书写，不暴露访问器名字。

**【定义 PY2.3.2｜D-PY2.3.2】（描述符协议）**

在**类**上定义了 `__get__(self, instance, owner)` / `__set__` / `__delete__` 的对象称为描述符。当类属性是描述符时，实例访问会转而调用它的 `__get__`，而不是直接返回描述符对象本身。优先级规则：

- **数据描述符**（同时有 `__get__` 和 `__set__`）优先于实例字典：读取时先问数据描述符；
- **非数据描述符**（只有 `__get__`）低于实例字典：只有实例字典没有同名项时才生效；
- 函数、`staticmethod`、`classmethod`、`property` 都是描述符；方法调用之所以能自动传入实例，正是函数描述符 `__get__` 返回了绑定方法。

因此完整的属性查找在【命题 PY2.1.3】之上叠加一层：`type(obj).__mro__` 上找到的类属性若是数据描述符则先由它处理，否则查实例字典，再否则用非数据描述符。

**【定义 PY2.3.3｜D-PY2.3.3】（`__slots__` 与内存布局）**

在类体中声明 `__slots__ = ("x", "y")` 后，实例不再拥有 `__dict__`，属性读写只能落到声明的槽位上。效果：

- 每个实例少一个字典对象，内存更省、属性访问更快；
- 阻止动态添加未声明属性（写入会抛 `AttributeError`），相当于把接口固定下来；
- 子类若未定义 `__slots__`，会重新获得 `__dict__`，父类的槽位收益被抵消；
- 与继承、多继承和弱引用等有若干限制，可作为“轻量记录类型”的优化。

对照 C++：`__slots__` 使实例近似于固定成员布局的结构体（无哈希表），而默认的 `__dict__` 则像一个按键存放属性的动态表。配套代码 `F-pre-py2.memory_footprint` 给出粗略对比。

**【例 PY2.3.4｜E-PY2.3.4】（`CharTokenizer.vocab_size` 是只读属性）**

`data.py` 中 `CharTokenizer.vocab_size`（代码 `F-b01-data.CharTokenizer`）用 `@property` 返回 `len(self.itos)`，没有 setter。于是 `tok.vocab_size` 可读，而 `tok.vocab_size = 3` 会抛 `AttributeError`。相对直接暴露 `self.vocab_size` 字段，这保证了词表大小与 `itos` 始终一致，是【定义 PY2.3.1】的最小实用场景。

## 4 数据类、类型标注与枚举

本节处理“数据结构声明”的三件工具：`dataclass` 消除样板、`typing` 提供可检查的接口描述、`enum` 表达具名常量。

**【定义 PY2.4.1｜D-PY2.4.1】（`dataclass`：样板代码生成器）**

`@dataclass` 读取类体中的**带注解**赋值，自动生成 `__init__`、`__repr__`、`__eq__`（默认开启，因此 `__hash__` 被置空）。常用开关：

- `frozen=True`：生成不可变实例（赋值抛 `FrozenInstanceError`），并据字段生成 `__hash__`；
- `order=True`：生成 `<`、`<=`、`>`、`>=`；
- `field(default_factory=list)`：为可变默认值提供工厂，避免所有实例共享同一 list；
- 仅注解、不赋值的字段没有默认值，必须由 `__init__` 参数提供。

对照 C++：接近“聚合体/记录类型”加编译器生成的构造与比较；`field` 的默认工厂对应“每个对象各自初始化其成员”。

**【命题 PY2.4.2｜P-PY2.4.2】（`data.py` 的 `Batch` 就是 dataclass）**

`data.py` 的 `Batch`（代码 `F-b01-data.Batch`）定义为：

```python
@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
```

`CharDataset.__getitem__` 返回 `Batch(x=chunk[:-1], y=chunk[1:])`（代码 `F-b01-data.CharDataset`），调用方按属性名取 `batch.x`/`batch.y`。相对位置元组，字段名让接口自解释，也减少“取错分量”的错误。`Batch` 的分组正是经验风险按 batch 平均的载体（【定义 1.2.4】）。

**【定义 PY2.4.3｜D-PY2.4.3】（`typing`：`Optional`/`Union`/`Sequence`/`Mapping`/`Protocol`）**

类型标注默认在运行时**不**求值（`from __future__ import annotations` 之后连字符串化都省去），它们主要供静态检查器（mypy、pyright）离线校验，不改变运行行为。常用构造：

- `Optional[X]` 与 `X | None` 等价，表示可能为 `None`；
- `Union[A, B]` 与 `A | B` 等价；`Literal["cpu", "cuda"]` 限定字面量取值；
- `Sequence[X]`/`Mapping[K, V]` 是只读抽象接口（协变），`list`/`dict` 是其具体实现；参数标注用抽象接口可放宽调用方传入的类型；
- `Protocol` 定义**结构化子类型**（duck typing 的类型化）：只要一个类实现了协议列出的方法与属性，它就满足该协议，无需继承；`@runtime_checkable` 后才允许 `isinstance` 检查（且只检查方法名是否存在）。

对照 C++20：`Protocol` 近似 concept 的结构性约束，但 Python 的检查可选、发生在类型检查器或（有限地）运行时。

**【例 PY2.4.4｜E-PY2.4.4】（用 `Protocol` 描述分词器接口）**

可以声明

```python
class TokenizerLike(Protocol):
    def encode(self, text: str, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str: ...
```

`CharTokenizer` 无需写 `class CharTokenizer(TokenizerLike)` 即满足该协议。参数 `ids: Sequence[int]` 允许 `list[int]` 或 `tuple[int, ...]`，这正对应【例 PY2.2.6】的“结构化接口”思想，配套代码 `F-pre-py2.SimpleTokenizer` 给出可运行示例。

**【定义 PY2.4.5｜D-PY2.4.5】（`enum`：具名常量）**

`class Color(enum.Enum): RED = 1; GREEN = 2; BLUE = 3` 中，每个成员都是单例：`Color.RED is Color.RED` 恒真，且 `Color.RED` 不是整数 1。`IntEnum` 让成员同时是整数（可与整数比较、可用于 `int` 上下文）。对照 C++：`enum class` 提供作用域与类型安全，Python 的 `Enum` 额外把成员做成真对象，支持迭代、按名查找与自定义方法，但由此带来运行时代价。

## 5 异常模型

Python 用异常统一处理错误：它既是错误通道，也是一种控制流。对 C 程序员，关键差异是“错误不再靠返回值逐层传递”。

**【定义 PY2.5.1｜D-PY2.5.1】（异常层级）**

异常类的根是 `BaseException`，其下 `Exception` 是常规可捕获异常的基类。必须记住：

- `KeyboardInterrupt`、`SystemExit`、`GeneratorExit` 直接继承 `BaseException` 而非 `Exception`，因此不应被 `except Exception` 吞掉；
- 裸 `except:` 会连上述控制流异常一并捕获，通常应避免；
- 库应定义自己的异常基类（如 `TokenizerError(Exception)`），让调用方能只捕获本库的错误。

对照 C++：近似 `std::exception` 的继承层级与 `catch` 的按类型匹配；区别是 Python 的 `BaseException` 给“用户中断/退出”留了不受常规捕获影响的顶层通道。

**【定义 PY2.5.2｜D-PY2.5.2】（`try/except/else/finally` 语义）**

- `try` 块正常结束时执行 `else`，它是“未发生异常后的后续步骤”，不应把可能抛异常的主逻辑放在这里；
- 无论是否异常，`finally` 都执行，用于释放资源；`finally` 中的 `return`/`raise` 会覆盖原有的异常或返回值（一般应避免）；
- `except (A, B) as e:` 可一次捕获多类；多个 `except` 按出现顺序匹配，故子类必须先于父类；
- 未被任何 `except` 匹配的异常会继续向上传播。

对照 C++：`try/catch` 没有 `else` 分支；`finally` 的作用在 C++ 中由 RAII 析构承担，Python 则用显式语句表达。

**【定义 PY2.5.3｜D-PY2.5.3】（`raise`、异常链与自定义异常）**

- `raise E(...)` 抛出异常实例或类；
- `raise E(...) from cause` 设置 `__cause__`，形成**显式**异常链，traceback 会打印“The above exception was the direct cause of…”；
- 在处理某异常的过程中再抛出新异常，会自动把原异常记入新异常的 `__context__`，形成**隐式**链；`raise ... from None` 可切断显示；
- 链式异常保留了根因，是调试“底层错误码被上层包装”场景的关键。

对照 C：C 习惯用错误码或 `errno` 逐层返回并检查，容易丢失上下文；Python 的异常链把每一层的原因都保留下来。

**【注 PY2.5.4｜R-PY2.5.4】（EAFP vs LBYL；对照 C 错误码）**

- **EAFP**（Easier to Ask Forgiveness than Permission）：直接执行操作，在 `except` 中处理失败；典型如直接查字典、捕获 `KeyError`；
- **LBYL**（Look Before You Leap）：先检查条件再操作，如先 `if key in d:`；
- EAFP 通常更快也更少竞态（检查与使用之间状态可能变化），适合“失败是异常路径”的场合；LBYL 在失败很常见或需要清晰分支时可读性更好。

对照 C：C 的 `int rc = f(); if (rc != 0) return rc;` 把错误通道与返回值耦合，每层都要显式传播；异常把错误通道独立出来，传播由展开栈自动完成，`finally`/`with` 负责清理。

**【例 PY2.5.5｜E-PY2.5.5】（`data.py` 的“捕获后降级”与“主动报错”）**

`data.py` 的 `load_text`（代码 `F-b01-data.load_text`）对 `urllib.request.urlopen` 使用 `try/except Exception`：网络失败时打印提示并回退到内置语料，属于“可恢复的外部错误就地降级”。相反，`get_batch`（代码 `F-b01-data.get_batch`）在语料短到一个 block 都切不出时主动 `raise ValueError(...)`，这是“调用方用法错误，应尽早失败”。两类处理分别对应 EAFP 的降级与 fail-fast。

**【代码 PY2.5.6｜Cd-PY2.5.6】（EAFP 风格的分词）**

```python
def encode_safe(stoi, text):
    out = []
    for ch in text:
        try:
            out.append(stoi[ch])     # EAFP：直接查表
        except KeyError:
            continue                 # 未知字符跳过
    return out
```

## 6 上下文管理器与 `with`

资源（文件、锁、临时状态）的获取与释放若靠手工配对，异常路径极易泄漏。`with` 语句把这一对操作交给上下文管理器。

**【定义 PY2.6.1｜D-PY2.6.1】（`with` 与 `__enter__`/`__exit__`）**

执行 `with ctx as v:` 时：

1. 调用 `ctx.__enter__()`，把返回值绑定到 `v`（不需要时写作 `with ctx:`）；
2. 执行块体；
3. 无论块体正常结束还是抛出异常，都调用 `ctx.__exit__(exc_type, exc, tb)`；正常结束时三个参数均为 `None`；
4. 若 `__exit__` 返回真值，异常被视为已处理，不再向外抛；返回假值（或 `None`）则异常继续传播。

对照 C++：`with` 的语义与 RAII 的作用域资源管理一致——构造时获取、离开作用域时释放；差异在于 Python 的清理是显式方法调用，且可在 `__exit__` 中决定是否吞掉异常。

**【命题 PY2.6.2｜P-PY2.6.2】（`contextlib.contextmanager`：用生成器写上下文管理器）**

`@contextmanager` 把一个“`yield` 恰好一次”的生成器转成上下文管理器：`yield` 之前的代码对应 `__enter__`，`yield` 的值绑定给 `as` 目标；`yield` 之后的代码对应 `__exit__`，应放在 `try/finally` 中以保证清理。若块体抛异常，异常在 `yield` 处被抛回生成器，因此可用 `try/except` 捕获并决定是否重抛。常用件还有：

- `contextlib.suppress(*exc)`：吞掉指定异常；
- `contextlib.ExitStack`：动态注册多个清理回调，适合清理数量运行时才确定的场景；
- `contextlib.closing(obj)`：离开时调用 `obj.close()`。

**【例 PY2.6.3｜E-PY2.6.3】（`data.py` 的原子写：临时文件 + `os.replace`）**

`CharTokenizer.save`（代码 `F-b01-data.CharTokenizer`）先把 JSON 写入 `path + ".tmp"`，再 `os.replace(tmp, path)` 原子改名；全程用 `with open(...)` 确保文件在异常时也关闭。要点：

- `open` 返回的对象本身就是上下文管理器，`with` 保证 `close`；
- `os.replace` 在同一文件系统上是原子操作，避免读者看到半截文件；
- 这一“临时文件 + 原子替换”模式也是本仓各类断点续传产物写盘时的通用约定。

**【代码 PY2.6.4｜Cd-PY2.6.4】（类式与生成器式两种上下文管理器）**

```python
class TimerCtx:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        return False    # 不吞异常

from contextlib import contextmanager

@contextmanager
def pushed(stack, item):
    stack.append(item)
    try:
        yield item
    finally:
        stack.pop()
```

## 7 装饰器与高阶函数

函数在 Python 中是普通对象，因此“把函数当参数传、把函数当返回值”无需特殊语法；装饰器正是这一能力在定义处的语法糖。

**【定义 PY2.7.1｜D-PY2.7.1】（函数是一等对象与高阶函数）**

函数对象可以赋值、放入容器、作为参数与返回值；`def` 只是“创建函数对象并绑定名字”的语句。接受或返回函数的函数称为高阶函数，例如 `sorted(key=...)`、`map`、`filter`、`functools.reduce`。函数对象自带 `__name__`、`__doc__`、`__defaults__`、`__closure__` 等元数据，闭包会捕获外层的自由变量。

对照 C/C++：近似函数指针或 `std::function`，但 Python 函数天然带闭包与反射元数据，且调用是动态的。

**【定义 PY2.7.2｜D-PY2.7.2】（装饰器：`f = deco(f)` 的语法糖）**

`@deco` 置于函数定义之上，等价于 `f = deco(f)`；装饰器接收函数并返回一个可调用对象（通常是包装函数）。应使用 `functools.wraps(f)` 把 `__name__`、`__doc__`、`__module__`、`__wrapped__` 等复制到包装函数上，否则日志、文档、框架反射与调试都会看到错误的名字。多个装饰器自下而上应用：`@a` `@b` 等价于 `a(b(f))`。

对照 C++：没有直接对应；语义上介于“编译期宏/模板包装”与“运行时代理”，但 Python 的替换发生在运行时且作用于对象。

**【定义 PY2.7.3｜D-PY2.7.3】（带参装饰器：三层结构）**

写 `@retry(3)` 时，先求值 `retry(3)` 得到一个“真正装饰器”，再把它作用到函数上，因此需要三层：

```python
def retry(times):            # 外层：收装饰器参数
    def deco(fn):            # 中层：收被装饰函数
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):   # 内层：收调用参数
            ...
        return wrapper
    return deco
```

注意无参装饰器只需两层（`deco(fn)`），带参装饰器需要三层；混淆两者会导致“把函数当参数错误地当成配置值”的隐蔽 bug。

**【定义 PY2.7.4｜D-PY2.7.4】（`functools.lru_cache`：记忆化）**

`@functools.lru_cache(maxsize=None)` 按调用参数缓存返回值，参数必须可哈希；`maxsize=None` 表示无界缓存。`fn.cache_info()` 返回命中/未命中统计，`fn.cache_clear()` 清空缓存。注意：用可变对象（list/dict）作参数会触发 `TypeError`；对带副作用或依赖可变全局状态的函数应谨慎缓存。

**【例 PY2.7.5｜E-PY2.7.5】（`data.py` 中的高阶用法）**

`get_batch`（代码 `F-b01-data.get_batch`）用列表推导 `[dataset[i].x for i in ix]` 逐样本取块，再 `torch.stack(...)` 聚合成 batch 张量；`CharTokenizer` 用 `sorted(set(chars))` 构建确定顺序的词表。像 `sorted`/`max`/`min` 这类函数接受 `key=` 回调，是典型的高阶用法（配套代码 `F-pre-py2.sort_by_abs` 示范 `key=abs`）。

**【代码 PY2.7.6｜Cd-PY2.7.6】（带参装饰器与 `lru_cache`）**

```python
import functools

def retry(times):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for k in range(times):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError:
                    if k == times - 1:
                        raise
        return wrapper
    return deco

@functools.lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
```

## 8 迭代器与生成器

迭代是 Python 最普遍的控制流：`for`、推导式、`sum`、`zip`、`DataLoader` 都建立在同一协议之上。生成器让“写一个迭代器”变得与写普通函数一样简单。

**【定义 PY2.8.1｜D-PY2.8.1】（迭代协议：`__iter__`/`__next__`/`StopIteration`）**

`for x in obj` 的执行是：

1. 调用 `iter(obj)` 得到迭代器（即调用 `obj.__iter__()`）；
2. 反复调用迭代器的 `__next__()`，把返回值绑定到 `x`；
3. 当 `__next__()` 抛 `StopIteration` 时，循环正常结束。

规则：`__iter__` 必须返回一个迭代器对象；类式迭代器通常 `return self` 并在自身实现 `__next__`，同时用一个内部游标记录进度。注意**迭代器是一次性的**，而“可迭代对象”（如 list）每次 `iter()` 都产生新迭代器。

对照 C++：近似 `begin()/end()` 加 `operator++`/`operator*` 的范围遍历；Python 用异常 `StopIteration` 表示结束，并把它包装进 `for`、推导式等语法。

**【定义 PY2.8.2｜D-PY2.8.2】（生成器函数与 `yield`：惰性求值）**

函数体内只要出现 `yield`，它就是生成器函数：调用它不会执行函数体，而是立即返回一个生成器对象。每次 `next()` 从上次暂停处恢复执行，直到下一个 `yield` 交出一个值；函数返回或走到末尾会抛 `StopIteration`。要点：

- 局部变量与执行位置在两次 `next` 之间被自动保存；
- 惰性：不预先计算整个序列，适合大文件、流式数据与无限序列；
- `yield from sub` 把子可迭代对象的产出转发出去，并正确传递 `send`/`throw`。

对照 C++20：概念上接近协程与惰性 range 视图；但 Python 生成器每次 `next` 恢复一个栈帧，实现与调度更轻。

**【命题 PY2.8.3｜P-PY2.8.3】（生成器表达式与列表推导式的区别）**

`[f(x) for x in it]` 立即构造完整 list；`(f(x) for x in it)` 返回惰性生成器，仅在被消费时计算，额外内存为 O(1)。因此：

- 传给 `sum`/`max`/`any`/`all` 等聚合函数时优先用生成器表达式，避免中间列表；
- 需要重复遍历或随机索引时用 list，因为生成器只能消费一次；
- 推导式中不要用带副作用的表达式。

对照 C++：近似 lazy range 与 eager `std::vector` 的取舍。

**【注 PY2.8.4｜R-PY2.8.4】（`itertools` 常用件）**

`itertools` 提供可组合的迭代器构件：`count`/`cycle`/`repeat`（无限流）、`islice`（惰性切片，不能负索引）、`chain`（顺序拼接多个可迭代对象）、`groupby`（按键分组，使用前必须先按同键排序）、`product`/`permutations`/`combinations`（组合枚举）、`tee`（把一个迭代器复制成多路）。它们都返回迭代器，可串成流水线且不物化中间结果。

**【例 PY2.8.5｜E-PY2.8.5】（读懂 `data.py` 的生成器与迭代用法）**

- `CharTokenizer.decode`（代码 `F-b01-data.CharTokenizer`）用生成器表达式 `"".join(self.itos[int(i)] for i in ids if ...)` 惰性解码，无中间 list；
- `CharDataset.__getitem__`（代码 `F-b01-data.CharDataset`）用切片 `self.data[i:i + block_size + 1]` 返回 `Batch`，配合 `__len__` 构成可迭代数据集（【例 PY2.2.6】）；
- `get_batch`（代码 `F-b01-data.get_batch`）用 `torch.stack([... for i in ix])` 把逐样本结果聚合成 batch 张量——逐样本“惰性取块”与批量“物化”的分界，正是本节强调的取舍；张量细节留待 PY5。

**【代码 PY2.8.6｜Cd-PY2.8.6】（手写生成器与 `itertools`）**

```python
import itertools

def windows(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield tuple(buf)
            buf.pop(0)

def take(it, n):
    return list(itertools.islice(it, n))
```

## 9 模块与包

一个文件、一个目录如何被组织成可复用的单元，以及 `import` 在运行时究竟做了什么。

**【定义 PY2.9.1｜D-PY2.9.1】（模块与 `import` 机制）**

模块就是被加载的 `.py` 文件所形成的对象，其全局命名空间即模块的 `__dict__`。`import m` 的语义：

1. 若 `m` 已在 `sys.modules` 中，直接使用缓存对象；
2. 否则定位并加载模块源、执行模块体、把模块对象放入 `sys.modules`；
3. 最后在当前命名空间绑定名字 `m`。

`from m import x` 额外绑定子对象 `x`；`import m as a` 只是起别名。因此**模块体只在首次导入时执行一次**，这也是模块级初始化代码（如创建词表）适合放的位置。

对照 C：`import` 兼有“头文件声明”和“链接实现”的作用，但发生在运行时的对象查找层面，没有编译期符号表，也不存在 ODR 那种重复定义错误（同名冲突表现为遮蔽，见【注 PY2.9.4】）。

**【定义 PY2.9.2｜D-PY2.9.2】（`__name__ == "__main__"` 与双入口）**

每个模块都有 `__name__`：被导入时是模块名，作为脚本直接运行时是 `"__main__"`。因此

```python
if __name__ == "__main__":
    main()
```

把“可复用的库代码”和“命令行入口”分开：导入时不会触发演示或副作用。对照 C：一个翻译单元既可被链接为库、也可提供 `main`；Python 用这行守卫在语言层表达同一区分。

**【定义 PY2.9.3｜D-PY2.9.3】（包与相对导入）**

含 `__init__.py` 的目录是包（较新版本支持命名空间包，可省略该文件）。`import pkg.mod`、`from pkg import mod` 为绝对导入；包内可用相对导入 `from . import sibling`、`from .. import parent`。相对导入依赖模块的 `__package__`；若把一个本应作为子模块的文件直接当脚本运行，`__package__` 为空，相对导入会失败。实践建议：入口脚本放在包外、使用绝对导入，库内部用相对导入。

**【注 PY2.9.4｜R-PY2.9.4】（`sys.path` 与查找顺序；对照链接）**

导入查找顺序大致为：内置模块 → `sys.path` 中的目录（脚本所在目录、`PYTHONPATH`、site-packages 等）。要点：

- 脚本所在目录被自动加入 `sys.path` 之首，因此**同名的本地文件可能遮蔽标准库或第三方库**，这是最常见的“模块名冲突”坑；
- 已导入模块缓存在 `sys.modules`，改了源码需重启解释器或显式重载；
- 对照 C：类似 `-I`/`-L` 的搜索路径加链接期符号解析；但 Python 的查找对象是“文件/包到模块对象”，并且结果被缓存。

**【例 PY2.9.5｜E-PY2.9.5】（`data.py` 与 `model.py` 的模块结构）**

- `data.py` 顶部 `from __future__ import annotations` 与 `import torch`，并 `from dataclasses import dataclass`（代码 `F-b01-data`）；
- 末尾 `if __name__ == "__main__":`（代码 `F-b01-data`）在直接运行时准备 tokenizer 并抽样 batch，被其它模块导入时不执行（【定义 PY2.9.2】）；
- `model.py` 用 `from configs import GPTConfig`（代码 `F-b01-model`）——在 `from_scratch/` 目录内以脚本运行时，同目录的 `configs.py` 因“脚本目录进 `sys.path`”而可直接导入（【注 PY2.9.4】）；它定义的 `MLP` 对应【定义 1.1.3】、其 `forward` 对应【代码 1.1.4】。

## 10 来源与许可

本章以官方文档为语义与术语的权威来源，正文与示例均为独立改写，未整段复制；具体搬运范围如下。

| 来源 | URL | 许可 | 搬运范围 |
| --- | --- | --- | --- |
| Python 教程 · Classes | https://docs.python.org/3/tutorial/classes.html | PSF License Version 2（示例另按 0BSD） | 类/继承/MRO/迭代器与生成器的术语与语义 |
| Python 教程 · Errors and Exceptions | https://docs.python.org/3/tutorial/errors.html | PSF License Version 2（示例另按 0BSD） | 异常层级、`try/except/else/finally`、`raise ... from` |
| Python 语言参考 · Data model | https://docs.python.org/3/reference/datamodel.html | PSF License Version 2（示例另按 0BSD） | 对象模型、特殊方法、描述符、`__slots__` |
| Python 语言参考 · The import system | https://docs.python.org/3/reference/import.html | PSF License Version 2（示例另按 0BSD） | 导入机制、包、`__name__` 语义 |
| Python 语言参考 · Compound statements | https://docs.python.org/3/reference/compound_stmts.html | PSF License Version 2（示例另按 0BSD） | `with`、`try`、`class`、函数定义语法 |
| `dataclasses` 标准库 | https://docs.python.org/3/library/dataclasses.html | PSF License Version 2（示例另按 0BSD） | `@dataclass`、`field`、`frozen`/`order` |
| `typing` 标准库 | https://docs.python.org/3/library/typing.html | PSF License Version 2（示例另按 0BSD） | `Optional/Union/Sequence/Mapping/Protocol` |
| `enum` 标准库 | https://docs.python.org/3/library/enum.html | PSF License Version 2（示例另按 0BSD） | `Enum`/`IntEnum` 语义 |
| `contextlib` 标准库 | https://docs.python.org/3/library/contextlib.html | PSF License Version 2（示例另按 0BSD） | `contextmanager`/`suppress`/`ExitStack`/`closing` |
| `functools` 标准库 | https://docs.python.org/3/library/functools.html | PSF License Version 2（示例另按 0BSD） | `wraps`/`lru_cache`/`total_ordering` |
| `itertools` 标准库 | https://docs.python.org/3/library/itertools.html | PSF License Version 2（示例另按 0BSD） | 迭代器构件清单 |
| Descriptor HowTo Guide | https://docs.python.org/3/howto/descriptor.html | PSF License Version 2（示例另按 0BSD） | 描述符协议与查找优先级 |

本仓内部材料（自有，按项目许可使用，仅作内链引用）：`v1.0/01-基础/code/from_scratch/data.py`（`F-b01-data`）与 `v1.0/01-基础/code/from_scratch/model.py`（`F-b01-model`）。

## 11 小结与自测

本章要点：类属性与实例属性的查找顺序（【命题 PY2.1.3】）、MRO 与 `super()`（【定义 PY2.1.4】、【注 PY2.1.5】）、特殊方法构成的协议（【定义 PY2.2.1】）、`property` 与描述符（【定义 PY2.3.1】、【定义 PY2.3.2】）、异常链（【定义 PY2.5.3】）、`with` 与 RAII（【定义 PY2.6.1】）、装饰器三层（【定义 PY2.7.3】）、迭代与惰性（【定义 PY2.8.1】、【定义 PY2.8.2】）、模块与双入口（【定义 PY2.9.1】、【定义 PY2.9.2】）。

**【练习 PY2.11.1｜Ex-PY2.11.1】（类属性共享陷阱与 `dataclass` 修复）**

写一个类，其类属性为 `tasks = []`，在两个实例上分别 `self.tasks.append(...)`，观察互相污染；然后用实例属性或 `field(default_factory=list)`（【定义 PY2.4.1】）修复，并解释【命题 PY2.1.3】的作用。

**【练习 PY2.11.2｜Ex-PY2.11.2】（给 `Buffer` 补全协议）**

在【代码 PY2.2.7】的 `Buffer` 上增加 `__contains__`、`__eq__` 与 `__hash__`（或说明为何不适合哈希），并验证 `x in buf`（【命题 PY2.2.3】）与 `buf1 == buf2`（【命题 PY2.2.4】）符合预期。

**【练习 PY2.11.3｜Ex-PY2.11.3】（用生成器与装饰器处理 `data.py` 的采样）**

参照 `F-b01-data.get_batch`，写一个惰性生成器函数 `sample_blocks(dataset, batch_size)`，每次 `yield` 一个 batch；再用带参装饰器 `@retry(times)`（【定义 PY2.7.3】）包装一个可能抛 `RuntimeError` 的模拟采样函数，验证重试语义。
