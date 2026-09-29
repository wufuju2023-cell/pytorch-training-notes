# 【源代码｜F-pre-py3】v1.0/00-预备/code/py3_engineering.py — 工程与环境教学骨架（argparse + logging + 可测函数）
# 相关文档：【文档｜DOC-PY3】；命令行【注 PY3.3.1】、logging【注 PY3.3.3】、随机种子【注 PY3.7.2】。
# 仅依赖标准库（numpy/torch 为可选的跨库播种）；CPU 可直接运行：
#   python py3_engineering.py --seed 0 --steps 3 --log-level DEBUG
"""PY3 工程与环境的最小可运行骨架。

覆盖三件事，且都按“可测试”的方式拆分：

1. ``argparse``：把 seed/steps/log-level 从源码里挪到命令行；
2. ``logging``：统一格式与级别，替代散落的 print；
3. 纯函数：``draw`` 与 ``summarize`` 不产生副作用，便于 pytest 与复现实验。

对应的理论条目见《PY3-工程与环境.md》第 3、7 节。
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics

# 【F-pre-py3.LOGGER｜常量】本模块 logger（不在 import 时配置，配置权交给入口）
LOGGER = logging.getLogger(__name__)

# 【F-pre-py3.DEFAULT_FORMAT｜常量】日志统一格式：时间 级别 模块名 消息
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


# 【F-pre-py3.configure_logging｜函数】配置根 logger 的级别与格式
def configure_logging(level: str = "INFO") -> None:
    """配置日志：级别取 DEBUG/INFO/WARNING/ERROR，格式用 DEFAULT_FORMAT。

    对照 C 的 printf + 日志级别宏；只配置一次，库代码不调用本函数。
    """
    logging.basicConfig(level=getattr(logging, level.upper()), format=DEFAULT_FORMAT)
    LOGGER.debug("logging configured at level=%s", level)


# 【F-pre-py3.seed_everything｜函数】固定 Python/numpy/torch 三处随机源
def seed_everything(seed: int) -> None:
    """固定随机源，保证同一 seed 得到同一结果。

    Python 的 ``random`` 必装；``numpy``/``torch`` 若存在则一并播种，
    不强依赖它们，保证本教学文件在纯标准库环境也能运行。
    """
    random.seed(seed)
    try:  # pragma: no cover - 可选依赖
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:  # pragma: no cover - 可选依赖
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# 【F-pre-py3.draw｜函数】按 seed 生成一组 [0,1) 随机数（纯函数式）
def draw(seed: int, n: int = 5) -> list:
    """先播种再采样，返回长度 n 的随机数列表；同 seed 必得同结果。"""
    seed_everything(seed)
    return [random.random() for _ in range(n)]


# 【F-pre-py3.summarize｜函数】对数值序列做无副作用汇总
def summarize(values) -> dict:
    """返回样本量、均值与总体标准差；空输入给出明确异常。"""
    values = list(values)
    if not values:
        raise ValueError("summarize() requires a non-empty sequence")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values),
    }


# 【F-pre-py3.parse_args｜函数】定义并解析命令行参数
def parse_args(argv=None) -> argparse.Namespace:
    """命令行接口：--seed / --steps / --log-level（对照 getopt）。"""
    parser = argparse.ArgumentParser(description="PY3 工程与环境教学骨架")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--steps", type=int, default=10, help="模拟步数（默认 10）")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    return parser.parse_args(argv)


# 【F-pre-py3.main｜函数】串起解析、日志、播种与汇总流程
def main(argv=None) -> int:
    """入口：解析参数 -> 配置日志 -> 播种 -> 逐步采样并汇总。"""
    args = parse_args(argv)
    configure_logging(args.log_level)
    seed_everything(args.seed)
    LOGGER.info("seed=%d steps=%d", args.seed, args.steps)

    for step in range(args.steps):
        stats = summarize(draw(args.seed + step, n=8))
        LOGGER.debug(
            "step=%d n=%d mean=%.4f stdev=%.4f",
            step,
            stats["n"],
            stats["mean"],
            stats["stdev"],
        )

    final = summarize(draw(args.seed, n=args.steps + 1))
    LOGGER.info(
        "summary n=%d mean=%.4f stdev=%.4f",
        final["n"],
        final["mean"],
        final["stdev"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
