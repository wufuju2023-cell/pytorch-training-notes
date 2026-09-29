#!/usr/bin/env python3
# 【源代码｜F-pre-py6】v1.0/00-预备/code/py6_torch_ecosystem.py — PyTorch 进阶与生态教学代码
# 相关文档：《00-预备/PY6-PyTorch进阶与生态.md》【文档｜DOC-PY6】；
#           条目对照【定义 PY6.1.1】梯度检查点、【定义 PY6.1.3】autocast、
#           【定义 PY6.2.1】torch.profiler、【定义 PY6.2.4】torch.compile、【定义 PY6.3.1】transformers。
"""PyTorch 进阶与生态（CPU 可跑；可选依赖缺失即跳过）教学代码。

覆盖主题
--------
* 梯度检查点 ``torch.utils.checkpoint``（用计算换显存）；
* 非连续张量与 ``view`` / ``reshape`` 的差异；
* 混合精度 ``autocast`` 与 ``GradScaler``（CPU 上只演示语义）；
* ``torch.profiler`` 与显存统计、``empty_cache`` 的适用边界；
* ``torch.compile`` 可用性探测（CPU 用 eager 后端避免长编译）；
* ``transformers`` / ``peft`` / ``trl`` / ``accelerate`` 的最小可读调用链。

用法
----
::

    python3 py6_torch_ecosystem.py --demo all        # 依次跑全部演示
    python3 py6_torch_ecosystem.py --demo checkpoint # 只跑其中一个

所有 ``import`` 都在函数内完成：缺少 ``torch`` 以外生态包时只打印提示并继续，
不会让整个文件导入失败。
"""

from __future__ import annotations

import argparse
import importlib.util


# 【F-pre-py6._require_torch｜函数】按需导入 torch，缺失时返回 None
def _require_torch():
    """导入 torch；不可用时打印提示并返回 ``None``。"""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError:
        print("[跳过] 未安装 torch：pip install torch")
        return None


# 【F-pre-py6.checkpoint_demo｜函数】梯度检查点前后向一致性验证
def checkpoint_demo() -> None:
    """用 ``torch.utils.checkpoint`` 重算中间激活，验证前向/梯度与普通前向一致。"""
    torch = _require_torch()
    if torch is None:
        return
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)
    seg = nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(),
        nn.Linear(32, 8), nn.ReLU(),
        nn.Linear(8, 1),
    )

    def run(t):
        return seg(t)

    x = torch.randn(8, 16)
    plain = run(x).detach()
    ckpt = checkpoint(run, x, use_reentrant=False).detach()
    print("[checkpoint] 前向一致:", bool(torch.allclose(plain, ckpt)))

    x1 = x.clone().requires_grad_(True)
    run(x1).sum().backward()
    x2 = x.clone().requires_grad_(True)
    checkpoint(run, x2, use_reentrant=False).sum().backward()
    print("[checkpoint] 梯度一致:", bool(torch.allclose(x1.grad, x2.grad, atol=1e-6)))


# 【F-pre-py6.noncontiguous_demo｜函数】非连续张量上 view/reshape/contiguous 的差别
def noncontiguous_demo() -> None:
    """转置后张量非连续：``view`` 报错，``reshape``/``contiguous().view`` 正常。"""
    torch = _require_torch()
    if torch is None:
        return
    x = torch.arange(24).reshape(2, 3, 4)
    xt = x.transpose(1, 2)
    print("[non-contig] transpose 后 is_contiguous:", xt.is_contiguous())
    try:
        xt.view(2, 12)
        print("[non-contig] view 意外成功")
    except RuntimeError as exc:
        print("[non-contig] view 在非连续张量上报错:", str(exc).splitlines()[0])
    print("[non-contig] reshape 兜底:", tuple(xt.reshape(2, 12).shape))
    print("[non-contig] contiguous().view:", tuple(xt.contiguous().view(2, 12).shape))


# 【F-pre-py6.autocast_demo｜函数】混合精度 autocast 与 GradScaler 语义
def autocast_demo() -> None:
    """CPU 用 bfloat16 演示 autocast；CUDA 上再展示 GradScaler 的用法。"""
    torch = _require_torch()
    if torch is None:
        return
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    x = torch.randn(64, 32, device=device)

    with torch.autocast(device_type=device, dtype=dtype):
        out = model(x)
    print(f"[autocast] device={device} 计算 dtype={dtype} 输出 dtype={out.dtype}")

    if device == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = out.float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        print("[autocast] GradScaler(fp16) 一步完成")
    else:
        print("[autocast] CPU 无 GradScaler 下溢问题，fp32 master weight 直接更新")


# 【F-pre-py6.profiler_demo｜函数】torch.profiler 统计算子耗时
def profiler_demo() -> None:
    """用 ``torch.profiler`` 打印 CPU 耗时最高的前几个算子。"""
    torch = _require_torch()
    if torch is None:
        return
    from torch import nn

    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    x = torch.randn(256, 64)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        for _ in range(5):
            model(x).sum().backward()
            model.zero_grad(set_to_none=True)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))


# 【F-pre-py6.memory_demo｜函数】显存统计与 empty_cache 的适用边界
def memory_demo() -> None:
    """CUDA 下报告 allocated/reserved；CPU 下说明 ``empty_cache`` 不适用。"""
    torch = _require_torch()
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1024, 1024, device="cuda")
        mib = 2 ** 20
        print(f"[mem] allocated={torch.cuda.memory_allocated()/mib:.1f}MiB")
        print(f"[mem] reserved ={torch.cuda.memory_reserved()/mib:.1f}MiB")
        del x
        torch.cuda.empty_cache()
        print(f"[mem] empty_cache 后 reserved={torch.cuda.memory_reserved()/mib:.1f}MiB")
    else:
        print("[mem] 无 CUDA：跳过显存统计；empty_cache 只作用于 CUDA 缓存分配器")


# 【F-pre-py6.compile_demo｜函数】torch.compile 可用性探测
def compile_demo() -> None:
    """CPU 上用 eager 后端（快速）演示 ``torch.compile``；CUDA 上用默认后端。"""
    torch = _require_torch()
    if torch is None:
        return
    from torch import nn

    if not hasattr(torch, "compile"):
        print("[compile] 当前 torch 无 torch.compile，跳过")
        return
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
    x = torch.randn(16, 32)
    backend = "eager" if not torch.cuda.is_available() else "inductor"
    try:
        compiled = torch.compile(model, backend=backend)
        y = compiled(x)
        print(f"[compile] backend={backend} 前向成功 shape={tuple(y.shape)}")
    except Exception as exc:  # noqa: BLE001
        print("[compile] 本环境不可用:", type(exc).__name__, str(exc)[:80])


# 【F-pre-py6.ecosystem_demo｜函数】transformers/peft/trl/accelerate 速览
def ecosystem_demo() -> None:
    """探测生态包版本，并在已安装时构造最小对象（不下载任何权重）。"""
    for name in ("transformers", "peft", "trl", "accelerate"):
        if importlib.util.find_spec(name) is None:
            print(f"[ecosystem] {name} 未安装：pip install {name}")
        else:
            mod = __import__(name)
            print(f"[ecosystem] {name} {getattr(mod, '__version__', '?')}")

    try:
        from peft import LoraConfig
        cfg = LoraConfig(r=8, lora_alpha=16, bias="none",
                         task_type="CAUSAL_LM",
                         target_modules=["q_proj", "v_proj"])
        print(f"[ecosystem] LoraConfig ok: r={cfg.r} alpha={cfg.lora_alpha}")
    except Exception:  # noqa: BLE001
        print("[ecosystem] peft 不可用，跳过 LoraConfig")

    try:
        from accelerate import Accelerator
        print("[ecosystem] Accelerator device:", Accelerator().device)
    except Exception:  # noqa: BLE001
        print("[ecosystem] accelerate 不可用，跳过 Accelerator")


_DEMOS = {
    "checkpoint": checkpoint_demo,
    "noncontig": noncontiguous_demo,
    "autocast": autocast_demo,
    "profiler": profiler_demo,
    "memory": memory_demo,
    "compile": compile_demo,
    "ecosystem": ecosystem_demo,
}


# 【F-pre-py6.main｜函数】命令行入口，按名调度各演示
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PyTorch 进阶与生态教学代码")
    ap.add_argument("--demo", default="all", choices=[*_DEMOS, "all"],
                    help="要运行的演示；默认 all")
    args = ap.parse_args(argv)

    names = list(_DEMOS) if args.demo == "all" else [args.demo]
    for name in names:
        print(f"\n===== {name} =====")
        _DEMOS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
