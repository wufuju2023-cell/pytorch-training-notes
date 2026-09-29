#!/usr/bin/env python3
# 【源代码｜F-pre-py5】v1.0/00-预备/code/py5_torch_from_zero.py — PyTorch 从零：线性回归与两层 MLP 训练闭环
# 相关文档：【定义 1.1.1｜D-1.1.1】线性模型、【定义 1.1.3｜D-1.1.3】多层感知机、【代码 1.1.4｜Cd-1.1.4】MLP.forward
#           【定义 1.2.4｜D-1.2.4】交叉熵与负对数似然、【定义 1.2.5｜D-1.2.5】均方误差、【算法 1.5.1｜A-1.5.1】中心差分梯度检查
# 配套文档：v1.0/00-预备/PY5-PyTorch从零.md（DOC-PY5）；配套 notebook：v1.0/notebooks/00-预备/N22_PyTorch从零.ipynb（N-22）
"""PyTorch 从零：两个可运行的最小训练闭环（纯 CPU 也完整可跑）。

包含两个自包含示例：

* :func:`run_linear_regression` —— 单层线性模型 + ``nn.MSELoss`` + ``SGD``，
  在 ``y = 3x + 2 + 噪声`` 上完整演示标准训练五步（【算法 PY5.7.1】）。
* :func:`run_mlp_classification` —— 两层 ReLU MLP（对照【定义 1.1.3】/【代码 1.1.4】）
  + 自定义 ``Dataset``/``DataLoader`` + ``nn.CrossEntropyLoss`` + ``Adam``，
  带训练/验证、checkpoint 保存与加载续训，并在 CUDA 可用时启用 AMP（【注 PY5.7.4】）。

运行::

    python3 py5_torch_from_zero.py                 # 线性回归 + MLP 分类
    python3 py5_torch_from_zero.py --mode linear   # 只跑线性回归
    python3 py5_torch_from_zero.py --mode mlp --epochs 10

设备选择：有 CUDA 用 GPU，否则自动退化为 CPU；本文件在 CPU 上不触碰任何 CUDA-only API。
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 0


# 【F-pre-py5.make_regression_data｜函数】生成 y = 3x + 2 + 高斯噪声 的一维回归数据
def make_regression_data(n: int = 200, noise: float = 0.1, seed: int = SEED):
    """返回 ``(x, y)``，形状均为 ``(n, 1)``，设备为 CPU。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 10.0 - 5.0
    y = 3.0 * x + 2.0 + noise * torch.randn(n, 1, generator=g)
    return x, y


# 【F-pre-py5.LinearRegressionModel｜类】单层线性回归：y = x W^T + b（对照【定义 1.1.1】）
class LinearRegressionModel(nn.Module):
    """最简单的 ``nn.Module``：只注册一个 ``nn.Linear`` 子模块。"""

    def __init__(self, in_features: int = 1, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


# 【F-pre-py5.TwoLayerMLP｜类】两层 ReLU MLP（对照【定义 1.1.3】三层结构与【代码 1.1.4】forward）
class TwoLayerMLP(nn.Module):
    """``input -> Linear -> ReLU -> Linear -> logits`` 的两层 MLP。"""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# 【F-pre-py5.ToyClassificationDataset｜类】三簇二维玩具分类集（Dataset 协议：__len__/__getitem__）
class ToyClassificationDataset(Dataset):
    """以三个高斯簇构造的三分类数据集，标签为 ``torch.long``。"""

    def __init__(self, n_per_class: int = 120, seed: int = SEED):
        g = torch.Generator().manual_seed(seed)
        centers = torch.tensor([[2.0, 0.0], [-1.0, 1.8], [-1.0, -1.8]])
        xs, ys = [], []
        for cls in range(centers.shape[0]):
            xs.append(centers[cls] + 0.6 * torch.randn(n_per_class, 2, generator=g))
            ys.append(torch.full((n_per_class,), cls, dtype=torch.long))
        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# 【F-pre-py5.train_one_epoch｜函数】一个 epoch 的标准训练五步（【算法 PY5.7.1】）
def train_one_epoch(model, loader, criterion, optimizer, device):
    """前向 -> 损失 -> 清零梯度 -> 反向 -> 更新；返回按样本数加权的平均损失。"""
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)                       # 1 前向
        loss = criterion(pred, yb)             # 2 损失
        optimizer.zero_grad(set_to_none=True)  # 3 清零梯度（必须在 backward 前）
        loss.backward()                        # 4 反向（等价于【定理 1.3.4】递推）
        optimizer.step()                       # 5 更新参数
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(total_n, 1)


# 【F-pre-py5.train_one_epoch_amp｜函数】CUDA 混合精度版单 epoch（CPU 路径不进入）
def train_one_epoch_amp(model, loader, criterion, optimizer, device, scaler):
    """与 :func:`train_one_epoch` 相同，但用 ``autocast`` + ``GradScaler``（仅 CUDA）。"""
    model.train()
    total_loss, total_n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(xb), yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()          # 放大损失，防 fp16 下溢
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(total_n, 1)


# 【F-pre-py5.evaluate｜函数】验证：eval() + no_grad()，只前向，返回 (loss, acc)
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        total_n += xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


# 【F-pre-py5.save_checkpoint｜函数】保存 model/optimizer/epoch 的完整 checkpoint
def save_checkpoint(path, model, optimizer, epoch):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(ckpt, path)


# 【F-pre-py5.load_checkpoint｜函数】加载 checkpoint（map_location 兜底跨设备）
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


# 【F-pre-py5.run_linear_regression｜函数】线性回归闭环：拟合 y = 3x + 2
def run_linear_regression(device, steps=300, lr=0.05, verbose=True):
    torch.manual_seed(SEED)
    x, y = make_regression_data()
    x, y = x.to(device), y.to(device)
    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if verbose and (step == 1 or step % 50 == 0):
            print(f"[linear] step {step:4d}  loss={loss.item():.5f}")

    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    if verbose:
        print(f"[linear] 拟合结果: weight={weight:.3f} (真值 3.0)  bias={bias:.3f} (真值 2.0)")
    return model, (weight, bias)


# 【F-pre-py5.run_mlp_classification｜函数】两层 MLP 分类闭环（Dataset/DataLoader + 验证 + checkpoint + 可选 AMP）
def run_mlp_classification(device, epochs=8, batch_size=64, lr=1e-2,
                           ckpt_path=None, verbose=True):
    torch.manual_seed(SEED)
    dataset = ToyClassificationDataset()
    n_val = len(dataset) // 5
    n_train = len(dataset) - n_val
    ds_train, ds_val = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = TwoLayerMLP(in_dim=2, hidden=32, out_dim=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    for epoch in range(1, epochs + 1):
        if scaler is not None and scaler.is_enabled():
            train_loss = train_one_epoch_amp(model, dl_train, criterion, optimizer, device, scaler)
        else:
            train_loss = train_one_epoch(model, dl_train, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dl_val, criterion, device)
        if verbose:
            print(f"[mlp] epoch {epoch:2d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

    if ckpt_path:
        save_checkpoint(ckpt_path, model, optimizer, epochs)
        if verbose:
            print(f"[mlp] checkpoint 已保存: {ckpt_path}")
        epoch_loaded = load_checkpoint(ckpt_path, model, optimizer, map_location="cpu")
        if verbose:
            print(f"[mlp] 已从 checkpoint 恢复 (epoch={epoch_loaded})，再训练 1 个 epoch")
        train_one_epoch(model, dl_train, criterion, optimizer, device)

    return model


# 【F-pre-py5.main｜函数】命令行入口：--mode linear|mlp|both
def main(argv=None):
    parser = argparse.ArgumentParser(description="PY5 · PyTorch 从零最小训练闭环")
    parser.add_argument("--mode", choices=["linear", "mlp", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=8, help="MLP 训练轮数")
    parser.add_argument("--steps", type=int, default=300, help="线性回归步数")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "py5_ckpt.pt"),
        help="MLP checkpoint 路径",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verbose = not args.quiet
    if verbose:
        print(f"device = {device}   torch = {torch.__version__}")

    if args.mode in ("linear", "both"):
        run_linear_regression(device, steps=args.steps, lr=0.05, verbose=verbose)
    if args.mode in ("mlp", "both"):
        run_mlp_classification(
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            ckpt_path=args.ckpt,
            verbose=verbose,
        )
    if verbose:
        print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
