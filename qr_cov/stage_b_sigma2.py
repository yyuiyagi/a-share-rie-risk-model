# -*- coding: utf-8 -*-
r"""
qr_cov/stage_b_sigma2.py —— 阶段 B 第一步：标准化能否消掉 σ² 校准？
======================================================================
要检验的命题：把每只股票标准化到单位方差后，噪声尺度 σ² 是否自动变成 1？

结论（预期）：不能。标准化把【总方差】设成 1，不是【噪声】设成 1。
当因子贡献比例 s>0 时，真实噪声尺度 σ²_idio = 1 - s < 1，
naive 的 σ²=1 会给出过宽的噪声带。

本脚本：
  [A] 合成数据（已知真实 s）：证明噪声 bulk 落在 1-s，而不是 1
  [B] 真实 A 股数据：估计 s，量化 naive vs 正确噪声带的差异

用法：python qr_cov/stage_b_sigma2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension


def mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def standardize(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return Z / sd


def synthetic_case(s: float, N: int = 500, T: int = 252, seed: int = 7):
    """等相关系数模型：Σ = (1-s) I + s 11ᵀ，因子方差份额 = s。

    返回样本相关矩阵的特征值、naive/correct 噪声带上界。
    """
    rng = np.random.default_rng(seed)
    Sigma = (1 - s) * np.eye(N) + s * np.ones((N, N))
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((T, N)) @ L.T
    Z = standardize(X)
    S = Z.T @ Z / T
    ev = np.linalg.eigvalsh(S)[::-1]
    gamma = N / T
    _, hi_naive = mp_edges(1.0, gamma)          # naive σ²=1
    _, hi_correct = mp_edges(1 - s, gamma)      # 正确 σ²=1-s
    # 噪声 bulk 的均值（去掉最大的一个信号特征值）
    bulk = ev[1:]
    return ev, bulk.mean(), hi_naive, hi_correct, gamma


def main() -> None:
    print("=" * 78)
    print("阶段 B 第一步：标准化能否消掉 σ² 校准？")
    print("=" * 78)

    # ---- [A] 合成数据 ----
    print("\n[A] 合成数据：真实因子份额 s 已知")
    print(f"    {'s':>6} {'真实σ²':>8} {'bulk均值':>9} {'naive λ₊':>10} "
          f"{'正确λ₊':>10} {'naive信号数':>11}")
    for s in [0.0, 0.2, 0.5, 0.8]:
        ev, bulk_mean, hi_naive, hi_correct, gamma = synthetic_case(s)
        n_naive = int((ev > hi_naive).sum())
        n_correct = int((ev > hi_correct).sum())
        print(f"    {s:>6.1f} {1-s:>8.2f} {bulk_mean:>9.3f} {hi_naive:>10.3f} "
              f"{hi_correct:>10.3f} {n_naive:>11d}")

    print("""
    读法：
    - bulk均值 应 ≈ 真实σ² = 1-s，而不是 1。这说明噪声尺度不是 1。
    - naive λ₊（用 σ²=1）比正确 λ₊（用 σ²=1-s）高，s 越大差越多。
    - 「naive信号数」是错误地把 σ²=1 当噪声尺度时判出的信号个数。""")

    # ---- [B] 真实 A 股数据 ----
    print("\n" + "=" * 78)
    print("[B] 真实 A 股数据（500 只，最近 252 天）")
    print("=" * 78)
    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    U = comp.head(500).index.tolist()
    X = Rimp[U].iloc[-252:]
    Z = standardize(X.to_numpy(dtype=float))
    S = Z.T @ Z / Z.shape[0]
    ev = np.linalg.eigvalsh(S)[::-1]
    T = Z.shape[0]      # 天数
    N = Z.shape[1]      # 资产数
    gamma = N / T
    _, hi_naive = mp_edges(1.0, gamma)
    n_signal = int((ev > hi_naive).sum())
    signal_sum = ev[:n_signal].sum()
    s_hat = signal_sum / N                       # 因子方差份额估计
    sigma2_idio = 1 - s_hat
    _, hi_correct = mp_edges(sigma2_idio, gamma)
    bulk = ev[n_signal:]
    print(f"    γ = {gamma:.3f}")
    print(f"    naive λ₊（σ²=1）        = {hi_naive:.3f}")
    print(f"    naive 信号方向数          = {n_signal}")
    print(f"    信号特征值合计占迹        = {signal_sum:.1f} / {N} = {s_hat*100:.1f}%")
    print(f"    → 估计 σ²_idio = 1 - {s_hat:.3f} = {sigma2_idio:.3f}")
    print(f"    正确 λ₊（σ²_idio）       = {hi_correct:.3f}")
    print(f"    noise bulk 均值           = {bulk.mean():.3f}  "
          f"（应接近 σ²_idio={sigma2_idio:.3f}）")
    gray = int(((ev > hi_correct) & (ev <= hi_naive)).sum())
    print(f"    灰区方向数（正确带之上、naive 带之下）= {gray}")

    print("""
    结论：
    - 标准化后噪声尺度不是 1，而是 ≈ 1-s（因子贡献了 s 的方差）。
    - 用 naive σ²=1 会把噪声带算得过宽（λ₊ 从正确值抬到 naive 值），
      导致灰区内的弱信号被误判为噪声。
    - 因此：标准化【不能】消掉 σ² 校准，只是把「总噪声尺度」变成
      「特质噪声占多少」，而后者仍需估计（s 或 bulk 均值）。""")


if __name__ == "__main__":
    main()
