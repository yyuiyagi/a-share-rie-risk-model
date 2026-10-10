# -*- coding: utf-8 -*-
r"""
qr_cov/stage_b_spectrum.py —— 真实数据特征值分布（可视化）
========================================================
选取 4 个 252 天窗口，画相关矩阵特征值的分布，并标出：
  - naive 噪声带上界 λ₊(σ²=1)
  - 正确噪声带上界 λ₊(σ²_idio=1-信号迹占比)
  - noise bulk 均值（σ²_idio 的候选估计）

输出：ASCII 直方图 + results/spectrum_hist_*.csv

用法：python qr_cov/stage_b_spectrum.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")


def mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def standardize(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return Z / sd


def ascii_hist(values: np.ndarray, bins: int, lo: float, hi: float,
               marks: dict[str, float]) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    maxc = max(counts.max(), 1)
    print(f"    直方图 [{lo:.2f}, {hi:.2f}]，{bins} 个 bin（'#' 高度按峰值归一）")
    print(f"    标记： " + "  ".join(f"{k}={v:.2f}" for k, v in marks.items()))
    for i in range(bins):
        bar = "#" * int(counts[i] / maxc * 56)
        mark = ""
        for m, v in marks.items():
            if edges[i] <= v < edges[i + 1]:
                mark = f"  <-- {m}"
        print(f"    {edges[i]:>6.2f} |{bar:<56}|{mark}")
    print()


def analyze_window(Rimp, universe, end_date, n=252, bins=40, lo=0.0, hi=6.0):
    X = Rimp[universe].loc[:end_date].iloc[-n:]
    Z = standardize(X.to_numpy(dtype=float))
    S = Z.T @ Z / Z.shape[0]
    ev = np.linalg.eigvalsh(S)[::-1]
    T = Z.shape[0]
    N = Z.shape[1]
    gamma = N / T
    _, hi_naive = mp_edges(1.0, gamma)
    n_sig = int((ev > hi_naive).sum())
    s_hat = ev[:n_sig].sum() / N
    sigma2_idio = 1.0 - s_hat
    _, hi_correct = mp_edges(sigma2_idio, gamma)
    bulk = ev[n_sig:]
    bulk_mean = bulk.mean()
    marks = {"naive λ₊": hi_naive, "correct λ₊": hi_correct, "bulk mean": bulk_mean}
    print(f"\n窗口 {X.index[0].date()} → {X.index[-1].date()}  (T={T}, N={N}, γ={gamma:.3f})")
    print(f"  λ₁={ev[0]:.1f}  信号方向数={n_sig}  信号占迹={s_hat*100:.1f}%")
    print(f"  σ²_idio={sigma2_idio:.3f}  bulk均值={bulk_mean:.3f}")
    ascii_hist(ev[ev <= hi], bins, lo, hi, marks)
    # 保存数据
    cnt, edges = np.histogram(ev, bins=bins, range=(lo, hi))
    df = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": cnt})
    df.to_csv(os.path.join(RESULT_DIR, f"spectrum_hist_{str(end_date)[:10]}.csv"),
              index=False)
    return ev, marks


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("真实 A 股相关矩阵特征值分布（4 个 252 天窗口）")
    print("=" * 78)
    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    universe = comp.head(500).index.tolist()

    for end in ["2016-06-30", "2021-12-31", "2024-09-30", "2026-09-30"]:
        analyze_window(Rimp, universe, end)

    print("=" * 78)
    print("读法：bulk 高峰在 0.5 附近（σ²_idio），naive λ₊=5.8 把弱信号误当噪声；")
    print("      correct λ₊≈3.2 更贴合真实噪声带。数据已存 results/spectrum_hist_*.csv")


if __name__ == "__main__":
    main()
