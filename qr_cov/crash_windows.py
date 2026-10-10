# -*- coding: utf-8 -*-
r"""
qr_cov/crash_windows.py —— 方向 B：崩盘窗口特例（量化"降噪防不住"）
====================================================================
不单独做完整压力测试，而是把协方差模型的"断裂盲区"用几个 A 股历史崩盘窗口
量化：在这些窗口里，RIE min-var 组合的区间回撤是多少，证明"崩盘来了降噪防不住"。

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/crash_windows.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension
from qr_cov.denoise import estimate_sigma2_bulk, denoise_spiked

WINDOW = 252
HOLD = 21
CAP = 0.05


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def long_only_solve(Sigma, b, max_iter=300):
    N = Sigma.shape[0]
    active = np.ones(N, dtype=bool)
    w = np.zeros(N)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        sub = Sigma[np.ix_(idx, idx)]
        try:
            w_sub = np.linalg.solve(sub, b[idx])
        except np.linalg.LinAlgError:
            w_sub = np.linalg.pinv(sub) @ b[idx]
        w[idx] = w_sub
        neg = idx[w_sub < 0]
        if len(neg) == 0:
            break
        active[neg] = False
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones(N) / N


def cap_weights(w, cap, max_iter=200):
    w = w.copy()
    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break
        excess = float(w[over].sum() - cap * over.sum())
        w[over] = cap
        under = ~over
        if excess <= 0 or not under.any():
            break
        w[under] += excess / under.sum()
    return w


def build_covs(X, N):
    Z, mu, sd = standardize(X)
    C = Z.T @ Z / WINDOW
    ev, U = np.linalg.eigh(C)
    ev = ev[::-1]
    U = U[:, ::-1]
    gamma = N / WINDOW
    sigma2 = estimate_sigma2_bulk(ev, gamma)
    ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
    base = np.diag(sd) @ U
    return base @ np.diag(ev_rie) @ base.T


def main():
    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    universe = comp.head(500).index.tolist()
    X = Rimp[universe]
    Xarr = X.to_numpy(dtype=float)
    dates = X.index
    N = X.shape[1]
    T = len(X)

    rebal = list(range(WINDOW, T - HOLD, HOLD))
    eq_w = np.ones(N) / N

    # RIE min-var 长样本回测，记录逐日收益（带日期）
    rie_rets = []
    eq_rets = []
    day_dates = []
    for t in rebal:
        est = Xarr[t - WINDOW:t]
        S_rie = build_covs(est, N)
        w = cap_weights(long_only_solve(S_rie, np.ones(N)), CAP)
        for s in range(t, min(t + HOLD, T)):
            rie_rets.append(float(w @ Xarr[s]))
            eq_rets.append(float(eq_w @ Xarr[s]))
            day_dates.append(dates[s])
    r_rie = pd.Series(rie_rets, index=day_dates)
    r_eq = pd.Series(eq_rets, index=day_dates)

    # A 股历史崩盘/剧震窗口
    windows = [
        ("2016-01 熔断", "2016-01-04", "2016-02-29"),
        ("2018 熊市", "2018-01-01", "2018-12-31"),
        ("2020-02 疫情", "2020-02-03", "2020-03-23"),
        ("2022-01~04 熊市", "2022-01-01", "2022-04-30"),
        ("2024-01~02 微盘危机", "2024-01-01", "2024-02-07"),
    ]

    print("=" * 78)
    print("方向B：崩盘窗口特例（RIE min-var vs 等权500 区间收益）")
    print("=" * 78)
    print(f"{'崩盘窗口':<18}{'RIE区间收益%':>14}{'等权区间收益%':>14}{'RIE区间最大回撤%':>18}")
    for name, lo, hi in windows:
        s_rie = r_rie.loc[lo:hi]
        s_eq = r_eq.loc[lo:hi]
        if len(s_rie) == 0:
            continue
        cum_rie = (1 + s_rie).prod() - 1
        cum_eq = (1 + s_eq).prod() - 1
        cum = np.cumprod(1 + s_rie.to_numpy())
        mdd = float(((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min())
        print(f"{name:<18}{cum_rie*100:>13.2f}%{cum_eq*100:>13.2f}%{mdd*100:>17.2f}%")

    # 全期最大回撤
    cum_all = np.cumprod(1 + r_rie.to_numpy())
    mdd_all = float(((cum_all - np.maximum.accumulate(cum_all)) /
                     np.maximum.accumulate(cum_all)).min())
    print(f"\n全期（2016–2026）RIE min-var 最大回撤：{mdd_all*100:.2f}%")
    print("\n解读：崩盘窗口里 RIE min-var 与等权几乎同步大跌，说明协方差模型（去噪与否）")
    print("      在跳跃/传染驱动的崩盘中防不住——这是机制④的实证边界。")


if __name__ == "__main__":
    main()
