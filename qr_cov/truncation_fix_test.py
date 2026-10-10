# -*- coding: utf-8 -*-
r"""
qr_cov/truncation_fix_test.py —— 方向 D 第二步：截断修正实验
=============================================================
诊断发现：A 股涨跌停「截断」把观测 σ 压低了 ≈26–28%（σ_obs/σ_true ≈ 0.72），
即方差 σ² 被低估约 48%。而我们的 RIE 用 bulk matching 估 σ² 来定「信号/噪声」
分界（λ± = σ²(1±√γ)²）。σ² 偏低 ⇒ λ+ 偏低 ⇒ 本属噪声的特征值被误判为信号
⇒ 去噪偏保守。

本实验：把 bulk σ² 乘一个修正因子（1.0=现状 / 1.5 / 1.93 / 2.5），看 min-var
的实现波动、回撤、换手是否随「更激进的去噪」而改善。

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/truncation_fix_test.py
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
START_BAR = 252 * 7
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


def max_drawdown(r):
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def ann_vol(r):
    return float(r.std(ddof=1) * np.sqrt(252))


def build_covs(X, N, trunc_corr=1.0):
    """返回 (RIE Σ, 修正后 bulk σ², 样本特征值)。trunc_corr 放大 σ² 修正截断低估。"""
    Z, mu, sd = standardize(X)
    C = Z.T @ Z / WINDOW
    ev, U = np.linalg.eigh(C)
    ev = ev[::-1]
    U = U[:, ::-1]
    gamma = N / WINDOW
    sigma2 = estimate_sigma2_bulk(ev, gamma) * trunc_corr
    ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
    base = np.diag(sd) @ U
    S_rie = base @ np.diag(ev_rie) @ base.T
    return S_rie, sigma2, ev


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

    configs = [("现状 σ²(×1.0)", 1.0), ("修正 ×1.5", 1.5),
               ("修正 ×1.93", 1.93), ("修正 ×2.5", 2.5)]

    print("=" * 86)
    print("方向D第二步：截断修正实验（bulk σ² × 修正因子 → min-var 表现）")
    print("=" * 86)

    for slabel, start_bar in [("短样本(2022起)", 252 * 7), ("长样本(2016起)", 252)]:
        rebal = list(range(start_bar, T - HOLD, HOLD))
        table = {}
        for label, tc in configs:
            daily = []
            turnovers = []
            sigmas2 = []
            n_spikes = []
            prev_w = None
            for t in rebal:
                est = Xarr[t - WINDOW:t]
                S_rie, sigma2, ev = build_covs(est, N, trunc_corr=tc)
                w = cap_weights(long_only_solve(S_rie, np.ones(N)), CAP)
                daily.append(Xarr[t:t + HOLD] @ w)
                if prev_w is not None:
                    turnovers.append(0.5 * float(np.abs(w - prev_w).sum()))
                prev_w = w
                sigmas2.append(sigma2)
                # spike 数：λ > λ+ = σ²(1+√γ)²
                gamma = N / WINDOW
                lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
                n_spikes.append(int((ev > lam_plus).sum()))
            r = np.concatenate(daily)
            table[label] = dict(vol=ann_vol(r), ret=float(r.mean() * 252),
                                mdd=max_drawdown(r), to=float(np.mean(turnovers) * 252 / HOLD),
                                sigma2=float(np.mean(sigmas2)),
                                nspike=float(np.mean(n_spikes)))

        print(f"\n--- {slabel}：{len(rebal)} 期（min-var，权重上限 {CAP*100:.0f}%）---")
        print(f"{'配置':<16}{'bulk σ²':>10}{'平均spike数':>12}{'实现波动%':>11}"
              f"{'年化收益%':>11}{'最大回撤%':>11}{'年换手':>9}{'夏普':>7}")
        for label, d in table.items():
            print(f"{label:<16}{d['sigma2']:>10.3f}{d['nspike']:>12.1f}{d['vol']*100:>10.2f}%"
                  f"{d['ret']*100:>10.2f}%{d['mdd']*100:>10.2f}%{d['to']:>9.1f}"
                  f"{d['ret']/d['vol']:>7.3f}")

    print("\n解读：σ² 放大 ⇒ λ+ 上移 ⇒ spike 变少（去噪更激进）。")
    print("      对比短/长样本下「修正 ×1.93 vs 现状」的差异是否一致，")
    print("      判断方向 D 的「截断修正无影响」结论是否稳健。")


if __name__ == "__main__":
    main()
