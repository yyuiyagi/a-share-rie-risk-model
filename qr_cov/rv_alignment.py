# -*- coding: utf-8 -*-
r"""
qr_cov/rv_alignment.py —— 方向 C：修好「风险对齐」这把尺子
==========================================================
病根：之前用「每 21 天一个 std」当实现波动，只有 51 个观测，采样噪声大，
      ρ≈0，分辨不出 sample 与 RIE 谁预测得更准。这把尺子不修，方向 A/B 的
      比较研究都建立在不可靠的度量上。

改进：
  1. 延长样本 —— 纯 min-var 不需要 α 训练集，OOS 从 2016（252 天后）就开始，
     把 51 期扩到 ~120 期，直接降低 ρ 估计的标准误。
  2. ρ 显著性 —— 报告 ρ 的标准误 SE 与 |ρ|/SE（t 值），而不是只看一个 ≈0 的点。

输出：短样本(2022 起) vs 长样本(2016 起) 的 bias/RMSE/ρ/t值，对比分辨力。

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/rv_alignment.py
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
    S_sample = base @ np.diag(ev) @ base.T
    S_rie = base @ np.diag(ev_rie) @ base.T
    return S_sample, S_rie


def spearman_rho(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def rho_se(rho, n):
    """Spearman ρ 的近似标准误（Fisher 变换）。"""
    rho = float(np.clip(rho, -0.999, 0.999))
    z = np.arctanh(rho)
    se_z = 1.0 / np.sqrt(n - 3)
    return se_z * (1 - rho ** 2)   # 反变换的近似


def align_summary(pred, real):
    n = len(pred)
    bias = float(np.mean(real) / np.mean(pred))
    rmse = float(np.sqrt(np.mean((np.asarray(real) - np.asarray(pred)) ** 2)))
    mae = float(np.mean(np.abs(np.asarray(real) - np.asarray(pred))))
    rho = spearman_rho(pred, real)
    se = rho_se(rho, n)
    t = abs(rho) / se if se > 0 else np.nan
    return bias, rmse, mae, rho, se, t


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

    print("=" * 86)
    print("方向 C：风险对齐尺子的分辨力（短样本 vs 长样本）")
    print("=" * 86)

    for label, start_bar in [("短样本(2022起)", 252 * 7), ("长样本(2016起)", 252)]:
        rebal = list(range(start_bar, T - HOLD, HOLD))
        rows = []
        for t in rebal:
            est = Xarr[t - WINDOW:t]
            S_sample, S_rie = build_covs(est, N)
            w_s = cap_weights(long_only_solve(S_sample, np.ones(N)), CAP)
            w_r = cap_weights(long_only_solve(S_rie, np.ones(N)), CAP)
            pred_s = float(np.sqrt(w_s @ S_sample @ w_s * 252))
            pred_r = float(np.sqrt(w_r @ S_rie @ w_r * 252))
            fut = Xarr[t:t + HOLD]
            rows.append((pred_s, pred_r,
                         np.std(fut @ w_s, ddof=1) * np.sqrt(252),
                         np.std(fut @ w_r, ddof=1) * np.sqrt(252)))
        d = pd.DataFrame(rows, columns=["pred_s", "pred_r", "real_s", "real_r"])

        print(f"\n--- {label}：{len(d)} 期 ---")
        print(f"{'Σ':<8}{'预测均值%':>10}{'实现均值%':>10}{'Bias':>8}{'RMSE%':>8}"
              f"{'MAE%':>8}{'ρ':>7}{'SE(ρ)':>7}{'|ρ|/SE':>8}")
        for name, p, r in [("sample", d.pred_s, d.real_s), ("RIE", d.pred_r, d.real_r)]:
            b, rm, ma, rho, se, t = align_summary(p, r)
            print(f"{name:<8}{p.mean()*100:>9.2f}%{r.mean()*100:>9.2f}%{b:>8.3f}"
                  f"{rm*100:>7.2f}%{ma*100:>7.2f}%{rho:>7.3f}{se:>7.3f}{t:>8.2f}")

    print("\n解读：|ρ|/SE > 2 表示 ρ 显著 ≠0（预测波动携带了对实现波动的信息）。")
    print("      若长样本下 |ρ|/SE 显著上升，说明之前 ρ≈0 主要是「样本太短 + 噪声稀释」，")
    print("      延长样本后这把尺子才真正有分辨力；若仍 ≈0，则说明月频对齐本身信号弱，")
    print("      应改用「实现波动/回撤/换手」（校验 2）做主判据。")


if __name__ == "__main__":
    main()
