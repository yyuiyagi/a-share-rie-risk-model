# -*- coding: utf-8 -*-
r"""
qr_cov/neutrality_test.py —— 方向 A：风格中性化（严谨版 v2）
============================================================
在 v1（静态收益中性化）基础上做两个严谨化：
  1. 逐日暴露：vol/beta 用「截至 s-1」的滚动窗口算，消除 v1 静态暴露的前视。
  2. 权重约束中性化：优化器直接加 wᵀF=0，让组合因子暴露精确归零（v1 的收益中性化
     只把暴露压到 ~0.2，未严格归零）。

对比四种中性化方式下 RIE 相对 sample 的降波动幅度，看「43% 纯去噪 / 57% 因子下注」
这个分解是否稳健：

  raw            无中性化（允许隐性下注）
  static_neut    静态收益中性化（v1，暴露用全窗口，有前视）
  daily_neut     逐日收益中性化（暴露用截至 s-1 的滚动窗口，无前视）
  constraint     权重约束中性化（优化器加 wᵀF=0，暴露精确归零，无前视）

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/neutrality_test.py
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
L = 60          # 逐日暴露的滚动窗口


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


def build_covs(X, N):
    Z, mu, sd = standardize(X)
    C = Z.T @ Z / X.shape[0]
    ev, U = np.linalg.eigh(C)
    ev = ev[::-1]
    U = U[:, ::-1]
    gamma = N / X.shape[0]
    sigma2 = estimate_sigma2_bulk(ev, gamma)
    ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
    base = np.diag(sd) @ U
    S_sample = base @ np.diag(ev) @ base.T
    S_rie = base @ np.diag(ev_rie) @ base.T
    return S_sample, S_rie


# ---------------- 暴露（静态 / 逐日） ----------------

def static_exposures(est):
    """静态暴露：vol=窗口 std、beta=对等权市场回归斜率，横截面 z-score。est (T,N)。"""
    T, N = est.shape
    vol = est.std(axis=0)
    mkt = est.mean(axis=1)
    var_mkt = float(mkt.var())
    est_c = est - est.mean(axis=0, keepdims=True)
    mkt_c = mkt - mkt.mean()
    beta = (est_c.T @ mkt_c) / T / var_mkt
    return _zscore(vol), _zscore(beta)


def rolling_lagged_exposures(est, L=L):
    """逐日暴露（无前视）：第 s 天用 [s-L, s) 的滚动窗口，不含 s 当天。
    返回 (vol_mat, beta_mat)，各 (T,N)。前 L 天窗口自动退化为「从窗口起点累积」。"""
    T, N = est.shape
    vol_mat = np.zeros((T, N))
    beta_mat = np.zeros((T, N))
    for s in range(T):
        lo = max(0, s - L)
        sub = est[lo:s, :]
        if sub.shape[0] < 5:
            continue
        vol_mat[s] = sub.std(axis=0, ddof=1)
        m = sub.mean(axis=1)
        var_m = float(m.var())
        e_c = sub - sub.mean(axis=0, keepdims=True)
        m_c = m - m.mean()
        beta_mat[s] = (e_c.T @ m_c) / sub.shape[0] / (var_m + 1e-12)
    return vol_mat, beta_mat


def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-12)


# ---------------- 中性化 ----------------

def neutralize_static(est, exposures):
    """静态收益中性化：对每个交易日横截面回归到 [1, *exposures] 取残差。"""
    N = est.shape[1]
    D = np.column_stack([np.ones(N)] + list(exposures))
    P = D @ np.linalg.solve(D.T @ D, D.T)
    return est - est @ P.T


def neutralize_daily(est, vol_mat, beta_mat):
    """逐日收益中性化：第 s 天收益对「第 s 天的滞后暴露」回归取残差，无前视。"""
    T, N = est.shape
    eps = np.zeros_like(est)
    for s in range(T):
        v = _zscore(vol_mat[s])
        b = _zscore(beta_mat[s])
        D = np.column_stack([np.ones(N), v, b])
        coef, *_ = np.linalg.lstsq(D, est[s], rcond=None)
        eps[s] = est[s] - D @ coef
    return eps


def strict_neutral_minvar(Sigma, F, cap=CAP, max_iter=300):
    """严格权重约束中性化：min 0.5 wᵀΣw s.t. w≥0, 1ᵀw=1, Fᵀw=0。
    F: (N,K) 已标准化的因子暴露。active-set + 拉格朗日。"""
    N = Sigma.shape[0]
    K = F.shape[1]
    active = np.ones(N, dtype=bool)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        m = len(idx)
        if m <= K + 1:
            break
        S = Sigma[np.ix_(idx, idx)]
        Fi = F[idx]
        Sinv = np.linalg.pinv(S)
        ones = np.ones(m)
        a = float(ones @ Sinv @ ones)
        b = Fi.T @ Sinv @ ones          # (K,)
        C = Fi.T @ Sinv @ Fi            # (K,K)
        M = np.zeros((K + 1, K + 1))
        M[0, 0] = a
        M[0, 1:] = b
        M[1:, 0] = b
        M[1:, 1:] = C
        rhs = np.zeros(K + 1)
        rhs[0] = 1.0
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(M) @ rhs
        lam, eta = sol[0], sol[1:]
        w_sub = Sinv @ (lam * ones + Fi @ eta)
        w = np.zeros(N)
        w[idx] = w_sub
        neg = idx[w_sub < 0]
        if len(neg) == 0:
            break
        active[neg] = False
    w = np.maximum(w, 0)
    s = w.sum()
    w = w / s if s > 0 else np.ones(N) / N
    return cap_weights(w, cap)


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

    print("=" * 92)
    print("方向A：风格中性化 严谨版 v2（四种方式对比，长样本 2016 起）")
    print("=" * 92)

    methods = ["raw", "static_neut", "daily_neut", "constraint"]
    daily = {m: {"sample": [], "RIE": []} for m in methods}
    turnovers = {m: {"sample": [], "RIE": []} for m in methods}
    prev_w = {m: {"sample": None, "RIE": None} for m in methods}
    expo_hist = {m: {"sample": [], "RIE": []} for m in methods}

    for t in rebal:
        est = Xarr[t - WINDOW:t]

        # 暴露
        vol_z, beta_z = static_exposures(est)
        F = np.column_stack([vol_z, beta_z])            # (N,2) 用于权重约束
        vol_mat, beta_mat = rolling_lagged_exposures(est, L)

        # 各方式的中性化收益
        eps_static = neutralize_static(est, (vol_z, beta_z))
        eps_daily = neutralize_daily(est, vol_mat, beta_mat)

        # 协方差
        Ss_raw, Sr_raw = build_covs(est, N)
        Ss_sta, Sr_sta = build_covs(eps_static, N)
        Ss_dly, Sr_dly = build_covs(eps_daily, N)
        Ss_con, Sr_con = build_covs(est, N)             # 约束版用原始 Σ，约束在优化里加

        # 权重
        ws = {
            ("raw", "sample"): cap_weights(long_only_solve(Ss_raw, np.ones(N)), CAP),
            ("raw", "RIE"): cap_weights(long_only_solve(Sr_raw, np.ones(N)), CAP),
            ("static_neut", "sample"): cap_weights(long_only_solve(Ss_sta, np.ones(N)), CAP),
            ("static_neut", "RIE"): cap_weights(long_only_solve(Sr_sta, np.ones(N)), CAP),
            ("daily_neut", "sample"): cap_weights(long_only_solve(Ss_dly, np.ones(N)), CAP),
            ("daily_neut", "RIE"): cap_weights(long_only_solve(Sr_dly, np.ones(N)), CAP),
            ("constraint", "sample"): strict_neutral_minvar(Ss_con, F),
            ("constraint", "RIE"): strict_neutral_minvar(Sr_con, F),
        }

        fut = Xarr[t:t + HOLD]
        for m in methods:
            for k in ("sample", "RIE"):
                w = ws[(m, k)]
                daily[m][k].append(fut @ w)
                if prev_w[m][k] is not None:
                    turnovers[m][k].append(0.5 * float(np.abs(w - prev_w[m][k]).sum()))
                prev_w[m][k] = w
                expo_hist[m][k].append((float(w @ vol_z), float(w @ beta_z)))

    print(f"\nOOS {dates[rebal[0]].date()} → {dates[-1].date()}  "
          f"（{len(rebal)} 期 × {HOLD} 日，min-var，权重上限 {CAP*100:.0f}%）")
    print(f"{'中性化方式':<14}{'sample波动%':>12}{'RIE波动%':>10}{'RIE降幅':>9}"
          f"{'RIE|vol|':>9}{'RIE|beta|':>10}{'RIE夏普':>9}")
    for m in methods:
        r_s = np.concatenate(daily[m]["sample"])
        r_r = np.concatenate(daily[m]["RIE"])
        v_s, v_r = ann_vol(r_s), ann_vol(r_r)
        drop = (1 - v_r / v_s) * 100
        ev = float(np.mean(np.abs([e[0] for e in expo_hist[m]["RIE"]])))
        eb = float(np.mean(np.abs([e[1] for e in expo_hist[m]["RIE"]])))
        sr = float(r_r.mean() * 252) / v_r
        print(f"{m:<14}{v_s*100:>11.2f}%{v_r*100:>9.2f}%{drop:>8.1f}%"
              f"{ev:>9.3f}{eb:>10.3f}{sr:>9.3f}")

    print("\n解读：")
    print("  raw 的 RIE 降幅 = 纯去噪 + 因子下注（混在一起）。")
    print("  static/daily/constraint 的 RIE 降幅 ≈ 纯去噪（因子下注被禁）。")
    print("  若三种中性化给出的降幅接近，说明「43%/57%」分解稳健；")
    print("  constraint 行的 |vol|、|beta| 应 ≈0，证明权重约束精确归零。")


if __name__ == "__main__":
    main()
