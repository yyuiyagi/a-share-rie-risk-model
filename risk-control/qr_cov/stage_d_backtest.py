# -*- coding: utf-8 -*-
r"""
qr_cov/stage_d_backtest.py —— 滚动样本外回测
============================================
对比 4 个策略在真实 A 股上的样本外波动率：
  equal   等权
  sample  样本协方差求逆（pinv，奇异）
  clip    MP 特征值裁剪（σ² 由 bulk matching 估）
  rie     RIE 尖峰去噪（σ² 由 bulk matching 估）

协议：
  - 估计窗 T=252，再平衡每 21 天
  - 权重用 ≤t 的数据，收益用 t+1..t+21 的真实收益（无前视）
  - 相关矩阵清洗后转回协方差 Σ = D C D

用法：python qr_cov/stage_d_backtest.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension
from qr_cov.denoise import estimate_sigma2_bulk, denoise_spiked, mp_edges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def minvar(Cov: np.ndarray) -> np.ndarray:
    try:
        w = np.linalg.solve(Cov, np.ones(Cov.shape[0]))
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(Cov) @ np.ones(Cov.shape[0])
    return w / w.sum()


def long_only_minvar(Cov: np.ndarray, max_iter: int = 300) -> np.ndarray:
    """long-only 最小方差（主动集消去法，逐次去掉负权重资产再解）。"""
    N = Cov.shape[0]
    active = np.ones(N, dtype=bool)
    w = np.zeros(N)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        sub = Cov[np.ix_(idx, idx)]
        try:
            w_sub = np.linalg.solve(sub, np.ones(len(idx)))
        except np.linalg.LinAlgError:
            w_sub = np.linalg.pinv(sub) @ np.ones(len(idx))
        w[idx] = w_sub
        neg = idx[w_sub < 0]
        if len(neg) == 0:
            break
        active[neg] = False
    w = np.maximum(w, 0)
    return w / w.sum()


def clip_corr(ev: np.ndarray, U: np.ndarray, sigma2: float, gamma: float) -> np.ndarray:
    """MP 裁剪：保留 spike，bulk 压成常数（保迹）。"""
    _, lam_plus = mp_edges(sigma2, gamma)
    spikes = ev > lam_plus
    evc = ev.copy()
    n_bulk = int((~spikes).sum())
    if n_bulk > 0:
        evc[~spikes] = (ev.sum() - ev[spikes].sum()) / n_bulk
    return U @ np.diag(evc) @ U.T


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("滚动样本外回测（4 策略）")
    print("=" * 78)

    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    universe = comp.head(500).index.tolist()
    X = Rimp[universe]
    Xarr = X.to_numpy(dtype=float)
    dates = X.index
    N = X.shape[1]
    T = len(X)
    gamma = N / 252.0

    methods = ["equal", "sample", "clip", "rie"]
    rp = {m: [] for m in methods}
    neff_hist = {m: [] for m in methods}
    sigma2_hist = []

    rebal = list(range(252, T - 21, 21))
    print(f"宇宙 {N} 只，估计窗 252 天，再平衡 {len(rebal)} 次，"
          f"OOS {dates[252].date()} → {dates[-1].date()}")

    for t in rebal:
        est = Xarr[t - 252:t]
        Z, mu, sd = standardize(est)
        C = Z.T @ Z / 252.0
        ev, U = np.linalg.eigh(C)
        ev = ev[::-1]
        U = U[:, ::-1]
        D = np.diag(sd)

        sigma2 = estimate_sigma2_bulk(ev, gamma)
        sigma2_hist.append(sigma2)

        # 各方法协方差
        Covs = {}
        Covs["equal"] = None
        Covs["sample"] = D @ C @ D
        C_clip = clip_corr(ev, U, sigma2, gamma)
        Covs["clip"] = D @ C_clip @ D
        ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
        C_rie = U @ np.diag(ev_rie) @ U.T
        Covs["rie"] = D @ C_rie @ D

        # 权重（long-only）
        ws = {}
        ws["equal"] = np.ones(N) / N
        for m in ["sample", "clip", "rie"]:
            ws[m] = long_only_minvar(Covs[m])
        for m in methods:
            neff_hist[m].append(1.0 / np.sum(ws[m] ** 2))

        # 持有 21 天
        for s in range(t, min(t + 21, T)):
            r = Xarr[s]
            for m in methods:
                rp[m].append(float(ws[m] @ r))

    print(f"\n{'方法':<10} {'年化波动%':>10} {'年化收益%':>10} {'夏普':>8} "
          f"{'有效持仓':>9}")
    for m in methods:
        r = np.array(rp[m])
        vol = r.std() * np.sqrt(252) * 100
        ret = r.mean() * 252 * 100
        shp = r.mean() / r.std() * np.sqrt(252)
        neff = np.mean(neff_hist[m])
        print(f"{m:<10} {vol:>10.2f} {ret:>10.2f} {shp:>8.3f} {neff:>9.1f}")

    # σ² 的时变
    sigma2_arr = np.array(sigma2_hist)
    print(f"\nσ²(bulk matching)：mean={sigma2_arr.mean():.3f} "
          f"min={sigma2_arr.min():.3f} max={sigma2_arr.max():.3f}")

    # 保存
    n = len(rp["equal"])
    out = pd.DataFrame({m: rp[m] for m in methods}, index=dates[252:252 + n])
    out.to_csv(os.path.join(RESULT_DIR, "stage_d_backtest_returns.csv"))
    print("\n已保存 results/stage_d_backtest_returns.csv")


if __name__ == "__main__":
    main()
