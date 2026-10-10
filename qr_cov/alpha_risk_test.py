# -*- coding: utf-8 -*-
r"""
qr_cov/alpha_risk_test.py —— 高收益 α + 风控（成本感知版 v3）
==============================================================
在 v2 有效前沿基础上加：
  - 换手率记录
  - 单边交易成本 10bp（cost_per_side=0.001）
  - 权重上限 5%（clip-and-renormalize 启发式）
  - 同时报告 gross 与 net

关键：成本对 naive 和 controlled 都扣，所以看「相对优势」是否存活，
而不是绝对 Sharpe。

用法：python qr_cov/alpha_risk_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension
from qr_cov.factors import load_aux, compute_factor_raw, zscore_cs
from qr_cov.denoise import estimate_sigma2_bulk, denoise_spiked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")

COST_PER_SIDE = 0.001   # 10bp 单边
CAP = 0.05              # 单票权重上限 5%
TOP_K = 50


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
    """clip-and-renormalize 权重上限（启发式，非精确 QP）。"""
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


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("高收益 α + 风控（成本感知 v3）")
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

    aux = load_aux(universe, X.index)
    raw = compute_factor_raw(X, aux)
    z = zscore_cs(raw)
    feat_names = list(z.keys())

    F = pd.concat([z[n].stack().rename(n) for n in feat_names], axis=1)
    cum = X.cumsum()
    F = F.join((cum.shift(-20) - cum).stack().rename("label"))
    F = F.dropna(subset=["label"])

    train_mask = F.index.get_level_values(0) < "2022-01-01"
    test_mask = F.index.get_level_values(0) >= "2022-01-01"
    import lightgbm as lgb
    model = lgb.LGBMRegressor(n_estimators=150, num_leaves=63, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=7,
                              n_jobs=-1, verbose=-1)
    model.fit(F.loc[train_mask, feat_names].to_numpy(),
              F.loc[train_mask, "label"].to_numpy())
    Fte = F.loc[test_mask].copy()
    Fte["pred"] = model.predict(Fte[feat_names].to_numpy())
    ic = Fte.groupby(level=0).apply(
        lambda g: g["pred"].corr(g["label"], method="spearman"))
    print(f"RankIC：均值 {ic.mean():.4f}，IC>0 {(ic>0).mean()*100:.1f}%")

    # ---- OOS 回测（带成本/换手/权重上限）----
    rebal = list(range(252 * 7, T - 21, 21))
    res = {"naive": {"gross": [], "net": []},
           "tangency": {"gross": [], "net": []},
           "minvar": {"gross": [], "net": []}}
    to_hist = {"naive": [], "tangency": [], "minvar": []}
    prev_w = {"naive": None, "tangency": None, "minvar": None}

    for t in rebal:
        dt = dates[t]
        g = Fte.loc[dt] if dt in Fte.index.get_level_values(0) else None
        if g is None or len(g) < TOP_K:
            continue
        est = Xarr[t - 252:t]
        Z, mu, sd = standardize(est)
        C = Z.T @ Z / 252.0
        ev, U = np.linalg.eigh(C)
        ev = ev[::-1]; U = U[:, ::-1]
        gamma = N / 252.0
        sigma2 = estimate_sigma2_bulk(ev, gamma)
        ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
        Sigma_rie = np.diag(sd) @ (U @ np.diag(ev_rie) @ U.T) @ np.diag(sd)

        mu_t = g["pred"].reindex(universe).fillna(0).to_numpy()

        w_naive = np.zeros(N)
        top = g["pred"].reindex(universe).fillna(-np.inf).argsort()[::-1][:TOP_K]
        w_naive[top] = 1.0 / TOP_K
        w_tang = cap_weights(long_only_solve(Sigma_rie, mu_t), CAP)
        w_minv = cap_weights(long_only_solve(Sigma_rie, np.ones(N)), CAP)

        ws = {"naive": w_naive, "tangency": w_tang, "minvar": w_minv}
        for m in ws:
            if prev_w[m] is None:
                turnover = 1.0
            else:
                turnover = 0.5 * np.sum(np.abs(ws[m] - prev_w[m]))
            to_hist[m].append(turnover)
            cost = 2.0 * turnover * COST_PER_SIDE   # 双边成本
            for s in range(t, min(t + 21, T)):
                gr = float(ws[m] @ Xarr[s])
                res[m]["gross"].append(gr)
                res[m]["net"].append(gr - (cost if s == t else 0.0))
            prev_w[m] = ws[m]

    def stat(r):
        return (r.std() * np.sqrt(252) * 100, r.mean() * 252 * 100,
                r.mean() / r.std() * np.sqrt(252))

    print(f"\nOOS {dates[rebal[0]].date()} → {dates[-1].date()}  "
          f"(成本 {COST_PER_SIDE*1e4:.0f}bp 单边，权重上限 {CAP*100:.0f}%)")
    print(f"{'组合':<14} {'波动%':>8} {'毛收益%':>9} {'净收益%':>9} "
          f"{'毛夏普':>7} {'净夏普':>7} {'年换手':>8}")
    for m in ["naive", "minvar", "tangency"]:
        g = np.array(res[m]["gross"]); n = np.array(res[m]["net"])
        vg, rg, sg = stat(g); vn, rn, sn = stat(n)
        to = np.mean(to_hist[m])
        print(f"{m:<14} {vn:>8.2f} {rg:>9.2f} {rn:>9.2f} {sg:>7.3f} {sn:>7.3f} "
              f"{to*252/21:>8.1f}")

    # 净收益有效前沿
    print("\n净收益有效前沿（min-var 与切点混合）")
    print(f"{'α(切点权重)':>12} {'波动%':>8} {'净收益%':>9} {'净夏普':>7}")
    r_min_net = np.array(res["minvar"]["net"])
    r_tang_net = np.array(res["tangency"]["net"])
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = alpha * r_tang_net + (1 - alpha) * r_min_net
        v, rt, s = stat(r)
        print(f"{alpha:>12.2f} {v:>8.2f} {rt:>9.2f} {s:>7.3f}")

    # 与 naive 净波动相同的前沿点
    r_naive_net = np.array(res["naive"]["net"])
    v_naive = stat(r_naive_net)[0]
    best = None
    for alpha in np.linspace(0, 1, 201):
        r = alpha * r_tang_net + (1 - alpha) * r_min_net
        v, rt, s = stat(r)
        if best is None or abs(v - v_naive) < abs(best[0] - v_naive):
            best = (v, rt, s, alpha)
    print(f"\n与 naive 同净波动({v_naive:.1f}%)的前沿点：")
    print(f"  波动 {best[0]:.2f}%  净收益 {best[1]:.2f}%  净夏普 {best[2]:.3f}  "
          f"(α={best[3]:.2f})")
    print(f"  vs naive：净收益 {stat(r_naive_net)[1]:.2f}% / "
          f"净夏普 {stat(r_naive_net)[2]:.3f}")


if __name__ == "__main__":
    main()
