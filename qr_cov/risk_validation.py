# -*- coding: utf-8 -*-
r"""
qr_cov/risk_validation.py —— 风险模型的「正确」验收（不再是 Sharpe）
====================================================================
背景：之前一直用「RIE 是否跑赢 naive 的 Sharpe」验收风控，这是用错了度量衡。
行业（Axioma / Barra / Pantaleo-Tumminello-Lillo-Mantegna 2011）验证风险模型看的是：

  (1) 实现风险对齐  —— 每期「预测波动」 vs 未来窗口「实现波动」：
        bias B = mean(realized)/mean(predicted)  （应≈1，无偏）
        RMSE / MAE / Spearman ρ                  （越低/越高越好）
  (2) 纯 min-var 压制 —— 去掉 α，只跑 w ∝ Σ⁻¹1：
        实现波动率、换手、最大回撤                （去噪后应更低/更稳）

这里唯一变量是「是否做谱去噪」：sample Σ 与 RIE Σ 用同一个 U、同一个 sd，
只把样本特征值 ev 换成 RIE 去噪特征值 ev_rie。所以差异 100% 来自去噪本身。

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/risk_validation.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")

CAP = 0.05          # 单票权重上限
WINDOW = 252        # 协方差估计窗（日）
HOLD = 21           # 重平衡周期（日），与 20 日 α 对齐
START_BAR = 252 * 7  # 第一个 OOS 重平衡点（≈2022 年初）


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def long_only_solve(Sigma, b, max_iter=300):
    """非负二次优化 w = argmin_{w≥0,Σw=1} 0.5 wᵀΣw - bᵀw 的 active-set 近似。"""
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
    """返回 (sample Σ, RIE Σ)。两者同 U、同 sd，唯一差异是特征值去噪。"""
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


def max_drawdown(r):
    cum = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cum)
    return float(((cum - peak) / peak).min())


def ann_vol(r):
    return float(r.std(ddof=1) * np.sqrt(252))


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 80)
    print("风险模型验收：实现风险对齐 + 纯 min-var 压制（sample vs RIE）")
    print("=" * 80)

    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    universe = comp.head(500).index.tolist()
    X = Rimp[universe]
    Xarr = X.to_numpy(dtype=float)
    dates = X.index
    N = X.shape[1]
    T = len(X)

    rebal = list(range(START_BAR, T - HOLD, HOLD))
    eq_w = np.ones(N) / N

    # ---- 逐期记录：预测波动 vs 实现波动 ----
    rows = []
    for t in rebal:
        dt = dates[t]
        est = Xarr[t - WINDOW:t]
        S_sample, S_rie = build_covs(est, N)

        w_s = cap_weights(long_only_solve(S_sample, np.ones(N)), CAP)
        w_r = cap_weights(long_only_solve(S_rie, np.ones(N)), CAP)

        pred_s = float(np.sqrt(w_s @ S_sample @ w_s * 252))
        pred_r = float(np.sqrt(w_r @ S_rie @ w_r * 252))

        fut = Xarr[t:t + HOLD]
        r_s = fut @ w_s
        r_r = fut @ w_r
        r_eq = fut @ eq_w
        pred_eq_s = float(np.sqrt(eq_w @ S_sample @ eq_w * 252))
        pred_eq_r = float(np.sqrt(eq_w @ S_rie @ eq_w * 252))
        rows.append(dict(dt=dt, pred_s=pred_s, pred_r=pred_r,
                         pred_eq_s=pred_eq_s, pred_eq_r=pred_eq_r,
                         r_s=r_s, r_r=r_r, r_eq=r_eq,
                         w_s=w_s, w_r=w_r))
    df = pd.DataFrame(rows)
    df["real_s"] = df["r_s"].apply(ann_vol)
    df["real_r"] = df["r_r"].apply(ann_vol)
    df["real_eq"] = df["r_eq"].apply(ann_vol)

    # ---- 校验 1：实现风险对齐 ----
    def align_stats(pred, real):
        bias = real.mean() / pred.mean()
        rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
        mae = float(np.mean(np.abs(real - pred)))
        rho = float(pd.Series(real).corr(pd.Series(pred), method="spearman"))
        return bias, rmse, mae, rho

    print(f"\nOOS {df['dt'].iloc[0].date()} → {df['dt'].iloc[-1].date()}  "
          f"（{len(df)} 期 × {HOLD} 日，min-var，权重上限 {CAP*100:.0f}%）")
    print("\n【校验 1a】固定等权组合的风险对齐（唯一变量是 Σ，最纯粹的评分）")
    print(f"{'Σ':<8} {'预测均值%':>10} {'实现均值%':>10} {'Bias(实/预)':>12} "
          f"{'RMSE%':>8} {'MAE%':>8} {'ρ':>7}")
    for name, p, r in [("sample", df.pred_eq_s, df.real_eq),
                       ("RIE", df.pred_eq_r, df.real_eq)]:
        b, rm, ma, rho = align_stats(p, r)
        print(f"{name:<8} {p.mean()*100:>10.2f} {r.mean()*100:>10.2f} {b:>12.3f} "
              f"{rm*100:>8.2f} {ma*100:>8.2f} {rho:>7.3f}")

    print("\n【校验 1b】min-var 组合的风险对齐（预测=优化后的最小波动，天然偏乐观）")
    print(f"{'Σ':<8} {'预测均值%':>10} {'实现均值%':>10} {'Bias(实/预)':>12} "
          f"{'RMSE%':>8} {'MAE%':>8} {'ρ':>7}")
    for name, p, r in [("sample", df.pred_s, df.real_s),
                       ("RIE", df.pred_r, df.real_r)]:
        b, rm, ma, rho = align_stats(p, r)
        print(f"{name:<8} {p.mean()*100:>10.2f} {r.mean()*100:>10.2f} {b:>12.3f} "
              f"{rm*100:>8.2f} {ma*100:>8.2f} {rho:>7.3f}")

    # 压力段（2022 熊市 / 2024 微盘危机 / 其余）
    def seg(lo, hi):
        m = (df.dt >= lo) & (df.dt < hi)
        return df[m]

    print("\n分压力段 Bias（实/预，应≈1；sample 越 >1 表示越低估风险）")
    segs = [("2022 熊市", "2022-01-01", "2023-01-01"),
            ("2024 微盘危机", "2024-01-01", "2024-03-01"),
            ("其余", "2023-01-01", "2026-10-01")]
    print(f"{'压力段':<14} {'sample Bias':>12} {'RIE Bias':>12}")
    for name, lo, hi in segs:
        s = seg(lo, hi)
        if len(s) == 0:
            continue
        bs = s.real_s.mean() / s.pred_s.mean()
        br = s.real_r.mean() / s.pred_r.mean()
        print(f"{name:<14} {bs:>12.3f} {br:>12.3f}")

    # ---- 校验 2：纯 min-var 压制（实现波动 / 换手 / 回撤）----
    def full_stats(rs, ws):
        r = np.concatenate(rs)
        vol = ann_vol(r)
        ret = float(r.mean() * 252)
        mdd = max_drawdown(r)
        to = []
        for i in range(1, len(ws)):
            to.append(0.5 * float(np.abs(ws[i] - ws[i - 1]).sum()))
        ann_to = float(np.mean(to) * 252 / HOLD)
        return vol, ret, mdd, ann_to

    r_eq = np.concatenate([Xarr[t:t + HOLD] @ eq_w for t in rebal])

    print("\n【校验 2】纯 min-var 压制（无 α，仅 Σ⁻¹1）")
    print(f"{'组合':<10} {'实现波动%':>10} {'年化收益%':>10} {'最大回撤%':>10} "
          f"{'年化换手':>10} {'夏普':>7}")
    vs, rs, md_s, to_s = full_stats([d["r_s"] for _, d in df.iterrows()],
                                    [d["w_s"] for _, d in df.iterrows()])
    vr, rr, md_r, to_r = full_stats([d["r_r"] for _, d in df.iterrows()],
                                    [d["w_r"] for _, d in df.iterrows()])
    veq, req, mdeq = ann_vol(r_eq), float(r_eq.mean() * 252), max_drawdown(r_eq)
    print(f"{'等权500':<10} {veq*100:>10.2f} {req*100:>10.2f} {mdeq*100:>10.2f} "
          f"{'—':>10} {req/veq:>7.3f}")
    print(f"{'sample':<10} {vs*100:>10.2f} {rs*100:>10.2f} {md_s*100:>10.2f} "
          f"{to_s:>10.1f} {rs/vs:>7.3f}")
    print(f"{'RIE':<10} {vr*100:>10.2f} {rr*100:>10.2f} {md_r*100:>10.2f} "
          f"{to_r:>10.1f} {rr/vr:>7.3f}")
    print(f"\nRIE vs sample：实现波动 {vs*100:.2f}%→{vr*100:.2f}%  "
          f"({(vr/vs-1)*100:+.1f}%)，最大回撤 {md_s*100:.2f}%→{md_r*100:.2f}%，"
          f"年化换手 {to_s:.1f}→{to_r:.1f}")


if __name__ == "__main__":
    main()
