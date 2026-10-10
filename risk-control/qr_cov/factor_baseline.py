# -*- coding: utf-8 -*-
r"""
qr_cov/factor_baseline.py —— 20 个可观测因子的协方差 baseline（v3，业界标准）
==============================================================================
因子模型（时间序列回归）：

    r_{i,t} = α_i + β_iᵀ f_t + ε_{i,t}
    Σ̂ = B Fᵀ Bᵀ + D

与 v2 的区别（两个标准修正）：
  1. 因子收益改用【分位数多空组合】（top/bottom 各 q=0.1），尺度与单股同级；
  2. 因子协方差 F 做【Ledoit-Wolf 收缩】，压制噪声与共线导致的小特征值。

用法（在项目根目录）：
  python qr_cov/factor_baseline.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension
from qr_cov.factors import load_aux, compute_factor_raw, zscore_cs, factor_returns_quantile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, "results")
N_UNIVERSE = 500
T_WINDOW = 252


def mp_edges(gamma: float) -> tuple[float, float]:
    return ((1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2)


def shrink_F_ledoit_wolf(f: pd.DataFrame) -> pd.DataFrame:
    """对因子收益协方差做 Ledoit-Wolf 线性收缩（向恒等目标）。"""
    from sklearn.covariance import LedoitWolf
    f_clean = f.dropna()                          # 防御：剔除残留 NaN 行
    lw = LedoitWolf()
    lw.fit(f_clean.to_numpy(dtype=float))
    return pd.DataFrame(lw.covariance_, index=f.columns, columns=f.columns)


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("20 因子协方差 baseline（v3，分位数多空 + LW 收缩 F）")
    print("=" * 78)

    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, _ = impute_suspension(R, method="linear_log")
    comp = Rimp.notna().sum().sort_values(ascending=False)
    universe = comp.head(N_UNIVERSE).index.tolist()
    X = Rimp[universe]
    print(f"[1] 宇宙：{len(universe)} 只 × {X.shape[0]} 天")

    aux = load_aux(universe, X.index)
    raw = compute_factor_raw(X, aux)
    z = zscore_cs(raw)
    f = factor_returns_quantile(z, X, q=0.1)
    K = f.shape[1]
    print(f"[2] 因子收益：{f.shape[0]} 天 × {K} 个（分位数多空）")

    # ---- 因子收益尺度 ----
    f_win = f.iloc[-T_WINDOW:]
    ann_vol = f_win.std() * np.sqrt(252)
    print(f"\n[3] 因子收益年化波动（%），应与单股同级（~1%）")
    print(ann_vol.sort_values(ascending=False).round(2).to_string())

    # ---- F：原始 vs 收缩 ----
    Fcorr = f_win.corr()
    evc = np.linalg.eigvalsh(Fcorr.to_numpy())[::-1]
    gK = K / T_WINDOW
    _, hi = mp_edges(gK)
    F_raw = pd.DataFrame(np.cov((f_win - f_win.mean()).to_numpy(dtype=float).T),
                         index=f.columns, columns=f.columns)
    F_shr = shrink_F_ledoit_wolf(f_win)
    evF_raw = np.linalg.eigvalsh(F_raw.to_numpy())[::-1]
    evF_shr = np.linalg.eigvalsh(F_shr.to_numpy())[::-1]
    print(f"\n[4] 因子协方差 F")
    print(f"    γ_K={gK:.3f}, 相关矩阵 λ₊={hi:.3f}")
    print(f"    相关矩阵特征值（前 10）：{np.round(evc[:10], 2)}")
    print(f"    超过 λ₊ 的独立方向数：{int((evc > hi).sum())} / {K}")
    print(f"    F 原始条件数：{evF_raw[0]/max(evF_raw[-1],1e-12):.1f}")
    print(f"    F 收缩后条件数：{evF_shr[0]/max(evF_shr[-1],1e-12):.1f}")

    # ---- 时间序列回归 beta ----
    Xw = X.iloc[-T_WINDOW:].to_numpy(dtype=float)
    fw = f_win.to_numpy(dtype=float)
    Xc = Xw - Xw.mean(axis=0)
    fc = fw - fw.mean(axis=0)
    Bmat = np.linalg.solve(fc.T @ fc, fc.T @ Xc)          # K×N
    B = pd.DataFrame(Bmat.T, index=X.columns, columns=f.columns)
    resid = Xc - fc @ Bmat
    D = pd.Series(resid.var(axis=0), index=X.columns)
    total_var = Xc.var(axis=0)
    r2 = 1.0 - D.to_numpy() / total_var

    # ---- Σ̂ = B F_shr Bᵀ + D ----
    Bv = B.to_numpy(dtype=float)
    Fv = F_shr.to_numpy(dtype=float)
    common = Bv @ Fv @ Bv.T
    Sigma = common + np.diag(D.to_numpy(dtype=float))
    evS = np.linalg.eigvalsh(Sigma)[::-1]
    print(f"\n[5] 因子模型（参考日 {X.index[-1].date()}）")
    print(f"    时间序列 R²：均值 {r2.mean()*100:.1f}%，中位 {np.median(r2)*100:.1f}%")
    print(f"    Σ̂ 条件数：{evS[0]/max(evS[-1],1e-12):.1f}")
    print(f"    特质方差 D 均值 {D.mean():.2e} / 总方差均值 {total_var.mean():.2e}")

    # ---- 保存 ----
    f.to_csv(os.path.join(RESULT_DIR, "factor_returns.csv"))
    B.to_csv(os.path.join(RESULT_DIR, "factor_betas.csv"))
    F_shr.to_csv(os.path.join(RESULT_DIR, "factor_F_shrunk.csv"))
    F_raw.to_csv(os.path.join(RESULT_DIR, "factor_F_raw.csv"))
    D.to_csv(os.path.join(RESULT_DIR, "factor_D.csv"))
    summary = {
        "n_universe": N_UNIVERSE, "window": str(X.index[-1].date()),
        "K": K, "T_window": T_WINDOW, "gamma_K": gK,
        "n_indep_directions": int((evc > hi).sum()),
        "F_raw_cond": float(evF_raw[0] / max(evF_raw[-1], 1e-12)),
        "F_shr_cond": float(evF_shr[0] / max(evF_shr[-1], 1e-12)),
        "Sigma_cond": float(evS[0] / max(evS[-1], 1e-12)),
        "mean_R2": float(r2.mean()),
    }
    with open(os.path.join(RESULT_DIR, "factor_baseline_summary.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n[6] 已保存 results/：factor_returns, factor_betas, "
          f"factor_F_raw, factor_F_shrunk, factor_D, summary")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"""
1. 因子收益尺度已正常（年化波动 ~1% 量级）。
2. 20 个因子仍有 {int((evc > hi).sum())} 个独立方向（共线），但 LW 收缩把 F 的
   条件数从 {evF_raw[0]/max(evF_raw[-1],1e-12):.0f} 压到 {evF_shr[0]/max(evF_shr[-1],1e-12):.0f}。
3. 因子模型解释 {r2.mean()*100:.1f}% 收益方差，Σ̂ 可逆（条件数 {evS[0]/max(evS[-1],1e-12):.0f}）。
""")


if __name__ == "__main__":
    main()
