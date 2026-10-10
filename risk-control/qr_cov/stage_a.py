# -*- coding: utf-8 -*-
r"""
qr_cov/stage_a.py —— 阶段 A：主战场决策备忘录
==============================================
回答三件事：
  1. 资产层 vs 因子层的 γ 各是多少，噪声带多宽 → 主战场在哪
  2. 真实 A 股的经验谱 vs MP 律，有多少「信号方向」
  3. λ₁(t)（市场连通性代理）是否时变

产出：
  - 控制台备忘录
  - results/stage_a_lambda1.csv   （λ₁(t) 序列，供后续画图/回归）

用法（在项目根目录）：
  python qr_cov/stage_a.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# 让脚本能 `from qr_cov...` 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns
from qr_cov.qr_suspension import impute_suspension

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_DIR = os.path.join(ROOT, "results")


def mp_edges(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP 谱支撑边界 λ± = σ²(1 ± √γ)²。"""
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def standardize(X: pd.DataFrame) -> np.ndarray:
    """窗口内按列标准化为均值 0、方差 1，得到相关矩阵输入。"""
    Z = X.to_numpy(dtype=float)
    mu = np.nanmean(Z, axis=0)
    sd = np.nanstd(Z, axis=0)
    sd[sd == 0] = 1.0
    Z = (Z - mu) / sd
    return Z


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("=" * 78)
    print("阶段 A：主战场决策备忘录")
    print("=" * 78)

    # ---------- 1. 数据 + 停牌插补 ----------
    R = load_returns("2015-01-01", "2026-09-30")
    Rimp, mask = impute_suspension(R, method="linear_log")
    print(f"\n[1] 数据：{Rimp.shape[0]} 天 × {Rimp.shape[1]} 只")
    print(f"    停牌插补格数：{int(mask.sum().sum()):,} "
          f"({mask.sum().sum()/Rimp.size*100:.2f}%)")

    # ---------- 2. 宇宙：选 500 只最完整的（memo 用，非最终 PIT） ----------
    completeness = Rimp.notna().sum().sort_values(ascending=False)
    universe = completeness.head(500).index
    X = Rimp[universe]
    print(f"\n[2] 宇宙：取完整度最高的 500 只（memo 便捷宇宙，⚠️ 非逐日 PIT）")
    print(f"    每只有效样本数：min={int(completeness[universe].min())} "
          f"median={int(completeness[universe].median())}")

    # ---------- 3. γ 表 ----------
    print("\n[3] γ = N/T 与噪声带宽度（N 只资产 vs T 窗口）")
    print(f"    {'N':>5} {'T':>5} {'γ':>8} {'λ₋':>8} {'λ₊':>8} {'4√γ':>7} {'状态'}")
    for N, T in [(500, 252), (800, 252), (1600, 252),
                 (500, 756), (500, 1260)]:
        g = N / T
        lo, hi = mp_edges(g)
        w = 4 * np.sqrt(g)
        state = ("奇异" if g > 1 else "病态" if hi / lo > 100 else "有噪声")
        print(f"    {N:>5} {T:>5} {g:>8.3f} {lo:>8.3f} {hi:>8.3f} {w:>7.2f} {state}")

    # ---------- 4. 资产层经验谱（最近 252 天） ----------
    print("\n[4] 资产层经验谱 vs MP（最近 252 个交易日，N=500）")
    Xwin = X.iloc[-252:]
    T = Xwin.shape[0]
    N = Xwin.shape[1]
    assert Xwin.notna().all().all(), "窗口内仍有 NaN，请检查插补"
    Z = standardize(Xwin)
    S = Z.T @ Z / T                      # 相关矩阵（标准化后）
    ev = np.linalg.eigvalsh(S)[::-1]     # 降序
    g = N / T
    lo, hi = mp_edges(g)
    bbp = 1.0 + np.sqrt(g)               # BBP 是「真实特征值」的阈值，非样本
    n_above_hi = int((ev > hi).sum())
    print(f"    γ = {g:.3f}   噪声带 = [{lo:.4f}, {hi:.4f}]")
    print(f"    λ₁ = {ev[0]:.2f}   占迹比 = {ev[0]/N*100:.1f}%")
    print(f"    前 8 个特征值：{np.round(ev[:8], 2)}")
    print(f"    样本特征值超过 λ₊（可判为信号）的方向数：{n_above_hi}")
    print(f"    BBP 总体阈值 = {bbp:.2f}（真实特征值需超过它，样本才可能冒头）")
    print(f"    最小特征值：{ev[-1]:.5f}   条件数：{ev[0]/max(ev[-1],1e-12):.1f}")

    # ---------- 5. 因子层 γ ----------
    print("\n[5] 因子层 γ（若用 K 个可观测因子，估 K×K 协方差）")
    print(f"    {'K':>4} {'γ=K/252':>10} {'λ₊':>8} {'结论'}")
    for K in [10, 20, 40, 100]:
        gK = K / 252
        _, hiK = mp_edges(gK)
        print(f"    {K:>4} {gK:>10.3f} {hiK:>8.3f} "
              f"{'几乎干净' if gK < 0.2 else '开始有噪声' if gK < 0.5 else '噪声不可忽略'}")
    print("    注意：F 干净 ≠ 整个因子模型干净；困难藏在 B（N×K 暴露）与 D（对角）里。")

    # ---------- 6. λ₁(t) 连通性代理 ----------
    print("\n[6] λ₁(t) 滚动（252 天，步长 21 天）")
    step = 21
    rows = []
    for i in range(0, len(X) - 252, step):
        w = X.iloc[i:i + 252]
        Z = standardize(w)
        S = Z.T @ Z / w.shape[0]
        ev = np.linalg.eigvalsh(S)
        lam1 = float(ev[-1])
        share = lam1 / w.shape[1] * 100
        rows.append((w.index[-1], lam1, share))
    lam = pd.DataFrame(rows, columns=["date", "lambda1", "share_pct"]).set_index("date")
    out = os.path.join(RESULT_DIR, "stage_a_lambda1.csv")
    lam.to_csv(out)
    print(f"    已保存 {out}  ({len(lam)} 个窗口)")
    print(f"    λ₁ 汇总：mean={lam['lambda1'].mean():.1f} "
          f"min={lam['lambda1'].min():.1f} max={lam['lambda1'].max():.1f}")
    print(f"    占迹比：mean={lam['share_pct'].mean():.1f}% "
          f"min={lam['share_pct'].min():.1f}% max={lam['share_pct'].max():.1f}%")
    print("\n    λ₁ 最高的 5 个窗口：")
    print(lam.nlargest(5, "lambda1").to_string())
    print("\n    λ₁ 最低的 5 个窗口：")
    print(lam.nsmallest(5, "lambda1").to_string())

    print("\n" + "=" * 78)
    print("决策备忘录（初稿）")
    print("=" * 78)
    print("""
1. 资产层 γ≈2 是病态区（S 奇异），噪声带把真实单位谱撑到 0~5.8；
   这是「噪声问题」真正存在的地方。
2. 因子层 K=20 时 γ≈0.08，F 本身几乎干净 —— 但 B 与 D 的估计误差
   才是隐藏的困难，不能因此认为因子模型「免清洗」。
3. 结论：主战场是【资产层协方差 / 因子模型的残差与暴露】，不是 K×K 的 F。
   这支撑 v3 主线：在病态区做噪声尺度校准 × 非线性收缩。
""")


if __name__ == "__main__":
    main()
