# -*- coding: utf-8 -*-
r"""
qr_cov/diagnose_structural.py —— 方向 D 第一步：A 股结构性缺陷诊断（v2）
========================================================================
v2 修正一个方向性错误：涨跌停是「截断」——把本该更偏离的收益强行压到边界，
所以它对 σ 的因果效应是**压低（低估）真实波动**，而不是抬高。

正确的量化方法：用「封板频率」反推未截断的真实 σ。
    若真实收益 ~ N(0, σ_true)，涨停阈值 thr，涨停概率 p_up = P(r ≥ thr)
        →  thr / σ_true = Φ⁻¹(1 - p_up)   ⇒  σ_true = thr / Φ⁻¹(1 - p_up)

然后对比观测 σ_obs 与反推 σ_true，得出「涨跌停把 σ 压低了多少」。

另把「长尾」与「涨跌停」两个问题分开表述：
    - 涨跌停（截断）：系统性**低估 σ**（偏差，方向：压低）
    - 长尾（重尾）  ：不系统性偏差 σ，但让样本方差对极端值**敏感（不稳定）**

用法：
    $env:PYTHONIOENCODING='utf-8'
    python qr_cov/diagnose_structural.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_cov.qr_data import load_returns


def excess_kurtosis(x):
    m = x - x.mean()
    m2 = float((m ** 2).mean())
    m4 = float((m ** 4).mean())
    return m4 / m2 ** 2 - 3.0          # 正态 = 0


def skewness(x):
    m = x - x.mean()
    m2 = float((m ** 2).mean())
    m3 = float((m ** 3).mean())
    return m3 / m2 ** 1.5


def winsorize(x, q):
    lo = np.percentile(x, q * 100)
    hi = np.percentile(x, 100 - q * 100)
    return np.clip(x, lo, hi)


def limit_thr(code: str) -> float:
    """按代码前缀定涨跌停阈值：创业板/科创板 20%，其余 10%（ST 无法从代码判断，忽略）。"""
    return 0.195 if code[:2] in ("30", "68") else 0.095


def implied_sigma(p_side: float, thr: float) -> float:
    """由单侧封板频率反推未截断真实 σ（正态假设，μ≈0）。"""
    p_side = float(np.clip(p_side, 1e-6, 0.499))
    z = norm.ppf(1.0 - p_side)
    return thr / z if z > 0 else np.nan


def main():
    R = load_returns("2015-01-01", "2026-09-30")
    comp = R.notna().sum().sort_values(ascending=False)
    all_cols = comp.index.tolist()
    g500 = comp.head(500).index.tolist()
    g_small = comp.iloc[500:1200].index.tolist()
    groups = [("500宇宙(大中盘)", g500), ("次500(小盘)", g_small), ("全量", all_cols)]

    print("=" * 84)
    print("方向D第一步：A股结构性缺陷诊断 v2（截断低估 σ / 长尾敏感度）")
    print("=" * 84)

    summary = {}
    for gname, cols in groups:
        rows = []
        for c in cols:
            r = R[c].dropna().to_numpy(dtype=float)
            if len(r) < 200:
                continue
            thr = limit_thr(c)
            is20 = thr > 0.15
            p_up = float((r >= thr).mean())
            p_dn = float((r <= -thr).mean())
            p_lim = p_up + p_dn
            p_lim20 = float((np.abs(r) >= 0.185).mean())
            p_lim5 = float((np.abs(r) >= 0.045).mean())

            s_obs = float(r.std(ddof=1))
            si_up = implied_sigma(p_up, thr)   # 用涨停频率反推
            si_dn = implied_sigma(p_dn, thr)   # 用跌停频率反推
            s_true = float(np.nanmean([si_up, si_dn]))
            under = s_obs / s_true             # <1 表示观测 σ 被低估

            s_w1 = float(winsorize(r, 0.01).std(ddof=1))
            s_w2 = float(winsorize(r, 0.02).std(ddof=1))
            rows.append(dict(ek=excess_kurtosis(r), sk=skewness(r),
                             is20=is20, thr=thr,
                             p_up=p_up, p_dn=p_dn, p_lim=p_lim,
                             p_lim20=p_lim20, p_lim5=p_lim5,
                             s_obs=s_obs, s_true=s_true, under=under,
                             s_w1=s_w1, s_w2=s_w2))
        df = pd.DataFrame(rows)
        df["sens_w1"] = 1.0 - df.s_w1 / df.s_obs   # 尾部 1% 对样本方差的敏感度
        summary[gname] = df

    # ---- 1. 分布形状 ----
    print("\n【1】收益分布形状（excess kurtosis，正态=0；越大越重尾）")
    print(f"{'组':<16}{'峰度中位':>9}{'峰度p90':>9}{'偏度中位':>9}{'峰度>5占比':>11}")
    for gname, df in summary.items():
        frac = (df.ek > 5).mean() * 100
        print(f"{gname:<16}{df.ek.median():>9.1f}{df.ek.quantile(0.9):>9.1f}"
              f"{df.sk.median():>9.2f}{frac:>10.0f}%")

    # ---- 2. 封板密度 ----
    print("\n【2】涨跌停封板密度（日均占比 %；阈值按板块 10%/20%）")
    print(f"{'组':<16}{'涨停':>8}{'跌停':>8}{'封板合计':>10}{'≥20%板':>10}{'≥5%板':>10}")
    for gname, df in summary.items():
        print(f"{gname:<16}{df.p_up.mean()*100:>7.2f}%{df.p_dn.mean()*100:>7.2f}%"
              f"{df.p_lim.mean()*100:>9.2f}%{df.p_lim20.mean()*100:>9.2f}%"
              f"{df.p_lim5.mean()*100:>9.2f}%")

    # ---- 3. 截断低估 σ（正确方向）----
    print("\n【3】涨跌停截断 → σ 被低估（用封板频率反推未截断的真实 σ）")
    print(f"{'组':<16}{'观测σ%':>9}{'反推真实σ%':>11}{'低估比σobs/σtrue':>16}{'压低幅度':>10}")
    for gname, df in summary.items():
        ratio = df.s_obs.mean() / df.s_true.mean()
        print(f"{gname:<16}{df.s_obs.mean()*100:>8.2f}%{df.s_true.mean()*100:>10.2f}%"
              f"{ratio:>15.3f}{(1-ratio)*100:>9.0f}%")

    # 主板 vs 20%板 分层（主板占绝大多数，是主要结论）
    print("\n   按板块分层（主板 10% 阈值 / 创业·科创 20% 阈值）：")
    print(f"{'板块':<14}{'观测σ%':>9}{'反推真实σ%':>11}{'低估比':>10}{'压低幅度':>10}{'股票数':>8}")
    for gname, df in [("全量", summary["全量"])]:
        for is20, label in [(False, "主板(10%)"), (True, "创业/科创(20%)")]:
            sub = df[df.is20 == is20]
            if len(sub) < 10:
                continue
            ratio = sub.s_obs.mean() / sub.s_true.mean()
            print(f"{label:<14}{sub.s_obs.mean()*100:>8.2f}%{sub.s_true.mean()*100:>10.2f}%"
                  f"{ratio:>9.3f}{(1-ratio)*100:>9.0f}%{len(sub):>8}")

    # ---- 4. 长尾敏感度（重新解读：不稳定，非偏差）----
    print("\n【4】长尾对 σ 估计的「敏感度」（winsorize 1% 后 σ 降幅，是稳定性指标，非偏差）")
    print(f"{'组':<16}{'σ_obs%':>9}{'σ_w1%':>9}{'σ_w2%':>9}{'尾部1%敏感度':>12}{'尾部2%敏感度':>12}")
    for gname, df in summary.items():
        print(f"{gname:<16}{df.s_obs.mean()*100:>8.2f}%{df.s_w1.mean()*100:>8.2f}%"
              f"{df.s_w2.mean()*100:>8.2f}%{df.sens_w1.mean()*100:>11.1f}%"
              f"{(1-df.s_w2.mean()/df.s_obs.mean())*100:>11.1f}%")

    print("\n解读：")
    print("  · 涨跌停 = 截断 = 把 σ 系统性**压低**（观测波动低于真实波动），量级见【3】的「压低幅度」。")
    print("  · 长尾 = 重尾 = 不系统性偏差 σ，但让样本方差对极端值敏感（【4】的敏感度），是「估计不稳定」。")
    print("  · 两者性质不同：一个改「准不准」（偏差），一个改「稳不稳」（方差）。")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "structural_diag_v2.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for gname, df in summary.items():
        df.to_csv(out.replace(".csv", f"_{gname}.csv"), index=False, encoding="utf-8-sig")
    print(f"\n明细已存：{out.replace('.csv', '_<组>.csv')}")


if __name__ == "__main__":
    main()
