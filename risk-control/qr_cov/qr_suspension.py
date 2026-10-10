# -*- coding: utf-8 -*-
r"""
qr_suspension.py —— 停牌日处理
==============================

背景
----
`qr_data.load_returns()` 返回的宽表里，停牌日表现为 NaN（腾讯直接省略停牌日），
且**复牌日的收益已经是「多日跳空压到一天」**——例如 sz300009 在 2015-09-09~15
停牌 5 天，复牌日收益 11.25%，超过创业板 10% 涨跌停，说明它是约 6 个交易日的
累计收益。

三种处理方式（对应三种偏差方向）：

    method="linear_log"   ★主方案：把复牌日的多日跳空均匀摊回「停牌日 + 复牌日」
                          （在几何平均意义下），总收益守恒、日历不变。
    method="zero"         零填充内部缺口（近似「停牌期间不涨不跌」），会稀释波动率。
    method="raw"          不插补（保持 NaN + 复牌日跳空），会让单日波动率虚高。

注意：
- 只处理**内部缺口**（首尾之间的 NaN）。上市前/退市后的整段 NaN 是结构缺失，
  不在这里处理——那应该用成分股掩码（qr_data.membership_mask）排除。
- 返回一个布尔掩码 `imputed_mask`，标记哪些格子被插补过，供敏感性检验用。

用法
----
    from qr_suspension import impute_suspension
    R_imp, mask = impute_suspension(R, method="linear_log")
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _redistribute_gap(s: pd.Series) -> pd.Series:
    """把每个内部停牌缺口的多日跳空摊回缺口 + 复牌日。

    输入：单只股票的日收益序列（index=日期），内部缺口为 NaN。
    输出：线性插值后的序列（几何平均意义），结构与输入一致。
    """
    out = s.copy()
    isna = s.isna().to_numpy()
    n = len(s)
    # 用「有效→缺失」的边界找内部 NaN run
    # 只处理「前面有有效值、后面也有有效值」的 run
    run_start = None
    for i in range(n):
        if isna[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                # run = [run_start, i-1]，缺失天数 k = i - run_start
                # 复牌日 = i（当前这个有效值），其值是多日累计收益 R
                k = i - run_start
                if run_start > 0:  # 前面必须有有效值，才不是上市前缺失
                    R = out.iloc[i]
                    # 几何平均摊回： (1+avg)^(k+1) = 1+R
                    base = 1.0 + R
                    if base <= 0:
                        # 理论上有跌停保护不会到这里；防御性回退为线性拆分
                        avg = R / (k + 1.0)
                    else:
                        avg = base ** (1.0 / (k + 1.0)) - 1.0
                    out.iloc[run_start:i + 1] = avg
                run_start = None
    return out


def impute_suspension(returns: pd.DataFrame,
                      method: str = "linear_log") -> tuple[pd.DataFrame, pd.DataFrame]:
    """停牌缺口插补。

    参数
    ----
    returns : date × instrument 的日收益宽表（含 NaN 表示停牌）
    method  : "linear_log" | "zero" | "raw"

    返回
    ----
    (imputed, imputed_mask)
        imputed      : 同形状的收益矩阵
        imputed_mask : 同形状布尔矩阵，True = 该格子被插补
    """
    if method not in ("linear_log", "zero", "raw"):
        raise ValueError(f"unknown method: {method}")

    R = returns.copy()
    mask = pd.DataFrame(False, index=R.index, columns=R.columns)

    if method == "raw":
        return R, mask

    for col in R.columns:
        s = R[col]
        valid = s.notna()
        if valid.sum() == 0:
            continue
        first = valid.idxmax()  # 第一个有效位置
        last = s[::-1].notna().idxmax()  # 最后一个有效位置
        seg = s.loc[first:last]
        if method == "zero":
            fill = seg.fillna(0.0)
            R.loc[first:last, col] = fill
            mask.loc[first:last, col] = seg.isna()
        elif method == "linear_log":
            # 记录原始内部缺失位置（用于 mask）
            internal_nan = seg.isna()
            redistributed = _redistribute_gap(seg)
            R.loc[first:last, col] = redistributed
            # 插补过的格子 = 原本是 NaN 的 + 被摊回修改的复牌日
            # 复牌日原本非 NaN 但值被改，也应标记
            changed = (seg != redistributed) | internal_nan
            mask.loc[first:last, col] = changed

    return R, mask


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from qr_data import load_returns

    R = load_returns("2015-01-01", "2016-12-31")
    c = "sz300009"
    print(f"自检标的: {c}")
    s = R[c]
    valid = s.dropna()
    first, last = valid.index.min(), valid.index.max()
    seg = s.loc[first:last]
    # 找第一个内部停牌段
    grp = (seg.isna() != seg.isna().shift()).cumsum()
    gap = None
    for g, sub in seg.groupby(grp):
        if sub.isna().all() and len(sub) >= 3:
            gap = sub
            break
    if gap is None:
        print("未找到内部停牌段")
        sys.exit(1)

    print(f"停牌段: {gap.index.min().date()} -> {gap.index.max().date()} "
          f"({len(gap)} 天)")
    k = len(gap)
    resume_idx = seg.index[seg.index.get_loc(gap.index.max()) + 1]
    R_orig = seg.loc[resume_idx]
    print(f"插补前复牌日收益: {R_orig:.4%}")

    imp, mask = impute_suspension(R, method="linear_log")
    seg2 = imp[c].loc[first:last]
    gap2 = seg2.loc[gap.index]
    resume2 = seg2.loc[resume_idx]
    avg = (1 + R_orig) ** (1.0 / (k + 1)) - 1.0
    print(f"插补后停牌段首值: {gap2.iloc[0]:.4%}  复牌日: {resume2:.4%}")
    print(f"理论摊回值 avg:  {avg:.4%}")

    # 总收益守恒检验
    prod_orig = 1 + R_orig
    prod_new = float((1 + gap2).prod() * (1 + resume2))
    print(f"总收益守恒: 原始 {prod_orig:.6f} vs 插补后 {prod_new:.6f} "
          f"(误差 {abs(prod_orig-prod_new):.2e})")

    # 内部 NaN 是否清零
    internal_after = imp[c].loc[first:last].isna().sum()
    print(f"内部剩余 NaN: {internal_after} (应为 0)")
    print(f"插补标记格数: {int(mask[c].sum())}")
