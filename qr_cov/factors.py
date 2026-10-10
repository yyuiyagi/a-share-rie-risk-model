# -*- coding: utf-8 -*-
r"""
qr_cov/factors.py —— 20 个可观测量价因子
========================================

只依赖价格/成交量/成交额/换手率（数据均来自 data_tx），不依赖财务数据。
因子值全部用「截至 t 日」的信息计算，交叉截面逐日标准化后得到暴露 B。

因子清单（20 个，分 5 家族）：

    动量/反转  mom20, mom60, mom240_20, rev5
    波动/风险  vol20, vol60, max20, skew60, beta60, idio_vol60
    价格位置  rsi14, ma_ratio20, ma_cross2060, price_pos20
    流动性    turnover20, turnover_chg, amihud20, amount_log20
    量能/资金  vol_ratio20, money_flow20

用法：
    from qr_cov.factors import compute_factor_raw, zscore_cs, factor_returns
    raw = compute_factor_raw(R, aux)      # dict[str, DataFrame]
    z = zscore_cs(raw)                    # 逐日截面 z-score
    f = factor_returns(z, R)              # 20 个因子收益序列（用 t-1 暴露 × t 收益）
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_aux(universe: list[str], index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """从 data_tx/raw 加载成交量/成交额/换手率，对齐到给定交易日历。

    raw 与 qfq 文件的 volume/amount/turnover 相同（未复权），这里用 raw 文件。
    停牌日（源里缺失的日期）保持 NaN。
    """
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW = os.path.join(ROOT, "data_tx", "raw")
    cols = {"volume": [], "amount": [], "turnover": []}
    for sym in universe:
        # universe 的元素已是带前缀符号（如 sh600004 / sz000005），文件名直接就是它
        fp = os.path.join(RAW, f"{sym}.raw.csv.gz")
        if not os.path.exists(fp):
            for k in cols:
                cols[k].append(pd.Series(np.nan, index=index, name=sym))
            continue
        d = pd.read_csv(fp, parse_dates=["date"],
                        usecols=["date", "volume", "amount", "turnover"])
        d = d.set_index("date").reindex(index)
        for k in cols:
            cols[k].append(d[k].rename(sym))
    return {k: pd.concat(v, axis=1).sort_index() for k, v in cols.items()}


def _market(R: pd.DataFrame) -> pd.Series:
    """等权市场收益（截面均值）。"""
    return R.mean(axis=1)


def compute_factor_raw(R: pd.DataFrame,
                       aux: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """计算 20 个因子的原始值（date × instrument）。

    R : 已插补停牌的日收益（宽表）
    aux : load_aux 返回的 volume/amount/turnover
    """
    mkt = _market(R)
    out: dict[str, pd.DataFrame] = {}

    # ---- 动量 / 反转 ----
    out["mom20"] = R.rolling(20).sum()
    out["mom60"] = R.rolling(60).sum()
    out["mom240_20"] = R.rolling(240).sum() - R.rolling(20).sum()
    out["rev5"] = -R.rolling(5).sum()

    # ---- 波动 / 风险 ----
    out["vol20"] = R.rolling(20).std()
    out["vol60"] = R.rolling(60).std()
    out["max20"] = R.rolling(20).max()
    out["skew60"] = R.rolling(60).skew()

    # beta：滚动 60 日与市场收益的协方差 / 市场方差
    cov = R.rolling(60).cov(mkt)
    var_mkt = mkt.rolling(60).var()
    beta = cov.div(var_mkt, axis=0)
    out["beta60"] = beta

    # 特质波动：剔除市场后的残差波动（滚动 60 日）
    resid = R - beta.mul(mkt, axis=0)
    out["idio_vol60"] = resid.rolling(60).std()

    # ---- 价格位置（用收益重构的合成价格指数） ----
    P = (1 + R).cumprod()
    out["rsi14"] = _rsi(R, 14)
    out["ma_ratio20"] = P / P.rolling(20).mean() - 1.0
    out["ma_cross2060"] = P.rolling(20).mean() / P.rolling(60).mean() - 1.0
    lo20 = P.rolling(20).min()
    hi20 = P.rolling(20).max()
    denom = hi20 - lo20
    # 20 日最高==最低（完全平坦）时除零会污染因子收益，置为中性位置 0.5
    out["price_pos20"] = ((P - lo20) / denom).where(denom != 0, 0.5)

    # ---- 流动性 ----
    turnover = aux["turnover"]
    amount = aux["amount"]
    volume = aux["volume"]
    out["turnover20"] = turnover.rolling(20).mean()
    out["turnover_chg"] = turnover / turnover.rolling(20).mean() - 1.0
    # Amihud 非流动性：|R| / 亿元成交额，取 20 日均值后 log1p
    amihud = (R.abs() / (amount / 1e8)).rolling(20).mean()
    out["amihud20"] = np.log1p(amihud)
    out["amount_log20"] = np.log(amount).rolling(20).mean()

    # ---- 量能 / 资金 ----
    out["vol_ratio20"] = volume / volume.rolling(20).mean() - 1.0
    out["money_flow20"] = (R * amount).rolling(20).sum() / amount.rolling(20).sum()

    return out


def _rsi(R: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    gain = R.clip(lower=0)
    loss = (-R).clip(lower=0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # avg_loss 为 0 时（全涨）RSI=100
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def zscore_cs(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """逐日截面 z-score（每列在每一天减去截面均值、除以截面标准差）。"""
    out = {}
    for name, df in factors.items():
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        out[name] = df.sub(mu, axis=0).div(sd, axis=0)
    return out


def factor_returns(z: dict[str, pd.DataFrame], R: pd.DataFrame) -> pd.DataFrame:
    """特征组合因子收益：f_{k,t} = Σ_i z_{i,k,t-1} r_{i,t} / Σ_i |z_{i,k,t-1}|。

    用 t-1 暴露 × t 收益，无前视。
    """
    names = list(z.keys())
    series = {}
    for name in names:
        zt = z[name].shift(1)
        denom = zt.abs().sum(axis=1)
        f = (zt * R).sum(axis=1) / denom
        series[name] = f
    return pd.DataFrame(series)


def factor_returns_quantile(z: dict[str, pd.DataFrame], R: pd.DataFrame,
                            q: float = 0.1) -> pd.DataFrame:
    """分位数多空因子收益（Fama-French 风格，业界标准）。

    对每个因子，在 t-1 日按因子值排序：
        多头 = 前 q 分位（因子值最大的一批）
        空头 = 后 q 分位（因子值最小的一批）
        f_t = mean(r_{i,t} for i in top) - mean(r_{i,t} for i in bottom)

    每个因子收益的波动与单只股票同级（约 1%/日），无前视（t-1 暴露 × t 收益）。
    """
    out = {}
    for name, zt in z.items():
        zprev = zt.shift(1)                       # t-1 暴露
        rank = zprev.rank(axis=1, pct=True)       # 截面分位
        top = rank >= (1.0 - q)
        bot = rank <= q
        ntop = top.sum(axis=1)
        nbot = bot.sum(axis=1)
        # 若某日分位为空（因子截面恒定），除以 1 而非 0，得到 0 收益（当日不交易）
        long_ret = (R * top).sum(axis=1) / ntop.replace(0, 1.0)
        short_ret = (R * bot).sum(axis=1) / nbot.replace(0, 1.0)
        out[name] = long_ret - short_ret
    return pd.DataFrame(out)
