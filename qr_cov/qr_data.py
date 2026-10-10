# -*- coding: utf-8 -*-
r"""
qr_data.py —— 项目 A 数据加载接口
==================================
把 data_tx 的存储细节封装掉，避免下游代码重复踩两个已知的坑：

  坑 1：instrument 必须以字符串读，否则 000005 会被 pandas 读成 5，
        与 000051 合并成同一只股票，产生虚假收益序列。
  坑 2：必须是「阶梯式」收益（qfq 优先、raw 兜底），而不是单一口径。
        qfq 对老股有负价格缺陷，raw 有拆股跳空。

用法：
    from qr_data import load_returns, load_source, load_quality, DEFAULT_WINDOW
    R = load_returns(*DEFAULT_WINDOW)      # 宽表 date × instrument
    S = load_source(*DEFAULT_WINDOW)       # 'q' / 'r' / 'x'
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# qr_data.py 位于 qr_cov/ 下，项目根目录在其上一级；数据 data_tx/ 与 qlib 数据在根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_tx")
P2 = os.path.join(DATA, "panel2")
QLIB = os.path.join(ROOT, r"Quant\data\qlib_data\cn_data")

# 数据质量实测结论（见 QR/数据补给说明.md）
#   前复权非正价格占比：1990-2014 = 12.78%、2015-2020 = 0.363%、2021-2026 = 0.007%
QUALITY_TIERS = {
    "primary":   ("2015-01-01", "2026-09-30"),   # 推荐，可用率 > 99.9%
    "clean":     ("2021-01-01", "2026-09-30"),   # 最干净，含 2024 微盘危机
    "extended":  ("2010-01-01", "2026-09-30"),   # 需清洗，谨慎使用
    "full":      ("1990-12-19", "2026-09-30"),   # 仅供历史参考，2015 前有缺陷
}
DEFAULT_WINDOW = QUALITY_TIERS["primary"]


def _read_wide(name: str, lo: str | None = None,
               hi: str | None = None) -> pd.DataFrame:
    """读宽表并做日期切片。宽表 index 为日期，columns 为 6 位代码字符串。"""
    base = os.path.join(P2, name)
    if os.path.exists(base + ".parquet"):
        df = pd.read_parquet(base + ".parquet")
    elif os.path.exists(base + ".csv.gz"):
        # low_memory=False 必需：分块类型推断会在宽表上抛 IndexError
        df = pd.read_csv(base + ".csv.gz", index_col=0, low_memory=False)
    else:
        raise FileNotFoundError(f"panel2 缺失 {name}，请先跑 qr_build_panel_v2.py")
    df.index = pd.to_datetime(df.index)
    # 关键：列名统一为 6 位字符串，杜绝前导零丢失
    if name.startswith("returns") or name.startswith("ret_source"):
        df.columns = [str(c).zfill(6) for c in df.columns]
    else:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.zfill(6)
    if lo or hi:
        df = df.loc[lo:hi]
    return df


def load_returns(lo: str | None = None, hi: str | None = None,
                 drop_short: int = 0) -> pd.DataFrame:
    """阶梯式日收益宽表（qfq 优先，raw 兜底）。index=日期，columns=6位代码。

    drop_short: 若 >0，剔除窗口内有效样本少于此数的标的。
    """
    r = _read_wide("returns_final", lo, hi)
    if drop_short:
        keep = r.notna().sum() >= drop_short
        r = r.loc[:, keep[keep].index]
    return r


def load_source(lo: str | None = None, hi: str | None = None) -> pd.DataFrame:
    """口径来源宽表：'q'=前复权, 'r'=不复权兜底, 'x'=剔除。"""
    return _read_wide("ret_source", lo, hi)


def load_quality() -> pd.DataFrame:
    """按年的口径采用统计。"""
    return pd.read_csv(os.path.join(P2, "quality_by_year.csv"))


def load_meta() -> dict:
    with open(os.path.join(P2, "panel_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_pit_membership() -> pd.DataFrame:
    """qlib 逐日成分股（无幸存者偏差）。columns: symbol, start, end。"""
    p = os.path.join(QLIB, "instruments", "csi500.txt")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                rows.append({
                    "symbol": parts[0],
                    "instrument": parts[0][-6:],
                    "start": pd.Timestamp(parts[1]),
                    "end": pd.Timestamp(parts[2]),
                })
    return pd.DataFrame(rows)


def membership_mask(dates: pd.DatetimeIndex,
                    instruments: list[str]) -> pd.DataFrame:
    """构造 (date × instrument) 的成分股布尔掩码，用于把面板收窄到中证500。

    注意：
    - 面板列名是带前缀小写符号（如 sh600004），与 qlib 成分表里的 SH600004
      只差大小写，这里统一转小写匹配。
    - qlib 的成分表覆盖 1999–2022-03；2022 年之后无官方历史成分，
      该段仍是当前快照池，存在成分选择偏差——报告里必须声明。
    """
    mem = load_pit_membership()
    mask = pd.DataFrame(False, index=dates, columns=instruments)
    for sym, g in mem.groupby("symbol"):
        code_lower = sym.lower()          # SH600004 -> sh600004
        if code_lower not in mask.columns:
            continue
        m = (dates >= g["start"].min()) & (dates <= g["end"].max())
        mask.loc[m, code_lower] = True
    return mask


def describe(lo: str, hi: str) -> None:
    r = load_returns(lo, hi)
    s = load_source(lo, hi)
    print(f"window {lo} ~ {hi}")
    print(f"  shape          : {r.shape}")
    n = r.size
    print(f"  有效收益       : {int(r.notna().to_numpy().sum()):,} / {n:,}")
    if not s.empty:
        vc = s.stack().value_counts()
        for k, v in vc.items():
            print(f"  口径 {k!r}        : {v:,} ({v/n*100:.2f}%)")
    per_day = r.notna().sum(axis=1)
    print(f"  每日有效标的   : 中位={int(per_day.median())} "
          f"min={int(per_day.min())} max={int(per_day.max())}")


if __name__ == "__main__":
    meta = load_meta()
    print("panel built:", meta["built"])
    print("known defects:")
    for k, v in meta["known_defects"].items():
        print(f"  {k}: {v}")
    print()
    for tier, (lo, hi) in QUALITY_TIERS.items():
        describe(lo, hi)
        print()
