# -*- coding: utf-8 -*-
r"""
构建统一日线面板 —— 项目 A 数据层
=================================
输入：data_tx/raw/<symbol>.<raw|qfq>.csv.gz   （由 qr_download_tx.py 产出）
输出：data_tx/panel/
        prices_raw.parquet    长表：不复权 OHLCV + turnover + amount
        returns_raw.parquet   宽表：不复权日收益  (date × instrument)
        returns_adj.parquet   宽表：前复权日收益  (date × instrument)
        adj_factor.parquet    长表：前复权因子 = qfq_close / raw_close
        flags.parquet         长表：停牌 / 涨跌停 标记
        coverage.csv          每只股票的覆盖统计
        validate_vs_qlib.csv  与 qlib cn_data 的交叉验证结果

设计决策（写进报告「数据」章节）
--------------------------------
1. **不复权为主口径**。前复权价会在每次分红时被整体重算，导致历史波动率
   被系统性低估；协方差估计对这一点敏感。故风险侧以不复权收益为准。
2. **前复权收益作为对照口径**。qfq 已剔除除权除息跳空，但腾讯对部分老股
   的 qfq 序列起点晚于不复权（无法构造早期复权因子），因此两种口径都要存。
3. **复权因子显式落盘**（qfq_close / raw_close）。这样任何一天都能在对数
   收益上加回分红跳空，不必重下数据。
4. **收益率用同一口径内的价格算**，绝不做跨口径相减。

用法：
    python qr_build_panel.py
    python qr_build_panel.py --validate-only
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = r"C:\Users\Yui_Y\Desktop\Work\Study Notes"
DATA = os.path.join(HERE, "data_tx")
RAW = os.path.join(DATA, "raw")
PANEL = os.path.join(DATA, "panel")
QLIB = os.path.join(HERE, r"Quant\data\qlib_data\cn_data")

PAT = re.compile(r"^(?P<sym>(?:sh|sz|bj)\d{6})\.(?P<adj>raw|qfq)\.csv\.gz$")

# 涨跌停幅度（按代码段近似，含历史制度变更的说明见 validate 输出）
def limit_pct(code: str, date: pd.Timestamp) -> float:
    """返回该股票在该日期的涨跌幅限制（小数）。"""
    if code.startswith("688"):                      # 科创板：2019-07 起 20%
        return 0.20
    if code.startswith("300") or code.startswith("301"):
        # 创业板：2020-08-24 起由 10% 改为 20%
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("8", "4")):                 # 北交所 30%
        return 0.30
    return 0.10


def load_raw() -> dict[str, pd.DataFrame]:
    files = sorted(glob.glob(os.path.join(RAW, "*.csv.gz")))
    print(f"[load] {len(files)} files")
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for i, fp in enumerate(files, 1):
        m = PAT.match(os.path.basename(fp))
        if not m:
            continue
        sym, adj = m.group("sym"), m.group("adj")
        try:
            # instrument 必须按字符串读：6 位代码的前导零会被推断成整数而丢失，
            # 导致 000005 与 000051 被并成同一只股票（会产生虚假收益率）。
            df = pd.read_csv(fp, parse_dates=["date"],
                             dtype={"instrument": str})
        except Exception as e:
            print(f"  [warn] {os.path.basename(fp)}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            continue
        df["instrument"] = df["instrument"].astype(str).str.zfill(6)
        df = df.sort_values("date").reset_index(drop=True)
        out.setdefault(sym, {})[adj] = df
        if i % 500 == 0:
            print(f"  ...{i}/{len(files)}")
    print(f"[load] symbols parsed: {len(out)}")
    return out


def build(data: dict[str, dict[str, pd.DataFrame]]):
    os.makedirs(PANEL, exist_ok=True)

    price_frames, ret_frames, ret_adj_frames, fac_frames, flag_frames, cov_rows = \
        [], [], [], [], [], []

    for sym, byk in sorted(data.items()):
        code = sym[-6:]
        raw = byk.get("raw")
        qfq = byk.get("qfq")

        if raw is None:
            cov_rows.append({"symbol": sym, "instrument": code, "has_raw": False,
                             "has_qfq": qfq is not None, "rows_raw": 0,
                             "rows_qfq": 0, "first": None, "last": None})
            continue

        r = raw[["date", "instrument", "open", "close", "high", "low",
                 "volume", "turnover", "amount"]].copy()
        r["ret_raw"] = r["close"].pct_change()
        # 首日收益：用 open→close 兜底（仅首行）
        r.loc[r.index[0], "ret_raw"] = np.nan

        # 复权因子与 qfq 收益
        if qfq is not None and not qfq.empty:
            q = qfq[["date", "close"]].rename(columns={"close": "close_qfq"})
            r = r.merge(q, on="date", how="left")
            r["adj_factor"] = r["close_qfq"] / r["close"]
            r["ret_adj"] = r["close_qfq"].pct_change()
        else:
            r["close_qfq"] = np.nan
            r["adj_factor"] = np.nan
            r["ret_adj"] = np.nan

        # 停牌近似：当日成交量为 0 或缺失（腾讯停牌日通常无记录，但保留兜底）
        r["is_suspended"] = r["volume"].isna() | (r["volume"] <= 0)

        # 涨跌停标记：按幅度上限 + 2 位小数取整容差
        lim = np.array([limit_pct(code, d) for d in r["date"]])
        prev = r["close"].shift(1)
        up = np.round(prev * (1 + lim), 2)
        dn = np.round(prev * (1 - lim), 2)
        r["limit_up"] = (r["close"] >= up - 1e-9) & prev.notna()
        r["limit_down"] = (r["close"] <= dn + 1e-9) & prev.notna()

        price_frames.append(r[["date", "instrument", "open", "close", "high",
                               "low", "volume", "turnover", "amount",
                               "is_suspended", "limit_up", "limit_down"]])
        ret_frames.append(r.set_index("date")["ret_raw"].rename(sym))
        if r["ret_adj"].notna().any():
            ret_adj_frames.append(r.set_index("date")["ret_adj"].rename(sym))
        fac_frames.append(r[["date", "instrument", "adj_factor"]].dropna(
            subset=["adj_factor"]))
        flag_frames.append(r[["date", "instrument", "is_suspended",
                              "limit_up", "limit_down"]])
        cov_rows.append({
            "symbol": sym, "instrument": code, "has_raw": True,
            "has_qfq": qfq is not None, "rows_raw": len(r),
            "rows_qfq": 0 if qfq is None else len(qfq),
            "first": str(r["date"].iloc[0].date()),
            "last": str(r["date"].iloc[-1].date()),
            "adj_cov": float(r["adj_factor"].notna().mean()),
            "n_limit_up": int(r["limit_up"].sum()),
            "n_limit_down": int(r["limit_down"].sum()),
            "n_suspended": int(r["is_suspended"].sum()),
        })

    print("[build] concatenating price/flag tables ...")
    prices = pd.concat(price_frames, ignore_index=True)
    flags = pd.concat(flag_frames, ignore_index=True)
    prices = prices.sort_values(["date", "instrument"]).reset_index(drop=True)
    flags = flags.sort_values(["date", "instrument"]).reset_index(drop=True)

    print("[build] building wide return matrices ...")
    rets = pd.concat(ret_frames, axis=1).sort_index()
    rets = rets.loc[:, ~rets.columns.duplicated()]
    rets_adj = (pd.concat(ret_adj_frames, axis=1).sort_index()
                if ret_adj_frames else pd.DataFrame())
    if not rets_adj.empty:
        rets_adj = rets_adj.loc[:, ~rets_adj.columns.duplicated()]

    fac = pd.concat(fac_frames, ignore_index=True) if fac_frames else pd.DataFrame()
    cov = pd.DataFrame(cov_rows)

    def dump(df: pd.DataFrame, name: str) -> None:
        if df is None or df.empty:
            print(f"  [skip] {name} (empty)")
            return
        p = os.path.join(PANEL, name)
        try:
            df.to_parquet(p, index=(name.startswith("returns")), compression="zstd")
        except Exception:
            p = p.replace(".parquet", ".csv.gz")
            # 宽收益矩阵保留日期索引；其余不写索引
            df.to_csv(p, index=name.startswith("returns"))
        print(f"  [out] {os.path.basename(p)}  shape={df.shape}")

    dump(prices, "prices_raw.parquet")
    dump(rets, "returns_raw.parquet")
    dump(rets_adj, "returns_adj.parquet")
    dump(fac, "adj_factor.parquet")
    dump(flags, "flags.parquet")
    cov.to_csv(os.path.join(PANEL, "coverage.csv"), index=False)
    print(f"  [out] coverage.csv  shape={cov.shape}")

    return prices, rets, rets_adj, cov


def validate_vs_qlib(prices: pd.DataFrame) -> None:
    """与 qlib cn_data 交叉验证不复权价：qlib 的 raw = $close / $factor。"""
    import struct

    cal = open(os.path.join(QLIB, "calendars", "day.txt"),
               encoding="utf-8").read().split()
    cal = pd.to_datetime(cal)

    def read_bin(p: str):
        with open(p, "rb") as f:
            raw = f.read()
        vals = np.frombuffer(raw, dtype="<f4")
        si = int(vals[0])
        return si, vals[1:].astype(np.float64)

    print("\n[validate] 与 qlib cn_data 交叉验证（不复权 close）")
    rows = []
    syms = sorted(prices["instrument"].unique())[:400]
    for code in syms[::8]:                      # 抽样 ~50 只
        for ex in ("sh", "sz"):
            d = os.path.join(QLIB, "features", f"{ex}{code}")
            if not os.path.isdir(d):
                continue
            try:
                si_c, c = read_bin(os.path.join(d, "close.day.bin"))
                si_f, f = read_bin(os.path.join(d, "factor.day.bin"))
            except Exception:
                continue
            n = min(len(c), len(f))
            q = pd.DataFrame({"date": cal[si_c:si_c + n],
                              "qlib_raw": c[:n] / f[:n]}).dropna()
            t = prices[prices["instrument"] == code][["date", "close"]]
            j = q.merge(t, on="date", how="inner")
            if len(j) < 200:
                continue
            diff = (j["close"] - j["qlib_raw"]).abs()
            rel = diff / j["qlib_raw"].replace(0, np.nan)
            rows.append({
                "instrument": code, "n_overlap": len(j),
                "median_abs_diff": float(diff.median()),
                "median_rel_diff": float(rel.median()),
                "p99_rel_diff": float(rel.quantile(0.99)),
                "max_rel_diff": float(rel.max()),
            })
            break
    if not rows:
        print("  (no overlap found)")
        return
    v = pd.DataFrame(rows)
    v.to_csv(os.path.join(PANEL, "validate_vs_qlib.csv"), index=False)
    print(f"  compared {len(v)} symbols on overlapping dates")
    print(f"  median rel diff : {v['median_rel_diff'].median():.3e}")
    print(f"  99th pct rel diff: {v['p99_rel_diff'].median():.3e}")
    print(f"  worst   rel diff : {v['max_rel_diff'].max():.3e}")
    good = (v["median_rel_diff"] < 1e-3).sum()
    print(f"  symbols with median rel diff < 1e-3: {good}/{len(v)}")
    print(f"  [out] validate_vs_qlib.csv")


def report(prices: pd.DataFrame, rets: pd.DataFrame, cov: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("面板摘要")
    print("=" * 70)
    print(f"symbols          : {prices['instrument'].nunique()}")
    print(f"date range       : {prices['date'].min().date()} ~ "
          f"{prices['date'].max().date()}")
    print(f"trading days     : {prices['date'].nunique()}")
    print(f"rows (long)      : {len(prices):,}")
    print(f"returns matrix   : {rets.shape}")
    print(f"limit_up total   : {int(prices['limit_up'].sum()):,}")
    print(f"limit_down total : {int(prices['limit_down'].sum()):,}")
    print(f"suspended rows   : {int(prices['is_suspended'].sum()):,}")

    print("\n-- 每年有效标的数（有收益记录） --")
    per_year = rets.notna().groupby(rets.index.year).sum().mean(axis=1)
    for y, v in per_year.items():
        print(f"  {y}: {v:6.0f}")

    print("\n-- 复权覆盖 --")
    if "adj_cov" in cov.columns:
        print(f"  symbols with any qfq overlap: "
              f"{(cov['adj_cov'] > 0).sum()}/{len(cov)}")
        print(f"  mean qfq coverage           : {cov['adj_cov'].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    if args.validate_only:
        p = os.path.join(PANEL, "prices_raw.csv.gz")
        if not os.path.exists(p):
            p = os.path.join(PANEL, "prices_raw.parquet")
        prices = (pd.read_parquet(p) if p.endswith(".parquet")
                  else pd.read_csv(p, parse_dates=["date"]))
        validate_vs_qlib(prices)
        return

    data = load_raw()
    prices, rets, rets_adj, cov = build(data)
    validate_vs_qlib(prices)
    report(prices, rets, cov)


if __name__ == "__main__":
    main()
