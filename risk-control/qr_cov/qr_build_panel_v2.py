# -*- coding: utf-8 -*-
r"""
构建质量可信的日收益面板 v2 —— 项目 A 数据层（最终版）
========================================================
背景：实测发现腾讯两个口径各有硬缺陷（见 QR/数据补给说明.md）：

  1. **前复权（qfq）对老股被过度调整，出现负价格**。
     非正价格占全样本 6.5%，涉及 559/1725 只（32.4%）。
     分布：1990-2014 占 12.78%、2015-2020 占 0.363%、2021-2026 仅 0.007%。
     负价格会让 pct_change 爆炸（这是 qfq 出现 91,158 个 |ret|>30% 的原因）。

  2. **不复权（raw）序列内含拆股跳空，且无法用复权因子识别**。
     sh600601 在 1991-03-11 有 -80% 的 5:1 拆股跳空，而当天复权因子无变化
     （实测：不复权 <-15% 的下跌中，0 次伴随复权因子变化）。
     全样本 |ret|>30% 共 3,712 次，2021-2026 仅 163 次。

因此本脚本构建**阶梯式收益**，并显式标注每一条观测的口径来源与可信度：

  优先级 1  qfq_ok   : qfq 当日及前一日价格 > 0，且 |ret| <= 该股涨跌幅上限
  优先级 2  raw_jump : qfq 不可用，raw 有效且 |ret| <= 上限
  优先级 3  (剔除)   : 两者都异常 → 收益置 NaN 并标记，绝不猜测

输出（data_tx/panel2/）
-----------------------
  returns_final.csv.gz  宽表：阶梯式日收益（协方差研究的直接输入）
  ret_source.csv.gz     宽表：口径代码  'q'=qfq  'r'=raw  'x'=剔除
  quality_by_year.csv   按年质量统计（写进报告用）
  panel_meta.json       口径定义与阈值快照

用法：python qr_build_panel_v2.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_tx")
RAW = os.path.join(DATA, "raw")
OUT = os.path.join(DATA, "panel2")

PAT = re.compile(r"^(?P<sym>(?:sh|sz|bj)\d{6})\.(?P<adj>raw|qfq)\.csv\.gz$")

# 涨跌幅上限（含制度变更），用于判定"不可能的单日收益"
def limit_pct(code: str, date: pd.Timestamp) -> float:
    if code.startswith("688"):                       # 科创板 20%
        return 0.20
    if code.startswith(("300", "301")):              # 创业板 2020-08-24 起 20%
        return 0.20 if date >= pd.Timestamp("2020-08-24") else 0.10
    if code.startswith(("8", "4")):                  # 北交所 30%
        return 0.30
    return 0.10

# 容差：新股上市首日、ST 变更、四舍五入等留 5% 余量
TOLERANCE = 0.05


def load() -> dict[str, dict[str, pd.DataFrame]]:
    files = sorted(glob.glob(os.path.join(RAW, "*.csv.gz")))
    print(f"[load] {len(files)} files")
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for i, fp in enumerate(files, 1):
        m = PAT.match(os.path.basename(fp))
        if not m:
            continue
        sym, adj = m.group("sym"), m.group("adj")
        try:
            # instrument 必须按字符串读，否则 000005 会被读成 5
            df = pd.read_csv(fp, parse_dates=["date"], dtype={"instrument": str},
                             usecols=["date", "instrument", "open", "close",
                                      "high", "low", "volume"])
        except Exception as e:
            print(f"  [warn] {os.path.basename(fp)}: {type(e).__name__}: {e}")
            continue
        if df.empty:
            continue
        df["instrument"] = df["instrument"].astype(str).str.zfill(6)
        out.setdefault(sym, {})[adj] = df.sort_values("date").reset_index(drop=True)
        if i % 800 == 0:
            print(f"  ...{i}/{len(files)}")
    print(f"[load] symbols: {len(out)}")
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    data = load()

    ret_cols: dict[str, pd.Series] = {}
    src_cols: dict[str, pd.Series] = {}
    stats = {"qfq_ok": 0, "raw_jump": 0, "dropped": 0, "total": 0}
    year_stat: dict[int, dict[str, int]] = {}

    for sym, byk in sorted(data.items()):
        code = sym[-6:]
        raw = byk.get("raw")
        if raw is None or raw.empty:
            continue
        r = raw.set_index("date")

        # 不复权收益
        rr = r["close"].pct_change()
        rpos = r["close"] > 0
        rr_ok = rpos & rpos.shift(1).fillna(False)
        rr = rr.where(rr_ok)

        # 前复权收益
        q = byk.get("qfq")
        if q is not None and not q.empty:
            qs = q.set_index("date")["close"]
            # 注意：qfq 的存在本身就是"当日有有效复权价"的依据
            qpos = qs > 0
            qr = qs.pct_change()
            qr = qr.where(qpos & qpos.shift(1).fillna(False))
            rq = qr.reindex(r.index)
        else:
            rq = pd.Series(np.nan, index=r.index)

        # 涨跌幅上限（逐日）
        lim = pd.Series([limit_pct(code, d) for d in r.index], index=r.index)
        cap = lim + TOLERANCE

        # 阶梯选择
        use_q = rq.notna() & (rq.abs() <= cap)
        use_r = (~use_q) & rr.notna() & (rr.abs() <= cap)

        final = rq.where(use_q, rr.where(use_r))
        source = pd.Series("x", index=r.index, dtype=object)
        source[use_r] = "r"
        source[use_q] = "q"
        # 无任何数据的日子（停牌等）保持 'x'
        source[rq.isna() & rr.isna() & (~r["close"].isna())] = "x"

        ret_cols[sym] = final
        src_cols[sym] = source

        stats["total"] += len(final)
        stats["qfq_ok"] += int(use_q.sum())
        stats["raw_jump"] += int(use_r.sum())
        stats["dropped"] += int((~use_q & ~use_r & r["close"].notna()).sum())

        ys = pd.DataFrame({"src": source, "y": r.index.year})
        for y, g in ys.groupby("y"):
            d = year_stat.setdefault(int(y), {"q": 0, "r": 0, "x": 0})
            vc = g["src"].value_counts()
            for k in ("q", "r", "x"):
                d[k] += int(vc.get(k, 0))

    print("[build] assembling wide matrices ...")
    rets = pd.DataFrame(ret_cols).sort_index()
    srcs = pd.DataFrame(src_cols).sort_index()
    rets = rets.loc[:, ~rets.columns.duplicated()]
    srcs = srcs.loc[:, ~srcs.columns.duplicated()]

    def dump(df, name, index=True):
        p = os.path.join(OUT, name)
        try:
            df.to_parquet(p, compression="zstd")
        except Exception:
            p = p.replace(".parquet", ".csv.gz")
            df.to_csv(p, index=index)
        print(f"  [out] {os.path.basename(p)}  shape={df.shape}")

    dump(rets, "returns_final.parquet")
    dump(srcs, "ret_source.parquet")

    qy = pd.DataFrame(year_stat).T.sort_index()
    qy.index.name = "year"
    qy["total"] = qy.sum(axis=1)
    qy["usable_pct"] = (qy["q"] + qy["r"]) / qy["total"].replace(0, np.nan) * 100
    qy["qfq_pct"] = qy["q"] / qy["total"].replace(0, np.nan) * 100
    qy.to_csv(os.path.join(OUT, "quality_by_year.csv"))
    print(f"  [out] quality_by_year.csv  shape={qy.shape}")

    meta = {
        "built": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": "Tencent newfqkline (raw + qfq)",
        "ladder": ["q: qfq valid & |ret|<=limit", "r: raw valid & |ret|<=limit",
                   "x: dropped (anomalous)"],
        "tolerance_over_limit": TOLERANCE,
        "limit_rules": {"688": 0.20, "300/301": "0.20 after 2020-08-24 else 0.10",
                        "8xx/4xx": 0.30, "default": 0.10},
        "stats": stats,
        "symbols": len(ret_cols),
        "known_defects": {
            "qfq_negative_prices": "559/1725 symbols; 12.78% pre-2015, "
                                   "0.363% 2015-2020, 0.007% 2021-2026",
            "raw_split_jumps": "3712 days with |ret|>0.30; not identifiable "
                               "via adj_factor (0 overlap)",
            "recommended_window": "2015-2026 (2021+ essentially clean)",
        },
    }
    with open(os.path.join(OUT, "panel_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [out] panel_meta.json")

    # 报告
    print("\n" + "=" * 74)
    print("阶梯式收益面板摘要")
    print("=" * 74)
    n = stats["total"]
    print(f"symbols        : {len(ret_cols)}")
    print(f"dates          : {len(rets)}  "
          f"{rets.index.min().date()} ~ {rets.index.max().date()}")
    print(f"total cells    : {n:,}")
    print(f"qfq 采用       : {stats['qfq_ok']:,} ({stats['qfq_ok']/n*100:.2f}%)")
    print(f"raw 兜底       : {stats['raw_jump']:,} ({stats['raw_jump']/n*100:.2f}%)")
    print(f"剔除           : {stats['dropped']:,} ({stats['dropped']/n*100:.2f}%)")

    v = rets.to_numpy(dtype=float)
    fl = v[~np.isnan(v)]
    print(f"\n最终收益: mean={fl.mean():.6f} std={fl.std():.6f}")
    for qq in (0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999):
        print(f"  q{qq:<8}: {np.quantile(fl, qq):+.6f}")
    print(f"\n|ret|>0.30 : {(np.abs(fl) > 0.30).sum():,}")
    print(f"|ret|>0.50 : {(np.abs(fl) > 0.50).sum():,}")

    print("\n-- 关键时刻可用性 --")
    print(qy.loc[[y for y in (2015, 2018, 2021, 2022, 2024, 2025, 2026)
                  if y in qy.index]][["q", "r", "x", "usable_pct"]].to_string())


if __name__ == "__main__":
    main()
