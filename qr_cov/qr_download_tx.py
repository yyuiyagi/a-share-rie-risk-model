# -*- coding: utf-8 -*-
r"""
下载腾讯日线（不复权 + 前复权双口径）—— 项目 A 数据补给
============================================================
背景
----
qlib 官方 cn_data 止于 2020-09-25；东财域在本机被限流封禁（整个 eastmoney 域
连接被重置，包括 push2his / spot，且非本机网络问题——新浪、腾讯均正常）。

本脚本使用腾讯端点 `newfqkline/get`：
  - 覆盖长历史（沪市主板可回溯至 1991），**包含已退市股票**
  - 包含科创板/创业板
  - 单次请求最多 2000 行（count>2000 返回 param error），用向前翻页拼接

相比 akshare 的 `stock_zh_a_hist_tx`（按年循环 → 每只 ~37 次请求，实测
18.9 秒/只），本实现每只约 5-9 次请求。

字段语义（已用 qt 当期快照交叉验证）
-----------------------------------
  行结构 [date, open, close, high, low, volume, {}, turnover, amount, 0, 0]
  volume 单位为「手」→ ×100 得股（688/指数已是股，不乘）
  turnover 为百分数 → ÷100
  amount 单位为「万元」→ ×10000 得元

输出
----
  data_tx/raw/<symbol>.<adjust>.parquet   每只每口径一个文件（可断点续传）
  data_tx/download_log.jsonl              追加式下载日志
  data_tx/_summary.json                   运行摘要

用法
----
  python qr_download_tx.py                 # 全量（可重复运行，已下载的跳过）
  python qr_download_tx.py --limit 20      # 冒烟测试
  python qr_download_tx.py --workers 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QLIB = os.path.join(ROOT, r"Quant\data\qlib_data\cn_data")
OUT = os.path.join(ROOT, "data_tx")
RAW = os.path.join(OUT, "raw")
LOG = os.path.join(OUT, "download_log.jsonl")
SUMMARY = os.path.join(OUT, "_summary.json")

URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://gu.qq.com/",
}
START = "1990-01-01"
END = dt.date.today().isoformat()
MAX_ROWS = 2000          # 服务端硬上限，>2000 返回 param error
PAGE_CAP = 12            # 单只最多翻页数（12*2000=24000 行，足够）
COLS = ["date", "open", "close", "high", "low", "volume", "turnover", "amount"]

_log_lock = threading.Lock()


def log(rec: dict) -> None:
    with _log_lock:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def to_tx(code: str) -> str:
    c = str(code).zfill(6)
    if c[0] in "69":
        return "sh" + c
    if c[0] in "03":
        return "sz" + c
    if c[0] in "48":
        return "bj" + c
    return "sh" + c


def build_universe() -> list[str]:
    """qlib 逐日成分股 ∪ 当前官方成分股 → 6 位代码列表。"""
    codes: set[str] = set()
    pit = os.path.join(QLIB, "instruments", "csi500.txt")
    with open(pit, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                codes.add(parts[0][-6:])
    n_pit = len(codes)
    try:
        import akshare as ak
        cons = ak.index_stock_cons_csindex(symbol="000905")
        for v in cons.iloc[:, 4]:
            codes.add(str(v).zfill(6))
        print(f"[universe] qlib PIT={n_pit}  current snapshot=500  "
              f"union={len(codes)}")
    except Exception as e:
        print(f"[universe] current snapshot failed ({type(e).__name__}); "
              f"using PIT only = {len(codes)}")
    return sorted(codes)


def _one_page(session: requests.Session, symbol: str, end: str,
              adjust: str, retries: int = 4):
    params = {
        "_var": f"kline_day{adjust}",
        "param": f"{symbol},day,{START},{end},{MAX_ROWS},{adjust}",
        "r": "0.8205512681390605",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(URL, params=params, timeout=30, headers=HEADERS)
            txt = r.text
            i = txt.find("={")
            if i < 0:
                raise ValueError(f"unexpected payload: {txt[:80]!r}")
            j = json.loads(txt[i + 1:])
            if j.get("msg"):
                # param error 之类，重试无益
                raise ValueError(f"api msg={j['msg']!r}")
            node = (j.get("data") or {}).get(symbol)
            if not isinstance(node, dict):
                return None
            for k in ("day", "qfqday", "hfqday"):
                if isinstance(node.get(k), list):
                    return node[k]
            return []
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (attempt + 1) + 0.2 * attempt)
    raise last_err


def _rows_to_df(rows: list, symbol: str) -> pd.DataFrame:
    recs = []
    for r in rows:
        if not isinstance(r, list) or len(r) < 9:
            continue
        recs.append(r[:5] + [r[5], r[7], r[8]])
    df = pd.DataFrame(recs, columns=COLS)
    for c in ("open", "close", "high", "low", "volume", "turnover", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 成交量：腾讯主板/创业板单位为手 → 股；科创板与指数已是股
    if not symbol.startswith(("sh688", "sz399", "sh000", "sz000")):
        df["volume"] = df["volume"] * 100
    df["turnover"] = df["turnover"] / 100.0
    df["amount"] = df["amount"] * 10000.0
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    df["instrument"] = symbol[-6:]
    df["adjust"] = ""
    return df


def fetch_symbol(session: requests.Session, code: str, adjust: str) -> pd.DataFrame:
    """向前翻页直到覆盖全历史。"""
    symbol = to_tx(code)
    collected: dict[str, list] = {}
    end = END
    for _ in range(PAGE_CAP):
        rows = _one_page(session, symbol, end, adjust)
        if rows is None:
            raise ValueError("no data node")
        if not rows:
            break
        added = 0
        for r in rows:
            if isinstance(r, list) and r and r[0] not in collected:
                collected[r[0]] = r
                added += 1
        earliest = min(r[0] for r in rows if isinstance(r, list) and r)
        if added == 0:
            break
        y, m, d = map(int, earliest.split("-"))
        end = (dt.date(y, m, d) - dt.timedelta(days=1)).isoformat()
        if end <= START:
            break
        time.sleep(0.25)
    if not collected:
        return pd.DataFrame(columns=COLS + ["instrument", "adjust"])
    rows = [collected[k] for k in sorted(collected)]
    df = _rows_to_df(rows, symbol)
    df["adjust"] = adjust
    return df


FMT = "parquet"   # 由 main() 覆盖；pyarrow 不可用时退化为 csv.gz


def _ext() -> str:
    return "parquet" if FMT == "parquet" else "csv.gz"


def process_one(code: str, adjust: str) -> dict:
    symbol = to_tx(code)
    path = os.path.join(RAW, f"{symbol}.{adjust or 'raw'}.{_ext()}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return {"symbol": symbol, "adjust": adjust, "status": "skipped"}
    session = requests.Session()
    t0 = time.time()
    try:
        df = fetch_symbol(session, code, adjust)
        if df.empty:
            return {"symbol": symbol, "adjust": adjust, "status": "empty",
                    "elapsed": round(time.time() - t0, 2)}
        tmp = path + ".tmp"
        if FMT == "parquet":
            df.to_parquet(tmp, index=False, compression="zstd")
        else:
            df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)
        return {"symbol": symbol, "adjust": adjust, "status": "ok",
                "rows": len(df), "first": str(df["date"].iloc[0]),
                "last": str(df["date"].iloc[-1]),
                "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"symbol": symbol, "adjust": adjust, "status": "fail",
                "error": f"{type(e).__name__}: {str(e)[:90]}",
                "elapsed": round(time.time() - t0, 2)}
    finally:
        session.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--adjusts", default=",qfq",
                    help="逗号分隔；空串代表不复权，如 ',qfq'")
    ap.add_argument("--format", default="auto", choices=["auto", "parquet", "csv"],
                    help="存储格式；auto 时 pyarrow 可用则用 parquet")
    args = ap.parse_args()

    global FMT
    if args.format == "auto":
        try:
            import pyarrow  # noqa: F401
            FMT = "parquet"
        except ImportError:
            FMT = "csv"
            print("[warn] pyarrow 不可用，回退到 csv.gz（后可用 convert 脚本转 parquet）")
    else:
        FMT = args.format
    print(f"[format] {FMT}")

    os.makedirs(RAW, exist_ok=True)
    adjusts = [a.strip() for a in args.adjusts.split(",")]

    codes = build_universe()
    if args.limit:
        codes = codes[:args.limit]
    tasks = [(c, a) for c in codes for a in adjusts]

    print(f"[plan] symbols={len(codes)}  adjust variants={adjusts}  "
          f"tasks={len(tasks)}  workers={args.workers}")
    print(f"[plan] window {START} .. {END}   output={RAW}")

    stats = {"ok": 0, "skipped": 0, "empty": 0, "fail": 0}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, c, a): (c, a) for c, a in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            stats[rec["status"]] = stats.get(rec["status"], 0) + 1
            log({**rec, "ts": dt.datetime.now().isoformat(timespec="seconds")})
            done += 1
            if done % 25 == 0 or rec["status"] == "fail":
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(tasks) - done) / rate / 60 if rate else 0
                print(f"  [{done}/{len(tasks)}] ok={stats['ok']} "
                      f"skip={stats['skipped']} fail={stats['fail']} "
                      f"empty={stats['empty']}  {rate:.2f} task/s  "
                      f"ETA {eta:.1f} min", flush=True)
                if rec["status"] == "fail":
                    print(f"     FAIL {rec['symbol']} {rec['adjust']!r}: "
                          f"{rec.get('error')}", flush=True)

    summary = {
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "symbols": len(codes), "tasks": len(tasks), **stats,
    }
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[done]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
