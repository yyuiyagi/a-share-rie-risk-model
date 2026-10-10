# -*- coding: utf-8 -*-
r"""
在 WSL 中用 qlib(Alpha158 + LightGBM) 构建 α，再接入 RIE 风控做有效前沿对比。

数据：/home/yui/Quant/data/qlib_data/cn_data（1999-2020）
流程：
  1. qlib.init + Alpha158 + DatasetH(train/valid/test) + LGBModel（复刻 config.yaml）
  2. test 段预测 μ（1 日前向收益的 rank 代理）
  3. 构建 test 段日收益面板 R（datetime × instrument）
  4. 月频再平衡：naive top-k 等权 vs 长仓 min-var vs 长仓切点（RIE 去噪协方差）
  5. 有效前沿（min-var 与切点混合）

运行（在 WSL）：
  /home/yui/Quant/.venv/bin/python 本脚本
"""
import os
import sys

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158


class Alpha158_20d(Alpha158):
    """Alpha158 因子，标签改成 20 日前向收益（匹配月频调仓）。"""

    def get_label_config(self):
        return ["Ref($close, -21)/Ref($close, -1) - 1"], ["LABEL0"]

# 我们的 RIE 去噪（纯 numpy）
sys.path.insert(0, "/mnt/c/Users/Yui_Y/Desktop/Work/Study Notes/qr_cov")
from denoise import estimate_sigma2_bulk, denoise_spiked


def long_only_solve(Sigma, b, max_iter=300):
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
    """clip-and-renormalize 权重上限（启发式）。"""
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


def static_exposures(est):
    """横截面暴露：vol=窗口 std、beta=对等权市场回归斜率，各横截面 z-score。est (T,N)。"""
    T, N = est.shape
    vol = est.std(axis=0)
    mkt = est.mean(axis=1)
    var_mkt = float(mkt.var())
    est_c = est - est.mean(axis=0, keepdims=True)
    mkt_c = mkt - mkt.mean()
    beta = (est_c.T @ mkt_c) / T / var_mkt

    def zs(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    return zs(vol), zs(beta)


def strict_neutral_solve(Sigma, b, F, cap=0.05, max_iter=300):
    """严格因子中性：min 0.5 wᵀΣw - bᵀw s.t. w≥0, 1ᵀw=1, Fᵀw=0。
    b=1 为 min-var，b=μ 为切点。active-set + 拉格朗日。"""
    N = Sigma.shape[0]
    K = F.shape[1]
    active = np.ones(N, dtype=bool)
    for _ in range(max_iter):
        idx = np.where(active)[0]
        m = len(idx)
        if m <= K + 1:
            break
        S = Sigma[np.ix_(idx, idx)]
        Fi = F[idx]
        bi = b[idx]
        Sinv = np.linalg.pinv(S)
        ones = np.ones(m)
        a = float(ones @ Sinv @ ones)
        c = Fi.T @ Sinv @ ones          # (K,)
        C = Fi.T @ Sinv @ Fi            # (K,K)
        m1 = float(ones @ Sinv @ bi)
        mF = Fi.T @ Sinv @ bi           # (K,)
        M = np.zeros((K + 1, K + 1))
        M[0, 0] = a; M[0, 1:] = c; M[1:, 0] = c; M[1:, 1:] = C
        rhs = np.zeros(K + 1)
        rhs[0] = 1.0 - m1
        rhs[1:] = -mF
        try:
            sol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.pinv(M) @ rhs
        lam, eta = sol[0], sol[1:]
        w_sub = Sinv @ (bi + lam * ones + Fi @ eta)
        w = np.zeros(N)
        w[idx] = w_sub
        neg = idx[w_sub < 0]
        if len(neg) == 0:
            break
        active[neg] = False
    w = np.maximum(w, 0)
    s = w.sum()
    w = w / s if s > 0 else np.ones(N) / N
    return cap_weights(w, cap)


def main():
    qlib.init(provider_uri="/home/yui/Quant/data/qlib_data/cn_data",
              region=REG_CN, auto_mount=False, redis_port=-1)

    segments = {"train": ["2008-01-01", "2014-12-31"],
                "valid": ["2015-01-01", "2016-12-31"],
                "test": ["2017-01-01", "2020-09-25"]}
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158_20d",
                "module_path": __name__,
                "kwargs": {
                    "start_time": "2008-01-01",
                    "end_time": "2020-09-25",
                    "fit_start_time": "2008-01-01",
                    "fit_end_time": "2016-12-31",
                    "instruments": "csi500",
                },
            },
            "segments": segments,
        },
    }
    model_config = {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.1,
            "subsample": 0.8789, "lambda_l1": 205.6999, "lambda_l2": 580.9768,
            "max_depth": 8, "num_leaves": 210, "num_threads": 16, "seed": 42,
        },
    }

    print("训练 LightGBM(Alpha158) ...")
    dataset = init_instance_by_config(dataset_config)
    model = init_instance_by_config(model_config)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("pred").reorder_levels(["instrument", "datetime"]).sort_index()
    print(f"预测 {len(pred)} 条")

    # ---- 收益面板 ----
    test_insts = D.list_instruments(D.instruments("csi500"), as_list=True,
                                    start_time="2017-01-01", end_time="2020-09-25")
    close = D.features(test_insts, ["$close"], start_time="2017-01-01",
                       end_time="2020-09-25", freq="day")["$close"]
    R = close.groupby(level="instrument").pct_change(fill_method=None)
    R_wide = R.unstack(level="instrument")          # datetime × instrument
    R_wide = R_wide.dropna(axis=1, thresh=int(R_wide.shape[0] * 0.6))
    print(f"收益面板 {R_wide.shape}")

    # 对齐 pred 到面板
    pred_wide = pred.unstack(level="instrument")    # datetime × instrument
    pred_wide = pred_wide.reindex(index=R_wide.index, columns=R_wide.columns)

    X = R_wide
    Xarr = X.to_numpy(dtype=float)
    # 固定宇宙：覆盖率 ≥70% 的列，剩余 NaN 补 0（避免换手计算维度漂移）
    coverage = (~np.isnan(Xarr)).mean(axis=0)
    global_valid = coverage >= 0.7
    Xarr = np.nan_to_num(Xarr[:, global_valid])
    pred_arr = np.nan_to_num(pred_wide.to_numpy(dtype=float))[:, global_valid]
    N = Xarr.shape[1]
    T = Xarr.shape[0]
    gamma = N / 252.0
    print(f"固定宇宙 {N} 只（覆盖率≥70%）")

    rebal = list(range(252, T - 21, 21))
    COST = 0.001
    CAP = 0.05
    res = {"naive": {"gross": [], "net": []},
           "tangency": {"gross": [], "net": []},
           "minvar": {"gross": [], "net": []},
           "tangency_neut": {"gross": [], "net": []},
           "minvar_neut": {"gross": [], "net": []}}
    to_hist = {"naive": [], "tangency": [], "minvar": [],
               "tangency_neut": [], "minvar_neut": []}
    prev_w = {"naive": None, "tangency": None, "minvar": None,
              "tangency_neut": None, "minvar_neut": None}
    top_k = 50
    for t in rebal:
        est = Xarr[t - 252:t]
        Z = est - est.mean(axis=0)
        sd = est.std(axis=0); sd[sd == 0] = 1.0
        Z = Z / sd
        C = Z.T @ Z / Z.shape[0]
        ev, U = np.linalg.eigh(C)
        ev = ev[::-1]; U = U[:, ::-1]
        sigma2 = estimate_sigma2_bulk(ev, gamma)
        ev_rie = denoise_spiked(ev, sigma2, gamma, preserve_trace=True)
        Sigma_rie = np.diag(sd) @ (U @ np.diag(ev_rie) @ U.T) @ np.diag(sd)

        mu_v = pred_arr[t]

        vol_z, beta_z = static_exposures(est)
        F = np.column_stack([vol_z, beta_z])

        w_naive = np.zeros(N)
        order = np.argsort(mu_v)[::-1]
        w_naive[order[:top_k]] = 1.0 / top_k
        w_tang = cap_weights(long_only_solve(Sigma_rie, mu_v), CAP)
        w_minv = cap_weights(long_only_solve(Sigma_rie, np.ones(N)), CAP)
        w_tang_neut = strict_neutral_solve(Sigma_rie, mu_v, F, CAP)
        w_minv_neut = strict_neutral_solve(Sigma_rie, np.ones(N), F, CAP)

        ws = {"naive": w_naive, "tangency": w_tang, "minvar": w_minv,
              "tangency_neut": w_tang_neut, "minvar_neut": w_minv_neut}
        for m in ws:
            turnover = 1.0 if prev_w[m] is None else 0.5 * np.sum(
                np.abs(ws[m] - prev_w[m]))
            to_hist[m].append(turnover)
            cost = 2.0 * turnover * COST
            for s in range(t, min(t + 21, T)):
                gr = float(ws[m] @ Xarr[s])
                res[m]["gross"].append(gr)
                res[m]["net"].append(gr - (cost if s == t else 0.0))
            prev_w[m] = ws[m]

    def stat(r):
        return (r.std() * np.sqrt(252) * 100, r.mean() * 252 * 100,
                r.mean() / r.std() * np.sqrt(252))

    print("\nOOS 组合对比（qlib Alpha158 + RIE，成本 10bp + 权重上限 5%）")
    print(f"{'组合':<14} {'波动%':>8} {'毛收益%':>9} {'净收益%':>9} "
          f"{'毛夏普':>7} {'净夏普':>7} {'年换手':>8}")
    for m in ["naive", "minvar", "tangency", "minvar_neut", "tangency_neut"]:
        g = np.array(res[m]["gross"]); n = np.array(res[m]["net"])
        vg, rg, sg = stat(g); vn, rn, sn = stat(n)
        to = np.mean(to_hist[m])
        print(f"{m:<14} {vn:>8.2f} {rg:>9.2f} {rn:>9.2f} {sg:>7.3f} {sn:>7.3f} "
              f"{to*252/21:>8.1f}")

    r_min_net = np.array(res["minvar"]["net"])
    r_tang_net = np.array(res["tangency"]["net"])
    print("\n净收益有效前沿（min-var 与切点混合）")
    print(f"{'α(切点权重)':>12} {'波动%':>8} {'净收益%':>9} {'净夏普':>7}")
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = alpha * r_tang_net + (1 - alpha) * r_min_net
        v, rt, s = stat(r)
        print(f"{alpha:>12.2f} {v:>8.2f} {rt:>9.2f} {s:>7.3f}")

    r_naive_net = np.array(res["naive"]["net"])
    v_naive = stat(r_naive_net)[0]
    best = None
    for alpha in np.linspace(0, 1, 201):
        r = alpha * r_tang_net + (1 - alpha) * r_min_net
        v, rt, s = stat(r)
        if best is None or abs(v - v_naive) < abs(best[0] - v_naive):
            best = (v, rt, s, alpha)
    print(f"\n与 naive 同净波动({v_naive:.1f}%)的前沿点：")
    print(f"  波动 {best[0]:.2f}%  净收益 {best[1]:.2f}%  净夏普 {best[2]:.3f}  "
          f"(α={best[3]:.2f})")
    print(f"  vs naive：净收益 {stat(r_naive_net)[1]:.2f}% / "
          f"净夏普 {stat(r_naive_net)[2]:.3f}")


if __name__ == "__main__":
    main()
