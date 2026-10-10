# -*- coding: utf-8 -*-
r"""
qr_rnd_lab/01_mp_experiment.py
==============================
Marchenko-Pastur 律的数值实验 —— 亲手验证，而不是相信。

跑法：
    python qr_rnd_lab/01_mp_experiment.py              # 只看控制台
    python qr_rnd_lab/01_mp_experiment.py --out r.txt  # 同时写入 UTF-8(BOM) 文件
依赖：numpy, scipy, scikit-learn（仅 LW 收缩用到）
输出：控制台表格

实验清单（对应学习文档第 4、5 节）：
  E1  纯噪声下的经验谱 vs MP 理论密度
  E2  最大/最小特征值的收敛：是不是 σ²(1±√γ)²？
  E3  【核心】真实 Σ = I 时，样本特征值仍然展开 → 虚假分散
  E4  病态程度：条件数 κ 随 γ 的变化
  E5  【核心】Σ = I 时样本求逆给出的最小方差权重是噪声
  E6  低秩信号 + 噪声：信号特征值何时能从噪声带里『冒出来』
  E7  真实规模（N=500, T=252）下的完整对比：强因子 vs 弱因子
"""
from __future__ import annotations

import sys
import argparse
import atexit

# Windows 控制台默认 GBK，输出 λ₋ / γ 这类字符会崩
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class _Tee:
    """把输出同时写到控制台和一个 UTF-8(BOM) 文件。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=None,
                 help="额外把输出写入该文件（UTF-8 with BOM，避免 shell 重定向乱码）")
_args = _ap.parse_args()

if _args.out:
    _f = open(_args.out, "w", encoding="utf-8-sig", newline="\n")
    sys.stdout = _Tee(sys.stdout, _f)
    atexit.register(lambda: (sys.stdout.flush(), _f.close()))

import numpy as np

rng = np.random.default_rng(20261004)


# ----------------------------------------------------------------------
# 基础工具：MP 律
# ----------------------------------------------------------------------
def mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    """MP 律的谱支撑边界 λ± = σ²(1 ± √γ)²。"""
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def mp_density(x: np.ndarray, sigma2: float, gamma: float) -> np.ndarray:
    """MP 密度（不含 x=0 处的原子）。

        p(λ) = √((λ₊-λ)(λ-λ₋)) / (2π σ² γ λ),   λ ∈ [λ₋, λ₊]

    当 γ > 1 时 λ₋ = 0，且在 0 处有质量 1 - 1/γ 的原子。
    """
    lo, hi = mp_edges(sigma2, gamma)
    d = np.zeros_like(x, dtype=float)
    m = (x >= lo) & (x <= hi) & (x > 0)
    d[m] = (np.sqrt((hi - x[m]) * (x[m] - lo))
            / (2 * np.pi * sigma2 * gamma * x[m]))
    return d


def header(t: str) -> None:
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def ledoit_wolf_linear(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 线性收缩，直接用 sklearn 的审计实现。

    返回 (收缩后的协方差 Σ̂ = (1-λ)S + λμI, 收缩强度 λ*)。
    """
    from sklearn.covariance import ledoit_wolf
    Sig, lam = ledoit_wolf(X, assume_centered=False)
    return Sig, float(lam)


def mp_clip(S: np.ndarray, gamma: float, sigma2: float | None = None):
    """MP 特征值裁剪（Laloux-Cizeau-Bouchaud-Potters 1999）。

    λ > λ₊ 的特征值视为携带真实信号 → 保留；
    其余视为噪声 → 统一替换为其平均值（保持迹不变 = 总方差不变）。

    返回 (裁剪后的 Σ, 保留的特征值个数)。
    """
    N = S.shape[0]
    ev, V = np.linalg.eigh(S)
    if sigma2 is None:
        sigma2 = ev.mean()
    _, hi = mp_edges(sigma2, gamma)
    keep = ev > hi
    n_keep = int(keep.sum())
    ev_clean = ev.copy()
    if n_keep < N:
        ev_clean[~keep] = ev[~keep].mean()
    return (V * ev_clean) @ V.T, n_keep


def min_var_weights(Sig: np.ndarray) -> np.ndarray:
    """w ∝ Σ⁻¹1，再归一化到 Σw = 1。"""
    w = np.linalg.solve(Sig, np.ones(Sig.shape[0]))
    return w / w.sum()


# ----------------------------------------------------------------------
# E1  纯噪声下的经验谱 vs MP 律
# ----------------------------------------------------------------------
header("E1  纯噪声下的经验谱 vs MP 理论密度")
print("设定：真实 Σ = I（资产之间零相关、等方差）")
print("      样本 S = XᵀX/T 的特征值分布应当收敛到 MP 律")
print()
print(f"{'N':>6} {'T':>6} {'γ=N/T':>8} {'λ₋理论':>9} {'λ₊理论':>9} "
      f"{'λ最小实测':>11} {'λ最大实测':>11} {'均值':>8}")
for N, T in [(50, 252), (150, 252), (500, 252), (500, 1000), (500, 5000)]:
    g = N / T
    X = rng.standard_normal((T, N))
    ev = np.linalg.eigvalsh(X.T @ X / T)
    lo, hi = mp_edges(1.0, g)
    print(f"{N:>6} {T:>6} {g:>8.3f} {lo:>9.4f} {hi:>9.4f} "
          f"{ev[0]:>11.4f} {ev[-1]:>11.4f} {ev.mean():>8.4f}")
print("\n观察：N=500, T=252 时 γ = N/T = 1.984 > 1，此时 λ₋ = 0，")
print("      MP 律在 0 处还有一个质量 1 - 1/γ ≈ 0.496 的原子（S 奇异）。")
print("      关键：『真实协方差是单位矩阵』，样本特征值照样从 0 铺到 λ₊。")
print("      这种分散完全是有限样本的产物，不是真实结构。")

# ----------------------------------------------------------------------
# E2  谱边界的收敛
# ----------------------------------------------------------------------
header("E2  最大/最小特征值的收敛速率")
print("固定 N=500，让 T 增长，看实测谱边界是否逼近理论值")
print()
print(f"{'T':>7} {'γ':>8} {'λ₋理论':>9} {'λ₋实测':>9} {'λ₊理论':>9} {'λ₊实测':>9}")
N = 500
for T in [200, 300, 500, 1000, 2000, 5000, 20000]:
    g = N / T
    X = rng.standard_normal((T, N))
    ev = np.linalg.eigvalsh(X.T @ X / T)
    lo, hi = mp_edges(1.0, g)
    print(f"{T:>7} {g:>8.3f} {lo:>9.4f} {ev[0]:>9.4f} {hi:>9.4f} {ev[-1]:>9.4f}")
print("\n观察：γ→0 时 λ± → 1，谱塌缩成一个点。")
print("      维数灾难不是『模型不够好』，而是 γ 不够小。")

# ----------------------------------------------------------------------
# E3  核心：Σ=I 时最小方差组合已经失效
# ----------------------------------------------------------------------
header("E3  核心：真实 Σ = I 时最小方差组合已经失效")
print("真实 Σ = I ⇒ 最小方差组合理论上就是等权 w = 1/N，权重全相等。")
print("现在用样本协方差求逆：w ∝ S⁻¹1，看它给出什么。")
print()
print(f"{'N':>6} {'T':>6} {'γ':>7} {'max w':>9} {'权重方差':>11} {'有效持仓数':>11}")
for N, T in [(20, 252), (50, 252), (100, 252), (300, 252), (500, 252)]:
    g = N / T
    X = rng.standard_normal((T, N))
    S = X.T @ X / T
    if np.linalg.matrix_rank(S) < N:
        # γ>1 时 S 奇异，用伪逆
        w = np.linalg.pinv(S) @ np.ones(N)
    else:
        w = np.linalg.inv(S) @ np.ones(N)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w ** 2)
    print(f"{N:>6} {T:>6} {g:>7.3f} {w.max():>9.4f} "
          f"{np.var(w):>11.3e} {n_eff:>11.1f}")
print("\n观察：理论答案是等权（有效持仓数 = N）。")
print("      但 N=500 时有效持仓数极小 —— 权重被堆到噪声方向上。")
print("      『真实协方差是单位阵』这个最简单的问题，样本求逆都做不对。")

# ----------------------------------------------------------------------
# E4  条件数
# ----------------------------------------------------------------------
header("E4  条件数：为什么求逆是灾难")
print(f"{'γ':>7} {'κ理论≈':>10} {'κ实测中位':>11}")
for g in [0.1, 0.3, 0.5, 0.8, 0.95, 1.0, 2.0, 4.0]:
    N = 400
    T = int(round(N / g))
    ks = []
    for _ in range(5):
        X = rng.standard_normal((T, N))
        ev = np.linalg.eigvalsh(X.T @ X / T)
        if ev[0] <= 0:
            continue
        ks.append(ev[-1] / ev[0])
    theo = ((1 + np.sqrt(g)) / (1 - np.sqrt(g))) ** 2 if g < 1 else np.inf
    med = np.median(ks) if ks else float("nan")
    print(f"{g:>7.2f} {theo:>10.2f} {med:>11.2f}")
print("\n观察：γ→1 时 κ→∞；γ>1 时 S 奇异（T<N），根本不可逆。")
print("      这就是为什么 N=500、T=252 时不能直接对 S 求逆。")

# ----------------------------------------------------------------------
# E5  裁剪与收缩
# ----------------------------------------------------------------------
header("E5  修复方式一：收缩（真实 Σ = I，N=500，T=252）")
N, T = 500, 252
X = rng.standard_normal((T, N))
S = X.T @ X / T
Sig_lw, lam = ledoit_wolf_linear(X)
print()
print(f"{'方法':<26} {'有效持仓数':>11} {'max w':>9} {'组合真实方差':>13}")
print("-" * 62)
w_eq = np.ones(N) / N
print(f"{'理论最优（等权）':<26} {1/np.sum(w_eq**2):>11.1f} "
      f"{w_eq.max():>9.5f} {w_eq @ np.eye(N) @ w_eq:>13.6f}")
w_s = min_var_weights(S)
print(f"{'样本 S 求逆':<26} {1/np.sum(w_s**2):>11.1f} "
      f"{w_s.max():>9.5f} {w_s @ np.eye(N) @ w_s:>13.6f}")
w_lw = min_var_weights(Sig_lw)
print(f"{'LW 收缩 (λ*=' + f'{lam:.3f}' + ')':<26} {1/np.sum(w_lw**2):>11.1f} "
      f"{w_lw.max():>9.5f} {w_lw @ np.eye(N) @ w_lw:>13.6f}")
print("\n观察：真实方差越小越好（理论最优 = 1/N = 0.002）。")
print("      样本求逆的组合真实方差被放大两个数量级，它在赌噪声方向。")
print("      收缩把噪声特征值拉平，几乎恢复到理论最优。")

# ----------------------------------------------------------------------
# E6  低秩信号能否冒出噪声带
# ----------------------------------------------------------------------
header("E6  低秩信号 + 噪声：信号能否从噪声带里冒出来")
print("真实模型：Σ = I + ρ·vvᵀ（一个真实因子，强度 ρ）")
print("问题：这个因子的样本特征值何时能超出噪声带上界 λ₊？")
print()
N, T = 300, 252
g = N / T
_, hi = mp_edges(1.0, g)
print(f"N={N}, T={T} ⇒ γ={g:.3f}, 噪声带上界 λ₊={hi:.4f}")
print()
print(f"{'因子强度 ρ':>12} {'真特征值':>10} {'样本λ最大':>11} {'冒出噪声带?':>14}")
for rho in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)
    Sigma = np.eye(N) + rho * np.outer(v, v)
    L = np.linalg.cholesky(Sigma)
    X = rng.standard_normal((T, N)) @ L.T
    ev = np.linalg.eigvalsh(X.T @ X / T)
    seen = "是" if ev[-1] > hi * 1.02 else "否（被淹没）"
    print(f"{rho:>12.1f} {1 + rho:>10.3f} {ev[-1]:>11.4f} {seen:>14}")
print("\n观察：真实特征值低于 λ₊ 的因子会被噪声带吞掉，样本谱看不见它。")
print("      只有真实特征值明显高于 λ₊ 的因子才会『冒出来』。")
print("      这正是特征值裁剪能工作的原因，也是它的适用边界。")

# ----------------------------------------------------------------------
# E7  真实规模：强因子 vs 弱因子
# ----------------------------------------------------------------------
header("E7  真实规模（N=500, T=252）下的完整对比")
print("从目标特征值直接反构真实 Σ = U diag(s) Uᵀ + σ²I（U 正交），")
print("以便精确控制信号相对噪声带 λ₊ 的位置。")
print()

N, T = 500, 252
GAMMA = N / T


def build_case(sig_eigs, idio, seed):
    g = np.random.default_rng(seed)
    A = g.standard_normal((N, len(sig_eigs)))
    U, _ = np.linalg.qr(A)
    U = U[:, :len(sig_eigs)]
    Sig = U @ np.diag(sig_eigs) @ U.T + idio * np.eye(N)
    L = np.linalg.cholesky(Sig)
    X = g.standard_normal((T, N)) @ L.T
    return Sig, X, X.T @ X / T


def report_case(name, sig_eigs, idio, seed):
    print("-" * 76)
    print(f"场景 {name}：信号特征值 = {sig_eigs}，特质方差 σ² = {idio}")
    print("-" * 76)
    Sig, X, S = build_case(sig_eigs, idio, seed)
    lo, hi = mp_edges(idio, GAMMA)
    ev_t = np.linalg.eigvalsh(Sig)
    ev_s = np.linalg.eigvalsh(S)
    print(f"  γ = {GAMMA:.3f}    噪声带 = [{lo:.3f}, {hi:.3f}]")
    print(f"  真实 Σ 最大特征值 = {ev_t[-1]:.3f}  "
          f"→ {'高于' if ev_t[-1] > hi else '低于'} λ₊（"
          f"{'可识别' if ev_t[-1] > hi else '会被噪声淹没'}）")
    print(f"  样本 S 最大特征值 = {ev_s[-1]:.3f}  "
          f"(相对真实值被推高 {(ev_s[-1] / ev_t[-1] - 1) * 100:+.0f}%)")
    print()
    print("  σ² 的估计误差如何影响裁剪判定：")
    bulk = np.sort(ev_s)[:int(N * 0.8)].mean()
    for nm, s2 in [("tr(S)/N（倾向高估）", np.trace(S) / N),
                   ("最小 80% 均值（倾向低估）", bulk),
                   ("真实值（上帝视角，不可得）", idio)]:
        _, h = mp_edges(s2, GAMMA)
        print(f"    {nm:<26} σ̂²={s2:>8.3f}  λ₊={h:>8.3f}  "
              f"判为信号的方向数={int((ev_s > h).sum()):>4}")
    print()
    print("  组合表现（用真实 Σ 评价，故为『上帝视角』）：")
    print(f"    {'方法':<28} {'有效持仓':>9} {'最大权重':>9} {'真实方差':>13}")
    Sig_lw, lam = ledoit_wolf_linear(X)
    Sig_clip, nk = mp_clip(S, GAMMA, sigma2=idio)
    rows = [("等权（参照）", np.ones(N) / N),
            ("真实 Σ（上帝视角）", min_var_weights(Sig)),
            ("样本 S 求逆", min_var_weights(S)),
            (f"LW 收缩 (λ*={lam:.3f})", min_var_weights(Sig_lw)),
            (f"MP 裁剪 (σ²已知,保留{nk})", min_var_weights(Sig_clip))]
    for nm, w in rows:
        rv = float(w @ Sig @ w)
        print(f"    {nm:<28} {1/np.sum(w**2):>9.1f} {w.max():>9.4f} {rv:>13.6e}")
    print()


print("两个场景对比：信号高于噪声带（容易） vs 信号落在噪声带内（困难）")
print()
report_case("A  强因子", [60.0, 40.0, 25.0, 15.0, 8.0], 0.70, 11)
report_case("B  弱因子", [2.3, 1.5, 1.0, 0.9, 0.8], 0.70, 21)

print("=" * 76)
print("总结：本实验建立的核心直觉")
print("=" * 76)
print("""
1. 样本协方差的特征值谱总是『展开』的，即使真实协方差是单位阵。
   这是有限样本的数学必然（MP 律），不是模型缺陷。

2. 展开幅度由 γ = N/T 控制。γ→1 时条件数爆炸，γ>1 时 S 奇异。

3. 最小方差组合对 Σ⁻¹ 的误差极其敏感：权重被堆到噪声特征向量上，
   表现为『有效持仓数远小于 N』、真实方差被放大若干倍。

4. 修复的本质是同一件事：压低噪声特征值的分散度。
   收缩（向 I）与裁剪（截断噪声带）是同一思想的两种实现。

5. 裁剪的前提是知道噪声带边界 λ₊ = σ²(1+√γ)²，而 σ² 的估计会被信号污染：
   高估 → 真信号被误当噪声抹掉；低估 → 噪声被误当真信号保留。
   这是论文到实盘之间最大的裂缝。

6. 真实特征值低于 λ₊ 的因子会被噪声带淹没，样本谱看不见它。
   这解释了因子挖掘存在信息论意义上的天花板 ——
   与你在 前置.md 里用互信息推出的结论是同一件事的两面。
""")
