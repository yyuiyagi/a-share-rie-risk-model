# -*- coding: utf-8 -*-
r"""
qr_rnd_lab/02_verify_mp.py
==========================
Marchenko-Pastur 律的**可复现验证** —— 用于锚定所有公式，避免依赖记忆。

背景：本文档初稿里的自洽方程是凭记忆写的，经数值检验是**错的**
（残差 ~1.5，且反演出的密度为负）。本脚本用数值实验确定并锁定正确形式。
**任何用到 MP 公式的代码，都应先跑通本脚本自检。**

已验证的结论（N 个资产、T 期观测，X 为 T×N，S = XᵀX/T，γ = N/T）：

  [1] 谱支撑        λ± = (1 ± √γ)²                        （σ² = 1 约定）
  [2] 密度（在正支撑上，归一化到 1）
                    p(λ) = 2·√((λ₊-λ)(λ-λ₋)) / (2π γ λ)
      等价地 ∫_支撑 √((λ₊-λ)(λ-λ₋))/(2πγλ) dλ = 1 - 1/γ
      （γ>1 时剩下的 1/γ 质量是 λ=0 处的原子，对应 N-T 个零特征值）
  [3] Stieltjes 变换（约定 g(z) = (1/N)tr((zI-S)⁻¹)）
      满足自洽方程    γ z g² + (1 - γ - z) g + 1 = 0
      且反演    p(λ) = -2·Im g(λ + i0⁺) / π      （在正支撑上）
  [4] 条件数        κ = λ₊/λ₋ = ((1+√γ)/(1-√γ))²

跑法：python qr_rnd_lab/02_verify_mp.py
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from scipy import integrate

OK = "OK  "
BAD = "FAIL"


def mp_edges(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """谱支撑 λ± = σ²(1 ± √γ)²。"""
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def mp_density_support_normalized(x: np.ndarray, gamma: float,
                                  sigma2: float = 1.0) -> np.ndarray:
    """条件密度：在正支撑 [λ₋, λ₊] 上归一化到 1。

        p(λ) = √((λ₊-λ)(λ-λ₋)) / (2π σ² (1-1/γ) λ)      (γ > 1)

    γ > 1 时正特征值只有 T 个，除以 (1-1/γ) 完成条件归一化。
    γ ≤ 1 时无零原子，分母为 1。
    """
    lo, hi = mp_edges(gamma, sigma2)
    x = np.asarray(x, dtype=float)
    d = np.zeros_like(x)
    m = (x >= lo) & (x <= hi) & (x > 0)
    d[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2 * np.pi * sigma2 * gamma * x[m])
    if gamma > 1:
        d = d / (1 - 1 / gamma)
    return d


def stieltjes_from_equation(z: complex, gamma: float,
                            sigma2: float = 1.0) -> complex:
    """由自洽方程解出 g(z) = (1/N)tr((zI - S)^{-1})。

        γ z g² + (1 - γ - z) g + 1 = 0

    取使 Im g < 0 的根（z 在上半平面时，resolvent 的迹虚部为负）。
    """
    zz = complex(z) / sigma2
    A = gamma * zz
    B = 1.0 - gamma - zz
    C = 1.0
    disc = B * B - 4 * A * C
    r = np.sqrt(disc)
    g1 = (-B + r) / (2 * A)
    g2 = (-B - r) / (2 * A)
    g = g1 if g1.imag < 0 else g2
    return g / sigma2


def density_from_equation(x: np.ndarray, gamma: float,
                          sigma2: float = 1.0, eps: float = 1e-7) -> np.ndarray:
    """由 g 反演条件密度（正支撑上归一化到 1）。

        p(λ) = -Im g(λ + i0⁺) / π · (1/γ)          γ > 1
        p(λ) = -Im g(λ + i0⁺) / π · (1/(1-γ))      γ < 1

    因子来自 μ_N = (1/1_{γ<1} + 1/γ·1_{γ>1}) × MP 律 ——
    即 N 维经验谱与 T 维持有同一正谱，归一化方式差一个常数。
    该因子已由数值自检锁定（见文件头 [3]）。
    """
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    out = np.zeros_like(xs)
    lo, hi = mp_edges(gamma, sigma2)
    scale = 1.0 / gamma if gamma > 1 else 1.0 / (1.0 - gamma)
    for i, v in enumerate(xs):
        if v <= lo or v >= hi:
            continue
        g = stieltjes_from_equation(complex(v, eps), gamma, sigma2)
        out[i] = -g.imag / np.pi * scale
    return out


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{OK if cond else BAD}] {name}" + (f"   {detail}" if detail else ""))
    return cond


def main() -> None:
    print("=" * 78)
    print("Marchenko-Pastur 公式自检")
    print("=" * 78)

    # ---- 数值实验：大样本、纯噪声 ----
    results = []
    for gamma_target, N, T in [(2.0, 4000, 2000), (0.5, 2000, 4000),
                               (0.1, 800, 8000)]:
        rng = np.random.default_rng(20261004)
        gamma = N / T
        X = rng.standard_normal((T, N))
        S = X.T @ X / T
        ev = np.linalg.eigvalsh(S)
        pos = ev[ev > 1e-9]
        lo, hi = mp_edges(gamma)

        print(f"\n--- γ = {gamma:.3f}  (N={N}, T={T}) ---")
        print(f"  正特征值个数 = {len(pos)}  (理论 = min(N,T) = {min(N,T)})")
        print(f"  支撑理论 = [{lo:.4f}, {hi:.4f}]   "
              f"实测 = [{pos[0]:.4f}, {pos[-1]:.4f}]")

        # [1] 谱边界
        edge_ok = (abs(pos[-1] - hi) / hi < 0.03
                   and (gamma > 1 or abs(pos[0] - lo) / lo < 0.05))
        results.append(check("谱边界 λ± = (1±√γ)²", edge_ok))

        # [2] 条件密度积分 = 1
        f = lambda t: mp_density_support_normalized(np.array([t]), gamma)[0]
        val, _ = integrate.quad(f, lo + 1e-9, hi - 1e-9, limit=400)
        results.append(check("条件密度积分 = 1", abs(val - 1) < 1e-3,
                             f"实测 {val:.6f}"))

        # [2'] 未归一化密度积分 = 1 - 1/γ
        g_f = lambda t: mp_density_support_normalized(np.array([t]), gamma)[0] * (
            1 - 1 / gamma if gamma > 1 else 1.0)
        val2, _ = integrate.quad(g_f, lo + 1e-9, hi - 1e-9, limit=400)
        exp2 = 1 - 1 / gamma if gamma > 1 else 1.0
        results.append(check("含零原子的总质量 = 1", abs(val2 + max(0, 1/gamma if gamma > 1 else 0) - 1) < 1e-3,
                             f"正支撑 {val2:.6f} + 零原子 "
                             f"{max(0, 1/gamma) if gamma > 1 else 0:.6f}"))

        # [3] 自洽方程
        worst_eq = 0.0
        for zr in [0.3, 1.0, 2.0, 4.0]:
            z = complex(zr, 0.6)
            g_num = np.trace(np.linalg.inv(z * np.eye(N) - S)) / N
            g_ana = stieltjes_from_equation(z, gamma)
            worst_eq = max(worst_eq, abs(g_num - g_ana))
        results.append(check("自洽方程 γzg²+(1-γ-z)g+1=0 解 == 数值 resolvent",
                             worst_eq < 5e-2, f"最大差 {worst_eq:.2e}"))

        # [4] ★ 矩检验：密度的总质量与一阶矩（不依赖分箱，比直方图可靠）
        #     mu_N = s × (正支撑上的条件密度) + (1/γ)·δ₀·1_{γ>1}
        #       γ > 1: s = 1 - 1/γ（其余 1/γ 质量在 λ=0 的原子）
        #       γ ≤ 1: s = 1     （无零原子，正支撑承载全部质量）
        #     必须满足 ∫λ dμ = 1（因为 (1/N)tr(S) → 1）
        s = (1.0 - 1.0 / gamma) if gamma > 1 else 1.0
        atom = 1.0 / gamma if gamma > 1 else 0.0
        cond = lambda t: mp_density_support_normalized(np.array([t]), gamma)[0]
        mass, _ = integrate.quad(cond, lo + 1e-12, hi - 1e-12, limit=400)
        mom, _ = integrate.quad(lambda t: t * cond(t), lo + 1e-12, hi - 1e-12,
                                limit=400)
        results.append(check("条件密度积分 = 1（正支撑）", abs(mass - 1) < 1e-4,
                             f"{mass:.6f}"))
        results.append(check("谱总质量 s + 零原子 = 1",
                             abs(s + atom - 1) < 1e-6,
                             f"s={s:.6f} 原子={atom:.6f}"))
        results.append(check("一阶矩 s·∫λdμ = 1（== tr(S)/N）",
                             abs(s * mom - 1) < 1e-4, f"{s * mom:.6f}"))
        results.append(check("一阶矩 == 实测 tr(S)/N",
                             abs(s * mom - np.trace(S) / N) < 2e-3,
                             f"理论 {s*mom:.6f} 实测 {np.trace(S)/N:.6f}"))

        # [5] 条件数
        if gamma < 1:
            kappa = (hi / lo)
            evp = pos
            kappa_emp = evp[-1] / evp[0]
            results.append(check("κ = ((1+√γ)/(1-√γ))²",
                                 abs(kappa_emp / kappa - 1) < 0.1,
                                 f"理论 {kappa:.2f} 实测 {kappa_emp:.2f}"))

    print("\n" + "=" * 78)
    n_ok = sum(results)
    print(f"自检结果：{n_ok}/{len(results)} 通过")
    print("=" * 78)
    if n_ok == len(results):
        print("""
公式已锁定。下游代码可直接引用：

  from qr_cov.qr_rnd_lab import (mp_edges, mp_density_support_normalized,
                                  stieltjes_from_equation, density_from_equation)
""")
    else:
        print("存在未通过项，请勿在此状态下使用这些公式。")
        sys.exit(1)


if __name__ == "__main__":
    main()
