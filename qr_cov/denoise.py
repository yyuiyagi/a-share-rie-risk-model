# -*- coding: utf-8 -*-
r"""
qr_cov/denoise.py —— 噪声尺度估计 + 尖峰 RIE 非线性收缩
========================================================
对应三步：
  step1  estimate_sigma2_bulk   迭代 bulk matching 求 σ²
  step2  denoise_spiked         RIE（spiked model）逐特征值非线性收缩
  step3  在 stage_b_denoise.py 里上真实数据 + 对比 factor baseline

数学（spiked model，Σ = σ²I + Σ_k θ_k u_k u_kᵀ）：
  样本 spike λ 与真实 spike θ 满足
      λ = θ + σ²(1+γ) + γ σ⁴ / θ
  反解（RIE）：
      θ = ( [λ - σ²(1+γ)] + sqrt([λ - σ²(1+γ)]² - 4 γ σ⁴) ) / 2
  去噪后的完整特征值 = σ² + θ。
"""
from __future__ import annotations

import numpy as np


def mp_edges(sigma2: float, gamma: float) -> tuple[float, float]:
    return (sigma2 * (1 - np.sqrt(gamma)) ** 2,
            sigma2 * (1 + np.sqrt(gamma)) ** 2)


def estimate_sigma2_bulk(ev: np.ndarray, gamma: float,
                         tol: float = 1e-6, max_iter: int = 200) -> float:
    """迭代 bulk matching 求 σ²。

    关键：γ>1 时只有 T=N/γ 个正特征值，其均值为 γσ²（不是 σ²），
    故要把正 bulk 均值除以 γ。γ≤1 时无需除。
    """
    ev = np.sort(ev)[::-1]
    corr = gamma if gamma > 1 else 1.0
    sigma2 = 1.0
    for _ in range(max_iter):
        _, lam_plus = mp_edges(sigma2, gamma)
        bulk = ev[(ev > 1e-12) & (ev <= lam_plus)]
        if len(bulk) == 0:
            break
        new = float(bulk.mean()) / corr
        if abs(new - sigma2) < tol:
            sigma2 = new
            break
        sigma2 = new
    return sigma2


def inverse_spike_full(lam: float, sigma2: float, gamma: float) -> float:
    """由样本 spike λ 反解真实完整特征值 σ²+θ。支持标量或数组。"""
    lam = np.asarray(lam, dtype=float)
    b = lam - sigma2 * (1.0 + gamma)
    disc = b * b - 4.0 * gamma * sigma2 * sigma2
    disc = np.maximum(disc, 0.0)
    theta = (b + np.sqrt(disc)) / 2.0
    return sigma2 + theta


def denoise_spiked(ev: np.ndarray, sigma2: float, gamma: float,
                   preserve_trace: bool = True) -> np.ndarray:
    """尖峰 RIE 非线性收缩。

    输入：样本特征值（降序）、σ²、γ
    输出：去噪后的特征值（与输入同序）
      - spike（λ > λ₊）：用 inverse_spike_full 反解真实特征值
      - bulk（λ ≤ λ₊）：压成常数（默认保迹）
    """
    ev = np.asarray(ev, dtype=float)
    _, lam_plus = mp_edges(sigma2, gamma)
    out = np.zeros_like(ev)
    spikes = ev > lam_plus
    out[spikes] = inverse_spike_full(ev[spikes], sigma2, gamma)
    n_bulk = int((~spikes).sum())
    if n_bulk > 0:
        if preserve_trace:
            bulk_const = (ev.sum() - out[spikes].sum()) / n_bulk
            bulk_const = max(bulk_const, 1e-12)
        else:
            bulk_const = sigma2
        out[~spikes] = bulk_const
    return out
