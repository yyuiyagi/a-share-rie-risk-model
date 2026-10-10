# -*- coding: utf-8 -*-
r"""
qr_rnd_lab —— 随机矩阵理论实验包
================================

配套文档：QR/AI 资料/RMT_学习文档.md

模块
----
01_mp_experiment.py   MP 律的数值实验（E1–E7），建立直觉
02_verify_mp.py       MP 公式自检（26 项），锁定所有公式

用法（在项目根目录执行）
----
    python qr_cov/qr_rnd_lab/01_mp_experiment.py     # 建立直觉
    python qr_cov/qr_rnd_lab/02_verify_mp.py         # 公式自检

作为包导入（供下游代码复用已验证的公式）：

    from qr_cov.qr_rnd_lab import mp_edges, mp_density_support_normalized, \\
        stieltjes_from_equation, density_from_equation

**重要**：这些公式是经数值自检锁定的（见 02_verify_mp.py）。
本文档初稿里的自洽方程曾是凭记忆写的错误版本，请勿再自行"回忆"公式——
一律从本包导入。
"""
from __future__ import annotations

import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mp = _load("02_verify_mp")

mp_edges = _mp.mp_edges
mp_density_support_normalized = _mp.mp_density_support_normalized
stieltjes_from_equation = _mp.stieltjes_from_equation
density_from_equation = _mp.density_from_equation

__all__ = [
    "mp_edges",
    "mp_density_support_normalized",
    "stieltjes_from_equation",
    "density_from_equation",
]
