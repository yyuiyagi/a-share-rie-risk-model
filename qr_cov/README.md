# qr_cov —— 项目 A 代码

A股协方差谱清洗项目（噪声尺度校准 × 非线性收缩）。数据与公式已就绪，可直接进入阶段 A。

## 目录结构

```
qr_cov/
├── __init__.py             # 使 qr_cov 成为可导入包
├── qr_data.py              # 数据加载接口（load_returns / load_source / membership_mask）
├── qr_suspension.py        # 停牌日处理（线性插值 / 零填充 / 不处理）
├── qr_download_tx.py       # 腾讯日线下载器（已跑完，一般不再需要）
├── qr_build_panel_v2.py    # 阶梯式收益面板构建器（已跑完）
├── deprecated/
│   └── qr_build_panel.py   # v1 构建器（含 instrument 前导零 bug，仅存档）
└── qr_rnd_lab/
    ├── __init__.py         # 已验证的 MP 公式，供下游 import
    ├── 01_mp_experiment.py # 数值实验 E1–E7（建立直觉）
    └── 02_verify_mp.py     # MP 公式自检（26/26 通过）
```

数据在根目录 `data_tx/`（不是代码，未移入）：`data_tx/panel2/returns_final.*` 是主数据。

## 运行方式（在项目根目录执行）

```powershell
# 数据概览
python qr_cov/qr_data.py

# 停牌处理自检
python qr_cov/qr_suspension.py

# 数值实验（写 UTF-8 文件避免乱码）
python qr_cov/qr_rnd_lab/01_mp_experiment.py --out result.txt

# MP 公式自检
python qr_cov/qr_rnd_lab/02_verify_mp.py
```

## 在分析脚本里导入

```python
from qr_cov.qr_data import load_returns, load_source, DEFAULT_WINDOW
from qr_cov.qr_suspension import impute_suspension
from qr_cov.qr_rnd_lab import mp_edges, mp_density_support_normalized
```

## 注意事项

- 数据 `data_tx/` 与 qlib 数据 `Quant/` 都在项目根目录，脚本内已用相对路径自动定位。
- 本机 PowerShell 的 pip 无外网，`dl` 环境装不了 sklearn；本项目的 lab 用
  **基础 Python（AppData 那个，3.12.10，含 numpy/scipy/sklearn）** 即可。
