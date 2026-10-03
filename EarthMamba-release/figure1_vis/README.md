# Figure 1 可视化包

路径：`E:\project_mamba\figure1_vis`

自包含目录：精选原数据 + 出图脚本，可独立重绘论文 Figure 1 四宫格各 panel。

## 目录结构

```
figure1_vis/
├── data/                  # 绘图用原数据
│   ├── fig2_noise_radar/
│   ├── fig_erf_rotation/  # ERF 曲线 + 旋转 RRI
│   ├── fig_downstream/
│   └── fig_efficiency/    # 论文采用效率摘要（非 raw benchmark）
├── scripts/               # 出图脚本与 common 模块
├── output/                # 运行后生成的 PDF/PNG
└── run_all.py             # 一键出四图
```

## 四个 Panel

| Panel | 脚本 | 输出 |
|-------|------|------|
| (a) 噪声雷达 | `scripts/plot_fig2_noise_radar.py` | `output/fig2_noise_radar/` |
| (b) ERF + 旋转 | `scripts/plot_erf_rotation.py` | `output/fig_erf_rotation/` |
| (c) 下游四任务 | `scripts/plot_downstream.py` | `output/fig_downstream/` |
| (d) 效率摘要 | `scripts/plot_efficiency_summary.py` | `output/fig_efficiency/` |

统一 panel 宽度：AAAI 单栏 × 0.32 ≈ **1.066 in**（与 `fig_erf_rotation.pdf` 一致）。  
Earth-Mamba 强调色：**#FF1F3D**。

## 重绘

```bash
cd E:\project_mamba\figure1_vis
python run_all.py
```

或单独运行各脚本，例如：

```bash
python scripts/plot_erf_rotation.py
```

## 依赖

Python 3.10+，`matplotlib`、`numpy`。
