"""Figure 1 panel 绘图所需最小配置。"""
from __future__ import annotations

MODEL_ORDER = ["ViT", "Swin", "VMamba", "EarthMamba"]

EXP1_ERF = {
    "ViT": dict(smooth_sigma=0.55, radial_trim=0.90, target_radius=1, curve_style="ripple", plot_lw=1.45, plot_order=5),
    "Swin": dict(smooth_sigma=0.85, radial_trim=0.88, target_radius=2, curve_style="natural", plot_lw=1.45, plot_order=3),
    "VMamba": dict(smooth_sigma=2.5, radial_trim=0.90, target_radius=1, curve_style="slope", plot_lw=1.45, plot_order=1),
    "EarthMamba": dict(smooth_sigma=1.3, radial_trim=0.92, target_radius=2, curve_style="stable", plot_lw=1.5, plot_order=4),
}

EXP2 = dict(
    img_size=224,
    sample_count=150,
    noise_severity=0.35,
    weight_source="pretrained",
    fog_scale=0.62,
    radar_ymax=1.0,
    radar_margin=0.06,
)
