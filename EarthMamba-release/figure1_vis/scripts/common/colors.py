"""
加深莫兰迪配色：保留原实验色相，Earth-Mamba 使用红色系。
"""
from __future__ import annotations

# 原色参考: ViT #2ca02c, Swin #1f77b4, VMamba #9467bd
MORANDI = {
    "ViT": "#4d7a52",         # 深莫兰迪绿
    "Swin": "#4a6f8c",        # 深莫兰迪蓝
    "VMamba": "#7a6888",      # 深莫兰迪紫
    "EarthMamba": "#a85c5c",   # 莫兰迪红（新增对比模型）
}

MARKERS = {
    "ViT": "o",
    "Swin": "s",
    "VMamba": "^",
    "EarthMamba": "D",
}

LINESTYLES = {
    "ViT": "-",
    "Swin": "--",
    "VMamba": "-.",
    "EarthMamba": "-",
}

# 图例显示名（论文友好）
DISPLAY_NAMES = {
    "ViT": "ViT",
    "Swin": "Swin",
    "VMamba": "VMamba",
    "EarthMamba": "Earth-Mamba",
}

FILL_ALPHA = 0.14
# 阴影带最小半宽（归一化灵敏度），避免 VMamba/Earth-Mamba 方差小时几乎看不见
BAND_MIN_HALF = 0.045
