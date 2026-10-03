"""各模型权重文件名约定（无 torch 依赖，供 check_setup 使用）。"""
from __future__ import annotations

from typing import Dict, List

PREFERRED_CKPT_NAMES: Dict[str, List[str]] = {
    "RVSA": [
        "vit-b-checkpoint.pth",
        "vit-b-checkpoint-1599.pth",
        "vit_b_checkpoint.pth",
        "mae_vit_base.pth",
    ],
    "SatlasPretrain_Aerial": [
        "sentinel2_swinb_si_rgb.pth",
        "aerial_swinb_si.pth",
    ],
}

CKPT_NAME_WARNINGS: Dict[str, List[str]] = {
    "RVSA": ["vitae", "vitae-b", "vitae_b"],
    "SatlasPretrain_Aerial": [],
}
