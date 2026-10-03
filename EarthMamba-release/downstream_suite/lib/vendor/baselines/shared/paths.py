"""baselines 与 downstream_code 的路径约定。

服务器目录（`/hy-tmp/` 下同级放置）::

  /hy-tmp/downstream_code/   ← EarthMamba 脚本
  /hy-tmp/baselines/         ← 本目录
  /hy-tmp/earth-mamba/       ← EarthMamba 模型代码（或与 downstream_code 同级子目录）
  /hy-tmp/task/  /hy-tmp/CKPT/  /hy-tmp/downstream_results/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# baselines/shared/paths.py → baselines/
BASELINES_ROOT = Path(__file__).resolve().parent.parent
# /hy-tmp/（baselines 与 downstream_code 的父目录）
HYTMP_ROOT = BASELINES_ROOT.parent
# EarthMamba 下游脚本
DOWNSTREAM_ROOT = HYTMP_ROOT / "downstream_code"


def resolve_earth_mamba_root() -> Optional[Path]:
    """earth-mamba 代码目录：优先 downstream_code/earth-mamba，其次 /hy-tmp/earth-mamba。"""
    for candidate in (
        DOWNSTREAM_ROOT / "earth-mamba",
        HYTMP_ROOT / "earth-mamba",
    ):
        if candidate.is_dir():
            return candidate
    return None


def ensure_earth_mamba_on_path() -> Optional[Path]:
    root = resolve_earth_mamba_root()
    if root is not None:
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)
    return root

# CLI --backbone 小写名 → 权重/模型代码子目录名
BACKBONE_DIR_NAMES: dict[str, Optional[str]] = {
    "earthmamba": None,
    "skysense": "SkySense",
    "satmae": "SatMAE",
    "dofa": "DOFA",
    "clay": "Clay",
    "roma": "RoMA",
    "rsmamba": "RSMamba",
}


def backbone_model_dir(backbone: str) -> Optional[Path]:
    """返回某 backbone 的模型代码/权重根目录；earthmamba 返回 None。"""
    folder = BACKBONE_DIR_NAMES.get(backbone.lower().replace("-", "").replace("_", ""))
    if folder is None:
        return None
    return BASELINES_ROOT / folder


def default_weights_dir(backbone: str) -> Optional[Path]:
    d = backbone_model_dir(backbone)
    return d / "weights" if d is not None else None


def find_ckpt_in_dir(ckpt_dir: Path) -> Optional[str]:
    """在 weights 目录中自动发现 checkpoint（.pth / .pt / .ckpt）。"""
    if not ckpt_dir.is_dir():
        return None
    for pattern in ("*.pth", "*.pt", "*.ckpt"):
        files = sorted(ckpt_dir.glob(pattern))
        if files:
            return str(files[0])
    return None


def resolve_backbone_ckpt(
    backbone: str,
    ckpt: Optional[str] = None,
    ckpt_dir: Optional[str] = None,
) -> Optional[str]:
    """优先 --ckpt，其次 --backbone_ckpt_dir，最后 baselines/{Model}/weights/。"""
    if ckpt:
        return ckpt
    if ckpt_dir:
        found = find_ckpt_in_dir(Path(ckpt_dir))
        if found:
            return found
    default = default_weights_dir(backbone)
    if default is not None:
        return find_ckpt_in_dir(default)
    return None
