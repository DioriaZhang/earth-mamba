"""baselines2 路径约定（仅本目录，不依赖 baselines/ / downstream_code/）。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

BASELINES2_ROOT = Path(__file__).resolve().parent.parent


def backbone_weights_dir(model_name: str) -> Path:
    return BASELINES2_ROOT / model_name / "weights"


def find_ckpt(weights_dir: Path) -> Optional[str]:
    if not weights_dir.is_dir():
        return None
    for pat in ("*.pth", "*.pt", "*.ckpt", "*.safetensors"):
        files = sorted(weights_dir.glob(pat))
        if files:
            return str(files[0])
    return None
