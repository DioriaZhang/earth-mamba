"""Checkpoint resolution for EarthMamba-S and EarthMamba-B."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from lib.earth_mamba_variants import DEFAULT_CKPTS, normalize_earth_variant

_DEFAULT_RUN_DIR = Path(
    os.environ.get(
        "EARTH_MAMBA_B_RUN",
        "/hy-tmp/pretrain_base/pretrain_20260702_1805",
    )
)
_DEFAULT_EPOCH = int(os.environ.get("EARTH_MAMBA_B_EPOCH", "30"))
_CKPT_NAMES = ("backbone.pth", "checkpoint.pth")


def _pick_ckpt_in_dir(directory: Path) -> Optional[Path]:
    for name in _CKPT_NAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_earth_mamba_b_ckpt(ckpt: Optional[str] = None) -> Optional[str]:
    """Resolve base encoder weights (``backbone.pth`` or ``checkpoint.pth``)."""
    candidates: list[Path] = []
    if ckpt:
        candidates.append(Path(ckpt).expanduser())
    env = os.environ.get("EARTH_MAMBA_B_CKPT", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(_DEFAULT_RUN_DIR / "epochs" / f"ep{_DEFAULT_EPOCH}")

    for path in candidates:
        if path.is_file():
            return str(path.resolve())
        if path.is_dir():
            found = _pick_ckpt_in_dir(path)
            if found is not None:
                return str(found)
            ep_nested = path / "epochs" / f"ep{_DEFAULT_EPOCH}"
            if ep_nested.is_dir():
                found = _pick_ckpt_in_dir(ep_nested)
                if found is not None:
                    return str(found)
    return None


def resolve_earth_mamba_ckpt(backbone: str, ckpt: Optional[str] = None) -> Optional[str]:
    """Resolve default checkpoint for ``earth-mamba`` (small) or ``earth-mamba-b`` (base)."""
    if ckpt:
        path = Path(ckpt).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return str(path.resolve())

    variant = normalize_earth_variant(backbone)
    if variant == "earth-mamba-b":
        resolved = resolve_earth_mamba_b_ckpt(None)
        if resolved:
            return resolved
        hint = DEFAULT_CKPTS["earth-mamba-b"]
        if Path(hint).is_file():
            return str(Path(hint).resolve())
        return None

    default = DEFAULT_CKPTS["earth-mamba"]
    if Path(default).is_file():
        return str(Path(default).resolve())
    return None
