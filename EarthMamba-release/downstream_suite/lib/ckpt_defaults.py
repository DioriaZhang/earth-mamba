"""Default pretrained checkpoints (aligned with main comparison tables)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lib.earth_mamba_ckpt import resolve_earth_mamba_ckpt
from lib.earth_mamba_variants import DEFAULT_CKPTS, is_earth_mamba

_BASELINE_CKPTS: dict[str, str] = {
    "satmae": "/hy-tmp/baselines/SatMAE/weights/pretrain-vit-base-e199.pth",
    "satlas_aerial": "/hy-tmp/baselines/SatlasPretrain/weights/sentinel2_swinb_si_rgb.pth",
    "rvsa": "/hy-tmp/baselines2/RVSA/weights/vit-b-checkpoint-1599.pth",
    "roma": "/hy-tmp/baselines/RoMA/weights/mamba-base.pth",
    "rsmamba": "/hy-tmp/baselines/RSMamba/weights/RSMamba-h_UC.pth",
}


def resolve_training_ckpt(
    backbone: str,
    ckpt: Optional[str],
    project_root: Path,
) -> Optional[str]:
    if ckpt:
        path = Path(ckpt).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"--ckpt not found: {path}")
        return str(path.resolve())

    if is_earth_mamba(backbone):
        resolved = resolve_earth_mamba_ckpt(backbone, None)
        if resolved:
            print(f"  [ckpt] {backbone}: {resolved}", flush=True)
            return resolved
        default = DEFAULT_CKPTS.get(backbone.replace("_", "-"))
        if default:
            print(f"  [ckpt] WARN: earth-mamba ckpt missing: {default}", flush=True)
        return None

    norm = backbone.lower().replace("-", "_")
    table_key = {
        "earth_mamba": "earth-mamba",
        "satlas": "satlas_aerial",
    }.get(norm, backbone)
    default = _BASELINE_CKPTS.get(table_key)
    if default and Path(default).is_file():
        print(f"  [ckpt] using table default: {default}", flush=True)
        return default

    resolved = _resolve_via_registry(backbone, project_root)
    if resolved:
        print(f"  [ckpt] registry resolved: {resolved}", flush=True)
        return resolved

    if default:
        print(f"  [ckpt] WARN: table default missing on disk: {default}", flush=True)
    else:
        print(f"  [ckpt] WARN: no default for backbone={backbone!r}; random init", flush=True)
    return None


def _resolve_via_registry(backbone: str, project_root: Path) -> Optional[str]:
    from bootstrap import VENDOR_BASELINES, VENDOR_BASELINES2

    norm = backbone.lower().replace("-", "_")
    if norm in ("rvsa", "satlas_aerial", "satlas"):
        baselines2_root = VENDOR_BASELINES2
        if baselines2_root.is_dir():
            import sys

            path_s = str(baselines2_root)
            inserted = path_s not in sys.path
            if inserted:
                sys.path.insert(0, path_s)
            try:
                from backbone_registry import resolve_ckpt

                name = "satlas_aerial" if norm in ("satlas_aerial", "satlas") else "rvsa"
                return resolve_ckpt(name, None, None)
            finally:
                if inserted and sys.path and sys.path[0] == path_s:
                    sys.path.pop(0)

    if norm in ("satmae", "roma", "rsmamba"):
        baselines_root = VENDOR_BASELINES
        if baselines_root.is_dir():
            import sys

            path_s = str(baselines_root)
            inserted = path_s not in sys.path
            if inserted:
                sys.path.insert(0, path_s)
            try:
                from shared.paths import resolve_backbone_ckpt

                return resolve_backbone_ckpt(norm, None, None)
            finally:
                if inserted and sys.path and sys.path[0] == path_s:
                    sys.path.pop(0)
    return None
