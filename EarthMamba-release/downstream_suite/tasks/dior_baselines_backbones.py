from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _vendor_baselines() -> Path:
    return ROOT / "lib" / "vendor" / "baselines"
BASELINE_DIOR_BACKBONES = ("satmae", "roma", "rsmamba", "scalemae")


def _is_baselines_root(path: Path) -> bool:
    return (path / "shared" / "backbone_registry.py").is_file()


def _candidate_baselines_roots(ckpt: Optional[str] = None) -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("V1_BASELINES_ROOT", "BASELINES_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())

    for base in (ROOT, Path.cwd(), _vendor_baselines()):
        candidates.extend([
            base,
            base / "baselines" if base.name != "baselines" else base,
            _vendor_baselines(),
        ])

    if ckpt:
        ckpt_path = Path(ckpt).expanduser()
        candidates.extend(parent for parent in ckpt_path.parents)
        for parent in ckpt_path.parents:
            if parent.name == "weights":
                candidates.append(parent.parent.parent)
            if parent.name in ("SatMAE", "RoMA", "RSMamba"):
                candidates.append(parent.parent)

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def resolve_baselines_root(ckpt: Optional[str] = None) -> Path:
    for path in _candidate_baselines_roots(ckpt):
        if _is_baselines_root(path):
            return path
    tried = "\n    ".join(str(p) for p in _candidate_baselines_roots(ckpt)[:16])
    raise ModuleNotFoundError(
        "Cannot locate baselines/shared/backbone_registry.py. "
        "Set V1_BASELINES_ROOT=/hy-tmp/baselines or run from the project root.\n"
        f"  tried:\n    {tried}"
    )


def _drop_wrong_shared_package(baselines_root: Path) -> None:
    shared = sys.modules.get("shared")
    if shared is None:
        return
    shared_file = getattr(shared, "__file__", "") or ""
    shared_path = str(Path(shared_file).resolve()) if shared_file else ""
    if shared_path and shared_path.startswith(str(baselines_root)):
        return
    for name in list(sys.modules):
        if name == "shared" or name.startswith("shared."):
            sys.modules.pop(name, None)


def import_baseline_build_encoder(ckpt: Optional[str] = None):
    baselines_root = resolve_baselines_root(ckpt)
    root_str = str(baselines_root)
    sys.path = [p for p in sys.path if p != root_str]
    sys.path.insert(0, root_str)
    _drop_wrong_shared_package(baselines_root)
    importlib.invalidate_caches()
    from shared.backbone_registry import build_encoder

    return build_encoder, baselines_root


def import_scalemae_build_encoder():
    scalemae_root = ROOT / "v3_scalemae"
    if not (scalemae_root / "scalemae_backbone.py").is_file():
        raise ModuleNotFoundError(
            "Cannot locate v3_scalemae/scalemae_backbone.py. "
            "Put the ScaleMAE adapter under project_root/v3_scalemae/."
        )
    root_str = str(scalemae_root)
    sys.path = [p for p in sys.path if p != root_str]
    sys.path.insert(0, root_str)
    importlib.invalidate_caches()
    from scalemae_backbone import build_encoder

    return build_encoder, scalemae_root


def build_baseline_encoder(
    backbone: str,
    img_size: int,
    ckpt: Optional[str],
    *,
    ssm_version: str = "mamba3",
    strict_load_report: bool = False,
    **kwargs,
) -> nn.Module:
    if backbone == "scalemae":
        build_encoder, scalemae_root = import_scalemae_build_encoder()
        print(f"  [scale-root] {scalemae_root}")
        return build_encoder(backbone, img_size, ckpt, ssm_version=ssm_version, **kwargs)

    if backbone not in BASELINE_DIOR_BACKBONES:
        raise ValueError(f"DIOR v1 baseline supports {BASELINE_DIOR_BACKBONES}, got {backbone!r}")
    build_encoder, baselines_root = import_baseline_build_encoder(ckpt)
    print(f"  [baseline-root] {baselines_root}")

    encoder = build_encoder(backbone, img_size, ckpt, ssm_version=ssm_version, **kwargs)
    if strict_load_report:
        total = sum(p.numel() for p in encoder.parameters())
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(
            f"  [baseline-load-report] backbone={backbone} ckpt={ckpt or '(none)'} "
            f"params={total/1e6:.1f}M trainable={trainable/1e6:.1f}M out_dims={getattr(encoder, 'out_dims', None)}"
        )
        if ckpt:
            _print_checkpoint_shape_summary(ckpt, encoder)
    return encoder


def _print_checkpoint_shape_summary(ckpt: str, encoder: nn.Module) -> None:
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(ckpt, map_location="cpu")
    sd = raw.get("model", raw.get("state_dict", raw))
    model_sd = encoder.state_dict()
    same_name = 0
    same_shape = 0
    mismatched = []
    for k, v in sd.items():
        key = str(k)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "encoder.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in model_sd:
            same_name += 1
            if tuple(model_sd[key].shape) == tuple(v.shape):
                same_shape += 1
            else:
                mismatched.append((key, tuple(v.shape), tuple(model_sd[key].shape)))
    print(
        f"  [baseline-load-report] ckpt_keys={len(sd)} same_name={same_name} "
        f"same_shape={same_shape} shape_mismatch={len(mismatched)}"
    )
    for key, a, b in mismatched[:8]:
        print(f"    shape_mismatch {key}: ckpt{a} model{b}")
