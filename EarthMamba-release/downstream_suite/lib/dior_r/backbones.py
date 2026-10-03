from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path


SUPPORTED_BACKBONES = (
    "earth-mamba", "earth-mamba-b", "satmae", "roma", "rsmamba", "rvsa", "satlas_aerial"
)
_BASELINES = {"earth-mamba", "earth-mamba-b", "satmae", "roma", "rsmamba"}
# normalize_backbone 统一用连字符；satlas 在 registry 侧再转下划线
_CANONICAL_BACKBONES = {
    "earth-mamba": "earth-mamba",
    "earthmamba": "earth-mamba",
    "earth-mamba-b": "earth-mamba-b",
    "earthmamba-b": "earth-mamba-b",
    "earth_mamba_b": "earth-mamba-b",
    "satmae": "satmae",
    "roma": "roma",
    "rsmamba": "rsmamba",
    "rvsa": "rvsa",
    "satlas-aerial": "satlas_aerial",
    "satlas_aerial": "satlas_aerial",
    "satlas": "satlas_aerial",
}

# 与 多模型对比主表.md 对齐的默认 encoder 参数（覆盖 ckpt peek 误判）
_ENCODER_KWARGS = {
    "satmae": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
    },
    "rsmamba": {
        "rsmamba_size": "huge",
    },
}


def normalize_backbone(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    value = _CANONICAL_BACKBONES.get(key)
    if value is None:
        raise ValueError(f"unsupported backbone {name!r}; choices={SUPPORTED_BACKBONES}")
    return value


def _registry_modules():
    return [
        name for name in sys.modules
        if name == "shared" or name.startswith("shared.") or name == "backbone_registry"
    ]


@contextmanager
def _isolated_import(root: Path, extras=()):
    old_path = list(sys.path)
    saved = {name: sys.modules[name] for name in _registry_modules()}
    for name in saved:
        sys.modules.pop(name, None)
    sys.path[:0] = [str(path) for path in (root, *extras) if path.is_dir()]
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in _registry_modules():
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = old_path
        importlib.invalidate_caches()


def build_backbone(
    name: str,
    checkpoint: str | Path,
    project_root: str | Path = "/hy-tmp",
    image_size: int = 512,
    ssm_version: str = "mamba3",
    earth_adapter: str = "gated_pyramid",
    debug_shapes: bool = False,
):
    name = normalize_backbone(name)
    root = Path(project_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    if name in ("earth-mamba", "earth-mamba-b") and earth_adapter != "none":
        from .earth_adapter import ADAPTER_CHOICES, build_earth_mamba_encoder
        from lib.earth_mamba_variants import resolve_model_size

        if earth_adapter not in ADAPTER_CHOICES:
            raise ValueError(f"unknown earth_adapter={earth_adapter!r}; choices={ADAPTER_CHOICES}")
        print(f"  [{name}] dense_adapter={earth_adapter}", flush=True)
        encoder = build_earth_mamba_encoder(
            checkpoint,
            root,
            image_size,
            model_size=resolve_model_size(name, "auto"),
            ssm_version=ssm_version,
            earth_adapter=earth_adapter,
            debug_shapes=debug_shapes,
        )
        channels = tuple(int(value) for value in encoder.out_dims)
        return encoder, channels

    encoder_kwargs = dict(_ENCODER_KWARGS.get(name, {}))

    if name in _BASELINES:
        from bootstrap import VENDOR_BASELINES

        registry_path = VENDOR_BASELINES / "shared" / "backbone_registry.py"
        if not registry_path.is_file():
            raise FileNotFoundError(f"baseline registry not found: {registry_path}")
        earth_roots = (
            Path("/root/earth-mamba"),
            root / "earth-mamba",
            root.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
            VENDOR_BASELINES.parents[1] / "earth-mamba",
        )
        with _isolated_import(VENDOR_BASELINES, earth_roots):
            registry = importlib.import_module("shared.backbone_registry")
            registry_name = "earthmamba" if name in ("earth-mamba", "earth-mamba-b") else name
            encoder = registry.build_encoder(
                registry_name,
                image_size,
                str(checkpoint),
                ssm_version=ssm_version,
                **encoder_kwargs,
            )
    else:
        from bootstrap import VENDOR_BASELINES2

        registry_path = VENDOR_BASELINES2 / "backbone_registry.py"
        if not registry_path.is_file():
            raise FileNotFoundError(f"baseline2 registry not found: {registry_path}")
        with _isolated_import(VENDOR_BASELINES2):
            registry = importlib.import_module("backbone_registry")
            encoder = registry.build_encoder(
                name.replace("-", "_"),
                image_size,
                str(checkpoint),
                **encoder_kwargs,
            )

    channels = tuple(int(value) for value in getattr(encoder, "out_dims", ()))
    if len(channels) != 4:
        raise RuntimeError(f"{name} returned invalid out_dims={channels}")
    return encoder, channels
