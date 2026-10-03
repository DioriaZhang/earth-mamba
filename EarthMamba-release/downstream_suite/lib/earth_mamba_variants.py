"""EarthMamba small / base presets (see ``EarthMamba 模型参数尺寸说明.md``)."""

from __future__ import annotations

# Main-table six-model comparison uses earth-mamba (small, ~91M).
# earth-mamba-b is the base encoder (~161M) for scale ablations.
EARTH_MAMBA_VARIANTS: tuple[str, ...] = ("earth-mamba", "earth-mamba-b")

SMALL_DEPTHS = [2, 2, 27, 2]
SMALL_DIMS = [96, 192, 384, 768]
BASE_DEPTHS = [2, 2, 27, 2]
BASE_DIMS = [128, 256, 512, 1024]

MODEL_PRESETS: dict[str, tuple[list[int], list[int]]] = {
    "small": (SMALL_DEPTHS, SMALL_DIMS),
    "base": (BASE_DEPTHS, BASE_DIMS),
}

DEFAULT_CKPTS: dict[str, str] = {
    "earth-mamba": "/hy-tmp/CKPT/PN-log13-ep14/backbone.pth",
    "earth-mamba-b": "/hy-tmp/pretrain_base/pretrain_20260702_1805/epochs/ep30/backbone.pth",
}

_ALIASES = {
    "earth-mamba": "earth-mamba",
    "earthmamba": "earth-mamba",
    "earth_mamba": "earth-mamba",
    "earth-mamba-s": "earth-mamba",
    "earthmamba-s": "earth-mamba",
    "earth_mamba_s": "earth-mamba",
    "earth-mamba-b": "earth-mamba-b",
    "earthmamba-b": "earth-mamba-b",
    "earth_mamba_b": "earth-mamba-b",
}


def normalize_earth_variant(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in _ALIASES:
        return _ALIASES[key]
    key2 = name.strip().lower()
    if key2 in _ALIASES:
        return _ALIASES[key2]
    raise ValueError(f"unsupported EarthMamba variant {name!r}")


def is_earth_mamba(backbone: str) -> bool:
    try:
        normalize_earth_variant(backbone)
        return True
    except ValueError:
        return False


def preset_tag(backbone: str) -> str:
    """Return ``small`` or ``base`` for a variant CLI name."""
    variant = normalize_earth_variant(backbone)
    return "base" if variant == "earth-mamba-b" else "small"


def resolve_model_size(backbone: str, model_size: str = "auto") -> str:
    """Merge ``--backbone`` with optional ``--model_size``."""
    size = (model_size or "auto").strip().lower()
    if size not in ("auto", "small", "base"):
        raise ValueError(f"model_size must be auto|small|base, got {model_size!r}")
    if size != "auto":
        return size
    return preset_tag(backbone)
