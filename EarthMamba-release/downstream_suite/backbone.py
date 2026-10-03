"""Unified backbone API for all main-table models + EarthMamba-B variant."""



from __future__ import annotations



import importlib

import sys

from contextlib import contextmanager

from pathlib import Path

from typing import Any, List, Optional



from bootstrap import SUITE_ROOT, VENDOR_BASELINES, VENDOR_BASELINES2

from lib.earth_mamba_variants import EARTH_MAMBA_VARIANTS, is_earth_mamba, resolve_model_size



# Six-model main table + earth-mamba-b scale ablation.

BACKBONE_CHOICES: tuple[str, ...] = (

    "earth-mamba",

    "earth-mamba-b",

    "rvsa",

    "satlas_aerial",

    "satmae",

    "roma",

    "rsmamba",

)



MAIN_TABLE_BACKBONES: tuple[str, ...] = tuple(

    b for b in BACKBONE_CHOICES if b != "earth-mamba-b"

)



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

    "satlas": "satlas_aerial",

    "satlas-aerial": "satlas_aerial",

    "rvsa": "rvsa",

    "satmae": "satmae",

    "roma": "roma",

    "rsmamba": "rsmamba",

}



_BASELINES1 = {"satmae", "roma", "rsmamba"}

_BASELINES2 = {"rvsa", "satlas_aerial"}



_ENCODER_KWARGS: dict[str, dict[str, Any]] = {

    "satmae": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},

    "rsmamba": {"rsmamba_size": "huge"},

}





def normalize_backbone(name: str) -> str:

    key = name.strip().lower().replace("_", "-")

    if key in _ALIASES:

        return _ALIASES[key]

    key2 = name.strip().lower()

    if key2 in _ALIASES:

        return _ALIASES[key2]

    raise ValueError(f"unsupported backbone {name!r}; choices={BACKBONE_CHOICES}")





def _registry_modules() -> list[str]:

    return [

        n for n in sys.modules

        if n == "shared" or n.startswith("shared.") or n == "backbone_registry"

    ]





@contextmanager

def _isolated_import(root: Path, extras: tuple[Path, ...] = ()):

    old_path = list(sys.path)

    saved = {n: sys.modules[n] for n in _registry_modules()}

    for n in saved:

        sys.modules.pop(n, None)

    for p in (root, *extras):

        ps = str(p)

        if p.is_dir() and ps not in sys.path:

            sys.path.insert(0, ps)

    importlib.invalidate_caches()

    try:

        yield

    finally:

        for n in _registry_modules():

            sys.modules.pop(n, None)

        sys.modules.update(saved)

        sys.path[:] = old_path

        importlib.invalidate_caches()





def resolve_ckpt(

    backbone: str,

    ckpt: Optional[str],

    ckpt_dir: Optional[str] = None,

) -> Optional[str]:

    name = normalize_backbone(backbone)

    if ckpt:

        p = Path(ckpt).expanduser()

        if not p.is_file():

            raise FileNotFoundError(f"checkpoint not found: {p}")

        return str(p.resolve())

    if is_earth_mamba(name):

        from lib.earth_mamba_ckpt import resolve_earth_mamba_ckpt



        resolved = resolve_earth_mamba_ckpt(name, None)

        if resolved:

            print(f"  [ckpt] {name}: {resolved}", flush=True)

            return resolved

        print(f"  [ckpt] WARN: no default ckpt for {name}; random init", flush=True)

        return None

    if name in _BASELINES2:

        with _isolated_import(VENDOR_BASELINES2):

            reg = importlib.import_module("backbone_registry")

            return reg.resolve_ckpt(name, None, ckpt_dir)

    if name in _BASELINES1:

        with _isolated_import(VENDOR_BASELINES, _earth_mamba_roots()):

            from shared.paths import resolve_backbone_ckpt



            return resolve_backbone_ckpt(name, None, ckpt_dir=ckpt_dir)

    return None





def _earth_mamba_roots() -> tuple[Path, ...]:

    roots: list[Path] = []

    for cand in (

        Path("/hy-tmp/earth-mamba"),

        Path("/root/earth-mamba"),

        SUITE_ROOT.parent / "earth-mamba",

    ):

        if (cand / "earth_mamba").is_dir():

            roots.append(cand.resolve())

    return tuple(roots)





def build_encoder(

    backbone: str,

    img_size: int,

    ckpt: Optional[str],

    *,

    ssm_version: str = "mamba3",

    ckpt_dir: Optional[str] = None,

    model_size: str = "auto",

    **kwargs: Any,

):

    """4-stage encoder with ``out_dims`` (classification & shared detection API)."""

    name = normalize_backbone(backbone)

    ckpt = resolve_ckpt(name, ckpt, ckpt_dir)



    if is_earth_mamba(name):

        from lib.earth_mamba_classifier import build_classifier_encoder



        return build_classifier_encoder(

            name,

            img_size,

            ckpt,

            model_size=resolve_model_size(name, model_size),

            ssm_version=ssm_version,

        )



    enc_kw = dict(_ENCODER_KWARGS.get(name, {}))

    enc_kw.update(kwargs)



    if name in _BASELINES2:

        with _isolated_import(VENDOR_BASELINES2):

            reg = importlib.import_module("backbone_registry")

            return reg.build_encoder(name, img_size, ckpt, **enc_kw)



    with _isolated_import(VENDOR_BASELINES, _earth_mamba_roots()):

        from shared.backbone_registry import build_encoder as b1_build



        return b1_build(name, img_size, ckpt, ssm_version=ssm_version, **enc_kw)





def build_segmentation_encoder(

    backbone: str,

    img_size: int,

    ckpt: Optional[str],

    *,

    ssm_version: str = "mamba3",

    dense_adapter: str = "gated_pyramid",

    model_size: str = "auto",

    debug_shapes: bool = False,

    **kwargs: Any,

):

    """Earth-Mamba variants use gated pyramid adapter; other backbones use ``build_encoder``."""

    name = normalize_backbone(backbone)

    ckpt = resolve_ckpt(name, ckpt, kwargs.get("ckpt_dir"))

    if is_earth_mamba(name) and dense_adapter != "none":

        from lib.earthmamba_adapter import build_earthmamba_encoder, ensure_registered



        ensure_registered()

        return build_earthmamba_encoder(

            ckpt,

            img_size,

            model_size=resolve_model_size(name, model_size),

            ssm_version=ssm_version,

            dense_adapter=dense_adapter,

            debug_shapes=debug_shapes,

        )

    return build_encoder(

        name,

        img_size,

        ckpt,

        ssm_version=ssm_version,

        model_size=model_size,

        **kwargs,

    )


