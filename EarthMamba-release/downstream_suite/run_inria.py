#!/usr/bin/env python3
"""INRIA building segmentation — unified ``--backbone`` dispatcher."""

from __future__ import annotations

import runpy
import sys

import bootstrap
from bootstrap import SUITE_ROOT, VENDOR_BASELINES, VENDOR_BASELINES2

bootstrap.setup()

import backbone as bb  # noqa: E402


def _argv_backbone() -> str | None:
    for i, arg in enumerate(sys.argv):
        if arg == "--backbone" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _patch_baselines2_registry() -> None:
    import lib.vendor.baselines2.backbone_registry as reg

    reg.build_encoder = lambda name, img_size, ckpt, **kw: bb.build_encoder(name, img_size, ckpt, **kw)
    reg.resolve_ckpt = lambda name, ckpt, ckpt_dir=None: bb.resolve_ckpt(name, ckpt, ckpt_dir)
    reg.BACKBONE_CHOICES = [b for b in bb.BACKBONE_CHOICES if b in ("rvsa", "satlas_aerial")]


def main() -> None:
    name_arg = _argv_backbone()
    if not name_arg:
        raise SystemExit("Missing required --backbone")
    name = bb.normalize_backbone(name_arg)

    if name in ("earth-mamba", "earth-mamba-b"):
        from tasks import inria_earth

        inria_earth.main()
        return

    if name in ("rvsa", "satlas_aerial"):
        sys.path.insert(0, str(VENDOR_BASELINES2))
        _patch_baselines2_registry()
        runpy.run_path(str(VENDOR_BASELINES2 / "run_inria.py"), run_name="__main__")
        return

    sys.path.insert(0, str(VENDOR_BASELINES))
    runpy.run_path(str(VENDOR_BASELINES / "baselines_inria.py"), run_name="__main__")


if __name__ == "__main__":
    main()
