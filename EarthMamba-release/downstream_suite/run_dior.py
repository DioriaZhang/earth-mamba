#!/usr/bin/env python3
"""DIOR horizontal-box detection — unified ``--backbone`` dispatcher."""

from __future__ import annotations

import runpy
import sys

import bootstrap
from bootstrap import VENDOR_BASELINES, VENDOR_BASELINES2

bootstrap.setup()

import backbone as bb  # noqa: E402


def _argv_backbone() -> str | None:
    for i, arg in enumerate(sys.argv):
        if arg == "--backbone" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main() -> None:
    name_arg = _argv_backbone()
    if not name_arg:
        raise SystemExit("Missing required --backbone")
    name = bb.normalize_backbone(name_arg)

    if name in ("earth-mamba", "earth-mamba-b"):
        from tasks import dior_earth

        dior_earth.main()
        return

    if name in ("rvsa", "satlas_aerial"):
        sys.path.insert(0, str(VENDOR_BASELINES2))
        import lib.vendor.baselines2.backbone_registry as reg

        reg.build_encoder = lambda n, img_size, ckpt, **kw: bb.build_encoder(n, img_size, ckpt, **kw)
        reg.resolve_ckpt = lambda n, ckpt, ckpt_dir=None: bb.resolve_ckpt(n, ckpt, ckpt_dir)
        runpy.run_path(str(VENDOR_BASELINES2 / "run_dior.py"), run_name="__main__")
        return

    sys.path.insert(0, str(VENDOR_BASELINES))
    from tasks import dior_baselines

    dior_baselines.main()


if __name__ == "__main__":
    main()
