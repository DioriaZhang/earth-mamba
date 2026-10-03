"""Add downstream_suite to sys.path and locate earth-mamba repo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parent
VENDOR_BASELINES = SUITE_ROOT / "lib" / "vendor" / "baselines"
VENDOR_BASELINES2 = SUITE_ROOT / "lib" / "vendor" / "baselines2"


def setup() -> Path:
    root = str(SUITE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    lib_s = str(SUITE_ROOT / "lib")
    if lib_s not in sys.path:
        sys.path.insert(0, lib_s)
    import common

    sys.modules.setdefault("downstream_common", common)
    import lib.earthmamba_adapter as _em_adapt

    sys.modules.setdefault("earthmamba_adapter", _em_adapt)
    for em in (
        os.environ.get("EARTH_MAMBA_ROOT", ""),
        str(SUITE_ROOT.parent / "earth-mamba"),
        str(SUITE_ROOT.parent.parent / "earth-mamba"),  # 仓库根 earth-mamba（两文件夹布局）
        "/hy-tmp/earth-mamba",
        "/root/earth-mamba",
    ):
        if em and (Path(em) / "earth_mamba").is_dir():
            em_s = str(Path(em).resolve())
            if em_s not in sys.path:
                sys.path.insert(0, em_s)
            break
    return SUITE_ROOT
