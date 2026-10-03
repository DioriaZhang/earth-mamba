"""在 baselines 任务脚本启动时注入 import 路径。"""

from __future__ import annotations

import sys

from .paths import BASELINES_ROOT, DOWNSTREAM_ROOT, ensure_earth_mamba_on_path


def setup_import_paths() -> None:
    for p in (str(DOWNSTREAM_ROOT), str(BASELINES_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    ensure_earth_mamba_on_path()
