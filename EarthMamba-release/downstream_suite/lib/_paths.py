from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    env_root = os.environ.get("DOWNSTREAM_SUITE_ROOT") or os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def add_project_paths(root: Path | None = None) -> Path:
    root = root or project_root()
    for path in (root, root / "lib"):
        if path.is_dir():
            ps = str(path.resolve())
            if ps not in sys.path:
                sys.path.insert(0, ps)
    return root
