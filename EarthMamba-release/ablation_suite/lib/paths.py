"""earth_mamba 模型代码路径（仅模型包，非训练脚本）。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]  # ablation_suite root
SUITE_ROOT = RUN_DIR


def earth_mamba_roots() -> list[Path]:
    env = os.environ.get("EARTH_MAMBA_ROOT")
    roots: list[Path] = []
    if env:
        roots.append(Path(env))
    proj = os.environ.get("PROJECT_ROOT")
    if proj:
        roots.append(Path(proj) / "earth-mamba")
    roots.extend([
        SUITE_ROOT.parent / "earth-mamba",
        SUITE_ROOT.parent.parent / "earth-mamba",  # 仓库根 earth-mamba（两文件夹布局）
    ])
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        s = str(r.resolve()) if r.exists() else str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def ensure_earth_mamba_on_path() -> Path:
    for cand in earth_mamba_roots():
        if (cand / "earth_mamba").is_dir():
            s = str(cand.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            return cand
    raise FileNotFoundError(
        "找不到 earth_mamba 包。请设置 EARTH_MAMBA_ROOT 指向含 earth_mamba/ 的目录。"
    )
