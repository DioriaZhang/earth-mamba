"""Python 侧环境初始化（不依赖 source env_a100.sh）。"""
from __future__ import annotations

import os
from pathlib import Path

from lib.linux_env import clean_str, normalize_linux_env

RUN_DIR = Path(__file__).resolve().parents[1]


def bootstrap_env() -> None:
    normalize_linux_env()
    project = RUN_DIR.parent.parent
    defaults = {
        "CKPT": "/hy-tmp/CKPT/PN-log13-ep14/checkpoint.pth",
        "DATA_INRIA": "/hy-tmp/task/INRIA",
        "DATA_DFC15": "/hy-tmp/task/DFC15",
        "DATA_DIOR": "/hy-tmp/task/DIOR",
        "RESULTS_ROOT": "/hy-tmp/downstream_results/ablation_study",
        "CUDA_VISIBLE_DEVICES": "0",
        "NUM_WORKERS": "8",
        "PREFETCH_FACTOR": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "EARTH_MAMBA_ROOT": str(project / "earth-mamba"),
        "PROJECT_ROOT": str(project),
    }
    for key, val in defaults.items():
        cur = os.environ.get(key)
        os.environ[key] = clean_str(cur if cur else val)
    os.environ["RUN_DIR"] = str(RUN_DIR)
    earth = clean_str(os.environ["EARTH_MAMBA_ROOT"])
    run = str(RUN_DIR)
    pp = clean_str(os.environ.get("PYTHONPATH", ""))
    parts = [p for p in (run, earth, pp) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(parts))
