#!/usr/bin/env python3
"""DIOR 消融批量跑 ID1-ID8（不依赖 .sh）。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import lib.linux_env  # noqa: F401
from lib.bootstrap_env import bootstrap_env
from lib.linux_env import clean_str
from run_variant import build_command


def main() -> None:
    import os
    bootstrap_env()
    data = clean_str(os.environ.get("DATA_DIOR", "/hy-tmp/task/DIOR"))
    ckpt = clean_str(os.environ.get("CKPT", "/hy-tmp/CKPT/PN-log13-ep14/checkpoint.pth"))
    root = clean_str(os.environ.get("RESULTS_ROOT", "/hy-tmp/downstream_results/ablation_study"))
    for v in ("ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8"):
        print(f"========== DIOR {v} ==========")
        cmd = build_command("dior", v, data, ckpt, f"{root}/dior/{v}", False, [])
        if not cmd:
            continue
        subprocess.run(cmd, cwd=str(RUN_DIR), check=True)
    print(f"DIOR done. Collect: python {RUN_DIR}/collect_results.py --results_root {root}")


if __name__ == "__main__":
    main()
