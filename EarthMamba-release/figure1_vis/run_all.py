#!/usr/bin/env python3
"""一键重绘 Figure 1 四个 panel。"""
from __future__ import annotations

import os
import subprocess
import sys

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(PKG_ROOT, "scripts")

PANELS = [
    ("plot_fig2_noise_radar.py", "panel (a) noise radar"),
    ("plot_erf_rotation.py", "panel (b) ERF + rotation"),
    ("plot_downstream.py", "panel (c) downstream"),
    ("plot_efficiency_summary.py", "panel (d) efficiency"),
]


def main() -> int:
    py = sys.executable
    for script, label in PANELS:
        path = os.path.join(SCRIPTS, script)
        print(f"\n=== {label} ===")
        rc = subprocess.call([py, path], cwd=SCRIPTS)
        if rc != 0:
            print(f"FAILED: {script}", file=sys.stderr)
            return rc
    print("\nAll panels written to figure1_vis/output/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
