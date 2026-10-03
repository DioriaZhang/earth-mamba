#!/usr/bin/env python3
"""检查 baselines2 权重与依赖（不依赖 baselines/ / downstream_code/）。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from shared.weights_manifest import CKPT_NAME_WARNINGS, PREFERRED_CKPT_NAMES
from shared.paths import BASELINES2_ROOT, backbone_weights_dir

MIN_BYTES = 100_000_000


def _status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def check_weight(model: str, folder: str) -> list[str]:
    lines: list[str] = []
    wdir = backbone_weights_dir(folder)
    preferred = PREFERRED_CKPT_NAMES.get(folder, [])
    pths = sorted(wdir.glob("*.pth")) if wdir.is_dir() else []

    lines.append(f"\n[{model}] {wdir}")
    if not pths:
        lines.append(f"  {_status(False)} 无 .pth 文件")
        lines.append(f"  期望: {', '.join(preferred) or '见 README'}")
        return lines

    has_preferred = False
    for p in pths:
        size_mb = p.stat().st_size / 1e6
        size_ok = p.stat().st_size >= MIN_BYTES
        warn_hints = [h for h in CKPT_NAME_WARNINGS.get(folder, []) if h in p.name.lower()]
        preferred_ok = p.name in preferred
        has_preferred = has_preferred or preferred_ok

        if warn_hints:
            flag = "WARN"
        elif preferred_ok and size_ok:
            flag = "OK"
        elif size_ok:
            flag = "WARN"
        else:
            flag = "FAIL"

        lines.append(f"  [{flag}] {p.name} ({size_mb:.0f} MB)")
        if warn_hints:
            lines.append(f"        文件名含 {warn_hints}，可能下错权重")
        if not size_ok:
            lines.append("        文件过小，可能下载不完整")

    if preferred and not has_preferred:
        lines.append(f"  [WARN] 无精确推荐文件名，将 fallback 扫描 *.pth")
    return lines


def check_modules() -> list[str]:
    lines = ["\n[本地模块]"]
    for mod in ("tasks.dfc15", "tasks.inria", "tasks.second", "tasks.dior", "downstream_common", "backbone_registry"):
        try:
            importlib.import_module(mod)
            lines.append(f"  [OK] {mod}")
        except Exception as e:
            lines.append(f"  [FAIL] {mod}: {e}")
    return lines


def check_imports() -> list[str]:
    lines = ["\n[Python 依赖]"]
    for mod, pkg in [("timm", "timm"), ("torchvision", "torchvision"), ("torch", "torch")]:
        try:
            importlib.import_module(mod)
            lines.append(f"  [OK] {pkg}")
        except ImportError:
            lines.append(f"  [FAIL] {pkg}")
    return lines


def main() -> int:
    print("=" * 60)
    print(f"baselines2 环境检查  root={BASELINES2_ROOT}")
    print("=" * 60)

    lines = check_modules() + check_imports()
    lines += check_weight("RVSA", "RVSA")
    lines += check_weight("SatlasPretrain-Aerial", "SatlasPretrain_Aerial")
    print("\n".join(lines))

    text = "\n".join(lines)
    if "FAIL" in text:
        print("\n结论: 有问题需修复")
        return 1
    if "WARN" in text:
        print("\n结论: 可运行（有警告）")
        return 0
    print("\n结论: 检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
