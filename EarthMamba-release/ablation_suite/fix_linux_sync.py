#!/usr/bin/env python3
"""去掉 ablation_study/run 下 .sh/.yaml/.py 的 Windows CRLF。

用法（不依赖任何 .sh，直接从 python 调用）:
  python3 ablation_study/run/fix_linux_sync.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
_SUFFIXES = {".sh", ".yaml", ".py", ".md"}


def fix_file(path: Path) -> bool:
    raw = path.read_bytes()
    if b"\r" not in raw:
        return False
    path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    return True


def main() -> None:
    fixed: list[str] = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in _SUFFIXES:
            continue
        if "__pycache__" in p.parts:
            continue
        if fix_file(p):
            fixed.append(str(p.relative_to(ROOT)))
    if fixed:
        print(f"[fix_linux_sync] fixed {len(fixed)} files:")
        for name in fixed:
            print(f"  {name}")
    else:
        print("[fix_linux_sync] all files already LF")
    print("[fix_linux_sync] done")


if __name__ == "__main__":
    main()
