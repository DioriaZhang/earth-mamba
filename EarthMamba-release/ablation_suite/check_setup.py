#!/usr/bin/env python3
"""Smoke test for ablation_suite imports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import variant_flags  # noqa: F401

print("variants:", ", ".join(variant_flags.VARIANT_ORDER))
print("train:", ", ".join(variant_flags.RUN_VARIANTS))
print("results:", "results/RESULTS.md" if (ROOT / "results" / "RESULTS.md").is_file() else "missing")
print("OK: ablation_suite")
