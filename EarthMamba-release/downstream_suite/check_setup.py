#!/usr/bin/env python3
"""Quick import-path check for downstream_suite."""

from __future__ import annotations

import bootstrap

bootstrap.setup()

import backbone  # noqa: F401

print("backbones:", ", ".join(backbone.BACKBONE_CHOICES))
print("main-table (6):", ", ".join(backbone.MAIN_TABLE_BACKBONES))
print("earth variants: earth-mamba (small ~91M), earth-mamba-b (base ~161M)")
print("OK: bootstrap + backbone")
