#!/usr/bin/env python3
"""DIOR-R oriented bounding-box detection — unified ``--backbone`` (six models)."""

from __future__ import annotations

import bootstrap

bootstrap.setup()

from lib.dior_r.run import main  # noqa: E402

if __name__ == "__main__":
    main()
