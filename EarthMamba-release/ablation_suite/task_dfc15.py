#!/usr/bin/env python3
import sys
from pathlib import Path

_RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(_RUN))
import lib.linux_env  # noqa: F401

from lib.dfc15 import main

if __name__ == "__main__":
    main()
