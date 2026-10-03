from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES2 = ROOT / "baselines2"
if str(BASELINES2) not in sys.path:
    sys.path.insert(0, str(BASELINES2))

from downstream_common import *  # noqa: F401,F403

