#!/usr/bin/env python3
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from variant_flags import FLAGS, RUN_VARIANTS, resolve_flags, should_run_training
from run_variant import build_command


def test_flags():
    assert resolve_flags("ID1") == (False, False, False)
    assert resolve_flags("ID9") == (True, True, True)
    assert resolve_flags("ID6") == (True, True, False)
    assert should_run_training("ID9") is False
    assert len(RUN_VARIANTS) == 7


def test_commands():
    for v in RUN_VARIANTS:
        cmd = build_command("dfc15", v, "/hy-tmp/task/DFC15", None, None, True, [])
        assert "task_dfc15.py" in " ".join(cmd)
        assert "--ablation_variant" in cmd
    skip = build_command("inria", "ID9", "/hy-tmp/task/INRIA", None, None, True, [])
    assert skip == []


if __name__ == "__main__":
    test_flags()
    test_commands()
    print("OK")
