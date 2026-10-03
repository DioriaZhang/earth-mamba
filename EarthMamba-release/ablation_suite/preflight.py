#!/usr/bin/env python3
"""启动前检查：变体表、LF 换行、命令路径、encoder/任务冒烟。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import lib.linux_env  # noqa: F401
from lib.bootstrap_env import bootstrap_env
from lib.linux_env import clean_str, normalize_linux_env
from run_variant import build_command
from variant_flags import RUN_VARIANTS, resolve_flags, should_run_training

CKPT = clean_str(os.environ.get("CKPT", "/hy-tmp/CKPT/PN-log13-ep14/checkpoint.pth"))
DATA_INRIA = clean_str(os.environ.get("DATA_INRIA", "/hy-tmp/task/INRIA"))
DATA_DIOR = clean_str(os.environ.get("DATA_DIOR", "/hy-tmp/task/DIOR"))
DATA_DFC15 = clean_str(os.environ.get("DATA_DFC15", "/hy-tmp/task/DFC15"))

_TEXT_GLOBS = ("*.sh", "*.yaml", "*.py")


def check_variant_table():
    assert resolve_flags("ID1") == (False, False, False)
    assert resolve_flags("ID5") == (False, False, True)
    assert resolve_flags("ID9") == (True, True, True)
    assert should_run_training("ID9") is False
    assert len(RUN_VARIANTS) == 7


def _files_with_cr() -> list[str]:
    bad: list[str] = []
    for pattern in _TEXT_GLOBS:
        for p in RUN_DIR.rglob(pattern):
            if "__pycache__" in p.parts:
                continue
            if b"\r" in p.read_bytes():
                bad.append(str(p.relative_to(RUN_DIR)))
    return sorted(bad)


def check_text_files_lf():
    bad = _files_with_cr()
    if bad:
        raise SystemExit(
            "以下文件含 CRLF（Linux 会炸 pipefail / 路径 / CUDA env）:\n  "
            + "\n  ".join(bad)
            + "\n修复: python3 ablation_study/run/fix_linux_sync.py"
        )
    print("  text files LF OK")


def check_commands_no_cr():
    for task, data in (("inria", DATA_INRIA), ("dior", DATA_DIOR), ("dfc15", DATA_DFC15)):
        for v in ("ID1", "ID8"):
            cmd = build_command(task, v, data, CKPT, None, True, [])
            if not cmd:
                continue
            for arg in cmd:
                if "\r" in arg:
                    raise SystemExit(f"命令参数含 \\r: {arg!r} (检查 configs/*.yaml)")


def check_earth_mamba_import():
    from lib.paths import ensure_earth_mamba_on_path
    root = ensure_earth_mamba_on_path()
    from earth_mamba.models.earth_mamba_block import EarthMambaBlock
    blk = EarthMambaBlock(hidden_dim=96, use_sparse_ssm=False, use_graph=False, use_armg=True)
    assert blk.sparse_enable == 0.0 and blk.graph_enable == 0.0 and blk.armg_enable == 1.0
    print(f"  earth_mamba OK @ {root}")


def check_encoder_build():
    from lib.earthmamba import build_cls_encoder, build_dense_encoder
    build_dense_encoder(None, 512, dense_adapter="gated_pyramid",
                        use_sparse_ssm=False, use_graph=False, use_armg=False)
    build_cls_encoder(None, 224, use_sparse_ssm=True, use_graph=True, use_armg=True)
    print("  encoder build OK")


def _subprocess_env() -> dict[str, str]:
    normalize_linux_env()
    env = {k: clean_str(v) if isinstance(v, str) else v for k, v in os.environ.items()}
    pp = env.get("PYTHONPATH", "")
    if str(RUN_DIR) not in pp.split(os.pathsep):
        env["PYTHONPATH"] = f"{RUN_DIR}{os.pathsep}{pp}" if pp else str(RUN_DIR)
    return env


def _run_dry(task: str, variant: str, data_dir: str) -> None:
    cmd = build_command(task, variant, data_dir, CKPT, None, True, [])
    if not cmd:
        return
    print(f"  smoke {task} {variant} ...")
    subprocess.run(cmd, cwd=str(RUN_DIR), check=True, env=_subprocess_env())


def check_task_dry_run():
    for task, data in (("inria", DATA_INRIA), ("dior", DATA_DIOR), ("dfc15", DATA_DFC15)):
        _run_dry(task, "ID1", data)


if __name__ == "__main__":
    bootstrap_env()
    check_variant_table()
    check_text_files_lf()
    check_commands_no_cr()
    check_earth_mamba_import()
    check_encoder_build()
    check_task_dry_run()
    print("preflight OK")
