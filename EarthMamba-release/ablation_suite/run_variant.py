#!/usr/bin/env python3
"""Unified ablation entry — runs task scripts inside this folder only."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import lib.linux_env  # noqa: F401
from lib.linux_env import clean_obj, clean_str, normalize_linux_env
from variant_flags import LABELS, LOGIC, RUN_VARIANTS, resolve_flags, should_run_training

try:
    import yaml
except ImportError:
    yaml = None

_VARIANTS_FALLBACK = {
    "checkpoint": "/hy-tmp/CKPT/PN-log13-ep14/checkpoint.pth",
    "variants": {v: {"run": v != "ID9"} for v in (
        "ID1", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9",
    )},
}

_TASK_SCRIPTS = {
    "inria": "task_inria.py",
    "dfc15": "task_dfc15.py",
    "dior": "task_dior.py",
}

_INRIA_HP = {"patch_size": 512, "epochs": 80, "batch_size": 6, "lr": 1e-3, "encoder_lr": 1e-5,
             "adapter_lr": 1e-3, "weight_decay": 1e-4, "dense_adapter": "gated_pyramid",
             "ssm_version": "mamba3", "seed": 42, "building_weight": 1.0, "dice_weight": 0.5,
             "samples_per_tile": 8, "val_ratio": 0.2, "amp_dtype": "bf16",
             "num_workers": 8, "prefetch_factor": 4, "persistent_workers": True}
_DFC15_HP = {"img_size": 224, "epochs": 80, "batch_size": 48, "lr": 1e-3, "encoder_lr": 1e-5,
             "weight_decay": 0.05, "ssm_version": "mamba3", "seed": 42, "amp_dtype": "bf16",
             "num_workers": 8, "prefetch_factor": 4, "persistent_workers": True}
_DIOR_HP = {"img_size": 512, "epochs": 20, "batch_size": 4, "lr": 2e-4, "encoder_lr": 1e-5,
            "adapter_lr": 2e-4, "weight_decay": 1e-4, "grad_accum": 2, "dense_adapter": "semantic_pyramid",
            "ssm_version": "mamba3", "seed": 42, "amp_dtype": "none", "warmup_epochs": 2, "eval_interval": 5,
            "num_workers": 8, "prefetch_factor": 4, "persistent_workers": True}


def _load_yaml(path: Path) -> dict:
    if yaml and path.is_file():
        with open(path, encoding="utf-8", newline="\n") as f:
            return clean_obj(yaml.safe_load(f) or {})
    return {}


def _hp(task: str) -> dict:
    cfg = _load_yaml(RUN_DIR / "configs" / f"{task}.yaml")
    hp = dict(cfg.get("hyperparams") or {"inria": _INRIA_HP, "dfc15": _DFC15_HP, "dior": _DIOR_HP}[task])
    hp = clean_obj(hp)
    if os.environ.get("NUM_WORKERS"):
        hp["num_workers"] = int(os.environ["NUM_WORKERS"])
    if os.environ.get("PREFETCH_FACTOR"):
        hp["prefetch_factor"] = int(os.environ["PREFETCH_FACTOR"])
    return hp


def _perf_cli(hp: dict) -> list[str]:
    args: list[str] = []
    if hp.get("num_workers") is not None:
        args += ["--num_workers", str(hp["num_workers"])]
    if hp.get("prefetch_factor") is not None:
        args += ["--prefetch_factor", str(hp["prefetch_factor"])]
    if hp.get("persistent_workers") is False:
        args.append("--no_persistent_workers")
    return args


def _ckpt(override: str | None) -> str:
    if override:
        return clean_str(override)
    v = _load_yaml(RUN_DIR / "configs" / "variants.yaml")
    return clean_str(v.get("checkpoint", _VARIANTS_FALLBACK["checkpoint"]))


def _out_dir(task: str, variant: str, override: str | None) -> str:
    if override:
        return clean_str(override)
    root = clean_str(os.environ.get("RESULTS_ROOT", str(RUN_DIR / "outputs")))
    return str(Path(root) / task / variant)


def _flags_cli(use_sparse: bool, use_graph: bool, use_armg: bool) -> list[str]:
    a: list[str] = []
    if not use_sparse:
        a.append("--no_sparse")
    if not use_graph:
        a.append("--no_graph")
    if not use_armg:
        a.append("--no_armg")
    return a


def build_command(task, variant, data_dir, ckpt, output_dir, dry_run, extra) -> list[str]:
    vdef = (_load_yaml(RUN_DIR / "configs" / "variants.yaml").get("variants") or {}).get(variant, {})
    if variant == "ID9" and not vdef.get("run", True):
        print(f"[skip] ID9 ({LOGIC['ID9']}) — 引用主表")
        return []
    if not should_run_training(variant):
        return []

    use_sparse, use_graph, use_armg = resolve_flags(variant)
    hp = _hp(task)
    script = RUN_DIR / _TASK_SCRIPTS[task]
    common = [
        sys.executable, str(script),
        "--data_dir", data_dir,
        "--ckpt", _ckpt(ckpt),
        "--output_dir", _out_dir(task, variant, output_dir),
        "--ablation_variant", variant,
    ] + _flags_cli(use_sparse, use_graph, use_armg)

    if task == "inria":
        cmd = common + [
            "--patch_size", str(hp["patch_size"]), "--epochs", str(hp["epochs"]),
            "--batch_size", str(hp["batch_size"]), "--lr", str(hp["lr"]),
            "--encoder_lr", str(hp["encoder_lr"]), "--adapter_lr", str(hp["adapter_lr"]),
            "--weight_decay", str(hp["weight_decay"]), "--dense_adapter", hp["dense_adapter"],
            "--ssm_version", hp["ssm_version"], "--seed", str(hp["seed"]),
            "--building_weight", str(hp["building_weight"]), "--dice_weight", str(hp["dice_weight"]),
            "--samples_per_tile", str(hp["samples_per_tile"]), "--val_ratio", str(hp["val_ratio"]),
            "--amp_dtype", hp.get("amp_dtype", "bf16"),
        ] + _perf_cli(hp)
    elif task == "dfc15":
        cmd = common + [
            "--img_size", str(hp["img_size"]), "--epochs", str(hp["epochs"]),
            "--batch_size", str(hp["batch_size"]), "--lr", str(hp["lr"]),
            "--encoder_lr", str(hp["encoder_lr"]), "--weight_decay", str(hp["weight_decay"]),
            "--ssm_version", hp["ssm_version"], "--seed", str(hp["seed"]),
            "--amp_dtype", hp.get("amp_dtype", "bf16"),
        ] + _perf_cli(hp)
    else:
        cmd = common + [
            "--img_size", str(hp["img_size"]), "--epochs", str(hp["epochs"]),
            "--batch_size", str(hp["batch_size"]), "--lr", str(hp["lr"]),
            "--encoder_lr", str(hp["encoder_lr"]), "--adapter_lr", str(hp["adapter_lr"]),
            "--weight_decay", str(hp["weight_decay"]), "--grad_accum", str(hp["grad_accum"]),
            "--dense_adapter", hp["dense_adapter"], "--ssm_version", hp["ssm_version"],
            "--seed", str(hp["seed"]), "--amp_dtype", hp.get("amp_dtype", "none"),
            "--warmup_epochs", str(hp["warmup_epochs"]), "--eval_interval", str(hp["eval_interval"]),
        ] + _perf_cli(hp)
    if dry_run:
        cmd.append("--dry_run")
    cmd.extend(extra)
    return cmd


def main():
    from variant_flags import VARIANT_ORDER
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["inria", "dfc15", "dior"])
    p.add_argument("--variant", required=True, choices=VARIANT_ORDER)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--print_only", action="store_true")
    args, extra = p.parse_known_args()
    args.data_dir = clean_str(args.data_dir)
    if args.ckpt:
        args.ckpt = clean_str(args.ckpt)
    if args.output_dir:
        args.output_dir = clean_str(args.output_dir)
    cmd = build_command(args.task, args.variant, args.data_dir, args.ckpt, args.output_dir, args.dry_run, extra)
    if not cmd:
        return
    print(" ".join(cmd))
    if args.print_only:
        return
    normalize_linux_env()
    env = {k: clean_str(v) if isinstance(v, str) else v for k, v in os.environ.items()}
    pp = env.get("PYTHONPATH", "")
    if str(RUN_DIR) not in pp:
        env["PYTHONPATH"] = f"{RUN_DIR}{os.pathsep}{pp}" if pp else str(RUN_DIR)
    sys.exit(subprocess.call(cmd, cwd=str(RUN_DIR), env=env))


if __name__ == "__main__":
    main()
