#!/usr/bin/env python3
"""归档消融 results.json：按 数据集 + variant 保存最优 epoch 的整条 history 记录。

每个条目保留：
  - dataset / variant 标识
  - results.json 路径
  - best_epoch
  - best_epoch_record  ← history 里该 epoch 的完整 {}
  - run_meta           ← 文件顶层的 args、best_* 等（不含完整 history，避免重复）

输出::
  {output_dir}/raw_archive.json          # 总存档（下载这个即可）
  {output_dir}/raw/{dataset}_{variant}.json  # 分片，便于核对

用法::
  python3 ablation_study/run/archive_raw_results.py \\
    --results_root /hy-tmp/downstream_results/ablation_study
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

RUN_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = RUN_DIR

sys.path.insert(0, str(RUN_DIR))
import lib.linux_env  # noqa: F401
from lib.linux_env import clean_obj, clean_str
from variant_flags import LOGIC, REFERENCE_METRICS, VARIANT_ORDER

try:
    import yaml
except ImportError:
    yaml = None

TASKS = ("dfc15", "inria", "dior")

# 用哪个字段在 history 里选最优 epoch
_BEST_KEYS: dict[str, tuple[str, ...]] = {
    "dfc15": ("macro_mAP", "macro_map", "mAP"),
    "inria": ("val_miou",),
    "dior": ("mAP@0.5",),
}

_REF_KEY: dict[str, str] = {
    "dfc15": "dfc15_macro_map",
    "inria": "inria_val_miou",
    "dior": "dior_map50",
}


def _load_variants() -> dict:
    path = RUN_DIR / "configs" / "variants.yaml"
    if yaml is not None and path.is_file():
        with open(path, encoding="utf-8", newline="\n") as f:
            return clean_obj(yaml.safe_load(f) or {})
    from run_variant import _VARIANTS_FALLBACK
    return _VARIANTS_FALLBACK


def _find_results_json(task_dir: Path, variant: str) -> Optional[Path]:
    direct = task_dir / variant / "results.json"
    if direct.is_file():
        return direct
    for pattern in (f"{variant}*/results.json", f"**/{variant}/results.json"):
        hits = sorted(task_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if hits:
            return hits[0]
    return None


def _metric_from_row(row: dict, keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in row and row[k] is not None:
            return float(row[k])
    return None


def _pick_best_row(history: list, task: str) -> tuple[Optional[dict], Optional[int], Optional[str], Optional[float]]:
    keys = _BEST_KEYS[task]
    best_row, best_ep, best_key, best_val = None, None, None, None
    for row in history:
        if task == "dior" and not row.get("evaluated", True):
            continue
        v = _metric_from_row(row, keys)
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_row, best_ep, best_key, best_val = row, int(row.get("epoch", 0)), keys[0], v
    return best_row, best_ep, best_key, best_val


def _run_meta_without_history(data: dict) -> dict:
    meta = {k: v for k, v in data.items() if k != "history"}
    return meta


def _reference_entry(task: str, variant: str, vdef: dict) -> dict:
    ref_key = _REF_KEY[task]
    ref_val = (vdef.get("reference") or {}).get(ref_key) or REFERENCE_METRICS.get(ref_key)
    return {
        "dataset": task,
        "variant": variant,
        "logic": LOGIC[variant],
        "source": "main_table_reference",
        "results_json": None,
        "best_epoch": None,
        "primary_metric": ref_key,
        "primary_metric_value": ref_val,
        "best_epoch_record": {
            "note": "ID9 未重跑，无 history；下列为引用主表数值",
            ref_key: ref_val,
        },
        "run_meta": {"reference": vdef.get("reference")},
    }


def _entry_from_file(task: str, variant: str, rpath: Path) -> dict:
    with open(rpath, encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history") or []
    best_row, best_ep, metric_key, metric_val = _pick_best_row(history, task)

    # history 为空时，用顶层 best_* 拼一条 synthetic record
    if best_row is None:
        best_ep = data.get("best_epoch")
        if task == "inria":
            metric_key, metric_val = "val_miou", data.get("best_val_miou")
        elif task == "dfc15":
            bm = data.get("best_metrics") or {}
            metric_key = "macro_mAP"
            metric_val = bm.get("macro_mAP") or bm.get("macro_map")
        else:
            metric_key, metric_val = "mAP@0.5", data.get("best_mAP@0.5")
        if metric_val is not None:
            best_row = {
                "epoch": best_ep,
                metric_key: metric_val,
                "_synthetic": True,
                "_note": "results.json 无可用 history，由顶层 best_* 字段重建",
            }

    return {
        "dataset": task,
        "variant": variant,
        "logic": LOGIC[variant],
        "source": "results.json",
        "results_json": str(rpath),
        "best_epoch": best_ep,
        "primary_metric": metric_key,
        "primary_metric_value": metric_val,
        "best_epoch_record": best_row,
        "run_meta": _run_meta_without_history(data),
    }


def build_archive(results_root: Path) -> dict[str, Any]:
    variants_cfg = _load_variants()
    archive: dict[str, Any] = {
        "meta": {
            "results_root": str(results_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tasks": list(TASKS),
            "variants": list(VARIANT_ORDER),
        },
        "by_dataset": {t: {} for t in TASKS},
        "flat": {},
    }

    for variant in VARIANT_ORDER:
        vdef = (variants_cfg.get("variants") or {}).get(variant, {})
        is_ref = variant == "ID9" and not vdef.get("run", True)

        for task in TASKS:
            key = f"{task}_{variant}"
            if is_ref:
                entry = _reference_entry(task, variant, vdef)
            else:
                task_dir = results_root / task
                rpath = _find_results_json(task_dir, variant) if task_dir.is_dir() else None
                if rpath is None:
                    entry = {
                        "dataset": task,
                        "variant": variant,
                        "logic": LOGIC[variant],
                        "source": "missing",
                        "results_json": None,
                        "best_epoch": None,
                        "primary_metric": None,
                        "primary_metric_value": None,
                        "best_epoch_record": None,
                        "run_meta": None,
                    }
                else:
                    entry = _entry_from_file(task, variant, rpath)

            archive["by_dataset"][task][variant] = entry
            archive["flat"][key] = entry

    return archive


def main() -> None:
    p = argparse.ArgumentParser(description="归档各 dataset/variant 最优 epoch 的完整 history 记录")
    p.add_argument("--results_root", default=os.environ.get("RESULTS_ROOT", str(RUN_DIR / "outputs")))
    p.add_argument("--output-dir", default=None, help="default: results/ inside this folder")
    args = p.parse_args()

    root = Path(clean_str(args.results_root))
    out_dir = Path(clean_str(args.output_dir)) if args.output_dir else ABLATION_ROOT / "results"
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive = build_archive(root)

    master = out_dir / "raw_archive.json"
    master.write_text(json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8")

    n_ok = 0
    for key, entry in archive["flat"].items():
        shard = raw_dir / f"{key}.json"
        shard.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8")
        if entry.get("source") == "results.json":
            n_ok += 1

    print(f"Wrote {master}")
    print(f"Wrote {len(archive['flat'])} shards under {raw_dir}/")
    print(f"  ({n_ok} from results.json, rest reference/missing)")
    print("\n下载 raw_archive.json 即可；分片示例: raw/dior_ID3.json")


if __name__ == "__main__":
    main()
