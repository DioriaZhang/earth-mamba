#!/usr/bin/env python3
"""汇总三数据集 × 各 variant 的 results.json：最优 epoch 及对应指标。

目录约定::
  {results_root}/dfc15/ID1/results.json
  {results_root}/inria/ID3/results.json
  {results_root}/dior/ID8/results.json

用法::
  python3 ablation_study/run/collect_results.py \\
    --results_root /hy-tmp/downstream_results/ablation_study
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

RUN_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = RUN_DIR
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(RUN_DIR.parent)))

sys.path.insert(0, str(RUN_DIR))
import lib.linux_env  # noqa: F401
from lib.linux_env import clean_str
from variant_flags import LOGIC, REFERENCE_METRICS, TABLE_COLUMNS, VARIANT_ORDER

try:
    import yaml
except ImportError:
    yaml = None

TASKS = ("dfc15", "inria", "dior")

TASK_SPEC: dict[str, dict[str, str]] = {
    "dfc15": {
        "out_metric": "dfc15_macro_mAP",
        "out_epoch": "dfc15_best_epoch",
        "header": "DFC15 macro-mAP",
    },
    "inria": {
        "out_metric": "inria_val_mIoU",
        "out_epoch": "inria_best_epoch",
        "header": "INRIA mIoU",
    },
    "dior": {
        "out_metric": "dior_mAP@0.5",
        "out_epoch": "dior_best_epoch",
        "header": "DIOR mAP@0.5",
    },
}


def _load_variants() -> dict:
    path = RUN_DIR / "configs" / "variants.yaml"
    if yaml is not None and path.is_file():
        with open(path, encoding="utf-8", newline="\n") as f:
            from lib.linux_env import clean_obj
            return clean_obj(yaml.safe_load(f) or {})
    from run_variant import _VARIANTS_FALLBACK
    return _VARIANTS_FALLBACK


def _to_percent(v: float) -> float:
    return float(v) * 100.0 if v <= 1.0 else float(v)


def _find_results_json(task_dir: Path, variant: str) -> Optional[Path]:
    direct = task_dir / variant / "results.json"
    if direct.is_file():
        return direct
    for pattern in (f"{variant}*/results.json", f"**/{variant}/results.json"):
        candidates = sorted(task_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def _best_from_history(history: list, metric_keys: tuple[str, ...]) -> tuple[Optional[float], Optional[int]]:
    best_v, best_ep = None, None
    for row in history:
        if not row.get("evaluated", True):
            continue
        v = None
        for k in metric_keys:
            if k in row and row[k] is not None:
                v = float(row[k])
                break
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v, best_ep = v, int(row.get("epoch", 0))
    if best_v is None:
        return None, None
    return _to_percent(best_v), best_ep


def _parse_inria(data: dict) -> tuple[Optional[float], Optional[int]]:
    miou = data.get("best_val_miou")
    ep = data.get("best_epoch")
    if miou is not None:
        return _to_percent(float(miou)), ep
    return _best_from_history(data.get("history") or [], ("val_miou",))


def _parse_dfc15(data: dict) -> tuple[Optional[float], Optional[int]]:
    bm = data.get("best_metrics") or {}
    m = bm.get("macro_mAP") or bm.get("macro_map") or bm.get("mAP")
    ep = data.get("best_epoch")
    if m is not None:
        return _to_percent(float(m)), ep
    hist = data.get("history") or []
    if hist:
        best = max(hist, key=lambda x: x.get("macro_mAP", x.get("macro_map", 0)))
        m2 = best.get("macro_mAP") or best.get("macro_map")
        if m2 is not None:
            return _to_percent(float(m2)), best.get("epoch")
    return None, ep


def _parse_dior(data: dict) -> tuple[Optional[float], Optional[int]]:
    m = data.get("best_mAP@0.5") or data.get("best_map50")
    ep = data.get("best_epoch")
    if m is not None and (float(m) > 0 or ep):
        return _to_percent(float(m)), ep
    return _best_from_history(data.get("history") or [], ("mAP@0.5", "map50"))


PARSERS = {"inria": _parse_inria, "dfc15": _parse_dfc15, "dior": _parse_dior}

REF_MAP = {
    "dfc15": ("dfc15_macro_map", "dfc15_macro_mAP"),
    "inria": ("inria_val_miou", "inria_val_mIoU"),
    "dior": ("dior_map50", "dior_mAP@0.5"),
}


def collect(results_root: Path) -> dict[str, dict[str, Any]]:
    variants_cfg = _load_variants()
    out: dict[str, dict[str, Any]] = {v: {"variant": v, "logic": LOGIC[v]} for v in VARIANT_ORDER}

    for variant in VARIANT_ORDER:
        vdef = (variants_cfg.get("variants") or {}).get(variant, {})
        is_ref = variant == "ID9" and not vdef.get("run", True)

        for task in TASKS:
            spec = TASK_SPEC[task]
            mk, ek = spec["out_metric"], spec["out_epoch"]

            if is_ref:
                ref_key, _ = REF_MAP[task]
                ref = (vdef.get("reference") or {}).get(ref_key) or REFERENCE_METRICS.get(ref_key)
                out[variant][mk] = float(ref) if ref is not None else None
                out[variant][ek] = None
                out[variant][f"{task}_source"] = "main_table_reference"
                continue

            task_dir = results_root / task
            rpath = _find_results_json(task_dir, variant) if task_dir.is_dir() else None
            if rpath is None:
                out[variant][mk] = None
                out[variant][ek] = None
                out[variant][f"{task}_source"] = "missing"
                continue

            with open(rpath, encoding="utf-8") as f:
                data = json.load(f)
            metric, ep = PARSERS[task](data)
            out[variant][mk] = metric
            out[variant][ek] = ep
            out[variant][f"{task}_source"] = str(rpath)
            out[variant][f"{task}_results"] = str(rpath)

    return out


def _fmt_metric(v: Any, with_pct: bool = True) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}%" if with_pct else f"{v:.2f}"
    return str(v)


def _fmt_ep(v: Any) -> str:
    if v is None:
        return "—"
    return str(int(v))


def render_table_md(data: dict[str, dict[str, Any]], *, with_epoch: bool = True) -> str:
    lines = [
        "# EarthMamba 三模块组合消融 — 最优 epoch 指标",
        "",
        "> 由 `ablation_study/run/collect_results.py` 从各 `results.json` 提取。",
        "> ID9（Full）指标引用主表，无 best_epoch。",
        "",
    ]
    if with_epoch:
        lines.append(
            "| ID | A | B | C | Logic | DFC15 macro-mAP | ep | INRIA mIoU | ep | DIOR mAP@0.5 | ep |"
        )
        lines.append("|:--:|:--:|:--:|:--:|:---|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| ID | A | B | C | Logic | DFC15 macro-mAP | INRIA mIoU | DIOR mAP@0.5 |")
        lines.append("|:--:|:--:|:--:|:--:|:---|---:|---:|---:|")

    for vid in VARIANT_ORDER:
        row = data[vid]
        cols = TABLE_COLUMNS[vid]
        num = vid[2:] if vid.startswith("ID") else vid
        a = "✓" if cols["sparse"] else ""
        b = "✓" if cols["graph"] else ""
        c = "✓" if cols["armg"] else ""
        dfc = _fmt_metric(row.get("dfc15_macro_mAP"))
        inr = _fmt_metric(row.get("inria_val_mIoU"))
        dio = _fmt_metric(row.get("dior_mAP@0.5"))
        if with_epoch:
            lines.append(
                f"| {num} | {a} | {b} | {c} | {LOGIC[vid]} | {dfc} | {_fmt_ep(row.get('dfc15_best_epoch'))} "
                f"| {inr} | {_fmt_ep(row.get('inria_best_epoch'))} | {dio} | {_fmt_ep(row.get('dior_best_epoch'))} |"
            )
        else:
            lines.append(f"| {num} | {a} | {b} | {c} | {LOGIC[vid]} | {dfc} | {inr} | {dio} |")
    lines.append("")
    return "\n".join(lines)


def render_per_task_md(data: dict[str, dict[str, Any]]) -> str:
    lines = ["# 分数据集最优结果", ""]
    for task in TASKS:
        spec = TASK_SPEC[task]
        lines.append(f"## {spec['header']}")
        lines.append("")
        lines.append("| Variant | best_epoch | metric | results.json |")
        lines.append("|:-------:|-----------:|-------:|:-------------|")
        mk, ek = spec["out_metric"], spec["out_epoch"]
        for vid in VARIANT_ORDER:
            row = data[vid]
            src = row.get(f"{task}_source", "")
            src_short = Path(src).as_posix() if src and src not in ("missing", "main_table_reference") else src
            lines.append(
                f"| {vid} | {_fmt_ep(row.get(ek))} | {_fmt_metric(row.get(mk))} | {src_short or '—'} |"
            )
        lines.append("")
    return "\n".join(lines)


def print_console(data: dict[str, dict[str, Any]]) -> None:
    print("\n=== Ablation best metrics (metric @ best_epoch) ===\n")
    for vid in VARIANT_ORDER:
        row = data[vid]
        print(f"[{vid}] {LOGIC[vid]}")
        for task in TASKS:
            spec = TASK_SPEC[task]
            mk, ek = spec["out_metric"], spec["out_epoch"]
            m, e = row.get(mk), row.get(ek)
            src = row.get(f"{task}_source", "")
            if m is None:
                print(f"  {task:6s}: —  ({src})")
            elif e is None:
                print(f"  {task:6s}: {m:.2f}%  (ref)")
            else:
                print(f"  {task:6s}: {m:.2f}% @ epoch {e}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(description="从 results.json 汇总各 variant 最优 epoch 指标")
    p.add_argument("--results_root", default=os.environ.get("RESULTS_ROOT", str(RUN_DIR / "outputs")))
    p.add_argument("--output-dir", default=None, help="default: results/ inside this folder")
    p.add_argument("--no-epoch-cols", action="store_true", help="主表不显示 epoch 列")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    root = Path(clean_str(args.results_root))
    out_dir = Path(clean_str(args.output_dir)) if args.output_dir else ABLATION_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect(root)

    summary_json = out_dir / "ablation_summary.json"
    summary_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    table_md = out_dir / "ablation_table.md"
    table_md.write_text(render_table_md(data, with_epoch=not args.no_epoch_cols), encoding="utf-8")

    per_task_md = out_dir / "ablation_by_task.md"
    per_task_md.write_text(render_per_task_md(data), encoding="utf-8")

    if not args.quiet:
        print_console(data)
    print(f"Wrote {summary_json}")
    print(f"Wrote {table_md}")
    print(f"Wrote {per_task_md}")


if __name__ == "__main__":
    main()
