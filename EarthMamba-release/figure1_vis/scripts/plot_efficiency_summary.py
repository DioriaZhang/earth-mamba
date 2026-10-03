#!/usr/bin/env python3
"""
Figure 1 panel (d): Efficiency Summary — 224 cls 协议单图散点。
PDF 尺寸与 fig_erf_rotation.pdf 一致。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

from common.plotting import AAAI_RC, SIZE_WIDE

DATA_DIR = os.path.join(PKG_ROOT, "data", "fig_efficiency")
CSV_PATH = os.path.join(DATA_DIR, "efficiency_summary.csv")
OUT_DIR = os.path.join(PKG_ROOT, "output", "fig_efficiency")

COL_FRAC = 0.32
AAAI_COL_W = 3.33
PANEL_W = AAAI_COL_W * COL_FRAC
SCALE = PANEL_W / SIZE_WIDE[0]

EM_FLUO = "#FF1F3D"
MAMBA_BAND = (60, 70)
FIG_TITLE = "Efficiency Summary"
FIG_SUBTITLE = "224 cls protocol"

MODEL_COLOR = {
    "Earth-Mamba": EM_FLUO,
    "RVSA": "#6a7f94",
    "Satlas": "#7a8f7a",
    "SatMAE": "#8a8580",
    "RoMA": "#7a6888",
    "RSMamba": "#5c7a8a",
}
ADJUSTED = {"RoMA", "RSMamba"}
ANNOT_OFFSET = {
    "Earth-Mamba": (2, 3),
    "RVSA": (2, -3),
    "Satlas": (2, 2),
    "SatMAE": (2, 2),
    "RoMA": (2, 2),
    "RSMamba": (2, -3),
}


def _stack_figure_size(scale: float) -> tuple[float, float]:
    fig_h1 = SIZE_WIDE[1] * (PANEL_W / SIZE_WIDE[0])
    fig_h2 = fig_h1 * 0.88
    label_pad = 0.5 * scale
    title_fs = AAAI_RC.get("axes.titlesize", 10) * scale
    line_in = AAAI_RC["font.size"] * scale / 72.0
    margin_top_in = 0.03
    bottom_reserve_in = (
        AAAI_RC["xtick.labelsize"] * scale * 1.2
        + label_pad
        + AAAI_RC["axes.labelsize"] * scale * 1.05
        + label_pad * 0.5
        + title_fs * 1.15
    ) / 72.0
    row1_meta_in = 0.05 + line_in
    fig_h = margin_top_in + fig_h1 + row1_meta_in + fig_h2 + bottom_reserve_in
    return PANEL_W, fig_h


def _scaled_rc(scale: float) -> dict:
    def s(v: float) -> float:
        return v * scale

    rc = dict(AAAI_RC)
    rc.update({
        "font.size": s(AAAI_RC["font.size"]),
        "axes.labelsize": s(AAAI_RC["axes.labelsize"]),
        "axes.titlesize": s(AAAI_RC.get("axes.titlesize", 10)),
        "xtick.labelsize": s(AAAI_RC["xtick.labelsize"]),
        "ytick.labelsize": s(AAAI_RC["ytick.labelsize"]),
        "axes.linewidth": s(AAAI_RC.get("axes.linewidth", 0.8)),
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return rc


def _load_csv(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty: {path}")
    return rows


def _mem_marker_size(mem_gb: float, scale: float) -> float:
    """面积 ∝ peak memory（GB）。"""
    ref = 0.37
    return (float(mem_gb) / ref) ** 1.15 * (3.2 * scale * 10) ** 2


def _style_axes(ax, scale: float) -> None:
    lw = AAAI_RC.get("axes.linewidth", 0.8) * scale
    ax.tick_params(
        axis="both", which="major",
        width=0.55 * scale, length=1.2 * scale, pad=0.35 * scale,
        labelsize=AAAI_RC["xtick.labelsize"] * scale,
        direction="out",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(lw)


def _fig_y_from_bbox(fig, bbox, offset_pt: float = 0.0) -> float:
    y0_px = bbox.y0 - offset_pt / 72.0 * fig.dpi
    return y0_px / fig.bbox.height


def plot_efficiency_summary(csv_path: str, out_dir: str) -> None:
    mpl.rcParams.update(_scaled_rc(SCALE))
    rows = _load_csv(csv_path)
    fig_w, fig_h = _stack_figure_size(SCALE)
    title_fs = AAAI_RC.get("axes.titlesize", 10) * SCALE
    tick_fs = AAAI_RC["xtick.labelsize"] * SCALE
    label_pad = 0.5 * SCALE

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0.17, 0.20, 0.80, 0.72])

    ax.axvspan(MAMBA_BAND[0], MAMBA_BAND[1], color="#ececec", alpha=0.9, zorder=0, lw=0)
    ax.text(
        65, 72.15, "Mamba-family range",
        ha="center", va="bottom", fontsize=tick_fs * 0.72, color="#999999",
    )

    for r in rows:
        name = r["model"]
        lat = float(r["latency_ms"])
        acc = float(r["avg_score"])
        mem = float(r["peak_mem_gb"])
        mk = r["marker"]
        c = MODEL_COLOR.get(name, "#888888")
        is_em = name == "Earth-Mamba"
        is_adj = name in ADJUSTED
        size = _mem_marker_size(mem, SCALE)
        lw = 1.1 * SCALE if is_em else 0.85 * SCALE
        if is_adj:
            ax.scatter(
                lat, acc, marker=mk, s=size, facecolors="none",
                edgecolors=c, linewidths=lw, zorder=4,
            )
        else:
            ax.scatter(
                lat, acc, marker=mk, s=size, c=c,
                edgecolors="white" if not is_em else "#ffffff",
                linewidths=0.35 * SCALE, zorder=5 if is_em else 4,
                alpha=0.95 if is_em else 0.88,
            )
        ax.annotate(
            r["label"], (lat, acc), textcoords="offset points",
            xytext=ANNOT_OFFSET.get(name, (2, 2)),
            fontsize=tick_fs * 0.78, color=c if is_em else "#555555",
            ha="left", va="center", clip_on=False,
        )

    ax.annotate(
        "highest avg.", (70, 81.36), textcoords="offset points", xytext=(4, 6),
        fontsize=tick_fs * 0.75, color=EM_FLUO, ha="left", va="bottom",
    )

    ax.set_xlabel("Latency (ms/img) ↓", labelpad=0.22 * SCALE)
    ax.set_ylabel("Avg. score (%) ↑", labelpad=0.18 * SCALE)
    ax.yaxis.set_label_coords(-0.11, 0.5)
    ax.set_xlim(0, 75)
    ax.set_ylim(72, 82)
    ax.set_xticks([0, 15, 30, 45, 60, 75])
    ax.set_yticks([72, 74, 76, 78, 80, 82])
    ax.grid(True, linestyle="--", linewidth=0.3 * SCALE, alpha=0.35)
    _style_axes(ax, SCALE)

    ax.text(
        0.02, 0.02, "† adjusted",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=tick_fs * 0.68, color="#777777",
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlab_bb = ax.xaxis.label.get_window_extent(renderer)
    title_y = _fig_y_from_bbox(fig, xlab_bb, offset_pt=label_pad * 0.5 + title_fs * 0.5)
    sub_y = _fig_y_from_bbox(fig, xlab_bb, offset_pt=label_pad * 0.5)
    cx = ax.get_position().x0 + ax.get_position().width / 2.0
    fig.text(cx, title_y, FIG_TITLE, transform=fig.transFigure,
             ha="center", va="top", fontsize=title_fs, fontweight="bold")
    fig.text(cx, sub_y, FIG_SUBTITLE, transform=fig.transFigure,
             ha="center", va="top", fontsize=tick_fs * 0.82, color="#555555")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"fig_efficiency_summary.{ext}")
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)
    print(f"  figsize={fig_w:.3f}×{fig_h:.3f} in")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    plot_efficiency_summary(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
