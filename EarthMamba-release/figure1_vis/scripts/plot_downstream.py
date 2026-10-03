#!/usr/bin/env python3
"""
下游 4 任务 2×2 柱状图 panel（Earth-Mamba vs 各任务第二名）。
PDF 尺寸与 fig_erf_rotation.pdf 一致。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Patch

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

from common.plotting import AAAI_RC, SIZE_WIDE

DATA_DIR = os.path.join(PKG_ROOT, "data", "fig_downstream")
CSV_PATH = os.path.join(DATA_DIR, "downstream_tasks.csv")
OUT_DIR = os.path.join(PKG_ROOT, "output", "fig_downstream")

COL_FRAC = 0.32
AAAI_COL_W = 3.33
PANEL_W = AAAI_COL_W * COL_FRAC
SCALE = PANEL_W / SIZE_WIDE[0]

ROW_GAP_LINES = 2.0
SUBTITLE_GAP_PT = 0.25  # 子标题距 x 轴刻度标注
EM_FLUO = "#FF1F3D"

RVSA_GREEN = "#4d7a52"

RUNNER_COLORS = {
    "rvsa": RVSA_GREEN,
    "satlas": "#6b8e6b",
    "satlas_aerial": "#6b8e6b",
}

LEGEND_FS_SCALE = 0.72
LEGEND_OFFSET_CM = 0.05
LEGEND_HANDLE_TEXT_PAD = 1.0
LEGEND_LINE_SPACING = 1.5


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


def _pt_to_in(pt: float) -> float:
    return pt / 72.0


def _load_tasks(path: str) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    rows.sort(key=lambda r: (int(r["grid_row"]), int(r["grid_col"])))
    return rows


def _layout_panel(scale: float) -> dict:
    """固定 fig 尺寸；2×2 子图，行间留 2 倍行距。"""
    fig_w, fig_h = _stack_figure_size(scale)
    fs = AAAI_RC["font.size"] * scale
    tick_fs = AAAI_RC["xtick.labelsize"] * scale
    line_in = _pt_to_in(fs)
    tick_in = _pt_to_in(tick_fs)
    subtitle_in = tick_in * 1.05
    row_gap_in = line_in * ROW_GAP_LINES

    margin_top_in = 0.03
    margin_bottom_in = 0.02

    ax_h_in = (
        fig_h - margin_top_in - margin_bottom_in
        - 2.0 * subtitle_in - row_gap_in
    ) / 2.0
    ax_h = ax_h_in / fig_h

    margin_x = 0.06
    col_gap = 0.06
    ax_w = (1.0 - 2.0 * margin_x - col_gap) / 2.0

    y_row1_bottom = 1.0 - margin_top_in / fig_h - ax_h

    axes_pos: dict[tuple[int, int], list[float]] = {}
    for col in range(2):
        x = margin_x + col * (ax_w + col_gap)
        axes_pos[(0, col)] = [x, y_row1_bottom, ax_w, ax_h]

    return dict(
        fig_w=fig_w, fig_h=fig_h, axes_pos=axes_pos,
        fs=fs, tick_fs=tick_fs,
        line_in=line_in, row_gap_in=row_gap_in, row_gap_f=row_gap_in / fig_h,
        ax_h=ax_h, ax_w=ax_w, margin_x=margin_x, col_gap=col_gap,
        margin_top_in=margin_top_in,
    )


def _style_axes(ax, scale: float) -> None:
    spine_lw = AAAI_RC.get("axes.linewidth", 0.8) * scale
    ax.tick_params(
        axis="both", which="major",
        width=0.55 * scale, length=1.2 * scale,
        pad=0.4 * scale,
        labelsize=AAAI_RC["xtick.labelsize"] * scale,
        direction="out",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)


def _runner_color(model_key: str) -> str:
    return RUNNER_COLORS.get(model_key, "#7a6888")


def _draw_panel(ax, task: dict, scale: float) -> None:
    em_val = float(task["earth_mamba"])
    ru_val = float(task["runner_up"])
    metric = task["metric_label"]

    vals = [em_val, ru_val]
    colors = [EM_FLUO, _runner_color(task["runner_up_model"])]

    x = np.arange(2)
    width = 0.52
    bars = ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=0.4 * scale, zorder=3)

    ymin = min(vals) - max(2.5, (max(vals) - min(vals)) * 0.35)
    ymax = max(vals) + max(1.0, (max(vals) - min(vals)) * 0.15)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=0)
    ax.set_ylabel(
        metric,
        fontsize=AAAI_RC["axes.labelsize"] * scale,
        labelpad=0.3 * scale,
    )
    ax.yaxis.set_label_coords(-0.08, 0.5)

    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.35 * scale, alpha=0.45, zorder=0)
    _style_axes(ax, scale)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + (ymax - ymin) * 0.02,
            f"{v:.1f}",
            ha="center", va="bottom",
            fontsize=AAAI_RC["xtick.labelsize"] * scale * 0.92,
            color="#333333",
        )



def _place_subtitle(fig, ax, text: str, tick_fs: float, gap_pt: float = SUBTITLE_GAP_PT) -> float:
    """子标题紧贴坐标轴底边下方（无横轴模型名）。"""
    pos = ax.get_position()
    gap_f = (gap_pt / 72.0) / fig.get_figheight()
    y_top = pos.y0 - gap_f
    cx = pos.x0 + pos.width / 2.0
    t = fig.text(
        cx, y_top, text,
        transform=fig.transFigure, ha="center", va="top",
        fontsize=tick_fs * 0.95, fontweight="bold", color="#333333",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sub_bb = t.get_window_extent(renderer)
    return sub_bb.y0 / fig.bbox.height


def _legend_upper_right(ax, fig, scale: float) -> None:
    """图1 右上角：色块为显眼长方形。"""
    off_in = LEGEND_OFFSET_CM / 2.54
    trans = mtransforms.offset_copy(
        ax.transAxes, fig=fig, x=-off_in, y=-off_in, units="inches",
    )
    leg_fs = AAAI_RC["legend.fontsize"] * scale * LEGEND_FS_SCALE
    edge_lw = 0.45 * scale
    handles = [
        Patch(facecolor=EM_FLUO, edgecolor="#dddddd", linewidth=edge_lw),
        Patch(facecolor=RVSA_GREEN, edgecolor="#dddddd", linewidth=edge_lw),
    ]
    labels = [" Earth-Mamba", " RVSA"]
    leg = ax.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        bbox_transform=trans,
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fontsize=leg_fs,
        handlelength=10.2 * scale,
        handleheight=3.36 * scale,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LINE_SPACING,
        borderpad=0.06 * scale,
        borderaxespad=0,
    )
    leg.get_frame().set_linewidth(0.5 * scale)


def _row2_axes_bottom(row1_sub_bottom: float, lay: dict) -> float:
    """第一行子标题底边与第二行图顶之间：恰好 2 倍行距。"""
    return row1_sub_bottom - lay["row_gap_f"] - lay["ax_h"]


def plot_downstream_panel(csv_path: str, out_dir: str) -> None:
    mpl.rcParams.update(_scaled_rc(SCALE))
    tasks = _load_tasks(csv_path)
    lay = _layout_panel(SCALE)
    row1_tasks = [t for t in tasks if int(t["grid_row"]) == 0]
    row2_tasks = [t for t in tasks if int(t["grid_row"]) == 1]

    fig = plt.figure(figsize=(lay["fig_w"], lay["fig_h"]), dpi=150)
    axes: dict[tuple[int, int], mpl.axes.Axes] = {}

    for task in row1_tasks:
        rc, cc = int(task["grid_row"]), int(task["grid_col"])
        ax = fig.add_axes(lay["axes_pos"][(rc, cc)])
        _draw_panel(ax, task, SCALE)
        axes[(rc, cc)] = ax

    fig.canvas.draw()

    row1_subtitle_bottoms: dict[tuple[int, int], float] = {}
    for task in row1_tasks:
        rc, cc = int(task["grid_row"]), int(task["grid_col"])
        y_bot = _place_subtitle(fig, axes[(rc, cc)], task["subtitle"], lay["tick_fs"])
        row1_subtitle_bottoms[(rc, cc)] = y_bot

    row1_sub_bottom = min(row1_subtitle_bottoms.values())
    y_row2_bottom = _row2_axes_bottom(row1_sub_bottom, lay)

    for task in row2_tasks:
        rc, cc = int(task["grid_row"]), int(task["grid_col"])
        x = lay["margin_x"] + cc * (lay["ax_w"] + lay["col_gap"])
        ax = fig.add_axes([x, y_row2_bottom, lay["ax_w"], lay["ax_h"]])
        _draw_panel(ax, task, SCALE)
        axes[(rc, cc)] = ax

    fig.canvas.draw()

    for task in row2_tasks:
        rc, cc = int(task["grid_row"]), int(task["grid_col"])
        _place_subtitle(fig, axes[(rc, cc)], task["subtitle"], lay["tick_fs"])

    _legend_upper_right(axes[(0, 0)], fig, SCALE)

    os.makedirs(out_dir, exist_ok=True)
    stem = "fig_downstream_panel"
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
        print(f"  -> {path}")

    plt.close(fig)
    print(f"  figsize={lay['fig_w']:.3f}×{lay['fig_h']:.3f} in")
    for t in tasks:
        print(
            f"  [{t['grid_row']},{t['grid_col']}] {t['dataset']}: "
            f"EM {t['earth_mamba']} vs {t['runner_up_display']} {t['runner_up']} "
            f"(Δ{t['gap_pp']} pp)"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    print(f"data: {args.csv}")
    plot_downstream_panel(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
