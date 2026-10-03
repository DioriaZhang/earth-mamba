#!/usr/bin/env python3
"""
ERF 衰减 + 旋转鲁棒性趋势（两行 panel）。
上行纵轴对数刻度；标题在 xlabel 下方，再单倍行距接下行。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
import numpy as np

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

from common.panel_config import EXP1_ERF, MODEL_ORDER
from common.colors import MORANDI, FILL_ALPHA, DISPLAY_NAMES, MARKERS, LINESTYLES
from common.plotting import AAAI_RC, SIZE_WIDE

DATA_DIR = os.path.join(PKG_ROOT, "data", "fig_erf_rotation")
OUT_DIR = os.path.join(PKG_ROOT, "output", "fig_erf_rotation")

ERF_JSON = os.path.join(DATA_DIR, "erf_decay_curves.json")
RRI_CSV = os.path.join(DATA_DIR, "rotation_rri_by_angle.csv")

COL_FRAC = 0.32
AAAI_COL_W = 3.33
PANEL_W = AAAI_COL_W * COL_FRAC
FIG_W = PANEL_W
FIG_H1 = SIZE_WIDE[1] * (PANEL_W / SIZE_WIDE[0])
FIG_H2 = SIZE_WIDE[1] * (PANEL_W / SIZE_WIDE[0]) * 0.88
SCALE = PANEL_W / SIZE_WIDE[0]
LEGEND_FS_SCALE = 0.72
LEGEND_OFFSET_CM = 0.05
LEGEND_HANDLE_TEXT_PAD = 1.0

# 上行纵轴：对数刻度，上界 10^0
FIG1_YHI = 1.0  # 10^0
ROW_GAP_LINES = 2.0  # 第一行标题与第二行图之间：两倍行距

EM_FLUO = "#FF1F3D"

TITLE_TOP = "Effective Receptive Field Decay"
TITLE_BOTTOM = "Rotation Robustness Trend"


def _model_color(name: str) -> str:
    return EM_FLUO if name == "EarthMamba" else MORANDI[name]


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
        "legend.fontsize": s(AAAI_RC["legend.fontsize"]),
        "lines.linewidth": s(AAAI_RC["lines.linewidth"]),
        "axes.linewidth": s(AAAI_RC.get("axes.linewidth", 0.8)),
        "grid.linewidth": s(AAAI_RC.get("grid.linewidth", 0.5)),
        "xtick.major.width": s(0.55),
        "ytick.major.width": s(0.55),
        "xtick.major.size": s(1.4),
        "ytick.major.size": s(1.4),
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return rc


def _style_axes(ax, scale: float) -> None:
    spine_lw = AAAI_RC.get("axes.linewidth", 0.8) * scale
    ax.tick_params(
        axis="both", which="major",
        width=0.55 * scale, length=1.4 * scale, pad=0.5 * scale,
        labelsize=AAAI_RC["xtick.labelsize"] * scale,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(spine_lw)


def _load_fig1(json_path: str) -> tuple[np.ndarray, dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    common_x = np.asarray(data["common_x"], dtype=float)
    stats = {}
    for name in MODEL_ORDER:
        if name not in data.get("models", {}):
            continue
        m = data["models"][name]
        stats[name] = dict(
            mean=np.asarray(m["mean"], dtype=float),
            lo=np.asarray(m["lo"], dtype=float),
            hi=np.asarray(m["hi"], dtype=float),
            decay=float(m["decay"]),
        )
    return common_x, stats


def _load_fig3(csv_path: str) -> dict[str, dict]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    curves: dict[str, dict] = {}
    for model in MODEL_ORDER:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        sub.sort(key=lambda r: int(r["angle_deg"]))
        curves[model] = {
            "angles": [int(r["angle_deg"]) for r in sub],
            "mean": [float(r["rri_mean"]) for r in sub],
            "se": [float(r["rri_se"]) for r in sub],
        }
    return curves


def _plot_order():
    return sorted(MODEL_ORDER, key=lambda n: EXP1_ERF.get(n, {}).get("plot_order", 0))


def _fig1_ylim_vmamba(stats: dict) -> tuple[float, float]:
    """纵轴：VMamba 全部曲线值（mean/lo/hi）的最小值 → 10^0。"""
    s = stats["VMamba"]
    vals: list[float] = []
    for key in ("lo", "hi", "mean"):
        vals.extend(np.clip(s[key], 1e-3, 1.0).tolist())
    return float(np.min(vals)), FIG1_YHI


def _linear_ylim_from_zero(vals: list[float], margin: float = 0.04) -> tuple[float, float, list[float]]:
    yhi = min(1.05, max(vals) + margin)
    step = 0.2 if yhi > 0.55 else 0.1
    ticks = list(np.arange(0.0, yhi + step * 0.01, step))
    return 0.0, yhi, ticks


def _angle_std(curves: dict, model: str) -> float:
    return float(np.std(curves[model]["mean"], ddof=0))


def _pt_to_in(pt: float) -> float:
    return pt / 72.0


def _fig_y_from_bbox(fig, bbox, offset_pt: float = 0.0) -> float:
    """显示 bbox 底边在 figure 坐标中的 y，再下移 offset_pt。"""
    y0_px = bbox.y0 - offset_pt / 72.0 * fig.dpi
    return y0_px / fig.bbox.height


def _place_title_below_xlabel(fig, ax, title: str, title_fs: float, label_pad: float) -> None:
    """标题放在 xlabel 下方 0.5×labelpad 处。"""
    renderer = fig.canvas.get_renderer()
    xlab_bb = ax.xaxis.label.get_window_extent(renderer)
    gap_pt = label_pad * 0.5
    title_y = _fig_y_from_bbox(fig, xlab_bb, offset_pt=gap_pt + title_fs * 0.55)
    cx = ax.get_position().x0 + ax.get_position().width / 2.0
    fig.text(cx, title_y, title, transform=fig.transFigure,
             ha="center", va="top", fontsize=title_fs, fontweight="bold")


def _layout_stack(scale: float) -> dict:
    """上行顶对齐；下行在放置标题后按实测定位。"""
    label_pad = 0.5 * scale
    title_fs = AAAI_RC.get("axes.titlesize", 10) * scale
    line_in = _pt_to_in(AAAI_RC["font.size"] * scale)
    row_gap_in = line_in * ROW_GAP_LINES
    margin_top_in = 0.03
    bottom_reserve_in = _pt_to_in(
        AAAI_RC["xtick.labelsize"] * scale * 1.2
        + label_pad
        + AAAI_RC["axes.labelsize"] * scale * 1.05
        + label_pad * 0.5
        + title_fs * 1.15
    )
    row1_meta_in = 0.05 + line_in
    fig_h = margin_top_in + FIG_H1 + row1_meta_in + FIG_H2 + bottom_reserve_in

    left = 0.10
    width = 0.86
    h_ax1 = FIG_H1 / fig_h
    h_ax2 = FIG_H2 / fig_h
    y_ax1 = 1.0 - margin_top_in / fig_h - h_ax1

    return dict(
        fig_h=fig_h, left=left, width=width,
        y_ax1=y_ax1, h_ax1=h_ax1, h_ax2=h_ax2,
        label_pad=label_pad, title_fs=title_fs, line_in=line_in,
        row_gap_in=row_gap_in,
        margin_top_in=margin_top_in, bottom_reserve_in=bottom_reserve_in,
    )


def _legend_lower_left(ax, fig, scale: float) -> None:
    off_in = LEGEND_OFFSET_CM / 2.54
    trans = mtransforms.offset_copy(ax.transAxes, fig=fig, x=off_in, y=off_in, units="inches")
    leg_fs = AAAI_RC["legend.fontsize"] * scale * LEGEND_FS_SCALE
    leg = ax.legend(
        frameon=True, loc="lower left", framealpha=0.92, edgecolor="#cccccc",
        fontsize=leg_fs,
        handlelength=1.4 * scale,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=0,
        borderpad=0.12 * scale,
        borderaxespad=0,
        bbox_to_anchor=(0, 0),
        bbox_transform=trans,
    )
    leg.get_frame().set_linewidth(0.5 * scale)


def plot_stack(json_path: str, csv_path: str, out_dir: str) -> None:
    mpl.rcParams.update(_scaled_rc(SCALE))
    common_x, stats = _load_fig1(json_path)
    curves = _load_fig3(csv_path)
    draw_order = _plot_order()
    lay = _layout_stack(SCALE)

    fig = plt.figure(figsize=(FIG_W, lay["fig_h"]), dpi=150)
    ax1 = fig.add_axes([lay["left"], lay["y_ax1"], lay["width"], lay["h_ax1"]])

    ylab_pad1 = 0.0  # unused; kept for clarity
    label_pad = lay["label_pad"]
    title_fs = lay["title_fs"]

    # ── 上行：ERF（纵轴与 fig1b 相同）────────────────────────────────
    for name in draw_order:
        if name not in stats:
            continue
        s = stats[name]
        c = _model_color(name)
        lw = EXP1_ERF.get(name, {}).get("plot_lw", 1.45) * SCALE
        if name == "EarthMamba":
            lw *= 1.08
        label = f" {DISPLAY_NAMES[name]} ($b$={s['decay']:.2f})"
        lo = np.clip(s["lo"], 1e-3, None)
        hi = np.clip(s["hi"], 1e-3, None)
        mean_clip = np.clip(s["mean"], 1e-3, 1.0)
        ax1.fill_between(
            common_x, lo, hi, color=c, alpha=FILL_ALPHA, linewidth=0,
            zorder=1, edgecolor="none",
        )
        ax1.plot(
            common_x, mean_clip, color=c, linewidth=lw, label=label,
            zorder=2 + EXP1_ERF.get(name, {}).get("plot_order", 0),
        )

    ylo1, yhi1 = _fig1_ylim_vmamba(stats)
    ax1.set_yscale("log")
    ax1.set_xlim(0.02, 1.0)
    ax1.set_ylim(ylo1, yhi1)
    ax1.yaxis.set_minor_locator(mticker.NullLocator())
    ax1.set_xlabel("Normalized Distance", labelpad=label_pad)
    ax1.set_ylabel("Normalized Sensitivity", labelpad=0)
    ax1.yaxis.set_label_coords(-0.012, 0.5)
    ax1.grid(True, which="both", linestyle="--", alpha=0.28, linewidth=0.5 * SCALE)
    _style_axes(ax1, SCALE)
    _legend_lower_left(ax1, fig, SCALE)

    fig.canvas.draw()
    _place_title_below_xlabel(fig, ax1, TITLE_TOP, title_fs, label_pad)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title1 = [t for t in fig.texts if t.get_text() == TITLE_TOP][0]
    t1_bottom = title1.get_window_extent(renderer).y0
    ax2_top_f = (t1_bottom - lay["row_gap_in"] * fig.dpi) / fig.bbox.height
    ax2 = fig.add_axes([
        lay["left"], ax2_top_f - lay["h_ax2"], lay["width"], lay["h_ax2"],
    ])

    # ── 下行：RRI ────────────────────────────────────────────────────
    angles = curves[MODEL_ORDER[0]]["angles"]
    x = np.arange(len(angles))
    lw_base = 1.45 * SCALE
    ms = 4.0 * SCALE
    cap = 2.0 * SCALE
    fig2_vals = []

    for name in draw_order:
        if name not in curves:
            continue
        c = curves[name]
        y = np.asarray(c["mean"], dtype=float)
        se = np.asarray(c["se"], dtype=float)
        fig2_vals.extend(y.tolist())
        col = _model_color(name)
        lw = lw_base * (1.08 if name == "EarthMamba" else 1.0)
        std_a = _angle_std(curves, name)
        ax2.errorbar(
            x, y, yerr=se,
            color=col, linewidth=lw,
            linestyle=LINESTYLES[name],
            marker=MARKERS[name],
            markersize=ms,
            markerfacecolor=col,
            markeredgecolor="white",
            markeredgewidth=0.3 * SCALE,
            capsize=cap, capthick=0.5 * SCALE, elinewidth=0.5 * SCALE,
            label=f" {DISPLAY_NAMES[name]} ($\\sigma$={std_a:.3f})",
            zorder=3,
        )

    ylo2, yhi2, yticks2 = _linear_ylim_from_zero(fig2_vals, margin=0.05)
    ax2.set_xticks(x)
    tick_angles = [a for a in angles if a % 90 == 0]
    ax2.set_xticks([angles.index(a) for a in tick_angles])
    ax2.set_xticklabels([str(a) for a in tick_angles])
    ax2.set_xlabel("Rotation Angle (deg)", labelpad=label_pad)
    ax2.set_ylabel("Rotation Robustness Index", labelpad=label_pad)
    ax2.set_ylim(ylo2, yhi2)
    ax2.set_yticks(yticks2)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.28, linewidth=0.5 * SCALE)
    _style_axes(ax2, SCALE)
    _legend_lower_left(ax2, fig, SCALE)

    fig.canvas.draw()
    _place_title_below_xlabel(fig, ax2, TITLE_BOTTOM, title_fs, label_pad)

    os.makedirs(out_dir, exist_ok=True)
    stem = "fig_erf_rotation"
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)

    stds = {n: _angle_std(curves, n) for n in MODEL_ORDER if n in curves}
    print(f"  fig1 ylim=[{ylo1:.4f}, {yhi1:.1f}] (log, VMamba min → 10^0)")
    print(f"  row gap={lay['row_gap_in']*72:.1f}pt ({ROW_GAP_LINES:.0f}× line) below title1")
    print(f"  fig3 ylim=[{ylo2:.2f}, {yhi2:.2f}] (linear, from 0)")
    print(f"  angle std: {stds}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--erf-json", default=ERF_JSON)
    p.add_argument("--rri-csv", default=RRI_CSV)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    plot_stack(args.erf_json, args.rri_csv, args.out_dir)


if __name__ == "__main__":
    main()
