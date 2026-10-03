#!/usr/bin/env python3
"""
Fig2 噪声鲁棒性雷达图（panel 版）。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)

from common.colors import MORANDI, MARKERS
from common.panel_config import MODEL_ORDER, EXP2
from common.plotting import AAAI_RC, SIZE_WIDE

DATA_DIR = os.path.join(PKG_ROOT, "data", "fig2_noise_radar")
CSV_PATH = os.path.join(DATA_DIR, "radar_values.csv")
OUT_DIR = os.path.join(PKG_ROOT, "output", "fig2_noise_radar")

COL_FRAC = 0.32
AAAI_COL_W = 3.33
PANEL_W = AAAI_COL_W * COL_FRAC
SCALE = PANEL_W / SIZE_WIDE[0]

LEGEND_FS_SCALE = 0.72
LEGEND_HANDLE_TEXT_PAD = 1.0
LEGEND_LINE_SPACING = 1.5
RADAR_SIZE_SCALE = 0.7
FILL_ALPHA_EM = 0.10
FILL_ALPHA_OTHER = 0.25
MODEL_LEGEND_LEFT_CM = 0.2
MODEL_LEGEND_TOP_CM = 0.2
NOISE_LEGEND_LEFT_CM = 0.2
NOISE_LEGEND_BOTTOM_CM = 0.2
STACK_ROW_GAP_LINES = 2.0
AXIS_LABEL_R_FRAC = 0.028

EM_FLUO = "#FF1F3D"
FIG_TITLE = "Feature Robustness under Physical Noise"

PLOT_ORDER = ["Gaussian", "Poisson", "Impulse", "Fog"]
ABBREV = {"Gaussian": "Ga", "Poisson": "Po", "Impulse": "Im", "Fog": "Fo"}
NOISE_FULL = {
    "Gaussian": "Additive Gaussian Noise",
    "Poisson": "Poisson Noise",
    "Impulse": "Impulse Noise",
    "Fog": "Atmospheric Scattering Noise",
}
MODEL_LEGEND = {
    "ViT": "ViT-B",
    "Swin": "Swin-B",
    "VMamba": "VMamba-B",
    "EarthMamba": "Earth-Mamba",
}
THETAS = np.array([np.pi / 2, 0.0, 3 * np.pi / 2, np.pi])


def _model_color(name: str) -> str:
    return EM_FLUO if name == "EarthMamba" else MORANDI[name]


def _fill_alpha(name: str) -> float:
    return FILL_ALPHA_EM if name == "EarthMamba" else FILL_ALPHA_OTHER


def _fill_rgba(name: str) -> tuple[float, float, float, float]:
    r, g, b = mcolors.to_rgb(_model_color(name))
    return (r, g, b, _fill_alpha(name))


def _cm_in_fig(cm: float, fig_size_in: float) -> float:
    return (cm / 2.54) / fig_size_in


def _stack_figure_size(scale: float) -> tuple[float, float]:
    """与 fig_erf_rotation.pdf 完全一致的 PDF 尺寸。"""
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


def _legend_fs(scale: float) -> float:
    return AAAI_RC["legend.fontsize"] * scale * LEGEND_FS_SCALE


def _title_fs(scale: float) -> float:
    return AAAI_RC.get("axes.titlesize", 10) * scale


def _one_line_in(scale: float) -> float:
    return AAAI_RC["font.size"] * scale / 72.0


def _figure_layout(scale: float) -> dict:
    """PDF 尺寸与 panel (b) 一致；雷达居中；图例固定于画布左上/左下角。"""
    fig_w, fig_h = _stack_figure_size(scale)
    radar_d = PANEL_W * RADAR_SIZE_SCALE
    ax_left_in = (fig_w - radar_d) / 2.0
    ax_bottom_in = (fig_h - radar_d) / 2.0

    return dict(
        fig_w=fig_w, fig_h=fig_h,
        ax_rect=[ax_left_in / fig_w, ax_bottom_in / fig_h, radar_d / fig_w, radar_d / fig_h],
        radar_d=radar_d,
        leg_fs=_legend_fs(scale),
        title_fs=_title_fs(scale),
    )


def _scaled_rc(scale: float) -> dict:
    def s(v: float) -> float:
        return v * scale

    rc = dict(AAAI_RC)
    rc.update({
        "font.size": s(AAAI_RC["font.size"]),
        "axes.labelsize": s(AAAI_RC["axes.labelsize"]),
        "xtick.labelsize": s(AAAI_RC["xtick.labelsize"]),
        "ytick.labelsize": s(AAAI_RC["ytick.labelsize"]),
        "legend.fontsize": s(AAAI_RC["legend.fontsize"]),
        "lines.linewidth": s(AAAI_RC["lines.linewidth"]),
        "axes.linewidth": s(AAAI_RC.get("axes.linewidth", 0.8)),
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return rc


def _load_csv(path: str) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    data: dict[str, dict[str, float]] = {}
    for model in MODEL_ORDER:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        data[model] = {r["axis"]: float(r["cosine_similarity"]) for r in sub}
    if not data:
        raise ValueError(f"未在 {path} 解析到数据")
    return data


def _radar_factory(thetas: np.ndarray):
    class RadarAxes(PolarAxes):
        name = "radar"

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if len(x) and x[0] != x[-1]:
                    line.set_data(np.append(x, x[0]), np.append(y, y[0]))
            return lines

    register_projection(RadarAxes)
    return thetas


def _auto_ylim(final: dict[str, list[float]], ymax: float) -> float:
    vals = [v for arr in final.values() for v in arr]
    margin = EXP2.get("radar_margin", 0.06)
    vmin = min(vals)
    ymin = max(0.30, vmin - margin)
    ymin = min(ymin, ymax - 0.25)
    ymin = max(0.15, min(vals) - 0.05)
    return ymin


def _axis_abbrev_labels(ax, ymax: float, ymin: float, fs: float) -> None:
    span = ymax - ymin
    label_r = ymax + span * AXIS_LABEL_R_FRAC
    ha_map = {0.0: "left", np.pi / 2: "center", np.pi: "right", 3 * np.pi / 2: "center"}
    va_map = {0.0: "center", np.pi / 2: "bottom", np.pi: "center", 3 * np.pi / 2: "top"}
    for key, th in zip(PLOT_ORDER, THETAS):
        ax.text(
            th, label_r, ABBREV[key],
            ha=ha_map[th], va=va_map[th],
            fontsize=fs, fontweight="bold", color="#333333", clip_on=False,
        )


def _model_legend(fig, leg_fs: float, scale: float) -> None:
    """画布左上角（距左、上各 0.2 cm）。"""
    trans = mtransforms.offset_copy(
        fig.transFigure, fig=fig,
        x=MODEL_LEGEND_LEFT_CM / 2.54,
        y=-MODEL_LEGEND_TOP_CM / 2.54,
        units="inches",
    )
    handles, labels = [], []
    for name in MODEL_ORDER:
        c = _model_color(name)
        lw = (1.5 if name == "EarthMamba" else 1.45) * scale
        handles.append(Line2D(
            [0], [0], color=c, linewidth=lw, marker=MARKERS[name],
            markersize=4.0 * scale, markerfacecolor=c,
            markeredgecolor="white", markeredgewidth=0.3 * scale,
        ))
        labels.append(f" {MODEL_LEGEND[name]}")
    leg = fig.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        bbox_transform=trans,
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        fontsize=leg_fs,
        handlelength=1.4 * scale,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LINE_SPACING,
        borderpad=0.06 * scale,
        borderaxespad=0,
    )
    leg.get_frame().set_linewidth(0.5 * scale)


def _noise_legend(fig, leg_fs: float) -> None:
    """画布左下角（距左、下各 0.2 cm）。"""
    trans = mtransforms.offset_copy(
        fig.transFigure, fig=fig,
        x=NOISE_LEGEND_LEFT_CM / 2.54,
        y=NOISE_LEGEND_BOTTOM_CM / 2.54,
        units="inches",
    )
    row_step_in = (leg_fs * LEGEND_LINE_SPACING) / 72.0
    for i, key in enumerate(PLOT_ORDER):
        fig.text(
            0, i * row_step_in,
            f"{ABBREV[key]}: {NOISE_FULL[key]}",
            transform=trans,
            fontsize=leg_fs, va="bottom", ha="left", color="#333333", zorder=10,
        )


def _title_below_im(fig, ax, title_fs: float) -> None:
    renderer = fig.canvas.get_renderer()
    im_art = next((t for t in ax.texts if t.get_text() == "Im"), None)
    if im_art is not None:
        im_bb = im_art.get_window_extent(renderer)
        title_y = (im_bb.y0 - title_fs * 0.15) / fig.bbox.height
    else:
        pos = ax.get_position()
        title_y = pos.y0 - title_fs / 72.0 / fig.get_figheight() * 0.9
    cx = ax.get_position().x0 + ax.get_position().width / 2.0
    fig.text(
        cx, title_y, FIG_TITLE,
        transform=fig.transFigure, ha="center", va="top",
        fontsize=title_fs, fontweight="bold",
    )


def plot_fig2_radar(csv_path: str, out_dir: str) -> None:
    mpl.rcParams.update(_scaled_rc(SCALE))
    lay = _figure_layout(SCALE)
    data = _load_csv(csv_path)
    final = {n: [data[n][k] for k in PLOT_ORDER] for n in MODEL_ORDER if n in data}

    ymax = 1.02
    ymin = _auto_ylim(final, ymax)
    span = ymax - ymin
    leg_fs = lay["leg_fs"]

    thetas = _radar_factory(THETAS)
    fig = plt.figure(figsize=(lay["fig_w"], lay["fig_h"]), dpi=150)
    ax = fig.add_axes(lay["ax_rect"], projection="radar")

    ax.set_ylim(ymin, ymax)
    ax.set_rgrids(
        [ymin + span * t for t in (0.33, 0.66, 1.0)],
        angle=30,
        fontsize=AAAI_RC["xtick.labelsize"] * SCALE * 0.95,
        color="#aaaaaa",
        alpha=0.75,
    )
    ax.set_thetagrids(np.degrees(THETAS), [""] * 4)
    ax.tick_params(axis="x", which="major", pad=0)
    ax.tick_params(axis="y", which="major", width=0.55 * SCALE, length=1.2 * SCALE)
    ax.spines["polar"].set_visible(False)
    ax.set_aspect("equal")
    _axis_abbrev_labels(ax, ymax, ymin, leg_fs)

    # 填充：大面积在后（先画）、小面积在前；各色 rgba 显式指定
    fill_order = sorted(
        [n for n in MODEL_ORDER if n in final],
        key=lambda n: float(np.mean(final[n])),
        reverse=True,
    )
    for zi, name in enumerate(fill_order):
        vals = np.clip(final[name], ymin, ymax)
        rgba = _fill_rgba(name)
        ax.fill(
            thetas, vals,
            facecolor=rgba, edgecolor=rgba,
            linewidth=0, zorder=2 + zi,
        )

    for zi, name in enumerate(MODEL_ORDER):
        if name not in final:
            continue
        c = _model_color(name)
        lw = (1.5 if name == "EarthMamba" else 1.45) * SCALE
        vals = np.clip(final[name], ymin, ymax)
        ax.plot(
            thetas, vals, color=c, linewidth=lw, marker=MARKERS[name],
            markersize=4.0 * SCALE, markerfacecolor=c,
            markeredgecolor="white", markeredgewidth=0.3 * SCALE,
            zorder=10 + zi, clip_on=False,
        )

    _model_legend(fig, leg_fs, SCALE)
    _noise_legend(fig, leg_fs)

    fig.canvas.draw()
    _title_below_im(fig, ax, lay["title_fs"])

    os.makedirs(out_dir, exist_ok=True)
    stem = "fig2_noise_radar"
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
        print(f"  -> {path}")
    plt.close(fig)
    fills = {n: _fill_rgba(n) for n in MODEL_ORDER if n in final}
    print(
        f"  figsize={lay['fig_w']:.3f}×{lay['fig_h']:.3f} in  "
        f"radar_d={lay['radar_d']:.3f} in  "
        f"model_legend=L{MODEL_LEGEND_LEFT_CM}/T{MODEL_LEGEND_TOP_CM}cm  "
        f"noise_legend=L{NOISE_LEGEND_LEFT_CM}/B{NOISE_LEGEND_BOTTOM_CM}cm"
    )
    print(f"  fill: {', '.join(f'{n}={fills[n]}' for n in fill_order)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    print(f"data: {args.csv}")
    plot_fig2_radar(args.csv, args.out_dir)


if __name__ == "__main__":
    main()
