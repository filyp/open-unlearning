# %%
"""Schematic for Fig 1, panels A and B.

Panel A: PC index (high-variance top, low-variance bottom) on y-axis;
activation energy on x-axis. Two translucent bands: forget (green) and retain
(red). Top PCs region annotated as shared/disruptive; baseline unlearning
arrow points into top region.

Panel B: same plot, but top-PC region is collapsed (shaded out / hatched);
RepSelect update arrow points into the bottom (forget-specific) region.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import json

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"

# light theme
plt.style.use("default")  # explicitly reset to light defaults

OUT = Path(__file__).parent / "fig1_schematic.pdf"

# --- profiles over PC index ----------------------------------------------
# y goes top -> bottom = high-variance -> low-variance PCs.
# We plot energy (x) as a function of PC index (y).
N = 1000
y = np.linspace(0, 1, N)  # 0 = top PC, 1 = bottom PC

# Forget: peaks at top, with a heavier tail toward bottom PCs.
forget = 0.85 * np.exp(-(y / 0.35) ** 1.4) + 0.15 * (1 - y) ** 0.4
forget = forget / forget.max()  # peak ~1.0

# Retain: at the top extends ~2x further right than forget, then decays fast.
retain = 2.0 * np.exp(-(y / 0.16) ** 1.7)

GREEN = "#2ca02c"
RED = "#d62728"
GREY = "#9aa0a6"

TOP_FRAC = 0.32  # boundary between "top PCs" and "bottom PCs" regions


def draw_panel(ax, *, collapsed: bool, title: str):
    if collapsed:
        mask = y >= TOP_FRAC
        yy = y[mask]
        rr = retain[mask]
        ff = forget[mask]
    else:
        yy, rr, ff = y, retain, forget
    ax.fill_betweenx(yy, 0, rr, color=RED, alpha=0.30, lw=0, label="Retain")
    ax.fill_betweenx(yy, 0, ff, color=GREEN, alpha=0.30, lw=0, label="Forget")
    ax.plot(rr, yy, color=RED, lw=1.2, alpha=0.9)
    ax.plot(ff, yy, color=GREEN, lw=1.2, alpha=0.9)

    # Top-PC region marker
    ax.axhline(TOP_FRAC, color=GREY, lw=0.8, ls="--", alpha=0.7)

    if collapsed:
        ax.text(
            1.15,
            TOP_FRAC / 2,
            "collapsed by\nRepSelect",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )
        ax.text(
            1.35,
            TOP_FRAC * 1.3,
            "Forget w/o retain\n(low disruption)",
            ha="center",
            va="center",
            fontsize=8,
            # color="#444",
        )
    else:
        ax.text(
            1.45,
            TOP_FRAC * 0.72,
            # "Forget-retain\noverlap\n(high disruption)",
            # "Overlapping\nforget & retain\n(high disruption)",
            "Overlapping\nforget & retain",
            ha="center",
            va="center",
            fontsize=8.2,
            # color="#444",
        )

    # Axis cosmetics
    ax.set_xlim(0, 2.25)
    ax.set_ylim(1, 0)  # invert: top PCs at top
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("variance", fontsize=10)
    ax.set_ylabel(r"bottom PCs  $\longleftrightarrow$  top PCs", fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=10, weight="bold", loc="left")


# --- panel C: robustness bars --------------------------------------------
BLUE = "#1f6feb"
METHODS = ["GradDiff", "SimNPO", "RepSelectSimple_forget"]
METHOD_LABELS = {
    "GradDiff": "GradDiff",
    "SimNPO": "SimNPO",
    "RepSelectSimple_forget": "RepSelect",
}

# Fine-tuning attack: pull from results.json used by main_grid (Llama-3.1-8B / WMDP-Bio).
# Match main_grid: take mean of the top-10 lowest scores per method.
_results_path = (
    Path(__file__).parent.parent / "benchmarks" / "wmdp_low_mi" / "results.json"
)
with open(_results_path) as _f:
    _ft_data = json.load(_f)
ft_values = {}
for m in METHODS:
    _scores = sorted(_ft_data[m]["Llama-3.1-8B"]["scores"])[:10]
    ft_values[m] = float(np.mean(_scores))

# No-unlearn baselines for Llama-3.1-8B / WMDP-Bio.
FT_BASELINE = 0.16739  # from community/benchmarks/wmdp_low_mi/baselines.yaml
FS_BASELINE = 0.549  # from user's few-shot table (no-unlearn, k=10)

# Few-shot k=10 attack on Llama-3.1-8B / WMDP-Bio (from user's table).
fs_values = {
    "GradDiff": 0.531,
    "SimNPO": 0.252,
    "RepSelectSimple_forget": 0.001,
}


def draw_panel_c(ax, values, baseline, title):
    """main_grid convention: bars right-anchored at no-unlearn baseline,
    extending leftward toward 0. Longer bar = more unlearning."""
    methods = METHODS
    # RepSelect at the bottom (longest bar = punchline at the bottom)
    y_pos = np.arange(len(methods))
    vals = [values[m] for m in methods]
    widths = [v - baseline for v in vals]  # negative -> grows leftward
    colors = [BLUE if m == "RepSelectSimple_forget" else GREY for m in methods]

    ax.barh(y_pos, widths, color=colors, height=0.7, left=baseline)

    # Method labels on the right (anchored at baseline). No per-bar numerics.
    for yp, m in zip(y_pos, methods):
        ax.text(
            baseline * 1.04,
            yp,
            METHOD_LABELS[m],
            ha="left",
            va="center",
            fontsize=8.5,
            color=BLUE if m == "RepSelectSimple_forget" else "#333",
            weight="bold" if m == "RepSelectSimple_forget" else "normal",
        )

    xmin = min(vals + [0]) - baseline * 0.05
    ax.set_xlim(xmin, baseline)
    # Extend the top of the y-axis so the sub-row title has room above the bars
    ax.set_ylim(-0.6, len(methods) + 0.3)
    ax.set_yticks([])
    # Show bottom + right spines (x-axis + baseline anchor); hide top/left.
    for spine in ("top", "left"):
        ax.spines[spine].set_visible(False)
    for spine in ("right", "bottom"):
        ax.spines[spine].set_color("#333")
        ax.spines[spine].set_linewidth(1.0)
    # Shorten the right spine so it doesn't cut through the sub-row title.
    ax.spines["right"].set_bounds(-0.4, len(methods) - 0.5)
    ax.tick_params(axis="x", which="both", labelsize=7, length=2.5, pad=2)

    # Sub-row label INSIDE the axes top-left (doesn't push the axes box).
    ax.text(
        0.02, 0.98, title,
        transform=ax.transAxes,
        fontsize=8.5, color="#333", ha="left", va="top",
    )


# --- figure ---------------------------------------------------------------
fig = plt.figure(figsize=(5.5, 3.0))
# Reserve right margin for C's method labels (they sit in the gutter outside
# the axes box). Margins set explicitly so saving without bbox_inches='tight'
# preserves the requested 5.5-in width exactly.
# Use 5 gridspec columns so the AB and BC gaps can differ. The B-side y-label
# sticks leftward into the AB gap, making it look smaller, so we make AB ~30%
# larger and BC ~30% smaller.
gs = GridSpec(
    1, 5, figure=fig,
    width_ratios=[1.075, 0.425, 1.075, 0.25, 0.75],
    wspace=0.0,
    top=0.84, bottom=0.16, left=0.04, right=0.86,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 2])

gs_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 4], hspace=0.6)
ax_c_ft = fig.add_subplot(gs_c[0, 0])
ax_c_fs = fig.add_subplot(gs_c[1, 0])

draw_panel(ax_a, collapsed=False, title=None)
draw_panel(ax_b, collapsed=True, title=None)

draw_panel_c(ax_c_ft, ft_values, FT_BASELINE, "Fine-tuning attack")
draw_panel_c(ax_c_fs, fs_values, FS_BASELINE, "Few-shot attack (k=10)")
ax_c_fs.set_xlabel(r"post-attack score  ↓", fontsize=10)
# Shift the entire few-shot sub-row upward so its x-label aligns vertically
# with the "variance" labels under panels A/B (which have no x-tick labels
# beneath them). Also nudge the x-label slightly to the right.
_FS_SHIFT_UP = 0.07
_bb_fs = ax_c_fs.get_position()
ax_c_fs.set_position([_bb_fs.x0, _bb_fs.y0 + _FS_SHIFT_UP, _bb_fs.width, _bb_fs.height])
ax_c_fs.xaxis.set_label_coords(0.72, -0.32)

# Legend inside panel A (bottom-right)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.45, label="Forget"),
    plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.45, label="Retain"),
]
ax_a.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

# Aligned super-titles via fig.text. A starts at the figure's left edge so its
# (longer) text doesn't crowd panel B's title.
fig.canvas.draw()
bb_a = ax_a.get_position()
bb_b = ax_b.get_position()
bb_c = ax_c_ft.get_position()
title_y = bb_a.y1 + 0.04
fig.text(0.0, title_y, "A   Why unlearning fails",
         fontsize=10, weight="bold", ha="left", va="bottom")
fig.text(bb_b.x0, title_y, "B   RepSelect",
         fontsize=10, weight="bold", ha="left", va="bottom")
fig.text(bb_c.x0, title_y, "C   Robustness",
         fontsize=10, weight="bold", ha="left", va="bottom")

fig.savefig(OUT)  # no bbox_inches='tight': preserves exact 5.5 in width
print(f"wrote {OUT}")
