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
from matplotlib.transforms import Bbox

import json

# this style set MUST be before font set, to not overwrite them
plt.style.use("default")  # reset before applying our rcParams

plt.rcParams["font.size"] = 10

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"  # serif math glyphs
plt.rcParams["pdf.fonttype"] = 42  # embed real TTF, not Type 3 paths


# plt.rcParams["font.family"] = "Times New Roman"

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times"]
# plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["text.latex.preamble"] = r"\usepackage{mathptmx}"

OUT = Path(__file__).parent / "fig1_schematic2.pdf"

BLACK = "#000000"
# RED = "#7a1717"    # dark red
RED = "#a31f1f"  # dark red
GREEN = "#2ca02c"
GREY = "#9aa0a6"

# --- spectra over PC index (0=top PC, 1=bottom PC) -----------------------
_N = 1000
_y_spec = np.linspace(0, 1, _N)
# Forget: peaks at top, with a heavier tail toward bottom PCs.
_forget_spec = 0.85 * np.exp(-(_y_spec / 0.35) ** 1.4) + 0.15 * (1 - _y_spec) ** 0.4
_forget_spec = _forget_spec / _forget_spec.max()
# Retain: at the top extends ~2x further right than forget, then decays fast.
_retain_spec = 2.0 * np.exp(-_y_spec / 0.18) + 0.12 * (1 - _y_spec) ** 0.6
# SHADE_AREA = "#cfe2f7"
SHADE_AREA = "#eeeeee"

TOP_FRAC = 0.48  # boundary between "top PCs" and "bottom PCs" regions
XLIM = (0, 2.25)

# (text, x, y) — y in [0,1] with 0 = top PCs (plot is inverted)
TOP_WORDS = [
    ("the", 0.35, 0.05),
    ("a", 1.65, 0.06),
    ("in", 1.0, 0.08),
    ("virus", 1.6, 0.16),
    # ("RNA",   0.48, 0.16),
    ("viral", 0.7, 0.19),
    ("protein", 1.8, 0.28),
    ("infection", 0.65, 0.30),
    ("epidemic", 1.40, 0.40),
]
BOTTOM_WORDS = [
    ("RV strain SA11", 1.2, 0.63),
    ("plasmid-only\nreverse genetics", 1.2, 0.83),
    # ("Bordetella\npertussis", 1.0, 0.97),
]


def _q(txt):
    """Quote the example tokens, so they read as text rather than labels."""
    return f"“{txt}”"


def draw_panel(ax, *, collapsed: bool, title: str):
    # Retain spectrum as a low-alpha background band.
    if collapsed:
        _mask = _y_spec >= TOP_FRAC
        _yy, _rr = _y_spec[_mask], _retain_spec[_mask]
    else:
        _yy, _rr = _y_spec, _retain_spec
    ax.fill_betweenx(_yy, 0, _rr, color=RED, alpha=0.15, lw=0, zorder=0)
    ax.plot(_rr, _yy, color=RED, lw=1.0, alpha=0.6, zorder=1)

    # Bottom words: shown in both panels.
    for txt, x, y in BOTTOM_WORDS:
        ax.text(x, y, _q(txt), ha="center", va="center", fontsize=10, color=BLACK,
                zorder=3)

    if collapsed:
        # Mask the top region.
        ax.axhspan(0, TOP_FRAC, color=SHADE_AREA, alpha=0.85, lw=0, zorder=2)
        ax.text(
            (XLIM[0] + XLIM[1]) / 2,
            TOP_FRAC / 2,
            "collapsed by\nRepSelect",
            ha="center",
            va="center",
            fontsize=10,
            color="black",
            zorder=4,
        )
    else:
        for txt, x, y in TOP_WORDS:
            ax.text(x, y, _q(txt), ha="center", va="center", fontsize=10, color=RED,
                    zorder=3)

    # Top-PC region marker
    ax.axhline(TOP_FRAC, color=GREY, lw=0.8, ls="--", alpha=0.7)

    # Axis cosmetics
    ax.set_xlim(*XLIM)
    ax.set_ylim(1, 0)  # invert: top PCs at top
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("retain set variance", fontsize=10)
    # ax.set_ylabel(r"bottom PCs  $\longleftrightarrow$  top PCs (on forget)", fontsize=10)
    ax.set_ylabel(r"forget set PCs", fontsize=10)
    # ax.yaxis.set_label_coords(-0.06, 0.40)
    if title is not None:
        ax.set_title(title, fontsize=10, weight="bold", loc="left")


# --- panel C: robustness bars --------------------------------------------
METHODS = ["RMU", "NPO", "RepSelectSimple_forget"]
METHOD_LABELS = {
    "RMU": "RMU",
    "NPO": "NPO",
    "RepSelectSimple_forget": "RepSelect",
}
# Use default matplotlib tab colors, assigned by method (RepSelect first).
_DEFAULT_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
METHOD_COLORS = {
    "RepSelectSimple_forget": _DEFAULT_CYCLE[0],  # blue
    "NPO": _DEFAULT_CYCLE[1],  # orange
    "RMU": _DEFAULT_CYCLE[2],  # green
}

# Fine-tuning attack: pull from results.json used by main_grid (Llama-3.1-8B / WMDP-Bio).
# Match main_grid: take mean of the top-10 lowest scores per method.
_results_path = (
    Path(__file__).parent.parent
    / "benchmarks"
    / "wmdp_low_mi"
    / "results_bio"
    / "Llama-3.1-8B.json"
)
with open(_results_path) as _f:
    _ft_data = json.load(_f)
ft_values = {}
for m in METHODS:
    _scores = sorted(_ft_data[m]["scores"])[:10]
    ft_values[m] = float(np.mean(_scores))

# No-unlearn baselines for Llama-3.1-8B / WMDP-Bio.
FT_BASELINE = 0.16739  # from community/benchmarks/wmdp_low_mi/baselines.yaml
FS_BASELINE = 0.517  # from user's few-shot table (no-unlearn, k=5)

# Few-shot k=5 attack on Llama-3.1-8B / WMDP-Bio (from user's table).
fs_values = {
    "RMU": 0.294,
    "NPO": 0.511,
    "RepSelectSimple_forget": 0.001,
}


def draw_panel_c(ax, values, baseline, title, show_labels=True):
    """main_grid convention: bars right-anchored at no-unlearn baseline,
    extending leftward toward 0. Longer bar = more unlearning."""
    methods = METHODS
    # RepSelect at the bottom (longest bar = punchline at the bottom)
    y_pos = np.arange(len(methods))
    vals = [values[m] for m in methods]
    widths = [v - baseline for v in vals]  # negative -> grows leftward
    colors = [METHOD_COLORS[m] for m in methods]

    ax.barh(y_pos, widths, color=colors, height=0.85, left=baseline)

    # Method labels on the right (anchored at baseline). No per-bar numerics.
    # Both C panels share method order and colours, so one label set is enough.
    for yp, m in zip(y_pos, methods if show_labels else []):
        ax.text(
            baseline * 1.04,
            yp,
            METHOD_LABELS[m],
            ha="left",
            va="center",
            fontsize=10,
            color=METHOD_COLORS[m],
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
        0.01,
        1.01,
        title,
        transform=ax.transAxes,
        fontsize=10,
        color="#333",
        ha="left",
        va="top",
    )


# --- figure ---------------------------------------------------------------
# Two rows: A and B side by side on top, C's two attack panels side by side
# below. Sized for a single column so the 10pt text matches the body text.
fig = plt.figure(figsize=(3.03, 3.6))

# Outer 2x1: top row = A|B schematics, bottom row = C bars.
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=[1.25, 0.62],
    hspace=0.45,
    top=0.91,
    bottom=0.13,
    left=0.09,
    right=0.99,
)

# Top row: A | gap | B. The B-side y-label sticks leftward into the gap.
gs_ab = GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[0, 0], width_ratios=[1.0, 0.30, 1.0], wspace=0.0
)
ax_a = fig.add_subplot(gs_ab[0, 0])
ax_b = fig.add_subplot(gs_ab[0, 2])

# Bottom row: the two attacks side by side. Right gap leaves room for the
# method labels of the right-hand panel, which sit outside its axes box.
gs_c = GridSpecFromSubplotSpec(
    1, 4, subplot_spec=gs[1, 0], width_ratios=[1.0, 0.30, 1.0, 0.72], wspace=0.0
)
ax_c_ft = fig.add_subplot(gs_c[0, 0])
ax_c_fs = fig.add_subplot(gs_c[0, 2])

draw_panel(ax_a, collapsed=False, title=None)
draw_panel(ax_b, collapsed=True, title=None)

draw_panel_c(ax_c_ft, ft_values, FT_BASELINE, "Fine-tuning attack", show_labels=False)
draw_panel_c(ax_c_fs, fs_values, FS_BASELINE, "Few-shot attack (k=5)")
for _ax in (ax_c_ft, ax_c_fs):
    _ax.set_xlabel(r"post-attack score ↓", fontsize=10)

# Aligned super-titles via fig.text. A starts at the figure's left edge so its
# (longer) text doesn't crowd panel B's title.
fig.canvas.draw()
bb_a = ax_a.get_position()
bb_b = ax_b.get_position()
bb_c = ax_c_ft.get_position()


def _panel_title(x, y, letter, text):
    t = fig.text(x, y, letter, fontsize=10, weight="bold", ha="left", va="bottom")
    # Place description right after the bold letter; use a renderer-based
    # offset so spacing is consistent across panels.
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.012, y, text, fontsize=10, ha="left", va="bottom")


_title_y_ab = bb_a.y1 + 0.02
_title_y_c = bb_c.y1 + 0.02
_panel_title(0.00, _title_y_ab, "A", "Why naive unlearning fails")
_panel_title(bb_b.x0 + 0.00, _title_y_ab, "B", "RepSelect")
_panel_title(0.00, _title_y_c, "C", "Robustness")

# Trim outer margins, but keep a little breathing room under the x-labels.
_bb = fig.get_tightbbox().padded(0.01)
fig.savefig(OUT, bbox_inches=Bbox.from_extents(_bb.x0, _bb.y0 - 0.04, _bb.x1, _bb.y1))
print(f"wrote {OUT}")
