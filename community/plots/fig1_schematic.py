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
from matplotlib.patches import FancyArrowPatch, Rectangle

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
            fontsize=9,
            color="black",
            zorder=4,
        )
        ax.text(
            1.5,
            TOP_FRAC * 1.25,
            "Forget w/o retain\n(low disruption)",
            ha="center",
            va="center",
            fontsize=9,
            # color="#444",
        )
    else:
        ax.text(
            1.5,
            TOP_FRAC * 0.75,
            "Forget-retain overlap\n(high disruption)",
            ha="center",
            va="center",
            fontsize=9,
            # color="#444",
        )

    # Axis cosmetics
    ax.set_xlim(0, 2.25)
    ax.set_ylim(1, 0)  # invert: top PCs at top
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("variance", fontsize=10)
    ax.set_ylabel(r"bottom PCs  $\longleftrightarrow$  top PCs", fontsize=10)
    ax.set_title(title, fontsize=10, weight="bold")


# --- figure ---------------------------------------------------------------
panel_w, panel_h = 2.4, 3.4  # ~1 : 1.4 ratio
fig, axes = plt.subplots(
    1, 2, figsize=(2 * panel_w + 1.6, panel_h), gridspec_kw={"wspace": 0.4}
)

draw_panel(axes[0], collapsed=False, title="A   Why naive unlearning fails")
draw_panel(axes[1], collapsed=True, title="B   RepSelect")

# Legend inside panel A (bottom-right)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.45, label="Forget"),
    plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.45, label="Retain"),
]
axes[0].legend(
    handles=handles,
    loc="lower right",
    frameon=False,
    fontsize=9,
)

# Short connecting arrow between panels A and B (figure coords)
ax_a, ax_b = axes
bb_a = ax_a.get_position()
bb_b = ax_b.get_position()
y_mid = (bb_a.y0 + bb_a.y1) / 2
gap_mid = (bb_a.x1 + bb_b.x0) / 2
half = 0.025  # half-length in figure-x units
arrow = FancyArrowPatch(
    (gap_mid - half*1.9, y_mid),
    (gap_mid + half*0.8, y_mid),
    transform=fig.transFigure,
    arrowstyle="-|>",
    mutation_scale=22,
    lw=3.0,
    color="black",
    zorder=10,
)
fig.patches.append(arrow)

fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
