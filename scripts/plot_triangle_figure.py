"""
Two figures, one per model: tiered cross-distribution variance (vertical bars)
+ weight-update energy in top-50 PCs (horizontal bars).

Usage:
  python scripts/plot_triangle_figure.py
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

SAVE_DIR    = Path(__file__).resolve().parent.parent / "saves" / "disco_motivation"
PINTERP_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"
OUT_DIR     = Path(__file__).resolve().parent.parent / "overleaf_repselect" / "neurips2026" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PCA tier data (top 5 tiers; "300+" omitted: per-dir value ~0.001%, invisible at this scale) ──
# Tiers ordered low→high ratio (left→right); "1-10" merges "1-3" + "4-10"
PCA_TIER_ORDER  = ["1-10", "11-30", "31-100", "101-300", "300+"]
PCA_TIER_LABELS = ["Top 1–10", "11–30", "31–100", "101–300", "300–400"]
PCA_TIER_SIZES  = {"1-10": 10, "11-30": 20, "31-100": 70, "101-300": 200, "300+": 100}

def load_pca_tiers(json_file, layer_key):
    with open(SAVE_DIR / json_file) as f:
        d = json.load(f)
    t = d[layer_key]["pca_tiers"]
    f13, r13     = t["1-3"]["forget"],  t["1-3"]["retain"]
    f410, r410   = t["4-10"]["forget"], t["4-10"]["retain"]
    t["1-10"] = {"forget": f13 + f410, "retain": r13 + r410,
                 "ratio": (f13 + f410) / (r13 + r410)}
    forget = [t[k]["forget"] * 100 / PCA_TIER_SIZES[k] for k in PCA_TIER_ORDER]
    retain = [t[k]["retain"] * 100 / PCA_TIER_SIZES[k] for k in PCA_TIER_ORDER]
    ratio  = [t[k]["ratio"]                              for k in PCA_TIER_ORDER]
    return forget, retain, ratio


# ── Panel C helpers ──
_METHOD_ORDER = ["GradDiff", "NPO", "SimNPO", "UNDIAL", "RepSelectSimple", "Attacker"]
_DISPLAY = {"GradDiff": "GradDiff", "NPO": "NPO", "SimNPO": "SimNPO",
            "UNDIAL": "UNDIAL", "RepSelectSimple": "RepSelect", "Attacker": "Attacker\n(fine-tune)"}
_COLORS  = {"GradDiff": "#d62728", "NPO": "#d62728", "SimNPO": "#d62728",
            "UNDIAL": "#d62728", "RepSelectSimple": "#2ca02c", "Attacker": "#ff7f0e"}
_FALLBACK = (
    ["GradDiff", "NPO", "SimNPO", "UNDIAL", "RepSelect", "Attacker\n(fine-tune)"],
    [28, 41, 40, 40, 12, 27],
    ["#d62728", "#d62728", "#d62728", "#d62728", "#2ca02c", "#ff7f0e"],
)

def load_panel_c(json_name):
    path = PINTERP_DIR / json_name
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    buckets = {}
    for r in d["methods"].values():
        buckets.setdefault(r["method"], []).append(r["top50_frac"])
    return {m: round(float(np.mean(v)) * 100) for m, v in buckets.items()}

def panel_c_arrays(raw):
    if raw and any(m in raw for m in _METHOD_ORDER):
        present = [m for m in _METHOD_ORDER if m in raw]
        return ([_DISPLAY[m] for m in present],
                [raw[m] for m in present],
                [_COLORS[m] for m in present])
    return _FALLBACK


# ── Drawing functions ──
def draw_tier_panel(ax, forget, retain):
    x = np.arange(len(PCA_TIER_LABELS))
    w = 0.3
    ax.bar(x - w/2, forget, w, label="Forget", color="#4c72b0", zorder=3)
    bars_r = ax.bar(x + w/2, retain, w, label="Retain", color="#dd8452", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(PCA_TIER_LABELS, fontsize=10)
    ax.set_ylabel("Variance per PC (%)", fontsize=10.5)
    ymax = max(forget) * 1.2
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g%%"))
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("PC tier  (high → low variance)", fontsize=10.5)
    ax.legend(fontsize=9.5, loc="upper right", bbox_to_anchor=(0.82, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # diagonal arrow tracking the declining retain bars
    r0, r_last = bars_r[0], bars_r[-1]
    x0 = r0.get_x() + r0.get_width() / 2
    x1 = r_last.get_x() + r_last.get_width() / 2
    off = ymax * 0.04
    y0, y1 = r0.get_height() + off, r_last.get_height() + off
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="#b05a1a", lw=1.4))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ymax * 0.03,
            "~4× less retain per PC", ha="center", va="bottom",
            fontsize=9.5, color="#b05a1a")


def draw_energy_panel(ax, methods_c, energies_c, colors_c):
    y = np.arange(len(methods_c))
    bars = ax.barh(y, energies_c, color=colors_c, height=0.55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(methods_c, fontsize=10)
    ax.set_xlabel("Weight-update norm in top-50 forget PCs (%)", fontsize=10.5)
    ax.set_xlim(0, max(energies_c) + 12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, energies_c):
        ax.text(val + 0.7, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=9.5)

    attacker_energy = energies_c[-1]
    ax.axvline(attacker_energy, color="#ff7f0e", linestyle=":", linewidth=1.2, zorder=2)
    ax.text(attacker_energy + 0.5, len(methods_c) - 0.3, "Attacker\nlevel",
            color="#ff7f0e", fontsize=7, va="top")

    # legend removed — colours identified in caption


def make_figure(json_file, layer_key, panel_c_json, model_label, out_path):
    forget, retain, ratio = load_pca_tiers(json_file, layer_key)
    raw = load_panel_c(panel_c_json)
    methods_c, energies_c, colors_c = panel_c_arrays(raw)
    source = panel_c_json if raw else "hardcoded fallback"
    print(f"{model_label} panel C: {source}")

    fig, (ax_tier, ax_c) = plt.subplots(1, 2, figsize=(9.5, 3.3),
                                         gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(wspace=0.18)

    draw_tier_panel(ax_tier, forget, retain)
    draw_energy_panel(ax_c, methods_c, energies_c, colors_c)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {out_path}")
    plt.close(fig)


# ── Generate both figures ──
make_figure(
    "disco_motivation_DISCO_MOTIVATION_LLAMA_3.1_8B_BIO.json", "layer_10",
    "baseline_cmp_PANEL_C_LLAMA8B_BIO.json",
    "Llama-3.1-8B",
    OUT_DIR / "triangle_figure_llama8b.pdf",
)

make_figure(
    "disco_motivation_DISCO_MOTIVATION_QWEN3.5_9B_BIO.json", "layer_10",
    "baseline_cmp_PANEL_C_QWEN35_BIO.json",
    "Qwen3.5-9B",
    OUT_DIR / "triangle_figure_qwen35.pdf",
)
