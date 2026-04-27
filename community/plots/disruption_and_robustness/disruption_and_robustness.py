# %%
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from main_grid import study_remap  # noqa: E402

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR.parent / "trajectories" / "trajectories.pkl"

OUT_DIR = SCRIPT_DIR / "methods"
OUT_DIR_PNG = SCRIPT_DIR / "methods_png"

titles_dict = {
    "RepSelectSimple_forget": "RepSelect",
    "NPO": "NPO",
    "RMU": "RMU",
    "UNDIAL": "UNDIAL",
    "SimNPO": "SimNPO",
    "GradDiff": "GradDiff",
}

DISR_METRIC = "train/wikitext_kl"
DISR_THRESHOLD = 0.01
REL_STEPS = 10
N_TRIALS = 30

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]
BENCHMARKS = [
    ("bio", "v5.3", "train/recall_prob"),
    ("animal_abuse", "v7.3", "train/holdout_harmful_prob"),
]

benchmark_display = {"bio": "WMDP-Bio", "animal_abuse": "Animal Abuse"}


def canonical_to_actual(version, model, benchmark, suffix):
    canonical = f"{version}_{model}_{benchmark}_{suffix}"
    return study_remap.get(canonical, canonical)


with open(CACHE_FILE, "rb") as f:
    trajectories = pickle.load(f)
print(f"Loaded {len(trajectories)} cached runs from {CACHE_FILE}")

# %%

OUT_DIR.mkdir(exist_ok=True)
OUT_DIR_PNG.mkdir(exist_ok=True)

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
method_to_color = {
    m: default_colors[i % len(default_colors)] for i, m in enumerate(titles_dict.keys())
}


def disruption_point(unl_hist):
    """Return KL at the last unlearning step where KL <= threshold; None if no such step."""
    kl = unl_hist[DISR_METRIC].astype(float).values
    valid = np.where(kl <= DISR_THRESHOLD)[0]
    if len(valid) == 0:
        return None
    return float(kl[valid[-1]])


for model in MODELS:
    for benchmark, version, metric in BENCHMARKS:
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        print(f"\n{model} / {benchmark}")

        all_x, all_y = [], []
        for method, display_name in titles_dict.items():
            actual = canonical_to_actual(version, model, benchmark, method)
            color = method_to_color[method]

            for trial_i in range(N_TRIALS):
                run_name = f"{actual}_{trial_i}"
                entry = trajectories.get(run_name)
                if entry is None:
                    continue
                unl_hist, rel_hist = entry
                if unl_hist is None or rel_hist is None:
                    continue

                x = disruption_point(unl_hist)
                if x is None:
                    continue

                rel_head = rel_hist[metric].astype(float).head(REL_STEPS).values * 100
                if len(rel_head) == 0:
                    continue
                y_start = float(rel_head[0])
                y_end = float(np.max(rel_head))

                ax.annotate(
                    "",
                    xy=(x, y_end),
                    xytext=(x, y_start),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7, lw=0.8),
                )
                all_x.extend([x])
                all_y.extend([y_start, y_end])

        if all_x:
            x_margin = (max(all_x) - min(all_x)) * 0.05 + 1e-6
            y_margin = (max(all_y) - min(all_y)) * 0.05 + 1e-6
            ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

        ax.set_xlabel(f"Disruption (Wikitext KL at last step ≤ {DISR_THRESHOLD})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="gray", alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

        ax.text(
            1.03, 0.5, benchmark_display.get(benchmark, benchmark),
            transform=ax.transAxes, rotation=-90, ha="left", va="center",
        )

        handles = [
            plt.Line2D([0], [0], color=method_to_color[m], label=titles_dict[m])
            for m in titles_dict
        ]
        n_methods = len(titles_dict)
        legend_ncol = (n_methods + 1) // 2
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=legend_ncol,
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
        )

        fig.suptitle(model)
        plt.tight_layout(rect=[0.04, 0.04, 1, 0.96])
        fig.supylabel("Post-Attack Answer Probability (%)", fontsize=10, x=0.03)
        save_path = OUT_DIR / f"{model}_{benchmark}.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        save_path_png = OUT_DIR_PNG / f"{model}_{benchmark}.png"
        fig.savefig(save_path_png, bbox_inches="tight", dpi=200)
        print(f"  saved {save_path} and {save_path_png}")
        plt.close(fig)

# %%
