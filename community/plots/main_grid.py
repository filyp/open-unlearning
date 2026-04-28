# %%
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

plot_name = "main_grid.pdf"
height = 3.2
titles_dict = {
    "RepSelectSimple_forget": "RepSelect",
    "RepSelect2_forget": "└ multi-epoch",
    "RepSelectSimple_forget_no_lora": "└ w/o LoRA",
    "NPO": "NPO",
    "RMU": "RMU",
    "UNDIAL": "UNDIAL",
    "SimNPO": "SimNPO",
    "GradDiff": "GradDiff",
}


# # old runs with some bad sweep ranges
# "RepSelectSimple2": "RepSelect",
# "RepSelect_forget": "└ multi-epoch",
# "RepSelectSimple_no_lora": "└ w/o LoRA",

# # actually, abandon this branch; collapse comparisons are much better served by collapse_grid.py
# plot_name = "ablations.pdf"
# height = 2.7
# titles_dict = {
#     "RepSelectSimple2": "Forget",
#     "RepSelectSimple_retain": "Retain",  # ignore retain here
#     "RepSelect_forget": "Cont. Forget",
#     "RepSelect_retain": "Cont. Retain",  # ignore retain here
#     "RepSelectSimple_no_lora": "no LoRA",
#     # "RepSelect_no_lora": "Cont. no LoRA",
#     "RepSelectSimple_no_pcs": "no collapse",
# }

# Per-benchmark trial scores from results.json (produced by
# community/benchmarks/dump_results.py). Shape: {dataset: {method: {model: [scores]}}}.
_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
trial_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
for _dataset, _subdir in [("bio", "wmdp_low_mi"), ("animal_abuse", "beavertails")]:
    with open(_BENCHMARKS_DIR / _subdir / "results.json") as _f:
        _data = json.load(_f)
    trial_scores[_dataset] = {
        method: {model: info["scores"] for model, info in by_model.items()}
        for method, by_model in _data.items()
    }


def get_stats(
    model: str,
    dataset: str,
    top_n: int = 10,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """Mean/SEM/std of the top N trials for each method, plus the baseline.

    Lower is better, so "top N" means the N lowest scores.
    """
    method_stats = {}
    for method in titles_dict:
        scores = trial_scores[dataset][method][model]
        top = sorted(scores)[:top_n]
        method_stats[method] = (np.mean(top), stats.sem(top), np.std(top))
    return method_stats, baselines[dataset][model]


def plot_grid(
    rows: List[list],
    figsize: Tuple[float, float],
    col_titles: List[str] = None,
    row_titles: List[str] = None,
    save_path: str = None,
    gap_before: List[str] = None,
):
    """
    Create a grid of horizontal bar plots. Each row is a model, columns are benchmarks.

    Each entry in `rows` is a list of (method_stats, baseline) tuples, one per column.

    Bars extend left from baseline (lower = better = longer bar).
    """
    if gap_before is None:
        gap_before = []
    method_names = list(titles_dict.keys())

    nrows = len(rows)
    ncols = len(rows[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {m: colors[i % len(colors)] for i, m in enumerate(method_names)}

    for row_idx, row in enumerate(rows):
        # Keep all methods present in at least one column; missing entries render as empty bars
        common_methods = [
            m for m in method_names if any(m in stats for stats, _ in row)
        ]

        # Y positions with optional gaps
        y_positions = []
        current_y = 0
        for m in reversed(common_methods):
            y_positions.append(current_y)
            current_y += 1
            if m in gap_before:
                current_y += 0.5
        y_positions = list(reversed(y_positions))

        for col_idx, (method_stats, baseline) in enumerate(row):
            ax = axes[row_idx][col_idx]
            baseline_pct = baseline * 100
            means = [
                method_stats[m][0] * 100 if m in method_stats else baseline_pct
                for m in common_methods
            ]
            stds = [
                method_stats[m][2] * 100 if m in method_stats else 0.0
                for m in common_methods
            ]
            widths = [m - baseline_pct for m in means]

            ax.barh(
                y_positions,
                widths,
                xerr=stds,
                height=1.0,
                capsize=3,
                color=[method_to_color[m] for m in common_methods],
                left=baseline_pct,
            )

            # Column titles on top row only
            if row_idx == 0 and col_titles:
                ax.set_title(col_titles[col_idx])

            # Row titles on left column only
            if col_idx == 0 and row_titles:
                ax.set_ylabel(row_titles[row_idx])

            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)

            min_val = min(means) - max(stds)
            min_val -= (baseline_pct - min_val) * 0.05
            ax.set_xlim(min_val, baseline_pct)

            y_min = min(y_positions) - 0.5
            y_max = max(y_positions) + 0.5
            ax.set_ylim(y_min, y_max)

            # Method labels on right side of right column only
            if col_idx == ncols - 1:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([titles_dict.get(m, m) for m in common_methods])
                ax.yaxis.tick_right()
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    fig.text(
        0.5, -0.03, "Post-Attack Answer Probability (%) ↓", ha="center", va="bottom"
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {save_path}")

    return fig


# === CELL 2: Plotting (fast, iterate here) ===

# Baselines from dedicated reference runs in wandb
# (no epochs of unlearning, i.e. just relearning from the base model).
# Shape: {dataset: {model: value}}.
_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
baselines: Dict[str, Dict[str, float]] = {}
for _path in [
    _BENCHMARKS_DIR / "wmdp_low_mi" / "baselines.yaml",
    _BENCHMARKS_DIR / "beavertails" / "baselines.yaml",
]:
    with open(_path) as _f:
        baselines.update(yaml.safe_load(_f))

# %%

if __name__ == "__main__":
    # Create grid: rows = models, columns = benchmarks
    fig = plot_grid(
        rows=[
            [
                get_stats("Llama-3.1-8B", "bio"),
                get_stats("gemma-4-E4B", "bio"),
                get_stats("DeepSeek-V2-Lite", "bio"),
                get_stats("Qwen3.5-9B", "bio"),
            ],
            [
                get_stats("Llama-3.1-8B", "animal_abuse"),
                get_stats("gemma-4-E4B", "animal_abuse"),
                get_stats("DeepSeek-V2-Lite", "animal_abuse"),
                get_stats("Qwen3.5-9B", "animal_abuse"),
            ],
        ],
        col_titles=["Llama-3.1-8B", "Gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"],
        row_titles=["WMDP-Bio", "Animal Abuse"],
        figsize=(5.5, height),
        save_path=plot_name,
    )

    plt.show()

# %%
