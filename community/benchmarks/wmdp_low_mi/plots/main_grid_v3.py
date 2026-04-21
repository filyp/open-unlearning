# %%
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy import stats

CACHE_DIR = Path(__file__).parent / ".study_cache"
CACHE_DIR.mkdir(exist_ok=True)

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"


# === CELL 1: Load studies (slow, run once) ===

storage = os.environ.get("OPTUNA_STORAGE_URL")
assert storage is not None, "OPTUNA_STORAGE_URL environment variable not set"

# Method names (keys) and display labels (values)
titles_dict = {
    "SimNPO": "SimNPO",
    "RMU": "RMU",
    "GradDiff": "GradDiff",
    "UNDIAL": "UNDIAL",
    "NPO": "NPO",
    "RepSelect_forget": "RepSelect",
    "RepSelect_retain": "RepSelect Retain",
    "RepSelect_no_lora": "RepSelect (no LoRA)",
    # "RepSelect_no_pcs": "RepSelect (no PCs)",
    "RepSelectSimple2": "RepSelect Simple",
}

# Canonical study name -> actual name in Optuna (for the weirdly-named runs).
# Canonical scheme: always v5.3/v7.3, RepSelect always carries explicit _forget.
study_remap = {
    # Llama bio: actual runs use v5 (no .3)
    "v5.3_Llama-3.1-8B_bio_GradDiff":         "v5_Llama-3.1-8B_bio_GradDiff",
    "v5.3_Llama-3.1-8B_bio_NPO":              "v5_Llama-3.1-8B_bio_NPO",
    "v5.3_Llama-3.1-8B_bio_RMU":              "v5_Llama-3.1-8B_bio_RMU",
    "v5.3_Llama-3.1-8B_bio_SimNPO":           "v5_Llama-3.1-8B_bio_SimNPO",
    "v5.3_Llama-3.1-8B_bio_UNDIAL":           "v5_Llama-3.1-8B_bio_UNDIAL",
    # Llama animal_abuse: actual runs use v7 (no .3) for non-RepSelect methods
    "v7.3_Llama-3.1-8B_animal_abuse_GradDiff": "v7_Llama-3.1-8B_animal_abuse_GradDiff",
    "v7.3_Llama-3.1-8B_animal_abuse_NPO":      "v7_Llama-3.1-8B_animal_abuse_NPO",
    "v7.3_Llama-3.1-8B_animal_abuse_RMU":      "v7_Llama-3.1-8B_animal_abuse_RMU",
    "v7.3_Llama-3.1-8B_animal_abuse_SimNPO":   "v7_Llama-3.1-8B_animal_abuse_SimNPO",
    "v7.3_Llama-3.1-8B_animal_abuse_UNDIAL":   "v7_Llama-3.1-8B_animal_abuse_UNDIAL",
    # Bio RepSelect runs are bare (no _forget suffix)
    "v5.3_Llama-3.1-8B_bio_RepSelect_forget": "v5_Llama-3.1-8B_bio_RepSelect",
    "v5.3_gemma-4-E4B_bio_RepSelect_forget":      "v5.3_gemma-4-E4B_bio_RepSelect",
    "v5.3_DeepSeek-V2-Lite_bio_RepSelect_forget": "v5.3_DeepSeek-V2-Lite_bio_RepSelect",
    # animal_abuse RepSelect retain runs are bare (no _retain suffix)
    "v7.3_Llama-3.1-8B_animal_abuse_RepSelect_retain":     "v7_Llama-3.1-8B_animal_abuse_RepSelect",
    "v7.3_gemma-4-E4B_animal_abuse_RepSelect_retain":      "v7.3_gemma-4-E4B_animal_abuse_RepSelect",
    "v7.3_DeepSeek-V2-Lite_animal_abuse_RepSelect_retain": "v7.3_DeepSeek-V2-Lite_animal_abuse_RepSelect",
}


def load_studies(study_pattern: str) -> Dict[str, optuna.Study]:
    """
    Load Optuna studies for each method.

    Args:
        study_pattern: Pattern like "v5_Llama-3.1-8B_{}_bio" where {} is replaced by method name
        method_names: List of method names to load
        storage: Optuna storage URL

    Returns:
        Dict mapping method name to Study object
    """
    studies = {}
    for method in titles_dict.keys():
        canonical = study_pattern.format(method)
        actual = study_remap.get(canonical, canonical)
        cache_file = CACHE_DIR / f"{actual}.pkl"
        if cache_file.exists():
            print(f"  loading cached: {cache_file.name}")
            with open(cache_file, "rb") as f:
                studies[method] = pickle.load(f)
            continue

        try:
            study = optuna.load_study(study_name=actual, storage=storage)
        except KeyError:
            import warnings
            warnings.warn(f"Study not found: {actual} (method={method}); skipping")
            studies[method] = None
            continue
        frozen = SimpleNamespace(trials=list(study.trials))
        with open(cache_file, "wb") as f:
            pickle.dump(frozen, f)
        studies[method] = frozen

    for study in studies.values():
        n_complete = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        print(f"  {method}: {n_complete} completed trials")
        assert n_complete == 30
    return studies


# %%
def get_stats_from_studies(
    study_pattern: str,
    top_n: int = 10,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """
    Get mean and SEM of top N trials for each study, plus the baseline.

    For minimization studies (lower is better), takes the N lowest values.
    The baseline is provided manually (from a dedicated reference run in wandb).
    The most common value across all runs is logged as a sanity check.

    Returns:
        Tuple of:
        - Dict mapping method name to (mean, sem, std) tuple
        - Reference value (passed in)
    """
    from collections import Counter

    studies = load_studies(study_pattern)
    reference = references[study_pattern.format("reference")]

    method_stats = {}
    all_values = []  # Collect ALL values from all studies for sanity check

    print("Method stats (top {} runs):".format(top_n))
    for method, study in studies.items():
        if study is None:
            continue
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        values = [t.value for t in completed_trials]

        # Collect all values for sanity check
        all_values.extend(values)

        # Track worst run for this method
        method_worst = max(values)

        # Sort ascending (best = lowest for minimization)
        values_sorted = sorted(values)
        top_n_values = values_sorted[:top_n]

        mean = np.mean(top_n_values)
        sem = stats.sem(top_n_values) if len(top_n_values) > 1 else 0
        std = np.std(top_n_values)
        method_stats[method] = (mean, sem, std)

        print(
            f"  {method}: mean={mean*100:.2f}%, sem={sem*100:.2f}%, std={std*100:.2f}%, worst={method_worst*100:.2f}%"
        )

    # Sanity check: log most common value across all runs
    counter = Counter(all_values)
    most_common_value, most_common_count = counter.most_common(1)[0]
    print(
        f"  Sanity check - most common value: {most_common_value*100:.2f}% (appears {most_common_count} times)"
    )
    print(f"  Reference (from dedicated run): {reference*100:.2f}%")

    return method_stats, reference


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
    fig, axes = plt.subplots(nrows, 2, figsize=figsize)
    if nrows == 1:
        axes = [axes]

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

            # Only show x-label on bottom row
            if row_idx == nrows - 1:
                ax.set_xlabel("Answer Probability (%) ↓")

            y_min = min(y_positions) - 0.5
            y_max = max(y_positions) + 0.5
            ax.set_ylim(y_min, y_max)

            # Method labels on right side of right column only
            if col_idx == 1:
                ax.set_yticks(y_positions)
                ax.set_yticklabels([titles_dict.get(m, m) for m in common_methods])
                ax.yaxis.tick_right()
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {save_path}")

    return fig


# === CELL 2: Plotting (fast, iterate here) ===

# Reference values from dedicated reference runs in wandb
# (no epochs of unlearning, i.e. just relearning from the base model)
references = {
    "v5.3_Llama-3.1-8B_bio_reference": 0.16739,  # taken from the v5_ version
    "v7.3_Llama-3.1-8B_animal_abuse_reference": 0.20943,  # taken from the v7_ version
    "v5.3_gemma-4-E4B_bio_reference": 0.15398,
    "v7.3_gemma-4-E4B_animal_abuse_reference": 0.19647,
    "v7.3_Llama-3.1-8B-Instruct_animal_abuse_reference": 0.20335,
    "v7.3_DeepSeek-V2-Lite_animal_abuse_reference": 0.21067,
    "v5.3_DeepSeek-V2-Lite_bio_reference": 0.063342,
}

# %%

# Create grid: rows = models, columns = benchmarks
fig = plot_grid(
    rows=[
        [
            get_stats_from_studies("v5.3_Llama-3.1-8B_bio_{}"),
            get_stats_from_studies("v7.3_Llama-3.1-8B_animal_abuse_{}"),
        ],
        [
            get_stats_from_studies("v5.3_gemma-4-E4B_bio_{}"),
            get_stats_from_studies("v7.3_gemma-4-E4B_animal_abuse_{}"),
        ],
        [
            get_stats_from_studies("v5.3_DeepSeek-V2-Lite_bio_{}"),
            get_stats_from_studies("v7.3_DeepSeek-V2-Lite_animal_abuse_{}"),
        ],
    ],
    col_titles=["WMDP - Bio", "BeaverTails - Animal Abuse"],
    row_titles=["Llama 3.1 8B", "Gemma 4 E4B", "DeepSeek-V2-Lite"],
    figsize=(5.5, 5.4),
    save_path="main_grid_v3.pdf",
)

plt.show()

# %%
