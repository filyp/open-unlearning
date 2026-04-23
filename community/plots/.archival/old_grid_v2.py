# %%
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy import stats

plt.style.use("default")


def get_storage():
    """Get Optuna storage from environment variable."""
    storage_url = os.environ.get("OPTUNA_STORAGE_URL")
    if storage_url is None:
        raise ValueError("OPTUNA_STORAGE_URL environment variable not set")
    return storage_url


def load_studies(
    study_pattern: str,
    method_names: List[str],
    storage: str,
) -> Dict[str, optuna.Study]:
    """
    Load Optuna studies for each method.

    Args:
        study_pattern: Pattern like "v1_Qwen2.5-3B_bio_{}" where {} is replaced by method name
        method_names: List of method names to load
        storage: Optuna storage URL

    Returns:
        Dict mapping method name to Study object
    """
    studies = {}
    for method in method_names:
        study_name = study_pattern.format(method)
        study = optuna.load_study(study_name=study_name, storage=storage)
        n_complete = sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )
        print(f"  {method}: {n_complete} completed trials")
        assert n_complete == 50
        studies[method] = study
    return studies


def get_stats_from_studies(
    studies: Dict[str, optuna.Study],
    reference_baseline: float,
    top_n: int = 5,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """
    Get mean and SEM of top N trials for each study.

    For minimization studies (lower is better), takes the N lowest values.
    The baseline is provided explicitly from a reference run.

    Args:
        studies: Dict mapping method name to Study object
        reference_baseline: Explicit baseline value from the reference run
        top_n: Number of top trials to average

    Returns:
        Tuple of:
        - Dict mapping method name to (mean, sem, std) tuple
        - Baseline value (passed through)
    """
    method_stats = {}

    print("Method stats (top {} runs):".format(top_n))
    for method, study in studies.items():
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        values = [t.value for t in completed_trials]

        # Sort ascending (best = lowest for minimization)
        values_sorted = sorted(values)
        top_n_values = values_sorted[:top_n]

        mean = np.mean(top_n_values)
        sem = stats.sem(top_n_values) if len(top_n_values) > 1 else 0
        std = np.std(top_n_values)
        method_stats[method] = (mean, sem, std)

        worst = max(values)
        print(f"  {method}: mean={mean*100:.2f}%, sem={sem*100:.2f}%, std={std*100:.2f}%, worst={worst*100:.2f}%")

    print(f"  Reference baseline: {reference_baseline*100:.2f}%")
    return method_stats, reference_baseline


def plot_comparison(
    left_stats: Dict[str, Tuple[float, float, float]],
    right_stats: Dict[str, Tuple[float, float, float]],
    left_baseline: float,
    right_baseline: float,
    method_names: List[str],
    left_title: str = "WMDP-Bio (Qwen2.5-3B)",
    right_title: str = "BeaverTails (gemma-2-2b)",
    left_xlabel: str = "WMDP-Bio Answer Probability (%)",
    right_xlabel: str = "BeaverTails Answer Probability (%)",
    titles_dict: Dict[str, str] = None,
    figsize: Tuple[float, float] = (10, 4),
    save_path: str = None,
    gap_before: List[str] = None,
):
    """
    Create dual horizontal bar plot comparing two benchmarks.

    Bars start from the baseline (reference run) on the right and extend left.
    Lower accuracy = better = longer bar extending left.
    """
    if titles_dict is None:
        titles_dict = {m: m for m in method_names}
    if gap_before is None:
        gap_before = []

    # Filter to methods that have data in both
    common_methods = [m for m in method_names if m in left_stats and m in right_stats]

    if not common_methods:
        print("No common methods found with data in both studies!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {m: colors[i % len(colors)] for i, m in enumerate(method_names)}

    # Y positions with optional gaps
    y_positions = []
    current_y = 0
    for m in reversed(common_methods):
        y_positions.append(current_y)
        current_y += 1
        if m in gap_before:
            current_y += 0.5
    y_positions = list(reversed(y_positions))

    for ax, method_stats, baseline, title, xlabel in [
        (ax1, left_stats, left_baseline, left_title, left_xlabel),
        (ax2, right_stats, right_baseline, right_title, right_xlabel),
    ]:
        means = [method_stats[m][0] * 100 for m in common_methods]
        sems = [method_stats[m][1] * 100 for m in common_methods]
        stds = [method_stats[m][2] * 100 for m in common_methods]
        baseline_pct = baseline * 100

        # Bars extend LEFT from baseline
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

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel + " ↓")
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        min_val = min(means) - max(stds) - 2
        ax.set_xlim(min_val, baseline_pct)

    # Y-axis limits
    y_min = min(y_positions) - 0.5
    y_max = max(y_positions) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Y-axis labels on the RIGHT side of ax2
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([titles_dict.get(m, m) for m in common_methods])
    ax2.yaxis.tick_right()

    ax1.set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {save_path}")

    return fig


# %%
# === CELL 1: Load studies (slow, run once) ===

storage = get_storage()

# Display labels for methods
titles_dict = {
    "SimNPO": "SimNPO",
    "UNDIAL2": "UNDIAL",
    "RMU2": "RMU",
    "GradDiff2": "GradDiff",
    "NPO": "NPO",
    "RepSelect": "RepSelect",
}
method_names = list(titles_dict.keys())

# WMDP-Bio on Qwen2.5-3B (from wmdp_low_mi/run.sh)
print("Loading WMDP-Bio studies:")
wmdp_studies = load_studies(
    study_pattern="v1_Qwen2.5-3B_bio_{}",
    method_names=method_names,
    storage=storage,
)

# BeaverTails animal_abuse on gemma-2-2b (from beavertails/run.sh)
print("\nLoading BeaverTails studies:")
bt_studies = load_studies(
    study_pattern="v5_gemma-2-2b_animal_abuse_{}",
    method_names=method_names,
    storage=storage,
)

# %%
# === CELL 2: Plotting (fast, iterate here) ===

top_n = 5

# Explicit reference baselines (from dedicated reference runs)
wmdp_reference = 0.13873
bt_reference = 0.19462

print("\nWMDP-Bio (Qwen2.5-3B):")
wmdp_stats, wmdp_baseline = get_stats_from_studies(wmdp_studies, reference_baseline=wmdp_reference, top_n=top_n)

print("\nBeaverTails animal_abuse (gemma-2-2b):")
bt_stats, bt_baseline = get_stats_from_studies(bt_studies, reference_baseline=bt_reference, top_n=top_n)

# Create the plot
fig = plot_comparison(
    left_stats=wmdp_stats,
    right_stats=bt_stats,
    left_baseline=wmdp_baseline,
    right_baseline=bt_baseline,
    method_names=method_names,
    left_title="WMDP-Bio (Qwen2.5-3B)",
    right_title="BeaverTails (gemma-2-2b)",
    left_xlabel="WMDP-Bio Answer Probability (%)",
    right_xlabel="BeaverTails Answer Probability (%)",
    titles_dict=titles_dict,
    figsize=(10, 3),
    save_path="main_grid_v2.pdf",
)

plt.show()

# todo, have separate file for beavertails

# %%
