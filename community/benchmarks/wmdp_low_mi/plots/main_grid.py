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
        study_pattern: Pattern like "v2_3B_{}_bio" where {} is replaced by method name
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
    top_n: int = 5,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """
    Get mean and SEM of top N trials for each study, plus the baseline.

    For minimization studies (lower is better), takes the N lowest values.
    The baseline is the most frequently occurring worst value across all methods
    (these are runs that hit the disruption budget and were stopped early).

    Returns:
        Tuple of:
        - Dict mapping method name to (mean, sem, std) tuple
        - Baseline (most common worst value, must appear at least 10 times)
    """
    from collections import Counter

    method_stats = {}
    all_values = []  # Collect ALL values from all studies to find the most common

    print("Method stats (top {} runs):".format(top_n))
    for method, study in studies.items():
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        values = [t.value for t in completed_trials]

        # Collect all values for baseline detection
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

        print(f"  {method}: mean={mean*100:.2f}%, sem={sem*100:.2f}%, std={std*100:.2f}%, worst={method_worst*100:.2f}%")

    # Find the most common value across ALL runs (baseline)
    # These are the runs where the first unlearning epoch already broke the model,
    # so the relearning starts from the base model, which we treat as the baseline.
    # Alternatively, (if the assert below fails), we could derive this baseline manually.
    counter = Counter(all_values)
    most_common_value, most_common_count = counter.most_common(1)[0]

    assert most_common_count >= 10, (
        f"Most common value {most_common_value*100:.2f}% only appears {most_common_count} times, "
        f"expected at least 10. Top values: {counter.most_common(5)}"
    )

    print(f"  Baseline (most common value): {most_common_value*100:.2f}% (appears {most_common_count} times)")
    return method_stats, most_common_value


def plot_wmdp_comparison(
    bio_stats: Dict[str, Tuple[float, float, float]],
    cyber_stats: Dict[str, Tuple[float, float, float]],
    bio_baseline: float,
    cyber_baseline: float,
    method_names: List[str],
    titles_dict: Dict[str, str] = None,
    figsize: Tuple[float, float] = (10, 4),
    save_path: str = None,
    gap_before: List[str] = None,  # Methods to add a gap before
):
    """
    Create dual horizontal bar plot comparing WMDP-Bio and WMDP-Cyber accuracy.

    Bars start from the worst (highest) accuracy on the right and extend left.
    Lower accuracy = better = longer bar extending left.

    Args:
        bio_stats: Dict mapping method name to (mean, sem, std) for WMDP-Bio
        cyber_stats: Dict mapping method name to (mean, sem, std) for WMDP-Cyber
        bio_baseline: Baseline (worst run) for WMDP-Bio (rightmost point)
        cyber_baseline: Baseline (worst run) for WMDP-Cyber (rightmost point)
        method_names: List of method names in order (top to bottom)
        titles_dict: Optional dict mapping method names to display labels
        figsize: Figure size
        save_path: Optional path to save the figure
        gap_before: List of method names to add a gap before
    """
    if titles_dict is None:
        titles_dict = {m: m for m in method_names}
    if gap_before is None:
        gap_before = []

    # Filter to methods that have data in both
    common_methods = [m for m in method_names if m in bio_stats and m in cyber_stats]

    if not common_methods:
        print("No common methods found with data in both studies!")
        return

    # Create figure with two subplots (no sharey, we'll manage y-axis manually)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Colors
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    method_to_color = {m: colors[i % len(colors)] for i, m in enumerate(method_names)}

    # Y positions: no gaps between bars, except before specified methods
    # Build positions from bottom to top (reversed order)
    y_positions = []
    current_y = 0
    for m in reversed(common_methods):
        y_positions.append(current_y)
        current_y += 1
        if m in gap_before:
            current_y += 0.5  # Add gap after this method (which is before in display order)
    y_positions = list(reversed(y_positions))  # Reverse back to match method order

    for ax, method_stats, baseline, title, xlabel in [
        (ax1, bio_stats, bio_baseline, "WMDP-Bio", "WMDP-Bio Accuracy (%)"),
        (ax2, cyber_stats, cyber_baseline, "WMDP-Cyber", "WMDP-Cyber Accuracy (%)"),
    ]:
        # Get values for common methods
        means = [method_stats[m][0] * 100 for m in common_methods]  # Convert to percentage
        sems = [method_stats[m][1] * 100 for m in common_methods]
        stds = [method_stats[m][2] * 100 for m in common_methods]
        baseline_pct = baseline * 100

        # Bars extend LEFT from baseline (negative width)
        # width = mean - baseline (negative, so bar goes left)
        widths = [m - baseline_pct for m in means]

        # Plot horizontal bars starting from baseline going left
        ax.barh(
            y_positions,
            widths,
            # xerr=sems,
            xerr=stds,
            height=1.0,  # No gaps between bars
            capsize=3,
            color=[method_to_color[m] for m in common_methods],
            left=baseline_pct,  # Start from the right (baseline)
        )

        # Styling
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel + " ↓")
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # X-axis: baseline on right, lower values on left (normal direction, not inverted)
        # min_val = min(means) - max(sems) - 2
        min_val = min(means) - max(stds) - 2
        ax.set_xlim(min_val, baseline_pct)  # End exactly at baseline, no extra space

    # Adjust y-axis limits to remove extra space
    y_min = min(y_positions) - 0.5
    y_max = max(y_positions) + 0.5
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Y-axis labels on the RIGHT side of ax2
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([titles_dict.get(m, m) for m in common_methods])
    ax2.yaxis.tick_right()

    # Hide y-ticks on left subplot
    ax1.set_yticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to {save_path}")

    return fig


# %%
# === CELL 1: Load studies (slow, run once) ===

storage = get_storage()

# Method names from run.sh
method_names = [
    "GradDiff",
    "NPO",
    "RMU",
    "SimNPO",
    "UNDIAL",
    "CIR",
    "NPOstrict",
    "CIRstrict",
]

bio_studies = load_studies(
    study_pattern="v3_3B_{}_bio",
    method_names=method_names,
    storage=storage,
)

cyber_studies = load_studies(
    study_pattern="v3_3B_{}_cyber",
    method_names=method_names,
    storage=storage,
)

# %%
# === CELL 2: Plotting (fast, iterate here) ===

top_n = 5

# Get stats from studies
print("\nWMDP-Bio:")
bio_stats, bio_baseline = get_stats_from_studies(bio_studies, top_n=top_n)

print("\nWMDP-Cyber:")
cyber_stats, cyber_baseline = get_stats_from_studies(cyber_studies, top_n=top_n)

# Display labels for methods
titles_dict = {
    "SimNPO": "SimNPO",
    "UNDIAL": "UNDIAL",
    "GradDiff": "GradDiff",
    "RMU": "RMU",
    "NPO": "NPO",
    "CIR": "CIR",
    "NPOstrict": "NPO low disr.",
    "CIRstrict": "CIR low disr.",
}

# Create the plot (method order comes from titles_dict keys)
fig = plot_wmdp_comparison(
    bio_stats=bio_stats,
    cyber_stats=cyber_stats,
    bio_baseline=bio_baseline,
    cyber_baseline=cyber_baseline,
    method_names=list(titles_dict.keys()),
    titles_dict=titles_dict,
    figsize=(10, 3),
    save_path="main_grid.pdf",
    gap_before=["NPOstrict"],  # Add gap before CIR (strict)
)

plt.show()

# %%
