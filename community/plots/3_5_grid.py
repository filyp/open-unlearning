# %%
# 3x5 grid: rows = Llama + Qwen + DeepSeek, columns = all 5 datasets.
# Cells without dumped results yet (e.g. DeepSeek on cyber/rwku/sycophancy) render blank.
# Trial scores come from the per-model jsons written by dump_results_wandb.py.
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from main_grid import baselines, load_trial_scores, plot_grid, titles_dict

MODELS = ["Llama-3.1-8B", "Qwen3.5-9B", "DeepSeek-V2-Lite"]

# (subdir, results_dir, dataset_key, display_name)
DATASETS = [
    ("wmdp_low_mi", "results_bio", "bio", "WMDP-Bio"),
    ("wmdp_low_mi", "results_cyber", "cyber", "WMDP-Cyber"),
    ("rwku", "results", "rwku", "RWKU"),
    ("sycophancy", "results", "sycophancy", "Sycophancy"),
    ("beavertails", "results", "animal_abuse", "Animal Abuse"),
]

trial_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
    dataset: load_trial_scores(subdir, results_dir)
    for subdir, results_dir, dataset, _ in DATASETS
}


def get_stats(
    model: str,
    dataset: str,
    top_n: int = 10,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """Like main_grid.get_stats, but tolerates methods absent from the reduced grid."""
    method_stats = {}
    for method in titles_dict:
        by_model = trial_scores[dataset].get(method, {})
        if model not in by_model:
            continue  # not run on this reduced grid; plot_grid renders an empty bar
        top = sorted(by_model[model])[:top_n]
        method_stats[method] = (np.mean(top), stats.sem(top), np.std(top))
    if not method_stats:
        return {}, None  # nothing dumped yet; plot_grid renders a blank panel
    return method_stats, baselines[dataset][model]


# %%

if __name__ == "__main__":
    height = 1.0 + 1.1 * len(MODELS)
    fig = plot_grid(
        rows=[
            [get_stats(model, dataset) for _, _, dataset, _ in DATASETS]
            for model in MODELS
        ],
        col_titles=[display for _, _, _, display in DATASETS],
        row_titles=MODELS,
        figsize=(5.5, height),
        save_path="3_5_grid.pdf",
    )

    plt.show()

# %%
