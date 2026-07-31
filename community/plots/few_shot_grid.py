# %%
# Few-shot robustness grid: rows = Llama + Qwen, columns = cyber + sycophancy.
# Like 2_5_grid, but the plotted value is the few-shot attack metric (relearn
# epoch 0), and the top N trials are selected by that few-shot metric.
# Trial data comes from the per-model jsons written by dump_results_wandb.py.
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from main_grid import baselines, plot_grid, titles_dict

_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"

MODELS = ["Llama-3.1-8B", "Qwen3.5-9B"]

# (subdir, results_dir, display, trial metric key, baselines.yaml tag)
DATASETS = [
    ("wmdp_low_mi", "results_cyber", "WMDP-Cyber", "fewshot5_acc_t0", "cyber_fewshot5_acc_t0"),
    ("sycophancy", "results", "Sycophancy", "fewshot5_prob", "sycophancy_fewshot5_prob"),
]

trials: Dict[str, Dict[str, Dict[str, List[dict]]]] = {}
for _subdir, _results_dir, _display, _, _ in DATASETS:
    trials[_display] = {}
    for _model in MODELS:
        _data = json.loads(
            (_BENCHMARKS_DIR / _subdir / _results_dir / f"{_model}.json").read_text()
        )
        trials[_display][_model] = {m: info["trials"] for m, info in _data.items()}


def get_stats(
    model: str,
    display: str,
    metric: str,
    baseline_tag: str,
    top_n: int = 10,
) -> Tuple[Dict[str, Tuple[float, float, float]], float]:
    """Mean/SEM/std of the top N trials per method, ranked BY THE FEW-SHOT metric."""
    method_stats = {}
    for method in titles_dict:
        method_trials = trials[display].get(model, {}).get(method, [])
        scores = [t[metric] for t in method_trials if metric in t]
        if not scores:
            continue  # plot_grid renders an empty bar
        top = sorted(scores)[:top_n]
        method_stats[method] = (np.mean(top), stats.sem(top), np.std(top))
    return method_stats, baselines[baseline_tag][model]


# %%

if __name__ == "__main__":
    height = 1.0 + 0.9 * len(MODELS)
    fig = plot_grid(
        rows=[
            [
                get_stats(model, display, metric, tag)
                for _, _, display, metric, tag in DATASETS
            ]
            for model in MODELS
        ],
        col_titles=[display for _, _, display, _, _ in DATASETS],
        row_titles=MODELS,
        figsize=(3.5, height),
        save_path="few_shot_grid.pdf",
        xlabel="Few-Shot Attack Score (%) ↓",
    )

    plt.show()

# %%
