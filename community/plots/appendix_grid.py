# %%
# Appendix grid: reduced model set (Llama + Qwen) on the additional datasets.
# Reuses plot_grid / titles_dict / baselines from main_grid.py; per-dataset
# trial scores come from results_{dataset}.json (dump_results.py, appendix part).
# Add new datasets to DATASETS as their sweeps finish.
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from main_grid import baselines, plot_grid, titles_dict

_BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"

MODELS = ["Llama-3.1-8B", "Qwen3.5-9B"]

# (subdir, dataset_key, display_name)
DATASETS = [
    ("wmdp_low_mi", "cyber", "WMDP-Cyber"),
    # ("rwku", "rwku", "RWKU"),
    # ("sycophancy", "sycophancy", "Sycophancy"),
]

trial_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
for _subdir, _dataset, _ in DATASETS:
    with open(_BENCHMARKS_DIR / _subdir / f"results_{_dataset}.json") as _f:
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
    """Like main_grid.get_stats, but tolerates methods absent from the reduced grid."""
    method_stats = {}
    for method in titles_dict:
        by_model = trial_scores[dataset].get(method, {})
        if model not in by_model:
            continue  # not run on this reduced grid; plot_grid renders an empty bar
        top = sorted(by_model[model])[:top_n]
        method_stats[method] = (np.mean(top), stats.sem(top), np.std(top))
    return method_stats, baselines[dataset][model]


# %%

if __name__ == "__main__":
    height = 1.0 + 0.9 * len(DATASETS)
    fig = plot_grid(
        rows=[
            [get_stats(model, dataset) for model in MODELS]
            for _, dataset, _ in DATASETS
        ],
        col_titles=MODELS,
        row_titles=[display for _, _, display in DATASETS],
        figsize=(3.5, height),
        save_path="appendix_grid.pdf",
    )

    plt.show()

# %%
