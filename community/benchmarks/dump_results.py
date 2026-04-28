"""Dump optimal hyperparameters and trial scores from Optuna studies.

For each benchmark (wmdp_low_mi, beavertails), writes into the benchmark dir:
  results.json      — for each (method, model): best hyperparameters + all 30 scores
  results_full.json — same, but also includes hyperparameters for every trial
"""

import json
import os
import re
from pathlib import Path

import optuna
from dotenv import load_dotenv

load_dotenv()

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]

# Older "legacy" methods: Llama uses the bare version prefix; everything else uses *.3.
LEGACY_METHODS = ["GradDiff", "NPO", "RMU", "SimNPO", "UNDIAL"]
REPSELECT_METHODS = [
    "RepSelectSimple_forget",
    "RepSelect2_forget",
    "RepSelectSimple_forget_no_lora",
]
METHODS = LEGACY_METHODS + REPSELECT_METHODS

BENCHMARKS = [
    # (subdir, dataset, legacy_version, new_version)
    ("wmdp_low_mi", "bio", "v5", "v5.3"),
    ("beavertails", "animal_abuse", "v7", "v7.3"),
]


def dumps_compact_leaves(obj, indent: int = 2) -> str:
    """json.dumps with primitive-only arrays collapsed onto a single line."""
    s = json.dumps(obj, indent=indent)
    return re.sub(
        r"\[\s+([^\[\]{}]+?)\s+\]",
        lambda m: "[" + ", ".join(p.strip() for p in m.group(1).split(",")) + "]",
        s,
    )


def study_name(method: str, model: str, legacy_v: str, new_v: str, dataset: str) -> str:
    if method in LEGACY_METHODS and model == "Llama-3.1-8B":
        version = legacy_v
    else:
        version = new_v
    return f"{version}_{model}_{dataset}_{method}"


def dump_benchmark(out_dir: Path, dataset: str, legacy_v: str, new_v: str, storage: str):
    results = {}
    results_full = {}

    for method in METHODS:
        results[method] = {}
        results_full[method] = {}
        for model in MODELS:
            name = study_name(method, model, legacy_v, new_v, dataset)
            print(f"  loading {name}")
            study = optuna.load_study(study_name=name, storage=storage)
            completed = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            assert len(completed) == 30, f"{name}: {len(completed)} completed trials"

            best = min(completed, key=lambda t: t.value)
            scores = [t.value for t in completed]

            results[method][model] = {
                "optimal_hyperparameters": best.params,
                "scores": scores,
            }
            results_full[method][model] = {
                "optimal_hyperparameters": best.params,
                "scores": scores,
                "trials": [
                    {"value": t.value, "params": t.params} for t in completed
                ],
            }

    (out_dir / "results.json").write_text(dumps_compact_leaves(results))
    (out_dir / "results_full.json").write_text(dumps_compact_leaves(results_full))
    print(f"  wrote {out_dir / 'results.json'}")
    print(f"  wrote {out_dir / 'results_full.json'}")


def main():
    storage = os.environ["OPTUNA_STORAGE_URL"]
    base = Path(__file__).parent

    for subdir, dataset, legacy_v, new_v in BENCHMARKS:
        print(f"\n=== {subdir} ({dataset}) ===")
        dump_benchmark(base / subdir, dataset, legacy_v, new_v, storage)


if __name__ == "__main__":
    main()
