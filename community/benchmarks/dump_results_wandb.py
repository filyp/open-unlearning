"""Dump per-trial results from wandb (supersedes the optuna-based dump_results.py).

For each benchmark and model it fetches the trial runs {study}_{i} and records
  value      — max of the benchmark metric over the whole relearn trajectory
               (identical to the optuna objective computed in src/train.py;
               sanity-checked against the old optuna-derived jsons where they
               exist)
  params     — the swept hyperparameters, recovered as the unlearning-run
               config keys whose values vary across the study's trials
  extras     — one-shot relearn-epoch-0 evals from the run summary
               (few-shot attack + MMLU general caps)

Output layout: one json per model, {subdir}/{results_dir}/{model}.json,
where results_dir is "results" ("results_bio"/"results_cyber" for wmdp).
Each file maps method -> {scores, trials}.
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

import wandb

load_dotenv(Path(__file__).parents[2] / ".env")

UNL_PROJECT = "filyp/selective-unlearning"
REL_PROJECT = "filyp/rel-selective-unlearning"
N_TRIALS = 30

ALL_MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]
REDUCED_MODELS = ["Llama-3.1-8B", "Qwen3.5-9B"]

# Older "legacy" methods: on bio and animal_abuse, Llama uses the bare legacy
# version prefix; everything else uses the new one.
LEGACY_METHODS = ["GradDiff", "NPO", "RMU", "SimNPO", "UNDIAL"]
METHODS = LEGACY_METHODS + [
    "RepSelectSimple_forget",
    "RepSelect2_forget",
    "RepSelectSimple_forget_no_lora",
]

BENCHMARKS = [
    # (subdir, results_dir, dataset, legacy_v, new_v, metric, models)
    ("wmdp_low_mi", "results_bio", "bio", "v5", "v5.3", "train/recall_prob", ALL_MODELS),
    ("wmdp_low_mi", "results_cyber", "cyber", "v5.3", "v5.3", "train/recall_prob", REDUCED_MODELS),
    ("rwku", "results", "rwku", "v1", "v1", "train/recall_cloze_prob", REDUCED_MODELS),
    ("sycophancy", "results", "sycophancy", "v1", "v1", "train/recall_prob", REDUCED_MODELS),
    ("beavertails", "results", "animal_abuse", "v7", "v7.3", "train/holdout_harmful_prob", ALL_MODELS),
]

# unlearning-run config keys never treated as swept hyperparameters,
# even if they vary across trials
_IGNORED_KEY_PARTS = ["run_name", "dir", "_wandb", "task_name", "id"]


def study_name(method, model, legacy_v, new_v, dataset):
    version = legacy_v if (method in LEGACY_METHODS and model == "Llama-3.1-8B") else new_v
    return f"{version}_{model}_{dataset}_{method}"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _fetch_study_runs(project, study):
    runs = api.runs(project, filters={"display_name": {"$regex": f"^{re.escape(study)}_[0-9]+$"}})
    by_trial = {}
    for r in runs:
        i = int(r.name.rsplit("_", 1)[1])
        if i in by_trial:
            print(f"  warning: duplicate run for {r.name}, keeping first")
            continue
        by_trial[i] = r
    return by_trial


def _swept_params(unl_runs):
    """Config keys whose values vary across the study's trials."""
    flats = {i: _flatten(r.config) for i, r in unl_runs.items()}
    keys = set().union(*(f.keys() for f in flats.values()))
    swept = []
    for k in sorted(keys):
        if any(p in k for p in _IGNORED_KEY_PARTS):
            continue
        # only consider runs that have the key: crashed trials with partial
        # configs must not make every key look like it varies
        vals = {json.dumps(f[k], sort_keys=True, default=str) for f in flats.values() if k in f}
        if len(vals) > 1:
            swept.append(k)
    # drop the bare learning_rate duplicate of args.learning_rate
    if "args.learning_rate" in swept and "learning_rate" in swept:
        swept.remove("learning_rate")
    return {i: {k: f[k] for k in swept if k in f} for i, f in flats.items()}


def dump_model(dataset, legacy_v, new_v, metric, model):
    results = {}
    for method in METHODS:
        study = study_name(method, model, legacy_v, new_v, dataset)
        print(f"  fetching {study}")
        rel_runs = _fetch_study_runs(REL_PROJECT, study)
        unl_runs = _fetch_study_runs(UNL_PROJECT, study)
        params_by_trial = _swept_params(unl_runs)

        trials = []
        for i in sorted(rel_runs):
            r = rel_runs[i]
            hist = r.history(keys=[metric])
            if metric not in hist.columns:
                print(f"  warning: {study}_{i}: no {metric} logged, skipping (crashed trial?)")
                continue
            head = hist[metric].dropna()  # full trajectory, like the optuna objective
            if len(head) == 0:
                print(f"  warning: {study}_{i}: no {metric} logged, skipping (crashed trial?)")
                continue
            extras = {
                k.removeprefix("train/"): v
                for k, v in r.summary.items()
                if ("fewshot" in k or "mmlu" in k) and "stderr" not in k
            }
            trials.append(
                {
                    "trial": i,
                    "value": float(head.max()),
                    "params": params_by_trial.get(i),
                    **extras,
                }
            )
        if len(trials) != N_TRIALS:
            print(f"  warning: {study}: {len(trials)} trials (expected {N_TRIALS})")

        best = min(trials, key=lambda t: t["value"]) if trials else None
        results[method] = {
            "optimal_hyperparameters": best["params"] if best else None,
            "scores": [t["value"] for t in trials],
            "trials": trials,
        }
    return results


def sanity_check(out_dir: Path, dataset: str, model: str, results: dict):
    """Compare against the old optuna-derived jsons, where they exist."""
    for name in [f"results_{dataset}_full.json", "results_full.json"]:
        ref_path = out_dir / name
        if ref_path.exists():
            break
    else:
        print(f"  no optuna reference file for {dataset}, skipping sanity check")
        return
    ref = json.loads(ref_path.read_text())
    for method, info in results.items():
        if method not in ref or model not in ref[method]:
            continue
        ours = sorted(info["scores"])
        theirs = sorted(ref[method][model]["scores"])
        ok = len(ours) == len(theirs) and all(
            abs(a - b) < 1e-9 for a, b in zip(ours, theirs)
        )
        print(f"  sanity {method}/{model}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print(f"    wandb:  {ours[:3]}...\n    optuna: {theirs[:3]}...")


if __name__ == "__main__":
    api = wandb.Api(timeout=3600)
    base = Path(__file__).parent
    for subdir, results_dir, dataset, legacy_v, new_v, metric, models in BENCHMARKS:
        out_dir = base / subdir / results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for model in models:
            print(f"\n=== {subdir} ({dataset}) / {model} ===")
            results = dump_model(dataset, legacy_v, new_v, metric, model)
            (out_dir / f"{model}.json").write_text(json.dumps(results, indent=2))
            print(f"  wrote {out_dir / f'{model}.json'}")
            sanity_check(base / subdir, dataset, model, results)
