"""Dump per-search GPU time from wandb, for the compute appendix.

For every study in the main 3x5 grid we sum the `_runtime` (seconds) of all
unlearning runs and all relearning runs. Per-run runtimes are used rather than
wall-clock spans between trials, because several searches were paused and
restarted, which would inflate any span-based measure.

Overhead outside wandb's timing (model download, container startup, evals run
before wandb.init) is not counted by `_runtime`. Empirically it comes to roughly
20% on top of the logged time, so we also report an `estimated_*` figure with
OVERHEAD_FACTOR applied.

The grid is expected to be complete: every study should have N_TRIALS unlearning
runs and N_TRIALS relearning runs. Anything short of that is reported as a gap
at the end, so a search that is still running (or silently missing) can't quietly
shrink the totals.

Output: community/benchmarks/compute.json
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

import wandb

from dump_results_wandb import (
    BENCHMARKS,
    METHODS,
    N_TRIALS,
    REL_PROJECT,
    UNL_PROJECT,
    study_name,
)

load_dotenv(Path(__file__).parents[2] / ".env")

OUT = Path(__file__).parent / "compute.json"

# `_runtime` misses setup outside the training loop; measured at ~20%.
OVERHEAD_FACTOR = 1.2

# GPU used per (model, method), from community/benchmarks/run2.sh
def gpu_for(model: str, method: str) -> str:
    if "DeepSeek" in model and "RepSelect" in method:
        return "H200 141GB"
    if "DeepSeek" in model:
        return "B200 180GB"
    if "Qwen3.5-9B" in model:
        return "H200 141GB"
    return "RTX PRO 6000 96GB"


def study_runtime(api, project, study):
    """Total `_runtime` seconds and run count for one study in one project."""
    runs = api.runs(
        project,
        filters={"display_name": {"$regex": f"^{re.escape(study)}_[0-9]+$"}},
    )
    total, n, seen = 0.0, 0, set()
    for r in runs:
        if r.name in seen:
            continue
        seen.add(r.name)
        rt = r.summary.get("_runtime")
        if rt is None:
            continue
        total += float(rt)
        n += 1
    return total, n


if __name__ == "__main__":
    api = wandb.Api(timeout=3600)
    records, gaps = [], []
    for _subdir, _results_dir, dataset, legacy_v, new_v, _metric, models in BENCHMARKS:
        for model in models:
            for method in METHODS:
                study = study_name(method, model, legacy_v, new_v, dataset)
                unl_s, unl_n = study_runtime(api, UNL_PROJECT, study)
                rel_s, rel_n = study_runtime(api, REL_PROJECT, study)
                if unl_n != N_TRIALS or rel_n != N_TRIALS:
                    gaps.append(
                        dict(study=study, n_unlearn_runs=unl_n, n_relearn_runs=rel_n)
                    )
                if unl_n == 0 and rel_n == 0:
                    print(f"  MISSING (no runs at all): {study}")
                    continue
                records.append(
                    dict(
                        dataset=dataset,
                        model=model,
                        method=method,
                        study=study,
                        gpu=gpu_for(model, method),
                        unlearn_hours=unl_s / 3600,
                        relearn_hours=rel_s / 3600,
                        total_hours=(unl_s + rel_s) / 3600,
                        n_unlearn_runs=unl_n,
                        n_relearn_runs=rel_n,
                    )
                )
                print(
                    f"  {study}: {(unl_s + rel_s) / 3600:6.1f} h "
                    f"({unl_n} unl + {rel_n} rel runs)"
                )

    by_gpu = defaultdict(float)
    by_model = defaultdict(float)
    for r in records:
        by_gpu[r["gpu"]] += r["total_hours"]
        by_model[r["model"]] += r["total_hours"]

    total = sum(r["total_hours"] for r in records)
    out = dict(
        searches=records,
        incomplete_studies=gaps,
        totals=dict(
            n_searches=len(records),
            n_incomplete_studies=len(gaps),
            total_wandb_hours=total,
            estimated_total_hours=total * OVERHEAD_FACTOR,
            overhead_factor=OVERHEAD_FACTOR,
            unlearn_hours=sum(r["unlearn_hours"] for r in records),
            relearn_hours=sum(r["relearn_hours"] for r in records),
            by_gpu=dict(by_gpu),
            by_gpu_estimated={k: v * OVERHEAD_FACTOR for k, v in by_gpu.items()},
            by_model=dict(by_model),
        ),
    )
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    print(json.dumps(out["totals"], indent=2))
    if gaps:
        print(f"\nWARNING: {len(gaps)} studies do not have {N_TRIALS} runs on both sides:")
        for g in gaps:
            print(f"  {g['study']}: {g['n_unlearn_runs']} unl, {g['n_relearn_runs']} rel")
    else:
        print(f"\nGrid complete: all {len(records)} studies have {N_TRIALS}+{N_TRIALS} runs.")
