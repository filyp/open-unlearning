# Fetch the baseline reference runs (launched by run_baselines.sh) from wandb
# and write their values into baselines.yaml.
# The value is the max of the first REL_STEPS points of train/recall_prob
# (probability of held-out sycophantic responses), plus the normal-response probe.
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

load_dotenv(Path(__file__).parents[2].parent / ".env")

REL_PROJECT = "anonymous/rel-selective-unlearning"
REL_STEPS = 11  # all relearn eval points (epochs 0-10), like the optuna objective
# "sycophancy" (sycophantic-response prob) is the robustness metric used by the
# plots; "sycophancy_normal" is the normal-response probe, populated for reference
METRICS = {
    "sycophancy": "train/recall_prob",
    "sycophancy_normal": "train/normal_probe_prob",
}

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]

BASELINES_PATH = Path(__file__).parent / "baselines.yaml"

if BASELINES_PATH.exists():
    with open(BASELINES_PATH) as f:
        baselines = yaml.safe_load(f) or {}
else:
    baselines = {}

api = wandb.Api(timeout=3600)
for tag, metric in METRICS.items():
    baselines.setdefault(tag, {})
    for model in MODELS:
        task = f"baseline_sycophancy_{model}"
        runs = list(api.runs(REL_PROJECT, filters={"display_name": task}))
        old = baselines[tag].get(model)
        if len(runs) == 0:
            print(f"{task}: no run found, keeping {old}")
            continue
        if len(runs) > 1:
            print(f"{task}: warning: {len(runs)} runs, taking first")
        hist = runs[0].history(keys=[metric])
        head = hist.head(REL_STEPS)[metric].dropna()
        if len(head) == 0:
            print(f"{task}: no {metric} logged yet, keeping {old}")
            continue
        new = float(head.max())
        print(f"{task} [{tag}]: {old} -> {new}")
        baselines[tag][model] = new
        # pre-attack (epoch-0) value, used by the PRE_ATTACK collapse plot
        baselines.setdefault(f"{tag}_initial", {})[model] = float(head.iloc[0])
        # few-shot attack value (logged once at relearn epoch 0)
        val = runs[0].summary.get("train/fewshot5_prob")
        if val is not None:
            baselines.setdefault("sycophancy_fewshot5_prob", {})[model] = float(val)

header = """\
# Values are the maximum answer probability during a relearning attack
# on the base model (no unlearning).
# To rederive: launch the reference runs with run_baselines.sh (same directory),
# then run populate_baselines.py to fetch them from wandb and rewrite this file.
"""
with open(BASELINES_PATH, "w") as f:
    f.write(header)
    yaml.dump(baselines, f, sort_keys=False)
print(f"Wrote {BASELINES_PATH}")
