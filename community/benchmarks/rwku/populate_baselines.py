# Fetch the baseline reference runs (launched by run_baselines.sh) from wandb
# and write their values into baselines.yaml.
# The value is the max of the first REL_STEPS points of train/recall_cloze_prob
# (the RWKU robustness metric), mirroring the wmdp_low_mi convention.
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

load_dotenv(Path(__file__).parents[2].parent / ".env")

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 10  # keep in sync with the grid plots
# "rwku" (cloze) is the robustness metric used by the plots; "rwku_qa" is the
# QA-probe variant, populated alongside for reference
METRICS = {
    "rwku": "train/recall_cloze_prob",
    "rwku_qa": "train/recall_prob",
}

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]

BASELINES_PATH = Path(__file__).parent / "baselines.yaml"

with open(BASELINES_PATH) as f:
    baselines = yaml.safe_load(f)

api = wandb.Api(timeout=3600)
for tag, metric in METRICS.items():
    baselines.setdefault(tag, {})
    for model in MODELS:
        task = f"baseline_rwku_{model}"
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
