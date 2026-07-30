# Fetch the baseline reference runs from wandb and write their values into
# baselines.yaml. Unlike wmdp/rwku there is no run_baselines.sh here: the
# reference runs were launched by run2.sh as {version}_{model}_animal_abuse_reference
# (v7 for the legacy Llama runs, v7.3 for the rest).
# The value is the max of the first REL_STEPS points of train/holdout_harmful_prob.
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

load_dotenv(Path(__file__).parents[2].parent / ".env")

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 10  # keep in sync with the grid plots
METRIC = "train/holdout_harmful_prob"

# model -> version prefix of its reference run
MODELS = {
    "Llama-3.1-8B": "v7",  # legacy
    "gemma-4-E4B": "v7.3",
    "DeepSeek-V2-Lite": "v7.3",
    "Qwen3.5-9B": "v7.3",
}

BASELINES_PATH = Path(__file__).parent / "baselines.yaml"

with open(BASELINES_PATH) as f:
    baselines = yaml.safe_load(f)

api = wandb.Api(timeout=3600)
baselines.setdefault("animal_abuse", {})
for model, version in MODELS.items():
    task = f"{version}_{model}_animal_abuse_reference"
    runs = list(api.runs(REL_PROJECT, filters={"display_name": task}))
    old = baselines["animal_abuse"].get(model)
    if len(runs) == 0:
        print(f"{task}: no run found, keeping {old}")
        continue
    if len(runs) > 1:
        print(f"{task}: warning: {len(runs)} runs, taking first")
    hist = runs[0].history(keys=[METRIC])
    head = hist.head(REL_STEPS)[METRIC].dropna()
    if len(head) == 0:
        print(f"{task}: no {METRIC} logged yet, keeping {old}")
        continue
    new = float(head.max())
    print(f"{task}: {old} -> {new}")
    baselines["animal_abuse"][model] = new
    # pre-attack (epoch-0) value, used by the PRE_ATTACK collapse plot
    baselines.setdefault("animal_abuse_initial", {})[model] = float(head.iloc[0])

header = """\
# Values are the maximum answer probability during a relearning attack
# on the base model (no unlearning).
# To rederive: run populate_baselines.py (same directory), which fetches the
# {version}_{model}_animal_abuse_reference runs from wandb and rewrites this file.
"""
with open(BASELINES_PATH, "w") as f:
    f.write(header)
    yaml.dump(baselines, f, sort_keys=False)
print(f"Wrote {BASELINES_PATH}")
