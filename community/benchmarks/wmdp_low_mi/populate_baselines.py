# Fetch the baseline reference runs (launched by run_baselines.sh) from wandb
# and write their values into baselines.yaml.
# The value is the max of the first REL_STEPS points of train/recall_prob,
# mirroring how community/plots/collapse/collapse_grid.py summarizes the attack trajectories.
from pathlib import Path

import yaml
from dotenv import load_dotenv

import wandb

load_dotenv(Path(__file__).parents[2].parent / ".env")

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 10  # keep in sync with collapse_grid.py
METRIC = "train/recall_prob"

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]
DOMAINS = ["bio", "cyber"]

BASELINES_PATH = Path(__file__).parent / "baselines.yaml"

with open(BASELINES_PATH) as f:
    baselines = yaml.safe_load(f)

api = wandb.Api(timeout=3600)
for domain in DOMAINS:
    baselines.setdefault(domain, {})
    for model in MODELS:
        task = f"baseline_{domain}_{model}"
        runs = list(api.runs(REL_PROJECT, filters={"display_name": task}))
        old = baselines[domain].get(model)
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
        baselines[domain][model] = new
        # pre-attack (epoch-0) value, used by the PRE_ATTACK collapse plot
        baselines.setdefault(f"{domain}_initial", {})[model] = float(head.iloc[0])

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
