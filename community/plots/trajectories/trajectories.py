# %%
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))
from main_grid import study_remap, titles_dict  # noqa: E402

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "trajectories.pkl"

OUT_DIR = SCRIPT_DIR / "methods"
# titles_dict is imported from main_grid.py above

# --- uncomment for ablations instead ---
# select best trial by: "relearn" (max relearning metric) or "unlearn"
# (last valid unlearning metric with KL <= threshold). Lower is better in both.
# SELECT_BY = "unlearn"
SELECT_BY = "relearn"
OUT_DIR = SCRIPT_DIR / ("ablations_optim_unlearn" if SELECT_BY == "unlearn" else "ablations")
titles_dict = {
    "RepSelectSimple2": "Forget",
    "RepSelectSimple_retain": "Retain",
    "RepSelect_forget": "Cont. Forget",
    "RepSelect_retain": "Cont. Retain",
    "RepSelectSimple_no_lora": "no LoRA",
    # "RepSelect_no_lora": "Cont. no LoRA",
    "RepSelectSimple_no_pcs": "no collapse",
}


UNL_PROJECT = "filyp/selective-unlearning"
REL_PROJECT = "filyp/rel-selective-unlearning"
DISR_METRIC = "train/wikitext_kl"
DISR_THRESHOLD = 0.01
REL_STEPS = 10
N_TRIALS = 30
MAX_WORKERS = 12

# main_grid.py's study_remap doesn't cover reference runs; add them (Llama bio/animal_abuse
# reference runs live under v5_/v7_ prefixes, matching the method runs for those setups).
study_remap = {
    **study_remap,
    "v5.3_Llama-3.1-8B_bio_reference": "v5_Llama-3.1-8B_bio_reference",
    "v7.3_Llama-3.1-8B_animal_abuse_reference": "v7_Llama-3.1-8B_animal_abuse_reference",
}

MODELS = ["Llama-3.1-8B", "gemma-4-E4B", "DeepSeek-V2-Lite", "Qwen3.5-9B"]
BENCHMARKS = [
    ("bio", "v5.3", "train/recall_prob"),
    ("animal_abuse", "v7.3", "train/holdout_harmful_prob"),
]


def canonical_to_actual(version, model, benchmark, suffix):
    canonical = f"{version}_{model}_{benchmark}_{suffix}"
    return study_remap.get(canonical, canonical)


# %%
# === CELL 1: fetch trajectories (slow, cached) ===

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} cached runs from {CACHE_FILE}")
else:
    trajectories = {}


def list_runs_by_regex(api, project, pattern):
    return list(api.runs(project, filters={"display_name": {"$regex": pattern}}))


def fetch_history_with_retry(run, keys):
    for attempt in range(10):
        try:
            return run.history(keys=keys)
        except Exception as e:
            print(f"  {run.name}: attempt {attempt} failed: {e}")
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch history for {run.name}")


def gather_runs():
    api = wandb.Api(timeout=3600)
    unl_by_name = {}
    rel_by_name = {}
    metric_by_name = {}
    is_ref = {}

    for model in MODELS:
        for benchmark, version, metric in BENCHMARKS:
            for method in titles_dict.keys():
                actual = canonical_to_actual(version, model, benchmark, method)
                pattern = f"^{re.escape(actual)}_\\d+$"
                print(f"  listing {actual}_*")
                for r in list_runs_by_regex(api, UNL_PROJECT, pattern):
                    unl_by_name[r.name] = r
                    metric_by_name[r.name] = metric
                    is_ref[r.name] = False
                for r in list_runs_by_regex(api, REL_PROJECT, pattern):
                    rel_by_name[r.name] = r
                    metric_by_name[r.name] = metric
                    is_ref[r.name] = False

            ref_actual = canonical_to_actual(version, model, benchmark, "reference")
            print(f"  listing {ref_actual}")
            for r in list_runs_by_regex(api, REL_PROJECT, f"^{re.escape(ref_actual)}$"):
                rel_by_name[r.name] = r
                metric_by_name[r.name] = metric
                is_ref[r.name] = True

    return unl_by_name, rel_by_name, metric_by_name, is_ref


def fetch_one(name, unl_run, rel_run, metric, reference):
    if reference:
        rel_hist = fetch_history_with_retry(rel_run, [metric])
        return name, (None, rel_hist)
    unl_hist = fetch_history_with_retry(unl_run, [metric, DISR_METRIC])
    rel_hist = fetch_history_with_retry(rel_run, [metric])
    return name, (unl_hist, rel_hist)


expected_names = set()
for model in MODELS:
    for benchmark, version, _ in BENCHMARKS:
        for method in titles_dict.keys():
            actual = canonical_to_actual(version, model, benchmark, method)
            for i in range(N_TRIALS):
                expected_names.add(f"{actual}_{i}")
        expected_names.add(canonical_to_actual(version, model, benchmark, "reference"))

missing = [n for n in expected_names if n not in trajectories]

if missing:
    print(f"Missing {len(missing)}/{len(expected_names)} runs; fetching run lists from wandb...")
    unl_by_name, rel_by_name, metric_by_name, is_ref = gather_runs()

    fetch_tasks = []
    for name in missing:
        if name not in rel_by_name:
            print(f"  skipping {name}: no rel run found")
            continue
        reference = is_ref[name]
        if not reference and name not in unl_by_name:
            print(f"  skipping {name}: no unl run found")
            continue
        fetch_tasks.append(
            (name, unl_by_name.get(name), rel_by_name[name], metric_by_name[name], reference)
        )

    print(f"Fetching history for {len(fetch_tasks)} runs with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, *t): t[0] for t in fetch_tasks}
        done = 0
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                name_out, result = fut.result()
                trajectories[name_out] = result
            except Exception as e:
                print(f"  {name}: giving up: {e}")
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(fetch_tasks)} done; saving intermediate cache")
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(trajectories, f)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} runs to {CACHE_FILE}")

# %%
# === CELL 2: select best trial per method, plot one PDF per setup ===

OUT_DIR.mkdir(exist_ok=True)

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
method_to_color = {
    m: default_colors[i % len(default_colors)] for i, m in enumerate(titles_dict.keys())
}


def trial_score(actual_name, i, metric):
    run_name = f"{actual_name}_{i}"
    entry = trajectories.get(run_name)
    if entry is None:
        return None
    unl_hist, rel_hist = entry
    if SELECT_BY == "unlearn":
        if unl_hist is None or metric not in unl_hist.columns or DISR_METRIC not in unl_hist.columns:
            return None
        hist = unl_hist.copy()
        hist[DISR_METRIC] = pd.to_numeric(hist[DISR_METRIC], errors="coerce")
        hist[metric] = pd.to_numeric(hist[metric], errors="coerce")
        hist = hist.dropna(subset=[DISR_METRIC, metric])
        valid = hist[hist[DISR_METRIC] <= DISR_THRESHOLD]
        if len(valid) == 0:
            return None
        score = valid[metric].iloc[-1]
    else:
        if rel_hist is None or metric not in rel_hist.columns or len(rel_hist) == 0:
            return None
        score = rel_hist[metric].head(REL_STEPS).max()
    if pd.isna(score):
        return None
    return score


for model in MODELS:
    for benchmark, version, metric in BENCHMARKS:
        fig, (ax_unl, ax_rel) = plt.subplots(1, 2, figsize=(5.5, 2.1))

        print(f"\n{model} / {benchmark}")
        for method, display_name in titles_dict.items():
            actual = canonical_to_actual(version, model, benchmark, method)
            scores = {
                i: s
                for i in range(N_TRIALS)
                if (s := trial_score(actual, i, metric)) is not None
            }
            if not scores:
                print(f"  {method}: no data")
                continue
            best_i = min(scores, key=scores.get)
            print(f"  {method}: best_i={best_i} score={scores[best_i] * 100:.2f}%")

            unl_hist, rel_hist = trajectories[f"{actual}_{best_i}"]
            color = method_to_color[method]

            if unl_hist is not None and DISR_METRIC in unl_hist.columns:
                hist = unl_hist.dropna(subset=[DISR_METRIC, metric])
                kl = hist[DISR_METRIC].values
                y = (hist[metric] * 100).values
                over = np.where(kl > DISR_THRESHOLD)[0]
                if len(over) == 0:
                    x_plot, y_plot = kl, y
                elif over[0] == 0:
                    x_plot, y_plot = kl[:0], y[:0]
                else:
                    j = over[0]
                    dx = kl[j] - kl[j - 1]
                    t = (DISR_THRESHOLD - kl[j - 1]) / dx if dx else 0.0
                    y_interp = y[j - 1] + t * (y[j] - y[j - 1])
                    x_plot = np.concatenate([kl[:j], [DISR_THRESHOLD]])
                    y_plot = np.concatenate([y[:j], [y_interp]])
                ax_unl.plot(
                    x_plot,
                    y_plot,
                    color=color,
                    alpha=0.7,
                    label=display_name,
                )

            rel_head = rel_hist.head(REL_STEPS)
            ax_rel.plot(
                range(len(rel_head)),
                rel_head[metric] * 100,
                color=color,
                alpha=0.7,
                label=display_name,
            )

        ref_actual = canonical_to_actual(version, model, benchmark, "reference")
        ref_entry = trajectories.get(ref_actual)
        if ref_entry is not None:
            _, ref_rel = ref_entry
            ref_head = ref_rel.head(REL_STEPS)
            ax_rel.plot(
                range(len(ref_head)),
                ref_head[metric] * 100,
                color="black",
                linestyle="--",
                alpha=0.7,
                label="no unlearning",
            )

        unl_lo, unl_hi = ax_unl.get_ylim()
        rel_lo, rel_hi = ax_rel.get_ylim()
        y_lo, y_hi = min(unl_lo, rel_lo), max(unl_hi, rel_hi)
        ax_unl.set_ylim(y_lo, y_hi)
        ax_rel.set_ylim(y_lo, y_hi)

        ax_unl.set_xticks([0.0, 0.005, 0.01])
        ax_rel.set_xticks(range(REL_STEPS))
        ax_unl.set_title("Unlearning")
        ax_rel.set_title("Relearning")
        ax_unl.set_xlabel("Wikitext KL")
        ax_rel.set_xlabel("Epochs")

        handles, labels = ax_rel.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=(len(titles_dict) + 2) // 2,
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
        )

        plt.tight_layout(rect=[0.04, 0.04, 1, 1])
        fig.supylabel("Post-Attack Answer Probability (%) ↓", fontsize=10, x=0.03)

        save_path = OUT_DIR / f"{model}_{benchmark}.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved {save_path}")
        plt.close(fig)

# %%
