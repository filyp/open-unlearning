# %%
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wandb

# mirroring main_grid.py
study_remap = {
    # Llama bio: actual runs use v5 (no .3)
    "v5.3_Llama-3.1-8B_bio_GradDiff": "v5_Llama-3.1-8B_bio_GradDiff",
    "v5.3_Llama-3.1-8B_bio_NPO": "v5_Llama-3.1-8B_bio_NPO",
    "v5.3_Llama-3.1-8B_bio_RMU": "v5_Llama-3.1-8B_bio_RMU",
    "v5.3_Llama-3.1-8B_bio_SimNPO": "v5_Llama-3.1-8B_bio_SimNPO",
    "v5.3_Llama-3.1-8B_bio_UNDIAL": "v5_Llama-3.1-8B_bio_UNDIAL",
    # Llama animal_abuse: actual runs use v7 (no .3) for non-RepSelect methods
    "v7.3_Llama-3.1-8B_animal_abuse_GradDiff": "v7_Llama-3.1-8B_animal_abuse_GradDiff",
    "v7.3_Llama-3.1-8B_animal_abuse_NPO": "v7_Llama-3.1-8B_animal_abuse_NPO",
    "v7.3_Llama-3.1-8B_animal_abuse_RMU": "v7_Llama-3.1-8B_animal_abuse_RMU",
    "v7.3_Llama-3.1-8B_animal_abuse_SimNPO": "v7_Llama-3.1-8B_animal_abuse_SimNPO",
    "v7.3_Llama-3.1-8B_animal_abuse_UNDIAL": "v7_Llama-3.1-8B_animal_abuse_UNDIAL",
    # Bio RepSelect runs are bare (no _forget suffix)
    "v5.3_Llama-3.1-8B_bio_RepSelect_forget": "v5_Llama-3.1-8B_bio_RepSelect",
    "v5.3_gemma-4-E4B_bio_RepSelect_forget": "v5.3_gemma-4-E4B_bio_RepSelect",
    "v5.3_DeepSeek-V2-Lite_bio_RepSelect_forget": "v5.3_DeepSeek-V2-Lite_bio_RepSelect",
    # animal_abuse RepSelect retain runs are bare (no _retain suffix)
    "v7.3_Llama-3.1-8B_animal_abuse_RepSelect_retain": "v7_Llama-3.1-8B_animal_abuse_RepSelect",
    "v7.3_gemma-4-E4B_animal_abuse_RepSelect_retain": "v7.3_gemma-4-E4B_animal_abuse_RepSelect",
    "v7.3_DeepSeek-V2-Lite_animal_abuse_RepSelect_retain": "v7.3_DeepSeek-V2-Lite_animal_abuse_RepSelect",
}

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "trajectories.pkl"

OUT_DIR = SCRIPT_DIR / "methods"
OUT_DIR_PNG = SCRIPT_DIR / "methods_png"

titles_dict = {
    "RepSelectSimple_forget": "RepSelect",
    # "RepSelect2_forget": "└ multi-epoch",
    # "RepSelectSimple_forget_no_lora": "└ w/o LoRA",
    "NPO": "NPO",
    "RMU": "RMU",
    # "UNDIAL": "UNDIAL",
    # "SimNPO": "SimNPO",
    # "GradDiff": "GradDiff",
}
for key, value in titles_dict.items():
    titles_dict[key] = value.replace("└ ", "RepSelect ")

# # select best trial by: "relearn" (max relearning metric) or "unlearn"
# # (last valid unlearning metric with KL <= threshold). Lower is better in both.
# SELECT_BY = "unlearn"
SELECT_BY = "relearn"

TOP_N = 10
REL_STEPS = 5
UNL_PROJECT = "filyp/selective-unlearning"
REL_PROJECT = "filyp/rel-selective-unlearning"
DISR_METRIC = "train/wikitext_kl"
DISR_THRESHOLD = 0.01
N_TRIALS = 30
MAX_WORKERS = 12
LINE_ALPHA = 1
LINE_WIDTH = 1
N_GRID = 50
BAND_ALPHA = 0.2

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
OUT_DIR_PNG.mkdir(exist_ok=True)

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
        valid = unl_hist[unl_hist[DISR_METRIC] <= DISR_THRESHOLD]
        if len(valid) == 0:
            return None
        return valid[metric].iloc[-1]
    return rel_hist[metric].head(REL_STEPS).max()


def interp_to_grid(kl, y, x_grid):
    a, b = kl[:-1, None], kl[1:, None]
    ya, yb = y[:-1, None], y[1:, None]
    in_seg = (np.minimum(a, b) <= x_grid) & (x_grid <= np.maximum(a, b))
    dx = np.where(a == b, 1.0, b - a)
    interp = ya + (x_grid - a) / dx * (yb - ya)
    interp = np.where(in_seg, interp, np.inf)
    out = interp.min(axis=0)
    return np.where(np.isinf(out), np.nan, out)


x_grid = np.linspace(0.0, DISR_THRESHOLD, N_GRID)


benchmark_display = {"bio": "WMDP-Bio", "animal_abuse": "Animal Abuse"}

for model in MODELS:
    nrows = len(BENCHMARKS)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(5.5, 1.6 * nrows),
        sharey="row", gridspec_kw={"wspace": 0.08, "hspace": 0.4},
    )
    if nrows == 1:
        axes = [axes]

    for row_idx, (benchmark, version, metric) in enumerate(BENCHMARKS):
        ax_unl, ax_rel = axes[row_idx]

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
            top_is = sorted(scores, key=scores.get)[:TOP_N]
            print(
                f"  {method}: top={top_is} "
                f"scores={[f'{scores[i] * 100:.2f}%' for i in top_is]}"
            )
            color = method_to_color[method]

            unl_grids = []
            rel_arrs = []
            for trial_i in top_is:
                unl_hist, rel_hist = trajectories[f"{actual}_{trial_i}"]
                kl = unl_hist[DISR_METRIC].astype(float).values
                y = unl_hist[metric].astype(float).values * 100
                unl_grids.append(interp_to_grid(kl, y, x_grid))
                rel_arrs.append(
                    rel_hist[metric].astype(float).head(REL_STEPS).values * 100
                )

            stacked = np.stack(unl_grids)
            mean_u = np.nanmean(stacked, axis=0)
            std_u = np.nanstd(stacked, axis=0)
            ax_unl.plot(
                x_grid, mean_u, color=color, alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH, label=display_name,
            )
            ax_unl.fill_between(
                x_grid, mean_u - std_u, mean_u + std_u,
                color=color, alpha=BAND_ALPHA, linewidth=0,
            )

            rel_stacked = np.stack(rel_arrs)
            mean_r = rel_stacked.mean(axis=0)
            std_r = rel_stacked.std(axis=0)
            xs = np.arange(REL_STEPS)
            ax_rel.plot(
                xs, mean_r, color=color, alpha=LINE_ALPHA,
                linewidth=LINE_WIDTH, label=display_name,
            )
            ax_rel.fill_between(
                xs, mean_r - std_r, mean_r + std_r,
                color=color, alpha=BAND_ALPHA, linewidth=0,
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

        ax_unl.set_xticks([0.0, 0.005, 0.01])
        ax_rel.set_xticks(range(REL_STEPS))

        if row_idx == 0:
            ax_unl.set_title("Unlearning")
            ax_rel.set_title("Relearning")
        if row_idx == nrows - 1:
            ax_unl.set_xlabel("Wikitext KL")
            ax_rel.set_xlabel("Epochs")

        ax_rel.text(
            1.03, 0.5, benchmark_display.get(benchmark, benchmark),
            transform=ax_rel.transAxes, rotation=-90, ha="left", va="center",
        )

    handles, labels = axes[0][1].get_legend_handles_labels()
    if "no unlearning" in labels:
        i = labels.index("no unlearning")
        handles = [handles[i]] + handles[:i] + handles[i + 1 :]
        labels = [labels[i]] + labels[:i] + labels[i + 1 :]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )

    plt.tight_layout(rect=[0.04, 0.04, 1, 1])
    fig.supylabel("Post-Attack Answer Probability (%) ↓", fontsize=10, x=0.03)

    save_path = OUT_DIR / f"{model}.pdf"
    fig.savefig(save_path, bbox_inches="tight")
    save_path_png = OUT_DIR_PNG / f"{model}.png"
    fig.savefig(save_path_png, bbox_inches="tight", dpi=200)
    print(f"  saved {save_path} and {save_path_png}")
    plt.close(fig)

# %%
