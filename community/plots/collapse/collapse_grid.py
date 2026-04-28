# %%
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

import wandb

# Baselines from dedicated reference runs in wandb. Shape: {dataset: {model: value}}.
_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
baselines: dict[str, dict[str, float]] = {}
for _path in [
    _BENCHMARKS_DIR / "wmdp_low_mi" / "baselines.yaml",
    _BENCHMARKS_DIR / "beavertails" / "baselines.yaml",
]:
    with open(_path) as _f:
        baselines.update(yaml.safe_load(_f))

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "collapse_cache.pkl"

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 10
SHOW_INITIAL = False  # draw error-bar-style marker from max back to initial prob
SHOW_MIXED = False  # include "r. act, f. grad" row in the AA benchmark

MODELS = [
    ("Llama-3.1-8B", "Llama-3.1-8B"),
    ("Gemma-4-E4B", "gemma-4-E4B"),
    ("DeepSeek-V2-Lite", "DeepSeek-V2-Lite"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
]

# (exp_name_in_task, display, reference_benchmark_tag, metric)
BENCHMARKS = [
    ("bio", "WMDP-Bio", "bio", "train/recall_prob"),
    ("AA", "Animal Abuse", "animal_abuse", "train/holdout_harmful_prob"),
]

# (label, task-suffix). forget_none == retain_none so we show one
# "no collapse" row at the bottom (pick forget arbitrarily to fetch).
BASE_CONFIGS = [
    ("forget / act", "forget_act"),
    ("forget / grad", "forget_grad"),
    ("forget / both", "forget_both"),
    ("retain / act", "retain_act"),
    ("retain / grad", "retain_grad"),
    ("retain / both", "retain_both"),
    ("no collapse", "forget_none"),
]
# Per-benchmark configs. AA has the extra mixed (retain-act, forget-grad) run.
BENCH_CONFIGS = {
    "bio": BASE_CONFIGS,
    "AA": BASE_CONFIGS + (
        [("r. act, f. grad", "actretain_gradforget")] if SHOW_MIXED else []
    ),
}


def task_name(exp_name, model, suffix):
    return f"collapse_{exp_name}_{model}_{suffix}"


# %%
# === CELL 1: fetch relearning trajectories (cached) ===

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded {len(cache)} cached runs from {CACHE_FILE}")
else:
    cache = {}

expected = []
for _, model_field in MODELS:
    for exp_name, _, _, _ in BENCHMARKS:
        for _, suffix in BENCH_CONFIGS[exp_name]:
            expected.append((exp_name, task_name(exp_name, model_field, suffix)))

missing = [(e, t) for e, t in expected if t not in cache]
if missing:
    api = wandb.Api(timeout=3600)
    for exp_name, t in missing:
        metric = next(m for en, _, _, m in BENCHMARKS if en == exp_name)
        print(f"Fetching {t} (metric={metric})...")
        runs = list(api.runs(REL_PROJECT, filters={"display_name": t}))
        if len(runs) == 0:
            print(f"  no run found; marking None")
            cache[t] = None
            continue
        if len(runs) > 1:
            print(f"  warning: {len(runs)} runs, taking first")
        for i in range(10):
            try:
                hist = runs[0].history(keys=[metric])
                break
            except Exception as e:
                print(f"  attempt {i} failed: {e}")
                time.sleep(2**i)
        else:
            raise RuntimeError(f"failed to fetch history for {t}")
        cache[t] = hist
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved {len(cache)} runs to {CACHE_FILE}")

# %%
# === CELL 2: plot ===

nrows = len(BENCHMARKS)
ncols = len(MODELS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 3.2))
if nrows == 1:
    axes = [axes]
if ncols == 1:
    axes = [[ax] for ax in axes]

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# Color by collapse kind so forget/retain pairs share color; retain is hatched.
collapse_color = {"act": colors[0], "grad": colors[1], "both": colors[2], "none": colors[3]}


def style_for(suffix):
    if suffix == "actretain_gradforget":
        return colors[4], "xxx"  # mixed: distinct color + cross-hatch
    dist, coll = suffix.split("_")
    return collapse_color[coll], ("///" if dist == "retain" else "")

for row_idx, (exp_name, bench_display, bench_tag, metric) in enumerate(BENCHMARKS):
    configs = BENCH_CONFIGS[exp_name]
    bar_colors = [style_for(s)[0] for _, s in configs]
    bar_hatches = [style_for(s)[1] for _, s in configs]
    # Top-to-bottom: first config is on top.
    y_positions = list(range(len(configs) - 1, -1, -1))

    for col_idx, (model_display, model_field) in enumerate(MODELS):
        ax = axes[row_idx][col_idx]
        baseline = baselines[bench_tag][model_field] * 100

        maxes = []
        initials = []
        for _, suffix in configs:
            t = task_name(exp_name, model_field, suffix)
            hist = cache.get(t)
            if hist is None or metric not in hist.columns or len(hist) == 0:
                # Render nothing: bar width = 0, initial = baseline.
                maxes.append(baseline)
                initials.append(baseline)
                continue
            head = hist.head(REL_STEPS)[metric].dropna()
            if len(head) == 0:
                maxes.append(baseline)
                initials.append(baseline)
                continue
            maxes.append(head.max() * 100)
            initials.append(head.iloc[0] * 100)

        widths = [m - baseline for m in maxes]
        barh_kwargs = dict(
            height=1.0,
            color=bar_colors,
            left=baseline,
        )
        if SHOW_INITIAL:
            # xerr extends from bar tip (at `maxes[i]`) toward `initials[i]`.
            xerr_lower = [max(mx - ini, 0) for mx, ini in zip(maxes, initials)]
            xerr_upper = [max(ini - mx, 0) for mx, ini in zip(maxes, initials)]
            barh_kwargs.update(
                xerr=[xerr_lower, xerr_upper],
                capsize=3,
                error_kw={"ecolor": "black", "elinewidth": 1.0, "alpha": 0.7},
            )

        bars = ax.barh(y_positions, widths, **barh_kwargs)
        for patch, hatch in zip(bars, bar_hatches):
            if hatch:
                patch.set_hatch(hatch)
                patch.set_edgecolor("white")

        if row_idx == 0:
            ax.set_title(model_display)
        if col_idx == 0:
            ax.set_ylabel(bench_display)

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)

        all_xs = maxes + initials if SHOW_INITIAL else maxes
        min_val = min(all_xs)
        min_val -= (baseline - min_val) * 0.05 if baseline > min_val else 0.5
        ax.set_xlim(min_val, baseline)
        ax.set_ylim(min(y_positions) - 0.5, max(y_positions) + 0.5)

        if col_idx == ncols - 1:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([label for label, _ in configs])
            ax.yaxis.tick_right()
        else:
            ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
fig.text(
    0.5, -0.02, "Post-Attack Answer Probability (%) ↓", ha="center", va="bottom"
)

save_path = SCRIPT_DIR / "collapse_grid.pdf"
fig.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"Saved plot to {save_path}")
plt.show()

# %%
