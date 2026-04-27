# %%
# results: just 10 samples already gives us over 50% (57%) of the 360 sample post-attack probability drop (for llama, for qwen it looks even more)
# at 90 samples, there are no further improvements visible, 90 samples is enough
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt

import wandb

plt.style.use("default")
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 10

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / "data_scaling_law.pkl"

UNL_PROJECT = "filyp/selective-unlearning"
REL_PROJECT = "filyp/rel-selective-unlearning"

# (range_size, color) — rainbow by range size
RANGES = [
    (10, "red"),
    (25, "orange"),
    # (45, "gold"),
    (90, "green"),
    # (180, "blue"),
    (360, "blue"),
]

# (display_name, model_field_in_task_name, ylim)
MODELS = [
    ("Llama-3.1-8B", "Llama-3.1-8B", (9, 22)),
    ("Qwen3.5-9B", "Qwen3.5-9B", (3.5, 23)),
]

TASK_TEMPLATE = "datascaling2_AA_{model}_range{r}"
# per-model reference run name (Llama ref was named before the v7.3 rename)
REF_NAMES = {
    "Llama-3.1-8B": "v7_Llama-3.1-8B_animal_abuse_reference",
    "Qwen3.5-9B": "v7.3_Qwen3.5-9B_animal_abuse_reference",
}

METRIC = "train/holdout_harmful_prob"
DISR_METRIC = "train/wikitext_kl"
DISR_THRESHOLD = 0.01
REL_STEPS = 5
METRIC_KEYS = [METRIC, DISR_METRIC]

ALL_TASKS = [
    TASK_TEMPLATE.format(model=m, r=r) for _, m, _ in MODELS for r, _ in RANGES
]
REF_TASKS = [REF_NAMES[m] for _, m, _ in MODELS]

# %%
# === CELL 1: Load trajectories from wandb (slow, cached) ===

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded cached trajectories from {CACHE_FILE}")
else:
    trajectories = {}

missing = [t for t in ALL_TASKS if t not in trajectories]
missing_refs = [t for t in REF_TASKS if t not in trajectories]
if missing or missing_refs:
    api = wandb.Api(timeout=3600)

    for task_name in missing:
        print(f"Fetching {task_name}...")
        unl_runs = api.runs(UNL_PROJECT, filters={"display_name": task_name})
        rel_runs = api.runs(REL_PROJECT, filters={"display_name": task_name})
        assert len(unl_runs) == 1, f"expected 1 unlearn run for {task_name}, got {len(unl_runs)}"
        assert len(rel_runs) == 1, f"expected 1 relearn run for {task_name}, got {len(rel_runs)}"

        for i in range(10):
            try:
                unl_hist = unl_runs[0].history(keys=METRIC_KEYS)
                rel_hist = rel_runs[0].history(keys=[METRIC])
                break
            except Exception as e:
                print(f"  {i}: error, retrying: {e}")
                time.sleep(2**i)
        trajectories[task_name] = (unl_hist, rel_hist)

    for task_name in missing_refs:
        print(f"Fetching reference {task_name}...")
        rel_runs = api.runs(REL_PROJECT, filters={"display_name": task_name})
        assert len(rel_runs) == 1, f"expected 1 reference run for {task_name}, got {len(rel_runs)}"

        for i in range(10):
            try:
                rel_hist = rel_runs[0].history(keys=[METRIC])
                break
            except Exception as e:
                print(f"  {i}: error, retrying: {e}")
                time.sleep(2**i)
        trajectories[task_name] = (None, rel_hist)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved trajectories to {CACHE_FILE}")

# %%
# === CELL 2: Plot ===

nrows = len(MODELS)
fig, axes = plt.subplots(
    nrows, 2, figsize=(5.5, 1.8 * nrows),
    sharey="row", gridspec_kw={"wspace": 0.08, "hspace": 0.4},
)
if nrows == 1:
    axes = [axes]

for row_idx, (model_display, model_field, ylim) in enumerate(MODELS):
    ax_unl, ax_rel = axes[row_idx]

    for r, color in RANGES:
        task_name = TASK_TEMPLATE.format(model=model_field, r=r)
        unl_hist, rel_hist = trajectories[task_name]

        # Unlearning: reject points where disruption exceeds threshold
        unl_valid = unl_hist[unl_hist[DISR_METRIC] <= DISR_THRESHOLD]
        ax_unl.plot(
            unl_valid[DISR_METRIC],
            unl_valid[METRIC] * 100,
            color=color,
            alpha=0.5,
            label=f"{r} samples",
        )

        # Relearning: first N epochs (one row per eval, plotted against row index)
        rel_head = rel_hist.head(REL_STEPS)
        ax_rel.plot(
            range(len(rel_head)), rel_head[METRIC] * 100, color=color, alpha=0.6
        )

    # Reference: black dashed, right subplot only
    ref_task = REF_NAMES[model_field]
    _, ref_rel = trajectories[ref_task]
    ref_head = ref_rel.head(REL_STEPS)
    ax_rel.plot(
        range(len(ref_head)),
        ref_head[METRIC] * 100,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="no unlearning",
    )

    ax_unl.set_xticks([0.0, 0.005, 0.01])
    ax_rel.set_xticks(range(REL_STEPS))
    ax_unl.set_ylim(*ylim)

    # Column titles only on the top row
    if row_idx == 0:
        ax_unl.set_title("Unlearning")
        ax_rel.set_title("Relearning")

    # x-axis labels only on the bottom row
    if row_idx == nrows - 1:
        ax_unl.set_xlabel("Wikitext KL")
        ax_rel.set_xlabel("Epochs")

    # Model name on the right side of the right subplot
    ax_rel.text(
        1.03,
        0.5,
        model_display,
        transform=ax_rel.transAxes,
        rotation=-90,
        ha="left",
        va="center",
    )

color_handles, color_labels = axes[0][0].get_legend_handles_labels()
ref_handles, ref_labels = axes[0][1].get_legend_handles_labels()
ref_only = [(h, lbl) for h, lbl in zip(ref_handles, ref_labels) if lbl == "no unlearning"]
handles = color_handles + [h for h, _ in ref_only]
labels = color_labels + [lbl for _, lbl in ref_only]
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.18),
    frameon=False,
)

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
fig.supylabel("Post-Attack Answer Probability (%) ↓", fontsize=10, x=0.03)

save_path = SCRIPT_DIR / "data_scaling_law.pdf"
fig.savefig(save_path, bbox_inches="tight")
print(f"Saved plot to {save_path}")

plt.show()

# %%
