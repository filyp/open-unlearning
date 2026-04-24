# %%
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
CACHE_FILE = SCRIPT_DIR / "hard_vs_soft_collapse.pkl"

REL_PROJECT = "filyp/rel-selective-unlearning"
REL_STEPS = 10

# (display, field)
MODELS = [
    ("Llama-3.1-8B", "Llama-3.1-8B"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
]
N_PCS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
HARD_SOFT = ["soft", "hard"]
DISTRIBUTIONS = [("forget", "", "-"), ("retain", "_retain", "--")]  # (label, suffix, linestyle)

# (exp_name_in_task, display, metric)
BENCHMARKS = [
    ("bio", "WMDP-Bio", "train/recall_prob"),
    ("aa", "Animal Abuse", "train/holdout_harmful_prob"),
]

TASK_TEMPLATE = "hardvssoft_{exp}_{model}_{n}_{hs}{suffix}"

# %%
# === CELL 1: fetch relearning trajectories (cached) ===

if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} cached runs from {CACHE_FILE}")
else:
    trajectories = {}

expected = []
for exp, _, metric in BENCHMARKS:
    for _, model_field in MODELS:
        for hs in HARD_SOFT:
            for _, suffix, _ in DISTRIBUTIONS:
                for n in N_PCS:
                    expected.append(
                        (metric, TASK_TEMPLATE.format(
                            exp=exp, model=model_field, n=n, hs=hs, suffix=suffix
                        ))
                    )

missing = [(m, t) for m, t in expected if t not in trajectories]
if missing:
    api = wandb.Api(timeout=3600)
    for metric, t in missing:
        print(f"Fetching {t} (metric={metric})...")
        runs = list(api.runs(REL_PROJECT, filters={"display_name": t}))
        if len(runs) == 0:
            print(f"  no run found; marking None")
            trajectories[t] = None
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
        trajectories[t] = hist
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} runs to {CACHE_FILE}")

# %%
# === CELL 2: plot ===

nrows = len(MODELS)
ncols = len(BENCHMARKS)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5, 1.8 * nrows))
if nrows == 1:
    axes = [axes]

hs_color = {"soft": "tab:blue", "hard": "tab:orange"}

for row_idx, (model_display, model_field) in enumerate(MODELS):
    for col_idx, (exp, bench_display, metric) in enumerate(BENCHMARKS):
        ax = axes[row_idx][col_idx]
        for hs in HARD_SOFT:
            for dist_label, suffix, linestyle in DISTRIBUTIONS:
                xs, ys = [], []
                for n in N_PCS:
                    t = TASK_TEMPLATE.format(
                        exp=exp, model=model_field, n=n, hs=hs, suffix=suffix
                    )
                    hist = trajectories.get(t)
                    if hist is None or metric not in hist.columns or len(hist) == 0:
                        continue
                    head = hist.head(REL_STEPS)[metric].dropna()
                    if len(head) == 0:
                        continue
                    xs.append(n)
                    ys.append(head.max() * 100)
                ax.plot(
                    xs, ys,
                    color=hs_color[hs], linestyle=linestyle,
                    label=f"{hs} ({dist_label})",
                )

        ax.set_xscale("log")
        ax.set_xticks(N_PCS)
        ax.set_xticklabels([str(n) for n in N_PCS], rotation=45)
        ax.xaxis.set_minor_locator(plt.NullLocator())

        if row_idx == 0:
            ax.set_title(bench_display)
        if row_idx == nrows - 1:
            ax.set_xlabel("n_pcs")
        else:
            ax.set_xticklabels([])

        # Model name on the right side of the right subplot
        if col_idx == ncols - 1:
            ax.text(
                1.03, 0.5, model_display,
                transform=ax.transAxes, rotation=-90, ha="left", va="center",
            )

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="lower center", ncol=len(HARD_SOFT) * len(DISTRIBUTIONS),
    bbox_to_anchor=(0.5, -0.04), frameon=False,
)

plt.tight_layout(rect=[0.04, 0.04, 1, 1])
fig.supylabel("Post-Attack Answer Probability (%) ↓", fontsize=10, x=0.03)

save_path = SCRIPT_DIR / "hard_vs_soft_collapse.pdf"
fig.savefig(save_path, bbox_inches="tight")
print(f"Saved plot to {save_path}")
plt.show()

# %%
