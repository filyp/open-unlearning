# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

plt.style.use("default")
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.size"] = 10


# %%
version = "v3"

split = "bio"
# split = "cyber"

file_name = Path(__file__).parent / f"{version}_3B__{split}.pkl"

# we load from the files created by relearning_trajectories.py
with open(file_name, "rb") as f:
    method_histories = pickle.load(f)

# %% Arrow plot: x = initial wikitext_loss, y = recall_loss trajectory

method_names = [
    "SimNPO",
    "UNDIAL",
    "GradDiff",
    "RMU",
    "NPO",
    "CIR",
    # "NPOstrict",
    # "CIRstrict",
]

metric_name = "train/recall_loss"
# metric_name = "train/forget_acc_t1"


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_map = {method: colors[i % len(colors)] for i, method in enumerate(method_names)}

fig, ax = plt.subplots(figsize=(8, 6))

max_x = max_y = 0
min_x = min_y = float("inf")

for method_name in method_names:
    for run_num, (unl_hist, rel_hist) in enumerate(method_histories[method_name]):
        # Find last datapoint where wikitext_kl is still below 0.005
        wikitext_kl = unl_hist["train/wikitext_kl"]
        valid_indices = [i for i, kl in enumerate(wikitext_kl) if kl < 0.1]
        last_valid_idx = valid_indices[-1]

        x = wikitext_kl[last_valid_idx]
        # x = rel_hist["train/wikitext_loss"][0]

        # Assert that wikitext_loss at this point matches the 0th datapoint of relearning
        assert (
            unl_hist["train/wikitext_loss"][last_valid_idx]
            == rel_hist["train/wikitext_loss"][0]
        )

        y_start = rel_hist[metric_name][0]
        if metric_name == "train/recall_loss":
            y_end = min(rel_hist[metric_name])
        elif metric_name == "train/forget_acc_t1":
            y_end = max(rel_hist[metric_name])

        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y_end, y_start)
        min_y = min(min_y, y_end, y_start)

        # Draw arrow from (x, y_start) to (x, y_end)
        ax.annotate(
            "",
            xy=(x, y_end),
            xytext=(x, y_start),
            arrowprops=dict(
                arrowstyle="->",
                color=color_map[method_name],
                # mutation_scale=5,
            ),
        )

# Set y-axis to logarithmic scale
if metric_name == "train/recall_loss":
    ax.set_yscale("log")

# Set axis limits with padding
x_margin = 0.002
y_margin = 0.005
ax.set_xlim(min_x - x_margin, max_x + x_margin)
ax.set_ylim(min_y - y_margin, max_y + y_margin)

# Create legend with dummy handles
handles = [plt.Line2D([0], [0], color=color_map[m], label=m) for m in method_names]
ax.legend(handles=handles, loc="upper right")

ax.set_xlabel("Disruption (KL divergence on WikiText after unlearning)")
if metric_name == "train/recall_loss":
    ax.set_ylabel("Recall loss (log)")
ax.set_title(f"Recall loss drop during relearning ({split})")
plt.tight_layout()
plt.show()

# %%
