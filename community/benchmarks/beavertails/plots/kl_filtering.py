# %%
"""
Visualize KL filtering masks across layers for a given sample.

Usage:
    python community/benchmarks/beavertails/plots/kl_filtering.py \
        --batch_idx 0 --sample_idx 0 --module gate_proj
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch as pt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from transformers import AutoTokenizer

run_name = "L1B_t0.75_0.1"
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
MASKS_DIR = REPO_ROOT / f"saves/unlearn/{run_name}/masks"
MODEL_ID = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def load_masks(batch_idx, module_type, step=0):
    """Load kl_mask_2d for all layers of a given module type.

    Args:
        step: training step index (0 = first saved, 1 = second, ...).

    Returns dict: layer_num -> mask data dict, sorted by layer.
    """
    batch_dir = MASKS_DIR / str(batch_idx)
    if not batch_dir.exists():
        raise FileNotFoundError(
            f"No masks found for batch_idx={batch_idx} at {batch_dir}"
        )

    masks = {}
    for module_dir in sorted(batch_dir.iterdir()):
        name = module_dir.name  # e.g. model.layers.0.mlp.gate_proj
        if not name.endswith(f".{module_type}"):
            continue
        # extract layer number
        parts = name.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        timestamps = sorted(module_dir.glob("*.pt"))
        if not timestamps or step >= len(timestamps):
            continue
        mask_path = timestamps[step]
        masks[layer_idx] = pt.load(mask_path, map_location="cpu", weights_only=True)

    return dict(sorted(masks.items()))


def load_masks_avg(batch_idx, module_type):
    """Load kl_mask_2d averaged across all steps for a given module type."""
    batch_dir = MASKS_DIR / str(batch_idx)
    if not batch_dir.exists():
        raise FileNotFoundError(
            f"No masks found for batch_idx={batch_idx} at {batch_dir}"
        )

    # count steps from any matching module dir
    sample_dir = next(
        d for d in sorted(batch_dir.iterdir()) if d.name.endswith(f".{module_type}")
    )
    n_steps = len(sorted(sample_dir.glob("*.pt")))

    # accumulate across steps (cast to float first since kl_mask_2d is bool)
    acc = None
    for step in range(n_steps):
        masks = load_masks(batch_idx, module_type, step)
        if acc is None:
            acc = masks
            for layer_num in acc:
                acc[layer_num]["kl_mask_2d"] = acc[layer_num]["kl_mask_2d"].float()
        else:
            for layer_num in acc:
                acc[layer_num]["kl_mask_2d"] += masks[layer_num]["kl_mask_2d"].float()

    for layer_num in acc:
        acc[layer_num]["kl_mask_2d"] /= n_steps

    return acc, n_steps


def add_loss_delta_column(ax, fig, loss_delta, tokens, layer_nums):
    """Add a colored Δloss column on the left and color token labels to match."""
    vmax = max(abs(loss_delta.min()), abs(loss_delta.max()), 1e-6)
    loss_cmap = LinearSegmentedColormap.from_list(
        "loss", ["#0088ff", "#000000", "#ff8800"], N=256
    )
    loss_norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    # draw loss column at x=-1 (left of the heatmap)
    for row in range(len(loss_delta)):
        color = loss_cmap(loss_norm(loss_delta[row]))
        ax.add_patch(
            plt.Rectangle((-1.5, row - 0.5), 1, 1, color=color)
        )
        ax.text(
            -1, row, f"{loss_delta[row]:+.1f}",
            ha="center", va="center", fontsize=5, fontfamily="monospace", color="white",
        )

    ax.set_xlim(-1.5, len(layer_nums) - 0.5)

    # update x ticks to include Δloss column
    ax.set_xticks([-1] + list(range(len(layer_nums))))
    ax.set_xticklabels(["Δloss"] + list(layer_nums), fontsize=7)

    # color token labels with the same loss colormap
    for row, label in enumerate(ax.get_yticklabels()):
        color = loss_cmap(loss_norm(loss_delta[row]))
        label.set_bbox(dict(facecolor=color, edgecolor="none", pad=1, alpha=0.8))
        label.set_color("white")


# %% plot_kl_mask averaged across all steps
# args:
batch_idx = 2
sample_idx = 4
module_type = "up_proj"

masks, n_steps = load_masks_avg(batch_idx, module_type)
layer_nums = list(masks.keys())

data = next(iter(masks.values()))
input_ids = data["input_ids"][sample_idx]
attention_mask = data["attention_mask"][sample_idx]

tokens = [tokenizer.decode(tid) for tid in input_ids]

valid = attention_mask.bool()
valid_indices = valid.nonzero(as_tuple=True)[0]
tokens = [tokens[i] for i in valid_indices]

matrix = pt.zeros(len(valid_indices), len(layer_nums))
for col, layer_num in enumerate(layer_nums):
    kl_mask = masks[layer_num]["kl_mask_2d"][sample_idx]
    matrix[:, col] = kl_mask[valid_indices].float()

fig_height = max(4, len(tokens) * 0.14)
fig_width = max(4, len(layer_nums) * 0.2)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

cmap = LinearSegmentedColormap.from_list("kl", ["#000000", "#ff0000"], N=256)
im = ax.imshow(
    matrix.numpy(),
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens, fontsize=7, fontfamily="monospace")

ax.set_xticks(range(len(layer_nums)))
ax.set_xticklabels(layer_nums, fontsize=7)
ax.set_xlabel("Layer")

ax.set_title(
    f"KL mask avg ({n_steps} steps) — batch {batch_idx}\nsample {sample_idx}, {module_type}"
)

# per-token loss delta as colored column on the left (from latest step)
last_step_masks = load_masks(batch_idx, module_type, -1)
last_data = next(iter(last_step_masks.values()))
if "token_loss_delta" in last_data:
    loss_delta = last_data["token_loss_delta"][sample_idx][valid_indices].numpy()
    add_loss_delta_column(ax, fig, loss_delta, tokens, layer_nums)

fig.tight_layout()

# %%
