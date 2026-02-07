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
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer

run_name = "L1B_t0.75"
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


# %% plot_kl_mask
# args:
batch_idx = 0
sample_idx = 1
module_type = "gate_proj"
step = 0


masks = load_masks(batch_idx, module_type, step)
layer_nums = list(masks.keys())

# get input_ids and token texts from any layer's data (they're all the same)
data = next(iter(masks.values()))
input_ids = data["input_ids"][sample_idx]
attention_mask = data["attention_mask"][sample_idx]

tokens = [tokenizer.decode(tid) for tid in input_ids]

# keep only tokens where attention_mask is 1
valid = attention_mask.bool()
valid_indices = valid.nonzero(as_tuple=True)[0]
tokens = [tokens[i] for i in valid_indices]

# build matrix: rows=tokens, cols=layers
matrix = pt.zeros(len(valid_indices), len(layer_nums))
for col, layer_num in enumerate(layer_nums):
    kl_mask = masks[layer_num]["kl_mask_2d"][sample_idx]
    matrix[:, col] = kl_mask[valid_indices].float()

# plot
fig_height = max(4, len(tokens) * 0.14)
fig_width = max(4, len(layer_nums) * 0.2)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# cmap = LinearSegmentedColormap.from_list("kl", ["#f0f0f0", "#d62728"], N=256)
cmap = LinearSegmentedColormap.from_list("kl", ["#000000", "#ff0000"], N=256)
im = ax.imshow(
    matrix.numpy(),
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

# token labels on y-axis
ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens, fontsize=7, fontfamily="monospace")

# layer labels on x-axis
ax.set_xticks(range(len(layer_nums)))
ax.set_xticklabels(layer_nums, fontsize=7)
ax.set_xlabel("Layer")

ax.set_title(f"KL mask — batch {batch_idx}, sample {sample_idx}, {module_type}")

fig.colorbar(im, ax=ax, shrink=0.5, label="kl_mask value")
fig.tight_layout()

# out_path = Path(__file__).parent / f"kl_mask_b{batch_idx}_s{sample_idx}_{module_type}.png"
# fig.savefig(out_path, dpi=150)
# print(f"Saved to {out_path}")
# plt.close(fig)

# %% plot_kl_mask averaged across all steps
# args:
batch_idx = 0
sample_idx = 0
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
    f"KL mask avg ({n_steps} steps) — batch {batch_idx}, sample {sample_idx}, {module_type}"
)

fig.colorbar(im, ax=ax, shrink=0.5, label="kl_mask frequency")
fig.tight_layout()
