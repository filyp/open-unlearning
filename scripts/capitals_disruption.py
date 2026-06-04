"""
Capitals Disruption Analysis (Section 4, Figure: disruption of superficially similar facts)

Shows how unlearning "The capital of France is Paris" disrupts other facts.
Measures disruption via cosine similarity of weight gradients.
Visualizes per-token activations and gradients at a gate_proj module.

Ported from Filip's script at:
  https://github.com/filyp/unlearning/blob/main/src/plotting/1_capitals.py

Usage:
    python scripts/capitals_disruption.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch as pt
from transformers import AutoModelForCausalLM, AutoTokenizer

plt.style.use("default")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10

DEVICE = "cuda" if pt.cuda.is_available() else "cpu"
MODEL_ID = "meta-llama/Llama-3.2-1B"
TARGET_MODULES = ["gate_proj", "up_proj", "down_proj", "k_proj", "v_proj", "q_proj", "o_proj"]
# Layer and module for activation/gradient visualization
VIS_LAYER = 8
VIS_MODULE_NAME = "gate_proj"
SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "plots"


# --- Hooks ---

def install_hooks(model):
    """Attach forward/backward hooks to save full activations and gradients."""
    for name, module in model.named_modules():
        if any(t in name for t in TARGET_MODULES) and hasattr(module, "weight"):
            module.register_forward_hook(_save_act_hook)
            module.register_full_backward_hook(_save_grad_hook)


def _save_act_hook(module, args, output):
    module.last_act_full = args[0].detach()


def _save_grad_hook(module, grad_input, grad_output):
    module.last_grad_full = grad_output[0].detach()


# --- Gradient computation ---

def prepare_answer_mask(tokenizer, beginning, ending):
    """Create a mask that's True only on the answer tokens."""
    begin_ids = tokenizer(beginning, return_tensors="pt")["input_ids"]
    full_ids = tokenizer(f"{beginning} {ending}", return_tensors="pt")["input_ids"]
    mask = pt.zeros_like(full_ids, dtype=pt.bool)
    mask[0, begin_ids.shape[1]:] = True
    return full_ids.to(DEVICE), mask.to(DEVICE)


def get_grad(model, input_ids, loss_mask):
    """Forward + backward, return per-parameter gradients as dict."""
    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=input_ids)

    # Cross-entropy loss only on answer tokens
    logits = outputs.logits[:, :-1]  # (1, seq_len-1, vocab)
    targets = input_ids[:, 1:]       # (1, seq_len-1)
    mask = loss_mask[:, 1:]          # shift mask to match

    loss_per_token = pt.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    )
    loss = (loss_per_token * mask.reshape(-1).float()).sum() / mask.sum().clamp(min=1)
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def tensor_dict_dot_product(a, b):
    acc = 0
    for k in a.keys():
        acc += (a[k].to(pt.float32) * b[k].to(pt.float32)).sum()
    return acc


def tensor_dict_cossim(a, b):
    a_dot_b = tensor_dict_dot_product(a, b)
    a_norm = tensor_dict_dot_product(a, a).sqrt()
    b_norm = tensor_dict_dot_product(b, b).sqrt()
    return a_dot_b / (a_norm * b_norm)


# --- Data ---

beginning_ending_pairs_groups1 = [
    [
        ("The capital of France is", "Paris"),
    ],
    [
        ("The capital of Spain is", "Madrid"),
        ("The capital of China is", "Beijing"),
        ("The capital of Ukraine is", "Kyiv"),
    ],
    [
        ("The capital of France is", "Madrid"),
        ("The capital of Spain is", "Beijing"),
        ("The capital of China is", "Kyiv"),
        ("The capital of Ukraine is", "Paris"),
    ],
    [
        ("The largest planet is", "Jupiter"),
        ("The author of 1984 is", "George Orwell"),
        ("Marie Curie discovered", "radium"),
        ("Prometheus stole", "fire"),
    ],
]

beginning_ending_pairs_groups2 = [
    [
        ("The capital of Skyrim is", "Solitude"),
        ("The capital of Rohan is", "Edoras"),
    ],
    [
        ("Die Hauptstadt von Frankreich ist", "Paris"),
        ("La capital de Francia es", "París"),
        ("Столица Франции", "Париж"),
        ("A capital de França é", "Paris"),
    ],
    [
        ("Water contains", "hydrogen"),
        ("Salt contains", "sodium"),
        ("Diamond contains", "carbon"),
        ("Air contains", "oxygen"),
    ],
    [
        ("Napoleon is", "French"),
        ("Napoleon was", "French"),
        ("Mozart is", "Austrian"),
        ("Mozart was", "Austrian"),
    ],
    [
        ("Gold is", "valuable"),
        ("Gold was", "valuable"),
    ],
    [
        ("The library is", "quiet"),
        ("The library was", "quiet"),
    ],
]


# --- Analysis ---

def get_infos(model, tokenizer, beginning_ending_pairs):
    """Compute gradient cosine similarity and capture activations/gradients."""
    # Reference: unlearning "The capital of France is Paris"
    ref_ids, ref_mask = prepare_answer_mask(tokenizer, "The capital of France is", "Paris")
    ref_grad = get_grad(model, ref_ids, ref_mask)

    # Find the visualization module
    vis_module = None
    for name, module in model.named_modules():
        if f"layers.{VIS_LAYER}.mlp.{VIS_MODULE_NAME}" == name.split("model.")[-1]:
            vis_module = module
            break
    if vis_module is None:
        # Try with full path
        for name, module in model.named_modules():
            if f"layers.{VIS_LAYER}" in name and name.endswith(VIS_MODULE_NAME):
                vis_module = module
                break

    infos = []
    for beginning, ending in beginning_ending_pairs:
        input_ids, loss_mask = prepare_answer_mask(tokenizer, beginning, ending)
        grad = get_grad(model, input_ids, loss_mask)

        cossim = tensor_dict_cossim(grad, ref_grad)

        # Get pre-answer token position activations/gradients
        begin_ids = tokenizer(beginning, return_tensors="pt")["input_ids"]
        pre_answer_pos = begin_ids.shape[1] - 1

        act = vis_module.last_act_full[0, pre_answer_pos] if hasattr(vis_module, "last_act_full") else pt.zeros(1)
        grad_vis = vis_module.last_grad_full[0, pre_answer_pos] if hasattr(vis_module, "last_grad_full") else pt.zeros(1)

        infos.append(dict(
            beginning=beginning,
            ending=ending,
            cossim=cossim,
            act=act,
            grad=grad_vis,
        ))
    return infos


# --- Plotting ---

def create_activation_visualization(act_tensor, max_vals=35):
    if isinstance(act_tensor, pt.Tensor):
        act_values = act_tensor.cpu().float().numpy()
    else:
        act_values = act_tensor
    act_values = act_values[:max_vals]
    max_abs = np.abs(act_values).max()
    if max_abs > 0:
        act_values = act_values / max_abs
    rgb_image = np.zeros((1, len(act_values), 3))
    rgb_image[0, :, 0] = np.clip(-act_values, 0, 1)
    rgb_image[0, :, 1] = np.clip(act_values, 0, 1)
    return rgb_image


def create_plot(group_infos):
    text_y_pos = 0.4
    total_rows = sum(len(group) for group in group_infos)
    num_gaps = len(group_infos) - 1

    height_ratios = []
    for i, group in enumerate(group_infos):
        height_ratios.extend([1] * len(group))
        if i < len(group_infos) - 1:
            height_ratios.append(0.35)

    fig = plt.figure(figsize=(5.5, total_rows * 0.2))

    gs = gridspec.GridSpec(
        total_rows + num_gaps, 4,
        width_ratios=[0.45, 0.1, 0.225, 0.225],
        height_ratios=height_ratios,
        hspace=-0.01, wspace=0.07,
        left=0.0, right=1.0, top=0.92, bottom=0.02,
    )

    def create_group_slice_image(group_infos, slice_type):
        group_slices = []
        for info in reversed(group_infos):
            slice_vis = create_activation_visualization(info[slice_type])
            group_slices.append(slice_vis[0])
        return np.stack(group_slices, axis=0)

    group_activation_images = [create_group_slice_image(g, "act") for g in group_infos]
    group_gradient_images = [create_group_slice_image(g, "grad") for g in group_infos]

    all_infos = []
    row_indices = []
    current_row = 0
    for group_idx, group in enumerate(group_infos):
        for info in group:
            all_infos.append(info)
            row_indices.append(current_row)
            current_row += 1
        if group_idx < len(group_infos) - 1:
            current_row += 1

    for i, info in enumerate(all_infos):
        actual_row = row_indices[i]

        ax1 = fig.add_subplot(gs[actual_row, 0])
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.axis("off")
        ax1.text(0.01, text_y_pos, info["beginning"], va="center", ha="left", color="black")
        ax1.text(0.99, text_y_pos, info["ending"], va="center", ha="right", color="purple")

        ax2 = fig.add_subplot(gs[actual_row, 1])
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
        cossim_val = info["cossim"]
        percentage = int(cossim_val * 100)
        clipped = cossim_val.clamp(0, 1).item()
        bg_color = (1.0, 1.0 - clipped, 1.0 - clipped)
        ax2.add_patch(patches.Rectangle((0, 0), 1, 1, linewidth=0, facecolor=bg_color))
        ax2.text(0.5, text_y_pos, f"{percentage}%", va="center", ha="center", color="black")

    current_row = 0
    for group_idx, group in enumerate(group_infos):
        group_size = len(group)

        ax3 = fig.add_subplot(gs[current_row:current_row + group_size, 2])
        ax3.imshow(group_activation_images[group_idx], aspect="auto", extent=[0, 60, group_size, 0])
        ax3.set_xlim(0, 60); ax3.set_ylim(0, group_size)
        ax3.set_xticks([]); ax3.set_yticks([])
        for spine in ax3.spines.values(): spine.set_visible(False)

        ax4 = fig.add_subplot(gs[current_row:current_row + group_size, 3])
        ax4.imshow(group_gradient_images[group_idx], aspect="auto", extent=[0, 60, group_size, 0])
        ax4.set_xlim(0, 60); ax4.set_ylim(0, group_size)
        ax4.set_xticks([]); ax4.set_yticks([])
        for spine in ax4.spines.values(): spine.set_visible(False)

        current_row += group_size
        if group_idx < len(group_infos) - 1:
            spacing_ax = fig.add_subplot(gs[current_row, :]); spacing_ax.axis("off")
            current_row += 1

    if total_rows > 0:
        y = 0.94
        fig.text(0.2, y, "Prompt", fontsize=12, ha="center")
        fig.text(0.49, y, "Transfer", fontsize=12, ha="center")
        fig.text(0.662, y, "Activations[:35]", fontsize=12, ha="center")
        fig.text(0.892, y, "Gradients[:35]", fontsize=12, ha="center")

    return fig


# --- Main ---

if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=pt.bfloat16, device_map=DEVICE,
    )
    model.config.use_cache = False
    for n, p in model.named_parameters():
        p.requires_grad = any(t in n for t in TARGET_MODULES)
    install_hooks(model)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: main capitals analysis
    print("Computing plot 1 (capitals)...")
    group_infos1 = [get_infos(model, tokenizer, g) for g in beginning_ending_pairs_groups1]
    fig1 = create_plot(group_infos1)
    fig1.savefig(SAVE_DIR / "1_capitals.pdf", bbox_inches=None, dpi=300)
    print(f"  Saved to {SAVE_DIR / '1_capitals.pdf'}")

    # Plot 2: extended analysis
    print("Computing plot 2 (extended)...")
    groups2 = beginning_ending_pairs_groups1[:1] + beginning_ending_pairs_groups2
    group_infos2 = [get_infos(model, tokenizer, g) for g in groups2]
    fig2 = create_plot(group_infos2)
    fig2.savefig(SAVE_DIR / "1_capitals_2.pdf", bbox_inches=None, dpi=300)
    print(f"  Saved to {SAVE_DIR / '1_capitals_2.pdf'}")

    print("Done!")
