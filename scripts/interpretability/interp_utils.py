"""Shared utilities for PCA interpretability experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as pt

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"
PLOT_DIR = Path(__file__).resolve().parent.parent / "saves" / "plots"

# Paper-ready matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def collect_pca_stats(model, trainer):
    """Run 1 epoch of PCA collection (no weight updates). Returns dict of stats per module."""
    trainer.train()
    stats = {}
    for name, module in model.named_modules():
        if not hasattr(module, "act_collapser"):
            continue
        col = module.act_collapser
        if col.online_cov._count > 0:
            col.process_saved_vecs()
        if not hasattr(col, "eig_vec"):
            continue
        stats[name] = {
            "eig_vec": col.eig_vec.float().cpu(),
            "eig_val": col.eig_val.float().cpu(),
            "mean": col.mean.float().cpu(),
        }
    return stats


def save_pca_stats(stats, path):
    """Save PCA stats dict as .pt file."""
    pt.save(stats, path)


def load_pca_stats(path):
    """Load PCA stats dict from .pt file."""
    return pt.load(path, map_location="cpu", weights_only=True)


def project_weight_delta(delta_W, eig_vec):
    """Project weight delta onto activation PCs.

    Args:
        delta_W: (out_features, in_features) weight change
        eig_vec: (in_features, k) PC eigenvectors

    Returns:
        energy: (k,) energy per PC = ||delta_W @ v_i||^2
    """
    proj = delta_W.float() @ eig_vec.float()  # (out_features, k)
    energy = (proj ** 2).sum(dim=0)  # (k,)
    return energy


def compute_cumulative_energy(energy):
    """Normalize energy and compute cumulative sum."""
    total = energy.sum()
    if total == 0:
        return np.zeros(len(energy))
    normed = (energy / total).numpy()
    return np.cumsum(normed)


def get_weight_delta(model_state_before, model_state_after, module_name):
    """Get weight delta for a specific module."""
    key = module_name + ".weight"
    if model_state_before is None or model_state_after is None:
        return None
    if key not in model_state_before or key not in model_state_after:
        return None
    return (model_state_after[key].float().cpu() - model_state_before[key].float().cpu())


def get_layer_from_module_name(name):
    """Extract layer index from module name like 'model.layers.5.mlp.gate_proj'."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    return -1


def select_layers(n_layers):
    """Pick early, middle, late layers."""
    return sorted(set([0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]))


def plot_cumulative_energy_grid(results, title, save_path):
    """Plot 2x2 grid of cumulative energy curves.

    results: dict of {label: {layer_idx: cumulative_energy_array}}
    """
    fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 3), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

    for col, (label, layer_data) in enumerate(results.items()):
        ax = axes[0, col]
        for i, (layer_idx, cum_energy) in enumerate(sorted(layer_data.items())):
            ax.plot(range(len(cum_energy)), cum_energy,
                    label=f"Layer {layer_idx}", color=colors[i], linewidth=1.5)
        ax.set_xlabel("PC index (sorted by eigenvalue)")
        ax.set_ylabel("Cumulative energy fraction")
        ax.set_title(label)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_energy_bars(method_energies, pc_bins, title, save_path):
    """Plot grouped bar chart of energy in PC bins per method.

    method_energies: dict of {method_name: energy_array (k,)}
    pc_bins: list of (start, end, label) tuples
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(pc_bins))
    width = 0.8 / len(method_energies)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, (method, energy) in enumerate(method_energies.items()):
        total = energy.sum()
        if total == 0:
            continue
        bin_fracs = []
        for start, end, _ in pc_bins:
            bin_fracs.append(energy[start:end].sum().item() / max(total.item(), 1e-12))
        ax.bar(x + i * width, bin_fracs, width, label=method, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(method_energies) - 1) / 2)
    ax.set_xticklabels([label for _, _, label in pc_bins])
    ax.set_ylabel("Fraction of weight update energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def save_json(data, path):
    """Save dict as JSON."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


# --- Activation collection ---

def collect_activations(model, dataset, collator, target_module):
    """Collect ALL token activations at target_module input. Forward only, no grad."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    captured = []

    def hook(module, args, output):
        captured.append(args[0].detach())

    handle = target_module.register_forward_hook(hook)
    all_acts = []

    model.eval()
    with pt.no_grad():
        for batch in loader:
            captured.clear()
            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device),
            }
            if "labels" in batch:
                inputs["labels"] = batch["labels"].to(model.device)
            _ = model(**inputs)
            if not captured:
                continue
            acts = captured[0].float()
            mask = batch["attention_mask"].bool().to(acts.device)
            for seq_idx in range(acts.shape[0]):
                valid = acts[seq_idx, mask[seq_idx]]
                all_acts.append(valid.cpu())

    handle.remove()
    return pt.cat(all_acts, dim=0)


def disable_grad_collapsers(model):
    """Replace grad collapsers with no-ops (for PCA-only collection)."""
    class _NoOp:
        def add_vecs(self, vecs): pass
        def collapse(self, vecs): return vecs
        def process_saved_vecs(self): pass
    for module in model.modules():
        if hasattr(module, "grad_collapser"):
            module.grad_collapser = _NoOp()


def load_wikitext(tokenizer, max_samples=2000):
    """Load and tokenize wikitext for cross-distribution analysis."""
    from datasets import load_dataset
    raw = load_dataset("filypo/wikitext_16k", split="train")
    raw = raw.select(range(min(max_samples, len(raw))))
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    ds = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names)
    ds.set_format("torch", columns=["input_ids", "attention_mask"])
    return ds


# --- Generalised eigenvalue (DISCO) ---

def solve_generalised_eigenvalue(Sigma_f, Sigma_r, reg_eps=1e-4):
    """Solve Sigma_f v = lambda Sigma_r v via Cholesky whitening. Returns descending."""
    D = Sigma_f.shape[0]
    device = Sigma_f.device
    Sigma_r_reg = Sigma_r + reg_eps * pt.eye(D, device=device)
    L = pt.linalg.cholesky(Sigma_r_reg)
    L_inv = pt.linalg.inv(L)
    M = L_inv @ Sigma_f @ L_inv.T
    eigenvalues, U = pt.linalg.eigh(M)
    eigenvectors = pt.linalg.solve_triangular(L.T, U, upper=True)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    eigenvectors = eigenvectors / eigenvectors.norm(dim=0, keepdim=True)
    return eigenvalues, eigenvectors


def compute_tiered_variance(acts, directions, mean, tiers):
    """Compute variance fraction explained by direction tiers on a dataset."""
    centered = acts - mean
    projections = centered @ directions
    per_dir_var = projections.var(dim=0)
    total_var = centered.var(dim=0).sum()
    results = {}
    for name, (start, end) in tiers.items():
        end = min(end, per_dir_var.shape[0])
        if start >= per_dir_var.shape[0]:
            results[name] = 0.0
            continue
        results[name] = (per_dir_var[start:end].sum() / total_var).item()
    return results
