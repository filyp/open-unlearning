"""
Exp 7: Cross-Distribution PCA Similarity

For each forget PC, measure its variance contribution on forget, retain, and wikitext.
Shows whether the collapsed subspace is "blind" to retain/wikitext data (Proposition 1).

Usage:
    python scripts/pca_cross_similarity.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepSelect \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no \
        trainer.args.report_to=none \
        ~trainer.method_args.cfg.lora_rank ~trainer.method_args.cfg.lora_lr \
        task_name=PCA_CROSS_SIM_LLAMA_BIO
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"
PLOT_DIR = Path(__file__).resolve().parent.parent / "saves" / "plots"


def collect_activations(model, dataset, collator, target_module):
    """Collect ALL token activations (not just last token) at target_module input."""
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
            # Flatten: keep only non-padding tokens
            for seq_idx in range(acts.shape[0]):
                valid = acts[seq_idx, mask[seq_idx]]  # (seq_len, hidden_dim)
                all_acts.append(valid.cpu())

    handle.remove()
    return pt.cat(all_acts, dim=0)  # (total_tokens, hidden_dim)


def compute_pc_variance(activations, eig_vec, mean):
    """Project activations onto PCs, return per-PC variance."""
    centered = activations.float() - mean.float()
    projections = centered @ eig_vec.float()  # (n_tokens, n_pcs)
    return projections.var(dim=0)  # (n_pcs,)


def plot_cross_variance(results, title, save_path):
    """Plot per-PC variance for forget/retain/wikitext."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3), squeeze=False)

    for col, (layer_name, data) in enumerate(results.items()):
        ax = axes[0, col]
        n_pcs = len(data["forget_variance"])
        x = np.arange(n_pcs)

        # Normalize each distribution's variance by its own total
        for dist_name, color, label in [
            ("forget_variance", "#e74c3c", "Forget"),
            ("retain_variance", "#3498db", "Retain"),
            ("wikitext_variance", "#2ecc71", "Wikitext"),
        ]:
            var = np.array(data[dist_name])
            total = var.sum()
            if total > 0:
                var_frac = var / total
            else:
                var_frac = var
            ax.plot(x, var_frac, label=label, color=color, linewidth=1.2, alpha=0.8)

        ax.set_xlabel("PC index (by forget eigenvalue)")
        ax.set_ylabel("Fraction of variance")
        ax.set_title(f"Layer {data['layer_idx']}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, min(50, n_pcs))
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "default")

    # --- 1. Setup & PCA collection ---
    model, tokenizer = get_model(cfg.model)
    model = model.cuda()
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    trainer, _ = load_trainer(
        trainer_cfg=cfg.trainer, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        processing_class=tokenizer, data_collator=collator,
        evaluators=None, template_args=cfg.model.template_args,
    )

    # No-op grad collapsers
    class _NoOp:
        def add_vecs(self, vecs): pass
        def collapse(self, vecs): return vecs
        def process_saved_vecs(self): pass
    for module in model.modules():
        if hasattr(module, "grad_collapser"):
            module.grad_collapser = _NoOp()

    print("=" * 60)
    print("Phase 1: Collecting PCA statistics...")
    print("=" * 60)
    trainer.train()
    for module in model.modules():
        if hasattr(module, "act_collapser") and module.act_collapser.online_cov._count > 0:
            module.act_collapser.process_saved_vecs()

    # --- 2. Determine layers ---
    n_layers = len(model.model.layers)
    layers = sorted(set([0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]))

    # --- 3. Load wikitext as third distribution ---
    # Retain is already in data["train"].retain, forget in data["train"].forget
    # For wikitext, load the eval split directly
    from datasets import load_dataset
    wikitext_raw = load_dataset("filypo/wikitext_16k", split="train")
    # Tokenize like the other datasets
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    wikitext_ds = wikitext_raw.map(tokenize_fn, batched=True, remove_columns=wikitext_raw.column_names)
    wikitext_ds.set_format("torch", columns=["input_ids", "attention_mask"])

    # --- 4. Analyze each layer ---
    train_data = data["train"]
    all_results = {}

    for layer_idx in layers:
        target_module = None
        module_name = None
        for name, module in model.named_modules():
            if (f"layers.{layer_idx}.mlp" in name
                    and name.endswith("gate_proj")
                    and hasattr(module, "act_collapser")):
                target_module = module
                module_name = name
                break

        if target_module is None or not hasattr(target_module.act_collapser, "eig_vec"):
            print(f"  Skipping layer {layer_idx}: no PCs found")
            continue

        eig_vec = target_module.act_collapser.eig_vec.float().cpu()
        eig_val = target_module.act_collapser.eig_val.float().cpu()
        mean = target_module.act_collapser.mean.float().cpu()

        print(f"\n--- Layer {layer_idx} ({module_name}) ---")

        # Collect activations from each distribution
        print(f"  Collecting forget activations...")
        forget_acts = collect_activations(model, train_data.forget, collator, target_module)
        print(f"  Collecting retain activations...")
        retain_acts = collect_activations(model, train_data.retain, collator, target_module)
        print(f"  Collecting wikitext activations...")
        wikitext_acts = collect_activations(model, wikitext_ds, collator, target_module)

        # Compute per-PC variance on each distribution
        forget_var = compute_pc_variance(forget_acts, eig_vec, mean)
        retain_var = compute_pc_variance(retain_acts, eig_vec, mean)
        wikitext_var = compute_pc_variance(wikitext_acts, eig_vec, mean)

        # Summary: overlap ratio for top-k PCs
        for k in [10, 50]:
            f_frac = forget_var[:k].sum() / forget_var.sum()
            r_frac = retain_var[:k].sum() / retain_var.sum()
            w_frac = wikitext_var[:k].sum() / wikitext_var.sum()
            print(f"  Top-{k} PCs: forget={f_frac:.1%}, retain={r_frac:.1%}, wikitext={w_frac:.1%}")

        all_results[module_name] = {
            "layer_idx": layer_idx,
            "forget_variance": forget_var.tolist(),
            "retain_variance": retain_var.tolist(),
            "wikitext_variance": wikitext_var.tolist(),
            "eigenvalues": eig_val.tolist(),
        }

        del forget_acts, retain_acts, wikitext_acts
        pt.cuda.empty_cache()

    # --- 5. Save ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVE_DIR / f"cross_similarity_{task_name}.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    plot_cross_variance(all_results, f"Cross-Distribution PC Variance ({task_name})",
                        PLOT_DIR / f"cross_similarity_{task_name}.pdf")


if __name__ == "__main__":
    main()
