"""
Exp 8: Per-Token Loss Comparison

After unlearning with each method, compare per-token loss on retain data vs original model.
Shows that baselines break generic domain tokens more than RepSelect.

Usage:
    python scripts/per_token_loss_comparison.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepSelect \
        trainer.args.eval_strategy=no \
        trainer.args.report_to=none \
        task_name=TOKEN_LOSS_LLAMA_BIO
"""

import copy
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import get_collators, get_data
from data.utils import prep_batch
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"
PLOT_DIR = Path(__file__).resolve().parent.parent / "saves" / "plots"
N_UNLEARN_EPOCHS = 3


def compute_per_token_loss(model, dataset, collator, tokenizer):
    """Compute per-token cross-entropy loss on dataset. Returns dict: token_id -> list of losses."""
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    token_losses = defaultdict(list)

    model.eval()
    with pt.no_grad():
        for batch in loader:
            batch = prep_batch(batch, model.device)
            outputs = model(**batch)
            logits = outputs.logits  # (B, T, V)

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["input_ids"][:, 1:].contiguous()
            shift_mask = batch["attention_mask"][:, 1:].bool()

            # Per-token CE loss
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            # Accumulate per token_id
            for b in range(shift_labels.shape[0]):
                for t in range(shift_labels.shape[1]):
                    if shift_mask[b, t]:
                        tid = shift_labels[b, t].item()
                        token_losses[tid].append(loss_per_token[b, t].item())

    # Average per token
    return {tid: np.mean(losses) for tid, losses in token_losses.items() if len(losses) >= 3}


def run_unlearn_method(cfg, method_name, model, W0, tokenizer, data, collator):
    """Run one unlearning method, return model (with modified weights)."""
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()}, strict=False)

    trainer_cfg = copy.deepcopy(cfg.trainer)
    trainer_cfg.handler = method_name
    trainer_cfg.args.num_train_epochs = N_UNLEARN_EPOCHS
    trainer_cfg.args.eval_strategy = "no"
    trainer_cfg.args.report_to = "none"

    if method_name == "GradDiff":
        trainer_cfg.method_args = OmegaConf.create({
            "gamma": 1.0, "alpha": 1.0, "retain_loss_type": "NLL"
        })
    elif method_name == "NPO":
        trainer_cfg.method_args = OmegaConf.create({
            "beta": 0.1, "gamma": 1.0, "alpha": 1.0, "retain_loss_type": "NLL"
        })
    elif method_name == "SimNPO":
        trainer_cfg.method_args = OmegaConf.create({
            "delta": 0.0, "beta": 4.5, "alpha": 1.0, "gamma": 0.125, "retain_loss_type": "NLL"
        })
    elif method_name == "RMU":
        trainer_cfg.method_args = OmegaConf.create({
            "gamma": 1.0, "steering_coeff": 2, "retain_loss_type": "EMBED_DIFF",
            "alpha": 1, "module_regex": "model\\.layers\\.7",
            "trainable_params_regex": [".*"],
        })
    elif method_name == "UNDIAL":
        trainer_cfg.method_args = OmegaConf.create({
            "gamma": 1.0, "alpha": 0.0, "beta": 10.0, "retain_loss_type": "NLL"
        })
    elif method_name == "RepSelect":
        if "lora_rank" in trainer_cfg.get("method_args", {}).get("cfg", {}):
            OmegaConf.set_struct(trainer_cfg, False)
            del trainer_cfg.method_args.cfg.lora_rank
            if "lora_lr" in trainer_cfg.method_args.cfg:
                del trainer_cfg.method_args.cfg.lora_lr
            OmegaConf.set_struct(trainer_cfg, True)

    trainer, _ = load_trainer(
        trainer_cfg=trainer_cfg, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        processing_class=tokenizer, data_collator=collator,
        evaluators=None, template_args=cfg.model.template_args,
    )
    print(f"  Training {method_name} for {N_UNLEARN_EPOCHS} epochs...")
    trainer.train()
    del trainer
    pt.cuda.empty_cache()


def plot_token_loss_increase(method_results, tokenizer, title, save_path, top_n=30):
    """Bar chart: tokens with largest loss increase per method."""
    fig, axes = plt.subplots(1, len(method_results), figsize=(5 * len(method_results), 4), squeeze=False)
    colors = {"GradDiff": "#e74c3c", "NPO": "#3498db", "RepSelect": "#2ecc71"}

    for col, (method, deltas) in enumerate(method_results.items()):
        ax = axes[0, col]
        # Sort by loss increase
        sorted_tokens = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:top_n]
        token_ids = [t[0] for t in sorted_tokens]
        increases = [t[1] for t in sorted_tokens]
        labels = [tokenizer.decode([tid]).strip() or f"[{tid}]" for tid in token_ids]

        ax.barh(range(len(labels)), increases, color=colors.get(method, "#999"))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Loss increase (vs original)")
        ax.set_title(method)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

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

    # --- 1. Setup ---
    model, tokenizer = get_model(cfg.model)
    model = model.cuda()
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    W0 = {k: v.cpu() for k, v in model.state_dict().items()}

    # --- 2. Baseline per-token loss on retain data ---
    print("=" * 60)
    print("Computing original model per-token loss on retain data...")
    print("=" * 60)
    original_losses = compute_per_token_loss(model, data["train"].retain, collator, tokenizer)
    print(f"  Tracked {len(original_losses)} unique tokens")

    # --- 3. Run each method and compute loss ---
    methods = ["GradDiff", "NPO", "SimNPO", "RMU", "UNDIAL", "RepSelect"]
    method_deltas = {}  # method -> {token_id: loss_increase}

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Running {method}...")
        print("=" * 60)
        try:
            run_unlearn_method(cfg, method, model, W0, tokenizer, data, collator)
            method_losses = compute_per_token_loss(model, data["train"].retain, collator, tokenizer)

            # Compute delta
            deltas = {}
            for tid in original_losses:
                if tid in method_losses:
                    deltas[tid] = method_losses[tid] - original_losses[tid]
            method_deltas[method] = deltas

            # Summary
            increases = [d for d in deltas.values() if d > 0]
            print(f"  {len(increases)}/{len(deltas)} tokens have increased loss")
            top5 = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:5]
            for tid, delta in top5:
                tok_str = tokenizer.decode([tid]).strip()
                print(f"    '{tok_str}' (id={tid}): +{delta:.3f}")

        except Exception as e:
            print(f"  WARNING: {method} failed: {e}")

    # --- 4. Save ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for method, deltas in method_deltas.items():
        # Convert token_ids to strings for JSON
        top_increased = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:100]
        top_decreased = sorted(deltas.items(), key=lambda x: x[1])[:100]
        results[method] = {
            "n_tokens": len(deltas),
            "mean_delta": np.mean(list(deltas.values())),
            "median_delta": np.median(list(deltas.values())),
            "n_increased": sum(1 for d in deltas.values() if d > 0),
            "top_increased": [
                {"token": tokenizer.decode([tid]).strip(), "token_id": tid, "delta": d}
                for tid, d in top_increased
            ],
            "top_decreased": [
                {"token": tokenizer.decode([tid]).strip(), "token_id": tid, "delta": d}
                for tid, d in top_decreased
            ],
        }

    save_path = SAVE_DIR / f"token_loss_{task_name}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    # --- 5. Plot ---
    plot_token_loss_increase(method_deltas, tokenizer,
                             f"Per-Token Loss Increase on Retain ({task_name})",
                             PLOT_DIR / f"token_loss_{task_name}.pdf")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, r in results.items():
        print(f"  {method}: mean_delta={r['mean_delta']:.4f}, "
              f"median={r['median_delta']:.4f}, "
              f"increased={r['n_increased']}/{r['n_tokens']}")


if __name__ == "__main__":
    main()
