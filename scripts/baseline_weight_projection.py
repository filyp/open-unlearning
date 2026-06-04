"""
Exp 5: Baseline vs RepCollapse Weight Update Projection

Shows that GradDiff/NPO modify weights along high-variance PCs (attacker's
subspace), while RepCollapse restricts updates to low-variance PCs.

Requires PCA stats from Exp 4 (pca_stats_*.pt files).

Usage:
    python scripts/baseline_weight_projection.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B \
        trainer.args.eval_strategy=no \
        task_name=BASELINE_CMP_LLAMA_BIO
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import get_collators, get_data
from data.utils import prep_batch
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

from interp_utils import (
    SAVE_DIR, PLOT_DIR, load_pca_stats,
    project_weight_delta, compute_cumulative_energy,
    get_weight_delta, get_layer_from_module_name, select_layers,
    plot_energy_bars, save_json,
)

N_UNLEARN_EPOCHS = 3
ATTACK_STEPS = 50
ATTACK_LR = 1e-5


def simulate_attacker(model, W0, forget_dataset, collator):
    """Run fine-tuning attack on forget data. Returns post-attack model state."""
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()})
    loader = DataLoader(forget_dataset, batch_size=4, collate_fn=collator, shuffle=True)
    optimizer = pt.optim.SGD(model.parameters(), lr=1e-4)  # SGD avoids Adam's 2x memory overhead
    model.train()
    step = 0
    for batch in loader:
        if step >= ATTACK_STEPS:
            break
        optimizer.zero_grad()
        output = model(**prep_batch(batch, model.device))
        output.loss.backward()
        optimizer.step()
        step += 1
        if step % 10 == 0:
            print(f"    Attacker step {step}/{ATTACK_STEPS}, loss={output.loss.item():.4f}")
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()})
    pt.cuda.empty_cache()
    return state


def run_method(cfg, method_name, model, W0, tokenizer, data, collator, template_args, n_epochs):
    """Run one unlearning method, return model state after training."""
    # Reset to original weights
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()})

    trainer_cfg = copy.deepcopy(cfg.trainer)
    trainer_cfg.handler = method_name
    trainer_cfg.args.num_train_epochs = n_epochs
    trainer_cfg.args.eval_strategy = "no"
    trainer_cfg.args.report_to = "none"

    # Method-specific config
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
    elif method_name == "UNDIAL":
        trainer_cfg.method_args = OmegaConf.create({
            "gamma": 1.0, "alpha": 0.0, "beta": 10.0, "retain_loss_type": "NLL"
        })
    elif method_name == "RMU":
        trainer_cfg.method_args = OmegaConf.create({
            "gamma": 1.0, "steering_coeff": 2, "retain_loss_type": "EMBED_DIFF",
            "alpha": 1, "module_regex": "model\\.layers\\.7",
            "trainable_params_regex": [".*"],
        })
    elif method_name == "RepCollapse":
        # Use existing RepCollapse config but no LoRA for fair comparison
        if "lora_rank" in trainer_cfg.get("method_args", {}).get("cfg", {}):
            OmegaConf.set_struct(trainer_cfg, False)
            del trainer_cfg.method_args.cfg.lora_rank
            if "lora_lr" in trainer_cfg.method_args.cfg:
                del trainer_cfg.method_args.cfg.lora_lr
            OmegaConf.set_struct(trainer_cfg, True)
    elif method_name == "RepSelectSimple":
        OmegaConf.set_struct(trainer_cfg, False)
        trainer_cfg.method_args = OmegaConf.create({
            "n_pcs": 400,
            "lora_lr": 0.01,
            "distribution": "forget",
            "collapse_on": "both",
            "use_lora": True,
            "hard_soft": "soft",
        })
        OmegaConf.set_struct(trainer_cfg, True)

    trainer, _ = load_trainer(
        trainer_cfg=trainer_cfg, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        processing_class=tokenizer, data_collator=collator,
        evaluators=None, template_args=template_args,
    )

    print(f"  Training {method_name} for {n_epochs} epochs...")
    trainer.train()
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    del trainer
    pt.cuda.empty_cache()
    return state


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "default")

    # --- 1. Load PCA stats ---
    # Try to find matching PCA stats file
    pca_candidates = list(SAVE_DIR.glob("pca_stats_*.pt"))
    if not pca_candidates:
        print("ERROR: No PCA stats found. Run attack_subspace.py first.")
        return

    # Pick matching model — match "LLAMA" or "QWEN" from task_name
    matched = [p for p in pca_candidates if task_name.split("_")[-2] in p.stem.upper()] if "_" in task_name else []
    pca_path = sorted(matched or pca_candidates, key=lambda p: p.stat().st_mtime)[-1]
    print(f"Loading PCA stats from: {pca_path}")
    pca_stats = load_pca_stats(pca_path)

    # --- 2. Get original model weights ---
    model, tokenizer = get_model(cfg.model)
    model = model.cuda()
    W0 = {k: v.cpu() for k, v in model.state_dict().items()}
    n_layers = len(model.model.layers)
    layers = select_layers(n_layers)

    # --- 3. Load data ---
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    # --- 4. Run each method ---
    methods = ["GradDiff", "NPO", "SimNPO", "RMU", "UNDIAL", "RepCollapse", "RepSelectSimple"]
    method_states = {}

    # Fine-tuning attacker (not a trainer, runs directly)
    print(f"\n{'=' * 60}")
    print("Running fine-tuning attacker (50 steps)...")
    print(f"{'=' * 60}")
    method_states["Attacker"] = simulate_attacker(model, W0, data["train"].forget, collator)

    for method in methods:
        print(f"\n{'=' * 60}")
        print(f"Running {method}...")
        print(f"{'=' * 60}")
        try:
            method_states[method] = run_method(
                cfg, method, model, W0, tokenizer, data, collator, cfg.model.template_args, N_UNLEARN_EPOCHS
            )
        except Exception as e:
            print(f"  WARNING: {method} failed: {e}")
            continue

    # --- 5. Project weight deltas ---
    print(f"\n{'=' * 60}")
    print("Projecting weight deltas onto PCs...")
    print(f"{'=' * 60}")

    results = {"task": task_name, "methods": {}}
    # Aggregate energy across layers for bar plot
    method_total_energy = {m: pt.zeros(400) for m in method_states}

    for module_name, stats in pca_stats.items():
        layer_idx = get_layer_from_module_name(module_name)
        if layer_idx not in layers:
            continue
        if "gate_proj" not in module_name:
            continue

        eig_vec = stats["eig_vec"]

        for method, W_method in method_states.items():
            delta = get_weight_delta(W0, W_method, module_name)
            if delta is None:
                continue
            energy = project_weight_delta(delta, eig_vec)
            k = min(len(energy), 400)
            method_total_energy[method][:k] += energy[:k]

            cum = compute_cumulative_energy(energy)
            key = f"{method}_layer_{layer_idx}"
            results["methods"][key] = {
                "method": method,
                "layer": layer_idx,
                "module": module_name,
                "top10_frac": energy[:10].sum().item() / max(energy.sum().item(), 1e-12),
                "top50_frac": energy[:50].sum().item() / max(energy.sum().item(), 1e-12),
                "cumulative": cum.tolist(),
            }
            print(f"  {method} Layer {layer_idx}: {energy[:10].sum().item() / max(energy.sum().item(), 1e-12):.1%} in top-10 PCs")

    # --- 6. Save & plot ---
    save_json(results, SAVE_DIR / f"baseline_cmp_{task_name}.json")

    # Energy bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc_bins = [(0, 10, "PC 1-10"), (10, 50, "PC 11-50"), (50, 100, "PC 51-100"), (100, 400, "PC 101-400")]
    plot_energy_bars(method_total_energy, pc_bins,
                     f"Weight update energy distribution ({task_name})",
                     PLOT_DIR / f"baseline_cmp_{task_name}.pdf")

    # Cumulative energy comparison (middle layer)
    mid_layer = layers[len(layers) // 2]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    colors = {"GradDiff": "#e74c3c", "NPO": "#f39c12", "RepCollapse": "#2ecc71"}

    for method in methods:
        key = f"{method}_layer_{mid_layer}"
        if key in results["methods"]:
            cum = results["methods"][key]["cumulative"]
            ax.plot(range(len(cum)), cum, label=method,
                    color=colors.get(method, "gray"), linewidth=1.5)

    ax.set_xlabel("PC index (sorted by eigenvalue, high → low)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title(f"Unlearning update energy on forget PCs (Layer {mid_layer})")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"baseline_cumulative_{task_name}.pdf")
    plt.close(fig)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    for key, r in results["methods"].items():
        print(f"  {r['method']} Layer {r['layer']}: top-10={r['top10_frac']:.1%}, top-50={r['top50_frac']:.1%}")


if __name__ == "__main__":
    main()
