"""
Exp 4+6: Attack Subspace Concentration + LoRA Adversary Depth Analysis

Shows that:
1. A fine-tuning attacker's weight deltas concentrate on high-variance forget PCs
2. RepSelect with LoRA pushes modifications deeper into low-variance PCs
3. The LoRA adapter itself targets high-variance PCs

Usage:
    python scripts/attack_subspace.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepSelect \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no \
        ~trainer.method_args.cfg.lora_rank \
        task_name=ATTACK_SUB_LLAMA_BIO
"""

import copy
import json
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
    SAVE_DIR, PLOT_DIR, collect_pca_stats, save_pca_stats,
    project_weight_delta, compute_cumulative_energy,
    get_weight_delta, get_layer_from_module_name, select_layers,
    plot_cumulative_energy_grid, save_json,
)

ATTACK_STEPS = 50
ATTACK_LR = 1e-5


def simulate_attack(model, forget_dataset, collator, n_steps, lr):
    """Run n_steps of CE fine-tuning on forget data. Returns model state after attack."""
    loader = DataLoader(forget_dataset, batch_size=4, collate_fn=collator, shuffle=True)
    optimizer = pt.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            optimizer.zero_grad()
            output = model(**prep_batch(batch, model.device))
            output.loss.backward()
            optimizer.step()
            step += 1
            if step % 10 == 0:
                print(f"    Attack step {step}/{n_steps}, loss={output.loss.item():.4f}")

    return {k: v.cpu() for k, v in model.state_dict().items()}


def run_repselect(model, W0, tokenizer, data, collator, template_args, trainer_cfg_base, n_epochs, with_lora=True):
    """Run RepSelect for n_epochs. Reuses existing model, resets to W0 first."""
    # Reset to original weights
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()}, strict=False)

    trainer_cfg = copy.deepcopy(trainer_cfg_base)
    trainer_cfg.args.num_train_epochs = n_epochs
    trainer_cfg.args.eval_strategy = "no"
    trainer_cfg.args.report_to = "none"

    if not with_lora and "lora_rank" in trainer_cfg.get("method_args", {}).get("cfg", {}):
        OmegaConf.set_struct(trainer_cfg, False)
        del trainer_cfg.method_args.cfg.lora_rank
        if "lora_lr" in trainer_cfg.method_args.cfg:
            del trainer_cfg.method_args.cfg.lora_lr
        OmegaConf.set_struct(trainer_cfg, True)

    trainer, _ = load_trainer(
        trainer_cfg=trainer_cfg, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        processing_class=tokenizer, data_collator=collator,
        evaluators=None, template_args=template_args,
    )
    trainer.train()
    state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Extract LoRA adapter weights if available
    lora_weights = {}
    if with_lora:
        for name, module in model.named_modules():
            if hasattr(module, "lora_module"):
                A = module.lora_module.lora_A.data.float().cpu()
                B = module.lora_module.lora_B.data.float().cpu()
                scaling = module.lora_module.scaling
                lora_weights[name] = (A @ B * scaling).T  # (out_features, in_features)

    del trainer
    pt.cuda.empty_cache()
    return state, lora_weights


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)

    # --- 1. Setup & PCA collection ---
    model, tokenizer = get_model(cfg.model)
    model = model.cuda()
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    # Use no-lora config for PCA collection
    pca_cfg = copy.deepcopy(cfg)
    if "lora_rank" in pca_cfg.trainer.get("method_args", {}).get("cfg", {}):
        OmegaConf.set_struct(pca_cfg, False)
        del pca_cfg.trainer.method_args.cfg.lora_rank
        if "lora_lr" in pca_cfg.trainer.method_args.cfg:
            del pca_cfg.trainer.method_args.cfg.lora_lr
        OmegaConf.set_struct(pca_cfg, True)

    pca_trainer_cfg = copy.deepcopy(pca_cfg.trainer)
    pca_trainer_cfg.args.num_train_epochs = 1
    pca_trainer_cfg.args.eval_strategy = "no"
    pca_trainer_cfg.args.report_to = "none"

    trainer, _ = load_trainer(
        trainer_cfg=pca_trainer_cfg, model=model,
        train_dataset=data.get("train", None), eval_dataset=data.get("eval", None),
        processing_class=tokenizer, data_collator=collator,
        evaluators=None, template_args=cfg.model.template_args,
    )

    # Replace grad collapsers with no-op
    class _NoOp:
        def add_vecs(self, vecs): pass
        def collapse(self, vecs): return vecs
        def process_saved_vecs(self): pass
    for module in model.modules():
        if hasattr(module, "grad_collapser"):
            module.grad_collapser = _NoOp()

    print("=" * 60)
    print("Phase 1: Collecting PCA statistics (1 epoch)...")
    print("=" * 60)
    pca_stats = collect_pca_stats(model, trainer)
    task_name = cfg.get("task_name", "default")
    save_pca_stats(pca_stats, SAVE_DIR / f"pca_stats_{task_name}.pt")

    # Save original weights
    W0 = {k: v.cpu() for k, v in model.state_dict().items()}
    del trainer
    pt.cuda.empty_cache()

    # --- 2. Simulate fine-tuning attack ---
    print("\n" + "=" * 60)
    print("Phase 2: Simulating fine-tuning attack...")
    print("=" * 60)
    # Reset model to original weights for attack simulation
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()}, strict=False)
    W_attack = simulate_attack(model, data["train"].forget, collator, ATTACK_STEPS, ATTACK_LR)
    # Restore original weights
    model.load_state_dict({k: v.to(model.device) for k, v in W0.items()}, strict=False)

    # --- 3. Run RepSelect WITH LoRA (if config has it) ---
    has_lora = "lora_lr" in cfg.trainer.get("method_args", {}).get("cfg", {})
    W_lora, lora_adapters = None, {}
    W_nolora = None

    if has_lora:
        try:
            print("\n" + "=" * 60)
            print("Phase 3a: Running RepSelect WITH LoRA (5 epochs)...")
            print("=" * 60)
            W_lora, lora_adapters = run_repselect(model, W0, tokenizer, data, collator, cfg.model.template_args, cfg.trainer, 5, with_lora=True)

            print("\nPhase 3b: Running RepSelect WITHOUT LoRA (5 epochs)...")
            W_nolora, _ = run_repselect(model, W0, tokenizer, data, collator, cfg.model.template_args, cfg.trainer, 5, with_lora=False)
        except Exception as e:
            print(f"WARNING: LoRA phase failed: {e}. Continuing with attack-only results.")

    # --- 4. Project weight deltas onto PCs ---
    print("\n" + "=" * 60)
    print("Phase 4: Projecting weight deltas onto PCs...")
    print("=" * 60)

    n_layers = len(model.model.layers)
    layers = select_layers(n_layers)
    results = {"task": task_name, "layers": {}}
    cumulative_attack = {}
    cumulative_lora_with = {}
    cumulative_lora_without = {}
    cumulative_lora_adapter = {}

    for module_name, stats in pca_stats.items():
        layer_idx = get_layer_from_module_name(module_name)
        if layer_idx not in layers:
            continue
        if "gate_proj" not in module_name:
            continue

        eig_vec = stats["eig_vec"]
        eig_val = stats["eig_val"]

        # Attack projection
        delta_attack = get_weight_delta(W0, W_attack, module_name)
        if delta_attack is None:
            continue
        energy_attack = project_weight_delta(delta_attack, eig_vec)
        cum_attack = compute_cumulative_energy(energy_attack)

        layer_result = {
            "module": module_name,
            "n_pcs": eig_vec.shape[1],
            "eigenvalue_range": [eig_val[0].item(), eig_val[-1].item()],
            "attack_energy_top10_frac": energy_attack[:10].sum().item() / max(energy_attack.sum().item(), 1e-12),
            "attack_energy_top50_frac": energy_attack[:50].sum().item() / max(energy_attack.sum().item(), 1e-12),
            "attack_cumulative": cum_attack.tolist(),
        }

        cumulative_attack[layer_idx] = cum_attack

        # RepSelect with/without LoRA
        if W_lora is not None:
            delta_lora = get_weight_delta(W0, W_lora, module_name)
            delta_nolora = get_weight_delta(W0, W_nolora, module_name)

            if delta_lora is not None:
                energy_lora = project_weight_delta(delta_lora, eig_vec)
                energy_nolora = project_weight_delta(delta_nolora, eig_vec)
                cum_lora = compute_cumulative_energy(energy_lora)
                cum_nolora = compute_cumulative_energy(energy_nolora)
                cumulative_lora_with[layer_idx] = cum_lora
                cumulative_lora_without[layer_idx] = cum_nolora

                layer_result["repselect_with_lora_top10_frac"] = energy_lora[:10].sum().item() / max(energy_lora.sum().item(), 1e-12)
                layer_result["repselect_without_lora_top10_frac"] = energy_nolora[:10].sum().item() / max(energy_nolora.sum().item(), 1e-12)
                layer_result["repselect_with_lora_cumulative"] = cum_lora.tolist()
                layer_result["repselect_without_lora_cumulative"] = cum_nolora.tolist()

            # LoRA adapter projection
            if module_name in lora_adapters:
                adapter_W = lora_adapters[module_name]
                energy_adapter = project_weight_delta(adapter_W, eig_vec)
                cum_adapter = compute_cumulative_energy(energy_adapter)
                cumulative_lora_adapter[layer_idx] = cum_adapter
                layer_result["lora_adapter_top10_frac"] = energy_adapter[:10].sum().item() / max(energy_adapter.sum().item(), 1e-12)
                layer_result["lora_adapter_cumulative"] = cum_adapter.tolist()

        results["layers"][str(layer_idx)] = layer_result
        print(f"  Layer {layer_idx}: attack top-10 PC energy = {layer_result['attack_energy_top10_frac']:.1%}")

    # --- 5. Save results & plot ---
    save_json(results, SAVE_DIR / f"attack_subspace_{task_name}.json")

    # Plot attack cumulative energy
    plot_data = {"Attack fine-tuning": cumulative_attack}
    if cumulative_lora_with:
        plot_data["RepSelect +LoRA"] = cumulative_lora_with
        plot_data["RepSelect -LoRA"] = cumulative_lora_without
    if cumulative_lora_adapter:
        plot_data["LoRA adapter"] = cumulative_lora_adapter

    # Single figure with all curves per layer (pick middle layer)
    mid_layer = layers[len(layers) // 2]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    styles = {"Attack fine-tuning": ("red", "-"), "RepSelect +LoRA": ("blue", "-"),
              "RepSelect -LoRA": ("blue", "--"), "LoRA adapter": ("orange", "-.")}

    for label, layer_data in plot_data.items():
        if mid_layer in layer_data:
            color, ls = styles.get(label, ("gray", "-"))
            ax.plot(range(len(layer_data[mid_layer])), layer_data[mid_layer],
                    label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_xlabel("PC index (sorted by eigenvalue, high → low)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title(f"Weight update energy on forget PCs (Layer {mid_layer})")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / f"attack_subspace_{task_name}.pdf")
    plt.close(fig)
    print(f"\nSaved plot: {PLOT_DIR / f'attack_subspace_{task_name}.pdf'}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for layer_str, lr in results["layers"].items():
        print(f"\nLayer {layer_str} ({lr['module']}):")
        print(f"  Attack: {lr['attack_energy_top10_frac']:.1%} energy in top-10 PCs, "
              f"{lr['attack_energy_top50_frac']:.1%} in top-50")
        if "repselect_with_lora_top10_frac" in lr:
            print(f"  RepSelect +LoRA: {lr['repselect_with_lora_top10_frac']:.1%} in top-10")
            print(f"  RepSelect -LoRA: {lr['repselect_without_lora_top10_frac']:.1%} in top-10")
        if "lora_adapter_top10_frac" in lr:
            print(f"  LoRA adapter: {lr['lora_adapter_top10_frac']:.1%} in top-10")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()
