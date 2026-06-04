"""
PCA Sequence-Level Interpretability: Multi-layer Activation PC analysis

For each layer's gate_proj, analyzes activation PCs.
Scores sequences using LAST TOKEN hidden state.
Computes forget-retain steering vector and checks which PCs align with it.

Usage:
    python scripts/pca_sequence_interp.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepCollapse \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no \
        ~trainer.method_args.cfg.lora_rank \
        task_name=PCA_SEQ_INTERP_LLAMA
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data import get_collators, get_data
from data.utils import prep_batch
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

N_PCS_TO_SHOW = 10
TOP_K_SEQUENCES = 20
SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"
# Analyze these layers (None = auto: early, middle, late)
LAYER_INDICES = None


def collect_last_hidden(model, tokenizer, dataset, collator, target_module, capture_output=False):
    """Forward pass, collect last-token hidden states at target_module's input or output.

    Args:
        capture_output: If False (default), capture input to module (hidden_dim space).
                        If True, capture output of module (intermediate_dim space).
    """
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    captured_acts = []

    def hook(module, args, output):
        if capture_output:
            captured_acts.append(output.detach())
        else:
            captured_acts.append(args[0].detach())

    handle = target_module.register_forward_hook(hook)

    all_hidden = []
    all_texts = []

    model.eval()
    with pt.no_grad():
        for batch in loader:
            captured_acts.clear()
            _ = model(**prep_batch(batch, model.device))
            if not captured_acts:
                continue

            acts = captured_acts[0].float()
            mask = batch["attention_mask"].bool().to(acts.device)

            for seq_idx in range(acts.shape[0]):
                last_pos = mask[seq_idx].sum() - 1
                all_hidden.append(acts[seq_idx, last_pos].cpu())

            for seq_idx in range(batch["input_ids"].shape[0]):
                ids = batch["input_ids"][seq_idx]
                mask_i = batch["attention_mask"][seq_idx].bool()
                all_texts.append(tokenizer.decode(ids[mask_i], skip_special_tokens=True))

    handle.remove()
    return pt.stack(all_hidden), all_texts


def analyze_pcs(eig_vec, eig_val, steering_vec_norm, forget_hidden, retain_hidden,
                forget_texts, retain_texts, mean, collapser_type):
    """Analyze one set of PCs (act): steering alignment + F/R ratio + top sequences."""
    n_pcs = eig_vec.shape[1]
    device = eig_vec.device

    high_pc_indices = list(range(min(N_PCS_TO_SHOW, n_pcs)))
    low_pc_indices = list(range(max(0, n_pcs - N_PCS_TO_SHOW), n_pcs))
    pc_indices = sorted(set(high_pc_indices + low_pc_indices))

    # --- Steering alignment for ALL PCs ---
    all_alignment = []
    for i in range(n_pcs):
        pc = eig_vec[:, i]
        cos_sim = (pc @ steering_vec_norm.to(device)).item()
        all_alignment.append({
            "pc_index": i,
            "eigenvalue": eig_val[i].item(),
            "cosine_with_steering": cos_sim,
            "abs_cosine": abs(cos_sim),
        })

    # --- F/R ratio + top sequences for selected PCs ---
    pc_details = []
    for pc_i in pc_indices:
        pc = eig_vec[:, pc_i].to(forget_hidden.device)
        mean_dev = mean.to(forget_hidden.device)

        # Score forget sequences (last token projection)
        f_proj = ((forget_hidden - mean_dev) @ pc)
        r_proj = ((retain_hidden - mean_dev) @ pc)

        f_signed_mean = f_proj.mean().item()
        r_signed_mean = r_proj.mean().item()
        f_abs_mean = f_proj.abs().mean().item()
        r_abs_mean = r_proj.abs().mean().item()
        ratio = f_abs_mean / r_abs_mean if r_abs_mean > 0 else float("inf")
        separation = f_signed_mean - r_signed_mean  # positive = forget is "more" in PC direction

        # Top sequences by absolute projection
        top_k = min(TOP_K_SEQUENCES, len(forget_texts))
        f_top = f_proj.abs().topk(top_k).indices.tolist()
        r_top = r_proj.abs().topk(min(TOP_K_SEQUENCES, len(retain_texts))).indices.tolist()

        pc_details.append({
            "pc_index": pc_i,
            "eigenvalue": eig_val[pc_i].item(),
            "variance_type": "both" if pc_i in high_pc_indices and pc_i in low_pc_indices else ("high" if pc_i in high_pc_indices else "low"),
            "forget_signed_mean": f_signed_mean,
            "retain_signed_mean": r_signed_mean,
            "forget_abs_mean": f_abs_mean,
            "retain_abs_mean": r_abs_mean,
            "forget_to_retain_ratio": ratio,
            "signed_separation": separation,
            "forget_top_sequences": [
                {"score": f_proj[i].item(), "abs_score": f_proj[i].abs().item(), "text": forget_texts[i]} for i in f_top
            ],
            "retain_top_sequences": [
                {"score": r_proj[i].item(), "abs_score": r_proj[i].abs().item(), "text": retain_texts[i]} for i in r_top
            ],
        })

    # Summary stats
    high_align = [a["abs_cosine"] for a in all_alignment if a["pc_index"] < N_PCS_TO_SHOW]
    low_align = [a["abs_cosine"] for a in all_alignment if a["pc_index"] >= n_pcs - N_PCS_TO_SHOW]

    sorted_alignment = sorted(all_alignment, key=lambda x: x["abs_cosine"], reverse=True)

    high_sep = [d["signed_separation"] for d in pc_details if d["pc_index"] < N_PCS_TO_SHOW]
    low_sep = [d["signed_separation"] for d in pc_details if d["pc_index"] >= n_pcs - N_PCS_TO_SHOW]

    return {
        "collapser_type": collapser_type,
        "n_pcs": n_pcs,
        "eigenvalue_range": [eig_val[0].item(), eig_val[-1].item()],
        "steering_alignment": {
            "high_var_mean_abs_cosine": sum(high_align) / len(high_align) if high_align else 0,
            "low_var_mean_abs_cosine": sum(low_align) / len(low_align) if low_align else 0,
            "top_aligned": sorted_alignment[:15],
            "all": all_alignment,
        },
        "signed_separation_summary": {
            "high_var_mean_abs_sep": sum(abs(s) for s in high_sep) / len(high_sep) if high_sep else 0,
            "low_var_mean_abs_sep": sum(abs(s) for s in low_sep) / len(low_sep) if low_sep else 0,
        },
        "pc_details": pc_details,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)

    # --- 1. Setup ---
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

    # Replace grad collapsers with no-op to save memory — this script only uses activation PCs
    class _NoOpCollapser:
        """Dummy collapser that stores nothing and returns input unchanged."""
        def add_vecs(self, vecs): pass
        def collapse(self, vecs): return vecs
        def process_saved_vecs(self): pass
    for module in model.modules():
        if hasattr(module, "grad_collapser"):
            module.grad_collapser = _NoOpCollapser()

    print("=" * 60)
    print("Training to collect PCA statistics...")
    print("=" * 60)

    n_epochs = cfg.trainer.args.num_train_epochs
    if n_epochs <= 1:
        # With 1 epoch, no weight updates happen (only PCA collection).
        # process_saved_vecs() won't fire automatically, so call it manually.
        trainer.train()
        for module in model.modules():
            if hasattr(module, "act_collapser") and module.act_collapser.online_cov._count > 0:
                module.act_collapser.process_saved_vecs()
            if hasattr(module, "grad_collapser") and hasattr(module.grad_collapser, "online_cov") and module.grad_collapser.online_cov._count > 0:
                module.grad_collapser.process_saved_vecs()
        print("Manually computed PCs after 1 epoch (no weight changes).")
    else:
        # With 2+ epochs, PCs are computed automatically but weights get modified.
        # Save and restore original weights.
        import copy
        original_state = copy.deepcopy(model.state_dict())
        trainer.train()
        model.load_state_dict(original_state)
        del original_state
        print("Restored original model weights for scoring.")

    # --- 2. Determine layers to analyze ---
    n_layers = len(model.model.layers)
    max_trained = n_layers - 1  # RepCollapse trains all layers

    if LAYER_INDICES is not None:
        layers = LAYER_INDICES
    else:
        # Early, middle, late within trained range
        layers = sorted(set([0, max_trained // 3, 2 * max_trained // 3, max_trained]))

    print(f"\nLayers to analyze: {layers} (out of {n_layers}, trained: 0-{max_trained})")

    # --- 3. Analyze each layer ---
    train_data = data["train"]
    all_results = {}

    for layer_idx in layers:
        # Find gate_proj — dense: model.layers.X.mlp.gate_proj
        #                  MoE:   model.layers.X.mlp.experts.0.gate_proj (use first expert)
        target_module = None
        module_name = None
        for name, module in model.named_modules():
            if (f"layers.{layer_idx}.mlp" in name
                    and name.endswith("gate_proj")
                    and hasattr(module, "act_collapser")):
                target_module = module
                module_name = name
                break  # take first match (expert 0 for MoE)

        if target_module is None or not hasattr(target_module.act_collapser, "eig_vec"):
            print(f"  Skipping layer {layer_idx}: no PCs found")
            continue

        print(f"\n--- Layer {layer_idx} ({module_name}) ---")

        # Collect last-token hidden states (input space)
        print(f"  Collecting forget hidden states...")
        forget_hidden, forget_texts = collect_last_hidden(
            model, tokenizer, train_data.forget, collator, target_module
        )
        print(f"  Collecting retain hidden states...")
        retain_hidden, retain_texts = collect_last_hidden(
            model, tokenizer, train_data.retain, collator, target_module
        )

        # Steering vector in hidden (input) space
        steering_vec = forget_hidden.mean(0) - retain_hidden.mean(0)
        steering_vec_norm = steering_vec / steering_vec.norm()

        layer_result = {
            "layer": layer_idx,
            "module": module_name,
            "steering_vector_norm": steering_vec.norm().item(),
        }

        # Analyze activation PCs (hidden_dim space)
        act_col = target_module.act_collapser
        if hasattr(act_col, "eig_vec"):
            print(f"  Analyzing activation PCs...")
            layer_result["activation_pcs"] = analyze_pcs(
                act_col.eig_vec.float(), act_col.eig_val.float(),
                steering_vec_norm, forget_hidden, retain_hidden,
                forget_texts, retain_texts, act_col.mean.float(), "activation"
            )

        # NOTE: Gradient PCs live in intermediate_dim space and would need actual
        # backprop gradients (not forward activations) for meaningful analysis.
        # Omitted for now — activation PCs already give strong interpretable results.

        all_results[f"layer_{layer_idx}"] = layer_result

    # --- 4. Save ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    task_name = cfg.get("task_name", "default")
    out_path = SAVE_DIR / f"sequence_interp_{task_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # --- 5. Print summary table ---
    print(f"\n{'=' * 100}")
    print(f"{'Layer':>6} {'Type':>10} {'High |cos|':>12} {'Low |cos|':>12} {'cos H/L':>10} {'High |sep|':>12} {'Low |sep|':>12} {'sep H/L':>10}")
    print(f"{'-' * 100}")
    for key, lr in all_results.items():
        if "activation_pcs" not in lr:
            continue
        sa = lr["activation_pcs"]["steering_alignment"]
        ss = lr["activation_pcs"]["signed_separation_summary"]
        h_cos = sa["high_var_mean_abs_cosine"]
        l_cos = sa["low_var_mean_abs_cosine"]
        cos_ratio = h_cos / l_cos if l_cos > 0 else float("inf")
        h_sep = ss["high_var_mean_abs_sep"]
        l_sep = ss["low_var_mean_abs_sep"]
        sep_ratio = h_sep / l_sep if l_sep > 0 else float("inf")
        print(f"  {lr['layer']:4d} {'act':>10} {h_cos:12.4f} {l_cos:12.4f} {cos_ratio:10.1f} {h_sep:12.4f} {l_sep:12.4f} {sep_ratio:10.1f}")

    print(f"\n{'=' * 100}")
    print("cos H/L: steering alignment ratio (high-var / low-var PCs)")
    print("sep H/L: signed separation ratio (high-var / low-var PCs)")
    print("If ratios << 1: low-var PCs discriminate forget/retain more")
    print(f"\nJSON saved to {out_path}")


if __name__ == "__main__":
    main()
