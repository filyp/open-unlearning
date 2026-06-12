"""
PCA Forget Sequence Interpretability (Interp Exp 3)

For each layer's gate_proj activation PCs, projects forget sequences
onto each PC and returns the top-scoring sequences. No retain set needed.

This answers: "What forget-domain content does each PC encode?"
High-variance PCs should activate on domain-characteristic sequences
(the attacker's subspace); low-variance PCs on atypical/idiosyncratic ones.

Usage:
    python scripts/pca_forget_seq_interp.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepSelect \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no \
        ~trainer.method_args.cfg.lora_rank \
        task_name=PCA_FORGET_SEQ_LLAMA_BIO
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
LAYER_INDICES = None  # None = auto: early, middle, late


def collect_last_hidden(model, tokenizer, dataset, collator, target_module):
    """Forward pass, collect last-token hidden states at target_module's input."""
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)
    captured_acts = []

    def hook(module, args, output):
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


def analyze_forget_sequences(eig_vec, eig_val, forget_hidden, forget_texts, mean):
    """For each selected PC, return top forget sequences by projection magnitude."""
    n_pcs = eig_vec.shape[1]

    high_pc_indices = list(range(min(N_PCS_TO_SHOW, n_pcs)))
    low_pc_indices = list(range(max(0, n_pcs - N_PCS_TO_SHOW), n_pcs))
    pc_indices = sorted(set(high_pc_indices + low_pc_indices))

    pc_details = []
    for pc_i in pc_indices:
        pc = eig_vec[:, pc_i].to(forget_hidden.device)
        mean_dev = mean.to(forget_hidden.device)

        projections = (forget_hidden - mean_dev) @ pc
        top_k = min(TOP_K_SEQUENCES, len(forget_texts))
        top_idx = projections.abs().topk(top_k).indices.tolist()

        pc_details.append({
            "pc_index": pc_i,
            "eigenvalue": eig_val[pc_i].item(),
            "variance_type": "high" if pc_i in high_pc_indices else "low",
            "projection_mean": projections.mean().item(),
            "projection_std": projections.std().item(),
            "top_forget_sequences": [
                {
                    "rank": rank + 1,
                    "score": projections[i].item(),
                    "abs_score": projections[i].abs().item(),
                    "text": forget_texts[i],
                }
                for rank, i in enumerate(top_idx)
            ],
        })

    return {
        "n_pcs": n_pcs,
        "eigenvalue_range": [eig_val[0].item(), eig_val[-1].item()],
        "pc_details": pc_details,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)

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

    # No-op grad collapsers to save memory
    class _NoOpCollapser:
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
        trainer.train()
        for module in model.modules():
            if hasattr(module, "act_collapser") and module.act_collapser.online_cov._count > 0:
                module.act_collapser.process_saved_vecs()
        print("Manually computed PCs after 1 epoch.")
    else:
        import copy
        original_state = copy.deepcopy(model.state_dict())
        trainer.train()
        model.load_state_dict(original_state)
        del original_state
        print("Restored original model weights.")

    # Determine layers
    n_layers = len(model.model.layers)
    max_trained = n_layers - 1
    if LAYER_INDICES is not None:
        layers = LAYER_INDICES
    else:
        layers = sorted(set([0, max_trained // 3, 2 * max_trained // 3, max_trained]))

    print(f"\nLayers to analyze: {layers}")

    # Analyze each layer — forget sequences only
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

        print(f"\n--- Layer {layer_idx} ({module_name}) ---")
        print(f"  Collecting forget hidden states...")
        forget_hidden, forget_texts = collect_last_hidden(
            model, tokenizer, train_data.forget, collator, target_module
        )

        act_col = target_module.act_collapser
        print(f"  Analyzing {act_col.eig_vec.shape[1]} PCs...")
        layer_result = {
            "layer": layer_idx,
            "module": module_name,
            "n_forget_sequences": len(forget_texts),
            "analysis": analyze_forget_sequences(
                act_col.eig_vec.float(), act_col.eig_val.float(),
                forget_hidden, forget_texts, act_col.mean.float(),
            ),
        }
        all_results[f"layer_{layer_idx}"] = layer_result

    # Save
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    task_name = cfg.get("task_name", "default")
    out_path = SAVE_DIR / f"forget_seq_interp_{task_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    for key, lr in all_results.items():
        print(f"\n{'=' * 80}")
        print(f"Layer {lr['layer']} ({lr['module']})")
        for pc in lr["analysis"]["pc_details"]:
            var_label = "HIGH-VAR" if pc["variance_type"] == "high" else "LOW-VAR"
            print(f"\n  [{var_label}] PC{pc['pc_index']} (lambda={pc['eigenvalue']:.4f})")
            for seq in pc["top_forget_sequences"][:5]:
                text_preview = seq["text"][:120].replace("\n", " ")
                print(f"    #{seq['rank']:2d} score={seq['score']:+.3f}  {text_preview}")

    print(f"\nDone! JSON saved to {out_path}")


if __name__ == "__main__":
    main()
