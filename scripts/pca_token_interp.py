"""
PCA Token-Level Interpretability: Vocabulary Projection

Projects each PC eigenvector through lm_head to find which tokens
each principal component represents. 

Saves results as JSON for
downstream analysis (e.g. LLM judge interpretation).

Expected:
High-variance PCs → general/shared tokens (the, is, of)
Low-variance PCs  → forget-specific tokens (sarin, toxin, ...)

Usage:
    python scripts/pca_token_interp.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.2-3B trainer=RepCollapse \
        trainer.args.num_train_epochs=1 \
        trainer.args.eval_strategy=no \
        ~trainer.method_args.cfg.lora_rank \
        task_name=PCA_TOKEN_INTERP_Llama
"""

import json
import sys
from pathlib import Path

# add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import hydra
import torch as pt
from omegaconf import DictConfig

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

N_PCS_TO_SHOW = 10   # how many PCs from each end of the spectrum
TOP_K_TOKENS = 20    # how many top tokens per PC
SAVE_DIR = Path(__file__).resolve().parent.parent / "saves" / "pca_interp"


def project_pc_to_vocab(pc_vec, lm_head, tokenizer, top_k):
    """Project a single PC through lm_head and return top-k tokens with logits."""
    logits = lm_head(pc_vec.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype).unsqueeze(0))[0].float()
    top = logits.topk(top_k)
    bot = logits.topk(top_k, largest=False)
    return {
        "top_tokens": [tokenizer.decode(t) for t in top.indices],
        "top_logits": top.values.tolist(),
        "bottom_tokens": [tokenizer.decode(t) for t in bot.indices],
        "bottom_logits": bot.values.tolist(),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)

    # --- 1. Load model & data, build trainer ---
    model, tokenizer = get_model(cfg.model)
    data = get_data(
        cfg.data,
        mode="unlearn",
        tokenizer=tokenizer,
        template_args=cfg.model.template_args,
    )
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    trainer, _ = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=None,
        template_args=cfg.model.template_args,
    )

    # --- 2. Run training to collect PCA statistics ---
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
            if hasattr(module, "grad_collapser") and module.grad_collapser.online_cov._count > 0:
                module.grad_collapser.process_saved_vecs()
        print("Manually computed PCs after 1 epoch (no weight changes).")
    else:
        # With 2+ epochs, PCs are computed automatically but weights get modified.
        import copy
        original_state = copy.deepcopy(model.state_dict())
        trainer.train()
        model.load_state_dict(original_state)
        del original_state
        print("Restored original model weights for analysis.")

    # --- 3. Extract eigenvectors and project through lm_head ---
    lm_head = model.lm_head
    all_results = {}

    for name, module in model.named_modules():
        if not hasattr(module, "act_collapser"):
            continue
        if not hasattr(module.act_collapser, "eig_vec"):
            continue
        # only gate_proj — lives in hidden_state space (same as lm_head)
        if "gate_proj" not in name:
            continue

        collapser = module.act_collapser
        eig_vec = collapser.eig_vec.float()  # (hidden_dim, n_pcs)
        eig_val = collapser.eig_val.float()  # (n_pcs,)
        n_pcs = eig_vec.shape[1]

        module_results = {
            "eigenvalue_range": [eig_val[0].item(), eig_val[-1].item()],
            "all_eigenvalues": eig_val.tolist(),
            "high_variance_pcs": [],
            "low_variance_pcs": [],
        }

        # High-variance PCs
        for i in range(min(N_PCS_TO_SHOW, n_pcs)):
            proj = project_pc_to_vocab(eig_vec[:, i], lm_head, tokenizer, TOP_K_TOKENS)
            entry = {"pc_index": i, "eigenvalue": eig_val[i].item(), **proj}
            module_results["high_variance_pcs"].append(entry)

        # Low-variance PCs
        for i in range(max(0, n_pcs - N_PCS_TO_SHOW), n_pcs):
            proj = project_pc_to_vocab(eig_vec[:, i], lm_head, tokenizer, TOP_K_TOKENS)
            entry = {"pc_index": i, "eigenvalue": eig_val[i].item(), **proj}
            module_results["low_variance_pcs"].append(entry)

        all_results[name] = module_results

    if not all_results:
        print("ERROR: No eigenvectors found. Did training run for >= 2 epochs?")
        return

    # --- 4. Save JSON ---
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    task_name = cfg.get("task_name", "default")
    out_path = SAVE_DIR / f"token_interp_{task_name}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # --- 5. Print summary ---
    for name, res in all_results.items():
        print(f"\n{'=' * 60}")
        print(f"Module: {name}")
        print(f"  Eigenvalue range: {res['eigenvalue_range'][0]:.4f} ... {res['eigenvalue_range'][1]:.6f}")

        print(f"\n  --- HIGH VARIANCE PCs (general/shared) ---")
        for pc in res["high_variance_pcs"]:
            print(f"    PC{pc['pc_index']:3d} (λ={pc['eigenvalue']:10.4f}): {pc['top_tokens'][:10]}")

        print(f"\n  --- LOW VARIANCE PCs (forget-specific) ---")
        for pc in res["low_variance_pcs"]:
            print(f"    PC{pc['pc_index']:3d} (λ={pc['eigenvalue']:10.6f}): {pc['top_tokens'][:10]}")

    print(f"\n{'=' * 60}")
    print(f"Done! JSON saved to {out_path}")
    print("We can now feed this JSON to an LLM judge for interpretation.")


if __name__ == "__main__":
    main()
