"""Check PCA tier forget/retain variance ratios for large tiers.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/check_pca_tiers.py \
      --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default \
      model=Llama-3.1-8B model.model_args.attn_implementation=sdpa \
      task_name=pca_tier_check
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch as pt
import hydra
from omegaconf import DictConfig
from model import get_model
from data import get_collators, get_data
from interp_utils import collect_activations
from trainer.utils import seed_everything

TIERS = [("Top 1-300", 0, 300), ("301-500", 300, 500), ("501-650", 500, 650)]

@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(42)
    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    layer_idx = 10
    module = model.model.layers[layer_idx].mlp.gate_proj
    print("Collecting activations...")
    f_acts = collect_activations(model, data["train"].forget, collator, module).float()
    r_acts = collect_activations(model, data["train"].retain, collator, module).float()

    mean_f = f_acts.mean(0)
    mean_r = r_acts.mean(0)
    Sf = ((f_acts - mean_f).T @ (f_acts - mean_f)) / (len(f_acts) - 1)

    print("Running SVD q=650...")
    _, _, V = pt.svd_lowrank(Sf.cuda(), q=650)
    V = V.cpu()

    f_total = (f_acts - mean_f).var(0).sum().item()
    r_total = (r_acts - mean_r).var(0).sum().item()
    print(f"\nforget total var: {f_total:.1f},  retain total var: {r_total:.1f},  retain/forget = {r_total/f_total:.2f}x")

    print(f"\n{'Tier':<14} {'F-frac%':>8} {'R-frac%':>8} {'Frac-ratio':>11} {'Abs-ratio':>10}")
    for name, lo, hi in TIERS:
        pf = (f_acts - mean_f) @ V[:, lo:hi]
        pr = (r_acts - mean_r) @ V[:, lo:hi]
        ff = pf.var(0).sum().item() / f_total
        rf = pr.var(0).sum().item() / r_total
        abs_ratio = (ff * f_total) / (rf * r_total) if rf > 0 else float('inf')
        print(f"{name:<14} {ff*100:>7.1f}% {rf*100:>7.1f}% {ff/rf:>10.2f}x {abs_ratio:>9.2f}x")

if __name__ == "__main__":
    main()
