"""Compute forget/retain variance ratio along PCA directions, tiered by PCA rank.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/compute_pca_tiered_ratio.py \
    --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=Llama-3.2-3B task_name=pca_tiered
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch as pt
import hydra
from omegaconf import DictConfig
sys.path.insert(0, str(Path(__file__).resolve().parent))
from interp_utils import collect_activations
from model import get_model
from data import get_collators, get_data
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)

    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()

    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    train_data = data["train"]

    n_layers = len(model.model.layers)
    layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    for layer_idx in layers:
        module = model.model.layers[layer_idx].mlp.gate_proj
        print(f"\n{'='*50}\nLayer {layer_idx} gate_proj\n{'='*50}")

        f_acts = collect_activations(model, train_data.forget, collator, module).float()
        r_acts = collect_activations(model, train_data.retain, collator, module).float()
        print(f"  forget: {f_acts.shape[0]} tokens, retain: {r_acts.shape[0]} tokens, dim={f_acts.shape[1]}")

        Sf = pt.cov(f_acts.T)
        Sr = pt.cov(r_acts.T)

        n_pcs = min(400, f_acts.shape[0] - 1, f_acts.shape[1])
        _, S, V = pt.svd_lowrank(Sf, q=n_pcs)
        fv = (V.T @ Sf @ V).diagonal()
        rv = (V.T @ Sr @ V).diagonal().clamp(min=1e-4)
        ratio = fv / rv

        # Spearman
        pca_rank = pt.arange(n_pcs)
        ratio_rank = ratio.argsort(descending=True).argsort()
        d = (pca_rank.float() - ratio_rank.float())
        n = len(ratio)
        spearman = 1 - 6 * (d ** 2).sum().item() / (n * (n ** 2 - 1))
        cv = rv.std().item() / rv.mean().item()

        print(f"  Spearman rho = {spearman:.4f}, Retain CV = {cv:.3f}")
        print(f"  {'Tier':<12} {'Ratio':>8} {'Forget var':>12} {'Retain var':>12}")
        for lo, hi in [(0, 3), (3, 10), (10, 30), (30, 100), (100, 300), (300, n_pcs)]:
            if hi > n_pcs: break
            r = ratio[lo:hi].mean().item()
            f = fv[lo:hi].mean().item()
            rv_tier = rv[lo:hi].mean().item()
            print(f"  {lo}-{hi:<8} {r:>7.1f}x {f:>12.0f} {rv_tier:>12.0f}")


if __name__ == "__main__":
    main()
