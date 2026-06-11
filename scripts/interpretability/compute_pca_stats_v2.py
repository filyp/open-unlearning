"""
Compute proper PCA stats from forget activations and measure attacker energy.

Replaces collect_pca_stats (which relied on the old collapser interface) with
a direct SVD on forget-set activations. Saves pca_stats compatible with
baseline_weight_projection.py, and outputs attacker top-50 energy per layer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/compute_pca_stats_v2.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default \
        model=Llama-3.1-8B \
        task_name=PCA_STATS_LLAMA8B_BIO
"""
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
from trainer.utils import seed_everything
from interp_utils import SAVE_DIR, collect_activations, load_wikitext, select_layers, save_json

N_PCS = 400
ATTACK_STEPS = 50
ATTACK_LR = 1e-5
TOP_M = 50


def compute_pca_from_acts(acts):
    """SVD on activation matrix → (eig_vec, eig_val, mean)."""
    mean = acts.mean(dim=0)
    centered = acts - mean
    _, S, V = pt.svd_lowrank(centered, q=N_PCS)
    # S^2 / (n-1) = eigenvalues of covariance
    eig_val = (S ** 2) / max(acts.shape[0] - 1, 1)
    return V.float(), eig_val.float(), mean.float()


def energy_in_top_m(delta_W, eig_vec, m):
    """Fraction of ||ΔW||_F^2 projecting onto top-m input-space PCs."""
    proj = delta_W.float() @ eig_vec[:, :m]   # (out, m)
    return (proj ** 2).sum().item() / max((delta_W.float() ** 2).sum().item(), 1e-12)


def simulate_attacker(model, forget_dataset, collator, n_steps, lr):
    """AdamW fine-tuning on forget data (matches paper attack setup). Returns weight delta dict (CPU)."""
    W0 = {n: p.data.clone().cpu() for n, p in model.named_parameters() if "mlp" in n}
    for n, p in model.named_parameters():
        p.requires_grad = True
    optimizer = pt.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(forget_dataset, batch_size=1, collate_fn=collator, shuffle=True)
    model.train()
    step = 0
    while step < n_steps:
        for batch in loader:
            if step >= n_steps:
                break
            optimizer.zero_grad()
            model(**prep_batch(batch, model.device)).loss.backward()
            optimizer.step()
            step += 1
    W1 = {n: p.data.clone().cpu() for n, p in model.named_parameters() if "mlp" in n}
    # restore original weights
    with pt.no_grad():
        for n, p in model.named_parameters():
            if n in W0:
                p.data.copy_(W0[n].to(p.device))
    for p in model.parameters():
        p.requires_grad = False
    return {n: W1[n] - W0[n] for n in W0}


@hydra.main(version_base=None, config_path="../configs", config_name="unlearn.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    task_name = cfg.get("task_name", "PCA_STATS")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = get_model(cfg.model)
    model = model.cuda().eval()

    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)

    layers = select_layers(len(model.model.layers))
    proj_names = ["gate_proj"]   # input-space PCA on gate_proj only

    # ── 1. Compute PCA stats per layer ──
    print("Computing PCA stats from forget activations...")
    pca_stats = {}
    for layer_idx in layers:
        for proj in proj_names:
            module = getattr(model.model.layers[layer_idx].mlp, proj)
            name = f"model.layers.{layer_idx}.mlp.{proj}"
            acts = collect_activations(model, data["train"].forget, collator, module).float()
            eig_vec, eig_val, mean = compute_pca_from_acts(acts)
            pca_stats[name] = {
                "eig_vec": eig_vec.cpu(),
                "eig_val": eig_val.cpu(),
                "mean":    mean.cpu(),
            }
            print(f"  Layer {layer_idx} {proj}: {acts.shape[0]} tokens, {N_PCS} PCs")
            del acts
            pt.cuda.empty_cache()

    pca_path = SAVE_DIR / f"pca_stats_{task_name}.pt"
    pt.save(pca_stats, pca_path)
    print(f"Saved PCA stats → {pca_path}")

    # ── 2. Simulate attacker ──
    print(f"\nSimulating attacker ({ATTACK_STEPS} SGD steps)...")
    model.train()
    attack_deltas = simulate_attacker(model, data["train"].forget, collator, ATTACK_STEPS, ATTACK_LR)
    model.eval()
    pt.cuda.empty_cache()

    # ── 3. Measure attacker energy in top-50 PCs ──
    results = {}
    for name, stats in pca_stats.items():
        layer_str = name.split(".")[2]
        # find matching delta key (parameter name uses weight)
        param_key = name + ".weight"
        delta = attack_deltas.get(param_key)
        if delta is None:
            continue
        frac = energy_in_top_m(delta, stats["eig_vec"], TOP_M)
        results[name] = {"layer": int(layer_str), "attack_top50_frac": frac}
        print(f"  Layer {layer_str}: attacker top-{TOP_M} energy = {frac:.1%}")

    save_json({"task": task_name, "layers": results},
              SAVE_DIR / f"pca_attacker_{task_name}.json")
    print("Done.")


if __name__ == "__main__":
    from data import get_collators
    main()
