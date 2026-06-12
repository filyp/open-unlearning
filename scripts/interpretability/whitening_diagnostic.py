"""Diagnostic: per-dimension forget/retain variance + attack subspace analysis.

Computes online stats (no full activation storage) to understand where whitening fails vs PCA.
"""
import os, sys
import torch as pt
import logging
from collections import defaultdict

os.environ["HF_HOME"] = "/VData/kebl6672/.cache/huggingface"
sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_stats(model, dataset, tokenizer, n_batches=15, batch_size=8):
    """Collect per-dimension mean and var online (no full activation storage)."""
    from data.utils import batched
    counts = defaultdict(int)
    means = {}
    m2s = {}  # for Welford online variance

    hooks = []
    current_acts = {}

    def make_hook(layer_idx, name):
        def hook_fn(module, args, output):
            current_acts[(layer_idx, name)] = args[0].detach().float().reshape(-1, args[0].shape[-1])
        return hook_fn

    for layer_idx, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        hooks.append(mlp.gate_proj.register_forward_hook(make_hook(layer_idx, "gate")))
        hooks.append(mlp.up_proj.register_forward_hook(make_hook(layer_idx, "up")))
        hooks.append(mlp.down_proj.register_forward_hook(make_hook(layer_idx, "down")))

    all_batches = list(batched(dataset, batch_size))[:n_batches]
    for batch in all_batches:
        max_len = max(len(s["input_ids"]) for s in batch)
        input_ids = pt.stack([
            pt.nn.functional.pad(s["input_ids"].clone().detach(), (0, max_len - len(s["input_ids"])), value=tokenizer.pad_token_id)
            if isinstance(s["input_ids"], pt.Tensor) else
            pt.nn.functional.pad(pt.tensor(s["input_ids"]), (0, max_len - len(s["input_ids"])), value=tokenizer.pad_token_id)
            for s in batch
        ])
        attention_mask = pt.stack([
            pt.nn.functional.pad(s["attention_mask"].clone().detach(), (0, max_len - len(s["attention_mask"])), value=0)
            if isinstance(s["attention_mask"], pt.Tensor) else
            pt.nn.functional.pad(pt.tensor(s["attention_mask"]), (0, max_len - len(s["attention_mask"])), value=0)
            for s in batch
        ])
        with pt.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

        # Online Welford update
        for key, acts in current_acts.items():
            n = acts.shape[0]
            if key not in means:
                means[key] = pt.zeros(acts.shape[1], device=acts.device)
                m2s[key] = pt.zeros(acts.shape[1], device=acts.device)
            counts[key] += n
            delta = acts - means[key]
            means[key] += delta.sum(0) / counts[key]
            delta2 = acts - means[key]
            m2s[key] += (delta * delta2).sum(0)
        current_acts.clear()

    for h in hooks:
        h.remove()

    # Convert to std
    stds = {k: (m2s[k] / counts[k]).clamp(min=1e-12).sqrt() for k in means}
    return means, stds


def collect_pca_and_energy(model, dataset, tokenizer, n_batches=10, batch_size=8, n_pcs=50):
    """Collect covariance for one sample layer, compute PCA, measure energy."""
    from data.utils import batched
    from trainer.unlearn.repselect.online_covariance import OnlineCovariance

    target_layer = len(model.model.layers) // 2  # middle layer
    cov_tracker = OnlineCovariance(dtype=pt.bfloat16)
    hooks = []

    def hook_fn(module, args, output):
        acts = args[0].detach().reshape(-1, args[0].shape[-1])
        cov_tracker.add_vecs(acts)

    mlp = model.model.layers[target_layer].mlp
    hooks.append(mlp.gate_proj.register_forward_hook(hook_fn))

    all_batches = list(batched(dataset, batch_size))[:n_batches]
    for batch in all_batches:
        max_len = max(len(s["input_ids"]) for s in batch)
        input_ids = pt.stack([
            pt.nn.functional.pad(s["input_ids"].clone().detach(), (0, max_len - len(s["input_ids"])), value=tokenizer.pad_token_id)
            if isinstance(s["input_ids"], pt.Tensor) else
            pt.nn.functional.pad(pt.tensor(s["input_ids"]), (0, max_len - len(s["input_ids"])), value=0)
            for s in batch
        ])
        attention_mask = pt.stack([
            pt.nn.functional.pad(s["attention_mask"].clone().detach(), (0, max_len - len(s["attention_mask"])), value=0)
            if isinstance(s["attention_mask"], pt.Tensor) else
            pt.nn.functional.pad(pt.tensor(s["attention_mask"]), (0, max_len - len(s["attention_mask"])), value=0)
            for s in batch
        ])
        with pt.no_grad():
            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

    for h in hooks:
        h.remove()

    cov = cov_tracker.get_cov().to(pt.float32)
    cov = (cov + cov.T) / 2
    _, S, V = pt.svd_lowrank(cov, q=n_pcs)
    return V, S, cov_tracker.mean.to(pt.float32), target_layer


def analyze(forget_means, forget_stds, retain_means, retain_stds):
    """Per-dimension diagnostic."""
    layers = sorted(set(k[0] for k in forget_stds.keys()))
    sample_layers = [layers[0], layers[len(layers)//4], layers[len(layers)//2],
                     layers[3*len(layers)//4], layers[-1]]

    print("\n=== PER-DIMENSION DIAGNOSTIC ===\n")
    print(f"{'Layer':>5} {'Mod':>5} | {'σ_f dyn':>8} | {'σ_r dyn':>8} | "
          f"{'ratio med':>9} | {'r>1.5':>5} {'0.8<r<1.5':>10} {'r<0.8':>6}")

    all_ratios = []
    for layer_idx in sample_layers:
        for mod_name in ["gate", "up", "down"]:
            key = (layer_idx, mod_name)
            if key not in forget_stds:
                continue
            sf = forget_stds[key]
            sr = retain_stds[key]
            ratio = sf / sr.clamp(min=1e-6)
            all_ratios.append(ratio)

            n_sel = (ratio > 1.5).sum().item()
            n_shared = ((ratio >= 0.8) & (ratio <= 1.5)).sum().item()
            n_ret = (ratio < 0.8).sum().item()

            print(f"L{layer_idx:>4} {mod_name:>5} | "
                  f"{sf.max()/sf.min():>7.0f}x | "
                  f"{sr.max()/sr.min():>7.0f}x | "
                  f"{ratio.median():>8.2f} | "
                  f"{n_sel:>5} {n_shared:>10} {n_ret:>6}")

    all_r = pt.cat(all_ratios)
    print(f"\n=== SUMMARY ===")
    print(f"Overall median ratio: {all_r.median():.3f}")
    print(f"Fraction selective (>1.5): {(all_r > 1.5).float().mean():.3f}")
    print(f"Fraction shared (0.8-1.5): {((all_r >= 0.8) & (all_r <= 1.5)).float().mean():.3f}")
    print(f"Fraction retain-dom (<0.8): {(all_r < 0.8).float().mean():.3f}")


def analyze_attack_subspace(forget_pca_V, forget_pca_S, forget_mean, forget_stds,
                            retain_stds, target_layer):
    """Compare whitening vs PCA in terms of attack subspace energy."""
    key = (target_layer, "gate")
    sf = forget_stds[key]
    sr = retain_stds[key]
    D = sf.shape[0]
    k = forget_pca_V.shape[1]

    print(f"\n=== ATTACK SUBSPACE ANALYSIS (Layer {target_layer}, gate_proj) ===\n")
    print(f"PCA top-{k} eigenvalue dynamic range: {forget_pca_S.max()/forget_pca_S.min():.0f}x")
    print(f"Per-dim σ_f dynamic range: {sf.max()/sf.min():.0f}x")
    print(f"Per-dim σ_f² dynamic range: {(sf**2).max()/(sf**2).min():.0f}x")

    # How much does whitening project onto PCA's top subspace?
    # If whitening W = diag(1/σ), then the "whitened basis" is just coordinate axes scaled by 1/σ
    # Energy of whitened coordinate axis e_i in PCA subspace V: ||V^T e_i||² / ||e_i||² = sum_j V_ji²
    # Weighted by whitening: w_i = 1/σ_i
    w_standard = 1.0 / sf
    w_power2 = 1.0 / sf**2
    w_ratio = sr / sf

    for name, w in [("standard 1/σ", w_standard), ("power 1/σ²", w_power2), ("ratio σ_r/σ_f", w_ratio)]:
        # For each whitened dim, how much energy goes to PCA top-k?
        # The whitened activation a' = w * (a - μ). The weight update ΔW = g ⊗ a'.
        # Energy in PCA subspace: ||a' @ V||² / ||a'||²
        # For random a ~ N(μ, Σ), E[||w*(a-μ) @ V||²] = sum_j (sum_i w_i² Σ_ii V_ij²)
        # Simplified: per-dim contribution to PCA subspace
        w2 = w ** 2
        var_f = sf ** 2
        # Energy per dim going to PCA subspace
        pca_proj = (forget_pca_V ** 2).sum(dim=1)  # (D,) — how much each dim contributes to PCA subspace
        weighted_in_pca = (w2 * var_f * pca_proj).sum()
        weighted_total = (w2 * var_f).sum()
        frac = (weighted_in_pca / weighted_total).item()
        print(f"  {name:>15}: {frac*100:.1f}% of whitened energy in top-{k} PCA subspace")

    # For reference: PCA Mahalanobis
    # After Mahalanobis, energy is redistributed to low-variance PCs
    # The collapsed activation lives in the ~D-k dimensional complement of top PCs
    print(f"  {'PCA Mahalanobis':>15}: ~{(1-k/D)*100:.0f}% energy in complement of top-{k} (by design)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=pt.bfloat16).cuda().eval()

    from data.custom_loaders import wmdp_low_mi, load_hf_and_tokenize
    from omegaconf import OmegaConf
    tok_cfg = OmegaConf.create({"max_length": 128})

    forget_cfg = OmegaConf.create({"dataset": "bio", 
                                    "num_examples_per_question": 3, "tokenizer": tok_cfg})
    forget_data = wmdp_low_mi(forget_cfg, tokenizer)
    forget_samples = forget_data.get("forget", list(forget_data.values())[0])

    retain_cfg = OmegaConf.create({"dataset_name": "retain", "range": [0, 200], "tokenizer": tok_cfg,
                                    "hf_args": {"path": "m-a-p/FineFineWeb",
                                                "data_files": ["biology/biology_000849.jsonl"]}})
    retain_data = load_hf_and_tokenize(retain_cfg, tokenizer)
    retain_samples = retain_data["retain"]

    print(f"Forget: {len(forget_samples)}, Retain: {len(retain_samples)}")

    # Collect per-dimension stats
    print("\nCollecting forget stats...")
    f_means, f_stds = collect_stats(model, forget_samples, tokenizer)
    print("Collecting retain stats...")
    r_means, r_stds = collect_stats(model, retain_samples, tokenizer)

    analyze(f_means, f_stds, r_means, r_stds)

    # Collect PCA for attack subspace analysis
    print("\nCollecting PCA for attack subspace analysis...")
    V, S, mean, target_layer = collect_pca_and_energy(model, forget_samples, tokenizer, n_pcs=50)
    analyze_attack_subspace(V, S, mean, f_stds, r_stds, target_layer)
