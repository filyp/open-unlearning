# %%
import os
import time

import hydra
import matplotlib.pyplot as plt
import torch as pt
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from welford_torch import OnlineCovariance

from data import get_collators, get_data
from evals import get_evaluators
from model import get_model
from trainer.unlearn.cir.cir_utils import (
    batched,
    mlp_breaking_loss,
    prep_batch,
    trainable_modules,
)

# %%
# * initialize hydra config
config_path = os.path.abspath("../configs")
with initialize_config_dir(version_base=None, config_dir=config_path):
    cfg = compose(
        config_name="unlearn.yaml",
        overrides=[
            "experiment=unlearn/wmdp_deduped/default",
            "trainer=CIR",
            "model=Llama-3.2-1B",  # for speed
            "data.custom_loaders.wmdp_bio_deduped.use_dev_split=true",
            "data.custom_loaders.wmdp_bio_deduped.eval_on_all_questions=true",
            "task_name=LAB",
            "trainer.method_args.cfg.target_modules=[gate_proj]",
            "trainer.method_args.cfg.layer_range=[0, 13]",
        ],
    )

# * load the stuff that train.py loads
mode = cfg.get("mode", "train")
template_args = cfg.model.template_args
model, tokenizer = get_model(cfg.model)
model.to("cuda")
data = get_data(cfg.data, mode=mode, tokenizer=tokenizer, template_args=template_args)
collator = get_collators(cfg.collator, tokenizer=tokenizer)
# evaluators = get_evaluators(...

# * set trainable params
mcfg = cfg.trainer.method_args.cfg
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in mcfg.target_modules)

# # * disable layers beyond 11 - useful for MLP breaking loss, when not all layers are used
# for n, p in model.named_parameters():
#     if p.requires_grad:
#         layer_num = int(n.split(".")[2])
#         if layer_num > 11:
#             p.requires_grad = False

# %%
def dotproduct(update1: dict, update2: dict) -> float:
    assert update1.keys() == update2.keys()
    return sum(
        (update1[k].to(pt.float32) * update2[k].to(pt.float32)).sum()
        for k in update1.keys()
    )


# for gate_proj and up_proj, this function is much better at predicting selectivity
# during a real training run, than simple dotproduct (also better than using dim=1 below,
# better than clipping negative values before summing, better than per weight abs,
# better than per module abs)
# see the note github.com/filyp/obsidian_shared/blob/main/unlearning/2025.12.16.md
# it first sums columns, then abs, then sums the rest
# so a low score means that the update "doesn't care" for what happens at each
# individual residual stream position
def dotproduct_abs(update1: dict, update2: dict) -> float:
    assert update1.keys() == update2.keys()
    return sum(
        (update1[k].to(pt.float32) * update2[k].to(pt.float32)).sum(dim=0).abs().sum()
        for k in update1.keys()
    )


def normalize(update: dict) -> None:
    norm = dotproduct(update, update) ** 0.5
    for k in update.keys():
        update[k] /= norm


def get_update_from_batches(batches: list[dict]) -> dict:
    model.zero_grad()
    for batch in batches:
        output = model(**prep_batch(batch, model.device))
        output.loss.backward()
    update = {name: module.weight.grad for name, module in trainable_modules(model)}
    normalize(update)
    return update


def get_precision_matrices_from_online_cov(
    online_covs: dict[str, OnlineCovariance], reg: float = 1e-6
):
    """Compute precision matrices (inverse covariance) and means from OnlineCovariance objects."""
    precisions = {}
    means = {}
    for n, online_cov in online_covs.items():
        pt.cuda.empty_cache()
        if online_cov.mean is None:
            continue

        mean = online_cov.mean.to("cuda").float()
        cov = online_cov.cov.to("cuda").float()

        # Add regularization and invert
        cov_reg = cov + reg * pt.eye(cov.shape[0], device=cov.device)
        precision = pt.linalg.inv(cov_reg)

        precisions[n] = precision.cpu()
        means[n] = mean.cpu()

        del mean, cov, cov_reg, precision

    return precisions, means


# Mahalanobis direction computation
# Given an activation vector x and precision matrix P (inverse covariance) from the forget distribution,
# the Mahalanobis direction is: P @ (x - mu)
# This direction indicates how "unusual" the activation is relative to the forget distribution.
def get_mahalanobis_directions(activations: pt.Tensor, precision: pt.Tensor, mean: pt.Tensor):
    """
    Compute Mahalanobis directions for a batch of activations.

    Args:
        activations: (N, D) tensor of activation vectors
        precision: (D, D) precision matrix (inverse covariance)
        mean: (D,) mean vector

    Returns:
        directions: (N, D) Mahalanobis directions for each activation
    """
    centered = activations - mean  # (N, D)
    directions = centered @ precision.T  # (N, D)
    return directions


def save_output_hook(module, args, output):
    """Hook to cache MLP outputs for mlp_breaking_loss."""
    module.cached_out = output


# ! compute the retain and recall updates, for later reference
# wikitext_update = get_update_from_batches(data["wikitext"][:50])
# bad = -dotproduct(unlearning_update, wikitext_update)
recall_update = get_update_from_batches(data["recall"])

retain_batches = [collator(f) for f in batched(data["train"].retain, 12)]
retain_update = get_update_from_batches(retain_batches[:50])

forget_batches = [collator(f) for f in batched(data["train"].forget, 12)]
# # unlearning update should be negated, because it aims to break forget performance
# unlearning_update = get_update_from_batches(forget_batches)
# for k in unlearning_update.keys():
#     unlearning_update[k] *= -1


def get_ratio(unlearning_update):
    neg_unlearning_update = {k: -v for k, v in unlearning_update.items()}
    # we negate dotproducts, because unlearning_update is negated:
    # values are positive when they improve forget performance
    # while in other updates, values are positive when break performance (increase CE loss)
    bad = dotproduct_abs(neg_unlearning_update, retain_update)
    # bad = dotproduct(neg_unlearning_update, retain_update)
    good = dotproduct(neg_unlearning_update, recall_update)
    ratio = bad / good
    print(f"ratio: {ratio:.3f}, bad: {bad:.3f}, good: {good:.3f}")
    # return ratio, bad, good


# %%
# * install MLP hooks and cache original outputs for mlp_breaking_loss
for layer_id in range(*mcfg.layer_range):
    model.model.layers[layer_id].mlp.register_forward_hook(save_output_hook)
for batch in forget_batches:
    with pt.no_grad():
        output = model(**prep_batch(batch, model.device))
    _mask = batch["attention_mask"].bool().clone()
    _mask[:, : mcfg.cut_off_tokens] = False
    batch["org_mlp_out"] = {}
    batch["org_mlp_out_norm"] = {}
    for layer_id in range(*mcfg.layer_range):
        mlp = model.model.layers[layer_id].mlp
        out = mlp.cached_out.detach()[_mask]
        batch["org_mlp_out"][layer_id] = out.cpu()
        batch["org_mlp_out_norm"][layer_id] = out.float().norm(dim=-1).mean().cpu()


# %% compute the PCA projections
from trainer.unlearn.cir.cir_core import get_last_act, get_last_grad, install_hooks

# Install hooks to capture activations and gradients
install_hooks(model)

start_time = time.time()
# Initialize online covariance objects for each trainable module
acts_online_cov = {n: OnlineCovariance() for n, _ in trainable_modules(model)}

# Go through forget batches and collect activations (online)
for batch in forget_batches:
    model.zero_grad()
    output = model(**prep_batch(batch, model.device))
    # loss = -output.loss
    # loss = mlp_breaking_loss(model, batch, mcfg)
    # loss.backward()

    for name, module in trainable_modules(model):
        acts = get_last_act(module, batch["attention_mask"])
        acts_online_cov[name].add_all(acts.cpu().float())

model.zero_grad()
pt.cuda.empty_cache()

# Compute precision matrices (inverse covariance) and means from online covariance
reg = 1e-2
act_precisions, act_means = get_precision_matrices_from_online_cov(acts_online_cov, reg=reg)
# grad_projections, grad_eigenvalues_dict = get_projections_exact(grads_list)
print(f"Time taken to compute precision matrices: {time.time() - start_time:.2f} seconds")


# %% eval the collapse of PCs
# Number of components to collapse (row 0 = mean, rows 1:n = PCA components)
# act_n_collapse = 20
# grad_n_collapse = 0

# threshold = 1.5
# epsilon = 0.03

# quantile = 0.2

# Accumulate collapsed updates - use module names
collapsed_update = {
    name: pt.zeros_like(module.weight) for name, module in trainable_modules(model)
}

for batch in forget_batches:
    model.zero_grad()
    output = model(**prep_batch(batch, model.device))
    loss = -output.loss
    # loss = mlp_breaking_loss(model, batch, mcfg)
    loss.backward()

    for name, module in trainable_modules(model):
        acts = get_last_act(module, batch["attention_mask"]).to(pt.float32)
        grads = get_last_grad(module, batch["attention_mask"]).to(pt.float32)

        # # ! Compute Mahalanobis directions
        # precision = act_precisions[name].to("cuda")
        # mean = act_means[name].to("cuda")
        # centered = acts - mean
        # mahal_dirs = get_mahalanobis_directions(acts, precision, mean)
        # acts = mahal_dirs

        # # ! cherry-picking PCs
        # act_mean = act_projections[name][0]
        # acts = acts - project_out(acts, act_mean)
        # eigenvectors = act_projections[name][1:]
        # pc_projections = (eigenvectors @ acts.T)
        # eigenvalues = act_eigenvalues_dict[name]
        # mask = pc_projections.abs() / (eigenvalues.reshape(-1,1) + epsilon) > threshold
        # mask[:20, :] = False
        # acts = (pc_projections * mask).T @ eigenvectors

        # # ! top PCs collapsing
        # for comp in act_projections[name][:act_n_collapse]:
        #     acts = acts - project_out(acts, comp)
        # for comp in grad_projections[name][:grad_n_collapse]:
        #     grads = grads - project_out(grads, comp)

        # Reconstruct gradient from collapsed acts/grads
        collapsed_grad = pt.einsum("ti,tj->ij", grads, acts)
        collapsed_update[name] += collapsed_grad

# # Normalize the collapsed update
# normalize(collapsed_update)

# print(f"{act_n_collapse= }   {grad_n_collapse= }   ", end="")
# print(f"{threshold= }   {epsilon= }   ", end="")
print(f"{reg=:.0e}   ", end="")
get_ratio(collapsed_update)

# %%

# %%

# %%

# %% Example usage: get Mahalanobis directions for activations from a batch
name = "model.layers.2.mlp.gate_proj"
module = model.get_submodule(name)

# Get activations from a batch
batch = forget_batches[0]
model.zero_grad()
output = model(**prep_batch(batch, model.device))
acts = get_last_act(module, batch["attention_mask"]).to(pt.float32)

# %%

# Compute Mahalanobis directions
precision = act_precisions[name].to("cuda")
mean = act_means[name].to("cuda")
mahal_dirs = get_mahalanobis_directions(acts, precision, mean)
print(mahal_dirs.norm(dim=1))
mahal_dirs /= mahal_dirs.norm(dim=1, keepdim=True)
proj_strenghts = (mahal_dirs * acts).sum(dim=1, keepdim=True)
acts = proj_strenghts * mahal_dirs

final_dists = get_mahalanobis_directions(acts, precision, mean).norm(dim=1)
quantile = 0.2
thresh_mah_dist = final_dists.quantile(quantile)
mask = final_dists > thresh_mah_dist
acts[mask].shape

# %%
# Get eigenvalues/eigenvectors from online covariance (eig_val is ascending order)
eigenvalues = acts_online_cov[name].eig_val.to("cuda").float()
eigenvectors = acts_online_cov[name].eig_vec.to("cuda").float().T  # transpose to match old format
# Reverse to get descending order (like get_projections_exact)
eigenvalues = eigenvalues.flip(0)
eigenvectors = eigenvectors.flip(0)
pc_projections = (eigenvectors @ acts.T)

# threshold = 1.5
# epsilon = 0  # 0.03
# mask = pc_projections.abs() / (eigenvalues.reshape(-1,1) + epsilon) > threshold

idx = 10
projslice = pc_projections.abs()[:,idx].to("cpu").numpy()
# maskslice = mask[:,idx].to("cpu").numpy()
plt.scatter(
    range(len(projslice)), 
    projslice, 
    # c=["red" if m else "blue" for m in maskslice], 
    c="red",
    s=1
)
plt.plot(eigenvalues.to("cpu").numpy())
plt.xlim(left=0.0, right=900)
plt.ylim(bottom=0.0)
plt.show()


# %%
projslice