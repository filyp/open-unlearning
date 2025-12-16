# %%
import os
import time

import hydra
import matplotlib.pyplot as plt
import torch as pt
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

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


def save_output_hook(module, args, output):
    """Hook to cache MLP outputs for mlp_breaking_loss."""
    module.cached_out = output


# %% compute the retain and recall updates, for later reference
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


# %%
def get_ratio(unlearning_update):
    neg_unlearning_update = {k: -v for k, v in unlearning_update.items()}
    # we negate dotproducts, because unlearning_update is negated:
    # values are positive when they improve forget performance
    # while in other updates, values are positive when break performance (increase CE loss)
    bad = dotproduct_abs(neg_unlearning_update, retain_update)
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


def get_projections_exact(vector_lists: dict[str, list[pt.Tensor]]):
    # vectors can be either acts or grads
    to_collapse = {}
    eigenvalues_dict = {}
    for n in list(vector_lists.keys()):
        pt.cuda.empty_cache()
        cached_vectors = vector_lists.pop(n)
        if not cached_vectors:
            continue
        v = pt.cat(cached_vectors)
        v = v.to("cuda").float()

        mean = v.mean(axis=0)

        # * compute the PCA components
        # Center the data
        v = v - mean
        # Compute covariance matrix
        cov = (v.T @ v) / (v.shape[0] - 1)
        # Compute eigenvalues and eigenvectors
        # * pt.linalg.eigh seems to leak memory!!
        eigenvalues, eigenvectors = pt.linalg.eigh(cov)
        # Sort in descending order
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pca_components = eigenvectors.T  # [:n_components]

        # return one tensor of mean and the pca components
        to_collapse[n] = pt.cat([mean.reshape(1, -1), pca_components], dim=0)
        eigenvalues_dict[n] = eigenvalues

        del v, mean, cov, eigenvalues, eigenvectors, pca_components

    return to_collapse, eigenvalues_dict


# %% compute the PCA projections
from trainer.unlearn.cir.cir_core import (
    get_last_act,
    get_last_grad,
    install_hooks,
    project_out,
)

# Install hooks to capture activations and gradients
install_hooks(model)

start_time = time.time()
# Initialize lists to accumulate acts/grads for each trainable module
acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}

# Go through forget batches and collect activations/gradients
for batch in forget_batches:
    model.zero_grad()
    output = model(**prep_batch(batch, model.device))
    loss = -output.loss
    # loss = mlp_breaking_loss(model, batch, mcfg)
    loss.backward()

    for name, module in trainable_modules(model):
        acts = get_last_act(module, batch["attention_mask"])
        grads = get_last_grad(module, batch["attention_mask"])
        acts_list[name].append(acts.clone().cpu())
        grads_list[name].append(grads.clone().cpu())

model.zero_grad()
pt.cuda.empty_cache()

# Compute PCA projections (mean + principal components)
act_projections, act_eigenvalues_dict = get_projections_exact(acts_list)
# grad_projections, grad_eigenvalues_dict = get_projections_exact(grads_list)
print(f"Time taken to compute PCA projections: {time.time() - start_time:.2f} seconds")

# %% eval the collapse of PCs
# Number of components to collapse (row 0 = mean, rows 1:n = PCA components)
# act_n_collapse = 20
# grad_n_collapse = 0

threshold = 1.
# epsilon = 0.03
epsilon = 0.0

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

        # collapse the mean
        act_mean = act_projections[name][0]
        acts = acts - project_out(acts, act_mean)

        eigenvectors = act_projections[name][1:]
        pc_projections = (eigenvectors @ acts.T)
        eigenvalues = act_eigenvalues_dict[name]
        mask = pc_projections.abs() / (eigenvalues.reshape(-1,1) + epsilon) > threshold
        mask[:20, :] = False
        acts = (pc_projections * mask).T @ eigenvectors

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
print(f"{threshold= }   {epsilon= }   ", end="")
get_ratio(collapsed_update)

# # %%
# name = "model.layers.8.mlp.gate_proj"
# module = model.get_submodule(name)
# # %%
# act_projections[name].shape
# %%

# %%

# for n, m in trainable_modules(model):
#     # print(eigenvalues_dict[n])

#     eig = act_eigenvalues_dict[n].to("cpu").numpy()

#     plt.plot(eig)
#     plt.xlim(left=0.0)
#     plt.ylim(bottom=0.0)
#     plt.show()

# %%
name = "model.layers.7.mlp.gate_proj"
module = model.get_submodule(name)
acts = get_last_act(module, batch["attention_mask"]).to(pt.float32)

act_mean = act_projections[name][0]
acts = acts - project_out(acts, act_mean)
# %%

eigenvectors = act_projections[name][1:]
pc_projections = (eigenvectors @ acts.T)
eigenvalues = act_eigenvalues_dict[name]

threshold = 1.5
epsilon = 0  # 0.03
mask = pc_projections.abs() / (eigenvalues.reshape(-1,1) + epsilon) > threshold

projslice = pc_projections.abs()[:,8].to("cpu").numpy()
maskslice = mask[:,8].to("cpu").numpy()
plt.scatter(
    range(len(projslice)), 
    projslice, 
    c=["red" if m else "blue" for m in maskslice], 
    s=1
)
plt.plot(eigenvalues.to("cpu").numpy())
plt.xlim(left=0.0, right=900)
plt.ylim(bottom=0.0)
plt.show()
# %%
pc_projections.shape
# %%
eigenvalues.shape
# %%
# (eigenvectors.T @ pc_projections).shape
# %%
acts = (pc_projections * mask).T @ eigenvectors