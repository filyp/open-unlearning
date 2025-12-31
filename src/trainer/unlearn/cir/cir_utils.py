# %%
from itertools import islice

import torch as pt

###################### hooks for caching acts and grads #######################


def _save_act_hook(module, args):
    module.last_act_full = args[0].detach().clone()


def _save_grad_hook(module, args):
    module.last_grad_full = args[0].detach().clone()


def install_act_and_grad_caching_hooks(model):
    for _, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            module.register_forward_pre_hook(_save_act_hook)
            module.register_full_backward_pre_hook(_save_grad_hook)


################################ torch utils #################################


def prep_batch(batch, device):
    return dict(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )


def batched(iterable, n):
    """Batch an iterable into chunks of size n.

    In python>=3.12,  it can be replaced with itertools.batched
    """
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


# @contextmanager
# def trim_layers(model, max_layer):
#     """Temporarily tell the model to use only the first max_layer layers."""
#     all_layers = model.model.layers
#     model.model.layers = model.model.layers[:max_layer]
#     try:
#         yield
#     finally:
#         model.model.layers = all_layers


def get_token_mask(batch):
    token_mask = batch["labels"] != -100
    token_mask[:, 0] = False  # ignore BOS token
    return token_mask


def compute_per_text_quantile_mask(
    dists: pt.Tensor, token_mask: pt.Tensor, quantile: float
) -> pt.Tensor:
    """Compute quantile threshold per text in batch and return relevance mask.

    Args:
        dists: Distance values for each token (1D tensor of length N where N is number of True values in token_mask)
        token_mask: Boolean mask indicating which tokens are valid (as in labels != -100) (2D tensor of shape [batch_size, seq_len])
        quantile: Quantile threshold (0-1) for filtering

    Returns:
        Boolean mask of same length as dists, True for tokens above their text's quantile threshold
    """
    batch_indices = pt.nonzero(token_mask)[:, 0].to(dists.device)
    act_relev_mask = pt.zeros(len(dists), dtype=pt.bool, device=dists.device)
    for text_idx in batch_indices.unique():
        text_mask = batch_indices == text_idx
        text_dists = dists[text_mask]
        threshold = text_dists.quantile(quantile)
        act_relev_mask[text_mask] = text_dists > threshold
    return act_relev_mask


class PreCachingDataLoader:
    def __init__(self, train_dataset, collator, batch_size):
        # * go through whole dataset, to prepare the batches in advance
        self.forget = []
        self.retain = []
        for f, r in zip(
            # prepare separately forget and retain, to support different batch sizes
            batched(train_dataset.forget, batch_size),
            batched(train_dataset.retain, batch_size),
        ):
            self.forget.append(collator(f))
            self.retain.append(collator(r))

    def __iter__(self):
        for fb, rb in zip(self.forget, self.retain):
            yield {"forget": fb, "retain": rb}

    def __len__(self):
        return len(self.forget)


################################ loss functions #################################


def mlp_breaking_loss(model, batch, cfg):
    _mask = get_token_mask(batch)

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        mlp = model.model.layers[layer_id].mlp
        out = mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / mlp.org_mlp_out_norm**2
        loss_acc += dotproducts.clip(min=0).mean()

    return loss_acc / len(range(*cfg.layer_range))


def cb_retain_loss(output, batch, cfg):
    # _mask = get_token_mask(batch)  # retains only on meaningful tokens
    _mask = batch["attention_mask"].bool()  # retains also on template on BOS tokens

    loss_acc = 0
    for layer_id in cfg.cb_retaining_layers:
        acts = output.hidden_states[layer_id][_mask].float()
        org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
        assert acts.shape == org_acts.shape
        assert len(acts.shape) == 2

        avg_act_norm = org_acts.norm(dim=-1).mean()
        dist = (acts - org_acts).norm(dim=-1).mean() / avg_act_norm

        loss_acc += dist**cfg.cb_retaining_pow

    return loss_acc / len(cfg.cb_retaining_layers)


################################ loss helpers #################################


def cache_activations_for_mlp_breaking_loss(model, batches, cfg):
    for layer_id in range(*cfg.layer_range):
        model.model.layers[layer_id].mlp.org_mlp_out_norm = 0

    for batch in batches:
        with pt.no_grad():
            model(**prep_batch(batch, model.device))
        _mask = get_token_mask(batch)
        batch["org_mlp_out"] = {}
        for layer_id in range(*cfg.layer_range):
            mlp = model.model.layers[layer_id].mlp
            out = mlp.cached_out.detach()[_mask]
            batch["org_mlp_out"][layer_id] = out
            mlp.org_mlp_out_norm += out.float().norm(dim=-1).mean().item()

    for layer_id in range(*cfg.layer_range):
        model.model.layers[layer_id].mlp.org_mlp_out_norm /= len(batches)

    for layer_id in range(*cfg.layer_range):
        # # Extract eigendecomposition and apply mahalanobis projection
        # # Compute covariance statistics per layer using all batches
        # if cfg.get("mlp_reg") is not None:
        #     oc = OnlineCovariance(device="cuda")
        #     for batch in batches:
        #         oc.add_all(batch["org_mlp_out"][layer_id].float())
        #     for batch in batches:
        #         out = batch["org_mlp_out"][layer_id].float()
        #         centered = out - oc.mean
        #         projected = centered @ oc.eig_vec
        #         # Scale reg by largest eigenvalue (last one from eigh) to be scale-invariant
        #         _reg = cfg.mlp_reg * oc.eig_val[-1]
        #         mahal_dirs = (projected / (oc.eig_val + _reg)) @ oc.eig_vec.T
        #         mahal_dirs_normal = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
        #         batch["org_mlp_out"][layer_id] = mahal_dirs_normal.to(model.dtype)

        # # MLP output filtering: zero out low-distance outputs per text
        # # can be useful for recall_loss breaking, on late layers
        # if cfg.get("mlp_quantile", 0) > 0:
        #     for batch in batches:
        #         dists = batch["org_mlp_out"][layer_id].float().norm(dim=1)
        #         quantile_mask = compute_per_text_quantile_mask(
        #             dists, get_token_mask(batch), cfg.mlp_quantile
        #         )
        #         # Zero out filtered outputs so they don't contribute to loss
        #         batch["org_mlp_out"][layer_id][~quantile_mask] = 0

        for batch in batches:
            batch["org_mlp_out"][layer_id] = batch["org_mlp_out"][layer_id].cpu()


def cache_activations_for_cb_retain_loss(model, batches, cfg):
    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device), output_hidden_states=True)
        batch["retain_acts"] = {
            l_num: output.hidden_states[l_num].detach().to("cpu")
            for l_num in cfg.cb_retaining_layers
        }


# def get_update_norm(model):
#     """L2 norm of weight.grad, computed across all the trainable weights."""
#     return (
#         sum(
#             m.weight.grad.to(pt.float32).norm() ** 2
#             for _, m in trainable_modules(model)
#             if m.weight.grad is not None
#         )
#         ** 0.5
#     )


# def scale_grads_(model, factor: float):
#     for p in model.parameters():
#         if p.grad is not None:
#             p.grad *= factor


# def PCA_gpu(v):
#     # Center the data
#     v = v - v.mean(axis=0)
#     # Compute covariance matrix
#     cov = (v.T @ v) / (v.shape[0] - 1)
#     # Compute eigenvalues and eigenvectors
#     # * pt.linalg.eigh seems to leak memory!!
#     eigenvalues, eigenvectors = pt.linalg.eigh(cov)
#     # Sort in descending order
#     idx = eigenvalues.argsort(descending=True)
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:, idx]
#     return eigenvectors.T  # [:n_components]


# def trainable_modules(model):
#     return [
#         (n, m)
#         for n, m in model.named_modules()
#         if "_proj" in n and m.weight.requires_grad
#     ]
