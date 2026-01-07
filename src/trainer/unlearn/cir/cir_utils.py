from itertools import islice

import torch as pt

import trainer.unlearn.cir.hooks as hooks


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
    # todo: apparently this is quite slow, so maybe compute only at the beginning and store the masks?
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


def normalize_grads(model):
    # L2 norm of weight.grad, computed across all the trainable weights
    update_norm = pt.sqrt(
        sum(
            p.grad.float().norm() ** 2 for p in model.parameters() if p.grad is not None
        )
    )
    # normalize the grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= update_norm


def get_relev_mask_with_caching(batch, name, acts, token_mask, quantile):
    if "relev_mask" not in batch:
        batch["relev_mask"] = {}

    if name in batch["relev_mask"]:
        # we use the caching, because recalculating these can be slow
        relev_mask = batch["relev_mask"][name]
    else:
        dists = acts.norm(dim=1)
        relev_mask = compute_per_text_quantile_mask(dists, token_mask, quantile)
        batch["relev_mask"][name] = relev_mask
    return relev_mask


def _sanitize_tensor(t, epsilon):
    sign = t.sign()
    sign[sign == 0] = 1
    return t + sign * epsilon


def get_grad_correction(model, token_mask, grad_collapsers, after_first_epoch):
    grad_corrections = {}
    for name, module in model.named_modules():
        if not name.endswith(".down_proj"):
            continue
        up_proj = model.get_submodule(name.replace(".down_proj", ".up_proj"))
        if not up_proj.weight.requires_grad:
            continue

        grad_input = module.last_grad_input[token_mask].detach().clone()
        grad_output = module.last_grad_output[token_mask].detach().clone()
        assert grad_input.shape == (token_mask.sum(), module.weight.shape[1])
        assert grad_output.shape == (token_mask.sum(), module.weight.shape[0])

        grad_collapsers[name].add_vecs(grad_output)

        if not after_first_epoch:
            continue  # first epoch, so only collect activations and not train

        out_collapsed = (
            grad_collapsers[name].collapse(grad_output).to(module.weight.dtype)
        )
        in_collapsed = out_collapsed @ module.weight  # backpropagation
        grad_correction = in_collapsed / _sanitize_tensor(grad_input, 1e-6)
        _parent_mlp_name = name.rsplit(".", 1)[0]
        grad_corrections[_parent_mlp_name] = grad_correction
    return grad_corrections


def install_hooks(model, layer_range, forget_loss):
    # hooks for forget loss
    for layer_id in range(*layer_range):
        mlp = model.model.layers[layer_id].mlp
        if forget_loss == "mlp_breaking":
            mlp.down_proj.register_forward_hook(hooks.save_act_output)
        elif forget_loss == "mlp_activation_breaking":
            mlp.down_proj.register_forward_hook(hooks.save_act_input)
        elif forget_loss in ["gate_and_up_breaking", "gate_and_up_breaking_approx"]:
            mlp.gate_proj.register_forward_hook(hooks.save_act_output)
            mlp.up_proj.register_forward_hook(hooks.save_act_output)

    for layer_id in range(layer_range[1] - 1):
        # hooks for component collapse
        mlp = model.model.layers[layer_id].mlp
        for module in [mlp.up_proj, mlp.gate_proj]:
            module.register_forward_hook(hooks.save_act_input)
            module.register_full_backward_hook(hooks.save_grad_output)
        # additional hooks for computing grad collapse more efficiently
        mlp.down_proj.register_full_backward_hook(hooks.save_grad_input)
        mlp.down_proj.register_full_backward_hook(hooks.save_grad_output)


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


# def cb_retain_loss(output, batch, cfg):
#     # _mask = get_token_mask(batch)  # retains only on meaningful tokens
#     _mask = batch["attention_mask"].bool()  # retains also on template on BOS tokens
#     loss_acc = 0
#     for layer_id in cfg.cb_retaining_layers:
#         acts = output.hidden_states[layer_id][_mask].float()
#         org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
#         assert acts.shape == org_acts.shape
#         assert len(acts.shape) == 2
#         avg_act_norm = org_acts.norm(dim=-1).mean()
#         dist = (acts - org_acts).norm(dim=-1).mean() / avg_act_norm
#         loss_acc += dist
#     return loss_acc / len(cfg.cb_retaining_layers)


# def cache_activations_for_cb_retain_loss(model, batches, cfg):
#     for batch in batches:
#         with pt.no_grad():
#             output = model(**prep_batch(batch, model.device), output_hidden_states=True)
#         batch["retain_acts"] = {
#             l_num: output.hidden_states[l_num].detach().to("cpu")
#             for l_num in cfg.cb_retaining_layers
#         }


# def trainable_modules(model):
#     return [
#         (n, m)
#         for n, m in model.named_modules()
#         if "_proj" in n and m.weight.requires_grad
#     ]

# @contextmanager
# def trim_layers(model, max_layer):
#     """Temporarily tell the model to use only the first max_layer layers."""
#     all_layers = model.model.layers
#     model.model.layers = model.model.layers[:max_layer]
#     try:
#         yield
#     finally:
#         model.model.layers = all_layers
