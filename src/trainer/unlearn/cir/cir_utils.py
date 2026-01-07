from itertools import islice

import torch as pt

###################### hooks for caching acts and grads #######################


def save_act_input_hook(module, args, output):
    assert isinstance(args, tuple)
    assert len(args) == 1
    if module.training:
        module.last_act_input = args[0]
    else:
        module.last_act_input = None  # clean automatically


def save_output_hook(module, args, output):
    assert isinstance(output, pt.Tensor)
    if module.training:
        module.cached_out = output
    else:
        module.cached_out = None  # clean automatically


def save_grad_input_hook(module, grad_input, grad_output):
    assert isinstance(grad_input, tuple)
    assert len(grad_input) == 1
    if module.training:
        module.last_grad_input = grad_input[0]
    else:
        module.last_grad_input = None  # clean automatically


def save_grad_output_hook(module, grad_input, grad_output):
    assert isinstance(grad_output, tuple)
    assert len(grad_output) == 1
    if module.training:
        module.last_grad_output = grad_output[0]
    else:
        module.last_grad_output = None  # clean automatically


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
    # print(f"update_norm: {update_norm}")


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


def sanitize_tensor(t, epsilon):
    sign = t.sign()
    sign[sign == 0] = 1
    return t + sign * epsilon


################################ loss functions #################################


def mlp_breaking_loss(model, batch, layer_range):
    # note that it transports the original outputs from RAM
    # which would normally be slow, but if it is called right after model.forward(),
    # it is done in parallel, so causes no slowdown
    _mask = get_token_mask(batch)

    if "org_mlp_out" not in batch:  # first epoch
        batch["org_mlp_out"] = {}

    loss_acc = 0
    for layer_id in range(*layer_range):
        mlp = model.model.layers[layer_id].mlp
        out = mlp.down_proj.cached_out[_mask]

        if layer_id not in batch["org_mlp_out"]:  # first epoch, so cache it
            batch["org_mlp_out"][layer_id] = out.detach().cpu()

        org_out = batch["org_mlp_out"][layer_id].to(out.device)
        org_out_norm = org_out.norm(dim=-1).mean()
        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / org_out_norm**2
        loss_acc += dotproducts.clip(min=0).mean()

    return loss_acc / len(range(*layer_range))


def mlp_activation_breaking_loss(model, batch, layer_range):
    # Similar to mlp_breaking_loss but targets the down_proj input activation
    # (the intermediate MLP activation before the down projection)
    _mask = get_token_mask(batch)

    if "org_down_proj_input" not in batch:  # first epoch
        batch["org_down_proj_input"] = {}

    loss_acc = 0
    for layer_id in range(*layer_range):
        down_proj = model.model.layers[layer_id].mlp.down_proj
        act = down_proj.last_act_input[_mask]

        if layer_id not in batch["org_down_proj_input"]:  # first epoch, so cache it
            batch["org_down_proj_input"][layer_id] = act.detach().cpu()

        org_act = batch["org_down_proj_input"][layer_id].to(act.device)
        org_act_norm = org_act.norm(dim=-1).mean()
        # dotproducts = pt.einsum("ts,ts->t", act, org_act)
        # dotproducts = dotproducts / org_act_norm**2
        # loss_acc += dotproducts.clip(min=0).mean()
        loss_acc += (act * org_act).clip(min=0).mean() / org_act_norm**2

    return loss_acc / len(range(*layer_range))


def neuron_breaking_loss(model, batch, layer_range, output):
    # Similar to mlp_activation_breaking_loss but uses gradients instead of cached activations
    # On first batch, we do an extra backward pass with -output.loss to get gradients on neurons
    _mask = get_token_mask(batch)

    if "org_down_proj_grad" not in batch:  # first epoch
        # Do backward pass with -loss to get gradients that would decrease the loss
        (-output.loss).backward(retain_graph=True)

        batch["org_down_proj_grad"] = {}
        for layer_id in range(*layer_range):
            down_proj = model.model.layers[layer_id].mlp.down_proj
            # last_grad_input is the gradient w.r.t. the input of down_proj
            grad = down_proj.last_grad_input[_mask]
            batch["org_down_proj_grad"][layer_id] = grad.detach().cpu()

        # Zero out the gradients so they don't affect the actual training step
        model.zero_grad()

    loss_acc = 0
    for layer_id in range(*layer_range):
        down_proj = model.model.layers[layer_id].mlp.down_proj
        act = down_proj.last_act_input[_mask]

        org_grad = batch["org_down_proj_grad"][layer_id].to(act.device)
        # activation * org_gradient, clipped and averaged
        loss_acc += (act * org_grad).clip(min=0).mean()

    return loss_acc / len(range(*layer_range))


################################ grad collapse #################################


def get_grad_correction(model, token_mask, grad_collapsers, collapsers_initialized):
    grad_corrections = {}
    for name, module in model.named_modules():
        if not name.endswith(".down_proj"):
            continue
        up_proj = model.get_submodule(name.replace(".down_proj", ".up_proj"))
        if not up_proj.weight.requires_grad:
            continue

        # grad_input == grad_output @ module.weight (from backpropagation)
        grad_input = module.last_grad_input[token_mask].detach().clone()
        grad_output = module.last_grad_output[token_mask].detach().clone()
        assert grad_input.shape == (token_mask.sum(), module.weight.shape[1])
        assert grad_output.shape == (token_mask.sum(), module.weight.shape[0])

        grad_collapsers[name].add_vecs(grad_output)

        if not collapsers_initialized:
            continue  # first epoch, so only collect activations and not train

        out_collapsed = (
            grad_collapsers[name].collapse(grad_output).to(module.weight.dtype)
        )
        in_collapsed = out_collapsed @ module.weight  # backprop
        grad_correction = in_collapsed / sanitize_tensor(grad_input, 1e-6)
        _parent_mlp_name = name.rsplit(".", 1)[0]
        grad_corrections[_parent_mlp_name] = grad_correction
    return grad_corrections


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

