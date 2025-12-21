# %%
from contextlib import contextmanager
from itertools import islice
import torch as pt

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


@contextmanager
def trim_layers(model, max_layer):
    """Temporarily tell the model to use only the first max_layer layers."""            
    all_layers = model.model.layers
    model.model.layers = model.model.layers[:max_layer]
    try:
        yield
    finally:
        model.model.layers = all_layers


def get_token_mask(batch):
    token_mask = batch["labels"] != -100
    token_mask[:, 0] = False  # ignore BOS token
    return token_mask


################################ loss functions #################################


def mlp_breaking_loss(model, batch, cfg):
    _mask = get_token_mask(batch)

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        out = model.model.layers[layer_id].mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        org_norm = batch["org_mlp_out_norm"][layer_id].to(out.device)
        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / org_norm**2
        # logging.debug(dotproducts)
        loss_acc += dotproducts.clip(min=cfg.mlp_floor).mean()
        # used to also do max=1, but that's catastrophic - stops unlearning but not disruption

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
    for batch in batches:
        with pt.no_grad():
            model(**prep_batch(batch, model.device))
        _mask = get_token_mask(batch)
        batch["org_mlp_out"] = {}
        batch["org_mlp_out_norm"] = {}
        for layer_id in range(*cfg.layer_range):
            mlp = model.model.layers[layer_id].mlp
            out = mlp.cached_out.detach()[_mask]
            batch["org_mlp_out"][layer_id] = out.cpu()
            batch["org_mlp_out_norm"][layer_id] = (
                out.float().norm(dim=-1).mean().cpu()
            )


def cache_activations_for_cb_retain_loss(model, batches, cfg):
    for batch in batches:
        with pt.no_grad():
            with trim_layers(model, max(cfg.cb_retaining_layers) + 1):
                output = model(
                    **prep_batch(batch, model.device), output_hidden_states=True
                )
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

