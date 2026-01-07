import torch as pt
from trainer.unlearn.cir.cir_utils import get_token_mask


def mlp_breaking(model, batch, layer_range):
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


def mlp_activation_breaking(model, batch, layer_range):
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
        loss_acc += (act * org_act).clip(min=0).mean() / org_act_norm**2

    return loss_acc / len(range(*layer_range))


def gate_and_up_breaking_approx(model, batch, layer_range):
    # ignores neurons where gate activation is negative
    # this way, we can store 2x less data than in gate_and_up_breaking
    # also, it turns out, ignoring negative gates actually yields better results
    _mask = get_token_mask(batch)

    if "org_act_out" not in batch:  # first epoch
        batch["org_act_out"] = {}

    loss_acc = 0
    for layer_id in range(*layer_range):
        mlp = model.model.layers[layer_id].mlp
        gate_out = mlp.gate_proj.cached_out[_mask]
        up_out = mlp.up_proj.cached_out[_mask]
        
        if layer_id not in batch["org_act_out"]:  # first epoch, so cache it
            act = gate_out.clip(min=0) * up_out
            batch["org_act_out"][layer_id] = act.detach().cpu()

        org_act = batch["org_act_out"][layer_id].to(up_out.device)
        norm = org_act.norm(dim=-1).mean()

        loss_acc += (up_out * org_act).clip(min=0).mean() / norm**1.5
        loss_acc += (gate_out.clip(min=0) * org_act.abs()).mean() / norm**1.5

    return loss_acc / len(range(*layer_range))


def gate_and_up_breaking(model, batch, layer_range):
    _mask = get_token_mask(batch)

    if "org_gate_out" not in batch:  # first epoch
        batch["org_gate_out"] = {}
        batch["org_up_out"] = {}

    loss_acc = 0
    for layer_id in range(*layer_range):
        mlp = model.model.layers[layer_id].mlp
        gate_out = mlp.gate_proj.cached_out[_mask]
        up_out = mlp.up_proj.cached_out[_mask]

        gate_out = mlp.act_fn(gate_out)

        if layer_id not in batch["org_gate_out"]:  # first epoch, so cache it
            batch["org_gate_out"][layer_id] = gate_out.detach().cpu()
            batch["org_up_out"][layer_id] = up_out.detach().cpu()

        org_gate_out = batch["org_gate_out"][layer_id].to(up_out.device)
        org_up_out = batch["org_up_out"][layer_id].to(up_out.device)
        gate_norm = org_gate_out.norm(dim=-1).mean()
        up_norm = org_up_out.norm(dim=-1).mean()

        loss_acc += (
            ((gate_out * org_gate_out).clip(min=0) * org_up_out.abs()).mean()
            / gate_norm**2
            / up_norm
        )
        loss_acc += (
            ((up_out * org_up_out).clip(min=0) * org_gate_out.abs()).mean()
            / up_norm**2
            / gate_norm
        )

    return loss_acc / len(range(*layer_range))


# # separate gate and up
# def gate_and_up_breaking(model, batch, layer_range):
#     # Similar to mlp_breaking_loss but targets the down_proj input activation
#     # (the intermediate MLP activation before the down projection)
#     _mask = get_token_mask(batch)

#     if "org_gate_out" not in batch:  # first epoch
#         batch["org_gate_out"] = {}
#         batch["org_up_out"] = {}

#     loss_acc = 0
#     for layer_id in range(*layer_range):
#         mlp = model.model.layers[layer_id].mlp
#         gate_out = mlp.gate_proj.cached_out[_mask]
#         up_out = mlp.up_proj.cached_out[_mask]

#         gate_out = gate_out.clip(min=0)

#         if layer_id not in batch["org_gate_out"]:  # first epoch, so cache it
#             # act = gate_out.clip(min=0) * up_out
#             batch["org_gate_out"][layer_id] = gate_out.detach().cpu()
#             batch["org_up_out"][layer_id] = up_out.detach().cpu()

#         org_gate_out = batch["org_gate_out"][layer_id].to(gate_out.device)
#         org_up_out = batch["org_up_out"][layer_id].to(up_out.device)

#         norm = org_gate_out.norm(dim=-1).mean()
#         loss_acc += (gate_out * org_gate_out).clip(min=0).mean() / norm**2
#         norm = org_up_out.norm(dim=-1).mean()
#         loss_acc += (up_out * org_up_out).clip(min=0).mean() / norm**2

#     return loss_acc / len(range(*layer_range))


# def neuron_breaking(model, batch, layer_range, output):
#     # It weighs how much neurons must be broken, by their gradient.
#     # Note: this works surprisingly bad; possibly there's some bug.
#     # Even if there's no bug, it would be useful to understand why it's so bad.

#     # Similar to mlp_activation_breaking_loss but uses gradients instead of cached activations
#     # On first batch, we do an extra backward pass with -output.loss to get gradients on neurons
#     _mask = get_token_mask(batch)

#     if "org_down_proj_grad" not in batch:  # first epoch
#         # Do backward pass with -loss to get gradients that would decrease the loss
#         (-output.loss).backward(retain_graph=True)

#         batch["org_down_proj_grad"] = {}
#         for layer_id in range(*layer_range):
#             down_proj = model.model.layers[layer_id].mlp.down_proj
#             # last_grad_input is the gradient w.r.t. the input of down_proj
#             grad = down_proj.last_grad_input[_mask]
#             batch["org_down_proj_grad"][layer_id] = grad.detach().cpu()

#         # Zero out the gradients so they don't affect the actual training step
#         model.zero_grad()

#     loss_acc = 0
#     for layer_id in range(*layer_range):
#         down_proj = model.model.layers[layer_id].mlp.down_proj
#         act = down_proj.last_act_input[_mask]

#         org_grad = batch["org_down_proj_grad"][layer_id].to(act.device)
#         # activation * org_gradient, clipped and averaged
#         loss_acc += (act * org_grad).clip(min=0).mean()

#     return loss_acc / len(range(*layer_range))

# neuron_breaking loss requires:
#             elif cfg.forget_loss == "neuron_breaking":
#                 mlp.down_proj.register_forward_hook(hooks.save_act_input)
#                 # note: overlaps some collapse hooks, but that's fine:
#                 mlp.down_proj.register_full_backward_hook(hooks.save_grad_input)
