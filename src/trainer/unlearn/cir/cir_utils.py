# %%
import torch as pt

################################ torch utils #################################


def trainable_modules(model):
    return [
        (n, m)
        for n, m in model.named_modules()
        if "_proj" in n and m.weight.requires_grad
    ]


def get_update_norm(model):
    """L2 norm of weight.grad, computed across all the trainable weights."""
    return (
        sum(
            m.weight.grad.to(pt.float32).norm() ** 2
            for _, m in trainable_modules(model)
            if m.weight.grad is not None
        )
        ** 0.5
    )


def scale_grads_(model, factor: float):
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= factor



def sanitize_batch(batch):
    return dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )


################################ loss functions #################################



def mlp_breaking_loss(model, batch, cfg):
    _mask = batch["attention_mask"] & (batch["labels"] != -100)
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        out = model.model.layers[layer_id].mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        org_norm = batch["org_mlp_out_norm"][layer_id].to(out.device)
        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / org_norm ** 2
        # logging.debug(dotproducts)
        loss_acc += dotproducts.clip(min=cfg.mlp_floor).mean()
        # used to also do max=1, but that's catastrophic - stops unlearning but not disruption

    return loss_acc / len(range(*cfg.layer_range))


def cb_retain_loss(output, batch, cfg):
    _mask = batch["attention_mask"] & (batch["labels"] != -100)
    _mask = batch["attention_mask"].bool().clone()
    # _mask[:, :cfg.cut_off_tokens] = False  # do not do it! retain everywhere!
    
    loss_acc = 0
    for layer_id in cfg.cb_retaining_layers:
        acts = output.hidden_states[layer_id][_mask].float()
        org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
        assert acts.shape == org_acts.shape
        assert len(acts.shape) == 2

        avg_act_norm = org_acts.norm(dim=-1).mean()
        dist = (acts - org_acts).norm(dim=-1).mean() / avg_act_norm

        loss_acc += dist ** cfg.cb_retaining_pow

    return loss_acc / len(cfg.cb_retaining_layers)
