import random
from datetime import datetime
from pathlib import Path

import torch as pt

from data.utils import batched


def get_token_mask(labels):
    token_mask = labels != -100
    token_mask[:, 0] = False  # ignore BOS token
    return token_mask


class PreCachingDataLoader:
    def __init__(self, train_dataset, collator, batch_size):
        # * go through whole dataset, to prepare the batches in advance
        self.forget = [collator(f) for f in batched(train_dataset.forget, batch_size)]
        self.retain = [collator(r) for r in batched(train_dataset.retain, batch_size)]
        assert len(self.forget) <= len(self.retain)
        print(f"{len(self.forget)=}, {len(self.retain)=}")

    def __iter__(self):
        for idx in range(len(self.forget)):
            yield {
                "forget": self.forget[idx],
                "retain": random.choice(self.retain),
                "idx": idx,
            }

    def __len__(self):
        return len(self.forget)


def normalize_grads(model):
    # L2 norm of weight.grad, computed across all the trainable weights
    update_norm = pt.sqrt(
        sum(
            param.grad.float().norm() ** 2
            for param in model.parameters()
            if param.grad is not None
        )
    )
    # normalize the grads
    for param in model.parameters():
        if param.grad is not None:
            param.grad /= update_norm


def sanitize_tensor(t, epsilon):
    sign = t.sign()
    sign[sign == 0] = 1
    return t + sign * epsilon


def mlp_iter(model, layer_range):
    model_type = model.config.model_type
    # if layer_range is None:
    #     layer_range = [0, len(model.model.layers)]

    if model_type in ["qwen3_moe"]:
        for layer_id in range(*layer_range):
            for expert in model.model.layers[layer_id].mlp.experts:
                yield expert
    else:
        for layer_id in range(*layer_range):
            yield model.model.layers[layer_id].mlp


def save_kl_mask(inputs, token_mask, kl_mask, save_path, token_loss_delta=None):
    masks_path = Path(save_path)
    masks_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_path = masks_path / f"{timestamp}.pt"

    token_mask_indices = token_mask.nonzero()
    kl_mask_indices = token_mask_indices[kl_mask.cpu()]
    kl_mask_2d = pt.zeros_like(token_mask)
    kl_mask_2d[kl_mask_indices[:, 0], kl_mask_indices[:, 1]] = 1
    kl_mask_2d = kl_mask_2d & token_mask

    data = {
        "input_ids": inputs["input_ids"].cpu(),
        "attention_mask": inputs["attention_mask"].cpu(),
        "kl_mask_2d": kl_mask_2d.cpu(),
    }
    if token_loss_delta is not None:
        data["token_loss_delta"] = token_loss_delta.cpu()
    pt.save(data, mask_path)


# def install_hooks(model, layer_range, forget_loss, train_to_layer):
#     # hooks for forget loss
#     for layer_id in range(*layer_range):
#         mlp = model.model.layers[layer_id].mlp
#         if forget_loss == "mlp_breaking":
#             mlp.down_proj.register_forward_hook(hooks.save_act_output)
#         elif forget_loss == "mlp_activation_breaking":
#             mlp.down_proj.register_forward_hook(hooks.save_act_input)
#         elif forget_loss.startswith("gate_and_up_breaking"):
#             mlp.gate_proj.register_forward_hook(hooks.save_act_output)
#             mlp.up_proj.register_forward_hook(hooks.save_act_output)

#     for mlp in mlp_iter(model, [0, train_to_layer]):
#         # hooks for component collapse
#         for module in [mlp.up_proj, mlp.gate_proj]:
#             module.register_forward_hook(hooks.save_act_input)
#             module.register_full_backward_hook(hooks.save_grad_output)
#         # additional hooks for computing grad collapse more efficiently
#         mlp.down_proj.register_full_backward_hook(hooks.save_grad_input)
#         mlp.down_proj.register_full_backward_hook(hooks.save_grad_output)


# def get_relev_mask_with_caching(batch, name, acts, token_mask, quantile):
#     if "relev_mask" not in batch:
#         batch["relev_mask"] = {}

#     if name in batch["relev_mask"]:
#         # we use the caching, because recalculating these can be slow
#         relev_mask = batch["relev_mask"][name]
#     else:
#         norms = acts.norm(dim=1)
#         relev_mask = compute_per_text_quantile_mask(norms, token_mask, quantile)
#         batch["relev_mask"][name] = relev_mask
#     return relev_mask


# def compute_per_text_quantile_mask(
#     dists: pt.Tensor, token_mask: pt.Tensor, quantile: float
# ) -> pt.Tensor:
#     """Compute quantile threshold per text in batch and return relevance mask.

#     Args:
#         dists: Distance values for each token (1D tensor of length N where N is number of True values in token_mask)
#         token_mask: Boolean mask indicating which tokens are valid (as in labels != -100) (2D tensor of shape [batch_size, seq_len])
#         quantile: Quantile threshold (0-1) for filtering

#     Returns:
#         Boolean mask of same length as dists, True for tokens above their text's quantile threshold
#     """
#     # apparently this is quite slow, so it's better to compute only at the beginning and store the masks
#     batch_indices = pt.nonzero(token_mask)[:, 0].to(dists.device)
#     act_relev_mask = pt.zeros(len(dists), dtype=pt.bool, device=dists.device)
#     for text_idx in batch_indices.unique():
#         text_mask = batch_indices == text_idx
#         text_dists = dists[text_mask]
#         threshold = text_dists.quantile(quantile)
#         act_relev_mask[text_mask] = text_dists > threshold
#     return act_relev_mask


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
#             dists, get_token_mask(batch["labels"]), cfg.mlp_quantile
#         )
#         # Zero out filtered outputs so they don't contribute to loss
#         batch["org_mlp_out"][layer_id][~quantile_mask] = 0


# def cb_retain_loss(output, batch, cfg):
#     # _mask = get_token_mask(batch["labels"])  # retains only on meaningful tokens
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
