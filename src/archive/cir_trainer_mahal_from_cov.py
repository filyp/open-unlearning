# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
from contextlib import contextmanager

import hydra
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer
from welford_torch import OnlineCovariance

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_core import (
    get_last_act,
    get_last_grad,
    install_hooks,
)
from trainer.unlearn.cir.cir_utils import (
    batched,
    cb_retain_loss,
    get_update_norm,
    mlp_breaking_loss,
    prep_batch,
    scale_grads_,
    trainable_modules,
)

logging.basicConfig(level=logging.INFO)


@contextmanager
def trim_layers(model, max_layer):
    """Temporarily tell the model to use only the first max_layer layers."""
    all_layers = model.model.layers
    model.model.layers = model.model.layers[:max_layer]
    try:
        yield
    finally:
        model.model.layers = all_layers


def save_output_hook(module, args, output):
    # install hooks for MLPs
    module.cached_out = output


def get_precision_matrices_from_online_cov(
    online_covs: dict[str, OnlineCovariance], reg: float = 1e-6
):
    """Compute precision matrices (inverse covariance) and means from OnlineCovariance objects."""
    precisions = {}
    means = {}
    for n, online_cov in online_covs.items():
        if online_cov.mean is None:
            continue

        cov = online_cov.cov

        # Add regularization
        cov_reg = cov + reg * pt.eye(cov.shape[0], device=cov.device)

        precisions[n] = pt.linalg.inv(cov_reg)  # Invert
        means[n] = online_cov.mean

    return precisions, means


def get_mahalanobis_directions(
    activations: pt.Tensor, precision: pt.Tensor, mean: pt.Tensor
):
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


# %%
class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        model = self.model
        self.max_layer = max(max(cfg.layer_range), max(cfg.cb_retaining_layers)) + 1
        self.unit_optimizer = pt.optim.SGD(model.parameters(), lr=1.0)

        # * set trainable params
        for n, p in model.named_parameters():
            p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

        install_hooks(model)

        for layer_id in range(*cfg.layer_range):
            model.model.layers[layer_id].mlp.register_forward_hook(save_output_hook)

        # * go through whole dataset, to prepare the batches in advance

        self.forget_batches = []
        self.retain_batches = []
        for f, r in zip(
            # prepare separately forget and retain, to support different batch sizes
            batched(self.train_dataset.forget, cfg.train_batch_size),
            batched(self.train_dataset.retain, cfg.retain_batch_size),
        ):
            self.forget_batches.append(self.data_collator(f))
            self.retain_batches.append(self.data_collator(r))
        del self.train_dataset

        # * cache the activations for circuit breaker retaining
        if cfg.get("retaining_rate", 0) > 0:
            for batch in self.retain_batches:
                with pt.no_grad():
                    with trim_layers(model, self.max_layer):
                        output = model(
                            **prep_batch(batch, model.device), output_hidden_states=True
                        )
                batch["retain_acts"] = {
                    l_num: output.hidden_states[l_num].detach().to("cpu")
                    for l_num in cfg.cb_retaining_layers
                }

        # * cache the activations for MLP breaking
        for batch in self.forget_batches:
            with pt.no_grad():
                output = model(**prep_batch(batch, model.device))
            _mask = batch["attention_mask"].bool().clone()
            _mask[:, : cfg.cut_off_tokens] = False
            batch["org_mlp_out"] = {}
            batch["org_mlp_out_norm"] = {}
            for layer_id in range(*cfg.layer_range):
                mlp = model.model.layers[layer_id].mlp
                out = mlp.cached_out.detach()[_mask]
                batch["org_mlp_out"][layer_id] = out.cpu()
                batch["org_mlp_out_norm"][layer_id] = (
                    out.float().norm(dim=-1).mean().cpu()
                )

        self.acts_online_cov = {
            n: OnlineCovariance(device="cuda") for n, _ in trainable_modules(model)
        }

    def train(self):
        model = self.model
        for epoch in range(self.cfg.max_num_epochs):
            # ! one epoch
            model.train()
            for fb, rb in zip(self.forget_batches, self.retain_batches):
                inputs = dict(forget=fb, retain=rb)
                self.training_step(model, inputs)

            # ! calculate precision matrices for Mahalanobis distance
            reg = self.cfg.get("mahal_reg", 1e-2)
            self.act_precisions, self.act_means = (
                get_precision_matrices_from_online_cov(
                    self.acts_online_cov, reg=reg
                )
            )

            # ! Reset online covariance for next epoch
            self.acts_online_cov = {
                n: OnlineCovariance(device="cuda") for n, _ in trainable_modules(model)
            }

            # ! get metrics
            res = self.evaluate()
            # if res["wikitext_loss"] > self.init_res["wikitext_loss"] * self.cfg.get(
            #     "loss_budget", 1.01
            # ):
            #     break

    def training_step(self, model, inputs):
        # ! unlearning loss
        batch = inputs["forget"]
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        with trim_layers(model, self.max_layer):
            output = model(**prep_batch(batch, model.device), output_hidden_states=True)
        loss = mlp_breaking_loss(model, batch, self.cfg)
        loss.backward()

        # ! here we modify the grad
        for name, module in trainable_modules(model):
            if module.weight.grad is None:
                continue
            acts = get_last_act(
                module, batch["attention_mask"], self.cfg.cut_off_tokens
            )
            grads = get_last_grad(
                module, batch["attention_mask"], self.cfg.cut_off_tokens
            )  # todo rename to bos_len
            self.acts_online_cov[name].add_all(acts.float())
            assert len(acts.shape) == len(grads.shape) == 2

            if not hasattr(self, "act_precisions"):
                continue  # first epoch

            acts = acts.to(pt.float32)
            grads = grads.to(pt.float32)

            # ! Compute Mahalanobis directions and filter by distance
            precision = self.act_precisions[name]
            mean = self.act_means[name]
            centered = acts - mean
            mahal_dirs = get_mahalanobis_directions(acts, precision, mean)
            mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
            proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
            acts = proj_strenghts * mahal_dirs_norm
            # acts = mahal_dirs

            # Apply mask based on Mahalanobis distance quantile
            # ! note: once, using centered worked better than centered_norm, so investigate it
            centered_norm = centered / centered.norm(dim=1, keepdim=True)

            # dists = (centered_norm * mahal_dirs_norm).sum(dim=1)

            dists = (centered * mahal_dirs_norm).sum(dim=1)
            
            # # get Mahalanobis distance (squared) of mahal_dirs_norm
            # dists = ((mahal_dirs_norm @ precision.T) * mahal_dirs_norm).sum(dim=1)
            
            # dists = (centered * mahal_dirs).sum(dim=1)

            # dists = (centered_norm * mahal_dirs).sum(dim=1)

            mask = dists > dists.quantile(self.cfg.mahal_quantile)
            acts = acts[mask]
            grads = grads[mask]

            # del precision, mean, centered, mahal_dirs, mahal_dists_sq

            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(module.weight.grad.dtype)

        if not hasattr(self, "act_precisions"):
            return  # first epoch

        # * normalize grads
        # norm = get_update_norm(model)
        # scale_grads_(model, self.cfg.unlearning_rate / norm)
        # print(self.cfg.unlearning_rate / norm)
        scale_grads_(model, self.cfg.unlearning_rate)
        self.unit_optimizer.step()  # unit_optimizer has lr=1.0

        # ! retain pass
        if self.cfg.get("retaining_rate", 0) > 0:
            model.zero_grad(set_to_none=True)
            r_batch = inputs["retain"]
            with trim_layers(model, self.max_layer):
                output = model(
                    **prep_batch(r_batch, model.device), output_hidden_states=True
                )
            loss = cb_retain_loss(output, r_batch, self.cfg)
            loss.backward()

            scale_grads_(model, self.cfg.retaining_rate)  # apply intended lr
            self.unit_optimizer.step()  # unit_optimizer has lr=1.0

        return 0  # mock training loss


# # minimal steps to run:
# model = AutoModelForCausalLM.from_pretrained(
#     cfg.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
# )
# model.config.use_cache = False
# trainer = CirTrainer(
#     model=model,
#     train_dataset=train_dataset,
# )
# trainer.train()
