# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
from dataclasses import dataclass

import torch as pt
from welford_torch import OnlineCovariance

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_core import get_last_act, get_last_grad, install_hooks
from trainer.unlearn.cir.cir_utils import (
    batched,
    cache_activations_for_cb_retain_loss,
    cache_activations_for_mlp_breaking_loss,
    cb_retain_loss,
    mlp_breaking_loss,
    prep_batch,
    scale_grads_,
    trainable_modules,
    trim_layers,
)

logging.basicConfig(level=logging.INFO)


def save_output_hook(module, args, output):
    # install hooks for MLPs
    module.cached_out = output


@dataclass
class DistributionStats:
    mean: pt.Tensor
    eigenvalues: pt.Tensor
    eigenvectors: pt.Tensor


def get_mahalanobis_directions(
    centered_vecs: pt.Tensor,
    stats: DistributionStats,
    reg: float = 1e-2,
    mahal_pow: float = 1.0,
):
    """Compute Mahalanobis directions using cached eigendecomposition.

    Mahalanobis direction = Σ⁻¹(x - μ) = V (Λ + reg)⁻¹ Vᵀ (x - μ)
    """
    # V (Λ + reg)⁻¹ Vᵀ (x - μ) in two matmuls
    projected = centered_vecs @ stats.eigenvectors  # (N, D)
    directions = (
        projected / (stats.eigenvalues + reg) ** mahal_pow
    ) @ stats.eigenvectors.T
    return directions


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

        cache_activations_for_mlp_breaking_loss(model, self.forget_batches, cfg)
        if cfg.get("retaining_rate", 0) > 0:
            cache_activations_for_cb_retain_loss(model, self.retain_batches, cfg)

        self.acts_online_cov = {
            n: OnlineCovariance(device="cuda") for n, _ in trainable_modules(model)
        }

    def train(self):
        model = self.model

        if self.args.eval_on_start:
            self.evaluate()

        for epoch in range(self.cfg.max_num_epochs):
            # ! one epoch
            model.train()
            for fb, rb in zip(self.forget_batches, self.retain_batches):
                inputs = dict(forget=fb, retain=rb)
                self.training_step(model, inputs)

            # ! Extract distribution stats and reset online covariance for next epoch
            self.distribution_stats = {
                name: DistributionStats(oc.mean, oc.eig_val, oc.eig_vec)
                for name, oc in self.acts_online_cov.items()
                if oc.mean is not None
            }
            self.acts_online_cov = {
                n: OnlineCovariance(device="cuda") for n, _ in trainable_modules(model)
            }

            # ! get metrics
            self.evaluate()
            if self.control.should_training_stop:
                break

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

            if not getattr(self, "distribution_stats", None):
                continue  # first epoch

            acts = acts.to(pt.float32)
            grads = grads.to(pt.float32)

            stats = self.distribution_stats[name]
            centered = acts - stats.mean
            projected = centered @ stats.eigenvectors  # (N, D)

            # # ! collapse top components
            # # works worse a bit than full mahalanobis, but is acceptable when you want more speed (can be optimized to compute only top N PCs)
            # projected[:, -self.cfg.act_collapse:] = 0  # collapse top components

            # filtered_acts = projected @ stats.eigenvectors.T  # skip any mahalanobis

            # ! Compute Mahalanobis directions using eigendecomposition
            mahal_dirs = (
                projected
                / (stats.eigenvalues + self.cfg.mahal_reg) ** self.cfg.mahal_pow
            ) @ stats.eigenvectors.T

            if self.cfg.project_to_mahal:
                mahal_dirs_norm = mahal_dirs / mahal_dirs.norm(dim=1, keepdim=True)
                proj_strenghts = (mahal_dirs_norm * centered).sum(dim=1, keepdim=True)
                filtered_acts = proj_strenghts * mahal_dirs_norm
            else:
                filtered_acts = mahal_dirs

            if self.cfg.filter:
                # act_norms = centered.norm(dim=1, keepdim=True)
                # rescaled_acts = acts * act_norms**(self.cfg.act_norm_pow)
                for_filtering = get_mahalanobis_directions(
                    centered, stats, self.cfg.filter_reg, self.cfg.filter_pow
                )
                for_filtering_normed = for_filtering / for_filtering.norm(
                    dim=1, keepdim=True
                )
                dists = (for_filtering_normed * centered).sum(dim=1)

                # act_norms = centered.norm(dim=1, keepdim=True)
                # rescaled_acts = acts * act_norms**(self.cfg.act_norm_pow)
                # dists = (rescaled_acts * mahal_dirs_norm).sum(dim=1)

                # Apply mask based on Mahalanobis distance quantile
                # ! note: once, using centered worked better than centered_norm, so investigate it
                # centered_norm = centered / centered.norm(dim=1, keepdim=True)

                # dists = (centered_norm * mahal_dirs_norm).sum(dim=1)

                # dists = (centered * mahal_dirs_norm).sum(dim=1)

                # dists = (centered * mahal_dirs).sum(dim=1)

                # dists = (centered_norm * mahal_dirs).sum(dim=1)

                # # get Mahalanobis distance (squared) using eigendecomposition
                # projected = mahal_dirs_norm @ eigenvectors
                # dists = (projected ** 2 / eigenvalues).sum(dim=1)

                mask = dists > dists.quantile(self.cfg.quantile)
                acts = filtered_acts[mask]
                grads = grads[mask]

            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(model.dtype)

        if not getattr(self, "distribution_stats", None):
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
