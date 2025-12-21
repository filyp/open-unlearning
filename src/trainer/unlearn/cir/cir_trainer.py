# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
from dataclasses import dataclass

import torch as pt
from welford_torch import OnlineCovariance

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_core import install_hooks
from trainer.unlearn.cir.cir_utils import *

logging.basicConfig(level=logging.INFO)


def save_output_hook(module, args, output):
    # install hooks for MLPs
    module.cached_out = output


@dataclass
class DistributionStats:
    mean: pt.Tensor
    eigenvalues: pt.Tensor
    eigenvectors: pt.Tensor


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        model = self.model
        self.max_layer = max(max(cfg.layer_range), max(cfg.cb_retaining_layers)) + 1
        self.optimizer = pt.optim.SGD(model.parameters(), lr=cfg.unlearning_rate)

        # * set trainable params
        for n, p in model.named_parameters():
            p.requires_grad = any(pattern in n for pattern in cfg.target_modules)
            # if p.requires_grad:  # * not training first n layers
            #     layer_num = int(n.split(".")[2])
            #     if layer_num < cfg.train_from_layer:
            #         p.requires_grad = False

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
            self.ret_optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retaining_rate)

        # todo if using only gate and up proj, use just one distr per MLP
        self.acts_online_cov = {
            n: OnlineCovariance(device="cuda")
            for n, m in model.named_modules()
            if hasattr(m, "weight") and m.weight.requires_grad
        }

    def train(self):
        model = self.model

        if self.args.eval_on_start:
            self.evaluate()

        for _ in range(self.cfg.max_num_epochs):
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

            # reset online covariance accumulators for the next epoch
            for name in self.acts_online_cov:
                # todo, for multi-GPU, maybe use weigth's device?
                self.acts_online_cov[name] = OnlineCovariance(device="cuda")

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
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight.grad is None:
                continue

            token_mask = get_token_mask(batch)
            acts = module.last_act_full[token_mask].to(pt.float32)
            grads = module.last_grad_full[token_mask].to(pt.float32)
            assert len(acts.shape) == len(grads.shape) == 2

            self.acts_online_cov[name].add_all(acts)

            if not getattr(self, "distribution_stats", None):
                continue  # first epoch, so only collect activations and not train

            stats = self.distribution_stats[name]
            centered = acts - stats.mean
            projected = centered @ stats.eigenvectors  # (N, D)

            # # ! collapse top components
            # # works worse a bit than full mahalanobis, but is acceptable when you want more speed (can be optimized to compute only top N PCs)
            # projected[:, -self.cfg.act_collapse:] = 0  # collapse top components

            # filtered_acts = projected @ stats.eigenvectors.T  # skip any mahalanobis

            # ! Compute Mahalanobis directions using eigendecomposition
            mahal_dirs = (
                projected / (stats.eigenvalues + self.cfg.mahal_reg)
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

                # get mahalanobis directions
                for_filtering = (
                    projected / (stats.eigenvalues + self.cfg.filter_reg)
                ) @ stats.eigenvectors.T

                for_filtering_normed = for_filtering / for_filtering.norm(
                    dim=1, keepdim=True
                )
                dists = (for_filtering_normed * centered).sum(dim=1)

                # compute quantile threshold per text in batch
                batch_indices = pt.nonzero(token_mask)[:, 0]  # which text each token belongs to
                act_relev_mask = pt.zeros(len(dists), dtype=pt.bool, device=dists.device)
                for text_idx in batch_indices.unique():
                    text_mask = batch_indices == text_idx
                    text_dists = dists[text_mask]
                    threshold = text_dists.quantile(self.cfg.quantile)
                    act_relev_mask[text_mask] = text_dists > threshold

                acts = filtered_acts[act_relev_mask]
                grads = grads[act_relev_mask]

            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(model.dtype)

        if not getattr(self, "distribution_stats", None):
            return  # first epoch, so only collect activations and not train

        # print(self.cfg.unlearning_rate / get_update_norm(model))
        self.optimizer.step()

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
            self.ret_optimizer.step()

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


# * old ways of filtering:
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


# def get_mahalanobis_directions(
#     centered_vecs: pt.Tensor,
#     stats: DistributionStats,
#     reg: float = 1e-2,
#     mahal_pow: float = 1.0,
# ):
#     """Compute Mahalanobis directions using cached eigendecomposition.

#     Mahalanobis direction = Σ⁻¹(x - μ) = V (Λ + reg)⁻¹ Vᵀ (x - μ)
#     """
#     # V (Λ + reg)⁻¹ Vᵀ (x - μ) in two matmuls
#     projected = centered_vecs @ stats.eigenvectors  # (N, D)
#     directions = (
#         projected / (stats.eigenvalues + reg) ** mahal_pow
#     ) @ stats.eigenvectors.T
#     return directions
