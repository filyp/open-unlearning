# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
from contextlib import contextmanager
from itertools import islice

import hydra
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_core import (
    get_last_act,
    get_last_grad,
    install_hooks,
    project_out,
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


def get_projections_exact(vector_lists: dict[str, list[pt.Tensor]]):
    """Compute exact PCA projections (mean + principal components) for each module."""
    to_collapse = {}
    eigenvalues_dict = {}
    for n in list(vector_lists.keys()):
        pt.cuda.empty_cache()
        cached_vectors = vector_lists.pop(n)
        if not cached_vectors:
            continue
        v = pt.cat(cached_vectors)
        v = v.to("cuda").float()

        mean = v.mean(axis=0)

        # Compute the PCA components
        # Center the data
        v = v - mean
        # Compute covariance matrix
        cov = (v.T @ v) / (v.shape[0] - 1)
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = pt.linalg.eigh(cov)
        # Sort in descending order
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pca_components = eigenvectors.T

        # return one tensor of mean and the pca components
        to_collapse[n] = pt.cat([mean.reshape(1, -1), pca_components], dim=0)
        eigenvalues_dict[n] = eigenvalues

        del v, mean, cov, eigenvectors, pca_components

    return to_collapse, eigenvalues_dict


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

        self.acts_list = {n: [] for n, _ in trainable_modules(model)}

    def train(self):
        model = self.model
        for epoch in range(self.cfg.max_num_epochs):
            # ! one epoch
            model.train()
            for fb, rb in zip(self.forget_batches, self.retain_batches):
                inputs = dict(forget=fb, retain=rb)
                self.training_step(model, inputs)

            if epoch % self.cfg.get("pca_every_n", 1) == 0:
                # ! calculate means and PCA components (exact)
                self.act_projections, self.act_eigenvalues = get_projections_exact(
                    self.acts_list
                )
            self.acts_list = {n: [] for n, _ in trainable_modules(model)}

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
            )
            self.acts_list[name].append(acts.clone().to("cpu"))
            assert len(acts.shape) == len(grads.shape) == 2

            if not hasattr(self, "act_projections"):
                continue  # first epoch

            acts = acts.to(pt.float32)
            grads = grads.to(pt.float32)

            # ! project out the mean from activations
            act_mean = self.act_projections[name][0]
            acts = acts - project_out(acts, act_mean)

            # ! apply masking based on eigenvalues and threshold
            eigenvectors = self.act_projections[name][1:]
            pc_projections = eigenvectors @ acts.T
            eigenvalues = self.act_eigenvalues[name]
            threshold = self.cfg.get("pca_threshold", 3)
            epsilon = self.cfg.get("pca_epsilon", 0.03)
            mask = pc_projections.abs() / (eigenvalues.reshape(-1, 1) + epsilon) > threshold
            # Keep the first act_proj_num components collapsed (masked out)
            mask[: self.cfg.act_proj_num, :] = False
            # mask[self.cfg.act_proj_num :, :] = True
            acts = (pc_projections * mask).T @ eigenvectors

            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(module.weight.grad.dtype)

        if not hasattr(self, "act_projections"):
            return  # first epoch

        # * normalize grads
        norm = get_update_norm(model)
        scale_grads_(model, self.cfg.unlearning_rate / norm)
        self.unit_optimizer.step()  # unit_optimizer has lr=1.0

        if self.cfg.get("retaining_rate", 0) > 0:
            model.zero_grad(set_to_none=True)
            r_batch = inputs["retain"]
            with trim_layers(model, self.max_layer):
                output = model(**prep_batch(r_batch, model.device), output_hidden_states=True)
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
