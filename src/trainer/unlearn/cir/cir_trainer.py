# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
from contextlib import contextmanager

import hydra
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_core import *
from trainer.unlearn.cir.cir_utils import (
    cb_retain_loss,
    get_update_norm,
    mlp_breaking_loss,
    scale_grads_,
    trainable_modules,
    sanitize_batch,
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
        
        # ! go through whole dataset, to prepare the batches in advance
        # forget and retain batches must be prepared separately, to support different forget and retain batch sizes
        self.forget_batches = []
        for i in range(0, len(self.train_dataset), cfg.train_batch_size):
            # Get batch_size samples
            samples = [self.train_dataset[j]["forget"] for j in range(i, min(i + cfg.train_batch_size, len(self.train_dataset)))]
            # Use collator to create proper batched tensors
            self.forget_batches.append(self.data_collator(samples))
        self.retain_batches = []
        for n in range(len(self.forget_batches)):
            start = n * cfg.retain_batch_size
            end = (n + 1) * cfg.retain_batch_size
            samples = [self.train_dataset[j]["retain"] for j in range(start, end)]
            self.retain_batches.append(self.data_collator(samples))
        del self.train_dataset

        # * cache the activations for circuit breaker retaining
        if cfg.get("retaining_rate", 0) > 0:
            for batch in self.retain_batches:
                with pt.no_grad():
                    with trim_layers(model, self.max_layer):
                        output = model(**sanitize_batch(batch), output_hidden_states=True)
                batch["retain_acts"] = {
                    l_num: output.hidden_states[l_num].detach().to("cpu")
                    for l_num in cfg.cb_retaining_layers
                }

        # * cache the activations for MLP breaking
        for batch in self.forget_batches:
            with pt.no_grad():
                output = model(**sanitize_batch(batch))
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
        self.grads_list = {n: [] for n, _ in trainable_modules(model)}

    def train(self):
        model = self.model
        for epoch in range(self.cfg.max_num_epochs):
            # ! one epoch
            model.train()
            for forget_batch, retain_batch in zip(self.forget_batches, self.retain_batches):
                inputs = dict(forget=forget_batch, retain=retain_batch)
                self.training_step(model, inputs)

            if epoch % self.cfg.get("pca_every_n", 1) == 0:
                # ! calculate means and PCA components
                self.act_to_collapse = get_projections(
                    self.acts_list, self.cfg.act_proj_num, self.cfg.cir_niter
                )
                self.grad_to_collapse = get_projections(
                    self.grads_list, self.cfg.grad_proj_num, self.cfg.cir_niter
                )
            self.acts_list = {n: [] for n, _ in trainable_modules(model)}
            self.grads_list = {n: [] for n, _ in trainable_modules(model)}

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
        with trim_layers(model, self.max_layer):
            output = model(**sanitize_batch(batch), output_hidden_states=True)
        loss = mlp_breaking_loss(model, batch, self.cfg)
        loss.backward()

        # ! here we modify the grad
        for name, module in trainable_modules(model):
            if module.weight.grad is None:
                continue
            acts = get_last_act(module, batch["attention_mask"], self.cfg.cut_off_tokens)
            grads = get_last_grad(module, batch["attention_mask"], self.cfg.cut_off_tokens)
            self.acts_list[name].append(acts.clone().to("cpu"))
            self.grads_list[name].append(grads.clone().to("cpu"))
            assert len(acts.shape) == len(grads.shape) == 2

            if not hasattr(self, "act_to_collapse"):
                continue  # first epoch

            # ! proj out the means and PCA components
            for comp in self.act_to_collapse[name]:
                acts -= project_out(acts, comp)
            for comp in self.grad_to_collapse[name]:
                grads -= project_out(grads, comp)
            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

        if not hasattr(self, "act_to_collapse"):
            return  # first epoch

        # * normalize grads
        norm = get_update_norm(model)
        scale_grads_(model, self.cfg.unlearning_rate / norm)
        self.unit_optimizer.step()  # unit_optimizer has lr=1.0

        if self.cfg.get("retaining_rate", 0) > 0:
            model.zero_grad(set_to_none=True)
            r_batch = inputs["retain"]
            with trim_layers(model, self.max_layer):
                output = model(**sanitize_batch(r_batch), output_hidden_states=True)
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