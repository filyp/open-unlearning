# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import re

import torch as pt
from transformers import TrainerCallback

import trainer.unlearn.cir.loss_fns as loss_fns
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import (
    PreCachingDataLoader,
    get_relev_mask_with_caching,
    normalize_grads,
    prep_batch,
    sanitize_tensor,
)
from trainer.unlearn.cir.kl_utils import KLComputor
import trainer.unlearn.cir.hooks as hooks
from trainer.unlearn.cir.collapsers import MahalanobisCollapser

logging.basicConfig(level=logging.INFO)


class CalculateDistributionStatsCallback(TrainerCallback):
    """Callback to extract distribution stats at epoch end."""

    def __init__(self, trainer, collapsers):
        self.trainer = trainer
        self.collapsers = collapsers

    def on_epoch_end(self, args, state, control, **kwargs):
        for collapser in self.collapsers:
            collapser.process_saved_vecs()
        self.trainer.collapsers_initialized = True


def layer_num(name):
    return int(re.search(r"\.layers\.(\d+)\.", name).group(1))


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.collapsers_initialized = False
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place

        # set trainable params
        train_to_layer = int(len(self.model.model.layers) * cfg.train_first_layers)
        for name, module in self.model.named_modules():
            if not hasattr(module, "weight"):
                continue
            module.weight.requires_grad = (
                any(pattern in name for pattern in cfg.target_modules)
                and layer_num(name) < train_to_layer
            )
            if module.weight.requires_grad:  # install hooks
                module.register_forward_hook(hooks.save_act_input)
                module.register_full_backward_hook(hooks.save_grad_output)

        # additional hooks for computing grad collapse more efficiently
        for layer_id in range(train_to_layer):
            mlp = self.model.model.layers[layer_id].mlp
            mlp.down_proj.register_full_backward_hook(hooks.save_grad_input)
            mlp.down_proj.register_full_backward_hook(hooks.save_grad_output)

        # pre-cache batches (handy for storing batch-related data later)
        self.batches = PreCachingDataLoader(
            self.train_dataset,
            self.data_collator,
            self.args.per_device_train_batch_size,
        )

        # register collapsers, that calculate distribution stats, and later collapse
        _all_collapsers = []
        if "act_pcs_to_use" in self.cfg:
            self.act_collapsers = {
                name: MahalanobisCollapser(cfg.act_pcs_to_use, module.weight.device)
                for name, module in self.model.named_modules()
                if name.endswith(".up_proj")
            }
            _all_collapsers += list(self.act_collapsers.values())
        if "grad_pcs_to_use" in self.cfg:
            self.grad_collapsers = {
                name: MahalanobisCollapser(cfg.grad_pcs_to_use, module.weight.device)
                for name, module in self.model.named_modules()
                if name.endswith(".down_proj")
            }
            _all_collapsers += list(self.grad_collapsers.values())
            
        self.add_callback(CalculateDistributionStatsCallback(self, _all_collapsers))

        # ! prepare retain
        if "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.batches.retain)
            for param in self.model.parameters():
                if param.requires_grad:
                    assert not hasattr(param, "reference_grad")
                    param.reference_grad = pt.zeros_like(param)

    def get_train_dataloader(self):
        """Return dataloader over pre-batched forget/retain pairs."""
        return self.batches

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ! retain pass
        if "retain_momentum" in self.cfg:
            r_batch = inputs["retain"]
            model.zero_grad(set_to_none=True)
            output = model(**prep_batch(r_batch, model.device))
            kl, ce_loss, num_tokens = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    param.reference_grad *= self.cfg.retain_momentum
                    param.reference_grad += param.grad * (1 - self.cfg.retain_momentum)

        # ! unlearning loss
        batch = inputs["forget"]
        token_mask = batch["attention_mask"].bool().clone()
        token_mask[:, 0] = False  # ignore BOS token

        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = loss_fns.label_logits(output, batch)
        forget_loss.backward()

        if "grad_pcs_to_use" in self.cfg:
            grad_corrections = self.get_grad_correction(token_mask)

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            acts = module.last_act_input[token_mask].detach()
            grads = module.last_grad_output[token_mask].detach()
            assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            if name.endswith(".up_proj"):
                self.act_collapsers[name].add_vecs(acts)

            if not self.collapsers_initialized:
                continue  # so only collect activations and not train

            if "grad_pcs_to_use" in self.cfg:
                grads *= grad_corrections[parent_mlp_name(name)]

            # gate_proj shares inputs with up_proj, so we can use up_proj's collapser
            _up_proj_name = name.replace(".gate_proj", ".up_proj")
            acts = self.act_collapsers[_up_proj_name].collapse(acts).to(model.dtype)
            
            # if self.cfg.get("act_quantile", 0) > 0:
            #     relev_mask = get_relev_mask_with_caching(
            #         batch, name, acts.float(), token_mask, self.cfg.act_quantile
            #     )
            #     acts = acts[relev_mask]
            #     grads = grads[relev_mask]

            # ! MUDMAN-like operation
            if "retain_momentum" in self.cfg:
                ref_grad = module.weight.reference_grad
                col_mask = pt.einsum("ij,ti->tj", ref_grad, grads) * acts > 0
                row_mask = pt.einsum("ij,tj->ti", ref_grad, acts) * grads > 0
                acts *= col_mask
                grads *= row_mask

            # without the projections, this is equivalent to normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

        normalize_grads(model)

        if not self.collapsers_initialized:
            # zero gradients so that optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()

    def get_grad_correction(self, token_mask):
        grad_corrections = {}
        for name, module in self.model.named_modules():
            if not name.endswith(".down_proj"):
                continue
            up_proj = self.model.get_submodule(name.replace(".down_proj", ".up_proj"))
            if not up_proj.weight.requires_grad:
                continue

            grad_input = module.last_grad_input[token_mask].detach().clone()
            grad_output = module.last_grad_output[token_mask].detach().clone()
            assert grad_input.shape == (token_mask.sum(), module.weight.shape[1])
            assert grad_output.shape == (token_mask.sum(), module.weight.shape[0])

            self.grad_collapsers[name].add_vecs(grad_output)

            if not self.collapsers_initialized:
                continue  # first epoch, so only collect activations and not train

            out_collapsed = (
                self.grad_collapsers[name].collapse(grad_output).to(module.weight.dtype)
            )
            in_collapsed = out_collapsed @ module.weight  # backpropagation
            grad_correction = in_collapsed / sanitize_tensor(grad_input, 1e-6)
            grad_corrections[parent_mlp_name(name)] = grad_correction
        return grad_corrections


def parent_mlp_name(name):
    parent_name = name.rsplit(".", 1)[0]
    assert parent_name.endswith(".mlp")
    return parent_name


# # minimal steps to run:
# model = AutoModelForCausalLM.from_pretrained(
#     cfg.model_id, torch_dtype=pt.bfloat16
# )
# trainer = CIR(model=model, train_dataset=train_dataset)
# trainer.train()
