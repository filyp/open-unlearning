# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import re

import torch as pt

import trainer.unlearn.cir.loss_fns as loss_fns
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import (
    PreCachingDataLoader,
    normalize_grads,
    prep_batch,
)
from trainer.unlearn.cir.kl_utils import KLComputor
import trainer.unlearn.cir.hooks as hooks

logging.basicConfig(level=logging.INFO)


def layer_num(name):
    return int(re.search(r"\.layers\.(\d+)\.", name).group(1))


class MUDMAN(UnlearnTrainer):
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

        # pre-cache batches (handy for storing batch-related data later)
        self.batches = PreCachingDataLoader(
            self.train_dataset,
            self.data_collator,
            self.args.per_device_train_batch_size,
        )

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
        r_batch = inputs["retain"]
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(r_batch, model.device))
        kl, ce_loss, num_tokens = self.kl_computor.get_kl(r_batch)
        kl.backward()

        # output.loss.backward()
        for param in self.model.parameters():
            if param.requires_grad:
                param.reference_grad *= self.cfg.retain_momentum
                param.reference_grad += param.grad * (1 - self.cfg.retain_momentum)
        model.zero_grad(set_to_none=True)

        # ! unlearning loss
        batch = inputs["forget"]

        token_mask = batch["attention_mask"].bool().clone()
        token_mask[:, 0] = False  # ignore BOS token

        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device), output_hidden_states=True)

        forget_loss = loss_fns.label_logits(output, batch)
        # forget_loss = -output.loss

        forget_loss.backward()

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            acts = module.last_act_input[token_mask].detach()
            grads = module.last_grad_output[token_mask].detach()
            assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            ref_grad = module.weight.reference_grad

            # _mask = wgrad.sign() != ref_grad.sign()
            # wgrad[_mask] = 0

            # # wgrad = pt.einsum("ti,tj->ij", grads, acts)
            # wgrad = module.weight.grad
            # ref_sim = ref_grad * wgrad
            # col_mask = ref_sim.mean(dim=0, keepdim=True) > 0
            # row_mask = ref_sim.mean(dim=1, keepdim=True) > 0
            # wgrad *= col_mask
            # wgrad *= row_mask
            # module.weight.grad = wgrad

            col_mask = pt.einsum("ij,ti->tj", ref_grad, grads) * acts > 0
            row_mask = pt.einsum("ij,tj->ti", ref_grad, acts) * grads > 0
            acts *= col_mask
            grads *= row_mask
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

        normalize_grads(model)

        return forget_loss.detach()
