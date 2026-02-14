# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from transformers import TrainerCallback

import trainer.unlearn.cir.hooks as hooks
from data.utils import batched, prep_batch
from trainer.utils import label_logits, normalize_grads
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.collapsers import MahalanobisCollapser
from trainer.unlearn.cir.kl_utils import KLComputor

logging.basicConfig(level=logging.INFO)


class CIR_MoE(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.after_first_epoch = False
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place

        # set trainable params
        self.model.requires_grad_(False)  # train only modules that we specify
        train_to_layer = int(len(self.model.model.layers) * cfg.train_first_layers)
        for name, module in self.model.named_modules():
            if not hasattr(module, "weight"):
                # logging.info(f"Skipping {name} because it doesn't have a weight")
                continue
            module.weight.requires_grad = (
                any(pattern in name for pattern in cfg.target_modules)
                and layer_num(name) < train_to_layer
            )
            if module.weight.requires_grad:
                # install hooks
                module.register_forward_hook(hooks.save_act_input)
                module.register_full_backward_hook(hooks.save_grad_output)
                module.last_act_input = None
                module.last_grad_output = None
                # install collapsers
                if "act_pcs_to_use" in self.cfg:
                    pass
                if "grad_pcs_to_use" in self.cfg:
                    module.grad_collapser = MahalanobisCollapser(
                        cfg.grad_pcs_to_use, module.weight.device
                    )

        self.add_callback(CalculateDistributionStatsCallback(self))

        # pre-cache batches (handy for storing batch-related data later)
        # ! prepare retain
        if "retain_momentum" in self.cfg:
            # pre-cache retain batches (needed for storing data for KL computation)
            self.retain_batches = [
                self.data_collator(r)
                for r in batched(
                    self.train_dataset.retain, self.args.per_device_train_batch_size
                )
            ]
            self.kl_computor = KLComputor(self.model, self.batches.retain)
            for param in self.model.parameters():
                if param.requires_grad:
                    assert not hasattr(param, "ref_grad")
                    param.ref_grad = quantize_blockwise(pt.zeros_like(param))

    def get_train_dataloader(self):
        """Return dataloader over pre-batched forget/retain pairs."""
        return self.batches

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ! retain pass
        if "retain_momentum" in self.cfg and self.after_first_epoch:
            # we ignore the input["retain"], and instead use the cached retain batches
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    ref = dequantize_blockwise(*param.ref_grad)
                    ref *= self.cfg.retain_momentum
                    ref += param.grad * (1 - self.cfg.retain_momentum)
                    param.ref_grad = quantize_blockwise(ref)

        # ! unlearning loss
        batch = inputs["forget"]
        token_mask = batch["attention_mask"].bool().clone()
        token_mask[:, 0] = False  # ignore BOS token
        # todo, implement token masking for moe

        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = label_logits(output.logits, batch["labels"])
        forget_loss.backward()

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            # acts = module.last_act_input[token_mask].detach()
            # grads = module.last_grad_output[token_mask].detach()
            # assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            # assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            if module.last_act_input is None or module.last_grad_output is None:
                # some experts may be never chosen, and so have no acts and grads
                continue

            acts = module.last_act_input.detach()
            grads = module.last_grad_output.detach()
            module.last_act_input = None
            module.last_grad_output = None
            # we need to cast, because sometimes the router causes upcast to float32
            acts = acts.to(model.dtype)
            grads = grads.to(model.dtype)

            # if "act_pcs_to_use" in self.cfg:
            #     if name.endswith(".up_proj"):
            #         self.act_collapsers[name].add_vecs(acts)
            if "grad_pcs_to_use" in self.cfg:
                module.grad_collapser.add_vecs(grads)

            if not self.after_first_epoch:
                continue  # so only collect activations and not train

            # if "act_pcs_to_use" in self.cfg:
            #     # gate_proj shares inputs with up_proj, so we can use up_proj's collapser
            #     _up_proj_name = name.replace(".gate_proj", ".up_proj")
            #     acts = self.act_collapsers[_up_proj_name].collapse(acts).to(model.dtype)
            if "grad_pcs_to_use" in self.cfg:
                if not hasattr(module.grad_collapser, "mean"):
                    breakpoint()
                grads = module.grad_collapser.collapse(grads).to(model.dtype)

            # ! MUDMAN-like operation
            if "retain_momentum" in self.cfg:
                ref_grad = dequantize_blockwise(*module.weight.ref_grad).to(model.dtype)
                token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
                kl_mask = token_disr > 0
                acts = acts[kl_mask]
                grads = grads[kl_mask]

            # if acts.dtype != grads.dtype:
            #     breakpoint()
            # without the projections, this is equivalent to normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
            # would be possible to optimize training by disabling the first grad computation,
            # since we discard these grads anyway

        normalize_grads(model)

        if not self.after_first_epoch:
            # zero gradients so that optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()


class CalculateDistributionStatsCallback(TrainerCallback):
    """Callback to extract distribution stats at epoch end."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        for module in self.trainer.model.modules():
            if hasattr(module, "act_collapser"):
                module.act_collapser.process_saved_vecs()
            if hasattr(module, "grad_collapser"):
                module.grad_collapser.process_saved_vecs()
        self.trainer.after_first_epoch = True
