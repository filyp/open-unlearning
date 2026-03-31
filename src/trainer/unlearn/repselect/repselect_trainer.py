# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=SAMPLE_UNLEARN
import logging
import math
import random
from typing import Iterable

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise

from data.utils import batched, prep_batch
from evals.kl_eval import KLComputor
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.collapsers import CovCollapser
from trainer.unlearn.repselect.utils import get_banned_tokens, ManualLoRA
from trainer.utils import label_logits, normalize_grads

logging.basicConfig(level=logging.INFO)


class RepSelect(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.use_hooks = False
        self.batch_idx = 0
        self.recalc_every = math.ceil(  # on default, recalculate every epoch
            len(self.train_dataset) / self.args.per_device_train_batch_size
        )
        logging.info(f"{self.recalc_every=}")
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place

        self.is_moe = hasattr(self.model.model.layers[0].mlp, "experts")

        self.model.requires_grad_(False)  # train only modules that we specify
        self.lora_params = []
        self.base_trainable_params = []
        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            experts = mlp.experts if self.is_moe else [mlp]
            assert isinstance(experts, Iterable), "For new MoE implementation, please use RepSelectMOE"  # fmt: skip
            for expert in experts:
                for module in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    module.weight.requires_grad = True
                    self.base_trainable_params.append(module.weight)

                    # install hooks
                    module.register_forward_hook(self.save_act_input_hook)
                    module.register_full_backward_hook(self.collapse_hook)

                    # initialize collapsers
                    if "n_pcs" in cfg:
                        # module.act_collapser = IncrementalPCACollapser(cfg.n_pcs)
                        # module.grad_collapser = IncrementalPCACollapser(cfg.n_pcs)
                        module.act_collapser = CovCollapser(cfg.n_pcs)
                        module.grad_collapser = CovCollapser(cfg.n_pcs)

                    # ! adversarial LoRA
                    if "lora_lr" in cfg:
                        module.lora_module = ManualLoRA(
                            module.weight.shape[1],  # in_features
                            module.weight.shape[0],  # out_features
                            cfg.lora_rank,
                        ).to(self.model.device, dtype=self.model.dtype)
                        self.lora_params.extend(module.lora_module.parameters())
                        module.register_forward_hook(self.lora_forward_hook)

        # ! prepare retain
        self.kl_computor = None
        if "retain_momentum" in self.cfg:
            # pre-cache retain batches (needed for storing data for KL computation)
            self.retain_batches = [
                self.data_collator(r)
                for r in batched(
                    self.train_dataset.retain, self.args.per_device_train_batch_size
                )
            ]

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # Lazy init KLComputor (model is guaranteed on CUDA here)
        if self.kl_computor is None and "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.retain_batches)

        # ! retain pass
        if "retain_momentum" in self.cfg and self.batch_idx >= self.recalc_every * 2:
            # we ignore the input["retain"], and instead use the cached retain batches
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.base_trainable_params:
                if hasattr(param, "ref_grad"):
                    ref = dequantize_blockwise(*param.ref_grad)
                else:  # initialize
                    ref = pt.zeros_like(param)
                if param.grad is not None:  # some experts may be not chosen
                    momentum = self.cfg.retain_momentum
                    ref = ref * momentum + param.grad * (1 - momentum)
                param.ref_grad = quantize_blockwise(ref)  # 8-bit quantization

        # ! unlearning loss
        batch = inputs["forget"]
        self.token_mask = batch["attention_mask"].bool().clone()
        self.token_mask[:, 0] = False  # omit unlearning on the BOS token
        if self.processing_class.chat_template is not None:  # omit template tokens
            for banned_token in get_banned_tokens(self.processing_class):
                self.token_mask &= batch["input_ids"] != banned_token

        self.use_hooks = True
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        # forget_loss = label_logits(output.logits, batch["labels"])
        forget_loss = -output.loss
        # we will backpropagate because the graph has been built by the forward pass
        # but backward() itself will not compute weight gradients for base params
        # instead, weights will remain with grad computed by the collapse_hook
        for p in self.base_trainable_params:
            p.requires_grad_(False)
        forget_loss.backward()
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        self.use_hooks = False

        # ! update LoRA adversarially (gradient ascent - adversary tries to relearn)
        normalize_grads(self.lora_params)
        for p in self.lora_params:
            p.data += self.cfg.lora_lr * self.args.learning_rate * p.grad
            p.grad = None

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "act_collapser"):
                    module.act_collapser.process_saved_vecs()
                if hasattr(module, "grad_collapser"):
                    module.grad_collapser.process_saved_vecs()

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if not self.use_hooks:
            return
        module.last_act_input = args[0].detach()

    def collapse_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return
        acts = module.last_act_input
        grads = grad_output[0]
        module.last_act_input = None

        if self.is_moe:
            # todo, in future MoE HF5 implementation, make sure we use actual self.token_mask
            token_mask = grads.norm(dim=1) != 0
            acts = acts[token_mask]
            grads = grads[token_mask]
            if acts.shape[0] == 0:
                # this expert wasn't selected for any tokens
                return
        else:
            acts = acts[self.token_mask]
            grads = grads[self.token_mask]

        # note: we could optimize and reuse the act collapser for gate_proj and up_proj, but for simplicity don't
        if "n_pcs" in self.cfg:
            module.act_collapser.add_vecs(acts)
            module.grad_collapser.add_vecs(grads)

        if self.batch_idx < self.recalc_every * 2:
            return  # too early to train, so only collect activations and return early

        if "n_pcs" in self.cfg:
            acts = module.act_collapser.collapse(acts)
            grads = module.grad_collapser.collapse(grads)

        # ! KL-masking, per token and per module
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)

            token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
            kl_mask = token_disr > 0
            acts = acts[kl_mask]
            grads = grads[kl_mask]

            # # the core of DisrCollapse that can be swapped for the block above:
            # disr_grad = acts @ ref_grad.T
            # disr_grad /= disr_grad.norm(dim=1, keepdim=True) + 1e-8
            # projections = pt.einsum("tg,tg->t", disr_grad, grads).unsqueeze(1)
            # projections = projections.clamp(max=0)
            # grads -= projections * disr_grad

        # without acts and grads modifications, this is equivalent to normal backprop
        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

    def lora_forward_hook(self, module, args, output):
        if self.use_hooks:
            return output + module.lora_module(args[0])
