# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=SAMPLE_UNLEARN
import logging
import math
from typing import Iterable

import torch as pt
from bitsandbytes.functional import dequantize_blockwise

from data.utils import batched, prep_batch
from evals.kl_eval import KLComputor
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.collapsers import InvSmallCovCollapser
from trainer.unlearn.repselect.utils import get_banned_tokens, ManualLoRA
from trainer.utils import label_logits, normalize_grads, require_grad, update_ref_grad

logging.basicConfig(level=logging.INFO)


class RepSelect(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.do_add_vecs = False
        self.do_collapse = False
        self.use_lora = False
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
                        collapser_class = InvSmallCovCollapser
                        module.act_collapser = collapser_class(cfg.n_pcs)
                        module.grad_collapser = collapser_class(cfg.n_pcs)

                    # ! adversarial LoRA
                    if "lora_lr" in cfg:
                        module.lora_module = ManualLoRA(
                            module.weight.shape[1],  # in_features
                            module.weight.shape[0],  # out_features
                            cfg.lora_rank,
                        ).to(self.model.device, dtype=self.model.dtype)
                        self.lora_params.extend(module.lora_module.parameters())
                        module.register_forward_hook(self.lora_forward_hook)

        # pre-cache batches (needed for storing data for KL computation)
        _bsize = self.args.per_device_train_batch_size
        self.forget_batches = [
            self.data_collator(r) for r in batched(self.train_dataset.forget, _bsize)
        ]
        self.retain_batches = [
            self.data_collator(r) for r in batched(self.train_dataset.retain, _bsize)
        ]

        self.kl_computor = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        idx = self.batch_idx % len(self.forget_batches)
        f_batch = self.forget_batches[idx]
        r_batch = self.retain_batches[idx]
        self.model.requires_grad_(False)  # train only modules that we specify

        # Lazy init KLComputor (model is guaranteed on CUDA here)
        if self.kl_computor is None and "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.retain_batches)

        # Pass A: retain momentum
        if "retain_momentum" in self.cfg and self.batch_idx >= self.recalc_every * 2:
            # # sanity check that the samples match
            # logging.info(f"RETAIN: {self.processing_class.decode(r_batch['input_ids'][0])}")
            # logging.info(f"FORGET: {self.processing_class.decode(f_batch['input_ids'][0])}")
            # logging.info(f"\n")

            model.zero_grad(set_to_none=True)
            with require_grad(self.base_trainable_params):
                kl_loss, _, _ = self.kl_computor.get_kl(r_batch)
                kl_loss.backward()
                for param in self.base_trainable_params:
                    update_ref_grad(param, self.cfg.retain_momentum)

        # Pass B: distribution collection (retain side)
        if self.cfg.use_distribution == "retain":
            model.zero_grad(set_to_none=True)
            self.token_mask = r_batch["attention_mask"].bool().clone()
            self.token_mask[:, 0] = False
            with require_grad(self.base_trainable_params):
                output = model(**prep_batch(r_batch, model.device))
            self.do_add_vecs = True
            # _loss = label_logits(output.logits, r_batch["labels"], clip=float("-inf"))
            _loss = -output.loss
            _loss.backward()
            self.do_add_vecs = False

        self.use_lora = True

        # Pass C: LoRA adversarial pass
        if "lora_lr" in self.cfg:
            model.zero_grad(set_to_none=True)
            with require_grad(self.lora_params):
                output = model(**prep_batch(f_batch, model.device))
                output.loss.backward()
            normalize_grads(self.lora_params)
            for p in self.lora_params:
                p.data -= self.cfg.lora_lr * self.args.learning_rate * p.grad

        # Pass D: forget forward+backward
        self.token_mask = f_batch["attention_mask"].bool().clone()
        self.token_mask[:, 0] = False  # omit unlearning on the BOS token
        if self.processing_class.chat_template is not None:  # omit template tokens
            for banned_token in get_banned_tokens(self.processing_class):
                self.token_mask &= f_batch["input_ids"] != banned_token

        model.zero_grad(set_to_none=True)
        with require_grad(self.base_trainable_params):
            output = model(**prep_batch(f_batch, model.device))
        self.do_collapse = True
        self.do_add_vecs = self.cfg.use_distribution == "forget"
        # forget_loss = label_logits(output.logits, f_batch["labels"])
        forget_loss = -output.loss
        # we will backpropagate because the graph has been built by the forward pass
        # but backward() itself will not compute weight gradients for base params
        # instead, weights will remain with grad computed by the collapse_hook
        forget_loss.backward()
        self.do_collapse = False
        self.do_add_vecs = False

        self.use_lora = False

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "act_collapser"):
                    module.act_collapser.fit()
                if hasattr(module, "grad_collapser"):
                    module.grad_collapser.fit()

        normalize_grads(self.base_trainable_params)
        for p in self.base_trainable_params:
            p.requires_grad_(True)  # so that the optimizer updates them
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        # if not (self.do_add_vecs or self.do_collapse):
        #     return
        module.last_act_input = args[0]

    def lora_forward_hook(self, module, args, output):
        if self.use_lora:
            return output + module.lora_module(args[0])

    def collapse_hook(self, module, grad_input, grad_output):
        if not (self.do_add_vecs or self.do_collapse):
            return
        acts = module.last_act_input.detach()
        grads = grad_output[0]
        module.last_act_input = None

        if self.is_moe:
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
        if self.do_add_vecs and "n_pcs" in self.cfg:
            module.act_collapser.add_vecs(acts)
            module.grad_collapser.add_vecs(grads)

        if not self.do_collapse:
            return
        if self.batch_idx < self.recalc_every * 2:
            return  # too early to train, so only collect activations and return early

        if "n_pcs" in self.cfg:
            acts = module.act_collapser.collapse(acts)
            grads = module.grad_collapser.collapse(grads)

        # ! KL-masking, per token and per module
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)
            disr_grad = acts @ ref_grad.T

            if self.cfg.kl_mask == "module":
                kl_mask = (disr_grad * grads).sum(dim=1) > 0
                acts = acts[kl_mask]
                grads = grads[kl_mask]
            elif self.cfg.kl_mask == "disrproj":
                # the core of DisrCollapse that can be swapped for the block above:
                disr_grad /= disr_grad.norm(dim=1, keepdim=True) + 1e-8
                projs = pt.einsum("tg,tg->t", disr_grad, grads).unsqueeze(1)
                projs = projs.clamp(max=0)
                grads -= projs * disr_grad
            else:
                raise ValueError(f"Invalid KL mask: {self.cfg.kl_mask}")

        # without acts and grads modifications, this is equivalent to normal backprop
        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
