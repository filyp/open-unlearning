# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=SAMPLE_UNLEARN
import logging
import math

import torch as pt

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.collapsers import InvSmallCovCollapser
from trainer.unlearn.repselect.utils import ManualLoRA
from trainer.utils import normalize_grads, npo_saturating_loss, require_grad

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

        assert not hasattr(self.model.model.layers[0].mlp, "experts")

        self.model.requires_grad_(False)  # train only modules that we specify
        self.lora_params = []
        self.base_trainable_params = []
        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        idx = self.batch_idx % len(self.forget_batches)
        f_batch = self.forget_batches[idx]
        r_batch = self.retain_batches[idx]
        self.model.requires_grad_(False)  # train only modules that we specify

        if self.cfg.lora_before_retain:
            self.use_lora = True  # todo, should lora be here or after retain pass?

        # Pass B: distribution collection (retain side)
        if self.cfg.use_distribution == "retain":
            model.zero_grad(set_to_none=True)
            # we will backpropagate because the graph has been built by the forward pass but backward() itself will not compute weight gradients for base params instead, weights will remain with grad computed by the collapse_hook
            with require_grad(self.base_trainable_params):
                output = model(**prep_batch(r_batch, model.device))
            self.do_add_vecs = True
            _loss = -output.loss
            _loss.backward()
            self.do_add_vecs = False

        if not self.cfg.lora_before_retain:
            self.use_lora = True  # should lora be here or before retain pass?

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
        model.zero_grad(set_to_none=True)
        with require_grad(self.base_trainable_params):
            output = model(**prep_batch(f_batch, model.device))
        self.do_collapse = self.batch_idx >= self.recalc_every * 2
        self.do_add_vecs = self.cfg.use_distribution == "forget"
        forget_loss = npo_saturating_loss(output, f_batch, self.cfg.npo_beta)
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

        token_mask = grads.norm(dim=-1) != 0
        # token_mask = token_mask & self.token_mask
        acts = acts[token_mask]
        grads = grads[token_mask]

        # note: we could optimize and reuse the act collapser for gate_proj and up_proj, but for simplicity don't
        if self.do_add_vecs and "n_pcs" in self.cfg:
            module.act_collapser.add_vecs(acts)
            module.grad_collapser.add_vecs(grads)

        if not self.do_collapse:
            return

        if "n_pcs" in self.cfg:
            acts = module.act_collapser.collapse(acts)
            grads = module.grad_collapser.collapse(grads)

        # without acts and grads modifications, this is equivalent to normal backprop
        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)


# # sanity check that the samples match
# logging.info(f"RETAIN: {self.processing_class.decode(r_batch['input_ids'][0])}")
# logging.info(f"FORGET: {self.processing_class.decode(f_batch['input_ids'][0])}")
# logging.info(f"\n")

# self.token_mask = get_token_mask(r_batch, self.processing_class)
# def get_token_mask(batch, processing_class):
#     token_mask = batch["attention_mask"].bool().clone()
#     token_mask[:, 0] = False  # omit unlearning on the BOS token
#     if processing_class.chat_template is not None:  # omit template tokens
#         for banned_token in get_banned_tokens(processing_class):
#             token_mask &= batch["input_ids"] != banned_token
