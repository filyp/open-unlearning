# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.collapsers import IncrementalPCACollapser, MahalanobisCollapser
from trainer.unlearn.cir.kl_utils import KLComputor
from trainer.utils import label_logits, no_weight_grads, normalize_grads

logging.basicConfig(level=logging.INFO)


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.use_hooks = False
        self.batch_idx = 0
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place

        # set trainable params
        self.model.requires_grad_(False)  # train only modules that we specify
        train_to_layer = int(len(self.model.model.layers) * cfg.train_first_layers)

        self.is_moe = hasattr(self.model.model.layers[0].mlp, "experts")

        for layer_num in range(train_to_layer):
            mlp = self.model.model.layers[layer_num].mlp
            experts = mlp.experts if self.is_moe else [mlp]
            for expert in experts:
                for module in [expert.gate_proj, expert.up_proj]:
                    module.weight.requires_grad = True

                    # install hooks
                    module.register_forward_hook(self.save_act_input_hook)
                    module.register_full_backward_hook(self.collapse_hook)

                    # install collapsers
                    if "act_pcs_to_use" in cfg:
                        # module.act_collapser = MahalanobisCollapser(
                        module.act_collapser = IncrementalPCACollapser(
                            cfg.act_pcs_to_use, self.model.device
                        )
                    if "grad_pcs_to_use" in cfg:
                        # module.grad_collapser = MahalanobisCollapser(
                        module.grad_collapser = IncrementalPCACollapser(
                            cfg.grad_pcs_to_use, self.model.device
                        )

                    # register latent attack hooks
                    if "latent_attack_strength" in cfg:
                        module.register_forward_hook(self.latent_attack_hook)
                        module.register_full_backward_hook(self.prep_latent_attack_hook)

        # ! prepare retain
        if "retain_momentum" in self.cfg:
            # pre-cache retain batches (needed for storing data for KL computation)
            self.retain_batches = [
                self.data_collator(r)
                for r in batched(
                    self.train_dataset.retain, self.args.per_device_train_batch_size
                )
            ]
            self.kl_computor = KLComputor(self.model, self.retain_batches)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ! retain pass
        if "retain_momentum" in self.cfg and self.batch_idx >= self.cfg.recompute_every:
            # we ignore the input["retain"], and instead use the cached retain batches
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in model.parameters():
                if param.requires_grad:
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
        self.token_mask[:, 0] = False  # ignore BOS token

        self.use_hooks = True
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = label_logits(output.logits, batch["labels"])
        with no_weight_grads(model):
            # we will backpropagate because the graph has been built by the forward pass
            # but backward() itself will not compute weight gradients
            # instead, weights will remain with grad computed by the collapse_hook
            forget_loss.backward()
        self.use_hooks = False

        self.batch_idx += 1
        if self.batch_idx % self.cfg.recompute_every == 0:
            for module in model.modules():
                if hasattr(module, "act_collapser"):
                    module.act_collapser.process_saved_vecs()
                if hasattr(module, "grad_collapser"):
                    module.grad_collapser.process_saved_vecs()

        normalize_grads(model)
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
            token_mask = grads.norm(dim=1) != 0
            acts = acts[token_mask]
            grads = grads[token_mask]
            if acts.shape[0] == 0:
                # this expert wasn't selected for any tokens
                return
        else:
            acts = acts[self.token_mask]
            grads = grads[self.token_mask]

        if "act_pcs_to_use" in self.cfg:
            module.act_collapser.add_vecs(acts)
        if "grad_pcs_to_use" in self.cfg:
            module.grad_collapser.add_vecs(grads)

        if self.batch_idx < self.cfg.recompute_every:
            return  # not initialized yet, so only collect activations and not train

        if "act_pcs_to_use" in self.cfg:
            acts = module.act_collapser.collapse(acts).to(module.weight.dtype)
        if "grad_pcs_to_use" in self.cfg:
            grads = module.grad_collapser.collapse(grads).to(module.weight.dtype)

        # # we need to cast, because sometimes the router causes upcast to float32
        # # but maybe with new MoEs that doesn't happen, so comment out for now
        # acts = acts.to(module.weight.dtype)
        # grads = grads.to(module.weight.dtype)

        # ! KL-masking, per token and per module
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)
            token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
            kl_mask = token_disr > 0
            acts = acts[kl_mask]
            grads = grads[kl_mask]

        # without acts and grads modifications, this is equivalent to normal backprop
        module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)

    def latent_attack_hook(self, module, args, output):
        if not self.use_hooks:
            return
        if not hasattr(module, "attack"):
            return
        normalized_attack = module.attack / module.attack.norm()
        attack_norm = output.norm(dim=-1).mean() * self.cfg.latent_attack_strength
        output = output + normalized_attack * attack_norm
        return output

    def prep_latent_attack_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return
        grads = grad_output[0][self.token_mask]
        module.attack = grads.mean(dim=0)
