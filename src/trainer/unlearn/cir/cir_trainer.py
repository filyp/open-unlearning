# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from transformers import TrainerCallback

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.collapsers import MahalanobisCollapser
from trainer.unlearn.cir.kl_utils import KLComputor
from trainer.utils import label_logits, normalize_grads, sanitize_tensor

logging.basicConfig(level=logging.INFO)


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        self.use_hooks = False
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place
        self.after_first_epoch = False
        self.add_callback(CalculateDistributionStatsCallback(self))

        # set trainable params
        self.model.requires_grad_(False)  # train only modules that we specify
        train_to_layer = int(len(self.model.model.layers) * cfg.train_first_layers)
        for layer_num in range(train_to_layer):
            mlp = self.model.model.layers[layer_num].mlp
            mlp.up_proj.weight.requires_grad = True
            mlp.gate_proj.weight.requires_grad = True
            mlp.up_proj.register_forward_hook(self.save_act_input_hook)
            mlp.gate_proj.register_forward_hook(self.save_act_input_hook)
            mlp.up_proj.register_full_backward_hook(self.collapse_hook)
            mlp.gate_proj.register_full_backward_hook(self.collapse_hook)

            # register collapsers, that calculate distribution stats, and later collapse
            if "act_pcs_to_use" in self.cfg:
                mlp.up_proj.act_collapser = MahalanobisCollapser(
                    cfg.act_pcs_to_use, mlp.up_proj.weight.device
                )
            # gate_proj will reuse up_proj's collapsers, so it needs a reference to it
            object.__setattr__(mlp.gate_proj, "mlp_ref", mlp)
            object.__setattr__(mlp.up_proj, "mlp_ref", mlp)
            if "grad_pcs_to_use" in self.cfg:
                mlp.down_proj.grad_collapser = MahalanobisCollapser(
                    cfg.grad_pcs_to_use, mlp.down_proj.weight.device
                )
                mlp.down_proj.register_full_backward_hook(self.grad_correction_hook)
                object.__setattr__(mlp.down_proj, "mlp_ref", mlp)

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
            for param in self.model.parameters():
                if param.requires_grad:
                    assert not hasattr(param, "ref_grad")
                    param.ref_grad = quantize_blockwise(pt.zeros_like(param))

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ! retain pass
        if "retain_momentum" in self.cfg and self.after_first_epoch:
            # we ignore the input["retain"], and instead use the cached retain batches
            r_batch = random.choice(self.retain_batches)
            model.zero_grad(set_to_none=True)
            output = model(**prep_batch(r_batch, model.device))
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    ref = dequantize_blockwise(*param.ref_grad)
                    momentum = self.cfg.retain_momentum
                    ref = ref * momentum + param.grad * (1 - momentum)
                    param.ref_grad = quantize_blockwise(ref)

        # ! unlearning loss
        batch = inputs["forget"]
        self.token_mask = batch["attention_mask"].bool().clone()
        self.token_mask[:, 0] = False  # ignore BOS token

        self.use_hooks = True
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = label_logits(output.logits, batch["labels"])
        forget_loss.backward()
        self.use_hooks = False

        for p in model.parameters():
            if hasattr(p, "tmpgrad"):
                p.grad = p.tmpgrad
                p.tmpgrad = None

        normalize_grads(model)

        if not self.after_first_epoch:
            # zero gradients so that optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if not self.use_hooks:
            return
        assert isinstance(args, tuple)
        assert len(args) == 1
        module.last_act_input = args[0]

    def collapse_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return
        acts = module.last_act_input[self.token_mask].detach()
        grads = grad_output[0][self.token_mask]
        assert acts.shape == (self.token_mask.sum(), module.weight.shape[1])
        assert grads.shape == (self.token_mask.sum(), module.weight.shape[0])
        module.last_act_input = None

        if hasattr(module, "act_collapser"):  # gate_proj doesn't have to add_vecs too
            module.act_collapser.add_vecs(acts)

        if not self.after_first_epoch:
            return  # first epoch, so only collect activations and not train

        if "act_pcs_to_use" in self.cfg:
            act_collapser = module.mlp_ref.up_proj.act_collapser
            acts = act_collapser.collapse(acts).to(module.weight.dtype)

        if "grad_pcs_to_use" in self.cfg:
            grads *= module.grad_correction
            module.grad_correction = None

        # ! KL-masking, per token and per module
        if "retain_momentum" in self.cfg:
            ref_grad = dequantize_blockwise(*module.weight.ref_grad)
            ref_grad = ref_grad.to(module.weight.dtype)
            token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
            kl_mask = token_disr > 0
            acts = acts[kl_mask]
            grads = grads[kl_mask]

        # without the projections, this is equivalent to normal backprop
        module.weight.tmpgrad = pt.einsum("ti,tj->ij", grads, acts)
        # would be possible to optimize training by disabling the first grad computation,
        # since we discard these grads anyway

    def grad_correction_hook(self, module, grad_input, grad_output):
        """Supposed to be run on down_proj. It will collapse the MLP output gradients,
        and then backpropagate them through down_proj and compare with non-collapsed input
        gradients. This way, we find out how much the up_proj and gate_proj output gradients
        need to be corrected to approximate their collapse.
        """
        if not self.use_hooks:
            return
        assert isinstance(grad_input, tuple)
        assert len(grad_input) == len(grad_output) == 1

        grad_input = grad_input[0][self.token_mask]
        grad_output = grad_output[0][self.token_mask]
        assert grad_input.shape == (self.token_mask.sum(), module.weight.shape[1])
        assert grad_output.shape == (self.token_mask.sum(), module.weight.shape[0])

        module.grad_collapser.add_vecs(grad_output)

        if not self.after_first_epoch:
            return  # first epoch, so only collect vectors and not train

        out_collapsed = module.grad_collapser.collapse(grad_output).to(
            module.weight.dtype
        )
        in_collapsed = out_collapsed @ module.weight  # backpropagation
        grad_correction = in_collapsed / sanitize_tensor(grad_input)

        module.mlp_ref.up_proj.grad_correction = grad_correction
        module.mlp_ref.gate_proj.grad_correction = grad_correction


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
