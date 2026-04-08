# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=SAMPLE_UNLEARN
import logging
import math
import types

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise

from data.utils import batched, prep_batch
from evals.kl_eval import KLComputor
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.collapsers import BatchedInvSmallCovCollapser
from trainer.unlearn.repselect.moe_patch import (
    Identity,
    hooked_grouped_mm_experts_forward,
)
from trainer.unlearn.repselect.utils import (  # noqa: F401
    BatchedManualLoRA,
    get_banned_tokens,
)
from trainer.utils import normalize_grads

logging.basicConfig(level=logging.INFO)


class RepSelectMOE(UnlearnTrainer):
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
        assert hasattr(self.model.model.layers[0].mlp, "experts")
        assert (
            getattr(self.model.config, "_experts_implementation", None) == "grouped_mm"
        ), "RepSelectMOE requires experts_implementation='grouped_mm'"

        self.model.requires_grad_(False)  # train only modules that we specify
        self.base_trainable_params = []
        self.lora_params = []
        for layer_num in range(len(self.model.model.layers)):
            e = self.model.model.layers[layer_num].mlp.experts

            for param in [e.gate_up_proj, e.down_proj]:
                param.requires_grad = True
                self.base_trainable_params.append(param)

            # Add Identity probes and patch forward
            e.gate_up_output_probe = Identity()
            e.down_output_probe = Identity()
            e.forward = types.MethodType(hooked_grouped_mm_experts_forward, e)

            # Install backward hooks on probes
            e.gate_up_output_probe._proj_type = "gate_up"
            e.down_output_probe._proj_type = "down"
            e.gate_up_output_probe._is_transposed = e.is_transposed
            e.down_output_probe._is_transposed = e.is_transposed
            e.gate_up_output_probe.register_full_backward_hook(self.collapse_hook)
            e.down_output_probe.register_full_backward_hook(self.collapse_hook)

            # Initialize batched collapsers and stash on probes
            if "n_pcs" in cfg:
                n, E = cfg.n_pcs, e.num_experts
                collapser_class = BatchedInvSmallCovCollapser
                e.gate_up_output_probe.act_collapser = collapser_class(n, E)
                e.gate_up_output_probe.grad_collapser = collapser_class(n, E)
                e.down_output_probe.act_collapser = collapser_class(n, E)
                e.down_output_probe.grad_collapser = collapser_class(n, E)

            # adversarial LoRA (per-expert, batched via _grouped_mm)
            if "lora_lr" in cfg:
                E = e.num_experts
                # is_transposed=True:  weight (E, in_dim, out_dim) → in=shape[-2], out=shape[-1]
                # is_transposed=False: weight (E, out_dim, in_dim) → in=shape[-1], out=shape[-2]
                if e.is_transposed:
                    gu_in, gu_out = e.gate_up_proj.shape[-2], e.gate_up_proj.shape[-1]
                    dn_in, dn_out = e.down_proj.shape[-2], e.down_proj.shape[-1]
                else:
                    gu_in, gu_out = e.gate_up_proj.shape[-1], e.gate_up_proj.shape[-2]
                    dn_in, dn_out = e.down_proj.shape[-1], e.down_proj.shape[-2]
                dev, dtype = self.model.device, self.model.dtype
                e.gate_up_output_probe.lora_module = BatchedManualLoRA(
                    E, gu_in, gu_out, cfg.lora_rank
                ).to(dev, dtype)
                e.down_output_probe.lora_module = BatchedManualLoRA(
                    E, dn_in, dn_out, cfg.lora_rank
                ).to(dev, dtype)
                self.lora_params.extend(e.gate_up_output_probe.lora_module.parameters())
                self.lora_params.extend(e.down_output_probe.lora_module.parameters())
                e.gate_up_output_probe.register_forward_hook(self.lora_forward_hook)
                e.down_output_probe.register_forward_hook(self.lora_forward_hook)

        # pre-cache batches (needed for storing data for KL computation)
        self.forget_batches = [
            self.data_collator(r)
            for r in batched(
                self.train_dataset.forget, self.args.per_device_train_batch_size
            )
        ]
        self.retain_batches = [
            self.data_collator(r)
            for r in batched(
                self.train_dataset.retain, self.args.per_device_train_batch_size
            )
        ]

        # KLComputor is deferred to first training_step,
        # because the model may not be on CUDA yet during __init__)
        self.kl_computor = None


    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        idx = self.batch_idx % len(self.forget_batches)
        f_batch = self.forget_batches[idx]
        r_batch = self.retain_batches[idx]

        # Lazy init KLComputor (model is guaranteed on CUDA here)
        if self.kl_computor is None and "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.retain_batches)

        # Retain pass
        if "retain_momentum" in self.cfg and self.batch_idx >= self.recalc_every * 2:
            model.zero_grad(set_to_none=True)
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.base_trainable_params:
                if hasattr(param, "ref_grad"):
                    ref = dequantize_blockwise(*param.ref_grad)
                else:
                    ref = pt.zeros_like(param)
                if param.grad is not None:
                    momentum = self.cfg.retain_momentum
                    ref = ref * momentum + param.grad * (1 - momentum)
                param.ref_grad = quantize_blockwise(ref)

        # Unlearning loss
        # token_mask = f_batch["attention_mask"].bool().clone()
        # token_mask[:, 0] = False  # omit BOS token
        # if self.processing_class.chat_template is not None:
        #     for banned_token in get_banned_tokens(self.processing_class):
        #         token_mask &= f_batch["input_ids"] != banned_token
        # self.flat_token_mask = token_mask.reshape(-1)

        self.use_hooks = True
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(f_batch, model.device))
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

        # update LoRA adversarially (gradient ascent - adversary tries to relearn)
        if self.lora_params:
            normalize_grads(self.lora_params)
            for p in self.lora_params:
                p.data += self.cfg.lora_lr * self.args.learning_rate * p.grad
                p.grad = None

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for m in model.modules():
                if isinstance(m, Identity) and hasattr(m, "act_collapser"):
                    m.act_collapser.fit()
                if isinstance(m, Identity) and hasattr(m, "grad_collapser"):
                    m.grad_collapser.fit()

        normalize_grads(self.base_trainable_params)
        return forget_loss.detach()

    def lora_forward_hook(self, module, args, output):
        if self.use_hooks:
            return output + module.lora_module(module._acts, module._offsets)

    def collapse_hook(self, module, grad_input, grad_output):
        if not self.use_hooks:
            return

        acts_sorted = module._acts.detach()  # (S, in_dim), stashed during forward
        fused_param = module._fused_param[0]  # unwrap from list (hidden from nn.Module)
        offsets = module._offsets  # (num_experts,) cumulative counts
        num_experts = offsets.shape[0]
        is_transposed = module._is_transposed

        grads_sorted = grad_output[0]  # (S, out_dim)
        module._acts = None  # free reference

        # Filter tokens globally: non-zero grads AND token_mask (BOS, padding, template)
        keep = grads_sorted.norm(dim=1) != 0
        # if hasattr(self, 'flat_token_mask'):
        #     keep = keep & self.flat_token_mask[module._token_idx_sorted]
        acts_sorted = acts_sorted[keep]
        grads_sorted = grads_sorted[keep]

        # Recompute offsets after filtering
        # Build per-token expert_ids from cumulative offsets, then filter and recount
        # NOTE: right=True is required in all bucketize calls
        assert offsets[-1] == keep.shape[0]
        expert_ids = pt.bucketize(
            pt.arange(keep.shape[0], device=keep.device), offsets, right=True
        )
        expert_ids = expert_ids[keep]
        tokens_per_expert = pt.bincount(expert_ids, minlength=num_experts)
        offsets = pt.cumsum(tokens_per_expert, dim=0, dtype=pt.int32)
        assert offsets[-1] == acts_sorted.shape[0]

        module.act_collapser.add_vecs(acts_sorted, offsets)
        module.grad_collapser.add_vecs(grads_sorted, offsets)

        if self.batch_idx < self.recalc_every * 2:
            return

        # Batched collapse via _grouped_mm (replaces per-expert loop)
        acts_sorted = module.act_collapser.collapse(acts_sorted, offsets)
        grads_sorted = module.grad_collapser.collapse(grads_sorted, offsets)

        # KL-masking (zero out instead of selecting, to preserve offsets)
        if "retain_momentum" in self.cfg:
            # Dequantize ref_grad once for all experts (instead of per-expert)
            ref_grad = dequantize_blockwise(*fused_param.ref_grad).to(fused_param.dtype)
            if is_transposed:
                ref_grad = ref_grad.transpose(-2, -1)  # (E, out_dim, in_dim)
            ends = offsets.tolist()
            starts = [0] + ends[:-1]
            token_disr = pt.zeros(acts_sorted.shape[0], device=acts_sorted.device)
            for expert_idx in range(num_experts):
                start, end = starts[expert_idx], ends[expert_idx]
                if start == end:
                    continue
                token_disr[start:end] = pt.einsum(
                    "ij,ti,tj->t",
                    ref_grad[expert_idx],
                    grads_sorted[start:end],
                    acts_sorted[start:end],
                )

            zero_mask = token_disr <= 0
            acts_sorted[zero_mask] = 0
            grads_sorted[zero_mask] = 0

        # Single _grouped_mm call: aggregates per-expert weight gradients via CUTLASS
        if is_transposed:
            fused_param.grad = pt._grouped_mm(acts_sorted.T, grads_sorted, offs=offsets)
        else:
            fused_param.grad = pt._grouped_mm(grads_sorted.T, acts_sorted, offs=offsets)
