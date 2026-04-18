# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=Contrast task_name=SAMPLE_UNLEARN
import logging

import torch as pt

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import normalize_grads, npo_saturating_loss, require_grad

logging.basicConfig(level=logging.INFO)


class Contrast(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.collect_retain = False
        self.do_collapse = False
        self.batch_idx = 0
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place
        assert not hasattr(self.model.model.layers[0].mlp, "experts")

        self.model.requires_grad_(False)
        self.base_trainable_params = []
        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                module.weight.requires_grad = True
                self.base_trainable_params.append(module.weight)
                module.register_forward_hook(self.save_act_input_hook)
                module.register_full_backward_hook(self.collapse_hook)

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
        self.model.requires_grad_(False)

        # Calibration epoch: populate npo_saturating_loss's initial_nll for every
        # cached batch before any weight updates. Forward-only, no retain, no collapse.
        if self.batch_idx < len(self.forget_batches):
            with pt.no_grad():
                output = model(**prep_batch(f_batch, model.device))
                forget_loss = npo_saturating_loss(output, f_batch, self.cfg.npo_beta)
            self.batch_idx += 1
            return forget_loss.detach()

        # Pass A: retain forward+backward to record per-token acts and grads
        self.collect_retain = True
        with require_grad(self.base_trainable_params):
            output = model(**prep_batch(r_batch, model.device))
            (-output.loss).backward()
        self.collect_retain = False

        # Pass B: forget forward+backward with contrast-collapse
        model.zero_grad(set_to_none=True)
        with require_grad(self.base_trainable_params):
            output = model(**prep_batch(f_batch, model.device))
        self.do_collapse = True
        forget_loss = npo_saturating_loss(output, f_batch, self.cfg.npo_beta)
        forget_loss.backward()
        self.do_collapse = False

        self.batch_idx += 1

        normalize_grads(self.base_trainable_params)
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        if self.collect_retain:
            module.retain_acts = args[0].detach()
        else:
            module.last_act_input = args[0].detach()

    def collapse_hook(self, module, grad_input, grad_output):
        if self.collect_retain:
            module.retain_grads = grad_output[0].detach()
            return
        if not self.do_collapse:
            return
        acts = module.last_act_input  # (B, T_f, D_in)
        grads = grad_output[0]  # (B, T_f, D_out)
        module.last_act_input = None
        retain_acts = module.retain_acts  # (B, T_r, D_in)
        module.retain_acts = None
        retain_grads = module.retain_grads  # (B, T_r, D_out)
        module.retain_grads = None

        forget_mask = grads.norm(dim=-1) != 0  # (B, T_f)
        retain_acts_mask = retain_acts.norm(dim=-1) > 0  # (B, T_r)
        retain_grads_mask = retain_grads.norm(dim=-1) > 0  # (B, T_r)
        eps = 1e-12

        collapsed_acts = self._top1_collapse(acts, retain_acts, retain_acts_mask, eps)
        collapsed_grads = self._top1_collapse(grads, retain_grads, retain_grads_mask, eps)

        # zero out masked forget tokens so they don't contribute to the weight grad
        collapsed_acts = collapsed_acts * forget_mask.unsqueeze(-1)
        collapsed_grads = collapsed_grads * forget_mask.unsqueeze(-1)

        module.weight.grad = pt.einsum("bti,btj->ij", collapsed_grads, collapsed_acts)

    @staticmethod
    def _top1_collapse(x, retain, retain_mask, eps):
        # For each token in x, find the retain token (same sample) with max dot product,
        # then remove the unit-vector projection of x onto that matched retain vector.
        sim = pt.einsum("btd,bsd->bts", x, retain)  # (B, T_x, T_r)
        sim = sim.masked_fill(~retain_mask.unsqueeze(1), float("-inf"))
        idx = sim.argmax(dim=-1)  # (B, T_x)
        D = retain.shape[-1]
        matched = retain.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))  # (B, T_x, D)
        m_hat = matched / matched.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x - (x * m_hat).sum(dim=-1, keepdim=True) * m_hat
