# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=WGradSVDJoint task_name=SAMPLE_UNLEARN
import logging
import math

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.collapsers import WeightGradSVDCollapser
from trainer.utils import normalize_grads, npo_saturating_loss, require_grad

logging.basicConfig(level=logging.INFO)


class WGradSVDJoint(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.do_accumulate = False
        self.do_collapse = False
        self.batch_idx = 0
        self.recalc_every = math.ceil(
            len(self.train_dataset) / self.args.per_device_train_batch_size
        )
        logging.info(f"{self.recalc_every=}")
        assert self.args.gradient_accumulation_steps == 1

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

                module.wgrad_collapser = WeightGradSVDCollapser(cfg.n_pcs)

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

        is_warmup_batch = self.batch_idx < self.recalc_every

        # Pass B: retain-side weight-gradient accumulation
        if self.cfg.use_distribution == "retain":
            model.zero_grad(set_to_none=True)
            with require_grad(self.base_trainable_params):
                r_output = model(**prep_batch(r_batch, model.device))
            self.do_accumulate = True
            r_output.loss.backward()
            self.do_accumulate = False

        # Pass D: forget-side backward + joint collapse + optimizer step
        model.zero_grad(set_to_none=True)
        with require_grad(self.base_trainable_params):
            output = model(**prep_batch(f_batch, model.device))
        self.do_accumulate = self.cfg.use_distribution == "forget"
        self.do_collapse = not is_warmup_batch
        forget_loss = npo_saturating_loss(output, f_batch, self.cfg.npo_beta)
        forget_loss.backward()
        self.do_accumulate = False
        self.do_collapse = False

        self.batch_idx += 1
        if self.batch_idx % self.recalc_every == 0:
            for module in model.modules():
                if hasattr(module, "wgrad_collapser"):
                    module.wgrad_collapser.fit()

        if is_warmup_batch:
            for p in self.base_trainable_params:
                p.grad = None
        else:
            normalize_grads(self.base_trainable_params)

        for p in self.base_trainable_params:
            p.requires_grad_(True)
        return forget_loss.detach()

    def save_act_input_hook(self, module, args, output):
        module.last_act_input = args[0]

    def collapse_hook(self, module, grad_input, grad_output):
        if not (self.do_accumulate or self.do_collapse):
            return
        acts = module.last_act_input.detach()
        grads = grad_output[0]
        module.last_act_input = None

        token_mask = grads.norm(dim=-1) != 0
        acts = acts[token_mask]
        grads = grads[token_mask]

        if self.do_accumulate:
            module.wgrad_collapser.add_weight_grad(acts, grads)

        if not self.do_collapse:
            return

        module.weight.grad = module.wgrad_collapser.collapse_joint(acts, grads).to(
            module.weight.dtype
        )
