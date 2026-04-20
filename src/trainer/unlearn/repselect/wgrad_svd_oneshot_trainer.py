# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=WGradSVDOneShot task_name=SAMPLE_UNLEARN
import logging

import torch as pt

from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.repselect.utils import ManualLoRA

logging.basicConfig(level=logging.INFO)


def _collapse(mat: pt.Tensor, eig_vec: pt.Tensor, eig_val: pt.Tensor) -> pt.Tensor:
    projected = mat @ eig_vec
    proj_diff = projected - projected / eig_val
    return mat - proj_diff @ eig_vec.T


class WGradSVDOneShot(UnlearnTrainer):
    """
    Single-shot variant of WGradSVD:
    - Epoch 1: standard forward+backward on the forget NPO loss, per-module
      weight.grad stashed into module.grad_acc, then weight.grad cleared so the
      trainer's optimizer.step is a no-op.
    - End of epoch 1: for each module, SVD the accumulated grad_acc for U, V,
      eig_val, then collapse grad_acc itself — first along acts side (D_in,
      using V), then along grads side (D_out, using U) — and stash the result.
    - Each subsequent epoch: on the first batch, set weight.grad = stashed and
      let the trainer step the optimizer; idle the remaining batches so the
      trainer still fires evaluate() at the epoch boundary.

    Because the per-batch collapse is linear, collapsing the accumulated
    grad_acc once is equivalent to collapsing per batch and summing.
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        assert not hasattr(self.model.model.layers[0].mlp, "experts")

        self.model.requires_grad_(False)
        self.base_trainable_params = []
        self.lora_params = []
        self.use_lora = False
        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            for module in [mlp.gate_proj, mlp.up_proj, mlp.down_proj]:
                module.weight.requires_grad = True
                self.base_trainable_params.append(module.weight)

                module.lora_module = ManualLoRA(
                    module.weight.shape[1],
                    module.weight.shape[0],
                    cfg.lora_rank,
                ).to(self.model.device, dtype=self.model.dtype)
                self.lora_params.extend(module.lora_module.parameters())
                module.register_forward_hook(self._lora_forward_hook)

    def _lora_forward_hook(self, module, args, output):
        if self.use_lora:
            return output + module.lora_module(args[0])

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        # todo: simplify forget_batches iteration
        # todo: use peft lora
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()
        forget_batches = [
            self.data_collator(r)
            for r in batched(
                self.train_dataset.forget, self.args.per_device_train_batch_size
            )
        ]

        # LoRA adversarial pre-training: one epoch, SGD descent on forget NLL
        for p in self.base_trainable_params:
            p.requires_grad_(False)
        self.use_lora = True
        for batch in forget_batches:
            self.model.zero_grad(set_to_none=True)
            output = self.model(**prep_batch(batch, self.model.device))
            output.loss.backward()
            for p in self.lora_params:
                p.data -= self.cfg.lora_lr * p.grad
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        for p in self.lora_params:
            p.requires_grad_(False)

        # one epoch: accumulate forget weight-gradient with LoRA active
        self.model.zero_grad(set_to_none=True)
        for batch in forget_batches:
            output = self.model(**prep_batch(batch, self.model.device))
            forget_loss = -output.loss
            forget_loss.backward()

        self.use_lora = False

        # SVD and collapse
        for weight in self.base_trainable_params:
            raw_grad = weight.grad.float()
            U, S, V = pt.svd_lowrank(raw_grad, q=self.cfg.n_pcs)
            eig_val = S / S.min()
            filtered = _collapse(raw_grad, V, eig_val)  # filter D_in side
            filtered = _collapse(filtered.mT, U, eig_val).mT  # filter D_out side
            weight.filtered_grad = filtered.to(weight.dtype)
            weight.grad = None

        self.evaluate()
        for epoch in range(self.args.num_train_epochs):
            for weight in self.base_trainable_params:
                weight.data -= weight.filtered_grad * self.args.learning_rate

            self.state.epoch = epoch + 1
            self.evaluate()
            if self.control.should_training_stop:
                break

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )
