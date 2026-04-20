# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=WGradSVDOneShot task_name=SAMPLE_UNLEARN
import logging

import torch as pt
from peft import LoraConfig, get_peft_model

from trainer.unlearn.base import UnlearnTrainer

logging.basicConfig(level=logging.INFO)


def _collapse(mat: pt.Tensor, eig_vec: pt.Tensor, eig_val: pt.Tensor) -> pt.Tensor:
    projected = mat @ eig_vec
    proj_diff = projected - projected / eig_val
    return mat - proj_diff @ eig_vec.T


def _prep_batch(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}


class RepSelectSimple(UnlearnTrainer):
    """
    Single-shot variant of WGradSVD, over MLP gate/up/down projections:
    1. Adversarial LoRA pretrain: freeze base, SGD-descent LoRA on forget NLL.
    2. Freeze LoRA, accumulate forget weight-gradient over one pass (LoRA
       still active in forward).
    3. Unload LoRA, SVD the accumulated grad, collapse its top principal
       components on both D_in (via V) and D_out (via U).
    4. Each training epoch: weight -= filtered_grad * lr, then evaluate.
    """

    # todo implement moe support
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        assert not hasattr(self.model.model.layers[0].mlp, "experts")

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)

        self.base_trainable_params = []
        for layer in self.model.base_model.model.model.layers:
            for module in [layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]:
                self.base_trainable_params.append(module.base_layer.weight)
        self.lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()

        # LoRA adversarial pre-training: one epoch, SGD descent on forget NLL
        self.model.requires_grad_(False)
        for p in self.lora_params:
            p.requires_grad_(True)
        for batch_pair in self.get_train_dataloader():
            self.model.zero_grad(set_to_none=True)
            f_batch = _prep_batch(batch_pair["forget"])
            output = self.model(**f_batch)
            output.loss.backward()
            for p in self.lora_params:
                p.data -= self.cfg.lora_lr * p.grad

        # one epoch: accumulate forget weight-gradient with LoRA active
        self.model.zero_grad(set_to_none=True)
        self.model.requires_grad_(False)
        for p in self.base_trainable_params:
            p.requires_grad_(True)
        for batch_pair in self.get_train_dataloader():
            f_batch = _prep_batch(batch_pair["forget"])
            output = self.model(**f_batch)
            (-output.loss).backward()

        # strip LoRA
        self.model = self.model.unload()

        # SVD and collapse
        for weight in self.base_trainable_params:
            raw_grad = weight.grad.float()
            U, S, V = pt.svd_lowrank(raw_grad, q=self.cfg.n_pcs)
            eig_val = S / S.min()
            filtered = _collapse(raw_grad, V, eig_val)  # filter D_in side
            filtered = _collapse(filtered.mT, U, eig_val).mT  # filter D_out side
            weight.filtered_grad = filtered.to(weight.dtype)
            weight.grad = None

        # perform dummy epochs, simply applying the filtered gradient
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
