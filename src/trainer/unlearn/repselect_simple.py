# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelectSimple task_name=SAMPLE_UNLEARN
import logging

import torch as pt
from peft import LoraConfig, get_peft_model

from trainer.unlearn.base import UnlearnTrainer

logging.basicConfig(level=logging.INFO)


def _collapse(mat: pt.Tensor, eig_vec: pt.Tensor, eig_val: pt.Tensor) -> pt.Tensor:
    projected = mat @ eig_vec
    proj_diff = projected - projected / eig_val.unsqueeze(-2)
    return mat - proj_diff @ eig_vec.mT


def _prep_batch(batch):
    return {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}


def _train_on(params, model):
    model.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)


class RepSelectSimple(UnlearnTrainer):
    """
    Single-shot variant of WGradSVD, over MLP gate/up/down projections:
    1. Adversarial LoRA pretrain: freeze base, SGD-descent LoRA on forget NLL.
    2. Freeze LoRA, accumulate forget weight-gradient over one pass (LoRA
       still active in forward).
    3. Unload LoRA, SVD the weight-gradient of the chosen `distribution`
       ("forget" or "retain"; "none" skips collapse), collapse its top
       principal components on both D_in (via V) and D_out (via U).
    4. Each training epoch: weight -= filtered_grad * lr, then evaluate.
    """

    def __init__(
        self,
        n_pcs,
        lora_lr,
        distribution="forget",
        collapse_on="both",
        use_lora=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_pcs = n_pcs
        self.lora_lr = lora_lr
        self.distribution = distribution
        self.collapse_on = collapse_on
        self.use_lora = use_lora
        assert distribution in ["forget", "retain"]
        assert collapse_on in ["act", "grad", "both", "none"]

        is_moe = any(hasattr(layer.mlp, "experts") for layer in self.model.model.layers)
        if is_moe:
            lora_config = LoraConfig(target_parameters=["mlp.experts.gate_up_proj"])
            self.model = get_peft_model(self.model, lora_config)
            self.base_trainable_params = [
                layer.mlp.experts.base_layer.gate_up_proj
                for layer in self.model.base_model.model.model.layers
                if hasattr(layer.mlp, "experts")
            ]
        else:
            lora_config = LoraConfig(
                target_modules=["gate_proj", "up_proj", "down_proj"]
            )
            self.model = get_peft_model(self.model, lora_config)
            self.base_trainable_params = [
                module.base_layer.weight
                for layer in self.model.base_model.model.model.layers
                for module in [
                    layer.mlp.gate_proj,
                    layer.mlp.up_proj,
                    layer.mlp.down_proj,
                ]
            ]

        self.lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n]

        if self.collapse_on == "none":
            # when not collapsing, interventions are much more disrputive, so adjust LR to keep the sweeper range valid
            self.args.learning_rate /= 500

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.model = self.accelerator.prepare(self.model)
        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )
        self.model.train()

        # retain epoch
        if self.distribution == "retain":
            self.model.zero_grad(set_to_none=True)
            _train_on(self.base_trainable_params, self.model)
            for batch_pair in self.get_train_dataloader():
                r_batch = _prep_batch(batch_pair["retain"])
                output = self.model(**r_batch)
                (-output.loss).backward()
            # retain SVD
            for weight in self.base_trainable_params:
                weight.USV = pt.svd_lowrank(weight.grad.float(), q=self.n_pcs)

        # LoRA adversarial pre-training: one epoch, SGD descent on forget NLL
        if self.use_lora:  # toggle for ablations
            _train_on(self.lora_params, self.model)
            for batch_pair in self.get_train_dataloader():
                self.model.zero_grad(set_to_none=True)
                f_batch = _prep_batch(batch_pair["forget"])
                output = self.model(**f_batch)
                output.loss.backward()
                for p in self.lora_params:
                    p.data -= self.lora_lr * p.grad

        # one epoch: accumulate forget weight-gradient with LoRA active
        self.model.zero_grad(set_to_none=True)
        _train_on(self.base_trainable_params, self.model)
        for batch_pair in self.get_train_dataloader():
            f_batch = _prep_batch(batch_pair["forget"])
            output = self.model(**f_batch)
            (-output.loss).backward()

        # strip LoRA
        self.model = self.model.unload()

        # forget SVD
        if self.distribution == "forget":
            for weight in self.base_trainable_params:
                weight.USV = pt.svd_lowrank(weight.grad.float(), q=self.n_pcs)

        # collapse
        for weight in self.base_trainable_params:
            grad = weight.grad.float()
            U, S, V = weight.USV
            eig_val = S / S.amin(dim=-1, keepdim=True)
            if self.collapse_on in ["act", "both"]:
                grad = _collapse(grad, V, eig_val)  # filter D_in side
            if self.collapse_on in ["grad", "both"]:
                grad = _collapse(grad.mT, U, eig_val).mT  # filter D_out side
            weight.filtered_grad = grad.to(weight.dtype)
            weight.grad = None

        self._apply_unlearn_loop()

        self.control = self.callback_handler.on_train_end(
            self.args, self.state, self.control
        )

    def _apply_unlearn_loop(self):
        # perform dummy epochs, simply applying the filtered gradient
        self.evaluate()
        for epoch in range(self.args.num_train_epochs):
            for weight in self.base_trainable_params:
                weight.data -= weight.filtered_grad * self.args.learning_rate

            self.state.epoch = epoch + 1
            self.evaluate()
            if self.control.should_training_stop:
                break
