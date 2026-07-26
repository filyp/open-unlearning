# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelectSimple task_name=SAMPLE_UNLEARN
import logging

import torch as pt
from peft import LoraConfig, get_peft_model

from trainer.unlearn.base import UnlearnTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _collapse(
    mat: pt.Tensor, eig_vec: pt.Tensor, S: pt.Tensor, hard_soft: str, n_pcs: int
) -> pt.Tensor:
    projected = mat @ eig_vec

    if hard_soft == "hard":
        # top N PCs projection
        return mat - projected @ eig_vec.mT

    if hard_soft == "soft":
        # Mahalanobis collapse: divide each PC by its singular value,
        # renormalized per module so the smallest kept one maps to 1
        S = S / S.amin(dim=-1, keepdim=True)
        proj_diff = projected - projected / S.unsqueeze(-2)
        return mat - proj_diff @ eig_vec.mT

    if hard_soft == "quadratic":
        # divide by S^2, the eigenvalues of Uf.T@Uf (the act/grad covariance
        # proxy), so it applies cov^-1 rather than cov^-1/2 — equivalent to
        # whitening with the SVD of Uf.T@Uf; same per-module renormalization
        S = S**2
        S = S / S.amin(dim=-1, keepdim=True)
        proj_diff = projected - projected / S.unsqueeze(-2)
        return mat - proj_diff @ eig_vec.mT

    if hard_soft == "ridge":
        # quadratic over the full-rank SVD with Tikhonov damping: faithful
        # (cov + eps*I)^-1 with one global scale, eps = the n_pcs-th squared
        # singular value. In-span components divided by S^2+eps, the
        # out-of-span complement (incl. the part of D_out that a rectangular
        # SVD never spans) by eps, no per-module renormalization —
        # cross-module weighting is left to the raw covariance scales and
        # absorbed into the single global alpha.
        S = S**2
        eps = S[..., min(n_pcs, S.shape[-1]) - 1].unsqueeze(-1)
        S = S + eps
        proj_diff = projected - projected * (eps / S).unsqueeze(-2)
        return (mat - proj_diff @ eig_vec.mT) / eps.unsqueeze(-2)

    if hard_soft == "ridge_module_retuning":
        # same damped spectrum as ridge, but renormalized per module so the
        # tail/complement passes through unchanged and cross-module weighting
        # stays that of the raw gradient
        S = S**2
        eps = S[..., min(n_pcs, S.shape[-1]) - 1].unsqueeze(-1)
        S = S + eps
        S = S / S.amin(dim=-1, keepdim=True)
        proj_diff = projected - projected / S.unsqueeze(-2)
        return mat - proj_diff @ eig_vec.mT

    raise ValueError(f"unknown hard_soft: {hard_soft}")


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
        hard_soft="soft",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_pcs = n_pcs
        self.lora_lr = lora_lr
        self.distribution = distribution
        self.collapse_on = collapse_on
        self.use_lora = use_lora
        self.hard_soft = hard_soft
        assert distribution in ["forget", "retain"]
        assert collapse_on in ["act", "grad", "both", "none"]
        assert hard_soft in [
            "hard",
            "soft",
            "quadratic",
            "ridge",
            "ridge_module_retuning",
        ]
        # note though, that for some MoE models act and grad dimensions may be transposed

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

    def _svd(self, grad):
        if self.hard_soft.startswith("ridge"):
            # full-rank factors are ~D_min/n_pcs times larger than low-rank
            # ones, so park them on CPU until the collapse step
            U, S, Vh = pt.linalg.svd(grad, full_matrices=False)
            return U.cpu(), S.cpu(), Vh.mT.cpu()
        return pt.svd_lowrank(grad, q=self.n_pcs)

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
                weight.USV = self._svd(weight.grad.float())

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
                weight.USV = self._svd(weight.grad.float())

        # collapse
        for weight in self.base_trainable_params:
            grad = weight.grad.float()
            U, S, V = (x.to(grad.device) for x in weight.USV)

            eps = S[..., min(self.n_pcs, S.shape[-1]) - 1] ** 2
            logger.info("eps=" + " ".join(f"{e:.2e}" for e in eps.flatten().tolist()))

            if self.collapse_on in ["act", "both"]:
                grad = _collapse(grad, V, S, self.hard_soft, self.n_pcs)
            if self.collapse_on in ["grad", "both"]:
                grad = _collapse(grad.mT, U, S, self.hard_soft, self.n_pcs).mT
            weight.filtered_grad = grad.to(weight.dtype)
            weight.grad = None
            weight.USV = None

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
