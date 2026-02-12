# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import re

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from transformers import TrainerCallback

import trainer.unlearn.cir.hooks as hooks
from data.utils import prep_batch
from trainer.utils import label_logits
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import (
    PreCachingDataLoader,
    normalize_grads,
)
from trainer.unlearn.cir.collapsers import MahalanobisCollapser
from trainer.unlearn.cir.kl_utils import KLComputor

logging.basicConfig(level=logging.INFO)


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


class CIR_MoE(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.after_first_epoch = False
        assert self.args.gradient_accumulation_steps == 1  # we modify grads in-place

        # set trainable params
        self.model.requires_grad_(False)  # to be sure bias params are not trained
        train_to_layer = int(len(self.model.model.layers) * cfg.train_first_layers)
        for name, module in self.model.named_modules():
            if not hasattr(module, "weight"):
                # logging.info(f"Skipping {name} because it doesn't have a weight")
                continue
            module.weight.requires_grad = (
                any(pattern in name for pattern in cfg.target_modules)
                and layer_num(name) < train_to_layer
            )
            if module.weight.requires_grad:
                # install hooks
                module.register_forward_hook(hooks.save_act_input)
                module.register_full_backward_hook(hooks.save_grad_output)
                module.last_act_input = None
                module.last_grad_output = None
                # install collapsers
                if "act_pcs_to_use" in self.cfg:
                    pass
                if "grad_pcs_to_use" in self.cfg:
                    module.grad_collapser = MahalanobisCollapser(
                        cfg.grad_pcs_to_use, module.weight.device
                    )

        self.add_callback(CalculateDistributionStatsCallback(self))

        # pre-cache batches (handy for storing batch-related data later)
        self.batches = PreCachingDataLoader(
            self.train_dataset,
            self.data_collator,
            self.args.per_device_train_batch_size,
        )

        # ! prepare retain
        if "retain_momentum" in self.cfg:
            self.kl_computor = KLComputor(self.model, self.batches.retain)
            for param in self.model.parameters():
                if param.requires_grad:
                    assert not hasattr(param, "ref_grad")
                    param.ref_grad = quantize_blockwise(pt.zeros_like(param))

    def get_train_dataloader(self):
        """Return dataloader over pre-batched forget/retain pairs."""
        return self.batches

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # ! retain pass
        if "retain_momentum" in self.cfg and self.after_first_epoch:
            r_batch = inputs["retain"]
            model.zero_grad(set_to_none=True)
            output = model(**prep_batch(r_batch, model.device))
            kl, _, _ = self.kl_computor.get_kl(r_batch)
            kl.backward()
            for param in self.model.parameters():
                if param.requires_grad:
                    ref = dequantize_blockwise(*param.ref_grad)
                    ref *= self.cfg.retain_momentum
                    ref += param.grad * (1 - self.cfg.retain_momentum)
                    param.ref_grad = quantize_blockwise(ref)

        # ! unlearning loss
        batch = inputs["forget"]
        token_mask = batch["attention_mask"].bool().clone()
        token_mask[:, 0] = False  # ignore BOS token
        # todo, implement token masking for moe

        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = label_logits(output.logits, batch["labels"])
        forget_loss.backward()

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            # acts = module.last_act_input[token_mask].detach()
            # grads = module.last_grad_output[token_mask].detach()
            # assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            # assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            if module.last_act_input is None or module.last_grad_output is None:
                # some experts may be never chosen, and so have no acts and grads
                continue

            acts = module.last_act_input.detach()
            grads = module.last_grad_output.detach()
            module.last_act_input = None
            module.last_grad_output = None
            # we need to cast, because sometimes the router causes upcast to float32
            acts = acts.to(model.dtype)
            grads = grads.to(model.dtype)

            # if "act_pcs_to_use" in self.cfg:
            #     if name.endswith(".up_proj"):
            #         self.act_collapsers[name].add_vecs(acts)
            if "grad_pcs_to_use" in self.cfg:
                module.grad_collapser.add_vecs(grads)

            if not self.after_first_epoch:
                continue  # so only collect activations and not train

            # if "act_pcs_to_use" in self.cfg:
            #     # gate_proj shares inputs with up_proj, so we can use up_proj's collapser
            #     _up_proj_name = name.replace(".gate_proj", ".up_proj")
            #     acts = self.act_collapsers[_up_proj_name].collapse(acts).to(model.dtype)
            if "grad_pcs_to_use" in self.cfg:
                if not hasattr(module.grad_collapser, "mean"):
                    breakpoint()
                grads = module.grad_collapser.collapse(grads).to(model.dtype)

            # ! MUDMAN-like operation
            if "retain_momentum" in self.cfg:
                ref_grad = dequantize_blockwise(*module.weight.ref_grad).to(model.dtype)
                token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
                kl_mask = token_disr > 0
                acts = acts[kl_mask]
                grads = grads[kl_mask]

            # if acts.dtype != grads.dtype:
            #     breakpoint()
            # without the projections, this is equivalent to normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
            # would be possible to optimize training by disabling the first grad computation,
            # since we discard these grads anyway

        normalize_grads(model)

        if not self.after_first_epoch:
            # zero gradients so that optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()


# def parent_mlp_name(name):
#     parent_name = name.rsplit(".", 1)[0]
#     assert parent_name.endswith(".mlp")
#     return parent_name


def layer_num(module_name):
    return int(re.search(r"\.layers\.(\d+)\.", module_name).group(1))


# # minimal steps to run:
# model = AutoModelForCausalLM.from_pretrained(
#     cfg.model_id, torch_dtype=pt.bfloat16
# )
# trainer = CIR(model=model, train_dataset=train_dataset)
# trainer.train()


# # ref_grad = module.weight.ref_grad
# for ref_grad in module.weight.ref_grad:
#     # gate_proj shares inputs with up_proj, so we can use up_proj's collapser
#     _up_proj_name = name.replace(".gate_proj", ".up_proj")
#     collapser = self.act_collapsers[_up_proj_name]
#     centered = acts - collapser.mean.to(model.dtype)
#     eig_vec = collapser.eig_vec.to(model.dtype)
#     # t: tokens, d: dimension of residual stream, c: components
#     projected_acts = pt.einsum("td,dc->tc", centered, eig_vec)
#     # h: hidden dimension of the MLP
#     act_disruptions = pt.einsum("th,hd->td", grads, ref_grad)
#     projected_disrs = pt.einsum("td,dc->tc", act_disruptions, eig_vec)
#     mask_out = projected_acts.sign() != projected_disrs.sign()
#     projected_acts[mask_out] = 0
#     acts = pt.einsum("tc,dc->td", projected_acts, eig_vec)  # p @ eig_vec.T


# if "disr_momentum" in self.cfg:
#     if "token_disr_acc" not in batch:
#         batch["token_disr_acc"] = {}
#     if name not in batch["token_disr_acc"]:
#         batch["token_disr_acc"][name] = pt.zeros_like(token_disr)
#     batch["token_disr_acc"][name] *= self.cfg.disr_momentum
#     batch["token_disr_acc"][name] += token_disr * (1 - self.cfg.disr_momentum)
#     mask = batch["token_disr_acc"][name] > 0
# else:

# # for calculating distribution stats on the RETAIN set
# token_mask = r_batch["attention_mask"].bool().clone()
# token_mask[:, 0] = False  # ignore BOS token
# if "act_pcs_to_use" in self.cfg and self.cfg.distribution == "retain":
#     for name, module in model.named_modules():
#         if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
#             continue
#         acts = module.last_act_input[token_mask].detach()
#         # grads = module.last_grad_output[token_mask].detach()
#         if name.endswith(".up_proj"):
#             self.act_collapsers[name].add_vecs(acts)

# # per-token CE loss [B, S], position i = loss for predicting token i
# with pt.no_grad():
#     logits = output.logits[:, :-1]
#     labels = batch["labels"][:, 1:].to(logits.device)
#     ce = F.cross_entropy(
#         logits.reshape(-1, logits.size(-1)),
#         labels.reshape(-1),
#         ignore_index=-100,
#         reduction="none",
#     ).reshape(logits.shape[:2])
#     per_token_loss = pt.zeros(batch["input_ids"].shape, device=ce.device)
#     per_token_loss[:, 1:] = ce
# if "initial_token_loss" not in batch:
#     batch["initial_token_loss"] = per_token_loss.cpu()
# token_loss_delta = per_token_loss.cpu() - batch["initial_token_loss"]

# if "initial_label_logits" not in batch:
#     assert not self.after_first_epoch, "epoch number != 0"
#     batch["initial_label_logits"] = loss_fns.get_label_logits(
#         output, batch
#     ).detach()  # .cpu()
# forget_loss = loss_fns.saturating_logits(
#     output.logits, batch["labels"], batch["initial_label_logits"], self.cfg.sat_speed
# )


# def create_optimizer(self):
#     params = [p for p in self.model.parameters() if p.requires_grad]
#     self.optimizer = pt.optim.SGD(
#         params,
#         lr=self.args.learning_rate,
#         momentum=self.cfg.get("sgd_momentum", 0.0),
#     )
#     return self.optimizer
# # sgd_momentum: 0.9


# per_seq_losses = []
# for i in range(output.logits.shape[0]):
#     s_logits = output.logits[i].unsqueeze(0)
#     s_labels = batch["labels"][i].unsqueeze(0)
#     s_forget_loss = loss_fns.label_logits(s_logits, s_labels)
#     per_seq_losses.append(s_forget_loss)
# per_seq_losses = pt.stack(per_seq_losses)
# if not self.after_first_epoch:
#     assert "init_seq_losses" not in batch
#     batch["init_seq_losses"] = per_seq_losses.detach()
# diff = per_seq_losses - batch["init_seq_losses"]
# sat_speed = 0.1
# unlearning_saturations = -F.logsigmoid(-sat_speed * diff) / sat_speed
# forget_loss = unlearning_saturations.mean()
