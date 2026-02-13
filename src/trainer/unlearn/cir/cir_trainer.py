# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=SAMPLE_UNLEARN
import logging
import random

import torch as pt
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from transformers import TrainerCallback

import trainer.unlearn.cir.hooks as hooks
from data.utils import batched, prep_batch
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import normalize_grads, sanitize_tensor
from trainer.unlearn.cir.collapsers import MahalanobisCollapser
from trainer.unlearn.cir.kl_utils import KLComputor
from trainer.utils import label_logits, layer_num

logging.basicConfig(level=logging.INFO)


class CalculateDistributionStatsCallback(TrainerCallback):
    """Callback to extract distribution stats at epoch end."""

    def __init__(self, trainer, collapsers):
        self.trainer = trainer
        self.collapsers = collapsers

    def on_epoch_end(self, args, state, control, **kwargs):
        # todo: simplify, and store collapsers inside modules, like in CIR_MoE
        for collapser in self.collapsers:
            collapser.process_saved_vecs()
        self.trainer.after_first_epoch = True


class CIR(UnlearnTrainer):
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
            if module.weight.requires_grad:  # install hooks
                module.register_forward_hook(hooks.save_act_input)
                module.register_full_backward_hook(hooks.save_grad_output)

        # additional hooks for computing grad collapse more efficiently
        if "grad_pcs_to_use" in self.cfg:
            for layer_id in range(train_to_layer):
                mlp = self.model.model.layers[layer_id].mlp
                mlp.down_proj.register_full_backward_hook(hooks.save_grad_input)
                mlp.down_proj.register_full_backward_hook(hooks.save_grad_output)

        # register collapsers, that calculate distribution stats, and later collapse
        _all_collapsers = []
        if "act_pcs_to_use" in self.cfg:
            self.act_collapsers = {
                name: MahalanobisCollapser(cfg.act_pcs_to_use, module.weight.device)
                for name, module in self.model.named_modules()
                if name.endswith(".up_proj")
            }
            _all_collapsers += list(self.act_collapsers.values())
        if "grad_pcs_to_use" in self.cfg:
            self.grad_collapsers = {
                name: MahalanobisCollapser(cfg.grad_pcs_to_use, module.weight.device)
                for name, module in self.model.named_modules()
                if name.endswith(".down_proj")
            }
            _all_collapsers += list(self.grad_collapsers.values())

        self.add_callback(CalculateDistributionStatsCallback(self, _all_collapsers))

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
                    ref *= self.cfg.retain_momentum
                    ref += param.grad * (1 - self.cfg.retain_momentum)
                    param.ref_grad = quantize_blockwise(ref)

        # ! unlearning loss
        batch = inputs["forget"]
        token_mask = batch["attention_mask"].bool().clone()
        token_mask[:, 0] = False  # ignore BOS token

        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device))
        forget_loss = label_logits(output.logits, batch["labels"])
        forget_loss.backward()

        if "grad_pcs_to_use" in self.cfg:
            grad_corrections = self.get_grad_correction(token_mask)

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            acts = module.last_act_input[token_mask].detach()
            grads = module.last_grad_output[token_mask].detach()
            assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            if "act_pcs_to_use" in self.cfg:
                if name.endswith(".up_proj"):
                    self.act_collapsers[name].add_vecs(acts)

            if not self.after_first_epoch:
                continue  # so only collect activations and not train

            if "act_pcs_to_use" in self.cfg:
                # gate_proj shares inputs with up_proj, so we can use up_proj's collapser
                _up_proj_name = name.replace(".gate_proj", ".up_proj")
                acts = self.act_collapsers[_up_proj_name].collapse(acts).to(model.dtype)

            if "grad_pcs_to_use" in self.cfg:
                grads *= grad_corrections[parent_mlp_name(name)]

            # ! MUDMAN-like operation
            if "retain_momentum" in self.cfg:
                ref_grad = dequantize_blockwise(*module.weight.ref_grad).to(model.dtype)

                token_disr = pt.einsum("ij,ti,tj->t", ref_grad, grads, acts)
                kl_mask = token_disr > 0
                acts = acts[kl_mask]
                grads = grads[kl_mask]

                # # col_mask = pt.einsum("ij,ti->tj", ref_grad, grads) * acts > 0
                # row_mask = pt.einsum("ij,tj->ti", ref_grad, acts) * grads > 0
                # # acts *= col_mask
                # grads *= row_mask

            # without the projections, this is equivalent to normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
            # would be possible to optimize training by disabling the first grad computation,
            # since we discard these grads anyway

        normalize_grads(model)

        if not self.after_first_epoch:
            # zero gradients so that optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()

    def get_grad_correction(self, token_mask):
        grad_corrections = {}
        for name, module in self.model.named_modules():
            if not name.endswith(".down_proj"):
                continue
            up_proj = self.model.get_submodule(name.replace(".down_proj", ".up_proj"))
            if not up_proj.weight.requires_grad:
                continue

            grad_input = module.last_grad_input[token_mask].detach().clone()
            grad_output = module.last_grad_output[token_mask].detach().clone()
            assert grad_input.shape == (token_mask.sum(), module.weight.shape[1])
            assert grad_output.shape == (token_mask.sum(), module.weight.shape[0])

            self.grad_collapsers[name].add_vecs(grad_output)

            if not self.after_first_epoch:
                continue  # first epoch, so only collect activations and not train

            out_collapsed = (
                self.grad_collapsers[name].collapse(grad_output).to(module.weight.dtype)
            )
            in_collapsed = out_collapsed @ module.weight  # backpropagation
            grad_correction = in_collapsed / sanitize_tensor(grad_input, 1e-6)
            grad_corrections[parent_mlp_name(name)] = grad_correction
        return grad_corrections


def parent_mlp_name(name):
    parent_name = name.rsplit(".", 1)[0]
    assert parent_name.endswith(".mlp")
    return parent_name


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
