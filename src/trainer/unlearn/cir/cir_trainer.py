# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging
import re

import torch as pt
from transformers import TrainerCallback

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import (
    PreCachingDataLoader,
    get_grad_correction,
    get_relev_mask_with_caching,
    get_token_mask,
    mlp_activation_breaking_loss,
    mlp_breaking_loss,
    neuron_breaking_loss,
    normalize_grads,
    prep_batch,
    save_act_input_hook,
    save_grad_input_hook,
    save_grad_output_hook,
    save_output_hook,
)
from trainer.unlearn.cir.collapsers import MahalanobisCollapser

logging.basicConfig(level=logging.INFO)


# limit RAM
import resource  # noqa: E402

resource.setrlimit(resource.RLIMIT_DATA, (18 * 1024**3, 20 * 1024**3))


class CIRCallback(TrainerCallback):
    """Callback to extract distribution stats at epoch end."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        for collapser in self.trainer.act_collapsers.values():
            collapser.process_saved_vecs()
        if "grad_pcs_to_use" in self.trainer.cfg:
            for collapser in self.trainer.grad_collapsers.values():
                collapser.process_saved_vecs()

        self.trainer.after_first_epoch = True


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.after_first_epoch = False
        model = self.model

        self.layer_range = cfg.get("layer_range", [0, len(model.model.layers)])
        logging.info(f"loss layer range: {self.layer_range}")

        # * set trainable params
        for n, p in model.named_parameters():
            p.requires_grad = any(pattern in n for pattern in cfg.target_modules)
            if p.requires_grad:
                # match .layers.X. pattern
                layer_num = re.search(r"\.layers\.(\d+)\.", n).group(1)
                # don't train last layer that's used for loss, and onwards
                # because for that last layer, we don't have down_proj output grads
                if int(layer_num) >= self.layer_range[1] - 1:
                    p.requires_grad = False

        self.batches = PreCachingDataLoader(
            self.train_dataset,
            self.data_collator,
            self.args.per_device_train_batch_size,
        )

        # hooks for forget loss
        for layer_id in range(*self.layer_range):
            mlp = model.model.layers[layer_id].mlp
            if cfg.forget_loss == "mlp_breaking":
                mlp.down_proj.register_forward_hook(save_output_hook)
            elif cfg.forget_loss == "mlp_activation_breaking":
                mlp.down_proj.register_forward_hook(save_act_input_hook)
            elif cfg.forget_loss == "neuron_breaking":
                mlp.down_proj.register_forward_hook(save_act_input_hook)
                # note: overlaps some collapse hooks, but that's fine:
                mlp.down_proj.register_full_backward_hook(save_grad_input_hook)

        # hooks for component collapse
        for layer_id in range(self.layer_range[1] - 1):
            mlp = model.model.layers[layer_id].mlp
            mlp.up_proj.register_forward_hook(save_act_input_hook)
            mlp.up_proj.register_full_backward_hook(save_grad_output_hook)
            mlp.gate_proj.register_forward_hook(save_act_input_hook)
            mlp.gate_proj.register_full_backward_hook(save_grad_output_hook)
            mlp.down_proj.register_full_backward_hook(save_grad_input_hook)
            mlp.down_proj.register_full_backward_hook(save_grad_output_hook)

        # if cfg.get("retaining_rate", 0) > 0:
        #     cache_activations_for_cb_retain_loss(model, self.batches.retain, cfg)

        self.act_collapsers = {
            name: MahalanobisCollapser(cfg.act_pcs_to_use, module.weight.device)
            for name, module in model.named_modules()
            if name.endswith(".up_proj")
        }
        if "grad_pcs_to_use" in self.cfg:
            self.grad_collapsers = {
                name: MahalanobisCollapser(cfg.grad_pcs_to_use, module.weight.device)
                for name, module in model.named_modules()
                if name.endswith(".down_proj")
            }

        self.add_callback(CIRCallback(self))

    def get_train_dataloader(self):
        """Return dataloader over pre-batched forget/retain pairs."""
        return self.batches

    def training_step(self, model, inputs):
        model.train()
        # ! unlearning loss
        batch = inputs["forget"]
        token_mask = get_token_mask(batch)
        model.zero_grad(set_to_none=True)
        output = model(**prep_batch(batch, model.device), output_hidden_states=True)

        if self.cfg.get("forget_loss") == "mlp_breaking":
            forget_loss = mlp_breaking_loss(model, batch, self.layer_range)
        elif self.cfg.get("forget_loss") == "mlp_activation_breaking":
            forget_loss = mlp_activation_breaking_loss(model, batch, self.layer_range)
        elif self.cfg.get("forget_loss") == "neuron_breaking":
            forget_loss = neuron_breaking_loss(model, batch, self.layer_range, output)
        else:
            forget_loss = -output.loss
        forget_loss.backward()
        # we could do backward(inputs=[some_early_weight]) and delete that grad later
        # to skip the weight.grad computation, while maintaining full backpropagation

        if "grad_pcs_to_use" in self.cfg:
            grad_corrections = get_grad_correction(
                model, token_mask, self.grad_collapsers, self.after_first_epoch
            )

        for name, module in model.named_modules():
            if (not hasattr(module, "weight")) or (not module.weight.requires_grad):
                continue

            acts = module.last_act_input[token_mask].detach().clone().float()
            grads = module.last_grad_output[token_mask].detach().clone().float()
            assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            assert grads.shape == (token_mask.sum(), module.weight.shape[0])

            if name.endswith(".up_proj"):
                self.act_collapsers[name].add_vecs(acts)

            if not self.after_first_epoch:
                continue  # so only collect activations and not train

            if "grad_pcs_to_use" in self.cfg:
                _parent_mlp_name = name.rsplit(".", 1)[0]
                grads *= grad_corrections[_parent_mlp_name].float()

            if self.cfg.get("act_quantile", 0) > 0:
                relev_mask = get_relev_mask_with_caching(
                    batch, name, acts, token_mask, self.cfg.act_quantile
                )
                acts = acts[relev_mask]
                grads = grads[relev_mask]

            _up_proj_name = name.replace(".gate_proj", ".up_proj")
            acts = self.act_collapsers[_up_proj_name].collapse(acts)

            # without the projections, this is equivalent to normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(model.dtype)

        # # ! retain pass
        # if self.cfg.get("retaining_rate", 0) > 0:
        #     r_batch = inputs["retain"]
        #     output = model(
        #         **prep_batch(r_batch, model.device), output_hidden_states=True
        #     )
        #     retain_loss = cb_retain_loss(output, r_batch, self.cfg)
        #     retain_loss *= self.cfg.retaining_rate
        #     retain_loss.backward()

        normalize_grads(model)

        if not self.after_first_epoch:
            # zero gradients so optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()


# # minimal steps to run:
# model = AutoModelForCausalLM.from_pretrained(
#     cfg.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
# )
# trainer = CIR(model=model, train_dataset=train_dataset)
# trainer.train()

# * old ways of filtering:
# dists = acts.norm(dim=1) * grads.norm(dim=1)
# dists = grads.norm(dim=1)
#
# dists = (centered_norm * mahal_dirs_norm).sum(dim=1)
# dists = (centered * mahal_dirs_norm).sum(dim=1)
# dists = (centered * mahal_dirs).sum(dim=1)
# dists = (centered_norm * mahal_dirs).sum(dim=1)
