# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging

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
    normalize_grads,
    prep_batch,
    save_act_input_hook,
    save_act_input_hook_nondetached,
    save_grad_output_hook,
    save_grad_input_and_output_hook,
    save_output_hook,
)
from trainer.unlearn.cir.collapsers import MahalanobisCollapser

logging.basicConfig(level=logging.INFO)


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

        self.trainer.collapsers_initialized = True


class CIR(UnlearnTrainer):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.collapsers_initialized = False
        model = self.model

        # * set trainable params
        for n, p in model.named_parameters():
            p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

        self.batches = PreCachingDataLoader(
            self.train_dataset,
            self.data_collator,
            self.args.per_device_train_batch_size,
        )

        if cfg.get("forget_loss") in ("mlp_breaking", "mlp_activation_breaking"):
            self.layer_range = cfg.get("layer_range", [0, len(model.model.layers)])
            logging.info(f"layer_range for {cfg.forget_loss}: {self.layer_range}")
            # install hooks for MLPs
            for layer_id in range(*self.layer_range):
                mlp = model.model.layers[layer_id].mlp
                if cfg.forget_loss == "mlp_breaking":
                    mlp.register_forward_hook(save_output_hook)
                elif cfg.forget_loss == "mlp_activation_breaking":
                    mlp.down_proj.register_forward_pre_hook(save_act_input_hook_nondetached)

        # if cfg.get("retaining_rate", 0) > 0:
        #     cache_activations_for_cb_retain_loss(model, self.batches.retain, cfg)

        for n, m in model.named_modules():
            if n.endswith(".up_proj") or n.endswith(".gate_proj"):
                m.register_forward_pre_hook(save_act_input_hook)
                m.register_full_backward_pre_hook(save_grad_output_hook)
            if n.endswith(".down_proj"):
                m.register_full_backward_hook(save_grad_input_and_output_hook)

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
        else:
            forget_loss = -output.loss
        forget_loss.backward()

        if "grad_pcs_to_use" in self.cfg:
            grad_corrections = get_grad_correction(
                model, token_mask, self.grad_collapsers, self.collapsers_initialized
            )

        for name, module in model.named_modules():
            if not (name.endswith(".up_proj") or name.endswith(".gate_proj")):
                continue
            if not hasattr(module, "last_grad_output"):
                continue

            acts = module.last_act_input[token_mask].float()
            grads = module.last_grad_output[token_mask].float()
            assert acts.shape == (token_mask.sum(), module.weight.shape[1])
            assert grads.shape == (token_mask.sum(), module.weight.shape[0])
            module.last_act_input = None
            module.last_grad_output = None

            if name.endswith(".up_proj"):
                self.act_collapsers[name].add_vecs(acts)

            if not self.collapsers_initialized:
                continue  # first epoch, so only collect activations and not train

            if "grad_pcs_to_use" in self.cfg:
                _parent_mlp_name = name.rsplit(".", 1)[0]
                if _parent_mlp_name not in grad_corrections:
                    # on last layer in layer_range, there is no grad correction
                    _last_layer = max(self.layer_range) - 1
                    assert f".{_last_layer}." in _parent_mlp_name
                    module.weight.grad = None
                    continue
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

        if not self.collapsers_initialized:
            # First epoch: zero gradients so optimizer.step() is no-op
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
