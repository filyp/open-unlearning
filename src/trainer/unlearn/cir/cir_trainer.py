# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_deduped/default trainer=CIR task_name=SAMPLE_UNLEARN mode=wmdp_deduped
import logging

import torch as pt
from transformers import TrainerCallback

from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.cir.cir_utils import (
    batched,
    cache_activations_for_cb_retain_loss,
    cache_activations_for_mlp_breaking_loss,
    cb_retain_loss,
    compute_per_text_quantile_mask,
    get_token_mask,
    mlp_breaking_loss,
    prep_batch,
    install_hooks,
)
from trainer.unlearn.cir.collapsers import TopPCsCollapser, MahalanobisCollapser

logging.basicConfig(level=logging.INFO)


def save_output_hook(module, args, output):
    # install hooks for MLPs
    module.cached_out = output


class CIRCallback(TrainerCallback):
    """Callback to extract distribution stats at epoch end."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        for collapser in self.trainer.acts_collapsers.values():
            collapser.process_saved_vecs()
        for collapser in self.trainer.grads_collapsers.values():
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

        install_hooks(model)
        for layer_id in range(*cfg.layer_range):
            model.model.layers[layer_id].mlp.register_forward_hook(save_output_hook)

        # * go through whole dataset, to prepare the batches in advance
        self.forget_batches = []
        self.retain_batches = []
        for f, r in zip(
            # prepare separately forget and retain, to support different batch sizes
            batched(self.train_dataset.forget, cfg.train_batch_size),
            batched(self.train_dataset.retain, cfg.retain_batch_size),
        ):
            self.forget_batches.append(self.data_collator(f))
            self.retain_batches.append(self.data_collator(r))
        del self.train_dataset

        cache_activations_for_mlp_breaking_loss(model, self.forget_batches, cfg)
        if cfg.get("retaining_rate", 0) > 0:
            cache_activations_for_cb_retain_loss(model, self.retain_batches, cfg)

        self.acts_collapsers = {
            n: MahalanobisCollapser(cfg.act_reg)
            for n, m in model.named_modules()
            if hasattr(m, "weight") and m.weight.requires_grad
        }

        if cfg.get("grad_proj_num", 0) > 0:
            self.grads_collapsers = {
                n: TopPCsCollapser(cfg.grad_proj_num)
                for n, m in model.named_modules()
                if hasattr(m, "weight") and m.weight.requires_grad
            }
        else:
            self.grads_collapsers = {}

        self.add_callback(CIRCallback(self))

    def get_train_dataloader(self):
        """Return dataloader over pre-batched forget/retain pairs."""

        class CIRDataLoader:
            def __init__(self, forget_batches, retain_batches):
                self.forget_batches = forget_batches
                self.retain_batches = retain_batches

            def __iter__(self):
                for fb, rb in zip(self.forget_batches, self.retain_batches):
                    yield {"forget": fb, "retain": rb}

            def __len__(self):
                return len(self.forget_batches)

        return CIRDataLoader(self.forget_batches, self.retain_batches)

    def training_step(self, model, inputs):
        # note that we may lose some functionality from the original trainer.training_step
        model.train()
        # ! unlearning loss
        batch = inputs["forget"]
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        output = model(**prep_batch(batch, model.device), output_hidden_states=True)
        forget_loss = mlp_breaking_loss(model, batch, self.cfg)
        forget_loss.backward()

        # ! here we modify the grad
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight.grad is None:
                continue

            token_mask = get_token_mask(batch)
            acts = module.last_act_full[token_mask].to(pt.float32)
            grads = module.last_grad_full[token_mask].to(pt.float32)
            assert len(acts.shape) == len(grads.shape) == 2

            self.acts_collapsers[name].add_vecs(acts)
            if self.grads_collapsers:
                self.grads_collapsers[name].add_vecs(grads)

            if not self.collapsers_initialized:
                continue  # first epoch, so only collect activations and not train

            centered = acts - self.acts_collapsers[name].mean

            if self.cfg.get("act_quantile", 0) > 0:
                dists = centered.norm(dim=1)
                # dists = collapsed_acts.norm(dim=1)  # slows unlearning!
                act_relev_mask = compute_per_text_quantile_mask(
                    dists, token_mask, self.cfg.act_quantile
                )
                acts = acts[act_relev_mask]
                grads = grads[act_relev_mask]

            acts = self.acts_collapsers[name].collapse(acts)
            if self.grads_collapsers:
                grads = self.grads_collapsers[name].collapse(grads)

            # without the projections, this is the equivalent of normal backprop
            module.weight.grad = pt.einsum("ti,tj->ij", grads, acts).to(model.dtype)

        # ! retain pass
        if self.cfg.get("retaining_rate", 0) > 0:
            model.zero_grad(set_to_none=True)
            r_batch = inputs["retain"]
            output = model(
                **prep_batch(r_batch, model.device), output_hidden_states=True
            )
            retain_loss = cb_retain_loss(output, r_batch, self.cfg)
            retain_loss *= self.cfg.retaining_rate
            retain_loss.backward()

        if not self.collapsers_initialized:
            # First epoch: zero gradients so optimizer.step() is no-op
            model.zero_grad()

        return forget_loss.detach()


# # minimal steps to run:
# model = AutoModelForCausalLM.from_pretrained(
#     cfg.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
# )
# model.config.use_cache = False
# trainer = CirTrainer(
#     model=model,
#     train_dataset=train_dataset,
# )
# trainer.train()


# * old ways of filtering:
# act_norms = centered.norm(dim=1, keepdim=True)
# rescaled_acts = acts * act_norms**(self.cfg.act_norm_pow)
# dists = (rescaled_acts * mahal_dirs_norm).sum(dim=1)

# Apply mask based on Mahalanobis distance quantile
# ! note: once, using centered worked better than centered_norm, so investigate it
# centered_norm = centered / centered.norm(dim=1, keepdim=True)

# dists = (centered_norm * mahal_dirs_norm).sum(dim=1)
# dists = (centered * mahal_dirs_norm).sum(dim=1)
# dists = (centered * mahal_dirs).sum(dim=1)
# dists = (centered_norm * mahal_dirs).sum(dim=1)

