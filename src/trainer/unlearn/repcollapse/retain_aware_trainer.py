# python src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RetainAwareCollapse task_name=test
"""RepCollapse with retain-aware eigenvalue ranking.

Identical to RepCollapse except:
- Act/grad collapsers use RetainAwareCovCollapser (PCA dirs ranked by forget/retain ratio)
- Epoch 0 collects retain activations via forward hook (one extra no-grad forward pass)
"""
import logging

import torch as pt

from data.utils import prep_batch
from trainer.unlearn.repcollapse.repcollapse_trainer import RepCollapse
from trainer.unlearn.repcollapse.retain_aware_collapser import RetainAwareCovCollapser
from trainer.unlearn.repcollapse.utils import get_banned_tokens

logging.basicConfig(level=logging.INFO)


class RetainAwareCollapse(RepCollapse):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.recording_retain = False

        # Replace CovCollapsers with RetainAwareCovCollapsers
        reg_eps = cfg.get("reg_eps", 1e-4)
        for layer_num in range(len(self.model.model.layers)):
            mlp = self.model.model.layers[layer_num].mlp
            experts = mlp.experts if self.is_moe else [mlp]
            for expert in experts:
                for module in [expert.gate_proj, expert.up_proj, expert.down_proj]:
                    if hasattr(module, "act_collapser"):
                        module.act_collapser = RetainAwareCovCollapser(cfg.n_pcs, reg_eps)
                        module.grad_collapser = RetainAwareCovCollapser(cfg.n_pcs, reg_eps)

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Collect retain activations (epoch 0 only, forward pass, no grad)
        if self.batch_idx < self.recalc_every:
            r_batch = inputs["retain"]
            self.retain_token_mask = r_batch["attention_mask"].bool().clone()
            self.retain_token_mask[:, 0] = False
            if self.processing_class.chat_template is not None:
                for banned_token in get_banned_tokens(self.processing_class):
                    self.retain_token_mask &= r_batch["input_ids"] != banned_token

            self.recording_retain = True
            with pt.no_grad():
                model(**prep_batch(r_batch, model.device))
            self.recording_retain = False

        # Everything else is standard RepCollapse
        return super().training_step(model, inputs, num_items_in_batch)

    def save_act_input_hook(self, module, args, output):
        # Intercept retain forward pass to collect retain activations
        if self.recording_retain:
            if hasattr(module, "act_collapser") and hasattr(module.act_collapser, "add_retain_vecs"):
                retain_acts = args[0].detach()
                retain_acts = retain_acts[self.retain_token_mask]
                module.act_collapser.add_retain_vecs(retain_acts)
            return
        # Standard RepCollapse hook
        return super().save_act_input_hook(module, args, output)
