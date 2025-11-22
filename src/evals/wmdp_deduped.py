import logging

import torch as pt

import wandb
from evals.base import Evaluator
from trainer.unlearn.cir.cir_utils import sanitize_batch
from trainer.unlearn.cir.wmdp_efficient import eval_on

logger = logging.getLogger("evaluator")


def _get_loss(model, batches):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**sanitize_batch(batch))
            loss_acc += output.loss.item()
    return loss_acc / len(batches)


class WMDPDedupedEvaluator(Evaluator):
    def __init__(self, eval_cfg, data, **kwargs):
        self.eval_cfg = eval_cfg

        # load data
        self.wikitext = data["wikitext"]
        self.recall_batches = data["recall"]
        self.eval_qs = data["eval_qs"]

        if eval_cfg.get("wandb"):
            # todo, finiching wandb should be handled at training end instead, evaluator will handle this, when deciding whether to terminate the run
            wandb.finish()  # finish any existing wandb run
            wandb.init(
                project=eval_cfg.wandb.project,
                name=eval_cfg.wandb.name,
                group=eval_cfg.wandb.group,
                # config=OmegaConf.to_container(cfg),
            )

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        res = {}
        model.eval()

        # * eval forget acc
        res["forget_acc_t0"], res["forget_acc_t1"] = eval_on(self.eval_qs, model)

        nb = self.eval_cfg.num_eval_batches
        res["wikitext_loss"] = _get_loss(model, self.wikitext[:nb])
        res["recall_loss"] = _get_loss(model, self.recall_batches)
        # res["retain_loss"] = _get_loss(model, [x["retain"] for x in train_dataset[:nb]])

        logging.info(res)
        if self.eval_cfg.get("wandb"):
            wandb.log(res)
        return res
