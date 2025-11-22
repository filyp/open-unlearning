import logging

import torch as pt
import wandb

from evals.base import Evaluator
from trainer.unlearn.cir.cir_utils import (
    cross_entropy,
    load_from_repo,
    load_wmdp_simple_set,
)
from trainer.unlearn.cir.wmdp_efficient import eval_on

logger = logging.getLogger("evaluator")


def _get_loss(model, batches, use_answer_mask=False):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            if use_answer_mask:
                answer_mask = batch["answer_mask"]
                loss_acc += cross_entropy(output, batch, answer_mask).item()
            else:
                loss_acc += cross_entropy(output, batch).item()
    return loss_acc / len(batches)


class WMDPDedupedEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        self.num_eval_batches = eval_cfg.num_eval_batches

        # ! load data
        wikitext = load_from_repo("wikitext_16k.jsonl")
        self.wikitext_batches = [
            kwargs["tokenizer"](x["text"], **eval_cfg.tokenizer)
            for x in wikitext.shuffle(seed=42).batch(eval_cfg.wikitext_batch_size)
        ]
        data_dict = load_wmdp_simple_set(
            eval_cfg.data, kwargs["tokenizer"]
        )
        self.recall_batches = data_dict["recall"]
        self.eval_qs = data_dict["eval"]

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

        nb = self.num_eval_batches
        res["wikitext_loss"] = _get_loss(model, self.wikitext_batches[:nb])
        res["recall_loss"] = _get_loss(model, self.recall_batches, use_answer_mask=True)
        # res["retain_loss"] = _get_loss(model, [x["retain"] for x in train_dataset[:nb]])

        logging.info(res)
        wandb.log(res)
        return res
