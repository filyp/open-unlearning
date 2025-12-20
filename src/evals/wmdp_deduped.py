import logging

import torch as pt
import lm_eval.tasks
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict

import wandb
from evals.base import Evaluator
from trainer.unlearn.cir.cir_utils import prep_batch

logger = logging.getLogger("evaluator")
# Suppress the specific warnings from lm_eval when loading an existing model to HFLM
logging.getLogger("lm_eval.models.huggingface").setLevel(logging.ERROR)
# Disable lm_eval spam
logging.getLogger("lm_eval.api.task").setLevel(logging.WARNING)
logging.getLogger("lm_eval.evaluator").setLevel(logging.WARNING)


# Disable lm_eval progress bars by patching tqdm - there are no other ways to do this
import lm_eval.models.huggingface as hf_module
import lm_eval.api.task as task_module
from functools import wraps
_original_tqdm = hf_module.tqdm
@wraps(_original_tqdm)
def _disabled_tqdm(*args, **kwargs):
    kwargs['disable'] = True
    return _original_tqdm(*args, **kwargs)
hf_module.tqdm = _disabled_tqdm
task_module.tqdm = _disabled_tqdm


def _get_loss(model, batches):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device))
            loss_acc += output.loss.item()
    return loss_acc / len(batches)


def _get_temperature_0_accuracy(lm_eval_results):
    return lm_eval_results["results"]["wmdp_bio"]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results):
    samples = lm_eval_results["samples"]["wmdp_bio"]
    target_logprobs = pt.tensor([s["resps"][s["target"]][0][0] for s in samples])
    target_probs = pt.exp(target_logprobs)
    return target_probs.mean().item()


class WMDPDedupedEvaluator(Evaluator):
    def __init__(self, eval_cfg, data, **kwargs):
        self.eval_cfg = eval_cfg

        # load data
        self.wikitext = data["wikitext"]
        self.recall_batches = data["recall"]
        self.eval_qs = data["eval_qs"]

        # Get the wmdp_bio task (uses the standard template)
        # Use include_defaults=False and only include wmdp tasks for a much faster init
        wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
        task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
        self.task_dict = get_task_dict(["wmdp_bio"], task_manager)
        # Modify the wmdp_bio task to use our custom questions
        task = self.task_dict["wmdp_bio"]
        task.dataset["test"] = data["eval_qs"]

        self.init_wikitext_loss = None

        if eval_cfg.get("wandb"):
            wandb.init(
                project=eval_cfg.wandb.project,
                name=eval_cfg.wandb.name,
                group=eval_cfg.wandb.group,
                # config=OmegaConf.to_container(cfg),
            )

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        res = {}
        model.eval()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        
        res["wikitext_loss"] = _get_loss(model, self.wikitext)
        res["recall_loss"] = _get_loss(model, self.recall_batches)
        # res["retain_loss"] = _get_loss(model, [x["retain"] for x in train_dataset[:nb]])

        # * eval forget acc
        lm = HFLM(pretrained=model, tokenizer=kwargs["tokenizer"], batch_size=8)
        lm_eval_results = evaluator.evaluate(
            lm=lm,
            task_dict=self.task_dict,
            log_samples=True,
        )
        res["forget_acc_t0"] = _get_temperature_0_accuracy(lm_eval_results)
        res["forget_acc_t1"] = _get_temperature_1_accuracy(lm_eval_results)
        
        # ! finished evaluating, now handle the results

        logging.info(res)
        
        assert kwargs["trainer"].args.eval_on_start, "eval_on_start must be True"
        if self.init_wikitext_loss is None:
            # this is the first evaluation, before training
            self.init_wikitext_loss = res["wikitext_loss"]
            
        # * check condition to stop training
        if res["wikitext_loss"] > self.init_wikitext_loss * self.eval_cfg.disr_budget:
            logging.info(f"Wikitext loss exceeded the disruption budget")
            kwargs["trainer"].control.should_training_stop = True
            return res

        self.last_valid_res = res

        if self.eval_cfg.get("wandb"):
            wandb.log(res)

        return res

    def final_score(self):
        if self.eval_cfg.get("wandb") and wandb.run is not None:
            wandb.finish()
        return self.last_valid_res["recall_loss"]