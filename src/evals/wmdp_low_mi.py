import logging

import lm_eval
import lm_eval.tasks
import torch as pt
from lm_eval.tasks import TaskManager, get_task_dict

from hub_retry import retry_on_rate_limit

# logger = logging.getLogger("evaluator")
# Suppress the specific warnings from lm_eval when loading an existing model to HFLM
logging.getLogger("lm_eval.models.huggingface").setLevel(logging.ERROR)
# Disable lm_eval spam
logging.getLogger("lm_eval.api.task").setLevel(logging.WARNING)
logging.getLogger("lm_eval.evaluator").setLevel(logging.WARNING)


# Disable lm_eval progress bars by patching tqdm - there are no other ways to do this
# fmt: off
from functools import wraps  # noqa
import lm_eval.models.huggingface as hf_module  # noqa
_original_tqdm = hf_module.tqdm
@wraps(_original_tqdm)
def _disabled_tqdm(*args, **kwargs):
    kwargs["disable"] = True
    return _original_tqdm(*args, **kwargs)
hf_module.tqdm = _disabled_tqdm
# fmt: on


def _get_temperature_0_accuracy(lm_eval_results, task):
    return lm_eval_results["results"][task]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results, task):
    samples = lm_eval_results["samples"][task]
    target_logprobs = pt.tensor([s["resps"][s["target"]][0][0] for s in samples])
    target_probs = pt.exp(target_logprobs)
    return target_probs.mean().item()


class WMDPLLowMIEvaluator:
    def __init__(self, eval_cfg, data, **kwargs):
        # load data
        self.eval_qs = data["eval_qs"]

        # Get the domain's wmdp task (uses the standard template; the tasks differ
        # only in the description line, e.g. "...about biology" vs "...about cybersecurity")
        # Use include_defaults=False and only include wmdp tasks for a much faster init
        self.task = eval_cfg.get("task", "wmdp_bio")
        wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
        task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
        self.task_dict = retry_on_rate_limit(get_task_dict, [self.task], task_manager)
        # Modify the task to use our custom questions
        self.task_dict[self.task].dataset["test"] = data["eval_qs"]

    def evaluate(self, model, tokenizer, trainer, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)

        # * eval forget acc
        lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=trainer.args.per_device_eval_batch_size,
        )
        lm_eval_results = lm_eval.evaluator.evaluate(
            lm=lm,
            task_dict=self.task_dict,
            log_samples=True,
        )

        return dict(
            forget_acc_t0=_get_temperature_0_accuracy(lm_eval_results, self.task),
            forget_acc_t1=_get_temperature_1_accuracy(lm_eval_results, self.task),
        )
