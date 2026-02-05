import logging

import lm_eval
import lm_eval.tasks
import torch as pt
from lm_eval.tasks import TaskManager, get_task_dict

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


def _get_temperature_0_accuracy(lm_eval_results):
    return lm_eval_results["results"]["wmdp_bio"]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results):
    samples = lm_eval_results["samples"]["wmdp_bio"]
    target_logprobs = pt.tensor([s["resps"][s["target"]][0][0] for s in samples])
    target_probs = pt.exp(target_logprobs)
    return target_probs.mean().item()


class WMDPLLowMIEvaluator:
    def __init__(self, eval_cfg, data, **kwargs):
        self.eval_cfg = eval_cfg
        assert not kwargs["template_args"].apply_chat_template, "model not supported"

        # load data
        self.eval_qs = data["eval_qs"]

        # Get the wmdp_bio task (uses the standard template)
        # Use include_defaults=False and only include wmdp tasks for a much faster init
        wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
        task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
        self.task_dict = get_task_dict(["wmdp_bio"], task_manager)
        # Modify the wmdp_bio task to use our custom questions
        # Note that this also supports eval_qs from wmdp_cyber - we only need
        # the task template which is the same for both tasks.
        self.task_dict["wmdp_bio"].dataset["test"] = data["eval_qs"]

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        trainer = kwargs["trainer"]

        # * eval forget acc
        lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=kwargs["tokenizer"],
            batch_size=trainer.args.per_device_eval_batch_size,
        )
        lm_eval_results = lm_eval.evaluator.evaluate(
            lm=lm,
            task_dict=self.task_dict,
            log_samples=True,
        )

        return dict(
            forget_acc_t0=_get_temperature_0_accuracy(lm_eval_results),
            forget_acc_t1=_get_temperature_1_accuracy(lm_eval_results),
        )

    def get_relearning_robustness_metric(self, eval_results_history):
        logging.info("Using max temperature=1 accuracy as the robustness metric")
        return max(res["forget_acc_t1"] for res in eval_results_history)
        # logging.info("Using min recall loss as the robustness metric")
        # return min(res["recall_loss"] for res in self.results)
