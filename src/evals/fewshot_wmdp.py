import gc
import logging

import lm_eval
import lm_eval.tasks
import torch as pt
from lm_eval.tasks import TaskManager, get_task_dict


logger = logging.getLogger("evaluator")


def _get_temperature_0_accuracy(lm_eval_results, task):
    return lm_eval_results["results"][task]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results, task):
    samples = lm_eval_results["samples"][task]
    target_logprobs = pt.tensor([s["resps"][s["target"]][0][0] for s in samples])
    target_probs = pt.exp(target_logprobs)
    return target_probs.mean().item()


class FewShotWMDPEvaluator:
    """Few-shot attack evaluator for WMDP.

    Prepends k in-context examples (from the relearn split) to each eval question,
    probing whether the unlearned model can recall forgotten knowledge without
    weight updates.
    """

    def __init__(self, eval_cfg, data, **kwargs):
        self.eval_qs = data["eval_qs"]
        self.fewshot_qs = data["fewshot_qs"]
        self.num_fewshot = eval_cfg.get("num_fewshot", 5)
        self.only_at_relearn_start = eval_cfg.get("only_at_relearn_start", False)
        self.mode = kwargs.get("mode")
        # long few-shot prompts (esp. cyber, ~9k tokens) OOM at the trainer's eval
        # batch size, so allow overriding it just for this eval
        self.batch_size = eval_cfg.get("batch_size", None)
        self.prefix = f"fewshot{self.num_fewshot}"

        self.task = eval_cfg.get("task", "wmdp_bio")
        wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
        task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
        self.task_dict = get_task_dict([self.task], task_manager)

        task = self.task_dict[self.task]
        # Set eval questions as test split
        task.dataset["test"] = self.eval_qs
        # Set relearn questions as training split (source for few-shot examples)
        task.dataset["train"] = self.fewshot_qs
        task._config.training_split = "train"
        task._config.num_fewshot = self.num_fewshot
        task.set_fewshot_seed(seed=42)  # deterministic sampling

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        tokenizer = kwargs["tokenizer"]
        trainer = kwargs["trainer"]
        if self.only_at_relearn_start and (self.mode != "relearn" or trainer.state.epoch):
            return {}
        model.eval()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()

        # Reset fewshot seed each call so the same demos are used every epoch
        self.task_dict[self.task].set_fewshot_seed(seed=42)

        lm = lm_eval.models.huggingface.HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=self.batch_size or trainer.args.per_device_eval_batch_size,
        )
        lm_eval_results = lm_eval.evaluator.evaluate(
            lm=lm,
            task_dict=self.task_dict,
            log_samples=True,
        )

        result = {
            f"{self.prefix}_acc_t0": _get_temperature_0_accuracy(lm_eval_results, self.task),
            f"{self.prefix}_acc_t1": _get_temperature_1_accuracy(lm_eval_results, self.task),
        }
        del lm, lm_eval_results
        gc.collect()
        pt.cuda.empty_cache()
        return result
