import gc
import logging

import lm_eval
import lm_eval.tasks
import torch as pt
from lm_eval.tasks import TaskManager, get_task_dict


logger = logging.getLogger("evaluator")


def _get_temperature_0_accuracy(lm_eval_results):
    return lm_eval_results["results"]["wmdp_bio"]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results):
    samples = lm_eval_results["samples"]["wmdp_bio"]
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
        self.prefix = f"fewshot{self.num_fewshot}"

        wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
        task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
        self.task_dict = get_task_dict(["wmdp_bio"], task_manager)

        task = self.task_dict["wmdp_bio"]
        # Set eval questions as test split
        task.dataset["test"] = self.eval_qs
        # Set relearn questions as training split (source for few-shot examples)
        task.dataset["train"] = self.fewshot_qs
        task._config.training_split = "train"
        task._config.num_fewshot = self.num_fewshot
        task.set_fewshot_seed(seed=42)  # deterministic sampling

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        tokenizer = kwargs["tokenizer"]
        trainer = kwargs["trainer"]

        # Reset fewshot seed each call so the same demos are used every epoch
        self.task_dict["wmdp_bio"].set_fewshot_seed(seed=42)

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

        result = {
            f"{self.prefix}_acc_t0": _get_temperature_0_accuracy(lm_eval_results),
            f"{self.prefix}_acc_t1": _get_temperature_1_accuracy(lm_eval_results),
        }
        del lm, lm_eval_results
        gc.collect()
        pt.cuda.empty_cache()
        return result
