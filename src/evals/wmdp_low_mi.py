import logging

import lm_eval
import lm_eval.tasks
import torch as pt
from lm_eval.tasks import TaskManager, get_task_dict

from evals.base import Evaluator
from trainer.unlearn.cir.cir_utils import batched, prep_batch
from trainer.unlearn.cir.kl_utils import KLComputor

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


class WMDPLLowMIEvaluator(Evaluator):
    def __init__(self, eval_cfg, data, **kwargs):
        self.eval_cfg = eval_cfg
        assert not kwargs["template_args"].apply_chat_template, "model not supported"

        # load data
        self.wikitext = data["wikitext"]
        self.recall_samples = data["recall"]
        self.eval_qs = data["eval_qs"]

        if self.eval_cfg.eval_mcq:
            # Get the wmdp_bio task (uses the standard template)
            # Use include_defaults=False and only include wmdp tasks for a much faster init
            wmdp_path = lm_eval.tasks.__path__[0] + "/wmdp"
            task_manager = TaskManager(include_path=wmdp_path, include_defaults=False)
            self.task_dict = get_task_dict(["wmdp_bio"], task_manager)
            # Modify the wmdp_bio task to use our custom questions
            # Note that this also supports eval_qs from wmdp_cyber - we only need
            # the task template which is the same for both tasks.
            self.task_dict["wmdp_bio"].dataset["test"] = data["eval_qs"]

        self.results = []
        self.first_eval = True

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        trainer = kwargs["trainer"]

        assert trainer.args.eval_on_start, "eval_on_start must be True"
        if self.first_eval:  # ! first evaluation, before training
            self.kl_computor = KLComputor(model, self.wikitext)

        res = {}
        res["wikitext_loss"], res["wikitext_kl"] = (
            self.kl_computor.eval_kl_many_batches(self.wikitext)
        )
        if self.first_eval:
            assert res["wikitext_kl"] == 0, (
                f"Initial KL should be 0, but got {res['wikitext_kl']}"
            )
            self.first_eval = False

        # collate recall samples to target batch size
        recall_batches = [
            trainer.data_collator(samples)
            for samples in batched(
                self.recall_samples, trainer.args.per_device_eval_batch_size
            )
        ]
        res["recall_loss"] = _get_loss(model, recall_batches)

        # res["retain_loss"] = _get_loss(model, [x["retain"] for x in train_dataset[:nb]])

        if self.eval_cfg.eval_mcq:
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
            res["forget_acc_t0"] = _get_temperature_0_accuracy(lm_eval_results)
            res["forget_acc_t1"] = _get_temperature_1_accuracy(lm_eval_results)

        # ! finished evaluating, now handle the results
        self.results.append(res)

        if self.eval_cfg.disr_budget is None:
            # used in relearning
            # don't stop training, don't keep track of the best valid model state
            return res

        # * check condition to stop training
        if res["wikitext_kl"] > self.eval_cfg.disr_budget:
            logging.info("Wikitext KL exceeded the disruption budget")
            trainer.control.should_training_stop = True
            return res

        # save the best model state, that doesn't exceed the disruption budget
        # this way relearning can start from this valid model state
        self.best_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        return res

    def get_relearning_robustness_metric(self):
        if self.eval_cfg.eval_mcq:
            logging.info("Using max temperature=1 accuracy as the robustness metric")
            return max(res["forget_acc_t1"] for res in self.results)
        else:
            logging.info("Using min recall loss as the robustness metric")
            return min(res["recall_loss"] for res in self.results)
