import copy
import logging

import lm_eval.tasks
import torch as pt
import torch.nn.functional as F
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager, get_task_dict

from evals.base import Evaluator
from trainer.unlearn.cir.cir_utils import batched, prep_batch

logger = logging.getLogger("evaluator")
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


def _cache_last_hidden_states(model, batches):
    """Cache last hidden states for each batch to compute KL divergence later."""
    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device), output_hidden_states=True)
            batch["cached_last_hidden"] = output.hidden_states[-1].detach()


def _get_loss_and_kl(model, batches, acts_to_logits):
    """Compute loss and KL divergence in a single pass.

    KL(P || Q) where P is the original model (cached) and Q is the current model.
    KL is averaged across all tokens (where labels != -100).
    """
    total_kl = 0.0
    total_tokens = 0
    loss_acc = 0.0

    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device))
            loss_acc += output.loss.item()

            current_logits = output.logits.float()

            # Reconstruct original logits from cached hidden states
            cached_hidden = batch["cached_last_hidden"].to(model.dtype)
            cached_logits = acts_to_logits(cached_hidden).float()
            assert current_logits.shape == cached_logits.shape  # (batch, seq, vocab)

            # Get mask for valid tokens (labels != -100)
            labels = batch["labels"].to(model.device)
            token_mask = labels != -100
            assert token_mask.shape == current_logits.shape[:2]

            # Mask first
            cached_logits_masked = cached_logits[token_mask]  # btw, bfloat is enough
            current_logits_masked = current_logits[token_mask]
            assert cached_logits_masked.shape == current_logits_masked.shape
            assert cached_logits_masked.ndim == 2  # (n_valid_tokens, vocab)

            # KL(P || Q) using kl_div (expects log_q as input, p as target)
            log_p = pt.nn.functional.log_softmax(cached_logits_masked, dim=-1)
            log_q = pt.nn.functional.log_softmax(current_logits_masked, dim=-1)

            total_kl += pt.nn.functional.kl_div(
                log_q, log_p, reduction="sum", log_target=True
            ).item()
            total_tokens += token_mask.sum().item()

    avg_loss = loss_acc / len(batches)
    avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_kl


def _get_temperature_0_accuracy(lm_eval_results):
    return lm_eval_results["results"]["wmdp_bio"]["acc,none"]


def _get_temperature_1_accuracy(lm_eval_results):
    samples = lm_eval_results["samples"]["wmdp_bio"]
    target_logprobs = pt.tensor([s["resps"][s["target"]][0][0] for s in samples])
    target_probs = pt.exp(target_logprobs)
    return target_probs.mean().item()


def _create_acts_to_logits(model):
    """Create a function to convert hidden states to logits.

    Handles models with lm_head and models with shared embeddings (e.g., MobileLLM).
    Copies the weights so they're preserved even if the model changes during training.
    """
    if hasattr(model, "lm_head"):
        cached_lm_head = copy.deepcopy(model.lm_head)
        return cached_lm_head
    else:
        # MobileLLM-style: use shared embedding weights
        cached_embed_weight = model.model.embed_tokens.weight.detach().clone()
        return lambda h: F.linear(h, cached_embed_weight)


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
            task = self.task_dict["wmdp_bio"]
            task.dataset["test"] = data["eval_qs"]
        
        self.results = []

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        trainer = kwargs["trainer"]

        assert trainer.args.eval_on_start, "eval_on_start must be True"
        first_eval = not hasattr(self, "acts_to_logits")
        if first_eval:  # ! first evaluation, before training
            # Cache last hidden states for wikitext KL divergence computation
            _cache_last_hidden_states(model, self.wikitext)
            # Create function to convert hidden states to logits (copies weights)
            self.acts_to_logits = _create_acts_to_logits(model)

        res = {}
        res["wikitext_loss"], res["wikitext_kl"] = _get_loss_and_kl(
            model, self.wikitext, self.acts_to_logits
        )

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
            lm = HFLM(
                pretrained=model,
                tokenizer=kwargs["tokenizer"],
                batch_size=trainer.args.per_device_eval_batch_size,
            )
            lm_eval_results = evaluator.evaluate(
                lm=lm,
                task_dict=self.task_dict,
                log_samples=True,
            )
            res["forget_acc_t0"] = _get_temperature_0_accuracy(lm_eval_results)
            res["forget_acc_t1"] = _get_temperature_1_accuracy(lm_eval_results)

        # ! finished evaluating, now handle the results
        self.results.append(res)

        if first_eval:
            assert res["wikitext_kl"] == 0, "Initial KL should be 0"

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
