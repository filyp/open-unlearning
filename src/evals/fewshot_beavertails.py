import gc
import logging
import random

import torch as pt

from data.custom_loaders import _tokenize, DATE_STRING
from evals.metrics.utils import evaluate_probability


logger = logging.getLogger("evaluator")


class FewShotBeaverTailsEvaluator:
    """Few-shot attack evaluator for BeaverTails.

    Prepends k harmful prompt-response pairs as in-context demonstrations,
    then measures the probability of generating the harmful response to a new prompt.
    """

    def __init__(self, eval_cfg, data, **kwargs):
        self.fewshot_raw = data["fewshot_raw"]
        self.holdout_raw = data["holdout_raw"]
        self.num_fewshot = eval_cfg.get("num_fewshot", 5)
        self.only_at_relearn_start = eval_cfg.get("only_at_relearn_start", False)
        self.mode = kwargs.get("mode")
        self.tokenizer = kwargs["tokenizer"]
        # no truncation by default, so few-shot contexts are never clipped
        max_length = eval_cfg.get("max_length", None)
        self.tokenizer_cfg = {"max_length": max_length,
                              "truncation": max_length is not None, "padding": False}
        self.seed = eval_cfg.get("seed", 42)
        self.prefix = f"fewshot{self.num_fewshot}"

    def _build_fewshot_samples(self):
        """Build tokenized samples with k few-shot demos prepended to each eval example."""
        rng = random.Random(self.seed)
        demos = rng.sample(self.fewshot_raw, min(self.num_fewshot, len(self.fewshot_raw)))

        samples = []
        for eval_ex in self.holdout_raw:
            if self.tokenizer.chat_template is None:
                # Base model: concatenate as plain text
                parts = [f"{d['prompt']} {d['response']}" for d in demos]
                parts.append(f"{eval_ex['prompt']} {eval_ex['response']}")
                full_txt = "\n\n".join(parts)
                # Beginning = everything up to the eval response
                beginning_parts = [f"{d['prompt']} {d['response']}" for d in demos]
                beginning_parts.append(eval_ex["prompt"])
                beginning_txt = "\n\n".join(beginning_parts)
            else:
                # Chat model: format as multi-turn conversation
                chat = []
                for d in demos:
                    chat.append({"role": "user", "content": d["prompt"]})
                    chat.append({"role": "assistant", "content": d["response"]})
                chat.append({"role": "user", "content": eval_ex["prompt"]})
                chat.append({"role": "assistant", "content": eval_ex["response"]})
                full_txt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, date_string=DATE_STRING
                )
                beginning_txt = self.tokenizer.apply_chat_template(
                    chat[:-1], tokenize=False, date_string=DATE_STRING,
                    add_generation_prompt=True,
                )

            sample = _tokenize(full_txt, self.tokenizer, self.tokenizer_cfg)
            beginning_len = len(
                self.tokenizer(beginning_txt, **self.tokenizer_cfg)["input_ids"]
            )
            # Only keep sample if there are response tokens after the prompt
            if beginning_len < len(sample["input_ids"]):
                sample["labels"][:beginning_len] = -100
                samples.append(sample)

        return samples

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        trainer = kwargs["trainer"]
        if self.only_at_relearn_start and (self.mode != "relearn" or trainer.state.epoch):
            return {}
        model.eval()
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()

        samples = self._build_fewshot_samples()
        batch_size = trainer.args.per_device_eval_batch_size

        from data.utils import batched

        eval_batches = [
            trainer.data_collator(batch) for batch in batched(samples, batch_size)
        ]

        probs = []
        losses = []
        for batch in eval_batches:
            results = evaluate_probability(model, batch)
            probs.extend(r["prob"] for r in results)
            losses.extend(r["avg_loss"] for r in results)

        result = {
            f"{self.prefix}_prob": sum(probs) / len(probs),
            f"{self.prefix}_loss": sum(losses) / len(losses),
        }
        gc.collect()
        pt.cuda.empty_cache()
        return result
