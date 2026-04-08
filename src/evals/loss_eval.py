import torch as pt

from data.utils import batched, prep_batch
from evals.metrics.utils import evaluate_probability


def get_loss(model, batches):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device))
            loss_acc += output.loss.item()
    return loss_acc / len(batches)


class LossEvaluator:
    def __init__(self, eval_cfg, data, **kwargs):
        self.dataset_name = eval_cfg.dataset_name
        self.eval_samples = data[self.dataset_name]

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        trainer = kwargs["trainer"]

        eval_batches = [
            trainer.data_collator(samples)
            for samples in batched(
                self.eval_samples, trainer.args.per_device_eval_batch_size
            )
        ]
        return {f"{self.dataset_name}_loss": get_loss(model, eval_batches)}


class ProbabilityEvaluator:
    """Evaluates mean per-sequence normalized probability (geometric mean of token probs).

    More robust than mean loss for unlearning: bounded in [0,1], so one sequence
    with very high loss can't mask others that remain memorized.
    """

    def __init__(self, eval_cfg, data, **kwargs):
        self.dataset_name = eval_cfg.dataset_name
        self.eval_samples = data[self.dataset_name]

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        model.eval()
        model.zero_grad(set_to_none=True)
        trainer = kwargs["trainer"]

        eval_batches = [
            trainer.data_collator(samples)
            for samples in batched(
                self.eval_samples, trainer.args.per_device_eval_batch_size
            )
        ]
        probs = []
        for batch in eval_batches:
            results = evaluate_probability(model, batch)
            probs.extend(r["prob"] for r in results)
        print("answer probabilities: ", [round(p, 3) for p in probs])
        return {f"{self.dataset_name}_prob": sum(probs) / len(probs)}
