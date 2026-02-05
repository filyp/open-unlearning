import torch as pt

from data.utils import batched, prep_batch


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
