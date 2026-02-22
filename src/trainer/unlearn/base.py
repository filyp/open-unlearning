from copy import deepcopy

from accelerate.utils import is_deepspeed_available

from trainer.base import FinetuneTrainer

if is_deepspeed_available():
    import deepspeed


class UnlearnTrainer(FinetuneTrainer):
    def prediction_step(self, *args, **kwargs):
        """Use standard loss during evaluation, not the unlearn loss.

        Subclasses override compute_loss() with custom unlearn logic, but that
        should only run during training. During evaluation (via prediction_step),
        we temporarily swap in the base FinetuneTrainer.compute_loss so the
        standard cross-entropy loss is used instead. This avoids having to copy
        the full prediction_step implementation from transformers, making it
        robust across library versions.
        """
        custom_compute_loss = self.compute_loss
        # __get__ binds the unbound class method to this instance, so it receives `self`
        self.compute_loss = FinetuneTrainer.compute_loss.__get__(self, type(self))
        try:
            return super().prediction_step(*args, **kwargs)
        finally:
            self.compute_loss = custom_compute_loss

    # Adapted from Huggingface DPO Trainer: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model
