import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
            data=data,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")
    
    relearning_cfg = cfg.get("relearning", None)
    if relearning_cfg:
        
        # unfreeze all parameters
        for p in model.parameters():
            p.requires_grad = True
        # remove all hooks
        for m in model.modules():
            m._forward_hooks = {}
            m._backward_hooks = {}

        from evals.wmdp_deduped import WMDPDedupedEvaluator
        import torch as pt
        from trainer.unlearn.cir.cir_utils import sanitize_batch
        from torch.utils.data import DataLoader

        ev = WMDPDedupedEvaluator(relearning_cfg.relearning_eval, data, tokenizer=tokenizer)

        retraining_optimizer = pt.optim.SGD(model.parameters(), lr=relearning_cfg.lr)
        relearn_loader = DataLoader(
            data["relearn"], 
            batch_size=relearning_cfg.relearn_batch_size,
            shuffle=False,
            collate_fn=collator
        )

        # * get metrics
        ev.evaluate(model)

        for epoch in range(relearning_cfg.num_epochs):
            model.train()
            for batch in relearn_loader:
                model.zero_grad(set_to_none=True)
                output = model(**sanitize_batch(batch))
                output.loss.backward()
                retraining_optimizer.step()

            # * get metrics
            ev.evaluate(model)



if __name__ == "__main__":
    main()
