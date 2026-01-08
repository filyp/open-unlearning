import os

from dotenv import load_dotenv

load_dotenv()  # here, because of TQDM_DISABLE, todo, move back down

import hydra
from omegaconf import DictConfig

from data import get_collators, get_data
from evals import get_evaluators
from model import get_model, reset_model
from trainer import load_trainer
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
    if trainer_cfg.get("save_final_state", True):
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")

    relearning_cfg = cfg.get("relearning_trainer", None)
    if relearning_cfg:
        # Get best model state dict from evaluator
        for evaluator in evaluators.values():
            if hasattr(evaluator, "best_model_state_dict"):
                best_model_state_dict = evaluator.best_model_state_dict
                break
        assert "best_model_state_dict" in locals(), "Relearning needs saved best model"

        # Create fresh model and load best weights
        model = reset_model(model)
        model.load_state_dict(best_model_state_dict, assign=True)

        # Finish current tracking runs and modify project names for relearning
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
        if "WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = "rel-" + os.environ["WANDB_PROJECT"]
        if "COMET_PROJECT_NAME" in os.environ:
            os.environ["COMET_PROJECT_NAME"] = "rel-" + os.environ["COMET_PROJECT_NAME"]

        relearning_evaluators = get_evaluators(
            eval_cfgs=relearning_cfg.relearning_eval,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
            data=data,
        )

        relearn_trainer, _ = load_trainer(
            trainer_cfg=relearning_cfg,
            model=model,
            train_dataset=data["relearn"],
            tokenizer=tokenizer,
            data_collator=collator,
            evaluators=relearning_evaluators,
            template_args=template_args,
        )
        relearn_trainer.train()

    # * get the final score (if defined), and return for potential Optuna optimization
    for evaluator in evaluators.values():
        if hasattr(evaluator, "get_final_score"):
            final_score = evaluator.get_final_score()
            print(f"Final score for {evaluator.__class__.__name__}: {final_score}")
            return final_score


if __name__ == "__main__":
    main()
