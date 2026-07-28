from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from data import get_collators, get_data
from evals import get_evaluators
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything

load_dotenv()


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
            mode=mode,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
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

    if "wandb" in (trainer_args.report_to or []):
        import wandb

        if wandb.run is not None:
            wandb.config.update(
                OmegaConf.to_container(cfg.trainer, resolve=True), allow_val_change=True
            )

    is_main = trainer.is_world_process_zero()

    # save last valid model (before any metric broke)
    comm_dir = Path(cfg.paths.tmp_comm_dir)
    if is_main and trainer.last_valid_model_state is not None and comm_dir.exists():
        model.load_state_dict(trainer.last_valid_model_state)
        model.save_pretrained(comm_dir / "last_valid_model")

    # Save per-epoch eval history (survives tmp_comm cleanup via output_dir)
    if is_main and trainer.eval_results_history:
        import json
        history_dir = Path(trainer_args.output_dir) / "eval_histories"
        history_dir.mkdir(parents=True, exist_ok=True)
        run_name = cfg.trainer.args.get("run_name", "run")
        history_path = history_dir / f"{run_name}.json"
        history_path.write_text(json.dumps(trainer.eval_results_history, indent=2))

    # * get the final score (if defined), and save to file for the pipeline script
    if is_main and trainer.eval_results_history and cfg.get("metric_to_optimize"):
        target_metric = cfg.metric_to_optimize
        if mode == "relearn":
            # relearning optimizes in the opposite direction to unlearning and the sweeper
            # so for example when unlearning maximizes loss, then robutsness=min(relearning loss)
            if cfg.optimize_direction == "minimize":
                robustness = max(res[target_metric] for res in trainer.eval_results_history if target_metric in res)
            elif cfg.optimize_direction == "maximize":
                robustness = min(res[target_metric] for res in trainer.eval_results_history if target_metric in res)
        else:
            # unlearn mode: return the metric directly (best value across epochs)
            if cfg.optimize_direction == "minimize":
                robustness = min(res[target_metric] for res in trainer.eval_results_history if target_metric in res)
            elif cfg.optimize_direction == "maximize":
                robustness = max(res[target_metric] for res in trainer.eval_results_history if target_metric in res)

        (comm_dir / "robustness.txt").write_text(str(robustness))
        


if __name__ == "__main__":
    main()
