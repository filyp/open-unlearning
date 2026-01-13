# example usage:
# python3 src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=1B_cir
import os
import shutil
import subprocess
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def _get_run_name(cfg: DictConfig) -> str:
    """Get run_name with hydra job number."""
    try:
        job_num = HydraConfig.get().job.num
        return f"{cfg.task_name}_{job_num}"
    except Exception:
        return cfg.task_name


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg.trainer.args.run_name = _get_run_name(cfg)
    cfg.relearning_trainer.args.run_name = _get_run_name(cfg)

    comm_dir = Path(cfg.paths.tmp_comm_dir) / cfg.task_name
    comm_dir.mkdir(parents=True, exist_ok=False)
    try:
        # ! unlearning #########################################################
        if "UNL_WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = os.environ["UNL_WANDB_PROJECT"]
        unlearning_cfg_path = comm_dir / "unlearning_cfg.yaml"
        OmegaConf.save(cfg, unlearning_cfg_path)
        subprocess.run(
            [
                "python3",
                "src/train.py",
                f"--config-path={comm_dir.absolute()}",
                "--config-name=unlearning_cfg.yaml",
            ],
            check=True,
        )

        # ! relearning #########################################################
        if "REL_WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = os.environ["REL_WANDB_PROJECT"]

        cfg.trainer = cfg.relearning_trainer
        cfg.eval = cfg.relearning_eval
        cfg.mode = "relearn"
        cfg.model.model_args.pretrained_model_name_or_path = str(
            (comm_dir / "best_model").absolute()
        )

        relearning_cfg_path = comm_dir / "relearning_cfg.yaml"
        OmegaConf.save(cfg, relearning_cfg_path)
        subprocess.run(
            [
                "python3",
                "src/train.py",
                f"--config-path={comm_dir.absolute()}",
                "--config-name=relearning_cfg.yaml",
            ],
            check=True,
        )

        robustness = float(open(comm_dir / "robustness.txt").read())
        print(f"Robustness: {robustness}")
        return robustness

    finally:
        shutil.rmtree(comm_dir)


if __name__ == "__main__":
    main()
