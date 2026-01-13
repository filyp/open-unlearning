# example usage:
# python3 src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=1B_cir
import os
import shutil
import subprocess

from pathlib import Path
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from omegaconf import OmegaConf

load_dotenv()


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    comm_dir = Path(cfg.paths.tmp_comm_dir) / cfg.task_name
    comm_dir.mkdir(parents=True, exist_ok=False)
    try:
        # ! unlearning #########################################################
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
        if "WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = "rel-" + os.environ["WANDB_PROJECT"]

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
