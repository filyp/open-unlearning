# Unlearn-only pipeline: no relearning attack. Returns best few-shot robustness metric.
# Usage: python3 src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=test
import os
import signal
import shutil
import subprocess
import uuid
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

load_dotenv()


def _get_run_name(cfg: DictConfig) -> str:
    try:
        job_num = HydraConfig.get().job.num
        return f"{cfg.task_name}_{job_num}"
    except Exception:
        return cfg.task_name


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    cfg.trainer.args.run_name = _get_run_name(cfg)

    suffix = cfg.task_name + "_" + uuid.uuid4().hex[:8]
    cfg.paths.tmp_comm_dir = str(Path(cfg.paths.tmp_comm_dir) / suffix)
    comm_dir = Path(cfg.paths.tmp_comm_dir)
    comm_dir.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: exit(1))

    try:
        if "UNL_WANDB_PROJECT" in os.environ:
            os.environ["WANDB_PROJECT"] = os.environ["UNL_WANDB_PROJECT"]
        unlearning_cfg_path = comm_dir / "unlearning_cfg.yaml"
        OmegaConf.save(cfg, unlearning_cfg_path)
        nproc = int(os.environ.get("NPROC_PER_NODE", "1"))
        if nproc > 1:
            cmd = [
                "torchrun", f"--nproc_per_node={nproc}", "--master_port=29501",
                "src/train.py",
                f"--config-path={comm_dir.absolute()}",
                "--config-name=unlearning_cfg.yaml",
            ]
        else:
            cmd = [
                "python3", "src/train.py",
                f"--config-path={comm_dir.absolute()}",
                "--config-name=unlearning_cfg.yaml",
            ]
        subprocess.run(cmd, check=True)

        # Return the optimisation metric from the last valid eval
        robustness_file = comm_dir / "robustness.txt"
        if robustness_file.exists():
            robustness = float(robustness_file.read_text())
            print(f"Robustness: {robustness}")
            return robustness
        return 0.0

    finally:
        shutil.rmtree(comm_dir)


if __name__ == "__main__":
    main()
