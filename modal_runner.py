# Run from repo root:
#   modal run modal_runner.py --args 'python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=CIR task_name=test'

import subprocess

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("peft==0.17.1")  # pin to compatible version with transformers 4.45.1
    .pip_install("lm-eval==0.4.8")
    .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    .add_local_dir(
        ".",
        remote_path="/root/code",
        ignore=[".git", ".venv", "**/__pycache__", "saves", "*.pyc", "profile.prof", "wandb", "multirun", ".ruff_cache", "build", "logs"],
    )
)

app = modal.App("open-unlearning", image=image)


@app.function(
    gpu="L40S",  # 48GB
    # gpu="A100-80GB",  # if needing 80GB
    # gpu="H100",  # it's not any faster than L40S, at least for wmdp_low_mi
    timeout=1 * 3600,
)
def run_training(args: str):
    cmd = f"cd /root/code && HF_HUB_DOWNLOAD_TIMEOUT=60 PYTHONUNBUFFERED=1 {args}"

    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


@app.local_entrypoint()
def main(args: str):
    run_training.remote(args)
