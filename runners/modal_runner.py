# Run from repo root:
#   modal run runners/modal_runner.py "python3 src/train.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer=RepSelect task_name=test"

import subprocess

import modal


def download_models():
    from huggingface_hub import snapshot_download
    # note that modal images are up to 30GB, so be mindful of what models you fit

    for model in [
        # "allenai/OLMoE-1B-7B-0125",
        "google/gemma-2-2b",
        # "meta-llama/Llama-3.2-3B",
    ]:
        snapshot_download(model)


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("lm-eval==0.4.11")
    # .pip_install("flash-attn==2.6.3", extra_options="--no-build-isolation")
    # .pip_install("flash-attn==2.8.3", extra_options="--no-build-isolation")
    # if we move to torch>2.5, we need to use pre-built wheels from here, because the build is painfully slow
    # also to support B200, flash-attn==2.6.3 is too old, we'd need to bump to e.g. 2.8.3
    .pip_install(
        # "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl"
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
    )
    # things above, are equivalent to using docker filyp/open-unlearning:latest
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    .run_function(download_models, secrets=[modal.Secret.from_dotenv()])
    .add_local_dir("data", remote_path="/root/code/data")
    .add_local_dir(".cache/load_hf", remote_path="/root/code/.cache/load_hf")
    .add_local_dir("configs", remote_path="/root/code/configs")
    .add_local_dir("src", remote_path="/root/code/src")
)

app = modal.App("open-unlearning", image=image)


@app.function(
    gpu="L40S",  # 48GB
    # gpu="A100-80GB",  # if needing 80GB
    # gpu="H100",  # 80GB
    # gpu="H200",  # 141GB
    # gpu="B200",  # 180GB
    # B300 262GB, but can't be set in modal
    timeout=3 * 3600,
    secrets=[modal.Secret.from_dotenv()],
)
def run_training(args: str):
    cmd = f"cd /root/code && HF_HUB_DOWNLOAD_TIMEOUT=60 PYTHONUNBUFFERED=1 {args}"

    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, executable="/bin/bash", check=True)


@app.local_entrypoint()
def main(args: str):
    run_training.remote(args)
