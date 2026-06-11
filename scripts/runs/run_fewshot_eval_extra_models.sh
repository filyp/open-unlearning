#!/bin/bash
# Run few-shot eval for Qwen3-8B and Gemma-3-1B
# Usage: bash scripts/run_fewshot_eval_extra_models.sh
# Parallelised across GPUs 0, 2, 3

set -e
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot.db"
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

version=fewshot_v1

wmdp_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default"
wmdp_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0"
bt_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small"
bt_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small trainer.args.num_train_epochs=0"

run_all_methods() {
    local common="$1"
    local ref="$2"
    local prefix="$3"

    ${ref} trainer=GradDiff task_name=${prefix}_reference
    ${common} trainer=RepSelect   hydra/sweeper=RepSelect   task_name=${prefix}_RepSelect
    ${common} trainer=GradDiff    hydra/sweeper=GradDiff    task_name=${prefix}_GradDiff
    ${common} trainer=NPO         hydra/sweeper=NPO         task_name=${prefix}_NPO
    ${common} trainer=SimNPO      hydra/sweeper=SimNPO      task_name=${prefix}_SimNPO
    ${common} trainer=RMU         hydra/sweeper=RMU         task_name=${prefix}_RMU
    ${common} trainer=UNDIAL      hydra/sweeper=UNDIAL      task_name=${prefix}_UNDIAL
}

###############################################################
# GPU 0: Qwen3-8B — WMDP-Bio + WMDP-Cyber
###############################################################
run_qwen_wmdp() {
    export CUDA_VISIBLE_DEVICES=0
    model=Qwen3-8B-Base

    # Bio
    prefix="${version}_${model}_bio"
    run_all_methods \
        "${wmdp_common} model=${model} wmdp_domain=bio" \
        "${wmdp_ref} model=${model} wmdp_domain=bio" \
        "${prefix}"

    # Cyber
    prefix="${version}_${model}_cyber"
    run_all_methods \
        "${wmdp_common} model=${model} wmdp_domain=cyber" \
        "${wmdp_ref} model=${model} wmdp_domain=cyber" \
        "${prefix}"
}

###############################################################
# GPU 2: Gemma-3-1B — WMDP-Bio + WMDP-Cyber
###############################################################
run_gemma_wmdp() {
    export CUDA_VISIBLE_DEVICES=2
    model=gemma-3-1b-pt

    # Bio
    prefix="${version}_${model}_bio"
    run_all_methods \
        "${wmdp_common} model=${model} wmdp_domain=bio" \
        "${wmdp_ref} model=${model} wmdp_domain=bio" \
        "${prefix}"

    # Cyber
    prefix="${version}_${model}_cyber"
    run_all_methods \
        "${wmdp_common} model=${model} wmdp_domain=cyber" \
        "${wmdp_ref} model=${model} wmdp_domain=cyber" \
        "${prefix}"
}

###############################################################
# GPU 3: Both models — BeaverTails
###############################################################
run_beavertails_extra() {
    export CUDA_VISIBLE_DEVICES=3
    category=animal_abuse

    # Qwen3-8B
    model=Qwen3-8B-Base
    prefix="${version}_${model}_${category}"
    run_all_methods \
        "${bt_common} model=${model} category=${category}" \
        "${bt_ref} model=${model} category=${category}" \
        "${prefix}"

    # Gemma-3-1B
    model=gemma-3-1b-pt
    prefix="${version}_${model}_${category}"
    run_all_methods \
        "${bt_common} model=${model} category=${category}" \
        "${bt_ref} model=${model} category=${category}" \
        "${prefix}"
}

###############################################################
# Launch
###############################################################
mkdir -p saves/logs

echo "Starting Qwen3-8B WMDP on GPU 0..."
run_qwen_wmdp &> saves/logs/fewshot_qwen_wmdp.log &

echo "Starting Gemma-3-1B WMDP on GPU 2..."
run_gemma_wmdp &> saves/logs/fewshot_gemma_wmdp.log &

echo "Starting BeaverTails (both models) on GPU 3..."
run_beavertails_extra &> saves/logs/fewshot_extra_beavertails.log &

echo "Running in background. Monitor:"
echo "  tail -f saves/logs/fewshot_qwen_wmdp.log"
echo "  tail -f saves/logs/fewshot_gemma_wmdp.log"
echo "  tail -f saves/logs/fewshot_extra_beavertails.log"
wait
echo "All extra model experiments finished."
