#!/bin/bash
# Run few-shot attack evaluation + ARC-Challenge for all methods
# Usage: bash scripts/run_fewshot_eval.sh
# Parallelised across 3 GPUs (0, 2, 3)

set -e
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot.db"
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

version=fewshot_v1

###############################################################
# WMDP experiments
###############################################################
wmdp_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default"
wmdp_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0"

# --- GPU 0: WMDP-Bio (Llama-3.2-3B) ---
run_wmdp_bio() {
    export CUDA_VISIBLE_DEVICES=0
    model=Llama-3.2-3B
    domain=bio
    prefix="${version}_${model}_${domain}"
    common="${wmdp_common} model=${model} wmdp_domain=${domain}"
    ref="${wmdp_ref} model=${model} wmdp_domain=${domain}"

    # Reference (no unlearning)
    ${ref} trainer=GradDiff task_name=${prefix}_reference

    # All methods
    ${common} trainer=RepSelect   hydra/sweeper=RepSelect   task_name=${prefix}_RepSelect
    ${common} trainer=GradDiff    hydra/sweeper=GradDiff    task_name=${prefix}_GradDiff
    ${common} trainer=NPO         hydra/sweeper=NPO         task_name=${prefix}_NPO
    ${common} trainer=SimNPO      hydra/sweeper=SimNPO      task_name=${prefix}_SimNPO
    ${common} trainer=RMU         hydra/sweeper=RMU         task_name=${prefix}_RMU
    ${common} trainer=UNDIAL      hydra/sweeper=UNDIAL      task_name=${prefix}_UNDIAL
}

# --- GPU 2: WMDP-Cyber (Llama-3.2-3B) ---
run_wmdp_cyber() {
    export CUDA_VISIBLE_DEVICES=2
    model=Llama-3.2-3B
    domain=cyber
    prefix="${version}_${model}_${domain}"
    common="${wmdp_common} model=${model} wmdp_domain=${domain}"
    ref="${wmdp_ref} model=${model} wmdp_domain=${domain}"

    ${ref} trainer=GradDiff task_name=${prefix}_reference

    ${common} trainer=RepSelect   hydra/sweeper=RepSelect   task_name=${prefix}_RepSelect
    ${common} trainer=GradDiff    hydra/sweeper=GradDiff    task_name=${prefix}_GradDiff
    ${common} trainer=NPO         hydra/sweeper=NPO         task_name=${prefix}_NPO
    ${common} trainer=SimNPO      hydra/sweeper=SimNPO      task_name=${prefix}_SimNPO
    ${common} trainer=RMU         hydra/sweeper=RMU         task_name=${prefix}_RMU
    ${common} trainer=UNDIAL      hydra/sweeper=UNDIAL      task_name=${prefix}_UNDIAL
}

# --- GPU 3: BeaverTails (Llama-3.2-3B) ---
run_beavertails() {
    export CUDA_VISIBLE_DEVICES=3
    model=Llama-3.2-3B
    category=animal_abuse
    prefix="${version}_${model}_${category}"
    bt_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=${model} category=${category}"
    bt_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=${model} category=${category} trainer.args.num_train_epochs=0"

    ${bt_ref} trainer=GradDiff task_name=${prefix}_reference

    ${bt_common} trainer=RepSelect   hydra/sweeper=RepSelect   task_name=${prefix}_RepSelect
    ${bt_common} trainer=GradDiff    hydra/sweeper=GradDiff    task_name=${prefix}_GradDiff
    ${bt_common} trainer=NPO         hydra/sweeper=NPO         task_name=${prefix}_NPO
    ${bt_common} trainer=SimNPO      hydra/sweeper=SimNPO      task_name=${prefix}_SimNPO
    ${bt_common} trainer=RMU         hydra/sweeper=RMU         task_name=${prefix}_RMU
    ${bt_common} trainer=UNDIAL      hydra/sweeper=UNDIAL      task_name=${prefix}_UNDIAL
}

###############################################################
# Launch all three in parallel
###############################################################
echo "Starting WMDP-Bio on GPU 0..."
run_wmdp_bio &> saves/logs/fewshot_wmdp_bio.log &
PID_BIO=$!

echo "Starting WMDP-Cyber on GPU 2..."
run_wmdp_cyber &> saves/logs/fewshot_wmdp_cyber.log &
PID_CYBER=$!

echo "Starting BeaverTails on GPU 3..."
run_beavertails &> saves/logs/fewshot_beavertails.log &
PID_BT=$!

echo "Running in background: PIDs ${PID_BIO} ${PID_CYBER} ${PID_BT}"
echo "Logs: saves/logs/fewshot_wmdp_{bio,cyber}.log, saves/logs/fewshot_beavertails.log"
wait
echo "All experiments finished."
