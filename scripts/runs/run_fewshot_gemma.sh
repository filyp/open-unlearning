#!/bin/bash
# Run few-shot eval for Gemma-3-1B after Llama finishes
# Usage: nohup bash scripts/run_fewshot_gemma.sh &> saves/logs/fewshot_gemma.log &

set -e
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot.db"
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

# Wait for Llama experiments to finish
echo "Waiting for Llama experiments to finish..."
while pgrep -f "run_fewshot_eval.sh" > /dev/null 2>&1; do
    sleep 60
done
echo "Llama experiments done. Starting Gemma-3-1B..."

version=fewshot_v1
model=gemma-3-1b-pt

wmdp_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=${model}"
wmdp_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${model} trainer.args.num_train_epochs=0"
bt_common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=${model}"
bt_ref="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=${model} trainer.args.num_train_epochs=0"

run_methods() {
    local common="$1" ref="$2" prefix="$3"
    ${ref} trainer=GradDiff task_name=${prefix}_reference
    ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect
    ${common} trainer=GradDiff  hydra/sweeper=GradDiff  task_name=${prefix}_GradDiff
    ${common} trainer=NPO       hydra/sweeper=NPO       task_name=${prefix}_NPO
    ${common} trainer=SimNPO    hydra/sweeper=SimNPO    task_name=${prefix}_SimNPO
    ${common} trainer=RMU       hydra/sweeper=RMU       task_name=${prefix}_RMU
    ${common} trainer=UNDIAL    hydra/sweeper=UNDIAL    task_name=${prefix}_UNDIAL
}

# GPU 0: WMDP-Bio
run_bio() {
    export CUDA_VISIBLE_DEVICES=0
    run_methods "${wmdp_common} wmdp_domain=bio" "${wmdp_ref} wmdp_domain=bio" "${version}_${model}_bio"
}

# GPU 2: WMDP-Cyber
run_cyber() {
    export CUDA_VISIBLE_DEVICES=2
    run_methods "${wmdp_common} wmdp_domain=cyber" "${wmdp_ref} wmdp_domain=cyber" "${version}_${model}_cyber"
}

# GPU 3: BeaverTails
run_bt() {
    export CUDA_VISIBLE_DEVICES=3
    run_methods "${bt_common} category=animal_abuse" "${bt_ref} category=animal_abuse" "${version}_${model}_animal_abuse"
}

run_bio &
run_cyber &
run_bt &
wait
echo "Gemma-3-1B experiments finished."
