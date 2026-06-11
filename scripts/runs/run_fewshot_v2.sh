#!/bin/bash
# GPU 0: Llama-3.2-3B WMDP-Bio
# GPU 2: Gemma-3-1B WMDP-Bio + BeaverTails
# GPU 3: Llama-3.2-3B BeaverTails

set -e
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot.db"
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

version=fewshot_v1

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

# GPU 0: Llama Bio
run_llama_bio() {
    export CUDA_VISIBLE_DEVICES=0
    local m=Llama-3.2-3B p=${version}_Llama-3.2-3B_bio
    run_methods \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=${m} wmdp_domain=bio" \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${m} wmdp_domain=bio trainer.args.num_train_epochs=0" \
        "${p}"
}

# GPU 2: Gemma Bio then BeaverTails
run_gemma() {
    export CUDA_VISIBLE_DEVICES=2
    local m=gemma-3-1b-pt

    # Bio
    run_methods \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=${m} wmdp_domain=bio" \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=${m} wmdp_domain=bio trainer.args.num_train_epochs=0" \
        "${version}_${m}_bio"

    # BeaverTails
    run_methods \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=${m} category=animal_abuse" \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=${m} category=animal_abuse trainer.args.num_train_epochs=0" \
        "${version}_${m}_animal_abuse"
}

# GPU 3: Llama BeaverTails
run_llama_bt() {
    export CUDA_VISIBLE_DEVICES=3
    local m=Llama-3.2-3B p=${version}_Llama-3.2-3B_animal_abuse
    run_methods \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=${m} category=animal_abuse" \
        "python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=${m} category=animal_abuse trainer.args.num_train_epochs=0" \
        "${p}"
}

mkdir -p saves/logs

echo "Starting Llama Bio on GPU 0..."
run_llama_bio &> saves/logs/fewshot_v2_llama_bio.log &

echo "Starting Gemma Bio+BT on GPU 2..."
run_gemma &> saves/logs/fewshot_v2_gemma.log &

echo "Starting Llama BT on GPU 3..."
run_llama_bt &> saves/logs/fewshot_v2_llama_bt.log &

echo "Logs: saves/logs/fewshot_v2_{llama_bio,gemma,llama_bt}.log"
wait
echo "All done."
