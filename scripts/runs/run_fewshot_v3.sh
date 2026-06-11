#!/bin/bash
# Few-shot eval: unlearn only (no fine-tuning relearning attack)
# k=5 and k=10 few-shot evaluated at every epoch
# Overrides metric_to_optimize via CLI so default configs stay unchanged
# GPU 0: Llama-3.2-3B Bio
# GPU 2: Gemma-3-1B Bio + BeaverTails
# GPU 3: Llama-3.2-3B BeaverTails

cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

version=fewshot_v3

# Override metric_to_optimize per benchmark (CLI overrides, not config changes)
wmdp_metric="metric_to_optimize=fewshot5_acc_t1"
bt_metric="metric_to_optimize=fewshot5_prob"

run_methods() {
    local common="$1" ref="$2" prefix="$3"
    ${ref} trainer=GradDiff task_name=${prefix}_reference || true
    ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect || true
    ${common} trainer=GradDiff  hydra/sweeper=GradDiff  task_name=${prefix}_GradDiff || true
    ${common} trainer=NPO       hydra/sweeper=NPO       task_name=${prefix}_NPO || true
    ${common} trainer=SimNPO    hydra/sweeper=SimNPO    task_name=${prefix}_SimNPO || true
    ${common} trainer=RMU       hydra/sweeper=RMU       task_name=${prefix}_RMU || true
    ${common} trainer=UNDIAL    hydra/sweeper=UNDIAL    task_name=${prefix}_UNDIAL || true
}

# GPU 0: Llama Bio
run_llama_bio() {
    export CUDA_VISIBLE_DEVICES=0
    export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot_v3_gpu0.db"
    local m=Llama-3.2-3B
    run_methods \
        "python src/unlearn_only.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=$m wmdp_domain=bio $wmdp_metric" \
        "python src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=$m wmdp_domain=bio trainer.args.num_train_epochs=0 $wmdp_metric" \
        "${version}_${m}_bio"
}

# GPU 2: Gemma Bio then BeaverTails (smaller eval batch to avoid OOM)
run_gemma() {
    export CUDA_VISIBLE_DEVICES=2
    export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot_v3_gpu2.db"
    local m=gemma-3-1b-pt
    local extra="trainer.args.per_device_eval_batch_size=4"
    run_methods \
        "python src/unlearn_only.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=$m wmdp_domain=bio $extra $wmdp_metric" \
        "python src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default model=$m wmdp_domain=bio trainer.args.num_train_epochs=0 $extra $wmdp_metric" \
        "${version}_${m}_bio"
    run_methods \
        "python src/unlearn_only.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=$m category=animal_abuse $extra $bt_metric" \
        "python src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=$m category=animal_abuse trainer.args.num_train_epochs=0 $extra $bt_metric" \
        "${version}_${m}_animal_abuse"
}

# GPU 3: Llama BeaverTails
run_llama_bt() {
    export CUDA_VISIBLE_DEVICES=3
    export OPTUNA_STORAGE_URL="sqlite:///optuna_fewshot_v3_gpu3.db"
    local m=Llama-3.2-3B
    run_methods \
        "python src/unlearn_only.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_small model=$m category=animal_abuse $bt_metric" \
        "python src/unlearn_only.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_small model=$m category=animal_abuse trainer.args.num_train_epochs=0 $bt_metric" \
        "${version}_${m}_animal_abuse"
}

mkdir -p saves/logs

echo "GPU 0: Llama Bio"
run_llama_bio &> saves/logs/fewshot_v3_llama_bio.log &

echo "GPU 2: Gemma Bio + BT"
run_gemma &> saves/logs/fewshot_v3_gemma.log &

echo "GPU 3: Llama BT"
run_llama_bt &> saves/logs/fewshot_v3_llama_bt.log &

echo "Logs: saves/logs/fewshot_v3_*.log"
wait
echo "All done."
