#!/bin/bash
# Optuna hyperparameter search for RepSelectSimple, 10 trials each.
# All four tasks sequential on GPU 0.
# RepSelectSimple uses SGD — fits Qwen 9B on one 80 GB GPU (~40 GB peak).

cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
unset NPROC_PER_NODE

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

mkdir -p saves/logs

COMMON="trainer=RepSelectSimple hydra/sweeper=RepSelectSimple hydra.sweeper.n_trials=10 trainer.args.per_device_eval_batch_size=1"
# Only run fewshot5 and fewshot10 evals; disable everything else
BIO_SKIP="eval.wmdp_low_mi=null eval.recall_prob=null eval.retain_eval=null eval.wikitext_kl=null"
BT_SKIP="eval.holdout_harmful=null eval.fewshot_attack_0=null eval.wikitext=null"

echo "=== Llama-3.1-8B bio ==="
export OPTUNA_STORAGE_URL="sqlite:///optuna_repselect_table4_llama8b.db"
python src/unlearn_only.py --config-name=unlearn.yaml --multirun \
    experiment=unlearn/wmdp_low_mi/default model=Llama-3.1-8B wmdp_domain=bio \
    metric_to_optimize=fewshot5_acc_t1 model.model_args.attn_implementation=sdpa \
    task_name=optuna_rs_Llama-3.1-8B_bio $COMMON $BIO_SKIP \
    2>&1 | tee saves/logs/optuna_rs_llama8b_bio.log

echo "=== Llama-3.1-8B bt ==="
python src/unlearn_only.py --config-name=unlearn.yaml --multirun \
    experiment=unlearn/beavertails/curated_small model=Llama-3.1-8B category=animal_abuse \
    metric_to_optimize=fewshot5_prob model.model_args.attn_implementation=sdpa \
    task_name=optuna_rs_Llama-3.1-8B_bt $COMMON $BT_SKIP \
    2>&1 | tee saves/logs/optuna_rs_llama8b_bt.log

echo "=== Qwen3.5-9B bio ==="
export OPTUNA_STORAGE_URL="sqlite:///optuna_repselect_table4_qwen9b.db"
python src/unlearn_only.py --config-name=unlearn.yaml --multirun \
    experiment=unlearn/wmdp_low_mi/default model=Qwen3.5-9B wmdp_domain=bio \
    metric_to_optimize=fewshot5_acc_t1 \
    task_name=optuna_rs_Qwen3.5-9B_bio $COMMON $BIO_SKIP \
    2>&1 | tee saves/logs/optuna_rs_qwen9b_bio.log

echo "=== Qwen3.5-9B bt ==="
python src/unlearn_only.py --config-name=unlearn.yaml --multirun \
    experiment=unlearn/beavertails/curated_small model=Qwen3.5-9B category=animal_abuse \
    metric_to_optimize=fewshot5_prob \
    task_name=optuna_rs_Qwen3.5-9B_bt $COMMON $BT_SKIP \
    2>&1 | tee saves/logs/optuna_rs_qwen9b_bt.log

echo "All done."
