#!/bin/bash
# MMLU eval for RepSelectSimple checkpoints.
# Usage: bash scripts/run_mmlu_eval_simple.sh [gpu_id]

GPU_ID=${1:-1}
cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${GPU_ID}
unset NPROC_PER_NODE

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

run_eval() {
    local TASK=$1; local MODEL=$2; shift 2
    echo "=== ${TASK}: MMLU eval ==="
    python src/eval.py \
        experiment=eval/lm_eval_general \
        model=${MODEL} \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK} \
        task_name=${TASK} \
        paths.output_dir=saves/eval/${TASK} \
        "$@" || { echo "FAILED: ${TASK}"; exit 1; }
}

run_eval mmlu_Llama-3.1-8B_bio_RepSelectSimple   Llama-3.1-8B model.model_args.attn_implementation=sdpa
run_eval mmlu_gemma-4-E4B_bio_RepSelectSimple     gemma-4-E4B
run_eval mmlu_Qwen3.5-9B_bio_RepSelectSimple      Qwen3.5-9B

echo "=== All MMLU evals done ==="
