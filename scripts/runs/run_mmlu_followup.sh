#!/bin/bash
# Follow-up MMLU evals after DeepSeek chain completes.
# Runs: Gemma RepSelect eval (saved model), Gemma reference, Llama RepSelect (retrain), Qwen RepSelect (retrain).
# Usage: bash scripts/run_mmlu_followup.sh [gpu_id]

cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

GPU_ID=${1:-3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
unset NPROC_PER_NODE

# Gemma RepSelect eval on saved checkpoint (correct HPs already used)
echo "=== mmlu_gemma-4-E4B_bio_RepSelect: MMLU eval ==="
python src/eval.py \
    experiment=eval/lm_eval_general \
    model=gemma-4-E4B \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/mmlu_gemma-4-E4B_bio_RepSelect \
    task_name=mmlu_gemma-4-E4B_bio_RepSelect \
    paths.output_dir=saves/eval/mmlu_gemma-4-E4B_bio_RepSelect && \
echo "=== mmlu_gemma-4-E4B_bio_RepSelect: removing saved checkpoint ===" && \
find saves/unlearn/mmlu_gemma-4-E4B_bio_RepSelect -name "*.safetensors" -delete

# Gemma-4-E4B reference (base model, direct eval)
echo "=== mmlu_gemma-4-E4B_bio_reference: MMLU eval ==="
python src/eval.py \
    experiment=eval/lm_eval_general \
    model=gemma-4-E4B \
    task_name=mmlu_gemma-4-E4B_bio_reference \
    paths.output_dir=saves/eval/mmlu_gemma-4-E4B_bio_reference

# Llama-3.1-8B RepSelect (retrain with community benchmark HPs + eval)
echo "=== Llama-3.1-8B RepSelect: retraining with correct HPs ==="
bash scripts/run_mmlu_llama8b.sh ${GPU_ID}

# Qwen3.5-9B RepSelect (reference skips if done; retrain with correct HPs + eval)
echo "=== Qwen3.5-9B RepSelect: retraining with correct HPs ==="
bash scripts/run_mmlu_qwen9b.sh ${GPU_ID}

echo "=== All follow-up MMLU evals done. ==="
