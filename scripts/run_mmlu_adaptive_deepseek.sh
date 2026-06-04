#!/bin/bash
# MMLU eval for DeepSeek-V2-Lite with RepSelectAdaptive (KL-budget controlled).
# Usage: bash scripts/run_mmlu_adaptive_deepseek.sh [gpu_id]

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

TASK=mmlu_DeepSeek-V2-Lite_bio_RepSelectAdaptive
MODEL_PATH=saves/unlearn/${TASK}

echo "=== ${TASK}: unlearning ==="
python src/unlearn_only.py --config-name=unlearn.yaml \
    experiment=unlearn/wmdp_low_mi/default wmdp_domain=bio \
    trainer=RepSelectAdaptive task_name=${TASK} \
    model=DeepSeek-V2-Lite \
    model.model_args.attn_implementation=sdpa \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.per_device_eval_batch_size=1 \
    trainer.args.gradient_checkpointing=true \
    eval.retain_eval=null eval.wmdp_low_mi=null \
    eval.recall_prob=null eval.fewshot_attack_0=null \
    eval.fewshot_attack=null eval.fewshot_attack_10=null \
    metric_to_optimize=null \
    trainer.args.eval_strategy=no \
    trainer.save_final_state=true \
    trainer.args.learning_rate=13.6977203864842 \
    trainer.method_args.lora_lr=1.04250201741007 || exit 1

echo "=== ${TASK}: MMLU eval ==="
python src/eval.py \
    experiment=eval/lm_eval_general \
    model=DeepSeek-V2-Lite \
    model.model_args.attn_implementation=sdpa \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    task_name=${TASK} \
    paths.output_dir=saves/eval/${TASK} && \
echo "=== ${TASK}: removing checkpoint ===" && \
find ${MODEL_PATH} -name "*.safetensors" -delete

echo "=== DeepSeek-V2-Lite RepSelectAdaptive done ==="
