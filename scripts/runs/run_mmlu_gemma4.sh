#!/bin/bash
# MMLU utility evaluation for Gemma-4-E4B WMDP-Bio.
# Strategy: save final model after unlearning, then run eval.py for MMLU.
# Results saved to saves/eval/mmlu_<task_name>/.
# Usage: bash scripts/run_mmlu_gemma4.sh [gpu_id]

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

mkdir -p saves/logs

run_bio() {
    local trainer=$1; local task=$2; shift 2; local extra="$@"
    local model_path="saves/unlearn/${task}"
    echo "=== $task: unlearning ==="
    python src/unlearn_only.py --config-name=unlearn.yaml \
        experiment=unlearn/wmdp_low_mi/default wmdp_domain=bio \
        trainer=$trainer task_name=$task \
        model=gemma-4-E4B \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.per_device_eval_batch_size=1 \
        trainer.args.gradient_checkpointing=true \
        eval.wikitext_kl=null eval.retain_eval=null eval.wmdp_low_mi=null \
        eval.recall_prob=null eval.fewshot_attack_0=null \
        eval.fewshot_attack=null eval.fewshot_attack_10=null \
        metric_to_optimize=null \
        trainer.args.eval_strategy=no \
        trainer.save_final_state=true \
        $extra || return 1

    echo "=== $task: MMLU eval ==="
    python src/eval.py \
        experiment=eval/lm_eval_general \
        model=gemma-4-E4B \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        task_name=${task} \
        paths.output_dir=saves/eval/${task} && \
    echo "=== $task: removing saved checkpoint ===" && \
    find ${model_path} -name "*.safetensors" -delete
}

# Reference: base Gemma-4-E4B MMLU (direct eval, no unlearning)
echo "=== mmlu_gemma-4-E4B_bio_reference: MMLU eval ==="
python src/eval.py \
    experiment=eval/lm_eval_general \
    model=gemma-4-E4B \
    task_name=mmlu_gemma-4-E4B_bio_reference \
    paths.output_dir=saves/eval/mmlu_gemma-4-E4B_bio_reference

# RepSelectSimple — best bio: lr=0.0268, lora_lr=0.032
run_bio RepSelectSimple mmlu_gemma-4-E4B_bio_RepSelect \
    trainer.args.learning_rate=0.0267561068095576 \
    trainer.method_args.lora_lr=0.0319953095819889


echo "=== All Gemma-4-E4B MMLU runs done. Results in saves/eval/mmlu_gemma-4-E4B_bio_*/ ==="
