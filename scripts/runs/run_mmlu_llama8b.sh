#!/bin/bash
# MMLU utility evaluation for Llama-3.1-8B WMDP-Bio.
# Strategy: save final model after unlearning, then run eval.py for MMLU.
# Results saved to saves/eval/mmlu_<task_name>/.
# Usage: bash scripts/run_mmlu_llama8b.sh [gpu_id]

cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

GPU_ID=${1:-1}
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
        model=Llama-3.1-8B model.model_args.attn_implementation=sdpa \
        trainer.args.per_device_train_batch_size=1 \
        trainer.args.per_device_eval_batch_size=1 \
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
        model=Llama-3.1-8B \
        model.model_args.attn_implementation=sdpa \
        model.model_args.pretrained_model_name_or_path=${model_path} \
        task_name=${task} \
        paths.output_dir=saves/eval/${task} && \
    echo "=== $task: removing saved checkpoint ===" && \
    find ${model_path} -name "*.safetensors" -delete
}

# RepSelectSimple — community benchmark HPs (optimised for fine-tuning attack)
run_bio RepSelectSimple mmlu_Llama-3.1-8B_bio_RepSelect \
    trainer.args.learning_rate=0.115643277655407 \
    trainer.method_args.lora_lr=0.050129327532378


echo "=== All done. MMLU results in saves/eval/mmlu_Llama-3.1-8B_bio_*/ ==="
