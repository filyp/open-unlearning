#!/bin/bash
# Final pass: finish all missing few-shot baselines.
#
# All runs use CUDA_VISIBLE_DEVICES=2 (single GPU, sequential).
# With gc.collect()+empty_cache() at start AND end of fewshot eval, peak GPU usage stays within 80 GB:
#   GradDiff/SimNPO: 54 GB training peak (model 18 + optimizer 18 + gradients 18)
#   NPO/UNDIAL: 72 GB training peak (+ ref model 18, no optimizer for ref)

cd /VData/kebl6672/open-unlearning
source .env
export HF_HOME HF_TOKEN
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

eval "$(/VData/kebl6672/miniconda3/bin/conda shell.bash hook 2>/dev/null)"
conda activate unlearning

mkdir -p saves/logs

run_if_missing() {
    local task_name=$1; shift
    if [ -d "saves/unlearn/${task_name}/eval_histories" ]; then
        echo "SKIP $task_name (already done)"
    else
        echo "RUN  $task_name"
        python src/unlearn_only.py --config-name=unlearn.yaml "$@" task_name=$task_name || true
    fi
}

MEM="trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=8 trainer.args.per_device_eval_batch_size=1 trainer.args.gradient_checkpointing=True"
# device_map=balanced splits the model across both visible GPUs (pipeline parallel).
# Single process => accelerator.num_processes=1 => fewshot eval runs normally.
DM="+model.model_args.device_map=balanced"

LLAMA_BIO="experiment=unlearn/wmdp_low_mi/default model=Llama-3.1-8B wmdp_domain=bio metric_to_optimize=fewshot5_acc_t1 model.model_args.attn_implementation=sdpa $MEM $DM"
LLAMA_BT="experiment=unlearn/beavertails/curated_small model=Llama-3.1-8B category=animal_abuse metric_to_optimize=fewshot5_prob model.model_args.attn_implementation=sdpa $MEM $DM"
QWEN_BIO="experiment=unlearn/wmdp_low_mi/default model=Qwen3.5-9B wmdp_domain=bio metric_to_optimize=fewshot5_acc_t1 $MEM $DM"
QWEN_BT="experiment=unlearn/beavertails/curated_small model=Qwen3.5-9B category=animal_abuse metric_to_optimize=fewshot5_prob $MEM $DM"

export CUDA_VISIBLE_DEVICES=0,1
unset NPROC_PER_NODE
echo "=== Qwen GradDiff + SimNPO (GPUs 0,1 pipeline-parallel) ==="

run_if_missing table4_Qwen3.5-9B_bio_GradDiff $QWEN_BIO \
    trainer=GradDiff trainer.args.learning_rate=3.429e-6 \
    trainer.method_args.alpha=1.453

run_if_missing table4_Qwen3.5-9B_bt_GradDiff $QWEN_BT \
    trainer=GradDiff trainer.args.learning_rate=4.173e-6 \
    trainer.method_args.alpha=1.716

run_if_missing table4_Qwen3.5-9B_bio_SimNPO $QWEN_BIO \
    trainer=SimNPO trainer.args.learning_rate=1.425e-7 \
    trainer.method_args.beta=4.309 trainer.method_args.delta=0.182 trainer.method_args.gamma=0.153

run_if_missing table4_Qwen3.5-9B_bt_SimNPO $QWEN_BT \
    trainer=SimNPO trainer.args.learning_rate=2.841e-7 \
    trainer.method_args.beta=4.258 trainer.method_args.delta=0.267 trainer.method_args.gamma=0.160

echo "=== Llama NPO + UNDIAL (GPUs 0,1 pipeline-parallel) ==="

run_if_missing table4_Llama-3.1-8B_bio_NPO $LLAMA_BIO \
    trainer=NPO trainer.args.learning_rate=3.848e-6 \
    trainer.method_args.alpha=1.513 trainer.method_args.beta=0.478

run_if_missing table4_Llama-3.1-8B_bt_NPO $LLAMA_BT \
    trainer=NPO trainer.args.learning_rate=9.491e-6 \
    trainer.method_args.alpha=1.315 trainer.method_args.beta=0.355

run_if_missing table4_Llama-3.1-8B_bio_UNDIAL $LLAMA_BIO \
    trainer=UNDIAL trainer.args.learning_rate=4.423e-6 \
    trainer.method_args.alpha=2.544 trainer.method_args.beta=5.769

run_if_missing table4_Llama-3.1-8B_bt_UNDIAL $LLAMA_BT \
    trainer=UNDIAL trainer.args.learning_rate=3.659e-7 \
    trainer.method_args.alpha=1.090 trainer.method_args.beta=5.002

echo "=== Qwen NPO + UNDIAL (GPUs 0,1 pipeline-parallel) ==="

run_if_missing table4_Qwen3.5-9B_bio_NPO $QWEN_BIO \
    trainer=NPO trainer.args.learning_rate=4.386e-6 \
    trainer.method_args.alpha=2.792 trainer.method_args.beta=0.111

run_if_missing table4_Qwen3.5-9B_bt_NPO $QWEN_BT \
    trainer=NPO trainer.args.learning_rate=1.300e-5 \
    trainer.method_args.alpha=1.452 trainer.method_args.beta=0.281

run_if_missing table4_Qwen3.5-9B_bio_UNDIAL $QWEN_BIO \
    trainer=UNDIAL trainer.args.learning_rate=3.208e-7 \
    trainer.method_args.alpha=1.684 trainer.method_args.beta=17.340

run_if_missing table4_Qwen3.5-9B_bt_UNDIAL $QWEN_BT \
    trainer=UNDIAL trainer.args.learning_rate=1.963e-7 \
    trainer.method_args.alpha=2.801 trainer.method_args.beta=4.173

echo "All done."
