#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# # Unlearn
# CUDA_VISIBLE_DEVICES=0 \
# python src/train.py --config-name=unlearn.yaml \
# experiment=${experiment} \
# trainer=${trainer} \
# task_name=${task_name} \
# model=${model} \
# forget_split=${forget_split} \
# retain_split=${retain_split} \
# model.model_args.pretrained_model_name_or_path=${model_path} \
# retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
# trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
# trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
# trainer.args.eval_strategy=no \
# trainer.args.eval_on_start=False \
# trainer.args.learning_rate=$lr \
# trainer.method_args.beta=$beta \
# trainer.method_args.alpha=$alpha


common="python3 src/unlearn_relearn.py \
--multirun \
--config-name=unlearn.yaml \
experiment=unlearn/wmdp_low_mi/default"

cyber="data.custom_loaders.0.dataset=cyber \
data.custom_loaders.1.hf_args.data_files=[computer_science_and_technology/computer_science_and_technology_000000.jsonl]"
ver=v3


# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch $HOME/open-unlearning/runners/slurm_runner.sh $common"
fi


$common model=Llama-3.2-3B trainer=GradDiff task_name=${ver}_3B_GradDiff_bio
$common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPO_bio
$common model=Llama-3.2-3B trainer=RMU task_name=${ver}_3B_RMU_bio
$common model=Llama-3.2-3B trainer=SimNPO task_name=${ver}_3B_SimNPO_bio
$common model=Llama-3.2-3B trainer=UNDIAL task_name=${ver}_3B_UNDIAL_bio
$common model=Llama-3.2-3B trainer=CIR task_name=${ver}_3B_CIR_bio
$common model=Llama-3.2-3B trainer=CIR task_name=${ver}_3B_CIRstrict_bio eval.wikitext_kl.disr_budget=0.005
$common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPOstrict_bio eval.wikitext_kl.disr_budget=0.005

$common model=Llama-3.2-3B trainer=GradDiff task_name=${ver}_3B_GradDiff_cyber $cyber
$common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPO_cyber $cyber
$common model=Llama-3.2-3B trainer=RMU task_name=${ver}_3B_RMU_cyber $cyber
$common model=Llama-3.2-3B trainer=SimNPO task_name=${ver}_3B_SimNPO_cyber $cyber
$common model=Llama-3.2-3B trainer=UNDIAL task_name=${ver}_3B_UNDIAL_cyber $cyber
$common model=Llama-3.2-3B trainer=CIR task_name=${ver}_3B_CIR_cyber $cyber
$common model=Llama-3.2-3B trainer=CIR task_name=${ver}_3B_CIRstrict_cyber eval.wikitext_kl.disr_budget=0.005 $cyber
$common model=Llama-3.2-3B trainer=NPO task_name=${ver}_3B_NPOstrict_cyber eval.wikitext_kl.disr_budget=0.005 $cyber

# alternatively, to run on SLURM
common="sbatch open-unlearning/job.sh"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml