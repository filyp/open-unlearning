#!/bin/bash

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

# gemma-4-E4B, Llama-3.1-8B, DeepSeek-V2-Lite, Qwen3.5-9B
model=$1

wmdp_domain='bio'
# wmdp_domain='cyber'

version=v5.3
# v5 uses 7e-6 relearning LR

common="run python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain}"
reference="run python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0 model=${model} wmdp_domain=${wmdp_domain}"
prefix="${version}_${model}_${wmdp_domain}"

# for running on verda:
run() {
  bash verda_runner.sh $*
}

# # for running on modal:
# source .venv/bin/activate
# run() {
#   modal run runners/modal_runner.py --args "$*"
# }

# MoE requires 30-100x larger LRs - other methods use adam so it's fine, but with sgd, we need to shift the range
case "${model}" in
  DeepSeek-V2-Lite|Qwen3-30B-A3B) rs_sweeper=RepSelectSimpleMoE ;;
  *)                              rs_sweeper=RepSelectSimple ;;
esac

###############################################################

${reference} trainer=GradDiff task_name=${prefix}_reference

# Main experiments
${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff
${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO
${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU
${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO
${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL

${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} task_name=${prefix}_RepSelectSimple2

# # ABLATIONS
# ${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} \
#   trainer.method_args.use_lora=false \
#   task_name=${prefix}_RepSelectSimple_no_lora
# ${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} \
#   trainer.method_args.distribution=none \
#   task_name=${prefix}_RepSelectSimple_no_pcs
# ${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} \
#   trainer.method_args.distribution=retain \
#   task_name=${prefix}_RepSelectSimple_retain

# # RepSelect old continuous version
# if [ "${model}" = "DeepSeek-V2-Lite" ]; then  # also add other MoE models here
#     ${common} trainer=RepSelect hydra/sweeper=RepSelectMoE task_name=${prefix}_RepSelect_forget trainer.method_args.cfg.use_distribution=forget trainer.handler=RepSelectMOE
#     ${common} trainer=RepSelect hydra/sweeper=RepSelectMoE task_name=${prefix}_RepSelect_retain trainer.method_args.cfg.use_distribution=retain trainer.handler=RepSelectMOE
# else
#     ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect_forget trainer.method_args.cfg.use_distribution=forget
#     ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect_retain trainer.method_args.cfg.use_distribution=retain
# fi