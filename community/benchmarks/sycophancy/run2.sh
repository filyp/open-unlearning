#!/bin/bash

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

# gemma-4-E4B, Llama-3.1-8B, DeepSeek-V2-Lite, Qwen3.5-9B
model=$1

version=v1

common="run python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/sycophancy/default model=${model}"
prefix="${version}_${model}_sycophancy"

# for running on verda:
run() {
  bash verda_runner.sh $*
}

# MoE requires 30-100x larger LRs - other methods use adam so it's fine, but with sgd, we need to shift the range
case "${model}" in
  DeepSeek-V2-Lite|Qwen3-30B-A3B) rs_sweeper=RepSelectSimpleMoE ;;
  *)                              rs_sweeper=RepSelectSimple ;;
esac

# # DeepSeek is run in two batches (compute constraints):
# #   1st pass: 10 trials per study (line below)
# #   2nd pass: comment the first line, uncomment the resume line ->
# #             20 more trials per study, wandb run names continue at _10
# if [ "${model}" = "DeepSeek-V2-Lite" ]; then
#   common="${common} hydra.sweeper.n_trials=10"
#   # common="${common} hydra.sweeper.n_trials=20 run_name_offset=10"  # resume
# fi

###############################################################

# (no ${prefix}_reference run: the baseline_sycophancy_${model} runs from
# run_baselines.sh already cover the no-unlearning reference)

# Main experiments
${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff
${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO
${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU
${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO
${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL

${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} task_name=${prefix}_RepSelectSimple_forget

# ABLATIONS
${common} trainer=RepSelectSimple hydra/sweeper=${rs_sweeper} \
  trainer.method_args.use_lora=false \
  task_name=${prefix}_RepSelectSimple_forget_no_lora

# RepSelect old continuous version
if [ "${model}" = "DeepSeek-V2-Lite" ]; then  # also add other MoE models here
    ${common} trainer=RepSelect hydra/sweeper=RepSelectMoE task_name=${prefix}_RepSelect2_forget trainer.handler=RepSelectMOE
else
    ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect2_forget
fi
