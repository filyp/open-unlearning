#!/bin/bash

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

model=gemma-4-E4B
# model=Llama-3.1-8B
# model=Qwen3-30B-A3B-Base

wmdp_domain='bio'
# wmdp_domain='cyber'

version=v3

common="run python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default model=${model} wmdp_domain=${wmdp_domain}"
reference="run python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0 model=${model} wmdp_domain=${wmdp_domain}"
prefix="${version}_${model}_${wmdp_domain}"

# # for running on verda:
run() {
  bash verda_runner.sh $*
}

# # for running on modal:
# source .venv/bin/activate
# run() {
#   modal run runners/modal_runner.py --args "$*"
# }


###############################################################

${reference} trainer=GradDiff task_name=${prefix}_reference

# Main experiments
${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff
${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO
${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU
${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO
${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL

# ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect

# # RepSelect ablations - todo, adapt the sweeper configs
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_no_lora '~trainer.method_args.cfg.lora_lr' task_name=${prefix}_RepSelect_no_lora
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_no_pcs '~trainer.method_args.cfg.n_pcs' task_name=${prefix}_RepSelect_no_pcs



# # High disruption experiments
# common="$common eval.wikitext.disr_budget=0.1"
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_highdisr task_name=${prefix}_RepSelect_highdisr
# ${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff2_highdisr
# ${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO_highdisr
# ${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU2_highdisr
# ${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO_highdisr
# ${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL2_highdisr


# for gemma: RTX 6000 Ada (48GB)
# but NPO,undial OOMs on it
# ok: graddiff, simnpo, rmu
# so actually, for gemma use A100 (80GB)