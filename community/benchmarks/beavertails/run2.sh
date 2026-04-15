#!/bin/bash

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

# model=gemma-4-E4B
model=Llama-3.1-8B
# model=Qwen3-30B-A3B-Base

category='animal_abuse'
# category='terrorism,organized_crime'

version=v7
# "no version" used the original beavertails dataset, where there is data duplication and mislabeling; also, it wasn't finished, got terminated before 50 trials
# v2 uses our curated high-quality subset
# v3 uses a 2x smaller LR during relearning, because it was too severe
# v4 tunes probability, not loss; also npo saturation was removed from repselect
# v5 evel smaller relearning LR
# v6 uses 1e-5 relearning LR, and non-contrastive set (curated.yaml)
# v7 uses 7e-6 relearning LR, and contrastive set (curated_contrast.yaml)

common="run python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated_contrast model=${model} category=${category}"
reference="run python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_contrast trainer.args.num_train_epochs=0 model=${model} category=${category}"
prefix="${version}_${model}_${category}"

# for running on verda:
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

# ${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect trainer.method_args.cfg.use_distribution=retain

# # RepSelect ablations
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_no_lora '~trainer.method_args.cfg.lora_lr' task_name=${prefix}_RepSelect_no_lora trainer.method_args.cfg.use_distribution=retain
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_no_pcs '~trainer.method_args.cfg.n_pcs' task_name=${prefix}_RepSelect_no_pcs trainer.method_args.cfg.use_distribution=retain


# # High disruption experiments
# common="$common eval.wikitext.disr_budget=0.1"
# ${common} trainer=RepSelect hydra/sweeper=RepSelect_highdisr task_name=${prefix}_RepSelect_highdisr trainer.method_args.cfg.use_distribution=retain
# ${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff2_highdisr
# ${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO_highdisr
# ${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU2_highdisr
# ${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO_highdisr
# ${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL2_highdisr