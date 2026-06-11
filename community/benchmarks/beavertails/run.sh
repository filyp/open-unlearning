#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/beavertails/curated"
reference="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated trainer.args.num_train_epochs=0"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch runners/slurm_runner.sh $common"
    reference="sbatch runners/slurm_runner.sh $reference"
fi

# model=gemma-3-270m
# model=gemma-3-4b-pt
model=gemma-2-2b

category='animal_abuse'
# category='terrorism,organized_crime'

version=v5
# "no version" used the original beavertails dataset, where there is data duplication and mislabeling; also, it wasn't finished, got terminated before 50 trials
# v2 uses our curated high-quality subset
# v3 uses a 2x smaller LR during relearning, because it was too severe
# v4 tunes probability, not loss; also npo saturation was removed from repselect
# v5 evel smaller relearning LR

###############################################################

common="${common} model=${model} category=${category}"
reference="${reference} model=${model} category=${category}"
prefix="${version}_${model}_${category}"

${reference} trainer=GradDiff task_name=${prefix}_reference

# Main experiments
${common} trainer=RepSelect hydra/sweeper=RepSelect task_name=${prefix}_RepSelect
${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff2
${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO
${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU2
${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO
${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL2

# RepSelect ablations (all use wide LR range for fair comparison)
${common} trainer=RepSelect hydra/sweeper=RepSelect_wide task_name=${prefix}_RepSelect_wide2
${common} trainer=RepSelect hydra/sweeper=RepSelect_no_lora '~trainer.method_args.cfg.lora_lr' task_name=${prefix}_RepSelect_no_lora2
${common} trainer=RepSelect hydra/sweeper=RepSelect_no_retain '~trainer.method_args.cfg.retain_momentum' task_name=${prefix}_RepSelect_no_retain2
${common} trainer=RepSelect hydra/sweeper=RepSelect_no_pcs '~trainer.method_args.cfg.n_pcs' task_name=${prefix}_RepSelect_no_pcs2

# High disruption experiments
common="$common eval.wikitext.disr_budget=0.1"
${common} trainer=RepSelect hydra/sweeper=RepSelect_highdisr task_name=${prefix}_RepSelect_highdisr
${common} trainer=GradDiff hydra/sweeper=GradDiff task_name=${prefix}_GradDiff2_highdisr
${common} trainer=NPO hydra/sweeper=NPO task_name=${prefix}_NPO_highdisr
${common} trainer=RMU hydra/sweeper=RMU task_name=${prefix}_RMU2_highdisr
${common} trainer=SimNPO hydra/sweeper=SimNPO task_name=${prefix}_SimNPO_highdisr
${common} trainer=UNDIAL hydra/sweeper=UNDIAL task_name=${prefix}_UNDIAL2_highdisr


${common} trainer=RepSelectCohen hydra/sweeper=RepSelect_wide task_name=${prefix}_RepSelectCohen
${common} trainer=RepSelectCohen hydra/sweeper=RepSelect_wide trainer.method_args.cfg.only_down=True task_name=${prefix}_RepSelectCohen_onlydown
${common} trainer=RepSelectCohen hydra/sweeper=RepSelect_no_retain '~trainer.method_args.cfg.retain_momentum' task_name=${prefix}_RepSelectCohen_no_retain
${common} trainer=RepSelectCohen hydra/sweeper=RepSelect_no_retain '~trainer.method_args.cfg.retain_momentum' trainer.method_args.cfg.only_down=True task_name=${prefix}_RepSelectCohen_no_retain_onlydown