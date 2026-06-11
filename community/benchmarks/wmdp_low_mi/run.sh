#!/bin/bash

# export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# echo "Master Port: $MASTER_PORT"

# note, experiments were done with adamw_8bit as the default optimizer in finetune.yaml

common="python src/unlearn_relearn.py --config-name=unlearn.yaml --multirun experiment=unlearn/wmdp_low_mi/default"
reference="python src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/wmdp_low_mi/default trainer.args.num_train_epochs=0"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch runners/slurm_runner.sh $common"
    reference="sbatch runners/slurm_runner.sh $reference"
fi

# model=Llama-3.2-3B
model=Qwen2.5-3B

wmdp_domain='bio'
# wmdp_domain='cyber'

version=v1

###############################################################

common="${common} model=${model} wmdp_domain=${wmdp_domain}"
reference="${reference} model=${model} wmdp_domain=${wmdp_domain}"
prefix="${version}_${model}_${wmdp_domain}"

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