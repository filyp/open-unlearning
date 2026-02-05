#!/bin/bash

# note no --multirun
# we use more epochs, to make sure we hit the disruption budget
common="python3 src/unlearn_relearn.py \
--config-name=unlearn.yaml \
experiment=unlearn/wmdp_low_mi/default \
model=Llama-3.2-3B \
trainer=CIR \
eval.wmdp_low_mi.kl_evals.0.disr_budget=0.005 \
trainer.args.learning_rate=0.3 \
trainer.args.num_train_epochs=50"

# todo, just have another config, with cyber override
cyber="data.custom_loaders.0.dataset=cyber \
data.custom_loaders.1.hf_args.data_files=[computer_science_and_technology/computer_science_and_technology_000000.jsonl]"

ver=0.3_bio

# ver=0.3_cyber
# common="$common $cyber"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    echo "Running on SLURM"
    common="sbatch $HOME/open-unlearning/runners/slurm_runner.sh $common"
fi

$common task_name=${ver}_coverage_neg_ce trainer.method_args.cfg.forget_loss=neg_cross_entropy
$common task_name=${ver}_coverage_mlp_breaking trainer.method_args.cfg.forget_loss=mlp_breaking
$common task_name=${ver}_coverage_mlp_activation_breaking trainer.method_args.cfg.forget_loss=mlp_activation_breaking
$common task_name=${ver}_coverage_gate_and_up_breaking_approx trainer.method_args.cfg.forget_loss=gate_and_up_breaking_approx
