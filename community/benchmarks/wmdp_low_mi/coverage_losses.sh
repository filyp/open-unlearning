#!/bin/bash

common="python3 src/unlearn_relearn.py \
--multirun \
--config-name=unlearn.yaml \
experiment=unlearn/wmdp_low_mi/default \
model=Llama-3.2-3B \
trainer=CIR \
eval.wmdp_low_mi.disr_budget=0.005 \
trainer.args.learning_rate=0.2 \
trainer.args.num_train_epochs=50"

# Auto-detect if we're on SLURM
if command -v sbatch &> /dev/null; then
    common="sbatch $HOME/open-unlearning/runners/slurm.sh $common"
fi

$common task_name=coverage_neg_ce trainer.method_args.cfg.forget_loss=neg_cross_entropy
$common task_name=coverage_mlp_breaking trainer.method_args.cfg.forget_loss=mlp_breaking
$common task_name=coverage_mlp_activation_breaking trainer.method_args.cfg.forget_loss=mlp_activation_breaking
$common task_name=coverage_gate_and_up_breaking_approx trainer.method_args.cfg.forget_loss=gate_and_up_breaking_approx