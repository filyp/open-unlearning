#!/bin/bash

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

model="Llama-3.2-1B-Instruct"

forget_split=forget10
retain_split=retain90

model_path=open-unlearning/tofu_${model}_full
echo ${task_name}: Unlearning ${model_path} using ${trainer}

# Unlearn
CUDA_VISIBLE_DEVICES=0 \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default.yaml \
trainer=GradDiff \
task_name=tofu_${model}_${forget_split}_${trainer}_lr${lr} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
trainer.args.learning_rate=1e-5 \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.eval_strategy=no \
trainer.args.eval_on_start=false



# # Eval
# CUDA_VISIBLE_DEVICES=0 python src/eval.py \
# experiment=eval/tofu/default.yaml \
# forget_split=${forget_split} \
# model=${model} \
# task_name=${task_name} \
# model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
# paths.output_dir=saves/unlearn/${task_name}/evals \
# retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json
