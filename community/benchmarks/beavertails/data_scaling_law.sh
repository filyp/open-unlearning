# based on values from v7.3_Llama-3.1-8B_animal_abuse_RepSelectSimple_no_lora, run 22
# n_pcs=500 (default), lr=0.1

source .venv/bin/activate
run() {
  modal run runners/modal_runner.py --args "$*"
}

common="run python3 src/unlearn_relearn.py --config-name=unlearn.yaml experiment=unlearn/beavertails/curated_contrast trainer=RepSelectSimple trainer.method_args.use_lora=false trainer.args.num_train_epochs=30 data.anchor=forget"

# run on A100 80GB
model=Llama-3.1-8B
${common} model=${model} data.custom_loaders.0.range=[0,10] trainer.args.learning_rate=0.64 task_name=datascaling2_AA_${model}_range10
${common} model=${model} data.custom_loaders.0.range=[0,25] trainer.args.learning_rate=0.32 task_name=datascaling2_AA_${model}_range25
${common} model=${model} data.custom_loaders.0.range=[0,45] trainer.args.learning_rate=0.24 task_name=datascaling2_AA_${model}_range45
${common} model=${model} data.custom_loaders.0.range=[0,90] trainer.args.learning_rate=0.12 task_name=datascaling2_AA_${model}_range90
${common} model=${model} data.custom_loaders.0.range=[0,180] trainer.args.learning_rate=0.06 task_name=datascaling2_AA_${model}_range180
${common} model=${model} data.custom_loaders.0.range=[0,360] trainer.args.learning_rate=0.03 task_name=datascaling2_AA_${model}_range360

# run on H200
model=Qwen3.5-9B
${common} model=${model} data.custom_loaders.0.range=[0,10] trainer.args.learning_rate=0.64 task_name=datascaling2_AA_${model}_range10
${common} model=${model} data.custom_loaders.0.range=[0,25] trainer.args.learning_rate=0.32 task_name=datascaling2_AA_${model}_range25
${common} model=${model} data.custom_loaders.0.range=[0,45] trainer.args.learning_rate=0.16 task_name=datascaling2_AA_${model}_range45
${common} model=${model} data.custom_loaders.0.range=[0,90] trainer.args.learning_rate=0.08 task_name=datascaling2_AA_${model}_range90
${common} model=${model} data.custom_loaders.0.range=[0,180] trainer.args.learning_rate=0.04 task_name=datascaling2_AA_${model}_range180
${common} model=${model} data.custom_loaders.0.range=[0,360] trainer.args.learning_rate=0.02 task_name=datascaling2_AA_${model}_range360
