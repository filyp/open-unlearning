
source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}
# run() {
#   modal run runners/modal_runner.py --args "$*"
# }

model=Llama-3.1-8B
# model=gemma-4-E4B
# model=DeepSeek-V2-Lite
# model=Qwen3.5-9B

experiment=unlearn/wmdp_low_mi/default exp_name=bio
# experiment=unlearn/beavertails/curated_contrast exp_name=aa

# hard_soft=soft
hard_soft=quadratic
# hard_soft=ridge

# for ridge, n_pcs only moves the damping shoulder (the SVD is full-rank
# anyway, so this costs nothing extra); variant is the task-name suffix
# n_pcs=1024
# n_pcs=3072
# n_pcs=256
n_pcs=512

for dist in forget retain forget_and_retain; do
# for dist in forget_and_retain; do
  # for collapse in act grad both none; do
  for collapse in act grad both; do
    run python src/unlearn_relearn.py --config-name=unlearn.yaml \
      trainer=RepSelectSimple \
      trainer.handler=RepSelectAdaptive \
      trainer.method_args.use_lora=false \
      relearning_trainer.args.num_train_epochs=10 \
      trainer.method_args.hard_soft=${hard_soft} \
      trainer.method_args.n_pcs=${n_pcs} \
      experiment=${experiment} \
      model=${model} \
      trainer.method_args.distribution=${dist} \
      trainer.method_args.collapse_on=${collapse} \
      task_name=collapse2_${exp_name}_${model}_${dist}_${collapse}_${hard_soft}
      # task_name=collapse2_${exp_name}_${model}_${dist}_${collapse}_${hard_soft}${n_pcs}
  done
done

# note that these experiments were run with n_pcs=500, not current 512

# # retain act + forget grad experiment; note that it uses outdated args from commit 4911cbb7dcdde3adaea4b4586cb047bacffd37ce
# run python src/unlearn_relearn.py --config-name=unlearn.yaml \
#   trainer=RepSelectSimple \
#   trainer.handler=RepSelectAdaptive \
#   trainer.method_args.use_lora=false \
#   experiment=${experiment} \
#   model=${model} \
#   trainer.method_args.act_collapse=retain \
#   trainer.method_args.grad_collapse=forget \
#   task_name=collapse_${exp_name}_${model}_actretain_gradforget