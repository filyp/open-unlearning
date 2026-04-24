
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

for dist in forget retain; do
  for collapse in act grad both none; do
    run python src/unlearn_relearn.py --config-name=unlearn.yaml \
      trainer=RepSelectSimple \
      trainer.handler=RepSelectAdaptive \
      trainer.method_args.use_lora=false \
      experiment=${experiment} \
      model=${model} \
      trainer.method_args.distribution=${dist} \
      trainer.method_args.collapse_on=${collapse} \
      task_name=collapse_${exp_name}_${model}_${dist}_${collapse}
  done
done

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