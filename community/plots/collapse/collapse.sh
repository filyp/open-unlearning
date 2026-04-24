
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

experiment=unlearn/wmdp_low_mi/default
run python src/unlearn_relearn.py --config-name=unlearn.yaml \
  trainer=RepSelectSimple \
  trainer.handler=RepSelectAdaptive \
  trainer.method_args.use_lora=false \
  experiment=${experiment} \
  model=${model} \
  task_name="collapse_bio_${model}_\${trainer.method_args.distribution}_\${trainer.method_args.collapse_on}" \
  hydra/sweeper=basic \
  --multirun \
  trainer.method_args.distribution=forget,retain \
  trainer.method_args.collapse_on=act,grad,both,none

experiment=unlearn/beavertails/curated_contrast
run python src/unlearn_relearn.py --config-name=unlearn.yaml \
  trainer=RepSelectSimple \
  trainer.handler=RepSelectAdaptive \
  trainer.method_args.use_lora=false \
  experiment=${experiment} \
  model=${model} \
  task_name="collapse_AA_${model}_\${trainer.method_args.distribution}_\${trainer.method_args.collapse_on}" \
  hydra/sweeper=basic \
  --multirun \
  trainer.method_args.distribution=forget,retain \
  trainer.method_args.collapse_on=act,grad,both,none
