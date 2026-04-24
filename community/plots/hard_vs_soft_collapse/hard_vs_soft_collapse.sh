
source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

# run on A100 80GB or RTX PRO 6000 96GB
model=Llama-3.1-8B
# model=Qwen3.5-9B

experiment=unlearn/wmdp_low_mi/default exp_name=bio
# experiment=unlearn/beavertails/curated_contrast exp_name=aa

for hard_soft in soft hard; do
    for n_pcs in 4 8 16 32 64 128 256 512 1024; do

        run python src/unlearn_relearn.py --config-name=unlearn.yaml \
          trainer=RepSelectSimple \
          trainer.handler=RepSelectAdaptive \
          trainer.method_args.use_lora=false \
          experiment=${experiment} \
          model=${model} \
          trainer.method_args.n_pcs=${n_pcs} \
          trainer.method_args.hard_soft=${hard_soft} \
          task_name=hardvssoft_${exp_name}_${model}_${n_pcs}_${hard_soft}
        echo $?

        run python src/unlearn_relearn.py --config-name=unlearn.yaml \
          trainer=RepSelectSimple \
          trainer.handler=RepSelectAdaptive \
          trainer.method_args.use_lora=false \
          experiment=${experiment} \
          model=${model} \
          trainer.method_args.n_pcs=${n_pcs} \
          trainer.method_args.hard_soft=${hard_soft} \
          trainer.method_args.distribution=retain \
          task_name=hardvssoft_${exp_name}_${model}_${n_pcs}_${hard_soft}_retain
        echo $?
    done
done