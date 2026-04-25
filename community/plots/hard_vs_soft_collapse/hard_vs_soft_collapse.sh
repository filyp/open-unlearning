
# run on A100 80GB or RTX PRO 6000 96GB
source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

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

# # weight SVD experiment - not working well
# # commit 9cab616c658f576e032633a54153564bb7092dd1
# # model=Llama-3.1-8B
# model=Qwen3.5-9B
# # experiment=unlearn/wmdp_low_mi/default exp_name=bio
# experiment=unlearn/beavertails/curated_contrast exp_name=aa
# hard_soft=soft
# distribution=weight
# for n_pcs in 4 8 16 32 64 128 256 512 1024; do
#     run python src/unlearn_relearn.py --config-name=unlearn.yaml \
#       trainer=RepSelectSimple \
#       trainer.handler=RepSelectAdaptive \
#       trainer.method_args.use_lora=false \
#       experiment=${experiment} \
#       model=${model} \
#       trainer.method_args.n_pcs=${n_pcs} \
#       trainer.method_args.hard_soft=${hard_soft} \
#       trainer.method_args.distribution=${distribution} \
#       task_name=hardvssoft_${exp_name}_${model}_${n_pcs}_${hard_soft}_${distribution}
# done


# # keep PC ranges
# # commit 15aad2bc6fddf7a30b46b61bf3ce279eaf21ed65
# # model=Llama-3.1-8B
# model=Qwen3.5-9B
# # experiment=unlearn/wmdp_low_mi/default exp_name=bio
# experiment=unlearn/beavertails/curated_contrast exp_name=aa
# for range in "0,4" "4,16" "16,64" "64,256" "256,1024"; do
#     lo=${range%,*}
#     hi=${range#*,}
#     run python src/unlearn_relearn.py --config-name=unlearn.yaml \
#       trainer=RepSelectSimple \
#       trainer.handler=RepSelectAdaptive \
#       trainer.method_args.use_lora=false \
#       experiment=${experiment} \
#       model=${model} \
#       trainer.method_args.n_pcs="[${lo},${hi}]" \
#       task_name=hardvssoft_${exp_name}_${model}_${lo}-${hi}_ranges
#     run python src/unlearn_relearn.py --config-name=unlearn.yaml \
#       trainer=RepSelectSimple \
#       trainer.handler=RepSelectAdaptive \
#       trainer.method_args.use_lora=false \
#       experiment=${experiment} \
#       model=${model} \
#       trainer.method_args.n_pcs="[${lo},${hi}]" \
#       trainer.method_args.distribution=retain \
#       task_name=hardvssoft_${exp_name}_${model}_${lo}-${hi}_ranges_retain
# done