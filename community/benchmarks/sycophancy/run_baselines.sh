
# Baseline reference runs: no unlearning (0 unlearning epochs), just the relearning
# attack on the base model.
# Once they finish, run populate_baselines.py to write their values into baselines.yaml.
# (num_train_epochs=0 rather than do_train=false, since the trainer only moves the
# model to GPU inside train(), and the evals need it there.)
source .venv/bin/activate
run() {
  bash verda_runner.sh $*
}

for model in Llama-3.1-8B gemma-4-E4B DeepSeek-V2-Lite Qwen3.5-9B; do
  run python src/unlearn_relearn.py --config-name=unlearn.yaml \
    experiment=unlearn/sycophancy/default \
    model=${model} \
    trainer=finetune \
    trainer.args.num_train_epochs=0 \
    task_name=baseline_sycophancy_${model}
done
