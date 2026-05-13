"""Regression check for UnlearnTrainer.prediction_step.

Run from repo root:
    python tests/prediction_step_regression.py

For now this just prints the outputs so we can capture a baseline.
Once we have a known-good output, asserts can be added below.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trainer.unlearn.base import UnlearnTrainer  # noqa: E402


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SEED = 0


def main():
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.eval()

    args = TrainingArguments(
        output_dir="/tmp/prediction_step_regression",
        per_device_eval_batch_size=2,
        report_to=[],
    )

    trainer = UnlearnTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
    )

    text = "The capital of France is Paris."
    enc = tokenizer([text, text], return_tensors="pt", padding=True)
    inputs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
    }

    loss, logits, labels = trainer.prediction_step(
        model, inputs, prediction_loss_only=False
    )

    print("=== prediction_step output ===")
    print(f"loss: {loss}")
    print(f"logits shape: {tuple(logits.shape) if logits is not None else None}")
    if logits is not None:
        print(f"logits[0, 0, :8]: {logits[0, 0, :8].tolist()}")
        print(f"logits sum: {logits.sum().item():.6f}")
    print(f"labels shape: {tuple(labels.shape) if labels is not None else None}")

    # Baseline captured on upstream (transformers pre-5.x).
    expected_loss = 2.7516109943389893
    expected_logits_shape = (2, 8, 128256)
    expected_logits_head = [
        2.8333027362823486,
        3.5809550285339355,
        7.026833534240723,
        3.282773494720459,
        2.5123724937438965,
        1.5481433868408203,
        2.607846736907959,
        3.727475881576538,
    ]
    expected_labels_shape = (2, 8)

    assert abs(loss.item() - expected_loss) < 1e-3, (loss.item(), expected_loss)
    assert tuple(logits.shape) == expected_logits_shape
    assert tuple(labels.shape) == expected_labels_shape
    head = logits[0, 0, :8].tolist()
    for got, exp in zip(head, expected_logits_head):
        assert abs(got - exp) < 1e-3, (got, exp)
    print("OK")


if __name__ == "__main__":
    main()
