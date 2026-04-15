"""Extract the text-only model from a Gemma 4 multimodal checkpoint.

Usage:
    python scripts/strip_multimodal.py google/gemma-4-E4B saves/models/gemma-4-E4B-text-only
    python scripts/strip_multimodal.py google/gemma-4-31B saves/models/gemma-4-31B-text-only

Loads only the safetensors state dict (no model instantiation), remaps the
language-model keys, drops vision/audio weights, writes a Gemma4ForCausalLM-
compatible checkpoint. Peak RAM ≈ 1x model size.
"""

import argparse
import json
import os

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer


PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
EMBED_KEY = "model.embed_tokens.weight"


def remap_state_dict(state_dict):
    """Keep only language model + lm_head weights, strip the prefix."""
    new_sd = {}
    for key, tensor in state_dict.items():
        if key == LM_HEAD_KEY:
            new_sd[key] = tensor
        elif key.startswith(PREFIX):
            new_key = "model." + key[len(PREFIX):]
            new_sd[new_key] = tensor
    return new_sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="HuggingFace model ID or local path")
    parser.add_argument("destination", help="Local path to save text-only checkpoint")
    args = parser.parse_args()

    os.makedirs(args.destination, exist_ok=True)

    # Resolve config
    if os.path.isdir(args.source):
        config_path = os.path.join(args.source, "config.json")
    else:
        config_path = hf_hub_download(args.source, "config.json")

    with open(config_path) as f:
        full_config = json.load(f)

    text_config = full_config["text_config"]
    text_config["architectures"] = ["Gemma4ForCausalLM"]
    text_config["model_type"] = "gemma4_text"
    # Preserve transformers_version if present
    if "transformers_version" in full_config:
        text_config["transformers_version"] = full_config["transformers_version"]

    with open(os.path.join(args.destination, "config.json"), "w") as f:
        json.dump(text_config, f, indent=2)
    print("Wrote config.json")

    # Load and remap weights (no model instantiation — just tensors in RAM)
    if os.path.isdir(args.source):
        safetensor_path = os.path.join(args.source, "model.safetensors")
    else:
        print(f"Downloading safetensors from {args.source} ...")
        safetensor_path = hf_hub_download(args.source, "model.safetensors")

    print("Loading state dict ...")
    state_dict = load_file(safetensor_path)

    print(f"Loaded {len(state_dict)} keys, remapping ...")
    new_sd = remap_state_dict(state_dict)
    del state_dict  # free the full state dict
    print(f"Kept {len(new_sd)} keys")

    out_path = os.path.join(args.destination, "model.safetensors")
    print(f"Saving to {out_path} ...")
    save_file(new_sd, out_path)
    del new_sd

    # Copy tokenizer
    print("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    tokenizer.save_pretrained(args.destination)

    print(f"\nDone. Load with:")
    print(f"  AutoModelForCausalLM.from_pretrained('{args.destination}')")


if __name__ == "__main__":
    main()
