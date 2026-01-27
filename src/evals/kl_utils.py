import copy

import torch as pt
import torch.nn.functional as F

from trainer.unlearn.cir.cir_utils import get_lm, prep_batch


def cache_last_hidden_states(model, batches):
    """Cache last hidden states for each batch to compute KL divergence later.
    
    Note: If GPU memory is tight, you may want to move them to RAM.
    """
    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device), output_hidden_states=True)
            batch["cached_last_hidden"] = output.hidden_states[-1].detach()


def get_loss_and_kl(model, batches, acts_to_logits):
    """Compute loss and KL divergence in a single pass.

    KL(P || Q) where P is the original model (cached) and Q is the current model.
    KL is averaged across all tokens (where labels != -100).
    """
    total_kl = 0.0
    total_tokens = 0
    loss_acc = 0.0

    for batch in batches:
        with pt.no_grad():
            output = model(**prep_batch(batch, model.device))
            loss_acc += output.loss.item()

            current_logits = output.logits.float()

            # Reconstruct original logits from cached hidden states
            cached_hidden = batch["cached_last_hidden"].to(model.dtype)
            cached_logits = acts_to_logits(cached_hidden).float()
            assert current_logits.shape == cached_logits.shape  # (batch, seq, vocab)

            # Get mask for valid tokens (labels != -100)
            labels = batch["labels"].to(model.device)
            token_mask = labels != -100
            assert token_mask.shape == current_logits.shape[:2]

            # Mask first
            cached_logits_masked = cached_logits[token_mask]  # btw, bfloat is enough
            current_logits_masked = current_logits[token_mask]
            assert cached_logits_masked.shape == current_logits_masked.shape
            assert cached_logits_masked.ndim == 2  # (n_valid_tokens, vocab)

            # # Filter out rows with NaN values - used for debugging gemma3 (currently broken)
            # valid_rows = ~(pt.isnan(cached_logits_masked).any(dim=-1))
            # # valid_rows2 = ~(pt.isnan(current_logits_masked).any(dim=-1))
            # # assert they are the same
            # # assert valid_rows == valid_rows2
            # cached_logits_masked = cached_logits_masked[valid_rows]
            # current_logits_masked = current_logits_masked[valid_rows]

            # KL(P || Q) using kl_div (expects log_q as input, p as target)
            log_p = pt.nn.functional.log_softmax(cached_logits_masked, dim=-1)
            log_q = pt.nn.functional.log_softmax(current_logits_masked, dim=-1)

            total_kl += pt.nn.functional.kl_div(
                log_q, log_p, reduction="sum", log_target=True
            ).item()
            total_tokens += token_mask.sum().item()

    avg_loss = loss_acc / len(batches)
    avg_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_kl


def create_acts_to_logits(model):
    """Create a function to convert hidden states to logits.

    Handles models with lm_head and models with shared embeddings (e.g., MobileLLM).
    Copies the weights so they're preserved even if the model changes during training.
    
    Note that some models (e.g. gemma2) have custom ways of converting hidden states to logits.
    To make sure that it works correctly, your evaluator should assert that on the evaluation
    before the training starts, the KL divergence is 0 (see WMDPLLowMIEvaluator).
    """
    model = get_lm(model)
    model_type = model.config.model_type

    if not hasattr(model, "lm_head"):
        # MobileLLM-style: use shared embedding weights
        cached_embed_weight = model.model.embed_tokens.weight.detach().clone()
        return lambda h: F.linear(h, cached_embed_weight)

    cached_lm_head = copy.deepcopy(model.lm_head)

    def _acts_to_logits(hidden_states):
        logits = cached_lm_head(hidden_states)

        if model_type == "gemma2" and model.config.final_logit_softcapping is not None:
            # based on: https://github.com/huggingface/transformers/blob/ab87f2445096554e1c28ffe896afd96fa9469444/src/transformers/models/gemma2/modeling_gemma2.py#L539-L542
            logits = logits / model.config.final_logit_softcapping
            logits = pt.tanh(logits)
            logits = logits * model.config.final_logit_softcapping

        return logits

    return _acts_to_logits
