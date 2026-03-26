from transformers.integrations.moe import grouped_mm_experts_forward, ALL_EXPERTS_FUNCTIONS, _grouped_linear
import torch

# original_grouped = grouped_mm_experts_forward

class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def hooked_grouped_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    # Reshape for easier indexing
    # S is the number of selected tokens-experts pairs (S = num_tokens * num_top_k)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_top_k).reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)
    expert_ids = top_k_index.reshape(-1)  # (S,)

    # Get current hidden states for selected samples
    selected_hidden_states = hidden_states[token_idx]

    # Sort by expert for grouped processing
    perm = torch.argsort(expert_ids)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)

    expert_ids_g = expert_ids[perm]
    sample_weights_g = sample_weights[perm]
    selected_hidden_states_g = selected_hidden_states[perm]

    # Compute offsets for grouped_mm
    # using histc instead of bincount to avoid cuda graph issues
    # With deterministic algorithms, CPU only supports float input, CUDA only supports int input.
    histc_input = expert_ids_g.float() if device.type == "cpu" else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=self.num_experts, min=0, max=self.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # Select expert weights and biases
    # NOTE: We keep all experts here and rely on offsets to target the active ones.
    # I have already implemented a version that only passes the active experts, but
    # to do so I had to use torch.unique which breaks the graph capture (data-dependent).
    # Also there were no speedup gains from it in my experiments, even in eager mode.
    if self.has_gate:
        selected_weights = self.gate_up_proj
        selected_biases = self.gate_up_proj_bias[expert_ids_g] if self.has_bias else None
    else:
        selected_weights = self.up_proj
        selected_biases = self.up_proj_bias[expert_ids_g] if self.has_bias else None

    # --- Up projection per expert (grouped) ---
    self.pre_gate_up = selected_hidden_states_g  # ! ADDED !!!!!!!!!!!!!!!!!!!!!!!!!!
    proj_out = _grouped_linear(
        selected_hidden_states_g, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  # (S, 2 * intermediate_dim) or  (S, intermediate_dim) depending on whether we have gating
    proj_out = self.gate_up_output_probe(proj_out)  # ! ADDED !!!!!!!!!!!!!!!!!!!!!!!!!!

    # Apply gating or activation
    if self.has_gate:
        # for gated experts we apply the custom/default gating mechanism
        proj_out = self._apply_gate(proj_out)  # (S, intermediate_dim)
    else:
        # for non-gated experts we just apply the activation function
        proj_out = self.act_fn(proj_out)  # (S, intermediate_dim)

    # Select down projection weights and biases
    selected_weights = self.down_proj
    selected_biases = self.down_proj_bias[expert_ids_g] if self.has_bias else None

    # --- Down projection per expert (grouped) ---
    self.pre_down = proj_out  # ! ADDED !!!!!!!!!!!!!!!!!!!!!!!!!!
    proj_out = _grouped_linear(
        proj_out, selected_weights, offsets, bias=selected_biases, is_transposed=self.is_transposed
    )  # (S, hidden_dim)
    proj_out = self.down_output_probe(proj_out)  # ! ADDED !!!!!!!!!!!!!!!!!!!!!!!!!!

    # Apply routing weights
    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)  # (S, hidden_dim)

    # Restore original order
    weighted_out = weighted_out[inv_perm]  # (S, hidden_dim)

    # Accumulate results using deterministic reshape+sum instead of index_add_
    # index_add_ with duplicate indices is non-deterministic on CUDA due to atomicAdd
    # index_add_ accumulates in-place using the dtype of the output tensor (fp16/bf16)
    # reshape+sum accumulates in fp32 which is more stable for low precision training/inference.
    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)

    return final_hidden_states.to(hidden_states.dtype)




# ALL_EXPERTS_FUNCTIONS._global_mapping["grouped_mm"] = hooked_grouped_mm_experts_forward