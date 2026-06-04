import torch as pt

from data.custom_loaders import DATE_STRING


def get_banned_tokens(processing_class):
    empty_chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]
    return set(
        processing_class.apply_chat_template(empty_chat, date_string=DATE_STRING)
    )


class ManualLoRA(pt.nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = pt.nn.Parameter(pt.empty(in_features, rank))
        self.lora_B = pt.nn.Parameter(pt.zeros(rank, out_features))
        pt.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.requires_grad_(True)

    def forward(self, x):
        return x @ self.lora_A @ self.lora_B * self.scaling


class BatchedManualLoRA(pt.nn.Module):
    def __init__(self, num_experts, in_features, out_features, rank, alpha=16):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = pt.nn.Parameter(pt.empty(num_experts, in_features, rank))
        self.lora_B = pt.nn.Parameter(pt.zeros(num_experts, rank, out_features))
        pt.nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.requires_grad_(True)

    def forward(self, x, offsets):
        # x: (S, in_features), offsets: (E,)
        mid = pt._grouped_mm(x, self.lora_A, offs=offsets)  # (S, rank)
        out = pt._grouped_mm(mid, self.lora_B, offs=offsets)  # (S, out_features)
        return out * self.scaling
