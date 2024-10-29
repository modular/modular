# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Llama testing utils.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_layer_norm(eps: float, x: torch.Tensor, weight):
    # Compute the mean and variance along the last dimension (features)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalize the input
    x_normalized = (x - mean) / torch.sqrt(variance + eps)

    after_weights = x_normalized * weight

    return after_weights


def torch_linear(weight, **kwargs):
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchVisionEncoderMLP(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.fc1 = torch_linear(w1, bias=False)
        self.fc2 = torch_linear(w2, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
