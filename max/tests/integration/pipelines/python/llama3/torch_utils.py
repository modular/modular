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
