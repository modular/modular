# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Llama testing utils."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_linear(weight: torch.Tensor, **kwargs: Any) -> nn.Linear:
    linear = nn.Linear(*weight.shape, **kwargs)
    linear.weight = nn.Parameter(weight)
    return linear


class TorchVisionEncoderMLP(nn.Module):
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        super().__init__()
        self.fc1 = torch_linear(w1, bias=False)
        self.fc2 = torch_linear(w2, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TorchVisionAttention(nn.Module):
    def __init__(
        self, attn_weight: torch.Tensor, hidden_size: int, attention_heads: int
    ) -> None:
        super().__init__()

        self.embed_dim = hidden_size
        self.num_heads = attention_heads
        self.head_dim = hidden_size // attention_heads

        self.q_proj = torch_linear(attn_weight, bias=False)
        self.k_proj = torch_linear(attn_weight, bias=False)
        self.v_proj = torch_linear(attn_weight, bias=False)
        self.o_proj = torch_linear(attn_weight, bias=False)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        batch_size, q_seq_len, _ = query.shape
        _, kv_seq_len, _ = key.shape

        query = query.view(
            batch_size, q_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, kv_seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_seq_len, -1)

        output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return output, attn_weights
