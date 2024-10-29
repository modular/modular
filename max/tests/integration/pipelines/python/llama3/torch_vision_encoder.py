# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
This vision encoder class serves as a Torch reference implementation for
our testing purposes.

These interfaces are largely copied from the transformers package:
transformers.models.mllama.modeling_mllama.py as of version 4.45.2.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch_utils import torch_layer_norm, TorchVisionEncoderMLP


class MllamaVisionEncoderLayer(nn.Module):
    def __init__(
        self,
        eps: float,
        mlp_fc1: torch.Tensor,
        mlp_fc2: torch.Tensor,
        encoder_layernorm_w1: torch.Tensor,
        encoder_layernorm_w2: torch.Tensor,
        is_gated: bool = False,
    ):
        super().__init__()

        self.eps = eps
        self.mlp = TorchVisionEncoderMLP(mlp_fc1, mlp_fc2)

        self.encoder_layernorm_w1 = encoder_layernorm_w1
        self.encoder_layernorm_w2 = encoder_layernorm_w2
        self.is_gated = is_gated

        if is_gated:
            self.gate_attn = nn.Parameter(torch.ones(1) * math.pi / 4)
            self.gate_ffn = nn.Parameter(torch.ones(1) * math.pi / 4)

    def forward(
        self,
        hidden_state: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_state

        hidden_state = torch_layer_norm(
            self.eps, hidden_state, self.encoder_layernorm_w1
        )

        # TODO: Marking this as a no-op for now.
        # hidden_state, attn_weights = self.self_attn(
        #     hidden_state, attention_mask=attention_mask
        # )
        if self.is_gated:
            hidden_state = self.gate_attn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        residual = hidden_state
        hidden_state = torch_layer_norm(
            self.eps, hidden_state, self.encoder_layernorm_w2
        )
        hidden_state = self.mlp(hidden_state)
        if self.is_gated:
            hidden_state = self.gate_ffn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)

        return outputs


class MllamaVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MllamaEncoderLayer`].
    """

    def __init__(
        self,
        eps: float,
        mlp_fc1,
        mlp_fc2,
        encoder_layernorm_w1,
        encoder_layernorm_w2,
        num_layers=32,
        is_gated=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MllamaVisionEncoderLayer(
                    eps=eps,
                    mlp_fc1=mlp_fc1,
                    mlp_fc2=mlp_fc2,
                    encoder_layernorm_w1=encoder_layernorm_w1,
                    encoder_layernorm_w2=encoder_layernorm_w2,
                    is_gated=is_gated,
                )
                for _ in range(num_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        # return_dict: bool | None = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_state=hidden_states,
                # attention_mask=attention_mask,
            )

            hidden_states = layer_outputs[0]

        # Always return like that for now.
        return hidden_states
