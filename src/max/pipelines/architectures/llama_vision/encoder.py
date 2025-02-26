# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Llama 3.2 Transformer Vision Encoder."""

from __future__ import annotations

from dataclasses import dataclass

from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.nn import LayerNorm
from max.pipelines.nn.layer import Layer

from .attention import Attention
from .mlp import MLP


@dataclass
class VisionEncoderLayer(Layer):
    """
    This class implements a layer within Llama 3.2 vision transformer encoder.
    """

    mlp: MLP
    input_layernorm: LayerNorm
    post_attention_layernorm: LayerNorm
    self_attn: Attention
    is_gated: bool = False
    gate_attn: TensorValueLike | None = None
    gate_ffn: TensorValueLike | None = None

    def __call__(
        self, hidden_state: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        # Self Attention.
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)

        hidden_state = self.self_attn(
            x=hidden_state,
            attention_mask=attention_mask,
        )

        if self.is_gated:
            hidden_state = ops.tanh(self.gate_attn) * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward.
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)

        # Gating criteria.
        if self.is_gated:
            hidden_state = ops.tanh(self.gate_ffn) * hidden_state
        return residual + hidden_state


@dataclass
class VisionEncoder(Layer):
    """
    Transformer encoder consisting of # of hidden self attention
    layers. Each layer is a [`VisionEncoderLayer`].
    """

    layers: list[VisionEncoderLayer]

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        output_hidden_states: bool,
    ) -> tuple[TensorValue, tuple[TensorValue] | None]:
        r"""
        Args:
            hidden_states (Tensor of shape `(batch_size, sequence_length, hidden_size)`):
            attention_mask (Tensor of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        encoder_states: tuple | None = () if output_hidden_states else None
        for encoder_layer in self.layers:
            if encoder_states is not None:
                encoder_states = encoder_states + (hidden_states,)
            hidden_states = encoder_layer(hidden_states, attention_mask)

        if encoder_states is not None:
            encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states
