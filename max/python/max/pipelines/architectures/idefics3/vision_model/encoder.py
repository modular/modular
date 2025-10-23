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
"""Idefics3 vision encoder implementation."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import LayerNorm, Linear, Module

from ..model_config import Idefics3VisionConfig
from .attention import Idefics3VisionAttention


class Idefics3VisionMLP(Module):
    """Vision MLP for Idefics3 encoder layer.

    This implements the standard transformer feed-forward network with
    GELU activation, used in the Idefics3 vision encoder layers.

    Architecture:
    - Linear layer (hidden_size -> intermediate_size)
    - GELU activation (pytorch_tanh variant)
    - Linear layer (intermediate_size -> hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = True,
    ) -> None:
        """Initialize the Idefics3 vision MLP.

        Args:
            hidden_size: The input/output dimension (embed_dim).
            intermediate_size: The hidden dimension of the MLP.
            dtype: Data type for the weights.
            device: Device to place the weights on.
            has_bias: Whether to include bias terms in linear layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.fc1 = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

        self.fc2 = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass of the MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # First linear layer
        x = self.fc1(x)

        # GELU activation (pytorch_tanh variant)
        # This matches the "gelu_pytorch_tanh" activation in the config
        x = ops.gelu(x, approximate="tanh")

        # Second linear layer
        x = self.fc2(x)

        return x


class Idefics3VisionEncoderLayer(Module):
    """Single encoder layer for the Idefics3 vision transformer.

    Each encoder layer implements the standard transformer architecture:
    - Multi-head self-attention
    - Layer normalization (pre-norm architecture)
    - Feed-forward network (MLP) with GELU activation
    - Residual connections around both attention and MLP blocks

    This follows the SigLIP/SiglipEncoderLayer architecture pattern used by Idefics3.
    """

    def __init__(
        self,
        vision_config: Idefics3VisionConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize an Idefics3 vision encoder layer.

        Args:
            vision_config: Vision configuration object containing model parameters.
            dtype: Data type for the weights.
            device: Device to place the weights on.
        """
        super().__init__()
        self.embed_dim = vision_config.hidden_size

        # Self-attention layer
        self.self_attn = Idefics3VisionAttention(
            hidden_size=vision_config.hidden_size,
            num_attention_heads=vision_config.num_attention_heads,
            devices=[device],
            dtype=dtype,
        )

        # Layer normalization layers (pre-norm architecture)
        self.layer_norm1 = LayerNorm(
            dims=self.embed_dim,
            device=device,
            dtype=dtype,
            eps=vision_config.layer_norm_eps,
            use_bias=True,
        )

        self.layer_norm2 = LayerNorm(
            dims=self.embed_dim,
            device=device,
            dtype=dtype,
            eps=vision_config.layer_norm_eps,
            use_bias=True,
        )

        # MLP (feed-forward network)
        self.mlp = Idefics3VisionMLP(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=True,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
        output_attentions: bool = False,
    ) -> TensorValue:
        """Forward pass of the encoder layer.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional attention mask tensor.
            output_attentions: Whether to return attention weights (not currently supported).

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # Store input for first residual connection
        residual = hidden_states

        # Pre-norm: Apply layer norm before attention
        hidden_states = self.layer_norm1(hidden_states)

        # Self-attention
        hidden_states = self.self_attn(
            hidden_states,  # First positional argument becomes 'x'
            attention_mask=attention_mask,
        )

        # First residual connection
        hidden_states = residual + hidden_states

        # Store for second residual connection
        residual = hidden_states

        # Pre-norm: Apply layer norm before MLP
        hidden_states = self.layer_norm2(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Second residual connection
        hidden_states = residual + hidden_states

        return hidden_states


class Idefics3VisionEncoder(Module):
    """Idefics3 vision encoder consisting of multiple encoder layers.

    This implements the complete vision encoder stack with multiple
    Idefics3VisionEncoderLayer instances.
    """

    def __init__(
        self,
        vision_config: Idefics3VisionConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        """Initialize the Idefics3 vision encoder.

        Args:
            vision_config: Vision configuration object containing model parameters.
            dtype: Data type for the weights.
            device: Device to place the weights on.
        """
        super().__init__()
        self.num_hidden_layers = vision_config.num_hidden_layers

        # Create the encoder layers as individual attributes (not a list)
        # This allows MAX's load_state_dict to properly discover them
        for i in range(self.num_hidden_layers):
            setattr(
                self,
                f"layers.{i}",
                Idefics3VisionEncoderLayer(
                    vision_config=vision_config,
                    dtype=dtype,
                    device=device,
                ),
            )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Forward pass through all encoder layers.

        Args:
            hidden_states: Input embeddings of shape [batch_size, seq_len, hidden_size].
            attention_mask: Optional attention mask tensor.

        Returns:
            Final hidden states of shape [batch_size, seq_len, hidden_size].
        """
        # Pass through each encoder layer
        for i in range(self.num_hidden_layers):
            layer = getattr(self, f"layers.{i}")
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        return hidden_states
