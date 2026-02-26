# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

import math
from typing import Any

from max.driver.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.module_v3 import Linear, Module
from max.nn.module_v3.norm import RMSNorm
from max.nn.module_v3.sequential import ModuleList

from .activations import GELU
from .embeddings import apply_rotary_emb

flash_attention_gpu = F.functional(_flash_attention_gpu)


class FluxAttention(Module[..., Tensor | tuple[Tensor, Tensor]]):
    """Flux attention mechanism with QK normalization and optional dual stream."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int | None = None,
        context_pre_only: bool | None = None,
        pre_only: bool = False,
    ):
        """Initialize Flux attention module.

        Args:
            query_dim: Dimension of query vectors.
            heads: Number of attention heads.
            dim_head: Dimension of each attention head.
            dropout: Dropout probability.
            bias: Whether to use bias in projections.
            added_kv_proj_dim: Optional dimension for additional key/value projections.
            added_proj_bias: Whether to use bias in additional projections.
            out_bias: Whether to use bias in output projection.
            eps: Epsilon for normalization layers.
            out_dim: Optional output dimension.
            context_pre_only: Whether to use context pre-processing only.
            pre_only: Whether to use pre-processing only.

        """
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = Linear(
            query_dim,
            self.inner_dim,
            bias=bias,
        )
        self.to_k = Linear(
            query_dim,
            self.inner_dim,
            bias=bias,
        )
        self.to_v = Linear(
            query_dim,
            self.inner_dim,
            bias=bias,
        )

        if not self.pre_only:
            self.to_out = Linear(
                self.inner_dim,
                self.out_dim,
                bias=out_bias,
            )

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm(dim_head, eps=eps)  # elementwise_affine
            self.norm_added_k = RMSNorm(dim_head, eps=eps)  # elementwise_affine
            self.add_q_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
            )
            self.add_k_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim,
                self.inner_dim,
                bias=added_proj_bias,
            )
            self.to_add_out = Linear(
                self.inner_dim,
                query_dim,
                bias=out_bias,
            )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Apply Flux attention to hidden states.

        Args:
            hidden_states: Input hidden states.
            encoder_hidden_states: Optional encoder hidden states for cross-attention.
            image_rotary_emb: Optional rotary embeddings for position encoding.

        Returns:
            Output hidden states after attention, or tuple of (hidden_states, encoder_hidden_states)
            if encoder states provided.
        """
        batch_size = hidden_states.shape[0]

        # get qkv projections
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        seq_len = query.shape[1]
        query = F.reshape(
            query, (batch_size, seq_len, self.heads, self.head_dim)
        )
        key = F.reshape(key, (batch_size, seq_len, self.heads, self.head_dim))
        value = F.reshape(
            value, (batch_size, seq_len, self.heads, self.head_dim)
        )

        query = self.norm_q(query)
        key = self.norm_k(key)

        encoder_query = encoder_key = encoder_value = None
        if (
            encoder_hidden_states is not None
            and self.added_kv_proj_dim is not None
        ):
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_seq_len = encoder_query.shape[1]
            encoder_query = F.reshape(
                encoder_query,
                (batch_size, encoder_seq_len, self.heads, self.head_dim),
            )
            encoder_key = F.reshape(
                encoder_key,
                (batch_size, encoder_seq_len, self.heads, self.head_dim),
            )
            encoder_value = F.reshape(
                encoder_value,
                (batch_size, encoder_seq_len, self.heads, self.head_dim),
            )

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = F.concat([encoder_query, query], axis=1)
            key = F.concat([encoder_key, key], axis=1)
            value = F.concat([encoder_value, value], axis=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # TODO: Support arbitrary attention mask
        hidden_states = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=math.sqrt(1.0 / self.head_dim),
        )

        total_seq_len = hidden_states.shape[1]
        hidden_states = F.reshape(
            hidden_states,
            (batch_size, total_seq_len, self.heads * self.head_dim),
        )

        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :encoder_seq_len, :]
            hidden_states = hidden_states[:, encoder_seq_len:, :]

            hidden_states = self.to_out(hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        return hidden_states


class FeedForward(Module[[Tensor], Tensor]):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim: int | None = None,
        bias: bool = True,
    ):
        """Initialize FeedForward module.

        Args:
            dim: Input dimension.
            dim_out: Optional output dimension. Defaults to dim if None.
            mult: Multiplier for hidden dimension.
            activation_fn: Activation function to use ("gelu" or "gelu-approximate").
            inner_dim: Optional inner dimension. Computed as dim * mult if None.
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
            )
        else:
            raise NotImplementedError(
                f"Activation function {activation_fn} is not implemented"
            )

        self.net: ModuleList[Module[..., Any]] = ModuleList(
            [
                act_fn,
                Linear(
                    inner_dim,
                    dim_out,
                    bias=bias,
                ),
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply feedforward network to hidden states.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Output hidden states after feedforward network.
        """
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class FluxPosEmbed(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Flux Position Embedding module for 3D rotary position embeddings.

    This module computes separate rotary embeddings for each spatial dimension
    (typically time, height, width) and concatenates them.

    Args:
        theta: Base value for frequency computation (typically 10000)
        axes_dim: List of dimensions for each axis (e.g., [16, 56, 56] for time, height, width)
    """

    def __init__(
        self, theta: int = 10000, axes_dim: tuple[int, int, int] = (16, 56, 56)
    ):
        """Initialize Flux position embedding module.

        Args:
            theta: Base value for frequency computation (typically 10000).
            axes_dim: Dimensions for each axis (e.g., [16, 56, 56] for time, height, width).
        """
        super().__init__()
        self.theta = float(theta)
        self.axes_dim = list(axes_dim)

    def _get_1d_rotary_pos_embed(
        self, dim: int, pos: Tensor, device: Device | DeviceRef
    ) -> tuple[Tensor, Tensor]:
        """Compute 1D rotary position embeddings for a single axis.

        Args:
            dim: Dimension of the embedding (should be even)
            pos: Position indices, shape [batch_size]
            device: Device to compute on

        Returns:
            Tuple of (freqs_cos, freqs_sin), each with shape [batch_size, dim]
        """
        # Ensure dim is even
        assert dim % 2 == 0, f"dim must be even, got {dim}"

        # Cast position to float32 for computation
        pos = F.cast(pos, DType.float32)

        # Compute frequencies: 1.0 / (theta ** (arange(0, dim, 2) / dim))
        # Shape: [dim/2]
        arange_vals = F.arange(
            0, dim, step=2, dtype=DType.float32, device=device
        )
        exponents = arange_vals / float(dim)

        # theta ** exponents
        theta_tensor = F.constant(self.theta, DType.float32, device=device)
        theta_powered = F.pow(theta_tensor, exponents)

        # 1.0 / theta_powered
        freqs = 1.0 / theta_powered  # Shape: [dim/2]

        # Outer product: pos [batch_size] x freqs [dim/2] = [batch_size, dim/2]
        freqs_outer = F.outer(pos, freqs)

        # Compute cos and sin
        freqs_cos_half = F.cos(freqs_outer)  # [batch_size, dim/2]
        freqs_sin_half = F.sin(freqs_outer)  # [batch_size, dim/2]

        # Repeat interleave to get full dimension
        # repeat_interleave(2, dim=1): [a, b, c] -> [a, a, b, b, c, c]
        # Since repeat_interleave is not supported on GPU, we use reshape + tile

        # 1. Unsqueeze: [batch_size, dim/2] -> [batch_size, dim/2, 1]
        freqs_cos_expanded = F.unsqueeze(freqs_cos_half, axis=2)
        freqs_sin_expanded = F.unsqueeze(freqs_sin_half, axis=2)

        # 2. Concat to duplicate: [batch_size, dim/2, 1] -> [batch_size, dim/2, 2]
        freqs_cos_tiled = F.concat(
            [freqs_cos_expanded, freqs_cos_expanded], axis=2
        )
        freqs_sin_tiled = F.concat(
            [freqs_sin_expanded, freqs_sin_expanded], axis=2
        )

        # 3. Reshape to flatten: [batch_size, dim/2, 2] -> [batch_size, dim]
        flattened_dim = freqs_cos_tiled.shape[1] * freqs_cos_tiled.shape[2]
        freqs_cos = F.reshape(
            freqs_cos_tiled, (freqs_cos_tiled.shape[0], flattened_dim)
        )
        freqs_sin = F.reshape(
            freqs_sin_tiled, (freqs_sin_tiled.shape[0], flattened_dim)
        )

        return freqs_cos, freqs_sin

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass to compute rotary position embeddings.

        Args:
            ids: Position indices tensor with shape [batch_size, n_axes]
                 where n_axes is the number of spatial dimensions (e.g., 3 for time/height/width)

        Returns:
            Tuple of (freqs_cos, freqs_sin) with concatenated embeddings from all axes
        """
        # Get number of axes from the last dimension
        n_axes = ids.shape[-1]
        device = ids.device

        cos_out = []
        sin_out = []

        # Compute embeddings for each axis
        for i in range(int(n_axes)):
            # Extract position for this axis: ids[:, i]
            pos = ids[:, i]

            # Compute 1D rotary embeddings for this axis
            cos_embed, sin_embed = self._get_1d_rotary_pos_embed(
                dim=self.axes_dim[i], pos=pos, device=device
            )

            cos_out.append(cos_embed)
            sin_out.append(sin_embed)

        # Concatenate embeddings from all axes along the last dimension
        freqs_cos = F.concat(cos_out, axis=-1)  # [batch_size, sum(axes_dim)]
        freqs_sin = F.concat(sin_out, axis=-1)  # [batch_size, sum(axes_dim)]

        return freqs_cos, freqs_sin
