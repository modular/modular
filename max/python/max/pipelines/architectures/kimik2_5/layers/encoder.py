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

"""Encoder layer for Kimi K2.5 vision tower."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike
from max.nn.layer import Module
from max.nn.norm import LayerNorm

from .attention import Attention
from .mlp import MLP2


class EncoderLayer(Module):
    """Vision encoder layer with QKV-packed self-attention and MLP.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension of the encoder.
        mlp_dim: Inner dimension of the feed-forward MLP.
        dtype: Data type for all layer weights.
        device: Device on which to allocate weights.
        has_bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
    ) -> None:
        super().__init__()

        self.norm0 = LayerNorm(dims=hidden_dim, devices=[device], dtype=dtype)
        self.norm1 = LayerNorm(dims=hidden_dim, devices=[device], dtype=dtype)
        self.attn = Attention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.mlp = MLP2(
            dim=(hidden_dim, mlp_dim, hidden_dim),
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

    def __call__(
        self,
        x: TensorValueLike,
        input_row_offsets: TensorValue,
        max_seq_len: TensorValue,
        rope_freqs_cis: TensorValue,
    ) -> TensorValue:
        """Full encoder forward pass.

        Args:
            x: Packed input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            rope_freqs_cis: Precomputed [cos, sin] pairs of shape
                (n_patches, head_dim // 2, 2).

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        x = TensorValue(x)
        residual = x
        x = self.norm0(x)
        x = self.attn(x, input_row_offsets, max_seq_len, rope_freqs_cis)
        x = residual + x

        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + x

        return x
