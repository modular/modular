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
"""InternVL attention layers with QK normalization support."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.multihead_attention import MultiheadAttention
from max.nn.norm import DistributedRMSNorm, RMSNorm


class InternVLMultiheadAttention(MultiheadAttention):
    """InternVL multihead attention with QK normalization support.
    Extends the base MultiheadAttention to add QK normalization layers
    as used in InternVL vision models.
    """

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        qk_normalization: bool = True,
        layer_norm_eps: float = 1e-6,
        scale: float | None = None,
        has_bias: bool = False,
        stacked_qkv: bool = True,
    ) -> None:
        """Initialize InternVL attention layer.
        Args:
            num_attention_heads: The number of attention heads.
            hidden_size: The dimension of the hidden states (embed_dim).
            devices: Device(s) to place the weights and run the computation.
                If multiple devices provided, uses distributed computation.
            dtype: DType of the QKV and output projection weights.
            qk_normalization: Whether to apply QK normalization.
            layer_norm_eps: Epsilon value for layer normalization.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            stacked_qkv: Whether to use a single stacked QKV weight matrix.
        """
        devices_list = list(devices) if devices else []

        super().__init__(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            devices=devices_list if devices_list else None,
            dtype=dtype,
            scale=scale,
            has_bias=has_bias,
            stacked_qkv=stacked_qkv,
        )

        self.qk_normalization = qk_normalization
        self.layer_norm_eps = layer_norm_eps

        # Type annotations for normalization layers
        self.q_norm: RMSNorm | DistributedRMSNorm
        self.k_norm: RMSNorm | DistributedRMSNorm

        # Initialize normalization layers
        if self.qk_normalization:
            if self.is_distributed:
                self.q_norm = DistributedRMSNorm(
                    dim=self.embed_dim,
                    dtype=dtype,
                    eps=layer_norm_eps,
                    devices=devices_list,
                )
                self.k_norm = DistributedRMSNorm(
                    dim=self.embed_dim,
                    dtype=dtype,
                    eps=layer_norm_eps,
                    devices=devices_list,
                )
            else:
                self.q_norm = RMSNorm(
                    dim=self.embed_dim,
                    dtype=dtype,
                    eps=layer_norm_eps,
                )
                self.k_norm = RMSNorm(
                    dim=self.embed_dim,
                    dtype=dtype,
                    eps=layer_norm_eps,
                )

    def _compute_qkv(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute Q, K, V projections with QK normalization."""
        # Fused in-projection for Q, K, V
        qkv = x @ self.wqkv.T

        q, k, v = ops.split(
            qkv, [self.embed_dim, self.embed_dim, self.embed_dim], axis=-1
        )

        # Apply QK normalization if enabled
        if self.qk_normalization and not self.is_distributed:
            if not isinstance(self.q_norm, RMSNorm):
                raise TypeError(
                    f"Expected RMSNorm for q_norm, got {type(self.q_norm)}"
                )
            if not isinstance(self.k_norm, RMSNorm):
                raise TypeError(
                    f"Expected RMSNorm for k_norm, got {type(self.k_norm)}"
                )

            # Type narrowing for mypy, cause its dumb and doesn't understand the previous isinstance checks
            q_norm = cast(RMSNorm, self.q_norm)
            k_norm = cast(RMSNorm, self.k_norm)
            q = q_norm(q)
            k = k_norm(k)

        # Reshape for multihead attention
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        return q, k, v
