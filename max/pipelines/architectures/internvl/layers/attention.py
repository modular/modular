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

import math
from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm


def compute_heads_per_device(
    *, total_heads: int, device_idx: int, num_devices: int
) -> int:
    """Compute number of heads for a specific device with uneven distribution.

    This function distributes heads across devices handling cases where the total
    number of heads is not evenly divisible by the number of devices. It follows
    the same logic as weight._compute_shard_range for consistency.

    Args:
        total_heads: Total number of attention heads.
        device_idx: The index of the current device (0-based).
        num_devices: Total number of devices.

    Returns:
        Number of heads assigned to the specified device.

    Example:
        For 25 heads across 2 devices:
        - Device 0: 13 heads (indices 0-12)
        - Device 1: 12 heads (indices 13-24)
    """
    base_heads, remainder = divmod(total_heads, num_devices)
    if device_idx < remainder:
        return base_heads + 1
    else:
        return base_heads


class InternVLMultiheadAttention(Module, Shardable):
    """InternVL multihead attention with QK normalization support.

    This implements multi-head attention specifically for InternVL vision models,
    with optional QK normalization layers. It supports single-device execution
    and can be sharded for tensor parallel execution.
    """

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        head_dim: int,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        qk_normalization: bool = True,
        layer_norm_eps: float = 1e-6,
        scale: float | None = None,
        qkv_has_bias: bool = False,
        o_proj_has_bias: bool = False,
        stacked_qkv: bool = True,
    ) -> None:
        """Initialize InternVL attention layer.
        Args:
            num_attention_heads: The number of attention heads.
            hidden_size: The dimension of the hidden states (embed_dim).
            head_dim: Head dimension for attention.
            devices: Device(s) to place the weights and run the computation.
            dtype: DType of the QKV and output projection weights.
            qk_normalization: Whether to apply QK normalization.
            layer_norm_eps: Epsilon value for layer normalization.
            scale: Value used to scale the results of the attention output.
            qkv_has_bias: Whether to use an attention bias.
            o_proj_has_bias: Whether to use an output projection bias.
            stacked_qkv: Whether to use a single stacked QKV weight matrix.
        """
        super().__init__()

        # Store parameters
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.embed_dim = hidden_size
        self.devices = devices if devices is not None else [DeviceRef.CPU()]
        self.device = self.devices[0] if self.devices else DeviceRef.CPU()
        self.dtype = dtype
        self.scale = (
            scale if scale is not None else 1.0 / math.sqrt(self.head_dim)
        )
        self.qkv_has_bias = qkv_has_bias
        self.o_proj_has_bias = o_proj_has_bias
        self.stacked_qkv = stacked_qkv

        # InternVL-specific attributes
        self.qk_normalization = qk_normalization
        self.layer_norm_eps = layer_norm_eps

        # Initialize weights
        self._init_weights()

        # Initialize QK normalization layers if needed
        if self.qk_normalization:
            self.q_norm = RMSNorm(
                dim=self.embed_dim, dtype=dtype, eps=layer_norm_eps
            )
            self.k_norm = RMSNorm(
                dim=self.embed_dim, dtype=dtype, eps=layer_norm_eps
            )

    def _init_weights(self) -> None:
        """Initialize the attention weights."""
        if self.stacked_qkv:
            self.qkv_proj = Weight(
                name="qkv_proj.weight",
                dtype=self.dtype,
                shape=(3 * self.embed_dim, self.embed_dim),
                device=self.device,
            )

            if self.qkv_has_bias:
                self.qkv_proj_bias = Weight(
                    name="qkv_proj.bias",
                    dtype=self.dtype,
                    shape=(3 * self.embed_dim,),
                    device=self.device,
                )
        else:
            self.q_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.qkv_has_bias,
                dtype=self.dtype,
                device=self.device,
            )
            self.k_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.qkv_has_bias,
                dtype=self.dtype,
                device=self.device,
            )
            self.v_proj = Linear(
                in_dim=self.embed_dim,
                out_dim=self.embed_dim,
                has_bias=self.qkv_has_bias,
                dtype=self.dtype,
                device=self.device,
            )

        self.o_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            has_bias=self.o_proj_has_bias,
            dtype=self.dtype,
            device=self.device,
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        if self.stacked_qkv:
            return self.qkv_proj
        else:
            wq: TensorValue = self.q_proj.weight
            wk: TensorValue = self.k_proj.weight
            wv: TensorValue = self.v_proj.weight
            return ops.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.qkv_has_bias:
            return None

        if self.stacked_qkv:
            return self.qkv_proj_bias

        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

        return ops.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def _compute_qkv(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute Q, K, V projections with QK normalization."""
        # Fused in-projection for Q, K, V
        qkv = x @ self.wqkv.T

        # Add bias if present
        if self.wqkv_bias is not None:
            qkv += self.wqkv_bias

        # For tensor parallel attention with uneven head distribution,
        # the QKV output dimension matches the weight's row dimension
        # which is 3 * (num_heads_for_this_device * head_dim)
        qkv_dim = qkv.shape[-1]
        split_size = qkv_dim // 3
        q, k, v = ops.split(qkv, [split_size, split_size, split_size], axis=-1)

        # Apply QK normalization if enabled
        if self.qk_normalization:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Reshape for multihead attention
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
        v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))

        return q, k, v

    def _apply_attention(
        self, q: TensorValue, k: TensorValue, v: TensorValue
    ) -> TensorValue:
        """Apply attention mechanism to Q, K, V."""
        attn_out = flash_attention_gpu(
            q, k, v, mask_variant=MHAMaskVariant.NULL_MASK, scale=self.scale
        )

        # Reshape back
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        attn_out = attn_out.reshape(
            (batch_size, seq_len, self.num_heads * self.head_dim)
        )

        return attn_out

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass for attention computation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after attention and projection.
        """
        # Compute QKV
        q, k, v = self._compute_qkv(x)

        # Apply attention
        attn_out = self._apply_attention(q, k, v)

        # Output projection
        output = self.o_proj(attn_out)

        return output

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the attention sharding strategy."""
        if self.stacked_qkv:
            return self.qkv_proj.sharding_strategy
        else:
            return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the attention weights.

        Args:
            strategy: The sharding strategy to apply.
        """
        if strategy.is_replicate:
            # For replicate strategy, all weights use the same strategy
            if self.stacked_qkv:
                self.qkv_proj.sharding_strategy = strategy
                if self.qkv_has_bias:
                    self.qkv_proj_bias.sharding_strategy = strategy
            else:
                self.q_proj.sharding_strategy = strategy
                self.k_proj.sharding_strategy = strategy
                self.v_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
        else:
            # For tensor parallel: QKV stacked sharding, output column-wise
            num_devices = strategy.num_devices
            if self.stacked_qkv:
                self.qkv_proj.sharding_strategy = ShardingStrategy.stacked_qkv(
                    num_devices, self.num_heads, self.head_dim
                )
                if self.qkv_has_bias:
                    self.qkv_proj_bias.sharding_strategy = (
                        ShardingStrategy.stacked_qkv(
                            num_devices, self.num_heads, self.head_dim
                        )
                    )
            else:
                self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
                self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
                self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.num_heads, self.head_dim
                )
            )

        # Set replicate strategy for QK norm weights, if present.
        # They operate on full embedding dimension, not per-head.
        if self.qk_normalization:
            self.q_norm.weight.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.k_norm.weight.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )

    def shard(
        self, shard_idx: int, device: DeviceRef
    ) -> InternVLMultiheadAttention:
        """Creates a sharded view of this attention layer for a specific device.

        Overrides the parent method to handle QK normalization layers.

        Args:
            shard_idx: The index of the shard (0 to num_devices-1).
            device: The device where this shard should reside.

        Returns:
            A sharded InternVLMultiheadAttention instance.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "InternVLMultiheadAttention layer cannot be sharded because no sharding strategy was provided."
            )

        # Calculate sharded dimensions - handle uneven head distribution
        sharded_num_heads = compute_heads_per_device(
            total_heads=self.num_heads,
            device_idx=shard_idx,
            num_devices=self.sharding_strategy.num_devices,
        )

        # Create new attention instance with sharded configuration
        sharded = InternVLMultiheadAttention(
            num_attention_heads=sharded_num_heads,
            hidden_size=self.embed_dim,
            head_dim=self.head_dim,
            devices=[device],
            dtype=self.q_proj.weight.dtype
            if hasattr(self, "q_proj")
            else self.qkv_proj.dtype,
            scale=self.scale,
            qkv_has_bias=self.qkv_has_bias,
            o_proj_has_bias=self.o_proj_has_bias,
            stacked_qkv=self.stacked_qkv,
            qk_normalization=self.qk_normalization,
            layer_norm_eps=self.layer_norm_eps,
        )

        # Shard weights using parent's logic
        if self.stacked_qkv:
            sharded.qkv_proj = self.qkv_proj.shard(shard_idx, device)
            if self.qkv_has_bias:
                sharded.qkv_proj_bias = self.qkv_proj_bias.shard(
                    shard_idx, device
                )
        else:
            sharded.q_proj = self.q_proj.shard(shard_idx, device)
            sharded.k_proj = self.k_proj.shard(shard_idx, device)
            sharded.v_proj = self.v_proj.shard(shard_idx, device)

        sharded.o_proj = self.o_proj.shard(shard_idx, device)

        # Shard QK normalization layers using replicate strategy.
        if self.qk_normalization:
            sharded.q_norm.weight = self.q_norm.weight.shard(shard_idx, device)
            sharded.k_norm.weight = self.k_norm.weight.shard(shard_idx, device)

        return sharded
