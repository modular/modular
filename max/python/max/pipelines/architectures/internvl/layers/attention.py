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
"""InternVL attention layers with QK normalization support."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention import num_heads_for_device
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.stacked_linear import StackedLinear


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
        self.qkv_proj = StackedLinear(
            in_dim=self.embed_dim,
            out_dims=[self.embed_dim, self.embed_dim, self.embed_dim],
            names=["q", "k", "v"],
            dtype=self.dtype,
            device=self.device,
            stacked=self.stacked_qkv,
            has_bias=self.qkv_has_bias,
        )

        self.o_proj = Linear(
            in_dim=self.embed_dim,
            out_dim=self.embed_dim,
            has_bias=self.o_proj_has_bias,
            dtype=self.dtype,
            device=self.device,
        )

    def _compute_qkv(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute Q, K, V projections with QK normalization."""
        # Fused in-projection for Q, K, V via StackedLinear.
        qkv = self.qkv_proj(x)

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
        return self.qkv_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the attention weights.

        Args:
            strategy: The sharding strategy to apply.
        """
        if strategy.is_replicate:
            self.qkv_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
        else:
            # For tensor parallel: QKV stacked sharding, output column-wise
            num_devices = strategy.num_devices
            if self.stacked_qkv:
                self.qkv_proj.sharding_strategy = ShardingStrategy.stacked_qkv(
                    num_devices, self.num_heads, self.head_dim
                )
            else:
                self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
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
        self, devices: Iterable[DeviceRef]
    ) -> list[InternVLMultiheadAttention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded InternVLMultiheadAttention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "InternVLMultiheadAttention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        qkv_proj_shards = self.qkv_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        q_norm_weight_shards = []
        k_norm_weight_shards = []
        if self.qk_normalization:
            q_norm_weight_shards = self.q_norm.weight.shard(devices)
            k_norm_weight_shards = self.k_norm.weight.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            sharded_num_heads = num_heads_for_device(
                num_heads=self.num_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            # Create new attention instance with sharded configuration
            sharded = InternVLMultiheadAttention(
                num_attention_heads=sharded_num_heads,
                hidden_size=self.embed_dim,
                head_dim=self.head_dim,
                devices=[device],
                dtype=self.dtype,
                scale=self.scale,
                qkv_has_bias=self.qkv_has_bias,
                o_proj_has_bias=self.o_proj_has_bias,
                stacked_qkv=self.stacked_qkv,
                qk_normalization=self.qk_normalization,
                layer_norm_eps=self.layer_norm_eps,
            )

            # Assign sharded weights
            sharded.qkv_proj = qkv_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign QK normalization weights
            if self.qk_normalization:
                sharded.q_norm.weight = q_norm_weight_shards[shard_idx]
                sharded.k_norm.weight = k_norm_weight_shards[shard_idx]

            shards.append(sharded)

        return shards
