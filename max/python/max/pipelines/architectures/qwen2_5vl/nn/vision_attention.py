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
"""Qwen2.5vVL attention layers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.pipelines.architectures.internvl.layers.attention import (
    compute_heads_per_device,
)


class DistributedVisionWindowAttention(Module, Shardable):
    """Sliding Window Vision Attention Layer for Qwen2.5vVL.

    This layer implements sliding window vision attention for Qwen2.5vVL.
    It does the following steps:
    1. Linear Projections Q, K, V
    2. Apply Rotary position embeddings on the Linear Projections Q, and K
    3. flash attention
    4. Final Linear projection layer

    It supports single-device and multi-device execution by tensor parallelism.
    """

    def __init__(
        self,
        dtype: DType,
        hidden_size: int,
        n_heads: int,
        head_dim: int,
        flash_attention: bool = False,
        devices: Sequence[DeviceRef] | None = None,
    ):
        super().__init__()
        self.dtype = dtype
        self.devices = devices if devices is not None else [DeviceRef.CPU()]
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Add explicit scaling factor
        self.scaling = math.sqrt(1.0 / self.head_dim)
        self.flash_attention = flash_attention

        self.qkv_proj = Weight(
            name="qkv.weight",
            dtype=self.dtype,
            shape=(3 * hidden_size, hidden_size),
            device=self.devices[0],
        )
        self.qkv_proj_bias = Weight(
            name="qkv.bias",
            dtype=self.dtype,
            shape=(3 * hidden_size,),
            device=self.devices[0],
        )

        self.proj = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            has_bias=True,
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        return self.qkv_proj

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        return self.qkv_proj_bias

    def _compute_qkv(
        self, x: TensorValue
    ) -> tuple[TensorValue, TensorValue, TensorValue]:
        """Compute stacked Q, K, V then reshape to match attention layout."""
        # x: [seq_len, hidden_size]
        seq_len = x.shape[0]

        # Fused in-projection for Q, K, V using stacked weights
        qkv = x @ self.wqkv.T
        if self.wqkv_bias is not None:
            qkv += self.wqkv_bias
        # For tensor parallel attention with uneven head distribution,
        # the QKV output dimension matches the weight's row dimension
        # which is 3 * (num_heads_for_this_device * head_dim)
        qkv_dim = qkv.shape[-1]
        split_size = qkv_dim // 3
        q, k, v = ops.split(qkv, [split_size, split_size, split_size], axis=-1)
        # Reshape for multihead attention
        q = q.reshape((seq_len, -1, self.head_dim))
        k = k.reshape((seq_len, -1, self.head_dim))
        v = v.reshape((seq_len, -1, self.head_dim))

        return q, k, v

    @staticmethod
    def apply_rotary_pos_emb_vision(
        q: TensorValue, k: TensorValue, cos: TensorValue, sin: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        def _rotate_half(x: TensorValue) -> TensorValue:
            """Rotates half the hidden dims of the input."""
            head_dim = x.shape[-1]
            half_dim = head_dim // 2
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            return ops.concat((-x2, x1), -1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q, k = ops.cast(q, DType.float32), ops.cast(k, DType.float32)
        cos, sin = (
            ops.cast(ops.unsqueeze(cos, -2), DType.float32),
            ops.cast(ops.unsqueeze(sin, -2), DType.float32),
        )
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        q_embed = ops.cast(q_embed, orig_q_dtype)
        k_embed = ops.cast(k_embed, orig_k_dtype)
        return q_embed, k_embed

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
    ) -> TensorValue:
        """Naive Sliding Window Vision Attention Layer for Qwen2.5vVL. It does the following steps:
            1. Linear Projections Q, K, V
            2. Apply Rotary position embeddings on the Linear Projections Q, and K
            3. Scaled dot product attention
            4. Final Linear projection layer

        Args:
            x: Input tensor of shape (seq_len, hidden_size)
            position_embeddings: Tuple of (cos, sin) tensors for rotary embeddings
            input_row_offsets: Tensor of shape [window_size + 1] with dtype uint32.
                Indicates where each window starts and ends in the ragged tensors.
                The values should be a prefix sum (cumulative sum) of window lengths.
            max_seqlen: Maximum window length for flash attention.

        Returns:
            The output of applying sliding window attention on input `x` using `attention_mask`.
            It applies rotary position embeddings `position_embeddings` in the process.

        Shapes:
            Input:
                x: (seq_len, hidden_size)
                position_embeddings: tuple of 2 tensors of shape (seq_len, head_dim)
                input_row_offsets: (window_size + 1)
                max_seqlen: (1)
            Output:
                - tensor: (seq_len, hidden_size)
        """
        seq_length = x.shape[0]
        # Compute Q, K, V shaped [seq_len, n_heads, head_dim]
        xq, xk, xv = self._compute_qkv(x)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        xq, xk = DistributedVisionWindowAttention.apply_rotary_pos_emb_vision(
            xq, xk, cos, sin
        )

        attn_output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scaling,
        )
        attn_output = attn_output.reshape((seq_length, -1))
        attn_output = self.proj(attn_output)
        return attn_output

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
            # For replicate strategy, all weights use the same strategy
            self.qkv_proj.sharding_strategy = strategy
            self.qkv_proj_bias.sharding_strategy = strategy
            self.proj.sharding_strategy = strategy
        else:
            # For tensor parallel: QKV stacked sharding, output column-wise
            num_devices = strategy.num_devices

            self.qkv_proj.sharding_strategy = ShardingStrategy.stacked_qkv(
                num_devices, self.n_heads, self.head_dim
            )
            self.qkv_proj_bias.sharding_strategy = ShardingStrategy.stacked_qkv(
                num_devices, self.n_heads, self.head_dim
            )
            self.proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.head_dim
                )
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[DistributedVisionWindowAttention]:
        """Creates sharded views of this attention layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded DistributedVisionWindowAttention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "DistributedVisionWindowAttention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        qkv_proj_shards = self.qkv_proj.shard(devices)
        qkv_proj_bias_shards = self.qkv_proj_bias.shard(devices)

        proj_shards = self.proj.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            sharded_num_heads = compute_heads_per_device(
                total_heads=self.n_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            # Create new attention instance with sharded configuration
            sharded = DistributedVisionWindowAttention(
                hidden_size=self.hidden_size,
                n_heads=sharded_num_heads,
                head_dim=self.head_dim,
                devices=[device],
                dtype=self.dtype,
                flash_attention=self.flash_attention,
            )

            # Assign sharded weights
            sharded.qkv_proj = qkv_proj_shards[shard_idx]
            sharded.qkv_proj_bias = qkv_proj_bias_shards[shard_idx]

            sharded.proj = proj_shards[shard_idx]

            shards.append(sharded)

        return shards
