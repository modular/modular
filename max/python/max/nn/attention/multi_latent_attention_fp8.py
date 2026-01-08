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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.support.math import ceildiv

from ..comm import Allreduce
from ..float8_config import Float8Config
from ..float8_ops import matmul_float8
from ..kernels import (
    flare_mla_prefill_plan,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul_scaled_float8,
    mla_decode_branch_fp8,
    mla_prefill_branch_fp8,
    mla_prefill_decode_graph_fp8,
    quantize_dynamic_scaled_float8,
    rms_norm_key_cache,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..norm import RMSNorm
from ..rotary_embedding import RotaryEmbedding
from .mask_config import MHAMaskVariant


class LatentAttentionWithRopeFp8(Module, Shardable):
    """Implementation of Latent Attention with Rope with FP8 weights."""

    rope: RotaryEmbedding

    _sharding_strategy: ShardingStrategy | None = None
    """The sharding strategy for the module."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        float8_config: Float8Config,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
    ) -> None:
        """Initializes the latent attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            dtype: DType of the weights, currently only bfloat16 is supported.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            q_lora_rank: Optional LoRA rank for Q projection.
            kv_lora_rank: LoRA rank for KV projections.
            qk_nope_head_dim: Head dimension for non-positional encoding part.
            qk_rope_head_dim: Head dimension for rope part.
            v_head_dim: Head dimension for value.
            buffer_size: Buffer size for storing the temporal results during
                prefill, in unit of tokens.
            graph_mode: Pipeline role to use for the attention layer. Should be
                "prefill", "decode", or "auto".
        """
        super().__init__()

        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or 'auto'."
            )
        if (
            not float8_config.weight_scale.is_block
            or not float8_config.input_scale.is_block
        ):
            raise ValueError(
                "Weight scale and input scale must be block-wise for LatentAttentionWithRopeFp8"
            )

        self.graph_mode = _role
        self.float8_config = float8_config

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.linear_cls = linear_cls

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim

        self.BUFFER_TOK_SIZE = buffer_size

        self._scale = scale if scale else math.sqrt(1.0 / self.qk_head_dim)
        self.scale = self.rope.compute_scale(self._scale)
        self.devices = devices or [DeviceRef.CPU()]
        assert float8_config.weight_scale.block_size is not None
        assert float8_config.input_scale.block_size is not None
        self.weight_block_size = float8_config.weight_scale.block_size

        if float8_config.input_scale.block_size[1] != self.weight_block_size[1]:
            raise ValueError(
                "Input scale and weight scale must have the same K block size"
            )
        self.scales_granularity_mnk = (
            float8_config.input_scale.block_size[0],
            self.weight_block_size[0],
            self.weight_block_size[1],
        )

        self.q_a_proj = Weight(
            name="q_a_proj.weight",
            dtype=DType.float8_e4m3fn,
            shape=(self.q_lora_rank, self.hidden_size),
            device=self.devices[0],
        )
        self.q_a_proj_scale = Weight(
            name="q_a_proj.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.q_a_proj.shape[0]),
                    float8_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.q_a_proj.shape[1]),
                    float8_config.weight_scale.block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.q_a_layernorm = RMSNorm(
            dim=self.q_lora_rank,
            dtype=DType.bfloat16,
            eps=1e-6,
            multiply_before_cast=False,
        )

        self.q_b_proj = Weight(
            name="q_b_proj.weight",
            dtype=DType.float8_e4m3fn,
            shape=(self.n_heads * self.qk_head_dim, self.q_lora_rank),
            device=self.devices[0],
        )
        self.q_b_proj_scale = Weight(
            name="q_b_proj.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.q_b_proj.shape[0]),
                    float8_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.q_b_proj.shape[1]),
                    float8_config.weight_scale.block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.kv_a_proj_layernorm = Weight(
            name="kv_a_layernorm.weight",
            dtype=DType.bfloat16,
            shape=(self.kv_lora_rank,),
            device=self.devices[0],
        )

        self.kv_a_proj_with_mqa = Weight(
            name="kv_a_proj_with_mqa.weight",
            dtype=DType.float8_e4m3fn,
            shape=(self.cache_head_dim, self.hidden_size),
            device=self.devices[0],
        )
        self.kv_a_proj_with_mqa_scale = Weight(
            name="kv_a_proj_with_mqa.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[0]),
                    self.weight_block_size[0],
                ),
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[1]),
                    self.weight_block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=DType.float8_e4m3fn,
            shape=(
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=self.devices[0],
        )
        self.kv_b_proj_scale = Weight(
            name="kv_b_proj.weight_scale",
            dtype=float8_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_b_proj.shape[0]),
                    self.weight_block_size[0],
                ),
                ceildiv(
                    int(self.kv_b_proj.shape[1]),
                    self.weight_block_size[1],
                ),
            ),
            device=self.devices[0],
        )

        self.o_proj = linear_cls(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            dtype=DType.float8_e4m3fn,
            device=self.devices[0],
            float8_config=float8_config,
        )

    def create_mla_inputs(
        self, input_row_offsets: TensorValue, kv_collection: PagedCacheValues
    ) -> list[TensorValue]:
        mla_inputs: list[TensorValue] = []
        (buffer_row_offsets, cache_offsets, buffer_lengths) = (
            flare_mla_prefill_plan(
                self.kv_params,
                input_row_offsets,
                kv_collection,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                self.BUFFER_TOK_SIZE,
                max_chunks=1,  # we only do one-shot prefill now.
            )
        )
        buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

        mla_inputs.append(buffer_row_offsets)
        mla_inputs.append(cache_offsets)
        mla_inputs.append(buffer_lengths_host)

        return mla_inputs

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Module sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the Module sharding strategy.

        Args:
            strategy: The strategy describing the Module sharding.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy

            if (self.n_heads / strategy.num_devices) % 16 != 0:
                raise ValueError(
                    "MLA head per device must be a multiple of 16."
                )

            self.q_a_proj.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.q_a_proj_scale.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.q_a_layernorm.weight.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )

            self.q_b_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.q_b_proj_scale.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )

            self.kv_a_proj_layernorm.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_a_proj_with_mqa.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_a_proj_with_mqa_scale.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )

            self.kv_b_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.kv_b_proj_scale.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )

            self.o_proj.weight.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
            if self.o_proj.input_scale is not None:
                self.o_proj.input_scale.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
            if self.o_proj.weight_scale is not None:
                self.o_proj.weight_scale.sharding_strategy = (
                    ShardingStrategy.columnwise(strategy.num_devices)
                )
        elif strategy.is_replicate:
            # Data parallelism: replicate the entire module's weights to each device.
            self._sharding_strategy = strategy

            weights = [
                self.q_a_proj,
                self.q_a_proj_scale,
                self.q_a_layernorm.weight,
                self.q_b_proj,
                self.q_b_proj_scale,
                self.kv_a_proj_layernorm,
                self.kv_a_proj_with_mqa,
                self.kv_a_proj_with_mqa_scale,
                self.kv_b_proj,
                self.kv_b_proj_scale,
                self.o_proj.weight,
            ]

            if self.o_proj.input_scale is not None:
                weights.append(self.o_proj.input_scale)
            if self.o_proj.weight_scale is not None:
                weights.append(self.o_proj.weight_scale)

            for weight in weights:
                weight.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[LatentAttentionWithRopeFp8]:
        """Creates sharded views of this Module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded LatentAttentionWithRope instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "LatentAttentionWithRope layer cannot be sharded because no sharding strategy was provided."
            )

        if self.sharding_strategy.is_tensor_parallel:
            q_a_proj_shards = self.q_a_proj.shard(devices)
            q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
            q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                devices
            )
            q_b_proj_shards = self.q_b_proj.shard(devices)
            q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_a_proj_with_mqa_scale_shards = (
                self.kv_a_proj_with_mqa_scale.shard(devices)
            )
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)

            o_proj_weight_shards = self.o_proj.weight.shard(devices)
            if self.o_proj.input_scale is not None:
                o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
            if self.o_proj.weight_scale is not None:
                o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(
                    devices
                )

            shards = []
            for shard_idx, device in enumerate(devices):
                sharded = LatentAttentionWithRopeFp8(
                    rope=self.rope,
                    num_attention_heads=self.n_heads
                    // self.sharding_strategy.num_devices,
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    float8_config=self.float8_config,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                )

                sharded.q_a_proj = q_a_proj_shards[shard_idx]
                sharded.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
                sharded.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                    shard_idx
                ]
                sharded.q_b_proj = q_b_proj_shards[shard_idx]
                sharded.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

                sharded.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                sharded.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                sharded.kv_a_proj_with_mqa_scale = (
                    kv_a_proj_with_mqa_scale_shards[shard_idx]
                )
                sharded.kv_b_proj = kv_b_proj_shards[shard_idx]
                sharded.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]

                sharded.o_proj.weight = o_proj_weight_shards[shard_idx]
                if self.o_proj.input_scale is not None:
                    sharded.o_proj.input_scale = o_proj_scale_shards[shard_idx]
                if self.o_proj.weight_scale is not None:
                    sharded.o_proj.weight_scale = o_proj_weight_scale_shards[
                        shard_idx
                    ]

                shards.append(sharded)

            return shards
        elif self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            q_a_proj_shards = self.q_a_proj.shard(devices)
            q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
            q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                devices
            )
            q_b_proj_shards = self.q_b_proj.shard(devices)
            q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_a_proj_with_mqa_scale_shards = (
                self.kv_a_proj_with_mqa_scale.shard(devices)
            )
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)
            o_proj_weight_shards = self.o_proj.weight.shard(devices)

            if self.o_proj.input_scale is not None:
                o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
            if self.o_proj.weight_scale is not None:
                o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(
                    devices
                )

            replicas: list[LatentAttentionWithRopeFp8] = []
            for shard_idx, device in enumerate(devices):
                replica = LatentAttentionWithRopeFp8(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    float8_config=self.float8_config,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                )

                replica.q_a_proj = q_a_proj_shards[shard_idx]
                replica.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
                replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                    shard_idx
                ]
                replica.q_b_proj = q_b_proj_shards[shard_idx]
                replica.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

                replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa_scale = (
                    kv_a_proj_with_mqa_scale_shards[shard_idx]
                )
                replica.kv_b_proj = kv_b_proj_shards[shard_idx]
                replica.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]
                replica.o_proj.weight = o_proj_weight_shards[shard_idx]
                if self.o_proj.input_scale is not None:
                    replica.o_proj.input_scale = o_proj_scale_shards[shard_idx]
                if self.o_proj.weight_scale is not None:
                    replica.o_proj.weight_scale = o_proj_weight_scale_shards[
                        shard_idx
                    ]

                replicas.append(replica)

            return replicas
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    @property
    def wqkv(self) -> tuple[TensorValue, TensorValue]:
        """The concatenation of q_a_proj and kv_a_proj_with_mqa weight vectors."""

        wqkv = ops.concat((self.q_a_proj, self.kv_a_proj_with_mqa))
        wqkv_scale = ops.concat(
            (self.q_a_proj_scale, self.kv_a_proj_with_mqa_scale)
        )

        return (wqkv, wqkv_scale)

    @property
    def w_uk_uv(self) -> list[TensorValue]:
        """The concatenation of q, k, and v weight vectors."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)

        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (
                self.kv_lora_rank,
                self.n_heads,
                (self.qk_nope_head_dim + self.v_head_dim),
            )
        )

        w_uk, w_uv = ops.split(
            kv_b_proj_weight, [self.qk_nope_head_dim, self.v_head_dim], axis=2
        )
        w_uv = w_uv.permute([1, 2, 0])
        w_uk_t = w_uk.transpose(0, 1)

        # Transform the scales the same way
        kv_b_proj_weight_scale = self.kv_b_proj_scale.transpose(0, 1)

        qk_nope_head_scale_dim = (
            self.qk_nope_head_dim // self.weight_block_size[0]
        )
        v_head_scale_dim = self.v_head_dim // self.weight_block_size[0]
        kv_b_proj_weight_scale = kv_b_proj_weight_scale.reshape(
            (
                self.kv_lora_rank // self.weight_block_size[0],
                self.n_heads,
                qk_nope_head_scale_dim + v_head_scale_dim,
            )
        )
        w_uk_scale, w_uv_scale = ops.split(
            kv_b_proj_weight_scale,
            [qk_nope_head_scale_dim, v_head_scale_dim],
            axis=2,
        )
        w_uv_scale = w_uv_scale.permute([1, 2, 0])
        w_uk_t_scale = w_uk_scale.transpose(0, 1)
        return [w_uk_t, w_uk_t_scale, w_uv, w_uv_scale]

    def _mla_impl(
        self,
        xq_nope: TensorValue,
        xq_rope: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        _mla_inputs: list[TensorValue] | None = None,
    ) -> TensorValue:
        # Prepare the inputs and weights for the prefill and decode branches.
        attn_kwargs: dict[str, Any] = {
            "q_nope": xq_nope,
            "q_rope": xq_rope,
            "input_row_offsets": input_row_offsets,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "float8_config": self.float8_config,
        }

        if self.graph_mode in ["prefill", "auto"]:
            if _mla_inputs is None or len(_mla_inputs) == 0:
                mla_prefill_inputs = self.create_mla_inputs(
                    input_row_offsets, kv_collection
                )
            else:
                assert len(_mla_inputs) == 3
                mla_prefill_inputs = _mla_inputs

            attn_kwargs["buffer_row_offsets"] = mla_prefill_inputs[0]
            attn_kwargs["cache_offsets"] = mla_prefill_inputs[1]
            attn_kwargs["buffer_length"] = mla_prefill_inputs[2]
            attn_kwargs["kv_b_proj"] = self.kv_b_proj
            attn_kwargs["kv_b_proj_scale"] = self.kv_b_proj_scale

        if self.graph_mode in ["decode", "auto"]:
            w_uk, w_uk_scale, w_uv, w_uv_scale = self.w_uk_uv
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode == "prefill":
            result = mla_prefill_branch_fp8(**attn_kwargs)
        elif self.graph_mode == "decode":
            result = mla_decode_branch_fp8(**attn_kwargs)
        else:
            result = mla_prefill_decode_graph_fp8(**attn_kwargs)

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_inputs: list[TensorValue] | None = None,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            self.float8_config.input_scale,
            self.float8_config.weight_scale,
            scales_type=self.kv_a_proj_with_mqa_scale.dtype,
            group_size_or_per_token=self.scales_granularity_mnk[2],
            out_type=self.kv_a_proj_with_mqa.dtype,
        )

        # First FP8 matmul: x @ q_a_proj.T, fused with x @ kv_a_proj_with_mqa.T
        wqkv, wqkv_scale = self.wqkv
        q_a_out = fused_qkv_ragged_matmul_scaled_float8(
            self.kv_params,
            x,
            input_row_offsets,
            wqkv,
            kv_collection,
            layer_idx,
            self.n_heads,
            x_scales,
            wqkv_scale,
            float8_config=self.float8_config,
            _output_dim=self.q_lora_rank,
        )

        # Apply layer norm
        q_a_normed = self.q_a_layernorm(q_a_out)

        # Second FP8 matmul: q_a_normed @ q_b_proj.T
        xq = matmul_float8(
            x=q_a_normed,
            weight=self.q_b_proj,
            weight_scale=self.q_b_proj_scale,
            input_scale=None,  # Dynamic scaling
            float8_config=self.float8_config,
            group_size_or_per_token=self.scales_granularity_mnk[2],
        )

        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.kv_a_proj_layernorm,
            epsilon=1e-6,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            rms_norm_cols=self.kv_lora_rank,
            weight_offset=0.0,
            multiply_before_cast=False,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        xq_nope, xq_rope = ops.split(
            xq, [self.qk_nope_head_dim, self.qk_rope_head_dim], axis=2
        )

        # Apply rope.
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        xq_rope = fused_qk_ragged_rope(
            self.kv_params,
            xq_rope,
            input_row_offsets,
            kv_collection,
            freqs_cis=freqs_cis,
            layer_idx=layer_idx,
            interleaved=True,
        )

        attn_out = self._mla_impl(
            xq_nope,
            xq_rope,
            kv_collection,
            layer_idx,
            input_row_offsets,
            mla_inputs,
        )

        return self.o_proj(attn_out)


class TensorParallelLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """Distributed tensor parallel implementation of the Latent Attention with
    Rope (FP8). Note that using tensor parallelism for MLA will cause the
    KV-cache to be duplicated across all devices, which is not efficient.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.allreduce = Allreduce(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_inputs(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[TensorValue]:
        multi_mla_inputs: list[TensorValue] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_inputs.extend(
                super().create_mla_inputs(input_row_offsets, kv_collection)
            )

        return multi_mla_inputs

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_inputs: list[TensorValue] | None = None,
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")
        if len(input_row_offsets) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} input_row_offsets, got {len(input_row_offsets)}"
            )
        if not all(isinstance(x, TensorValue) for x in input_row_offsets):
            raise TypeError(
                "All elements in input_row_offsets must be TensorValue instances"
            )

        n = len(self.devices)
        inputs: list[TensorValue] = []
        for i in range(n):
            mla_inputs_i: list[TensorValue] | None
            if mla_inputs is not None and len(mla_inputs) == 3 * n:
                mla_inputs_i = mla_inputs[3 * i : 3 * (i + 1)]
            else:
                mla_inputs_i = mla_inputs
            inputs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_inputs=mla_inputs_i,
                )
            )

        return self.allreduce(
            inputs=inputs,
            signal_buffers=signal_buffers,
        )


class DataParallelLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """Data-parallel implementation of Latent Attention with RoPE.

    This replicates the attention module across devices and runs each replica on
    its local inputs (x, kv, freqs_cis, input_row_offsets). No collective ops
    are required; KV-cache remains local to each device.

    Notes:
      - `signal_buffers` is accepted for interface parity with the distributed
        implementation but is not used here.
      - Assumes the caller has already distributed `xs`, `kv_collections`,
        `freqs_cis`, and `input_row_offsets` so that index i corresponds to
        device i, with `input_row_offsets[i]` rebased to start at 0.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_inputs(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[TensorValue]:
        multi_mla_inputs: list[TensorValue] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_inputs.extend(
                super().create_mla_inputs(input_row_offsets, kv_collection)
            )

        return multi_mla_inputs

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_inputs: list[TensorValue] | None = None,
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, freqs_cis, and input_row_offsets must all have "
                f"length equal to number of devices ({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

            mla_inputs_i: list[TensorValue] | None
            if mla_inputs is not None and len(mla_inputs) == 3 * n:
                mla_inputs_i = mla_inputs[3 * i : 3 * (i + 1)]
            else:
                mla_inputs_i = mla_inputs

            outs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_inputs=mla_inputs_i,
                )
            )
        return outs
