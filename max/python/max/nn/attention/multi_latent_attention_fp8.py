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

from ..float8_config import Float8Config, nvfp4_packed_k
from ..float8_ops import matmul_float8
from ..kernels import (
    flare_mla_prefill_plan,
    fused_qkv_ragged_matmul_scaled_float8,
    mla_decode_graph,
    mla_prefill_decode_graph,
    mla_prefill_graph,
    quantize_dynamic_scaled_float8,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..norm import RMSNorm
from ..rotary_embedding import RotaryEmbedding
from .mask_config import MHAMaskVariant
from .multi_latent_attention import MLAPrefillMetadata


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

        self._scale = (
            scale if scale is not None else math.sqrt(1.0 / self.qk_head_dim)
        )
        self.scale = self.rope.compute_scale(self._scale)
        self.devices = devices or [DeviceRef.CPU()]
        assert float8_config.weight_scale.block_size is not None
        assert float8_config.input_scale.block_size is not None
        self.weight_block_size = float8_config.weight_scale.block_size

        k_block = self.weight_block_size[1]
        input_k_block = float8_config.input_scale.block_size[1]
        if input_k_block != k_block:
            raise ValueError(
                "Input scale and weight scale must have the same K block size"
            )

        self.scales_granularity_mnk = (
            float8_config.input_scale.block_size[0],
            self.weight_block_size[0],
            input_k_block,
        )

        proj_dtype = DType.float8_e4m3fn
        self.q_a_proj = Weight(
            name="q_a_proj.weight",
            dtype=proj_dtype,
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
                    input_k_block,
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
            dtype=proj_dtype,
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
                    input_k_block,
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
            dtype=proj_dtype,
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
                    input_k_block,
                ),
            ),
            device=self.devices[0],
        )

        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=proj_dtype,
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
                    input_k_block,
                ),
            ),
            device=self.devices[0],
        )

        o_proj_float8_config = float8_config
        o_proj_in_dim = nvfp4_packed_k(
            self.n_heads * self.v_head_dim, float8_config
        )
        self.o_proj = linear_cls(
            in_dim=o_proj_in_dim,
            out_dim=self.hidden_size,
            dtype=proj_dtype,
            device=self.devices[0],
            float8_config=o_proj_float8_config,
        )

    def create_mla_prefill_metadata(
        self, input_row_offsets: TensorValue, kv_collection: PagedCacheValues
    ) -> MLAPrefillMetadata:
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

        return MLAPrefillMetadata(
            buffer_row_offsets=buffer_row_offsets,
            cache_offsets=cache_offsets,
            buffer_lengths=buffer_lengths,
        )

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
        if strategy.is_replicate:
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
                "Only replicate sharding strategy is supported for LatentAttentionWithRopeFp8"
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
    def _kv_b_proj_weight(self) -> TensorValue:
        """Returns `kv_b_proj` reshaped for per-head projection slicing."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)
        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (self.kv_lora_rank, self.n_heads, -1)
        )
        return kv_b_proj_weight

    @property
    def _qk_nope_head_scale_dim(self) -> int:
        return self.qk_nope_head_dim // self.weight_block_size[0]

    @property
    def _v_head_scale_dim(self) -> int:
        return self.v_head_dim // self.weight_block_size[0]

    @property
    def _kv_b_proj_weight_scale(self) -> TensorValue:
        """Returns reshaped `kv_b_proj_scale` aligned with `_kv_b_proj_weight`."""
        kv_b_proj_weight_scale = self.kv_b_proj_scale.transpose(0, 1)
        kv_b_proj_weight_scale = kv_b_proj_weight_scale.reshape(
            (self.kv_lora_rank // self.weight_block_size[0], self.n_heads, -1)
        )
        return kv_b_proj_weight_scale

    @property
    def w_uk(self) -> tuple[TensorValue, TensorValue]:
        """Returns decode K-projection tensor/scale with shape [H, kv_rank, qk_nope_dim]."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        w_uk = w_uk_base.transpose(0, 1)

        w_uk_scale_base = self._kv_b_proj_weight_scale[
            ..., : self._qk_nope_head_scale_dim
        ]
        w_uk_scale = w_uk_scale_base.transpose(0, 1)
        return (w_uk, w_uk_scale)

    @property
    def w_uv(self) -> tuple[TensorValue, TensorValue]:
        """Returns decode V-projection tensor/scale with shape [H, v_dim, kv_rank]."""
        w_uv_base = self._kv_b_proj_weight[..., self.qk_nope_head_dim :]
        w_uv = w_uv_base.permute([1, 2, 0])

        w_uv_scale_base = self._kv_b_proj_weight_scale[
            ..., self._qk_nope_head_scale_dim :
        ]
        w_uv_scale = w_uv_scale_base.permute([1, 2, 0])
        return (w_uv, w_uv_scale)

    @property
    def w_k(self) -> tuple[TensorValue, TensorValue]:
        """Returns prefill K-projection tensor/scale with shape [H*qk_nope_dim, kv_rank]."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        w_k = w_uk_base.permute([1, 2, 0]).reshape((-1, self.kv_lora_rank))

        w_uk_scale_base = self._kv_b_proj_weight_scale[
            ..., : self._qk_nope_head_scale_dim
        ]
        w_k_scale = w_uk_scale_base.permute([1, 2, 0]).reshape(
            (-1, self.kv_lora_rank // self.weight_block_size[0])
        )
        return (w_k, w_k_scale)

    def _mla_impl(
        self,
        xq: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_a_proj_layernorm: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> TensorValue:
        # Prepare the inputs and weights for the prefill and decode branches.
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_a_proj_layernorm,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": 1e-6,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "float8_config": self.float8_config,
        }

        w_k, w_k_scale = self.w_k
        w_uk, w_uk_scale = self.w_uk
        w_uv, w_uv_scale = self.w_uv
        if self.graph_mode in ["prefill", "auto"]:
            if _mla_prefill_metadata is None:
                mla_prefill_metadata = self.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            else:
                mla_prefill_metadata = _mla_prefill_metadata

            attn_kwargs["buffer_row_offsets"] = (
                mla_prefill_metadata.buffer_row_offsets
            )
            attn_kwargs["cache_offsets"] = mla_prefill_metadata.cache_offsets
            attn_kwargs["buffer_length"] = (
                mla_prefill_metadata.buffer_lengths.to(DeviceRef.CPU())
            )
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_k_scale"] = w_k_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode in ["decode", "auto"]:
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode == "prefill":
            result = mla_prefill_graph(**attn_kwargs)
        elif self.graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs)
        else:
            result = mla_prefill_decode_graph(**attn_kwargs)

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
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

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        # QK RoPE and RMSNorm of K cache are handled inside the MLA kernel.
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        attn_out = self._mla_impl(
            xq,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            mla_prefill_metadata,
        )

        return self.o_proj(attn_out)


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

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
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

            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                assert (
                    mla_prefill_metadata is None
                    or len(mla_prefill_metadata) == 0
                )
                mla_prefill_metadata_i = None

            outs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_prefill_metadata=mla_prefill_metadata_i,
                )
            )
        return outs
