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
"""Qwen3VL-MoE attention layer for the language model decoder (text)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.float8_config import Float8Config
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    fused_qkv_ragged_matmul_scaled_float8,
    quantize_dynamic_scaled_float8,
    rms_norm_key_cache,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues, uses_opaque
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

from .text_rotary import Qwen3VLTextRotaryEmbedding


class Qwen3VLMoEDecoderAttentionWithRope(Module, Shardable):
    """Qwen3VLMoE-style attention with RoPE and per-head Q/K RMSNorm (text)."""

    rope: Qwen3VLTextRotaryEmbedding

    def __init__(
        self,
        *,
        rope: Qwen3VLTextRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        float8_config: Float8Config | None = None,
    ) -> None:
        super().__init__()

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / float(self.kv_params.head_dim))
        )

        self.devices = devices or [DeviceRef.CPU()]
        self._sharding_strategy: ShardingStrategy | None = None
        self.float8_config = float8_config

        if not uses_opaque(self.kv_params.cache_strategy):
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy is not supported "
                "in Qwen3VLMoEDecoderAttentionWithRope."
            )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )

        # Per-head RMSNorm for Q and K.
        self.q_norm = RMSNorm(
            dim=self.kv_params.head_dim,
            dtype=kv_params.dtype,
            eps=rms_norm_eps,
            multiply_before_cast=True,
        )
        self.k_norm = RMSNorm(
            dim=self.kv_params.head_dim,
            dtype=kv_params.dtype,
            eps=rms_norm_eps,
            multiply_before_cast=True,
        )

    @property
    def wqkv(self) -> TensorValue:
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wq, wk, wv))

    @property
    def wqkv_bias(self) -> TensorValue | None:
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    @property
    def wqkv_scale(self) -> TensorValue | None:
        """Concatenated QKV weight scales for FP8 models."""
        if self.q_proj.weight_scale is None:
            return None
        assert self.k_proj.weight_scale is not None
        assert self.v_proj.weight_scale is not None
        return ops.concat(
            (
                self.q_proj.weight_scale,
                self.k_proj.weight_scale,
                self.v_proj.weight_scale,
            )
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass over a batch of tokens in ragged form.

        Args:
            layer_idx: Layer index for KV cache.
            x: Flattened input [T, H] for all sequences in the batch, where
                T = sum_i L_i over all sequences.
            kv_collection: KV cache handle.
            freqs_cis: RoPE coefficient table, shape [max_seq_len * 2, rope_dim].
            input_row_offsets: Ragged offsets [0, L0, L0+L1, ...]. For a single
                contiguous sequence of length L this is simply [0, L].
        """
        total_seq_len = x.shape[0]

        # Keep attention stack in BF16.
        self.kv_params.dtype = DType.bfloat16

        # Make sure activations are BF16 for the fused kernel.
        x_in = x if x.dtype == DType.bfloat16 else ops.cast(x, DType.bfloat16)

        # Get concatenated QKV weights and optional bias
        wqkv = self.wqkv
        wqkv_bias = (
            self.wqkv_bias.to(x_in.device)
            if self.wqkv_bias is not None
            else None
        )

        # Project QKV using FP8 or BF16 path
        if self.float8_config and self.q_proj.weight.dtype.is_float8():
            # FP8 path: dynamically quantize input and use FP8 fused kernel
            weight_scale = self.wqkv_scale
            assert weight_scale is not None

            # Get K block size for dynamic quantization (matches DeepSeek pattern)
            weight_block_size = self.float8_config.weight_scale.block_size
            k_block_size = (
                weight_block_size[1] if weight_block_size is not None else -1
            )

            x_fp8, x_scales = quantize_dynamic_scaled_float8(
                x_in,
                self.float8_config.input_scale,
                self.float8_config.weight_scale,
                scales_type=weight_scale.dtype,
                group_size_or_per_token=k_block_size,
                out_type=self.q_proj.weight.dtype,
            )

            xq = fused_qkv_ragged_matmul_scaled_float8(
                self.kv_params,
                input=x_fp8,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                input_scale=x_scales.to(x_fp8.device),
                weight_scale=weight_scale.to(x_fp8.device),
                float8_config=self.float8_config,
            )
        else:
            # BF16 path: regular fused QKV matmul
            if wqkv.dtype != DType.bfloat16:
                wqkv = ops.cast(wqkv, DType.bfloat16)

            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x_in,
                wqkv=wqkv,
                bias=wqkv_bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )

        # [T, n_heads, head_dim]; per-head RMSNorm on Q.
        xq = ops.reshape(xq, (-1, self.n_heads, self.kv_params.head_dim))
        xq = self.q_norm(xq)

        # Apply learned K RMSNorm in-place on new cache keys.
        rms_norm_key_cache(
            kv_params=self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight,
            epsilon=self.rms_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            rms_norm_cols=None,
            multiply_before_cast=True,
            per_head_norm=True,
        )

        # Apply RoPE to Q and (read) K; positions are derived inside the fused
        # kernel as cache_length + token_idx for each token.
        position_ids = ops.unsqueeze(
            ops.range(0, xq.shape[0], 1, device=xq.device, dtype=DType.uint32),
            0,
        )
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis.to(xq.device),
            layer_idx=layer_idx,
            interleaved=self.rope.interleaved,
            mrope_section=None,
            position_ids=position_ids,
        )

        # Flash attention over Q and normalized/rotated K/V.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Merge heads + output projection.
        # The Linear layer handles FP8 internally via dynamic_scaled_matmul.
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.q_proj.sharding_strategy = sharding_strategy
            self.k_proj.sharding_strategy = sharding_strategy
            self.v_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy
            self.q_norm.sharding_strategy = sharding_strategy
            self.k_norm.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
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
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            # Note: For FP8 block-wise scales with columnwise weight sharding,
            # Linear.sharding_strategy setter automatically uses columnwise
            # sharding for the scale's K dimension to match the sharded input.
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "Qwen3VLMoEDecoderAttentionWithRope only supports tensor parallel "
                "and replicate sharding strategies."
            )

        self._sharding_strategy = sharding_strategy

    def shard(
        self,
        devices: Iterable[DeviceRef],
    ) -> list[Qwen3VLMoEDecoderAttentionWithRope]:
        if not self.sharding_strategy:
            raise ValueError(
                "Qwen3VLMoEDecoderAttentionWithRope layer cannot be sharded "
                "because no sharding strategy was provided."
            )

        devices_list = list(devices)

        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[Qwen3VLMoEDecoderAttentionWithRope] = []
        num_shards = len(devices_list)

        for shard_idx, device in enumerate(devices_list):
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, num_shards
            )
            sharded_num_heads = head_end - head_start

            kv_head_start, kv_head_end = _compute_shard_range(
                self.kv_params.n_kv_heads, shard_idx, num_shards
            )
            sharded_num_kv_heads = kv_head_end - kv_head_start

            # Use the same Linear subclass as q_proj; the actual shard
            # modules are assigned immediately below.
            sharded = Qwen3VLMoEDecoderAttentionWithRope(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                dtype=self.q_proj.weight.dtype,
                devices=[device],
                linear_cls=self.q_proj.__class__,
                scale=self.scale,
                has_bias=self.has_bias,
                rms_norm_eps=self.rms_norm_eps,
                float8_config=self.float8_config,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]
            shards.append(sharded)

        return shards
