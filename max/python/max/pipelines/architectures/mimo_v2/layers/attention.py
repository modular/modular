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
"""MiMo-V2 attention: fused chunk-order QKV, partial RoPE, sinks, SWA."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import RotaryEmbedding

from ..weight_adapters import QkvChunkLayout


def _split_heads(x: TensorValue, head_dim: int) -> TensorValue:
    """Views ``[tokens, chunks, heads * head_dim]`` as its heads.

    Returns ``[tokens, chunks, heads, head_dim]``. When a whole chunk is a
    whole number of ``head_dim`` rows, this stays a view of the chunk slice,
    which the pad's concat reads in place; otherwise the compiler copies the
    slice first. Merging ``chunks`` and ``heads`` before the pad copies it
    either way.
    """
    tokens, chunks, rows = x.shape
    return ops.reshape(x, [tokens, chunks, int(rows) // head_dim, head_dim])


def _pad_heads(x: TensorValue, depth: int) -> TensorValue:
    """Zero-pads each head of ``[tokens, chunks, heads, dim]`` to ``depth``.

    Returns ``[tokens, chunks * heads, depth]``.
    """
    tokens, chunks, heads, dim = x.shape
    if depth != int(dim):
        zeros = ops.broadcast_to(
            ops.constant(0, x.dtype, device=x.device),
            [tokens, chunks, heads, depth - int(dim)],
        )
        x = ops.concat([x, zeros], axis=-1)
    return ops.reshape(x, [tokens, int(chunks) * int(heads), depth])


class MiMoV2Attention(Module, Shardable):
    """Attention for one MiMo-V2 decoder layer.

    ``qkv_proj`` is FP8 in Xiaomi's chunk order: each chunk holds a slice of
    the query heads, the KV heads they read, and zero rows up to a whole FP8
    block. The pad rows' outputs are dropped here. Q and K are ``head_dim``
    wide and V is ``v_head_dim`` wide; each is zero-padded to the KV cache
    head dim after the projection, which leaves ``QK^T`` unchanged, and the
    output is cut back to ``v_head_dim`` before ``o_proj``. RoPE rotates the
    leading ``rope.head_dim`` dims, NeoX style.
    """

    attention_sink_bias: Weight | None = None

    def __init__(
        self,
        *,
        layout: QkvChunkLayout,
        head_dim: int,
        v_head_dim: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        rope: RotaryEmbedding,
        scale: float,
        value_scale: float,
        sliding_window: int | None,
        has_sinks: bool,
        quant_config: QuantConfig,
        devices: list[DeviceRef],
        is_sharding: bool = False,
    ) -> None:
        """Initializes the layer.

        Args:
            layout: The ``qkv_proj`` chunk layout this module holds.
            head_dim: The Q and K head dim.
            v_head_dim: The V head dim.
            hidden_size: The model width.
            kv_params: The KV cache group this layer writes.
            layer_idx: The layer's index within its KV group.
            rope: The rotary embedding of the layer's group.
            scale: The softmax scale.
            value_scale: The factor V is multiplied by before the cache write.
            sliding_window: Keys attended to, the query included; ``None``
                for full attention.
            has_sinks: Whether the layer has a learned per-head sink logit.
            quant_config: The dense FP8 config of ``qkv_proj``.
            devices: The devices; only the first is used by this instance.
            is_sharding: Whether :meth:`shard` is building this instance and
                will assign its weights.
        """
        super().__init__()
        self.layout = layout
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_size = hidden_size
        self.kv_params = kv_params
        self.layer_idx = layer_idx
        self.rope = rope
        self.scale = scale
        self.value_scale = value_scale
        self.sliding_window = sliding_window
        self.has_sinks = has_sinks
        self.quant_config = quant_config
        self.devices = devices
        self.n_heads = layout.chunks * layout.q_rows // head_dim
        self.n_kv_heads = layout.chunks * layout.k_rows // head_dim
        self._sharding_strategy: ShardingStrategy | None = None
        if is_sharding:
            return
        self.qkv_proj = Linear(
            in_dim=hidden_size,
            out_dim=layout.chunks * layout.padded_rows,
            dtype=DType.float8_e4m3fn,
            device=devices[0],
            quant_config=quant_config,
        )
        self.o_proj = Linear(
            in_dim=self.n_heads * v_head_dim,
            out_dim=hidden_size,
            dtype=DType.bfloat16,
            device=devices[0],
        )
        if has_sinks:
            self.attention_sink_bias = Weight(
                "attention_sink_bias",
                DType.bfloat16,
                [self.n_heads],
                device=devices[0],
            )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        layout = self.layout
        tokens = x.shape[0]
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        qkv = ops.reshape(
            self.qkv_proj(x),
            [tokens, layout.chunks, layout.padded_rows],
        )
        k_start = layout.q_rows
        v_start = k_start + layout.k_rows
        # Q and K are split into heads together, so a chunk that is not a
        # whole number of heads (a sliding one) costs one copy, not two.
        qk = _split_heads(qkv[:, :, :v_start], self.head_dim)
        q = qk[:, :, : k_start // self.head_dim]
        k = qk[:, :, k_start // self.head_dim :]
        v = _split_heads(qkv[:, :, v_start : layout.rows], self.v_head_dim)
        # Scaled in float32 and rounded once, as a reference multiplying a
        # BF16 tensor by a Python float does. The cache holds scaled V.
        v = ops.cast(ops.cast(v, DType.float32) * self.value_scale, v.dtype)

        depth = self.kv_params.head_dim
        q = _pad_heads(q, depth)
        store_k_cache_ragged(
            kv_collection, _pad_heads(k, depth), input_row_offsets, layer_idx
        )
        store_v_cache_ragged(
            kv_collection, _pad_heads(v, depth), input_row_offsets, layer_idx
        )
        freqs_cis = ops.cast(self.rope.freqs_cis, q.dtype).to(q.device)
        q = fused_qk_ragged_rope(
            self.kv_params,
            q,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn = flash_attention_ragged(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=(
                MHAMaskVariant.CAUSAL_MASK
                if self.sliding_window is None
                else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            ),
            scale=self.scale,
            local_window_size=self.sliding_window or -1,
            sink_weights=self.attention_sink_bias,
        )
        attn = ops.reshape(
            attn[:, :, : self.v_head_dim],
            [tokens, self.n_heads * self.v_head_dim],
        )
        return self.o_proj(attn)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """The attention sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "MiMoV2Attention only supports tensor parallelism."
            )
        n = strategy.num_devices
        if self.layout.chunks % n:
            raise ValueError(
                f"MiMoV2Attention: {n} devices do not divide the "
                f"{self.layout.chunks} qkv_proj chunks."
            )
        # Rows split on chunk boundaries, so each device gets whole chunks and
        # whole FP8 blocks, and its heads in order.
        self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(n)
        self.o_proj.sharding_strategy = ShardingStrategy.head_aware_columnwise(
            n, self.n_heads, self.v_head_dim
        )
        if self.attention_sink_bias is not None:
            self.attention_sink_bias.sharding_strategy = (
                ShardingStrategy.rowwise(n)
            )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[MiMoV2Attention]:
        """Returns one attention module per device, each with whole chunks."""
        if self._sharding_strategy is None:
            raise ValueError("MiMoV2Attention has no sharding strategy.")
        devices = list(devices)
        layout = dataclasses.replace(
            self.layout, chunks=self.layout.chunks // len(devices)
        )
        qkv_shards = self.qkv_proj.shard(devices)
        o_shards = self.o_proj.shard(devices)
        sink_shards = (
            self.attention_sink_bias.shard(devices)
            if self.attention_sink_bias is not None
            else None
        )
        shards = []
        for i, device in enumerate(devices):
            shard = MiMoV2Attention(
                layout=layout,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                rope=self.rope,
                scale=self.scale,
                value_scale=self.value_scale,
                sliding_window=self.sliding_window,
                has_sinks=self.has_sinks,
                quant_config=self.quant_config,
                devices=[device],
                is_sharding=True,
            )
            shard.qkv_proj = qkv_shards[i]
            shard.o_proj = o_shards[i]
            if sink_shards is not None:
                shard.attention_sink_bias = sink_shards[i]
            shards.append(shard)
        return shards
