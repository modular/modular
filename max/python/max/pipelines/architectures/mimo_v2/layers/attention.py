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
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_chunked_qkv_rope_store_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear, linear
from max.nn.quant_config import QuantConfig

from ..quant import FP8_BLOCK


@dataclass(frozen=True)
class QkvChunkLayout:
    """Row layout of one layer's fused ``qkv_proj`` in Xiaomi's chunk order.

    The weight is ``chunks`` repeats of ``[q | k | v]``: chunk ``r`` holds the
    ``r``-th equal slice of the query heads and of the KV heads, the rows one
    rank of the tensor-parallel run that saved the checkpoint held. Each
    chunk's FP8 block scales cover it padded to a whole 128-row block.
    """

    chunks: int
    q_rows: int
    k_rows: int
    v_rows: int

    @property
    def rows(self) -> int:
        """Rows of real weights per chunk."""
        return self.q_rows + self.k_rows + self.v_rows

    @property
    def padded_rows(self) -> int:
        """Rows per chunk on the FP8 block-scale grid."""
        return -(-self.rows // FP8_BLOCK) * FP8_BLOCK


class ChunkedQkvProj(Module, Shardable):
    """``qkv_proj`` as the checkpoint stores it, padded per chunk in the graph.

    The checkpoint's FP8 scales cover each chunk padded to whole 128-row
    blocks, but its weight holds only the real rows, and the block-scaled
    matmul reads row ``r``'s scale at ``r // 128``. The pad depends on the
    weight alone, so the graph compiler runs it once, at model init. The pad
    rows' outputs are zero.
    """

    weight: Weight
    weight_scale: Weight

    def __init__(
        self,
        layout: QkvChunkLayout,
        hidden_size: int,
        quant_config: QuantConfig,
        device: DeviceRef,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the projection.

        Args:
            layout: The chunk layout this module holds.
            hidden_size: The model width.
            quant_config: The dense FP8 block-scaled config.
            device: The device the projection runs on.
            is_sharding: Whether :meth:`shard` is building this instance and
                will assign its weights.
        """
        super().__init__()
        self.layout = layout
        self.hidden_size = hidden_size
        self.quant_config = quant_config
        if is_sharding:
            return
        self.weight = Weight(
            "weight",
            DType.float8_e4m3fn,
            [layout.chunks * layout.rows, hidden_size],
            device=device,
        )
        self.weight_scale = Weight(
            "weight_scale",
            quant_config.weight_scale.dtype,
            [
                layout.chunks * layout.padded_rows // FP8_BLOCK,
                hidden_size // FP8_BLOCK,
            ],
            device=DeviceRef.CPU(),
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Returns ``[tokens, chunks * padded_rows]``, the pad columns zero."""
        layout = self.layout
        weight: TensorValue = self.weight.to(x.device)
        if layout.padded_rows != layout.rows:
            chunks = weight.reshape(
                [layout.chunks, layout.rows, self.hidden_size]
            )
            chunks = ops.pad(
                chunks, [0, 0, 0, layout.padded_rows - layout.rows, 0, 0]
            )
            weight = chunks.reshape(
                [layout.chunks * layout.padded_rows, self.hidden_size]
            )
        return linear(
            x,
            weight,
            quant_config=self.quant_config,
            weight_scale=self.weight_scale,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """The weight sharding strategy."""
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_rowwise:
            raise ValueError("ChunkedQkvProj only shards by rows.")
        if self.layout.chunks % strategy.num_devices:
            raise ValueError(
                f"ChunkedQkvProj: {strategy.num_devices} devices do not "
                f"divide the {self.layout.chunks} qkv_proj chunks."
            )
        # An even row split of either tensor is whole chunks per device.
        self.weight.sharding_strategy = strategy
        self.weight_scale.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[ChunkedQkvProj]:
        """Returns one projection per device, each with whole chunks."""
        devices = list(devices)
        layout = dataclasses.replace(
            self.layout, chunks=self.layout.chunks // len(devices)
        )
        shards = []
        for device, weight, scale in zip(
            devices,
            self.weight.shard(devices),
            self.weight_scale.shard(devices),
            strict=True,
        ):
            shard = ChunkedQkvProj(
                layout,
                self.hidden_size,
                self.quant_config,
                device,
                is_sharding=True,
            )
            shard.weight = weight
            shard.weight_scale = scale
            shards.append(shard)
        return shards


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


def prepare_qkv_unfused(
    qkv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    value_scale: float,
    *,
    kv_params: KVCacheParams,
    layout: QkvChunkLayout,
    head_dim: int,
    v_head_dim: int,
) -> TensorValue:
    """Prepares chunked queries and stores rotated keys and scaled values.

    This is the default path and the reference for the fused operator.
    The projection includes FP8 block padding after each chunk's real rows.

    Args:
        qkv: Projection with shape ``[tokens, chunks, padded_chunk_rows]``.
        input_row_offsets: Ragged sequence boundaries on the projection device.
        freqs_cis: Partial NeoX RoPE table on the projection device.
        kv_collection: Paged cache receiving K and V for the current layer.
        layer_idx: CPU scalar selecting the layer within its cache group.
        value_scale: FP32 multiplier applied before V is rounded to BF16.
        kv_params: Parameters of the layer's cache group.
        layout: Chunk counts and Q/K/V row extents on this device.
        head_dim: Unpadded query and key width.
        v_head_dim: Unpadded value width.

    Returns:
        Rotated queries padded to the cache head width.
    """
    assert qkv.shape[1] == layout.chunks
    k_start = layout.q_rows
    v_start = k_start + layout.k_rows
    # Q and K are split into heads together, so a chunk that is not a
    # whole number of heads (a sliding one) costs one copy, not two.
    qk = _split_heads(qkv[:, :, :v_start], head_dim)
    q = qk[:, :, : k_start // head_dim]
    k = qk[:, :, k_start // head_dim :]
    v = _split_heads(qkv[:, :, v_start : layout.rows], v_head_dim)
    # Scaled in float32 and rounded once, as a reference multiplying a
    # BF16 tensor by a Python float does. The cache holds scaled V.
    v = ops.cast(ops.cast(v, DType.float32) * value_scale, v.dtype)

    depth = kv_params.head_dim
    q = _pad_heads(q, depth)
    store_k_cache_ragged(
        kv_collection,
        _pad_heads(k, depth),
        input_row_offsets,
        layer_idx,
    )
    store_v_cache_ragged(
        kv_collection,
        _pad_heads(v, depth),
        input_row_offsets,
        layer_idx,
    )
    return fused_qk_ragged_rope(
        kv_params,
        q,
        input_row_offsets,
        kv_collection,
        freqs_cis,
        layer_idx,
        interleaved=False,
    )


class MiMoV2Attention(Module, Shardable):
    """Attention for one MiMo-V2 decoder layer.

    ``qkv_proj`` is FP8 in Xiaomi's chunk order: each chunk holds a slice of
    the query heads and the KV heads they read (see :class:`ChunkedQkvProj`).
    The pad rows' outputs are dropped here. Q and K are ``head_dim`` wide and
    V is ``v_head_dim`` wide; each is zero-padded to the KV cache head dim
    after the projection, which leaves ``QK^T`` unchanged, and the output is
    cut back to ``v_head_dim`` before ``o_proj``. RoPE rotates the leading
    dims that ``freqs_cis`` covers, NeoX style.

    The layer's KV-group index and ``freqs_cis`` are call arguments, so that
    every layer of a group can share one compiled subgraph.
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
        scale: float,
        value_scale: float,
        sliding_window: int | None,
        has_sinks: bool,
        quant_config: QuantConfig,
        devices: list[DeviceRef],
        fused_sliding_qkv: bool = False,
        fused_full_qkv: bool = False,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the layer.

        Args:
            layout: The ``qkv_proj`` chunk layout this module holds.
            head_dim: The Q and K head dim.
            v_head_dim: The V head dim.
            hidden_size: The model width.
            kv_params: The KV cache group this layer writes.
            scale: The softmax scale.
            value_scale: The factor V is multiplied by before the cache write.
            sliding_window: Keys attended to, the query included; ``None``
                for full attention.
            has_sinks: Whether the layer has a learned per-head sink logit.
            quant_config: The dense FP8 config of ``qkv_proj``.
            devices: The devices; only the first is used by this instance.
            fused_sliding_qkv: Whether sliding layers use the fused QKV
                preparation op, which raises on an unsupported layout.
                Defaults to ``False``.
            fused_full_qkv: Whether full layers use the fused QKV preparation
                op, which raises on an unsupported layout. Defaults to ``False``.
            is_sharding: Whether :meth:`shard` is building this instance and
                will assign its weights.
        """
        super().__init__()
        self.fused_sliding_qkv = fused_sliding_qkv
        self.fused_full_qkv = fused_full_qkv
        self.layout = layout
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_size = hidden_size
        self.kv_params = kv_params
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
        self.qkv_proj = ChunkedQkvProj(
            layout, hidden_size, quant_config, devices[0]
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
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the layer.

        Args:
            layer_idx: The layer's index within its KV group, a ``uint32``
                scalar on the host.
            x: The ``[tokens, hidden]`` input.
            kv_collection: The layer's KV group on this device.
            freqs_cis: The group's RoPE table on this device, in ``x.dtype``.
            input_row_offsets: Each sequence's first row in ``x``.

        Returns:
            This device's partial ``o_proj`` output.
        """
        layout = self.layout
        tokens = x.shape[0]

        qkv = ops.reshape(
            self.qkv_proj(x),
            [tokens, layout.chunks, layout.padded_rows],
        )
        fused_qkv = (
            self.fused_full_qkv
            if self.sliding_window is None
            else self.fused_sliding_qkv
        )
        if fused_qkv:
            q = fused_chunked_qkv_rope_store_ragged(
                qkv,
                input_row_offsets,
                freqs_cis,
                kv_collection,
                layer_idx,
                self.value_scale,
                num_q_heads=self.n_heads,
                qk_head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
                chunk_padding=(
                    layout.padded_rows - layout.rows
                    if self.sliding_window is None
                    else 0
                ),
            )
        else:
            q = prepare_qkv_unfused(
                qkv,
                input_row_offsets,
                freqs_cis,
                kv_collection,
                layer_idx,
                self.value_scale,
                kv_params=self.kv_params,
                layout=layout,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
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
        # Each device gets whole chunks, so its heads in order.
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
                scale=self.scale,
                value_scale=self.value_scale,
                sliding_window=self.sliding_window,
                has_sinks=self.has_sinks,
                quant_config=self.quant_config,
                devices=[device],
                fused_sliding_qkv=self.fused_sliding_qkv,
                fused_full_qkv=self.fused_full_qkv,
                is_sharding=True,
            )
            shard.qkv_proj = qkv_shards[i]
            shard.o_proj = o_shards[i]
            if sink_shards is not None:
                shard.attention_sink_bias = sink_shards[i]
            shards.append(shard)
        return shards
