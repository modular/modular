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
"""Inkling2 attention: qkvr, L2 clips, K/V short convs, relative bias, sinks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorValue, ops
from max.nn import Module
from max.nn._graph_ops import store_k_cache_ragged, store_v_cache_ragged
from max.nn._graph_ops.state_space import short_conv_ring
from max.nn.common_layers.kv_cache import PagedCacheValues
from max.nn.common_layers.linear import RowParallelLinear, col_parallel
from max.nn.kv_cache import KVCacheParams
from max.nn.kv_cache import PagedCacheValues as PagedCacheGraphValues
from max.nn.linear import Linear
from max.tensor import Tensor

from ..model_config import InklingTextConfig
from .placement import as_sharded, shard_axis

_FP32 = DType.float32

_EPS_DENOM = 1e-6
"""Added to the L2 norm before dividing (:147, :162 of the reference)."""

_SCALE_CAP = 1.0
"""``clamp(max=1.0)``: a vector already inside the threshold keeps scale 1."""

_MASKED_LOGIT = -1e30
"""Additive stand-in for ``-inf`` on a masked logit."""

_QUERY_CHUNK = 32
"""Query rows one attention pass carries."""


def _l2_clip_fp32(x: TensorValue, threshold: float) -> TensorValue:
    norm = ops.sqrt(ops.sum(x * x, axis=-1))
    return x * ops.min(threshold / (norm + _EPS_DENOM), _SCALE_CAP)


def _scaled_l2_clip_fp32(
    x: TensorValue, scale: TensorValue, threshold: float
) -> TensorValue:
    return _l2_clip_fp32(x * ops.cast(scale, _FP32), threshold)


def _unweighted_rms_norm_fp32(x: TensorValue, eps: float) -> TensorValue:
    return x * ops.rsqrt(ops.mean(x * x, axis=-1) + eps)


def _token_index(total_tokens: Dim | int, device: DeviceRef) -> TensorValue:
    ones = ops.broadcast_to(
        ops.constant(1, DType.int64, device=device), [total_tokens]
    )
    return ops.cumsum(ones, axis=0) - 1


def _sequence_ids(
    input_row_offsets: TensorValue, total_tokens: Dim | int
) -> TensorValue:
    offsets = ops.cast(input_row_offsets, DType.int64)
    index = _token_index(total_tokens, offsets.device)
    started = ops.cast(
        ops.reshape(offsets, [1, offsets.shape[0]])
        <= ops.reshape(index, [index.shape[0], 1]),
        DType.int64,
    )
    return ops.sum(started, axis=1) - 1


def _sequence_starts(
    input_row_offsets: TensorValue, total_tokens: Dim | int
) -> TensorValue:
    return ops.gather(
        ops.cast(input_row_offsets, DType.int64),
        _sequence_ids(input_row_offsets, total_tokens),
        axis=0,
    )


def _short_conv_static(
    x: TensorValue,
    taps: TensorValue,
    conv_ring: BufferValue,
    conv_rows: TensorValue,
    layer_row: TensorValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    *,
    kernel_size: int,
    commit: bool,
) -> TensorValue:
    convolved = short_conv_ring(
        ops.cast(x, _FP32),
        ops.cast(taps, _FP32),
        conv_ring,
        input_row_offsets,
        positions,
        conv_rows,
        layer_row,
        commit=commit,
    )
    return ops.cast(convolved * (1.0 / math.sqrt(kernel_size + 1)), x.dtype)


def _short_conv_dynamic(
    x: TensorValue,
    dynamic_input: TensorValue,
    static_w: TensorValue,
    weight_proj: TensorValue,
    conv_ring: BufferValue,
    conv_rows: TensorValue,
    layer_row: TensorValue,
    input_row_offsets: TensorValue,
    positions: TensorValue,
    *,
    channels: int,
    kv_heads: int,
    head_dim: int,
    kernel_size: int,
    commit: bool,
) -> TensorValue:
    total_tokens = x.shape[0]
    device = x.device
    x32 = ops.cast(x, _FP32)

    # The ring as it stood before this chunk: this layer's slot per sequence,
    # then that slot's index for every (token, lag) the chunk needs.
    ring = ops.buffer_load(conv_ring)
    ring_len = int(ring.shape[1])
    slots = ops.gather(
        ops.reshape(
            ops.gather(
                ops.cast(conv_rows, DType.int64),
                ops.reshape(ops.cast(layer_row, DType.int64).to(device), [1]),
                axis=0,
            ),
            [conv_rows.shape[1]],
        ),
        _sequence_ids(input_row_offsets, total_tokens),
        axis=0,
    )

    index = _token_index(total_tokens, device)
    lags = ops.constant(list(range(kernel_size)), DType.int64, device=device)
    position = ops.reshape(ops.cast(positions, DType.int64), [total_tokens, 1])
    source = position - ops.reshape(lags, [1, kernel_size])
    row = ops.reshape(index, [total_tokens, 1]) - ops.reshape(
        lags, [1, kernel_size]
    )
    starts = _sequence_starts(input_row_offsets, total_tokens)
    from_chunk = ops.gather(x32, ops.max(row, 0), axis=0)
    from_ring = ops.gather(
        ops.reshape(ring, [ring.shape[0] * ring_len, channels]),
        ops.reshape(slots, [total_tokens, 1]) * ring_len
        + (ops.max(source, 0) % ring_len),
        axis=0,
    )
    windows = ops.where(
        ops.reshape(
            row >= ops.reshape(starts, [total_tokens, 1]),
            [total_tokens, kernel_size, 1],
        ),
        from_chunk,
        from_ring,
    )
    windows = ops.where(
        ops.reshape(source >= 0, [total_tokens, kernel_size, 1]),
        windows,
        ops.constant(0.0, _FP32, device=device),
    )

    # Commit this chunk's last ``ring_len`` rows for the next step. Zero taps
    # leave the kernel's own convolution term at zero, so the call only writes.
    short_conv_ring(
        x32,
        ops.broadcast_to(
            ops.constant(0.0, _FP32, device=device), [channels, kernel_size]
        ),
        conv_ring,
        input_row_offsets,
        positions,
        conv_rows,
        layer_row,
        commit=commit,
    )

    # Per-token taps: [total, heads * k] -> [total, heads, 1, k].
    dynamic = ops.reshape(
        ops.matmul(
            ops.cast(dynamic_input, _FP32),
            ops.permute(ops.cast(weight_proj, _FP32), [1, 0]),
        ),
        [total_tokens, kv_heads, 1, kernel_size],
    )
    # Static taps: [k, channels] -> [heads, head_dim, k].
    static = ops.permute(
        ops.reshape(
            ops.cast(static_w, _FP32), [kernel_size, kv_heads, head_dim]
        ),
        [1, 2, 0],
    )
    weights = ops.broadcast_to(
        static, [total_tokens, kv_heads, head_dim, kernel_size]
    ) + ops.broadcast_to(
        dynamic, [total_tokens, kv_heads, head_dim, kernel_size]
    )

    windows = ops.permute(
        ops.reshape(windows, [total_tokens, kernel_size, kv_heads, head_dim]),
        [0, 2, 3, 1],
    )
    convolved = ops.reshape(
        ops.sum(windows * weights, axis=-1),
        [total_tokens, channels],
    )
    return ops.cast(
        (convolved + x32) * (1.0 / math.sqrt(2 * kernel_size + 1)), x.dtype
    )


def _head_major(x: TensorValue, *, kv_heads: int, group: int) -> TensorValue:
    total, head_dim = x.shape[0], x.shape[2]
    kv_major = ops.permute(x, [1, 0, 2])
    wide = ops.broadcast_to(
        ops.reshape(kv_major, [kv_heads, 1, total, head_dim]),
        [kv_heads, group, total, head_dim],
    )
    return ops.reshape(wide, [kv_heads * group, total, head_dim])


def _cache_plane(
    kv_collection: PagedCacheGraphValues, plane: int, row: int
) -> TensorValue:
    blocks = kv_collection.kv_blocks
    if kv_collection.kv_scales is not None:
        raise ValueError(
            "the cached-prefix read addresses packed cache pages, and a "
            "quantized cache pads them; this attention needs an unquantized "
            "KV cache"
        )
    if int(blocks.shape[1]) != 2:
        raise ValueError(
            "expected a cache holding keys and values side by side, got "
            f"{int(blocks.shape[1])} planes"
        )
    if not 0 <= row < int(blocks.shape[2]):
        raise ValueError(
            f"the KV cache declares {int(blocks.shape[2])} layer rows and "
            f"this layer asked for row {row}"
        )
    return ops.slice_tensor(
        ops.buffer_load(blocks), [slice(None), plane, row, Ellipsis]
    )


def _screen(x: TensorValue, last: TensorValue) -> TensorValue:
    return ops.min(ops.max(x, 0), last)


@dataclass(frozen=True)
class _DeviceAttention:
    kv_params: KVCacheParams
    out_dims: tuple[int, int, int, int]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    scale: float
    qk_clip_threshold: float
    rel_clip_threshold: float
    rms_norm_eps: float
    sconv_kernel_size: int
    local_window_size: int | None
    applies_log_scaling: bool
    static_conv: bool
    commit: bool
    key_slot: int
    """Tokens per cache page; the prefix window is page-aligned on it."""
    key_span: int
    """Key positions the cached-prefix read materializes per query token.

    Page-aligned, and large enough for every key the mask can keep: the
    sliding window (plus the page it starts inside) on a local layer, the
    serving context on a global one.
    """
    query_rows: int
    """Query rows one bounded pass carries, rounded up from the budget."""


def _distance_bias(
    rel_table: TensorValue, distance: TensorValue, *, block: _DeviceAttention
) -> tuple[TensorValue, TensorValue]:
    total_tokens, keys = distance.shape[0], distance.shape[1]
    device = distance.device
    heads = block.num_heads
    rel_extent = block.rel_extent
    index = ops.min(ops.max(distance, 0), rel_extent - 1)
    heads_axis = ops.broadcast_to(
        ops.reshape(
            ops.constant(list(range(heads)), DType.int64, device=device),
            [1, heads],
        ),
        [total_tokens, heads],
    )
    row = ops.reshape(_token_index(total_tokens, device), [total_tokens, 1])
    base = (row * heads + heads_axis) * rel_extent
    lookup = ops.reshape(base, [total_tokens, heads, 1]) + ops.broadcast_to(
        ops.reshape(index, [total_tokens, 1, keys]),
        [total_tokens, heads, keys],
    )
    bias = ops.reshape(
        ops.gather(
            ops.reshape(rel_table, [total_tokens * heads * rel_extent]),
            ops.reshape(lookup, [total_tokens * heads, keys]),
            axis=0,
        ),
        [total_tokens, heads, keys],
    )
    in_extent = ops.cast(
        (
            ops.cast(distance >= 0, DType.int32)
            * ops.cast(distance < rel_extent, DType.int32)
        ),
        _FP32,
    )
    return ops.permute(bias, [1, 0, 2]), ops.reshape(
        in_extent, [1, total_tokens, keys]
    )


def _bounded_sink_attention(
    query_states: TensorValue,
    key_states: TensorValue,
    value_states: TensorValue,
    rel_table: TensorValue,
    sink_logits: TensorValue,
    positions: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheGraphValues,
    cache_row: int,
    requests: TensorValue,
    cached_lengths: TensorValue,
    *,
    block: _DeviceAttention,
) -> TensorValue:
    total_tokens = query_states.shape[0]
    device = query_states.device
    dtype = query_states.dtype
    heads, head_dim = block.num_heads, block.head_dim
    kv_heads = block.num_kv_heads
    group = heads // kv_heads
    chunk = _QUERY_CHUNK
    span, slot = block.key_span, block.key_slot
    # Whole pages, so the window is gathered a page at a time: the tile covers
    # ceil(span / slot) pages, the last one partly outside the window (masked
    # below by the distance and the cached length).
    page_span = -(-span // slot)
    keys = page_span * slot
    fp32 = _FP32

    blocks = kv_collection.kv_blocks
    slot_dim = int(blocks.shape[3])
    if int(blocks.shape[4]) != kv_heads:
        raise ValueError(
            f"the KV cache holds {int(blocks.shape[4])} heads per device but "
            f"this layer has {kv_heads}: the cached rows would not line up "
            "with the query heads"
        )
    if slot_dim != slot:
        raise ValueError(
            f"the KV cache page holds {slot_dim} slots but the cached-prefix "
            f"window is page-aligned to {slot}; the window arithmetic would "
            "address the wrong slot"
        )

    # Shared by every chunk: the batch's own keys and values, the batch's
    # start positions, and the page walk of the cache leaf.
    q_all = ops.permute(ops.cast(query_states, fp32), [1, 0, 2])
    k_all = _head_major(
        ops.cast(key_states, fp32), kv_heads=kv_heads, group=group
    )
    v_all = _head_major(value_states, kv_heads=kv_heads, group=group)
    sink_all = ops.reshape(
        ops.permute(ops.cast(sink_logits, fp32), [1, 0]), [heads, total_tokens]
    )
    starts = ops.reshape(
        _sequence_starts(input_row_offsets, total_tokens), [total_tokens]
    )
    index_pos = _token_index(total_tokens, device)
    last_row = ops.max(index_pos, axis=0)
    offset_all = ops.cast(positions, DType.int64)
    requests = ops.reshape(requests, [total_tokens])
    lengths_all = ops.reshape(
        ops.cast(cached_lengths, DType.int64), [total_tokens]
    )
    key_pages = _cache_plane(kv_collection, 0, cache_row)
    value_pages = _cache_plane(kv_collection, 1, cache_row)
    # The page table is [requests, max_num_pages]; its column count is
    # algebraic, so the clip value comes from an index ramp, not the shape.
    page_table = kv_collection.lookup_table
    pages = page_table.shape[1]
    page_last = ops.max(_token_index(pages, device), axis=0)
    page_table = ops.reshape(
        ops.cast(page_table, DType.int64), [page_table.shape[0] * pages]
    )
    # Static tables: the page columns of the window, and the key offset of
    # each tile slot from the window's first page.
    column_at = ops.constant(list(range(page_span)), DType.int64, device=device)
    keys_at = ops.constant(list(range(keys)), DType.int64, device=device)
    row_offsets = ops.constant(list(range(chunk)), DType.int64, device=device)
    zero_kv = ops.constant(0.0, dtype, device=device)
    masked_logit = ops.constant(_MASKED_LOGIT, fp32, device=device)
    zero_score = ops.constant(0.0, fp32, device=device)

    chunks = []
    for first_row in range(0, block.query_rows, chunk):
        # A tail chunk repeats the last row; its results are dropped by the
        # final row select, and the repeat keeps every shape static.
        rows = ops.min(row_offsets + first_row, last_row)
        q = ops.gather(q_all, rows, axis=1)
        sink = ops.reshape(
            ops.gather(sink_all, rows, axis=1), [heads, chunk, 1]
        )
        rel = ops.gather(rel_table, rows, axis=0)
        pos = ops.gather(offset_all, rows, axis=0)
        lengths = ops.gather(lengths_all, rows, axis=0)
        row_starts = ops.gather(starts, rows, axis=0)
        req = ops.gather(requests, rows, axis=0)

        # This step's own keys: causality, the sequence boundary, the window.
        distance = ops.reshape(pos, [chunk, 1]) - ops.reshape(
            offset_all, [1, total_tokens]
        )
        scores = ops.matmul(q, ops.permute(k_all, [0, 2, 1])) * block.scale
        bias, extent = _distance_bias(rel, distance, block=block)
        scores = scores + bias * extent
        visible = ops.cast(distance >= 0, DType.int32) * ops.cast(
            ops.reshape(row_starts, [chunk, 1])
            <= ops.reshape(index_pos, [1, total_tokens]),
            DType.int32,
        )
        if block.local_window_size is not None:
            visible = visible * ops.cast(
                distance < block.local_window_size, DType.int32
            )
        scores = scores + ops.broadcast_to(
            ops.where(
                ops.reshape(visible > 0, [1, chunk, total_tokens]),
                zero_score,
                masked_logit,
            ),
            [heads, chunk, total_tokens],
        )

        # The cached prefix, over the window its mask can still reach.
        if block.local_window_size is None:
            first_page = ops.broadcast_to(
                ops.constant(0, DType.int64, device=device), [chunk]
            )
        else:
            first_page = ops.max(pos - (block.local_window_size - 1), 0) // slot
        key_pos = ops.reshape(first_page * slot, [chunk, 1]) + ops.reshape(
            keys_at, [1, keys]
        )
        # A masked lane can name a column the request never allocated: clip it
        # into the table, and the zeroing below drops it.
        pages_at = ops.gather(
            page_table,
            ops.reshape(req, [chunk, 1]) * (page_last + 1)
            + _screen(
                ops.reshape(first_page, [chunk, 1])
                + ops.reshape(column_at, [1, page_span]),
                page_last,
            ),
            axis=0,
        )
        inside = key_pos < ops.reshape(lengths, [chunk, 1])
        live = ops.broadcast_to(
            ops.reshape(inside, [chunk, keys, 1, 1]),
            [chunk, keys, kv_heads, head_dim],
        )
        # The cached rows are read twice (a key pass and a value pass) and the
        # two passes are kept apart on purpose: the gathered rows are the
        # largest tensor in the graph, so only one of them is ever live.
        cached_keys = ops.where(
            live,
            ops.reshape(
                ops.gather(key_pages, pages_at, axis=0),
                [chunk, keys, kv_heads, head_dim],
            ),
            zero_kv,
        )

        # One query row and one kv head per matmul batch: the group expansion
        # is the matmul's shape, never a materialized
        # ``[heads, chunk, keys, head_dim]`` copy of the cached rows.
        prefix_scores = (
            ops.matmul(
                ops.reshape(
                    ops.permute(
                        ops.reshape(q, [kv_heads, group, chunk, head_dim]),
                        [0, 2, 1, 3],
                    ),
                    [kv_heads * chunk, group, head_dim],
                ),
                ops.permute(
                    ops.reshape(
                        ops.permute(ops.cast(cached_keys, fp32), [2, 0, 1, 3]),
                        [kv_heads * chunk, keys, head_dim],
                    ),
                    [0, 2, 1],
                ),
            )
            * block.scale
        )
        prefix_scores = ops.reshape(
            ops.permute(
                ops.reshape(prefix_scores, [kv_heads, chunk, group, keys]),
                [0, 2, 1, 3],
            ),
            [heads, chunk, keys],
        )
        prefix_distance = ops.reshape(pos, [chunk, 1]) - key_pos
        prefix_bias, prefix_extent = _distance_bias(
            rel, prefix_distance, block=block
        )
        prefix_scores = prefix_scores + prefix_bias * prefix_extent
        prefix_visible = ops.cast(inside, DType.int32) * ops.cast(
            prefix_distance >= 0, DType.int32
        )
        if block.local_window_size is not None:
            prefix_visible = prefix_visible * ops.cast(
                prefix_distance < block.local_window_size, DType.int32
            )
        prefix_scores = ops.where(
            ops.broadcast_to(
                ops.reshape(prefix_visible > 0, [1, chunk, keys]),
                [heads, chunk, keys],
            ),
            prefix_scores,
            masked_logit,
        )

        # One row maximum over both ranges and the sink, which is the same
        # softmax as the reference's concatenated logits (:391-392).
        row_max = ops.max(ops.max(scores, axis=-1), sink)
        row_max = ops.max(row_max, ops.max(prefix_scores, axis=-1))
        shifted = ops.exp(scores - row_max)
        prefix_shifted = ops.exp(prefix_scores - row_max)
        denominator = (
            ops.sum(shifted, axis=-1)
            + ops.exp(sink - row_max)
            + ops.sum(prefix_shifted, axis=-1)
        )
        # The reference casts the probabilities to the query dtype before the
        # value matmul (:392).
        probabilities = ops.cast(shifted / denominator, dtype)
        out = ops.matmul(probabilities, v_all)
        prefix_probabilities = ops.cast(prefix_shifted / denominator, dtype)
        cached_values = ops.where(
            live,
            ops.reshape(
                ops.gather(value_pages, pages_at, axis=0),
                [chunk, keys, kv_heads, head_dim],
            ),
            zero_kv,
        )
        out = out + ops.reshape(
            ops.permute(
                ops.reshape(
                    ops.matmul(
                        ops.reshape(
                            ops.permute(
                                ops.reshape(
                                    prefix_probabilities,
                                    [kv_heads, group, chunk, keys],
                                ),
                                [0, 2, 1, 3],
                            ),
                            [kv_heads * chunk, group, keys],
                        ),
                        ops.reshape(
                            ops.permute(
                                ops.cast(cached_values, dtype), [2, 0, 1, 3]
                            ),
                            [kv_heads * chunk, keys, head_dim],
                        ),
                    ),
                    [kv_heads, chunk, group, head_dim],
                ),
                [0, 2, 1, 3],
            ),
            [heads, chunk, head_dim],
        )
        chunks.append(out)

    joined = chunks[0] if len(chunks) == 1 else ops.concat(chunks, axis=1)
    out = ops.gather(joined, index_pos, axis=1)
    return ops.reshape(
        ops.permute(out, [1, 0, 2]), [total_tokens, heads * head_dim]
    )


def _attention_on_device(
    query_states: TensorValue,
    key_states: TensorValue,
    value_states: TensorValue,
    relative_states: TensorValue,
    key_side: TensorValue,
    value_side: TensorValue,
    gate: TensorValue,
    sink_logits: TensorValue,
    positions: TensorValue,
    log_scaling: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheGraphValues,
    q_scale: TensorValue,
    rel_scale: TensorValue,
    k_conv_taps: TensorValue,
    k_conv_proj: TensorValue,
    v_conv_taps: TensorValue,
    v_conv_proj: TensorValue,
    rel_logits_proj: TensorValue,
    k_conv_ring: BufferValue,
    v_conv_ring: BufferValue,
    k_conv_rows: TensorValue,
    v_conv_rows: TensorValue,
    k_layer_row: TensorValue,
    v_layer_row: TensorValue,
    cache_layer_idx: TensorValue,
    cache_row: int,
    *,
    block: _DeviceAttention,
) -> TensorValue:
    q_dim, k_dim, v_dim, _ = block.out_dims
    total_tokens = query_states.shape[0]
    heads_shape: list[int | Dim] = [
        total_tokens,
        block.num_heads,
        block.head_dim,
    ]
    kv_heads_shape: list[int | Dim] = [
        total_tokens,
        block.num_kv_heads,
        block.head_dim,
    ]

    # conv_idx 0 and 1 (:424-427 sliding, :522-537 full).
    if block.static_conv:
        key = _short_conv_static(
            key_states,
            k_conv_taps,
            k_conv_ring,
            k_conv_rows,
            k_layer_row,
            input_row_offsets,
            positions,
            kernel_size=block.sconv_kernel_size,
            commit=block.commit,
        )
        value = _short_conv_static(
            value_states,
            v_conv_taps,
            v_conv_ring,
            v_conv_rows,
            v_layer_row,
            input_row_offsets,
            positions,
            kernel_size=block.sconv_kernel_size,
            commit=block.commit,
        )
    else:
        key = _short_conv_dynamic(
            key_states,
            key_side,
            k_conv_taps,
            k_conv_proj,
            k_conv_ring,
            k_conv_rows,
            k_layer_row,
            input_row_offsets,
            positions,
            channels=k_dim,
            kv_heads=block.num_kv_heads,
            head_dim=block.head_dim,
            kernel_size=block.sconv_kernel_size,
            commit=block.commit,
        )
        value = _short_conv_dynamic(
            value_states,
            value_side,
            v_conv_taps,
            v_conv_proj,
            v_conv_ring,
            v_conv_rows,
            v_layer_row,
            input_row_offsets,
            positions,
            channels=v_dim,
            kv_heads=block.num_kv_heads,
            head_dim=block.head_dim,
            kernel_size=block.sconv_kernel_size,
            commit=block.commit,
        )

    tau = None
    if block.applies_log_scaling:
        tau = ops.reshape(log_scaling, [total_tokens, 1, 1])

    # q: learned per-head scale, the parameter-free clip, then tau on a full
    # layer only (:465, :595-596).
    query = ops.reshape(query_states, heads_shape)
    query = ops.cast(
        _scaled_l2_clip_fp32(
            ops.cast(query, _FP32), q_scale, block.qk_clip_threshold
        ),
        query_states.dtype,
    )
    if tau is not None:
        query = ops.cast(ops.cast(query, _FP32) * tau, query.dtype)

    # k: the parameter-free clip, no learned scale (:466, :597).
    key = ops.reshape(key, kv_heads_shape)
    key = ops.cast(
        _l2_clip_fp32(ops.cast(key, _FP32), block.qk_clip_threshold),
        key.dtype,
    )

    # v: an unweighted RMS norm over the value head dim, in float32 (:468-469,
    # :599-600).
    value = ops.reshape(value, kv_heads_shape)
    value = ops.cast(
        _unweighted_rms_norm_fp32(ops.cast(value, _FP32), block.rms_norm_eps),
        value.dtype,
    )

    # r: learned per-head scale then clip (:470, :601), then tau before the
    # projection, which is linear in r (:589-602).
    relative = ops.reshape(
        relative_states, [total_tokens, block.num_heads, block.d_rel]
    )
    relative = ops.cast(
        _scaled_l2_clip_fp32(
            ops.cast(relative, _FP32), rel_scale, block.rel_clip_threshold
        ),
        relative_states.dtype,
    )
    if tau is not None:
        relative = ops.cast(ops.cast(relative, _FP32) * tau, relative.dtype)
    rel_table = ops.reshape(
        ops.matmul(
            ops.reshape(
                ops.cast(relative, _FP32),
                [total_tokens * block.num_heads, block.d_rel],
            ),
            ops.cast(rel_logits_proj, _FP32),
        ),
        [total_tokens, block.num_heads, block.rel_extent],
    )

    # Stored at `cache_lengths + token_idx`, so a one-token decode step attends
    # to its whole history rather than to itself (see _bounded_sink_attention).
    store_k_cache_ragged(kv_collection, key, input_row_offsets, cache_layer_idx)
    store_v_cache_ragged(
        kv_collection, value, input_row_offsets, cache_layer_idx
    )
    requests = _sequence_ids(input_row_offsets, total_tokens)
    cached_lengths = ops.gather(
        ops.cast(kv_collection.cache_lengths, DType.int64), requests, axis=0
    )

    attn_out = _bounded_sink_attention(
        query,
        key,
        value,
        rel_table,
        sink_logits,
        positions,
        input_row_offsets,
        kv_collection,
        cache_row,
        requests,
        cached_lengths,
        block=block,
    )

    # The gate multiplies the pre-projection output in float32 and casts back
    # (:496, :628); it is a sigmoid, not a silu.
    attn_out = ops.cast(
        ops.cast(attn_out, _FP32) * ops.sigmoid(ops.cast(gate, _FP32)),
        attn_out.dtype,
    )
    return ops.reshape(attn_out, [total_tokens, q_dim])


_attention = F.functional(_attention_on_device)


class _ScaledL2Clip(Module[[], None]):
    def __init__(self, *, num_heads: int, head_dim: int) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = Tensor.zeros([num_heads, head_dim])


class _RelLogitsProj(Module[[], None]):
    def __init__(self, *, d_rel: int, rel_extent: int) -> None:
        self.proj = Tensor.zeros([d_rel, rel_extent])


class ShortConvolution(Module[[], None]):
    """Weights of a static depthwise causal conv (:247-305)."""

    def __init__(
        self, *, channels: int, kernel_size: int, commit_conv_state: bool = True
    ) -> None:
        self.channels = channels
        self.kernel_size = kernel_size
        self.commit_conv_state = commit_conv_state
        self.weight = Tensor.zeros([channels, 1, kernel_size])

    @property
    def taps(self) -> Tensor:
        """The weight as ``[channels, kernel_size]``."""
        return self.weight.reshape([self.channels, self.kernel_size])


class DynamicShortConvolution(Module[[], None]):
    """Weights of an input-dependent causal conv (:308-353)."""

    def __init__(
        self,
        *,
        channels: int,
        num_heads: int,
        input_size: int,
        kernel_size: int,
    ) -> None:
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.input_size = input_size
        self.kernel_size = kernel_size
        # The checkpoint's own leaves: no `.weight` suffix on disk.
        self.static_w = Tensor.zeros([kernel_size, channels])
        self.weight_proj = Tensor.zeros([num_heads * kernel_size, input_size])


class InklingAttention(Module[..., Tensor]):
    """One decoder layer's attention block, sliding or full."""

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: KVCacheParams,
        num_devices: int,
        max_seq_len: int,
        max_query_rows: int,
        is_local: bool | None = None,
        commit_conv_state: bool = True,
    ) -> None:
        if is_local is None:
            is_local = text_config.is_local_attention(layer_idx)
        num_heads = text_config.num_heads(is_local)
        num_kv_heads = text_config.num_kv_heads(is_local)
        if num_heads % num_devices or num_kv_heads % num_devices:
            raise ValueError(
                f"layer {layer_idx} has {num_heads} query heads and "
                f"{num_kv_heads} KV heads, which do not both divide over "
                f"{num_devices} devices"
            )
        # The head split is only readable on concrete cache types; the base
        # ``KVCacheParams`` reaches here, so check it when it is answerable.
        try:
            cached_per_device: int | None = kv_params.n_kv_heads_per_device
        except (AttributeError, NotImplementedError):
            cached_per_device = None
        if (cached_per_device is not None) and (
            cached_per_device != num_kv_heads // num_devices
        ):
            raise ValueError(
                f"layer {layer_idx} is sharded {num_devices} ways into "
                f"{num_kv_heads // num_devices} KV heads per rank, but its "
                f"cache holds {cached_per_device} per device; "
                "the cache and the projection must be split the same way"
            )

        self.text_config = text_config
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.num_devices = num_devices
        self.is_local = is_local
        hidden = text_config.hidden_size
        head_dim = text_config.head_dim_for(is_local)
        q_dim, k_dim, v_dim, r_dim = text_config.qkvr_out_dims(is_local)
        conv_channels = text_config.kv_conv_dim(is_local)
        # Global widths, kept for the projection splits: ``F.split`` takes
        # GLOBAL sizes on a distributed tensor and hands each rank its share.
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.r_dim = r_dim

        # The cached-prefix window: page-aligned, and no wider than the keys the
        # mask can keep.
        page_slot = int(getattr(kv_params, "page_size", 128) or 128)
        window = text_config.attention_window(is_local)
        needed = max_seq_len if window is None else window + page_slot - 1
        key_pages = -(-needed // page_slot)
        query_rows = -(-max(1, max_query_rows) // _QUERY_CHUNK) * _QUERY_CHUNK

        self.block = _DeviceAttention(
            kv_params=kv_params,
            out_dims=(
                q_dim // num_devices,
                k_dim // num_devices,
                v_dim // num_devices,
                r_dim // num_devices,
            ),
            num_heads=num_heads // num_devices,
            num_kv_heads=num_kv_heads // num_devices,
            head_dim=head_dim,
            d_rel=text_config.d_rel,
            rel_extent=text_config.rel_extent_for(is_local),
            # Inkling2 scales by head_dim**-0.5 (:414, :513), not by v1's muP
            # 1/head_dim.
            scale=text_config.attn_scale(is_local),
            qk_clip_threshold=text_config.qk_l2_clip_threshold,
            rel_clip_threshold=text_config.rel_l2_clip_threshold,
            rms_norm_eps=text_config.rms_norm_eps,
            sconv_kernel_size=text_config.sconv_kernel_size,
            local_window_size=window,
            applies_log_scaling=text_config.applies_log_scaling(is_local),
            static_conv=is_local,
            commit=commit_conv_state,
            key_slot=page_slot,
            key_span=key_pages * page_slot,
            query_rows=query_rows,
        )

        # Column-parallel: each rank keeps its own heads of every projection.
        # Bias-free throughout (:419-424).
        self.wq_du = col_parallel(Linear(hidden, q_dim, bias=False))
        self.wk_dv = col_parallel(Linear(hidden, k_dim, bias=False))
        self.wv_dv = col_parallel(Linear(hidden, v_dim, bias=False))
        self.wr_du = col_parallel(Linear(hidden, r_dim, bias=False))

        # conv_idx 0 and 1 (:424-427 sliding, :522-537 full). Both are drawn
        # from the same conv_state["k"]/["v"] leaves in either flavor.
        self.k_sconv: ShortConvolution | DynamicShortConvolution
        self.v_sconv: ShortConvolution | DynamicShortConvolution
        if is_local:
            self.k_sconv = ShortConvolution(
                channels=conv_channels // num_devices,
                kernel_size=text_config.sconv_kernel_size,
                commit_conv_state=commit_conv_state,
            )
            self.v_sconv = ShortConvolution(
                channels=conv_channels // num_devices,
                kernel_size=text_config.sconv_kernel_size,
                commit_conv_state=commit_conv_state,
            )
            # The channel axis carries the head, so a split here hands each
            # rank whole heads.
            self.k_sconv.weight = shard_axis(self.k_sconv.weight, 0)
            self.v_sconv.weight = shard_axis(self.v_sconv.weight, 0)
        else:
            conv_kwargs = {
                "channels": conv_channels // num_devices,
                "num_heads": num_kv_heads // num_devices,
                "input_size": hidden,
                "kernel_size": text_config.sconv_kernel_size,
            }
            self.k_sconv = DynamicShortConvolution(**conv_kwargs)
            self.v_sconv = DynamicShortConvolution(**conv_kwargs)
            # static_w is [tap, channel]; weight_proj's rows are (head, tap)
            # pairs, so a row split hands each rank whole heads.
            for conv in (self.k_sconv, self.v_sconv):
                conv.static_w = shard_axis(conv.static_w, 1)
                conv.weight_proj = shard_axis(conv.weight_proj, 0)

        # The checkpoint ships clip *scales* for q and r and nothing for k
        # (:430-435), so there is no q_norm/k_norm.
        self.q_l2_clip = _ScaledL2Clip(
            num_heads=num_heads // num_devices, head_dim=head_dim
        )
        self.rel_l2_clip = _ScaledL2Clip(
            num_heads=num_heads // num_devices, head_dim=text_config.d_rel
        )
        # One scale row per head, so the split must follow heads.
        self.q_l2_clip.scale = shard_axis(self.q_l2_clip.scale, 0)
        self.rel_l2_clip.scale = shard_axis(self.rel_l2_clip.scale, 0)
        # Replicated: every rank looks up the same [d_rel, rel_extent] bank.
        self.rel_logits_proj = _RelLogitsProj(
            d_rel=text_config.d_rel, rel_extent=self.block.rel_extent
        )
        # gate_proj and sink_proj read the layer's attention input (:454-455)
        # and are column-parallel like q.
        self.gate_attn = col_parallel(
            Linear(hidden, text_config.attn_gate_dim(is_local), bias=False)
        )
        self.attn_sink_proj = col_parallel(
            Linear(hidden, text_config.num_sinks(is_local), bias=False)
        )
        self.wo_ud = RowParallelLinear(q_dim, hidden, bias=False)

    def _project(self, x: Tensor, projections: tuple[Linear, ...]) -> Tensor:
        weight = F.concat([proj.weight for proj in projections], axis=0)
        out = x @ weight.T
        biases = [proj.bias for proj in projections]
        if all(isinstance(bias, Tensor) for bias in biases):
            out = out + F.concat(biases, axis=0)
        return out

    def forward(
        self,
        x: Tensor,
        *,
        kv_collection: PagedCacheValues,
        input_row_offsets: Tensor,
        positions: Tensor,
        log_scaling: Tensor,
        k_conv_ring: Tensor,
        k_conv_rows: Tensor,
        k_layer_row: Tensor,
        v_conv_ring: Tensor,
        v_conv_rows: Tensor,
        v_layer_row: Tensor,
        cache_layer_idx: Tensor,
        cache_row: int,
        key_hidden_states: Tensor | None = None,
        value_hidden_states: Tensor | None = None,
    ) -> Tensor:
        """Runs the block over one ragged batch."""
        if self.is_local:
            q_pre, k_pre, v_pre, r_pre = F.split(
                self._project(
                    x, (self.wq_du, self.wk_dv, self.wv_dv, self.wr_du)
                ),
                [self.q_dim, self.k_dim, self.k_dim, self.r_dim],
                axis=-1,
            )
            k_sconv, v_sconv = self.k_sconv, self.v_sconv
            assert isinstance(k_sconv, ShortConvolution)
            assert isinstance(v_sconv, ShortConvolution)
            # A static conv reads no side state; the projection output stands
            # in so both flavors share one call signature.
            k_side, v_side = k_pre, v_pre
            k_taps = k_proj = k_sconv.taps
            v_taps = v_proj = v_sconv.taps
        else:
            if key_hidden_states is None or value_hidden_states is None:
                raise ValueError(
                    f"layer {self.layer_idx} is a full-attention layer: its "
                    "attention takes three states (modeling_inkling2.py:565-571"
                    ", :850-852), and key_hidden_states/value_hidden_states "
                    "were not given. Projecting all three from one state is "
                    "what the sliding layers do."
                )
            q_pre, r_pre = F.split(
                self._project(x, (self.wq_du, self.wr_du)),
                [self.q_dim, self.r_dim],
                axis=-1,
            )
            k_pre = self._project(key_hidden_states, (self.wk_dv,))
            v_pre = self._project(value_hidden_states, (self.wv_dv,))
            k_sconv, v_sconv = self.k_sconv, self.v_sconv
            assert isinstance(k_sconv, DynamicShortConvolution)
            assert isinstance(v_sconv, DynamicShortConvolution)
            k_side, v_side = key_hidden_states, value_hidden_states
            k_taps, k_proj = k_sconv.static_w, k_sconv.weight_proj
            v_taps, v_proj = v_sconv.static_w, v_sconv.weight_proj

        attn_out = _attention(
            q_pre,
            k_pre,
            v_pre,
            r_pre,
            k_side,
            v_side,
            self.gate_attn(x),
            self.attn_sink_proj(x),
            positions,
            log_scaling,
            input_row_offsets,
            kv_collection,
            self.q_l2_clip.scale,
            self.rel_l2_clip.scale,
            k_taps,
            k_proj,
            v_taps,
            v_proj,
            self.rel_logits_proj.proj,
            k_conv_ring,
            v_conv_ring,
            k_conv_rows,
            v_conv_rows,
            k_layer_row,
            v_layer_row,
            cache_layer_idx,
            cache_row,
            block=self.block,
        )
        return self.wo_ud(as_sharded(attn_out, x.mesh, axis=1))
