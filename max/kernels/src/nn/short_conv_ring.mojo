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
"""Depthwise causal short convolution over a ring of past inputs.

A request's conv state is a ring indexed by absolute position: the input at
position `p` lives at `ring[p % R]`. A forward is two launches that never
touch the same entry:

- `short_conv_ring_fwd`: `x + conv(x)`. A tap at `p - j` reads `x` when it
  is inside the chunk, the ring when it is before the chunk, and zero before
  position zero. Writes nothing.
- `short_conv_ring_commit`: writes each sequence's last `R` inputs into
  their entries. Reads nothing.

One code path therefore covers decode, prefill, mixed batches and
speculative verify. A rejected draft token's entry is overwritten once the
sequence passes its position again, so `R` must be `width - 1` plus the
largest rollback.
"""

from std.math import ceildiv
from std.sys.info import align_of, simd_width_of
from std.utils.numerics import get_accum_type

from max.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_gpu
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import TileTensor
from layout.coord import Coord
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from nn._ragged_utils import get_batch_and_token_idx_from_row_offsets
from nn.normalization import _rms_norm_warp_tiling_subkernel

comptime _BLOCK = 256
"""Threads per block."""


def _vector_width[dtype: DType, a: Int, b: Int]() -> Int:
    """The GPU vector width when `a` and `b` are static and divide by it,
    else 1."""
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    if a == -1 or b == -1 or a % simd_width != 0 or b % simd_width != 0:
        return 1
    return simd_width


@inline(.always)
def _short_conv_ring_step[
    x_dtype: DType,
    ring_dtype: DType,
    //,
    width: Int,
    accum_dtype: DType,
    simd_width: Int = 1,
](
    x: TileTensor[mut=False, x_dtype, ...],
    weight: TileTensor[mut=False, x_dtype, ...],
    ring: TileTensor[ring_dtype, ...],
    token_idx: Int,
    col: Int,
    channel: Int,
    slot: Int,
    position: Int,
    chunk_start: Int,
) -> SIMD[accum_dtype, simd_width]:
    """`x + conv(x)` for one token of `simd_width` adjacent channels.

    `x[token_idx, col]` is the token's input. Tap `j` reads position
    `position - j` from `x` when it is at or after `chunk_start`, from
    `ring[slot, (position - j) % R, channel]` when it is before the chunk,
    else zero. `weight[channel, width - 1]` multiplies the current input,
    `weight[channel, width - 1 - j]` the input `j` back, as in
    `causal_conv1d`.
    """
    comptime ring_len = ring.static_shape[1]
    comptime assert ring_len != -1, "the conv ring needs a static length"
    comptime assert (
        ring_len >= width - 1
    ), "the conv ring must hold at least width - 1 positions of history"

    # weight is [channels, width], so this thread's taps are one contiguous
    # run; load it once and transpose into one vector per tap.
    var taps = Array[SIMD[accum_dtype, simd_width], width](fill=0)
    comptime if width & (width - 1) == 0:
        var run = weight.load[
            width=simd_width * width,
            alignment=align_of[SIMD[x_dtype, simd_width]](),
        ](Coord(channel, 0)).cast[accum_dtype]()
        comptime for i in range(simd_width):
            comptime for w in range(width):
                taps[w][i] = run[i * width + w]
    else:
        comptime for w in range(width):
            comptime for i in range(simd_width):
                taps[w][i] = weight.load[width=1](Coord(channel + i, w)).cast[
                    accum_dtype
                ]()[0]

    var x_cur = x.load[width=simd_width](Coord(token_idx, col)).cast[
        accum_dtype
    ]()
    var acc = x_cur + x_cur * taps[width - 1]
    comptime for j in range(1, width):
        var tap_pos = position - j
        var value = SIMD[accum_dtype, simd_width](0)
        if tap_pos >= chunk_start:
            value = x.load[width=simd_width](Coord(token_idx - j, col)).cast[
                accum_dtype
            ]()
        elif tap_pos >= 0:
            value = ring.load[width=simd_width](
                Coord(slot, tap_pos % ring_len, channel)
            ).cast[accum_dtype]()
        acc += value * taps[width - 1 - j]
    return acc


def short_conv_ring_fwd[
    dtype: DType,
    ring_dtype: DType,
    //,
    target: StaticString,
](
    x: TileTensor[mut=False, dtype, ...],
    weight: TileTensor[mut=False, dtype, ...],
    ring: TileTensor[ring_dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """`x + conv(x)` over a ragged batch; reads the ring, writes nothing.

    `x`, `output`: `[total_seq_len, channels]`. `weight`: `[channels,
    width]`. `ring`: `[slots, R, channels]`. `positions` is per token,
    `conv_row` per sequence.
    """
    comptime assert is_gpu[target](), "short_conv_ring_fwd is GPU-only"
    comptime assert x.flat_rank == 2, "x must be [total_seq_len, channels]"
    comptime assert output.flat_rank == 2, "output must match x"
    comptime assert weight.flat_rank == 2, "weight must be [channels, width]"
    comptime assert ring.flat_rank == 3, "ring must be [slots, R, channels]"
    comptime width = weight.static_shape[1]
    comptime assert width != -1, "Need static shape for weight"
    comptime accum_dtype = get_accum_type[dtype]()
    comptime vec = _vector_width[
        dtype, x.static_shape[1], ring.static_shape[2]
    ]()

    var total_seq_len = Int(x.dim[0]())
    var channels = Int(x.dim[1]())
    if total_seq_len == 0:
        return
    # Kernel captures must be fixed-width; Int is not device-passable.
    var channel_blocks = Int32(ceildiv(channels, _BLOCK * vec))
    var channels_dev = Int32(channels)

    def kernel() {
        var x,
        var weight,
        var ring,
        var input_row_offsets,
        var positions,
        var conv_row,
        var output,
        var channel_blocks,
        var channels_dev,
    }:
        var token_idx = Int(block_idx.x) // Int(channel_blocks)
        var channel = (
            (Int(block_idx.x) % Int(channel_blocks)) * Int(block_dim.x)
            + Int(thread_idx.x)
        ) * vec
        if channel >= Int(channels_dev):
            return
        var batch_idx, idx_in_seq = get_batch_and_token_idx_from_row_offsets(
            input_row_offsets, token_idx
        )
        var position = Int(positions.load[width=1](Coord(token_idx))[0])
        var value = _short_conv_ring_step[width, accum_dtype, vec](
            x,
            weight,
            ring,
            token_idx,
            channel,
            channel,
            Int(conv_row.load[width=1](Coord(batch_idx))[0]),
            position,
            position - idx_in_seq,
        )
        output.store[width=vec](Coord(token_idx, channel), value.cast[dtype]())

    with Trace[TraceLevel.OP, target=target](
        "short_conv_ring_fwd.channels_" + String(channels),
        task_id=get_safe_task_id(context),
    ):
        context.enqueue_function(
            kernel,
            grid_dim=total_seq_len * Int(channel_blocks),
            block_dim=_BLOCK,
        )


@inline(.always)
def _commit_sequence_tail[
    x_dtype: DType, //
](
    x: TileTensor[mut=False, x_dtype, ...],
    ring: TileTensor[mut=True, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    batch_idx: Int,
    col: Int,
    channel: Int,
):
    """Writes one channel of a sequence's last `R` inputs into its slot.

    Earlier inputs share entries with later ones, so only the tail is
    written and every entry has one writer.
    """
    comptime ring_len = ring.static_shape[1]
    comptime assert ring_len != -1, "the conv ring needs a static length"

    var start = Int(input_row_offsets.load[width=1](Coord(batch_idx))[0])
    var end = Int(input_row_offsets.load[width=1](Coord(batch_idx + 1))[0])
    var slot = Int(conv_row.load[width=1](Coord(batch_idx))[0])
    for token_idx in range(max(start, end - ring_len), end):
        var position = Int(positions.load[width=1](Coord(token_idx))[0])
        var value = x.load[width=1](Coord(token_idx, col))
        ring.store[width=1](
            Coord(slot, position % ring_len, channel),
            value.cast[ring.dtype](),
        )


def short_conv_ring_commit[
    dtype: DType,
    //,
    target: StaticString,
](
    x: TileTensor[mut=False, dtype, ...],
    ring: TileTensor[mut=True, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    conv_row: TileTensor[mut=False, .uint32, ...],
    context: DeviceContext,
) raises:
    """Writes each sequence's last `R` rows of `x` into its ring slot.

    `x`: `[total_seq_len, channels]`. `ring`: `[slots, R, channels]`.
    Launch after every reader of the ring in the same forward.
    """
    comptime assert is_gpu[target](), "short_conv_ring_commit is GPU-only"
    comptime assert x.flat_rank == 2, "x must be [total_seq_len, channels]"
    comptime assert ring.flat_rank == 3, "ring must be [slots, R, channels]"
    comptime channels = ring.static_shape[2]
    comptime assert channels != -1, "the conv ring needs a static width"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    if batch_size <= 0 or x.dim[0]() == 0:
        return

    def kernel() {
        var x,
        var ring,
        var input_row_offsets,
        var positions,
        var conv_row,
    }:
        var channel = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        if channel >= channels:
            return
        _commit_sequence_tail(
            x,
            ring,
            input_row_offsets,
            positions,
            conv_row,
            Int(block_idx.y),
            channel,
            channel,
        )

    with Trace[TraceLevel.OP, target=target](
        "short_conv_ring_commit.channels_" + String(channels),
        task_id=get_safe_task_id(context),
    ):
        context.enqueue_function(
            kernel,
            grid_dim=(ceildiv(channels, _BLOCK), batch_size),
            block_dim=_BLOCK,
        )


def short_conv_ring_commit_kv[
    dtype: DType,
    ring_dtype: DType,
    //,
    target: StaticString,
    k_col: Int,
](
    qkvr: TileTensor[mut=False, dtype, ...],
    k_ring: TileTensor[mut=True, ring_dtype, ...],
    v_ring: TileTensor[mut=True, ring_dtype, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    k_conv_row: TileTensor[mut=False, .uint32, ...],
    v_conv_row: TileTensor[mut=False, .uint32, ...],
    context: DeviceContext,
) raises:
    """Commits the K and V conv inputs of an attention block in one launch.

    K channels start at column `k_col` of `qkvr` and V channels follow them.
    Grid z picks the site.
    """
    comptime assert is_gpu[target](), "short_conv_ring_commit_kv is GPU-only"
    comptime assert qkvr.flat_rank == 2, "qkvr must be rank 2"
    comptime assert k_ring.flat_rank == 3, "k_ring must be [slots, R, C]"
    comptime assert v_ring.flat_rank == 3, "v_ring must be [slots, R, C]"
    comptime channels = k_ring.static_shape[2]
    comptime assert channels != -1, "the conv ring needs a static width"
    comptime assert (
        v_ring.static_shape[2] == channels
    ), "K and V conv rings must have the same width"
    comptime assert (
        v_ring.static_shape[1] == k_ring.static_shape[1]
    ), "K and V conv rings must have the same length"

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    if batch_size <= 0 or qkvr.dim[0]() == 0:
        return

    def kernel() {
        var qkvr,
        var k_ring,
        var v_ring,
        var input_row_offsets,
        var positions,
        var k_conv_row,
        var v_conv_row,
    }:
        var channel = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
        if channel >= channels:
            return
        # Distinct operands never share a TileTensor type, so no select.
        if block_idx.z == 0:
            _commit_sequence_tail(
                qkvr,
                k_ring,
                input_row_offsets,
                positions,
                k_conv_row,
                Int(block_idx.y),
                k_col + channel,
                channel,
            )
        else:
            _commit_sequence_tail(
                qkvr,
                v_ring,
                input_row_offsets,
                positions,
                v_conv_row,
                Int(block_idx.y),
                k_col + channels + channel,
                channel,
            )

    with Trace[TraceLevel.OP, target=target](
        "short_conv_ring_commit_kv.channels_" + String(channels),
        task_id=get_safe_task_id(context),
    ):
        context.enqueue_function(
            kernel,
            grid_dim=(ceildiv(channels, _BLOCK), batch_size, 2),
            block_dim=_BLOCK,
        )


# HACK: the kernel closure stores through both `k_cache` and `v_cache`, the
# disjoint halves of one `blocks` buffer that share the collection's mutable
# origins, which the exclusivity checker cannot tell apart. Same stopgap as
# `_rope_split_store_ragged_impl`; the proper fix is disjoint k/v origins.
@__unsafe_nested_origins_read_only
def _launch_fused_qk_rms_norm_short_conv[
    dtype: DType,
    ring_dtype: DType,
    cache_t: KVCacheT,
    //,
    head_dim: Int,
    width: Int,
    multiply_before_cast: Bool,
    apply_log_scaling: Bool,
](
    qkvr: TileTensor[mut=False, dtype, ...],
    k_cache: cache_t,
    v_cache: cache_t,
    q_gamma: TileTensor[mut=False, dtype, ...],
    k_gamma: TileTensor[mut=False, dtype, ...],
    k_weight: TileTensor[mut=False, dtype, ...],
    v_weight: TileTensor[mut=False, dtype, ...],
    k_conv_ring: TileTensor[ring_dtype, ...],
    v_conv_ring: TileTensor[ring_dtype, ...],
    epsilon: Float32,
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    k_conv_row: TileTensor[mut=False, .uint32, ...],
    v_conv_row: TileTensor[mut=False, .uint32, ...],
    log_scaling: TileTensor[mut=False, .float32, ...],
    q_output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    comptime kv_num_heads = cache_t.kv_params.num_heads
    comptime accum_dtype = get_accum_type[dtype]()
    comptime vec = _vector_width[dtype, head_dim, qkvr.static_shape[1]]()
    comptime warps_per_block = ceildiv(head_dim // vec, WARP_SIZE)
    comptime assert (
        warps_per_block
        <= context.default_device_info.max_thread_block_size // WARP_SIZE
    ), "fused short-conv prologue block size exceeds device max warps per block"

    # Captured scalars must be device-passable, hence `Int32` rather than `Int`.
    var total_seq_len = Int(qkvr.dim[0]())
    var q_num_heads = Int32(q_output.dim[1]())
    var q_rows = Int32(total_seq_len) * q_num_heads
    var kv_rows = Int32(total_seq_len * kv_num_heads)

    # One block per (token, head) row, Q rows first, then K, then V; one
    # thread per `vec` elements of the head. Q and K are normed, V is not.
    def kernel() {
        var qkvr,
        var q_output,
        var k_cache,
        var v_cache,
        var q_gamma,
        var k_gamma,
        var k_weight,
        var v_weight,
        var k_conv_ring,
        var v_conv_ring,
        var input_row_offsets,
        var positions,
        var k_conv_row,
        var v_conv_row,
        var log_scaling,
        var epsilon,
        var q_num_heads,
        var q_rows,
        var kv_rows,
    }:
        var n_q_rows = Int(q_rows)
        var n_kv_rows = Int(kv_rows)
        var row = Int(block_idx.x)
        var idx = Int(thread_idx.x) * vec
        var in_head = idx < head_dim
        var is_q = row < n_q_rows
        var is_k = not is_q and row < n_q_rows + n_kv_rows

        var token_idx: Int
        var head_idx: Int
        if is_q:
            token_idx = row // Int(q_num_heads)
            head_idx = row % Int(q_num_heads)
        else:
            var kv_row = row - n_q_rows - (0 if is_k else n_kv_rows)
            token_idx = kv_row // kv_num_heads
            head_idx = kv_row % kv_num_heads

        var value = SIMD[accum_dtype, vec](0)
        var gamma_val = SIMD[dtype, vec](0)
        var batch_idx = 0
        var tok_in_seq = 0
        if is_q:
            if in_head:
                value = qkvr.load[width=vec](
                    Coord(token_idx, head_idx * head_dim + idx)
                ).cast[accum_dtype]()
                gamma_val = q_gamma.load[width=vec](Coord(idx))
        else:
            batch_idx, tok_in_seq = get_batch_and_token_idx_from_row_offsets(
                input_row_offsets, token_idx
            )
            var position = Int(positions.load[width=1](Coord(token_idx))[0])
            var chunk_start = position - tok_in_seq
            var channel = head_idx * head_dim + idx
            var col = (Int(q_num_heads) + (0 if is_k else kv_num_heads)) * (
                head_dim
            ) + channel
            # Distinct operands never share a TileTensor type, so no select.
            if in_head and is_k:
                value = _short_conv_ring_step[width, accum_dtype, vec](
                    qkvr,
                    k_weight,
                    k_conv_ring,
                    token_idx,
                    col,
                    channel,
                    Int(k_conv_row.load[width=1](Coord(batch_idx))[0]),
                    position,
                    chunk_start,
                )
                gamma_val = k_gamma.load[width=vec](Coord(idx))
            elif in_head:
                value = _short_conv_ring_step[width, accum_dtype, vec](
                    qkvr,
                    v_weight,
                    v_conv_ring,
                    token_idx,
                    col,
                    channel,
                    Int(v_conv_row.load[width=1](Coord(batch_idx))[0]),
                    position,
                    chunk_start,
                )

        var out_val: SIMD[dtype, vec]
        if is_q or is_k:
            out_val = _rms_norm_warp_tiling_subkernel[
                warps_per_block, multiply_before_cast
            ](
                row,
                idx,
                value,
                gamma_val,
                epsilon,
                Scalar[accum_dtype](0),
                head_dim,
            )
        else:
            out_val = value.cast[dtype]()

        if not in_head:
            return
        if is_q:
            comptime if apply_log_scaling:
                # Rounds to dtype before scaling, as the unfused graph did.
                var factor = log_scaling.load[width=1](Coord(token_idx))
                out_val = (out_val.cast[DType.float32]() * factor).cast[dtype]()
            q_output.store[width=vec](Coord(token_idx, head_idx, idx), out_val)
            return
        var cache = k_cache if is_k else v_cache
        cache.store(
            bs=batch_idx,
            head_idx=head_idx,
            tok_idx=tok_in_seq + cache.cache_length(batch_idx),
            head_dim_idx=idx,
            val=out_val.cast[cache_t.dtype](),
        )

    context.enqueue_function(
        kernel,
        grid_dim=Int(q_rows + 2 * kv_rows),
        block_dim=warps_per_block * WARP_SIZE,
    )


def fused_qk_rms_norm_short_conv_ragged_paged[
    dtype: DType,
    ring_dtype: DType,
    params: KVCacheStaticParams,
    page_size: Int,
    cache_dtype: DType,
    //,
    target: StaticString,
    multiply_before_cast: Bool,
    apply_log_scaling: Bool,
](
    qkvr: TileTensor[mut=False, dtype, ...],
    kv_collection: PagedKVCacheCollection[
        cache_dtype,
        params,
        page_size,
        ...,
    ],
    q_gamma: TileTensor[mut=False, dtype, ...],
    k_gamma: TileTensor[mut=False, dtype, ...],
    k_weight: TileTensor[mut=False, dtype, ...],
    v_weight: TileTensor[mut=False, dtype, ...],
    k_conv_ring: TileTensor[ring_dtype, ...],
    v_conv_ring: TileTensor[ring_dtype, ...],
    epsilon: Float32,
    layer_idx: UInt32,
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    positions: TileTensor[mut=False, .uint32, ...],
    k_conv_row: TileTensor[mut=False, .uint32, ...],
    v_conv_row: TileTensor[mut=False, .uint32, ...],
    log_scaling: TileTensor[mut=False, .float32, ...],
    q_output: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    """A short-conv attention block's prologue in one GPU launch.

    Reads Q, K and V by column offset from `qkvr` `[total_seq_len, q_dim +
    k_dim + v_dim + ...]`. Q: per-head RMSNorm into `q_output`. K, V:
    depthwise causal conv with residual, taps before the chunk from the conv
    rings; K is then RMSNormed; both are stored into the paged cache for
    `layer_idx`. With `apply_log_scaling`, each token's Q is then scaled by
    `log_scaling[token]`.

    Exact for any chunk length. Only reads the rings; commit them afterwards
    with `short_conv_ring_commit_kv`.
    """
    comptime assert is_gpu[
        target
    ](), "fused_qk_rms_norm_short_conv_ragged_paged is GPU-only"
    comptime assert qkvr.flat_rank == 2, "qkvr must be rank 2"
    comptime assert q_output.flat_rank == 3, "q_output must be rank 3"
    comptime head_dim = params.head_size
    comptime assert (
        q_gamma.static_shape[0] == head_dim
        and k_gamma.static_shape[0] == head_dim
    ), "fused short-conv prologue requires full per-head normalization"
    comptime width = k_weight.static_shape[1]
    comptime assert width != -1, "Need static shape for k_weight"
    comptime assert (
        v_weight.static_shape[1] == width
    ), "k_weight and v_weight must have the same conv width"

    if qkvr.dim[0]() == 0:
        return

    with Trace[TraceLevel.OP, target=target](
        "fused_qk_rms_norm_short_conv_ragged_paged.kv_nhead_"
        + String(params.num_heads)
        + ".hdim_"
        + String(head_dim),
        task_id=get_safe_task_id(context),
    ):
        _launch_fused_qk_rms_norm_short_conv[
            head_dim=head_dim,
            width=width,
            multiply_before_cast=multiply_before_cast,
            apply_log_scaling=apply_log_scaling,
        ](
            qkvr,
            kv_collection.get_key_cache(Int(layer_idx)),
            kv_collection.get_value_cache(Int(layer_idx)),
            q_gamma,
            k_gamma,
            k_weight,
            v_weight,
            k_conv_ring,
            v_conv_ring,
            epsilon,
            input_row_offsets,
            positions,
            k_conv_row,
            v_conv_row,
            log_scaling,
            q_output,
            context,
        )
