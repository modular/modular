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
"""One-launch epilogues of a DSA lightning indexer with SSMax.

`indexer_q_ssmax_rope_quantize` replaces the query chain an SSMax indexer runs
after its query projection (SSMax scaling, interleaved RoPE over the leading
columns of each head, per-head FP8 quantization with an E8M0 scale), and
`indexer_k_quantize_store` replaces the whole key chain from the raw key
projection on (a biased LayerNorm, interleaved RoPE over the leading columns,
per-key FP8 quantization with an E8M0 scale, then the paged-cache stores of
the key and its scale).

Each step rounds where the graph-op chain it replaces rounds, so the outputs
are bit-identical to that chain: SSMax's factor is rounded to BF16 after the
log, after the scale and after the bias, the scaled query and the rotated
columns are rounded to BF16, the normalized key is rounded to BF16 before the
rotation, and the quantization derives its scale from the BF16 values exactly
as `quantize_dynamic_scaled_fp8` does. Each row is computed on its own, so a
row's result does not depend on how many rows the launch holds.

The key's LayerNorm statistics are the bar's sharpest edge.
`mo.reduce.layer_norm` dispatches to the extensibility kernel
`builtin_kernels.reductions.LayerNorm`, which runs
`builtin_kernels.rowwise_normalization.layer_norm`; there
`rowwise.pick_simd_width[Welford[float32, 1], target, 96, ...]` picks the width.
The module's `keep_dtype=False` casts the key to float32 before the op, so that
pick is 4 -- float32's natural width, which the 96-byte budget leaves alone --
and 128 channels sit under the warp tier's ceiling, so every row takes one warp.
The reduction is Welford's online state: one four-wide fold per lane, a
sequential combine of the four lanes, then the warp's distance-halving
butterfly. `_welford_fold`/`_welford_join` below restate those two formulas
expression for expression, and the launch keeps the same lane-to-column map.

The tree's association order is what the byte comparison does not pin: the
butterfly's stride order and the four-lane sequential combine alike. These keys
are BF16 values upcast to float32 and every warp-level join combines equal
counts, so its ratio is exactly one half and the statistics come out the same
bit for bit whatever order the combines take -- a swapped stride order passes
the tests unchanged. The order below mirrors the reference's by construction;
the byte comparison pins the rounding points, not the association.

The key chain is also specified by the Moka component in
`Kernels/test/gpu/Moka/kernels/indexer/indexer_k_tail_plan.mojo`, whose plan and
sync certificate pin the launch the shipped graph op stages.
"""

from std.bit import log2_floor
from std.math import ceildiv, log, rsqrt
from std.utils.index import StaticTuple
from std.utils.numerics import max_finite
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
import max.gpu.primitives.warp as warp
from internal_utils.fp8_utils import fp8_quantize
from kv_cache.types import KVCacheT
from layout import TileTensor
from nn.rope import _rope

comptime INDEX_HEAD_DIM = 128
"""Channels of one index head, which is also the quantization group."""

comptime _LANE_WIDTH = INDEX_HEAD_DIM // WARP_SIZE
"""Channels each lane of a head's warp holds."""

comptime _WARPS_PER_BLOCK = 4

comptime _BLOCK_THREADS = _WARPS_PER_BLOCK * WARP_SIZE

comptime _WARP_LEVELS = log2_floor(WARP_SIZE)
"""Tree levels of the warp reduction the reference LayerNorm uses."""


@inline(.always)
def _welford_fold[
    w: Int
](
    mut count: SIMD[.float32, w],
    mut mean: SIMD[.float32, w],
    mut m2: SIMD[.float32, w],
    val: SIMD[.float32, w],
):
    """Folds one full-width tile into a Welford state, as the graph's row
    reduction does.

    A tile as wide as the state advances every lane's count in lock step, so
    the reciprocal is one scalar division splatted across the lanes rather
    than a per-lane one. `na` is the count before the fold, which is what the
    sum of squared deviations is scaled by.
    """
    var na = count
    var new_count = na + SIMD[.float32, w](1)
    var delta = val - mean
    var inv = SIMD[.float32, w](1.0 / new_count[0])
    mean = mean + delta * inv
    m2 = m2 + delta * delta * na * inv
    count = new_count


@inline(.always)
def _welford_join[
    w: Int
](
    mut count: SIMD[.float32, w],
    mut mean: SIMD[.float32, w],
    mut m2: SIMD[.float32, w],
    other_count: SIMD[.float32, w],
    other_mean: SIMD[.float32, w],
    other_m2: SIMD[.float32, w],
):
    """Combines two Welford states with Chan et al.'s update, as the graph's
    row reduction does.

    The sum of squared deviations takes the new total into account before
    the mean moves, and the mean ratio is floored at one sample so an empty
    side cannot divide by zero.
    """
    var na = count
    var nb = other_count
    var new_count = na + nb
    var safe_n = max(new_count, SIMD[.float32, w](1))
    var delta = other_mean - mean
    var n_ratio = nb / safe_n
    m2 = m2 + other_m2 + delta * delta * na * n_ratio
    mean = mean + delta * n_ratio
    count = new_count


@inline(.always)
def _batch_of(
    row_offsets: ImmPointer[UInt32, ImmUntrackedOrigin],
    num_offsets: Int,
    row: Int,
) -> Int:
    """The request a ragged row belongs to, as `get_batch_from_row_offsets`.

    That helper takes a `TileTensor` and asserts the row is in range; these
    kernels reach their inputs through pointers, where a check is already
    done before the call, so they keep this pointer form.
    """
    var low = 0
    var high = num_offsets - 1
    while low + 1 != high:
        var mid = (low + high) // 2
        if row >= Int(row_offsets[unsafe_offset=mid]):
            low = mid
        else:
            high = mid
    return low


@inline(.always)
def _e8m0_scale(
    values: SIMD[.float32, _LANE_WIDTH]
) -> Tuple[Scalar[.float8_e8m0fnu], Float32]:
    """The head's E8M0 scale and its reciprocal, reduced over the warp.

    The expressions are `_QuantizeFp8Kernel._group_scale`'s with no amax
    floor, so the scale matches the standalone quantizer's bit for bit. It
    keeps its own body rather than calling that method because it also owns
    the cross-lane max the quantizer reduces before the call.
    """
    var group_max = warp.lane_group_max[WARP_SIZE](abs(values).reduce_max())
    comptime fp8_max = max_finite[.float8_e4m3fn]()
    var scale = max(
        max(group_max, Float32(0)) / fp8_max.cast[.float32](),
        Float32(1e-10),
    ).cast[.float8_e8m0fnu]()
    var recip = Float32(0) if group_max == 0.0 else 1.0 / scale.cast[.float32]()
    return (scale, recip)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK_THREADS))
)
@__name(t"indexer_q_ssmax_rope_fp8_r{rope_dim}_h{num_heads}")
def _indexer_q_kernel[
    num_heads: Int,
    rope_dim: Int,
](
    q_out: MutPointer[Scalar[.float8_e4m3fn], MutUntrackedOrigin],
    scale_out: MutPointer[Scalar[.float8_e8m0fnu], MutUntrackedOrigin],
    q: ImmPointer[BFloat16, ImmUntrackedOrigin],
    positions: ImmPointer[UInt32, ImmUntrackedOrigin],
    ssmax_s: ImmPointer[BFloat16, ImmUntrackedOrigin],
    ssmax_b: ImmPointer[BFloat16, ImmUntrackedOrigin],
    freqs_cis: ImmPointer[BFloat16, ImmUntrackedOrigin],
    row_offsets: ImmPointer[UInt32, ImmUntrackedOrigin],
    start_pos: ImmPointer[UInt32, ImmUntrackedOrigin],
    num_offsets_arg: Int32,
    num_rows_arg: Int32,
    q_row_stride_arg: Int32,
    freqs_row_stride_arg: Int32,
):
    var num_offsets = Int(num_offsets_arg)
    var num_rows = Int(num_rows_arg)
    var q_row_stride = Int(q_row_stride_arg)
    var freqs_row_stride = Int(freqs_row_stride_arg)
    var unit = block_idx.x * _WARPS_PER_BLOCK + warp_id()
    var row = unit // num_heads
    var head = unit % num_heads
    var col = lane_id() * _LANE_WIDTH
    with PDL():
        # A row past the last offset belongs to no request: its rope batch
        # search would answer with the last one and read its bytes out of
        # bounds. Mirrors the key kernel's bound.
        if row >= num_rows or row >= Int(
            row_offsets[unsafe_offset=num_offsets - 1]
        ):
            return
        # Only the row stride is taken: a head's 128 channels are contiguous,
        # as the projection's output is.
        var src = q.unsafe_offset(row * q_row_stride + head * INDEX_HEAD_DIM)
        # The projection's zero bias is folded away in the graph, negative
        # zeros included, so it is not applied here. Each SSMax factor step
        # rounds to BF16 as the graph's chain rounds it.
        var v = src.unsafe_load[width=_LANE_WIDTH](col)
        var log_pos = log(
            positions[unsafe_offset=row].cast[.float32]() + Float32(1)
        ).cast[.bfloat16]()
        var factor = (
            log_pos * ssmax_s[unsafe_offset=head] + ssmax_b[unsafe_offset=head]
        )
        v = v * factor

        comptime if rope_dim > 0:
            if col < rope_dim:
                var batch = _batch_of(row_offsets, num_offsets, row)
                var pos = Int(start_pos[unsafe_offset=batch]) + (
                    row - Int(row_offsets[unsafe_offset=batch])
                )
                var freq = freqs_cis.unsafe_offset(
                    pos * freqs_row_stride
                ).unsafe_load[width=_LANE_WIDTH](col)
                v = _rope(v, freq)

        var values = v.cast[.float32]()
        var scale, recip = _e8m0_scale(values)
        var unit_offset = row * num_heads + head
        q_out.unsafe_offset(unit_offset * INDEX_HEAD_DIM).unsafe_store(
            col, fp8_quantize[.float8_e4m3fn](values, recip)
        )
        if lane_id() == 0:
            scale_out[unsafe_offset=unit_offset] = scale


def indexer_q_ssmax_rope_quantize[
    rope_dim: Int,
](
    q_out: TileTensor[mut=True, .float8_e4m3fn, ...],
    scale_out: TileTensor[mut=True, .float8_e8m0fnu, ...],
    q: TileTensor[.bfloat16, ...],
    positions: TileTensor[.uint32, ...],
    ssmax_s: TileTensor[.bfloat16, ...],
    ssmax_b: TileTensor[.bfloat16, ...],
    freqs_cis: TileTensor[.bfloat16, ...],
    input_row_offsets: TileTensor[.uint32, ...],
    start_pos: TileTensor[.uint32, ...],
    ctx: DeviceContext,
) raises:
    """Scales, rotates and FP8-quantizes an indexer query in one launch.

    Computes, per row `t` at absolute position `positions[t]` and head `h`,
    `q[t, h] * (s[h] * log(positions[t] + 1) + b[h])`, rotates its leading
    `rope_dim` columns by the interleaved RoPE at the row's cache position,
    and quantizes the head to FP8 with one E8M0 scale. A row past the last
    ragged offset belongs to no request and is skipped, as the key op's rows
    are.

    `start_pos` has to hold an entry per request, or the rope's batch lookup
    would read past it, so a shorter buffer is refused as the key op refuses
    one. `q_out` and `scale_out` are addressed flat, so a non-contiguous
    layout is refused too.

    Parameters:
        rope_dim: Leading columns of each head the RoPE rotates; 0 rotates
            none.

    Args:
        q_out: `[rows, heads, 128]` FP8 query.
        scale_out: `[rows, heads]` E8M0 scale of each head.
        q: `[rows, heads, 128]` BF16 query projection, rows contiguous.
        positions: `[rows]` absolute 0-based positions, for SSMax.
        ssmax_s: `[heads]` SSMax scale.
        ssmax_b: `[heads]` SSMax bias.
        freqs_cis: `[positions, rope_dim]` interleaved RoPE table, rows
            contiguous.
        input_row_offsets: `[batch + 1]` ragged row offsets.
        start_pos: `[batch]` cached tokens per request, the RoPE start.
        ctx: The device context.
    """
    comptime assert q.flat_rank == 3, "q must be [rows, heads, head_dim]"
    comptime num_heads = Int(q.static_shape[1])
    comptime assert (
        Int(q.static_shape[2]) == INDEX_HEAD_DIM
    ), "an index head is 128 channels"
    comptime assert rope_dim % _LANE_WIDTH == 0 and rope_dim <= INDEX_HEAD_DIM
    var num_rows = Int(q.dim[0]())
    var num_offsets = Int(input_row_offsets.dim[0]())
    if num_offsets < 1:
        raise Error("input_row_offsets must hold at least one offset")
    # Both outputs are addressed flat in the kernel, so they have to be
    # contiguous; the input `q` is read through its row stride below.
    if (
        Int(q_out.dim[2]()) != INDEX_HEAD_DIM
        or Int(q_out.dynamic_stride(2)) != 1
        or Int(q_out.dynamic_stride(1)) != INDEX_HEAD_DIM
        or Int(q_out.dynamic_stride(0)) != Int(q_out.dim[1]()) * INDEX_HEAD_DIM
    ):
        raise Error("q_out must be a contiguous [rows, heads, 128] tensor")
    comptime num_heads = Int(q.static_shape[1])
    if (
        Int(scale_out.dim[1]()) != num_heads
        or Int(scale_out.dynamic_stride(1)) != 1
        or Int(scale_out.dynamic_stride(0)) != num_heads
    ):
        raise Error("scale_out must be a contiguous [rows, heads] tensor")
    # The rope start is read by batch index, so the buffer has to cover every
    # request the offsets describe. A lower bound rather than an equality: a
    # step allocates it at the batch it was compiled for. The key op refuses a
    # short one for the same reason.
    if Int(start_pos.dim[0]()) < num_offsets - 1:
        raise Error(
            "start_pos holds ",
            Int(start_pos.dim[0]()),
            " entries for ",
            num_offsets - 1,
            " requests",
        )
    if num_rows == 0:
        return
    comptime kernel = _indexer_q_kernel[num_heads, rope_dim]
    ctx.enqueue_function[kernel](
        q_out.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
        scale_out.ptr.unsafe_origin_cast[MutUntrackedOrigin](),
        q.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        positions.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        ssmax_s.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        ssmax_b.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        freqs_cis.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        input_row_offsets.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        start_pos.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        Int32(input_row_offsets.dim[0]()),
        Int32(num_rows),
        Int32(q.dynamic_stride(0)),
        Int32(freqs_cis.dynamic_stride(0)),
        grid_dim=ceildiv(num_rows * num_heads, _WARPS_PER_BLOCK),
        block_dim=_BLOCK_THREADS,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK_THREADS))
)
@__name(t"indexer_k_norm_rope_fp8_store_r{rope_dim}_o{key_col_offset}")
def _indexer_k_kernel[
    cache_t: KVCacheT,
    gamma_dtype: DType,
    beta_dtype: DType,
    rope_dim: Int,
    key_col_offset: Int,
](
    k_cache: cache_t,
    k: ImmPointer[BFloat16, ImmUntrackedOrigin],
    gamma: ImmPointer[Scalar[gamma_dtype], ImmUntrackedOrigin],
    beta: ImmPointer[Scalar[beta_dtype], ImmUntrackedOrigin],
    freqs_cis: ImmPointer[BFloat16, ImmUntrackedOrigin],
    row_offsets: ImmPointer[UInt32, ImmUntrackedOrigin],
    start_pos: ImmPointer[UInt32, ImmUntrackedOrigin],
    epsilon: Float32,
    num_offsets_arg: Int32,
    num_rows_arg: Int32,
    k_row_stride_arg: Int32,
    freqs_row_stride_arg: Int32,
):
    var num_offsets = Int(num_offsets_arg)
    var num_rows = Int(num_rows_arg)
    var k_row_stride = Int(k_row_stride_arg)
    var freqs_row_stride = Int(freqs_row_stride_arg)
    # The tree below assumes the reference's lane-to-column map at a 32-lane
    # warp: one four-wide fold per lane, then strides 16, 8, 4, 2, 1. Below 32
    # lanes the reference keeps its four-wide state and grid-strides the row in
    # chunk pairs, where `_LANE_WIDTH` would widen instead, so the two part; at
    # 64 they coincide again for this row width. Refusing every other warp
    # width is therefore conservative rather than required -- this model runs
    # B200 only -- and it keeps the op off hardware nobody has measured it on.
    comptime assert (
        WARP_SIZE == 32
    ), "the indexer key's LayerNorm assumes a 32-lane warp"
    var row = block_idx.x * _WARPS_PER_BLOCK + warp_id()
    var col = lane_id() * _LANE_WIDTH
    with PDL():
        if row >= num_rows or row >= Int(
            row_offsets[unsafe_offset=num_offsets - 1]
        ):
            return
        # The key's 128 channels are contiguous from `key_col_offset` on: the
        # merged projection puts the latent's columns first and the key's
        # after them, so the op reads its slice where the GEMM wrote it.
        var x = (
            k.unsafe_offset(row * k_row_stride + key_col_offset)
            .unsafe_load[width=_LANE_WIDTH](col)
            .cast[.float32]()
        )

        # The graph's row reduction, at the same lane-to-column map: this
        # lane's four channels folded into the wide state, the state's four
        # lanes combined in order, then the warp's distance-halving tree.
        var welford_count = SIMD[.float32, _LANE_WIDTH](0)
        var welford_mean = SIMD[.float32, _LANE_WIDTH](0)
        var welford_m2 = SIMD[.float32, _LANE_WIDTH](0)
        _welford_fold[_LANE_WIDTH](welford_count, welford_mean, welford_m2, x)
        var count = welford_count[0]
        var mean = welford_mean[0]
        var m2 = welford_m2[0]
        comptime for j in range(1, _LANE_WIDTH):
            _welford_join[1](
                count,
                mean,
                m2,
                welford_count[j],
                welford_mean[j],
                welford_m2[j],
            )
        comptime for level in range(_WARP_LEVELS):
            var stride = UInt32(WARP_SIZE >> (level + 1))
            _welford_join[1](
                count,
                mean,
                m2,
                warp.shuffle_xor(count, stride),
                warp.shuffle_xor(mean, stride),
                warp.shuffle_xor(m2, stride),
            )
        var inv = rsqrt(m2 / count + epsilon)

        var gamma_val = (
            gamma.unsafe_offset(col)
            .unsafe_load[width=_LANE_WIDTH]()
            .cast[.float32]()
        )
        var beta_val = (
            beta.unsafe_offset(col)
            .unsafe_load[width=_LANE_WIDTH]()
            .cast[.float32]()
        )
        # The graph writes the normalized row in float32 and rounds it to the
        # input dtype; that one rounding is the quantization's input.
        var normed = (
            (x - SIMD[.float32, _LANE_WIDTH](mean))
            * SIMD[.float32, _LANE_WIDTH](inv)
            * gamma_val
            + beta_val
        ).cast[.bfloat16]()

        var batch = _batch_of(row_offsets, num_offsets, row)
        comptime if rope_dim > 0:
            if col < rope_dim:
                var pos = Int(start_pos[unsafe_offset=batch]) + (
                    row - Int(row_offsets[unsafe_offset=batch])
                )
                var freq = freqs_cis.unsafe_offset(
                    pos * freqs_row_stride
                ).unsafe_load[width=_LANE_WIDTH](col)
                normed = _rope(normed, freq)

        var values = normed.cast[.float32]()
        var scale, recip = _e8m0_scale(values)
        var token = (
            row
            - Int(row_offsets[unsafe_offset=batch])
            + k_cache.cache_length(batch)
        )
        k_cache.store(
            batch,
            0,
            token,
            col,
            fp8_quantize[cache_t.dtype](values, recip),
        )
        if lane_id() == 0:
            k_cache.store_scale(batch, 0, token, 0, scale.cast[.float32]())


def indexer_k_quantize_store[
    cache_t: KVCacheT,
    gamma_dtype: DType,
    beta_dtype: DType,
    rope_dim: Int,
    key_col_offset: Int,
](
    k_cache: cache_t,
    k: TileTensor[.bfloat16, ...],
    gamma: TileTensor[gamma_dtype, ...],
    beta: TileTensor[beta_dtype, ...],
    epsilon: Float32,
    freqs_cis: TileTensor[.bfloat16, ...],
    input_row_offsets: TileTensor[.uint32, ...],
    start_pos: TileTensor[.uint32, ...],
    ctx: DeviceContext,
) raises:
    """Normalizes, rotates, FP8-quantizes index keys and stores them and their
    scales in the cache, in one launch.

    Replaces, bit for bit, a biased LayerNorm over each 128-channel key, the
    interleaved RoPE over the leading `rope_dim` columns, the per-key FP8
    quantization with one E8M0 scale, and the two paged-cache stores. The key
    lands at its request's next cache position, and its scale, as float32, in
    the cache's scale page at the same position. Rows past the last offset
    belong to no request and are skipped.

    The key's channels are read out of `k` in place, starting at
    `key_col_offset`, so the merged projection can be handed over whole: the
    op takes its 128 columns where the GEMM wrote them and no copy of them is
    launched. A caller with a packed key passes offset 0.

    `start_pos` has to hold an entry per request, or the kernel would read past
    it; a shorter buffer is refused. The chain this replaces tolerated one when
    the trailing requests carried no rows, so that pathological case is the one
    input the fused op refuses and the chain did not.

    Parameters:
        cache_t: The paged key cache type, with float32 scales (inferred).
        gamma_dtype: Dtype of the normalization's scale weight.
        beta_dtype: Dtype of the normalization's bias weight.
        rope_dim: Leading columns the RoPE rotates; 0 rotates none.
        key_col_offset: The column `k`'s key starts at; 0 for a packed key.

    Args:
        k_cache: The layer's key cache.
        k: `[rows, key_col_offset + 128]` BF16 channels as the projection left
            them: the merged projection up to and including the key's last
            column, or just the key when the offset is 0. The row stride is
            read from the tile rather than assumed, so a projection that keeps
            its latent columns beside the key's works without a copy.
        gamma: `[128]` LayerNorm scale weight.
        beta: `[128]` LayerNorm bias weight.
        epsilon: The LayerNorm epsilon.
        freqs_cis: `positions x rope_dim` interleaved RoPE table, rows
            contiguous.
        input_row_offsets: `[batch + 1]` ragged row offsets.
        start_pos: `[batch]` cached tokens per request, the RoPE start.
        ctx: The device context.
    """
    comptime assert k.flat_rank == 2, "k must be [rows, key_col_offset + 128]"
    comptime assert (
        Int(k.static_shape[1]) == key_col_offset + INDEX_HEAD_DIM
    ), "k ends at the key's last channel"
    comptime assert cache_t.dtype == .float8_e4m3fn, "an FP8 key cache"
    comptime assert gamma.flat_rank == 1 and beta.flat_rank == 1
    comptime assert (
        Int(gamma.static_shape[0]) == INDEX_HEAD_DIM
    ), "the normalization spans an index key"
    comptime assert (
        Int(beta.static_shape[0]) == INDEX_HEAD_DIM
    ), "the normalization shifts an index key"
    comptime assert rope_dim % _LANE_WIDTH == 0 and rope_dim <= INDEX_HEAD_DIM
    comptime assert freqs_cis.flat_rank == 2
    comptime assert Int(freqs_cis.static_shape[1]) >= rope_dim
    comptime assert start_pos.flat_rank == 1, "start_pos must be [batch]"
    var num_rows = Int(k.dim[0]())
    var num_offsets = Int(input_row_offsets.dim[0]())
    if num_offsets < 1:
        raise Error("input_row_offsets must hold at least one offset")
    # A request's rope start is read by batch index, so the buffer has to cover
    # every request the offsets describe. A lower bound rather than an equality:
    # a step allocates it at the batch it was compiled for.
    if Int(start_pos.dim[0]()) < num_offsets - 1:
        raise Error(
            "start_pos holds ",
            Int(start_pos.dim[0]()),
            " entries for ",
            num_offsets - 1,
            " requests",
        )
    if num_rows == 0:
        return
    comptime kernel = _indexer_k_kernel[
        cache_t, gamma_dtype, beta_dtype, rope_dim, key_col_offset
    ]
    ctx.enqueue_function[kernel](
        k_cache,
        k.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        gamma.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        beta.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        freqs_cis.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        input_row_offsets.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        start_pos.ptr.as_imm().unsafe_origin_cast[ImmUntrackedOrigin](),
        epsilon,
        Int32(num_offsets),
        Int32(num_rows),
        Int32(k.dynamic_stride(0)),
        Int32(freqs_cis.dynamic_stride(0)),
        grid_dim=ceildiv(num_rows, _WARPS_PER_BLOCK),
        block_dim=_BLOCK_THREADS,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
