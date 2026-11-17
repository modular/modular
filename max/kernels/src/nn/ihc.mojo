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
"""Identity hyper-connection (iHC) site kernels.

An iHC site keeps `hc_mult` residual streams per token. Before its sublayer it
projects the flattened streams (`K = hc_mult * hidden` values) onto `num_mix`
float32 weight rows, scales the logits by the streams' inverse RMS, turns them
into sigmoid gates, folds the streams into one sequence with the first
`hc_mult` gates (`pre`), and RMS-normalizes that sequence.

Two kernels do it with two reads of the streams, which stay in their own
dtype throughout:

- `ihc_mix_gemv`: the projection and the streams' sum of squares, split over
  `K`, on tensor cores straight from the bfloat16 streams (the float32
  weight as a bfloat16 pair), rather than converting the streams for a
  float32 (TF32) GEMM.
- `ihc_collapse_rms_norm`: the gates, the collapse and the RMSNorm, one block
  per token.

`nn.mhc.mhc_pre` fuses the same kind of pre site for mHC, with the comb mixer
and its Sinkhorn projection, and only on gfx950; iHC has neither. Two
bfloat16 terms of the weight (where `mhc_pre` takes three) already exceed the
TF32 a float32 matmul would use, and keep the gates within 1e-5.
"""

from std.math import ceildiv, rsqrt
from std.sys.info import size_of
from std.utils.index import StaticTuple

import max.gpu.primitives.warp as warp
from layout import TensorLayout, TileTensor, row_major
from layout import stack_allocation as tt_stack_allocation
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.compute.mma import mma
from max.gpu.primitives import block
from max.gpu.sync import barrier
from max.runtime.tracing import Trace, TraceLevel

from nn.activations import sigmoid

comptime _GEMV_WARP_K = 128
"""Columns of the flattened streams one warp of `ihc_mix_gemv` owns."""
comptime _GEMV_MAX_WARPS = 8
comptime _GEMV_TILES = 4
"""16-row tiles one `ihc_mix_gemv` block projects."""
comptime _COLLAPSE_THREADS = 256

comptime IHC_POISONS = 4
"""Planted faults, `poison` 1 through 4, each of which a test must catch:
1 drops the low half of the split weight, 2 sums only half of a row pair's
squares, 3 scales `pre` by the `post` band's scale, 4 skips the last row."""


@inline(.always)
def ihc_mix_gemv_block[
    dtype: DType,
    num_mix: Int,
    num_warps: Int,
    poison: Int = 0,
](
    partials: TileTensor[mut=True, .float32, ...],
    streams: TileTensor[dtype, ...],
    weight: TileTensor[.float32, ...],
    num_rows: Int,
    row_block: Int,
    split: Int,
    tid: Int,
):
    """Projects `_GEMV_TILES` 16-row tiles onto one `K` split on tensor cores.

    One CTA's work: `tid` is the thread in a block of `num_warps` warps that
    takes rows `row_block * 64 ...` and split `split` of the columns. Warp
    `w` owns `_GEMV_WARP_K` columns of the flattened streams and runs
    `m16n8k16` MMAs over them for every row tile. The float32 weight is split
    into a bfloat16 `hi` and `lo = weight - hi`, two MMAs per tile, which keeps
    about 16 bits of it; the bfloat16 streams are exact.

    `K` is a reduction axis, so it can be permuted freely as long as A and B
    agree. Lane `(g, t) = (lane // 4, lane % 4)` loads 8 contiguous columns
    `8t .. 8t+7` of each 32-column chunk for rows `g` and `g + 8`; the first
    four are its fragment of the chunk's first MMA, the last four of the
    second, and it loads row `g` of the weight at the same columns.

    Parameters:
        dtype: The stream dtype.
        num_mix: Number of `weight` rows.
        num_warps: Warps per block.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        partials: Split `s`'s share of each row's projections, then of its
            sum of squares. Shape: `[splits, num_rows, num_mix + 1]`.
        streams: The flattened streams. Shape: `[num_rows, k]`.
        weight: The projection weight. Shape: `[num_mix, k]`.
        num_rows: Number of rows.
        row_block: The block of `_GEMV_TILES * 16` rows.
        split: The block of `num_warps * _GEMV_WARP_K` columns.
        tid: The thread in the block.
    """
    comptime chunks = _GEMV_WARP_K // 32
    comptime num_out = num_mix + 1
    comptime per_tile = 16 * num_out
    comptime assert dtype == DType.bfloat16, "the MMA takes bfloat16 streams"
    comptime assert WARP_SIZE == 32, "needs 32-lane warps"
    comptime assert num_mix <= 8 and num_mix % 2 == 0

    var warp_idx = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var g = lane // 4
    var t = lane % 4
    var k0 = (split * num_warps + warp_idx) * _GEMV_WARP_K + 8 * t

    var w_hi = StaticTuple[SIMD[DType.bfloat16, 8], chunks]()
    var w_lo = StaticTuple[SIMD[DType.bfloat16, 8], chunks]()
    comptime for c in range(chunks):
        var w = SIMD[DType.float32, 8](0)
        if g < num_mix:
            w = weight.load[width=8]((g, k0 + 32 * c))
        w_hi[c] = w.cast[.bfloat16]()
        w_lo[c] = (w - w_hi[c].cast[.float32]()).cast[.bfloat16]()

    # Every warp's sums for every tile, reduced across warps once at the end.
    var sums = tt_stack_allocation[dtype=.float32, address_space=.SHARED](
        row_major[_GEMV_TILES, num_warps, 16, num_out]()
    )
    var row_base = row_block * _GEMV_TILES * 16

    for tile in range(_GEMV_TILES):
        # Block-uniform; the reduction below never writes a skipped tile's rows.
        if row_base + tile * 16 >= num_rows:
            break
        var row_g = row_base + tile * 16 + g
        var top = StaticTuple[SIMD[dtype, 8], chunks]()
        var bottom = StaticTuple[SIMD[dtype, 8], chunks]()
        comptime for c in range(chunks):
            top[c] = 0
            bottom[c] = 0
            if row_g < num_rows:
                top[c] = streams.load[width=8]((row_g, k0 + 32 * c))
            if row_g + 8 < num_rows:
                bottom[c] = streams.load[width=8]((row_g + 8, k0 + 32 * c))

        var acc = SIMD[DType.float32, 4](0)
        var sq = SIMD[DType.float32, 2](0)
        comptime for c in range(chunks):
            comptime for half in range(2):
                comptime o = 4 * half
                # A fragment order: (g, k), (g + 8, k), (g, k + 8), (g + 8,
                # k + 8), two values each.
                var a = (
                    top[c]
                    .slice[2, offset=o]()
                    .join(bottom[c].slice[2, offset=o]())
                    .join(
                        top[c]
                        .slice[2, offset=o + 2]()
                        .join(bottom[c].slice[2, offset=o + 2]())
                    )
                )
                mma(acc, a, w_hi[c].slice[4, offset=o](), acc)
                comptime if poison != 1:
                    mma(acc, a, w_lo[c].slice[4, offset=o](), acc)
            var ft = top[c].cast[.float32]()
            var fb = bottom[c].cast[.float32]()
            sq[0] += (ft * ft).reduce_add()
            sq[1] += (fb * fb).reduce_add()
        # The four lanes of a group hold one row pair's columns between them.
        comptime for step in range(1 if poison == 2 else 2):
            sq += SIMD[DType.float32, 2](
                warp.shuffle_xor(sq[0], UInt32(1 << step)),
                warp.shuffle_xor(sq[1], UInt32(1 << step)),
            )

        # The C fragment: rows g and g + 8, columns 2t and 2t + 1.
        if 2 * t < num_mix:
            sums[tile, warp_idx, g, 2 * t] = acc[0]
            sums[tile, warp_idx, g, 2 * t + 1] = acc[1]
            sums[tile, warp_idx, g + 8, 2 * t] = acc[2]
            sums[tile, warp_idx, g + 8, 2 * t + 1] = acc[3]
        if t == 0:
            sums[tile, warp_idx, g, num_mix] = sq[0]
            sums[tile, warp_idx, g + 8, num_mix] = sq[1]
    barrier()

    for i in range(tid, _GEMV_TILES * per_tile, num_warps * WARP_SIZE):
        var tile = i // per_tile
        var r = (i % per_tile) // num_out
        var col = i % num_out
        var total = Float32(0)
        comptime for w in range(num_warps):
            total += rebind[Float32](sums[tile, w, r, col])
        var row = row_base + tile * 16 + r
        if row < num_rows:
            partials[split, row, col] = total


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(num_warps * WARP_SIZE)
    )
)
# Without a residency bound ptxas holds the kernel to 70 registers and
# serializes its stream loads; with it, 94, and 0.65 us less at one row.
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"ihc_mix_gemv_{dtype}_m{num_mix}_k{k}_w{num_warps}")
def ihc_mix_gemv_kernel[
    dtype: DType,
    PartialsLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    WeightLayoutType: TensorLayout,
    num_mix: Int,
    k: Int,
    num_warps: Int,
    poison: Int,
](
    partials: TileTensor[
        mut=True, .float32, PartialsLayoutType, MutUntrackedOrigin
    ],
    streams: TileTensor[dtype, StreamsLayoutType, ImmUntrackedOrigin],
    weight: TileTensor[.float32, WeightLayoutType, ImmUntrackedOrigin],
    num_rows: Int32,
):
    """Runs `ihc_mix_gemv_block` for block `(row_block, split)`.

    Parameters:
        dtype: The stream dtype.
        PartialsLayoutType: Layout of `partials`.
        StreamsLayoutType: Layout of `streams`.
        WeightLayoutType: Layout of `weight`.
        num_mix: Number of `weight` rows.
        k: Width of the flattened streams, `hc_mult * hidden`.
        num_warps: Warps per block.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        partials: Shape: `[k // (num_warps * _GEMV_WARP_K), num_rows,
            num_mix + 1]`.
        streams: The flattened streams. Shape: `[num_rows, k]`.
        weight: The projection weight. Shape: `[num_mix, k]`.
        num_rows: Number of rows.
    """
    ihc_mix_gemv_block[dtype, num_mix, num_warps, poison](
        partials,
        streams,
        weight,
        Int(num_rows),
        block_idx.x,
        block_idx.y,
        thread_idx.x,
    )


@inline(.always)
def ihc_collapse_rms_norm_block[
    dtype: DType,
    hc_mult: Int,
    num_mix: Int,
    num_splits: Int,
    hidden: Int,
    block_size: Int,
    poison: Int = 0,
](
    output: TileTensor[mut=True, dtype, ...],
    gates: TileTensor[mut=True, .float32, ...],
    streams: TileTensor[dtype, ...],
    partials: TileTensor[.float32, ...],
    base: TileTensor[.float32, ...],
    scale: TileTensor[.float32, ...],
    gamma: TileTensor[dtype, ...],
    hc_magnitude: Float32,
    hc_eps: Float32,
    streams_eps: Float32,
    norm_eps: Float32,
    row: Int,
    tid: Int,
):
    """Collapses one token's streams and RMS-normalizes the result.

    One CTA's work: `tid` is the thread in a block of `block_size` that takes
    token `row`. The gates come from `partials` alone, so the streams are read
    once, straight into the collapse.

    Parameters:
        dtype: The stream, output and `gamma` dtype.
        hc_mult: Number of residual streams.
        num_mix: Gate logits per token: `hc_mult` (`pre` only) or
            `2 * hc_mult` (`pre` then `post`).
        num_splits: Leading dimension of `partials`.
        hidden: Width of one stream.
        block_size: Threads per block.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        output: `rms_norm(sum_j pre_j * streams[:, j])`, the collapse rounded
            to `dtype` first. Shape: `[num_rows, hidden]`.
        gates: `pre`, then `post` when `num_mix == 2 * hc_mult`. Shape:
            `[num_rows, num_mix]`.
        streams: Shape: `[num_rows, hc_mult, hidden]`.
        partials: `ihc_mix_gemv`'s output. Shape:
            `[num_splits, num_rows, num_mix + 1]`.
        base: Per-logit bias. Shape: `[num_mix]`.
        scale: One scale per band. Shape: `[num_mix // hc_mult]`.
        gamma: The RMSNorm weight. Shape: `[hidden]`.
        hc_magnitude: Upper bound of `post`.
        hc_eps: Added to every gate.
        streams_eps: Epsilon of the streams' RMS.
        norm_eps: Epsilon of the output RMSNorm.
        row: The token.
        tid: The thread in the block.
    """
    comptime width = 16 // size_of[dtype]()
    comptime vecs = ceildiv(hidden, width * block_size)
    comptime assert hidden % width == 0, "hidden must fill whole vectors"
    comptime assert num_mix == hc_mult or num_mix == 2 * hc_mult
    comptime assert num_mix + 1 <= block_size
    comptime pre_band = 1 if poison == 3 and num_mix > hc_mult else 0

    var sums = tt_stack_allocation[dtype=.float32, address_space=.SHARED](
        row_major[num_mix + 1]()
    )
    if tid <= num_mix:
        var total = Float32(0)
        comptime for s in range(num_splits):
            total += rebind[Float32](partials.load((s, row, tid)))
        sums[tid] = total
    barrier()

    var inv_rms = rsqrt(sums[num_mix] / Float32(hc_mult * hidden) + streams_eps)
    if tid < num_mix:
        var g = sigmoid(
            sums[tid] * inv_rms * rebind[Float32](scale.load((tid // hc_mult,)))
            + rebind[Float32](base.load((tid,)))
        )
        gates[row, tid] = (hc_magnitude * g if tid >= hc_mult else g) + hc_eps

    var pre = SIMD[DType.float32, hc_mult]()
    comptime for j in range(hc_mult):
        pre[j] = (
            sigmoid(
                sums[j] * inv_rms * rebind[Float32](scale.load((pre_band,)))
                + rebind[Float32](base.load((j,)))
            )
            + hc_eps
        )

    var collapsed = StaticTuple[SIMD[dtype, width], vecs]()
    var sum_sq = Float32(0)
    comptime for i in range(vecs):
        var d = (tid + i * block_size) * width
        var acc = SIMD[DType.float32, width](0)
        if d < hidden:
            comptime for j in range(hc_mult):
                acc += (
                    pre[j]
                    * streams.load[width=width]((row, j, d)).cast[.float32]()
                )
        collapsed[i] = acc.cast[dtype]()
        var f = collapsed[i].cast[.float32]()
        sum_sq += (f * f).reduce_add()

    var factor = rsqrt(
        block.sum[block_size=block_size](sum_sq) / Float32(hidden) + norm_eps
    )
    comptime for i in range(vecs):
        var d = (tid + i * block_size) * width
        if d < hidden:
            # Llama-style, as `mo.reduce.rms_norm`: round, then scale by gamma.
            output.store(
                (row, d),
                (collapsed[i].cast[.float32]() * factor).cast[dtype]()
                * gamma.load[width=width]((d,)),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"ihc_collapse_rms_norm_{dtype}_h{hc_mult}_m{num_mix}_d{hidden}")
def ihc_collapse_rms_norm_kernel[
    dtype: DType,
    OutputLayoutType: TensorLayout,
    GatesLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    PartialsLayoutType: TensorLayout,
    BaseLayoutType: TensorLayout,
    ScaleLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    hc_mult: Int,
    num_mix: Int,
    num_splits: Int,
    hidden: Int,
    block_size: Int,
    poison: Int,
](
    output: TileTensor[mut=True, dtype, OutputLayoutType, MutUntrackedOrigin],
    gates: TileTensor[mut=True, .float32, GatesLayoutType, MutUntrackedOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmUntrackedOrigin],
    partials: TileTensor[.float32, PartialsLayoutType, ImmUntrackedOrigin],
    base: TileTensor[.float32, BaseLayoutType, ImmUntrackedOrigin],
    scale: TileTensor[.float32, ScaleLayoutType, ImmUntrackedOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, ImmUntrackedOrigin],
    hc_magnitude: Float32,
    hc_eps: Float32,
    streams_eps: Float32,
    norm_eps: Float32,
):
    """Runs `ihc_collapse_rms_norm_block` for token `block_idx.x`.

    Parameters:
        dtype: The stream, output and `gamma` dtype.
        OutputLayoutType: Layout of `output`.
        GatesLayoutType: Layout of `gates`.
        StreamsLayoutType: Layout of `streams`.
        PartialsLayoutType: Layout of `partials`.
        BaseLayoutType: Layout of `base`.
        ScaleLayoutType: Layout of `scale`.
        GammaLayoutType: Layout of `gamma`.
        hc_mult: Number of residual streams.
        num_mix: Gate logits per token.
        num_splits: Leading dimension of `partials`.
        hidden: Width of one stream.
        block_size: Threads per block.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        output: Shape: `[num_rows, hidden]`.
        gates: Shape: `[num_rows, num_mix]`.
        streams: Shape: `[num_rows, hc_mult, hidden]`.
        partials: Shape: `[num_splits, num_rows, num_mix + 1]`.
        base: Shape: `[num_mix]`.
        scale: Shape: `[num_mix // hc_mult]`.
        gamma: Shape: `[hidden]`.
        hc_magnitude: Upper bound of `post`.
        hc_eps: Added to every gate.
        streams_eps: Epsilon of the streams' RMS.
        norm_eps: Epsilon of the output RMSNorm.
    """
    ihc_collapse_rms_norm_block[
        dtype, hc_mult, num_mix, num_splits, hidden, block_size, poison
    ](
        output,
        gates,
        streams,
        partials,
        base,
        scale,
        gamma,
        hc_magnitude,
        hc_eps,
        streams_eps,
        norm_eps,
        block_idx.x,
        thread_idx.x,
    )


@inline(.always)
def ihc_mix_gemv_warps[k: Int]() -> Int:
    """Returns the warps per block of `ihc_mix_gemv`.

    Parameters:
        k: Width of the flattened streams.

    Returns:
        The warps per block.
    """
    return min(_GEMV_MAX_WARPS, k // _GEMV_WARP_K)


@inline(.always)
def ihc_mix_gemv_splits[k: Int]() -> Int:
    """Returns the leading dimension of `ihc_mix_gemv`'s partials.

    Parameters:
        k: Width of the flattened streams.

    Returns:
        The number of column splits of the flattened streams.
    """
    comptime cols = ihc_mix_gemv_warps[k]() * _GEMV_WARP_K
    comptime assert k % cols == 0, "k must split into whole warps"
    return k // cols


@inline(.always)
def ihc_mix_gemv[
    k: Int,
    num_mix: Int,
    poison: Int = 0,
](
    partials: TileTensor[mut=True, .float32, ...],
    streams: TileTensor[mut=False, ...],
    weight: TileTensor[mut=False, .float32, ...],
    context: DeviceContext,
) raises:
    """Launches the split-K iHC projection and sum of squares.

    Parameters:
        k: Width of the flattened streams, `hc_mult * hidden`.
        num_mix: Number of `weight` rows.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        partials: Shape: `[ihc_mix_gemv_splits[k](), num_rows, num_mix + 1]`.
        streams: The flattened streams. Shape: `[num_rows, k]`.
        weight: The projection weight. Shape: `[num_mix, k]`.
        context: The device context.

    Raises:
        If a shape disagrees with the parameters or the rows.
    """
    comptime assert streams.flat_rank == 2 and weight.flat_rank == 2
    comptime dtype = streams.dtype
    if (
        Int(streams.dim(1)) != k
        or Int(weight.dim(0)) != num_mix
        or Int(weight.dim(1)) != k
        or Int(partials.dim(0)) != ihc_mix_gemv_splits[k]()
        or Int(partials.dim(1)) != Int(streams.dim(0))
        or Int(partials.dim(2)) != num_mix + 1
    ):
        raise Error("ihc_mix_gemv: a shape disagrees with its op")
    comptime splits = ihc_mix_gemv_splits[k]()
    comptime num_warps = ihc_mix_gemv_warps[k]()

    var num_rows = Int(streams.dim(0)) - Int(poison == 4)
    if num_rows <= 0:
        return

    with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
        "hyper_connection.ihc.mix_gemv", task_id=Int(context.device().id)
    ):
        comptime kernel = ihc_mix_gemv_kernel[
            dtype,
            partials.LayoutType,
            streams.LayoutType,
            weight.LayoutType,
            num_mix,
            k,
            num_warps,
            poison,
        ]
        context.enqueue_function[kernel](
            partials,
            streams,
            weight,
            Int32(num_rows),
            grid_dim=(ceildiv(num_rows, _GEMV_TILES * 16), splits),
            block_dim=num_warps * WARP_SIZE,
        )


@inline(.always)
def ihc_collapse_rms_norm[
    hc_mult: Int,
    num_mix: Int,
    num_splits: Int,
    hidden: Int,
    poison: Int = 0,
](
    output: TileTensor[mut=True, ...],
    gates: TileTensor[mut=True, .float32, ...],
    streams: TileTensor[mut=False, output.dtype, ...],
    partials: TileTensor[mut=False, .float32, ...],
    base: TileTensor[mut=False, .float32, ...],
    scale: TileTensor[mut=False, .float32, ...],
    gamma: TileTensor[mut=False, output.dtype, ...],
    hc_magnitude: Float32,
    hc_eps: Float32,
    streams_eps: Float32,
    norm_eps: Float32,
    context: DeviceContext,
) raises:
    """Launches the iHC collapse + RMSNorm kernel, one block per token.

    Parameters:
        hc_mult: Number of residual streams.
        num_mix: Gate logits per token, `hc_mult` or `2 * hc_mult`.
        num_splits: Leading dimension of `partials`.
        hidden: Width of one stream.
        poison: A planted fault (`IHC_POISONS`), 0 for none.

    Args:
        output: The normalized collapse. Shape: `[num_rows, hidden]`.
        gates: `pre`, then `post` if `partials` carries it. Shape:
            `[num_rows, num_mix]`.
        streams: Shape: `[num_rows, hc_mult, hidden]`.
        partials: `ihc_mix_gemv`'s output. Shape:
            `[num_splits, num_rows, num_mix + 1]`.
        base: Per-logit bias. Shape: `[num_mix]`.
        scale: One scale per band. Shape: `[num_mix // hc_mult]`.
        gamma: The RMSNorm weight. Shape: `[hidden]`.
        hc_magnitude: Upper bound of `post`.
        hc_eps: Added to every gate.
        streams_eps: Epsilon of the streams' RMS.
        norm_eps: Epsilon of the output RMSNorm.
        context: The device context.

    Raises:
        If a width disagrees with the parameters or a row count with the
        streams.
    """
    comptime assert streams.flat_rank == 3 and partials.flat_rank == 3

    if (
        Int(streams.dim(1)) != hc_mult
        or Int(streams.dim(2)) != hidden
        or Int(gamma.dim(0)) != hidden
        or Int(partials.dim(0)) != num_splits
        or Int(partials.dim(2)) != num_mix + 1
        or Int(output.dim(1)) != hidden
        or Int(gates.dim(1)) != num_mix
    ):
        raise Error("ihc_collapse_rms_norm: a width disagrees with its op")
    var rows = Int(streams.dim(0))
    if (
        Int(output.dim(0)) != rows
        or Int(gates.dim(0)) != rows
        or Int(partials.dim(1)) != rows
    ):
        raise Error("ihc_collapse_rms_norm: a row count disagrees")
    if Int(base.dim(0)) != num_mix or Int(scale.dim(0)) != num_mix // hc_mult:
        raise Error(
            "expected base of size ",
            num_mix,
            " and scale of size ",
            num_mix // hc_mult,
            " but got ",
            Int(base.dim(0)),
            " and ",
            Int(scale.dim(0)),
        )

    var num_rows = Int(streams.dim(0)) - Int(poison == 4)
    if num_rows <= 0:
        return

    with Trace[TraceLevel.OP, target=context.target.device_kind_label()](
        "hyper_connection.ihc.collapse_rms_norm",
        task_id=Int(context.device().id),
    ):
        comptime kernel = ihc_collapse_rms_norm_kernel[
            output.dtype,
            output.LayoutType,
            gates.LayoutType,
            streams.LayoutType,
            partials.LayoutType,
            base.LayoutType,
            scale.LayoutType,
            gamma.LayoutType,
            hc_mult,
            num_mix,
            num_splits,
            hidden,
            _COLLAPSE_THREADS,
            poison,
        ]
        context.enqueue_function[kernel](
            output,
            gates,
            streams,
            partials,
            base,
            scale,
            gamma,
            hc_magnitude,
            hc_eps,
            streams_eps,
            norm_eps,
            grid_dim=num_rows,
            block_dim=_COLLAPSE_THREADS,
        )
