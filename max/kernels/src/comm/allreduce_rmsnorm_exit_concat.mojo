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
"""Fused allreduce sum + RMSNorm + column-shard cast + SiTU-and-mul concat.

One launch replaces the four-kernel exit chain of a tensor-parallel latent
MoE (Kimi K3 at TP8):

  1. `allreduce.sum` over every device's `[rows, cols]` partial (float32 or
     bfloat16),
  2. `rms_norm` of every reduced row over all `cols` columns,
  3. the column-shard slice/cast: device `r` keeps columns
     `[r * cols / ngpus, (r + 1) * cols / ngpus)` of the normed row, in bf16,
  4. `moe.exit.act.concat`: that shard written beside the SiTU-and-mul of the
     device's own `[rows, 2 * half]` gated shared-expert projection.

The result is `[rows, cols / ngpus + half]` bf16 per device, column-sharded --
exactly the exit GEMM's activation.

## Shape of the kernel

Every rank holds a *partial* of every column, so the collective is a
reduce-scatter into per-rank scratch (the bytes behind each `Signal` header)
followed by a gather that rebuilds whole rows:

  Stage 0 (before the start barrier, no peer data): each block writes the
  SiTU half of its rows from the LOCAL `gate_up`. It needs nothing from a
  peer, so it hides behind the barrier.

  Stage 1 (reduce-scatter): block `b` on rank `r` reduces columns
  `[r * cols / ngpus, (r + 1) * cols / ngpus)` of its rows from every
  peer's partials in float32 and stores the sum, rounded to the input dtype
  as the plain allreduce rounds it, in rank `r`'s own scratch, laid out
  `[rows, cols / ngpus]`.

  Barrier 2 (`need_fence=True`): the barriers index their counters by block
  id, so block `b` may only read what block `b` wrote on the other ranks;
  the row-ownership below is a function of the block id alone, which is
  what makes that hold.

  Stage 2: block `b` reads the whole row back, slice `p` from peer `p`, to
  fold the mean square in registers, then re-reads its OWN slice (local
  bytes) to write the normalized bf16 shard. Stage 0 already wrote those
  rows' SiTU half.

## Bit-identity

Both the sum and the mean square are computed in the same order the ops this
replaces use, so the fused output is byte-identical to the unfused chain
wherever that chain's `allreduce.sum` takes its 2-stage kernel -- the
1-stage caveat in `Shape dispatch` below applies to it too:

* The peer sum starts at the rank that OWNS each element's SIMD vector in
  `allreduce`'s flat reduce-scatter partition and rotates from there, which
  is the order `_load_reduce` sums in (the 2-stage allreduce hands it the
  tiles already rotated by the partition owner). The owner of row `m`'s
  slice-`r` vectors is `(m * ngpus + r) // rows`: with the partition's
  contiguous chunks, a row's column slice lies inside one chunk, and every
  row inside a chunk shares it.

* The mean square folds a lane-wise `SIMD[f32, simd_width]` accumulator over
  the row in ascending column tiles, collapses it with `reduce_add`, and
  block-reduces that scalar -- `ReduceSum`'s `accumulate` / `reduce` /
  `join_parallel` in `builtin_kernels/monoids.mojo`, which is what the graph
  compiler's generated `rowwise` block-tier norm kernel runs. That tier's
  block size is a fixed 256 threads and its tile stride
  `BLOCK_SIZE * simd_width`, so thread `t` folds the same columns in the same
  order here.

  The norm itself is `ops.rms_norm`'s Llama form: `x * rsqrt(mean(x**2) +
  eps)` in f32, rounded to the input dtype, times gamma, cast to the output
  dtype last (`weight_offset` 0, `multiply_before_cast` False).

* The SiTU half repeats `moe.exit.act.concat`'s body in the same operation
  order. `comm` cannot import `nn`, so the math is copied, not called.

## Shape dispatch

The fused kernel gives a block one row, so its reduce-scatter pulls `cols`
bytes per row from every peer in `ngpus` messages of `cols / ngpus` bytes. That
is the right shape for a decode step (a few dozen rows: the messages ride the
latency, and folding the norm into the collective's epilogue deletes two
launches), and the wrong shape for a prefill chunk (thousands of rows: the
per-row messages leave the fabric idle and the norm's block reduction runs on
`cols / ngpus` columns per row instead of a whole row). In isolation the
fused kernel is still ahead of the chain at every row count measured on MI355
at 2 and at 8 devices, so a microbenchmark alone cannot place a gate. The gate
exists because routing prefill chunks through the fused kernel raised Kimi
K3's end-to-end time to first token on 8 devices, while decode-sized calls
gained.

So the launcher gates on size, at run time, from `rows`/`cols` -- the same
value on every rank, so every rank takes the same branch:

* at or below `EXIT_CONCAT_FUSE_ROWS` (`EXIT_CONCAT_FUSE_THRESHOLD` bytes) the
  fused kernel runs, unchanged;
* every other call -- above the cut, or at a world size the threshold was
  not calibrated for (any but 2, 4 or 8 devices) -- runs the chain this op
  replaced: the plain allreduce into a full `[rows, cols]` sum, then
  `_allreduce_rmsnorm_exit_concat_chain_kernel`, which folds the mean square
  over the local sum, normalizes this rank's slice and writes the SiTU half.

Where the chain's plain allreduce takes its 2-stage kernel, both arms write
the same bytes: its reduce-scatter sums each partition from the partition's
owner first, rotating exactly as the fused kernel does, the chain reads that
peer sum rounded to `in_dtype` (which is what the fused kernel's
reduce-scatter writes) and folds the mean square in the same tile order with
the same block reduction. The plain allreduce's 1-stage kernel sums in
global rank order instead, and CDNA3's tuning keeps 1-stage at every size,
so on such a world the two arms can differ by the rounding of the float32
sum alone -- ULP-level, far under the norm's own tolerance. The chain's
`[rows, cols]` sum is a per-device workspace (`comm.workspace`), not the
payload behind this rank's `Signal` header: only the plain allreduce's
partition staging has to fit there, so a prefill chunk's signal buffers stay
the plain allreduce's size.

A vendor-CCL build and a host without P2P run the chain too, with the vendor
allreduce or the plain allreduce's staged fallback in front of it -- the same
two-launch form `allreduce_rms_norm` takes in those two situations. Neither
consults the size gate, so neither can pick the fused kernel.

The plain `allreduce` kernels are untouched: this file adds its own kernels and
its own scratch layout.
"""

from std.collections import Array
from std.math import exp, rsqrt, tanh
from std.sys import align_of, get_defined_bool, simd_width_of, size_of
from std.utils import StaticTuple

from layout import Coord, TensorLayout, TileTensor, row_major
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block

from .allreduce import allreduce, allreduce_tuning_table
from .device_query import dispatch_select_comm_config, get_sm_version
from .reducescatter import _target_address_space
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    circular_add,
    is_p2p_enabled,
)
from .workspace import staging_workspace
import comm.vendor.ccl as vendor_ccl

comptime ARNEC_THREADS = 256
"""Threads per block.

Fixed rather than tuned: the mean square has to be block-reduced in the same
tree shape as the norm it replaces, whose rowwise block tier hard-codes
`BLOCK_SIZE = 256` (`algorithm/rowwise.mojo`'s `launch`), and that tier's
per-thread tile stride is `BLOCK_SIZE * simd_width` -- this kernel folds each
thread's columns in the same order only at the same block size.
"""


comptime EXIT_CONCAT_CALIBRATED_COLS = 3584
"""Cols the fuse threshold is stated at (Kimi K3's latent).

A different width moves the cut in bytes, so recalibrate before trusting it
there.
"""

comptime EXIT_CONCAT_FUSE_ROWS = 128
"""Rows at `EXIT_CONCAT_CALIBRATED_COLS` at or below which the fused kernel runs.

A decode/prefill cut, not a measured crossover (see the module docstring). A
decode step is 32 rows at batch 4 with DSpark spec decode, well below it, and
a prefill chunk is thousands of rows, well above it. A perf cut, not
correctness: both arms write the same bytes.

`EXIT_CONCAT_FUSE_THRESHOLD` is that row count as `[rows, cols]` payload bytes,
the meter `AR_NORM_FUSE_THRESHOLD` uses: a fused arm's cost is set by the
traffic per row, `cols` bytes from every peer, which does not shrink with the
world size, unlike the plain allreduce's per-rank partition. It is stated in
float32 bytes, so a bfloat16 call fuses up to twice the rows.
"""

comptime EXIT_CONCAT_FUSE_THRESHOLD = (
    EXIT_CONCAT_FUSE_ROWS
    * EXIT_CONCAT_CALIBRATED_COLS
    * size_of[DType.float32]()
)
"""`[rows, cols]` payload bytes at or below which the fused kernel runs."""


def _fuse_exit_concat[
    in_dtype: DType,
    ngpus: Int,
](rows: Int, cols: Int, ctx: DeviceContext) raises -> Bool:
    """Returns whether this launch takes the fused kernel or the chain.

    Reads only `rows`, `cols`, `in_dtype`, `ngpus` and the target, all of which
    the launcher has already checked are identical on every rank (an allreduce
    needs same-shaped operands). The two arms issue different barriers on the
    shared `rank_sigs`, so a per-rank disagreement would deadlock -- this is
    the one decision that must stay shape-only.

    Parameters:
        in_dtype: The partials' element type.
        ngpus: Number of cooperating devices.

    Args:
        rows: Rows in every partial.
        cols: Columns in every partial.
        ctx: Device context, for the target.

    Returns:
        True to run the fused kernel, False to run the chain.

    Raises:
        On an element type the threshold was not derived for: a guess here
        would silently route traffic to the slower arm. A world size it was
        not calibrated for returns False instead, as the off-AMD target does.
    """
    comptime assert in_dtype in (
        DType.bfloat16,
        DType.float32,
    ), "the fused exit collective takes bf16 or float32 partials"
    # The threshold is calibrated at 2 devices and, through the K3
    # end-to-end run, at 8; any other world size takes the chain rather
    # than a guess at where the cut should sit.
    if ngpus not in (2, 4, 8):
        return False
    # Off AMD nothing is calibrated, so the chain always serves.
    comptime if not ctx.target.is_amd_gpu():
        return False

    var payload_bytes = rows * cols * size_of[in_dtype]()
    return payload_bytes <= EXIT_CONCAT_FUSE_THRESHOLD


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ARNEC_THREADS))
)
@__name(
    t"allreduce_rmsnorm_exit_concat_{in_dtype}_{dtype}_{gamma_dtype}_{ngpus}"
)
def _allreduce_rmsnorm_exit_concat_kernel[
    in_dtype: DType,
    gamma_dtype: DType,
    dtype: DType,
    gamma_layout: TensorLayout,
    gamma_origin: Origin,
    gate_up_layout: TensorLayout,
    gate_up_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[gamma_dtype, gamma_layout, gamma_origin],
    gate_up: TileTensor[dtype, gate_up_layout, gate_up_origin],
    output: TileTensor[mut=True, dtype, out_layout, out_origin],
    epsilon: Float32,
    beta: Float32,
    linear_beta: Float32,
    rows_dev: Int32,
    cols_dev: Int32,
    half_dev: Int32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank_dev: Int32,
):
    """Reduces, norms and re-shards every row, and activates the exit's tail.

    Parameters:
        in_dtype: The partials' element type (bfloat16 or float32).
        gamma_dtype: The norm weight's element type: `in_dtype`, or
            bfloat16/float16 on a float32 input (widened in registers).
        dtype: The gated projection's and the output's element type (bf16).
        gamma_layout: Layout of `gamma`.
        gamma_origin: Origin of `gamma`.
        gate_up_layout: Layout of `gate_up`.
        gate_up_origin: Origin of `gate_up`.
        out_layout: Layout of `output`.
        out_origin: Origin of `output`.
        ngpus: Number of cooperating devices (the mesh's TP width).
        simd_width: Elements per thread's vector; the same width the replaced
            norm kernel tiles at.
        threads_per_block: Threads per block; `ARNEC_THREADS`.

    Args:
        src_ptrs: Every device's `[rows, cols]` partial, indexed by rank.
        gamma: This device's copy of the norm's `[cols]` weight, in
            `in_dtype`.
        gate_up: This device's `[rows, 2 * half]` shared gate/up projection,
            gate half first.
        output: This device's `[rows, cols / ngpus + half]` destination: the
            normed column shard of this rank, then the activated half.
        epsilon: The norm's epsilon.
        beta: The SiTU gate's beta.
        linear_beta: The SiTU up-half saturation beta; 0 passes the up half
            through unchanged.
        rows_dev: Rows in every partial.
        cols_dev: Columns in every partial (the whole latent width).
        half_dev: The shared expert's half width (the output's tail).
        rank_sigs: Every device's `Signal` pointer, indexed by rank. The
            bytes behind each header are this kernel's reduce-scatter scratch.
        my_rank_dev: This device's rank in `[0, ngpus)`.
    """
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var half = Int(half_dev)
    var my_rank = Int(my_rank_dev)
    var slice_cols = cols // ngpus
    var slice_start = my_rank * slice_cols
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # The plain allreduce's accumulation type: float32 for either input.
    comptime accum_type = DType.float32
    # The replaced norm kernel's per-thread tile stride (`BLOCK_SIZE *
    # simd_width`), which sets the order each thread folds its columns in.
    comptime block_span = threads_per_block * simd_width

    var tid = thread_idx.x
    var num_blocks = grid_dim.x

    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gate_up.flat_rank == 2, "gate_up must have rank 2"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    # Scratch, indexed by GLOBAL rank: rank p's scratch holds the reduced
    # columns `[p * slice_cols, (p + 1) * slice_cols)` of every row, laid out
    # `[rows, slice_cols]`. `unsafe_offset(1)` steps past the Signal header.
    comptime PtrType = MutPointer[Scalar[in_dtype], MutAnyOrigin]
    var scratch_ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: (
            rank_sigs[i].unsafe_offset(1)
        ).unsafe_bitcast[Scalar[in_dtype]]()
    )
    var my_scratch = scratch_ptrs[my_rank]

    # --- Stage 0: the SiTU half, from this device's own gate/up ---
    # No peer data, so it runs before the start barrier and hides behind it.
    for row in range(block_idx.x, rows, num_blocks):
        for base in range(0, half, block_span):
            var j = base + tid * simd_width
            if j < half:
                var gate = gate_up.load[width=simd_width]((row, j)).cast[
                    .float32
                ]()
                var up = gate_up.load[width=simd_width]((row, j + half)).cast[
                    .float32
                ]()
                # `beta * tanh(gate / beta) * sigmoid(gate) * up`, in the
                # operation order of `moe.exit.act.concat`'s body.
                var situ = beta * tanh(gate / beta)
                situ = situ * (Float32(1.0) / (Float32(1.0) + exp(-gate)))
                if linear_beta != 0.0:
                    up = linear_beta * tanh(up / linear_beta)
                output.store((row, slice_cols + j), (situ * up).cast[dtype]())

    # Start barrier: every rank's partials must be final before any rank
    # reads them, and this barrier also retires the previous collective's use
    # of the scratch written below.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # --- Stage 1: reduce-scatter this rank's column slice ---
    for row in range(block_idx.x, rows, num_blocks):
        # The rank that owns this row's slice in `allreduce`'s flat
        # partition, where the peer sum starts (see the module docstring).
        var owner = Int(UInt32(row * ngpus + my_rank) // UInt32(rows))
        var row_start = row * cols + slice_start
        for base in range(0, slice_cols, block_span):
            var j = base + tid * simd_width
            if j < slice_cols:
                var elem = row_start + j
                # `_load_reduce`'s order: the owner's value first, then the
                # peers rotating from it.
                var accum = (
                    src_ptrs[owner]
                    .unsafe_address_space_cast[_target_address_space]()
                    .unsafe_load[
                        width=simd_width, alignment=align, invariant=True
                    ](elem)
                    .cast[accum_type]()
                )
                comptime for i in range(1, ngpus):
                    var peer = circular_add[ngpus](owner, i)
                    accum += (
                        src_ptrs[peer]
                        .unsafe_address_space_cast[_target_address_space]()
                        .unsafe_load[
                            width=simd_width, alignment=align, invariant=True
                        ](elem)
                        .cast[accum_type]()
                    )
                my_scratch.unsafe_address_space_cast[
                    _target_address_space
                ]().unsafe_store[width=simd_width, alignment=align](
                    row * slice_cols + j, accum.cast[in_dtype]()
                )

    # Mid barrier with release/acquire semantics: block `b`'s scratch writes
    # on every rank are visible to block `b` on every rank.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # --- Stage 2: fold the mean square, then normalize this rank's slice ---
    for row in range(block_idx.x, rows, num_blocks):
        var sum_acc = SIMD[accum_type, simd_width](0)
        for base in range(0, cols, block_span):
            var c = base + tid * simd_width
            if c < cols:
                # Column `c` lives in peer `c // slice_cols`'s scratch. A
                # vector never straddles two slices (`slice_cols` is a
                # multiple of `simd_width`). Both fit 32 bits, so the
                # divide is the short 32-bit sequence, not the 64-bit one.
                var p = Int(UInt32(c) // UInt32(slice_cols))
                var j = c - p * slice_cols
                var v = (
                    scratch_ptrs[p]
                    .unsafe_address_space_cast[_target_address_space]()
                    .unsafe_load[
                        width=simd_width, alignment=align, invariant=True
                    ](row * slice_cols + j)
                ).cast[accum_type]()
                # Lane-wise fold, as `ReduceSum.accumulate` does.
                sum_acc += v * v
        # ... then `ReduceSum.reduce` (the horizontal `reduce_add`) and
        # `ReduceSum.join_parallel` (one block-wide sum, broadcast back).
        var thread_m2 = sum_acc.reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        # Divide by `cols` first, add epsilon after: the norm op's order.
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )

        # The normalized value is this rank's own slice, already in local
        # scratch, so this pass does not touch the fabric.
        for base in range(0, slice_cols, block_span):
            var j = base + tid * simd_width
            if j < slice_cols:
                var v = (
                    my_scratch.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_load[
                        width=simd_width, alignment=align, invariant=True
                    ](
                        row * slice_cols + j
                    )
                ).cast[accum_type]()
                # A narrower gamma widens exactly to the input dtype.
                var g = gamma.load[
                    width=simd_width,
                    alignment=align_of[SIMD[gamma_dtype, simd_width]](),
                ]((slice_start + j,)).cast[in_dtype]()
                # `multiply_before_cast=False`: `(x * inv)` rounds to the
                # input dtype before the gamma multiply (a no-op for float32
                # input), and the result rounds to bf16 last.
                var normed = ((v * norm_factor).cast[in_dtype]() * g).cast[
                    dtype
                ]()
                output.store((row, j), normed)

    # No end barrier: peers only P2P-read the bytes behind `Signal` headers,
    # and the next launch's start barrier orders this rank's scratch reuse
    # behind every peer's Stage 2 -- as the 2-stage allreduce's gather does.


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(ARNEC_THREADS))
)
@__name(t"allreduce_rmsnorm_exit_concat_chain_{in_dtype}_{dtype}_{gamma_dtype}")
def _allreduce_rmsnorm_exit_concat_chain_kernel[
    in_dtype: DType,
    gamma_dtype: DType,
    dtype: DType,
    gamma_layout: TensorLayout,
    gamma_origin: Origin,
    summed_layout: TensorLayout,
    summed_origin: Origin,
    gate_up_layout: TensorLayout,
    gate_up_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
](
    summed: TileTensor[in_dtype, summed_layout, summed_origin],
    gamma: TileTensor[gamma_dtype, gamma_layout, gamma_origin],
    gate_up: TileTensor[dtype, gate_up_layout, gate_up_origin],
    output: TileTensor[mut=True, dtype, out_layout, out_origin],
    epsilon: Float32,
    beta: Float32,
    linear_beta: Float32,
    rows_dev: Int32,
    cols_dev: Int32,
    half_dev: Int32,
    my_rank_dev: Int32,
):
    """Normalizes an already-reduced row and writes its shard beside the SiTU half.

    The chain arm's second launch: `summed` holds the whole `[rows, cols]` sum
    the plain allreduce produced on this device, so no peer data is read and
    there is no barrier. The rounds are the fused kernel's Stage 0 and Stage 2
    verbatim -- same tile order, same block reduction, same operation order --
    which is what makes the two arms byte-identical.

    Parameters:
        in_dtype: The sum's element type (the partials' type).
        gamma_dtype: The norm weight's element type.
        dtype: The gated projection's and the output's element type (bf16).
        gamma_layout: Layout of `gamma`.
        gamma_origin: Origin of `gamma`.
        summed_layout: Layout of `summed`.
        summed_origin: Origin of `summed`.
        gate_up_layout: Layout of `gate_up`.
        gate_up_origin: Origin of `gate_up`.
        out_layout: Layout of `output`.
        out_origin: Origin of `output`.
        ngpus: Number of cooperating devices (the column shard count).
        simd_width: Elements per thread's vector.
        threads_per_block: Threads per block; `ARNEC_THREADS`.

    Args:
        summed: This device's `[rows, cols]` reduced sum.
        gamma: This device's `[cols]` norm weight.
        gate_up: This device's `[rows, 2 * half]` gated projection.
        output: This device's `[rows, cols / ngpus + half]` destination.
        epsilon: The norm's epsilon.
        beta: The SiTU gate's beta.
        linear_beta: The SiTU up-half saturation beta; 0 passes it through.
        rows_dev: Rows in the sum.
        cols_dev: Columns in the sum (the whole latent width).
        half_dev: The shared expert's half width (the output's tail).
        my_rank_dev: This device's rank in `[0, ngpus)`.
    """
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var half = Int(half_dev)
    var my_rank = Int(my_rank_dev)
    var slice_cols = cols // ngpus
    var slice_start = my_rank * slice_cols
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # Fold in float32, the plain allreduce's accumulation type for either
    # input.
    comptime accum_type = DType.float32
    # The replaced norm kernel's per-thread tile stride, as in the fused kernel.
    comptime block_span = threads_per_block * simd_width

    var tid = thread_idx.x
    var num_blocks = grid_dim.x

    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert summed.flat_rank == 2, "the sum must have rank 2"
    comptime assert gate_up.flat_rank == 2, "gate_up must have rank 2"
    comptime assert output.flat_rank == 2, "output must have rank 2"

    for row in range(block_idx.x, rows, num_blocks):
        # The SiTU half, from this device's own gate/up: no peer data.
        for base in range(0, half, block_span):
            var j = base + tid * simd_width
            if j < half:
                var gate = gate_up.load[width=simd_width]((row, j)).cast[
                    .float32
                ]()
                var up = gate_up.load[width=simd_width]((row, j + half)).cast[
                    .float32
                ]()
                var situ = beta * tanh(gate / beta)
                situ = situ * (Float32(1.0) / (Float32(1.0) + exp(-gate)))
                if linear_beta != 0.0:
                    up = linear_beta * tanh(up / linear_beta)
                output.store((row, slice_cols + j), (situ * up).cast[dtype]())

        # Fold the mean square over the whole row, in the order the fused
        # kernel's Stage 2 does: the same column tiles, the same lane-wise
        # accumulator, the same block tree.
        var sum_acc = SIMD[accum_type, simd_width](0)
        for base in range(0, cols, block_span):
            var c = base + tid * simd_width
            if c < cols:
                var v = summed.load[width=simd_width, alignment=align](
                    (row, c)
                ).cast[accum_type]()
                sum_acc += v * v
        var thread_m2 = sum_acc.reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )

        for base in range(0, slice_cols, block_span):
            var j = base + tid * simd_width
            if j < slice_cols:
                var v = summed.load[width=simd_width, alignment=align](
                    (row, slice_start + j)
                ).cast[accum_type]()
                var g = gamma.load[
                    width=simd_width,
                    alignment=align_of[SIMD[gamma_dtype, simd_width]](),
                ]((slice_start + j,)).cast[in_dtype]()
                var normed = ((v * norm_factor).cast[in_dtype]() * g).cast[
                    dtype
                ]()
                output.store((row, j), normed)


# --- Launcher ---


def allreduce_rmsnorm_exit_concat[
    in_dtype: DType,
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
](
    input_buffers: Array[TileTensor[in_dtype, in_layout, in_origin], ngpus],
    gamma: TileTensor[...],
    gate_up: TileTensor[dtype, ...],
    output: TileTensor[mut=True, dtype, ...],
    epsilon: Float32,
    beta: Float32,
    linear_beta: Float32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    my_rank: Optional[Int] = None,
    force_arm: Int = 0,
) raises:
    """Fused allreduce sum + RMSNorm + exit SiTU-and-mul concat.

    Runs the fused kernel at or below the size gate
    (`EXIT_CONCAT_FUSE_THRESHOLD` bytes) and the chain it replaced above it;
    see the module docstring for both arms and why they are bit-identical. A
    vendor-CCL build and a host without P2P access run the chain too, with the
    vendor or the plain allreduce reducing in front of it.

    Parameters:
        in_dtype: The partials' element type; bfloat16 or float32.
        dtype: The gated projection's and the output's type; bfloat16 only.
        ngpus: Number of cooperating devices.
        in_layout: Layout of the input partial TileTensors.
        in_origin: Origin of the input partial TileTensors.

    Args:
        input_buffers: Every device's `[rows, cols]` partial, indexed by
            GLOBAL device rank (the fused arm needs peer access).
        gamma: This device's `[cols]` norm weight, in `in_dtype` or, on a
            float32 input, bfloat16/float16.
        gate_up: This device's `[rows, 2 * half]` shared gate/up projection.
        output: This device's `[rows, cols / ngpus + half]` destination.
        epsilon: The norm's epsilon.
        beta: The SiTU gate's beta.
        linear_beta: The SiTU up-half saturation beta; 0 disables it.
        rank_sigs: Every device's `Signal` pointer, indexed by GLOBAL rank.
            The bytes behind each header hold the collective's staging scratch,
            never its sum.
        ctx: Device context for THIS device.
        my_rank: Optional GLOBAL rank of THIS device in `[0, ngpus)`; defaults
            to the physical device id.
        force_arm: Override for threshold experiments. `0` uses the size gate,
            `1` forces the fused kernel and `2` forces the chain. Every rank of
            a call must agree, exactly as the gate itself requires.

    Raises:
        On any shape, dtype, rank or arm this op does not implement. A world
        size the size gate has no threshold for takes the chain.
    """
    comptime assert ngpus >= 2, "the fused exit collective needs 2+ devices"
    if ngpus > MAX_GPUS:
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: ngpus (",
                ngpus,
                ") exceeds the supported maximum (",
                MAX_GPUS,
                ")",
            )
        )
    comptime assert in_dtype in (
        DType.bfloat16,
        DType.float32,
    ), "allreduce_rmsnorm_exit_concat: the partials must be bf16 or float32"
    comptime assert gamma.dtype == in_dtype or (
        in_dtype == DType.float32
        and gamma.dtype in (DType.bfloat16, DType.float16)
    ), (
        "allreduce_rmsnorm_exit_concat: gamma must be the partials' dtype, or"
        " bfloat16/float16 on float32 partials"
    )
    comptime assert dtype == DType.bfloat16, (
        "allreduce_rmsnorm_exit_concat: the gated projection and the output"
        " must be bfloat16"
    )
    comptime threads_per_block = ARNEC_THREADS

    var global_rank = my_rank.or_else(Int(ctx.device().id))
    if not 0 <= global_rank < ngpus:
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: my_rank (",
                global_rank,
                ") must be in [0, ",
                ngpus,
                ")",
            )
        )

    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[global_rank].dim[last_dim_idx]())
    var rows = input_buffers[global_rank].num_elements() // cols
    var half = Int(gate_up.dim[1]()) // 2

    # Fail fast on every shape the kernel does not implement rather than
    # computing something plausible: a ragged partition would corrupt the
    # rounds silently. The vector's own width is what the dispatch at the end
    # picks a kernel instantiation for.
    if cols % ngpus != 0:
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: cols (",
                cols,
                ") must be a multiple of ngpus (",
                ngpus,
                ")",
            )
        )
    var slice_cols = cols // ngpus
    if (
        Int(output.dim[0]()) != rows
        or Int(output.dim[1]()) != slice_cols + half
    ):
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: the output must be [",
                rows,
                ", ",
                slice_cols + half,
                "], got [",
                Int(output.dim[0]()),
                ", ",
                Int(output.dim[1]()),
                "]",
            )
        )
    if Int(gate_up.dim[0]()) != rows:
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: the gated projection must have",
                " one row per partial row (",
                rows,
                "), got ",
                Int(gate_up.dim[0]()),
            )
        )
    if Int(gamma.dim[0]()) != cols:
        raise Error(
            String(
                "allreduce_rmsnorm_exit_concat: gamma (",
                Int(gamma.dim[0]()),
                ") must have one entry per column (",
                cols,
                ")",
            )
        )

    for i in range(ngpus):
        if input_buffers[i].num_elements() != rows * cols:
            raise Error(
                "allreduce_rmsnorm_exit_concat: every device's partial must"
                " have the same shape"
            )

    # A batch can legitimately arrive with no rows (a DP replica with an empty
    # batch); `enqueue_function` rejects the zero grid that follows, and
    # skipping the barrier pair is group-uniform because `rows` is identical
    # on every rank.
    if rows == 0:
        return

    @inline(.always)
    def launch[simd_width: Int]() raises {imm}:
        """Runs one arm of the launch, vectorizing the rows at `simd_width`.

        Parameters:
            simd_width: Elements per thread's vector. The launcher passes the
                target's own width where it tiles the column shard and the
                shared half evenly, and the scalar width otherwise.
        """
        # A vector straddling a column shard or a ragged shared half would
        # corrupt the rounds silently. The dispatch below only picks a width
        # that divides both, so this cannot fire from there: it guards the
        # invariant rather than the caller.
        if cols % (ngpus * simd_width) != 0:
            raise Error(
                String(
                    "allreduce_rmsnorm_exit_concat: cols (",
                    cols,
                    ") must be a multiple of ngpus * simd_width (",
                    ngpus * simd_width,
                    ")",
                )
            )
        if half % simd_width != 0:
            raise Error(
                String(
                    "allreduce_rmsnorm_exit_concat: the shared half width (",
                    half,
                    ") must be a multiple of simd_width (",
                    simd_width,
                    ")",
                )
            )

        comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
        var src_ptrs = Array[_, ngpus](
            fill_with=lambda (i: Int) -> PtrType: input_buffers[i]
            ._storage.as_imm()
            .as_unsafe_any_origin()
        )

        # The gate's threshold is calibrated for the in-house P2P collective,
        # so a vendor-CCL build and a host without P2P skip it and take the
        # chain, whose reduction is the fallback each of them uses.
        var fuse = False
        comptime if not get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
            if is_p2p_enabled():
                fuse = _fuse_exit_concat[in_dtype, ngpus](rows, cols, ctx)

        # `force_arm` is a test hook that overrides the gate, not a knob.
        if force_arm == 1:
            fuse = True
        elif force_arm == 2:
            fuse = False
        elif force_arm != 0:
            raise Error(
                String(
                    "allreduce_rmsnorm_exit_concat: force_arm (",
                    force_arm,
                    ") must be 0 (gate), 1 (fused) or 2 (chain)",
                )
            )

        if not fuse:
            # --- Chain arm: allreduce, then the norm + exit concat. ---
            # The reduced sum is this device's own workspace rather than the
            # payload behind its `Signal` header: the header only has to hold
            # the plain allreduce's partition staging, so a prefill chunk's
            # signal buffers stay the plain allreduce's size.
            # TODO: The sum this arm stages is unplanned device memory: a
            # prefill chunk parks up to `max_batch_input_tokens` rows of the
            # latent here -- 112 MiB per device at K3's defaults, and growth
            # leaves old mappings behind (see `comm.workspace`). Move it into
            # planned memory, a per-device scratch operand planned like the
            # signal buffers or a discarded `summed` output as
            # `allreduce_rms_norm` has, before this op gains callers on
            # smaller devices.
            var sum_tensor = TileTensor(
                staging_workspace[in_dtype](ctx, rows * cols),
                row_major(Coord(rows, cols)),
            )
            comptime chain_kernel = (
                _allreduce_rmsnorm_exit_concat_chain_kernel[
                    in_dtype=in_dtype,
                    gamma_dtype=gamma.dtype,
                    dtype=dtype,
                    gamma_layout=gamma.LayoutType,
                    gamma_origin=gamma.origin,
                    summed_layout=sum_tensor.LayoutType,
                    summed_origin=sum_tensor.origin,
                    gate_up_layout=gate_up.LayoutType,
                    gate_up_origin=gate_up.origin,
                    out_layout=output.LayoutType,
                    out_origin=output.origin,
                    ngpus=ngpus,
                    simd_width=simd_width,
                    threads_per_block=threads_per_block,
                ]
            )
            # Load the norm kernel before the reduction: a JIT module load
            # blocks on the CUDA context, and the reduction is already spinning
            # at a multi-GPU barrier by then.
            _ = ctx.compile_function[chain_kernel]()

            comptime if get_defined_bool["MODULAR_USE_VENDOR_CCL", False]():
                # `_get_global_comms` has a check-then-create race: two
                # threads seeing null simultaneously would both call
                # ncclCommInitAll and leak one set of communicators. Only
                # device 0 initializes; the others spin-wait.
                if Int(ctx.device().id) == 0:
                    vendor_ccl.init_comms(ngpus)
                else:
                    vendor_ccl.wait_for_comms(ngpus)
                # The vendor collective takes default-engine tiles.
                comptime VendorInput = TileTensor[
                    in_dtype, in_layout, ImmUntrackedOrigin
                ]
                var vendor_inputs = Array[VendorInput, ngpus](
                    fill_with_unrolled=lambda [i: Int]() -> VendorInput: rebind[
                        VendorInput
                    ](input_buffers[i])
                )
                vendor_ccl.allreduce[ngpus=ngpus](
                    vendor_inputs, sum_tensor, rank_sigs, ctx
                )
            else:
                # Without P2P the plain allreduce stages the peers itself,
                # which is why this arm serves a no-P2P host.
                allreduce[ngpus=ngpus](
                    input_buffers,
                    sum_tensor,
                    rank_sigs,
                    ctx,
                    my_rank=global_rank,
                )

            var grid = min(rows, MAX_NUM_BLOCKS_UPPER_BOUND)
            ctx.enqueue_function[chain_kernel](
                sum_tensor,
                gamma,
                gate_up,
                output,
                epsilon,
                beta,
                linear_beta,
                Int32(rows),
                Int32(cols),
                Int32(half),
                Int32(global_rank),
                grid_dim=grid,
                block_dim=threads_per_block,
            )
            return

        # One block per row, grid-strided past the dispatch cap. The cap
        # depends only on the payload, so every rank launches the same grid --
        # which the per-block barriers require.
        comptime sm_version = get_sm_version()
        var payload_bytes = rows * cols * size_of[in_dtype]()
        var max_blocks = min(
            dispatch_select_comm_config[
                ngpus, sm_version, allreduce_tuning_table
            ](payload_bytes).get_num_blocks(),
            MAX_NUM_BLOCKS_UPPER_BOUND,
        )
        var grid_size = min(rows, max_blocks)

        comptime kernel = _allreduce_rmsnorm_exit_concat_kernel[
            in_dtype=in_dtype,
            gamma_dtype=gamma.dtype,
            dtype=dtype,
            gamma_layout=gamma.LayoutType,
            gamma_origin=gamma.origin,
            gate_up_layout=gate_up.LayoutType,
            gate_up_origin=gate_up.origin,
            out_layout=output.LayoutType,
            out_origin=output.origin,
            ngpus=ngpus,
            simd_width=simd_width,
            threads_per_block=threads_per_block,
        ]
        ctx.enqueue_function[kernel](
            src_ptrs,
            gamma,
            gate_up,
            output,
            epsilon,
            beta,
            linear_beta,
            Int32(rows),
            Int32(cols),
            Int32(half),
            rank_sigs,
            Int32(global_rank),
            grid_dim=grid_size,
            block_dim=threads_per_block,
        )

    # The target's own vector width tiles the column shard and the shared half
    # on every shipped shape; a width it cannot tile -- one element per
    # thread, as the activation kernel this replaces fell back to -- keeps the
    # op working for a width the model chooses differently.
    comptime target_simd_width = simd_width_of[in_dtype, target=ctx.target]()
    if (
        cols % (ngpus * target_simd_width) == 0
        and half % target_simd_width == 0
    ):
        launch[target_simd_width]()
    else:
        launch[1]()
