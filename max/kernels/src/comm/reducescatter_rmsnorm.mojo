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
"""Fused reduce-scatter + RMSNorm (bf16 in/out, no FP8).

Reduce-scatters a `[rows, cols]` stream across `ngpus` GPUs and RMSNorms each
owned row, keeping the sum in registers so there is no HBM round-trip between
the two. The sum is rounded to `in_dtype` before the norm so the result is
bit-for-bit the standalone `reduce-scatter -> bf16 shard -> rms_norm` path
(norming the wider f32 sum silently shifts model behavior). Emits two
`[rank_units, cols]` shards: `sum_out` (the reduce-scatter sum / residual
stream) and `normed_out` (its RMSNorm). Inherently `multiply_before_cast=True`.

Factored from the 2-stage AR kernel (`allreduce_residual_rmsnorm.mojo`); keep
the reduce/norm math in sync with it.
"""

from std.collections import InlineArray
from std.math import ceildiv, rsqrt
from std.sys import (
    align_of,
    simd_width_of,
    size_of,
)

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.primitives import block
from layout import Coord, TensorLayout, TileTensor
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .allreduce import allreduce_tuning_table
from .device_query import dispatch_select_comm_config, get_sm_version
from .reducescatter import ReduceScatterConfig, _target_address_space
from .sync import MAX_GPUS, Signal, _multi_gpu_barrier, is_p2p_enabled


# Per-rank shard bytes at/below which `_dispatch_rs_norm` fuses. Fused
# in-register `block.sum` matches `rms_norm_gpu`'s row-count-dependent reduction
# bit-for-bit only up to a crossover: measured M <= 512 at H=6144 (4xMI355, TP4,
# bf16, wo=1.0, mbc=True), diverging 1 bf16 ULP at M=1024. Value = M=512 shard
# (128 rows * 6144 * 2 B). H-specific; recalibrate before fusing another H.
comptime RS_NORM_FUSE_THRESHOLD = 128 * 6144 * size_of[DType.bfloat16]()


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(t"reducescatter_rmsnorm_{in_dtype}_{ngpus}")
def _reducescatter_rmsnorm_kernel[
    mut: Bool,
    origin: Origin[mut=mut],
    GammaLayoutType: TensorLayout,
    normed_origin: MutOrigin,
    NormedLayoutType: TensorLayout,
    sum_origin: MutOrigin,
    SumLayoutType: TensorLayout,
    in_dtype: DType,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    gamma: TileTensor[in_dtype, GammaLayoutType, origin],
    normed_out: TileTensor[mut=True, in_dtype, NormedLayoutType, normed_origin],
    sum_out: TileTensor[mut=True, in_dtype, SumLayoutType, sum_origin],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows: Int,
    cols: Int,
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
):
    """Reduce-scatter each owned row in f32, then RMSNorm it in registers.

    Blocks stride over the ragged partition's local rows; `normed_out`/`sum_out`
    are this rank's `[rank_units, cols]` shards, indexed by local row.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    # 2D stores below use Coord(local_row, col_idx).
    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var tid = thread_idx.x
    var col_idx = tid * simd_width
    var is_valid = col_idx < cols
    var num_blocks = grid_dim.x

    # Ragged partition (matches `ReduceScatterConfig`, NOT ceildiv): remainder
    # rows to low ranks -> drop-in for standalone RS, no OOB at rows%ngpus!=0.
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var my_start = config.rank_unit_start(my_rank)
    var my_count = config.rank_units(my_rank)

    # Round-robin peer order (RS's `circular_add`): peer 0 is self, so accum
    # from 0 over all peers is bit-for-bit RS's `accum = peer[0]` init (AMD
    # non-multimem).
    var ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)
    comptime for i in range(ngpus):
        ptrs[i] = src_ptrs[(my_rank + i) % ngpus]

    # Preload gamma in f32 before the barrier (local data, latency-hidden).
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(col_idx)).cast[
                accum_type
            ]()
            + weight_offset.cast[accum_type]()
        )

    # Start barrier: we P2P-read peers' inputs, so all ranks must be ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # Grid-strided over local owned rows; the ragged partition bounds them, so
    # no `row < rows` guard is needed.
    for local_row in range(block_idx.x, my_count, num_blocks):
        var row = my_start + local_row

        # P2P load from all GPUs, accumulate in f32.
        var accum = SIMD[accum_type, simd_width](0)
        if is_valid:
            var global_elem = row * cols + col_idx
            comptime for gpu_idx in range(ngpus):
                accum += (
                    ptrs[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=align,
                        invariant=True,
                    ](global_elem)
                    .cast[accum_type]()
                )

        # Round to bf16 BEFORE norming: matches the two-launch path (RS emits a
        # bf16 shard); norming the wider f32 sum silently shifts MXFP4 accuracy.
        # Invalid lanes hold 0.
        var reduced = accum.cast[in_dtype]()
        var reduced_f = reduced.cast[accum_type]()
        if is_valid:
            # `reduced` == the standalone reduce-scatter shard (residual stream).
            sum_out.store[width=simd_width](Coord(local_row, col_idx), reduced)

        # Mean-square over the full row. Divide by `cols` (full H) FIRST, add
        # epsilon SECOND (not the shard count; epsilon outside the division).
        var thread_m2 = (reduced_f**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )

        if is_valid:
            # (x * norm) * gamma in f32, cast to in_dtype last: mbc=True.
            var normalized = (reduced_f * norm_factor) * gamma_vec
            normed_out.store[width=simd_width](
                Coord(local_row, col_idx), normalized.cast[in_dtype]()
            )

    # No end barrier (matches the AR+norm kernel): local-GPU output consumers
    # rely on stream ordering (a remote consumer must add its own). The input is
    # subtler -- peers P2P-read this rank's `src_ptrs`, so it must not be reused
    # until the next collective's start barrier gates slow peers.


# --- Launcher ---


def _reducescatter_rmsnorm_launch[
    simd_width: Int,
    in_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
](
    rows: Int,
    cols: Int,
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    ctx: DeviceContext,
) raises:
    """Launch the fused reduce-scatter + RMSNorm kernel."""
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    # Grid capped at rank 0's max shard (ceildiv(rows, ngpus)) so every block
    # has work. Signal scratch only.
    var grid_size = min(ceildiv(rows, ngpus), max_blocks)
    var block_dim = threads_per_block

    comptime assert normed_out.flat_rank >= 2
    comptime assert sum_out.flat_rank >= 2

    comptime kernel = _reducescatter_rmsnorm_kernel[
        mut=gamma.mut,
        origin=gamma.origin,
        GammaLayoutType=gamma.LayoutType,
        normed_origin=normed_out.origin,
        NormedLayoutType=normed_out.LayoutType,
        sum_origin=sum_out.origin,
        SumLayoutType=sum_out.LayoutType,
        in_dtype=in_dtype,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        normed_out,
        sum_out,
        epsilon,
        weight_offset,
        rows,
        cols,
        rank_sigs,
        my_rank,
        grid_dim=grid_size,
        block_dim=block_dim,
    )


# --- Public API ---


def reducescatter_rmsnorm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
](
    input_buffers: InlineArray[
        TileTensor[in_dtype, in_layout, in_origin], ngpus
    ],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Fused reduce-scatter + RMSNorm across `ngpus` GPUs (bf16 in/out).

    Reduce-scatters `input_buffers` (each `[rows, cols]`, one per GPU) along
    rows and RMSNorm-normalizes each owned row. Writes this GPU's
    `[rank_units, cols]` shards: the reduce-scatter sum to `sum_out` and its
    RMSNorm to `normed_out`. `weight_offset` (1.0 for M3, Gemma-style) is folded
    into gamma in f32.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors (peer access
            required).
        normed_out: This GPU's normed output shard `[rank_units, cols]`.
        sum_out: This GPU's reduce-scatter sum shard `[rank_units, cols]` (the
            residual stream).
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.

    Note:
        No end barrier is issued: the outputs are safe to read only on the
        local GPU (a remote-GPU consumer must insert its own). The input
        buffers are P2P-read by peers and become dead after this op; their
        reuse must wait for the next collective's start barrier, so callers
        must not overwrite them before then.
    """
    comptime assert ngpus >= 2, "reducescatter_rmsnorm requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"

    if not is_p2p_enabled():
        raise Error("reducescatter_rmsnorm requires P2P access between GPUs")

    # Compute rows/cols from the full (pre-scatter) input.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Raw peer pointers, origin erased to ImmutAnyOrigin (matches standalone RS).
    var src_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)
    comptime for i in range(ngpus):
        src_ptrs[i] = input_buffers[i].ptr.as_immutable().as_unsafe_any_origin()

    # Each thread owns `simd_width` cols; H=6144 fits the base width
    # (64*8*16=8192) on all targets (no AR two-width dispatch). Assert fit +
    # divisibility below.
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_supported_cols = WARP_SIZE * simd_width * max_warps_per_block
    if cols > max_supported_cols:
        raise Error(
            String(
                "reducescatter_rmsnorm: cols (",
                cols,
                ") exceeds max supported (",
                max_supported_cols,
                ") for the warp-tiling kernel",
            )
        )
    if cols % simd_width != 0:
        raise Error(
            String(
                "reducescatter_rmsnorm: cols (",
                cols,
                ") must be a multiple of simd_width (",
                simd_width,
                ")",
            )
        )

    _reducescatter_rmsnorm_launch[
        simd_width, in_dtype, ngpus, threads_per_block
    ](
        rows,
        cols,
        src_ptrs,
        normed_out,
        sum_out,
        gamma,
        epsilon,
        weight_offset,
        rank_sigs,
        Int(ctx.id()),
        ctx,
    )


# --- Dispatch (A/B selector) ---


def _dispatch_rs_norm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    //,
    two_launch: def() raises capturing -> None,
](
    input_buffers: InlineArray[
        TileTensor[in_dtype, in_layout, in_origin], ngpus
    ],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    threshold: Int = RS_NORM_FUSE_THRESHOLD,
) raises:
    """Runtime-select the fused kernel vs a caller-supplied two-launch path.

    `two_launch` (standalone reduce-scatter + `rms_norm`) is caller-supplied so
    `comm` stays free of `nn` (nn -> comm already exists). The graph op and the
    bench share this one selector.

    Parameters:
        in_dtype: Input/output data type (bf16).
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        two_launch: Caller-supplied standalone reduce-scatter + RMSNorm closure.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        normed_out: This GPU's normed output shard `[rank_units, cols]`.
        sum_out: This GPU's reduce-scatter sum shard `[rank_units, cols]`.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        threshold: Per-rank-bytes fuse threshold; fuse at/below, else
            `two_launch`. Defaults to `RS_NORM_FUSE_THRESHOLD`.
    """
    # Threshold is a bf16 row-count crossover in bytes; another element size
    # maps to the wrong row count and could fuse a diverging shape. Fail loud.
    comptime assert (
        in_dtype == DType.bfloat16
    ), "_dispatch_rs_norm fuse threshold is bf16-calibrated (bf16 in/out only)"

    # Fuse-vs-two-launch MUST be rank-invariant: the paths issue different
    # barriers/grids on the shared `rank_sigs`, so disagreement deadlocks. Gate
    # on rank 0's shard (ceildiv(rows, ngpus)), NEVER `rank_units(ctx.id())` --
    # at `rows % ngpus != 0` low ranks own an extra row and could straddle the
    # threshold.
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = input_buffers[0].num_elements() // cols
    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=rows, unit_numel=cols, threads_per_gpu=0
    )
    var per_rank_bytes = config.rank_units(0) * cols * size_of[in_dtype]()
    var use_fused = per_rank_bytes <= threshold

    if use_fused:
        reducescatter_rmsnorm(
            input_buffers,
            normed_out,
            sum_out,
            gamma,
            epsilon,
            weight_offset,
            rank_sigs,
            ctx,
        )
    else:
        two_launch()
