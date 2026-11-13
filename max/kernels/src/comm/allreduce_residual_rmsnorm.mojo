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
"""Fused allreduce + RMSNorm (+ optional FP8 quantization) kernel.

Combines P2P allreduce, RMSNorm normalization, and optional FP8 dynamic
quantization into a single kernel launch. Data stays in registers
throughout, with no global memory intermediate between allreduce and RMSNorm.

Quantization is fused in only when the output dtype differs from the input
dtype (`out_dtype != in_dtype`). When `out_dtype == in_dtype` (e.g. a bf16
input with a bf16 output) the FP8 phases (per-row max, dynamic scale, and
quantize) are skipped at compile time, the normalized value is written
directly in the input dtype, and no per-row scale is produced.

Four dispatch paths:

Lamport (latency-bound, non-quantizing payloads): the barrier-free Lamport
  kernel in `allreduce_lamport_rmsnorm.mojo`, selected by the same byte
  buckets as the plain allreduce's Lamport path. Each rank pushes its packs
  into its peers' Lamport regions and polls its own, so no start barrier and no
  peer-input reads are needed.

1-stage (small payloads): Each thread loads from all GPUs, accumulates
  in float32 registers, then applies RMSNorm + FP8 quantization. Simple
  but O(N x ngpus) P2P traffic.

2-stage (large payloads, non-residual or medium residual): Single kernel
  with two stages separated by a per-block barrier:
    Stage 1 (RS + RMSNorm + FP8): each block reduces rows in its partition
      from all GPUs in f32 registers, then normalizes and quantizes to FP8
      immediately, with no f32 scratch. Writes compact fp8 data + per-row scale
      (+ optional bf16 residual) to local scratch.
    Stage 2 (AG copy): each block reads compact fp8/scale/bf16 from the
      owning GPU's scratch (P2P) and writes to local output buffers. No
      compute, just copies of data that is 4x smaller than f32.
  Both stages use the same row-to-block mapping, so the per-block barrier
  correctly synchronizes all data dependencies. Total P2P traffic is
  N * sizeof(in_dtype) for Stage 1 (same as 1-stage) plus
  N * sizeof(fp8) [+ N * sizeof(in_dtype) if residual] for Stage 2.
  The AG phase copies fp8 (1 byte/elem) instead of f32 (4 bytes/elem),
  reducing Stage 2 bandwidth by ~4x for the non-residual case.

Split (large residual payloads): Two separate kernel launches:
  allreduce with add epilogue followed by fused rmsnorm+fp8. Avoids
  carrying bf16 residual data through scratch buffers in both stages,
  which nearly doubles Stage 2 copy bandwidth in the fused path. The
  caller-provided residual_output buffer serves as the intermediate,
  so no extra allocation is needed.

Each block processes multiple rows via a grid-strided loop, allowing
row counts beyond the hardware-tuned block limit. Gamma weights are
preloaded once and reused across all rows in the loop.
"""

from std.builtin.device_passable import DevicePassable
from std.collections import Array
from std.math import align_up, ceildiv, rsqrt
from std.sys import (
    align_of,
    get_defined_bool,
    get_defined_int,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, default_accelerator
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import (
    Coord,
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.tile_tensor import _ComptimeConditionalTileTensor
from std.utils import StaticTuple
from std.utils.numerics import get_accum_type


from internal_utils.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

from .rms_norm_fp8 import rms_norm_fused_fp8

from .allreduce import (
    AllReduceAlgorithm,
    _load_peers_rotated,
    allreduce,
    allreduce_tuning_table,
)
from .allreduce_lamport_rmsnorm import (
    _lamport_allreduce_rmsnorm_launch,
    lamport_allreduce_rmsnorm_fits,
)
from .allreduce_mcast_rmsnorm import (
    MCAR_GROUP,
    allreduce_mcast_rmsnorm,
    mcar_supports_cols,
)
from .device_query import get_sm_version, dispatch_select_comm_config
from .lamport import Lamport
from .multicast_window import McarWindowRank, mcar_window_rows
from .reducescatter import _target_address_space
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)


# --- PDL helpers ---


@inline(.always)
def _pdl_wait[pdl_level: PDLLevel]():
    """Waits for the predecessor grid and, for `OVERLAP_AT_BEGINNING`, releases
    the dependent grid right away.

    The dependent's own `wait_on_dependent_grids` blocks until this grid has
    completed and its stores are visible, so releasing at the start only
    decides when the dependent becomes resident. Releasing after the wait
    keeps at most one dependent resident ahead of the running grid. Every
    thread of the block must call it.
    """
    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()
        comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
            launch_dependent_grids()


@inline(.always)
def _pdl_release_at_end[pdl_level: PDLLevel]():
    """Releases the dependent grid after the last global access, unless
    `_pdl_wait` already did.

    The trigger follows the barrier that retires the peers' P2P reads of this
    GPU's input, which the next kernel may overwrite. Only the first warp spins
    on the peers' flags inside that barrier, so the block syncs again before
    any thread releases the dependent. Every thread of the block must call it.
    """
    comptime if (
        pdl_level > PDLLevel.OFF and pdl_level != PDLLevel.OVERLAP_AT_BEGINNING
    ):
        barrier()
        launch_dependent_grids()


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(
    t"allreduce_rmsnorm_fp8_warp_tiling_{in_dtype}_{out_dtype}_{ngpus}_{has_residual}_{multiply_before_cast}",
)
def _allreduce_rmsnorm_fp8_kernel_warp_tiling[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ResidualLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualOutLayoutType: TensorLayout,
    residual_out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel,
    OutputFnType: ImplicitlyCopyable
    & DevicePassable
    & RegisterPassable
    & def[width: SIMDLength](Int, Int, SIMD[out_dtype, width]) -> None,
    *,
    GammaEngine: TensorEngine,
    ScaleBufferEngine: TensorEngine,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[in_dtype, LayoutType, origin, Engine=GammaEngine],
    scale_buffer: TileTensor[
        mut=True,
        scales_dtype,
        ScaleLayoutType,
        scale_origin,
        Engine=ScaleBufferEngine,
    ],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows: Int32,
    cols: Int32,
    scale_ub: Float32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int32,
    residual: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualLayoutType,
        residual_origin,
        engaged=has_residual,
    ],
    residual_output: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualOutLayoutType,
        residual_out_origin,
        engaged=has_residual,
    ],
    output_fn: OutputFnType,
):
    """Fused allreduce + RMSNorm + FP8 kernel using warp-tiling.

    Each thread block processes one or more rows via a persistent row loop
    (stride = grid_dim). Data stays in registers across all four phases:
    P2P load+reduce, mean-square, normalize+max, quantize+write.
    Gamma weights are preloaded once and reused across all rows.
    """
    var _rows = Int(rows)
    var _cols = Int(cols)
    var _my_rank = Int(my_rank)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    # Provide evidence that flat_rank >= 1 for the Coord(...) loads below.
    comptime assert gamma.flat_rank >= 1
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...)
    # loads/stores on residual and residual_output below.
    comptime assert residual.T.flat_rank >= 2
    comptime assert residual_output.T.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # Fuse FP8 quantization only when the output dtype differs from the input.
    # When out_dtype == in_dtype the max/scale/quantize phases are skipped.
    comptime quantize = out_dtype != in_dtype

    var tid = thread_idx.x
    var idx = tid * simd_width
    var is_valid = idx < _cols

    # Peers P2P-read the input the predecessor (the producing GEMM) writes, and
    # the start barrier below tells them it is ready. Wait first so the barrier
    # is never signalled before those stores land.
    _pdl_wait[pdl_level]()

    # Preload gamma weights into registers. Gamma is not guaranteed to be a
    # checkpoint buffer: the graph can pass a computed value (for example a
    # cast of the weight), so it is read only after the wait. Its latency hides
    # behind the start barrier. It depends only on the column index, so it is
    # loaded once and reused across all _rows.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align]((idx,)).cast[
                accum_type
            ]()
            + _weight_offset.cast[accum_type]()
        )

    # Barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # Input pointers in global rank order on every rank, so every rank rounds
    # the reduced sum identically; see `_load_peers_rotated` for the load order.
    var ptrs = src_ptrs.copy()
    comptime if ngpus == 2:
        # Two terms sum to the same bits in either order, so keep the
        # round-robin order, which also balances the reads.
        ptrs = Array[_, ngpus](
            fill_with_unrolled=lambda [i: Int]() -> ImmPointer[
                Scalar[in_dtype], ImmutAnyOrigin
            ]: src_ptrs[(_my_rank + i) % ngpus]
        )

    # Row loop: each block processes _rows with stride = grid_dim.
    # For _rows <= grid_dim, the loop body runs exactly once per block.
    var num_blocks = grid_dim.x
    for row in range(block_idx.x, _rows, num_blocks):
        # Phase 0: P2P load from all GPUs + accumulate in float32 regs.
        # Gamma load latency from above is hidden during P2P stalls.
        var vec_data = SIMD[accum_type, simd_width](0)
        if is_valid:
            var elem_idx = row * _cols + idx

            comptime if (is_nvidia_gpu() or is_amd_gpu()) and ngpus > 2:
                comptime for r in range(ngpus):
                    if _my_rank == r:
                        var vals = _load_peers_rotated[
                            ngpus, r, simd_width, alignment=simd_width
                        ](ptrs, elem_idx)
                        var accum = SIMD[accum_type, simd_width](0)
                        comptime for g in range(ngpus):
                            accum += vals[g].cast[accum_type]()
                        vec_data = accum
            else:
                comptime for gpu_idx in range(ngpus):
                    vec_data += (
                        ptrs[gpu_idx]
                        .unsafe_address_space_cast[_target_address_space]()
                        .unsafe_load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](elem_idx)
                        .cast[accum_type]()
                    )

            # Add residual and write pre-norm sum (compile-time gated).
            comptime if has_residual:
                vec_data += (
                    residual[]
                    .load[width=simd_width]((row, idx))
                    .cast[accum_type]()
                )
                # Write bf16 pre-normalization sum for the residual stream.
                residual_output[].store((row, idx), vec_data.cast[in_dtype]())

        # Phase 1: Compute mean-square.
        var thread_m2 = (vec_data**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](_cols)) + _epsilon.cast[accum_type]()
        )

        # Phase 2: Normalize (preloaded gamma, no global load).
        var normalized = SIMD[accum_type, simd_width](0)
        if is_valid:
            normalized = (vec_data * norm_factor) * gamma_vec

        comptime if quantize:
            # Phase 2b (FP8 only): find row max and compute the dynamic scale.
            var thread_max = Scalar[accum_type](0)
            if is_valid:
                thread_max = abs(normalized).reduce_max()
            var row_max = block.max[
                block_size=threads_per_block, broadcast=True
            ](thread_max)
            var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                out_dtype
            ](row_max, _scale_ub)
            if tid == 0:
                scale_buffer[row] = scale_factor

            # Phase 3: Quantize and write (normalized values in registers).
            if is_valid:
                var output_fp8 = fp8_quantize[out_dtype](
                    normalized, scale_factor_recip
                )
                output_fn[simd_width](row, idx, output_fp8)
        else:
            # Phase 3 (no quant): write the normalized value directly in
            # out_dtype (== in_dtype). No per-row scale is produced.
            if is_valid:
                comptime if multiply_before_cast:
                    output_fn[simd_width](
                        row, idx, normalized.cast[out_dtype]()
                    )
                else:
                    output_fn[simd_width](
                        row,
                        idx,
                        (
                            (vec_data * norm_factor).cast[in_dtype]()
                            * gamma_vec.cast[in_dtype]()
                        ).cast[out_dtype](),
                    )

    # Peers read this GPU's input over P2P, so hold it until they finish.
    _multi_gpu_barrier[ngpus, is_start=False](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    _pdl_release_at_end[pdl_level]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(
    t"allreduce_rmsnorm_fp8_2stage_{in_dtype}_{out_dtype}_{ngpus}_{has_residual}_{multiply_before_cast}",
)
def _allreduce_rmsnorm_fp8_kernel_2stage[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_origin: MutOrigin,
    ScaleLayoutType: TensorLayout,
    ResidualLayoutType: TensorLayout,
    residual_origin: Origin,
    ResidualOutLayoutType: TensorLayout,
    residual_out_origin: MutOrigin,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel,
    OutputFnType: ImplicitlyCopyable
    & DevicePassable
    & RegisterPassable
    & def[width: SIMDLength](Int, Int, SIMD[out_dtype, width]) -> None,
    *,
    GammaEngine: TensorEngine,
    ScaleBufferEngine: TensorEngine,
](
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    gamma: TileTensor[in_dtype, LayoutType, origin, Engine=GammaEngine],
    scale_buffer: TileTensor[
        mut=True,
        scales_dtype,
        ScaleLayoutType,
        scale_origin,
        Engine=ScaleBufferEngine,
    ],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rows: Int32,
    cols: Int32,
    scale_ub: Float32,
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int32,
    residual: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualLayoutType,
        residual_origin,
        engaged=has_residual,
    ],
    residual_output: _ComptimeConditionalTileTensor[
        in_dtype,
        ResidualOutLayoutType,
        residual_out_origin,
        engaged=has_residual,
    ],
    output_fn: OutputFnType,
):
    """Single-kernel 2-stage fused RS + RMSNorm + FP8 + AG.

    Stage 1 (RS + RMSNorm + FP8): Each block reduces rows in its partition
    from all GPUs in f32 registers, normalizes, quantizes to FP8, and writes
    compact results (fp8 + scale + optional bf16 residual) to local scratch.
    No f32 scratch is written: data stays in registers through RS → Norm → FP8.

    Inter-stage: Per-block barrier with release/acquire fence ensures all
    scratch writes from block B on every GPU are visible before block B
    on any GPU starts Stage 2.

    Stage 2 (AG copy): Each block iterates over each GPU's partition and
    reads fp8/scale/bf16 from that GPU's scratch via P2P, then writes to
    local output buffers. Consecutive reads from the same peer improve
    NVLink pipelining. No compute, just copies of compact data.

    Scratch layout per GPU (after Signal header):
      [fp8:      rows_per_rank * cols bytes]
      [scales:   align_up(rows_per_rank * sizeof(scales_dtype), simd_width) bytes]
      [residual: rows_per_rank * cols * sizeof(in_dtype) bytes] (if has_residual)

    The scale section is padded to a multiple of simd_width bytes so the
    residual section starts at a simd_width-byte aligned offset (required
    for SIMD residual loads/stores with alignment=simd_width).

    Minimum signal buffer size per GPU:
        sizeof(Signal)
        + rows_per_rank * cols                                          (fp8 data)
        + align_up(rows_per_rank * sizeof(scales_dtype), simd_width)   (scales + pad)
        + rows_per_rank * cols * sizeof(in_dtype)                       (if has_residual)

    Both stages iterate over local rows within each partition using
    `for local_row in range(block_idx.x, rows_per_rank, num_blocks)`.
    Block B processes local_rows {B, B+num_blocks, B+2*num_blocks, ...}
    in both stages, so the per-block barrier correctly synchronizes:
    block B on GPU C writes those local rows in Stage 1, and block B
    on any GPU reads them from GPU C's scratch in Stage 2.
    """
    var _rows = Int(rows)
    var _cols = Int(cols)
    var _my_rank = Int(my_rank)
    var _epsilon = Scalar[in_dtype](epsilon)
    var _weight_offset = Scalar[in_dtype](weight_offset)
    var _scale_ub = Scalar[scales_dtype](scale_ub)
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert scale_buffer.flat_rank == 1, "scale_buffer must have rank 1"
    # Provide evidence that flat_rank >= 1 for the Coord(...) loads below.
    comptime assert gamma.flat_rank >= 1
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...)
    # loads/stores on residual and residual_output below.
    comptime assert residual.T.flat_rank >= 2
    comptime assert residual_output.T.flat_rank >= 2
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()
    # Fuse FP8 quantization only when the output dtype differs from the input.
    # When out_dtype == in_dtype the scratch carries the normalized value in
    # out_dtype (== in_dtype) and no scale section is reserved or written.
    comptime quantize = out_dtype != in_dtype

    var tid = thread_idx.x
    var col_idx = tid * simd_width
    var is_valid = col_idx < _cols
    var num_blocks = grid_dim.x

    # Row-aligned partitioning: each GPU owns ceildiv(_rows, ngpus) _rows.
    var rows_per_rank = ceildiv(_rows, ngpus)
    var my_row_start = _my_rank * rows_per_rank

    # --- Scratch pointers (after Signal header) ---
    # Layout: [fp8_data | scales (padded) | bf16_residual (optional)]
    # See kernel docstring for the full size formula.
    var fp8_per_rank = rows_per_rank * _cols
    comptime if quantize:
        assert (
            fp8_per_rank % 4 == 0
        ), "fp8 scratch must be 4-byte aligned for scale stores"

    # Round up the scale element count to the next multiple of
    # (simd_width / sizeof(scales_dtype)) so that the residual scratch
    # pointer (scale_ptr + scale_pad_elements) is simd_width-byte aligned.
    # Without this, rows_per_rank * sizeof(scales_dtype) may not be
    # divisible by simd_width, causing a misaligned SIMD residual store
    # when ceildiv(_rows, ngpus) % (simd_width / sizeof(scales_dtype)) != 0.
    # When not quantizing there is no scale section, so the residual scratch
    # follows the output scratch directly (scale_pad_elements = 0).
    comptime scales_per_simd_chunk = simd_width // size_of[scales_dtype]()
    var scale_pad_elements: Int
    comptime if quantize:
        scale_pad_elements = (
            ceildiv(rows_per_rank, scales_per_simd_chunk)
            * scales_per_simd_chunk
        )
    else:
        scale_pad_elements = 0

    # Local scratch pointers (for writes in Stage 1).
    # +1 advances the Signal pointer by sizeof(Signal) bytes, stepping
    # past the Signal header to reach the scratch region of the buffer.
    var scratch_fp8 = (
        rank_sigs[_my_rank].unsafe_offset(1).unsafe_bitcast[Scalar[out_dtype]]()
    )
    var scratch_scale = scratch_fp8.unsafe_offset(fp8_per_rank).unsafe_bitcast[
        Scalar[scales_dtype]
    ]()
    var scratch_residual = scratch_scale.unsafe_offset(
        scale_pad_elements
    ).unsafe_bitcast[Scalar[in_dtype]]()

    # P2P scratch pointers for all GPUs (for reads in Stage 2).
    # +1 advances by sizeof(Signal) bytes (see local scratch note above).
    comptime Fp8PtrType = MutPointer[Scalar[out_dtype], MutAnyOrigin]
    var fp8_ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> Fp8PtrType: (
            rank_sigs[i].unsafe_offset(1)
        ).unsafe_bitcast[Scalar[out_dtype]]()
    )

    comptime ScalePtrType = MutPointer[Scalar[scales_dtype], MutAnyOrigin]
    var scale_ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> ScalePtrType: (
            rank_sigs[i]
            .unsafe_offset(1)
            .unsafe_bitcast[Scalar[out_dtype]]()
            .unsafe_offset(fp8_per_rank)
        ).unsafe_bitcast[Scalar[scales_dtype]]()
    )
    var residual_ptrs = Array[
        MutPointer[Scalar[in_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime if has_residual:
        comptime for i in range(ngpus):
            residual_ptrs[i] = (
                scale_ptrs[i].unsafe_offset(scale_pad_elements)
            ).unsafe_bitcast[Scalar[in_dtype]]()

    # Round-robin P2P input pointers for NVLink load-balancing.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var ptrs = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: src_ptrs[
            (_my_rank + i) % ngpus
        ]
    )

    # The predecessor writes the input peers P2P-read below and the residual,
    # so wait before signalling the start barrier.
    _pdl_wait[pdl_level]()

    # Preload gamma weights into registers BEFORE the start barrier. Gamma is
    # not guaranteed to be a checkpoint buffer: the graph can pass a computed
    # value (for example a cast of the weight), so it is read only after the
    # wait. Its latency hides behind the start barrier.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align]((col_idx,)).cast[
                accum_type
            ]()
            + _weight_offset.cast[accum_type]()
        )

    # Start barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # === Stage 1: RS + RMSNorm + FP8 (partition _rows only) ===
    # Each block iterates over its partition _rows directly. With
    # grid_dim <= rows_per_rank, every block has at least one row.
    for local_row in range(block_idx.x, rows_per_rank, num_blocks):
        var row = my_row_start + local_row
        if row < _rows:
            # P2P load from all GPUs + accumulate in f32 registers.
            var accum = SIMD[accum_type, simd_width](0)
            if is_valid:
                var global_elem = row * _cols + col_idx

                comptime for gpu_idx in range(ngpus):
                    accum += (
                        ptrs[gpu_idx]
                        .unsafe_address_space_cast[_target_address_space]()
                        .unsafe_load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](global_elem)
                        .cast[accum_type]()
                    )

                # Add residual and write residual to scratch (compile-time gated).
                comptime if has_residual:
                    accum += (
                        residual[]
                        .load[width=simd_width]((row, col_idx))
                        .cast[accum_type]()
                    )
                    var local_elem = local_row * _cols + col_idx
                    scratch_residual.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_store[width=simd_width, alignment=simd_width](
                        local_elem, accum.cast[in_dtype]()
                    )

            # RMSNorm: compute mean-square.
            var thread_m2 = (accum**2).reduce_add()
            var row_m2 = block.sum[
                block_size=threads_per_block, broadcast=True
            ](thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](_cols))
                + _epsilon.cast[accum_type]()
            )

            # Normalize.
            var normalized = SIMD[accum_type, simd_width](0)
            if is_valid:
                normalized = (accum * norm_factor) * gamma_vec

            comptime if quantize:
                # Find max for FP8 scale, write scale + fp8 to local scratch.
                var thread_max = Scalar[accum_type](0)
                if is_valid:
                    thread_max = abs(normalized).reduce_max()

                var row_max = block.max[
                    block_size=threads_per_block, broadcast=True
                ](thread_max)
                var scale_factor, scale_factor_recip = (
                    compute_dynamic_fp8_scale[out_dtype](row_max, _scale_ub)
                )

                # Write scale to local scratch.
                if tid == 0:
                    scratch_scale.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_store(local_row, scale_factor)

                # Quantize and write fp8 to local scratch.
                if is_valid:
                    var output_fp8 = fp8_quantize[out_dtype](
                        normalized, scale_factor_recip
                    )
                    var local_elem = local_row * _cols + col_idx
                    scratch_fp8.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_store[width=simd_width, alignment=simd_width](
                        local_elem, output_fp8
                    )
            else:
                # No quant: write normalized value (out_dtype == in_dtype) to
                # local scratch. No scale is computed or written.
                if is_valid:
                    var local_elem = local_row * _cols + col_idx
                    var out_vec: SIMD[out_dtype, simd_width]
                    comptime if multiply_before_cast:
                        out_vec = normalized.cast[out_dtype]()
                    else:
                        out_vec = (
                            (accum * norm_factor).cast[in_dtype]()
                            * gamma_vec.cast[in_dtype]()
                        ).cast[out_dtype]()
                    scratch_fp8.unsafe_address_space_cast[
                        _target_address_space
                    ]().unsafe_store[width=simd_width, alignment=simd_width](
                        local_elem, out_vec
                    )

    # Per-block barrier with fence: block B waits for block B on all GPUs.
    # `syncthreads` plus the release/acquire atomics inside `_multi_gpu_barrier`
    # ensure all of block B's Stage 1 writes on every GPU are visible before
    # Stage 2 reads them.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, rank_sigs[_my_rank], _my_rank
    )

    # === Stage 2: All-Gather (lightweight P2P copy, no compute) ===
    # Batch reads by GPU: consecutive reads from the same peer improve
    # NVLink pipelining. No runtime owner_rank division per row.
    comptime for gpu in range(ngpus):
        var gpu_row_start = gpu * rows_per_rank
        var gpu_row_end = min(gpu_row_start + rows_per_rank, _rows)
        var gpu_rows = gpu_row_end - gpu_row_start
        for local_row in range(block_idx.x, gpu_rows, num_blocks):
            var row = gpu_row_start + local_row
            var local_elem = local_row * _cols + col_idx

            # Read output value (fp8 when quantizing, else out_dtype) from
            # gpu's scratch and write to output.
            if is_valid:
                var out_val = (
                    fp8_ptrs[gpu]
                    .unsafe_address_space_cast[_target_address_space]()
                    .unsafe_load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=True,
                    ](local_elem)
                )
                output_fn[simd_width](row, col_idx, out_val)

            # Thread 0: read scale from gpu's scratch → scale_buffer (FP8 only).
            comptime if quantize:
                if tid == 0:
                    scale_buffer[row] = (
                        scale_ptrs[gpu]
                        .unsafe_address_space_cast[_target_address_space]()
                        .unsafe_load[invariant=True](local_row)
                    )

            # Read bf16 residual from gpu's scratch (compile-time gated).
            comptime if has_residual:
                if is_valid:
                    var bf16_val = (
                        residual_ptrs[gpu]
                        .unsafe_address_space_cast[_target_address_space]()
                        .unsafe_load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](local_elem)
                    )
                    residual_output[].store((row, col_idx), bf16_val)

    # No end barrier. Every block waited at the mid barrier for its peer
    # blocks to finish Stage 1, so all P2P reads of the inputs are done by
    # the time this kernel exits. Stage 2 only reads peer scratch inside
    # the Signal buffer, which the next collective's start barrier protects.

    _pdl_release_at_end[pdl_level]()


# --- Launcher ---

comptime _ZeroSizedLayout = type_of(row_major(Idx[0], Idx[0]))


def _allreduce_rmsnorm_fp8_launch[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int,
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Launch the fused allreduce + RMSNorm + FP8 kernel."""
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    var grid_dim = min(rows, max_blocks)
    var block_dim = threads_per_block

    # Provide evidence that flat_rank >= 2 for output.store((..., ...)).
    comptime assert output.flat_rank >= 2

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](row: Int, col: Int, val: SIMD[out_dtype, width]) {var output}:
        output.store((row, col), val)

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_warp_tiling[
        mut=gamma.mut,
        origin=gamma.origin,
        LayoutType=gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_origin=scale_output.origin,
        ScaleLayoutType=scale_output.LayoutType,
        ResidualLayoutType=residual.T.LayoutType,
        residual_origin=residual.T.origin,
        ResidualOutLayoutType=residual_output.T.LayoutType,
        residual_out_origin=residual_output.T.origin,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        multiply_before_cast=multiply_before_cast,
        pdl_level=pdl_level,
        OutputFnType=type_of(output_fn),
        GammaEngine=gamma.Engine,
        ScaleBufferEngine=scale_output.Engine,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        scale_output,
        epsilon.cast[.float32](),
        weight_offset,
        Int32(rows),
        Int32(cols),
        Float32(scale_ub),
        rank_sigs,
        Int32(my_rank),
        residual,
        residual_output,
        output_fn,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(pdl_level),
    )


def _allreduce_rmsnorm_fp8_launch_2stage[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int,
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Launch the single-kernel 2-stage fused RS + RMSNorm + FP8 + AG.

    A single kernel with two stages separated by a per-block barrier:
      Stage 1 (RS + RMSNorm + FP8): each block reduces rows in its partition
        from all GPUs in f32 registers, normalizes, quantizes to FP8, and
        writes compact results (fp8 + scale + optional bf16) to local scratch.
      Stage 2 (AG copy): each block iterates over each GPU's partition and
        reads compact fp8/scale/bf16 via P2P, then writes to local output.

    Grid is sized to rows_per_rank (not total rows) so every block has
    work in Stage 1. Both stages use the same local_row-to-block mapping,
    so the per-block barrier correctly synchronizes all data dependencies.
    """
    # Cap grid to rows_per_rank so every block has work in Stage 1.
    # With `rows` blocks, 1-1/ngpus of them would be idle during the
    # RS+Norm+FP8 stage (e.g. 75% idle at 4 GPUs). Using rows_per_rank
    # ensures 100% block utilization in both stages.
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = dispatch_select_comm_config[
        ngpus, sm_version, allreduce_tuning_table
    ](payload_bytes).get_num_blocks()
    var grid_dim = min(ceildiv(rows, ngpus), max_blocks)
    var block_dim = threads_per_block

    # Fuse FP8 quantization only when the output dtype differs from the input.
    comptime quantize = out_dtype != in_dtype

    # Validate that the signal buffer scratch layout is correctly aligned.
    # The 2-stage kernel places the output data immediately after the Signal
    # header (fp8 when quantizing, else out_dtype == in_dtype). When
    # quantizing, the scale values (padded to simd_width bytes) follow; with no
    # quantization there is no scale section. The residual (if has_residual)
    # comes last. For scale values to be properly aligned, the output block
    # size must be a multiple of sizeof(scales_dtype); the scale section is
    # padded to simd_width bytes so the residual section is simd_width-byte
    # aligned.
    # Callers must ensure each rank_sigs[i] buffer has at least:
    #   sizeof(Signal) + output_scratch [+ scale_scratch_padded if quantizing]
    # (plus residual_scratch = rows_per_rank * cols * sizeof(in_dtype) if
    # has_residual). See the kernel docstring for the full formula.
    var _rows_per_rank = ceildiv(rows, ngpus)
    var _out_scratch = _rows_per_rank * cols * size_of[out_dtype]()
    var _scale_scratch = 0
    comptime if quantize:
        var _scale_scratch_raw = _rows_per_rank * size_of[scales_dtype]()
        # Pad scale section to simd_width bytes (matches scale_pad_elements).
        _scale_scratch = align_up(_scale_scratch_raw, simd_width)
    var _min_buf = size_of[Signal]() + _out_scratch + _scale_scratch
    comptime if has_residual:
        _min_buf += _rows_per_rank * cols * size_of[in_dtype]()
    comptime if quantize:
        assert _out_scratch % size_of[scales_dtype]() == 0, String(
            t"2-stage: output scratch ({_out_scratch} B) must be a"
            t" multiple of sizeof(scales_dtype) for scale pointer"
            t" alignment; rank_sigs[i] must be >= {_min_buf} bytes"
        )
    comptime if has_residual:
        assert (_out_scratch + _scale_scratch) % simd_width == 0, String(
            t"2-stage: residual scratch offset (output={_out_scratch} B +"
            t" scales_padded={_scale_scratch} B = "
            t"{_out_scratch + _scale_scratch} B) must be a multiple of"
            t" simd_width ({simd_width} B) for SIMD residual stores"
        )

    # Provide evidence that flat_rank >= 2 for output.store((..., ...)).
    comptime assert output.flat_rank >= 2

    @inline(.always)
    def output_fn[
        width: SIMDLength
    ](row: Int, col: Int, val: SIMD[out_dtype, width]) {var output}:
        output.store((row, col), val)

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_2stage[
        mut=gamma.mut,
        origin=gamma.origin,
        LayoutType=gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_origin=scale_output.origin,
        ScaleLayoutType=scale_output.LayoutType,
        ResidualLayoutType=residual.T.LayoutType,
        residual_origin=residual.T.origin,
        ResidualOutLayoutType=residual_output.T.LayoutType,
        residual_out_origin=residual_output.T.origin,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        multiply_before_cast=multiply_before_cast,
        pdl_level=pdl_level,
        OutputFnType=type_of(output_fn),
        GammaEngine=gamma.Engine,
        ScaleBufferEngine=scale_output.Engine,
    ]
    ctx.enqueue_function[kernel](
        src_ptrs,
        gamma,
        scale_output,
        epsilon.cast[.float32](),
        weight_offset,
        Int32(rows),
        Int32(cols),
        Float32(scale_ub),
        rank_sigs,
        Int32(my_rank),
        residual,
        residual_output,
        output_fn,
        grid_dim=grid_dim,
        block_dim=block_dim,
        attributes=pdl_launch_attributes(pdl_level),
    )


# --- Split (2-kernel) path ---


@inline(.always)
def _rms_norm_rows[
    dtype: DType,
    simd_width: Int,
    threads_per_block: Int,
    multiply_before_cast: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    epsilon: Float32,
    weight_offset: Scalar[dtype],
):
    comptime accum_type = get_accum_type[dtype]()
    comptime stride = threads_per_block * simd_width
    var first_col = thread_idx.x * simd_width

    for row in range(block_idx.x, rows, grid_dim.x):
        var row_in = input.unsafe_offset(row * cols)
        var thread_m2 = Scalar[accum_type](0)
        for col in range(first_col, cols, stride):
            var x = row_in.unsafe_load[width=simd_width](col).cast[accum_type]()
            thread_m2 += (x * x).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            row_m2 / Scalar[accum_type](cols) + epsilon.cast[accum_type]()
        )

        var row_out = output.unsafe_offset(row * cols)
        for col in range(first_col, cols, stride):
            var x = row_in.unsafe_load[width=simd_width](col).cast[accum_type]()
            var gamma_vec = (
                gamma.unsafe_load[width=simd_width](col).cast[accum_type]()
                + weight_offset.cast[accum_type]()
            )
            var y: SIMD[dtype, simd_width]
            comptime if multiply_before_cast:
                y = (x * norm_factor * gamma_vec).cast[dtype]()
            else:
                y = (x * norm_factor).cast[dtype]() * gamma_vec.cast[dtype]()
            row_out.unsafe_store[width=simd_width](col, y)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
@__name(
    t"rms_norm_rows_{dtype}_{simd_width}_{multiply_before_cast}",
)
def _rms_norm_rows_kernel[
    dtype: DType,
    simd_width: Int,
    threads_per_block: Int,
    multiply_before_cast: Bool,
](
    output: MutPointer[Scalar[dtype], MutAnyOrigin],
    input: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int32,
    cols: Int32,
    epsilon: Float32,
    weight_offset: Scalar[dtype],
):
    """RMSNorm over each row of a local `[rows, cols]` buffer, one block per
    row, with the fused kernels' gamma and cast-order arithmetic."""
    var _rows = Int(rows)
    var _cols = Int(cols)
    if _cols % simd_width == 0:
        _rms_norm_rows[
            dtype, simd_width, threads_per_block, multiply_before_cast
        ](output, input, gamma, _rows, _cols, epsilon, weight_offset)
    else:
        _rms_norm_rows[dtype, 1, threads_per_block, multiply_before_cast](
            output, input, gamma, _rows, _cols, epsilon, weight_offset
        )


def _launch_split_allreduce_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel(),
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output_2d: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int,
    ctx: DeviceContext,
    residual: TileTensor[mut=False, in_dtype, ...],
    residual_output: TileTensor[mut=True, in_dtype, ...],
) raises:
    """Two-kernel path: allreduce+add epilogue, then RMSNorm [+ FP8].

    Faster than fused 2-stage for large quantizing with-residual payloads
    because it avoids carrying bf16 residual data through scratch buffers.
    It is also the only path without P2P access: the allreduce then falls
    back to staged copies, and the norm reads only local memory.
    """
    comptime quantize = out_dtype != in_dtype
    # Construct TileTensor inputs for allreduce.
    comptime _TT = type_of(TileTensor(src_ptrs[0], row_major(rows, cols)))
    var input_buffers = Array[_, ngpus](
        fill_with=lambda (i: Int) -> _TT: TileTensor(
            src_ptrs[i], row_major(rows, cols)
        )
    )

    var residual_2d = residual.reshape(row_major(rows, cols))
    var residual_output_2d = residual_output.reshape(row_major(rows, cols))

    # Define input_fn for RMSNorm (reads from residual_output after allreduce).
    @inline(.always)
    def input_fn[
        width: Int
    ](coords: Coord) {var residual_output_2d} -> SIMD[in_dtype, width]:
        return residual_output_2d.load[width=width, alignment=width](coords)

    var shape = (rows, cols)
    var scale_output_2d = scale_output_1d.reshape(row_major(rows, Idx[1]))

    comptime norm_threads = 256
    comptime norm_simd_width = simd_width_of[in_dtype, target=ctx.target]()
    comptime NormKernel = _rms_norm_rows_kernel[
        in_dtype, norm_simd_width, norm_threads, multiply_before_cast
    ]

    # Pre-compile the RMSNorm kernel before launching allreduce.
    # This avoids a deadlock where cuModuleLoadDataEx (JIT compilation)
    # blocks on the CUDA context while the allreduce kernel is waiting
    # at a multi-GPU barrier for other ranks to be enqueued.
    comptime if quantize:
        rms_norm_fused_fp8[
            in_dtype,
            out_dtype,
            scales_dtype,
            2,
            compile_only=True,
            pdl_level=pdl_level,
        ](
            input_fn,
            shape,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            ctx,
            scale_ub,
            scale_output_2d,
        )
    var norm_kernel = ctx.compile_function[NormKernel]()

    # Step 1: Allreduce with add epilogue → residual_output.
    @inline(.always)
    def add_epilogue[
        _dtype: DType,
        _width: SIMDLength,
        *,
        alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, length=_width]) {
        var residual_2d, var residual_output_2d
    } -> None:
        var il = coord_to_index_list(coords)
        var rc = Coord(il[0], il[1])
        var res = residual_2d.load[width=_width, alignment=alignment](rc)
        # Add in f32 for precision parity with the fused kernel,
        # which accumulates allreduce + residual in f32 before casting.
        var sum_f32 = (
            val.cast[.float32]()
            + rebind[SIMD[_dtype, _width]](res).cast[.float32]()
        )
        residual_output_2d.store[alignment=alignment](
            rc, rebind[SIMD[in_dtype, _width]](sum_f32.cast[_dtype]())
        )

    allreduce[ngpus=ngpus, pdl_level=pdl_level](
        input_buffers,
        residual_output.layout,
        add_epilogue,
        rank_sigs,
        ctx,
        my_rank=my_rank,
    )

    # Step 2: RMSNorm [+ FP8] on residual_output (kernel already compiled).
    comptime if quantize:
        rms_norm_fused_fp8[
            in_dtype, out_dtype, scales_dtype, 2, pdl_level=pdl_level
        ](
            input_fn,
            shape,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            ctx,
            scale_ub,
            scale_output_2d,
        )
    else:
        ctx.enqueue_function(
            norm_kernel,
            rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](output_2d.ptr),
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                residual_output.ptr
            ),
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](gamma.ptr),
            Int32(rows),
            Int32(cols),
            epsilon,
            weight_offset,
            grid_dim=min(rows, MAX_NUM_BLOCKS_UPPER_BOUND),
            block_dim=norm_threads,
        )


# --- Dispatch ---


def _2stage_scratch_bytes[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    has_residual: Bool,
](rows: Int, cols: Int) -> Int:
    """Returns the scratch the 2-stage kernel stages through after each
    `Signal` header (see `_allreduce_rmsnorm_fp8_launch_2stage`)."""
    var rows_per_rank = ceildiv(rows, ngpus)
    var num_bytes = rows_per_rank * cols * size_of[out_dtype]()
    comptime if out_dtype != in_dtype:
        # The scale section is padded to the SIMD width; 32 B covers any width.
        num_bytes += align_up(rows_per_rank * size_of[scales_dtype](), 32)
    comptime if has_residual:
        num_bytes += rows_per_rank * cols * size_of[in_dtype]()
    return num_bytes


def _fused_kernel_tiles_cols[
    in_dtype: DType
](cols: Int, ctx: DeviceContext) -> Bool:
    """Whether the fused kernels' warp-tiling dispatch can handle `cols`.

    Depends on shape only, so every rank agrees; `_dispatch_ar_norm` uses it
    to route untileable widths to the two-launch fallback.
    """
    comptime warp_size = ctx.default_device_info.warp_size
    comptime max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // warp_size
    )
    comptime sw1 = simd_width_of[in_dtype, target=ctx.target]()
    comptime sw2 = sw1 * 2
    return cols <= (warp_size * sw1 * max_warps_per_block) or (
        cols <= (warp_size * sw2 * max_warps_per_block) and cols % sw2 == 0
    )


def _dispatch_fused_kernel[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output_2d: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    my_rank: Int,
    scratch_bytes: Int,
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        ImmutAnyOrigin,
        engaged=False,
    ](),
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ] = _ComptimeConditionalTileTensor[
        in_dtype,
        _ZeroSizedLayout,
        MutAnyOrigin,
        engaged=False,
    ](),
) raises:
    """Dispatch the fused kernel with appropriate simd width and stage count.

    Centralizes the dispatch logic shared by allreduce_rmsnorm and
    allreduce_residual_rmsnorm. Selects simd width based on column count
    and Lamport, 1-stage, or 2-stage based on payload size. Paths that
    stage through more than `scratch_bytes` after each `Signal` header fall
    back to the 1-stage kernel.
    """
    # The kernels and the split path always wait for the predecessor, so the
    # no-wait level would silently mean the opposite of its name. Checked
    # before any fallback return so every path rejects it.
    comptime assert (
        pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
    ), "pdl_level must be OFF, ON or OVERLAP_AT_BEGINNING"

    # A step that skips speculative drafting hands every rank a zero-row
    # batch, and a zero grid is rejected by `enqueue_function`. Every rank
    # sees the same `rows`, so skipping the barrier pair strands no peer.
    if rows == 0:
        return

    # Every fused kernel reads its peers' inputs directly. Without P2P the
    # plain allreduce falls back to staged copies, so reduce with it and
    # normalize separately, as the unfused graph would.
    comptime if has_residual:
        if not is_p2p_enabled():
            _launch_split_allreduce_rmsnorm[
                in_dtype,
                out_dtype,
                scales_dtype,
                ngpus,
                multiply_before_cast,
                pdl_level,
            ](
                rows,
                cols,
                src_ptrs,
                output_2d,
                gamma,
                epsilon,
                weight_offset,
                scale_ub,
                scale_output_1d,
                rank_sigs,
                my_rank,
                ctx,
                residual[],
                residual_output[],
            )
            return

    comptime warp_size = ctx.default_device_info.warp_size
    comptime max_warps_per_block = (
        ctx.default_device_info.max_thread_block_size // warp_size
    )
    comptime threads_per_block = max_warps_per_block * warp_size
    comptime base_simd_width = simd_width_of[in_dtype, target=ctx.target]()
    comptime sw1 = base_simd_width
    comptime sw2 = base_simd_width * 2

    # Fuse FP8 quantization only when the output dtype differs from the input.
    # With P2P the split (2-kernel) path is picked only when quantizing: its
    # crossover is measured against the compact FP8 all-gather. So when not
    # quantizing the dispatch picks the Lamport, 1-stage, or 2-stage kernels.
    comptime quantize = out_dtype != in_dtype
    comptime assert multiply_before_cast or not quantize, (
        "the FP8 epilogue applies gamma before the cast, so"
        " multiply_before_cast=False is supported without quantization only"
    )

    # Use 2-stage (reduce-scatter + fused all-gather) for large payloads.
    # Threshold is per-rank payload: each GPU processes ceildiv(rows, ngpus)
    # full rows, so per-rank bytes stays roughly constant across column widths.
    var per_rank_bytes = ceildiv(rows, ngpus) * cols * size_of[in_dtype]()

    # Override for threshold tuning experiments:
    #   0 = use default threshold (production)
    #   1 = force 1-stage for all sizes
    #   2 = force 2-stage for all sizes
    #   3 = force split (2-kernel) path for all sizes (residual only)
    #   4 = force the Lamport kernel (no-quant only; the shape must fit it)
    comptime force_stage = get_defined_int["FORCE_ALLREDUCE_STAGE", 0]()

    # Latency-bound payloads take the barrier-free Lamport kernel wherever the
    # plain allreduce's tuning table routes the same byte count to Lamport.
    # That kernel has no FP8 epilogue, so quantizing calls stay on 1-stage,
    # and its sentinel exists only for 16- and 32-bit floats.
    comptime lamport_dtype = in_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    )
    comptime if (
        not quantize
        and lamport_dtype
        and (force_stage == 0 or force_stage == 4)
    ):
        comptime sm_version = get_sm_version()
        var use_lamport = force_stage == 4 or (
            dispatch_select_comm_config[
                ngpus, sm_version, allreduce_tuning_table
            ](rows * cols * size_of[in_dtype]()).algorithm
            == AllReduceAlgorithm.LAMPORT
            and lamport_allreduce_rmsnorm_fits[in_dtype, ngpus](rows, cols)
        )
        if use_lamport:
            comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
            _lamport_allreduce_rmsnorm_launch[
                in_dtype,
                ngpus,
                has_residual=has_residual,
                multiply_before_cast=multiply_before_cast,
            ](
                my_rank,
                src_ptrs[my_rank],
                rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](
                    output_2d.ptr
                ),
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](gamma.ptr),
                rank_sigs,
                rows,
                cols,
                Scalar[in_dtype](epsilon),
                weight_offset,
                ctx,
                residual,
                residual_output,
            )
            return

    # Per-rank byte thresholds for switching to 2-stage allreduce.
    # Derived from empirical sweep data (bf16, per-rank crossover points):
    #   MI355 4GPU: 128 KB non-res,  96 KB residual
    #   MI355 8GPU:  80 KB non-res,  96 KB residual
    #   B200  4GPU: 512 KB non-res, 256 KB residual
    #   B200  8GPU:  80 KB non-res, 80/100 KB residual (column-aware, see below)
    def _rank_4_per_rank_thresh() -> Int:
        comptime if ctx.target.is_amd_gpu():
            return 128 * 1024 if not has_residual else 96 * 1024
        else:
            return 512 * 1024 if not has_residual else 256 * 1024

    def _rank_8_per_rank_thresh() -> Int:
        comptime if ctx.target.is_amd_gpu():
            return 80 * 1024 if not has_residual else 96 * 1024
        else:
            return 80 * 1024

    # B200 8-GPU residual: the 1-stage/2-stage crossover is NOT a constant
    # per-rank byte count -- it rises with column width because the 1-stage
    # cost is dominated by a fixed per-launch floor plus a row-count term, so
    # for the same per-rank bytes a wider row (fewer rows) keeps 1-stage ahead
    # longer. Measured crossovers (B200 8xGPU, bf16, residual):
    #   cols 4096 -> ~72 KB ; cols 7168 -> ~104 KB ; cols 8192 -> ~104 KB.
    # A flat 80 KB mis-routes the wide-column (Kimi hidden=7168) M=40..56 band
    # onto the slower 2-stage path (e.g. M=48: 2-stage 41.2us vs 1-stage
    # 36.3us, a 12% loss). Use a column-aware threshold: 100 KB for wide
    # columns (>= 6144, the large-hidden regime), 80 KB otherwise (matches the
    # validated narrow-column crossover; cols=4096 crosses near 72 KB).
    def _rank_8_residual_thresh_for_cols(c: Int) -> Int:
        comptime if ctx.target.is_amd_gpu():
            return _rank_8_per_rank_thresh()
        else:
            return 100 * 1024 if c >= 6144 else 80 * 1024

    var threshold: Int
    comptime if ngpus <= 2 and has_residual and not quantize:
        # 2-stage ships the payload twice, moving 2*(ngpus-1)/ngpus of what
        # 1-stage moves over the fabric, plus a second barrier and a scratch
        # round trip (~9 us on 2xB200). At two ranks that ratio is exactly 1,
        # so this is never a win and we always use the 1-stage kernel.
        threshold = Int.MAX
    elif ngpus <= 4:
        threshold = _rank_4_per_rank_thresh()
    elif has_residual and not ctx.target.is_amd_gpu():
        threshold = _rank_8_residual_thresh_for_cols(cols)
    else:
        threshold = _rank_8_per_rank_thresh()

    # Per-rank byte thresholds for switching to split (2-kernel) path
    # when has_residual=True. Above this, allreduce+add + rmsnorm_fp8 beats
    # fused 2-stage because it avoids carrying bf16 residual through scratch.
    #   MI355: column-dependent; 2-stage beats split for cols <= 8192,
    #          split beats 2-stage for cols > 8192. See dispatch below.
    #   B200 4GPU: 1536 KB per-rank crossover
    #   B200 8GPU: conservative, same as 2-stage threshold
    def _rank_4_split_thresh() -> Int:
        comptime if ctx.target.is_amd_gpu():
            return _rank_4_per_rank_thresh()
        else:
            return 1536 * 1024

    def _rank_8_split_thresh() -> Int:
        # For 8 GPUs the split threshold equals the 2-stage threshold on
        # both AMD and NVIDIA.  This intentionally means the 2-stage
        # *residual* path is never selected at 8 GPUs: once per-rank
        # payload crosses threshold the dispatch jumps straight to split,
        # skipping the `elif per_rank_bytes >= threshold` arm below.
        # Benchmarks show the split (2-kernel) path is always faster than
        # fused 2-stage with residual at 8-GPU scale because the extra
        # bf16 scratch traffic in Stage 2 outweighs the launch overhead.
        comptime if ctx.target.is_amd_gpu():
            return _rank_8_per_rank_thresh()
        else:
            return _rank_8_per_rank_thresh()

    comptime split_threshold = _rank_4_split_thresh() if ngpus <= 4 else _rank_8_split_thresh()

    var use_2stage: Bool
    var use_split: Bool = False
    comptime match force_stage:
        case 1:
            use_2stage = False
        case 2:
            use_2stage = True
        case 3:
            # Force split (2-kernel) path (FP8/quantizing residual only).
            use_2stage = False
            comptime if has_residual and quantize:
                use_split = True
        case _:
            comptime if has_residual and quantize:
                comptime if ctx.target.is_amd_gpu():
                    # MI355: 2-stage fused beats split for cols <= 8192,
                    # split beats 2-stage for wider columns (16384+).
                    if per_rank_bytes >= threshold:
                        if cols > 8192:
                            use_split = True
                            use_2stage = False
                        else:
                            use_2stage = True
                    else:
                        use_2stage = False
                else:
                    # NVIDIA residual dispatch.  For 8 GPUs split_threshold
                    # == threshold, so the elif is unreachable and the
                    # 2-stage residual path is intentionally skipped (see
                    # _rank_8_split_thresh comment).  For 4 GPUs the two
                    # thresholds differ and all three paths are reachable.
                    if per_rank_bytes >= split_threshold:
                        use_split = True
                        use_2stage = False
                    elif per_rank_bytes >= threshold:
                        use_2stage = True
                    else:
                        use_2stage = False
            elif has_residual:
                # Non-quant residual: the split path is FP8-only, so pick between
                # the 1-stage and 2-stage fused kernels by per-rank payload only.
                use_2stage = ngpus <= 8 and per_rank_bytes >= threshold
            else:
                use_2stage = ngpus <= 8 and per_rank_bytes >= threshold

            # The 2-stage kernel and the split path's allreduce stage through
            # the scratch after each Signal header. The 1-stage kernel reads
            # peers directly, so it is the fallback when the scratch is short.
            if (
                use_2stage
                and _2stage_scratch_bytes[
                    in_dtype, out_dtype, scales_dtype, ngpus, has_residual
                ](rows, cols)
                > scratch_bytes
            ):
                use_2stage = False
            # Matches the plain allreduce's 32-element partitioning.
            var split_scratch = (
                ceildiv(ceildiv(rows * cols, 32), ngpus)
                * 32
                * size_of[in_dtype]()
            )
            if use_split and split_scratch > scratch_bytes:
                use_split = False

    def launch_1stage[sw: Int]() raises {imm}:
        _allreduce_rmsnorm_fp8_launch[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
            multiply_before_cast=multiply_before_cast,
            pdl_level=pdl_level,
        ](
            rows,
            cols,
            src_ptrs,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            scale_ub,
            scale_output_1d,
            rank_sigs,
            my_rank,
            ctx,
            residual,
            residual_output,
        )

    def launch_2stage[sw: Int]() raises {imm}:
        _allreduce_rmsnorm_fp8_launch_2stage[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
            multiply_before_cast=multiply_before_cast,
            pdl_level=pdl_level,
        ](
            rows,
            cols,
            src_ptrs,
            output_2d,
            gamma,
            epsilon,
            weight_offset,
            scale_ub,
            scale_output_1d,
            rank_sigs,
            my_rank,
            ctx,
            residual,
            residual_output,
        )

    if use_split:
        # use_split is only ever set when has_residual and quantize, and the
        # comptime guard keeps the split launcher out of the other
        # instantiations.
        comptime if has_residual and quantize:
            _launch_split_allreduce_rmsnorm[
                in_dtype,
                out_dtype,
                scales_dtype,
                ngpus,
                multiply_before_cast,
                pdl_level,
            ](
                rows,
                cols,
                src_ptrs,
                output_2d,
                gamma,
                epsilon,
                weight_offset,
                scale_ub,
                scale_output_1d,
                rank_sigs,
                my_rank,
                ctx,
                residual[],
                residual_output[],
            )
        return

    # Warp-tiling: each thread handles simd_width elements. Wider simd for
    # larger column counts.
    if not _fused_kernel_tiles_cols[in_dtype](cols, ctx):
        comptime max_cols = warp_size * sw2 * max_warps_per_block
        raise Error(
            String(
                t"allreduce_rmsnorm: cols ({cols}) exceeds max supported ("
                t"{max_cols}) for warp-tiling kernel"
            )
        )
    if cols <= (warp_size * sw1 * max_warps_per_block):
        if use_2stage:
            launch_2stage[sw1]()
        else:
            launch_1stage[sw1]()
    elif use_2stage:
        launch_2stage[sw2]()
    else:
        launch_1stage[sw2]()


# --- Public API ---


def allreduce_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    //,
    multiply_before_cast: Bool = True,
    # PDL is verified on 2 GPUs only; wider worlds stay OFF until measured.
    pdl_level: PDLLevel = (
        PDLLevel.OVERLAP_AT_BEGINNING if ngpus <= 2 else PDLLevel.OFF
    ),
](
    input_buffers: Array[
        TileTensor[in_dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    my_rank: Optional[Int] = None,
    scratch_bytes: Optional[Int] = None,
) raises:
    """Fused allreduce + RMSNorm with optional FP8 quantization.

    Combines a P2P allreduce across GPUs, RMSNorm, and (when the output dtype
    differs from the input dtype) FP8 dynamic quantization into a single
    kernel launch, eliminating the global memory round-trip between allreduce
    output and RMSNorm input. When `out_dtype == in_dtype` the quantization is
    skipped: the normalized value is written directly in the input dtype and
    `scale_ub` and `scale_output` are ignored.

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: Output data type. Either a float8 type (fuses
            quantization) or equal to `in_dtype` (no quantization).
        scales_dtype: Scale factor data type (e.g. float32). Ignored when
            `out_dtype == in_dtype`.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        multiply_before_cast: Matches `ops.rms_norm`. True applies gamma in
            float32 before the single cast to the output dtype; False casts
            the normalized value to `in_dtype` first and applies gamma in
            `in_dtype`. False requires `out_dtype == in_dtype`.
        pdl_level: Programmatic dependent launch level. The fused kernels
            wait for their predecessor before they signal their peers, and
            release the next kernel after the barrier that retires the peers'
            reads (`ON`) or right after the wait (`OVERLAP_AT_BEGINNING`). On
            the split path the allreduce always releases the norm at its exit,
            and only the norm honors `OVERLAP_AT_BEGINNING`. That level
            requires the next kernel not to write this kernel's inputs before
            its own PDL wait, because peers may still be reading them. The
            Lamport branch always uses PDL.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        output: Output buffer (FP8 values when quantizing, else `in_dtype`).
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping (ignored when not
            quantizing).
        scale_output: Output buffer for per-row FP8 scales (ignored, and not
            written, when not quantizing).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        my_rank: This GPU's index in `input_buffers` and `rank_sigs`.
            Defaults to `ctx.device().id`, which is right only when device
            ordinals match list positions.
        scratch_bytes: Bytes after each `Signal` header that kernels may
            stage through. Paths that need more take the 1-stage kernel.
            None means the caller sized it for every path.

    Note:
        Both kernels finish all P2P reads of the input buffers before they
        exit (the 1-stage kernel with an end barrier, the 2-stage kernel
        through its mid barrier), so the inputs are free to be reused once
        this op retires. The output and scale buffers are safe to read only
        on the local GPU (stream ordering guarantees visibility). If a
        remote GPU needs to read these outputs, the caller must insert an
        explicit barrier.

        Signal buffer sizing:
          1-stage path (payload < threshold): size_of[Signal]() only.
          2-stage path (payload > threshold):
            size_of[Signal]()
            + ceildiv(rows, ngpus) * cols * sizeof(out_dtype)          (output)
            + align_up(ceildiv(rows, ngpus) * sizeof(scales_dtype),
                       simd_width)                          (scales + pad, if
                                                             quantizing)
    """
    comptime assert ngpus >= 2, "allreduce_rmsnorm requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert (out_dtype == in_dtype) or out_dtype.is_float8(), (
        "out_dtype must be a float8 type (fuse quantization) or equal to"
        " in_dtype (skip quantization)"
    )

    if not is_p2p_enabled():
        raise Error("allreduce_rmsnorm requires P2P access between GPUs")

    var rank = my_rank.or_else(Int(ctx.device().id))
    if not 0 <= rank < ngpus:
        raise Error(
            t"allreduce_rmsnorm: my_rank ({rank}) must be in [0, {ngpus})"
        )

    # Compute rows/cols from TileTensor dims.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Extract raw pointers from TileTensors.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: rebind[
            ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
        ](input_buffers[i].ptr)
    )

    # Create internal 2D/1D TileTensor views for _dispatch_fused_kernel.
    var output_2d = output.reshape(row_major(rows, cols))
    var scale_output_1d = scale_output.reshape(row_major(rows))

    _dispatch_fused_kernel[
        in_dtype,
        out_dtype,
        scales_dtype,
        ngpus,
        multiply_before_cast=multiply_before_cast,
        pdl_level=pdl_level,
    ](
        rows,
        cols,
        src_ptrs,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        scale_ub,
        scale_output_1d,
        rank_sigs,
        rank,
        scratch_bytes.or_else(Int.MAX),
        ctx,
    )


# Cols the fuse threshold was measured at (MI355, 8 GPUs, bf16); a different
# width maps the byte threshold to a different row count, so recalibrate
# before trusting the crossover there.
comptime AR_NORM_CALIBRATED_COLS = 3584

# `[rows, cols]` bf16 bytes at/below which `_dispatch_ar_norm` fuses: the
# measured MI355 8-GPU crossover at 3584 cols, where the fused kernel wins
# through 512 rows and loses from 768. A perf cut, not correctness -- above
# it the fabric-saturated plain allreduce beats the norm-in-epilogue, so the
# dispatch hands off to a standalone allreduce + norm.
comptime AR_NORM_FUSE_THRESHOLD = (
    512 * AR_NORM_CALIBRATED_COLS * size_of[DType.bfloat16]()
)


def _dispatch_ar_norm[
    in_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    TwoLaunchFnType: def() raises -> None,
    //,
    multiply_before_cast: Bool = True,
](
    input_buffers: Array[
        TileTensor[in_dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    normed_out: TileTensor[mut=True, in_dtype, ...],
    sum_out: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    two_launch: TwoLaunchFnType,
    my_rank: Optional[Int] = None,
    scratch_bytes: Optional[Int] = None,
) raises:
    """Runtime-select the fused allreduce+norm vs a two-launch fallback.

    Fuses while the full `[rows, cols]` payload fits `AR_NORM_FUSE_THRESHOLD`;
    above it, hand off to `two_launch` (a standalone allreduce into `sum_out`
    followed by a separate norm), caller-supplied so `comm` stays free of
    `nn`. Nothing is calibrated off-AMD, so those targets always take
    `two_launch`, as does any host without P2P access: the fused kernels
    read their peers' inputs directly, while the plain allreduce falls back
    to staged copies.

    Widths the fused kernels cannot warp-tile also take `two_launch`. The
    gate reads only `rows` and `cols`, which are identical on every rank
    (an allreduce needs same-shaped operands): the two arms issue different
    barriers on shared `rank_sigs`, so a per-rank disagreement would
    deadlock. `sum_out` is written by `two_launch` only.
    """
    comptime assert (
        in_dtype == .bfloat16
    ), "_dispatch_ar_norm fuse threshold is bf16-calibrated (bf16 in/out only)"

    # The fused kernels read their peers' inputs directly; without P2P the
    # plain allreduce falls back to staged copies, so run the two-launch
    # fallback instead of the fused arm's hard P2P requirement.
    # `is_p2p_enabled` is a process-wide property, so every rank takes the
    # same arm.
    if not is_p2p_enabled():
        two_launch()
        return

    comptime threshold = (
        AR_NORM_FUSE_THRESHOLD if ctx.target.is_amd_gpu() else Int(0)
    )

    var payload_bytes = input_buffers[0].num_elements() * size_of[in_dtype]()
    comptime last_dim_idx = in_layout.rank - 1
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    if payload_bytes <= threshold and _fused_kernel_tiles_cols[in_dtype](
        cols, ctx
    ):
        # in == out dtype, so the kernel skips its FP8 epilogue and never
        # touches the scale operand; `sum_out` (unwritten on this arm)
        # doubles as the dummy.
        allreduce_rmsnorm[multiply_before_cast=multiply_before_cast](
            input_buffers,
            normed_out,
            gamma,
            epsilon,
            weight_offset,
            Float32(0.0),
            sum_out,
            rank_sigs,
            ctx,
            my_rank=my_rank,
            scratch_bytes=scratch_bytes,
        )
    else:
        two_launch()


comptime MCAR_NORM_MAX_ROWS = 16
"""Rows past which the arm's normed-row epilogue hands back to the shipped arms.

Equal work, four B200s at TP4 and a 4096-wide hidden, residual and pre-norm
stream on every arm: 7.4/9.3/10.3 (mcast) against 8.3/10.9/11.5 (Lamport) and
10.7/13.0/13.6 (1-stage) us a call at 1/8/16 rows, a dead heat at 32, and the
Lamport arm ahead from 64 (14.4 against 13.0, 20.9 against 15.9 at 128).

This one is a cost bound, and it is the only reason the arm stops before the
band: the epilogue past it is the same arithmetic on the same rows.
"""

comptime MCAR_GROUPED_MAX_ROWS = 16
"""Rows past which the arm's grouped FP8 epilogue hands back to its staging path.

The bound is the speculative decoder's acceptance, not the arm's time. The arm's
kernel is its own algorithm (an fp32 residual add and its own reduction tree),
so its group amax is close to the unfused chain's rather than byte-equal, and on
MiMo-V2.6 that moves draft acceptance by a few percent with a sign that flips
with the row count. The cap keeps the arm where it was measured to win; past it
a grouped call runs the unfused chain, which is byte-inert (`distributed.mojo`).

Not a hard ceiling on the kernel: `mcar_rmsnorm` serves any row count the
window holds, and its oracle runs past this cap. What the cap bounds is the
fold, i.e. which arm the graph compiler's epilogue may take.
"""


def _mcar_transport_supported[dtype: DType]() -> Bool:
    """Whether the multicast boundary carries a `dtype` message at all.

    The arm's kernel stores through `multimem`, which only SM90+ NVIDIA GPUs
    have. On any other target this is false at compile time, so no caller
    instantiates that kernel there.

    Parameters:
        dtype: Element type of the message.
    """
    return (
        default_accelerator()._is_sm_9x_or_newer() and dtype == DType.bfloat16
    )


def mcar_rmsnorm_fits[
    dtype: DType, ngpus: Int, quantized: Bool
](rows: Int, cols: Int) -> Bool:
    """Returns whether a `[rows, cols]` boundary belongs to the multicast arm.

    The band is the bucket the arm replaces: the per-rank message has to fit a
    slot of the symmetric window, which is the same 1 MiB the Lamport arm
    reserves. Neither epilogue reaches it -- both are capped first, at
    `MCAR_NORM_MAX_ROWS` and `MCAR_GROUPED_MAX_ROWS` -- so what decides a call
    here is the epilogue's cap, and past it the existing path serves, unchanged.

    Parameters:
        dtype: Element type of the message.
        ngpus: Number of participating ranks.
        quantized: Whether the call's epilogue is the grouped FP8 row. Both
            epilogues are capped below the band, for different reasons: the
            normed row where a shipped arm is cheaper at equal work
            (`MCAR_NORM_MAX_ROWS`), the grouped one where the arm costs draft
            acceptance (`MCAR_GROUPED_MAX_ROWS`).

    Args:
        rows: Number of rows (tokens) the call reduces.
        cols: Row width; RMSNorm runs over it.

    Returns:
        True when the multicast arm serves the shape.
    """
    comptime if not _mcar_transport_supported[dtype]():
        return False
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime epilogue_cap = (
        MCAR_GROUPED_MAX_ROWS if quantized else MCAR_NORM_MAX_ROWS
    )
    var row_cap = min(mcar_window_rows[dtype](cols), epilogue_cap)
    return (
        2 <= ngpus <= MAX_GPUS
        and mcar_supports_cols[ngpus](cols)
        and cols % atomic_width == 0
        and rows <= row_cap
    )


def normalized_scale_group[
    group_size_or_per_token: Int, static_cols: Int
]() -> Int:
    """Returns the FP8 scale granularity with one scale a row spelled `-1`.

    The dialect spells a per-token scale as the row width, so the attribute's
    own contract makes `-1` and the row width the same granularity, and neither
    is a group that happens to cover the row. Normalizing once keeps every test
    downstream a test of grouping (`<= 0`, `== MCAR_GROUP`) rather than a test
    of spelling: a caller that passes the row width reaches the same fused
    epilogue and the same arm that `-1` reaches.

    Reads only compile-time values, so a caller can gate on it at compile time.

    Parameters:
        group_size_or_per_token: The attribute as the graph carries it.
        static_cols: The row width when the graph knows it, else `-1`.

    Returns:
        `-1` when the granularity is one scale a row, else the attribute's
        group size.
    """
    comptime if group_size_or_per_token <= 0:
        return -1
    comptime if static_cols > 0 and group_size_or_per_token == static_cols:
        return -1
    return group_size_or_per_token


def mcar_epilogue_serves[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    group_size_or_per_token: Int,
]() -> Bool:
    """Whether the multicast arm's epilogue is the one this call writes.

    The arm carries two epilogues: the normed row in the message dtype, and the
    1x128-group E4M3 row with float32 scales its port implements. Every other
    epilogue -- a per-token scale, another group size, another FP8 flavour or
    scale type -- is the standalone quantizer's, which a caller runs as its own
    launch over the normed rows.

    Reads only compile-time values, so a caller can gate on it at compile time.

    Parameters:
        in_dtype: Message element type of the boundary.
        out_dtype: Element type of the quantized output.
        scales_dtype: Element type of the quantized scales.
        group_size_or_per_token: Columns per FP8 scale (`-1` for one a row).

    Returns:
        True when the arm's own epilogue writes this call's output.
    """
    comptime if not _mcar_transport_supported[in_dtype]():
        return False
    comptime if group_size_or_per_token <= 0:
        return out_dtype == in_dtype
    return (
        group_size_or_per_token == MCAR_GROUP
        and out_dtype == DType.float8_e4m3fn
        and scales_dtype == DType.float32
    )


def mcar_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    has_residual: Bool = False,
    multiply_before_cast: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    *,
    group_size_or_per_token: Int = -1,
](
    rows: Int,
    cols: Int,
    src_ptrs: Array[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus],
    output_2d: TileTensor[mut=True, out_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    my_rank: Int,
    ctx: DeviceContext,
    residual: _ComptimeConditionalTileTensor[
        in_dtype, engaged=has_residual, ...
    ],
    residual_output: _ComptimeConditionalTileTensor[
        mut=True, in_dtype, engaged=has_residual, ...
    ],
    window: McarWindowRank,
) raises:
    """The multicast arm of the fused residual boundary.

    Everything `allreduce_residual_rmsnorm` does, exchanged through the team's
    symmetric window instead of the Lamport slots: each rank multicasts its own
    pack once (the fabric performs the fan-out), reads the peers' slots from its
    own copy, and runs the residual add, the RMSNorm and the group-scaled FP8
    quant in the same launch. `window` is this rank's views of that window, and
    its device-resident call state is what rotates the generations, so a
    captured graph replays without a host argument.

    Only the shapes `mcar_rmsnorm_fits` accepts, and only the epilogues
    `mcar_epilogue_serves` names, reach here; every other call stays on the
    shipped paths, which is why this arm cannot regress one.

    Parameters:
        in_dtype: Input data type (bf16 on every device the arm serves).
        out_dtype: Output data type: `in_dtype`, or a float8 type when the
            call quantizes.
        scales_dtype: Scale factor data type.
        ngpus: Number of participating ranks.
        has_residual: Whether the residual add is engaged.
        multiply_before_cast: Matches `ops.rms_norm`.
        pdl_level: Programmatic dependent launch level.
        group_size_or_per_token: Scale granularity of the quantized epilogue
            this kernel serves (`-1` is one scale a row).

    Args:
        rows: Rows (tokens) the call reduces.
        cols: Row width.
        src_ptrs: Per-rank input pointers.
        output_2d: `[rows, cols]` output.
        gamma: RMSNorm weight.
        epsilon: RMSNorm epsilon.
        weight_offset: Additive offset for the RMSNorm weight.
        scale_ub: Upper bound for the quantized scale.
        scale_output: Quantized scales in the `[cols / K, rows]` grouped
            layout the consuming block-scaled GEMM reads (unused by the normed
            row epilogue).
        my_rank: This rank's index.
        ctx: Device context for this rank.
        residual: `[rows, cols]` residual, replicated on every rank.
        residual_output: `[rows, cols]` pre-norm sum.
        window: This rank's views of the team's symmetric window.
    """
    comptime assert mcar_epilogue_serves[
        in_dtype, out_dtype, scales_dtype, group_size_or_per_token
    ](), "only the epilogues the arm ports reach the multicast launch"
    comptime assert has_residual, "the multicast arm is the residual boundary's"
    comptime assert (
        in_dtype == DType.bfloat16
    ), "the multicast arm transports bf16"
    var scale_stride = 0
    comptime if group_size_or_per_token > 0:
        comptime assert (
            scale_output.rank == 2
        ), "a grouped epilogue writes a rank-2 scale tensor"
        # The consuming block-scaled GEMM pads the scale tensor's row
        # dimension, so the stride is the tensor's own, never `rows`.
        scale_stride = Int(scale_output.dim[1]())
    # The unused epilogue output still needs a valid pointer: the kernel's
    # `quant` variant writes one of the two, so both name the one output and
    # its origin is erased rather than borrowed twice. The tensors are global
    # memory; their pointers only carry the address space as a parameter.
    var output_ptr = output_2d.ptr.unsafe_address_space_cast[
        .GENERIC
    ]().unsafe_origin_cast[MutUntrackedOrigin]()
    var normed_ptr = output_ptr.unsafe_bitcast[Scalar[DType.bfloat16]]()
    var q_ptr = output_ptr.unsafe_bitcast[Scalar[DType.float8_e4m3fn]]()
    var scales_ptr = scale_output.ptr.unsafe_address_space_cast[.GENERIC]()
    var residual_ptr = residual[].ptr.unsafe_address_space_cast[.GENERIC]()
    var residual_out_ptr = residual_output[].ptr.unsafe_address_space_cast[
        .GENERIC
    ]()
    var gamma_ptr = gamma.ptr.unsafe_address_space_cast[.GENERIC]()
    allreduce_mcast_rmsnorm[
        ngpus,
        quant=group_size_or_per_token > 0,
        multiply_before_cast=multiply_before_cast,
    ](
        src_ptrs[my_rank].unsafe_bitcast[Scalar[DType.bfloat16]](),
        residual_ptr.unsafe_bitcast[Scalar[DType.bfloat16]](),
        residual_out_ptr.unsafe_bitcast[Scalar[DType.bfloat16]](),
        normed_ptr,
        gamma_ptr.unsafe_bitcast[Scalar[DType.bfloat16]](),
        q_ptr,
        scales_ptr.unsafe_bitcast[Float32](),
        scale_stride,
        rows,
        cols,
        epsilon,
        weight_offset.cast[DType.float32](),
        scale_ub,
        my_rank,
        window,
        ctx,
        pdl_level,
    )


def allreduce_residual_rmsnorm[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    in_engine: TensorEngine,
    //,
    multiply_before_cast: Bool = True,
    *,
    group_size_or_per_token: Int = -1,
](
    input_buffers: Array[
        TileTensor[in_dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    residual: TileTensor[mut=False, in_dtype, ...],
    output: TileTensor[mut=True, out_dtype, ...],
    residual_output: TileTensor[mut=True, in_dtype, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: TileTensor[mut=True, scales_dtype, ...],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], ngpus],
    ctx: DeviceContext,
    my_rank: Optional[Int] = None,
    scratch_bytes: Optional[Int] = None,
    window: Optional[McarWindowRank] = None,
) raises:
    """Fused allreduce + residual add + RMSNorm with optional FP8 quantization.

    Combines a P2P allreduce, a residual add, RMSNorm, and (when the output
    dtype differs from the input dtype) FP8 dynamic quantization into a fused
    kernel. When `out_dtype == in_dtype` the quantization is skipped: the
    normalized value is written directly in the input dtype and `scale_ub` and
    `scale_output` are ignored (the per-row scale is neither computed nor
    written). Without P2P access it runs the plain allreduce, which stages
    peer inputs through copies, followed by a separate RMSNorm kernel.

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: Output data type. Either a float8 type (fuses
            quantization) or equal to `in_dtype` (no quantization).
        scales_dtype: Scale factor data type (e.g. float32). Ignored when
            `out_dtype == in_dtype`.
        ngpus: Number of GPUs participating.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        multiply_before_cast: Matches `ops.rms_norm`. True applies gamma in
            float32 before the single cast to the output dtype; False casts
            the normalized value to `in_dtype` first and applies gamma in
            `in_dtype`. False requires `out_dtype == in_dtype`.
        group_size_or_per_token: Scale granularity of the quantized epilogue:
            `K` is one scale per `K` columns of a row, and `-1` (or the row
            width) is one per row, the row width normalized to `-1` here. Only
            the multicast arm serves a grouped epilogue (1x128 groups, float32
            scales); another grouped epilogue is the caller's to run over this
            kernel's normed output.

    Args:
        input_buffers: Per-GPU input buffers as TileTensors.
        residual: Residual buffer as a TileTensor.
        output: Output buffer (FP8 values when quantizing, else `in_dtype`).
        residual_output: Output buffer for pre-norm sum as a TileTensor.
        gamma: RMSNorm gamma weights (1D TileTensor).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping (ignored when not
            quantizing).
        scale_output: Output buffer for per-row FP8 scales (ignored, and not
            written, when not quantizing).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.
        my_rank: This GPU's index in `input_buffers` and `rank_sigs`.
            Defaults to `ctx.device().id`, which is right only when device
            ordinals match list positions.
        scratch_bytes: Bytes after each `Signal` header that kernels may
            stage through. Paths that need more take the 1-stage kernel.
            None means the caller sized it for every path.
        window: This rank's views of the team's symmetric multicast window,
            when the caller allocated one. Its presence is what lets the
            dispatch take the multicast arm for the shapes that arm serves;
            its absence is the fallback, and every path this op had before
            still runs.
    """
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert (out_dtype == in_dtype) or out_dtype.is_float8(), (
        "out_dtype must be a float8 type (fuse quantization) or equal to"
        " in_dtype (skip quantization)"
    )

    var rank = my_rank.or_else(Int(ctx.device().id))
    if not 0 <= rank < ngpus:
        raise Error(
            t"allreduce_residual_rmsnorm: my_rank ({rank}) must be in [0,"
            t" {ngpus})"
        )

    # Compute rows/cols from TileTensor dims. The internal dispatch flattens
    # to 2D, so we only need the last dim for cols and total/cols for rows.
    var in_num_elems = input_buffers[0].num_elements()
    comptime last_dim_idx = in_layout.rank - 1
    # The graph's own spelling of one scale a row is the row width, so the
    # granularity is normalized before anything tests it. `_shape_types` carries
    # the same static width the composite reads, and `-1` when it is dynamic.
    comptime group = normalized_scale_group[
        group_size_or_per_token,
        in_layout._shape_types[last_dim_idx].static_value,
    ]()
    var cols = Int(input_buffers[0].dim[last_dim_idx]())
    var rows = in_num_elems // cols

    # Extract raw pointers from TileTensors.
    comptime PtrType = ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
    var src_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> PtrType: rebind[
            ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]
        ](input_buffers[i].ptr)
    )

    # Create internal 2D/1D TileTensor views for _dispatch_fused_kernel.
    var output_2d = output.reshape(row_major(rows, cols))
    var residual_2d = (
        residual.as_imm().as_unsafe_any_origin().reshape(row_major(rows, cols))
    )
    var residual_output_2d = residual_output.as_unsafe_any_origin().reshape(
        row_major(rows, cols)
    )
    # PDL is verified on 2 GPUs only; 4/8-GPU runs stay OFF until someone
    # measures them. The fused kernels wait for their predecessor before they
    # signal their peers, and release the next kernel right after that wait.
    comptime pdl_level = (
        PDLLevel.OVERLAP_AT_BEGINNING if ngpus <= 2 else PDLLevel.OFF
    )

    # The multicast arm takes the bucket the Lamport arm serves, for the two
    # epilogues it ports (`mcar_epilogue_serves`). `window` is absent unless the
    # caller allocated one, so a run that cannot carry a multicast mapping
    # simply never reaches this branch. The define is what puts the arm in the
    # binary at all: it is validated on one model, and a graph compiled without
    # it never allocates a window (`distributed.mojo`), so instantiating the
    # kernels there would be codegen nothing can reach.
    comptime if get_defined_bool[
        "MODULAR_MULTICAST_ALLREDUCE_RMS_NORM", False
    ]() and mcar_epilogue_serves[in_dtype, out_dtype, scales_dtype, group]():
        if case .Some(rank_window) = window:
            if (
                mcar_rmsnorm_fits[in_dtype, ngpus, quantized=group > 0](
                    rows, cols
                )
                and rank_window.rows >= rows
            ):
                mcar_rmsnorm[
                    in_dtype,
                    out_dtype,
                    scales_dtype,
                    ngpus,
                    has_residual=True,
                    multiply_before_cast=multiply_before_cast,
                    pdl_level=pdl_level,
                    group_size_or_per_token=group,
                ](
                    rows,
                    cols,
                    src_ptrs,
                    output_2d,
                    gamma,
                    epsilon,
                    weight_offset,
                    scale_ub,
                    scale_output,
                    rank,
                    ctx,
                    residual_2d,
                    residual_output_2d,
                    rank_window,
                )
                return

    # A grouped epilogue is not this kernel's to write below the arm: the paths
    # that follow take one scale a row. A caller asking for groups runs the
    # boundary in the message dtype and the quantizer over its output, so
    # reaching here means the caller lost that branch.
    comptime if group > 0:
        raise Error(
            "a grouped quantized epilogue needs the multicast boundary arm;"
            " callers that miss it run the quantizer themselves"
        )

    # The paths below address one scale a row (a grouped epilogue keeps the
    # rank-2 tensor its consumer reads, and only the multicast arm serves it),
    # so the flatten happens once a grouped call has been routed away.
    var scale_output_1d = scale_output.reshape(row_major(rows))

    _dispatch_fused_kernel[
        in_dtype,
        out_dtype,
        scales_dtype,
        ngpus,
        has_residual=True,
        multiply_before_cast=multiply_before_cast,
        pdl_level=pdl_level,
    ](
        rows,
        cols,
        src_ptrs,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        scale_ub,
        scale_output_1d,
        rank_sigs,
        rank,
        scratch_bytes.or_else(Int.MAX),
        ctx,
        residual_2d,
        residual_output_2d,
    )
