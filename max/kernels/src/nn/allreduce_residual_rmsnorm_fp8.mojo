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
"""Fused allreduce + RMSNorm + FP8 quantization kernel.

Combines P2P allreduce, RMSNorm normalization, and FP8 dynamic quantization
into a single kernel launch. Data stays in registers throughout — no global
memory intermediate between allreduce and RMSNorm.

Three dispatch paths:

1-stage (small payloads): Each thread loads from all GPUs, accumulates
  in float32 registers, then applies RMSNorm + FP8 quantization. Simple
  but O(N x ngpus) P2P traffic.

2-stage (large payloads, non-residual or medium residual): Single kernel
  with two stages separated by a per-block barrier:
    Stage 1 (RS + RMSNorm + FP8): each block reduces rows in its partition
      from all GPUs in f32 registers, then normalizes and quantizes to FP8
      immediately — no f32 scratch. Writes compact fp8 data + per-row scale
      (+ optional bf16 residual) to local scratch.
    Stage 2 (AG copy): each block reads compact fp8/scale/bf16 from the
      owning GPU's scratch (P2P) and writes to local output buffers. No
      compute — just copies of data that is 4x smaller than f32.
  Both stages use the same row-to-block mapping, so the per-block barrier
  correctly synchronizes all data dependencies. Total P2P traffic is
  N * sizeof(in_dtype) for Stage 1 (same as 1-stage) plus
  N * sizeof(fp8) [+ N * sizeof(in_dtype) if residual] for Stage 2.
  The AG phase copies fp8 (1 byte/elem) instead of f32 (4 bytes/elem),
  reducing Stage 2 bandwidth by ~4x for the non-residual case.

Split (large residual payloads): Two separate kernel launches —
  allreduce with add epilogue followed by fused rmsnorm+fp8. Avoids
  carrying bf16 residual data through scratch buffers in both stages,
  which nearly doubles Stage 2 copy bandwidth in the fused path. The
  caller-provided residual_output buffer serves as the intermediate,
  so no extra allocation is needed.

Each block processes multiple rows via a grid-strided loop, allowing
row counts beyond the hardware-tuned block limit. Gamma weights are
preloaded once and reused across all rows in the loop.
"""

from collections import InlineArray, Optional
from math import ceildiv, rsqrt
from sys import (
    align_of,
    env_get_int,
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
)
from sys.info import _accelerator_arch

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu.host import DeviceContext, get_gpu_target
from gpu.primitives import block
from layout import Coord, Idx, TileTensor
from layout._layout import TensorLayout, row_major
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import Layout as LegacyLayout
from layout.layout_tensor import LayoutTensor as LegacyLayoutTensor
from layout.runtime_layout import RuntimeLayout
from utils import IndexList, StaticTuple
from utils.index import Index
from utils.numerics import get_accum_type, max_finite

from runtime.asyncrt import DeviceContextPtr
from linalg.fp8_utils import compute_dynamic_fp8_scale, fp8_quantize

from .normalization import rms_norm_fused_fp8

from comm.allreduce import allreduce, elementwise_epilogue_type
from comm.device_query import get_sm_version, _dispatch_max_num_blocks
from comm.reducescatter import _target_address_space
from comm.sync import (
    MAX_GPUS,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)


# --- GPU Kernel ---


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
fn _allreduce_rmsnorm_fp8_kernel_warp_tiling[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_layout: LegacyLayout,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    output_fn: fn[width: Int](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    gamma: TileTensor[in_dtype, LayoutType, origin],
    scale_buffer: LegacyLayoutTensor[
        mut=True, scales_dtype, scale_layout, MutAnyOrigin
    ],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    rows: Int,
    cols: Int,
    scale_ub: Scalar[scales_dtype],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin],
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin],
):
    """Fused allreduce + RMSNorm + FP8 kernel using warp-tiling.

    Each thread block processes one or more rows via a persistent row loop
    (stride = grid_dim). Data stays in registers across all four phases:
    P2P load+reduce, mean-square, normalize+max, quantize+write.
    Gamma weights are preloaded once and reused across all rows.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var tid = Int(thread_idx.x)
    var idx = tid * simd_width
    var is_valid = idx < cols

    # Barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # Preload gamma weights into registers. The global memory latency is
    # hidden behind the P2P loads below (~200+ cycles per GPU).
    # Gamma depends only on column index — preload once, reuse across all rows.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](Coord(Idx(idx))).cast[
                accum_type
            ]()
            + weight_offset.cast[accum_type]()
        )

    # Round-robin access pattern for NVLink load-balancing.
    var ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        var target = (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

    # Row loop: each block processes rows with stride = grid_dim.
    # For rows <= grid_dim, the loop body runs exactly once per block.
    var num_blocks = Int(grid_dim.x)
    for row in range(Int(block_idx.x), rows, num_blocks):
        # Phase 0: P2P load from all GPUs + accumulate in float32 regs.
        # Gamma load latency from above is hidden during P2P stalls.
        var vec_data = SIMD[accum_type, simd_width](0)
        if is_valid:
            var elem_idx = row * cols + idx

            comptime for gpu_idx in range(ngpus):
                vec_data += (
                    ptrs[gpu_idx]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=True,
                    ](elem_idx)
                    .cast[accum_type]()
                )

            # Add residual and write pre-norm sum (compile-time gated).
            comptime if has_residual:
                vec_data += residual.load[width=simd_width](
                    IndexList[2](row, idx)
                ).cast[accum_type]()
                # Write bf16 pre-normalization sum for the residual stream.
                residual_output.store[width=simd_width](
                    IndexList[2](row, idx), vec_data.cast[in_dtype]()
                )

        # Phase 1: Compute mean-square.
        var thread_m2 = (vec_data**2).reduce_add()
        var row_m2 = block.sum[block_size=threads_per_block, broadcast=True](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
        )

        # Phase 2: Normalize + find max (preloaded gamma, no global load).
        var normalized = SIMD[accum_type, simd_width](0)
        var thread_max = Scalar[accum_type](0)

        if is_valid:
            normalized = (vec_data * norm_factor) * gamma_vec
            thread_max = abs(normalized).reduce_max()

        # Find maximum and compute scale.
        var row_max = block.max[block_size=threads_per_block, broadcast=True](
            thread_max
        )
        var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
            out_dtype
        ](row_max, scale_ub)
        if tid == 0:
            scale_buffer[row] = scale_factor

        # Phase 3: Quantize and write (normalized values in registers).
        if is_valid:
            var output_fp8 = fp8_quantize[out_dtype](
                normalized, scale_factor_recip
            )
            output_fn[simd_width](row, idx, output_fp8)

    # NOTE: No end barrier needed. The FP8 output is consumed only by the
    # local GPU, so stream ordering guarantees writes complete before the
    # next kernel reads them. The start barrier of the NEXT allreduce call
    # protects the input buffers (which ARE read by remote GPUs).


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(threads_per_block)
    )
)
fn _allreduce_rmsnorm_fp8_kernel_2stage[
    mut: Bool,
    origin: Origin[mut=mut],
    LayoutType: TensorLayout,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    scale_layout: LegacyLayout,
    //,
    ngpus: Int,
    simd_width: Int,
    threads_per_block: Int,
    has_residual: Bool,
    output_fn: fn[width: Int](
        row: Int, col: Int, val: SIMD[out_dtype, width]
    ) capturing -> None,
](
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    gamma: TileTensor[in_dtype, LayoutType, origin],
    scale_buffer: LegacyLayoutTensor[
        mut=True, scales_dtype, scale_layout, MutAnyOrigin
    ],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    rows: Int,
    cols: Int,
    scale_ub: Scalar[scales_dtype],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin],
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin],
):
    """Single-kernel 2-stage fused RS + RMSNorm + FP8 + AG.

    Stage 1 (RS + RMSNorm + FP8): Each block reduces rows in its partition
    from all GPUs in f32 registers, normalizes, quantizes to FP8, and writes
    compact results (fp8 + scale + optional bf16 residual) to local scratch.
    No f32 scratch is written — data stays in registers through RS → Norm → FP8.

    Inter-stage: Per-block barrier with release/acquire fence ensures all
    scratch writes from block B on every GPU are visible before block B
    on any GPU starts Stage 2.

    Stage 2 (AG copy): Each block iterates over each GPU's partition and
    reads fp8/scale/bf16 from that GPU's scratch via P2P, then writes to
    local output buffers. Consecutive reads from the same peer improve
    NVLink pipelining. No compute — just copies of compact data.

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
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime accum_type = get_accum_type[in_dtype]()
    comptime align = align_of[SIMD[in_dtype, simd_width]]()

    var tid = Int(thread_idx.x)
    var col_idx = tid * simd_width
    var is_valid = col_idx < cols
    var num_blocks = Int(grid_dim.x)

    # Row-aligned partitioning: each GPU owns ceildiv(rows, ngpus) rows.
    var rows_per_rank = ceildiv(rows, ngpus)
    var my_row_start = my_rank * rows_per_rank

    # --- Scratch pointers (after Signal header) ---
    # Layout: [fp8_data | scales (padded) | bf16_residual (optional)]
    # See kernel docstring for the full size formula.
    var fp8_per_rank = rows_per_rank * cols
    debug_assert(
        fp8_per_rank % 4 == 0,
        "fp8 scratch must be 4-byte aligned for scale stores",
    )

    # Round up the scale element count to the next multiple of
    # (simd_width / sizeof(scales_dtype)) so that the residual scratch
    # pointer (scale_ptr + scale_pad_elements) is simd_width-byte aligned.
    # Without this, rows_per_rank * sizeof(scales_dtype) may not be
    # divisible by simd_width, causing a misaligned SIMD residual store
    # when ceildiv(rows, ngpus) % (simd_width / sizeof(scales_dtype)) != 0.
    comptime scales_per_simd_chunk = simd_width // size_of[scales_dtype]()
    var scale_pad_elements = (
        ceildiv(rows_per_rank, scales_per_simd_chunk) * scales_per_simd_chunk
    )

    # Local scratch pointers (for writes in Stage 1).
    # +1 advances the Signal pointer by sizeof(Signal) bytes, stepping
    # past the Signal header to reach the scratch region of the buffer.
    var scratch_fp8 = (
        rank_sigs[my_rank].address_space_cast[AddressSpace.GENERIC]() + 1
    ).bitcast[Scalar[out_dtype]]()
    var scratch_scale = (scratch_fp8 + fp8_per_rank).bitcast[
        Scalar[scales_dtype]
    ]()
    var scratch_residual = (scratch_scale + scale_pad_elements).bitcast[
        Scalar[in_dtype]
    ]()

    # P2P scratch pointers for all GPUs (for reads in Stage 2).
    var fp8_ptrs = InlineArray[
        UnsafePointer[Scalar[out_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var scale_ptrs = InlineArray[
        UnsafePointer[Scalar[scales_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)
    var residual_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], MutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        # +1 advances by sizeof(Signal) bytes (see local scratch note above).
        var base_i = (
            rank_sigs[i].address_space_cast[AddressSpace.GENERIC]() + 1
        ).bitcast[Scalar[out_dtype]]()
        fp8_ptrs[i] = base_i
        scale_ptrs[i] = (base_i + fp8_per_rank).bitcast[Scalar[scales_dtype]]()
        comptime if has_residual:
            residual_ptrs[i] = (scale_ptrs[i] + scale_pad_elements).bitcast[
                Scalar[in_dtype]
            ]()

    # Round-robin P2P input pointers for NVLink load-balancing.
    var ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        var target = (my_rank + i) % ngpus
        ptrs[i] = src_ptrs[target]

    # Preload gamma weights into registers BEFORE start barrier.
    # Gamma is local data — safe to load early, hides latency behind barrier.
    var gamma_vec = SIMD[accum_type, simd_width](0)
    if is_valid:
        gamma_vec = (
            gamma.load[width=simd_width, alignment=align](
                Coord(Idx(col_idx))
            ).cast[accum_type]()
            + weight_offset.cast[accum_type]()
        )

    # Start barrier: wait for all GPUs to have their input data ready.
    _multi_gpu_barrier[ngpus, is_start=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # === Stage 1: RS + RMSNorm + FP8 (partition rows only) ===
    # Each block iterates over its partition rows directly. With
    # grid_dim <= rows_per_rank, every block has at least one row.
    for local_row in range(Int(block_idx.x), rows_per_rank, num_blocks):
        var row = my_row_start + local_row
        if row < rows:
            # P2P load from all GPUs + accumulate in f32 registers.
            var accum = SIMD[accum_type, simd_width](0)
            if is_valid:
                var global_elem = row * cols + col_idx

                comptime for gpu_idx in range(ngpus):
                    accum += (
                        ptrs[gpu_idx]
                        .address_space_cast[_target_address_space]()
                        .load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](global_elem)
                        .cast[accum_type]()
                    )

                # Add residual and write residual to scratch (compile-time gated).
                comptime if has_residual:
                    accum += residual.load[width=simd_width](
                        IndexList[2](row, col_idx)
                    ).cast[accum_type]()
                    var local_elem = local_row * cols + col_idx
                    scratch_residual.address_space_cast[
                        _target_address_space
                    ]().store[width=simd_width, alignment=simd_width](
                        local_elem, accum.cast[in_dtype]()
                    )

            # RMSNorm: compute mean-square.
            var thread_m2 = (accum**2).reduce_add()
            var row_m2 = block.sum[
                block_size=threads_per_block, broadcast=True
            ](thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](cols)) + epsilon.cast[accum_type]()
            )

            # Normalize + find max for FP8 scale.
            var normalized = SIMD[accum_type, simd_width](0)
            var thread_max = Scalar[accum_type](0)
            if is_valid:
                normalized = (accum * norm_factor) * gamma_vec
                thread_max = abs(normalized).reduce_max()

            var row_max = block.max[
                block_size=threads_per_block, broadcast=True
            ](thread_max)
            var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                out_dtype
            ](row_max, scale_ub)

            # Write scale to local scratch.
            if tid == 0:
                scratch_scale.address_space_cast[_target_address_space]().store(
                    local_row, scale_factor
                )

            # Quantize and write fp8 to local scratch.
            if is_valid:
                var output_fp8 = fp8_quantize[out_dtype](
                    normalized, scale_factor_recip
                )
                var local_elem = local_row * cols + col_idx
                scratch_fp8.address_space_cast[_target_address_space]().store[
                    width=simd_width, alignment=simd_width
                ](local_elem, output_fp8)

    # Per-block barrier with fence: block B waits for block B on all GPUs.
    # syncthreads + store_release + load_acquire ensures all of block B's
    # Stage 1 writes on every GPU are visible before Stage 2 reads them.
    _multi_gpu_barrier[ngpus, is_start=False, need_fence=True](
        rank_sigs, rank_sigs[my_rank], my_rank
    )

    # === Stage 2: All-Gather (lightweight P2P copy, no compute) ===
    # Batch reads by GPU: consecutive reads from the same peer improve
    # NVLink pipelining. No runtime owner_rank division per row.
    comptime for gpu in range(ngpus):
        var gpu_row_start = gpu * rows_per_rank
        var gpu_row_end = min(gpu_row_start + rows_per_rank, rows)
        var gpu_rows = gpu_row_end - gpu_row_start
        for local_row in range(Int(block_idx.x), gpu_rows, num_blocks):
            var row = gpu_row_start + local_row
            var local_elem = local_row * cols + col_idx

            # Read fp8 from gpu's scratch and write to output.
            if is_valid:
                var fp8_val = (
                    fp8_ptrs[gpu]
                    .address_space_cast[_target_address_space]()
                    .load[
                        width=simd_width,
                        alignment=simd_width,
                        invariant=True,
                    ](local_elem)
                )
                output_fn[simd_width](row, col_idx, fp8_val)

            # Thread 0: read scale from gpu's scratch → scale_buffer.
            if tid == 0:
                scale_buffer[row] = (
                    scale_ptrs[gpu]
                    .address_space_cast[_target_address_space]()
                    .load[invariant=True](local_row)
                )

            # Read bf16 residual from gpu's scratch (compile-time gated).
            comptime if has_residual:
                if is_valid:
                    var bf16_val = (
                        residual_ptrs[gpu]
                        .address_space_cast[_target_address_space]()
                        .load[
                            width=simd_width,
                            alignment=simd_width,
                            invariant=True,
                        ](local_elem)
                    )
                    residual_output.store[width=simd_width](
                        IndexList[2](row, col_idx), bf16_val
                    )

    # NOTE: No end barrier needed (same reasoning as 1-stage kernel).


# --- Launcher ---


fn _allreduce_rmsnorm_fp8_launch[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    output: NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    ctx: DeviceContext,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin] = NDBuffer[
        in_dtype, 2, ImmutAnyOrigin
    ](),
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin] = NDBuffer[
        in_dtype, 2, MutAnyOrigin
    ](),
) raises:
    """Launch the fused allreduce + RMSNorm + FP8 kernel."""
    comptime sm_version = get_sm_version()
    var payload_bytes = rows * cols * size_of[in_dtype]()
    var max_blocks = _dispatch_max_num_blocks[ngpus, sm_version](payload_bytes)
    var grid_dim = min(rows, max_blocks)
    var block_dim = threads_per_block

    @always_inline
    @parameter
    @__copy_capture(output)
    fn output_fn[width: Int](row: Int, col: Int, val: SIMD[out_dtype, width]):
        output.store[width=width](IndexList[2](row, col), val)

    # Create a scale buffer tensor from scale_output.
    comptime layout_1d = LegacyLayout.row_major(UNKNOWN_VALUE)
    var scale_buffer_tensor = LegacyLayoutTensor[
        mut=True,
        scales_dtype,
        layout_1d,
        MutAnyOrigin,
        address_space = scale_output.address_space,
    ](scale_output.data, RuntimeLayout[layout_1d].row_major(Index(rows)))

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_warp_tiling[
        mut = gamma.mut,
        origin = gamma.origin,
        LayoutType = gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_layout=layout_1d,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        output_fn=output_fn,
    ]
    ctx.enqueue_function[kernel, kernel](
        src_ptrs,
        gamma,
        scale_buffer_tensor,
        epsilon,
        weight_offset,
        rows,
        cols,
        scale_ub.cast[scales_dtype](),
        rank_sigs,
        my_rank,
        residual,
        residual_output,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


fn _allreduce_rmsnorm_fp8_launch_2stage[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    output: NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    my_rank: Int,
    ctx: DeviceContext,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin] = NDBuffer[
        in_dtype, 2, ImmutAnyOrigin
    ](),
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin] = NDBuffer[
        in_dtype, 2, MutAnyOrigin
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
    var max_blocks = _dispatch_max_num_blocks[ngpus, sm_version](payload_bytes)
    var grid_dim = min(ceildiv(rows, ngpus), max_blocks)
    var block_dim = threads_per_block

    # Validate that the signal buffer scratch layout is correctly aligned.
    # The 2-stage kernel places fp8 data immediately after the Signal header,
    # then scale values (padded to simd_width bytes) immediately after the
    # fp8 block, then the residual (if has_residual). For scale values to
    # be properly aligned, the fp8 block size must be a multiple of
    # sizeof(scales_dtype). The scale section is padded to simd_width bytes
    # so the residual section is simd_width-byte aligned.
    # Callers must also ensure each rank_sigs[i] buffer has at least:
    #   sizeof(Signal) + fp8_scratch + scale_scratch_padded bytes
    # (plus residual_scratch = rows_per_rank * cols * sizeof(in_dtype) if
    # has_residual). See the kernel docstring for the full formula.
    var _rows_per_rank = ceildiv(rows, ngpus)
    var _fp8_scratch = _rows_per_rank * cols
    var _scale_scratch_raw = _rows_per_rank * size_of[scales_dtype]()
    # Pad scale section to simd_width bytes (matches scale_pad_elements in kernel).
    var _scale_scratch = ceildiv(_scale_scratch_raw, simd_width) * simd_width
    var _min_buf = size_of[Signal]() + _fp8_scratch + _scale_scratch
    comptime if has_residual:
        _min_buf += _rows_per_rank * cols * size_of[in_dtype]()
    debug_assert(
        _fp8_scratch % size_of[scales_dtype]() == 0,
        String("2-stage: fp8 scratch (")
        + String(_fp8_scratch)
        + " B) must be a multiple of sizeof(scales_dtype) for scale pointer"
        + " alignment; rank_sigs[i] must be >= "
        + String(_min_buf)
        + " bytes",
    )
    comptime if has_residual:
        debug_assert(
            (_fp8_scratch + _scale_scratch) % simd_width == 0,
            String("2-stage: residual scratch offset (fp8=")
            + String(_fp8_scratch)
            + " B + scales_padded="
            + String(_scale_scratch)
            + " B = "
            + String(_fp8_scratch + _scale_scratch)
            + " B) must be a multiple of simd_width ("
            + String(simd_width)
            + " B) for SIMD residual stores",
        )

    @always_inline
    @parameter
    @__copy_capture(output)
    fn output_fn[width: Int](row: Int, col: Int, val: SIMD[out_dtype, width]):
        output.store[width=width](IndexList[2](row, col), val)

    # Create a scale buffer tensor from scale_output.
    comptime layout_1d = LegacyLayout.row_major(UNKNOWN_VALUE)
    var scale_buffer_tensor = LegacyLayoutTensor[
        mut=True,
        scales_dtype,
        layout_1d,
        MutAnyOrigin,
        address_space = scale_output.address_space,
    ](scale_output.data, RuntimeLayout[layout_1d].row_major(Index(rows)))

    comptime kernel = _allreduce_rmsnorm_fp8_kernel_2stage[
        mut = gamma.mut,
        origin = gamma.origin,
        LayoutType = gamma.LayoutType,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        scales_dtype=scales_dtype,
        scale_layout=layout_1d,
        ngpus=ngpus,
        simd_width=simd_width,
        threads_per_block=threads_per_block,
        has_residual=has_residual,
        output_fn=output_fn,
    ]
    ctx.enqueue_function[kernel, kernel](
        src_ptrs,
        gamma,
        scale_buffer_tensor,
        epsilon,
        weight_offset,
        rows,
        cols,
        scale_ub.cast[scales_dtype](),
        rank_sigs,
        my_rank,
        residual,
        residual_output,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


# --- Split (2-kernel) path ---


fn _launch_split_allreduce_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
](
    rows: Int,
    cols: Int,
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    output_2d: NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin],
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin],
) raises:
    """Two-kernel fallback: allreduce+add epilogue, then rmsnorm+fp8.

    Faster than fused 2-stage for large with-residual payloads because
    it avoids carrying bf16 residual data through scratch buffers.
    """
    # Reconstruct NDBuffer inputs for allreduce.
    var input_buffers = InlineArray[
        NDBuffer[in_dtype, 2, ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        input_buffers[i] = NDBuffer[in_dtype, 2, ImmutAnyOrigin](
            src_ptrs[i], DimList(rows, cols)
        )

    var res_ptr = residual.data
    var res_out_ptr = residual_output.data
    var _cols = cols

    # Define input_fn for RMSNorm (reads from residual_output after allreduce).
    @__copy_capture(res_out_ptr, _cols)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[in_dtype, width]:
        var li = idx[0] * _cols + idx[1]
        return res_out_ptr.load[width=width, alignment=width](li)

    var shape = IndexList[2](rows, cols)
    var scale_output_2d = NDBuffer[mut=True, scales_dtype, 2, MutAnyOrigin](
        scale_output_1d.data, DimList(rows, 1)
    )

    # Pre-compile the RMSNorm+FP8 kernel before launching allreduce.
    # This avoids a deadlock where cuModuleLoadDataEx (JIT compilation)
    # blocks on the CUDA context while the allreduce kernel is waiting
    # at a multi-GPU barrier for other ranks to be enqueued.
    rms_norm_fused_fp8[
        in_dtype, out_dtype, scales_dtype, 2, input_fn, compile_only=True
    ](
        shape,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        DeviceContextPtr(ctx),
        scale_ub,
        scale_output_2d,
    )

    # Step 1: Allreduce with add epilogue → residual_output.
    @__copy_capture(res_ptr, res_out_ptr, _cols)
    @always_inline
    @parameter
    fn add_epilogue[
        _dtype: DType,
        _rank: Int,
        _width: Int,
        *,
        _alignment: Int,
    ](coords: IndexList[_rank], val: SIMD[_dtype, size=_width]) -> None:
        var li = coords[0] * _cols + coords[1]
        var res = res_ptr.load[width=_width, alignment=_alignment](li)
        # Add in f32 for precision parity with the fused kernel,
        # which accumulates allreduce + residual in f32 before casting.
        var sum_f32 = (
            val.cast[DType.float32]()
            + rebind[SIMD[_dtype, _width]](res).cast[DType.float32]()
        )
        res_out_ptr.store[width=_width, alignment=_alignment](
            li,
            rebind[SIMD[in_dtype, _width]](sum_f32.cast[_dtype]()),
        )

    allreduce[
        ngpus=ngpus,
        output_lambda = Optional[elementwise_epilogue_type](add_epilogue),
    ](input_buffers, residual_output, rank_sigs, ctx)

    # Step 2: Fused RMSNorm + FP8 on residual_output (kernel already compiled).
    rms_norm_fused_fp8[in_dtype, out_dtype, scales_dtype, 2, input_fn](
        shape,
        output_2d,
        gamma,
        epsilon,
        weight_offset,
        DeviceContextPtr(ctx),
        scale_ub,
        scale_output_2d,
    )


# --- Dispatch ---


fn _dispatch_fused_kernel[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    has_residual: Bool = False,
](
    rows: Int,
    cols: Int,
    src_ptrs: InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ],
    output_2d: NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output_1d: NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    residual: NDBuffer[in_dtype, 2, ImmutAnyOrigin] = NDBuffer[
        in_dtype, 2, ImmutAnyOrigin
    ](),
    residual_output: NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin] = NDBuffer[
        in_dtype, 2, MutAnyOrigin
    ](),
) raises:
    """Dispatch the fused kernel with appropriate simd width and stage count.

    Centralizes the dispatch logic shared by allreduce_rmsnorm_fp8 and
    allreduce_residual_rmsnorm_fp8. Selects simd width based on column count
    and 1-stage vs 2-stage based on payload size.
    """
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime base_simd_width = simd_width_of[
        in_dtype, target = get_gpu_target()
    ]()
    comptime sw1 = base_simd_width
    comptime sw2 = base_simd_width * 2

    # Use 2-stage (reduce-scatter + fused all-gather) for large payloads.
    # Threshold is per-rank payload: each GPU processes ceildiv(rows, ngpus)
    # full rows, so per-rank bytes stays roughly constant across column widths.
    var per_rank_bytes = ceildiv(rows, ngpus) * cols * size_of[in_dtype]()

    # Override for threshold tuning experiments:
    #   0 = use default threshold (production)
    #   1 = force 1-stage for all sizes
    #   2 = force 2-stage for all sizes
    #   3 = force split (2-kernel) path for all sizes (residual only)
    comptime force_stage = env_get_int["FORCE_ALLREDUCE_STAGE", 0]()

    # Per-rank byte thresholds for switching to 2-stage allreduce.
    # Derived from empirical sweep data (bf16, per-rank crossover points):
    #   MI355 4GPU: 128 KB non-res,  96 KB residual
    #   MI355 8GPU:  80 KB non-res,  96 KB residual
    #   B200  4GPU: 512 KB non-res, 256 KB residual
    #   B200  8GPU:  80 KB non-res,  80 KB residual
    @parameter
    fn _rank_4_per_rank_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return 128 * 1024 if not has_residual else 96 * 1024
        else:
            return 512 * 1024 if not has_residual else 256 * 1024

    @parameter
    fn _rank_8_per_rank_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return 80 * 1024 if not has_residual else 96 * 1024
        else:
            return 80 * 1024

    comptime threshold = _rank_4_per_rank_thresh() if ngpus <= 4 else _rank_8_per_rank_thresh()

    # Per-rank byte thresholds for switching to split (2-kernel) path
    # when has_residual=True. Above this, allreduce+add + rmsnorm_fp8 beats
    # fused 2-stage because it avoids carrying bf16 residual through scratch.
    #   MI355: column-dependent; 2-stage beats split for cols <= 8192,
    #          split beats 2-stage for cols > 8192. See dispatch below.
    #   B200 4GPU: 1536 KB per-rank crossover
    #   B200 8GPU: conservative, same as 2-stage threshold
    @parameter
    fn _rank_4_split_thresh() -> Int:
        comptime if has_amd_gpu_accelerator():
            return _rank_4_per_rank_thresh()
        else:
            return 1536 * 1024

    @parameter
    fn _rank_8_split_thresh() -> Int:
        # For 8 GPUs the split threshold equals the 2-stage threshold on
        # both AMD and NVIDIA.  This intentionally means the 2-stage
        # *residual* path is never selected at 8 GPUs: once per-rank
        # payload crosses threshold the dispatch jumps straight to split,
        # skipping the `elif per_rank_bytes >= threshold` arm below.
        # Benchmarks show the split (2-kernel) path is always faster than
        # fused 2-stage with residual at 8-GPU scale because the extra
        # bf16 scratch traffic in Stage 2 outweighs the launch overhead.
        comptime if has_amd_gpu_accelerator():
            return _rank_8_per_rank_thresh()
        else:
            return _rank_8_per_rank_thresh()

    comptime split_threshold = _rank_4_split_thresh() if ngpus <= 4 else _rank_8_split_thresh()

    var use_2stage: Bool
    var use_split: Bool = False
    comptime if force_stage == 1:
        use_2stage = False
    elif force_stage == 2:
        use_2stage = True
    elif force_stage == 3:
        # Force split (2-kernel) path.
        use_2stage = False
        comptime if has_residual:
            use_split = True
    else:
        comptime if has_residual:
            comptime if has_amd_gpu_accelerator():
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
        else:
            use_2stage = ngpus <= 8 and per_rank_bytes >= threshold

    @parameter
    fn launch_1stage[sw: Int]() raises:
        _allreduce_rmsnorm_fp8_launch[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
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
            Int(ctx.id()),
            ctx,
            residual,
            residual_output,
        )

    @parameter
    fn launch_2stage[sw: Int]() raises:
        _allreduce_rmsnorm_fp8_launch_2stage[
            sw,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
            has_residual=has_residual,
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
            Int(ctx.id()),
            ctx,
            residual,
            residual_output,
        )

    if use_split:
        comptime if has_residual:
            _launch_split_allreduce_rmsnorm_fp8[
                in_dtype, out_dtype, scales_dtype, ngpus
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
                ctx,
                residual,
                residual_output,
            )
        return

    # Warp-tiling: each thread handles simd_width elements. Wider simd for
    # larger column counts.
    if cols <= (WARP_SIZE * sw1 * max_warps_per_block):
        if use_2stage:
            launch_2stage[sw1]()
        else:
            launch_1stage[sw1]()
    elif cols <= (WARP_SIZE * sw2 * max_warps_per_block) and cols % sw2 == 0:
        if use_2stage:
            launch_2stage[sw2]()
        else:
            launch_1stage[sw2]()
    else:
        comptime max_cols = WARP_SIZE * sw2 * max_warps_per_block
        raise Error(
            "allreduce_rmsnorm_fp8: cols ("
            + String(cols)
            + ") exceeds max supported ("
            + String(max_cols)
            + ") for warp-tiling kernel"
        )


# --- Public API ---


fn allreduce_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    ngpus: Int,
    //,
](
    input_buffers: InlineArray[NDBuffer[in_dtype, rank, ImmutAnyOrigin], ngpus],
    output: NDBuffer[mut=True, out_dtype, rank, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: NDBuffer[mut=True, scales_dtype, rank, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Fused allreduce + RMSNorm + FP8 quantization.

    Combines P2P allreduce across GPUs, RMSNorm normalization, and FP8
    dynamic quantization into a single kernel launch. Eliminates the
    global memory round-trip between allreduce output and RMSNorm input.

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: FP8 output data type (e.g. float8_e4m3fn).
        scales_dtype: Scale factor data type (e.g. float32).
        rank: Tensor rank of input/output/scale buffers.
        ngpus: Number of GPUs participating.

    Args:
        input_buffers: Per-GPU input buffers (last dim = cols).
        output: Output buffer for FP8 values (same shape as input).
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping.
        scale_output: Output buffer for per-row FP8 scales (last dim = 1).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.

    Note:
        This kernel does not issue an end barrier. The FP8 output and
        scale buffers are safe to read only on the local GPU (stream
        ordering guarantees visibility). If a remote GPU needs to read
        these outputs, the caller must insert an explicit barrier.
        The start barrier of the NEXT allreduce call protects the
        input buffers that are read by remote GPUs.

        Signal buffer sizing:
          1-stage path (payload < threshold): size_of[Signal]() only.
          2-stage path (payload > threshold):
            size_of[Signal]()
            + ceildiv(rows, ngpus) * cols                               (fp8 data)
            + align_up(ceildiv(rows, ngpus) * sizeof(scales_dtype),
                       simd_width)                                       (scales + pad)
    """
    comptime assert ngpus >= 2, "allreduce_rmsnorm_fp8 requires at least 2 GPUs"
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"

    if not is_p2p_enabled():
        raise Error("allreduce_rmsnorm_fp8 requires P2P access between GPUs")

    var cols = input_buffers[0].dim(rank - 1)
    var rows = input_buffers[0].num_elements() // cols

    # Extract raw pointers from NDBuffers.
    var src_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        src_ptrs[i] = input_buffers[i].data

    # Create internal 2D/1D views and dispatch.
    var output_2d = NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin](
        output.data, DimList(rows, cols)
    )
    var scale_output_1d = NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin](
        scale_output.data, DimList(rows)
    )

    _dispatch_fused_kernel[in_dtype, out_dtype, scales_dtype, ngpus](
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
        ctx,
    )


fn allreduce_residual_rmsnorm_fp8[
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    rank: Int,
    ngpus: Int,
    //,
](
    input_buffers: InlineArray[NDBuffer[in_dtype, rank, ImmutAnyOrigin], ngpus],
    residual: NDBuffer[in_dtype, rank, ImmutAnyOrigin],
    output: NDBuffer[mut=True, out_dtype, rank, ...],
    residual_output: NDBuffer[mut=True, in_dtype, rank, ...],
    gamma: TileTensor[in_dtype, ...],
    epsilon: Scalar[in_dtype],
    weight_offset: Scalar[in_dtype],
    scale_ub: Float32,
    scale_output: NDBuffer[mut=True, scales_dtype, rank, ...],
    rank_sigs: InlineArray[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
) raises:
    """Fused allreduce + residual add + RMSNorm + FP8 quantization.

    Combines P2P allreduce across GPUs, element-wise residual addition,
    RMSNorm normalization, and FP8 dynamic quantization. Produces two outputs:
      - residual_output: pre-normalization sum stored as `in_dtype` (allreduce + residual),
        carried forward as the new residual state.
      - output: FP8 quantized post-normalization values consumed by the
        next linear layer.

    For small payloads, uses a single fused kernel (1-stage or 2-stage).
    For large payloads, automatically falls back to a split (2-kernel) path
    — allreduce with add epilogue followed by fused rmsnorm+fp8 — which
    avoids carrying bf16 residual data through scratch buffers.

    Parameters:
        in_dtype: Input data type (e.g. bfloat16).
        out_dtype: FP8 output data type (e.g. float8_e4m3fn).
        scales_dtype: Scale factor data type (e.g. float32).
        rank: Tensor rank of input/output/scale buffers.
        ngpus: Number of GPUs participating.

    Args:
        input_buffers: Per-GPU input buffers (last dim = cols).
        residual: Residual buffer to add (same shape as input). Should be
            allocated on the local GPU for optimal performance. In 1-stage
            mode, all rows are read; in 2-stage mode, only the local
            partition rows are read.
        output: Output buffer for FP8 values (same shape as input).
        residual_output: Output buffer for pre-norm sum stored as `in_dtype` (same shape).
            Also serves as the intermediate buffer in the split path.
        gamma: RMSNorm gamma weights (1D TileTensor of length cols).
        epsilon: RMSNorm epsilon for numerical stability.
        weight_offset: Additive offset for gamma weights.
        scale_ub: Upper bound for FP8 scale clamping.
        scale_output: Output buffer for per-row FP8 scales (last dim = 1).
        rank_sigs: Per-GPU signal pointers for synchronization.
        ctx: Device context for this GPU.

    Note:
        residual_output is written by the kernel on each GPU independently.
        In multi-GPU usage, each GPU should pass its own local residual_output
        buffer. All GPUs compute the same pre-norm sum, so the contents will
        be identical across ranks.

        Precondition: ``residual`` must contain identical data across all
        participating GPUs. In the 2-stage path each GPU reads only its own
        partition of ``residual`` during Stage 1 and broadcasts those
        pre-norm sums to peers via scratch; Stage 2 assembles the full
        ``residual_output`` from those broadcasts. If ``residual`` differs
        across ranks (e.g. sharded activations), ``residual_output`` will
        be silently incorrect. In standard tensor-parallel usage activations
        are replicated, so this condition holds by construction.

        Signal buffer sizing for 2-stage path (payload > threshold):
          size_of[Signal]()
          + ceildiv(rows, ngpus) * cols                                   (fp8 data)
          + align_up(ceildiv(rows, ngpus) * sizeof(scales_dtype),
                     simd_width)                                           (scales + pad)
          + ceildiv(rows, ngpus) * cols * sizeof(in_dtype)                (residual)

        The split path uses allreduce's own signal buffer sizing, which is
        strictly smaller than the fused 2-stage with residual sizing above.

        The scale section is padded to a simd_width-byte boundary so the
        residual scratch is simd_width-byte aligned for SIMD stores. The
        padding is at most simd_width-1 bytes per signal buffer.
    """
    comptime assert (
        in_dtype.is_floating_point()
    ), "in_dtype must be floating point"
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"

    if not is_p2p_enabled():
        raise Error(
            "allreduce_residual_rmsnorm_fp8 requires P2P access between GPUs"
        )

    var cols = input_buffers[0].dim(rank - 1)
    var rows = input_buffers[0].num_elements() // cols

    # Extract raw pointers from NDBuffers.
    var src_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    comptime for i in range(ngpus):
        src_ptrs[i] = input_buffers[i].data

    # Create internal 2D/1D views and dispatch.
    var output_2d = NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin](
        output.data, DimList(rows, cols)
    )
    var residual_2d = NDBuffer[in_dtype, 2, ImmutAnyOrigin](
        residual.data, DimList(rows, cols)
    )
    var residual_output_2d = NDBuffer[mut=True, in_dtype, 2, MutAnyOrigin](
        residual_output.data, DimList(rows, cols)
    )
    var scale_output_1d = NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin](
        scale_output.data, DimList(rows)
    )

    _dispatch_fused_kernel[
        in_dtype, out_dtype, scales_dtype, ngpus, has_residual=True
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
        ctx,
        residual_2d,
        residual_output_2d,
    )
