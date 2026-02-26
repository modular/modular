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

Design:
  1. P2P loads from all GPUs (like 1-stage allreduce's load_reduce)
  2. Accumulation in float32 registers (for bfloat16 inputs)
  3. RMSNorm computation (warp-tiling: persistent row loop)
  4. FP8 dynamic per-row quantization

Each block processes multiple rows via a grid-strided loop, allowing
row counts beyond MAX_NUM_BLOCKS_UPPER_BOUND (512). Gamma weights are
preloaded once and reused across all rows in the loop.
"""

from collections import InlineArray
from math import clamp, rsqrt
from sys import align_of, is_amd_gpu, simd_width_of, size_of
from sys.info import _accelerator_arch

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu.host import DeviceContext, get_gpu_target
from gpu.primitives import block
from layout._coord import Coord, Idx
from layout._layout import TensorLayout, row_major
from layout._tile_tensor import TileTensor
from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import Layout as LegacyLayout
from layout.layout_tensor import LayoutTensor as LegacyLayoutTensor
from layout.runtime_layout import RuntimeLayout
from utils import IndexList
from utils.index import Index
from utils.numerics import get_accum_type, max_finite, min_finite

from .reducescatter import _target_address_space
from .sync import (
    MAX_GPUS,
    MAX_NUM_BLOCKS_UPPER_BOUND,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
)


# --- Inlined FP8 utilities (from linalg/fp8_utils.mojo) ---
# Inlined to avoid circular dependency: linalg depends on comm.
# TODO(KERN-2477): Move fp8_utils to a shared lower-level package
# (e.g. internal_utils/) so both linalg and comm can import them.


@always_inline
fn _compute_dynamic_fp8_scale[
    out_dtype: DType,
](
    row_max: Scalar,
    scale_ub: Scalar,
) -> Tuple[
    type_of(scale_ub), type_of(row_max)
]:
    """Compute dynamic FP8 scale factor and its reciprocal from a row max.

    See linalg/fp8_utils.mojo:compute_dynamic_fp8_scale for canonical version.
    """
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"

    comptime fp8_max = max_finite[out_dtype]()
    var scale_factor = (
        min(row_max.cast[scale_ub.dtype](), scale_ub)
        / fp8_max.cast[scale_ub.dtype]()
    )
    var scale_factor_recip = type_of(row_max)(
        0.0 if scale_factor == 0.0 else 1.0 / scale_factor.cast[row_max.dtype]()
    )
    return (scale_factor, scale_factor_recip)


@always_inline
fn _fp8_quantize[
    out_dtype: DType,
    *,
    use_clamp: Bool = is_amd_gpu(),
](values: SIMD, scale_recip: Scalar[values.dtype]) -> SIMD[
    out_dtype, values.size
]:
    """Quantize values to FP8, optionally clamping to the representable range.

    See linalg/fp8_utils.mojo:fp8_quantize for canonical version.
    """
    comptime assert out_dtype.is_float8(), "out_dtype must be float8"
    var result = values * scale_recip

    @parameter
    if use_clamp:
        comptime min_val = SIMD[values.dtype, values.size](
            min_finite[out_dtype]()
        )
        comptime max_val = SIMD[values.dtype, values.size](
            max_finite[out_dtype]()
        )
        return clamp(result, min_val, max_val).cast[out_dtype]()
    else:
        return result.cast[out_dtype]()


# --- GPU Kernel ---


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

    @parameter
    for i in range(ngpus):
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

            @parameter
            for gpu_idx in range(ngpus):
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
        var scale_factor, scale_factor_recip = _compute_dynamic_fp8_scale[
            out_dtype
        ](row_max, scale_ub)
        if tid == 0:
            scale_buffer[row] = scale_factor

        # Phase 3: Quantize and write (normalized values in registers).
        if is_valid:
            var output_fp8 = _fp8_quantize[out_dtype](
                normalized, scale_factor_recip
            )
            output_fn[simd_width](row, idx, output_fp8)

    # NOTE: No end barrier needed. The FP8 output is consumed only by the
    # local GPU, so stream ordering guarantees writes complete before the
    # next kernel reads them. The start barrier of the NEXT allreduce call
    # protects the input buffers (which ARE read by remote GPUs).


# --- Launcher ---


fn _allreduce_rmsnorm_fp8_launch[
    simd_width: Int,
    in_dtype: DType,
    out_dtype: DType,
    scales_dtype: DType,
    ngpus: Int,
    threads_per_block: Int,
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
) raises:
    """Launch the fused allreduce + RMSNorm + FP8 kernel."""
    var grid_dim = min(rows, MAX_NUM_BLOCKS_UPPER_BOUND)
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
        grid_dim=grid_dim,
        block_dim=block_dim,
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
    """
    if not is_p2p_enabled():
        raise Error("allreduce_rmsnorm_fp8 requires P2P access between GPUs")

    var cols = input_buffers[0].dim(rank - 1)
    var rows = input_buffers[0].num_elements() // cols

    # Extract raw pointers from NDBuffers.
    var src_ptrs = InlineArray[
        UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin], ngpus
    ](uninitialized=True)

    @parameter
    for i in range(ngpus):
        src_ptrs[i] = input_buffers[i].data

    # Create internal 2D/1D views for the launcher.
    var output_2d = NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin](
        output.data, DimList(rows, cols)
    )
    var scale_output_1d = NDBuffer[mut=True, scales_dtype, 1, MutAnyOrigin](
        scale_output.data, DimList(rows)
    )

    # Dispatch based on column count.
    comptime hw_max_warps = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime max_warps_per_block = hw_max_warps
    comptime threads_per_block = max_warps_per_block * WARP_SIZE
    comptime base_simd_width = simd_width_of[
        in_dtype, target = get_gpu_target()
    ]()
    comptime sw1 = base_simd_width
    comptime sw2 = base_simd_width * 2

    # Warp-tiling: each thread handles simd_width elements. Wider simd for
    # larger column counts.
    if cols <= (WARP_SIZE * sw1 * max_warps_per_block):
        _allreduce_rmsnorm_fp8_launch[
            sw1,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
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
        )
    elif cols <= (WARP_SIZE * sw2 * max_warps_per_block) and cols % sw2 == 0:
        _allreduce_rmsnorm_fp8_launch[
            sw2,
            in_dtype,
            out_dtype,
            scales_dtype,
            ngpus,
            threads_per_block,
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
        )
    else:
        comptime max_cols = WARP_SIZE * sw2 * max_warps_per_block
        raise Error(
            "allreduce_rmsnorm_fp8: cols ("
            + String(cols)
            + ") exceeds max supported ("
            + String(max_cols)
            + ") for warp-tiling kernel"
        )
