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

"""Provides block-scaled quantization kernels for NVFP4, MXFP4, and MXFP8."""

from std.math import align_up, ceildiv, rsqrt
from std.math.uutils import uceildiv, udivmod, ufloordiv
from std.memory import unsafe_stack_allocation
from max.gpu import (
    block_idx,
    thread_idx,
    grid_dim,
    block_dim,
    global_idx,
    lane_id,
    warp_id,
    WARP_SIZE,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from max.gpu.host import DeviceContext, FuncAttribute
from layout import (
    coord,
    Coord,
    Idx,
    RowMajorLayout,
    DefaultEngine,
    ImmTileTensor,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.tile_layout import TensorLayout
from std.collections import OptionalReg
from std.logger import Logger
from max.gpu.primitives import warp
from max.gpu.primitives.warp import lane_group_max, shuffle_down, shuffle_xor
from std.complex import ComplexSIMD
from std.math import recip
from .block_scaled_utils import compute_mxfp8_block_scale
from .swiglu_activation import (
    apply_swiglu_activation,
    swiglu,
    swiglu_activation_type,
)
from .fp4_utils import (
    cast_float_to_fp4e2m1_amd,
    cast_fp32_to_fp4e2m1,
    cast_f4e2m1x2_to_fp16x2,
    compute_mxfp4_even_scale,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP4_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
    get_scale_factor,
    get_scaling_kind,
)
from max.gpu.host.info import B200, MI355X, _is_sm10x_gpu, _is_sm12x_gpu
from std.utils import StaticTuple
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from std.utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from std.memory import bitcast
from max.gpu.sync import named_barrier
from max.gpu.intrinsics import warpgroup_reg_dealloc
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from max.gpu.memory import (
    cp_async_bulk_prefetch,
    external_memory,
    fence_async_view_proxy,
)
from max.gpu.sync import barrier
from std.sys import size_of, align_of, get_defined_int
from layout.swizzle import make_swizzle
from max.algorithm import elementwise
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.sys import get_defined_bool
from std.sys.intrinsics import llvm_intrinsic
from linalg.matmul.gpu.sm100.block_scaled_dispatch import (
    heuristic_and_outliers_dispatch,
)
from max.gpu.primitives.grid_controls import PDLLevel
from linalg.matmul.gpu.sm100_structured.default.dispatch import DISPATCH_HIT
from max.gpu.primitives.grid_controls import (
    PDL,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.runtime.tracing import Trace, TraceLevel, trace_arg, get_safe_task_id
from std.collections.string.string_span import get_static_string
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100.tile_scheduler import RasterOrder
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)


@inline(.always)
def _rope_rotate[
    dtype: DType, width: SIMDLength, //, *, inverse: Bool
](val: SIMD[dtype, width], freq: SIMD[DType.float32, width]) -> SIMD[
    dtype, width
]:
    """Rotates the interleaved `(re, im)` pairs of `val` by the `(cos, sin)`
    pairs of `freq`, or by their conjugates when `inverse`, in fp32.

    The arithmetic matches `rope_ragged`, so a rotation followed by a
    quantize gives the same bytes as the two kernels in sequence.
    """
    var x_re, x_im = val.cast[DType.float32]().deinterleave()
    var f_re, f_im = freq.deinterleave()
    var f = ComplexSIMD(f_re, f_im)
    comptime if inverse:
        f = f.conj()
    var r = ComplexSIMD(x_re, x_im) * f
    return rebind[SIMD[dtype, width]](r.re.interleave(r.im).cast[dtype]())


########################################################
# Dynamic scaled NVFP4 quantization
########################################################

comptime logger = Logger()


@inline(.always)
def quantize_dynamic_scaled_fp4fp8[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 512,
    gated: Bool = False,
    activation: swiglu_activation_type = swiglu[0.0],
    rope_dim: Int = 0,
    rope_head_dim: Int = 0,
    rope_inverse: Bool = False,
    rope_layout: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
    grid_by_occupancy: Bool = False,
](
    ctx: DeviceContext,
    output_tile: TileTensor[mut=True, out_dtype, address_space=.GENERIC, ...],
    scales_tile: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    input_tile: TileTensor[mut=False, in_dtype, address_space=.GENERIC, ...],
    num_cols: Int,
    num_cols_padded: Int,
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
    rope_freqs: OptionalReg[
        TileTensor[.float32, rope_layout, ImmUnsafeAnyOrigin]
    ] = None,
) raises:
    """Launches the SM100 kernel that quantizes BF16 input to NVFP4, MXFP4, or MXFP8 with per-block scale factors.

    Each thread processes `ELEMENTS_PER_THREAD` elements, cooperatively
    reduces the per-group maximum across warp lanes, derives the scale
    factor, and writes the quantized output and scales tensors.

    Parameters:
        out_dtype: Element type of the quantized output tensor. Must
            be `uint8` for NVFP4 or MXFP4, or `float8_e4m3fn` for MXFP8
            (inferred).
        scales_dtype: Element type of the block scale-factor tensor.
            Must be `float8_e4m3fn` for NVFP4 or `float8_e8m0fnu` for
            MXFP4 and MXFP8 (inferred).
        in_dtype: Element type of the input activation tensor. Must
            be `bfloat16` (inferred).
        SF_VECTOR_SIZE: Number of input elements covered by each block
            scale factor: 16 for NVFP4 or MXFP4, 32 for MXFP8
            (defaults to 16).
        num_max_threads: Maximum number of threads per block for the
            launch grid (defaults to 512).
        gated: Whether each input row is `[gate | up]`, twice the output
            width, and the kernel quantizes `activation(gate, up)` rounded
            to `in_dtype` in place of the input.
        activation: The gated activation; read only when `gated`.
        rope_dim: Trailing columns of each head that the kernel rotates by
            `rope_freqs` before it quantizes; 0 rotates nothing.
        rope_head_dim: Columns per head; read only when `rope_dim > 0`.
        rope_inverse: Whether to rotate by the conjugate of `rope_freqs`.
        rope_layout: Layout of `rope_freqs` (inferred).
        grid_by_occupancy: Whether to size the grid by the blocks an SM
            holds at the kernel's register use, instead of by threads.
    Args:
        ctx: Device context used to enqueue the kernel.
        output_tile: Output tile for the quantized elements of shape
            `[num_rows, num_cols // 2]` for packed FP4 `uint8` or
            `[num_rows, num_cols]` for MXFP8.
        scales_tile: Output tile for the per-block scale factors.
        input_tile: Input `bfloat16` activation tile of shape
            `[num_rows, num_cols]`, or `[num_rows, 2 * num_cols]` when
            `gated`.
        num_cols: Number of columns in the input tensor before
            padding.
        num_cols_padded: Number of columns in the input tensor after
            padding to a multiple of `SF_VECTOR_SIZE`; padded columns
            are zeroed in the output and scales.
        tensor_sf: Tensor-wise scale factor applied to the
            quantization (defaults to 1.0).
        rope_freqs: `[num_rows, rope_dim]` `float32` interleaved
            `(cos, sin)` pairs, one row per input row; required when
            `rope_dim > 0`. Each head's rotated columns are rounded to
            `in_dtype` before they are quantized.
    """
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100 or SM120"
    comptime assert in_dtype in (
        DType.bfloat16,
    ), "input dtype should be bfloat16"

    comptime assert (
        (
            out_dtype == .uint8
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
            and scales_dtype == .float8_e4m3fn
        )
        or (
            out_dtype == .uint8
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
            and scales_dtype == .float8_e8m0fnu
        )
        or (
            out_dtype == .float8_e4m3fn
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
            and scales_dtype == .float8_e8m0fnu
        )
    ), "output dtype should be uint8 for NVFP4/MXFP4 or float8_e4m3fn for MXFP8"

    comptime N = input_tile.LayoutType.static_shape[1] // (2 if gated else 1)

    comptime if SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE:
        comptime assert N % SF_VECTOR_SIZE == 0, "N must be a multiple of 32"
    else:
        comptime assert (
            N % (SF_VECTOR_SIZE // 2) == 0
        ), "N must be a multiple of 8"

    comptime ELEMENTS_PER_THREAD = 8
    comptime num_SMs = B200.sm_count

    var num_rows = Int(input_tile.dim[0]())
    if num_rows == 0 or num_cols == 0:
        return
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)

    comptime kernel = quantize_dynamic_scaled_fp4fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        output_tile.LayoutType,
        scales_tile.LayoutType,
        input_tile.LayoutType,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        num_max_threads=num_max_threads,
        gated=gated,
        activation=activation,
        rope_dim=rope_dim,
        rope_head_dim=rope_head_dim,
        rope_inverse=rope_inverse,
        rope_layout=rope_layout,
    ]

    var block_dim = (
        min(num_cols // ELEMENTS_PER_THREAD, num_max_threads),
        1,
        1,
    )
    var num_blocks_per_SM = max(
        1, B200.gpu_info.threads_per_multiprocessor // block_dim[0]
    )
    comptime if grid_by_occupancy:
        # One resident wave: a second, partial wave of blocks waits for the
        # first one's rows.
        var func = ctx.compile_function[kernel]()
        num_blocks_per_SM = max(
            1,
            func.occupancy_max_active_blocks_per_multiprocessor(
                block_dim[0], 0
            ),
        )
    var num_slots = num_SMs * num_blocks_per_SM
    var grid_x = min(num_rows_padded, num_slots)
    # A row wider than one block's pass loops over its columns one load
    # latency at a time. With few rows, more blocks per row issue those loads
    # at once. Blocks past the real rows zero the padding rows' scales, so no
    # block walks the padding alone. Up to one block per SM: with two, the
    # pair on an SM can be one row's blocks, leaving half the SMs idle.
    var col_blocks = min(
        ceildiv(
            align_up(num_cols_padded, SF_VECTOR_SIZE * SF_ATOM_K)
            // ELEMENTS_PER_THREAD,
            block_dim[0],
        ),
        max(1, num_slots // num_rows),
    )
    if col_blocks > 1:
        grid_x = min(num_rows_padded, max(num_rows, num_SMs // col_blocks))
    var grid_dim = (grid_x, col_blocks, 1)

    ctx.enqueue_function[kernel](
        output_tile,
        scales_tile,
        input_tile,
        Int32(num_cols),
        Int32(num_cols_padded),
        tensor_sf,
        rope_freqs,
        block_dim=block_dim,
        grid_dim=grid_dim,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
@__name(
    t"quantize_dynamic_scaled_fp4fp8_{out_dtype}_{scales_dtype}_{in_dtype}_sf{SF_VECTOR_SIZE}{"_gated" if gated else ""}{"_rope" if rope_dim > 0 else ""}",
)
def quantize_dynamic_scaled_fp4fp8_kernel[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    *,
    SF_VECTOR_SIZE: Int = 16,
    ELEMENTS_PER_THREAD: Int = 8,
    num_max_threads: Int = 512,
    gated: Bool = False,
    activation: swiglu_activation_type = swiglu[0.0],
    rope_dim: Int = 0,
    rope_head_dim: Int = 0,
    rope_inverse: Bool = False,
    rope_layout: TensorLayout = RowMajorLayout[
        *Coord[Int64, Int64].element_types
    ],
](
    output: TileTensor[out_dtype, output_layout, MutAnyOrigin],
    scales: TileTensor[scales_dtype, scales_layout, MutAnyOrigin],
    input: TileTensor[in_dtype, input_layout, ImmutAnyOrigin],
    num_cols: Int32,
    num_cols_padded: Int32,
    tensor_sf: Float32,
    rope_freqs: OptionalReg[
        TileTensor[.float32, rope_layout, ImmUnsafeAnyOrigin]
    ] = None,
):
    comptime assert SF_VECTOR_SIZE in (16, 32) and ELEMENTS_PER_THREAD == 8, (
        "Currently only supports NVFP4 (SF_VECTOR_SIZE = 16) and MXFP8"
        " (SF_VECTOR_SIZE = 32) with 8 elements per thread"
    )

    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
    comptime assert NUM_THREADS_PER_SF in (
        2,
        4,
    ), "NUM_THREADS_PER_SF must be 2 or 4"
    comptime OUTPUT_WIDTH = 4 if out_dtype == .uint8 else 8

    comptime assert (
        input_layout.static_shape[1] % ELEMENTS_PER_THREAD == 0
    ), "num_cols must be a multiple of ELEMENTS_PER_THREAD (8 for NVFP4/MXFP8)"
    comptime if rope_dim > 0:
        comptime assert not gated, "the gated and RoPE modes are exclusive"
        # A thread's elements then lie all inside or all outside a head's
        # rotated columns.
        comptime assert (
            rope_dim % ELEMENTS_PER_THREAD == 0
            and rope_head_dim % ELEMENTS_PER_THREAD == 0
            and rope_dim <= rope_head_dim
            and input_layout.static_shape[1] % rope_head_dim == 0
        ), "RoPE columns must split into whole heads of whole thread chunks"

    # Unsigned 32-bit indices: with signed 64-bit Int, every loop bound,
    # division and scale-offset term costs a multi-instruction sequence.
    var num_rows = input.dim[0]().cast[DType.uint32]()
    var num_rows_padded = align_up(num_rows, UInt32(SF_MN_GROUP_SIZE))
    var num_sf_cols = align_up(
        num_cols_padded.cast[DType.uint32](), UInt32(SF_VECTOR_SIZE * SF_ATOM_K)
    )

    var num_col_threads = num_cols.cast[DType.uint32]() // UInt32(
        ELEMENTS_PER_THREAD
    )
    var num_padded_col_threads = num_cols_padded.cast[DType.uint32]() // UInt32(
        ELEMENTS_PER_THREAD
    )
    var num_sf_threads = num_sf_cols // UInt32(ELEMENTS_PER_THREAD)

    # Releasing at entry lets the block-scaled GEMM that consumes this output
    # stream its weights while the quantize runs. A quantize that follows
    # another one directly gets packed onto fewer SMs this way, but models
    # place a GEMM after each quantize.
    with PDL[overlap_at_beginning=True]():
        var row = UInt32(block_idx.x)
        while row < num_rows_padded:
            # A block keeps one row's loads in flight, too few to hide DRAM
            # latency at large M, so it pulls its next row into L2.
            var pf_row = row + UInt32(grid_dim.x)
            if thread_idx.x == 0 and block_idx.y == 0 and pf_row < num_rows:
                cp_async_bulk_prefetch(
                    input.ptr.unsafe_offset(
                        Int(pf_row) * Int(input_layout.static_shape[1])
                    ),
                    Int32(input_layout.static_shape[1] * size_of[in_dtype]()),
                )
            var is_padded_row = row >= num_rows
            var col_thread = UInt32(block_idx.y * block_dim.x + thread_idx.x)
            while col_thread < num_sf_threads:
                var col = col_thread * UInt32(ELEMENTS_PER_THREAD)
                var owns_scale = col % UInt32(SF_VECTOR_SIZE) == 0
                var sf_offset = scales.layout[linear_idx_type=.uint32](
                    Coord(
                        row // UInt32(SF_MN_GROUP_SIZE),
                        col // UInt32(SF_VECTOR_SIZE * SF_ATOM_K),
                        row % UInt32(SF_ATOM_M[0]),
                        row % UInt32(SF_MN_GROUP_SIZE) // UInt32(SF_ATOM_M[0]),
                        col // UInt32(SF_VECTOR_SIZE) % UInt32(SF_ATOM_K),
                    )
                )

                if is_padded_row or col_thread >= num_col_threads:
                    # Tensor cores read the scales of padding rows and
                    # columns, so they must be 0. Padding rows only exist in
                    # the scales tensor.
                    if (
                        not is_padded_row
                        and col_thread < num_padded_col_threads
                    ):
                        output.store(
                            (Int(row), Int(col_thread * UInt32(OUTPUT_WIDTH))),
                            SIMD[out_dtype, OUTPUT_WIDTH](0),
                        )
                    if owns_scale:
                        scales.raw_store(sf_offset, Scalar[scales_dtype](0.0))
                else:
                    var input_vector: SIMD[in_dtype, ELEMENTS_PER_THREAD]
                    comptime if gated:
                        comptime up_offset = input_layout.static_shape[1] // 2
                        input_vector = apply_swiglu_activation[activation](
                            input.load[width=ELEMENTS_PER_THREAD](
                                (Int(row), Int(col))
                            ),
                            input.load[width=ELEMENTS_PER_THREAD](
                                (Int(row), up_offset + Int(col))
                            ),
                        )
                    else:
                        input_vector = input.load[width=ELEMENTS_PER_THREAD](
                            (Int(row), Int(col))
                        )
                        comptime if rope_dim > 0:
                            comptime rope_start = UInt32(
                                rope_head_dim - rope_dim
                            )
                            var head_col = col % UInt32(rope_head_dim)
                            if head_col >= rope_start:
                                input_vector = _rope_rotate[
                                    inverse=rope_inverse
                                ](
                                    input_vector,
                                    rope_freqs.value().load[
                                        width=ELEMENTS_PER_THREAD
                                    ]((Int(row), Int(head_col - rope_start))),
                                )

                    # each thread finds maximum value in its local 8 elements
                    var thread_max = abs(input_vector).reduce_max()
                    # find the maximum value among all 16 elements (two threads for 16)
                    thread_max = max(shuffle_xor(thread_max, 1), thread_max)

                    comptime if NUM_THREADS_PER_SF == 4:
                        thread_max = max(shuffle_xor(thread_max, 2), thread_max)

                    var group_max = thread_max.cast[.float32]()

                    var fp8_scale_factor: Scalar[scales_dtype]
                    var output_scale = Float32(0.0)
                    var block_is_dead = False
                    comptime if out_dtype == .uint8:
                        # get the scale factor for these 16 elements by dividing it by the maximum value of fp4-e2m1
                        # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know that scale_factor is always positive, so we can use E4M3 instead of UE4M3.
                        fp8_scale_factor = (
                            tensor_sf * (group_max * recip(Float32(6.0)))
                        ).cast[scales_dtype]()

                        # find the quantization scale factor for these 16 elements (scale_factor = scale_factor / tensor_sf)
                        # we divide input by this scale factor which is same as multiplying by the reciprocal of the scale factor
                        if group_max != 0:
                            output_scale = recip(
                                fp8_scale_factor.cast[.float32]()
                                * recip(tensor_sf)
                            )
                    else:
                        fp8_scale_factor, output_scale, block_is_dead = (
                            compute_mxfp8_block_scale[scales_dtype](group_max)
                        )

                    if owns_scale:
                        scales.raw_store(sf_offset, fp8_scale_factor)

                    var input_f32 = input_vector.cast[.float32]() * output_scale

                    var output_vector: SIMD[out_dtype, OUTPUT_WIDTH]

                    comptime if out_dtype == .uint8:
                        output_vector = bitcast[out_dtype, OUTPUT_WIDTH](
                            cast_fp32_to_fp4e2m1(input_f32)
                        )
                    else:
                        output_vector = rebind[SIMD[out_dtype, OUTPUT_WIDTH]](
                            input_f32.cast[out_dtype]()
                        )

                    # A dead MXFP8 block's multiplier is 0; storing zeros
                    # avoids `inf * 0.0 = NaN`.
                    if block_is_dead:
                        output_vector = SIMD[out_dtype, OUTPUT_WIDTH](0)

                    output.store(
                        (Int(row), Int(col_thread * UInt32(OUTPUT_WIDTH))),
                        output_vector,
                    )

                col_thread += UInt32(grid_dim.y * block_dim.x)
            row += UInt32(grid_dim.x)


@inline(.always)
def block_scales_interleave_fp4[
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    ctx: DeviceContext,
    input_scales_tile: TileTensor[
        mut=False, scales_dtype, address_space=.GENERIC, ...
    ],
    output_scales_tile: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
) raises:
    """Launches the SM100 kernel that reinterleaves rank-2 scale factors into the 5D TCGEN layout.

    Converts the flat scale-factor tensor into the interleaved 5D layout
    expected by the tensor-core scale-factor feed.

    Parameters:
        scales_dtype: Element type of the block scale-factor tensors
            (inferred).
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor (defaults to 16).
        num_max_threads: Maximum number of threads per block for the
            launch grid (defaults to 1024).
    Args:
        ctx: Device context used to enqueue the kernel.
        input_scales_tile: Input rank-2 scale-factor tile in the flat
            row-major layout.
        output_scales_tile: Output rank-5 scale-factor tile in the 5D
            TCGEN interleaved layout.
    """
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100 or SM120"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
        MXFP4_SF_DTYPE,
    ), "scales dtype should be float8_e4m3fn (NVFP4) or float8_e8m0fnu (MXFP4)"

    comptime num_SMs = B200.sm_count

    var num_rows = Int(input_scales_tile.dim[0]())
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)
    var num_cols = Int(input_scales_tile.dim[1]())
    var num_col_padded = align_up(num_cols, SF_ATOM_K)

    # each thread handle just one scale factor for SF_VECTOR_SIZE of elements
    var block_dim = (min(num_col_padded, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, 2 * B200.gpu_info.threads_per_multiprocessor // block_dim[0]
    )
    var grid_dim = (min(num_rows_padded, num_SMs * num_blocks_per_SM), 1, 1)

    comptime kernel = block_scales_interleave_fp4_kernel[
        scales_dtype,
        input_scales_tile.LayoutType,
        output_scales_tile.LayoutType,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        num_max_threads=num_max_threads,
    ]

    ctx.enqueue_function[kernel](
        input_scales_tile,
        output_scales_tile,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
@__name(t"block_scales_interleave_fp4_{scales_dtype}_{SF_VECTOR_SIZE}")
def block_scales_interleave_fp4_kernel[
    scales_dtype: DType,
    input_scales_layout: TensorLayout,
    output_scales_layout: TensorLayout,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    input_scales: TileTensor[scales_dtype, input_scales_layout, ImmutAnyOrigin],
    output_scales: TileTensor[scales_dtype, output_scales_layout, MutAnyOrigin],
):
    """GPU kernel that reinterleaves rank-2 scale factors into the 5D TCGEN layout.

    Each thread reads one scale factor from the flat input and writes it
    into the interleaved output at the swizzled position required by the
    tensor-core scale-factor feed.

    Parameters:
        scales_dtype: Element type of the scale-factor tensors.
        input_scales_layout: Layout of the input scale-factor tensor.
        output_scales_layout: Layout of the output scale-factor tensor.
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor (defaults to 16).
        num_max_threads: Maximum number of threads per block for the
            launch grid (defaults to 1024).
    Args:
        input_scales: Input rank-2 scale-factor tensor in the flat
            row-major layout.
        output_scales: Output rank-5 scale-factor tensor in the 5D TCGEN
            interleaved layout.
    """
    var num_rows = Int(input_scales.dim[0]())
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)
    var num_cols = Int(input_scales.dim[1]())
    var num_col_padded = align_up(num_cols, SF_ATOM_K)

    for row_idx in range(block_idx.x, num_rows_padded, grid_dim.x):
        for col_idx in range(thread_idx.x, num_col_padded, block_dim.x):
            var scale_factor = Scalar[scales_dtype](0.0)
            if row_idx < num_rows and col_idx < num_cols:
                scale_factor = input_scales[Int(row_idx), Int(col_idx)]

            set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                output_scales,
                row_idx,
                col_idx * SF_VECTOR_SIZE,
                scale_factor,
            )


def naive_block_scaled_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    scaling_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    accum_type: DType = .float32,
    transpose_b: Bool = True,
    BLOCK_DIM: Int = 16,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[a_type, address_space=.GENERIC, ...],
    b: TileTensor[b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[b_scales_type, address_space=.GENERIC, ...],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """Reference block-scaled matmul that emulates TCGEN scale-factor accumulation on SM100 hardware.

    Validates input and scale dimensions, then enqueues the
    `naive_block_scaled_matmul_kernel` with a 16x16 thread block grid.

    Parameters:
        c_type: Element type of the output matrix (inferred).
        a_type: Element type of the LHS input matrix (inferred).
        b_type: Element type of the RHS input matrix (inferred).
        a_scales_type: Element type of the `a_scales` block scale-factor
            tensor (inferred).
        b_scales_type: Element type of the `b_scales` block scale-factor
            tensor (inferred).
        scaling_kind: `UMMAKind` variant selecting the block-scaled MMA
            instruction.
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor.
        accum_type: Accumulator element type (defaults to
            `DType.float32`).
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
        BLOCK_DIM: Thread block tile dimension in rows and columns
            (defaults to 16).
    Args:
        c: Output matrix of shape `[M, N]`.
        a: LHS input matrix of shape `[M, K]` for FP8 or `[M, K//2]`
            for packed FP4 `uint8`.
        b: RHS input matrix of shape `[N, K]` for FP8 or `[N, K//2]`
            for packed FP4 `uint8`, stored transposed.
        a_scales: Block scale factors for `a` in the 5D TCGEN
            interleaved layout.
        b_scales: Block scale factors for `b` in the 5D TCGEN
            interleaved layout.
        ctx: Device context used to enqueue the kernel.
        alpha: Scalar multiplier applied to the matmul result
            (defaults to 1.0).
    """
    comptime assert transpose_b, "Only transpose_b = True is supported for now"
    comptime assert accum_type in (
        DType.float32,
    ), "Only float32 is supported for accumulation for scaled matmul"
    comptime assert (
        a_type == b_type
    ), "Only same input dtype is supported for block scaled matmul"
    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be same for block scaled matmul"
    comptime assert (
        (
            scaling_kind == UMMAKind.KIND_MXF4NVF4
            and a_type == .uint8
            and a_scales_type == NVFP4_SF_DTYPE
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF4
            and a_type == .uint8
            and a_scales_type == MXFP4_SF_DTYPE
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF8F6F4
            and a_type == .float8_e4m3fn
            and a_scales_type == MXFP8_SF_DTYPE
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        )
    ), (
        "Only support NVFP4 (KIND_MXF4NVF4), MXFP4 (KIND_MXF4),"
        " or MXFP8 (KIND_MXF8F6F4) for block scaled matmul"
    )
    comptime assert c_type in (DType.bfloat16, DType.float32), (
        "Only bfloat16 or float32 is supported for output dtype for block"
        " scaled matmul matmul"
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )
    var K = Int(a.dim[1]()) * 2 if is_fp4 else Int(a.dim[1]())

    if M == 0 or N == 0 or K == 0:
        return

    if (
        Int(a_scales.dim[0]()) != ceildiv(M, SF_MN_GROUP_SIZE)
        or Int(b_scales.dim[0]()) != ceildiv(N, SF_MN_GROUP_SIZE)
        or Int(a_scales.dim[1]()) != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        or Int(b_scales.dim[1]()) != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        or (Int(a_scales.dim[2]()) != Int(b_scales.dim[2]()) != SF_ATOM_M[0])
        or (Int(a_scales.dim[3]()) != Int(b_scales.dim[3]()) != SF_ATOM_M[1])
        or (Int(a_scales.dim[4]()) != Int(b_scales.dim[4]()) != SF_ATOM_K)
    ):
        raise Error("Invalid A/B scales dimensions.")

    logger.info("Executing Naive Block Scaled NVFP4 GEMM")
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]", sep="")
    logger.info(
        "A Scales Shape: [",
        a_scales.dim(0),
        ", ",
        a_scales.dim(1),
        ", ",
        a_scales.dim(2),
        ", ",
        a_scales.dim(3),
        ", ",
        a_scales.dim(4),
        "]",
        sep="",
    )
    logger.info(
        "B Scales Shape: [",
        b_scales.dim(0),
        ", ",
        b_scales.dim(1),
        ", ",
        b_scales.dim(2),
        ", ",
        b_scales.dim(3),
        ", ",
        b_scales.dim(4),
        "]",
        sep="",
    )

    comptime kernel = naive_block_scaled_matmul_kernel[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        a.LayoutType,
        b.LayoutType,
        c.LayoutType,
        a_scales.LayoutType,
        b_scales.LayoutType,
        scaling_kind=scaling_kind,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ]

    ctx.enqueue_function[kernel](
        c,
        a.as_imm(),
        b.as_imm(),
        a_scales.as_imm(),
        b_scales.as_imm(),
        alpha,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


@__name(t"naive_block_scaled_matmul")
def naive_block_scaled_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    c_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
    scaling_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: TileTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: TileTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
    alpha: Float32,
):
    """Naive GPU kernel that emulates a block-scaled matmul using TCGEN scale factors.

    Both A and B must be in K-major format with 5D TCGEN scale-factor
    layouts. Each thread accumulates one output element by iterating over
    K and applying per-block scale factors.

    Parameters:
        c_type: Element type of the output matrix `c`.
        a_type: Element type of the LHS input matrix `a`.
        b_type: Element type of the RHS input matrix `b`.
        a_scales_type: Element type of the `a_scales` block scale-factor
            tensor.
        b_scales_type: Element type of the `b_scales` block scale-factor
            tensor.
        accum_type: Element type used for the dot-product accumulator.
        a_layout: Memory layout of the LHS input matrix `a`.
        b_layout: Memory layout of the RHS input matrix `b`.
        c_layout: Memory layout of the output matrix `c`.
        a_scale_layout: Memory layout of the `a_scales` block
            scale-factor tensor.
        b_scale_layout: Memory layout of the `b_scales` block
            scale-factor tensor.
        scaling_kind: `UMMAKind` variant selecting the block-scaled MMA
            instruction.
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor.
        transpose_b: Whether `b` is stored transposed (defaults to
            `True`).
    Args:
        c: Output matrix of shape `[M, N]`.
        a: LHS input matrix of shape `[M, K]` for FP8 or `[M, K//2]`
            for packed FP4 `uint8`.
        b: RHS input matrix of shape `[N, K]` for FP8 or `[N, K//2]`
            for packed FP4 `uint8`, stored transposed.
        a_scales: Block scale factors for `a` in the 5D TCGEN
            interleaved layout.
        b_scales: Block scale factors for `b` in the 5D TCGEN
            interleaved layout.
        alpha: Scalar multiplier applied to the matmul result.
    """
    # Note: This is a naive kernel that emulates a block scaled matmul with TCGEN scale factors.
    # Assumptions:
    # 1. both A and B should be in K-major format
    # 2. both a_scales and b_scales should be in TCGEN scale factors layout (5D tensors)

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )
    comptime K_STEPS = 2 if is_fp4 else 1
    var K = Int(a.dim[1]()) * K_STEPS

    var row_idx = Int(global_idx.x)
    var col_idx = Int(global_idx.y)

    if row_idx >= M or col_idx >= N:
        return

    var accum = Scalar[accum_type](0.0)
    for k in range(0, K, K_STEPS):
        var a_scale = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            a_scales, row_idx, k
        )
        var b_scale = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            b_scales, col_idx, k
        )

        comptime if is_fp4:
            # Each uint8 element has two Float4-E2M1 values.
            var a_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](a[row_idx, k // K_STEPS])
            ).cast[accum_type]()
            var b_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](b[col_idx, k // K_STEPS])
            ).cast[accum_type]()

            comptime for k_idx in range(K_STEPS):
                var a_val = a_val_fp16x2[k_idx]
                var b_val = b_val_fp16x2[k_idx]
                var a_scale_val = abs(a_scale.cast[accum_type]())
                var b_scale_val = abs(b_scale.cast[accum_type]())
                accum += a_val * b_val * a_scale_val * b_scale_val
        else:
            # MXFP8: one float8 value per byte.
            var a_val = a[row_idx, k // K_STEPS].cast[accum_type]()
            var b_val = b[col_idx, k // K_STEPS].cast[accum_type]()
            var a_scale_val = abs(
                rebind[Scalar[accum_type]](a_scale.cast[accum_type]())
            )
            var b_scale_val = abs(
                rebind[Scalar[accum_type]](b_scale.cast[accum_type]())
            )
            accum += a_val * b_val * a_scale_val * b_scale_val

    accum *= alpha.cast[accum_type]()

    c[row_idx, col_idx] = accum.cast[c_type]()


@__llvm_arg_metadata(input_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(output_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(scales_tma_op, `nvvm.grid_constant`)
@__name(
    t"quantize_dynamic_scaled_async_kernel_{output_dtype}_sf{SF_VECTOR_SIZE}"
)
def quantize_dynamic_scaled_async_kernel[
    input_dtype: DType,
    input_tile_shape: Coord,
    input_desc_shape: Coord,
    output_dtype: DType,
    output_tile_shape: Coord,
    output_desc_shape: Coord,
    scales_dtype: DType,
    scales_tile_shape: Coord,
    scales_desc_shape: Coord,
    input_swizzle_mode: TensorMapSwizzle,
    output_swizzle_mode: TensorMapSwizzle,
    scales_swizzle_mode: TensorMapSwizzle,
    SF_VECTOR_SIZE: Int,
    NUM_PIPELINES_STAGES: Int,
](
    input_tma_op: TMATensorTile[
        input_dtype, input_tile_shape, input_desc_shape
    ],
    output_tma_op: TMATensorTile[
        output_dtype, output_tile_shape, output_desc_shape
    ],
    scales_tma_op: TMATensorTile[
        scales_dtype, scales_tile_shape, scales_desc_shape
    ],
    tensor_sf: Float32,  # tensor-wise scale factor
):
    """GPU kernel that quantizes BF16 tiles to NVFP4 or MXFP8 using TMA async copies and warp-specialized PDL.

    One warp issues TMA loads while a warpgroup computes per-group scale
    factors and quantizes, one row per thread, then stores results back via
    TMA async stores. A CTA covers 128 rows and one scale atom's
    `SF_ATOM_K` groups, loaded as `NUM_PIPELINES_STAGES` 64-column stages.

    Parameters:
        input_dtype: Element type of the input activation tensor.
        input_tile_shape: Per-dimension shape of one stage's input TMA tile.
        input_desc_shape: Per-dimension descriptor shape of the input TMA
            tile.
        output_dtype: Element type of the quantized output tensor: `uint8`
            for NVFP4 or `float8_e4m3fn` for MXFP8.
        output_tile_shape: Per-dimension shape of the output TMA tile.
        output_desc_shape: Per-dimension descriptor shape of the output
            TMA tile.
        scales_dtype: Element type of the block scale-factor tensor.
        scales_tile_shape: Per-dimension shape of the scales TMA tile.
        scales_desc_shape: Per-dimension descriptor shape of the scales
            TMA tile.
        input_swizzle_mode: `TensorMapSwizzle` mode for input TMA loads.
        output_swizzle_mode: `TensorMapSwizzle` mode for output TMA stores.
        scales_swizzle_mode: `TensorMapSwizzle` mode for scales TMA stores.
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor: 16 for NVFP4 or 32 for MXFP8.
        NUM_PIPELINES_STAGES: Number of 64-column input stages a CTA
            loads: 1 for NVFP4 or 2 for MXFP8.
    Args:
        input_tma_op: TMA tile descriptor for async input loads.
        output_tma_op: TMA tile descriptor for async output stores.
        scales_tma_op: TMA tile descriptor for async scale-factor stores.
        tensor_sf: Tensor-wise scale factor applied to the NVFP4
            quantization; unused for MXFP8.
    """
    comptime is_mxfp8 = output_dtype == .float8_e4m3fn

    var smem_storage = rebind[
        MutPointer[Scalar[input_dtype], MutAnyOrigin, address_space=.SHARED]
    ](
        external_memory[
            Scalar[input_dtype],
            address_space=.SHARED,
            alignment=128,
        ]()
    )

    comptime input_stage_tile_size = _idx_product[input_tile_shape]()
    comptime input_smem_tile_size = (
        input_stage_tile_size * NUM_PIPELINES_STAGES
    )
    comptime output_smem_tile_size = _idx_product[output_tile_shape]()
    comptime scales_smem_tile_size = _idx_product[scales_tile_shape]()

    comptime SF_K_GROUP_SIZE: Int = SF_VECTOR_SIZE * Int(SF_ATOM_K)
    comptime STAGE_GROUP_SIZE = SF_K_GROUP_SIZE // NUM_PIPELINES_STAGES
    comptime GROUPS_PER_STAGE = STAGE_GROUP_SIZE // SF_VECTOR_SIZE
    # Output elements one 16-byte smem store holds.
    comptime OUT_VEC = 16 // size_of[output_dtype]()
    # Output elements one stage yields per row: packed FP4 halves them.
    comptime STAGE_OUT_ELEMS = STAGE_GROUP_SIZE // (1 if is_mxfp8 else 2)
    comptime OUT_ROW_ELEMS = STAGE_OUT_ELEMS * NUM_PIPELINES_STAGES

    comptime assert (
        STAGE_GROUP_SIZE == 64
    ), "a stage must be 64 input columns, the 128-byte swizzle span of bfloat16"
    comptime assert (
        is_mxfp8
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        and scales_dtype == MXFP8_SF_DTYPE
    ) or (
        output_dtype == .uint8
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        and scales_dtype == NVFP4_SF_DTYPE
    ), (
        "expected NVFP4 (uint8 output, 16-wide float8_e4m3fn scales) or MXFP8"
        " (float8_e4m3fn output, 32-wide float8_e8m0fnu scales)"
    )

    var input_smem_ptr = smem_storage
    var output_smem_ptr = smem_storage.unsafe_offset(
        input_smem_tile_size
    ).unsafe_bitcast[Scalar[output_dtype]]()
    var scales_smem_ptr = (
        output_smem_ptr.unsafe_offset(output_smem_tile_size)
    ).unsafe_bitcast[Scalar[scales_dtype]]()
    var mbar_ptr = scales_smem_ptr.unsafe_offset(scales_smem_tile_size)

    var output_smem = TileTensor(output_smem_ptr, row_major(output_tile_shape))
    var scales_smem = TileTensor(scales_smem_ptr, row_major(scales_tile_shape))
    var output_smem_flat = output_smem.reshape(
        row_major[output_smem_tile_size]()
    )
    var scales_smem_flat = scales_smem.reshape(
        row_major[scales_smem_tile_size]()
    )

    var tma_mbar = mbar_ptr.unsafe_bitcast[SharedMemBarrier]()

    var local_row_idx = thread_idx.x

    if thread_idx.x == 0:
        comptime for idx in range(NUM_PIPELINES_STAGES):
            tma_mbar[idx].init()

    var tma_phase = SIMD[.uint32, NUM_PIPELINES_STAGES](0)

    barrier()

    comptime expected_bytes = (
        _idx_product[input_tile_shape]() * size_of[input_dtype]()
    )

    with PDL():
        if thread_idx.x >= 128:
            warpgroup_reg_dealloc[24]()

            comptime for iter_idx in range(NUM_PIPELINES_STAGES):
                var smem_tile = TileTensor(
                    input_smem_ptr.unsafe_offset(
                        iter_idx * input_stage_tile_size
                    ),
                    row_major(input_tile_shape),
                )

                if lane_id() == 0:
                    tma_mbar[iter_idx].expect_bytes(Int32(expected_bytes))
                    input_tma_op.async_copy(
                        smem_tile,
                        tma_mbar[iter_idx],
                        (
                            (block_idx.y * SF_K_GROUP_SIZE)
                            + (iter_idx * STAGE_GROUP_SIZE),
                            block_idx.x * SF_MN_GROUP_SIZE,
                        ),
                    )

        else:
            var scale_factors = SIMD[scales_dtype, SF_ATOM_K]()

            comptime for iter_idx in range(NUM_PIPELINES_STAGES):
                var smem_tile = TileTensor(
                    input_smem_ptr.unsafe_offset(
                        iter_idx * input_stage_tile_size
                    ),
                    row_major(input_tile_shape),
                )

                var smem_tile_flat = smem_tile.reshape(
                    row_major[input_stage_tile_size]()
                )

                tma_mbar[iter_idx].wait(tma_phase[iter_idx])
                var quantized_elements = SIMD[output_dtype, STAGE_OUT_ELEMS]()

                comptime for group_idx in range(GROUPS_PER_STAGE):
                    var group_elements = SIMD[input_dtype, SF_VECTOR_SIZE]()

                    comptime for col_idx in range(SF_VECTOR_SIZE // 8):
                        var swizzle_offset = (
                            local_row_idx * STAGE_GROUP_SIZE
                            + (group_idx * SF_VECTOR_SIZE)
                            + col_idx * 8
                        )

                        comptime input_swizzle = make_swizzle[
                            input_dtype, input_swizzle_mode
                        ]()
                        var swizzle_idx = input_swizzle(swizzle_offset)
                        var temp = smem_tile_flat.load[
                            width=8,
                            alignment=align_of[SIMD[input_dtype, 8]](),
                        ]((swizzle_idx,))

                        group_elements = group_elements.insert[
                            offset=col_idx * 8
                        ](temp)

                    var group_max = (
                        abs(group_elements).reduce_max().cast[.float32]()
                    )

                    var fp8_scale_factor: Scalar[scales_dtype]
                    var output_scale = Float32(0.0)
                    comptime if is_mxfp8:
                        var block_is_dead: Bool
                        fp8_scale_factor, output_scale, block_is_dead = (
                            compute_mxfp8_block_scale[scales_dtype](group_max)
                        )
                        # A dead block's multiplier is 0; zeroing avoids
                        # storing `inf * 0.0 = NaN`.
                        if block_is_dead:
                            group_elements = type_of(group_elements)(0)
                    else:
                        fp8_scale_factor = (
                            tensor_sf * group_max * recip(Float32(6.0))
                        ).cast[scales_dtype]()
                        if fp8_scale_factor.cast[.float32]() != 0:
                            output_scale = recip(
                                fp8_scale_factor.cast[.float32]()
                                * recip(tensor_sf)
                            )

                    scale_factors[
                        iter_idx * GROUPS_PER_STAGE + group_idx
                    ] = fp8_scale_factor

                    var scaled = group_elements.cast[.float32]() * output_scale
                    comptime if is_mxfp8:
                        quantized_elements = quantized_elements.insert[
                            offset=group_idx * SF_VECTOR_SIZE
                        ](scaled.cast[output_dtype]())
                    else:
                        comptime for slice_idx in range(SF_VECTOR_SIZE // 8):
                            quantized_elements = quantized_elements.insert[
                                offset=(
                                    group_idx * SF_VECTOR_SIZE + slice_idx * 8
                                )
                                // 2
                            ](
                                bitcast[output_dtype, 4](
                                    cast_fp32_to_fp4e2m1(
                                        scaled.slice[8, offset=slice_idx * 8]()
                                    )
                                )
                            )

                comptime for idx in range(STAGE_OUT_ELEMS // OUT_VEC):
                    comptime output_swizzle = make_swizzle[
                        output_dtype, output_swizzle_mode
                    ]()
                    var swizzle_offset = (
                        local_row_idx * OUT_ROW_ELEMS
                        + iter_idx * STAGE_OUT_ELEMS
                        + idx * OUT_VEC
                    )
                    var output_swizzle_idx = output_swizzle(swizzle_offset)
                    output_smem_flat.store[
                        alignment=align_of[SIMD[output_dtype, OUT_VEC]]()
                    ](
                        (output_swizzle_idx,),
                        quantized_elements.slice[
                            OUT_VEC, offset=idx * OUT_VEC
                        ](),
                    )

            scales_smem_flat.store[
                alignment=align_of[SIMD[scales_dtype, SF_ATOM_K]]()
            ](
                (
                    (local_row_idx % 32) * 16
                    + (local_row_idx // 32) * SF_ATOM_K,
                ),
                scale_factors,
            )

            named_barrier[128](1)

            if thread_idx.x == 0:
                fence_async_view_proxy()

                scales_tma_op.async_store(
                    scales_smem,
                    StaticTuple[UInt32, 4](
                        0,
                        0,
                        UInt32(block_idx.y),
                        UInt32(block_idx.x),
                    ),
                )

                output_tma_op.async_store(
                    output_smem,
                    StaticTuple[UInt32, 2](
                        UInt32(block_idx.y * OUT_ROW_ELEMS),
                        UInt32(block_idx.x) * UInt32(SF_MN_GROUP_SIZE),
                    ),
                )
                output_tma_op.commit_group()

            output_tma_op.wait_group[0]()


def quantize_dynamic_scaled_fp4_async[
    input_dtype: DType,
    output_dtype: DType,
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext,
    output_tensor_tile: TileTensor[
        mut=True, output_dtype, address_space=.GENERIC, ...
    ],
    scales_tensor_tile: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    input_tensor_tile: TileTensor[
        mut=False, input_dtype, address_space=.GENERIC, ...
    ],
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
) raises:
    """Launches the TMA-based NVFP4 or MXFP8 quantization kernel on SM100 hardware.

    Sets up TMA tile descriptors for input, output, and scales, then
    enqueues the `quantize_dynamic_scaled_async_kernel` with
    warp-specialized PDL.

    Parameters:
        input_dtype: Element type of the input activation tensor. Must
            be `bfloat16` (inferred).
        output_dtype: Element type of the quantized output tensor:
            `uint8` for NVFP4 or `float8_e4m3fn` for MXFP8 (inferred).
        scales_dtype: Element type of the block scale-factor tensor:
            `float8_e4m3fn` for NVFP4 or `float8_e8m0fnu` for MXFP8
            (inferred).
        SF_VECTOR_SIZE: Number of input elements covered by each block
            scale factor: 16 for NVFP4 or 32 for MXFP8.
    Args:
        ctx: Device context used to enqueue the kernel.
        output_tensor_tile: Output tile for the quantized elements: packed
            NVFP4 `uint8` with half the input columns, or MXFP8 with as
            many columns as the input. Its column count must be a
            multiple of 16 bytes.
        scales_tensor_tile: Output 5D scale-factor tile in the TCGEN
            interleaved layout.
        input_tensor_tile: Input `bfloat16` activation tile.
        tensor_sf: Tensor-wise scale factor applied to the NVFP4
            quantization (defaults to 1.0); unused for MXFP8.
    """
    comptime assert input_dtype == .bfloat16, "input_dtype must be bfloat16"

    comptime is_mxfp8 = output_dtype == .float8_e4m3fn
    comptime assert (
        output_dtype == .uint8
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        and scales_dtype == NVFP4_SF_DTYPE
    ) or (
        is_mxfp8
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        and scales_dtype == MXFP8_SF_DTYPE
    ), (
        "expected NVFP4 (uint8 output, float8_e4m3fn scales) or MXFP8"
        " (float8_e4m3fn output, float8_e8m0fnu scales)"
    )

    comptime input_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    # One row of the output tile: 128 MXFP8 bytes, or 64 packed FP4 values
    # in 32 bytes.
    comptime output_swizzle_mode = (
        TensorMapSwizzle.SWIZZLE_128B if is_mxfp8 else TensorMapSwizzle.SWIZZLE_32B
    )
    comptime scales_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE  # 16 elements / 1 elements per float8_e4m3fn = 16 elements per 16B

    var M = Int(input_tensor_tile.dim[0]())
    var N = Int(input_tensor_tile.dim[1]())

    comptime output_N = output_tensor_tile.LayoutType.static_shape[1]
    comptime input_N = input_tensor_tile.LayoutType.static_shape[1]
    comptime out_per_in = 1 if is_mxfp8 else 2
    comptime assert (
        output_N * size_of[output_dtype]() % 16 == 0
    ), "output_tensor rows must be a multiple of 16 bytes"
    comptime assert (
        input_N == output_N * out_per_in
    ), "input_tensor N must be output_tensor N times the elements a byte packs"

    comptime SF_K_GROUP_SIZE = SF_VECTOR_SIZE * SF_ATOM_K
    # A stage is 64 bfloat16 columns, the 128-byte swizzle span.
    comptime STAGE_GROUP_SIZE = 64
    comptime NUM_PIPELINES_STAGES = SF_K_GROUP_SIZE // STAGE_GROUP_SIZE

    comptime input_tma_tile_shape = coord[SF_MN_GROUP_SIZE, STAGE_GROUP_SIZE]
    var input_tma_op = create_tensor_tile[
        input_tma_tile_shape,
        swizzle_mode=input_swizzle_mode,
    ](ctx, input_tensor_tile)

    comptime output_tma_tile_shape = coord[
        SF_MN_GROUP_SIZE, SF_K_GROUP_SIZE // out_per_in
    ]
    var output_tma_op = create_tensor_tile[
        output_tma_tile_shape,
        swizzle_mode=output_swizzle_mode,
    ](ctx, output_tensor_tile)

    comptime assert scales_tensor_tile.rank == 5, "scales must be 5D tensors"

    comptime scales_layout = scales_tensor_tile.LayoutType
    comptime assert scales_layout.static_shape[2] == SF_ATOM_M[0], ""
    comptime assert scales_layout.static_shape[3] == SF_ATOM_M[1], ""
    comptime assert scales_layout.static_shape[4] == SF_ATOM_K, ""

    # The TMA box covers one (M1, K) atom, so the two innermost scale dims
    # are folded into a single contiguous extent for a rank-4 descriptor.
    var scales_4d_tensor = scales_tensor_tile.reshape(
        row_major(
            Int(scales_tensor_tile.dim[0]()),
            Int(scales_tensor_tile.dim[1]()),
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1] * SF_ATOM_K],
        )
    )

    comptime scales_tma_tile_shape = coord[
        1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    ]
    var scales_tma_op = create_tensor_tile[
        scales_tma_tile_shape,
        swizzle_mode=scales_swizzle_mode,
    ](ctx, scales_4d_tensor)

    comptime smem_use = (
        input_tma_tile_shape.element_types[0].static_value
        * input_tma_tile_shape.element_types[1].static_value
        * size_of[input_dtype]()
        * NUM_PIPELINES_STAGES
    ) + (
        output_tma_tile_shape.element_types[0].static_value
        * output_tma_tile_shape.element_types[1].static_value
        * size_of[output_dtype]()
    ) + (
        size_of[SharedMemBarrier]() * NUM_PIPELINES_STAGES
    ) + (
        SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K * size_of[scales_dtype]()
    )

    comptime kernel = quantize_dynamic_scaled_async_kernel[
        type_of(input_tma_op).dtype,
        type_of(input_tma_op).tile_shape,
        type_of(input_tma_op).desc_shape,
        type_of(output_tma_op).dtype,
        type_of(output_tma_op).tile_shape,
        type_of(output_tma_op).desc_shape,
        type_of(scales_tma_op).dtype,
        type_of(scales_tma_op).tile_shape,
        type_of(scales_tma_op).desc_shape,
        input_swizzle_mode,
        output_swizzle_mode,
        scales_swizzle_mode,
        SF_VECTOR_SIZE,
        NUM_PIPELINES_STAGES=NUM_PIPELINES_STAGES,
    ]

    ctx.enqueue_function[kernel, dump_asm=False](
        input_tma_op,
        output_tma_op,
        scales_tma_op,
        tensor_sf,
        grid_dim=(
            ceildiv(M, SF_MN_GROUP_SIZE),
            ceildiv(N, SF_K_GROUP_SIZE),
            1,
        ),
        block_dim=(SF_MN_GROUP_SIZE + 32),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@inline(.always)
def _scale_tile_expert[
    num_experts: Int
](
    row_offsets: ImmTileTensor[.uint32, ...],
    scales_offsets: ImmTileTensor[.uint32, ...],
    scale_tile_idx: Int,
) -> Int:
    """Returns the last expert whose scale tiles start at or before
    `scale_tile_idx`.

    The tiles of expert `e` start at
    `row_offsets[e] // SF_MN_GROUP_SIZE + scales_offsets[e]`, and these
    starts never decrease. A padding tile, between two experts' tiles or
    past the last expert's, has no owner: this returns the expert before
    it. Callers must check that the tile falls inside the returned expert's
    `ceildiv(row_offsets[e + 1] - row_offsets[e], SF_MN_GROUP_SIZE)` tiles
    before using it.

    One warp does the search in two rounds and puts the result in shared
    memory. Call this from all threads of the block, because it contains a
    barrier.

    Parameters:
        num_experts: Number of experts in `scales_offsets`.

    Args:
        row_offsets: Row boundaries per expert, `[num_experts + 1]`.
        scales_offsets: Per-expert offset into the scales tiles.
        scale_tile_idx: The scale tile to find the expert of.

    Returns:
        The index of that expert.

    Constraints:
        `row_offsets` and `scales_offsets` must be rank 1, and
        `ceildiv(num_experts, WARP_SIZE)` must not exceed `WARP_SIZE`.
    """
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be rank 1"
    comptime assert (
        scales_offsets.flat_rank == 1
    ), "scales_offsets must be rank 1"
    comptime probe_stride = uceildiv(num_experts, WARP_SIZE)
    # The second round gives one lane to each expert in a stride. A wider
    # stride skips experts and returns the wrong one.
    comptime assert probe_stride <= WARP_SIZE, (
        "expert count exceeds what a two-round warp search covers; add a"
        " round or widen the stride handling"
    )
    # Keep this slot in its own 16 bytes. No barrier separates its read from
    # the next shared-memory write of the caller.
    var search_smem = unsafe_stack_allocation[
        1, Int32, alignment=16, address_space=.SHARED
    ]()
    if thread_idx.x < WARP_SIZE:
        var lane = thread_idx.x
        var target = scale_tile_idx

        var coarse = lane * probe_stride
        # Expert 0 always qualifies. This keeps the span at zero or more when
        # the offsets do not start at zero.
        var best = 0 if lane == 0 else -1
        if coarse < num_experts:
            if (
                ufloordiv(
                    Int(row_offsets.load[width=1]((coarse,))),
                    SF_MN_GROUP_SIZE,
                )
                + Int(scales_offsets.load[width=1]((coarse,)))
            ) <= target:
                best = coarse
        var span = Int(lane_group_max[WARP_SIZE](Int32(best)))

        comptime if probe_stride > 1:
            var fine = span + lane
            var refined = span
            if lane < probe_stride and fine < num_experts:
                if (
                    ufloordiv(
                        Int(row_offsets.load[width=1]((fine,))),
                        SF_MN_GROUP_SIZE,
                    )
                    + Int(scales_offsets.load[width=1]((fine,)))
                ) <= target:
                    refined = fine
            span = Int(lane_group_max[WARP_SIZE](Int32(refined)))

        if thread_idx.x == 0:
            search_smem.unsafe_store(0, Int32(span))
    barrier()
    return Int(search_smem.unsafe_load(0))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(scales_tma_op, `nvvm.grid_constant`)
@__name("grouped_quantize_dynamic_scaled_async_fp4_kernel")
def grouped_quantize_dynamic_scaled_fp4_async_kernel[
    output_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    scales_tile_shape: Coord,
    scales_desc_shape: Coord,
    scales_swizzle_mode: TensorMapSwizzle,
    output_layout: TensorLayout,
    input_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    sf_layout: TensorLayout,
    OutputEngine: TensorEngine,
    InputEngine: TensorEngine,
    RowOffsetsEngine: TensorEngine,
    ScalesOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SfEngine: TensorEngine,
    row_indices_layout: TensorLayout = RowMajorLayout[
        *Coord[Int64].element_types
    ],
    RowIndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    row_indices_dtype: DType = .uint32,
    row_index_divisor: Int = 1,
    num_threads: Int = 128,
    k_tiles_per_block: Int = 1,
](
    output_tensor: TileTensor[
        output_dtype, output_layout, MutAnyOrigin, Engine=OutputEngine
    ],
    scales_tma_op: TMATensorTile[
        scales_dtype, scales_tile_shape, scales_desc_shape
    ],
    input_tensor: TileTensor[
        input_dtype, input_layout, ImmutAnyOrigin, Engine=InputEngine
    ],
    row_offsets: TileTensor[
        .uint32, row_offsets_layout, ImmutAnyOrigin, Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        .uint32,
        scales_offsets_layout,
        ImmutAnyOrigin,
        Engine=ScalesOffsetsEngine,
    ],
    expert_ids: TileTensor[
        .int32, expert_ids_layout, ImmutAnyOrigin, Engine=ExpertIdsEngine
    ],
    sf_tensor: TileTensor[
        .float32, sf_layout, ImmutAnyOrigin, Engine=SfEngine
    ],  # tensor-wise scale factor
    row_indices: OptionalReg[
        TileTensor[
            row_indices_dtype,
            row_indices_layout,
            ImmutAnyOrigin,
            Engine=RowIndicesEngine,
        ]
    ] = None,
):
    """GPU kernel that quantizes per-expert BF16 activation tiles to NVFP4/MXFP4/MXFP8 with TMA-based scale-factor stores.

    Each block locates its assigned expert with a warp-parallel search over
    `row_offsets` and `scales_offsets`, then quantizes the expert's
    activation tile and writes scale factors back through TMA async
    stores.

    Parameters:
        output_dtype: Element type of the quantized output tensor
            (inferred).
        scales_dtype: Element type of the block scale-factor tensor
            (inferred).
        input_dtype: Element type of the input activation tensor
            (inferred).
        scales_tile_shape: Per-dimension tile shape of the scales
            TMA descriptor (inferred).
        scales_desc_shape: Per-dimension descriptor shape of the
            scales TMA descriptor (inferred).
        scales_swizzle_mode: Swizzle mode applied to the scales TMA
            descriptor (inferred).
        output_layout: TileTensor layout of the quantized output
            tensor (inferred).
        input_layout: TileTensor layout of the input activation
            tensor (inferred).
        row_offsets_layout: TileTensor layout of the per-expert row
            offsets tensor (inferred).
        scales_offsets_layout: TileTensor layout of the per-expert
            scales offsets tensor (inferred).
        expert_ids_layout: TileTensor layout of the expert IDs
            tensor (inferred).
        sf_layout: TileTensor layout of the per-expert tensor-wise
            scale factor tensor (inferred).
        OutputEngine: Engine policy of the `output_tensor`.
        InputEngine: Engine policy of the `input_tensor`.
        RowOffsetsEngine: Engine policy of the `row_offsets` tensor.
        ScalesOffsetsEngine: Engine policy of the `scales_offsets` tensor.
        ExpertIdsEngine: Engine policy of the `expert_ids` tensor.
        SfEngine: Engine policy of the `sf_tensor`.
        row_indices_layout: TileTensor layout of the optional source row
            indices tensor.
        RowIndicesEngine: Engine policy of the `row_indices` tensor.
        row_indices_dtype: Integer element type of the `row_indices` tensor.
        row_index_divisor: Divisor applied to each `row_indices` entry to get
            the source row.
        num_threads: Number of threads per block in the launch grid
            (defaults to 128).
        k_tiles_per_block: Column tiles handled by one block. Batching them
            amortizes the per-block expert search and shrinks the grid, which
            dominates at decode where a block's payload is one routed row.
    Args:
        output_tensor: Output quantized tensor of packed FP4 `uint8`
            or `float8_e4m3fn` for MXFP8.
        scales_tma_op: TMA tile descriptor used for async stores of
            the block scale factors.
        input_tensor: Input BF16 activation tensor.
        row_offsets: Contiguous row boundaries per expert of shape
            `[num_experts + 1]` as `uint32`, where range `i` spans
            rows `row_offsets[i]` through `row_offsets[i + 1]`.
        row_indices: Optional source row indices of shape `[padded_rows]`.
            When present, permuted output row `i` reads input row
            `row_indices[i] // row_index_divisor`, fusing the MoE expert
            permutation gather into the quantize loads. The output layout and
            scale tiles are unchanged; absent means the input is already in
            permuted row order.
        scales_offsets: Per-expert offset into the scales tensor of
            shape `[num_experts]` as `uint32`.
        expert_ids: Expert ID for each active range of shape
            `[num_experts]` as `int32`.
        sf_tensor: Per-expert tensor-wise scale factor of shape
            `[num_experts]` as `float32`.
    """
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be rank 1"
    comptime assert (
        scales_offsets.flat_rank == 1
    ), "scales_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert sf_tensor.flat_rank == 1, "sf_tensor must be rank 1"
    comptime assert OutputEngine.element_size == 1
    comptime assert InputEngine.element_size == 1
    comptime assert RowOffsetsEngine.element_size == 1
    comptime assert ScalesOffsetsEngine.element_size == 1
    comptime assert ExpertIdsEngine.element_size == 1
    comptime assert SfEngine.element_size == 1
    comptime assert RowIndicesEngine.element_size == 1
    comptime assert scales_offsets_layout.all_dims_known
    comptime num_experts = scales_offsets_layout.static_shape[0]

    comptime SF_VECTOR_SIZE = 16 if scales_dtype == NVFP4_SF_DTYPE else 32
    comptime SF_K_GROUP_SIZE = SF_VECTOR_SIZE * SF_ATOM_K

    var scale_tile_idx = block_idx.x

    wait_on_dependent_grids()

    var expert_idx = _scale_tile_expert[num_experts](
        row_offsets, scales_offsets, scale_tile_idx
    )

    var curr_expert_start = Int(row_offsets[expert_idx][0])
    var curr_expert_end = Int(row_offsets[expert_idx + 1][0])
    var expert_tiles_start = ufloordiv(
        curr_expert_start, SF_MN_GROUP_SIZE
    ) + Int(scales_offsets[expert_idx][0])
    var expert_num_tiles = uceildiv(
        curr_expert_end - curr_expert_start, SF_MN_GROUP_SIZE
    )
    # Padded tiles have no payload, so they exit without releasing dependents.
    if scale_tile_idx >= expert_tiles_start + expert_num_tiles:
        return

    var expert_id = expert_ids[expert_idx][0]
    var input_sf = sf_tensor[expert_id][0]

    var token_start = (
        curr_expert_start
        + (scale_tile_idx - expert_tiles_start) * SF_MN_GROUP_SIZE
    )
    var num_tokens = min(curr_expert_end - token_start, SF_MN_GROUP_SIZE)

    comptime scales_smem_tile_size = align_up(
        Int(scales_tile_shape.product()), 128
    )
    var smem_ptr = unsafe_stack_allocation[
        scales_smem_tile_size * k_tiles_per_block,
        Scalar[scales_dtype],
        alignment=128,
        address_space=.SHARED,
    ]()

    comptime ELEMENTS_PER_THREAD = 8
    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
    comptime OUTPUT_WIDTH = 4 if output_dtype == .uint8 else 8
    comptime num_threads_per_row = SF_K_GROUP_SIZE // ELEMENTS_PER_THREAD

    # Threads span the block's column tiles as well as its rows. A row is
    # num_threads_per_row threads wide, so a decode tile holding one routed
    # row occupies eight threads of the 128 on its own.
    comptime slots_per_iter = num_threads // num_threads_per_row
    comptime rows_per_iter = slots_per_iter // k_tiles_per_block
    comptime num_iters = SF_MN_GROUP_SIZE // rows_per_iter
    # Both divisions truncate. A block covering a non-integral number of
    # rows per pass leaves the remainder of every tile at its stale scale
    # factor, with no runtime failure.
    comptime assert (
        slots_per_iter % k_tiles_per_block == 0
        and rows_per_iter >= 1
        and SF_MN_GROUP_SIZE % rows_per_iter == 0
    ), (
        "block threads, column tiles per block and the 128-row scale tile"
        " must divide evenly, or rows are left unquantized"
    )

    # The shuffle masks below reach only bits within col_thread_idx, so a
    # lane and its partner share a slot, and with it a column tile and a
    # row. The reduction therefore stays inside one scale group.
    var slot, col_thread_idx = udivmod(thread_idx.x, num_threads_per_row)
    # This divisor order gives a warp one row and all its column tiles,
    # so its global accesses cover one contiguous 512B span. The tiles sit
    # a multiple of the bank count apart, so the scale store at the bottom
    # of the loop serializes 4 ways, a few cycles against that load.
    var row_in_iter, kt_local = udivmod(Int(slot), k_tiles_per_block)

    var num_cols = input_tensor.dim(1)
    var num_k_tiles = uceildiv(Int(num_cols), SF_K_GROUP_SIZE)
    var k_idx = block_idx.y * k_tiles_per_block + kt_local

    var scales_smem = TileTensor(
        smem_ptr.unsafe_offset(kt_local * scales_smem_tile_size),
        row_major(scales_tile_shape),
    )
    var scales_smem_flat = scales_smem.reshape(
        row_major[Int(scales_tile_shape.product())]()
    )
    var input_col = (
        k_idx * SF_K_GROUP_SIZE + col_thread_idx * ELEMENTS_PER_THREAD
    )
    var output_col = (
        ufloordiv(input_col, 2) if output_dtype == .uint8 else input_col
    )
    # A column tile beyond num_k_tiles starts at or after num_cols, so
    # this single test also excludes tiles the input does not have. An
    # invalid lane contributes a zero group max and skips its stores,
    # leaving its scale slot at the zero written below.
    var col_is_valid = Int(input_col) < Int(num_cols)

    # Padding rows must carry a zero scale factor, or the tensor cores
    # read garbage from them. At decode nearly every row of a tile is
    # padding, so the block clears every tile it owns in one contiguous
    # pass over the whole region.
    comptime assert (
        scales_smem_tile_size * k_tiles_per_block
    ) % num_threads == 0, (
        "scale tile elements must divide evenly across the block, or the"
        " tail of the region keeps stale scale factors"
    )
    comptime zeros_per_thread = (
        scales_smem_tile_size * k_tiles_per_block
    ) // num_threads
    comptime assert (
        zeros_per_thread & (zeros_per_thread - 1) == 0
    ), "zero-fill store width must be a power of two"
    smem_ptr.unsafe_store(
        thread_idx.x * zeros_per_thread,
        SIMD[scales_dtype, zeros_per_thread](0),
    )
    barrier()

    # num_tokens is uniform across the block, so the trip count is uniform
    # and the shuffle reduction below keeps the full participation its
    # member mask claims. At decode a tile holds one routed token and 127
    # padding rows, so one iteration of num_iters runs.
    var active_iters = min(num_iters, uceildiv(num_tokens, rows_per_iter))
    for iter_idx in range(active_iters):
        var local_row = iter_idx * rows_per_iter + row_in_iter
        var is_valid = local_row < num_tokens and col_is_valid

        var input_vector = SIMD[input_dtype, ELEMENTS_PER_THREAD](0)
        if is_valid:
            var source_row = token_start + local_row
            if case .Some(row_indices) = row_indices:
                source_row = Int(row_indices[source_row]) // row_index_divisor
            input_vector = input_tensor.load[width=ELEMENTS_PER_THREAD](
                (source_row, input_col)
            )

        var thread_max = abs(input_vector).reduce_max()
        thread_max = max(shuffle_xor(thread_max, 1), thread_max)
        comptime if NUM_THREADS_PER_SF == 4:
            thread_max = max(shuffle_xor(thread_max, 2), thread_max)
        var group_max = thread_max.cast[.float32]()

        var scale_factor: Float32
        comptime if output_dtype == .uint8:
            scale_factor = input_sf * group_max * recip(Float32(6.0))
        else:
            scale_factor = group_max * recip(Float32(448.0))
        var fp8_scale_factor = scale_factor.cast[scales_dtype]()

        var output_scale = Float32(0.0)
        if group_max != 0:
            comptime if output_dtype == .uint8:
                output_scale = recip(
                    fp8_scale_factor.cast[.float32]() * recip(input_sf)
                )
            else:
                output_scale = recip(fp8_scale_factor.cast[.float32]())

        if is_valid:
            var global_row = token_start + local_row
            var input_f32 = input_vector.cast[.float32]() * output_scale
            var output_vector: SIMD[output_dtype, OUTPUT_WIDTH]
            comptime if output_dtype == .uint8:
                output_vector = bitcast[output_dtype, OUTPUT_WIDTH](
                    cast_fp32_to_fp4e2m1(input_f32)
                )
            else:
                output_vector = rebind[SIMD[output_dtype, OUTPUT_WIDTH]](
                    input_f32.cast[output_dtype]()
                )
            output_tensor.store((global_row, output_col), output_vector)

        if col_thread_idx % NUM_THREADS_PER_SF == 0:
            var sf_group = col_thread_idx // NUM_THREADS_PER_SF
            var smem_idx = (
                (local_row % 32) * 16 + (local_row // 32) * SF_ATOM_K + sf_group
            )
            scales_smem_flat.store((smem_idx,), fp8_scale_factor)

    barrier()

    # Every tile this block owns is in shared memory, so their stores go
    # out as one group under a single wait.
    if thread_idx.x == 0:
        fence_async_view_proxy()
        for kt in range(k_tiles_per_block):
            var store_k_idx = block_idx.y * k_tiles_per_block + kt
            if store_k_idx >= num_k_tiles:
                break
            scales_tma_op.async_store_4d(
                TileTensor(
                    smem_ptr.unsafe_offset(kt * scales_smem_tile_size),
                    row_major(scales_tile_shape),
                ),
                (0, 0, store_k_idx, scale_tile_idx),
            )
        scales_tma_op.commit_group()
        scales_tma_op.wait_group[0]()

    launch_dependent_grids()


def grouped_quantize_dynamic_scaled_fp4_async[
    input_dtype: DType,
    output_dtype: DType,
    scales_dtype: DType,
    //,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    RowOffsetsEngine: TensorEngine,
    ScalesOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SfEngine: TensorEngine,
    RowIndicesLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64].element_types
    ],
    RowIndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    row_indices_dtype: DType = .uint32,
    row_index_divisor: Int = 1,
](
    output_tensor: TileTensor[
        mut=True, output_dtype, address_space=.GENERIC, ..., Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        mut=True,
        scales_dtype,
        address_space=.GENERIC,
        ...,
        Engine=ScalesEngine,
    ],
    input_tensor: TileTensor[
        mut=False, input_dtype, address_space=.GENERIC, ..., Engine=InputEngine
    ],
    row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ..., Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        mut=False,
        .uint32,
        address_space=.GENERIC,
        ...,
        Engine=ScalesOffsetsEngine,
    ],
    expert_ids: TileTensor[
        mut=False, .int32, address_space=.GENERIC, ..., Engine=ExpertIdsEngine
    ],
    sf_tensor: TileTensor[
        mut=False, .float32, address_space=.GENERIC, ..., Engine=SfEngine
    ],
    ctx: DeviceContext,
    row_indices: OptionalReg[
        TileTensor[
            mut=False,
            row_indices_dtype,
            RowIndicesLayoutType,
            ImmutAnyOrigin,
            Engine=RowIndicesEngine,
        ]
    ] = None,
) raises:
    """Launches the grouped per-expert quantization kernel for NVFP4/MXFP4/MXFP8 on SM100 hardware.

    Sets up the TMA scale-factor descriptor and enqueues the
    `grouped_quantize_dynamic_scaled_fp4_async_kernel` over a grid
    spanning the scale-factor tile dimensions.

    Parameters:
        input_dtype: Element type of the input activation tensor
            (inferred).
        output_dtype: Element type of the quantized output tensor
            (inferred).
        scales_dtype: Element type of the block scale-factor tensor
            (inferred).
        OutputEngine: Engine policy of the `output_tensor`.
        ScalesEngine: Engine policy of the `scales_tensor`.
        InputEngine: Engine policy of the `input_tensor`.
        RowOffsetsEngine: Engine policy of the `row_offsets` tensor.
        ScalesOffsetsEngine: Engine policy of the `scales_offsets` tensor.
        ExpertIdsEngine: Engine policy of the `expert_ids` tensor.
        SfEngine: Engine policy of the `sf_tensor`.
        RowIndicesLayoutType: Layout of the optional `row_indices` tensor.
        RowIndicesEngine: Engine policy of the `row_indices` tensor.
        row_indices_dtype: Integer element type of the `row_indices` tensor.
        row_index_divisor: Divisor applied to each `row_indices` entry to get
            the source row.
    Args:
        output_tensor: Output quantized tensor of packed FP4 `uint8`
            or `float8_e4m3fn` for MXFP8.
        scales_tensor: Output block scale-factor tensor in the 5D
            TCGEN interleaved layout.
        input_tensor: Input BF16 activation tensor.
        row_offsets: Contiguous row boundaries per expert of shape
            `[num_experts + 1]` as `uint32`, where range `i` spans
            rows `row_offsets[i]` through `row_offsets[i + 1]`.
        scales_offsets: Per-expert offset into `scales_tensor` of
            shape `[num_experts]` as `uint32`.
        expert_ids: Expert ID for each active range of shape
            `[num_experts]` as `int32`.
        sf_tensor: Per-expert tensor-wise scale factor of shape
            `[num_experts]` as `float32`.
        ctx: Device context used to enqueue the kernel.
        row_indices: Optional source row indices of shape `[padded_rows]`.
            When present, permuted output row `i` reads input row
            `row_indices[i] // row_index_divisor`, fusing the MoE expert
            permutation gather into the quantize loads; the output stays in
            permuted row order.
    """
    comptime assert (
        row_indices_dtype.is_integral()
    ), "row_indices must hold integers"
    comptime assert row_index_divisor > 0, "row_index_divisor must be positive"

    # The kernel masks columns at 8-element lane granularity, so a column
    # count that is not a multiple of 8 would still load and store partially
    # out of bounds within the straddling lane.
    debug_assert(
        Int(input_tensor.dim(1)) % 8 == 0,
        "num_cols must be a multiple of ELEMENTS_PER_THREAD (8 for NVFP4)",
    )

    # Blocks write only the column tiles implied by num_cols, so a scales
    # tensor wider than that keeps stale factors in its trailing tiles.
    comptime _SF_VECTOR_SIZE = 16 if scales_dtype == NVFP4_SF_DTYPE else 32
    debug_assert(
        Int(scales_tensor.dim[1]())
        == uceildiv(Int(input_tensor.dim(1)), _SF_VECTOR_SIZE * SF_ATOM_K),
        (
            "scales column-tile count must match the column count the kernel"
            " infers, or trailing tiles keep stale scale factors"
        ),
    )

    # The TMA box covers one (M1, K) atom, so the two innermost scale dims
    # are folded into a single contiguous extent for a rank-4 descriptor.
    var scales_4d_tensor = scales_tensor.reshape(
        row_major(
            Int(scales_tensor.dim[0]()),
            Int(scales_tensor.dim[1]()),
            Int(scales_tensor.dim[2]()),
            Int(scales_tensor.dim[3]()) * Int(scales_tensor.dim[4]()),
        )
    )

    comptime scales_tma_tile_shape = coord[
        1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    ]
    var scales_tma_op = create_tensor_tile[
        scales_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, scales_4d_tensor)

    # 64 threads leave too little work per block to hide latency, and 256
    # leave most of the block idle on a decode tile that holds one row.
    comptime BLOCK_THREADS = 128
    # Batching column tiles amortizes the per-block expert search but divides
    # the grid, so it pays only once the unbatched grid already fills the
    # machine. 4 measures fastest across the decode and prefill shapes, and
    # both 2 and 8 are slower on either side of it.
    #
    # The threshold counts allocated scale tiles, which is a worst case bound
    # on the tiles an expert layout actually occupies. A layout parking its
    # rows on a handful of experts clears the threshold on the bound alone and
    # gets batched into a grid too small to fill the machine, costing up to
    # 1.4x. Tightening this needs the row offsets on the host, so the sparse
    # layouts keep the penalty.
    comptime hw_info = ctx.default_device_info
    comptime blocks_per_sm = max(
        1, hw_info.threads_per_multiprocessor // BLOCK_THREADS
    )
    var resident_blocks = hw_info.sm_count * blocks_per_sm
    var unbatched_blocks = Int(scales_tensor.dim[0]()) * Int(
        scales_tensor.dim[1]()
    )

    def launch_quant_fp4_kernel[k_tiles_per_block: Int]() raises {imm}:
        comptime kernel = grouped_quantize_dynamic_scaled_fp4_async_kernel[
            output_dtype,
            scales_dtype,
            input_dtype,
            scales_tma_op.tile_shape,
            scales_tma_op.desc_shape,
            TensorMapSwizzle.SWIZZLE_NONE,
            output_tensor.LayoutType,
            input_tensor.LayoutType,
            row_offsets.LayoutType,
            scales_offsets.LayoutType,
            expert_ids.LayoutType,
            sf_tensor.LayoutType,
            OutputEngine,
            InputEngine,
            RowOffsetsEngine,
            ScalesOffsetsEngine,
            ExpertIdsEngine,
            SfEngine,
            RowIndicesLayoutType,
            RowIndicesEngine,
            row_indices_dtype,
            row_index_divisor,
            num_threads=BLOCK_THREADS,
            k_tiles_per_block=k_tiles_per_block,
        ]

        ctx.enqueue_function[kernel](
            output_tensor,
            scales_tma_op,
            input_tensor,
            row_offsets,
            scales_offsets,
            expert_ids,
            sf_tensor,
            row_indices,
            grid_dim=(
                Int(scales_tensor.dim[0]()),
                uceildiv(Int(scales_tensor.dim[1]()), k_tiles_per_block),
                1,
            ),
            block_dim=(BLOCK_THREADS,),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

    if unbatched_blocks >= resident_blocks:
        launch_quant_fp4_kernel[k_tiles_per_block=4]()
    else:
        launch_quant_fp4_kernel[k_tiles_per_block=1]()


########################################################
# Per-row-scaled NVFP4 quantization with the row amax in the kernel
########################################################

# A warp quantizes one row, and a block holds this many consecutive rows.
# Small blocks spread decode's few hundred rows over many SMs.
comptime _ROW_AMAX_ROWS_PER_BLOCK = 4
# A lane loads 8 columns per chunk, so rows of up to 12 x 256 = 3072 columns
# stay in registers between the amax and the quantize.
comptime _ROW_AMAX_CACHED_CHUNKS = 12
# The largest E2M1 value times the largest E4M3 value.
comptime _NVFP4_ROW_RANGE = Float32(6.0 * 448.0)


comptime nvfp4_row_input_fn_type = def[dtype: DType, width: SIMDLength](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width]
"""Elementwise transform applied to each input vector before quantizing.

`grouped_quantize_nvfp4_row_amax` quantizes `input_fn(x)` in place of `x`,
for both the row amax and the payload; the result is rounded to the input
dtype, as the graph op it replaces would round it. The graph op passes the
Nemotron-H `relu(x) * relu(x)` built from `nn.activations.relu`.
"""


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(rows_per_block * WARP_SIZE)
    )
)
@__name("grouped_quantize_nvfp4_row_amax_kernel")
def grouped_quantize_nvfp4_row_amax_kernel[
    output_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    row_scales_layout: TensorLayout,
    input_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    RowScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    RowOffsetsEngine: TensorEngine,
    ScalesOffsetsEngine: TensorEngine,
    input_fn: Optional[nvfp4_row_input_fn_type],
    row_indices_layout: TensorLayout = RowMajorLayout[
        *Coord[Int64].element_types
    ],
    RowIndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    row_indices_dtype: DType = .uint32,
    row_index_divisor: Int = 1,
    rows_per_block: Int = _ROW_AMAX_ROWS_PER_BLOCK,
](
    output_tensor: TileTensor[
        output_dtype, output_layout, MutAnyOrigin, Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        scales_dtype, scales_layout, MutAnyOrigin, Engine=ScalesEngine
    ],
    row_scales: TileTensor[
        .bfloat16, row_scales_layout, MutAnyOrigin, Engine=RowScalesEngine
    ],
    input_tensor: TileTensor[
        input_dtype, input_layout, ImmutAnyOrigin, Engine=InputEngine
    ],
    row_offsets: TileTensor[
        .uint32, row_offsets_layout, ImmutAnyOrigin, Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        .uint32,
        scales_offsets_layout,
        ImmutAnyOrigin,
        Engine=ScalesOffsetsEngine,
    ],
    row_indices: OptionalReg[
        TileTensor[
            row_indices_dtype,
            row_indices_layout,
            ImmutAnyOrigin,
            Engine=RowIndicesEngine,
        ]
    ] = None,
):
    """Quantizes BF16 rows to NVFP4, each with its own global scale.

    This is the fusion of the graph-op chain

    ```
    amax = max(abs(f32(x)), axis=-1)
    row_scale = bf16(max(amax / 2688, 1e-30))
    scaled = bf16(f32(x) * (1 / f32(row_scale)))
    grouped_quantize_dynamic_scaled_fp4(scaled, sf_tensor=1)
    ```

    with each rounding to BF16 kept, so that the payload and the block scales
    match the chain bit for bit. A warp owns a row: it loads the row into
    registers with 16-byte loads, reduces the row's amax, derives
    `row_scale`, and quantizes the row in 16-element blocks from those
    registers, so the row is read from memory once. Rows wider than the
    register cache are read a second time for the quantize.

    The grid scales with the rows rather than the scale tiles: the first
    `ceildiv(rows, rows_per_block)` blocks give a warp to each output row,
    which finds its expert, and so its scale tile, from the expert offsets
    in one round of loads. One more block per allocated scale tile clears
    the scale slots of that tile's padding rows, which must be zero. No two
    warps write the same byte, and the scale stores go through
    `scales_layout`.

    Parameters:
        output_dtype: Element type of the packed output (`uint8`).
        scales_dtype: Element type of the block scales (`float8_e4m3fn`).
        input_dtype: Element type of the input (`bfloat16`).
        output_layout: Layout of `output_tensor`.
        scales_layout: Layout of `scales_tensor`.
        row_scales_layout: Layout of `row_scales`.
        input_layout: Layout of `input_tensor`.
        row_offsets_layout: Layout of `row_offsets`.
        scales_offsets_layout: Layout of `scales_offsets`.
        OutputEngine: Engine policy of the `output_tensor`.
        ScalesEngine: Engine policy of the `scales_tensor`.
        RowScalesEngine: Engine policy of `row_scales`.
        InputEngine: Engine policy of the `input_tensor`.
        RowOffsetsEngine: Engine policy of the `row_offsets` tensor.
        ScalesOffsetsEngine: Engine policy of the `scales_offsets` tensor.
        input_fn: Optional transform quantized in place of each input
            vector, such as relu2.
        row_indices_layout: Layout of the optional `row_indices`.
        RowIndicesEngine: Engine policy of the `row_indices` tensor.
        row_indices_dtype: Integer element type of `row_indices`.
        row_index_divisor: Divisor applied to each `row_indices` entry to get
            the source row.
        rows_per_block: Rows (one per warp) each block quantizes.

    Args:
        output_tensor: Packed FP4 output, `[rows, K / 2]`.
        scales_tensor: Block scales in the 5D tcgen interleaved layout.
        row_scales: Each output row's BF16 global scale, `[rows]`.
        input_tensor: BF16 input, `[source rows, K]`.
        row_offsets: Row boundaries per expert, `[num_experts + 1]`.
        scales_offsets: Per-expert offset into the scale tiles.
        row_indices: Optional source row of each output row.
    """
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be rank 1"
    comptime assert (
        scales_offsets.flat_rank == 1
    ), "scales_offsets must be rank 1"
    comptime assert row_scales.flat_rank == 1, "row_scales must be rank 1"
    comptime assert scales_tensor.flat_rank == 5, "scales_tensor must be rank 5"
    comptime assert scales_dtype == NVFP4_SF_DTYPE
    comptime assert output_dtype == .uint8
    comptime assert OutputEngine.element_size == 1
    comptime assert ScalesEngine.element_size == 1
    comptime assert RowScalesEngine.element_size == 1
    comptime assert InputEngine.element_size == 1
    comptime assert RowOffsetsEngine.element_size == 1
    comptime assert ScalesOffsetsEngine.element_size == 1
    comptime assert RowIndicesEngine.element_size == 1
    comptime assert scales_offsets_layout.all_dims_known
    comptime num_experts = scales_offsets_layout.static_shape[0]

    comptime assert rows_per_block > 0, "rows_per_block must be positive"

    comptime SF_VECTOR_SIZE = NVFP4_SF_VECTOR_SIZE
    comptime SF_K_GROUP_SIZE = SF_VECTOR_SIZE * SF_ATOM_K
    comptime ELEMENTS_PER_THREAD = 8
    comptime COLS_PER_PASS = WARP_SIZE * ELEMENTS_PER_THREAD
    comptime CACHED_CHUNKS = _ROW_AMAX_CACHED_CHUNKS
    comptime InVec = SIMD[input_dtype, ELEMENTS_PER_THREAD]

    var num_cols = Int(input_tensor.dim(1))
    var num_k_tiles = uceildiv(num_cols, SF_K_GROUP_SIZE)
    var padded_cols = num_k_tiles * SF_K_GROUP_SIZE
    var num_rows = Int(output_tensor.dim(0))
    var num_row_blocks = uceildiv(num_rows, rows_per_block)
    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE

    # The expert offsets can come from the kernel this one overlaps under
    # PDL, so they are read after the wait.
    with PDL():
        if block_idx.x >= num_row_blocks:
            # A tile block: clear the scale slots of the tile's padding rows,
            # which must be zero.
            var scale_tile_idx = block_idx.x - num_row_blocks
            var expert_idx = _scale_tile_expert[num_experts](
                row_offsets, scales_offsets, scale_tile_idx
            )
            var expert_start = Int(row_offsets[expert_idx][0])
            var expert_end = Int(row_offsets[expert_idx + 1][0])
            var tile_in_expert = scale_tile_idx - (
                ufloordiv(expert_start, SF_MN_GROUP_SIZE)
                + Int(scales_offsets[expert_idx][0])
            )
            # Tiles past the expert's last are never read by the matmul.
            if tile_in_expert < uceildiv(
                expert_end - expert_start, SF_MN_GROUP_SIZE
            ):
                var num_tokens = min(
                    expert_end
                    - expert_start
                    - tile_in_expert * SF_MN_GROUP_SIZE,
                    SF_MN_GROUP_SIZE,
                )
                # Atom row `lane` holds tile rows `lane + 32 * c` for the 4
                # atom columns `c`; the first `valid_cols` carry tokens.
                var valid_cols = min(
                    SF_ATOM_M[1],
                    ceildiv(max(num_tokens - lane, 0), SF_ATOM_M[0]),
                )
                # The scale stores go through `scales_layout`, so a strided
                # view is honored.
                for k_tile in range(warp_id, num_k_tiles, rows_per_block):
                    for atom_col in range(valid_cols, SF_ATOM_M[1]):
                        scales_tensor.store(
                            (scale_tile_idx, k_tile, lane, atom_col, 0),
                            SIMD[scales_dtype, SF_ATOM_K](0),
                        )
        else:
            var row = block_idx.x * rows_per_block + warp_id
            # The row's expert is the last one whose rows start at or
            # before it. Each lane reads the offsets of a few experts, so
            # the warp finds it in one round of loads.
            comptime EXPERTS_PER_LANE = ceildiv(num_experts, WARP_SIZE)
            var lane_expert = Int32(-1)
            var lane_start = Int32(0)
            var lane_scales_offset = Int32(0)
            comptime for j in range(EXPERTS_PER_LANE):
                var e = lane * EXPERTS_PER_LANE + j
                if e < num_experts:
                    var start = Int32(row_offsets[e][0])
                    var offset = Int32(scales_offsets[e][0])
                    if Int(start) <= row:
                        lane_expert = Int32(e)
                        lane_start = start
                        lane_scales_offset = offset
            var expert_idx = Int(
                lane_group_max[num_lanes=WARP_SIZE](lane_expert)
            )
            var owner = lane_expert == Int32(expert_idx)
            var expert_start = Int(
                lane_group_max[num_lanes=WARP_SIZE](
                    lane_start if owner else Int32(-1)
                )
            )
            var expert_scales_offset = Int(
                lane_group_max[num_lanes=WARP_SIZE](
                    lane_scales_offset if owner else Int32(-1)
                )
            )
            var rows_end = Int(row_offsets[num_experts][0])
            if row < num_rows and row < rows_end and expert_idx >= 0:
                var row_in_expert = row - expert_start
                var scale_tile_idx = (
                    ufloordiv(expert_start, SF_MN_GROUP_SIZE)
                    + expert_scales_offset
                    + ufloordiv(row_in_expert, SF_MN_GROUP_SIZE)
                )
                var row_in_tile = row_in_expert % SF_MN_GROUP_SIZE
                # Tile row `r` sits at atom row `r % 32`, atom column
                # `r // 32` of each k-tile's 32 x 4 x 4 scale atom.
                var atom_row = row_in_tile % SF_ATOM_M[0]
                var atom_col = row_in_tile // SF_ATOM_M[0]
                var global_row = row
                var source_row = global_row
                if case .Some(row_indices) = row_indices:
                    source_row = (
                        Int(row_indices[global_row]) // row_index_divisor
                    )

                # Quantizes the 8 columns from `col` with the row's inverse
                # scale. Called by the whole warp for a warp-uniform `col - 8 *
                # lane`, with zeros past the end of the row, so that the scale
                # columns the last k-tile does not fill are written as zero.
                @always_inline
                def quantize_chunk(
                    v: InVec, col: Int, inv_row_scale: Float32
                ) {imm}:
                    var scaled = (v.cast[.float32]() * inv_row_scale).cast[
                        input_dtype
                    ]()
                    var thread_max = abs(scaled).reduce_max()
                    thread_max = max(shuffle_xor(thread_max, 1), thread_max)
                    var group_max = thread_max.cast[.float32]()
                    # The graph chain passes sf_tensor == 1, which these
                    # multiplications and reciprocals leave exact.
                    var scale_factor = group_max * recip(Float32(6.0))
                    var fp8_scale_factor = scale_factor.cast[scales_dtype]()
                    var output_scale = Float32(0.0)
                    if group_max != 0:
                        output_scale = recip(
                            fp8_scale_factor.cast[.float32]()
                            * recip(Float32(1.0))
                        )

                    if col < num_cols:
                        var scaled_f32 = scaled.cast[.float32]() * output_scale
                        output_tensor.store(
                            (global_row, col // 2),
                            bitcast[output_dtype, 4](
                                cast_fp32_to_fp4e2m1(scaled_f32)
                            ),
                        )
                    # Lanes 8j..8j+7 cover the four 16-column blocks of one
                    # k-tile; lane 8j gathers their four scale bytes, which are
                    # contiguous in the atom, into one 4-byte store.
                    var sf_byte = UInt32(
                        bitcast[DType.uint8, 1](fp8_scale_factor)
                    )
                    var sf_word = (
                        sf_byte
                        | (shuffle_down(sf_byte, 2) << 8)
                        | (shuffle_down(sf_byte, 4) << 16)
                        | (shuffle_down(sf_byte, 6) << 24)
                    )
                    if col < padded_cols and lane % 8 == 0:
                        scales_tensor.store(
                            (
                                scale_tile_idx,
                                ufloordiv(col, SF_K_GROUP_SIZE),
                                atom_row,
                                atom_col,
                                0,
                            ),
                            bitcast[scales_dtype, SF_ATOM_K](sf_word),
                        )

                var amax = Float32(0)
                if num_cols <= CACHED_CHUNKS * COLS_PER_PASS:
                    # Load the whole row first so that all loads are in flight
                    # together, then quantize from registers.
                    var chunks = Array[InVec, CACHED_CHUNKS](fill=InVec(0))
                    comptime for c in range(CACHED_CHUNKS):
                        var col = c * COLS_PER_PASS + lane * ELEMENTS_PER_THREAD
                        if col < num_cols:
                            var v = input_tensor.load[
                                width=ELEMENTS_PER_THREAD
                            ]((source_row, col))
                            comptime if input_fn:
                                comptime transform = input_fn.value()
                                v = transform(v)
                            chunks[c] = v
                    comptime for c in range(CACHED_CHUNKS):
                        amax = max(
                            amax, abs(chunks[c]).reduce_max().cast[.float32]()
                        )
                    amax = lane_group_max[num_lanes=WARP_SIZE](amax)

                    # The two BF16 roundings the graph chain makes: the row
                    # scale, and the input divided by it. The reciprocal is a
                    # true division so that it rounds as the chain's does.
                    var row_scale = max(
                        amax / _NVFP4_ROW_RANGE, Float32(1e-30)
                    ).cast[.bfloat16]()
                    var inv_row_scale = (
                        Float32(1.0) / row_scale.cast[.float32]()
                    )
                    if lane == 0:
                        row_scales[global_row] = row_scale

                    comptime for c in range(CACHED_CHUNKS):
                        if c * COLS_PER_PASS < padded_cols:
                            quantize_chunk(
                                chunks[c],
                                c * COLS_PER_PASS + lane * ELEMENTS_PER_THREAD,
                                inv_row_scale,
                            )
                else:
                    # A row too wide for the registers: read it twice.
                    for col0 in range(0, num_cols, COLS_PER_PASS):
                        var col = col0 + lane * ELEMENTS_PER_THREAD
                        if col < num_cols:
                            var v = input_tensor.load[
                                width=ELEMENTS_PER_THREAD
                            ]((source_row, col))
                            comptime if input_fn:
                                comptime transform = input_fn.value()
                                v = transform(v)
                            amax = max(
                                amax, abs(v).reduce_max().cast[.float32]()
                            )
                    amax = lane_group_max[num_lanes=WARP_SIZE](amax)

                    var row_scale = max(
                        amax / _NVFP4_ROW_RANGE, Float32(1e-30)
                    ).cast[.bfloat16]()
                    var inv_row_scale = (
                        Float32(1.0) / row_scale.cast[.float32]()
                    )
                    if lane == 0:
                        row_scales[global_row] = row_scale

                    for col0 in range(0, padded_cols, COLS_PER_PASS):
                        var col = col0 + lane * ELEMENTS_PER_THREAD
                        var v = InVec(0)
                        if col < num_cols:
                            v = input_tensor.load[width=ELEMENTS_PER_THREAD](
                                (source_row, col)
                            )
                            comptime if input_fn:
                                comptime transform = input_fn.value()
                                v = transform(v)
                        quantize_chunk(v, col, inv_row_scale)


def grouped_quantize_nvfp4_row_amax[
    input_dtype: DType,
    output_dtype: DType,
    scales_dtype: DType,
    //,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    RowScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    RowOffsetsEngine: TensorEngine,
    ScalesOffsetsEngine: TensorEngine,
    *,
    input_fn: Optional[nvfp4_row_input_fn_type] = None,
    RowIndicesLayoutType: TensorLayout = RowMajorLayout[
        *Coord[Int64].element_types
    ],
    RowIndicesEngine: TensorEngine = DefaultEngine[element_width=1],
    row_indices_dtype: DType = .uint32,
    row_index_divisor: Int = 1,
    rows_per_block: Int = _ROW_AMAX_ROWS_PER_BLOCK,
](
    output_tensor: TileTensor[
        mut=True, output_dtype, address_space=.GENERIC, ..., Engine=OutputEngine
    ],
    scales_tensor: TileTensor[
        mut=True,
        scales_dtype,
        address_space=.GENERIC,
        ...,
        Engine=ScalesEngine,
    ],
    row_scales: TileTensor[
        mut=True, .bfloat16, address_space=.GENERIC, ..., Engine=RowScalesEngine
    ],
    input_tensor: TileTensor[
        mut=False, input_dtype, address_space=.GENERIC, ..., Engine=InputEngine
    ],
    row_offsets: TileTensor[
        mut=False, .uint32, address_space=.GENERIC, ..., Engine=RowOffsetsEngine
    ],
    scales_offsets: TileTensor[
        mut=False,
        .uint32,
        address_space=.GENERIC,
        ...,
        Engine=ScalesOffsetsEngine,
    ],
    ctx: DeviceContext,
    row_indices: OptionalReg[
        TileTensor[
            mut=False,
            row_indices_dtype,
            RowIndicesLayoutType,
            ImmutAnyOrigin,
            Engine=RowIndicesEngine,
        ]
    ] = None,
) raises:
    """Quantizes grouped BF16 rows to NVFP4, each row with its own scale.

    Equal to a chain of a row amax, `row_scale = bf16(max(amax / 2688,
    1e-30))`, `bf16(x * (1 / row_scale))` and
    `grouped_quantize_dynamic_scaled_fp4_async` with a tensor-wise scale of 1,
    but in one kernel with the same bits out. `row_scales` is the matmul's
    `a_row_scales`. SM100 only.

    Parameters:
        input_dtype: Element type of the input (inferred, `bfloat16`).
        output_dtype: Element type of the packed output (inferred).
        scales_dtype: Element type of the block scales (inferred).
        OutputEngine: Engine policy of the `output_tensor`.
        ScalesEngine: Engine policy of the `scales_tensor`.
        RowScalesEngine: Engine policy of `row_scales`.
        InputEngine: Engine policy of the `input_tensor`.
        RowOffsetsEngine: Engine policy of the `row_offsets` tensor.
        ScalesOffsetsEngine: Engine policy of the `scales_offsets` tensor.
        input_fn: Optional elementwise transform quantized in place of the
            input, such as relu2 (see `nvfp4_row_input_fn_type`).
        RowIndicesLayoutType: Layout of the optional `row_indices` tensor.
        RowIndicesEngine: Engine policy of the `row_indices` tensor.
        row_indices_dtype: Integer element type of the `row_indices` tensor.
        row_index_divisor: Divisor applied to each `row_indices` entry to get
            the source row.
        rows_per_block: Rows (one per warp) each block quantizes.

    Args:
        output_tensor: Packed FP4 output, `[rows, K / 2]`.
        scales_tensor: Block scales, `[tiles, ceil(K / 64), 32, 4, 4]`.
        row_scales: Each row's BF16 global scale, `[rows]`; may be strided.
        input_tensor: BF16 input, `[source rows, K]`.
        row_offsets: Row boundaries per expert, `[num_experts + 1]`.
        scales_offsets: Per-expert offset into `scales_tensor`.
        ctx: Device context used to enqueue the kernel.
        row_indices: Optional source row of each output row; output row `i`
            reads input row `row_indices[i] // row_index_divisor`.
    """
    comptime assert (
        row_indices_dtype.is_integral()
    ), "row_indices must hold integers"
    comptime assert row_index_divisor > 0, "row_index_divisor must be positive"
    debug_assert(
        Int(input_tensor.dim(1)) % 8 == 0,
        "num_cols must be a multiple of 8",
    )
    debug_assert(
        Int(scales_tensor.dim[1]())
        == uceildiv(Int(input_tensor.dim(1)), NVFP4_SF_VECTOR_SIZE * SF_ATOM_K),
        "scales column-tile count must match the column count",
    )

    comptime kernel = grouped_quantize_nvfp4_row_amax_kernel[
        output_dtype,
        scales_dtype,
        input_dtype,
        output_tensor.LayoutType,
        scales_tensor.LayoutType,
        row_scales.LayoutType,
        input_tensor.LayoutType,
        row_offsets.LayoutType,
        scales_offsets.LayoutType,
        OutputEngine,
        ScalesEngine,
        RowScalesEngine,
        InputEngine,
        RowOffsetsEngine,
        ScalesOffsetsEngine,
        input_fn,
        RowIndicesLayoutType,
        RowIndicesEngine,
        row_indices_dtype,
        row_index_divisor,
        rows_per_block,
    ]
    # Row blocks, one warp per output row, then one block per scale tile for
    # its padding rows. An empty batch (no rows, so no tiles) has nothing to
    # write, and a zero-sized grid would fail to launch.
    var num_blocks = uceildiv(Int(output_tensor.dim(0)), rows_per_block) + Int(
        scales_tensor.dim[0]()
    )
    if num_blocks == 0:
        return
    ctx.enqueue_function[kernel](
        output_tensor,
        scales_tensor,
        row_scales,
        input_tensor,
        row_offsets,
        scales_offsets,
        row_indices,
        grid_dim=(num_blocks, 1, 1),
        block_dim=(rows_per_block * ctx.default_device_info.warp_size,),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


########################################################
# SM100 Block Scaled matmul kernel dispatch
########################################################


########################################################
# SM100 Block Scaled matmul with normal epilogue kernel dispatch
########################################################


def block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    a_scales: TileTensor[mut=False, scales_dtype, ...],
    b_scales: TileTensor[mut=False, scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Our sm100 block scaled matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 block scaled matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations. Callers must allocate `c`; with `has_epilogue_fn` or
    `has_compute_fn` the matmul result is written into `c` and then read
    back by the closure.

    Parameters:
        c_type: Element type of the output matrix (inferred).
        a_type: Element type of the LHS input matrix (inferred).
        b_type: Element type of the RHS input matrix (inferred).
        scales_dtype: Element type of the block scale-factor tensors
            (inferred).
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor: 16 for NVFP4 or 32 for MXFP8.
        transpose_b: Whether `b` is stored transposed (defaults to `True`).
        has_epilogue_fn: Whether `epilogue_fn` receives the matmul result
            after it is written to `c`.
        has_compute_fn: Whether `compute_fn` transforms the matmul result
            in `c`.
    Args:
        c: Output matrix of shape `[M, N]`.
        a: LHS input matrix of shape `[M, K]` for FP8 or `[M, K//2]` for
            packed FP4 `uint8`.
        b: RHS input matrix of shape `[N, K]` for FP8 or `[N, K//2]` for
            packed FP4 `uint8`, stored transposed.
        a_scales: Block scale factors for `a` in the 5D TCGEN interleaved
            layout.
        b_scales: Block scale factors for `b` in the 5D TCGEN interleaved
            layout.
        tensor_sf: Scalar multiplier applied to the matmul result.
        ctx: Device context used to enqueue kernels.
        epilogue_fn: With `has_epilogue_fn`, receives each output value
            at its `(m, n)`.
        compute_fn: With `has_compute_fn`, maps each output value before it
            is stored back into `c`.
    """

    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100 or SM120"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "pass at most one of the epilogue closure or compute closure"

    # The vendor (cuBLASLt) block-scaled backend supports both NVFP4
    # (float8_e4m3fn scales, 16-vector) and MXFP8 (float8_e8m0fnu scales,
    # 32-vector); see `_cublasLt_matmul` in `linalg.matmul.vendor.blas`.
    comptime is_mxfp8_scaling = (
        scales_dtype == MXFP8_SF_DTYPE
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    )
    comptime assert scales_dtype == NVFP4_SF_DTYPE or is_mxfp8_scaling, (
        "Only NVFP4 (float8_e4m3fn scales, SF_VECTOR_SIZE=16) or MXFP8"
        " (float8_e8m0fnu scales, SF_VECTOR_SIZE=32) scales are supported."
    )
    comptime assert (
        SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE or is_mxfp8_scaling
    ), (
        "SF_VECTOR_SIZE must be NVFP4_SF_VECTOR_SIZE (16) for NVFP4 or"
        " MXFP8_SF_VECTOR_SIZE (32) for MXFP8"
    )

    comptime assert (
        a_scales.static_shape[1] == b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var _k = Int(a.dim[1]()) * 2 if a_type == .uint8 else Int(a.dim[1]())
    if m == 0 or n == 0:
        return

    @inline(.always)
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            "(gpu",
            ";", trace_arg("A", (m, _k), a_type),
            ";", trace_arg("B", (_k, n), b_type),
            ";", trace_arg("C", (m, n), c_type),
            ";A_scales=[", a_scales.dim[0](), ",", a_scales.dim[1](), "]",
            ";B_scales=[", b_scales.dim[0](), ",", b_scales.dim[1](), "]",
            ";transpose_b=", transpose_b,
            ";tensor_sf=", tensor_sf,
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        get_static_string[
            "block_scaled_matmul_with_epilogue_",
            String("nvfp4_" if a_type == .uint8 else "mxfp8_"),
            String(SF_VECTOR_SIZE) + String("_sfvs"),
        ](),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        comptime if has_epilogue_fn or has_compute_fn:
            # Nvidia GPUs >= sm_100 arch support 32B load/store to global memory.
            comptime simd_size = 32 // size_of[c_type]()

            def value_epilogue_wrapper[
                simd_width: Int, alignment: Int = 1
            ](idx: Coord) {var c, var epilogue_fn, var compute_fn}:
                var c_val = c.load[width=simd_width](idx)
                comptime if has_epilogue_fn:
                    epilogue_fn[c_type, simd_width, alignment=alignment](
                        idx, c_val
                    )
                else:
                    c.store[alignment=alignment * size_of[c_type]()](
                        idx,
                        compute_fn[c_type, simd_width, alignment=alignment](
                            idx, c_val
                        ),
                    )

            matmul[scales_type=scales_dtype](
                ctx,
                c,
                a,
                b,
                a_scales=a_scales,
                b_scales=b_scales,
                alpha=tensor_sf,
                transpose_b=True,
                c_row_major=True,
            )
            elementwise[
                simd_size,
                target="gpu",
                _trace_description="elementwise_block_scaled_matmul_epilogue",
            ](value_epilogue_wrapper, (m, n), ctx)
        else:
            matmul[scales_type=scales_dtype](
                ctx,
                c,
                a,
                b,
                a_scales=a_scales,
                b_scales=b_scales,
                transpose_b=True,
                c_row_major=True,
                alpha=tensor_sf,
            )


# ===----------------------------------------------------------------------=== #
# TileTensor primary implementations
# ===----------------------------------------------------------------------=== #


@inline(.always)
def block_scaled_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn = type_of(no_epilogue_fn),
    ComputeFnType: ElementwiseComputeFn = type_of(no_compute_fn),
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    transpose_a: Bool = False,
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    compute_fn_needs_vectors: Bool = False,
    compute_fn_col_period: Int = 0,
    compute_fn_col_start: Int = 0,
    pdl_level: PDLLevel = PDLLevel.OFF,
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c_device: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a_device: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b_device: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=.GENERIC, ...
    ],
    b_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=.GENERIC, ...
    ],
    tensor_sf: Float32,
    ctx: DeviceContext,
    epilogue_fn: EpilogueFnType = no_epilogue_fn,
    compute_fn: ComputeFnType = no_compute_fn,
) raises:
    """Dispatches the SM100 block-scaled matmul for NVFP4 or MXFP8, selecting between the Mojo heuristic kernel and the vendor fallback.

    For small M or MXFP8 shapes the Mojo `heuristic_and_outliers_dispatch`
    kernel is tried first; on a dispatch miss the vendor (cuBLASLt)
    block-scaled matmul is used as a fallback.

    Parameters:
        c_type: Element type of the output matrix (inferred).
        a_type: Element type of the LHS input matrix (inferred).
        b_type: Element type of the RHS input matrix (inferred).
        scales_dtype: Element type of the block scale-factor tensors
            (inferred).
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor: 16 for NVFP4 or 32 for MXFP8.
        transpose_b: Whether `b_device` is stored transposed (defaults
            to `True`).
        transpose_a: Whether `a_device` is stored transposed (defaults
            to `False`).
        has_epilogue_fn: Whether `epilogue_fn` stores the output instead
            of the kernel writing `c_device`.
        has_compute_fn: Whether `compute_fn` maps each output value before
            it is stored.
        compute_fn_needs_vectors: Whether `compute_fn` reads each value
            with its neighbors in the row, so it must get consecutive columns
            rather than one element at a time.
        compute_fn_col_period: When nonzero, `compute_fn` returns its
            input unchanged at every column whose index modulo this
            period is below `compute_fn_col_start`, so an unfused pass
            visits only the remaining columns.
        compute_fn_col_start: The first column of each period that
            `compute_fn` may change.
        pdl_level: Programmatic dependency launch level for the
            heuristic dispatch kernel (defaults to `PDLLevel.OFF`).
        _trace_description: Extra label appended to the trace event
            for profiling (defaults to `""`).
        target: Trace target string identifying the device in
            profiling output (defaults to `"cpu"`).
    Args:
        c_device: Output matrix of shape `[M, N]`.
        a_device: LHS input matrix of shape `[M, K]` for FP8 or
            `[M, K//2]` for packed FP4 `uint8`.
        b_device: RHS input matrix of shape `[N, K]` for FP8 or
            `[N, K//2]` for packed FP4 `uint8`, stored transposed.
        a_scales_device: Block scale factors for `a_device` in the 5D
            TCGEN interleaved layout.
        b_scales_device: Block scale factors for `b_device` in the 5D
            TCGEN interleaved layout.
        tensor_sf: Scalar multiplier applied to the matmul result.
        ctx: Device context used to enqueue kernels.
        epilogue_fn: With `has_epilogue_fn`, stores each output value at
            its `(m, n)`. `c_device` still backs the vendor fallback.
        compute_fn: With `has_compute_fn`, maps each output value before it
            is stored into `c_device`.
    """
    comptime assert c_device.rank == 2 and c_device.flat_rank == 2
    comptime assert a_device.rank == 2 and a_device.flat_rank == 2
    comptime assert b_device.rank == 2 and b_device.flat_rank == 2
    comptime assert a_scales_device.rank == 5 and a_scales_device.flat_rank == 5
    comptime assert b_scales_device.rank == 5 and b_scales_device.flat_rank == 5

    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100 or SM120"

    comptime assert transpose_b, "Only support transposed B"

    # NVFP4 (float8_e4m3fn scales, 16-vector) and MXFP8 (float8_e8m0fnu scales,
    # 32-vector) are both lowered to the SM100 block-scaled MMA below. The
    # vendor/epilogue fallback path is still NVFP4-only, so MXFP8 is routed to
    # the Mojo `heuristic_and_outliers_dispatch` kernel (see below).
    comptime is_mxfp8_scaling = (
        scales_dtype == MXFP8_SF_DTYPE
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    )
    comptime assert scales_dtype == NVFP4_SF_DTYPE or is_mxfp8_scaling, (
        "Only NVFP4 (float8_e4m3fn scales, SF_VECTOR_SIZE=16) or MXFP8"
        " (float8_e8m0fnu scales, SF_VECTOR_SIZE=32) are supported."
    )

    comptime assert (
        SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE or is_mxfp8_scaling
    ), (
        "SF_VECTOR_SIZE must be NVFP4_SF_VECTOR_SIZE (16) for NVFP4 or"
        " MXFP8_SF_VECTOR_SIZE (32) for MXFP8"
    )

    var c = c_device.as_unsafe_any_origin()
    var a = a_device.as_unsafe_any_origin()
    var b = b_device.as_unsafe_any_origin()
    var a_scales = a_scales_device.as_unsafe_any_origin()
    var b_scales = b_scales_device.as_unsafe_any_origin()

    comptime assert (
        a_scales.static_shape[1] == b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]()) * 2 if a_type == .uint8 else Int(a.dim[1]())

    if m == 0 or n == 0:
        return

    logger.info(
        "------ Dispatching to SM100 (B200+) Block Scaled matmul kernel ------"
    )
    logger.info(
        "Input Data Types: ",
        a_type,
        ", ",
        b_type,
        " Output Data Type: ",
        c_type,
        " Problem Shape: MNK=[",
        m,
        ", ",
        n,
        ", ",
        k,
        "]",
    )

    comptime assert not (
        has_epilogue_fn and has_compute_fn
    ), "pass at most one of the epilogue closure or compute closure"

    comptime static_N = c_device.static_shape[1]
    comptime static_K = a_device.static_shape[1] * (
        2 if a_type == .uint8 else 1
    )
    comptime static_NK = Index(static_N, static_K)

    comptime if get_defined_bool["AUTOTUNING_MODE", False]():
        comptime BM = get_defined_int["TUNE_BM", 128]()
        comptime BN = get_defined_int["TUNE_BN", 128]()
        comptime BK = (
            TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]()
        )
        comptime MMA_K = 32
        comptime CLUSTER_DIM_X = get_defined_int["TUNE_CLUSTER_DIM_X", 2]()
        comptime CLUSTER_DIM_Y = get_defined_int["TUNE_CLUSTER_DIM_Y", 1]()
        comptime CLUSTER_DIM_Z = get_defined_int["TUNE_CLUSTER_DIM_Z", 1]()
        comptime CLUSTER_DIM = Index(
            CLUSTER_DIM_X, CLUSTER_DIM_Y, CLUSTER_DIM_Z
        )
        comptime BLOCK_SWIZZLE_SIZE = get_defined_int[
            "TUNE_BLOCK_SWIZZLE_SIZE", 0
        ]()
        comptime RASTERIZE_ORDER = get_defined_int["TUNE_RASTER_ORDER", 1]()
        comptime CTA_GROUP = get_defined_int["TUNE_CTA_GROUP", 2]()
        comptime K_GROUP_SIZE = get_defined_int["TUNE_K_GROUP_SIZE", 1]()
        comptime AB_SWAPPED = get_defined_bool["TUNE_AB_SWAPPED", False]()
        # Two stages of an MMA_N = 256 accumulator fill all 512 TMEM columns
        # before the scale factors, so N = 256 tiles need 1.
        comptime NUM_ACCUM_STAGES = get_defined_int[
            "TUNE_NUM_ACCUM_STAGES", 2
        ]()

        comptime umma_shape = Index(BM * CTA_GROUP, BN * CTA_GROUP, MMA_K)

        comptime config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=get_scaling_kind[
                a_type, scales_dtype, SF_VECTOR_SIZE, b_type
            ](),
            mma_shape=umma_shape,
            cluster_shape=CLUSTER_DIM,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            raster_order=RasterOrder(Int32(RASTERIZE_ORDER)),
            cta_group=CTA_GROUP,
            AB_swapped=AB_SWAPPED,
            k_group_size=K_GROUP_SIZE,
            num_accum_pipeline_stages=NUM_ACCUM_STAGES,
        )

        return blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            K=a_device.static_shape[1],
            config=config,
        ](c, a, b, a_scales, b_scales, ctx)

    comptime if get_defined_bool[
        "ENABLE_EXPERIMENTAL_SM100_BLOCK_SCALED_MATMUL", False
    ]():
        var status = _heuristic_and_outliers_dispatch_any_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
            compute_fn_needs_vectors=compute_fn_needs_vectors,
            compute_fn_col_period=compute_fn_col_period,
            compute_fn_col_start=compute_fn_col_start,
            pdl_level=pdl_level,
        ](
            c_device,
            a_device,
            b_device,
            a_scales,
            b_scales,
            tensor_sf,
            epilogue_fn,
            compute_fn,
            ctx,
        )

        if status == DISPATCH_HIT:
            logger.info("Executing SM100 Block Scaled matmul kernel")
            return
        else:
            raise Error("Heuristic and outliers dispatch failed")

    @inline(.always)
    def description_fn() {imm} -> String:
        # fmt: off
        return String(
            "(",
            target,
            ";", trace_arg("A", (m, k), a_type),
            ";", trace_arg("B", (k, n), b_type),
            ";", trace_arg("C", (m, n), c_type),
            ";A_scales=[", a_scales.dim[0](), ",", a_scales.dim[1](), ",", a_scales.dim[2](), ",", a_scales.dim[3](), ",", a_scales.dim[4](), "]",
            ";B_scales=[", b_scales.dim[0](), ",", b_scales.dim[1](), ",", b_scales.dim[2](), ",", b_scales.dim[3](), ",", b_scales.dim[4](), "]",
            ";transpose_a=", True,
            ";transpose_b=", transpose_b,
            ";tensor_sf=", tensor_sf,
            ")"
        )
        # fmt: on

    with Trace[TraceLevel.OP, target=target](
        # Create a string literal so that the event label works with the
        # AsyncRT profiler, whose event labels must be `StaticString`s.
        get_static_string[
            "block_scaled_matmul_",
            String("nvfp4_" if a_type == .uint8 else "mxfp8_"),
            String(SF_VECTOR_SIZE) + String("_sfvs"),
            _trace_description if _trace_description else "",
        ](),
        Trace[TraceLevel.OP]._get_detail_str(description_fn),
        task_id=get_safe_task_id(ctx),
    ):
        # For these large-N shapes on B200, Mojo also wins at M=256.
        comptime is_widened_shape = (
            static_K == 7168 and static_N in (18432, 36864)
        )
        comptime mojo_m_cap = 256 if is_widened_shape else 128
        # MXFP8 always tries the Mojo block-scaled kernel first (its heuristic
        # config table is tuned for the shapes we serve). On a dispatch miss it
        # falls through to the vendor (cuBLASLt) block-scaled matmul below, the
        # same fallback NVFP4 uses for large M.
        # SM100 (B200) tries the Mojo block-scaled kernel first; consumer
        # Blackwell (sm_120 / sm_121) has no SM100 kernel, so it is skipped at
        # compile time and control falls straight to the vendor fallback below.
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            if m <= mojo_m_cap or is_mxfp8_scaling:
                var status = _heuristic_and_outliers_dispatch_any_epilogue[
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                    transpose_b=transpose_b,
                    has_epilogue_fn=has_epilogue_fn,
                    has_compute_fn=has_compute_fn,
                    compute_fn_needs_vectors=compute_fn_needs_vectors,
                    compute_fn_col_period=compute_fn_col_period,
                    compute_fn_col_start=compute_fn_col_start,
                    pdl_level=pdl_level,
                ](
                    c_device,
                    a_device,
                    b_device,
                    a_scales,
                    b_scales,
                    tensor_sf,
                    epilogue_fn,
                    compute_fn,
                    ctx,
                )
                if status == DISPATCH_HIT:
                    return

        # Vendor (cuBLASLt) block-scaled fallback, used whenever the Mojo
        # heuristic dispatch above has no config for this shape. This covers
        # NVFP4 large-M shapes as well as MXFP8 shapes the heuristic config
        # table does not enumerate (e.g. the 30k-token prefill in KERN-3024,
        # whose M exceeds the table's M<=8192 build range).
        block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
            epilogue_fn,
            compute_fn,
        )


@inline(.always)
def _heuristic_and_outliers_dispatch_any_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool,
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    compute_fn_needs_vectors: Bool,
    compute_fn_col_period: Int,
    compute_fn_col_start: Int,
    pdl_level: PDLLevel,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises -> Int:
    comptime if has_epilogue_fn:
        return heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            pdl_level=pdl_level,
        ](c, a, b, a_scales, b_scales, tensor_sf, epilogue_fn, ctx)
    elif has_compute_fn:
        return heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            pdl_level=pdl_level,
            compute_fn_needs_vectors=compute_fn_needs_vectors,
            compute_fn_col_period=compute_fn_col_period,
            compute_fn_col_start=compute_fn_col_start,
        ](c, a, b, a_scales, b_scales, tensor_sf, compute_fn, ctx)
    else:
        return heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            pdl_level=pdl_level,
        ](c, a, b, a_scales, b_scales, tensor_sf, ctx)


@inline(.always)
def quantize_dynamic_block_scaled[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_device: TileTensor[mut=True, out_dtype, address_space=.GENERIC, ...],
    scales_device: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    input_device: TileTensor[mut=False, in_dtype, address_space=.GENERIC, ...],
    tensor_sf: Float32,  # tensor-wise scale factor
    ctx: DeviceContext,
) raises:
    """Dispatches dynamic block-scaled quantization to the appropriate hardware-specific kernel.

    Routes to the SM100 NVFP4 and MXFP8 TMA-async path, the SM100
    `quantize_dynamic_scaled_fp4fp8` path, or the AMD CDNA4 MXFP4 path
    based on the target device and scale-factor format.

    Parameters:
        out_dtype: Element type of the quantized output tensor
            (inferred). `uint8` for NVFP4/MXFP4 or `float8_e4m3fn`
            for MXFP8.
        scales_dtype: Element type of the block scale-factor tensor
            (inferred). `float8_e4m3fn` for NVFP4 or
            `float8_e8m0fnu` for MXFP8.
        in_dtype: Element type of the input activation tensor
            (inferred). Must be `bfloat16`.
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor: 16 for NVFP4 or 32 for MXFP8.
        target: Trace target string identifying the device in
            profiling output (defaults to `"cpu"`).
    Args:
        output_device: Output quantized tensor of packed FP4 `uint8`
            or `float8_e4m3fn` for MXFP8.
        scales_device: Output block scale-factor tensor in the 5D
            TCGEN interleaved layout on SM100 or rank-2 layout on AMD
            CDNA4.
        input_device: Input BF16 activation tensor.
        tensor_sf: Tensor-wise scale factor multiplied into each
            per-group scale factor.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert output_device.rank == 2 and output_device.flat_rank == 2
    comptime assert (
        scales_device.rank == 2 or scales_device.rank == 5
    ), "scales must be rank 2 (AMD) or rank 5 (SM100)"
    comptime assert input_device.rank == 2 and input_device.flat_rank == 2

    comptime assert in_dtype in (
        DType.bfloat16,
    ), "input dtype should be bfloat16"
    comptime assert out_dtype in (
        DType.uint8,
        DType.float8_e4m3fn,
    ), "output dtype should be uint8 or float8_e4m3fn"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
        MXFP8_SF_DTYPE,
    ), (
        "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn) or"
        " MXFP8_SF_DTYPE (float8_e8m0fnu)"
    )
    comptime assert (
        SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        or SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    ), (
        "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4) or"
        " MXFP8_SF_VECTOR_SIZE (32 for MXFP8)"
    )

    var input_tensor = input_device.as_unsafe_any_origin()
    var output_tensor = output_device.as_unsafe_any_origin()
    var scales_tensor = scales_device.as_unsafe_any_origin()

    var num_rows = input_tensor.dim(0)
    var num_cols = input_tensor.dim(1)
    if num_rows == 0 or num_cols == 0:
        return

    comptime static_input_N = input_tensor.static_shape[1]
    comptime static_output_N = output_tensor.static_shape[1]
    comptime is_nvfp4 = out_dtype == .uint8 and scales_dtype == NVFP4_SF_DTYPE and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    comptime is_mxfp4 = out_dtype == .uint8 and scales_dtype == MXFP4_SF_DTYPE and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
    comptime is_fp8 = out_dtype == .float8_e4m3fn and scales_dtype == MXFP8_SF_DTYPE and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    comptime assert is_nvfp4 or is_mxfp4 or is_fp8, "invalid scaling kind"

    comptime is_packed_fp4 = is_nvfp4 or is_mxfp4
    comptime if is_packed_fp4:
        comptime assert static_output_N == static_input_N // 2, (
            "output.dim(1) must be equal to input.dim(1) // 2 (each output"
            " element (uint8) is 2 fp4-e2m1fn values)"
        )

    comptime if _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ):
        # SM100 / SM120 path: rank-5 interleaved scales.
        comptime assert (
            scales_device.rank == 5 and scales_device.flat_rank == 5
        ), "SM100 requires rank-5 interleaved scales"

        comptime if (is_nvfp4 or is_fp8) and static_input_N % 32 == 0:
            # One block a row keeps every SM busy at small M, where 128-row
            # TMA tiles leave most SMs idle. Past this size the TMA kernel
            # streams faster.
            comptime if is_fp8 and static_input_N <= 8192:
                if num_rows * num_cols <= 3 * 2**23:
                    quantize_dynamic_scaled_fp4fp8[
                        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                        num_max_threads=1024,
                    ](
                        ctx,
                        output_tensor,
                        scales_tensor,
                        input_tensor,
                        num_cols=Int(num_cols),
                        num_cols_padded=Int(num_cols),
                        tensor_sf=tensor_sf,
                    )
                    return
            quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                ctx,
                output_tensor,
                scales_tensor,
                input_tensor,
                tensor_sf=tensor_sf,
            )
        else:
            comptime assert (
                static_input_N % (SF_VECTOR_SIZE // 2) == 0
            ), "input.dim(1) must be a multiple of (SF_VECTOR_SIZE // 2)"

            quantize_dynamic_scaled_fp4fp8[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                num_max_threads=512,
            ](
                ctx,
                output_tensor,
                scales_tensor,
                input_tensor,
                num_cols=Int(input_tensor.dim(1)),
                num_cols_padded=Int(input_tensor.dim(1)),
                tensor_sf=tensor_sf,
            )
    elif (
        out_dtype == .float8_e4m3fn
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        and ctx.default_device_info == MI355X
    ):
        # AMD CDNA4 MXFP8 path: unpacked E4M3 output, rank-2 E8M0 scales.
        comptime assert (
            scales_device.rank == 2 and scales_device.flat_rank == 2
        ), "MXFP8 requires rank-2 scales on CDNA4"
        comptime assert (
            static_input_N % MXFP8_SF_VECTOR_SIZE == 0
        ), "input.dim(1) must be a multiple of 32 (MXFP8_SF_VECTOR_SIZE)"
        comptime assert static_output_N == static_input_N, (
            "MXFP8 output.dim(1) must equal input.dim(1): one e4m3 byte per"
            " element, not the FP4 packed half-width"
        )
        quantize_mx_amd(ctx, output_tensor, scales_tensor, input_tensor)
    elif is_mxfp4 and ctx.default_device_info == MI355X:
        # AMD CDNA4 MXFP4 path: rank-2 scales.
        comptime assert (
            scales_device.rank == 2 and scales_device.flat_rank == 2
        ), "MXFP4 requires rank-2 scales on CDNA4"
        comptime assert (
            static_input_N % MXFP4_SF_VECTOR_SIZE == 0
        ), "input.dim(1) must be a multiple of 32 (MXFP4_SF_VECTOR_SIZE)"
        quantize_mx_amd(ctx, output_tensor, scales_tensor, input_tensor)
    else:
        comptime assert False, (
            "Unsupported hardware/format combination for block-scaled"
            " quantization"
        )


def swiglu_quantize_dynamic_block_scaled[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    activation: swiglu_activation_type,
](
    output_device: TileTensor[mut=True, out_dtype, address_space=.GENERIC, ...],
    scales_device: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    gate_up_device: TileTensor[
        mut=False, in_dtype, address_space=.GENERIC, ...
    ],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Quantizes the gated activation of a `[gate | up]` input to NVFP4 or MXFP8 with block scales.

    One kernel computes what `activation(gate, up)` rounded to `in_dtype`
    and then `quantize_dynamic_block_scaled` would, without writing the
    activation out. SM100 and SM120 only.

    Parameters:
        out_dtype: Element type of the quantized output (inferred). `uint8`
            for NVFP4 or `float8_e4m3fn` for MXFP8.
        scales_dtype: Element type of the block scales (inferred).
            `float8_e4m3fn` for NVFP4 or `float8_e8m0fnu` for MXFP8.
        in_dtype: Element type of the input (inferred). Must be `bfloat16`.
        SF_VECTOR_SIZE: Elements per block scale: 16 for NVFP4, 32 for
            MXFP8.
        activation: The gated activation, applied to each `(gate, up)`
            pair.

    Args:
        output_device: Quantized output of shape `[M, N // 2]` for packed
            FP4 or `[M, N]` for MXFP8.
        scales_device: Block scales in the 5D TCGEN interleaved layout.
        gate_up_device: Input of shape `[M, 2 * N]`: `gate` in the first
            `N` columns, `up` in the last `N`.
        tensor_sf: Tensor-wise scale factor multiplied into each block
            scale.
        ctx: Device context used to enqueue the kernel.

    Raises:
        Error: If `output_device` is not `[M, N]` for MXFP8 or `[M, N // 2]`
            for packed FP4.
    """
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "gated block-scaled quantization only runs on SM100 or SM120"
    comptime assert (
        output_device.rank == 2 and gate_up_device.rank == 2
    ), "output and gate_up must be rank 2"
    comptime assert (
        scales_device.rank == 5 and scales_device.flat_rank == 5
    ), "scales must be rank 5"
    comptime assert (
        gate_up_device.static_shape[1] % 2 == 0
    ), "gate_up must hold equal gate and up halves"
    # The kernel reduces each block's max across the lanes that hold it, so
    # a partial last block would read a lane outside the row.
    comptime assert (
        gate_up_device.static_shape[1] // 2 % SF_VECTOR_SIZE == 0
    ), "each gate_up half must be a whole number of scale blocks"

    comptime is_fp8 = out_dtype == .float8_e4m3fn
    var num_cols = Int(gate_up_device.dim(1)) // 2
    var output_cols = num_cols if is_fp8 else num_cols // 2
    if (
        Int(output_device.dim(0)) != Int(gate_up_device.dim(0))
        or Int(output_device.dim(1)) != output_cols
    ):
        raise Error(
            "output must be [M, N] for MXFP8 or [M, N // 2] for packed FP4,"
            " where gate_up is [M, 2 * N]"
        )
    if Int(gate_up_device.dim(0)) == 0 or num_cols == 0:
        return

    quantize_dynamic_scaled_fp4fp8[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        num_max_threads=1024 if is_fp8 else 512,
        gated=True,
        activation=activation,
    ](
        ctx,
        output_device.as_unsafe_any_origin(),
        scales_device.as_unsafe_any_origin(),
        gate_up_device.as_unsafe_any_origin(),
        num_cols=num_cols,
        num_cols_padded=num_cols,
        tensor_sf=tensor_sf,
    )


def rope_quantize_dynamic_block_scaled[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    head_dim: Int,
    inverse: Bool,
](
    output_device: TileTensor[mut=True, out_dtype, address_space=.GENERIC, ...],
    scales_device: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    input_device: TileTensor[mut=False, in_dtype, address_space=.GENERIC, ...],
    freqs_device: TileTensor[
        mut=False, DType.float32, address_space=.GENERIC, ...
    ],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Rotates the trailing columns of each head by RoPE and quantizes the result to NVFP4 or MXFP8 with block scales.

    One kernel computes what rotating each row's heads with `rope_ragged`'s
    interleaved arithmetic and then `quantize_dynamic_block_scaled` would,
    without writing the rotated input out. SM100 and SM120 only.

    Parameters:
        out_dtype: Element type of the quantized output (inferred). `uint8`
            for NVFP4 or `float8_e4m3fn` for MXFP8.
        scales_dtype: Element type of the block scales (inferred).
            `float8_e4m3fn` for NVFP4 or `float8_e8m0fnu` for MXFP8.
        in_dtype: Element type of the input (inferred). Must be `bfloat16`.
        SF_VECTOR_SIZE: Elements per block scale: 16 for NVFP4, 32 for
            MXFP8.
        head_dim: Columns per head. The input's width is a whole number of
            heads.
        inverse: Whether to rotate by the conjugate of each frequency.

    Args:
        output_device: Quantized output of shape `[M, N // 2]` for packed
            FP4 or `[M, N]` for MXFP8.
        scales_device: Block scales in the 5D TCGEN interleaved layout.
        input_device: Input of shape `[M, N]`.
        freqs_device: `[M, rope_dim]` interleaved `(cos, sin)` pairs, one
            row per input row. Each head's last `rope_dim` columns are
            rotated.
        tensor_sf: Tensor-wise scale factor multiplied into each block
            scale.
        ctx: Device context used to enqueue the kernel.

    Raises:
        Error: If `freqs_device` does not hold one row per input row, or
            `output_device` is not `[M, N]` for MXFP8 or `[M, N // 2]` for
            packed FP4.
    """
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "RoPE block-scaled quantization only runs on SM100 or SM120"
    comptime assert (
        output_device.rank == 2
        and input_device.rank == 2
        and freqs_device.rank == 2
    ), "output, input and freqs must be rank 2"
    comptime assert (
        scales_device.rank == 5 and scales_device.flat_rank == 5
    ), "scales must be rank 5"

    # A dynamic width reads as -1, which would skip the rotation.
    comptime assert (
        freqs_device.static_shape[1] > 0 and head_dim > 0
    ), "freqs must have a static width and head_dim must be positive"

    comptime is_fp8 = out_dtype == .float8_e4m3fn
    var num_cols = Int(input_device.dim(1))
    var output_cols = num_cols if is_fp8 else num_cols // 2
    if (
        Int(output_device.dim(0)) != Int(input_device.dim(0))
        or Int(output_device.dim(1)) != output_cols
    ):
        raise Error(
            "output must be [M, N] for MXFP8 or [M, N // 2] for packed FP4,"
            " where input is [M, N]"
        )
    if Int(input_device.dim(0)) == 0 or num_cols == 0:
        return
    if Int(freqs_device.dim(0)) != Int(input_device.dim(0)):
        raise Error("freqs must hold one row per input row")

    comptime FreqsTensor = TileTensor[
        DType.float32, freqs_device.LayoutType, ImmUnsafeAnyOrigin
    ]
    var freqs = rebind[FreqsTensor](freqs_device.as_unsafe_any_origin())

    def launch[num_max_threads: Int, grid_by_occupancy: Bool]() raises {imm}:
        quantize_dynamic_scaled_fp4fp8[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            num_max_threads=num_max_threads,
            rope_dim=freqs_device.static_shape[1],
            rope_head_dim=head_dim,
            rope_inverse=inverse,
            rope_layout=freqs_device.LayoutType,
            grid_by_occupancy=grid_by_occupancy,
        ](
            ctx,
            output_device.as_unsafe_any_origin(),
            scales_device.as_unsafe_any_origin(),
            input_device.as_unsafe_any_origin(),
            num_cols=num_cols,
            num_cols_padded=num_cols,
            tensor_sf=tensor_sf,
            rope_freqs=freqs,
        )

    comptime if is_fp8:
        # A 512-thread block fits three times an SM and a 1024-thread block
        # once. The smaller blocks win at few rows and at many. Between
        # them, one wave of the larger blocks wins up to about ten rows a
        # block, then the two waves of a grid sized by threads.
        var num_rows = Int(input_device.dim(0))
        if num_rows <= 256 or num_rows > 4096:
            launch[512, True]()
        elif num_rows <= 1536:
            launch[1024, True]()
        else:
            launch[1024, False]()
    else:
        launch[512, False]()


@inline(.always)
def block_scales_interleave[
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_scales_device: TileTensor[
        mut=True, scales_dtype, address_space=.GENERIC, ...
    ],
    input_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    """Reinterleaves rank-2 scale factors into the 5D TCGEN layout on SM100 hardware.

    Delegates to `block_scales_interleave_fp4` after validating that the
    output scales are rank-5 and the input scales are rank-2.

    Parameters:
        scales_dtype: Element type of the block scale-factor tensors
            (inferred).
        SF_VECTOR_SIZE: Number of elements covered by each block scale
            factor: 16 for NVFP4 or 32 for MXFP4.
        target: Trace target string identifying the device in
            profiling output (defaults to `"cpu"`).
    Args:
        output_scales_device: Output rank-5 scale-factor tensor in the
            5D TCGEN interleaved layout.
        input_scales_device: Input rank-2 scale-factor tensor in the
            flat row-major layout.
        ctx: Device context used to enqueue kernels.
    """
    comptime assert (
        output_scales_device.rank == 5 and output_scales_device.flat_rank == 5
    )
    comptime assert (
        input_scales_device.rank == 2 and input_scales_device.flat_rank == 2
    )

    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100 or SM120"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
        MXFP4_SF_DTYPE,
    ), "scales dtype should be float8_e4m3fn (NVFP4) or float8_e8m0fnu (MXFP4)."

    var output = output_scales_device.as_unsafe_any_origin()
    var input = input_scales_device.as_unsafe_any_origin()

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE,](
        ctx, input, output
    )


########################################################
# AMD MXFP4 quantization (CDNA4 / MI355X)
########################################################


@inline(.always)
def quantize_mxfp8_lane_group[
    in_dtype: DType,
    width: Int,
    //,
    out_dtype: DType,
    scales_dtype: DType,
    *,
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
](val: SIMD[in_dtype, width]) -> Tuple[
    SIMD[out_dtype, width], Scalar[scales_dtype]
]:
    """Quantizes one thread's slice of an MX block to MXFP8, cooperatively.

    The block max spans `SF_VECTOR_SIZE // width` lanes, so the caller's
    thread-to-column map must land each MX block on one aligned lane group.

    Parameters:
        in_dtype: Element type of the incoming values (bfloat16).
        width: Number of elements this thread holds.
        out_dtype: Quantized element type (`float8_e4m3fn`).
        scales_dtype: Block-scale type (`float8_e8m0fnu`).
        SF_VECTOR_SIZE: Elements covered by one block scale (32).

    Args:
        val: This thread's `width` contiguous elements.

    Returns:
        `(quantized, e8m0_scale)`. Every lane in the group returns the same
        scale; the caller stores it once, from the group's first lane.
    """
    comptime num_lanes = SF_VECTOR_SIZE // width
    comptime assert (
        num_lanes * width == SF_VECTOR_SIZE
    ), "width must divide SF_VECTOR_SIZE"
    comptime assert in_dtype == .bfloat16, "input dtype should be bfloat16"
    comptime assert (
        out_dtype == .float8_e4m3fn
    ), "output dtype should be float8_e4m3fn"
    comptime assert (
        scales_dtype == MXFP8_SF_DTYPE
    ), "scales dtype should be MXFP8_SF_DTYPE (float8_e8m0fnu)"
    # `lane_group_max` reduces over `2 ** log2_floor(num_lanes)` lanes, so a
    # non-power-of-two group silently drops the remainder from the block max.
    comptime assert (
        num_lanes.is_power_of_two() and num_lanes <= WARP_SIZE
    ), "SF_VECTOR_SIZE // width must be a power of two no larger than the warp"

    var thread_max = lane_group_max[num_lanes=num_lanes](abs(val).reduce_max())
    var e8m0_scale: Scalar[scales_dtype]
    var out_scale: Float32
    var block_is_dead: Bool
    e8m0_scale, out_scale, block_is_dead = compute_mxfp8_block_scale[
        scales_dtype
    ](thread_max.cast[.float32]())

    var data = val.cast[.float32]()
    # A dead block's multiplier is 0; zeroing avoids storing `inf * 0.0 = NaN`.
    if block_is_dead:
        data = type_of(data)(0.0)
    return ((data * out_scale).cast[out_dtype](), e8m0_scale)


comptime _MXFP8_Q_DTYPE = DType.float8_e4m3fn


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((q_cols + kv_cols) // 8)
    )
)
@__name("rms_norm_quantize_mxfp8_q_kv_kernel")
def rms_norm_quantize_mxfp8_q_kv_kernel[
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    kv_layout: TensorLayout,
    input_layout: TensorLayout,
    q_gamma_layout: TensorLayout,
    kv_gamma_layout: TensorLayout,
    *,
    q_cols: Int,
    kv_cols: Int,
](
    output: TileTensor[_MXFP8_Q_DTYPE, output_layout, MutAnyOrigin],
    scales: TileTensor[MXFP8_SF_DTYPE, scales_layout, MutAnyOrigin],
    kv_output: TileTensor[DType.bfloat16, kv_layout, MutAnyOrigin],
    input: TileTensor[DType.bfloat16, input_layout, ImmutAnyOrigin],
    q_gamma: TileTensor[DType.bfloat16, q_gamma_layout, ImmutAnyOrigin],
    kv_gamma: TileTensor[DType.bfloat16, kv_gamma_layout, ImmutAnyOrigin],
    epsilon: Float32,
    num_rows: Int32,
):
    comptime ELEMENTS_PER_THREAD = 8
    comptime q_threads = q_cols // ELEMENTS_PER_THREAD
    comptime q_warps = q_threads // WARP_SIZE
    comptime num_warps = (q_cols + kv_cols) // ELEMENTS_PER_THREAD // WARP_SIZE
    comptime assert (
        q_cols % (ELEMENTS_PER_THREAD * WARP_SIZE) == 0
        and kv_cols % (ELEMENTS_PER_THREAD * WARP_SIZE) == 0
    ), "q and kv rows must be whole warps of 8-element chunks"
    # Scale columns past the row's last block would need zeroing.
    comptime assert (
        q_cols % (MXFP8_SF_VECTOR_SIZE * SF_ATOM_K) == 0
    ), "q row must be whole scale atoms"

    var row = UInt32(block_idx.x)
    var tid = UInt32(thread_idx.x)
    var is_q = tid < UInt32(q_threads)
    var col = (tid if is_q else tid - UInt32(q_threads)) * UInt32(
        ELEMENTS_PER_THREAD
    )
    var sf_offset = scales.layout[linear_idx_type=.uint32](
        Coord(
            row // UInt32(SF_MN_GROUP_SIZE),
            col // UInt32(MXFP8_SF_VECTOR_SIZE * SF_ATOM_K),
            row % UInt32(SF_ATOM_M[0]),
            row % UInt32(SF_MN_GROUP_SIZE) // UInt32(SF_ATOM_M[0]),
            col // UInt32(MXFP8_SF_VECTOR_SIZE) % UInt32(SF_ATOM_K),
        )
    )

    with PDL[overlap_at_beginning=True]():
        if row >= UInt32(num_rows):
            # Tensor cores read the scales of padding rows, so they must be 0.
            if is_q and col % UInt32(MXFP8_SF_VECTOR_SIZE) == 0:
                scales.raw_store(sf_offset, Scalar[MXFP8_SF_DTYPE](0.0))
        else:
            var x: SIMD[DType.float32, ELEMENTS_PER_THREAD] = input.load[
                width=ELEMENTS_PER_THREAD
            ]((Int(row), Int(col) + (0 if is_q else q_cols))).cast[.float32]()

            # Loaded before the reduction so its latency overlaps the barrier.
            var gamma: SIMD[DType.bfloat16, ELEMENTS_PER_THREAD]
            if is_q:
                gamma = q_gamma.load[width=ELEMENTS_PER_THREAD]((Int(col),))
            else:
                gamma = kv_gamma.load[width=ELEMENTS_PER_THREAD]((Int(col),))

            # One reduction per row: q warps, then kv warps, of one block.
            var warp_sums = unsafe_stack_allocation[
                num_warps, DType.float32, address_space=.SHARED
            ]()
            var warp_ssq = warp.sum((x * x).reduce_add())
            if lane_id() == 0:
                warp_sums[warp_id()] = warp_ssq
            barrier()
            var ssq = Float32(0)
            if is_q:
                comptime for w in range(q_warps):
                    ssq += warp_sums[w]
            else:
                comptime for w in range(q_warps, num_warps):
                    ssq += warp_sums[w]
            var inv_rms = rsqrt(
                ssq / Float32(q_cols if is_q else kv_cols) + epsilon
            )

            # `rms_norm` rounds the normalized row to bfloat16 before it is
            # quantized or stored.
            var normed: SIMD[DType.bfloat16, ELEMENTS_PER_THREAD] = (
                (x * inv_rms) * gamma.cast[.float32]()
            ).cast[DType.bfloat16]()

            if is_q:
                var group_max = lane_group_max[
                    num_lanes=MXFP8_SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
                ](abs(normed).reduce_max()).cast[.float32]()
                var e8m0_scale: Scalar[MXFP8_SF_DTYPE]
                var out_scale: Float32
                var block_is_dead: Bool
                e8m0_scale, out_scale, block_is_dead = (
                    compute_mxfp8_block_scale[MXFP8_SF_DTYPE](group_max)
                )
                var quantized = (normed.cast[.float32]() * out_scale).cast[
                    _MXFP8_Q_DTYPE
                ]()
                # A dead block's multiplier is 0; store zeros, not inf * 0.
                if block_is_dead:
                    quantized = type_of(quantized)(0)
                output.store((Int(row), Int(col)), quantized)
                if col % UInt32(MXFP8_SF_VECTOR_SIZE) == 0:
                    scales.raw_store(sf_offset, e8m0_scale)
            else:
                kv_output.store((Int(row), Int(col)), normed)


def rms_norm_quantize_mxfp8_q_kv(
    output_device: TileTensor[
        mut=True, _MXFP8_Q_DTYPE, address_space=.GENERIC, ...
    ],
    scales_device: TileTensor[
        mut=True, MXFP8_SF_DTYPE, address_space=.GENERIC, ...
    ],
    kv_output_device: TileTensor[
        mut=True, DType.bfloat16, address_space=.GENERIC, ...
    ],
    input_device: TileTensor[
        mut=False, DType.bfloat16, address_space=.GENERIC, ...
    ],
    q_gamma_device: TileTensor[
        mut=False, DType.bfloat16, address_space=.GENERIC, ...
    ],
    kv_gamma_device: TileTensor[
        mut=False, DType.bfloat16, address_space=.GENERIC, ...
    ],
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """RMS-normalizes the q and kv parts of each row and quantizes the q part to MXFP8 with block scales.

    One block per row holds the whole `[q | kv]` row in registers. Its q warps
    reduce the sum of squares of the q part, its kv warps that of the kv part.
    The q part is quantized as `quantize_dynamic_block_scaled` would, and the
    kv part is written normalized in bfloat16. Both normalized parts are
    rounded to bfloat16 where `rms_norm` rounds them, so the bytes match the
    unfused kernels up to the order of the sum of squares. SM100 and SM120
    only.

    Args:
        output_device: Quantized q of shape `[M, q_cols]`.
        scales_device: Block scales of q in the 5D TCGEN interleaved layout.
        kv_output_device: Normalized kv of shape `[M, kv_cols]`.
        input_device: Input of shape `[M, q_cols + kv_cols]`, q then kv.
        q_gamma_device: The q weight, `[q_cols]`.
        kv_gamma_device: The kv weight, `[kv_cols]`.
        epsilon: Added to each mean square. Rounded to bfloat16 first, as
            `rms_norm` does.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert _is_sm10x_gpu(ctx.default_device_info) or _is_sm12x_gpu(
        ctx.default_device_info
    ), "RMS norm MXFP8 quantization only runs on SM100 or SM120"
    comptime assert (
        output_device.rank == 2
        and kv_output_device.rank == 2
        and input_device.rank == 2
        and q_gamma_device.rank == 1
        and kv_gamma_device.rank == 1
    ), "wrong tensor ranks"
    comptime assert (
        scales_device.rank == 5 and scales_device.flat_rank == 5
    ), "scales must be rank 5"
    comptime q_cols = output_device.static_shape[1]
    comptime kv_cols = kv_output_device.static_shape[1]
    comptime assert (
        q_cols > 0
        and kv_cols > 0
        and input_device.static_shape[1] == q_cols + kv_cols
    ), "the input must hold the static q and kv widths"

    var num_rows = Int(input_device.dim(0))
    if num_rows == 0:
        return

    comptime kernel = rms_norm_quantize_mxfp8_q_kv_kernel[
        output_device.LayoutType,
        scales_device.LayoutType,
        kv_output_device.LayoutType,
        input_device.LayoutType,
        q_gamma_device.LayoutType,
        kv_gamma_device.LayoutType,
        q_cols=q_cols,
        kv_cols=kv_cols,
    ]
    ctx.enqueue_function[kernel](
        output_device.as_unsafe_any_origin(),
        scales_device.as_unsafe_any_origin(),
        kv_output_device.as_unsafe_any_origin(),
        input_device.as_unsafe_any_origin(),
        q_gamma_device.as_unsafe_any_origin(),
        kv_gamma_device.as_unsafe_any_origin(),
        epsilon.cast[DType.bfloat16]().cast[.float32](),
        Int32(num_rows),
        grid_dim=align_up(num_rows, SF_MN_GROUP_SIZE),
        block_dim=(q_cols + kv_cols) // 8,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
@__name(t"quantize_mx_amd_kernel_{out_dtype}")
def _quantize_mx_amd_kernel[
    out_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    *,
    ELEMENTS_PER_THREAD: Int = 8,
    SF_VECTOR_SIZE: Int = MXFP4_SF_VECTOR_SIZE,  # 32 for MXFP4 and MXFP8 alike
    num_max_threads: Int = 512,
](
    output: TileTensor[
        out_dtype, output_layout, MutAnyOrigin, Engine=OutputEngine
    ],
    scales: TileTensor[
        .float8_e8m0fnu, scales_layout, MutAnyOrigin, Engine=ScalesEngine
    ],
    input: TileTensor[
        .bfloat16, input_layout, MutAnyOrigin, Engine=InputEngine
    ],
    num_rows: Int32,
    num_cols: Int32,
):
    """AMD MX quantization kernel: BF16 -> MXFP4 or MXFP8 + 2D E8M0 scales.

    Each thread processes ELEMENTS_PER_THREAD (8) BF16 values. Four threads
    cooperate to compute one E8M0 block scale over SF_VECTOR_SIZE (32)
    elements via warp shuffle.

    `out_dtype` selects the format: `uint8` packs two FP4 nibbles per byte via
    V_CVT_SCALEF32_PK_FP4_BF16 (CDNA4+), `float8_e4m3fn` stores one byte per
    element. Only the scale derivation and the store width differ.
    """
    # 2 for MXFP4 (nibble-packed), 1 for MXFP8 (one byte per element).
    comptime elems_per_byte = 2 if out_dtype == .uint8 else 1
    var _num_rows = Int(num_rows)
    var _num_cols = Int(num_cols)
    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
    comptime assert (
        NUM_THREADS_PER_SF == 4
    ), "MX quantization requires 4 threads per scale factor group"
    comptime assert OutputEngine.element_size == 1
    comptime assert ScalesEngine.element_size == 1
    comptime assert InputEngine.element_size == 1

    var num_col_threads = _num_cols // ELEMENTS_PER_THREAD

    for global_row_idx in range(block_idx.x, _num_rows, grid_dim.x):
        for col_thread_idx in range(thread_idx.x, num_col_threads, block_dim.x):
            var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD

            # 1. Load 8x BF16.
            var input_vector = input.load[ELEMENTS_PER_THREAD](
                (global_row_idx, global_col_idx)
            )

            # 2-6. Derive the E8M0 scale, convert, and store. MXFP4 rounds the
            # scale even-mode and hands it to the packing intrinsic; MXFP8
            # divides by 448 (E4M3 maxabs) and scales the data by the exact
            # reciprocal of the resulting power of two.
            var e8m0_scale: Float8_e8m0fnu
            comptime if elems_per_byte == 1:
                var quantized: SIMD[out_dtype, ELEMENTS_PER_THREAD]
                quantized, e8m0_scale = quantize_mxfp8_lane_group[
                    out_dtype,
                    DType.float8_e8m0fnu,
                    SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                ](input_vector)
                output.store((global_row_idx, global_col_idx), quantized)
            else:
                var thread_max = lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                    abs(input_vector).reduce_max()
                )
                var group_max = thread_max.cast[.float32]()
                e8m0_scale = compute_mxfp4_even_scale(group_max)
                var packed = cast_float_to_fp4e2m1_amd(
                    input_vector, e8m0_scale.cast[.float32]()
                )
                output.store(
                    (global_row_idx, col_thread_idx * 4),
                    bitcast[out_dtype, 4](packed),
                )

            # 7. First thread in the 4-thread group stores the scale.
            if global_col_idx % SF_VECTOR_SIZE == 0:
                var scale_col = global_col_idx // SF_VECTOR_SIZE
                scales.store((global_row_idx, scale_col), e8m0_scale)


@inline(.always)
def quantize_mx_amd[
    out_dtype: DType = .uint8,
    scales_dtype: DType = .float8_e8m0fnu,
    in_dtype: DType = .bfloat16,
    //,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    *,
    num_max_threads: Int = 512,
](
    ctx: DeviceContext,
    output_tile: TileTensor[
        mut=True, out_dtype, address_space=.GENERIC, ..., Engine=OutputEngine
    ],
    scales_tile: TileTensor[
        mut=True,
        scales_dtype,
        address_space=.GENERIC,
        ...,
        Engine=ScalesEngine,
    ],
    input_tile: TileTensor[
        mut=False, in_dtype, address_space=.GENERIC, ..., Engine=InputEngine
    ],
) raises:
    """Quantize BF16 activations to MXFP4 or MXFP8 on AMD CDNA4 (MI355X).

    Produces 2D E8M0 block scales compatible with dequant_mxfp4() and
    V_MFMA_SCALE_F32_16X16X128_F8F6F4.

    NOTE: The 2D scales layout is a stand-in. The optimized CDNA4 layout
    will likely be 6D (32x32 tiles) or 7D (16x16 tiles), mirroring how
    SM100 uses a 5D interleaved layout for its tensor core scale feed.

    Parameters:
        out_dtype: Element type of the quantized output tensor: `uint8` for
            packed MXFP4 or `float8_e4m3fn` for MXFP8 (inferred).
        scales_dtype: Element type of the block scale-factor tensor.
            Must be `float8_e8m0fnu` (inferred).
        in_dtype: Element type of the input activation tensor. Must
            be `bfloat16` (inferred).
        OutputEngine: Engine policy of the `output_tile` tensor.
        ScalesEngine: Engine policy of the `scales_tile` tensor.
        InputEngine: Engine policy of the `input_tile` tensor.
        num_max_threads: Maximum number of threads per block for the
            launch grid (defaults to 512).
    Args:
        ctx: Device context.
        output_tile: Output [M, K//2] uint8 (MXFP4) or [M, K] float8_e4m3fn
            (MXFP8).
        scales_tile: Output [M, K//32] float8_e8m0fnu (block scales).
        input_tile: Input [M, K] bfloat16.
    """
    comptime assert (
        out_dtype == .uint8 or out_dtype == .float8_e4m3fn
    ), "output must be uint8 (MXFP4) or float8_e4m3fn (MXFP8)"
    comptime assert (
        scales_dtype == .float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype == .bfloat16, "input must be bfloat16"
    comptime assert output_tile.flat_rank >= 2, "output must be rank 2"
    comptime assert scales_tile.flat_rank >= 2, "scales must be rank 2"
    comptime assert input_tile.flat_rank >= 2, "input must be rank 2"

    var num_rows = Int(input_tile.dim[0]())
    var num_cols = Int(input_tile.dim[1]())
    if num_rows == 0 or num_cols == 0:
        return

    comptime ELEMENTS_PER_THREAD = 8
    debug_assert(
        num_cols % MXFP4_SF_VECTOR_SIZE == 0,
        "num_cols must be a multiple of 32 (MXFP4_SF_VECTOR_SIZE)",
    )
    comptime _gpu = ctx.default_device_info

    var num_col_threads = ceildiv(num_cols, ELEMENTS_PER_THREAD)
    var block_dim_val = (min(num_col_threads, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, _gpu.threads_per_multiprocessor // block_dim_val[0]
    )
    var grid_dim_val = (
        min(num_rows, _gpu.sm_count * num_blocks_per_SM),
        1,
        1,
    )

    var input_tt = rebind[
        TileTensor[
            .bfloat16,
            type_of(input_tile).LayoutType,
            MutAnyOrigin,
            Engine=InputEngine,
        ]
    ](input_tile)

    comptime kernel = _quantize_mx_amd_kernel[
        out_dtype,
        type_of(output_tile).LayoutType,
        type_of(scales_tile).LayoutType,
        type_of(input_tt).LayoutType,
        OutputEngine,
        ScalesEngine,
        InputEngine,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        num_max_threads=num_max_threads,
    ]

    ctx.enqueue_function[kernel](
        output_tile,
        scales_tile,
        input_tt,
        Int32(num_rows),
        Int32(num_cols),
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
    )


@__name("quantize_dynamic_block_scaled_mxfp4_kernel")
def quantize_dynamic_block_scaled_mxfp4_kernel[
    in_dtype: DType,
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    OutputEngine: TensorEngine,
    ScalesEngine: TensorEngine,
    InputEngine: TensorEngine,
    *,
    elements_per_thread: Int,
](
    output: TileTensor[
        .uint8, output_layout, MutAnyOrigin, Engine=OutputEngine
    ],
    output_scales: TileTensor[
        .float8_e8m0fnu, scales_layout, MutAnyOrigin, Engine=ScalesEngine
    ],
    input: TileTensor[
        in_dtype, input_layout, ImmutAnyOrigin, Engine=InputEngine
    ],
    num_elements: Int32,
):
    var _num_elements = Int(num_elements)
    comptime threads_per_group = MXFP4_SF_VECTOR_SIZE // elements_per_thread

    var n = global_idx.x * elements_per_thread
    if n >= _num_elements:
        return

    var loaded_vec = input.reshape(row_major(input.num_elements())).load[
        width=8, alignment=align_of[in_dtype](), invariant=False
    ]((n,))
    var thread_max = abs(loaded_vec).reduce_max().cast[.float32]()
    var group_max = warp.lane_group_max[num_lanes=threads_per_group](thread_max)

    var fp8_scale_factor = compute_mxfp4_even_scale(group_max)
    var scale_f32 = fp8_scale_factor.cast[.float32]()

    if thread_idx.x % threads_per_group == 0:
        output_scales.reshape(row_major(output_scales.num_elements())).store(
            (n // MXFP4_SF_VECTOR_SIZE,), fp8_scale_factor
        )

    output.reshape(row_major(output.num_elements())).store[alignment=4](
        (n // 2,),
        bitcast[.uint8, 4](cast_float_to_fp4e2m1_amd(loaded_vec, scale_f32)),
    )


@inline(.always)
def quantize_dynamic_block_scaled_mxfp4[
    in_dtype: DType
](
    output: TileTensor[mut=True, .uint8, ...],
    output_scales: TileTensor[mut=True, .float8_e8m0fnu, ...],
    input: TileTensor[mut=False, in_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Launches the AMD CDNA4 MXFP4 quantization kernel over a flat input buffer.

    Enqueues `quantize_dynamic_block_scaled_mxfp4_kernel` with a
    512-thread block grid, emitting a trace event for profiling.

    Parameters:
        in_dtype: Element type of the input activation tensor
            (inferred). Must be `bfloat16`.
    Args:
        output: Output packed MXFP4 buffer of `uint8` values, where
            each byte holds two FP4-E2M1 values.
        output_scales: Output E8M0 block scale-factor buffer, with
            one scale per `MXFP4_SF_VECTOR_SIZE` (32) elements.
        input: Input BF16 activation buffer.
        ctx: Device context used to enqueue the kernel.
    """
    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "quantize_dynamic_block_scaled_mxfp4",
        task_id=get_safe_task_id(ctx),
    ):
        comptime elements_per_thread = 8
        comptime BLOCK_DIM = 512

        var num_elements = input.num_elements()

        if num_elements == 0:
            return
        if num_elements % MXFP4_SF_VECTOR_SIZE != 0:
            raise Error("unexpected input tensor size")

        comptime kernel = quantize_dynamic_block_scaled_mxfp4_kernel[
            in_dtype,
            type_of(output).LayoutType,
            type_of(output_scales).LayoutType,
            type_of(input).LayoutType,
            type_of(output).Engine,
            type_of(output_scales).Engine,
            type_of(input).Engine,
            elements_per_thread=elements_per_thread,
        ]

        ctx.enqueue_function[kernel](
            output,
            output_scales,
            input,
            Int32(num_elements),
            grid_dim=ceildiv(num_elements // elements_per_thread, BLOCK_DIM),
            block_dim=BLOCK_DIM,
        )


@inline(.always)
def _mxfp4_dotprod[
    out_dtype: DType,
    //,
    BLOCK_N: Int,
](
    c: TileTensor[mut=True, out_dtype, ...],
    c_offset: Int,
    a: TileTensor[mut=False, .uint8, ...],
    a_offset: Int,
    b: TileTensor[mut=False, .uint8, ...],
    b_offset: Int,
    a_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    a_scales_offset: Int,
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    b_scales_offset: Int,
    K: Int,
):
    @inline(.always)
    def cast_fp2em1x2_to_bf16x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[.bfloat16, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4", SIMD[.bfloat16, 2]
        ](packed, scale, Int32(byte_select))

    @inline(.always)
    def dotprod_bf16x2(
        a: SIMD[.bfloat16, 2], b: SIMD[.bfloat16, 2], c: Float32
    ) -> Float32:
        return llvm_intrinsic["llvm.amdgcn.fdot2.f32.bf16", Float32](
            a, b, c, False
        )

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var a_flat = a.reshape(row_major(a.num_elements()))
    var b_flat = b.reshape(row_major(b.num_elements()))
    var a_scales_flat = a_scales.reshape(row_major(a_scales.num_elements()))
    var b_scales_flat = b_scales.reshape(row_major(b_scales.num_elements()))

    var a_local_offset = a_offset
    var b_local_offset = b_offset

    var accum = SIMD[.float32, BLOCK_N](0)

    for ko in range(k_groups):
        var a_scale = (
            a_scales_flat.load[width=1, invariant=False](
                (a_scales_offset + ko,)
            )
        ).cast[.float32]()
        var b_scale = Array[_, BLOCK_N](
            fill_with_unrolled=lambda [bn: Int]() {
                imm b_scales_flat, imm b_scales_offset, imm k_groups, imm ko
            } -> Float32: b_scales_flat.load[width=1, invariant=False](
                (b_scales_offset + bn * k_groups + ko,)
            ).cast[
                .float32
            ]()
        )

        comptime for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[.int32, 1](
                a_flat.load[
                    width=4, alignment=align_of[DType.uint8](), invariant=False
                ]((a_local_offset + ki,))
            )
            var b_data = Array[_, BLOCK_N](
                fill_with_unrolled=lambda [bn: Int]() {
                    imm b_flat, imm b_local_offset, imm K
                } -> Int32: bitcast[.int32, 1](
                    b_flat.load[
                        width=4,
                        alignment=align_of[DType.uint8](),
                        invariant=False,
                    ]((b_local_offset + bn * (K // 2) + ki,))
                )
            )

            comptime for byte_select in range(4):
                var a_slice_bf16x2 = cast_fp2em1x2_to_bf16x2[byte_select](
                    a_data, a_scale
                )

                comptime for bn in range(BLOCK_N):
                    accum[bn] = dotprod_bf16x2(
                        a_slice_bf16x2,
                        cast_fp2em1x2_to_bf16x2[byte_select](
                            b_data[bn], b_scale[bn]
                        ),
                        accum[bn],
                    )

        a_local_offset += MXFP4_SF_VECTOR_SIZE // 2
        b_local_offset += MXFP4_SF_VECTOR_SIZE // 2

    c.reshape(row_major(c.num_elements())).store[
        alignment=align_of[out_dtype]()
    ]((c_offset,), accum.cast[out_dtype]())


@inline(.always)
def _mxfp4_dotprod_block_size(static_N: Int) -> Int:
    comptime target_block_size = 16
    return target_block_size if (static_N % target_block_size) == 0 else 1


@__name("matmul_dynamic_block_scaled_amd_kernel")
def matmul_dynamic_block_scaled_amd_kernel[
    out_dtype: DType,
    BLOCK_N: Int,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scales_layout: TensorLayout,
    b_scales_layout: TensorLayout,
    CEngine: TensorEngine,
    AEngine: TensorEngine,
    BEngine: TensorEngine,
    AScalesEngine: TensorEngine,
    BScalesEngine: TensorEngine,
](
    c: TileTensor[out_dtype, c_layout, MutAnyOrigin, Engine=CEngine],
    a: TileTensor[.uint8, a_layout, ImmutAnyOrigin, Engine=AEngine],
    b: TileTensor[.uint8, b_layout, ImmutAnyOrigin, Engine=BEngine],
    a_scales: TileTensor[
        .float8_e8m0fnu, a_scales_layout, ImmutAnyOrigin, Engine=AScalesEngine
    ],
    b_scales: TileTensor[
        .float8_e8m0fnu, b_scales_layout, ImmutAnyOrigin, Engine=BScalesEngine
    ],
    M: Int32,
    N: Int32,
    K: Int32,
):
    var _M = Int(M)
    var _N = Int(N)
    var _K = Int(K)
    var n = global_idx.x * BLOCK_N
    var m = global_idx.y

    if m >= _M or n >= _N:
        return

    var k_groups = _K // MXFP4_SF_VECTOR_SIZE

    _mxfp4_dotprod[BLOCK_N](
        c,
        m * _N + n,
        a,
        m * (_K // 2),
        b,
        n * (_K // 2),
        a_scales,
        m * k_groups,
        b_scales,
        n * k_groups,
        _K,
    )


@inline(.always)
def matmul_dynamic_block_scaled_amd[
    out_dtype: DType
](
    c: TileTensor[mut=True, out_dtype, ...],
    a: TileTensor[mut=False, .uint8, ...],
    b: TileTensor[mut=False, .uint8, ...],
    a_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    """Launches the AMD CDNA4 MXFP4 block-scaled matmul kernel.

    Selects a `BLOCK_N` of 16 when the N dimension is divisible by 16,
    otherwise falls back to 1, and enqueues
    `matmul_dynamic_block_scaled_amd_kernel` over a 2D grid.

    Parameters:
        out_dtype: Element type of the output matrix `c`.
    Args:
        c: Output matrix of shape `[M, N]`.
        a: LHS input matrix of packed MXFP4 `uint8` with shape
            `[M, K//2]`.
        b: RHS input matrix of packed MXFP4 `uint8` with shape
            `[N, K//2]`.
        a_scales: Block scale factors for `a` in `float8_e8m0fnu`
            format.
        b_scales: Block scale factors for `b` in `float8_e8m0fnu`
            format.
        ctx: Device context used to enqueue the kernel.
    """
    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "matmul_dynamic_block_scaled_amd",
        task_id=get_safe_task_id(ctx),
    ):
        comptime BLOCK_DIM = 16
        comptime BLOCK_N = _mxfp4_dotprod_block_size(c.static_shape[1])

        var M = Int(c.dim[0]())
        var N = Int(c.dim[1]())
        var K = Int(b.dim[1]()) * 2

        if M == 0 or N == 0:
            return

        comptime kernel = matmul_dynamic_block_scaled_amd_kernel[
            out_dtype,
            BLOCK_N,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(a_scales).LayoutType,
            type_of(b_scales).LayoutType,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(b).Engine,
            type_of(a_scales).Engine,
            type_of(b_scales).Engine,
        ]

        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            a_scales,
            b_scales,
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(N // BLOCK_N, BLOCK_DIM), ceildiv(M, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )


@__name("grouped_matmul_block_scaled_amd_kernel")
def grouped_matmul_block_scaled_amd_kernel[
    out_dtype: DType,
    BLOCK_N: Int,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scales_layout: TensorLayout,
    b_scales_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    CEngine: TensorEngine,
    AEngine: TensorEngine,
    BEngine: TensorEngine,
    AScalesEngine: TensorEngine,
    BScalesEngine: TensorEngine,
    RowOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
](
    c: TileTensor[out_dtype, c_layout, MutAnyOrigin, Engine=CEngine],
    a: TileTensor[.uint8, a_layout, ImmutAnyOrigin, Engine=AEngine],
    b: TileTensor[.uint8, b_layout, ImmutAnyOrigin, Engine=BEngine],
    a_scales: TileTensor[
        .float8_e8m0fnu, a_scales_layout, ImmutAnyOrigin, Engine=AScalesEngine
    ],
    b_scales: TileTensor[
        .float8_e8m0fnu, b_scales_layout, ImmutAnyOrigin, Engine=BScalesEngine
    ],
    row_offsets: TileTensor[
        .uint32, row_offsets_layout, ImmutAnyOrigin, Engine=RowOffsetsEngine
    ],
    expert_ids: TileTensor[
        .int32, expert_ids_layout, ImmutAnyOrigin, Engine=ExpertIdsEngine
    ],
    num_active_experts: Int32,
    M: Int32,
    N: Int32,
    K: Int32,
):
    var _num_active_experts = Int(num_active_experts)
    var _M = Int(M)
    var _N = Int(N)
    var _K = Int(K)
    var n = global_idx.x * BLOCK_N
    var m = global_idx.y

    if m >= _M or n >= _N:
        return

    var expert_id = -1
    var row_offsets_flat = row_offsets.reshape(
        row_major(row_offsets.num_elements())
    )
    var expert_ids_flat = expert_ids.reshape(
        row_major(expert_ids.num_elements())
    )
    for i in range(_num_active_experts):
        if m >= Int(
            rebind[UInt32](row_offsets_flat.load[invariant=False]((i,)))
        ) and m < Int(
            rebind[UInt32](row_offsets_flat.load[invariant=False]((i + 1,)))
        ):
            expert_id = Int(
                rebind[Int32](expert_ids_flat.load[invariant=False]((i,)))
            )
            break

    if expert_id == -1:
        c.reshape(row_major(c.num_elements())).store[
            alignment=align_of[out_dtype]()
        ]((m * _N + n,), SIMD[out_dtype, BLOCK_N](0))
        return

    var k_groups = _K // MXFP4_SF_VECTOR_SIZE

    _mxfp4_dotprod[BLOCK_N](
        c,
        m * _N + n,
        a,
        m * (_K // 2),
        b,
        (expert_id * _N + n) * (_K // 2),
        a_scales,
        m * k_groups,
        b_scales,
        (expert_id * _N + n) * k_groups,
        _K,
    )


@inline(.always)
def grouped_matmul_block_scaled_amd[
    out_dtype: DType,
](
    c: TileTensor[mut=True, out_dtype, ...],
    a: TileTensor[mut=False, .uint8, ...],
    b: TileTensor[mut=False, .uint8, ...],
    a_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, .float8_e8m0fnu, ...],
    row_offsets: TileTensor[mut=False, .uint32, ...],
    expert_ids: TileTensor[mut=False, .int32, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launches the grouped per-expert MXFP4 block-scaled matmul kernel on AMD CDNA4.

    Enqueues `grouped_matmul_block_scaled_amd_kernel` over a 2D grid
    sized by the output M and N dimensions, emitting a trace event for
    profiling.

    Parameters:
        out_dtype: Element type of the output matrix.
    Args:
        c: Output matrix of shape `[M, N]`.
        a: LHS activation matrix of shape `[M, K//2]` as packed MXFP4
            `uint8`.
        b: RHS expert weight matrix of shape `[E, N, K//2]` as packed
            MXFP4 `uint8`, where `E` is the number of experts.
        a_scales: Block scale factors for `a` of shape `[M, K//32]` as
            `float8_e8m0fnu`.
        b_scales: Block scale factors for `b` of shape `[E, N, K//32]`
            as `float8_e8m0fnu`.
        row_offsets: Contiguous row boundaries per expert of shape
            `[num_active_experts + 1]` as `uint32`, where range `i`
            spans rows `row_offsets[i]` through `row_offsets[i + 1]`.
        expert_ids: Expert ID for each active range of shape
            `[num_active_experts]` as `int32`.
        num_active_experts: Number of active experts to search over.
        ctx: Device context used to enqueue the kernel.
    """
    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "grouped_matmul_block_scaled_amd",
        task_id=get_safe_task_id(ctx),
    ):
        comptime BLOCK_DIM = 16
        comptime BLOCK_N = _mxfp4_dotprod_block_size(c.static_shape[1])

        var M = Int(c.dim[0]())
        var N = Int(c.dim[1]())
        var K = Int(b.dim[2]()) * 2

        if M == 0 or N == 0:
            return

        comptime kernel = grouped_matmul_block_scaled_amd_kernel[
            out_dtype,
            BLOCK_N,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(a_scales).LayoutType,
            type_of(b_scales).LayoutType,
            type_of(row_offsets).LayoutType,
            type_of(expert_ids).LayoutType,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(b).Engine,
            type_of(a_scales).Engine,
            type_of(b_scales).Engine,
            type_of(row_offsets).Engine,
            type_of(expert_ids).Engine,
        ]

        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            a_scales,
            b_scales,
            row_offsets,
            expert_ids,
            Int32(num_active_experts),
            Int32(M),
            Int32(N),
            Int32(K),
            grid_dim=(ceildiv(N // BLOCK_N, BLOCK_DIM), ceildiv(M, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )
