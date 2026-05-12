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

from std.math import align_up, ceildiv
from std.math.uutils import uceildiv, udivmod, ufloordiv
from std.memory import stack_allocation
from std.gpu import (
    block_idx,
    thread_idx,
    grid_dim,
    block_dim,
    global_idx,
    lane_id,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from layout.tile_layout import TensorLayout
from std.logger import Logger
from std.gpu.primitives import warp
from std.gpu.primitives.warp import lane_group_max, shuffle_xor
from std.math import recip
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
from std.gpu.host.info import B200, MI355X, _is_sm10x_gpu
from std.utils import StaticTuple
from std.collections import Optional
from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)
from std.utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from std.memory import bitcast
from std.gpu.sync import named_barrier
from std.gpu.intrinsics import warpgroup_reg_dealloc
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _idx_product,
    create_tensor_tile,
)
from layout.layout_tensor import LayoutTensorIter
from std.gpu.memory import external_memory, fence_async_view_proxy
from std.gpu import barrier
from std.sys import size_of, align_of, simd_width_of, get_defined_int
from layout.swizzle import make_swizzle
from std.algorithm import elementwise
from std.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.sys import get_defined_bool
from std.sys.intrinsics import llvm_intrinsic
from linalg.matmul.gpu.sm100.block_scaled_dispatch import (
    heuristic_and_outliers_dispatch,
)
from std.gpu.primitives.grid_controls import PDLLevel
from linalg.matmul.gpu.sm100_structured.default.dispatch import DISPATCH_HIT
from std.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from std.runtime.tracing import Trace, TraceLevel, trace_arg, get_safe_task_id
from std.collections.string.string_slice import get_static_string
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from linalg.matmul.gpu.sm100.tile_scheduler import RasterOrder
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)

########################################################
# Dynamic scaled NVFP4 quantization
########################################################

comptime logger = Logger()


@always_inline
def quantize_dynamic_scaled_fp4fp8[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 512,
](
    ctx: DeviceContext,
    output_tile: TileTensor[
        mut=True, out_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_tile: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_tile: TileTensor[
        mut=False, in_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    num_cols: Int,
    num_cols_padded: Int,
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
) raises:
    var output = output_tile.to_layout_tensor()
    var scales = scales_tile.to_layout_tensor()
    var input = input_tile.to_layout_tensor()
    comptime output_layout = output.layout
    comptime scales_layout = scales.layout
    comptime input_layout = input.layout

    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"
    comptime assert in_dtype in (
        DType.bfloat16,
    ), "input dtype should be bfloat16"

    comptime assert (
        (
            out_dtype == DType.uint8
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
            and scales_dtype == DType.float8_e4m3fn
        )
        or (
            out_dtype == DType.uint8
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
            and scales_dtype == DType.float8_e8m0fnu
        )
        or (
            out_dtype == DType.float8_e4m3fn
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
            and scales_dtype == DType.float8_e8m0fnu
        )
    ), "output dtype should be uint8 for NVFP4/MXFP4 or float8_e4m3fn for MXFP8"

    comptime N = input_layout.shape[1].value()

    comptime if SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE:
        comptime assert N % SF_VECTOR_SIZE == 0, "N must be a multiple of 32"
    else:
        comptime assert (
            N % (SF_VECTOR_SIZE // 2) == 0
        ), "N must be a multiple of 8"

    comptime ELEMENTS_PER_THREAD = 8
    comptime num_SMs = B200.sm_count

    var num_rows = input.dim(0)
    if num_rows == 0 or num_cols == 0:
        return
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)

    var block_dim = (
        min(num_cols // ELEMENTS_PER_THREAD, num_max_threads),
        1,
        1,
    )
    var num_blocks_per_SM = max(
        1, B200.threads_per_multiprocessor // block_dim[0]
    )
    var grid_dim = (min(num_rows_padded, num_SMs * num_blocks_per_SM), 1, 1)

    comptime kernel = quantize_dynamic_scaled_fp4fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        output_layout,
        scales_layout,
        input_layout,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        num_max_threads=num_max_threads,
    ]

    ctx.enqueue_function[kernel](
        output,
        scales,
        input,
        num_cols,
        num_cols_padded,
        tensor_sf,
        block_dim=block_dim,
        grid_dim=grid_dim,
        attributes=pdl_launch_attributes(PDLLevel(1)),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
def quantize_dynamic_scaled_fp4fp8_kernel[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    input_layout: Layout,
    *,
    SF_VECTOR_SIZE: Int = 16,
    ELEMENTS_PER_THREAD: Int = 8,
    num_max_threads: Int = 512,
](
    output: LayoutTensor[out_dtype, output_layout, MutAnyOrigin],
    scales: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    input: LayoutTensor[in_dtype, input_layout, ImmutAnyOrigin],
    num_cols: Int,
    num_cols_padded: Int,
    tensor_sf: Float32,
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
    comptime OUTPUT_WIDTH = 4 if out_dtype == DType.uint8 else 8

    comptime assert (
        input.shape[1]() % ELEMENTS_PER_THREAD == 0
    ), "num_cols must be a multiple of ELEMENTS_PER_THREAD (8 for NVFP4/MXFP8)"

    var num_rows = input.dim(0)
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)
    var num_sf_cols = align_up(num_cols_padded, SF_VECTOR_SIZE * SF_ATOM_K)

    var num_col_threads = num_cols // ELEMENTS_PER_THREAD
    var num_padded_col_threads = num_cols_padded // ELEMENTS_PER_THREAD
    var num_sf_threads = num_sf_cols // ELEMENTS_PER_THREAD

    with PDL():
        for global_row_idx in range(block_idx.x, num_rows_padded, grid_dim.x):
            var is_padded_row = global_row_idx >= num_rows

            for col_idx in range(thread_idx.x, num_sf_threads, block_dim.x):
                var global_col_idx = col_idx * ELEMENTS_PER_THREAD

                if is_padded_row:
                    # This row is entirely padding, so zero out scale factors.
                    # Note: Padding rows do NOT exist in the output tensor (which is sized [num_rows, K]),
                    # they only exist in the scale factor tensor. Tensor cores expects these scale factors to be 0.
                    # there will be accuracy issues if we don't zero out the scale factors for padding rows.
                    if global_col_idx % SF_VECTOR_SIZE == 0:
                        set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                            scales,
                            global_row_idx,
                            global_col_idx,
                            Scalar[scales_dtype](0.0),
                        )

                else:
                    # this is only needed if we do padding in the output tensor N dimension
                    if (
                        col_idx >= num_col_threads
                        and col_idx < num_padded_col_threads
                    ):
                        output.store[width=OUTPUT_WIDTH](
                            global_row_idx,
                            col_idx * OUTPUT_WIDTH,
                            SIMD[out_dtype, OUTPUT_WIDTH](0),
                        )

                    if col_idx >= num_col_threads:
                        if global_col_idx % SF_VECTOR_SIZE == 0:
                            set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                                scales,
                                global_row_idx,
                                global_col_idx,
                                Scalar[scales_dtype](0.0),
                            )

                    # This row contains actual data
                    else:
                        var input_vector = input.load[ELEMENTS_PER_THREAD](
                            global_row_idx, global_col_idx
                        )

                        # each thread finds maximum value in its local 8 elements
                        var thread_max = abs(input_vector).reduce_max()
                        # find the maximum value among all 16 elements (two threads for 16)
                        thread_max = max(shuffle_xor(thread_max, 1), thread_max)

                        comptime if NUM_THREADS_PER_SF == 4:
                            thread_max = max(
                                shuffle_xor(thread_max, 2), thread_max
                            )

                        var group_max = thread_max.cast[DType.float32]()

                        # get the scale factor for these 16/32 elements by dividing it by the maximum value of fp4-e2m1/fp8-e4m3
                        var scale_factor: Float32
                        scale_factor = tensor_sf * (
                            group_max * recip(Float32(6.0))
                        ) if out_dtype == DType.uint8 else (
                            group_max * recip(Float32(448.0))
                        )

                        # NOTE: NVFP4 uses FP8-UE4M3 format for the scale factor but we know that scale_factor is always positive, so we can use E4M3 instead of UE4M3.
                        var fp8_scale_factor = scale_factor.cast[scales_dtype]()

                        # find the quantization scale factor for these 16 elements (scale_factor = scale_factor / tensor_sf)
                        # we divide input by this scale factor which is same as multiplying by the reciprocal of the scale factor
                        var output_scale = Float32(0.0)
                        if group_max != 0:
                            output_scale = recip(
                                fp8_scale_factor.cast[DType.float32]()
                                * recip(tensor_sf)
                            ) if out_dtype == DType.uint8 else (
                                recip(fp8_scale_factor.cast[DType.float32]())
                            )

                        # write back the scale factor
                        if global_col_idx % SF_VECTOR_SIZE == 0:
                            set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                                scales,
                                global_row_idx,
                                global_col_idx,
                                fp8_scale_factor,
                            )

                        var input_f32 = (
                            input_vector.cast[DType.float32]() * output_scale
                        )

                        var output_vector: SIMD[out_dtype, OUTPUT_WIDTH]

                        comptime if out_dtype == DType.uint8:
                            output_vector = bitcast[out_dtype, OUTPUT_WIDTH](
                                cast_fp32_to_fp4e2m1(input_f32)
                            )
                        else:
                            output_vector = rebind[
                                SIMD[out_dtype, OUTPUT_WIDTH]
                            ](input_f32.cast[out_dtype]())

                        output.store[width=OUTPUT_WIDTH](
                            global_row_idx,
                            col_idx * OUTPUT_WIDTH,
                            output_vector,
                        )


@always_inline
def block_scales_interleave_fp4[
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    ctx: DeviceContext,
    input_scales_tile: TileTensor[
        mut=False, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    output_scales_tile: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
) raises:
    var input_scales = input_scales_tile.to_layout_tensor()
    var output_scales = output_scales_tile.to_layout_tensor()
    comptime input_scales_layout = input_scales.layout
    comptime output_scales_layout = output_scales.layout
    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
        MXFP4_SF_DTYPE,
    ), "scales dtype should be float8_e4m3fn (NVFP4) or float8_e8m0fnu (MXFP4)"

    comptime num_SMs = B200.sm_count

    var num_rows = input_scales.dim(0)
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)
    var num_cols = input_scales.dim(1)
    var num_col_padded = align_up(num_cols, SF_ATOM_K)

    # each thread handle just one scale factor for SF_VECTOR_SIZE of elements
    var block_dim = (min(num_col_padded, num_max_threads), 1, 1)
    var num_blocks_per_SM = max(
        1, 2 * B200.threads_per_multiprocessor // block_dim[0]
    )
    var grid_dim = (min(num_rows_padded, num_SMs * num_blocks_per_SM), 1, 1)

    comptime kernel = block_scales_interleave_fp4_kernel[
        scales_dtype,
        input_scales_layout,
        output_scales_layout,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        num_max_threads=num_max_threads,
    ]

    ctx.enqueue_function[kernel](
        input_scales,
        output_scales,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
@__name(
    t"block_scales_interleave_fp4_{scales_dtype}_{SF_VECTOR_SIZE}", mangle=True
)
def block_scales_interleave_fp4_kernel[
    scales_dtype: DType,
    input_scales_layout: Layout,
    output_scales_layout: Layout,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    input_scales: LayoutTensor[
        scales_dtype, input_scales_layout, ImmutAnyOrigin
    ],
    output_scales: LayoutTensor[
        scales_dtype, output_scales_layout, MutAnyOrigin
    ],
):
    var num_rows = input_scales.dim(0)
    var num_rows_padded = align_up(num_rows, SF_MN_GROUP_SIZE)
    var num_cols = input_scales.dim(1)
    var num_col_padded = align_up(num_cols, SF_ATOM_K)

    for row_idx in range(block_idx.x, num_rows_padded, grid_dim.x):
        for col_idx in range(thread_idx.x, num_col_padded, block_dim.x):
            var scale_factor = Scalar[scales_dtype](0.0)
            if row_idx < num_rows and col_idx < num_cols:
                scale_factor = rebind[Scalar[scales_dtype]](
                    input_scales[row_idx, col_idx]
                )

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
    accum_type: DType = DType.float32,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    BLOCK_DIM: Int = 16,
](
    c: LayoutTensor[c_type, address_space=AddressSpace.GENERIC, ...],
    a: LayoutTensor[a_type, address_space=AddressSpace.GENERIC, ...],
    b: LayoutTensor[b_type, address_space=AddressSpace.GENERIC, ...],
    a_scales: LayoutTensor[
        a_scales_type, address_space=AddressSpace.GENERIC, ...
    ],
    b_scales: LayoutTensor[
        b_scales_type, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
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
            and a_type == DType.uint8
            and a_scales_type == NVFP4_SF_DTYPE
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF4
            and a_type == DType.uint8
            and a_scales_type == MXFP4_SF_DTYPE
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF8F6F4
            and a_type == DType.float8_e4m3fn
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

    var M = c.dim(0)
    var N = c.dim(1)
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )
    var K = a.dim(1) * 2 if is_fp4 else a.dim(1)

    if M == 0 or N == 0 or K == 0:
        return

    if (
        a_scales.dim(0) != ceildiv(M, SF_MN_GROUP_SIZE)
        or b_scales.dim(0) != ceildiv(N, SF_MN_GROUP_SIZE)
        or a_scales.dim(1) != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        or b_scales.dim(1) != ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
        or (a_scales.dim(2) != b_scales.dim(2) != SF_ATOM_M[0])
        or (a_scales.dim(3) != b_scales.dim(3) != SF_ATOM_M[1])
        or (a_scales.dim(4) != b_scales.dim(4) != SF_ATOM_K)
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
        type_of(a).layout,
        type_of(b).layout,
        type_of(c).layout,
        type_of(a_scales).layout,
        type_of(b_scales).layout,
        scaling_kind=scaling_kind,
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        alpha,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


@__name(t"naive_block_scaled_matmul", mangle=True)
def naive_block_scaled_matmul_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    scaling_kind: UMMAKind,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, MutAnyOrigin],
    alpha: Float32,
):
    # Note: This is a naive kernel that emulates a block scaled matmul with TCGEN scale factors.
    # Assumptions:
    # 1. both A and B should be in K-major format
    # 2. both a_scales and b_scales should be in TCGEN scale factors layout (5D tensors)

    var M = c.dim(0)
    var N = c.dim(1)
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )
    comptime K_STEPS = 2 if is_fp4 else 1
    var K = a.dim(1) * K_STEPS

    var row_idx = global_idx.x
    var col_idx = global_idx.y

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
                var a_val = rebind[Scalar[accum_type]](a_val_fp16x2[k_idx])
                var b_val = rebind[Scalar[accum_type]](b_val_fp16x2[k_idx])
                var a_scale_val = abs(
                    rebind[Scalar[accum_type]](a_scale.cast[accum_type]())
                )
                var b_scale_val = abs(
                    rebind[Scalar[accum_type]](b_scale.cast[accum_type]())
                )
                accum += a_val * b_val * a_scale_val * b_scale_val
        else:
            # MXFP8: one float8 value per byte.
            var a_val = rebind[Scalar[a_type]](a[row_idx, k // K_STEPS]).cast[
                accum_type
            ]()
            var b_val = rebind[Scalar[b_type]](b[col_idx, k // K_STEPS]).cast[
                accum_type
            ]()
            var a_scale_val = abs(
                rebind[Scalar[accum_type]](a_scale.cast[accum_type]())
            )
            var b_scale_val = abs(
                rebind[Scalar[accum_type]](b_scale.cast[accum_type]())
            )
            accum += a_val * b_val * a_scale_val * b_scale_val

    accum *= alpha.cast[accum_type]()

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(row_idx, col_idx), accum.cast[c_type]()
        )
    else:
        c[row_idx, col_idx] = accum.cast[c_type]()


@__llvm_arg_metadata(input_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(output_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(scales_tma_op, `nvvm.grid_constant`)
@__name("quantize_dynamic_scaled_async_fp4_kernel", mangle=True)
def quantize_dynamic_scaled_async_fp4_kernel[
    input_dtype: DType,
    input_tile_rank: Int,
    input_tile_shape: IndexList[input_tile_rank],
    input_desc_shape: IndexList[input_tile_rank],
    output_dtype: DType,
    output_tile_rank: Int,
    output_tile_shape: IndexList[output_tile_rank],
    output_desc_shape: IndexList[output_tile_rank],
    scales_dtype: DType,
    scales_tile_rank: Int,
    scales_tile_shape: IndexList[scales_tile_rank],
    scales_desc_shape: IndexList[scales_tile_rank],
    input_swizzle_mode: TensorMapSwizzle,
    output_swizzle_mode: TensorMapSwizzle,
    scales_swizzle_mode: TensorMapSwizzle,
    SF_VECTOR_SIZE: Int,
    NUM_PIPELINES_STAGES: Int,
](
    input_tma_op: TMATensorTile[
        input_dtype, input_tile_rank, input_tile_shape, input_desc_shape
    ],
    output_tma_op: TMATensorTile[
        output_dtype, output_tile_rank, output_tile_shape, output_desc_shape
    ],
    scales_tma_op: TMATensorTile[
        scales_dtype, scales_tile_rank, scales_tile_shape, scales_desc_shape
    ],
    tensor_sf: Float32,  # tensor-wise scale factor
):
    var smem_storage = rebind[
        UnsafePointer[
            Scalar[input_dtype],
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
    ](
        external_memory[
            Scalar[input_dtype],
            address_space=AddressSpace.SHARED,
            alignment=128,
        ]()
    )

    comptime input_smem_tile_size = _idx_product[
        input_tile_rank, input_tile_shape
    ]() * NUM_PIPELINES_STAGES
    comptime output_smem_tile_size = _idx_product[
        output_tile_rank, output_tile_shape
    ]()
    comptime scales_smem_tile_size = _idx_product[
        scales_tile_rank, scales_tile_shape
    ]()

    comptime SF_K_GROUP_SIZE: Int = SF_VECTOR_SIZE * Int(SF_ATOM_K)
    comptime STAGE_GROUP_SIZE = SF_K_GROUP_SIZE // NUM_PIPELINES_STAGES

    comptime assert (
        STAGE_GROUP_SIZE == 64
        and NUM_PIPELINES_STAGES == 1
        and SF_VECTOR_SIZE == Int(NVFP4_SF_VECTOR_SIZE)
    ), (
        "STAGE_GROUP_SIZE must be 64 and NUM_PIPELINES_STAGES must be 1 and"
        " SF_VECTOR_SIZE must be 16"
    )
    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "scales_dtype must be float8_e4m3fn"

    var input_smem_ptr = smem_storage
    var output_smem_ptr = (smem_storage + input_smem_tile_size).bitcast[
        Scalar[output_dtype]
    ]()
    var scales_smem_ptr = (output_smem_ptr + output_smem_tile_size).bitcast[
        Scalar[scales_dtype]
    ]()
    var mbar_ptr = scales_smem_ptr + scales_smem_tile_size

    var input_smem = LayoutTensorIter[
        input_dtype,
        Layout.row_major(input_tile_shape),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        input_smem_ptr,
        input_smem_tile_size,
    )

    var output_smem = LayoutTensor[
        output_dtype,
        Layout.row_major(output_tile_shape),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        output_smem_ptr,
    )

    var scales_smem = LayoutTensor[
        scales_dtype,
        Layout.row_major(scales_tile_shape),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
        alignment=128,
    ](
        scales_smem_ptr,
    )

    var tma_mbar = mbar_ptr.bitcast[SharedMemBarrier]()

    var local_row_idx = thread_idx.x

    if thread_idx.x == 0:
        comptime for idx in range(NUM_PIPELINES_STAGES):
            tma_mbar[idx].init()

    var tma_phase = SIMD[DType.uint32, NUM_PIPELINES_STAGES](0)

    barrier()

    comptime expected_bytes = _idx_product[
        input_tile_rank, input_tile_shape
    ]() * size_of[input_dtype]()

    with PDL():
        if thread_idx.x >= 128:
            warpgroup_reg_dealloc[24]()

            comptime for iter_idx in range(NUM_PIPELINES_STAGES):
                var smem_tile = input_smem.next(iter_idx)[]

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
                var smem_tile = input_smem.next(iter_idx)[]

                tma_mbar[iter_idx].wait(tma_phase[iter_idx])
                var quantized_elements = SIMD[DType.uint32, 8]()

                comptime for group_idx in range(
                    STAGE_GROUP_SIZE // SF_VECTOR_SIZE
                ):
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
                        var temp = smem_tile.ptr.load[
                            width=8,
                            alignment=align_of[SIMD[input_dtype, 8]](),
                        ](swizzle_idx)

                        group_elements = group_elements.insert[
                            offset=col_idx * 8
                        ](temp)

                    var group_max = (
                        abs(group_elements).reduce_max().cast[DType.float32]()
                    )

                    var scale_factor = (
                        tensor_sf * group_max * recip(Float32(6.0))
                    )
                    var fp8_scale_factor = scale_factor.cast[scales_dtype]()

                    scale_factors[
                        iter_idx * NUM_PIPELINES_STAGES + group_idx
                    ] = fp8_scale_factor

                    var output_scale = Float32(0.0)
                    if fp8_scale_factor.cast[DType.float32]() != 0:
                        output_scale = recip(
                            fp8_scale_factor.cast[DType.float32]()
                            * recip(tensor_sf)
                        )

                    comptime for slice_idx in range(2):
                        var slice_elements = group_elements.slice[
                            8, offset=slice_idx * 8
                        ]()
                        quantized_elements[
                            group_idx * 2 + slice_idx
                        ] = cast_fp32_to_fp4e2m1(
                            slice_elements.cast[DType.float32]() * output_scale
                        )

                comptime for idx in range(2):
                    var slice_elements = quantized_elements.slice[
                        4, offset=idx * 4
                    ]()
                    comptime output_swizzle = make_swizzle[
                        output_dtype, output_swizzle_mode
                    ]()
                    var swizzle_offset = (
                        local_row_idx * ufloordiv(STAGE_GROUP_SIZE, 2)
                        + idx * SF_VECTOR_SIZE
                    )
                    var output_swizzle_idx = output_swizzle(swizzle_offset)
                    output_smem.ptr.store[
                        alignment=align_of[SIMD[output_dtype, SF_VECTOR_SIZE]]()
                    ](
                        output_swizzle_idx,
                        bitcast[output_dtype, SF_VECTOR_SIZE](slice_elements),
                    )

                scales_smem.ptr.store[
                    alignment=align_of[SIMD[scales_dtype, SF_ATOM_K]]()
                ](
                    (local_row_idx % 32) * 16
                    + (local_row_idx // 32) * SF_ATOM_K,
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
                        UInt32(ufloordiv(block_idx.y * SF_K_GROUP_SIZE, 2)),
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
        mut=True, output_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_tensor_tile: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_tensor_tile: TileTensor[
        mut=False, input_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
) raises:
    var output_tensor = output_tensor_tile.to_layout_tensor()
    var scales_tensor = scales_tensor_tile.to_layout_tensor()
    var input_tensor = input_tensor_tile.to_layout_tensor()
    comptime output_layout = output_tensor.layout
    comptime scales_layout = scales_tensor.layout
    comptime input_layout = input_tensor.layout
    comptime assert (
        input_dtype == DType.bfloat16
    ), "input_dtype must be bfloat16"

    comptime assert (
        output_dtype == DType.uint8
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        and scales_dtype == NVFP4_SF_DTYPE
    ), (
        "output dtype should be uint8 (fp4-e2m1fnX2) for NVFP4 and scales_dtype"
        " must be float8_e4m3fn"
    )

    comptime input_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_32B  # 64 elements / 2 elements per uint8 = 32 elements per 32B
    comptime scales_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE  # 16 elements / 1 elements per float8_e4m3fn = 16 elements per 16B

    var M = input_tensor.dim(0)
    var N = input_tensor.dim(1)

    comptime output_N = output_layout.shape[1].value()
    comptime assert (
        output_N % 32 == 0
    ), "output_tensor N must be a multiple of 32"
    comptime input_N = input_layout.shape[1].value()
    comptime assert (
        input_N // output_N == 2
    ), "input_tensor N must be a multiple of 2 * output_tensor N"

    comptime SF_K_GROUP_SIZE = SF_VECTOR_SIZE * SF_ATOM_K
    comptime NUM_PIPELINES_STAGES = 1

    comptime input_tma_tile_shape = Index(128, SF_K_GROUP_SIZE)
    var input_tma_op = create_tensor_tile[
        input_tma_tile_shape,
        swizzle_mode=input_swizzle_mode,
    ](ctx, input_tensor)

    comptime output_tma_tile_shape = Index(128, 32)
    var output_tma_op = create_tensor_tile[
        output_tma_tile_shape,
        swizzle_mode=output_swizzle_mode,
    ](ctx, output_tensor)

    comptime assert scales_tensor.rank == 5, "scales must be 5D tensors"

    comptime assert scales_layout.shape[2].value() == SF_ATOM_M[0], ""
    comptime assert scales_layout.shape[3].value() == SF_ATOM_M[1], ""
    comptime assert scales_layout.shape[4].value() == SF_ATOM_K, ""

    comptime scales_4d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var scales_4d_tensor = LayoutTensor[
        scales_dtype, scales_4d_layout[scales_layout], MutAnyOrigin
    ](
        scales_tensor.ptr,
        RuntimeLayout[scales_4d_layout[scales_layout]].row_major(
            IndexList[4](
                scales_tensor.dim(0),
                scales_tensor.dim(1),
                scales_tensor.dim(2),
                scales_tensor.dim(3) * scales_tensor.dim(4),
            ),
        ),
    )

    comptime scales_tma_tile_shape = Index(
        1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    var scales_tma_op = create_tensor_tile[
        scales_tma_tile_shape,
        swizzle_mode=scales_swizzle_mode,
    ](ctx, scales_4d_tensor)

    comptime smem_use = (
        input_tma_tile_shape[0]
        * input_tma_tile_shape[1]
        * size_of[input_dtype]()
        * NUM_PIPELINES_STAGES
    ) + (
        output_tma_tile_shape[0]
        * output_tma_tile_shape[1]
        * size_of[output_dtype]()
    ) + (
        size_of[SharedMemBarrier]() * NUM_PIPELINES_STAGES
    ) + (
        SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K * size_of[scales_dtype]()
    )

    comptime kernel = quantize_dynamic_scaled_async_fp4_kernel[
        type_of(input_tma_op).dtype,
        type_of(input_tma_op).rank,
        type_of(input_tma_op).tile_shape,
        type_of(input_tma_op).desc_shape,
        type_of(output_tma_op).dtype,
        type_of(output_tma_op).rank,
        type_of(output_tma_op).tile_shape,
        type_of(output_tma_op).desc_shape,
        type_of(scales_tma_op).dtype,
        type_of(scales_tma_op).rank,
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
        attributes=pdl_launch_attributes(PDLLevel(1)),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__llvm_arg_metadata(scales_tma_op, `nvvm.grid_constant`)
@__name("grouped_quantize_dynamic_scaled_async_fp4_kernel", mangle=True)
def grouped_quantize_dynamic_scaled_fp4_async_kernel[
    output_dtype: DType,
    scales_dtype: DType,
    input_dtype: DType,
    scales_tile_rank: Int,
    scales_tile_shape: IndexList[scales_tile_rank],
    scales_desc_shape: IndexList[scales_tile_rank],
    scales_swizzle_mode: TensorMapSwizzle,
    output_layout: TensorLayout,
    input_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    scales_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    sf_layout: TensorLayout,
    num_threads: Int = 128,
](
    output_tensor: TileTensor[output_dtype, output_layout, MutAnyOrigin],
    scales_tma_op: TMATensorTile[
        scales_dtype, scales_tile_rank, scales_tile_shape, scales_desc_shape
    ],
    input_tensor: TileTensor[input_dtype, input_layout, ImmutAnyOrigin],
    row_offsets: TileTensor[DType.uint32, row_offsets_layout, ImmutAnyOrigin],
    scales_offsets: TileTensor[
        DType.uint32, scales_offsets_layout, ImmutAnyOrigin
    ],
    expert_ids: TileTensor[DType.int32, expert_ids_layout, ImmutAnyOrigin],
    sf_tensor: TileTensor[
        DType.float32, sf_layout, ImmutAnyOrigin
    ],  # tensor-wise scale factor
):
    comptime assert row_offsets.flat_rank == 1, "row_offsets must be rank 1"
    comptime assert (
        scales_offsets.flat_rank == 1
    ), "scales_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert sf_tensor.flat_rank == 1, "sf_tensor must be rank 1"
    comptime assert scales_offsets_layout.all_dims_known
    comptime num_experts = scales_offsets_layout.static_shape[0]

    comptime SF_VECTOR_SIZE = 16 if scales_dtype == NVFP4_SF_DTYPE else 32
    comptime SF_K_GROUP_SIZE = SF_VECTOR_SIZE * SF_ATOM_K

    # Each block process a SF_MN_GROUP_SIZE x SF_K_GROUP_SIZE tile
    var scale_tile_idx = block_idx.x
    var k_idx = block_idx.y

    # Finds expert id for current block through bisect search
    # row_offsets is of size num_experts + 1, scales_offsets is of size num_experts
    # For a given expert_idx, it's corresponding scales tiles start at:
    # row_offsets[expert_idx] // SF_MN_GROUP_SIZE + scales_offsets[expert_idx]
    # and there will be ceildiv(row_offsets[expert_idx + 1] -
    # row_offsets[expert_idx], SF_MN_GROUP_SIZE) scales tiles for this expert.
    var low = 0
    var high = num_experts
    while low + 1 != high:
        var mid = ufloordiv(low + high, 2)
        var mid_start = ufloordiv(
            Int(row_offsets[mid]), SF_MN_GROUP_SIZE
        ) + Int(scales_offsets[mid])
        if scale_tile_idx >= mid_start:
            low = mid
        else:
            high = mid
    var expert_idx = low

    # Tail tiles beyond last expert's range: exit early.
    var curr_expert_start = Int(row_offsets[expert_idx])
    var curr_expert_end = Int(row_offsets[expert_idx + 1])
    var expert_tiles_start = ufloordiv(
        curr_expert_start, SF_MN_GROUP_SIZE
    ) + Int(scales_offsets[expert_idx])
    var expert_num_tiles = uceildiv(
        curr_expert_end - curr_expert_start, SF_MN_GROUP_SIZE
    )

    var expert_id = expert_ids[expert_idx]
    var input_sf = sf_tensor[expert_id]

    var token_start = (
        curr_expert_start
        + (scale_tile_idx - expert_tiles_start) * SF_MN_GROUP_SIZE
    )
    var num_tokens = min(curr_expert_end - token_start, SF_MN_GROUP_SIZE)

    comptime scales_smem_tile_size = align_up(
        Int(Coord(scales_tile_shape).product()), 128
    )
    var smem_ptr = stack_allocation[
        scales_smem_tile_size,
        Scalar[scales_dtype],
        alignment=128,
        address_space=AddressSpace.SHARED,
    ]()

    var scales_smem = TileTensor(
        smem_ptr,
        row_major(Coord(scales_tile_shape)),
    )

    # We can safetly prefetch the row_offsets and scales_offsets before
    # `wait_on_dependent_grids()`.
    with PDL():
        if scale_tile_idx >= expert_tiles_start + expert_num_tiles:
            return

        comptime ELEMENTS_PER_THREAD = 8
        comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
        comptime OUTPUT_WIDTH = 4 if output_dtype == DType.uint8 else 8
        comptime num_threads_per_row = SF_K_GROUP_SIZE // ELEMENTS_PER_THREAD
        comptime rows_per_iter = num_threads // num_threads_per_row
        comptime num_iters = SF_MN_GROUP_SIZE // rows_per_iter

        var row_in_iter, col_thread_idx = udivmod(
            thread_idx.x, num_threads_per_row
        )
        var input_col = (
            k_idx * SF_K_GROUP_SIZE + col_thread_idx * ELEMENTS_PER_THREAD
        )
        var output_col = (
            ufloordiv(input_col, 2) if output_dtype
            == DType.uint8 else input_col
        )

        comptime for iter_idx in range(num_iters):
            var local_row = iter_idx * rows_per_iter + row_in_iter
            var is_valid = local_row < num_tokens

            var input_vector = SIMD[input_dtype, ELEMENTS_PER_THREAD](0)
            if is_valid:
                var global_row = token_start + local_row
                input_vector = input_tensor.load[width=ELEMENTS_PER_THREAD](
                    (Idx(global_row), Idx(input_col))
                )

            var thread_max = abs(input_vector).reduce_max()
            thread_max = max(shuffle_xor(thread_max, 1), thread_max)
            comptime if NUM_THREADS_PER_SF == 4:
                thread_max = max(shuffle_xor(thread_max, 2), thread_max)
            var group_max = thread_max.cast[DType.float32]()

            var scale_factor: Float32
            comptime if output_dtype == DType.uint8:
                scale_factor = input_sf * group_max * recip(Float32(6.0))
            else:
                scale_factor = group_max * recip(Float32(448.0))
            var fp8_scale_factor = scale_factor.cast[scales_dtype]()

            var output_scale = Float32(0.0)
            if group_max != 0:
                comptime if output_dtype == DType.uint8:
                    output_scale = recip(
                        fp8_scale_factor.cast[DType.float32]() * recip(input_sf)
                    )
                else:
                    output_scale = recip(fp8_scale_factor.cast[DType.float32]())

            if is_valid:
                var global_row = token_start + local_row
                var input_f32 = (
                    input_vector.cast[DType.float32]() * output_scale
                )
                var output_vector: SIMD[output_dtype, OUTPUT_WIDTH]
                comptime if output_dtype == DType.uint8:
                    output_vector = bitcast[output_dtype, OUTPUT_WIDTH](
                        cast_fp32_to_fp4e2m1(input_f32)
                    )
                else:
                    output_vector = rebind[SIMD[output_dtype, OUTPUT_WIDTH]](
                        input_f32.cast[output_dtype]()
                    )
                output_tensor.store(
                    (Idx(global_row), Idx(output_col)), output_vector
                )

            if col_thread_idx % NUM_THREADS_PER_SF == 0:
                var sf_group = col_thread_idx // NUM_THREADS_PER_SF
                var smem_idx = (
                    (local_row % 32) * 16
                    + (local_row // 32) * SF_ATOM_K
                    + sf_group
                )
                scales_smem.ptr.store(smem_idx, fp8_scale_factor)

        barrier()

        if thread_idx.x == 0:
            fence_async_view_proxy()
            scales_tma_op.async_store(
                scales_smem,
                StaticTuple[UInt32, 4](
                    0, 0, UInt32(k_idx), UInt32(scale_tile_idx)
                ),
            )
            scales_tma_op.commit_group()
            scales_tma_op.wait_group[0]()


def grouped_quantize_dynamic_scaled_fp4_async[
    input_dtype: DType,
    output_dtype: DType,
    scales_dtype: DType,
    //,
](
    output_tensor: TileTensor[
        mut=True, output_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_tensor: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_tensor: TileTensor[
        mut=False, input_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    row_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scales_offsets: TileTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    expert_ids: TileTensor[
        mut=False, DType.int32, address_space=AddressSpace.GENERIC, ...
    ],
    sf_tensor: TileTensor[
        mut=False, DType.float32, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    var scales_tensor_lt = scales_tensor.to_layout_tensor()
    comptime scales_lt_layout = scales_tensor_lt.layout

    comptime scales_4d_layout[layout: Layout] = Layout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var scales_4d_tensor = LayoutTensor[
        scales_dtype, scales_4d_layout[scales_lt_layout], MutAnyOrigin
    ](
        scales_tensor.ptr,
        RuntimeLayout[scales_4d_layout[scales_lt_layout]].row_major(
            IndexList[4](
                scales_tensor_lt.dim(0),
                scales_tensor_lt.dim(1),
                scales_tensor_lt.dim(2),
                scales_tensor_lt.dim(3) * scales_tensor_lt.dim(4),
            ),
        ),
    )

    comptime scales_tma_tile_shape = Index(
        1, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    var scales_tma_op = create_tensor_tile[
        scales_tma_tile_shape,
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, scales_4d_tensor)

    comptime kernel = grouped_quantize_dynamic_scaled_fp4_async_kernel[
        output_dtype,
        scales_dtype,
        input_dtype,
        scales_tma_op.rank,
        scales_tma_op.tile_shape,
        scales_tma_op.desc_shape,
        TensorMapSwizzle.SWIZZLE_NONE,
        output_tensor.LayoutType,
        input_tensor.LayoutType,
        row_offsets.LayoutType,
        scales_offsets.LayoutType,
        expert_ids.LayoutType,
        sf_tensor.LayoutType,
    ]

    ctx.enqueue_function[kernel](
        output_tensor,
        scales_tma_op,
        input_tensor,
        row_offsets,
        scales_offsets,
        expert_ids,
        sf_tensor,
        grid_dim=(
            scales_tensor.dim[0](),
            scales_tensor.dim[1](),
            1,
        ),
        block_dim=(128,),
        attributes=pdl_launch_attributes(PDLLevel(1)),
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
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Our sm100 block scaled matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 block scaled matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations. Callers must allocate `c`; when an `elementwise_lambda_fn`
    is supplied the matmul result is written into `c` and then read back
    by the lambda.
    """

    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "Only support NVFP4_SF_DTYPE (float8_e4m3fn) for scales for now."

    comptime assert SF_VECTOR_SIZE in (
        NVFP4_SF_VECTOR_SIZE,
    ), "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4)"

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
    var _k = Int(a.dim[1]()) * 2 if a_type == DType.uint8 else Int(a.dim[1]())
    if m == 0 or n == 0:
        return

    @always_inline
    @parameter
    def description_fn() -> String:
        # fmt: off
        return String(
            "(gpu",
            ";", trace_arg("A", IndexList[2](m, _k), a_type),
            ";", trace_arg("B", IndexList[2](_k, n), b_type),
            ";", trace_arg("C", IndexList[2](m, n), c_type),
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
            String("nvfp4_" if a_type == DType.uint8 else "mxfp8_"),
            String(SF_VECTOR_SIZE) + String("_sfvs"),
        ](),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(ctx),
    ):
        comptime if not elementwise_lambda_fn:
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
        else:
            comptime epilogue = elementwise_lambda_fn.value()
            # Nvidia GPUs >= sm_100 arch support 32B load/store to global memory.
            comptime use_32b_simd = True
            comptime simd_size = 32 // size_of[c_type]() if use_32b_simd else (
                simd_width_of[c_type, target=get_gpu_target()]()
            )

            @parameter
            @__copy_capture(c, n)
            def epilogue_wrapper[
                simd_width: Int, rank: Int, alignment: Int = 1
            ](idx: IndexList[rank]):
                var c_coord = Index(idx[0], idx[1])
                var c_val = rebind[SIMD[c_type, simd_width]](
                    c.raw_load[width=simd_width](idx[0] * n + idx[1])
                )
                epilogue[c_type, simd_width, alignment=alignment](
                    c_coord, c_val
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
            elementwise[epilogue_wrapper, simd_size, target="gpu"](
                Index(m, n), ctx
            )


# ===----------------------------------------------------------------------=== #
# TileTensor primary implementations
# ===----------------------------------------------------------------------=== #


@always_inline
def block_scaled_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    transpose_a: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    pdl_level: PDLLevel = PDLLevel(0),
    _trace_description: StaticString = "",
    target: StaticString = "cpu",
](
    c_device: TileTensor[
        mut=True, c_type, address_space=AddressSpace.GENERIC, ...
    ],
    a_device: TileTensor[
        mut=False, a_type, address_space=AddressSpace.GENERIC, ...
    ],
    b_device: TileTensor[
        mut=False, b_type, address_space=AddressSpace.GENERIC, ...
    ],
    a_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    b_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    comptime assert c_device.rank == 2 and c_device.flat_rank == 2
    comptime assert a_device.rank == 2 and a_device.flat_rank == 2
    comptime assert b_device.rank == 2 and b_device.flat_rank == 2
    comptime assert a_scales_device.rank == 5 and a_scales_device.flat_rank == 5
    comptime assert b_scales_device.rank == 5 and b_scales_device.flat_rank == 5

    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "Only support NVFP4_SF_DTYPE (float8_e4m3fn) for scales for now."

    comptime assert (
        SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    ), "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4)"

    var c = c_device.as_any_origin()
    var a = a_device.as_any_origin()
    var b = b_device.as_any_origin()
    var a_scales = a_scales_device.as_any_origin()
    var b_scales = b_scales_device.as_any_origin()

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
    var k = Int(a.dim[1]()) * 2 if a_type == DType.uint8 else Int(a.dim[1]())

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

    comptime assert (
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None
    ), "Either the epilogue lambda or the compute lambda can be used"

    # vendor block scaled matmul kernels don't support compute lambda, so we wrap it around an epilogue lambda instead.
    @parameter
    @always_inline
    @__copy_capture(c)
    def compute_lambda_wrapper[
        _dtype: DType, _width: Int, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[_dtype, _width]):
        comptime if elementwise_compute_lambda_fn:
            comptime compute_lambda = elementwise_compute_lambda_fn.value()
            var output = compute_lambda(coords, val)
            comptime assert (
                output.dtype == c_type
            ), "compute epilogue lambda output and c type mismatch"
            c.store_linear[alignment=alignment * size_of[c_type]()](
                coords, rebind[SIMD[c_type, _width]](output)
            )

    comptime elementwise_lambda_wrapper = Optional[elementwise_epilogue_type](
        compute_lambda_wrapper
    ) if elementwise_compute_lambda_fn else elementwise_lambda_fn

    comptime static_N = c_device.static_shape[1]
    comptime static_K = a_device.static_shape[1] * (
        2 if a_type == DType.uint8 else 1
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

        comptime umma_shape = Index(BM * CTA_GROUP, BN * CTA_GROUP, MMA_K)

        comptime config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=get_scaling_kind[
                a_type, scales_dtype, SF_VECTOR_SIZE
            ](),
            mma_shape=umma_shape,
            cluster_shape=CLUSTER_DIM,
            block_swizzle_size=BLOCK_SWIZZLE_SIZE,
            raster_order=RasterOrder(Int32(RASTERIZE_ORDER)),
            cta_group=CTA_GROUP,
            AB_swapped=AB_SWAPPED,
            k_group_size=K_GROUP_SIZE,
        )

        return blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            K=a_device.static_shape[1],
            config=config,
        ](c, a, b, a_scales, b_scales, ctx)

    comptime if get_defined_bool[
        "ENABLE_EXPERIMENTAL_SM100_BLOCK_SCALED_MATMUL", False
    ]():
        var status = heuristic_and_outliers_dispatch[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            pdl_level=pdl_level,
        ](c_device, a_device, b_device, a_scales, b_scales, tensor_sf, ctx)

        if status == DISPATCH_HIT:
            logger.info("Executing SM100 Block Scaled matmul kernel")
            return
        else:
            raise Error("Heuristic and outliers dispatch failed")

    @always_inline
    @parameter
    def description_fn() -> String:
        # fmt: off
        return String(
            "(",
            target,
            ";", trace_arg("A", IndexList[2](m, k), a_type),
            ";", trace_arg("B", IndexList[2](k, n), b_type),
            ";", trace_arg("C", IndexList[2](m, n), c_type),
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
            String("nvfp4_" if a_type == DType.uint8 else "mxfp8_"),
            String(SF_VECTOR_SIZE) + String("_sfvs"),
            _trace_description if _trace_description else "",
        ](),
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(ctx),
    ):
        # For these large-N shapes on B200, Mojo also wins at M=256.
        comptime is_widened_shape = (
            static_K == 7168 and static_N in (18432, 36864)
        )
        comptime mojo_m_cap = 256 if is_widened_shape else 128
        if m <= mojo_m_cap:
            var status = heuristic_and_outliers_dispatch[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                pdl_level=pdl_level,
            ](
                c_device,
                a_device,
                b_device,
                a_scales,
                b_scales,
                tensor_sf,
                ctx,
            )
            if status == DISPATCH_HIT:
                return

        # vendor matmul only supports epilogue lambda, so we wrap it around an epilogue lambda instead.
        block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_wrapper,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
        )


@always_inline
def quantize_dynamic_block_scaled[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_device: TileTensor[
        mut=True, out_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_device: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_device: TileTensor[
        mut=False, in_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    tensor_sf: Float32,  # tensor-wise scale factor
    ctx: DeviceContext,
) raises:
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

    var input_tensor = input_device.as_any_origin()
    var output_tensor = output_device.as_any_origin()
    var scales_tensor = scales_device.as_any_origin()

    var num_rows = input_tensor.dim(0)
    var num_cols = input_tensor.dim(1)
    if num_rows == 0 or num_cols == 0:
        return

    comptime static_input_N = input_tensor.static_shape[1]
    comptime static_output_N = output_tensor.static_shape[1]
    comptime is_nvfp4 = out_dtype == DType.uint8 and scales_dtype == NVFP4_SF_DTYPE and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    comptime is_mxfp4 = out_dtype == DType.uint8 and scales_dtype == MXFP4_SF_DTYPE and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
    comptime is_fp8 = out_dtype == DType.float8_e4m3fn and scales_dtype == MXFP8_SF_DTYPE and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    comptime assert is_nvfp4 or is_mxfp4 or is_fp8, "invalid scaling kind"

    comptime is_packed_fp4 = is_nvfp4 or is_mxfp4
    comptime if is_packed_fp4:
        comptime assert static_output_N == static_input_N // 2, (
            "output.dim(1) must be equal to input.dim(1) // 2 (each output"
            " element (uint8) is 2 fp4-e2m1fn values)"
        )

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        # NVIDIA SM100 path: rank-5 interleaved scales.
        comptime assert (
            scales_device.rank == 5 and scales_device.flat_rank == 5
        ), "SM100 requires rank-5 interleaved scales"

        comptime if is_nvfp4 and static_input_N % 32 == 0:
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
    elif is_mxfp4 and ctx.default_device_info == MI355X:
        # AMD CDNA4 MXFP4 path: rank-2 scales.
        comptime assert (
            scales_device.rank == 2 and scales_device.flat_rank == 2
        ), "MXFP4 requires rank-2 scales on CDNA4"
        comptime assert (
            static_input_N % MXFP4_SF_VECTOR_SIZE == 0
        ), "input.dim(1) must be a multiple of 32 (MXFP4_SF_VECTOR_SIZE)"
        quantize_mxfp4_amd(ctx, output_tensor, scales_tensor, input_tensor)
    else:
        comptime assert False, (
            "Unsupported hardware/format combination for block-scaled"
            " quantization"
        )


@always_inline
def block_scales_interleave[
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_scales_device: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_scales_device: TileTensor[
        mut=False, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    comptime assert (
        output_scales_device.rank == 5 and output_scales_device.flat_rank == 5
    )
    comptime assert (
        input_scales_device.rank == 2 and input_scales_device.flat_rank == 2
    )

    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
        MXFP4_SF_DTYPE,
    ), "scales dtype should be float8_e4m3fn (NVFP4) or float8_e8m0fnu (MXFP4)."

    var output = output_scales_device.as_any_origin()
    var input = input_scales_device.as_any_origin()

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE,](
        ctx, input, output
    )


########################################################
# AMD MXFP4 quantization (CDNA4 / MI355X)
########################################################


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
@__name("quantize_mxfp4_amd_kernel", mangle=True)
def _quantize_mxfp4_amd_kernel[
    output_layout: TensorLayout,
    scales_layout: TensorLayout,
    input_layout: TensorLayout,
    *,
    ELEMENTS_PER_THREAD: Int = 8,
    SF_VECTOR_SIZE: Int = MXFP4_SF_VECTOR_SIZE,  # 32
    num_max_threads: Int = 512,
](
    output: TileTensor[DType.uint8, output_layout, MutAnyOrigin],
    scales: TileTensor[DType.float8_e8m0fnu, scales_layout, MutAnyOrigin],
    input: TileTensor[DType.bfloat16, input_layout, MutAnyOrigin],
    num_rows: Int,
    num_cols: Int,
):
    """AMD MXFP4 quantization kernel: BF16 -> packed uint8 + 2D E8M0 scales.

    Each thread processes ELEMENTS_PER_THREAD (8) BF16 values. Four threads
    cooperate to compute one E8M0 block scale over SF_VECTOR_SIZE (32)
    elements via warp shuffle.

    Uses V_CVT_SCALEF32_PK_FP4_BF16 (CDNA4+) for hardware FP4 packing.
    """
    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD
    comptime assert (
        NUM_THREADS_PER_SF == 4
    ), "MXFP4 requires 4 threads per scale factor group"

    var num_col_threads = num_cols // ELEMENTS_PER_THREAD

    for global_row_idx in range(block_idx.x, num_rows, grid_dim.x):
        for col_thread_idx in range(thread_idx.x, num_col_threads, block_dim.x):
            var global_col_idx = col_thread_idx * ELEMENTS_PER_THREAD

            # 1. Load 8x BF16.
            var input_vector = input.load[ELEMENTS_PER_THREAD](
                Coord(Idx(global_row_idx), Idx(global_col_idx))
            )

            # 2. Find per-thread max absolute value.
            var thread_max = abs(input_vector).reduce_max()

            # 3. Reduce across 4 threads to get 32-element block max.
            thread_max = lane_group_max[num_lanes=NUM_THREADS_PER_SF](
                thread_max
            )

            var group_max = thread_max.cast[DType.float32]()

            # 4. Derive E8M0 scale with MXFP4 even-mode rounding.
            var e8m0_scale = compute_mxfp4_even_scale(group_max)
            var scale_f32 = e8m0_scale.cast[DType.float32]()

            # 5. Pack 8 BF16 -> 8 FP4 nibbles using AMD hardware intrinsic.
            var packed = cast_float_to_fp4e2m1_amd(
                rebind[SIMD[DType.bfloat16, 8]](input_vector), scale_f32
            )

            # 6. Store packed output.
            var packed_bytes = bitcast[DType.uint8, 4](packed)
            output.store[width=4](
                Coord(Idx(global_row_idx), Idx(col_thread_idx * 4)),
                packed_bytes,
            )

            # 7. First thread in the 4-thread group stores the scale.
            if global_col_idx % SF_VECTOR_SIZE == 0:
                var scale_col = global_col_idx // SF_VECTOR_SIZE
                scales.store(
                    Coord(Idx(global_row_idx), Idx(scale_col)), e8m0_scale
                )


@always_inline
def quantize_mxfp4_amd[
    out_dtype: DType = DType.uint8,
    scales_dtype: DType = DType.float8_e8m0fnu,
    in_dtype: DType = DType.bfloat16,
    //,
    *,
    num_max_threads: Int = 512,
](
    ctx: DeviceContext,
    output_tile: TileTensor[
        mut=True, out_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    scales_tile: TileTensor[
        mut=True, scales_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    input_tile: TileTensor[
        mut=False, in_dtype, address_space=AddressSpace.GENERIC, ...
    ],
) raises:
    """Quantize BF16 activations to MXFP4 on AMD CDNA4 (MI355X).

    Produces packed uint8 output and 2D E8M0 block scales compatible
    with dequant_mxfp4() and V_MFMA_SCALE_F32_16X16X128_F8F6F4.

    NOTE: The 2D scales layout is a stand-in. The optimized CDNA4 layout
    will likely be 6D (32x32 tiles) or 7D (16x16 tiles), mirroring how
    SM100 uses a 5D interleaved layout for its tensor core scale feed.

    Args:
        ctx: Device context.
        output_tile: Output [M, K//2] uint8 (packed FP4).
        scales_tile: Output [M, K//32] float8_e8m0fnu (block scales).
        input_tile: Input [M, K] bfloat16.
    """
    comptime assert out_dtype == DType.uint8, "output must be uint8"
    comptime assert (
        scales_dtype == DType.float8_e8m0fnu
    ), "scales must be float8_e8m0fnu"
    comptime assert in_dtype == DType.bfloat16, "input must be bfloat16"
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
        TileTensor[DType.bfloat16, type_of(input_tile).LayoutType, MutAnyOrigin]
    ](input_tile)

    comptime kernel = _quantize_mxfp4_amd_kernel[
        type_of(output_tile).LayoutType,
        type_of(scales_tile).LayoutType,
        type_of(input_tt).LayoutType,
        ELEMENTS_PER_THREAD=ELEMENTS_PER_THREAD,
        num_max_threads=num_max_threads,
    ]

    ctx.enqueue_function[kernel](
        output_tile,
        scales_tile,
        input_tt,
        num_rows,
        num_cols,
        block_dim=block_dim_val,
        grid_dim=grid_dim_val,
    )


@__name("quantize_dynamic_block_scaled_mxfp4_kernel", mangle=True)
def quantize_dynamic_block_scaled_mxfp4_kernel[
    in_dtype: DType,
    *,
    elements_per_thread: Int,
](
    output_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    output_scales_ptr: UnsafePointer[Float8_e8m0fnu, MutAnyOrigin],
    input_ptr: UnsafePointer[Scalar[in_dtype], ImmutAnyOrigin],
    num_elements: Int,
):
    comptime threads_per_group = MXFP4_SF_VECTOR_SIZE // elements_per_thread

    var n = global_idx.x * elements_per_thread
    if n >= num_elements:
        return

    var loaded_vec = input_ptr.load[width=8](n)
    var thread_max = abs(loaded_vec).reduce_max().cast[DType.float32]()
    var group_max = warp.lane_group_max[num_lanes=threads_per_group](thread_max)

    var fp8_scale_factor = compute_mxfp4_even_scale(group_max)
    var scale_f32 = fp8_scale_factor.cast[DType.float32]()

    if thread_idx.x % threads_per_group == 0:
        output_scales_ptr[n // MXFP4_SF_VECTOR_SIZE] = fp8_scale_factor

    output_ptr.store[alignment=4](
        n // 2,
        bitcast[DType.uint8, 4](
            cast_float_to_fp4e2m1_amd(loaded_vec, scale_f32)
        ),
    )


@always_inline
def quantize_dynamic_block_scaled_mxfp4[
    in_dtype: DType
](
    output: TileTensor[mut=True, DType.uint8, ...],
    output_scales: TileTensor[mut=True, DType.float8_e8m0fnu, ...],
    input: TileTensor[mut=False, in_dtype, ...],
    ctx: DeviceContext,
) raises:
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
            elements_per_thread=elements_per_thread,
        ]

        ctx.enqueue_function[kernel](
            output.ptr,
            output_scales.ptr,
            input.ptr,
            num_elements,
            grid_dim=ceildiv(num_elements // elements_per_thread, BLOCK_DIM),
            block_dim=BLOCK_DIM,
        )


@always_inline
def _mxfp4_dotprod[
    out_dtype: DType,
    //,
    BLOCK_N: Int,
](
    c_ptr: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    a_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    b_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    K: Int,
):
    @always_inline
    def cast_fp2em1x2_to_bf16x2[
        byte_select: Int
    ](packed: Int32, scale: Float32) -> SIMD[DType.bfloat16, 2]:
        return llvm_intrinsic[
            "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4", SIMD[DType.bfloat16, 2]
        ](packed, scale, Int32(byte_select))

    @always_inline
    def dotprod_bf16x2(
        a: SIMD[DType.bfloat16, 2], b: SIMD[DType.bfloat16, 2], c: Float32
    ) -> Float32:
        return llvm_intrinsic["llvm.amdgcn.fdot2.f32.bf16", Float32](
            a, b, c, False
        )

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    var a_local_ptr = a_ptr
    var b_local_ptr = b_ptr

    var accum = SIMD[DType.float32, BLOCK_N](0)

    for ko in range(k_groups):
        var a_scale = a_scales_ptr[ko].cast[DType.float32]()
        var b_scale = InlineArray[Float32, BLOCK_N](uninitialized=True)

        comptime for bn in range(BLOCK_N):
            b_scale[bn] = b_scales_ptr[bn * k_groups + ko].cast[DType.float32]()

        comptime for ki in range(0, MXFP4_SF_VECTOR_SIZE // 2, 4):
            var a_data = bitcast[DType.int32, 1](a_local_ptr.load[width=4](ki))
            var b_data = InlineArray[Int32, BLOCK_N](uninitialized=True)

            comptime for bn in range(BLOCK_N):
                b_data[bn] = bitcast[DType.int32, 1](
                    b_local_ptr.load[width=4](bn * (K // 2) + ki)
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

        a_local_ptr += MXFP4_SF_VECTOR_SIZE // 2
        b_local_ptr += MXFP4_SF_VECTOR_SIZE // 2

    c_ptr.store(accum.cast[out_dtype]())


@always_inline
def _mxfp4_dotprod_block_size(static_N: Int) -> Int:
    comptime target_block_size = 16
    return target_block_size if (static_N % target_block_size) == 0 else 1


@__name("matmul_dynamic_block_scaled_mxfp4_kernel", mangle=True)
def matmul_dynamic_block_scaled_mxfp4_kernel[
    out_dtype: DType, BLOCK_N: Int
](
    c_ptr: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    a_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    b_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    var n = global_idx.x * BLOCK_N
    var m = global_idx.y

    if m >= M or n >= N:
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    _mxfp4_dotprod[BLOCK_N](
        c_ptr + m * N + n,
        a_ptr + m * (K // 2),
        b_ptr + n * (K // 2),
        a_scales_ptr + m * k_groups,
        b_scales_ptr + n * k_groups,
        K,
    )


@always_inline
def matmul_dynamic_block_scaled_mxfp4[
    out_dtype: DType
](
    c: TileTensor[mut=True, out_dtype, ...],
    a: TileTensor[mut=False, DType.uint8, ...],
    b: TileTensor[mut=False, DType.uint8, ...],
    a_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    ctx: DeviceContext,
) raises:
    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "matmul_dynamic_block_scaled_mxfp4",
        task_id=get_safe_task_id(ctx),
    ):
        comptime BLOCK_DIM = 16
        comptime BLOCK_N = _mxfp4_dotprod_block_size(c.static_shape[1])

        var M = Int(c.dim[0]())
        var N = Int(c.dim[1]())
        var K = Int(b.dim[1]()) * 2

        if M == 0 or N == 0:
            return

        comptime kernel = matmul_dynamic_block_scaled_mxfp4_kernel[
            out_dtype, BLOCK_N
        ]

        ctx.enqueue_function[kernel](
            c.ptr,
            a.ptr,
            b.ptr,
            a_scales.ptr,
            b_scales.ptr,
            M,
            N,
            K,
            grid_dim=(ceildiv(N // BLOCK_N, BLOCK_DIM), ceildiv(M, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )


@__name("grouped_matmul_block_scaled_mxfp4_kernel", mangle=True)
def grouped_matmul_block_scaled_mxfp4_kernel[
    out_dtype: DType, BLOCK_N: Int
](
    c_ptr: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    a_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    b_scales_ptr: UnsafePointer[Scalar[DType.float8_e8m0fnu], ImmutAnyOrigin],
    row_offsets_ptr: UnsafePointer[Scalar[DType.uint32], ImmutAnyOrigin],
    expert_ids_ptr: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    num_active_experts: Int,
    M: Int,
    N: Int,
    K: Int,
):
    var n = global_idx.x * BLOCK_N
    var m = global_idx.y

    if m >= M or n >= N:
        return

    var expert_id = -1
    for i in range(num_active_experts):
        if m >= Int(row_offsets_ptr[i]) and m < Int(row_offsets_ptr[i + 1]):
            expert_id = Int(expert_ids_ptr[i])
            break

    if expert_id == -1:
        c_ptr.store(m * N + n, SIMD[out_dtype, BLOCK_N](0))
        return

    var k_groups = K // MXFP4_SF_VECTOR_SIZE

    _mxfp4_dotprod[BLOCK_N](
        c_ptr + m * N + n,
        a_ptr + m * (K // 2),
        b_ptr + (expert_id * N + n) * (K // 2),
        a_scales_ptr + m * k_groups,
        b_scales_ptr + (expert_id * N + n) * k_groups,
        K,
    )


@always_inline
def grouped_matmul_block_scaled_mxfp4[
    out_dtype: DType,
](
    c: TileTensor[mut=True, out_dtype, ...],
    a: TileTensor[mut=False, DType.uint8, ...],
    b: TileTensor[mut=False, DType.uint8, ...],
    a_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    b_scales: TileTensor[mut=False, DType.float8_e8m0fnu, ...],
    row_offsets: TileTensor[mut=False, DType.uint32, ...],
    expert_ids: TileTensor[mut=False, DType.int32, ...],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "grouped_matmul_block_scaled_mxfp4",
        task_id=get_safe_task_id(ctx),
    ):
        comptime BLOCK_DIM = 16
        comptime BLOCK_N = _mxfp4_dotprod_block_size(c.static_shape[1])

        var M = Int(c.dim[0]())
        var N = Int(c.dim[1]())
        var K = Int(b.dim[2]()) * 2

        if M == 0 or N == 0:
            return

        comptime kernel = grouped_matmul_block_scaled_mxfp4_kernel[
            out_dtype, BLOCK_N
        ]

        ctx.enqueue_function[kernel](
            c.ptr,
            a.ptr,
            b.ptr,
            a_scales.ptr,
            b_scales.ptr,
            row_offsets.ptr,
            expert_ids.ptr,
            num_active_experts,
            M,
            N,
            K,
            grid_dim=(ceildiv(N // BLOCK_N, BLOCK_DIM), ceildiv(M, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )
