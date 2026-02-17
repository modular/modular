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

from math import align_up, ceildiv
from gpu import (
    block_idx,
    thread_idx,
    grid_dim,
    block_dim,
    global_idx,
    MAX_THREADS_PER_BLOCK_METADATA,
    lane_id,
)
from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from layout import Layout, LayoutTensor
from logger import Logger
from gpu.primitives.warp import shuffle_xor
from math import recip
from .fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_f4e2m1x2_to_fp16x2,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    MXFP8_SF_DTYPE,
    set_scale_factor,
    get_scale_factor,
)
from gpu.host.info import B200
from utils import StaticTuple
from collections import Optional
from linalg.utils import (
    elementwise_epilogue_type,
    elementwise_compute_lambda_type,
)
from utils.index import Index, IndexList
from linalg.matmul.vendor.blas import matmul
from buffer import Dim, NDBuffer
from layout._ndbuffer_stub import from_ndbuffer_row_major
from memory import bitcast
from gpu.sync import named_barrier
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import SharedMemBarrier, TMATensorTile, create_tensor_tile
from layout.layout_tensor import LayoutTensorIter
from gpu.memory import external_memory, fence_async_view_proxy
from gpu import barrier
from sys import size_of, align_of, simd_width_of
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, RuntimeTuple
from memory import LegacyUnsafePointer
from layout.swizzle import make_swizzle
from algorithm import elementwise
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind

########################################################
# Dynamic scaled NVFP4 quantization
########################################################

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime logger = Logger()


@always_inline
fn quantize_dynamic_scaled_fp4fp8[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    input_layout: Layout,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 512,
](
    ctx: DeviceContext,
    output: LayoutTensor[out_dtype, output_layout, MutAnyOrigin],
    scales: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    input: LayoutTensor[in_dtype, input_layout, MutAnyOrigin],
    num_cols: Int,
    num_cols_padded: Int,
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
) raises:
    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"
    comptime assert in_dtype in (
        DType.bfloat16,
    ), "input dtype should be bfloat16"

    comptime assert (
        out_dtype == DType.uint8
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        and scales_dtype == DType.float8_e4m3fn
    ) or (
        out_dtype == DType.float8_e4m3fn
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        and scales_dtype == DType.float8_e8m0fnu
    ), (
        "output dtype should be uint8 (fp4-e2m1fnX2) for NVFP4 or"
        " float8_e4m3fnuz for MXFP8"
    )

    comptime N = input_layout.shape[1].value()

    @parameter
    if SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE:
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

    ctx.enqueue_function[kernel, kernel](
        output,
        scales,
        input,
        num_cols,
        num_cols_padded,
        tensor_sf,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
fn quantize_dynamic_scaled_fp4fp8_kernel[
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
    input: LayoutTensor[in_dtype, input_layout, MutAnyOrigin],
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

                    @parameter
                    if NUM_THREADS_PER_SF == 4:
                        thread_max = max(shuffle_xor(thread_max, 2), thread_max)

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

                    @parameter
                    if out_dtype == DType.uint8:
                        output_vector = bitcast[out_dtype, OUTPUT_WIDTH](
                            cast_fp32_to_fp4e2m1(input_f32)
                        )
                    else:
                        output_vector = rebind[SIMD[out_dtype, OUTPUT_WIDTH]](
                            input_f32.cast[out_dtype]()
                        )

                    output.store[width=OUTPUT_WIDTH](
                        global_row_idx,
                        col_idx * OUTPUT_WIDTH,
                        output_vector,
                    )


@always_inline
fn block_scales_interleave_fp4[
    scales_dtype: DType,
    input_scales_layout: Layout,
    output_scales_layout: Layout,
    //,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    ctx: DeviceContext,
    input_scales: LayoutTensor[scales_dtype, input_scales_layout, MutAnyOrigin],
    output_scales: LayoutTensor[
        scales_dtype, output_scales_layout, MutAnyOrigin
    ],
) raises:
    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
    ), "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn)"

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

    ctx.enqueue_function[kernel, kernel](
        input_scales,
        output_scales,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_max_threads))
)
fn block_scales_interleave_fp4_kernel[
    scales_dtype: DType,
    input_scales_layout: Layout,
    output_scales_layout: Layout,
    *,
    SF_VECTOR_SIZE: Int = 16,
    num_max_threads: Int = 1024,
](
    input_scales: LayoutTensor[scales_dtype, input_scales_layout, MutAnyOrigin],
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


fn naive_block_scaled_matmul[
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
    c: LayoutTensor[c_type, address_space = AddressSpace.GENERIC, ...],
    a: LayoutTensor[a_type, address_space = AddressSpace.GENERIC, ...],
    b: LayoutTensor[b_type, address_space = AddressSpace.GENERIC, ...],
    a_scales: LayoutTensor[
        a_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    b_scales: LayoutTensor[
        b_scales_type, address_space = AddressSpace.GENERIC, ...
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
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        and a_type == DType.uint8
        and a_scales_type == NVFP4_SF_DTYPE
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    ) or (
        scaling_kind == UMMAKind.KIND_MXF8F6F4
        and a_type == DType.float8_e4m3fn
        and a_scales_type == MXFP8_SF_DTYPE
        and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    ), (
        "Only MXF4NVF4 scaling kind is supported for NVFP4 input dtype with"
        " NVFP4 scales and MXF8F6F4 scaling kind is supported for MXFP8 input"
        " dtype with MXFP8 scales for block scaled matmul"
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
    var K = a.dim(1) * 2 if scaling_kind == UMMAKind.KIND_MXF4NVF4 else a.dim(1)

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

    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        alpha,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


fn naive_block_scaled_matmul_kernel[
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
    comptime K_STEPS = 2 if scaling_kind == UMMAKind.KIND_MXF4NVF4 else 1
    var K = a.dim(1) * K_STEPS

    var row_idx = global_idx.x
    var col_idx = global_idx.y

    if row_idx >= UInt(M) or col_idx >= UInt(N):
        return

    var accum = Scalar[accum_type](0.0)
    for k in range(0, K, K_STEPS):
        var a_scale = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            a_scales, Int(row_idx), k
        )
        var b_scale = get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            b_scales, Int(col_idx), k
        )

        @parameter
        if scaling_kind == UMMAKind.KIND_MXF4NVF4:
            # each uint8 element has two Float4-E2M1 values,
            var a_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](a[row_idx, k // K_STEPS])
            ).cast[accum_type]()
            var b_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
                rebind[UInt8](b[col_idx, k // K_STEPS])
            ).cast[accum_type]()

            @parameter
            for k_idx in range(K_STEPS):
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

    @parameter
    if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(row_idx, col_idx), accum.cast[c_type]()
        )
    else:
        c[row_idx, col_idx] = accum.cast[c_type]()


fn quantize_dynamic_block_scaled[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_device: NDBuffer[mut=True, out_dtype, 2, MutAnyOrigin, _],
    scales_device: NDBuffer[mut=True, scales_dtype, 5, MutAnyOrigin, _],
    input_device: NDBuffer[in_dtype, 2, MutAnyOrigin, _],
    tensor_sf: Float32,  # tensor-wise scale factor
    ctx: DeviceContext,
) raises:
    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"
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

    var input_tensor = from_ndbuffer_row_major(input_device)
    var output_tensor = from_ndbuffer_row_major(output_device)
    var scales_tensor = from_ndbuffer_row_major(scales_device)

    var num_rows = input_tensor.dim(0)
    var num_cols = input_tensor.dim(1)
    if num_rows == 0 or num_cols == 0:
        return

    comptime input_layout = input_tensor.layout
    comptime output_layout = output_tensor.layout
    comptime is_fp4 = out_dtype == DType.uint8 and scales_dtype == NVFP4_SF_DTYPE and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
    comptime is_fp8 = out_dtype == DType.float8_e4m3fn and scales_dtype == MXFP8_SF_DTYPE and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
    comptime assert is_fp4 or is_fp8, "invalid scaling kind"

    comptime static_N = input_layout.shape[1].value()

    @parameter
    if is_fp4:
        comptime assert (
            output_layout.shape[1].value() == input_layout.shape[1].value() // 2
        ), (
            "output.dim(1) must be equal to input.dim(1) // 2 (each output"
            " element (uint8) is 2 fp4-e2m1fn values)"
        )

    @parameter
    if is_fp4 and static_N % 32 == 0:
        quantize_dynamic_scaled_fp4_async[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            ctx,
            output_tensor,
            scales_tensor,
            input_tensor,
            tensor_sf=tensor_sf,
        )
    else:
        comptime assert (
            static_N % (SF_VECTOR_SIZE // 2) == 0
        ), "input.dim(1) must be a multiple of (SF_VECTOR_SIZE // 2)"

        quantize_dynamic_scaled_fp4fp8[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            num_max_threads=512,
        ](
            ctx,
            output_tensor,
            scales_tensor,
            input_tensor,
            num_cols=input_tensor.dim(1),
            num_cols_padded=input_tensor.dim(1),
            tensor_sf=tensor_sf,
        )


fn block_scales_interleave[
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    target: StaticString = "cpu",
](
    output_scales_device: NDBuffer[mut=True, scales_dtype, 5, MutAnyOrigin, _],
    input_scales_device: NDBuffer[scales_dtype, 2, MutAnyOrigin, _],
    ctx: DeviceContext,
) raises:
    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"
    comptime assert scales_dtype in (
        NVFP4_SF_DTYPE,
    ), "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn) for now."

    var output = from_ndbuffer_row_major(output_scales_device)
    var input = from_ndbuffer_row_major(input_scales_device)

    block_scales_interleave_fp4[SF_VECTOR_SIZE=SF_VECTOR_SIZE,](
        ctx, input, output
    )


@__llvm_arg_metadata(input_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(output_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(scales_tma_op, `nvvm.grid_constant`)
fn quantize_dynamic_scaled_async_fp4_kernel[
    input_dtype: DType,
    input_cta_tile_layout: Layout,
    input_desc_layout: Layout,
    output_dtype: DType,
    output_cta_tile_layout: Layout,
    output_desc_layout: Layout,
    scales_dtype: DType,
    scales_tma_tile_layout: Layout,
    scales_desc_layout: Layout,
    input_swizzle_mode: TensorMapSwizzle,
    output_swizzle_mode: TensorMapSwizzle,
    scales_swizzle_mode: TensorMapSwizzle,
    SF_VECTOR_SIZE: UInt,
    NUM_PIPELINES_STAGES: UInt,
](
    input_tma_op: TMATensorTile[
        input_dtype, input_cta_tile_layout, input_desc_layout
    ],
    output_tma_op: TMATensorTile[
        output_dtype, output_cta_tile_layout, output_desc_layout
    ],
    scales_tma_op: TMATensorTile[
        scales_dtype, scales_tma_tile_layout, scales_desc_layout
    ],
    tensor_sf: Float32,  # tensor-wise scale factor
):
    var smem_storage = rebind[
        UnsafePointer[Scalar[input_dtype], address_space = AddressSpace.SHARED]
    ](
        external_memory[
            Scalar[input_dtype],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]()
    )

    comptime input_smem_tile_size = input_cta_tile_layout.size() * Int(
        NUM_PIPELINES_STAGES
    )
    comptime output_smem_tile_size = output_cta_tile_layout.size()
    comptime scales_smem_tile_size = scales_tma_tile_layout.size()

    comptime SF_K_GROUP_SIZE: UInt = SF_VECTOR_SIZE * SF_ATOM_K
    comptime STAGE_GROUP_SIZE = SF_K_GROUP_SIZE // NUM_PIPELINES_STAGES

    comptime assert (
        STAGE_GROUP_SIZE == 64
        and NUM_PIPELINES_STAGES == 1
        and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
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
        input_cta_tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        input_smem_ptr,
        input_smem_tile_size,
    )

    var output_smem = LayoutTensor[
        output_dtype,
        output_cta_tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        output_smem_ptr,
    )

    var scales_smem = LayoutTensor[
        scales_dtype,
        scales_tma_tile_layout,
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ](
        scales_smem_ptr,
    )

    var tma_mbar = mbar_ptr.bitcast[SharedMemBarrier]()

    var local_row_idx = Int(thread_idx.x)

    if thread_idx.x == 0:

        @parameter
        for idx in range(NUM_PIPELINES_STAGES):
            tma_mbar[idx].init()

    var tma_phase = SIMD[DType.uint32, Int(NUM_PIPELINES_STAGES)](0)

    barrier()

    comptime expected_bytes = input_cta_tile_layout.size() * size_of[
        input_dtype
    ]()

    if thread_idx.x >= 128:
        warpgroup_reg_dealloc[24]()

        @parameter
        for iter_idx in range(NUM_PIPELINES_STAGES):
            var smem_tile = input_smem.next(iter_idx)[]

            if lane_id() == 0:
                tma_mbar[iter_idx].expect_bytes(Int32(expected_bytes))
                input_tma_op.async_copy(
                    smem_tile,
                    tma_mbar[iter_idx],
                    (
                        Int(
                            (block_idx.y * SF_K_GROUP_SIZE)
                            + (iter_idx * STAGE_GROUP_SIZE)
                        ),
                        Int(block_idx.x) * SF_MN_GROUP_SIZE,
                    ),
                )

    else:
        var scale_factors = SIMD[scales_dtype, SF_ATOM_K]()

        @parameter
        for iter_idx in range(NUM_PIPELINES_STAGES):
            var smem_tile = input_smem.next(iter_idx)[]

            tma_mbar[iter_idx].wait(tma_phase[Int(iter_idx)])
            var quantized_elements = SIMD[DType.uint32, 8]()

            @parameter
            for group_idx in range(STAGE_GROUP_SIZE // SF_VECTOR_SIZE):
                var group_elements = SIMD[input_dtype, Int(SF_VECTOR_SIZE)]()

                @parameter
                for col_idx in range(SF_VECTOR_SIZE // 8):
                    var swizzle_offset = (
                        local_row_idx * Int(STAGE_GROUP_SIZE)
                        + Int(group_idx * SF_VECTOR_SIZE)
                        + Int(col_idx * 8)
                    )

                    comptime input_swizzle = make_swizzle[
                        input_dtype, input_swizzle_mode
                    ]()
                    var swizzle_idx = input_swizzle(swizzle_offset)
                    var temp = smem_tile.ptr.load[
                        width=8, alignment = align_of[SIMD[input_dtype, 8]]()
                    ](swizzle_idx)

                    group_elements = group_elements.insert[
                        offset = Int(col_idx * 8)
                    ](temp)

                var group_max = (
                    abs(group_elements).reduce_max().cast[DType.float32]()
                )

                var scale_factor = tensor_sf * group_max * recip(Float32(6.0))
                var fp8_scale_factor = scale_factor.cast[scales_dtype]()

                scale_factors[
                    Int(iter_idx) * Int(NUM_PIPELINES_STAGES) + Int(group_idx)
                ] = fp8_scale_factor

                var output_scale = Float32(0.0)
                if fp8_scale_factor.cast[DType.float32]() != 0:
                    output_scale = recip(
                        fp8_scale_factor.cast[DType.float32]()
                        * recip(tensor_sf)
                    )

                @parameter
                for slice_idx in range(2):
                    var slice_elements = group_elements.slice[
                        8, offset = slice_idx * 8
                    ]()
                    quantized_elements[
                        Int(group_idx) * 2 + slice_idx
                    ] = cast_fp32_to_fp4e2m1(
                        slice_elements.cast[DType.float32]() * output_scale
                    )

            @parameter
            for idx in range(2):
                var slice_elements = quantized_elements.slice[
                    4, offset = idx * 4
                ]()
                comptime output_swizzle = make_swizzle[
                    output_dtype, output_swizzle_mode
                ]()
                var swizzle_offset = local_row_idx * Int(
                    STAGE_GROUP_SIZE // 2
                ) + idx * Int(SF_VECTOR_SIZE)
                var output_swizzle_idx = output_swizzle(swizzle_offset)
                output_smem.ptr.store[
                    alignment = align_of[
                        SIMD[output_dtype, Int(SF_VECTOR_SIZE)]
                    ]()
                ](
                    output_swizzle_idx,
                    bitcast[output_dtype, Int(SF_VECTOR_SIZE)](slice_elements),
                )

            scales_smem.ptr.store[
                alignment = align_of[SIMD[scales_dtype, SF_ATOM_K]]()
            ](
                (local_row_idx % 32) * 16 + (local_row_idx // 32) * SF_ATOM_K,
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
                    UInt32(block_idx.y * (SF_K_GROUP_SIZE) // 2),
                    UInt32(block_idx.x) * UInt32(SF_MN_GROUP_SIZE),
                ),
            )
            output_tma_op.commit_group()

        output_tma_op.wait_group[0]()


fn quantize_dynamic_scaled_fp4_async[
    input_dtype: DType,
    output_dtype: DType,
    scales_dtype: DType,
    input_layout: Layout,
    output_layout: Layout,
    scales_layout: Layout,
    //,
    SF_VECTOR_SIZE: Int,
](
    ctx: DeviceContext,
    output_tensor: LayoutTensor[output_dtype, output_layout, MutAnyOrigin],
    scales_tensor: LayoutTensor[scales_dtype, scales_layout, MutAnyOrigin],
    input_tensor: LayoutTensor[input_dtype, input_layout, MutAnyOrigin],
    tensor_sf: Float32 = 1.0,  # tensor-wise scale factor
) raises:
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
        __tile_layout = Layout.row_major(input_tma_tile_shape),
    ](ctx, input_tensor)

    comptime output_tma_tile_shape = Index(128, 32)
    var output_tma_op = create_tensor_tile[
        output_tma_tile_shape,
        swizzle_mode=output_swizzle_mode,
        __tile_layout = Layout.row_major(output_tma_tile_shape),
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
        __tile_layout = Layout.row_major(scales_tma_tile_shape),
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
        type_of(input_tma_op).layout,
        type_of(input_tma_op).desc_layout,
        type_of(output_tma_op).dtype,
        type_of(output_tma_op).layout,
        type_of(output_tma_op).desc_layout,
        type_of(scales_tma_op).dtype,
        type_of(scales_tma_op).layout,
        type_of(scales_tma_op).desc_layout,
        input_swizzle_mode,
        output_swizzle_mode,
        scales_swizzle_mode,
        UInt(SF_VECTOR_SIZE),
        NUM_PIPELINES_STAGES = UInt(NUM_PIPELINES_STAGES),
    ]

    ctx.enqueue_function[kernel, kernel, dump_asm=False](
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
    )


########################################################
# SM100 Block Scaled matmul kernel dispatch
########################################################


fn block_scaled_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    target: StaticString = "cpu",
](
    c_device: NDBuffer[mut=True, c_type, 2, MutAnyOrigin, _],
    a_device: NDBuffer[a_type, 2, MutAnyOrigin, _],
    b_device: NDBuffer[b_type, 2, MutAnyOrigin, _],
    a_scales_device: NDBuffer[scales_dtype, 5, MutAnyOrigin, _],
    b_scales_device: NDBuffer[scales_dtype, 5, MutAnyOrigin, _],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
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

    var c = from_ndbuffer_row_major(c_device)
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var a_scales = from_ndbuffer_row_major(a_scales_device)
    var b_scales = from_ndbuffer_row_major(b_scales_device)

    comptime sfa_layout = a_scales.layout
    comptime sfb_layout = b_scales.layout

    comptime assert (
        sfa_layout.shape[1].value() == sfb_layout.shape[1].value()
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        sfa_layout.shape[2].value()
        == sfb_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        sfa_layout.shape[3].value()
        == sfb_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        sfa_layout.shape[4].value() == sfb_layout.shape[4].value() == SF_ATOM_K
    ), ""

    block_scaled_matmul_with_epilogue[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=transpose_b,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        tensor_sf,
        ctx,
    )


########################################################
# SM100 Block Scaled matmul with normal epilogue kernel dispatch
########################################################


fn block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: LayoutTensor[scales_dtype, sfa_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[scales_dtype, sfb_layout, ImmutAnyOrigin],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Our sm100 block scaled matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 block scaled matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations.
    """

    comptime assert (
        ctx.default_device_info.compute == B200.compute
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        scales_dtype == NVFP4_SF_DTYPE
    ), "Only support NVFP4_SF_DTYPE (float8_e4m3fn) for scales for now."

    comptime assert SF_VECTOR_SIZE in (
        NVFP4_SF_VECTOR_SIZE,
    ), "SF_VECTOR_SIZE must be equal to NVFP4_SF_VECTOR_SIZE (16 for NVFP4)"

    comptime assert (
        sfa_layout.shape[1].value() == sfb_layout.shape[1].value()
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        sfa_layout.shape[2].value()
        == sfb_layout.shape[2].value()
        == SF_ATOM_M[0]
    ), ""
    comptime assert (
        sfa_layout.shape[3].value()
        == sfb_layout.shape[3].value()
        == SF_ATOM_M[1]
    ), ""
    comptime assert (
        sfa_layout.shape[4].value() == sfb_layout.shape[4].value() == SF_ATOM_K
    ), ""

    var m = c.dim(0)
    var n = c.dim(1)
    if m == 0 or n == 0:
        return

    @parameter
    if not elementwise_lambda_fn:
        if not c.ptr:
            raise "c must be allocated!"

        matmul(
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
            simd_width_of[c_type, target = get_gpu_target()]()
        )

        @parameter
        @__copy_capture(c)
        fn epilogue_wrapper[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            var c_coord = Index(idx[0], idx[1])
            var c_val = c.load[width=simd_width,](c_coord)
            epilogue[c_type, simd_width, alignment=alignment](c_coord, c_val)

        # If c is already allocated, we can just use the sm100 blockwise scaled fp8 matmul and
        # apply the epilogue.
        if c.ptr:
            var m = c.dim[0]()
            var n = c.dim[1]()

            matmul(
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
            return

        # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
        var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](c.size())
        var c_tmp = c
        c_tmp.ptr = tmp_device_buffer.unsafe_ptr()

        block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](
            c_tmp,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            ctx,
        )

        _ = tmp_device_buffer^
