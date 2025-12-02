# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
)
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from logger import Logger
from gpu.warp import shuffle_xor
from math import recip
from .fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_uint32_to_fp4e2m1,
    cast_fp_to_fp4e2m1,
    cast_f4e2m1x2_to_fp16x2,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    _set_scale_factor,
    _get_scale_factor,
)
from gpu.host.info import B200
from utils import StaticTuple
from collections import OptionalReg
from linalg.utils import elementwise_epilogue_type
from utils.index import Index, IndexList

########################################################
# Dynamic scaled NVFP4 quantization
########################################################

comptime logger = Logger()


@always_inline
fn quantize_dynamic_scaled_fp4[
    out_dtype: DType,
    scales_dtype: DType,
    in_dtype: DType,
    output_layout: Layout,
    scales_layout: Layout,
    input_layout: Layout, //,
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
    constrained[
        ctx.default_device_info.compute == B200.compute,
        "This kernel is only supported on SM100",
    ]()
    constrained[
        in_dtype in (DType.bfloat16,),
        "input dtype should be bfloat16",
    ]()
    constrained[
        out_dtype in (DType.uint32,),
        "output dtype should be uint32",
    ]()
    constrained[
        scales_dtype in (NVFP4_SF_DTYPE,),
        "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn)",
    ]()

    comptime ELEMENTS_PER_THREAD = SF_VECTOR_SIZE // 2
    comptime num_SMs = B200.sm_count

    var num_rows = input.dim(0)
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

    comptime kernel = quantize_dynamic_scaled_fp4_kernel[
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

    ctx.enqueue_function_checked[kernel, kernel](
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
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_max_threads)
)
fn quantize_dynamic_scaled_fp4_kernel[
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
    constrained[
        SF_VECTOR_SIZE == 16 and ELEMENTS_PER_THREAD == SF_VECTOR_SIZE // 2,
        (
            "Currently only supports NVFP4 (SF_VECTOR_SIZE = 16) with 8"
            " elements per thread"
        ),
    ]()

    comptime NUM_THREADS_PER_SF = SF_VECTOR_SIZE // ELEMENTS_PER_THREAD

    constrained[
        input.shape[1]() % ELEMENTS_PER_THREAD == 0,
        "num_cols must be a multiple of ELEMENTS_PER_THREAD (8 for NVFP4)",
    ]()

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
                    _set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
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
                    output[global_row_idx, col_idx] = rebind[Scalar[out_dtype]](
                        Scalar[out_dtype](0)
                    )

                if col_idx >= num_col_threads:
                    if global_col_idx % SF_VECTOR_SIZE == 0:
                        _set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
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
                    var group_max = max(
                        shuffle_xor(thread_max, 1), thread_max
                    ).cast[DType.float32]()

                    # get the scale factor for these 16 elements by dividing it by the maximum value of e2m1
                    var scale_factor = tensor_sf * (
                        group_max * recip(Float32(6.0))
                    )

                    # NVFP4 uses FP8-UE4M3 format for the scale factor but we know that scale_factor is always positive, so we can use E4M3 instead of UE4M3.
                    var fp8_scale_factor = scale_factor.cast[scales_dtype]()

                    # find the quantization scale factor for these 16 elements (scale_factor = scale_factor / tensor_sf)
                    # we divide input by this scale factor which is same as multiplying by the reciprocal of the scale factor
                    var output_scale = Float32(0.0)
                    if group_max != 0:
                        output_scale = recip(
                            fp8_scale_factor.cast[DType.float32]()
                            * recip(tensor_sf)
                        )

                    # write back the scale factor
                    if global_col_idx % SF_VECTOR_SIZE == 0:
                        _set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                            scales,
                            global_row_idx,
                            global_col_idx,
                            fp8_scale_factor,
                        )

                    var input_f32 = (
                        input_vector.cast[DType.float32]() * output_scale
                    )
                    var e2m1_vector = cast_fp32_to_fp4e2m1(input_f32)

                    output[global_row_idx, col_idx] = rebind[Scalar[out_dtype]](
                        e2m1_vector
                    )


@always_inline
fn block_scales_interleave_fp4[
    scales_dtype: DType,
    input_scales_layout: Layout,
    output_scales_layout: Layout, //,
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
    constrained[
        ctx.default_device_info.compute == B200.compute,
        "This kernel is only supported on SM100",
    ]()
    constrained[
        scales_dtype in (NVFP4_SF_DTYPE,),
        "scales dtype should be NVFP4_SF_DTYPE (float8_e4m3fn)",
    ]()

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

    ctx.enqueue_function_checked[kernel, kernel](
        input_scales,
        output_scales,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_max_threads)
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

            _set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                output_scales,
                row_idx,
                col_idx * SF_VECTOR_SIZE,
                scale_factor,
            )


fn naive_block_scaled_nvfp4_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType, //,
    *,
    SF_VECTOR_SIZE: Int,
    accum_type: DType = DType.float32,
    transpose_b: Bool = True,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    BLOCK_DIM: Int = 16,
](
    c: LayoutTensor[c_type, address_space = AddressSpace.GENERIC, **_],
    a: LayoutTensor[a_type, address_space = AddressSpace.GENERIC, **_],
    b: LayoutTensor[b_type, address_space = AddressSpace.GENERIC, **_],
    a_scales: LayoutTensor[
        a_scales_type, address_space = AddressSpace.GENERIC, **_
    ],
    b_scales: LayoutTensor[
        b_scales_type, address_space = AddressSpace.GENERIC, **_
    ],
    ctx: DeviceContext,
) raises:
    constrained[
        transpose_b,
        "Only transpose_b = True is supported for now",
    ]()
    constrained[
        accum_type in (DType.float32,),
        "Only float32 is supported for accumulation for scaled matmul",
    ]()
    constrained[
        a_type == b_type == DType.uint8,
        (
            "Only Float4-E2M1x2(i.e, uint8) is supported for input dtype for"
            " block scaled NVFP4 matmul"
        ),
    ]()
    constrained[
        a_scales_type == b_scales_type and a_scales_type == NVFP4_SF_DTYPE,
        (
            "input A and B scales dtype should be same and should be"
            " NVFP4_SF_DTYPE (float8_e4m3fn)"
        ),
    ]()
    constrained[
        c_type in (DType.bfloat16,),
        (
            "Only float32 is supported for output dtype for block scaled NVFP4"
            " matmul"
        ),
    ]()

    var M = c.dim(0)
    var N = c.dim(1)
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    var K = a.dim(1) * 2

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

    alias kernel = naive_block_scaled_nvfp4_matmul_kernel[
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
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]

    ctx.enqueue_function_checked[kernel, kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


fn naive_block_scaled_nvfp4_matmul_kernel[
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
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, MutAnyOrigin],
):
    # Note: This is a naive kernel that emulates a block scaled NVFP4 matmul.
    # Assumptions:
    # 1. both A and B should be in K-major format
    # 2. both a_scales and b_scales should be in NVFP4 scale factors layout (5D tensors)

    var M = c.dim(0)
    var N = c.dim(1)
    # TODO (KERN-2238): uint8 is a proxy data type for two Float4-E2M1 values for now.
    # We need to double the K dimension as we are allocating for uint8 input data type.
    # Remove this when GENAI-337 is fixed.
    var K = a.dim(1) * 2

    var row_idx = global_idx.x
    var col_idx = global_idx.y

    if row_idx >= UInt(M) or col_idx >= UInt(N):
        return

    var accum = Scalar[accum_type](0.0)
    for k in range(0, K, 2):
        var a_scale = _get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            a_scales, Int(row_idx), k
        )
        var b_scale = _get_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
            b_scales, Int(col_idx), k
        )

        # each uint8 element has two Float4-E2M1 values,
        var a_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
            rebind[UInt8](a[row_idx, k // 2])
        ).cast[accum_type]()
        var b_val_fp16x2 = cast_f4e2m1x2_to_fp16x2(
            rebind[UInt8](b[col_idx, k // 2])
        ).cast[accum_type]()

        @parameter
        for k_idx in range(2):
            var a_val = rebind[Scalar[accum_type]](a_val_fp16x2[k_idx])
            var b_val = rebind[Scalar[accum_type]](b_val_fp16x2[k_idx])
            var a_scale_val = abs(
                rebind[Scalar[accum_type]](a_scale.cast[accum_type]())
            )
            var b_scale_val = abs(
                rebind[Scalar[accum_type]](b_scale.cast[accum_type]())
            )
            accum += a_val * b_val * a_scale_val * b_scale_val

    @parameter
    if elementwise_lambda_fn:
        alias elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](
            Index(row_idx, col_idx), accum.cast[c_type]()
        )
    else:
        c[row_idx, col_idx] = accum.cast[c_type]()
