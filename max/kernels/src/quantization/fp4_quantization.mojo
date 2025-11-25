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

from math import align_up
from gpu import (
    block_idx,
    thread_idx,
    grid_dim,
    block_dim,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from logger import Logger
from gpu.warp import shuffle_xor
from math import recip
from quantization.fp4_utils import (
    cast_fp32_to_fp4e2m1,
    E2M1_TO_FLOAT32,
    cast_uint32_to_fp4e2m1,
    cast_fp_to_fp4e2m1,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    _set_scale_factor,
    _get_scale_factor,
)
from gpu.host.info import B200


########################################################
# Dynamic scaled NVFP4 quantization
########################################################

comptime logger = Logger()
comptime BLOCK_SIZE = 512


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

    var block_dim = (min(num_cols // ELEMENTS_PER_THREAD, BLOCK_SIZE), 1, 1)
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
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](BLOCK_SIZE)
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

                    # dequantize the input using the global scale factor and get the scale factor for these 16 elements by dividing it by the maximum value of e2m1
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
