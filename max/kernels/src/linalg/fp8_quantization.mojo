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


from collections import OptionalReg
from collections.string.string_slice import get_static_string
from math import ceildiv
from sys import simdwidthof
from sys.info import _is_sm_9x_or_newer

import gpu.warp as warp
from algorithm.functional import _elementwise_impl_gpu
from buffer import Dim, NDBuffer
from buffer.dimlist import DimList
from gpu import WARP_SIZE, barrier, block_idx, lane_id, thread_idx, warp_id
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext
from gpu.host._compile import get_gpu_target
from gpu.memory import AddressSpace
from linalg.matmul import matmul
from linalg.utils_gpu import MatmulConfig
from memory import stack_allocation
from runtime.tracing import trace_arg

from utils.index import IndexList
from utils.numerics import max_finite, min_finite

########################################################
# Static scaled fp8 quantization
########################################################


@always_inline
fn quantize_static_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    is_scale_inverted: Bool = True,
](
    out_buffer: NDBuffer[mut=True, out_dtype, 2, *_],
    in_buffer: NDBuffer[in_dtype, 2, *_],
    scale: Float32,
    context: DeviceContext,
) raises:
    constrained[
        in_dtype in (DType.float32, DType.float16, DType.bfloat16),
        "input type should be float16, bfloat16 or float32",
    ]()
    constrained[
        out_dtype is DType.float8_e4m3fn, "output type should be float8_e4m3fn"
    ]()

    @always_inline
    @parameter
    @__copy_capture(out_buffer, in_buffer, scale)
    fn scaled_fp8_quant[width: Int, rank: Int](idx_arg: IndexList[rank]):
        constrained[
            _is_sm_9x_or_newer(),
            "this kernel is only supported on sm90 or newer",
        ]()
        constrained[rank == 2, "rank should be equal to 2"]()

        var idx = rebind[IndexList[2]](idx_arg)
        var in_vec_f32 = in_buffer.load[width=width](idx).cast[DType.float32]()

        var inversed_scale: Float32 = 1.0 / scale

        @parameter
        for i in range(width):
            var scaled_input_f32: Float32

            scaled_input_f32 = in_vec_f32[i] * inversed_scale
            in_vec_f32[i] = max(
                Float32(min_finite[out_dtype]()),
                min(Float32(max_finite[out_dtype]()), scaled_input_f32),
            )

        var scaled_in_vec = in_vec_f32.cast[out_dtype]()
        out_buffer.store(idx, rebind[SIMD[out_dtype, width]](scaled_in_vec))

    alias compile_target = get_gpu_target()
    alias target_simd_width = simdwidthof[in_dtype, target=compile_target]()

    _elementwise_impl_gpu[func=scaled_fp8_quant, simd_width=target_simd_width](
        IndexList[2](in_buffer.dim[0](), in_buffer.dim[1]()), context
    )


########################################################
# dynamic scaled fp8 quantization
########################################################


@always_inline
fn quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType, //,
    group_size_or_per_token: Int,
](
    scaled_output: NDBuffer[mut=True, out_dtype, 2, *_],
    scales: NDBuffer[mut=True, scales_dtype, 2, *_],
    input: NDBuffer[in_dtype, 2, *_],
    scale_ub: Float32,
    ctx: DeviceContext,
) raises:
    constrained[
        scales_dtype in (DType.bfloat16, DType.float16, DType.float32),
        "scales type should be bfloat16, float16 or float32",
    ]()
    constrained[
        out_dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "output type should be float8_e4m3fn or float8_e4m3fnuz",
    ]()

    alias group_size = input.shape.get[
        1
    ]() if group_size_or_per_token == -1 else group_size_or_per_token
    alias n_groups = input.shape.get[1]() // group_size
    alias simd_width = simdwidthof[in_dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.device_info.max_thread_block_size // WARP_SIZE
    alias warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )

    alias kernel = quantize_fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        warps_per_block,
        group_size,
    ]

    ctx.enqueue_function[kernel](
        scaled_output,
        scales,
        input,
        scale_ub,
        grid_dim=(input.dim[0](), n_groups, 1),
        block_dim=warps_per_block * WARP_SIZE,
        attributes=pdl_launch_attributes(),
    )


@always_inline
fn block_reduce[
    type: DType, //, warps_per_block: Int
](val: Scalar[type]) -> Scalar[type]:
    var max_smem = stack_allocation[
        warps_per_block, type, address_space = AddressSpace.SHARED
    ]()
    var max_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var warp_id = warp_id()
    var lane_idx = lane_id()

    var warp_max = warp.max(val)

    if tid < warps_per_block:
        max_smem[tid] = 0
    barrier()

    if lane_idx == 0:
        max_smem[warp_id] = warp_max
    barrier()

    if warp_id == 0:
        var warp_max: Scalar[type]
        if lane_idx < warps_per_block:
            warp_max = max_smem[lane_idx]
        else:
            warp_max = 0
        # the shuffle function only support shuffle a whole warp
        var block_max = warp.lane_group_max[num_lanes=WARP_SIZE](warp_max)
        if lane_idx == 0:
            max_broadcast[0] = block_max
    barrier()

    return max_broadcast[0]


fn quantize_fp8_kernel[
    out_type: DType,
    scales_type: DType,
    in_type: DType,
    warps_per_block: Int,
    group_size: Int,
](
    output: NDBuffer[mut=True, out_type, 2, MutableAnyOrigin],
    scales: NDBuffer[mut=True, scales_type, 2, MutableAnyOrigin],
    input: NDBuffer[in_type, 2, MutableAnyOrigin],
    scale_ub: Scalar[scales_type],
):
    alias simd_width = simdwidthof[in_type]()
    alias num_threads = warps_per_block * WARP_SIZE
    alias use_warp_tiling = group_size <= num_threads * simd_width
    alias fp8_max = Scalar[out_type].MAX_FINITE

    var input_vec = SIMD[in_type, simd_width](0)
    var thread_max = Scalar[in_type](0)

    var tid = thread_idx.x
    var row = block_idx.x
    var group_idx = block_idx.y

    with PDL():
        for i in range(tid, group_size // simd_width, num_threads):
            var idx = i * simd_width + group_idx * group_size
            input_vec = input.load[width=simd_width](row, idx)
            thread_max = max(thread_max, abs(input_vec).reduce_max())

        var group_max: Scalar[in_type]

        @parameter
        if warps_per_block > 1:
            group_max = block_reduce[warps_per_block](thread_max)
        else:
            group_max = warp.lane_group_max_and_broadcast[WARP_SIZE](thread_max)

        var scale_factor = (
            max(group_max.cast[scales_type](), scale_ub)
            / fp8_max.cast[scales_type]()
        )

        if tid == 0:
            scales.store[width=1](IndexList[2](row, group_idx), scale_factor)

        for i in range(tid, group_size // simd_width, num_threads):
            var idx = i * simd_width + group_idx * group_size

            @parameter
            if use_warp_tiling:
                pass
            else:
                input_vec = input.load[width=simd_width](row, idx)

            var output_vec = input_vec.cast[scales_type]() / scale_factor

            output_vec = max(
                SIMD[scales_type, simd_width](-fp8_max),
                min(SIMD[scales_type, simd_width](fp8_max), output_vec),
            )
            output.store[width=simd_width](
                IndexList[2](row, idx), output_vec.cast[out_type]()
            )


########################################################
# scaled fp8 matmul
########################################################


@always_inline
fn matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType, //,
    transpose_b: Bool = False,
    config: OptionalReg[
        MatmulConfig[a_type, b_type, c_type, transpose_b]
    ] = None,
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, c_type, 2, _, _],
    a: NDBuffer[a_type, 2, _, _],
    b: NDBuffer[b_type, 2, _, _],
    a_scales: NDBuffer[a_scales_type, 2, _, _],
    b_scales: NDBuffer[b_scales_type, 2, _, _],
    ctx: DeviceContext,
) raises:
    constrained[
        a_type in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "input A type should be float8_e4m3fn, float8_e4m3fnuz",
    ]()
    constrained[
        b_type in (DType.float8_e4m3fn, DType.float8_e4m3fnuz),
        "input B type should be float8_e4m3fn, float8_e4m3fnuz",
    ]()
    constrained[
        a_scales_type in (DType.bfloat16, DType.float16, DType.float32),
        "input A scales type should be bfloat16, float16 or float32",
    ]()
    constrained[
        b_scales_type in (DType.bfloat16, DType.float16, DType.float32),
        "input B scales type should be bfloat16, float16 or float32",
    ]()

    alias b_k_axis = 1 if transpose_b else 0
    alias b_row_axis = 0 if transpose_b else 1
    alias N = b.shape.get[b_row_axis]()
    var M = a.dim[0]()

    alias _trace_string = get_static_string[
        trace_arg(
            "A_scales",
            IndexList[2](a_scales.shape.get[0](), a_scales.shape.get[1]()),
            a_scales.type,
        ),
        ";",
        trace_arg(
            "B_scales",
            IndexList[2](b_scales.shape.get[0](), b_scales.shape.get[1]()),
            b_scales.type,
        ),
    ]()

    # create a dummy buffer to instruct the matmul kernel to output values
    # in the correct type
    var c_dummy = NDBuffer[
        DType.float32, 2, MutableAnyOrigin, DimList(Dim(), N)
    ](
        UnsafePointer[Scalar[DType.float32]](),
        IndexList[2](M, N),
    )

    @parameter
    @__copy_capture(c, a, b, a_scales, b_scales)
    @always_inline
    fn scaled_output_fn[
        type_: DType, width: Int, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[type_, width]):
        var a_scale = a_scales.load[width=1](idx[0], 0).cast[type_]()
        var b_scale: SIMD[type_, width]

        @parameter
        if transpose_b:
            b_scale = b_scales.load[width=width](idx[1], 0).cast[type_]()
        else:
            b_scale = b_scales.load[width=width](0, idx[1]).cast[type_]()

        var scaled_val = val * a_scale * b_scale

        c.store[width=width, alignment=alignment](
            idx, scaled_val.cast[c_type]()
        )

    matmul[
        target=target,
        transpose_b=transpose_b,
        elementwise_lambda_fn=scaled_output_fn,
        _trace_description=_trace_string,
    ](c_dummy, a, b, Optional[DeviceContext](ctx))
