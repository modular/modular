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

from collections.string.string_slice import get_static_string
from math import ceildiv
from sys import simd_width_of, has_nvidia_gpu_accelerator
from sys import align_of, size_of
import gpu.primitives.block
from algorithm.functional import _elementwise_impl_gpu
from buffer import Dim, NDBuffer
from buffer.dimlist import DimList
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    thread_idx,
)
from gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext, get_gpu_target
from gpu.host.info import B200, H100
from layout import IntTuple, Layout, LayoutTensor
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout._coord import coord_to_index_list
from layout._tile_tensor import TileTensor
from logger import Logger
from memory import LegacyUnsafePointer, bitcast

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from runtime.tracing import Trace, TraceLevel, trace_arg
from std.bit import log2_floor
from algorithm import elementwise
from utils.index import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type, max_finite, min_finite

from .matmul import matmul
from .matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)
from .matmul.gpu.sm100_structured.structured_kernels.tile_types import lt_to_tt
from .utils import elementwise_epilogue_type
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from .fp8_utils import compute_dynamic_fp8_scale, fp8_quantize


comptime logger = Logger()


########################################################
# static scaled fp8 quantization
########################################################


@always_inline
fn quantize_static_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scale_is_inverted: Bool = True,
](
    out_buffer: NDBuffer[mut=True, out_dtype, 2, _, _, _],
    in_buffer: NDBuffer[in_dtype, 2, _, _, _],
    scale: Float32,
    context: DeviceContext,
) raises:
    comptime assert in_dtype in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
    ), "input dtype should be float16, bfloat16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    @always_inline
    @parameter
    @__copy_capture(out_buffer, in_buffer, scale)
    fn scaled_fp8_quant[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var idx = rebind[IndexList[2]](idx_arg)
        var in_vec_f32 = in_buffer.load[width=width](idx).cast[DType.float32]()
        var inversed_scale: Float32 = 1.0 / scale
        out_buffer.store(
            idx,
            fp8_quantize[out_dtype, use_clamp=True](in_vec_f32, inversed_scale),
        )

    comptime target_simd_width = simd_width_of[
        in_dtype, target = get_gpu_target()
    ]()

    _elementwise_impl_gpu[
        func=scaled_fp8_quant, simd_width = UInt(target_simd_width)
    ](IndexList[2](in_buffer.dim[0](), in_buffer.dim[1]()), context)


########################################################
# dynamic scaled fp8 quantization
########################################################


@always_inline
fn quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    //,
    input_fn: fn[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[in_dtype, width],
    group_size_or_per_token: Int,
    num_cols: Int,
](
    scaled_output: NDBuffer[mut=True, out_dtype, 2, _],
    scales: NDBuffer[mut=True, scales_dtype, 2, _],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
) raises:
    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
        DType.float8_e8m0fnu,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    comptime assert (scales_dtype != DType.float8_e8m0fnu) or (
        out_dtype == DType.float8_e4m3fn
    ), "float8_e8m0fnu is only supported for float8_e4m3fn output dtype"

    comptime group_size = num_cols if group_size_or_per_token == -1 else group_size_or_per_token
    comptime simd_width = 16 if group_size % 16 == 0 else 8 if group_size % 8 == 0 else 4
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )
    comptime num_threads = warps_per_block * WARP_SIZE

    comptime assert (
        group_size % simd_width == 0
    ), "group size must be multiple of simd size"

    with Trace[TraceLevel.OP, target = StaticString("gpu")](
        "quantize_dynamic_scaled_fp8",
        task_id=Int(ctx.id()),
    ):
        if num_rows == 0:
            return

        comptime kernel = quantize_fp8_kernel[
            out_dtype,
            scales_dtype,
            in_dtype,
            input_fn,
            num_threads,
            group_size,
            simd_width,
        ]

        ctx.enqueue_function[kernel, kernel](
            scaled_output,
            scales,
            scale_ub.cast[scales_dtype](),
            grid_dim=(num_rows, num_cols // group_size, 1),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn quantize_fp8_kernel[
    out_type: DType,
    scales_type: DType,
    in_type: DType,
    input_fn: fn[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[in_type, width],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
](
    output: NDBuffer[mut=True, out_type, 2, MutAnyOrigin],
    scales: NDBuffer[mut=True, scales_type, 2, MutAnyOrigin],
    scale_ub: Scalar[scales_type],
):
    comptime use_warp_tiling = group_size <= num_threads * simd_width
    comptime fp8_max = Scalar[out_type].MAX_FINITE
    comptime accum_type = get_accum_type[in_type]()

    comptime assert (scales_type != DType.float8_e8m0fnu) or (
        accum_type == DType.float32
    ), "float8_e8m0fnu quantization is only supported for float32 accum type"

    var input_vec = SIMD[accum_type, simd_width](0)
    var thread_max = Scalar[accum_type](0)

    var tid = thread_idx.x
    var row = Int(block_idx.x)
    var group_idx = Int(block_idx.y)

    with PDL():
        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size
            input_vec = input_fn[simd_width, simd_width](row, idx).cast[
                accum_type
            ]()
            thread_max = max(thread_max, abs(input_vec).reduce_max())

        var group_max = block.max[block_size=num_threads, broadcast=True](
            thread_max
        )

        var scale_factor: Scalar[scales_type]
        var scale_factor_recip: Scalar[accum_type]

        comptime if scales_type == DType.float8_e8m0fnu:
            scale_factor = max(
                group_max / fp8_max.cast[accum_type](),
                Scalar[accum_type](1e-10),
            ).cast[scales_type]()
            scale_factor_recip = (
                0.0 if group_max
                == 0.0 else 1.0 / scale_factor.cast[accum_type]()
            )
        else:
            scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
                out_type
            ](group_max, scale_ub)

        if tid == 0:
            scales.store(Index(group_idx, row), scale_factor)

        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size

            comptime if use_warp_tiling:
                pass
            else:
                input_vec = input_fn[simd_width, simd_width](row, idx).cast[
                    accum_type
                ]()

            output.store(
                Index(row, idx),
                fp8_quantize[out_type](input_vec, scale_factor_recip),
            )


@always_inline
fn batched_quantize_dynamic_scaled_fp8[
    out_dtype: DType,
    in_dtype: DType,
    scales_dtype: DType,
    //,
    input_fn: fn[width: Int, alignment: Int](
        batch: Int, row: Int, col: Int
    ) capturing -> SIMD[in_dtype, width],
    group_size_or_per_token: Int,
    num_cols: Int,
](
    scaled_output: NDBuffer[mut=True, out_dtype, 3, _],
    scales: NDBuffer[mut=True, scales_dtype, 3, _],
    scale_ub: Float32,
    ctx: DeviceContext,
    num_rows: Int,
    batch_size: Int,
) raises:
    comptime assert scales_dtype in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "scales dtype should be bfloat16, float16 or float32"
    comptime assert out_dtype in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "output dtype should be float8_e4m3fn or float8_e4m3fnuz"

    comptime group_size = num_cols if group_size_or_per_token == -1 else group_size_or_per_token
    comptime simd_width = 16 if group_size % 16 == 0 else 8 if group_size % 8 == 0 else 4
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime warps_per_block = min(
        ceildiv(group_size // simd_width, WARP_SIZE), max_warps_per_block
    )
    comptime num_threads = warps_per_block * WARP_SIZE

    comptime assert (
        group_size % simd_width == 0
    ), "group size must be multiple of simd size"

    if batch_size == 0 or num_rows == 0:
        return

    comptime kernel = batched_quantize_fp8_kernel[
        out_dtype,
        scales_dtype,
        in_dtype,
        input_fn,
        num_threads,
        group_size,
        simd_width,
    ]

    ctx.enqueue_function[kernel, kernel](
        scaled_output,
        scales,
        scale_ub.cast[scales_dtype](),
        grid_dim=(num_rows, num_cols // group_size, batch_size),
        block_dim=num_threads,
        attributes=pdl_launch_attributes(),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
fn batched_quantize_fp8_kernel[
    out_type: DType,
    scales_type: DType,
    in_type: DType,
    input_fn: fn[width: Int, alignment: Int](
        batch: Int, row: Int, col: Int
    ) capturing -> SIMD[in_type, width],
    num_threads: Int,
    group_size: Int,
    simd_width: Int,
](
    output: NDBuffer[mut=True, out_type, 3, MutAnyOrigin],
    scales: NDBuffer[mut=True, scales_type, 3, MutAnyOrigin],
    scale_ub: Scalar[scales_type],
):
    comptime use_warp_tiling = group_size <= num_threads * simd_width
    comptime accum_type = get_accum_type[in_type]()

    var input_vec = SIMD[accum_type, simd_width](0)
    var thread_max = Scalar[accum_type](0)

    var tid = thread_idx.x
    var row = Int(block_idx.x)
    var group_idx = Int(block_idx.y)
    var batch_idx = Int(block_idx.z)

    with PDL():
        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size
            input_vec = input_fn[simd_width, simd_width](
                batch_idx, row, idx
            ).cast[accum_type]()
            thread_max = max(thread_max, abs(input_vec).reduce_max())

        var group_max = block.max[block_size=num_threads, broadcast=True](
            thread_max
        )

        var scale_factor, scale_factor_recip = compute_dynamic_fp8_scale[
            out_type
        ](group_max, scale_ub)

        if tid == 0:
            scales.store(Index(batch_idx, group_idx, row), scale_factor)

        for i in range(tid, group_size // simd_width, num_threads):
            var idx: Int = i * simd_width + group_idx * group_size

            comptime if use_warp_tiling:
                pass
            else:
                input_vec = input_fn[simd_width, simd_width](
                    batch_idx, row, idx
                ).cast[accum_type]()

            output.store(
                Index(batch_idx, row, idx),
                fp8_quantize[out_type](input_vec, scale_factor_recip),
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
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: TileTensor[mut=True, c_type, address_space = AddressSpace.GENERIC, ...],
    a: TileTensor[a_type, address_space = AddressSpace.GENERIC, ...],
    b: TileTensor[b_type, address_space = AddressSpace.GENERIC, ...],
    a_scales: TileTensor[
        a_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    b_scales: TileTensor[
        b_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    comptime assert c.rank == 2
    comptime assert a.rank == 2
    comptime assert b.rank == 2
    comptime assert a_scales.rank == 2
    comptime assert b_scales.rank == 2
    comptime dim[i: Int] = Dim(i) if i > -1 else Dim()

    comptime c_shape = DimList(dim[c.static_shape[0]], dim[c.static_shape[1]])
    comptime c_stride = DimList(
        dim[c.static_stride[0]], dim[c.static_stride[1]]
    )
    var c_buf = NDBuffer[c_type, 2, _, c_shape, c_stride](
        c.ptr,
        rebind[IndexList[2]](coord_to_index_list(c.layout.shape_coord())),
        rebind[IndexList[2]](coord_to_index_list(c.layout.stride_coord())),
    )
    comptime a_shape = DimList(dim[a.static_shape[0]], dim[a.static_shape[1]])
    comptime a_stride = DimList(
        dim[a.static_stride[0]], dim[a.static_stride[1]]
    )
    var a_buf = NDBuffer[a_type, 2, ImmutAnyOrigin, a_shape, a_stride](
        a.ptr,
        rebind[IndexList[2]](coord_to_index_list(a.layout.shape_coord())),
        rebind[IndexList[2]](coord_to_index_list(a.layout.stride_coord())),
    )
    comptime b_shape = DimList(dim[b.static_shape[0]], dim[b.static_shape[1]])
    comptime b_stride = DimList(
        dim[b.static_stride[0]], dim[b.static_stride[1]]
    )
    var b_buf = NDBuffer[b_type, 2, ImmutAnyOrigin, b_shape, b_stride](
        b.ptr,
        rebind[IndexList[2]](coord_to_index_list(b.layout.shape_coord())),
        rebind[IndexList[2]](coord_to_index_list(b.layout.stride_coord())),
    )
    comptime a_scales_shape = DimList(
        dim[a_scales.static_shape[0]], dim[a_scales.static_shape[1]]
    )
    comptime a_scales_stride = DimList(
        dim[a_scales.static_stride[0]], dim[a_scales.static_stride[1]]
    )
    var a_scales_buf = NDBuffer[
        a_scales_type, 2, _, a_scales_shape, a_scales_stride
    ](
        a_scales.ptr,
        rebind[IndexList[2]](
            coord_to_index_list(a_scales.layout.shape_coord())
        ),
        rebind[IndexList[2]](
            coord_to_index_list(a_scales.layout.stride_coord())
        ),
    )
    comptime b_scales_shape = DimList(
        dim[b_scales.static_shape[0]], dim[b_scales.static_shape[1]]
    )
    comptime b_scales_stride = DimList(
        dim[b_scales.static_stride[0]], dim[b_scales.static_stride[1]]
    )
    var b_scales_buf = NDBuffer[
        b_scales_type, 2, _, b_scales_shape, b_scales_stride
    ](
        b_scales.ptr,
        rebind[IndexList[2]](
            coord_to_index_list(b_scales.layout.shape_coord())
        ),
        rebind[IndexList[2]](
            coord_to_index_list(b_scales.layout.stride_coord())
        ),
    )

    matmul_dynamic_scaled_fp8[
        input_scale_granularity,
        weight_scale_granularity,
        m_scale_granularity,
        n_scale_granularity,
        k_scale_granularity,
        transpose_b,
        target,
    ](
        c_buf.make_dims_unknown(),
        a_buf,
        b_buf,
        a_scales_buf,
        b_scales_buf,
        ctx,
    )


@always_inline
fn matmul_dynamic_scaled_fp8[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    input_scale_granularity: StaticString,
    weight_scale_granularity: StaticString,
    m_scale_granularity: Int,
    n_scale_granularity: Int,
    k_scale_granularity: Int,
    transpose_b: Bool = False,
    target: StaticString = "cpu",
](
    c: NDBuffer[mut=True, c_type, 2, _, _, _],
    a: NDBuffer[mut=False, a_type, 2, _, _],
    b: NDBuffer[mut=False, b_type, 2, _, _],
    a_scales: NDBuffer[a_scales_type, 2, _, _, _],
    b_scales: NDBuffer[b_scales_type, 2, _, _, _],
    ctx: DeviceContext,
) raises:
    comptime assert a_type == b_type, "input A and B dtype should be the same"
    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be the same"

    comptime assert a_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input A dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert b_type in (
        DType.float8_e4m3fn,
        DType.float8_e4m3fnuz,
    ), "input B dtype should be float8_e4m3fn, float8_e4m3fnuz"
    comptime assert a_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input A scales dtype should be bfloat16, float16 or float32"
    comptime assert b_scales_type in (
        DType.bfloat16,
        DType.float16,
        DType.float32,
    ), "input B scales dtype should be bfloat16, float16 or float32"

    comptime b_k_axis = 1 if transpose_b else 0
    comptime b_row_axis = 0 if transpose_b else 1
    comptime N = b.shape.get[b_row_axis]()
    var M = a.dim[0]()
    # var K = a.dim[1]()

    if M == 0:
        return

    comptime _trace_string = get_static_string[
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

    # Tensorwise and Channelwise scaling
    comptime if (
        input_scale_granularity == "colwise"
        and weight_scale_granularity == "rowwise"
    ) or (input_scale_granularity == weight_scale_granularity == "tensor"):
        logger.info(
            "Dispatching Matmul Dynamic Scaled FP8. Input Scale Granularity: ",
            input_scale_granularity,
            ", Weight Scale Granularity: ",
            weight_scale_granularity,
        )

        comptime if ctx.default_device_info == B200:

            @parameter
            @always_inline
            @__copy_capture(a_scales, b_scales)
            fn scale_compute_lambda_fn[
                _dtype: DType,
                width: Int,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
                _dtype, width
            ]:
                var a_scale = a_scales.load[width=1](0, idx[0]).cast[
                    DType.float32
                ]()
                var b_scale: SIMD[DType.float32, width]

                comptime if transpose_b:
                    b_scale = b_scales.load[width=width](idx[1], 0).cast[
                        DType.float32
                    ]()
                else:
                    b_scale = b_scales.load[width=width](0, idx[1]).cast[
                        DType.float32
                    ]()

                var scaled_val = val.cast[DType.float32]() * a_scale * b_scale
                return scaled_val.cast[_dtype]()

            matmul[
                target=target,
                transpose_b=transpose_b,
                elementwise_compute_lambda_fn=scale_compute_lambda_fn,
                _trace_description=_trace_string,
            ](c, a, b, Optional[DeviceContext](ctx))

        else:
            # create a dummy buffer to instruct the matmul kernel to output values
            # in the correct dtype
            var c_dummy = NDBuffer[
                DType.float32, 2, MutAnyOrigin, DimList(Dim(), N)
            ](
                UnsafePointer[Scalar[DType.float32]](),
                IndexList[2](M, N),
            )

            @parameter
            @__copy_capture(c, a, b, a_scales, b_scales)
            @always_inline
            fn scaled_output_fn[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                var a_scale = a_scales.load[width=1](0, idx[0]).cast[dtype]()
                var b_scale: SIMD[dtype, width]

                comptime if transpose_b:
                    b_scale = b_scales.load[width=width](idx[1], 0).cast[
                        dtype
                    ]()
                else:
                    b_scale = b_scales.load[width=width](0, idx[1]).cast[
                        dtype
                    ]()

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

    elif (
        input_scale_granularity == "block"
        and weight_scale_granularity == "block"
    ):
        var a_tensor = from_ndbuffer_row_major(a)
        var b_tensor = from_ndbuffer_row_major(b)
        var c_tensor = from_ndbuffer_row_major(c)
        var a_scales_tensor = from_ndbuffer_row_major(a_scales)
        var b_scales_tensor = from_ndbuffer_row_major(b_scales)

        blockwise_scaled_fp8_with_epilogue[
            transpose_b=transpose_b,
            scales_granularity_mnk = IndexList[3](
                m_scale_granularity,
                n_scale_granularity,
                k_scale_granularity,
            ),
        ](c_tensor, a_tensor, b_tensor, a_scales_tensor, b_scales_tensor, ctx)

    else:
        comptime assert False, (
            "Unsupported scaling mode: input_scale_granularity="
            + input_scale_granularity
            + ", weight_scale_granularity="
            + weight_scale_granularity
        )


fn naive_blockwise_scaled_fp8_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    BLOCK_DIM: Int = 16,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    scales_granularity_mnk: Optional[IndexList[3]] = None,
](
    c: LayoutTensor[
        mut=True, c_type, address_space = AddressSpace.GENERIC, ...
    ],
    a: LayoutTensor[
        mut=False, a_type, address_space = AddressSpace.GENERIC, ...
    ],
    b: LayoutTensor[
        mut=False, b_type, address_space = AddressSpace.GENERIC, ...
    ],
    a_scales: LayoutTensor[
        mut=False, a_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    b_scales: LayoutTensor[
        mut=False, b_scales_type, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    comptime assert a_type == b_type == DType.float8_e4m3fn, (
        "Only float8_e4m3fn is supported for input dtype for blockwise"
        " scaled fp8 matmul"
    )

    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be same"

    comptime assert (
        accum_type == DType.float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1)

    var a_scales_dim0 = a_scales.dim(0)
    var b_scales_dim0 = b_scales.dim(0)
    var b_scales_dim1 = b_scales.dim(1)

    if M == 0 or N == 0 or K == 0:
        return

    # these checks are only applicable when A_SCALES_SIZE and B_SCALES_SIZE are not provided
    comptime if not scales_granularity_mnk:
        if K % a_scales_dim0 != 0:
            raise Error(
                "K must be divisible by a_scales.dim(0) if A_SCALES_SIZE is not"
                " provided"
            )

        if transpose_b and (K % b_scales_dim1 != 0 or N % b_scales_dim0 != 0):
            raise Error(
                "K must be divisible by b_scales.dim(1) and N must be divisible"
                " by b_scales.dim(0) if B_SCALES_SIZE is not provided"
            )

        if not transpose_b and (
            K % b_scales_dim0 != 0 or N % b_scales_dim1 != 0
        ):
            raise Error(
                "K must be divisible by b_scales.dim(0) and N must be divisible"
                " by b_scales.dim(1) if B_SCALES_SIZE is not provided"
            )

    logger.info("Executing Naive Blockwise Scaled FP8 GEMM")
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]", sep="")
    logger.info(
        "A Scales Shape: [", a_scales.dim(0), ", ", a_scales.dim(1), "]", sep=""
    )
    logger.info(
        "B Scales Shape: [", b_scales.dim(0), ", ", b_scales.dim(1), "]", sep=""
    )

    comptime kernel = naive_blockwise_scaled_fp8_matmul_kernel[
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
        BLOCK_DIM=BLOCK_DIM,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ]

    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


fn naive_blockwise_scaled_fp8_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    c_shape: DimList,
    a_shape: DimList,
    b_shape: DimList,
    a_scale_shape: DimList,
    b_scale_shape: DimList,
    //,
    *,
    BLOCK_DIM: Int = 16,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    accum_type: DType = get_accum_type[c_type](),
    scales_granularity_mnk: Optional[IndexList[3]] = None,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    a_scales_device: NDBuffer[a_scales_type, 2, _, a_scale_shape],
    b_scales_device: NDBuffer[b_scales_type, 2, _, b_scale_shape],
    ctx: DeviceContext,
) raises:
    comptime assert a_type == b_type == DType.float8_e4m3fn, (
        "Only float8_e4m3fn is supported for input dtype for blockwise"
        " scaled fp8 matmul"
    )

    comptime assert (
        a_scales_type == b_scales_type
    ), "input A and B scales dtype should be same"

    comptime assert (
        accum_type == DType.float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)
    var a_scales = from_ndbuffer_row_major(a_scales_device)
    var b_scales = from_ndbuffer_row_major(b_scales_device)

    var M = c_device.dim(0)
    var N = c_device.dim(1)
    var K = a_device.dim(1)

    var a_scales_dim0 = a_scales.dim(0)
    var b_scales_dim0 = b_scales.dim(0)
    var b_scales_dim1 = b_scales.dim(1)

    if M == 0 or N == 0 or K == 0:
        return

    # these checks are only applicable when A_SCALES_SIZE and B_SCALES_SIZE are not provided
    comptime if not scales_granularity_mnk:
        if K % a_scales_dim0 != 0:
            raise Error(
                "K must be divisible by a_scales.dim(0) if A_SCALES_SIZE is not"
                " provided"
            )

        if transpose_b and (K % b_scales_dim1 != 0 or N % b_scales_dim0 != 0):
            raise Error(
                "K must be divisible by b_scales.dim(1) and N must be divisible"
                " by b_scales.dim(0) if B_SCALES_SIZE is not provided"
            )

        if not transpose_b and (
            K % b_scales_dim0 != 0 or N % b_scales_dim1 != 0
        ):
            raise Error(
                "K must be divisible by b_scales.dim(0) and N must be divisible"
                " by b_scales.dim(1) if B_SCALES_SIZE is not provided"
            )

    logger.info("Executing Naive Blockwise Scaled FP8 GEMM")
    logger.info("Problem Shape: MNK=[", M, ", ", N, ", ", K, "]", sep="")
    logger.info(
        "A Scales Shape: [", a_scales.dim(0), ", ", a_scales.dim(1), "]", sep=""
    )
    logger.info(
        "B Scales Shape: [", b_scales.dim(0), ", ", b_scales.dim(1), "]", sep=""
    )

    comptime kernel = naive_blockwise_scaled_fp8_matmul_kernel[
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
        BLOCK_DIM=BLOCK_DIM,
        transpose_b=transpose_b,
        elementwise_lambda_fn=elementwise_lambda_fn,
        scales_granularity_mnk=scales_granularity_mnk,
    ]

    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM), 1),
        block_dim=(BLOCK_DIM, BLOCK_DIM, 1),
    )


fn naive_blockwise_scaled_fp8_matmul_kernel[
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
    BLOCK_DIM: Int,
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    # Note: This is a naive kernel that supports a generalized blockwise scaled
    # fp8 matmul.
    # Currently, it supports two modes:
    # 1. [1 x SCALE_SIZE_K] x [SCALE_SIZE_K x SCALE_SIZE_N] blockwise scaling if
    #    scales_granularity_mnk is not provided. In this mode, the kernel will infer
    #    the scale sizes from the input and scale tensor shapes. The input shapes
    #    must be divisible by the scale sizes otherwise it will raise an error.
    # 2. [SCALE_SIZE_M x SCALE_SIZE_K] x [SCALE_SIZE_K x SCALE_SIZE_N] blockwise scaling if
    #    scales_granularity_mnk is provided. In this mode, the kernel will use the
    #    provided scale sizes to compute the scaled matmul.
    #
    # Assumptions:
    # 1. a should be always in K-major format
    # 2. b should be in K-major format if transpose_b is True otherwise it is in N-major format
    # 3. a_scales should be always in M-major format
    # 4. b_scales should be in K-major format if transpose_b is True otherwise it is in N-major format

    comptime assert (
        accum_type == DType.float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var M = c.dim(0)
    var N = c.dim(1)
    var K = a.dim(1)

    var x = global_idx.x
    var y = global_idx.y

    if x >= UInt(M) or y >= UInt(N):
        return

    var MAT_A_ROWS_SCALE_SIZE: UInt
    var MAT_A_COLS_SCALE_SIZE: UInt
    var MAT_B_ROWS_SCALE_SIZE: UInt
    var MAT_B_COLS_SCALE_SIZE: UInt

    comptime if scales_granularity_mnk:
        comptime scales_granularity = scales_granularity_mnk.value()
        MAT_A_ROWS_SCALE_SIZE = UInt(scales_granularity[2])
        MAT_A_COLS_SCALE_SIZE = UInt(scales_granularity[0])
        MAT_B_ROWS_SCALE_SIZE = UInt(
            scales_granularity[1]
        ) if transpose_b else UInt(scales_granularity[2])
        MAT_B_COLS_SCALE_SIZE = UInt(
            scales_granularity[2]
        ) if transpose_b else UInt(scales_granularity[1])

    else:
        var a_scale_0 = a_scales.dim(0)
        # var a_scale_1 = a_scales.dim(1)
        var b_scale_0 = b_scales.dim(0)
        var b_scale_1 = b_scales.dim(1)
        MAT_A_ROWS_SCALE_SIZE = UInt(K // a_scale_0)
        # MAT_A_COLS_SCALE_SIZE = UInt(M // a_scale_1)
        MAT_A_COLS_SCALE_SIZE = 1
        MAT_B_ROWS_SCALE_SIZE = UInt(N // b_scale_0) if transpose_b else UInt(
            K // b_scale_0
        )
        MAT_B_COLS_SCALE_SIZE = UInt(K // b_scale_1) if transpose_b else UInt(
            N // b_scale_1
        )

    var accum = Scalar[accum_type](0)
    for k in range(K):
        var a_val = rebind[Scalar[a_type]](a[x, k]).cast[accum_type]()
        var a_scale_factor = rebind[Scalar[a_scales_type]](
            a_scales[
                k // Int(MAT_A_ROWS_SCALE_SIZE), x // MAT_A_COLS_SCALE_SIZE
            ]
        ).cast[accum_type]()

        var b_val: Scalar[accum_type]
        var b_scale_factor: Scalar[accum_type]

        comptime if transpose_b:
            b_val = rebind[Scalar[b_type]](b[y, k]).cast[accum_type]()
            b_scale_factor = rebind[Scalar[b_scales_type]](
                b_scales[
                    y // MAT_B_ROWS_SCALE_SIZE, k // Int(MAT_B_COLS_SCALE_SIZE)
                ]
            ).cast[accum_type]()
        else:
            b_val = rebind[Scalar[b_type]](b[k, y]).cast[accum_type]()
            b_scale_factor = rebind[Scalar[b_scales_type]](
                b_scales[
                    k // Int(MAT_B_ROWS_SCALE_SIZE), y // MAT_B_COLS_SCALE_SIZE
                ]
            ).cast[accum_type]()

        accum += a_val * b_val * a_scale_factor * b_scale_factor

    comptime if elementwise_lambda_fn:
        comptime elementwise_lambda = elementwise_lambda_fn.value()
        elementwise_lambda[c_type, 1](Index(x, y), accum.cast[c_type]())
    else:
        c[x, y] = accum.cast[c_type]()


fn naive_blockwise_scaled_fp8_grouped_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    //,
    BLOCK_DIM_N: Int = 32,
    BLOCK_DIM_M: Int = 16,
    transpose_b: Bool = True,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
    a_offsets: LayoutTensor[a_offsets_type, a_offsets_layout, ImmutAnyOrigin],
    expert_ids: LayoutTensor[
        expert_ids_type, expert_ids_layout, ImmutAnyOrigin
    ],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    comptime accum_type = get_accum_type[a_type]()

    comptime assert (
        transpose_b
    ), "Only support transposed B in grouped fp8 matmul."
    comptime assert a_type == b_type == DType.float8_e4m3fn, (
        "Only float8_e4m3fn is supported for inputs in grouped blockwise"
        " scaled fp8 matmul"
    )
    comptime assert (
        accum_type == DType.float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    comptime assert a_offsets_type == DType.uint32, (
        "Only uint32 is supported for a_offsets in grouped blockwise scaled"
        " fp8 matmul"
    )
    comptime assert expert_ids_type == DType.int32, (
        "Only int32 is supported for expert_ids in grouped blockwise scaled"
        " fp8 matmul"
    )

    if max_num_tokens_per_expert == 0:
        return

    logger.info("Executing Naive Grouped Blockwise Scaled FP8 GEMM")

    comptime kernel = naive_blockwise_scaled_fp8_grouped_matmul_kernel[
        c_layout,
        a_layout,
        b_layout,
        a_scale_layout,
        b_scale_layout,
        a_offsets_layout,
        expert_ids_layout,
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        a_offsets_type,
        expert_ids_type,
        accum_type,
        transpose_b,
        scales_granularity_mnk,
        elementwise_lambda_fn,
    ]

    ctx.enqueue_function[kernel, kernel](
        c,
        a,
        b,
        a_offsets,
        expert_ids,
        a_scales,
        b_scales,
        grid_dim=(
            ceildiv(c.dim(1), BLOCK_DIM_N),
            ceildiv(max_num_tokens_per_expert, BLOCK_DIM_M),
            num_active_experts,
        ),
        block_dim=(BLOCK_DIM_N, BLOCK_DIM_M, 1),
    )


fn naive_blockwise_scaled_fp8_grouped_matmul_kernel[
    c_layout: Layout,
    a_layout: Layout,
    b_layout: Layout,
    a_scale_layout: Layout,
    b_scale_layout: Layout,
    a_offsets_layout: Layout,
    expert_ids_layout: Layout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    a_offsets_type: DType,
    expert_ids_type: DType,
    accum_type: DType,
    transpose_b: Bool = True,
    scales_granularity_mnk: Optional[IndexList[3]] = None,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, ImmutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, ImmutAnyOrigin],
    a_offsets: LayoutTensor[a_offsets_type, a_offsets_layout, ImmutAnyOrigin],
    expert_ids: LayoutTensor[
        expert_ids_type, expert_ids_layout, ImmutAnyOrigin
    ],
    a_scales: LayoutTensor[a_scales_type, a_scale_layout, ImmutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scale_layout, ImmutAnyOrigin],
):
    comptime assert (
        accum_type == DType.float32
    ), "Only float32 is supported for accumulation for scaled matmul"

    var N = b.dim[1]()
    var K = b.dim[2]()

    # Indices in current expert's matmul tile
    var n = Int(global_idx.x)
    var m_local = Int(global_idx.y)

    var expert_idx = Int(block_idx.z)

    # Determine rows for this expert
    var M_local: Int = Int(a_offsets[expert_idx + 1] - a_offsets[expert_idx])
    if n >= N or m_local >= M_local:
        return

    var MAT_A_ROWS_SCALE_SIZE: UInt
    var MAT_A_COLS_SCALE_SIZE: UInt
    var MAT_B_ROWS_SCALE_SIZE: UInt
    var MAT_B_COLS_SCALE_SIZE: UInt

    comptime if scales_granularity_mnk:
        comptime scales_granularity = scales_granularity_mnk.value()
        MAT_A_ROWS_SCALE_SIZE = UInt(scales_granularity[2])
        MAT_A_COLS_SCALE_SIZE = UInt(scales_granularity[0])
        MAT_B_ROWS_SCALE_SIZE = UInt(scales_granularity[1])
        MAT_B_COLS_SCALE_SIZE = UInt(scales_granularity[2])

    else:
        var a_s0 = a_scales.dim(0)
        var a_s1 = a_scales.dim(1)
        var b_s0 = b_scales.dim(1)
        var b_s1 = b_scales.dim(2)
        MAT_A_ROWS_SCALE_SIZE = UInt(K // a_s0)
        MAT_A_COLS_SCALE_SIZE = UInt(c.dim(0) // a_s1)
        MAT_B_ROWS_SCALE_SIZE = UInt(N // b_s0)
        MAT_B_COLS_SCALE_SIZE = UInt(K // b_s1)

    var a_start_row = Int(a_offsets[expert_idx])
    var expert = Int(expert_ids[expert_idx])
    var skip = expert == -1
    var accum = Scalar[accum_type](0)

    if not skip:
        var m_global = a_start_row + m_local
        var a_row_ptr = a.ptr + m_global * K
        var b_expert_ptr = b.ptr + expert * N * K
        for k in range(K):
            var a_val = rebind[Scalar[a_type]](a_row_ptr[k]).cast[accum_type]()
            var a_scale = rebind[Scalar[accum_type]](
                a_scales[
                    k // Int(MAT_A_ROWS_SCALE_SIZE),
                    m_global // Int(MAT_A_COLS_SCALE_SIZE),
                ]
            )
            var b_val = rebind[Scalar[b_type]](b_expert_ptr[n * K + k]).cast[
                accum_type
            ]()
            var b_scale = rebind[Scalar[accum_type]](
                b_scales[
                    UInt(expert),
                    n // Int(MAT_B_ROWS_SCALE_SIZE),
                    k // Int(MAT_B_COLS_SCALE_SIZE),
                ]
            )
            accum += a_val * b_val * a_scale * b_scale

    comptime if elementwise_lambda_fn:
        comptime ep = elementwise_lambda_fn.value()
        ep[c_type, 1](Index(a_start_row + m_local, n), accum.cast[c_type]())
    else:
        var c_ptr = c.ptr + a_start_row * N
        c_ptr[m_local * N + n] = accum.cast[c_type]()


########################################################
# FP8 E4M3FN to E4M3FNUZ Conversion for AMD GPUs
########################################################


@always_inline
fn convert_e4m3fn_to_e4m3fnuz(
    input_buffer: LayoutTensor[DType.float8_e4m3fn, ...],
    output_buffer: LayoutTensor[mut=True, DType.float8_e4m3fnuz, ...],
    context: DeviceContext,
) raises:
    """Convert E4M3FN weights to E4M3FNUZ format for AMD GPU compatibility.

    This conversion handles the key differences between E4M3FN and E4M3FNUZ:
    1. The bit pattern 10000000 (-128) represents zero in E4M3FN but NaN in E4M3FNUZ

    Args:
        input_buffer: Input tensor in E4M3FN format.
        output_buffer: Output tensor to store E4M3FNUZ format.
        context: Device context for kernel execution.
    """
    comptime assert (
        input_buffer.layout.shape == output_buffer.layout.shape
    ), "Input and output shapes must match"

    @always_inline
    @parameter
    @__copy_capture(input_buffer, output_buffer)
    fn convert_kernel[
        width: Int, rank: Int, alignment: Int = 1
    ](idx_arg: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var idx = rebind[IndexList[2]](idx_arg)

        var input_vec_e4m3fn = input_buffer.load[width=width](idx)
        var input_vec_int8 = bitcast[DType.int8](input_vec_e4m3fn)

        comptime ROCM_FP8_NAN_AS_INT = -128

        input_vec_int8 = input_vec_int8.eq(ROCM_FP8_NAN_AS_INT).select(
            Int8(0), input_vec_int8
        )
        var output_vec = bitcast[DType.float8_e4m3fnuz](input_vec_int8)
        output_buffer.store(idx, output_vec)

    comptime target_simd_width = simd_width_of[
        DType.float8_e4m3fn, target = get_gpu_target()
    ]()

    _elementwise_impl_gpu[
        func=convert_kernel, simd_width = UInt(target_simd_width)
    ](IndexList[2](input_buffer.dim[0](), input_buffer.dim[1]()), context)


########################################################
# SM100 Blockwise Scaled FP8 + FP32 with normal epilogue kernel dispatch
########################################################


fn blockwise_scaled_fp8_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    scales_granularity_mnk: IndexList[3],
    transpose_b: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[
        mut=True, c_type, _, _, address_space = AddressSpace.GENERIC, ...
    ],
    a: LayoutTensor[
        mut=False, a_type, _, _, address_space = AddressSpace.GENERIC, ...
    ],
    b: LayoutTensor[
        mut=False, b_type, _, _, address_space = AddressSpace.GENERIC, ...
    ],
    a_scales: LayoutTensor[
        mut=False,
        a_scales_type,
        _,
        _,
        address_space = AddressSpace.GENERIC,
        ...,
    ],
    b_scales: LayoutTensor[
        mut=False,
        b_scales_type,
        _,
        _,
        address_space = AddressSpace.GENERIC,
        ...,
    ],
    ctx: DeviceContext,
) raises:
    """Our sm100 blockwise scaled fp8 matmul kernel still does not support fusion of elementwise
    operations. This is a temporary implementation that uses our sm100 blockwise scaled fp8 matmul
    kernel and dispatch a separate epilogue kernel to apply the elementwise
    operations. For non B200 GPUs, we use the naive blockwise scaled fp8 matmul which support normal epilogue natively.
    """

    # 1D/2D (1x128)x(128x128) blockwise scaling
    comptime if (
        ctx.default_device_info == B200
        and transpose_b
        and c_type == DType.bfloat16
        and scales_granularity_mnk[0] == 1
        and scales_granularity_mnk[1] == scales_granularity_mnk[2] == 128
    ):
        comptime BK = 128
        comptime MMA_K = 32
        comptime block_tile_shape = Index(64, 96, BK)
        comptime umma_shape = Index(128, 192, MMA_K)
        comptime cluster_shape = Index(2, 1, 1)
        comptime matmul_config = MatmulConfig[
            a_type, b_type, c_type, transpose_b
        ](
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=umma_shape,
            cta_group=2,
        )

        comptime if not elementwise_lambda_fn:
            if not c.ptr:
                raise "c must be allocated!"

            blockwise_fp8_matmul[
                transpose_b=transpose_b,
                a_scales_type=a_scales_type,
                b_scales_type=b_scales_type,
                config=matmul_config,
            ](
                lt_to_tt(c),
                lt_to_tt(a),
                lt_to_tt(b),
                lt_to_tt(a_scales),
                lt_to_tt(b_scales),
                ctx,
            )
        else:
            comptime epilogue = elementwise_lambda_fn.value()
            # We hardcode simd width to 16B for Nvidia GPUs but >= sm_100
            # arch support 32B load/store to global memory, see KERN-2037.
            comptime use_32b_simd = (
                has_nvidia_gpu_accelerator()
                and ctx.default_device_info.compute >= B200.compute
            )
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
                epilogue[c_type, simd_width, alignment=alignment](
                    c_coord, c_val
                )

            # If c is already allocated, we can just use the sm100 blockwise scaled fp8 matmul and
            # apply the epilogue.
            if c.ptr:
                var m = c.dim[0]()
                var n = c.dim[1]()

                blockwise_fp8_matmul[
                    transpose_b=transpose_b,
                    a_scales_type=a_scales_type,
                    b_scales_type=b_scales_type,
                    config=matmul_config,
                ](
                    lt_to_tt(c),
                    lt_to_tt(a),
                    lt_to_tt(b),
                    lt_to_tt(a_scales),
                    lt_to_tt(b_scales),
                    ctx,
                )
                elementwise[epilogue_wrapper, simd_size, target="gpu"](
                    Index(m, n), ctx
                )
                return

            # Otherwise, we need to allocate a new buffer for c and apply the epilogue.
            var tmp_device_buffer = ctx.enqueue_create_buffer[c_type](c.size())
            var c_tmp = c
            c_tmp.ptr = tmp_device_buffer.unsafe_ptr()

            blockwise_scaled_fp8_with_epilogue[
                transpose_b=transpose_b,
                elementwise_lambda_fn=elementwise_lambda_fn,
                scales_granularity_mnk=scales_granularity_mnk,
            ](
                c_tmp,
                a,
                b,
                a_scales,
                b_scales,
                ctx,
            )

            _ = tmp_device_buffer^

    else:
        # For non B200 GPUs, we use the naive blockwise scaled fp8 matmul which support normal epilogue natively.
        naive_blockwise_scaled_fp8_matmul[
            transpose_b=transpose_b,
            scales_granularity_mnk=scales_granularity_mnk,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, b, a_scales, b_scales, ctx)
        return
