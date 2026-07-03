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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from layout import IntTuple
from linalg.fp8_quantization import convert_e4m3fn_to_e4m3fnuz
from nn.conv.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv.conv import pack_filter_shape as pack_filter_shape_conv
from nn.conv.conv_transpose import (
    conv_transpose_shape,
    conv_transposed_cpu,
    conv_transposed_gpu,
)
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pad_gpu import pad_constant as pad_constant_gpu
from nn.pool import avg_pool, pool_shape, pool_shape_ceil
from extensibility import InputTensor, OutputTensor
from extensibility import (
    _FusedOutputTensor as FusedOutputTensor,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils import IndexList
from std.utils.index import Index

# ===-----------------------------------------------------------------------===#
from .kernels import *
from .kernels import (
    _layout_transform_conv_filter_from_fcrs,
)


@compiler.register("mo.convert_e4m3fn_to_e4m3fnuz")
struct ConvertE4M3FNToE4M3FNUZ:
    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: OutputTensor[dtype=DType.float8_e4m3fnuz, rank=2, ...],
        input: InputTensor[dtype=DType.float8_e4m3fn, rank=2, ...],
        ctx: DeviceContext,
    ) raises:
        convert_e4m3fn_to_e4m3fnuz(
            input.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            ctx,
        )


@compiler.register_shape_function("mo.convert_e4m3fn_to_e4m3fnuz")
def convert_e4m3fn_to_e4m3fnuz_shape(
    input: InputTensor[dtype=DType.float8_e4m3fn, rank=2, ...],
) -> IndexList[2]:
    return IndexList[2](input.dim_size[0](), input.dim_size[1]())


@compiler.register("mo.avg_pool")
struct AvgPool:
    @staticmethod
    def execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            False,
            ctx,
        )


@compiler.register_shape_function("mo.avg_pool")
def avg_pool_shape[
    dtype: DType,
    int_type: DType,
](
    input: InputTensor[dtype=dtype, rank=4, ...],
    filter: InputTensor[dtype=int_type, rank=1, ...],
    strides: InputTensor[dtype=int_type, rank=1, ...],
    dilations: InputTensor[dtype=int_type, rank=1, ...],
    paddings: InputTensor[dtype=int_type, rank=1, ...],
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        pool_shape(
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    @staticmethod
    def execute[
        count_boundary: Bool,
        dtype: DType,
        int_type: DType,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=4, ...],
        input: InputTensor[dtype=dtype, rank=4, ...],
        filter: InputTensor[dtype=int_type, rank=1, ...],
        strides: InputTensor[dtype=int_type, rank=1, ...],
        dilations: InputTensor[dtype=int_type, rank=1, ...],
        paddings: InputTensor[dtype=int_type, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output.to_tile_tensor[DType.int64](),
            True,
            ctx,
        )


@compiler.register_shape_function("mo.avg_pool_ceil_mode_true")
def avg_pool_ceil_mode_true_shape[
    dtype: DType,
    int_type: DType,
](
    input: InputTensor[dtype=dtype, rank=4, ...],
    filter: InputTensor[dtype=int_type, rank=1, ...],
    strides: InputTensor[dtype=int_type, rank=1, ...],
    dilations: InputTensor[dtype=int_type, rank=1, ...],
    paddings: InputTensor[dtype=int_type, rank=1, ...],
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        pool_shape_ceil(
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("mo.pad.constant")
struct PadConstant:
    @staticmethod
    def execute[
        dtype: DType, rank: Int, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
        constant: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) raises:
        var paddings_ptr = padding._ptr

        comptime if is_cpu[target]():
            pad_constant(
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                paddings_ptr,
                constant,
            )
        elif is_gpu[target]():
            pad_constant_gpu(
                output._ptr,
                output.shape(),
                input._ptr,
                input.shape(),
                paddings_ptr,
                constant,
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target


@compiler.register_shape_function("mo.pad.constant")
def pad_constant_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    padding: InputTensor[rank=1, ...],
    constant: Scalar[dtype=dtype],
) raises -> IndexList[rank]:
    # rebind is required because mojo can't figure out that
    # input.static_spec.to_layout_tensor().rank == input.rank
    return rebind[IndexList[rank]](
        pad_shape(
            input.to_tile_tensor[DType.int64](),
            padding.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("mo.pad.repeat")
struct PadRepeat:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ):
        var paddings_ptr = padding._ptr
        pad_repeat(
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            paddings_ptr,
        )


@compiler.register_shape_function("mo.pad.repeat")
def pad_repeat_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    padding: InputTensor[rank=1, ...],
) raises -> IndexList[rank]:
    return rebind[IndexList[rank]](
        pad_shape(
            input.to_tile_tensor[DType.int64](),
            padding.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("mo.pad.reflect")
struct PadReflect:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        padding: InputTensor[rank=1, ...],
    ):
        var paddings_ptr = padding._ptr
        pad_reflect(
            output.to_tile_tensor[DType.int64](),
            input.to_tile_tensor[DType.int64](),
            paddings_ptr,
        )


@compiler.register_shape_function("mo.pad.reflect")
def pad_reflect_shape[
    dtype: DType,
    rank: Int,
](
    input: InputTensor[dtype=dtype, rank=rank, ...],
    padding: InputTensor[rank=1, ...],
) raises -> IndexList[rank]:
    return rebind[IndexList[rank]](
        pad_shape(
            input.to_tile_tensor[DType.int64](),
            padding.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("mo.conv")
struct Conv:
    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        has_epilogue_fusion: Bool,
        static_strides: IntTuple,
        static_dilations: IntTuple,
        static_padding: IntTuple,
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[rank=output.rank, ...],
        filter: InputTensor,
        strides: InputTensor,
        dilation: InputTensor,
        paddings: InputTensor,
        num_groups: Scalar,
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        @__copy_capture(output)
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        comptime assert (
            strides.dtype.is_integral() and dilation.dtype.is_integral()
        ), "stride and dilation must have integral type"

        comptime assert (
            input_layout == "NHWC"
        ), "only NHWC input layout is supported"

        if strides.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv strides")

        if dilation.size() != input.rank - 2:
            raise Error("(input_rank-2) values expected in conv dilation")

        if paddings.size() != 2 * (input.rank - 2):
            raise Error("(2*(input_rank-2)) value expected in conv paddings")

        var stride_tuple = IndexList[input.rank - 2](0)
        var dilation_tuple = IndexList[input.rank - 2](0)

        comptime for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        if dilation_tuple != IndexList[input.rank - 2](1):
            raise Error("Non-unit dilation is not supported yet.")

        var pad_d_tuple = IndexList[2](0)
        var pad_h_tuple = IndexList[2](0)
        var pad_w_tuple = IndexList[2](0)

        comptime if input.rank == 3:
            pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
        elif input.rank == 4:
            pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
        elif input.rank == 5:
            pad_d_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_h_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            pad_w_tuple = Index(paddings._ptr[4], paddings._ptr[5])

        comptime input_shape_val = Int(
            input._static_shape_tuple[input.rank - 1]
        )  # input C, NHWC
        comptime filter_shape_val = Int(
            filter._static_shape_tuple[filter.rank - 2]
        )  # filter C, RSCF or FRSCf
        comptime conv_attr = ConvInfoStatic[input.rank - 2](
            static_padding,
            static_strides,
            static_dilations,
            input_shape_val,
            filter_shape_val,
        )

        comptime filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        comptime filter_is_fcrs = filter_layout == "FCRS"

        var input_tt = input.to_tile_tensor[DType.int64]()
        var filter_tt = filter.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()

        comptime if is_cpu[target]():
            comptime assert (
                not filter_is_fcrs
            ), "Filter layout FCRS is not supported on CPU"
            # Pass LayoutTensor layouts explicitly so ConvDirectNHWC gets the
            # same compile-time shape/stride info as before the TileTensor
            # migration.
            comptime _input_layout = input.static_spec.to_layout()
            comptime _filter_layout = filter.static_spec.to_layout()
            comptime _output_layout = output.static_spec.to_layout()
            conv_nhwc_direct[
                _input_layout,
                _filter_layout,
                _output_layout,
                input.dtype,
                filter.dtype,
                output.dtype,
                filter_packed,
                conv_attr,
                has_epilogue_fusion,
                output_fn,
            ](
                input_tt,
                filter_tt,
                output_tt,
                stride_tuple,
                dilation_tuple,
                pad_d_tuple,
                pad_h_tuple,
                pad_w_tuple,
                Int(num_groups),
                Optional[DeviceContext](ctx),
            )
        else:
            comptime assert (input.rank == 4 and filter.rank == 4) or (
                input.rank == 5 and filter.rank == 5
            ), "only rank 4 or 5 tensor is supported on cuda gpu"
            comptime assert (
                filter_packed == False
            ), "only unpacked filter is supported on cuda gpu"

            var pad_tuple = IndexList[2 * (input.rank - 2)](0)

            comptime for i in range(2 * (input.rank - 2)):
                pad_tuple[i] = Int(paddings._ptr[i])

            conv_gpu[
                input.dtype,
                filter.dtype,
                output.dtype,
                output_fn,
                filter_is_fcrs,
            ](
                input_tt,
                filter_tt,
                output_tt,
                stride_tuple,
                dilation_tuple,
                pad_tuple,
                Int(num_groups),
                ctx,
            )


@compiler.register_shape_function("mo.conv")
def mo_conv_shape(
    input: InputTensor,
    filter: InputTensor,
    strides: InputTensor[rank=1, ...],
    dilations: InputTensor[rank=1, ...],
    paddings: InputTensor[rank=1, ...],
    num_groups: Scalar,
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        conv_shape(
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            num_groups,
        )
    )


@compiler.register("conv2d_residual_add")
struct Conv2dResidualAdd:
    """Fused conv2d + TMA residual add + bias for SM100 (Blackwell).

    Computes: D = Conv(input, filter) + bias + source
    The residual (source) is loaded via TMA pre-fetch overlapped with MMA,
    and the bias is applied in the epilogue.

    This op is intended for ResNet-style skip connections where a residual
    tensor is added to the convolution output.
    """

    @staticmethod
    def execute[
        stride_h: Int,
        stride_w: Int,
        pad_top: Int,
        pad_bottom: Int,
        pad_left: Int,
        pad_right: Int,
        has_bias: Bool,
        target: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[dtype=output.dtype, rank=4, ...],
        filter: InputTensor[rank=4, ...],
        source: InputTensor[dtype=output.dtype, rank=4, ...],
        bias: InputTensor[dtype=output.dtype, rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        @parameter
        @always_inline
        @__copy_capture(output, bias)
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            var result = val

            comptime if has_bias:
                var c_idx = coords[_rank - 1]
                var bias_vec = (bias.unsafe_ptr() + c_idx).load[width=_width]()
                result = val + bias_vec.cast[_dtype]()

            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](result),
            )

        comptime assert not is_cpu[
            target
        ](), "conv2d_residual_add is only supported on GPU"

        var input_tt = input.to_tile_tensor[DType.int64]()
        var filter_tt = filter.to_tile_tensor[DType.int64]()
        var output_tt = output.to_tile_tensor[DType.int64]()

        var pad_tuple = IndexList[4](pad_top, pad_bottom, pad_left, pad_right)
        var stride_tuple = IndexList[2](stride_h, stride_w)
        var dilation_tuple = IndexList[2](1, 1)

        conv_gpu[
            input.dtype,
            filter.dtype,
            output.dtype,
            output_fn,
            True,  # filter_is_fcrs
            has_residual=True,
        ](
            input_tt,
            filter_tt,
            output_tt,
            stride_tuple,
            dilation_tuple,
            pad_tuple,
            1,  # num_groups
            ctx,
            source.unsafe_ptr().as_unsafe_any_origin(),
            Float32(1.0),  # beta
        )


@compiler.register_shape_function("conv2d_residual_add")
def conv2d_residual_add_shape(
    input: InputTensor[rank=4, ...],
    filter: InputTensor[rank=4, ...],
    source: InputTensor[rank=4, ...],
    bias: InputTensor[rank=1, ...],
) raises -> IndexList[4]:
    # Output shape is the same as source shape (residual tensor).
    return source.shape()


@compiler.register("mo.conv_transpose")
struct ConvTranspose:
    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        has_epilogue_fusion: Bool,
        target: StaticString,
    ](
        output: FusedOutputTensor[...],
        input: InputTensor[rank=output.rank, ...],
        filter: InputTensor,
        strides: InputTensor[rank=1, ...],
        dilation: InputTensor[rank=1, ...],
        paddings: InputTensor[rank=1, ...],
        output_paddings: InputTensor[rank=1, ...],
        ctx: DeviceContext,
    ) capturing raises:
        comptime assert (
            strides.dtype.is_integral()
            and dilation.dtype.is_integral()
            and output_paddings.dtype.is_integral()
        )

        if strides.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose stride"
            )

        if dilation.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose dilation"
            )

        if output_paddings.size() != input.rank - 2:
            raise Error(
                "(input_rank-2) values expected in convTranspose output"
                " paddings"
            )

        if paddings.size() != 2 * (input.rank - 2):
            raise Error(
                "(2*(input_rank-2)) value expected in convTranspose paddings"
            )

        var stride_tuple = IndexList[
            type_of(input.to_tile_tensor[DType.int64]()).rank - 2
        ](0)
        var dilation_tuple = IndexList[
            type_of(input.to_tile_tensor[DType.int64]()).rank - 2
        ](0)

        comptime for i in range(input.rank - 2):
            stride_tuple[i] = Int(strides._ptr[i])
            dilation_tuple[i] = Int(dilation._ptr[i])

        var pad_d = IndexList[2](0)
        var pad_h = IndexList[2](0)
        var pad_w = IndexList[2](0)

        comptime if input.rank == 3:
            pad_w = Index(paddings[0], paddings[1])
        elif input.rank == 4:
            pad_h = Index(paddings[0], paddings[1])
            pad_w = Index(paddings[2], paddings[3])
        elif input.rank == 5:
            pad_d = Index(paddings[0], paddings[1])
            pad_h = Index(paddings[2], paddings[3])
            pad_w = Index(paddings[4], paddings[5])

        @parameter
        @always_inline
        def output_fn[
            _dtype: DType, _rank: Int, _width: SIMDSize, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            output._lambda_store[width=_width, element_alignment=_alignment](
                rebind[IndexList[output.rank]](coords),
                rebind[SIMD[output.dtype, _width]](val),
            )

        comptime filter_packed = filter_layout == "FRSCf" or filter_layout == "FQRSCf"
        comptime filter_is_cfrs = filter_layout == "CFRS"

        comptime if is_cpu[target]():
            conv_transposed_cpu[
                filter_packed,
                filter_is_cfrs,
                has_epilogue_fusion,
                output_fn,
            ](
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                stride_tuple,
                dilation_tuple,
                pad_d,
                pad_h,
                pad_w,
                Optional[DeviceContext](ctx),
            )
        else:
            comptime assert (
                input.rank == 4 and filter.rank == 4
            ), "only rank 4 tensor is supported on cuda gpu"
            comptime assert (
                filter_packed == False
            ), "only unpacked filter is supported on cuda gpu"

            var pad_tuple = IndexList[
                type_of(input.to_tile_tensor[DType.int64]()).rank - 2
            ](0)

            comptime if input.rank == 4:
                pad_tuple[0] = pad_h[0]
                pad_tuple[1] = pad_w[0]

            conv_transposed_gpu[
                input.dtype,
                filter.dtype,
                output.dtype,
                elementwise_epilogue=Optional[elementwise_simd_epilogue_type](
                    output_fn
                ) if has_epilogue_fusion else Optional[
                    elementwise_simd_epilogue_type
                ](),
            ](
                output.to_tile_tensor[DType.int64](),
                input.to_tile_tensor[DType.int64](),
                filter.to_tile_tensor[DType.int64](),
                stride_tuple,
                dilation_tuple,
                pad_tuple,
                ctx,
            )


@compiler.register_shape_function("mo.conv_transpose")
def mo_conv_transpose_shape[
    dtype: DType
](
    input: InputTensor[dtype=dtype, ...],
    filter: InputTensor[dtype=dtype, ...],
    strides: InputTensor[rank=1, ...],
    dilations: InputTensor[rank=1, ...],
    paddings: InputTensor[rank=1, ...],
    output_paddings: InputTensor[rank=1, ...],
) raises -> IndexList[input.rank]:
    return rebind[IndexList[input.rank]](
        conv_transpose_shape(
            input.to_tile_tensor[DType.int64](),
            filter.to_tile_tensor[DType.int64](),
            strides.to_tile_tensor[DType.int64](),
            dilations.to_tile_tensor[DType.int64](),
            paddings.to_tile_tensor[DType.int64](),
            output_paddings.to_tile_tensor[DType.int64](),
        )
    )


@compiler.register("layout_transform_RSFC_to_FRSCf")
struct LayoutTransformRSFC2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("layout_transform_QRSFC_to_FQRSCf")
struct LayoutTransformQRSFC2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@compiler.register("pack_conv_filter_shape")
struct PackConvFilterShape:
    @always_inline
    @staticmethod
    def execute(filter_buf: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")


@compiler.register_shape_function("pack_conv_filter_shape")
def pack_conv_filter_shape_fn[
    rank: Int,
    filter_type: DType,
    input_shape: IntTuple,
    filter_shape: IntTuple,
    output_shape: IntTuple,
    strides: IntTuple,
    dilations: IntTuple,
    paddings: IntTuple,
    num_groups: Int,
](filter_buf: InputTensor[dtype=filter_type, rank=rank, ...]) -> IndexList[
    rank + 1
]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
        rank: Rank of the un-packed filter.
        filter_type: Type of the filter.
        input_shape: NHWC layout.
        filter_shape: Filter shape.
        output_shape: NHWC layout.
        strides: Should be rank 1 size 2.
        dilations: Should be rank 1 size 2.
        paddings: Should be rank 1 size 4.
        num_groups: The number of groups in the convolution.

    Args:
        filter_buf: The filter to be packed.

    Returns:
        The output shape.
    """

    return rebind[IndexList[rank + 1]](
        pack_filter_shape_conv[
            filter_type,
            input_shape,
            filter_shape,
            output_shape,
            strides,
            dilations,
            paddings,
            num_groups,
        ](filter_buf.to_tile_tensor[DType.int64]())
    )


@compiler.register("layout_transform_QRSCF_to_FQRSCf")
struct LayoutTransformQRSCF2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_RSCF_to_FRSCf")
struct LayoutTransformRSCF2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_filter_common[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_FCRS_to_FRSCf")
struct LayoutTransformFCRS2FRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        _layout_transform_conv_filter_from_fcrs[num_groups=num_groups](
            packed_filter, filter
        )


@compiler.register("layout_transform_FCQRS_to_FQRSCf")
struct LayoutTransformFCQRS2FQRSCf:
    @always_inline
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_rank: Int, num_groups: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        _layout_transform_conv_filter_from_fcrs[num_groups=num_groups](
            packed_filter, filter
        )
