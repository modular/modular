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

"""Registers convolution graph ops and dispatches them to the `nn.conv` kernels."""

import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#

from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu, is_gpu
from layout import Coord, IntTuple, TileTensor, coord_to_index_list, row_major
from linalg.fp8_quantization import convert_e4m3fn_to_e4m3fnuz
from nn.conv.conv import ConvInfoStatic, conv_gpu, conv_nhwc_direct, conv_shape
from nn.conv.conv import pack_filter_shape as pack_filter_shape_conv
from nn.conv.conv_transpose import (
    conv_transpose_shape,
    conv_transposed_cpu,
    conv_transposed_gpu,
)
from nn.conv.conv_utils import ConvSimdEpilogueFn, no_conv_simd_epilogue_fn
from nn.pad import pad_constant, pad_reflect, pad_repeat, pad_shape
from nn.pad_gpu import (
    pad_constant as pad_constant_gpu,
    pad_reflect as pad_reflect_gpu,
    pad_repeat as pad_repeat_gpu,
)
from nn.pool import avg_pool, pool_shape, pool_shape_ceil
from extensibility import (
    Fused,
    Input,
    InputTensor,
    Output,
    OutputTensor,
    Storable,
    StoreAccess,
    Tensor,
    TileTensorable,
    UnsafeFusedView,
    unsafe_fused_view,
)
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


@extensibility.register("mo.convert_e4m3fn_to_e4m3fnuz")
struct ConvertE4M3FNToE4M3FNUZ:
    """Registers the `mo.convert_e4m3fn_to_e4m3fnuz` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        target: StaticString,
        _trace_name: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert type_of(output).rank == 2
        comptime assert type_of(input).rank == 2
        convert_e4m3fn_to_e4m3fnuz(
            input.to_tile_tensor[.float8_e4m3fn](),
            output.to_tile_tensor[.float8_e4m3fnuz](),
            ctx,
        )


@extensibility.register_shape_function("mo.convert_e4m3fn_to_e4m3fnuz")
def convert_e4m3fn_to_e4m3fnuz_shape(
    input: Some[Tensor],
) -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.convert_e4m3fn_to_e4m3fnuz` graph op.
    """
    comptime assert (
        type_of(input).dtype == .float8_e4m3fn
    ), "input dtype must be float8_e4m3fn"
    comptime assert type_of(input).rank == 2, "input must be rank 2"
    return rebind[IndexList[type_of(input).rank]](
        coord_to_index_list(input.shape())
    )


@extensibility.register("mo.avg_pool")
struct AvgPool:
    """Registers the `mo.avg_pool` graph op with the graph compiler."""

    @staticmethod
    def execute[
        count_boundary: Bool,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilations: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilations), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime int_type = type_of(filter).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 4
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[dtype](),
            filter.to_tile_tensor[int_type](),
            strides.to_tile_tensor[int_type](),
            dilations.to_tile_tensor[int_type](),
            paddings.to_tile_tensor[int_type](),
            output.to_tile_tensor(),
            False,
            ctx,
        )


@extensibility.register_shape_function("mo.avg_pool")
def avg_pool_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.avg_pool` graph op."""
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(filter).dtype == type_of(strides).dtype
    ), "filter dtype must match strides dtype"
    comptime assert (
        type_of(filter).dtype == type_of(dilations).dtype
    ), "filter dtype must match dilations dtype"
    comptime assert (
        type_of(filter).dtype == type_of(paddings).dtype
    ), "filter dtype must match paddings dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


@extensibility.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    """Registers the `mo.avg_pool_ceil_mode_true` graph op with the graph compiler.
    """

    @staticmethod
    def execute[
        count_boundary: Bool,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilations: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilations), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime dtype = type_of(output).dtype
        comptime int_type = type_of(filter).dtype
        comptime assert type_of(output).rank == 4
        comptime assert type_of(input).rank == 4
        avg_pool[count_boundary=count_boundary, target=target](
            input.to_tile_tensor[dtype](),
            filter.to_tile_tensor[int_type](),
            strides.to_tile_tensor[int_type](),
            dilations.to_tile_tensor[int_type](),
            paddings.to_tile_tensor[int_type](),
            output.to_tile_tensor(),
            True,
            ctx,
        )


@extensibility.register_shape_function("mo.avg_pool_ceil_mode_true")
def avg_pool_ceil_mode_true_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.avg_pool_ceil_mode_true` graph op.
    """
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 1, "filter must be rank 1"
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(filter).dtype == type_of(strides).dtype
    ), "filter dtype must match strides dtype"
    comptime assert (
        type_of(filter).dtype == type_of(dilations).dtype
    ), "filter dtype must match dilations dtype"
    comptime assert (
        type_of(filter).dtype == type_of(paddings).dtype
    ), "filter dtype must match paddings dtype"
    return rebind[IndexList[type_of(input).rank]](
        pool_shape_ceil(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
        )
    )


@extensibility.register("mo.pad.constant")
struct PadConstant:
    """Registers the `mo.pad.constant` graph op with the graph compiler."""

    @staticmethod
    def execute[
        dtype: DType, target: StaticString
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        padding: Some[Input & Tensor],
        constant: Scalar[dtype=dtype],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(padding), TileTensorable)
        comptime rank = type_of(output).rank
        comptime out_dtype = type_of(output).dtype
        comptime assert out_dtype == dtype
        comptime assert type_of(input).rank == rank
        comptime assert type_of(padding).rank == 1

        var output_tile = output.to_tile_tensor()
        # The helpers take one dtype for both tiles, so the input restates
        # the output's dtype expression (MOCO-4355 stops the checker seeing
        # that the two are equal).
        var input_tile = input.to_tile_tensor[out_dtype]()
        # `paddings` stays a raw pointer: the pad kernels index it directly
        # rather than as a tensor. Under a storage policy that cannot produce a
        # flat pointer this aborts at run time rather than compile time.
        var paddings_ptr = padding.to_tile_tensor().ptr

        comptime if is_cpu[target]():
            pad_constant(
                output_tile,
                input_tile,
                paddings_ptr,
                constant,
            )
        elif is_gpu[target]():
            pad_constant_gpu(
                output_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(output_tile.layout.shape_coord())
                ),
                input_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(input_tile.layout.shape_coord())
                ),
                paddings_ptr,
                rebind[Scalar[out_dtype]](constant),
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target


@extensibility.register_shape_function("mo.pad.constant")
def pad_constant_shape[
    constant_dtype: DType,
](
    input: Some[TileTensorable],
    padding: Some[TileTensorable],
    # `constant` binds its own dtype parameter. The graph compiler's
    # scalar-attribute binding cannot resolve a trait projection like
    # `type_of(input).dtype`. The `comptime assert` below restores the tie.
    constant: Scalar[dtype=constant_dtype],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.pad.constant` graph op."""
    comptime assert type_of(padding).rank == 1, "padding must be rank 1"
    comptime assert (
        constant_dtype == type_of(input).dtype
    ), "constant dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        pad_shape(
            input.to_tile_tensor(),
            padding.to_tile_tensor(),
        )
    )


@extensibility.register("mo.pad.repeat")
struct PadRepeat:
    """Registers the `mo.pad.repeat` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        padding: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(padding), TileTensorable)
        comptime rank = type_of(output).rank
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == rank
        comptime assert type_of(padding).rank == 1

        var output_tile = output.to_tile_tensor()
        var input_tile = input.to_tile_tensor[dtype]()
        # See the note in `mo.pad.constant` on the raw `paddings` pointer.
        var paddings_ptr = padding.to_tile_tensor().ptr

        comptime if is_cpu[target]():
            pad_repeat(output_tile, input_tile, paddings_ptr)
        elif is_gpu[target]():
            pad_repeat_gpu(
                output_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(output_tile.layout.shape_coord())
                ),
                input_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(input_tile.layout.shape_coord())
                ),
                paddings_ptr,
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target


@extensibility.register_shape_function("mo.pad.repeat")
def pad_repeat_shape(
    input: Some[TileTensorable], padding: Some[TileTensorable]
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.pad.repeat` graph op."""
    comptime assert type_of(padding).rank == 1, "padding must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        pad_shape(
            input.to_tile_tensor(),
            padding.to_tile_tensor(),
        )
    )


@extensibility.register("mo.pad.reflect")
struct PadReflect:
    """Registers the `mo.pad.reflect` graph op with the graph compiler."""

    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        padding: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(padding), TileTensorable)
        comptime rank = type_of(output).rank
        comptime dtype = type_of(output).dtype
        comptime assert type_of(input).rank == rank
        comptime assert type_of(padding).rank == 1

        var output_tile = output.to_tile_tensor()
        var input_tile = input.to_tile_tensor[dtype]()
        # See the note in `mo.pad.constant` on the raw `paddings` pointer.
        var paddings_ptr = padding.to_tile_tensor().ptr

        comptime if is_cpu[target]():
            pad_reflect(output_tile, input_tile, paddings_ptr)
        elif is_gpu[target]():
            pad_reflect_gpu(
                output_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(output_tile.layout.shape_coord())
                ),
                input_tile.ptr,
                rebind[IndexList[rank]](
                    coord_to_index_list(input_tile.layout.shape_coord())
                ),
                paddings_ptr,
                ctx,
            )
        else:
            comptime assert False, "Unknown target " + target


@extensibility.register_shape_function("mo.pad.reflect")
def pad_reflect_shape(
    input: Some[TileTensorable], padding: Some[TileTensorable]
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.pad.reflect` graph op."""
    comptime assert type_of(padding).rank == 1, "padding must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        pad_shape(
            input.to_tile_tensor(),
            padding.to_tile_tensor(),
        )
    )


def _index_list[size: Int](values: InputTensor) -> IndexList[size]:
    var result = IndexList[size](0)

    comptime for i in range(size):
        result[i] = Int(values._ptr[i])
    return result


def _check_conv_attrs[
    spatial_rank: Int
](strides: InputTensor, dilation: InputTensor, paddings: InputTensor) raises:
    comptime assert (
        strides.dtype.is_integral() and dilation.dtype.is_integral()
    ), "stride and dilation must have integral type"

    if strides.size() != spatial_rank:
        raise Error("(input_rank-2) values expected in conv strides")

    if dilation.size() != spatial_rank:
        raise Error("(input_rank-2) values expected in conv dilation")

    if paddings.size() != 2 * spatial_rank:
        raise Error("(2*(input_rank-2)) value expected in conv paddings")


@extensibility.register("mo.conv", type="cpu")
struct ConvCPU:
    """Registers the `mo.conv` graph op for CPU devices."""

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
    ) raises:
        @inline(.always)
        def output_fn[
            _dtype: DType, _width: SIMDLength, _alignment: Int = 1
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output}:
            output._lambda_store[width=_width, element_alignment=_alignment](
                coords, rebind[SIMD[output.dtype, _width]](val)
            )

        comptime assert (
            input_layout == "NHWC"
        ), "only NHWC input layout is supported"
        comptime assert (
            filter_layout != "FCRS"
        ), "Filter layout FCRS is not supported on CPU"

        _check_conv_attrs[input.rank - 2](strides, dilation, paddings)

        var pad_d_tuple = IndexList[2](0)
        var pad_h_tuple = IndexList[2](0)
        var pad_w_tuple = IndexList[2](0)

        comptime match input.rank:
            case 3:
                pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            case 4:
                pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
                pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            case 5:
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

        conv_nhwc_direct[
            input.dtype,
            filter.dtype,
            output.dtype,
            filter_layout == "FRSCf" or filter_layout == "FQRSCf",
            conv_attr,
            has_epilogue_fusion,
        ](
            input.to_tile_tensor[.int64](),
            filter.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            _index_list[input.rank - 2](strides),
            _index_list[input.rank - 2](dilation),
            pad_d_tuple,
            pad_h_tuple,
            pad_w_tuple,
            Int(num_groups),
            output_fn,
            Optional(ctx.assert_cpu_target()),
        )


@extensibility.register("mo.conv", type="gpu")
struct ConvGPU:
    """Registers the `mo.conv` graph op for GPU devices."""

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
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
    ) raises:
        @inline(.always)
        def output_fn[
            _dtype: DType, _width: SIMDLength, _alignment: Int = 1
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output}:
            output._lambda_store[width=_width, element_alignment=_alignment](
                coords, rebind[SIMD[output.dtype, _width]](val)
            )

        comptime assert (
            input_layout == "NHWC"
        ), "only NHWC input layout is supported"
        comptime assert (input.rank == 4 and filter.rank == 4) or (
            input.rank == 5 and filter.rank == 5
        ), "only rank 4 or 5 tensor is supported on cuda gpu"
        comptime assert (
            filter_layout != "FRSCf" and filter_layout != "FQRSCf"
        ), "only unpacked filter is supported on cuda gpu"

        _check_conv_attrs[input.rank - 2](strides, dilation, paddings)

        conv_gpu[
            input.dtype,
            filter.dtype,
            output.dtype,
            has_epilogue_fn=True,
            filter_is_fcrs=filter_layout == "FCRS",
        ](
            input.to_tile_tensor[.int64](),
            filter.to_tile_tensor[.int64](),
            output.to_tile_tensor[.int64](),
            _index_list[input.rank - 2](strides),
            _index_list[input.rank - 2](dilation),
            _index_list[2 * (input.rank - 2)](paddings),
            Int(num_groups),
            ctx,
            epilogue_fn=output_fn,
        )


@extensibility.register_shape_function("mo.conv")
def mo_conv_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
    num_groups: Scalar,
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.conv` graph op."""
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    return rebind[IndexList[type_of(input).rank]](
        conv_shape(
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
            num_groups,
        )
    )


@extensibility.register("conv2d_residual_add")
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
    ) raises:
        @inline(.always)
        def output_fn[
            _dtype: DType, _width: SIMDLength, _alignment: Int = 1
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output, var bias}:
            var result = val

            comptime if has_bias:
                var c_idx = Int(coords[coords.rank - 1])
                var bias_vec = (
                    bias.unsafe_ptr()
                    .unsafe_offset(c_idx)
                    .unsafe_load[width=_width]()
                )
                result = val + bias_vec.cast[_dtype]()

            output._lambda_store[width=_width, element_alignment=_alignment](
                coords, rebind[SIMD[output.dtype, _width]](result)
            )

        comptime assert not is_cpu[
            target
        ](), "conv2d_residual_add is only supported on GPU"

        var input_tt = input.to_tile_tensor[.int64]()
        var filter_tt = filter.to_tile_tensor[.int64]()
        var output_tt = output.to_tile_tensor[.int64]()

        var pad_tuple = IndexList[4](pad_top, pad_bottom, pad_left, pad_right)
        var stride_tuple = IndexList[2](stride_h, stride_w)
        var dilation_tuple = IndexList[2](1, 1)

        conv_gpu[
            input.dtype,
            filter.dtype,
            output.dtype,
            has_epilogue_fn=True,
            filter_is_fcrs=True,
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
            output_fn,
        )


@extensibility.register_shape_function("conv2d_residual_add")
def conv2d_residual_add_shape(
    input: Some[Tensor],
    filter: Some[Tensor],
    source: Some[Tensor],
    bias: Some[Tensor],
) raises -> IndexList[type_of(source).rank]:
    """Computes the output shape for the `conv2d_residual_add` graph op."""
    comptime assert type_of(input).rank == 4, "input must be rank 4"
    comptime assert type_of(filter).rank == 4, "filter must be rank 4"
    comptime assert type_of(source).rank == 4, "source must be rank 4"
    comptime assert type_of(bias).rank == 1, "bias must be rank 1"
    # Output shape is the same as source shape (residual tensor).
    return rebind[IndexList[type_of(source).rank]](
        coord_to_index_list(source.shape())
    )


def _check_conv_transpose_attrs[
    spatial_rank: Int
](
    strides: TileTensor,
    dilation: TileTensor,
    paddings: TileTensor,
    output_paddings: TileTensor,
) raises:
    comptime assert (
        strides.rank == 1
        and dilation.rank == 1
        and paddings.rank == 1
        and output_paddings.rank == 1
    )
    comptime assert (
        strides.dtype.is_integral()
        and dilation.dtype.is_integral()
        and output_paddings.dtype.is_integral()
    )

    if strides.num_elements() != spatial_rank:
        raise Error("(input_rank-2) values expected in convTranspose stride")

    if dilation.num_elements() != spatial_rank:
        raise Error("(input_rank-2) values expected in convTranspose dilation")

    if output_paddings.num_elements() != spatial_rank:
        raise Error(
            "(input_rank-2) values expected in convTranspose output paddings"
        )

    if paddings.num_elements() != 2 * spatial_rank:
        raise Error(
            "(2*(input_rank-2)) value expected in convTranspose paddings"
        )


def _index_list[size: Int](values: TileTensor) -> IndexList[size]:
    var result = IndexList[size](0)

    comptime for i in range(size):
        result[i] = Int(values.load((i,)))
    return result


# Shared by both `ConvTransposeCPU.execute` overloads: `output` is the buffer
# the kernel accumulates into and applies `epilogue_fn` over.
@inline(.always)
def _conv_transpose_cpu[
    EpilogueFnType: ConvSimdEpilogueFn,
    //,
    filter_layout: StaticString,
    has_epilogue_fusion: Bool,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, address_space=.GENERIC, ...],
    filter: TileTensor[address_space=.GENERIC, ...],
    strides: TileTensor,
    dilation: TileTensor,
    paddings: TileTensor,
    output_paddings: TileTensor,
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises:
    comptime spatial_rank = input.rank - 2

    _check_conv_transpose_attrs[spatial_rank](
        strides, dilation, paddings, output_paddings
    )

    var pads = _index_list[2 * spatial_rank](paddings)
    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

    comptime if input.rank == 3:
        pad_w = Index(pads[0], pads[1])
    elif input.rank == 4:
        pad_h = Index(pads[0], pads[1])
        pad_w = Index(pads[2], pads[3])
    elif input.rank == 5:
        pad_d = Index(pads[0], pads[1])
        pad_h = Index(pads[2], pads[3])
        pad_w = Index(pads[4], pads[5])

    conv_transposed_cpu[
        filter_layout == "FRSCf" or filter_layout == "FQRSCf",
        filter_layout == "CFRS",
        has_epilogue_fusion,
    ](
        output,
        input,
        filter,
        _index_list[spatial_rank](strides),
        _index_list[spatial_rank](dilation),
        pad_d,
        pad_h,
        pad_w,
        epilogue_fn,
        Optional(ctx.assert_cpu_target()),
    )


@extensibility.register("mo.conv_transpose", type="cpu")
struct ConvTransposeCPU:
    """Registers the `mo.conv_transpose` graph op for CPU devices.

    Two `execute` overloads under one registration: the graph compiler calls
    the fused one when any output has a fused epilogue and the plain one when
    none has. The CPU kernel accumulates in its output buffer and applies the
    epilogue in a pass over it, so the fused form accumulates in scratch and
    stores into the fused output, while the plain form writes the output
    directly. The pair is grandfathered on the graph compiler's allow list;
    new kernels declare one `execute` with a `StoreAccess` output.
    """

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Fused & StoreAccess],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilation: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        output_paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(output), Storable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilation), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime assert conforms_to(type_of(output_paddings), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        comptime dtype = type_of(output).dtype

        # The epilogue closure copies its capture into the kernel's loops, so
        # refine the output to the register-passable value it arrives as.
        comptime assert conforms_to(type_of(output), TrivialRegisterPassable)

        @inline(.always)
        def output_fn[
            _dtype: DType, _width: SIMDLength, _alignment: Int = 1
        ](coords: Coord, val: SIMD[_dtype, _width]) {var output}:
            output.store[width=Int(_width), element_alignment=_alignment](
                coords, rebind[SIMD[dtype, Int(_width)]](val)
            )

        var scratch_buf = ctx.enqueue_create_buffer[dtype](
            Int(output.shape().product())
        )
        var scratch = TileTensor(scratch_buf, row_major(output.shape()))
        _conv_transpose_cpu[filter_layout, has_epilogue_fusion=True](
            scratch,
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilation.to_tile_tensor(),
            paddings.to_tile_tensor(),
            output_paddings.to_tile_tensor(),
            output_fn,
            ctx,
        )

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilation: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        output_paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises where not conforms_to(type_of(output), Fused):
        comptime assert conforms_to(type_of(output), TileTensorable)
        comptime assert conforms_to(type_of(input), TileTensorable)
        comptime assert conforms_to(type_of(filter), TileTensorable)
        comptime assert conforms_to(type_of(strides), TileTensorable)
        comptime assert conforms_to(type_of(dilation), TileTensorable)
        comptime assert conforms_to(type_of(paddings), TileTensorable)
        comptime assert conforms_to(type_of(output_paddings), TileTensorable)
        comptime assert type_of(input).rank == type_of(output).rank
        _conv_transpose_cpu[filter_layout, has_epilogue_fusion=False](
            output.to_tile_tensor(),
            input.to_tile_tensor(),
            filter.to_tile_tensor(),
            strides.to_tile_tensor(),
            dilation.to_tile_tensor(),
            paddings.to_tile_tensor(),
            output_paddings.to_tile_tensor(),
            no_conv_simd_epilogue_fn,
            ctx,
        )


# Shared by both `ConvTransposeGPU.execute` overloads: validates the attributes
# and returns the stride, dilation, and padding cuDNN takes.
def _conv_transpose_gpu_attrs[
    filter_layout: StaticString
](
    input: TileTensor,
    filter: TileTensor,
    strides: TileTensor,
    dilation: TileTensor,
    paddings: TileTensor,
    output_paddings: TileTensor,
) raises -> Tuple[
    IndexList[input.rank - 2],
    IndexList[input.rank - 2],
    IndexList[input.rank - 2],
]:
    comptime assert (
        input.rank == 4 and filter.rank == 4
    ), "only rank 4 tensor is supported on cuda gpu"
    comptime assert (
        filter_layout != "FRSCf" and filter_layout != "FQRSCf"
    ), "only unpacked filter is supported on cuda gpu"
    comptime spatial_rank = input.rank - 2

    _check_conv_transpose_attrs[spatial_rank](
        strides, dilation, paddings, output_paddings
    )

    var padding = IndexList[spatial_rank](0)
    padding[0] = Int(paddings.load((0,)))
    padding[1] = Int(paddings.load((2,)))
    return (
        _index_list[spatial_rank](strides),
        _index_list[spatial_rank](dilation),
        padding,
    )


@extensibility.register("mo.conv_transpose", type="gpu")
struct ConvTransposeGPU:
    """Registers the `mo.conv_transpose` graph op for GPU devices.

    Two `execute` overloads under one registration: the graph compiler calls
    the fused one when any output has a fused epilogue and the plain one when
    none has. cuDNN writes a whole output buffer, so the fused form has it
    write scratch and stores from there into the fused output, while the plain
    form has cuDNN write the output directly. The pair is grandfathered on the
    graph compiler's allow list; new kernels declare one `execute` with a
    `StoreAccess` output.
    """

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Fused & StoreAccess],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilation: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        output_paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            conforms_to(type_of(input), TileTensorable)
            and conforms_to(type_of(filter), TileTensorable)
            and conforms_to(type_of(strides), TileTensorable)
            and conforms_to(type_of(dilation), TileTensorable)
            and conforms_to(type_of(paddings), TileTensorable)
            and conforms_to(type_of(output_paddings), TileTensorable)
        )
        comptime dtype = type_of(output).dtype

        # The epilogue runs in a device kernel the library launches, so it
        # copy-captures the access-enabled view rather than the host proxy.
        comptime assert conforms_to(type_of(output), UnsafeFusedView)
        var out_view = unsafe_fused_view(output)
        comptime assert conforms_to(type_of(out_view), Storable)

        @inline(.always)
        def output_fn[
            _dtype: DType, _width: SIMDLength, _alignment: Int = 1
        ](coords: Coord, val: SIMD[_dtype, _width]) {var out_view}:
            out_view.store[width=Int(_width), element_alignment=_alignment](
                coords,
                rebind[SIMD[type_of(out_view).dtype, Int(_width)]](val),
            )

        var input_tt = input.to_tile_tensor()
        var filter_tt = filter.to_tile_tensor()
        var attrs = _conv_transpose_gpu_attrs[filter_layout](
            input_tt,
            filter_tt,
            strides.to_tile_tensor(),
            dilation.to_tile_tensor(),
            paddings.to_tile_tensor(),
            output_paddings.to_tile_tensor(),
        )
        conv_transposed_gpu[type_of(input).dtype, type_of(filter).dtype, dtype](
            row_major(output.shape()),
            input_tt,
            filter_tt,
            attrs[0],
            attrs[1],
            attrs[2],
            output_fn,
            ctx,
        )

    @staticmethod
    def execute[
        input_layout: StaticString,
        filter_layout: StaticString,
        target: StaticString,
    ](
        output: Some[Output & Tensor],
        input: Some[Input & Tensor],
        filter: Some[Input & Tensor],
        strides: Some[Input & Tensor],
        dilation: Some[Input & Tensor],
        paddings: Some[Input & Tensor],
        output_paddings: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises where not conforms_to(type_of(output), Fused):
        comptime assert (
            conforms_to(type_of(output), TileTensorable)
            and conforms_to(type_of(input), TileTensorable)
            and conforms_to(type_of(filter), TileTensorable)
            and conforms_to(type_of(strides), TileTensorable)
            and conforms_to(type_of(dilation), TileTensorable)
            and conforms_to(type_of(paddings), TileTensorable)
            and conforms_to(type_of(output_paddings), TileTensorable)
        )
        var input_tt = input.to_tile_tensor()
        var filter_tt = filter.to_tile_tensor()
        var attrs = _conv_transpose_gpu_attrs[filter_layout](
            input_tt,
            filter_tt,
            strides.to_tile_tensor(),
            dilation.to_tile_tensor(),
            paddings.to_tile_tensor(),
            output_paddings.to_tile_tensor(),
        )
        conv_transposed_gpu[
            type_of(input).dtype, type_of(filter).dtype, type_of(output).dtype
        ](
            output.to_tile_tensor(),
            input_tt,
            filter_tt,
            attrs[0],
            attrs[1],
            attrs[2],
            ctx,
        )


@extensibility.register_shape_function("mo.conv_transpose")
def mo_conv_transpose_shape(
    input: Some[TileTensorable],
    filter: Some[TileTensorable],
    strides: Some[TileTensorable],
    dilations: Some[TileTensorable],
    paddings: Some[TileTensorable],
    output_paddings: Some[TileTensorable],
) raises -> IndexList[type_of(input).rank]:
    """Computes the output shape for the `mo.conv_transpose` graph op."""
    comptime assert type_of(strides).rank == 1, "strides must be rank 1"
    comptime assert type_of(dilations).rank == 1, "dilations must be rank 1"
    comptime assert type_of(paddings).rank == 1, "paddings must be rank 1"
    comptime assert (
        type_of(output_paddings).rank == 1
    ), "output_paddings must be rank 1"
    comptime assert (
        type_of(filter).dtype == type_of(input).dtype
    ), "filter dtype must match input dtype"
    return rebind[IndexList[type_of(input).rank]](
        conv_transpose_shape(
            input.to_tile_tensor(),
            filter.to_tile_tensor().bitcast[type_of(input).dtype](),
            strides.to_tile_tensor(),
            dilations.to_tile_tensor(),
            paddings.to_tile_tensor(),
            output_paddings.to_tile_tensor(),
        )
    )


@extensibility.register("layout_transform_RSFC_to_FRSCf")
struct LayoutTransformRSFC2FRSCf:
    """Registers the `layout_transform_RSFC_to_FRSCf` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@extensibility.register("layout_transform_QRSFC_to_FQRSCf")
struct LayoutTransformQRSFC2FQRSCf:
    """Registers the `layout_transform_QRSFC_to_FQRSCf` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute[
        dtype: DType, filter_rank: Int, packed_filter_rank: Int
    ](
        packed_filter: OutputTensor[dtype=dtype, rank=packed_filter_rank, ...],
        filter: InputTensor[dtype=dtype, rank=filter_rank, ...],
    ):
        layout_transform_conv_transpose_filter_common(packed_filter, filter)


@extensibility.register("pack_conv_filter_shape")
struct PackConvFilterShape:
    """Registers the `pack_conv_filter_shape` graph op with the graph compiler.
    """

    @inline(.always)
    @staticmethod
    def execute(filter_buf: InputTensor) raises:
        raise Error("Only meant to be used for shape function!")


@extensibility.register_shape_function("pack_conv_filter_shape")
def pack_conv_filter_shape_fn[
    input_shape: IntTuple,
    filter_shape: IntTuple,
    output_shape: IntTuple,
    strides: IntTuple,
    dilations: IntTuple,
    paddings: IntTuple,
    num_groups: Int,
](filter_buf: Some[TileTensorable]) -> IndexList[type_of(filter_buf).rank + 1]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
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

    return rebind[IndexList[type_of(filter_buf).rank + 1]](
        pack_filter_shape_conv[
            type_of(filter_buf).dtype,
            input_shape,
            filter_shape,
            output_shape,
            strides,
            dilations,
            paddings,
            num_groups,
        ](filter_buf.to_tile_tensor())
    )


@extensibility.register("layout_transform_QRSCF_to_FQRSCf")
struct LayoutTransformQRSCF2FQRSCf:
    """Registers the `layout_transform_QRSCF_to_FQRSCf` graph op with the graph compiler.
    """

    @inline(.always)
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


@extensibility.register("layout_transform_RSCF_to_FRSCf")
struct LayoutTransformRSCF2FRSCf:
    """Registers the `layout_transform_RSCF_to_FRSCf` graph op with the graph compiler.
    """

    @inline(.always)
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


@extensibility.register("layout_transform_FCRS_to_FRSCf")
struct LayoutTransformFCRS2FRSCf:
    """Registers the `layout_transform_FCRS_to_FRSCf` graph op with the graph compiler.
    """

    @inline(.always)
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


@extensibility.register("layout_transform_FCQRS_to_FQRSCf")
struct LayoutTransformFCQRS2FQRSCf:
    """Registers the `layout_transform_FCQRS_to_FQRSCf` graph op with the graph compiler.
    """

    @inline(.always)
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
