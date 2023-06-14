# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert
from BuildInfo import is_relwithdebinfo_build, is_debug_build
from Buffer import NDBuffer
from List import DimList
from DType import DType
from Image import (
    ImageData,
    Image2DLayout,
)
from Index import StaticIntTuple, Index
from Math import min
from TargetInfo import (
    has_avx512f,
    has_neon,
    os_is_macos,
    dtype_simd_width,
    dtype_sizeof,
)

# conv uses a different kernel than matmul
fn get_conv_a_row_size() -> Int:
    @parameter
    if has_neon():
        return 8
    elif has_avx512f():
        return 5
    return 3


fn get_conv_pack_inner_size() -> Int:
    @parameter
    if has_neon():
        return 2
    elif has_avx512f():
        return 4
    return 4


@register_passable("trivial")
struct ConvShape:
    """A shape struct describing the convolution dimensions"""

    # Input dimensions.
    var n: Int  # Input batch size.
    var h: Int  # Input height.
    var w: Int  # Input width.
    var c: Int  # Input channel count.
    var out_h: Int  # Output height.
    var out_w: Int  # Output width.

    # Filter dimensions.
    var f: Int  # Filter count.
    var r: Int  # Filter height.
    var s: Int  # Filter width.
    # Convolution parameters.
    var stride: StaticIntTuple[2]  # Stride on [H, W]
    var dilation: StaticIntTuple[2]  # Dilation on [H, W]
    var pad_h: StaticIntTuple[2]  # Padding on H dimension in (Low, High)
    var pad_w: StaticIntTuple[2]  # Padding on W dimension in (Low, High)


@adaptive
fn get_conv2d_shape[
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Image2DLayout.NCHW]()
    assert_param[filter_layout == Image2DLayout.NCHW]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[2](),
        w: input.dim[3](),
        c: input.dim[1](),
        out_h: output.dim[2](),
        out_w: output.dim[3](),
        f: filter.dim[0](),
        r: filter.dim[2](),
        s: filter.dim[3](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@adaptive
fn get_conv2d_shape[
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Image2DLayout.NHWC]()
    assert_param[filter_layout == Image2DLayout.NHWC]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[1](),
        w: input.dim[2](),
        c: input.dim[3](),
        out_h: output.dim[1](),
        out_w: output.dim[2](),
        f: filter.dim[0](),
        r: filter.dim[1](),
        s: filter.dim[2](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


@adaptive
fn get_conv2d_shape[
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[4, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Image2DLayout.NHWC]()
    assert_param[filter_layout == Image2DLayout.RSCF]()

    return ConvShape {
        n: input.dim[0](),
        h: input.dim[1](),
        w: input.dim[2](),
        c: input.dim[3](),
        out_h: output.dim[1](),
        out_w: output.dim[2](),
        f: filter.dim[3](),
        r: filter.dim[0](),
        s: filter.dim[1](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }
