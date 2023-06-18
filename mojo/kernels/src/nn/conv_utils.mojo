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
from Math import min, max, clamp
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


fn get_conv_tile_size[type: DType]() -> Int:
    alias KB = 1024

    # See MatmulUtils for context on tile size for debug built and macos.
    @parameter
    if is_relwithdebinfo_build() or is_debug_build():
        return 4 * KB // dtype_sizeof[type]()

    @parameter
    if os_is_macos():
        return 64 * KB // dtype_sizeof[type]()

    @parameter
    if has_neon() or has_avx512f():
        # This should be 1/2 of L2 cache size. Graviton 2 and Skylake server
        # have a 1 MiB L1 cache AMD Rome has a 512 KiB L2 cache.
        # Use 576 instead of 512 to accommodate important shapes in resnet.
        return 576 * KB // dtype_sizeof[type]()

    return 256 * KB // dtype_sizeof[type]()


fn get_conv_tile_shape[
    type: DType, micro_kernel_width: Int
](conv_shape: ConvShape) -> StaticIntTuple[2]:
    """Compute the (c, f) tile shape in L2.
    Assume NHWC layout, the tile shape is (R, S, c_tile, f_tile). R and S are
    by default fully covered. The heuristic tried to block in C as much as
    possible. If C is small, it would start to block F.
    """
    alias simd_size = dtype_simd_width[type]()

    # Number of elements in tile.
    let tile_size = get_conv_tile_size[type]()
    # Number of elements in micro kernel's f dimension.
    let micro_kernel_f = micro_kernel_width * simd_size
    # Max C tile size, assuming R, S, and micro_kernel_f are covered.
    # Round up to multiple simd_size
    let CF_tile_size = tile_size // (conv_shape.r * conv_shape.s)
    let max_c_tile_size = (
        CF_tile_size // micro_kernel_f // simd_size
    ) * simd_size
    # C tile size is bounded by the input channels.
    let c_tile_size = min(max_c_tile_size, conv_shape.c)
    # F tile size is rounded up to multiple micro_kernel_f.
    let rounded_f_tile_size = (
        CF_tile_size // c_tile_size // micro_kernel_f
    ) * micro_kernel_f
    let f_tile_size: Int = clamp[DType.int32, 1](
        rounded_f_tile_size, conv_shape.f, micro_kernel_f
    ).value

    return Index(c_tile_size, f_tile_size)


fn get_direct_conv_micro_kernel_height() -> Int:
    @parameter
    if has_avx512f():
        return 6
    elif has_neon():
        return 8
    return 4


fn get_direct_conv_micro_kernel_width() -> Int:
    @parameter
    if has_avx512f():
        return 4
    elif has_neon():
        return 2
    return 3
