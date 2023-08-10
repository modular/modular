# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param, debug_assert
from BuildInfo import is_debug_build
from Buffer import NDBuffer
from List import DimList
from DType import DType
from Image import (
    ImageData,
    Image2DLayout,
)
from Index import StaticIntTuple, Index
from Math import min, max, clamp, sqrt, div_ceil
from MatmulUtils import partition_work
from SIMD import SIMD, Float32
from TargetInfo import (
    has_avx512f,
    has_neon,
    os_is_macos,
    simdwidthof,
    sizeof,
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


fn get_conv2d_shape[
    filter_rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[4, output_shape, type],
    input: NDBuffer[4, input_shape, type],
    filter: NDBuffer[filter_rank, filter_shape, type],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
) -> ConvShape:
    assert_param[data_layout == Image2DLayout.NHWC]()
    assert_param[
        (filter_rank == 4 and filter_layout == Image2DLayout.RSCF)
        or (filter_rank == 5 and filter_layout == Image2DLayout.FRSCf)
    ]()

    @parameter
    if filter_rank == 4 and filter_layout == Image2DLayout.RSCF:
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

    # default case: filter is packed, FRSCf
    return ConvShape {
        n: input.dim[0](),
        h: input.dim[1](),
        w: input.dim[2](),
        c: input.dim[3](),
        out_h: output.dim[1](),
        out_w: output.dim[2](),
        f: output.dim[3](),
        r: filter.dim[1](),
        s: filter.dim[2](),
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
    }


fn get_conv_tile_size[type: DType]() -> Int:
    alias KB = 1024

    # See MatmulUtils for context on tile size for debug built and macos.
    @parameter
    if is_debug_build():
        return 4 * KB // sizeof[type]()

    @parameter
    if os_is_macos():
        return 64 * KB // sizeof[type]()

    @parameter
    if has_neon() or has_avx512f():
        # This should be 1/2 of L2 cache size. Graviton 2 and Skylake server
        # have a 1 MiB L1 cache AMD Rome has a 512 KiB L2 cache.
        # Use 576 instead of 512 to accommodate important shapes in resnet.
        return 576 * KB // sizeof[type]()

    return 256 * KB // sizeof[type]()


fn get_conv_tile_shape[
    type: DType, micro_kernel_width: Int
](conv_shape: ConvShape) -> StaticIntTuple[2]:
    """Compute the (c, f) tile shape in L2.
    Assume NHWC layout, the tile shape is (R, S, c_tile, f_tile). R and S are
    by default fully covered. The heuristic tried to block in C as much as
    possible. If C is small, it would start to block F.
    """
    alias simd_size = simdwidthof[type]()

    # Number of elements in tile.
    let tile_size = get_conv_tile_size[type]()
    # Number of elements in micro kernel's f dimension.
    let micro_kernel_f = micro_kernel_width * simd_size
    # Max C tile size, assuming R, S, and micro_kernel_f are covered.
    # Round up to multiple simd_size
    let CF_tile_size = tile_size // (conv_shape.r * conv_shape.s)
    let max_c_tile_size = max(
        CF_tile_size // micro_kernel_f // simd_size, 1
    ) * simd_size
    # C tile size is bounded by the input channels.
    let c_tile_size = min(max_c_tile_size, conv_shape.c)
    # F tile size is rounded up to multiple micro_kernel_f.
    let f_tile_size = max(
        CF_tile_size // c_tile_size // micro_kernel_f, 1
    ) * micro_kernel_f

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


# ===----------------------------------------------------------------------===#
# Partition Heuristics
# ===----------------------------------------------------------------------===#


fn get_conv_num_tasks(num_threads: Int, conv_shape: ConvShape) -> Int:
    # Currently use matmul's min task size but the optimal value
    # for direct conv may be different.
    alias min_task_size = 64 * 1024
    # fmt: off
    let complexity = conv_shape.n * conv_shape.out_h * conv_shape.out_w \
                   * conv_shape.r * conv_shape.s * conv_shape.c \
                   * conv_shape.f
    # fmt: on
    # Ensure at most one task per thread.
    return min(div_ceil(complexity, min_task_size), num_threads)


fn get_conv_num_partitions[
    micro_kernel_w: Int, micro_kernel_f: Int
](num_tasks: Int, conv_shape: ConvShape) -> StaticIntTuple[3]:
    @always_inline
    @noncapturing
    fn int_sqrt_floor(val: Int) -> Int:
        return Int(sqrt(Float32(val)).cast[DType.index]().value)

    alias min_rows_per_task = (196 // micro_kernel_w) * micro_kernel_w
    alias min_c_per_task = 64

    let matmul_M = conv_shape.n * conv_shape.out_h * conv_shape.out_w
    let matmul_N = conv_shape.f
    let matmul_K = conv_shape.r * conv_shape.s * conv_shape.c

    # Accessing A is more expensive in im2col than accessing B.
    # Time a factor to M to let the heuristic bias on partitioning M.
    # TODO: make this bias factor part of function parameter/argument and
    # unifies interface with matmul partition, e.x. bias=1 for matmul.
    let bias = 0.25
    let matmul_M_biased = max(
        (Float32(matmul_M) * bias).cast[DType.index]().value, 1
    )

    # The ideal partition in theory is to balance the cost of memory access in
    # M and N dimensions using square sub-matrix (after applying the bias).
    let ideal_num_col_tasks = int_sqrt_floor(
        div_ceil(matmul_N * num_tasks, matmul_M_biased)
    )
    var num_row_tasks = num_tasks // ideal_num_col_tasks
    var num_col_tasks = ideal_num_col_tasks

    # There must at least have enough elements to support a micro kernel.
    var max_num_col_tasks = min(div_ceil(matmul_N, micro_kernel_f), num_tasks)
    if ideal_num_col_tasks > max_num_col_tasks:
        num_col_tasks = max_num_col_tasks
        num_row_tasks = num_tasks // num_col_tasks
    # In this branch, not all threads get used for ideal_num_col_tasks
    # Check for alternative factorizations use the most threads.
    elif num_tasks % ideal_num_col_tasks != 0:
        # Set 20% deviation.
        let eps = div_ceil(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            let num_row_tasks_tmp = num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp > num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    let max_num_row_tasks = max(matmul_M // min_rows_per_task, 1)
    num_row_tasks = min(max_num_row_tasks, num_row_tasks)

    let max_num_channel_tasks = max(conv_shape.c // min_c_per_task, 1)
    let num_channel_tasks = min(
        max_num_channel_tasks,
        num_tasks // (num_row_tasks * num_col_tasks),
    )

    return Index(num_row_tasks, num_channel_tasks, num_col_tasks)


# ===----------------------------------------------------------------------===#
# Convolution Algorithms Selection
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct ConvAlgorithm:
    var value: Int
    alias Default = ConvAlgorithm(0)  # statically unknown layout.
    alias Im2Col = ConvAlgorithm(1)  # channels first layout.
    alias Direct = ConvAlgorithm(2)  # TF filter layout for channels last input.

    @always_inline("nodebug")
    fn __eq__(self, rhs: ConvAlgorithm) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: ConvAlgorithm) -> Bool:
        return self.value != rhs.value
