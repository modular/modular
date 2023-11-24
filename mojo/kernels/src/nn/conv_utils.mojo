# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, clamp, div_ceil, max, min, sqrt
from sys._build import is_debug_build
from sys.info import (
    has_avx2,
    has_avx512f,
    has_neon,
    is_neoverse_n1,
    os_is_macos,
    simdwidthof,
    sizeof,
)

from Image import Image2DLayout, ImageData
from MatmulUtils import partition_work
from memory.buffer import NDBuffer

from utils.index import Index, StaticIntTuple
from utils.list import DimList, Dim
from utils._optional_param import OptionalParamInt, OptionalParamInts


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
    """A shape struct describing the convolution dimensions."""

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
    var num_groups: Int

    @always_inline
    fn c_per_group(self) -> Int:
        """Returns the number of channels per group. Channel count must be divisible by group size.
        """
        return self.c // self.num_groups

    @always_inline
    fn f_per_group(self) -> Int:
        """Returns the number of filters per group. Filter count must be divisible by group size.
        """
        return self.f // self.num_groups

    @always_inline
    fn f_to_group(self, f_idx: Int) -> Int:
        """Given a global filter idx, returns the group idx of the group the filter belongs to.
        """
        return f_idx // self.f_per_group()

    @always_inline
    fn c_to_group(self, c_idx: Int) -> Int:
        """Given a global channel idx, returns the group idx of the group the channel belongs to.
        """
        return c_idx // self.c_per_group()

    @always_inline
    fn f_in_group(self, f_idx: Int) -> Int:
        """Given a global filter idx, returns the offset of the filter in its group.
        """
        return f_idx % self.f_per_group()

    @always_inline
    fn c_in_group(self, c_idx: Int) -> Int:
        """Given a global channel idx, returns the offset of the channel in its group.
        """
        return c_idx % self.c_per_group()


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
    num_groups: Int,
) -> ConvShape:
    constrained[data_layout == Image2DLayout.NCHW]()
    constrained[filter_layout == Image2DLayout.NCHW]()

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
        num_groups: num_groups,
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
    num_groups: Int,
) -> ConvShape:
    constrained[data_layout == Image2DLayout.NHWC]()
    constrained[filter_layout == Image2DLayout.NHWC]()

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
        num_groups: num_groups,
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
    num_groups: Int,
) -> ConvShape:
    constrained[data_layout == Image2DLayout.NHWC]()
    constrained[filter_layout == Image2DLayout.RSCF]()

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
        num_groups: num_groups,
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
    num_groups: Int,
) -> ConvShape:
    constrained[data_layout == Image2DLayout.NHWC]()
    constrained[
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
            num_groups: num_groups,
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
        num_groups: num_groups,
    }


fn get_conv_tile_size[type: DType]() -> Int:
    # The rule-of-thumb is 1/2 of L2 cache size. It's common to have 3x3
    # filter window in convolution. So the cache tile size (in terms of
    # elements) is rounded up to multiple of 9.
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
        #  Graviton 2 and Skylake server
        # have a 1 MiB L2 cache
        return 576 * KB // sizeof[type]()

    # AMD Rome has a 512 KiB L2 cache.
    return 288 * KB // sizeof[type]()


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


# must be register passable because it is used as a parameter
@value
@register_passable("trivial")
struct ConvInfoStatic:
    var pad_h: DimList
    var pad_w: DimList
    var stride: DimList
    var dilation: DimList
    var num_groups: Dim

    @always_inline
    fn all_known(self) -> Bool:
        return (
            self.pad_h.all_known[2]()
            and self.pad_w.all_known[2]()
            and self.stride.all_known[2]()
            and self.dilation.all_known[2]()
            and self.num_groups.has_value()
        )

    @always_inline
    fn pad_left(self) -> Int:
        return self.pad_w.at[0]().get()

    @always_inline
    fn pad_bottom(self) -> Int:
        return self.pad_h.at[0]().get()

    @always_inline
    fn strides(self) -> StaticIntTuple[2]:
        return Index(self.stride.at[0]().get(), self.stride.at[1]().get())

    @always_inline
    fn dilations(self) -> StaticIntTuple[2]:
        return Index(self.dilation.at[0]().get(), self.dilation.at[1]().get())

    @always_inline
    @staticmethod
    fn create_unknown() -> Self:
        return rebind[Self](
            ConvInfoStatic(
                DimList.create_unknown[2](),
                DimList.create_unknown[2](),
                DimList.create_unknown[2](),
                DimList.create_unknown[2](),
                Dim(),
            )
        )


struct ConvInfo[conv_info_static: ConvInfoStatic]:
    var pad_h: OptionalParamInts[2, conv_info_static.pad_h]
    var pad_w: OptionalParamInts[2, conv_info_static.pad_w]
    var stride: OptionalParamInts[2, conv_info_static.stride]
    var dilation: OptionalParamInts[2, conv_info_static.dilation]
    var num_groups: OptionalParamInt[conv_info_static.num_groups]

    fn __init__(
        inout self,
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        num_groups: Int,
    ):
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride = stride
        self.dilation = dilation
        self.num_groups = num_groups


fn get_direct_conv_micro_kernel_height() -> Int:
    @parameter
    if has_avx512f():
        return 6
    elif is_neoverse_n1():
        return 8
    elif has_neon():  # neon other than neoverse-N1
        return 6
    return 4


fn get_direct_conv_micro_kernel_width() -> Int:
    @parameter
    if has_avx512f():
        return 4
    elif is_neoverse_n1():
        return 2
    elif has_neon():  # neon other than neoverse-N1
        return 4
    return 3


fn get_micro_kernel_shape[
    WO: Dim, F: Dim, conv_attr: ConvInfoStatic, simd_size: Int
]() -> StaticIntTuple[2]:
    alias optimize_static_shapes = WO.has_value() and F.has_value() and conv_attr.all_known()

    # Number of named simd registers for each architecture.
    # TODO: configure micro kernel shape are other architectures.
    alias num_avx512_registers = 32
    alias num_avx2_registers = 16

    @parameter
    if optimize_static_shapes:
        alias WO_val = WO.get()
        alias F_val = F.get()
        alias pad_h_val = Index(
            conv_attr.pad_h.at[0]().get(), conv_attr.pad_h.at[1]().get()
        )
        alias pad_w_val = Index(
            conv_attr.pad_w.at[0]().get(), conv_attr.pad_w.at[1]().get()
        )
        alias has_padding = pad_h_val != Index(0, 0) or pad_w_val != Index(0, 0)

        @parameter
        if has_avx512f():
            # The micro tile is m rows by n*simd_size columns.
            # The register usage in tiling for avx512/avx2:
            #   (1) load n registers in F dimension.
            #   (2) broadcast 1 element from each row into 1 register. The same
            #       is used for all rows. This doesn't serialize the accumulation
            #       because register renaming can resolve RAR dependence.
            #   (3) accumulate m * n registers.
            # There are in total m*n + n + 1 registers needed.
            # Iterating n from 2, we get possible (m, n) combinations including
            # (14, 2), (9, 3), (6, 4), and (5, 5).

            # Static shapes enable a better algorithm for padding, which can choose micro
            # kernel shape based on input and output sizes.
            if has_padding:
                # Traverse the possible combinations (14, 2), (9, 3), (6, 4), and (5, 5).
                for n in range(2, 6):
                    let m = (num_avx512_registers - 1) // n - 1
                    # Short circuit if the row fit in one micro kernel and F is divisible.
                    # E.x. for WO=7 and F=512, 7x2 can be a better micro kernel than 7x3
                    # for multi-threading due to partition granularity (kernel width) in F.
                    if F_val % (n * simd_size) == 0 and WO_val <= m:
                        return Index(WO_val, n)
            # Use 6x4 by default as it achieves the best performance for most shapes.
            return Index(6, 4)

        @parameter
        if has_avx2():
            if has_padding:
                # Register usage formula is the same as avx512.
                # There are in total 16 named simd registers, the viable micro kernels
                # are (6, 2) and (4, 3).

                # The heuristic searchs the micro kernel shape leading to the
                # least remainder. The following values will be overwritten since
                # the residual is at most 2 * WO * F.
                var min_num_residual = 3 * WO_val * F_val
                var micro_kernel_height = -1
                var micro_kernel_width = -1
                for n in range(2, 3):
                    let m = (num_avx2_registers - 1) // n - 1
                    let num_residual = WO_val * (F_val % (n * simd_size)) + (
                        WO_val % m
                    ) * F_val
                    if num_residual < min_num_residual:
                        micro_kernel_height = m
                        micro_kernel_width = n
                        min_num_residual = num_residual
                return Index(micro_kernel_height, micro_kernel_width)
            return Index(6, 2)

        return Index(6, 2)

    else:  # Default options for dynamic shapes.

        @parameter
        if has_avx512f():
            return Index(6, 4)
        elif is_neoverse_n1():
            return Index(8, 2)
        elif has_neon():  # neon other than neoverse-N1
            return Index(6, 4)
        # default, including AVX2
        else:
            return Index(4, 3)


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
](max_num_tasks: Int, conv_shape: ConvShape) -> StaticIntTuple[4]:
    """Partition the worload in (batch, C, F, HOWO) dimensions.
    HOWO is the combination of HO and WO dimensions.
    The actual number of tasks are the product of return num_partitions.
    """

    # Heuristic parameters for partitioning
    # AVX512, partitioning channel can be beneficial for some shapes.
    alias min_rows_per_task_avx512 = align_down(196, micro_kernel_w)
    alias min_c_per_task_avx512 = 64
    # Otherwise, discourage partitioning channel.
    alias min_rows_per_task = min_rows_per_task_avx512 if has_avx512f() else align_down(
        64, micro_kernel_w
    )
    alias min_c_per_task = min_c_per_task_avx512 if has_avx512f() else 1024

    # alias min_rows_per_task = (196 // micro_kernel_w) * micro_kernel_w
    # alias min_c_per_task = 64

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
    let ideal_num_col_tasks = sqrt(
        div_ceil(matmul_N * max_num_tasks, matmul_M_biased)
    )
    var num_row_tasks = max_num_tasks // ideal_num_col_tasks
    var num_col_tasks = ideal_num_col_tasks

    # There must at least have enough elements to support a micro kernel.
    # Do not partition F when num_groups > 1.
    var max_num_col_tasks = min(
        div_ceil(matmul_N, micro_kernel_f), max_num_tasks
    ) if conv_shape.num_groups == 1 else 1
    if ideal_num_col_tasks > max_num_col_tasks:
        num_col_tasks = max_num_col_tasks
        num_row_tasks = max_num_tasks // num_col_tasks
    # In this branch, not all threads get used for ideal_num_col_tasks
    # Check for alternative factorizations use the most threads.
    elif max_num_tasks % ideal_num_col_tasks != 0:
        # Set 20% deviation.
        let eps = div_ceil(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            max_num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            let num_row_tasks_tmp = max_num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp >= num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    let max_num_row_tasks = max(matmul_M // min_rows_per_task, 1)
    num_row_tasks = min(max_num_row_tasks, num_row_tasks)

    # Do not partition channels when num_groups > 1.
    let max_num_channel_tasks = max(
        conv_shape.c // min_c_per_task, 1
    ) if conv_shape.num_groups == 1 else 1
    let num_channel_tasks = min(
        max_num_channel_tasks,
        max_num_tasks // (num_row_tasks * num_col_tasks),
    )

    let num_batch_tasks = min(conv_shape.n, num_row_tasks)

    num_row_tasks = num_row_tasks // num_batch_tasks

    return Index(
        num_batch_tasks, num_channel_tasks, num_col_tasks, num_row_tasks
    )


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
