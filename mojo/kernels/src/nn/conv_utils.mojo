# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, ceildiv, sqrt
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

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from linalg.utils import partition_work

from utils.index import Index, IndexList

from .image import Image2DLayout

# ===----------------------------------------------------------------------=== #
# Epilogue Helper                                                              #
# ===----------------------------------------------------------------------=== #


# Elementwise epilogue signature
alias elementwise_epilogue_type = fn[rank: Int] (
    coords: IndexList[rank],
    f_size: Int,
) capturing -> None

alias elementwise_simd_epilogue_type = fn[type: DType, rank: Int, width: Int] (
    IndexList[rank], SIMD[type, width]
) capturing -> None


# ===----------------------------------------------------------------------=== #
# Wrapper for  Convolution Shape                                               #
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct ConvShape[rank: Int]:
    """A shape struct describing the convolution dimensions."""

    var n: Int  # Input batch size.

    var input_dims: IndexList[rank]  # Ex H and W for 2D
    var output_dims: IndexList[rank]  # Ex HO and WO for 2D.
    var filter_dims: IndexList[rank]  # Ex R and S for 2D.

    var c: Int  # Input channel.
    var f: Int  # Output channel.

    var stride: IndexList[rank]

    var dilation: IndexList[rank]

    # TODO: change paddings to
    # pad_lower: IndexList[rank]
    # pad_upper: IndexList[rank]
    var pad_d: IndexList[2]
    var pad_h: IndexList[2]
    var pad_w: IndexList[2]

    var num_groups: Int

    @always_inline
    fn d(self) -> Int:
        """Input depth."""

        @parameter
        if rank >= 3:
            return self.input_dims[rank - 3]
        else:
            return 1

    @always_inline
    fn h(self) -> Int:
        """Input height."""

        @parameter
        if rank >= 2:
            return self.input_dims[rank - 2]
        else:
            return 1

    @always_inline
    fn w(self) -> Int:
        """Input width."""
        return self.input_dims[rank - 1]

    @always_inline
    fn do(self) -> Int:
        """Output depth."""

        @parameter
        if rank >= 3:
            return self.output_dims[rank - 3]
        else:
            return 1

    @always_inline
    fn ho(self) -> Int:
        """Output height."""

        @parameter
        if rank >= 2:
            return self.output_dims[rank - 2]
        else:
            return 1

    @always_inline
    fn wo(self) -> Int:
        """Output width."""
        return self.output_dims[rank - 1]

    @always_inline
    fn q(self) -> Int:
        """Filter window depth."""

        @parameter
        if rank >= 3:
            return self.filter_dims[rank - 3]
        else:
            return 1

    @always_inline
    fn r(self) -> Int:
        """Filter window height."""

        @parameter
        if rank >= 2:
            return self.filter_dims[rank - 2]
        else:
            return 1

    @always_inline
    fn s(self) -> Int:
        """Filter windown width."""
        return self.filter_dims[rank - 1]

    @always_inline
    fn filter_window_flat_size(self) -> Int:
        return self.filter_dims.flattened_length()

    @always_inline
    fn input_image_flat_size(self) -> Int:
        return self.input_dims.flattened_length()

    @always_inline
    fn output_image_flat_size(self) -> Int:
        return self.output_dims.flattened_length()

    @always_inline
    fn output_space_dims(self) -> IndexList[rank]:
        return self.output_dims

    @always_inline
    fn output_flat_coord_to_input_offset(
        self, n: Int, output_flat_coord: Int
    ) -> Int:
        constrained[
            rank == 1 or rank == 2 or rank == 3,
            "Only support 1d, 2d, and 3d convolution.",
        ]()

        @parameter
        if rank == 1:
            var w = output_flat_coord * self.stride[0] - self.pad_w[0]

            return self.c * w

        elif rank == 2:
            # Unpack output coordinates
            var ho = output_flat_coord // self.wo()
            var wo = output_flat_coord % self.wo()

            # Input coordinates
            var h = ho * self.stride[0] - self.pad_h[0]
            var w = wo * self.stride[1] - self.pad_w[0]

            return self.c * (w + self.w() * (h + n * self.h()))

        elif rank == 3:
            # Unpack output coordinates
            var doho = output_flat_coord // self.wo()
            var wo = output_flat_coord % self.wo()
            var do = doho // self.ho()
            var ho = doho % self.ho()

            # Input coordinates
            var d = do * self.stride[0] - self.pad_d[0]
            var h = ho * self.stride[1] - self.pad_h[0]
            var w = wo * self.stride[2] - self.pad_w[0]

            return self.c * (
                w + self.w() * (h + self.h() * (d + self.d() * self.n))
            )

        else:
            # Pass compile.
            return -1

    @always_inline
    fn matmul_M(self) -> Int:
        return self.n * self.output_dims.flattened_length() * self.num_groups

    @always_inline
    fn matmul_N(self) -> Int:
        return self.f // self.num_groups

    @always_inline
    fn matmul_K(self) -> Int:
        return self.c * self.filter_dims.flattened_length() // self.num_groups

    @always_inline
    fn padded(self) -> Bool:
        return (
            self.pad_w != Index(0, 0)
            or self.pad_h != Index(0, 0)
            or self.pad_d != Index(0, 0)
        )

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


@always_inline
fn get_conv_shape[
    rank: Int,
    filter_packed: Bool,
](
    output: NDBuffer,
    input: NDBuffer,
    filter: NDBuffer,
    stride: IndexList[rank],
    dilation: IndexList[rank],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) -> ConvShape[rank]:
    var output_dims = IndexList[rank](0)
    var input_dims = IndexList[rank](0)
    var filter_dims = IndexList[rank](0)

    @parameter
    for i in range(rank):
        output_dims[i] = output.dim[i + 1]()
        input_dims[i] = input.dim[i + 1]()

        @parameter
        if filter_packed:
            filter_dims[i] = filter.dim[i + 1]()
        else:
            filter_dims[i] = filter.dim[i]()

    return ConvShape[rank](
        n=input.dim[0](),
        input_dims=input_dims,
        output_dims=output_dims,
        filter_dims=filter_dims,
        c=input.dim[rank + 1](),
        f=output.dim[rank + 1](),
        stride=stride,
        dilation=dilation,
        pad_d=pad_d,
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )


fn get_conv2d_shape[
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[mut=True, type, 4, _, output_shape],
    input: NDBuffer[type, 4, _, input_shape],
    filter: NDBuffer[type, 4, _, filter_shape],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    stride: IndexList[2],
    dilation: IndexList[2],
    num_groups: Int,
) -> ConvShape[2]:
    constrained[
        data_layout == Image2DLayout.NHWC
        and filter_layout == Image2DLayout.RSCF,
        "only support NHWC and RSCF layout for conv2D.",
    ]()

    return ConvShape[2](
        n=input.dim[0](),
        input_dims=Index(input.dim[1](), input.dim[2]()),
        output_dims=Index(output.dim[1](), output.dim[2]()),
        filter_dims=Index(filter.dim[0](), filter.dim[1]()),
        c=input.dim[3](),
        f=output.dim[3](),
        stride=stride,
        dilation=dilation,
        pad_d=Index(0, 0),
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )


fn get_conv2d_shape[
    filter_rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    filter_shape: DimList,
    type: DType,
    data_layout: Image2DLayout,
    filter_layout: Image2DLayout,
](
    output: NDBuffer[mut=True, type, 4, _, output_shape],
    input: NDBuffer[type, 4, _, input_shape],
    filter: NDBuffer[type, filter_rank, _, filter_shape],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    stride: IndexList[2],
    dilation: IndexList[2],
    num_groups: Int,
) -> ConvShape[2]:
    constrained[data_layout == Image2DLayout.NHWC]()
    constrained[
        (filter_rank == 4 and filter_layout == Image2DLayout.RSCF)
        or (filter_rank == 5 and filter_layout == Image2DLayout.FRSCf)
    ]()

    var filter_dims: IndexList[2]

    @parameter
    if filter_rank == 4 and filter_layout == Image2DLayout.RSCF:
        filter_dims = Index(filter.dim[0](), filter.dim[1]())
    else:
        filter_dims = Index(filter.dim[1](), filter.dim[2]())

    return ConvShape[2](
        n=input.dim[0](),
        input_dims=Index(input.dim[1](), input.dim[2]()),
        output_dims=Index(output.dim[1](), output.dim[2]()),
        filter_dims=filter_dims,
        c=input.dim[3](),
        f=output.dim[3](),
        stride=stride,
        dilation=dilation,
        pad_d=Index(0, 0),
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )


@always_inline
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


@always_inline
fn get_conv_tile_shape[
    type: DType,
](c: Int, filter_window_size: Int, micro_kernel_width: Int,) -> IndexList[2]:
    """Compute the (c, f) tile shape in L2.
    Assume NHWC layout, the tile shape is (R, S, c_tile, f_tile). R and S are
    by default fully covered. The heuristic tried to block in C as much as
    possible. If C is small, it would start to block F.
    """
    alias simd_size = simdwidthof[type]()

    # Number of elements in tile.
    var tile_size = get_conv_tile_size[type]()
    # Number of elements in micro kernel's f dimension.
    var micro_kernel_f = micro_kernel_width * simd_size
    # Max C tile size, assuming R, S, and micro_kernel_f are covered.
    # Round up to multiple simd_size
    var CF_tile_size = tile_size // filter_window_size
    var max_c_tile_size = max(
        CF_tile_size // micro_kernel_f // simd_size, 1
    ) * simd_size
    # C tile size is bounded by the input channels.
    var c_tile_size = min(max_c_tile_size, c)
    # F tile size is rounded up to multiple micro_kernel_f.
    var f_tile_size = max(
        CF_tile_size // c_tile_size // micro_kernel_f, 1
    ) * micro_kernel_f

    return Index(c_tile_size, f_tile_size)


@always_inline
fn extend_shape[
    rank: Int
](in_shape: IndexList[rank], first: Int, last: Int) -> IndexList[rank + 2]:
    """Extend input shape by inserting `first` and `last` at both ends."""
    var out_shape = IndexList[rank + 2](0)
    out_shape[0] = first
    out_shape[rank + 1] = last

    @parameter
    for i in range(rank):
        out_shape[i + 1] = in_shape[i]

    return out_shape


@always_inline
fn append_shape[
    rank: Int
](in_shape: IndexList[rank], last2nd: Int, last: Int) -> IndexList[rank + 2]:
    """Append input shape by inserting `last2nd` and `last` at the end."""
    var out_shape = IndexList[rank + 2](0)
    out_shape[rank] = last2nd
    out_shape[rank + 1] = last

    @parameter
    for i in range(rank):
        out_shape[i] = in_shape[i]

    return out_shape


@always_inline
fn reorder_padding[rank: Int](pad: DimList) -> DimList:
    @parameter
    if rank == 1:
        return pad
    elif rank == 2:
        return DimList(pad.at[0](), pad.at[2](), pad.at[1](), pad.at[3]())
    else:
        return DimList(
            pad.at[0](),
            pad.at[2](),
            pad.at[4](),
            pad.at[1](),
            pad.at[3](),
            pad.at[5](),
        )


struct ConvInfoStatic[rank: Int]:
    var pad: DimList
    var stride: DimList
    var dilation: DimList
    var num_groups: Dim

    @always_inline
    fn __init__(out self):
        self.pad = DimList.create_unknown[2 * rank]()
        self.stride = DimList.create_unknown[rank]()
        self.dilation = DimList.create_unknown[rank]()
        self.num_groups = Dim()

    @always_inline
    fn __init__(
        out self,
        pad: DimList,
        stride: DimList,
        dilation: DimList,
        num_groups: Dim,
    ):
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.num_groups = num_groups

    @always_inline
    fn __init__(
        out self,
        pad: DimList,
        stride: DimList,
        dilation: DimList,
        input_c: Dim,
        filter_c: Dim,
    ):
        constrained[
            rank == 3 or rank == 2 or rank == 1,
            "Only support 1d/2d/3d/ conv attributes",
        ]()

        var num_groups = Dim()
        if input_c.has_value() and filter_c.has_value():
            num_groups = Dim(input_c.get() // filter_c.get())

        self.pad = reorder_padding[rank](pad)
        self.stride = stride
        self.dilation = dilation
        self.num_groups = num_groups

    @always_inline
    fn all_known(self) -> Bool:
        return (
            self.pad.all_known[2 * rank]()
            and self.stride.all_known[rank]()
            and self.dilation.all_known[rank]()
            and self.num_groups.has_value()
        )

    @always_inline
    fn pad_left(self) -> Int:
        # TODO: extend to 1d/3d.
        return self.pad.get[1]()

    @always_inline
    fn pad_bottom(self) -> Int:
        # TODO: extend to 1d/3d.
        return self.pad.get[0]()

    @always_inline
    fn strides(self) -> IndexList[2]:
        return Index(self.stride.get[0](), self.stride.get[1]())

    @always_inline
    fn dilations(self) -> IndexList[2]:
        return Index(self.dilation.get[0](), self.dilation.get[1]())


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
    rank: Int, WO: Dim, F: Dim, conv_attr: ConvInfoStatic[rank], simd_size: Int
]() -> IndexList[2]:
    alias optimize_static_shapes = WO.has_value() and F.has_value() and conv_attr.all_known()

    # Number of named simd registers for each architecture.
    # TODO: configure micro kernel shape are other architectures.
    alias num_avx512_registers = 32
    alias num_avx2_registers = 16

    @parameter
    if optimize_static_shapes:
        alias WO_val = WO.get()
        alias F_val = F.get()
        # TODO: extend to 1d/3d.
        alias pad_h_val = Index(conv_attr.pad.get[0](), conv_attr.pad.get[2]())
        alias pad_w_val = Index(conv_attr.pad.get[1](), conv_attr.pad.get[3]())
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
                    var m = (num_avx512_registers - 1) // n - 1
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
                for n in range(2, 4):
                    var m = (num_avx2_registers - 1) // n - 1
                    var num_residual = WO_val * (F_val % (n * simd_size)) + (
                        WO_val % m
                    ) * F_val
                    if num_residual < min_num_residual:
                        micro_kernel_height = m
                        micro_kernel_width = n
                        min_num_residual = num_residual
                return Index(micro_kernel_height, micro_kernel_width)
            return Index(4, 3)

        @parameter
        if is_neoverse_n1():
            return Index(8, 2)
        elif has_neon():  # neon other than neoverse-N1
            return Index(6, 4)

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


# ===-----------------------------------------------------------------------===#
# Partition Heuristics
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct ConvPartition:
    """Work range for a partition."""

    # Batch and group dims are merged into one.
    var ng_offset: Int
    var ng_size: Int

    # Output channel range.
    var f_offset: Int
    var f_size: Int

    # Input dim.
    # For point-wise conv, ho and wo dims are merged and partitioned.
    # For others, only ho is partitioned.
    var ho_or_howo_offset: Int
    var ho_or_howo_size: Int

    # Input Channel dim.
    var c_offset: Int
    var c_size: Int

    @always_inline
    fn empty(self) -> Bool:
        # fmt: off
        return self.ng_size <= 0 or \
               self.f_size <= 0 or \
               self.ho_or_howo_size <= 0 or \
               self.c_size <= 0
        # fmt: on


@always_inline
fn get_conv_num_tasks(num_threads: Int, conv_shape: ConvShape) -> Int:
    # Currently use matmul's min task size but the optimal value
    # for direct conv may be different.
    alias min_task_size = 64 * 1024
    # fmt: off
    var complexity = conv_shape.matmul_M() * conv_shape.matmul_N() \
                   * conv_shape.matmul_K()
    # fmt: on
    # Ensure at most one task per thread.
    return min(ceildiv(complexity, min_task_size), num_threads)


fn get_conv_num_partitions[
    micro_kernel_w: Int, micro_kernel_f: Int
](num_threads: Int, conv_shape: ConvShape) -> IndexList[4]:
    """Partition the worload in (batch, C, F, HOWO) dimensions.
    HOWO is the combination of HO and WO dimensions.
    The actual number of tasks are the product of return num_partitions.
    """

    var max_num_tasks = get_conv_num_tasks(num_threads, conv_shape)

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

    var matmul_M = conv_shape.matmul_M()
    var matmul_N = conv_shape.matmul_N()
    # var matmul_K = conv_shape.matmul_K()

    # Accessing A is more expensive in im2col than accessing B.
    # Time a factor to M to var the heuristic bias on partitioning M.
    # TODO: make this bias factor part of function parameter/argument and
    # unifies interface with matmul partition, e.x. bias=1 for matmul.
    alias bias = 0.25
    var matmul_M_biased = Int(max(Float64(matmul_M) * bias, 1))

    # The ideal partition in theory is to balance the cost of memory access in
    # M and N dimensions using square sub-matrix (after applying the bias).
    var ideal_num_col_tasks = sqrt(
        ceildiv(matmul_N * max_num_tasks, matmul_M_biased)
    )
    var num_row_tasks = max_num_tasks // ideal_num_col_tasks
    var num_col_tasks = ideal_num_col_tasks

    # There must at least have enough elements to support a micro kernel.
    # Do not partition F when num_groups > 1.
    var max_num_col_tasks = min(
        ceildiv(matmul_N, micro_kernel_f), max_num_tasks
    )
    if ideal_num_col_tasks > max_num_col_tasks:
        num_col_tasks = max_num_col_tasks
        num_row_tasks = max_num_tasks // num_col_tasks
    # In this branch, not all threads get used for ideal_num_col_tasks
    # Check for alternative factorizations use the most threads.
    elif max_num_tasks % ideal_num_col_tasks != 0:
        # Set 20% deviation.
        var eps = ceildiv(2 * ideal_num_col_tasks, 10)
        max_num_col_tasks = min(max_num_col_tasks, ideal_num_col_tasks + eps)
        var num_col_tasks_tmp = max(ideal_num_col_tasks - eps, 1)
        var num_threads_used = (
            max_num_tasks // ideal_num_col_tasks
        ) * ideal_num_col_tasks
        while num_col_tasks_tmp <= max_num_col_tasks:
            var num_row_tasks_tmp = max_num_tasks // num_col_tasks_tmp
            if num_row_tasks_tmp * num_col_tasks_tmp >= num_threads_used:
                num_col_tasks = num_col_tasks_tmp
                num_row_tasks = num_row_tasks_tmp
                num_threads_used = num_row_tasks_tmp * num_col_tasks_tmp
            num_col_tasks_tmp += 1

    var max_num_row_tasks = max(matmul_M // min_rows_per_task, 1)
    num_row_tasks = min(max_num_row_tasks, num_row_tasks)

    # Do not partition channels when num_groups > 1.
    var max_num_channel_tasks = max(
        conv_shape.c // min_c_per_task, 1
    ) if conv_shape.num_groups == 1 and conv_shape.rank == 2 else 1
    var num_channel_tasks = min(
        max_num_channel_tasks,
        max_num_tasks // (num_row_tasks * num_col_tasks),
    )

    var num_batch_group_tasks = min(
        conv_shape.n * conv_shape.num_groups, num_row_tasks
    )

    num_row_tasks = num_row_tasks // num_batch_group_tasks

    return Index(
        num_batch_group_tasks, num_channel_tasks, num_col_tasks, num_row_tasks
    )


@always_inline
fn get_partition(
    task_id: Int,
    num_partitions: IndexList[4],
    conv_shape: ConvShape,
    micro_kernel_height: Int,
    micro_kernel_f_size: Int,
) -> ConvPartition:
    var task_id_f = task_id % num_partitions[2]
    var quotient = task_id // num_partitions[2]
    var task_id_c = quotient % num_partitions[1]
    quotient = quotient // num_partitions[1]
    var task_id_howo = quotient % num_partitions[3]
    var task_id_ng = quotient // num_partitions[3]

    var ng_range = partition_work(
        task_id_ng, num_partitions[0], conv_shape.n * conv_shape.num_groups, 1
    )

    var c_range = partition_work(task_id_c, num_partitions[1], conv_shape.c, 1)

    var f_range = partition_work(
        task_id_f,
        num_partitions[2],
        conv_shape.f // conv_shape.num_groups,
        micro_kernel_f_size,
    )

    # Merge output space loops when there is no padding and 2D.
    # Otherwise the partition granularity is a row.
    # TODO: generalize to 1D and 3D.
    var merge_loop = not conv_shape.padded() and conv_shape.rank == 2
    var work_unit = micro_kernel_height if merge_loop else 1
    var work_load = conv_shape.output_image_flat_size() if merge_loop else conv_shape.ho()
    var howo_range = partition_work(
        task_id_howo, num_partitions[3], work_load, work_unit
    )

    return ConvPartition(
        ng_offset=ng_range[0],
        ng_size=ng_range[1],
        f_offset=f_range[0],
        f_size=f_range[1],
        ho_or_howo_offset=howo_range[0],
        ho_or_howo_size=howo_range[1],
        c_offset=c_range[0],
        c_size=c_range[1],
    )


# ===-----------------------------------------------------------------------===#
# Convolution Algorithms Selection
# ===-----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------=== #
# align_down_residual
# ===----------------------------------------------------------------------=== #


@always_inline
fn align_down_residual(value: Int, alignment: Int) -> Int:
    """Returns the remainder after aligning down value to alignment.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        The remainder after aligning down value to the closest multiple of
        alignment. In other words, value - align_down(value, alignment).
    """
    return value - align_down(value, alignment)
