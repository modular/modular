# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_down_residual, div_ceil, fma, max, min
from sys.info import (
    alignof,
    has_avx2,
    has_avx512f,
    has_neon,
    simdbytewidth,
    simdwidthof,
)
from sys.intrinsics import PrefetchOptions

from algorithm import (
    sync_parallelize,
    sync_parallelize,
    tile,
    tile_middle_unswitch_boundaries,
    unroll,
    unswitch,
    vectorize,
    vectorize_unroll,
)
from .AccumulateSIMD import (
    accumulate,
    init_register_tile,
    load_register_tile,
    store_register_tile,
)
from .ConvUtils import (
    ConvInfo,
    ConvInfoStatic,
    ConvShape,
    ConvPartition,
    get_conv2d_shape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
    get_partition,
)
from .Image import Image2DLayout, ImageData, ImageShape
from Matmul import (
    GemmShape,
    MatmulInnerLoopBPacked,
    PackMatrixCols,
    PackMatrixRows,
    calculate_tile_n_k,
)
from MatmulUtils import (
    PartitionHeuristic,
    get_matmul_prefetch_b_distance_k,
    get_min_task_size,
    get_partitioned_matmul,
    get_partitioned_matmul_im2col,
    partition_work,
)
from memory import memset_zero, stack_allocation
from memory.buffer import (
    Buffer,
    DynamicRankBuffer,
    NDBuffer,
    _compute_ndbuffer_offset,
    partial_simd_load,
    partial_simd_store,
    prod_dims,
)
from memory.unsafe import DTypePointer
from .ShapeFuncUtils import get_sliding_window_out_dim

from utils.index import Index, StaticIntTuple
from utils.list import Dim, DimList
from utils._optional import Optional
from runtime.llcl import Runtime


@value
struct Naive2dConvolution[
    output_type: DType,
    input_type: DType,
    filter_type: DType,
]:
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: DTypePointer[output_type]
    var input: DTypePointer[input_type]
    var filter: DTypePointer[filter_type]
    var pad_d: StaticIntTuple[2]
    var pad_h: StaticIntTuple[2]
    var pad_w: StaticIntTuple[2]
    var stride: StaticIntTuple[3]
    var dilation: StaticIntTuple[3]
    var num_groups: Int

    # Derived params.
    var output_shape: StaticIntTuple[5]  # NDHWC layout.
    var input_shape: StaticIntTuple[5]  # NDHWC layout.
    var filter_shape: StaticIntTuple[5]  # QRSCF layout.

    @staticmethod
    fn run(
        output: DTypePointer[output_type],
        input: DTypePointer[input_type],
        filter: DTypePointer[filter_type],
        output_shape: StaticIntTuple[5],
        input_shape: StaticIntTuple[5],
        filter_shape: StaticIntTuple[5],
        pad_d: StaticIntTuple[2],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[3],
        dilation: StaticIntTuple[3],
        num_groups: Int,
    ):
        # Create an instance of the convolution op.
        let naive2d_convolution = Naive2dConvolution[
            output_type, input_type, filter_type
        ](
            output,
            input,
            filter,
            output_shape,
            input_shape,
            filter_shape,
            pad_d,
            pad_h,
            pad_w,
            stride,
            dilation,
            num_groups,
        )

        # Run the actual loops and computations.
        naive2d_convolution._outer_loop()

    fn __init__(
        inout self,
        output: DTypePointer[output_type],
        input: DTypePointer[input_type],
        filter: DTypePointer[filter_type],
        output_shape: StaticIntTuple[5],
        input_shape: StaticIntTuple[5],
        filter_shape: StaticIntTuple[5],
        pad_d: StaticIntTuple[2],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[3],
        dilation: StaticIntTuple[3],
        num_groups: Int,
    ):
        self.output = output
        self.input = input
        self.filter = filter
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.pad_d = pad_d
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride = stride
        self.dilation = dilation
        self.num_groups = num_groups

    fn _outer_loop(self):
        """Implementation of the outermost loop of a convolution operator with
        loops covering the iteration space of batch, filter count, height and wi-
        dth dimensions.
        """
        # Iterate on output batch dimension.
        for n in range(self.output_shape[0]):
            # Iterate on filter dimension.
            for f in range(self.output_shape[4]):
                # Iterate on output H dimension.
                for do in range(self.output_shape[1]):
                    # Iterate on output H dimension.
                    for ho in range(self.output_shape[2]):
                        # Iterate on output W dimension.
                        for wo in range(self.output_shape[3]):
                            # Compute the result value at this specific output posit-
                            #  ion.
                            self._compute_point(n, do, ho, wo, f)

    fn _compute_point(self, n: Int, do: Int, ho: Int, wo: Int, f: Int):
        """Implementation of the inner loop computation of a conv2d operator
        producing a single scalar value at the given output tensor index.
        """
        # Initialize the result of this point.
        var value: SIMD[output_type, 1] = 0

        # Input dims.
        let D = self.input_shape[1]
        let H = self.input_shape[2]
        let W = self.input_shape[3]
        let C = self.input_shape[4]
        let image_bound = Index(D, H, W)
        let C_per_group = C // self.num_groups

        # Filter dims.
        let Q = self.filter_shape[0]
        let R = self.filter_shape[1]
        let S = self.filter_shape[2]

        # Output dims.
        let DO = self.output_shape[1]
        let HO = self.output_shape[2]
        let WO = self.output_shape[3]
        let F = self.output_shape[4]

        let g = f // (F // self.num_groups)

        for q in range(Q):
            for r in range(R):
                for s in range(S):
                    # Compute input access index, on the H and W dimension.
                    let dhw = (
                        # Output HxW with striding.
                        Index(do, ho, wo) * self.stride
                        +
                        # Filter RxS with dilation.
                        (Index(q, r, s) * self.dilation)
                        -
                        # Padding offset, using the left padding only here.
                        Index(self.pad_d[0], self.pad_h[0], self.pad_w[0])
                    )

                    # Check that the current image index is within valid range
                    #  on the input image data tensor.
                    if Index(0, 0, 0) <= dhw < image_bound:
                        # Iterate on channels dimension.
                        for c in range(C_per_group * g, C_per_group * (g + 1)):
                            # Accumulate product of input data filter data.
                            let input_val = self.input[
                                c
                                + C
                                * (dhw[2] + W * (dhw[1] + H * (dhw[0] + D * n)))
                            ]
                            let c_in_group = c % C_per_group
                            let filter_val = self.filter[
                                f
                                + F
                                * (
                                    c_in_group
                                    + C_per_group * (s + S * (r + R * q))
                                )
                            ]
                            value += (
                                input_val.cast[output_type]()
                                * filter_val.cast[output_type]()
                            )

        # Store the computed output at the given output position..
        self.output.simd_store(
            f + F * (wo + WO * (ho + HO * (do + DO * n))), value
        )


# ===----------------------------------------------------------------------=== #
# Direct convolution helpers
# ===----------------------------------------------------------------------=== #


@always_inline
fn _m_to_n_ho_wo_nhwc(m: Int, WO: Int, HO: Int) -> StaticIntTuple[3]:
    """Converts post-im2col m dimension index to pre-im2col coordinates on
    (N, Hout, Wout) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            The translated 3d indices in (N, Hout, Wout) format.
    TODO(Fixel): This utility should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    let n = m // (HO * WO)
    let ho = (m % (HO * WO)) // WO
    let wo = m % WO
    return Index(n, ho, wo)


# Elementwise epilogue signature
alias elementwise_epilogue_type = fn (Int, Int, Int, Int, Int) escaping -> None


# Reduce helper when the input channel dimension is partitioned.
@always_inline
fn _reduce_output[
    simd_size: Int,
    elementwise_epilogue_enabled: Bool,
](
    scratch: DTypePointer,
    output: DTypePointer[scratch.type],
    N: Int,
    output_space_dims: StaticIntTuple,
    F: Int,
    num_partitions: Int,
    num_threads: Int,
    elementwise_epilogue_fn: elementwise_epilogue_type,
):
    let num_rows = N * output_space_dims.flattened_length()
    let buf_size = num_rows * F

    # Reduce from the output scratch buffer to the actual output.
    @parameter
    @always_inline
    fn reduce_task(tid: Int):
        # Use all threads in reduction.
        let reduce_range = partition_work(tid, num_threads, num_rows, 1)

        @parameter
        @always_inline
        fn sum[width: Int](offset: Int):
            let tid_output_offset = reduce_range[0] * F + offset
            var vec = scratch.simd_load[width](tid_output_offset)
            # The number of partitions here is typically small.
            # There may not be much benefit from unrolling the reduction axis.
            # Only unroll the last dimension.
            for i in range(1, num_partitions):
                vec += scratch.simd_load[width](
                    tid_output_offset + i * buf_size
                )
            output.simd_store[width](tid_output_offset, vec)

        vectorize_unroll[simd_size, 4, sum](reduce_range[1] * F)

        @parameter
        if elementwise_epilogue_enabled:
            for m in range(reduce_range[0], reduce_range[0] + reduce_range[1]):
                let nhowo = _m_to_n_ho_wo_nhwc(
                    m, output_space_dims[0], output_space_dims[1]
                )
                elementwise_epilogue_fn(nhowo[0], nhowo[1], nhowo[2], 0, F)

    # NOTE: synchronous, so use of locally allocated output_ptr is safe.
    sync_parallelize[reduce_task](num_threads)


# ===----------------------------------------------------------------------=== #
# Direct Convolution Entry Point                                               #
# ===----------------------------------------------------------------------=== #


@value
struct ConvDirectNHWC[
    input_rank: Int,
    filter_rank: Int,
    output_rank: Int,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    conv_attr: ConvInfoStatic,
    elementwise_epilogue_enabled: Bool,
]:
    """Implement the outer loops for direct convolution.
    Collapse N, HO, WO into one dimension n_ho_wo. Tile n_ho_wo, C, and F.
    The tile factor for C and F are chosen by a heuristic prioritizing C.
    n_ho_wo is tiled by micro kernel's height.

    If n_ho_wo is large enough to spill LLC, we may need to tile n_ho_wo as the
    outer most loop with a factor fit in LLC.

    Assume F is divisible at least by simd_size.
    """

    var output: NDBuffer[output_rank, output_shape, output_type]
    var input: NDBuffer[input_rank, input_shape, input_type]
    var filter: NDBuffer[filter_rank, filter_shape, filter_type]

    var conv_shape: ConvShape[input_rank - 2]

    # Support partition in 4 dims: (n, c, f, ho_or_howo). If the input is
    # padded, the output spacial dims are merged into one as howo. If not
    # padded, only ho is partitioned for now.
    var partition: ConvPartition

    var cf_tile_size: StaticIntTuple[2]

    var elementwise_epilogue_fn: elementwise_epilogue_type

    # If shapes and attributes are known at compile time
    alias packed_and_fully_static = conv_attr.all_known() and input_shape.all_known[
        1, input_rank
    ]() and output_shape.all_known[
        1, output_rank
    ]() and filter_shape.all_known[
        filter_rank
    ]() and filter_packed

    @staticmethod
    fn run(
        output: NDBuffer[output_rank, output_shape, output_type],
        input: NDBuffer[input_rank, input_shape, input_type],
        filter: NDBuffer[filter_rank, filter_shape, filter_type],
        conv_shape: ConvShape[input_rank - 2],
    ) raises:
        fn direct_null_elementwise_epilogue(
            n: Int, ho: Int, wo: Int, f_offset: Int, f_size: Int
        ):
            pass

        Self.run(
            output,
            input,
            filter,
            conv_shape,
            direct_null_elementwise_epilogue,
        )

    @staticmethod
    fn run(
        output: NDBuffer[output_rank, output_shape, output_type],
        input: NDBuffer[input_rank, input_shape, input_type],
        filter: NDBuffer[filter_rank, filter_shape, filter_type],
        conv_shape: ConvShape[input_rank - 2],
        elementwise_epilogue_fn: elementwise_epilogue_type,
    ) raises:
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_shape = get_micro_kernel_shape[
            output_shape.at[output_rank - 2](),  # WO
            output_shape.at[output_rank - 1](),  # F
            conv_attr,
            simd_size,
        ]()
        alias micro_kernel_height = micro_kernel_shape[0]
        alias micro_kernel_width = micro_kernel_shape[1]
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        let cf_tile_size = get_conv_tile_shape[filter_type](
            conv_shape.c,
            conv_shape.filter_window_flat_size(),
            micro_kernel_width,
        )

        @parameter
        if conv_attr.num_groups.has_value():
            constrained[
                filter_packed or conv_attr.num_groups == Dim(1),
                (
                    "if number of conv groups is statically known, conv filter"
                    " must be prepacked when num_groups > 1"
                ),
            ]()

        if conv_shape.num_groups > 1 and not filter_packed:
            raise Error("grouped conv requires packed filter")
        if conv_shape.c % conv_shape.num_groups != 0:
            raise Error("channel count must be divisible by group count")
        if conv_shape.f % conv_shape.num_groups != 0:
            raise Error("filter count must be divisible by group count")

        # Number of partitions in n, ho_wo, c, f dimensions.
        let num_threads = Runtime().parallelism_level()
        let num_partitions = get_conv_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_threads, conv_shape)
        let num_tasks = num_partitions.flattened_length()

        # Wrap the pointer inside NDBuffer so it can be properly captured by async closure.
        var output_ptr = output.data
        let output_size = prod_dims[0, output_rank](output)
        let scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = DTypePointer[output_type].alloc(scratch_size)
        let output_scratch = Buffer[Dim(), output_type](
            output_ptr, scratch_size
        )

        @parameter
        @always_inline
        fn task_func(task_id: Int):
            let partition = get_partition(
                task_id,
                num_partitions,
                conv_shape,
                micro_kernel_height,
                micro_kernel_f_size,
            )

            if partition.empty():
                return

            let task_tile_size = Index(
                min(cf_tile_size[0], partition.c_size), cf_tile_size[1]
            )

            # TODO: Need to have a more robust way to compute task_id_c
            let task_id_c = (task_id // num_partitions[2]) % num_partitions[1]
            let task_output = NDBuffer[output_rank, output_shape, output_type](
                output_scratch.data.offset(task_id_c * output_size),
                output.dynamic_shape,
            )

            let instance = ConvDirectNHWC[
                input_rank,
                filter_rank,
                output_rank,
                input_shape,
                filter_shape,
                output_shape,
                input_type,
                filter_type,
                output_type,
                filter_packed,
                conv_attr,
                elementwise_epilogue_enabled,
            ](
                task_output,
                input,
                filter,
                conv_shape,
                partition,
                task_tile_size,
                elementwise_epilogue_fn,
            )
            instance._batch_group_loop()

        if num_partitions[1] > 1:
            sync_parallelize[task_func](num_tasks)

            # Reduce from the output scratch buffer to the actual output.
            _reduce_output[
                simd_size, elementwise_epilogue_enabled and input_rank == 4
            ](
                output_scratch.data,
                output.data,
                conv_shape.n,
                conv_shape.output_space_dims(),
                conv_shape.f,
                num_partitions[1],
                num_threads,
                elementwise_epilogue_fn,
            )

            output_ptr.free()
        else:
            # Use sync to work around #12624
            sync_parallelize[task_func](num_tasks)

    fn _batch_group_loop(self):
        """Loop over the batch and group dimensions. The two dimension are
        merged and partitioned for parallelism."""

        @always_inline
        @parameter
        fn body[padded: Bool]():
            for ng in range(
                self.partition.ng_offset,
                self.partition.ng_offset + self.partition.ng_size,
            ):
                let n = ng // self.conv_shape.num_groups
                let g = ng % self.conv_shape.num_groups
                self._c_tile_loop[padded](n, g, self.cf_tile_size[0])

        unswitch[body](self.conv_shape.padded())

    fn _c_tile_loop[padded: Bool](self, n: Int, g: Int, tile_size: Int):
        """Loop over C tiles."""

        alias apply_static_shape_optimization = self.packed_and_fully_static and padded and conv_attr.num_groups == Dim(
            1
        )

        @always_inline
        @parameter
        fn c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
            # Only apply static shape optimizations to shapes with padding since
            # there is a fast path for pointwise (no padding) conv with strides.
            # Grouped conv logic has not been plumbed into static specialized funcs yet.
            @parameter
            if apply_static_shape_optimization:
                self._f_tile_loop_static[False](n, c_tile_offset, c_tile_size)
            else:
                self._f_tile_loop[padded, False](
                    n, g, c_tile_offset, c_tile_size
                )

        # Can't fuse epilogue inside conv if C is partitioned
        if self.partition.c_size < self.conv_shape.c:
            tile[c_tile_iteration](
                self.partition.c_offset,
                self.partition.c_offset + self.partition.c_size,
                tile_size,
            )
        # C is not partitioned, fuse epilogue in the last C tile.
        else:
            # for g in range(self.conv_shape.num_groups):
            let c_start = g * self.conv_shape.c_per_group()
            let c_round_by_tile = align_down(
                (self.conv_shape.c_per_group() - 1), tile_size
            )
            let c_round_by_tile_residual = self.conv_shape.c_per_group() - c_round_by_tile
            tile[c_tile_iteration](
                c_start,
                c_start + c_round_by_tile,
                tile_size,
            )

            # Update the last c tile with fusion
            @parameter
            if apply_static_shape_optimization:
                self._f_tile_loop_static[True](
                    n,
                    c_start + c_round_by_tile,
                    c_round_by_tile_residual,
                )
            else:
                self._f_tile_loop[padded, True](
                    n,
                    g,
                    c_start + c_round_by_tile,
                    c_round_by_tile_residual,
                )

    fn _f_tile_loop[
        padded: Bool, last_c_tile: Bool
    ](self, n: Int, g: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        alias micro_kernel_width = get_direct_conv_micro_kernel_width()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # TODO: Extend the merged loop to support 1d and 3d.
        # For now, only merge HO and WO dims for 2D conv w/o padding.
        alias merge_output_space_loops = (not padded) and input_rank == 4

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            @parameter
            if not merge_output_space_loops:
                self.output_space_loop[
                    micro_kernel_height, size // simd_size, False, last_c_tile
                ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)
            else:
                self.output_space_flat_loop[size, False, last_c_tile](
                    n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

        let f_per_group = self.conv_shape.f_per_group()

        # The partition heuristic sees F_per_group and may partition it.
        # The partition's F_offset should be added to the group's F offset to
        # get the actually offset in output's F dim.
        let group_f_offset = g * f_per_group + self.partition.f_offset

        let group_f_end_align_simd = group_f_offset + align_down(
            self.partition.f_size, simd_size
        )

        # The first tile size is based on cache size. Within the tile
        # it's stepped by the micro kernel size in F. The rest is stepped
        # by simd_size. If F is not multiple of simd_size, the residual
        # is padded with 0 to fit a simd vector in the packed filter.
        tile[
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
            f_tile_iteration,
        ](
            group_f_offset,
            group_f_end_align_simd,
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
        )

        # If this is the last partition in F and it's not a multiple of simd_size.
        # The partition is aligned by micro_kernel_f_size, so only the last
        # partition is possible to have residual.
        let residual = align_down_residual(f_per_group, simd_size)
        if (
            self.partition.f_offset + self.partition.f_size == f_per_group
            and residual > 0
        ):

            @parameter
            if not merge_output_space_loops:
                self.output_space_loop[
                    micro_kernel_height, 1, True, last_c_tile
                ](
                    n,
                    group_f_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )
            else:
                self.output_space_flat_loop[simd_size, True, last_c_tile](
                    n,
                    group_f_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )

    @always_inline
    fn is_new_c_accum(self, c_idx: Int) -> Bool:
        # returns true when processing first C in a group or first C in a C partition
        if self.conv_shape.num_groups > 1:
            return self.conv_shape.c_in_group(c_idx) == 0
        return c_idx == self.partition.c_offset

    fn update_output_tile_no_padding[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        output_flat_coord: Int,
    ):
        constrained[
            not has_residual or (has_residual and micro_kernel_width == 1),
            "Use Height x 1 kernel for residual in F.",
        ]()

        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # Base input offsets.
        let input_base_offsets = Buffer[
            micro_kernel_height, DType.int32
        ].stack_allocation()

        @unroll
        for i in range(micro_kernel_height):
            input_base_offsets.simd_store[1](
                i,
                self.conv_shape.output_flat_coord_to_input_offset(
                    n, output_flat_coord + i
                )
                + c_tile_offset,
            )

        alias alignment = alignof[SIMD[output_type, simd_size]]()
        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ].aligned_stack_allocation[alignment]()

        let output_offset = self.conv_shape.f * (
            n * self.conv_shape.output_image_flat_size() + output_flat_coord
        ) + f_tile_offset

        if self.is_new_c_accum(c_tile_offset):
            self._init_output_micro_tile[
                micro_kernel_height, micro_kernel_width, simd_size
            ](output_micro_tile)
        else:
            self._load_output_micro_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](self.output.data + output_offset, output_micro_tile)

        var filter_ptr: DTypePointer[filter_type] = self.filter.data

        @parameter
        if filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter,
                self.conv_shape.c_to_group(c_tile_offset),  # group index
                self.conv_shape.f_per_group(),
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr.offset(
                # Jump over f_tile_offset in current group.
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.r()
                * self.conv_shape.s()
                * self.conv_shape.c_per_group()
                # Jump over c_tile_offset in current group.
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )

        for r in range(self.conv_shape.r()):
            for s in range(self.conv_shape.s()):
                let input_offset = self.conv_shape.c * (
                    s + self.conv_shape.w() * r
                )

                # Unpacked version. For each (r, s), we first offset the
                # filter pointer by (r, s) plus c_tile_offset. Later for
                # each c, we access micro_kernel_f_size contiguous elements.
                # These contiguous segments are strided by F.
                @parameter
                if not filter_packed:
                    filter_ptr = self.filter.data.offset(
                        (s + r * self.conv_shape.s())
                        * self.conv_shape.c
                        * self.conv_shape.f
                        + c_tile_offset * self.conv_shape.f
                        + f_tile_offset
                    )

                self._accumulate[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    c_fully_cached,
                    has_residual,
                    # prefetch offset
                    4 if not has_neon() else -1,
                ](
                    input_base_offsets,
                    input_offset,
                    c_tile_size,
                    filter_ptr,
                    output_micro_tile,
                )

                # Shift C*f to get the next point in stencil (s+1) for FRSCf layout.
                if filter_packed:
                    filter_ptr = filter_ptr.offset(
                        self.conv_shape.c_per_group() * micro_kernel_f_size
                    )

        self._store_output_micro_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            has_residual,
        ](output_micro_tile, self.output.data + output_offset)

        @parameter
        if elementwise_epilogue_enabled and last_c_tile:
            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            let f_tile_size_bounded: Int

            @parameter
            if has_residual:
                f_tile_size_bounded = (
                    self.conv_shape.f_per_group()
                    - self.conv_shape.f_in_group(f_tile_offset)
                )
            else:
                f_tile_size_bounded = f_tile_size

            for m in range(
                output_flat_coord, output_flat_coord + micro_kernel_height
            ):
                # The micro tile may cover points in different rows/images.
                # Convert the 1D index back to (n, ho, wo).
                self.elementwise_epilogue_fn(
                    n,
                    m // self.conv_shape.wo(),
                    m % self.conv_shape.wo(),
                    f_tile_offset,
                    f_tile_size_bounded,
                )

    @always_inline
    fn _init_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
    ](
        self,
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
    ):
        """Initialize a micro tile to zero.
        Arguments:
            n_ho_wo: offset of micro tile in fused (n, ho, wo) dimension.
            f: offset of micro tile in F dimension.
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """

        @always_inline
        @parameter
        fn body[idx0: Int, idx1: Int]():
            output_micro_tile.simd_store[simd_size](
                Index(idx0, idx1 * simd_size), SIMD[output_type, simd_size](0.0)
            )

        unroll[micro_kernel_height, micro_kernel_width, body]()

    @always_inline
    fn _load_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_base: DTypePointer[output_type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
    ):
        """Load a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. The residual
              is loaded and padded with zero to fit a simd vector.

        Arguments:
            output_base: Point to micro tile start, (n, ho, wo, f).
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """
        var output_ptr = output_base

        @unroll
        for i in range(micro_kernel_height):

            @unroll
            for j in range(micro_kernel_width):

                @parameter
                if has_residual:
                    let residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    output_micro_tile.simd_store[simd_size](
                        Index(i, j * simd_size),
                        partial_simd_load[simd_size](
                            output_ptr.offset(j * simd_size), 0, residual, 0.0
                        ),
                    )
                else:
                    output_micro_tile.simd_store[simd_size](
                        Index(i, j * simd_size),
                        output_ptr.offset(j * simd_size).simd_load[simd_size](),
                    )

            @parameter
            if output_shape.at[output_rank - 1]().has_value():
                alias F = output_shape.get[output_rank - 1]()
                output_ptr = output_ptr.offset(F)
            else:
                output_ptr = output_ptr.offset(self.conv_shape.f)

    @always_inline
    fn _store_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
        output_base: DTypePointer[output_type],
    ):
        """Store a micro tile from the output buffer.
        Parameters:
            has_residual: True when F is not multiple of simd_size. Only the
              residual elements within the simd vector are stored to output.

        Arguments:
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
            output_base: Point to micro tile start, (n, ho, wo, f).
        """
        var output_ptr = output_base

        @unroll
        for i in range(micro_kernel_height):

            @unroll
            for j in range(micro_kernel_width):
                let output_vec = output_micro_tile.simd_load[simd_size](
                    Index(i, j * simd_size)
                )

                @parameter
                if has_residual:
                    let residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    partial_simd_store[simd_size](
                        output_ptr.offset(j * simd_size),
                        0,
                        residual,
                        output_vec,
                    )
                else:
                    output_ptr.offset(j * simd_size).simd_store[simd_size](
                        output_vec
                    )

            @parameter
            if output_shape.at[output_rank - 1]().has_value():
                alias F = output_shape.get[output_rank - 1]()
                output_ptr = output_ptr.offset(F)
            else:
                output_ptr = output_ptr.offset(self.conv_shape.f)

    @always_inline
    fn _load_filter_vec[
        has_residual: Bool, simd_size: Int
    ](self, filter_ptr: DTypePointer[filter_type], offset: Int) -> SIMD[
        filter_type, simd_size
    ]:
        """Load a simd vector from the filter.
        There may be residual elements i.e. F - offset < simd_size. Partial
        simd load instrinc is used if the filter is not packed.  Otherwise,
        it's safe to load a vector since the filter has been properly padded.
        """
        let filter_vec: SIMD[filter_type, simd_size]

        # Partial load if F is not multiple of simd_size.
        @parameter
        if has_residual and not filter_packed:
            let residual = self.conv_shape.f - (
                self.conv_shape.f // simd_size
            ) * simd_size
            # TODO: Follow #20211 to optimize it for NEON.
            filter_vec = partial_simd_load[simd_size](
                filter_ptr, 0, residual, 0.0
            )
        # It's always safe to load a full vector from packed filter because
        # the filter is padded to multiple simd_size during pre-packing.
        else:
            filter_vec = filter_ptr.simd_load[simd_size](offset)

        return filter_vec

    @always_inline
    fn _accumulate_default[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: Buffer[micro_kernel_height, DType.int32],
        input_offset: Int,
        c_tile_size: Int,
        filter_base: DTypePointer[filter_type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
    ):
        """Accumulate with register tiling on SIMD ISAs other than NEON.
        It has been optimized for AVX512 and AVX2.
        """
        constrained[not has_neon()]()

        alias micro_kernel_f_size = micro_kernel_width * simd_size

        var offset = input_offset
        var filter_ptr = filter_base

        for c in range(c_tile_size):

            @parameter
            if prefetch_offset > 0:
                # fmt: off
                let dist = prefetch_offset * micro_kernel_f_size \
                            if c_fully_cached or c < c_tile_size - prefetch_offset \
                            else ( \
                                prefetch_offset + self.conv_shape.c - c_tile_size \
                            ) * micro_kernel_f_size
                # fmt: on
                @unroll
                for idx in range(micro_kernel_width):
                    filter_ptr.offset(dist + idx * simd_size).prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ]()

            # micro kernel with register blocking
            @unroll
            for i in range(micro_kernel_height):
                let input_val = self.input.data.offset(
                    input_base_offsets[i].value + offset
                ).simd_load[1]()
                let input_vec = SIMD[input_type, simd_size](input_val)

                @unroll
                for j in range(micro_kernel_width):
                    let filter_vec = self._load_filter_vec[
                        has_residual, simd_size
                    ](filter_ptr, j * simd_size)

                    # The following should be lifted to registers and show up as
                    # FMA instructions.
                    let output_micro_idx = Index(i, j * simd_size)
                    var output_vec = output_micro_tile.simd_load[simd_size](
                        output_micro_idx
                    )
                    output_vec = fma[output_type, simd_size](
                        input_vec.cast[output_type](),
                        filter_vec.cast[output_type](),
                        output_vec,
                    )
                    output_micro_tile.simd_store[simd_size](
                        output_micro_idx, output_vec
                    )

            # FRSCf: micro f segments are packed continuously.
            @parameter
            if filter_packed:
                filter_ptr = filter_ptr.offset(micro_kernel_f_size)
            # RSCF: jump to the next row for the next f segment.
            else:
                filter_ptr = filter_ptr.offset(self.conv_shape.f)

            offset += 1

    @always_inline
    fn _accumulate_neon[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: Buffer[micro_kernel_height, DType.int32],
        input_offset: Int,
        c_tile_size: Int,
        filter_base: DTypePointer[filter_type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
    ):
        """Accumulate with register tiling on NEON architectures."""

        constrained[has_neon()]()

        alias micro_kernel_f_size = micro_kernel_width * simd_size

        @parameter
        @always_inline
        fn micro_kernel[num_lanes: Int](offset: Int):
            let input_vecs = stack_allocation[
                micro_kernel_height, SIMD[input_type, num_lanes]
            ]()

            # Load vectors of size num_lanes from input.
            @unroll
            for i in range(micro_kernel_height):
                # input_base_offset + input_offset is the actual offset
                # for the accumulation.
                let input_vec = self.input.data.offset(
                    input_base_offsets[i].value + input_offset + offset
                ).simd_load[num_lanes]()
                input_vecs[i] = input_vec

            var filter_ptr: DTypePointer[filter_type]

            @parameter
            if filter_packed:
                filter_ptr = filter_base.offset(offset * micro_kernel_f_size)
            else:
                filter_ptr = filter_base.offset(offset * self.conv_shape.f)

            @unroll
            for lane in range(num_lanes):

                @unroll
                for j in range(micro_kernel_width):
                    let filter_vec = self._load_filter_vec[
                        has_residual, simd_size
                    ](filter_ptr, j * simd_size)

                    @unroll
                    for i in range(micro_kernel_height):
                        let input_vec = input_vecs[i]
                        let output_micro_idx = Index(i, j * simd_size)
                        var output_vec = output_micro_tile.simd_load[simd_size](
                            output_micro_idx
                        )
                        # Neon can broadcast from an element in simd vector.
                        output_vec = fma[output_type, simd_size](
                            input_vec[lane].cast[output_type](),
                            filter_vec.cast[output_type](),
                            output_vec,
                        )
                        output_micro_tile.simd_store[simd_size](
                            output_micro_idx, output_vec
                        )

                @parameter
                if filter_packed:
                    filter_ptr = filter_ptr.offset(micro_kernel_f_size)
                else:
                    filter_ptr = filter_ptr.offset(self.conv_shape.f)

        tile[micro_kernel, VariadicList[Int](simd_size, 1)](0, c_tile_size)

    @always_inline
    fn _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        c_fully_cached: Bool,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: Buffer[micro_kernel_height, DType.int32],
        input_offset: Int,
        c_tile_size: Int,
        filter_base: DTypePointer[filter_type],
        output_micro_tile: NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ],
    ):
        @parameter
        if has_neon():
            self._accumulate_neon[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                c_fully_cached,
                has_residual,
                prefetch_offset,
            ](
                input_base_offsets,
                input_offset,
                c_tile_size,
                filter_base,
                output_micro_tile,
            )
        else:
            self._accumulate_default[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                c_fully_cached,
                has_residual,
                prefetch_offset,
            ](
                input_base_offsets,
                input_offset,
                c_tile_size,
                filter_base,
                output_micro_tile,
            )

    @always_inline
    fn _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        c_tile_size: Int,
        input_stride: Int,
        input_base: DTypePointer[input_type],
        filter_base: DTypePointer[filter_type],
        output_ptr: DTypePointer[output_type],
    ):
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        let F = self.output.dim[3]()
        let filter_stride = micro_kernel_f_size if filter_packed else F

        accumulate[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            prefetch_offset=prefetch_offset,
            partial_load_b = has_residual and not filter_packed,
        ](
            c_tile_size,
            output_ptr,
            input_base,
            input_stride,
            filter_base,
            filter_stride,
            F % simd_size,
        )

    fn output_space_flat_loop[
        micro_kernel_f_size: Int, has_residual: Bool, last_c_tile: Bool
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias micro_kernel_width = micro_kernel_f_size // simd_size

        @always_inline
        @parameter
        fn iteration[tile_size: Int](output_flat_coord: Int):
            @always_inline
            @parameter
            fn body[c_fully_cached: Bool]():
                self.update_output_tile_no_padding[
                    tile_size,  # micro kernel height
                    micro_kernel_width,
                    c_fully_cached,
                    has_residual,
                    last_c_tile,
                ](
                    n,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    output_flat_coord,
                )

            # c_fully_cached means the C dimension is fully covered in the
            # cache tile.
            unswitch[body](self.conv_shape.c == c_tile_size)

        # After the loop can't be stepped with micro_kernel_height,
        # it will step by 5, 4, 3, 2, 1.
        tile[iteration, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),](
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        )

    @always_inline
    fn output_space_loop[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # Current group index.
        let g = self.conv_shape.f_to_group(f_tile_offset)

        # Filter pointer to the current cf tile offset location.
        var filter_ptr: DTypePointer[filter_type]

        @parameter
        if filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter, g, self.conv_shape.f_per_group()
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr.offset(
                # Jump over f_tile_offset in current group.
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.c_per_group()
                * self.conv_shape.filter_window_flat_size()
                # Jump over c_tile_offset in current group.
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )
        else:
            filter_ptr = self.filter.data.offset(
                c_tile_offset * self.conv_shape.f + f_tile_offset
            )

        # Pointer to input and output of the current sample (batch dim).
        # fmt: off
        let input_ptr  = self.input.data + c_tile_offset \
                       + self.conv_shape.input_image_flat_size() \
                       * self.conv_shape.c * n

        let output_ptr = self.output.data + f_tile_offset \
                       + self.conv_shape.output_image_flat_size() \
                       * self.conv_shape.f * n
        # fmt: on

        # Divide each row into three part:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        let left_pad_impact_end = div_ceil(
            self.conv_shape.pad_w[0], self.conv_shape.stride[input_rank - 3]
        )
        let right_pad_impact_start = (
            self.conv_shape.w()
            + self.conv_shape.pad_w[0]
            - self.conv_shape.s() * self.conv_shape.dilation[input_rank - 3]
        ) // self.conv_shape.stride[input_rank - 3] + 1

        @parameter
        if input_rank == 3:
            self.output_space_loop_1d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr,
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif input_rank == 4:
            self.output_space_loop_2d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr,
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif input_rank == 5:
            self.output_space_loop_3d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr,
                input_ptr,
                filter_ptr,
                n,
                self.is_new_c_accum(c_tile_offset),
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )

    @always_inline
    fn output_space_loop_1d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        output: DTypePointer,
        input: DTypePointer,
        filter: DTypePointer,
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        alias simd_size = simdwidthof[output_type]()

        # Offset by -pad_w because s loop starts from the leftmost neightbor
        # in padding. The kernel skip the padding point and increment the
        # pointer.
        var input_base = input - self.conv_shape.c * self.conv_shape.pad_w[0]

        # Points output to the start of the row
        var output_base = output

        @parameter
        @always_inline
        fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
            conv1d_update_wo_tile[
                height,
                micro_kernel_width,
                simd_size,
                filter_packed,
                effected_by_padding,
                has_residual,
                last_c_tile,
                # TODO: Enable epilogue for 1D and 3D.
                elementwise_epilogue_enabled and input_rank == 4,
            ](
                output_base,
                input_base,
                filter,
                first_c_tile_in_group,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                rebind[ConvShape[1]](self.conv_shape),
                n,
                wo,
                self.elementwise_epilogue_fn,
            )

            input_base = input_base.offset(
                height * self.conv_shape.stride[0] * self.conv_shape.c,
            )
            output_base = output_base.offset(height * self.conv_shape.f)

        tile_middle_unswitch_boundaries[
            work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
        ](
            0,
            left_pad_impact_end,
            right_pad_impact_start,
            self.conv_shape.wo(),
        )

    @always_inline
    fn output_space_loop_2d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        output: DTypePointer,
        input: DTypePointer,
        filter: DTypePointer,
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        alias simd_size = simdwidthof[output_type]()

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]

            # Points input to the start of the row.
            # Offset by -pad_w because s loop starts from the leftmost neightbor
            # in padding. The kernel skip the padding point and increment the
            # pointer.
            var input_base = input + self.conv_shape.c * (
                -self.conv_shape.pad_w[0] + self.conv_shape.w() * h
            )

            # Points output to the start of the row
            var output_base = output + self.conv_shape.f * self.conv_shape.wo() * ho

            @parameter
            @always_inline
            fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                conv2d_update_wo_tile[
                    height,
                    micro_kernel_width,
                    simd_size,
                    filter_packed,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
                    elementwise_epilogue_enabled,
                ](
                    output_base,
                    input_base,
                    filter,
                    first_c_tile_in_group,
                    c_tile_size,
                    f_tile_offset,
                    f_tile_size,
                    rebind[ConvShape[2]](self.conv_shape),
                    n,
                    Index(ho, wo),
                    self.elementwise_epilogue_fn,
                )

                input_base = input_base.offset(
                    height * self.conv_shape.stride[1] * self.conv_shape.c,
                )
                output_base = output_base.offset(height * self.conv_shape.f)

            tile_middle_unswitch_boundaries[
                work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.wo(),
            )

    @always_inline
    fn output_space_loop_3d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        output: DTypePointer,
        input: DTypePointer,
        filter: DTypePointer,
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        alias simd_size = simdwidthof[output_type]()

        for do in range(0, self.conv_shape.do()):
            let d = do * self.conv_shape.stride[0] - self.conv_shape.pad_d[0]

            for ho in range(
                self.partition.ho_or_howo_offset,
                self.partition.ho_or_howo_offset
                + self.partition.ho_or_howo_size,
            ):
                # fmt: off
                let h = ho * self.conv_shape.stride[1] - self.conv_shape.pad_h[0]
                # fmt: on

                # Points input to the start of the row.
                # Offset by -pad_w because s loop starts from the leftmost neightbor
                # in padding. The kernel skip the padding point and increment the
                # pointer.
                var input_base = input + self.conv_shape.c * (
                    -self.conv_shape.pad_w[0]
                    + self.conv_shape.w() * (h + self.conv_shape.h() * d)
                )

                # Points output to the start of the row
                var output_base = output + self.conv_shape.f * self.conv_shape.wo() * (
                    ho + self.conv_shape.ho() * do
                )

                @parameter
                @always_inline
                fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                    conv3d_update_wo_tile[
                        height,
                        micro_kernel_width,
                        simd_size,
                        filter_packed,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
                        # TODO: Enable epilogue for 1D and 3D.
                        elementwise_epilogue_enabled and input_rank == 4,
                    ](
                        output_base,
                        input_base,
                        filter,
                        first_c_tile_in_group,
                        c_tile_size,
                        f_tile_offset,
                        f_tile_size,
                        rebind[ConvShape[3]](self.conv_shape),
                        n,
                        Index(do, ho, wo),
                        self.elementwise_epilogue_fn,
                    )

                    input_base = input_base.offset(
                        height * self.conv_shape.stride[2] * self.conv_shape.c,
                    )
                    output_base = output_base.offset(height * self.conv_shape.f)

                tile_middle_unswitch_boundaries[
                    work_fn,
                    VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.wo(),
                )

    fn _f_tile_loop_static[
        last_c_tile: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        alias WO = output_shape.get[2]()  # NHWC
        alias F = output_shape.get[3]()  # NHWC
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_shape = get_micro_kernel_shape[
            WO, F, conv_attr, simd_size
        ]()
        alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size

        let f_round_by_simd = (
            (self.partition.f_offset + self.partition.f_size) // simd_size
        ) * simd_size

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            self._h_loop_static[
                micro_kernel_shape[0],
                size // simd_size,
                False,
                last_c_tile,
            ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)

        tile[
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
            f_tile_iteration,
        ](
            self.partition.f_offset,
            f_round_by_simd,
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
        )

        let residual = F - f_round_by_simd
        if (
            self.partition.f_offset + self.partition.f_size == F
            and residual > 0
        ):
            self._h_loop_static[
                micro_kernel_shape[0],
                1,
                True,
                last_c_tile,
            ](n, f_round_by_simd, simd_size, c_tile_offset, c_tile_size)

    @always_inline
    fn _h_loop_static[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        """Loop over H dimension
        Each row is divied into three parts: (1) effected by left padding, (2)
        not effected by padding, (3) effected by right padding. Use pointwise
        micro kernel 1 x micro_kernel_width for (1) and (3) and exploits the
        default micro kernel for (2).
        """
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        alias H = input_shape.get[1]()  # NHWC
        alias W = input_shape.get[2]()  # NHWC
        alias C = input_shape.get[3]()  # NHWC
        alias R = filter_shape.get[1]()  # FRSCf
        alias S = filter_shape.get[2]()  # FRSCf
        alias HO = output_shape.get[1]()  # NHWC
        alias WO = output_shape.get[2]()  # NHWC
        alias F = output_shape.get[3]()  # NHWC

        let filter_base: DTypePointer[filter_type]

        @parameter
        if filter_packed:
            filter_base = self.filter.data.offset(
                f_tile_offset * C * R * S + c_tile_offset * micro_kernel_f_size
            )
        else:
            filter_base = self.filter.data.offset(
                c_tile_offset * F + f_tile_offset
            )

        let input_curr_image = self.input.data.offset(n * W * H * C)
        let output_curr_image = self.output.data.offset(n * WO * HO * F)

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            let h = ho * conv_attr.strides()[0] - conv_attr.pad_bottom()
            # Point to (n, 0, ho, c_tile_offset) mapped in input
            var input_base = input_curr_image.offset(
                c_tile_offset + C * (-conv_attr.pad_left() + W * h)
            )
            # Point to (n, 0, ho, f_tile_offset) mapped in input
            var output_base = output_curr_image.offset(
                f_tile_offset + F * WO * ho
            )

            # The entire row fits in one micro kernel.
            @parameter
            if WO == micro_kernel_height:
                self._inner_loops_static[
                    micro_kernel_height,
                    micro_kernel_width,
                    True,
                    True,
                    has_residual,
                    last_c_tile,
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    0,  # wo
                )
            # The row is split into multiple micro kernels.
            else:
                # micro kernel height for left and right boundaries.
                # IF WO is just 1-2 points more than micro kernel height, the
                # following would divide the row evely by two micro kernels.
                alias micro_kernel_height_lbound = min(
                    micro_kernel_height, WO // 2
                )
                alias micro_kernel_height_rbound = min(
                    micro_kernel_height, WO - WO // 2
                )
                # Left boundary
                self._inner_loops_static[
                    micro_kernel_height_lbound,
                    micro_kernel_width,
                    True,
                    False,
                    has_residual,
                    last_c_tile,
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    0,  # beginning of wo dimension
                )
                input_base = input_base.offset(
                    micro_kernel_height_lbound * conv_attr.strides()[1] * C
                )
                output_base = output_base.offset(micro_kernel_height_lbound * F)

                # Update middle points if any. They aren't effected by padding.
                @always_inline
                @parameter
                fn update_middle[height: Int](wo: Int):
                    self._inner_loops_static[
                        height,
                        micro_kernel_width,
                        False,
                        False,
                        has_residual,
                        last_c_tile,
                    ](
                        input_base,
                        filter_base,
                        output_base,
                        f_tile_offset,
                        f_tile_size,
                        c_tile_offset,
                        c_tile_size,
                        n,
                        ho,
                        wo,
                    )
                    input_base = input_base.offset(
                        height * conv_attr.strides()[1] * C
                    )
                    output_base = output_base.offset(height * F)

                # Middle points are the points not updated by micro kernels
                # on left or right boundary
                alias num_middle_points = WO - micro_kernel_height_lbound - micro_kernel_height_rbound
                # `tile` can't handle zero tile size.
                alias micro_kernel_height_middle = num_middle_points % micro_kernel_height if num_middle_points % micro_kernel_height > 0 else 1
                tile[
                    update_middle,
                    VariadicList[Int](
                        micro_kernel_height, micro_kernel_height_middle
                    ),
                ](micro_kernel_height_lbound, WO - micro_kernel_height_rbound)

                # Right boundary.
                self._inner_loops_static[
                    micro_kernel_height_rbound,
                    micro_kernel_width,
                    False,
                    True,
                    has_residual,
                    last_c_tile,
                ](
                    input_base,
                    filter_base,
                    output_base,
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    WO - micro_kernel_height_rbound,  # offset in wo dimension
                )

    @always_inline
    fn _inner_loops_static[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        padded_left: Bool,
        padded_right: Bool,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        input_base: DTypePointer[
            input_type
        ],  # points to (ho, wo) mapped in input
        filter_base: DTypePointer[filter_type],  # point to filter in cf tile
        output_base: DTypePointer[output_type],  # point to (ho, wo) in output
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        n: Int,  # batch Index
        ho: Int,  # index in output height
        wo: Int,  # index in output width
    ):
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        alias R = filter_shape.get[1]()  # FRSCf
        alias S = filter_shape.get[2]()  # FRSCf
        alias C = input_shape.get[3]()  # NHWC
        alias s_stride_in_input = conv_attr.dilations()[1] * C
        alias wo_stride_in_input = conv_attr.strides()[1] * C
        alias filter_S_stride = C * micro_kernel_f_size
        alias filter_F_stride = R * S * filter_S_stride

        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ].stack_allocation()

        # Initialize micro tile with 0 for its first use
        if self.is_new_c_accum(c_tile_offset):
            self._init_output_micro_tile[
                micro_kernel_height, micro_kernel_width, simd_size
            ](output_micro_tile)
        # Load micro tile from output buffer.
        else:
            self._load_output_micro_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](output_base, output_micro_tile)

        alias W = input_shape.get[2]()  # NHWC
        alias H = input_shape.get[1]()  # NHWC
        alias WO = output_shape.get[2]()  # NHWC
        # Shift in input H when shifting 1 in filter stencil' R dimension.
        var h_shift = 0
        # h index in input image
        let h = ho * conv_attr.strides()[0] - conv_attr.pad_bottom()
        for r in range(R):
            # Skip if row falls in padding.
            if h + h_shift < 0 or h + h_shift >= H:
                h_shift += conv_attr.dilations()[0]
                continue

            var input_ptr = input_base.offset(h_shift * C * W)
            var filter_ptr = filter_base.offset(r * S * filter_S_stride)
            var w = wo * conv_attr.strides()[1] - conv_attr.pad_left()

            @parameter
            @always_inline
            fn body[s: Int]():
                # Adjustment of micro kernel height for left padding
                # The first left_adjust x micro_kernel_width registers are
                # ignored because they fall in padding.
                alias left_adjust = max(
                    div_ceil(
                        conv_attr.pad_left() - s * conv_attr.dilations()[1],
                        conv_attr.strides()[1],
                    ),
                    0,
                ) if padded_left else 0
                # Adjustment of micro kernel height for right padding
                # The last left_adjust x micro_kernel_width registers are ignored.
                # fmt: off
                alias right_adjust = max(
                    WO - 1 - (W - 1 + conv_attr.pad_left() - s * conv_attr.dilations()[1])
                             // conv_attr.strides()[1],
                    0,
                ) if padded_right else 0
                # fmt: on
                self._accumulate[
                    micro_kernel_height - left_adjust - right_adjust,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                    # prefetch offset, default to 4 for now
                    4,
                ](
                    c_tile_size,
                    wo_stride_in_input,
                    input_ptr.offset(left_adjust * wo_stride_in_input),
                    filter_ptr,
                    output_micro_tile.data.offset(
                        left_adjust * micro_kernel_f_size
                    ),
                )

                filter_ptr = filter_ptr.offset(filter_S_stride)
                input_ptr = input_ptr.offset(s_stride_in_input)

            unroll[S, body]()

            h_shift += conv_attr.dilations()[0]

        # Store the micro tile
        self._store_output_micro_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            has_residual,
        ](output_micro_tile, output_base)

        # Apply elmentwise epilogue to the
        alias F = output_shape.get[3]()  # NHWC

        @parameter
        if elementwise_epilogue_enabled and last_c_tile:
            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            let f_tile_size_bounded = F - f_tile_offset if has_residual else f_tile_size
            for wo_idx in range(wo, wo + micro_kernel_height):
                self.elementwise_epilogue_fn(
                    n, ho, wo_idx, f_tile_offset, f_tile_size_bounded
                )

        return


# ===----------------------------------------------------------------------=== #
# Direct Convolution 1D Resigter Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
fn accumulate_wo_tile_1d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
](
    c_tile_size: Int,
    S: Int,
    output: DTypePointer,
    input: DTypePointer,
    input_stride: Int,
    input_stride_to_nbr: Int,
    filter: DTypePointer,
    filter_stride: Int,
    filter_stride_to_nbr: Int,
    partial_load_filter_size: Int,
    w: Int,
    W: Int,
    dilation: Int,
):
    """Update one row in the output for a given (c, f) tile.

    Parameters:
        micro_kernel_height: Number of input points in register tiling.
        micro_kernel_width: Number of SIMD resgiters assigned to F.
        simd_size: Number of elements in a SIMD register.
        partial_load_filter: Whether using partial load for filter.
        effected_by_padding: Whether the tile is effected by padding.

    Args:
        c_tile_size: Tile size in input channel.
        S: Filter window width.
        output: Output registers.
        input: Pointer to the first input point in WO tile.
        input_stride: Stride between two input points, i.e., C w/ NHWC layout.
        input_stride_to_nbr: Stride between an input point and its neighbor.
        filter: Pointer to the first coef in the filter window.
        filter_stride: Stride between two segments of size `micro_kernel_width * simd_size`.
        filter_stride_to_nbr: Stride between between two neighbor coefs, i.e.,
            CF w/ RSCF layout.
        partial_load_filter_size: Size of partial load for filter.
        w: Coordinate in an input row.
        W: Input width.
        dilation: Convolution dilation.
    """

    for s in range(S):
        # Offset in the input row.

        let input_ptr = input + s * input_stride_to_nbr
        let filter_ptr = filter + s * filter_stride_to_nbr

        # When effected by padding, we update 1 output point a time.
        # Skip this point's neighbor if it's in padding.
        @parameter
        if effected_by_padding:
            constrained[
                micro_kernel_height == 1,
                "The tile must only have 1 point when effected bypadding.",
            ]()
            let w_nbr = w + s * dilation
            if w_nbr < 0 or w_nbr >= W:
                continue

        # Accumulat in output registers.
        accumulate[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            prefetch_offset=4,
            partial_load_b=partial_load_filter,
        ](
            c_tile_size,
            output,
            input_ptr,
            input_stride,
            filter_ptr,
            filter_stride,
            partial_load_filter_size,
        )


@always_inline
fn conv1d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    elementwise_epilogue_enabled: Bool,
](
    output: DTypePointer,
    input: DTypePointer,
    filter: DTypePointer,
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape,
    n: Int,
    wo: Int,
    elementwise_epilogue_fn: elementwise_epilogue_type,
):
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride when s increments by 1
    let input_stride_by_s = conv_shape.dilation[0] * conv_shape.c

    # Filter stride when s increments by 1.
    let filter_stride_by_s: Int

    @parameter
    if filter_packed:  # FSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # SCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    # Filter stride in F dimension in FRSCf
    let filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    let w = wo * conv_shape.stride[0] - conv_shape.pad_w[0]

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    let register_tile = NDBuffer[
        2,
        DimList(micro_kernel_height, micro_kernel_width * simd_size),
        output.type,
    ].stack_allocation()

    if first_c_tile:
        init_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ](register_tile.data)
    else:
        load_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load=has_residual,
        ](
            register_tile.data,
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_1d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        conv_shape.s(),
        register_tile.data,
        input,
        conv_shape.c * conv_shape.stride[0],
        input_stride_by_s,
        filter,
        filter_stride,
        filter_stride_by_s,
        conv_shape.f % simd_size,
        w,
        conv_shape.w(),
        conv_shape.dilation[0],
    )

    # Store the micro tile
    store_register_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        partial_store=has_residual,
    ](
        output,
        conv_shape.f,
        register_tile.data,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elementwise epilogue if necessary
    @parameter
    if elementwise_epilogue_enabled and last_c_tile:
        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        let f_tile_size_bounded = conv_shape.f - f_tile_offset if has_residual else f_tile_size
        alias ho = 1
        for wo_idx in range(wo, wo + micro_kernel_height):
            elementwise_epilogue_fn(
                n, ho, wo_idx, f_tile_offset, f_tile_size_bounded
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution 2D Register Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
fn accumulate_wo_tile_2d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
](
    c_tile_size: Int,
    RS: StaticIntTuple[2],
    output: DTypePointer,
    input: DTypePointer,
    input_stride: Int,
    input_stride_to_nbr: StaticIntTuple[2],
    filter: DTypePointer,
    filter_stride: Int,
    filter_stride_to_nbr: StaticIntTuple[2],
    partial_load_filter_size: Int,
    hw: StaticIntTuple[2],
    HW: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
):
    for r in range(RS[0]):
        # Skip the row if it falls into padding.
        let h_nbr = hw[0] + r * dilation[0]
        if h_nbr < 0 or h_nbr >= HW[0]:
            continue

        let input_ptr = input + r * input_stride_to_nbr[0]
        let filter_ptr = filter + r * filter_stride_to_nbr[0]

        accumulate_wo_tile_1d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            RS[1],
            output,
            input_ptr,
            input_stride,
            input_stride_to_nbr[1],
            filter_ptr,
            filter_stride,
            filter_stride_to_nbr[1],
            partial_load_filter_size,
            hw[1],
            HW[1],
            dilation[1],
        )


@always_inline
fn conv2d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    elementwise_epilogue_enabled: Bool,
](
    output: DTypePointer,
    input: DTypePointer,
    filter: DTypePointer,
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[2],
    n: Int,
    howo: StaticIntTuple[2],
    elementwise_epilogue_fn: elementwise_epilogue_type,
):
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (R, S).
    let input_stride_by_s = conv_shape.dilation[1] * conv_shape.c
    let input_stride_by_r = conv_shape.dilation[
        0
    ] * conv_shape.w() * conv_shape.c

    # Filter stride when s increments by 1.
    let filter_stride_by_s: Int

    @parameter
    if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    let filter_stride_by_r = conv_shape.s() * filter_stride_by_s

    # Filter stride in F dimension in FRSCf
    let filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    let hw = Index(
        howo[0] * conv_shape.stride[0] - conv_shape.pad_h[0],
        howo[1] * conv_shape.stride[1] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    let register_tile = NDBuffer[
        2,
        DimList(micro_kernel_height, micro_kernel_width * simd_size),
        output.type,
    ].stack_allocation()

    if first_c_tile:
        init_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ](register_tile.data)
    else:
        load_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load=has_residual,
        ](
            register_tile.data,
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_2d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        Index(conv_shape.r(), conv_shape.s()),
        register_tile.data,
        input,
        conv_shape.c * conv_shape.stride[1],
        Index(input_stride_by_r, input_stride_by_s),
        filter,
        filter_stride,
        Index(filter_stride_by_r, filter_stride_by_s),
        conv_shape.f % simd_size,
        hw,
        Index(conv_shape.h(), conv_shape.w()),
        conv_shape.dilation,
    )

    # Store the micro tile
    store_register_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        partial_store=has_residual,
    ](
        output,
        conv_shape.f,
        register_tile.data,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    @parameter
    if elementwise_epilogue_enabled and last_c_tile:
        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        let f_tile_size_bounded: Int

        @parameter
        if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(howo[1], howo[1] + micro_kernel_height):
            elementwise_epilogue_fn(
                n, howo[0], wo_idx, f_tile_offset, f_tile_size_bounded
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution 3D Resigter Tiling
# ===----------------------------------------------------------------------=== #


# TODO: Simplify this with a rank parameter + recursion.
@always_inline
fn accumulate_wo_tile_3d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load_filter: Bool,
    effected_by_padding: Bool,
](
    c_tile_size: Int,
    QRS: StaticIntTuple[3],
    output: DTypePointer,
    input: DTypePointer,
    input_stride: Int,
    input_stride_to_nbr: StaticIntTuple[3],
    filter: DTypePointer,
    filter_stride: Int,
    filter_stride_to_nbr: StaticIntTuple[3],
    partial_load_filter_size: Int,
    dhw: StaticIntTuple[3],
    DHW: StaticIntTuple[3],
    dilation: StaticIntTuple[3],
):
    for q in range(QRS[0]):
        let d_nbr = dhw[0] + q * dilation[0]
        if d_nbr < 0 or d_nbr >= DHW[0]:
            continue

        let input_ptr = input + q * input_stride_to_nbr[0]
        let filter_ptr = filter + q * filter_stride_to_nbr[0]

        accumulate_wo_tile_2d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            Index(QRS[1], QRS[2]),
            output,
            input_ptr,
            input_stride,
            Index(input_stride_to_nbr[1], input_stride_to_nbr[2]),
            filter_ptr,
            filter_stride,
            Index(filter_stride_to_nbr[1], filter_stride_to_nbr[2]),
            partial_load_filter_size,
            Index(dhw[1], dhw[2]),
            Index(DHW[1], DHW[2]),
            Index(dilation[1], dilation[2]),
        )


@always_inline
fn conv3d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    elementwise_epilogue_enabled: Bool,
](
    output: DTypePointer,
    input: DTypePointer,
    filter: DTypePointer,
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[3],
    n: Int,
    dohowo: StaticIntTuple[3],
    elementwise_epilogue_fn: elementwise_epilogue_type,
):
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (Q, R, S).
    # fmt: off
    let input_stride_by_s = conv_shape.dilation[2] * conv_shape.c
    let input_stride_by_r = conv_shape.dilation[1] * conv_shape.w() * conv_shape.c
    let input_stride_by_q = conv_shape.dilation[0] * conv_shape.w() * conv_shape.h() * conv_shape.c
    # fmt: on

    # Filter stride when s increments by 1.
    let filter_stride_by_s: Int

    @parameter
    if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    let filter_stride_by_r = conv_shape.s() * filter_stride_by_s
    let filter_stride_by_q = conv_shape.r() * filter_stride_by_r

    # Filter stride in F dimension in FRSCf
    let filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    let dhw = Index(
        dohowo[0] * conv_shape.stride[0] - conv_shape.pad_d[0],
        dohowo[1] * conv_shape.stride[1] - conv_shape.pad_h[0],
        dohowo[2] * conv_shape.stride[2] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    let register_tile = NDBuffer[
        2,
        DimList(micro_kernel_height, micro_kernel_width * simd_size),
        output.type,
    ].stack_allocation()

    if first_c_tile:
        init_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ](register_tile.data)
    else:
        load_register_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load=has_residual,
        ](
            register_tile.data,
            output,
            conv_shape.f,
            conv_shape.f_per_group() % simd_size,
        )

    accumulate_wo_tile_3d[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        has_residual and not filter_packed,
        effected_by_padding,
    ](
        c_tile_size,
        conv_shape.filter_dims,
        register_tile.data,
        input,
        conv_shape.c * conv_shape.stride[2],
        Index(input_stride_by_q, input_stride_by_r, input_stride_by_s),
        filter,
        filter_stride,
        Index(filter_stride_by_q, filter_stride_by_r, filter_stride_by_s),
        conv_shape.f % simd_size,
        dhw,
        conv_shape.input_dims,
        conv_shape.dilation,
    )

    # Store the micro tile
    store_register_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        partial_store=has_residual,
    ](
        output,
        conv_shape.f,
        register_tile.data,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    @parameter
    if elementwise_epilogue_enabled and last_c_tile:
        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        let f_tile_size_bounded: Int

        @parameter
        if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(dohowo[2], dohowo[2] + micro_kernel_height):
            elementwise_epilogue_fn(
                n, dohowo[1], wo_idx, f_tile_offset, f_tile_size_bounded
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution Filter Packing                                            #
# ===----------------------------------------------------------------------=== #


@always_inline
fn pack_filter_shape_impl[
    filter_type: DType
](R: Int, S: Int, C: Int, F: Int, num_groups: Int) -> StaticIntTuple[5]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Args:
        R: Original R filter dimension.
        S: Original S filter dimension.
        C: Original C filter dimension.
        F: Original F filter dimension.
        num_groups: Number of groups in the convolution.

    Returns:
        The output shape.
    """
    alias simd_size = simdwidthof[filter_type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    let F_per_group = F // num_groups

    var output_shape = StaticIntTuple[5]()
    output_shape[0] = num_groups * div_ceil(F_per_group, micro_kernel_f_size)
    output_shape[1] = R
    output_shape[2] = S
    output_shape[3] = C
    output_shape[4] = micro_kernel_f_size

    return output_shape


@always_inline
fn pack_conv_filter_shape[
    single_thread_blocking_override: Bool,
](filter: NDBuffer, num_groups: Int) -> StaticIntTuple[filter.rank + 1]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        filter: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """

    alias simd_size = simdwidthof[filter.type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Filter is in RSCF layout. The last dim is F no matter it's 1d, 2d, or 3d.
    let F = filter.dim[filter.rank - 1]()

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    let F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = StaticIntTuple[filter.rank + 1]()
    packed_shape[0] = num_groups * div_ceil(F_per_group, micro_kernel_f_size)
    packed_shape[filter.rank] = micro_kernel_f_size

    @always_inline
    @parameter
    fn assign[i: Int]():
        packed_shape[i + 1] = filter.dim[i]()

    unroll[filter.rank - 1, assign]()

    return packed_shape


@always_inline
fn _get_group_filter_base(
    packed_filter: NDBuffer, group_idx: Int, f_per_group: Int
) -> DTypePointer[packed_filter.type, packed_filter.address_space]:
    """Returns the pointer of the input group's start in the packed filter."""
    # Each group is zero padded to
    #     div_ceil(F_per_group, micro_kernel_width)
    #   * filter_window_size
    #   * C
    #   * micro_kernel_f_width
    # Output pointer points to the start of the current group.

    let micro_kernel_f_size = packed_filter.dim[packed_filter.rank - 1]()
    alias rank = packed_filter.rank

    var filter_window_size = 1

    # The packed filter has layout e.x. FRSCf. The [1, rank-2) dims are filter
    # window sizes.
    @parameter
    @always_inline
    fn multiply[i: Int]():
        filter_window_size *= packed_filter.dim[i + 1]()

    unroll[rank - 3, multiply]()

    # Size of one group's packed filter.
    # fmt: off
    let group_size = div_ceil(f_per_group , micro_kernel_f_size) \
                   * filter_window_size * packed_filter.dim[rank-2]() \
                   * micro_kernel_f_size
    # fmt: on

    return packed_filter.data + group_idx * group_size


@always_inline
fn pack_filter(
    filter: NDBuffer,
    packed_filter: NDBuffer,
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.
    Use the default micro kernel size for dynamic shapes."""

    constrained[
        filter.type == packed_filter.type,
        "Type mismatch between the filter and the packed filter.",
    ]()

    alias simd_size = simdwidthof[filter.type]()
    alias f_size_default = get_direct_conv_micro_kernel_width() * simd_size

    @parameter
    if packed_filter.shape.at[packed_filter.rank - 1]().has_value():
        alias f_size = packed_filter.shape.get[packed_filter.rank - 1]()
        pack_filter[simd_size, f_size](filter, packed_filter, num_groups)
    else:
        pack_filter[simd_size, f_size_default](
            filter, packed_filter, num_groups
        )


@always_inline
fn pack_filter[
    simd_size: Int,
    micro_kernel_f_size: Int,
](filter: NDBuffer, packed_filter: NDBuffer, num_groups: Int):
    """This packs the filter form RSCF to FRSCf.

    Parameters:
        simd_size: Can differ from the simd size of the input type.
        micro_kernel_f_size: The size of the last dimension in FRSCf, which is
            equals the size of the micro kernel's F dimension.

    Args:
        filter: Filter in RSCF layout (if 2D).
        packed_filter: Packed filter in FRSCf layout (if 2D).
            F       - the index of continuous segments in micro kernel.
            R, S, C - original R, S, C.
            f       - the index within a continuous segments.
        num_groups: The number of groups in the convolution.

    F is first broken down to segements of size micro_kernel_f_size, then the
    remainder is further divided by simd_size. The last residual elements if
    any is padded with zero to fill simd_size.
    """

    # The micro kernel should be multiple of simd_size in F dimension.
    constrained[micro_kernel_f_size % simd_size == 0]()

    # The input simd size should not exceed filter type's simd size.
    # E.x. we can pack int8 filter based on int32 simd size.
    constrained[simd_size <= simdwidthof[filter.type]()]()

    # Product of filter dims upto (rank - 1).
    var outer_dims_prod = 1

    @always_inline
    @parameter
    fn multiply[i: Int]():
        outer_dims_prod *= filter.dim[i]()

    unroll[filter.rank - 1, multiply]()

    let F = filter.dim[filter.rank - 1]()
    let F_per_group = F // num_groups

    packed_filter.zero()

    # Each group is zero padded to
    #
    #                   div_ceil(F_per_group, micro_kernel_f_size)
    #                 * outer_dims_prod
    #                 * micro_kernel_f_size.
    #
    # There can be a remainder: F_per_group % micro_kernel_f_size. That's further
    # tiled by simd_size. The elements beyond the remainder is set to 0. E.x.
    # micro_kernel_f_size = 8, simd_size = 2, 21 values in total, follows
    #
    #                       |--------|--------|--|--|-0|00|

    for g in range(num_groups):
        let group_start = _get_group_filter_base(packed_filter, g, F_per_group)

        @always_inline
        @parameter
        fn pack[f_tile_size: Int](f_tile_start: Int):
            var packed_filter_ptr = group_start + f_tile_start * outer_dims_prod

            for row in range(outer_dims_prod):
                let filter_ptr = filter.data + row * F + g * F_per_group + f_tile_start

                @unroll
                for i in range(f_tile_size // simd_size):
                    packed_filter_ptr.simd_store(
                        i * simd_size,
                        filter_ptr.simd_load[simd_size](i * simd_size).cast[
                            packed_filter.type
                        ](),
                    )

                packed_filter_ptr += f_tile_size

        # If F % simd_size != 0, the following won't touch the remainder.
        tile[pack, VariadicList[Int](micro_kernel_f_size, simd_size)](
            0, F_per_group
        )

    # Check the remainder if any
    let F_round_by_simd = align_down(F_per_group, simd_size)
    let residual = F_per_group - F_round_by_simd

    # Handle the remainder if any
    if residual > 0:
        for g in range(num_groups):
            let group_start = _get_group_filter_base(
                packed_filter, g, F_per_group
            )
            var packed_filter_ptr = group_start + F_round_by_simd * outer_dims_prod

            for row in range(outer_dims_prod):
                let filter_ptr = filter.data + row * F + g * F_per_group + F_round_by_simd

                # Load remainder elements and pad with zero to
                # to fill a simd vector.
                let filter_vec = partial_simd_load[simd_size](
                    filter_ptr, 0, residual, 0.0
                ).cast[packed_filter.type]()
                packed_filter_ptr.simd_store(filter_vec)

                # Hence, packed filter is incremented by simd_size
                packed_filter_ptr = packed_filter_ptr + simd_size


@always_inline
fn conv_shape[
    input_rank: Int,
    filter_rank: Int,
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    num_groups_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    filter_buf: NDBuffer[
        filter_rank, DimList.create_unknown[filter_rank](), filter_type
    ],
    strides_buf: NDBuffer[1, DimList.create_unknown[1](), strides_type],
    dilations_buf: NDBuffer[1, DimList.create_unknown[1](), dilations_type],
    paddings_buf: NDBuffer[1, DimList.create_unknown[1](), paddings_type],
    num_groups_buf: NDBuffer[1, DimList.create_unknown[1](), num_groups_type],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `conv` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        filter_rank: Rank of the filter tensor.
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.
        num_groups_type: Type of the num_groups tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter tensor.
        strides_buf: The strides tensor.
        dilations_buf: The dilations tensor.
        paddings_buf: The paddings tensor.
        num_groups_buf: The num_groups tensor.

    Returns:
        The output shape.
    """

    if input_rank != 4:
        raise Error("input rank must be 4")
    if input_rank != filter_rank:
        raise Error("input rank must match filter rank")
    if (
        strides_buf.dim(0) != input_rank - 2
        or dilations_buf.dim(0) != input_rank - 2
    ):
        raise Error("strides and dilations size must be input rank - 2")
    if paddings_buf.dim(0) != 2 * (input_rank - 2):
        raise Error("paddings size must be 2 * (input rank - 2)")

    # Assume input has layout NHWC
    let batch_size = input_buf.dim(0)
    let input_channels = input_buf.dim(3)
    # Assume filter has layout RSCF
    let filter_channels = filter_buf.dim(2)
    let num_groups = int(num_groups_buf[0])
    let output_channels = filter_buf.dim(3)

    if input_channels != (num_groups * filter_channels):
        raise Error(
            "input channels and groups times filter channels must match"
        )
    if (output_channels % num_groups) != 0:
        raise Error("output_channels must be divisible by the number of groups")

    # compute and return the output shape
    let output_height = get_sliding_window_out_dim(
        input_buf.dim(1),
        filter_buf.dim(0),
        int(dilations_buf[0]),
        int(strides_buf[0]),
        int(paddings_buf[0] + paddings_buf[1]),
    )
    let output_width = get_sliding_window_out_dim(
        input_buf.dim(2),
        filter_buf.dim(1),
        int(dilations_buf[1]),
        int(strides_buf[1]),
        int(paddings_buf[2] + paddings_buf[3]),
    )

    if output_height <= 0:
        raise Error("Convolution output height must be positive")
    if output_width <= 0:
        raise Error("Convolution output width must be positive")

    var output_shape = StaticIntTuple[input_rank](
        batch_size, output_height, output_width, output_channels
    )

    return output_shape


alias elementwise_lambda_fn_sig_type = fn[type: DType, width: Int] (
    StaticIntTuple[4], SIMD[type, width]
) capturing -> None


fn conv_2d_nhwc_direct[
    filter_rank: Int,
    filter_packed: Bool,
    conv_info_static: ConvInfoStatic,
    lambdas_have_fusion: Bool,
    epilogue_wrapper: elementwise_lambda_fn_sig_type,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
](
    input: NDBuffer[4, input_shape, input_type],
    filter: NDBuffer[filter_rank, filter_shape, filter_type],
    output: NDBuffer[4, output_shape, output_type],
    conv_info: ConvInfo[conv_info_static],
) raises:
    constrained[
        input_type == filter_type and input_type == output_type,
        "conv input/output/filter types must be the same",
    ]()
    constrained[
        (filter_packed and filter_rank == 5)
        or (not filter_packed and filter_rank == 4),
        "unexpected filter rank for filter layout",
    ]()

    let output_rebind = rebind[NDBuffer[4, output_shape, input_type]](output)
    let filter_rebind = rebind[NDBuffer[filter_rank, filter_shape, input_type]](
        filter
    )

    alias filter_layout = Image2DLayout.FRSCf if filter_packed else Image2DLayout.RSCF
    let conv_shape = get_conv2d_shape[
        filter_rank,
        output_shape,
        input_shape,
        filter_shape,
        input_type,
        Image2DLayout.NHWC,
        filter_layout,
    ](
        output_rebind,
        input,
        filter_rebind,
        conv_info.pad_h.get(),
        conv_info.pad_w.get(),
        conv_info.stride.get(),
        conv_info.dilation.get(),
        conv_info.num_groups.get(),
    )

    # The closure updates a row segment of the output.
    fn elementwise_epilogue_closure(
        n: Int,
        ho: Int,
        wo: Int,
        f_offset: Int,
        f_size: Int,
    ):
        alias simd_size = simdwidthof[output_type]()

        @always_inline
        @parameter
        fn body[width: Int](idx: Int):
            let coords = Index(n, ho, wo, f_offset + idx)
            let vec = output.simd_load[width](coords)
            epilogue_wrapper[output_type, width](coords, vec)

        vectorize[simd_size, body](f_size)

    ConvDirectNHWC[
        4,  # input_rank
        filter_rank,
        4,  # output_rank
        input_shape,
        filter_shape,
        output_shape,
        input_type,
        filter_type,
        output_type,
        filter_packed,
        conv_info_static,
        lambdas_have_fusion,
    ].run(
        output,
        input,
        filter,
        conv_shape,
        elementwise_epilogue_closure,
    )
