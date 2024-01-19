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
from AccumulateSIMD import accumulate_x86_simd, accumulate_neon
from ConvUtils import (
    ConvInfo,
    ConvInfoStatic,
    ConvShape,
    get_conv2d_shape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)
from Image import Image2DLayout, ImageData, ImageShape
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
)
from memory.unsafe import DTypePointer
from ShapeFuncUtils import get_sliding_window_out_dim

from utils.index import Index, StaticIntTuple
from utils.list import Dim, DimList
from runtime.llcl import Runtime

alias MAX_NUM_CHANNELS_TILE = 384


@value
struct Naive2dConvolution[
    static_output_shape: DimList,
    static_filter_shape: DimList,
    static_input_shape: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    static_data_layout: Image2DLayout,
    static_filter_layout: Image2DLayout,
]:
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: ImageData[static_output_shape, output_type, static_data_layout]
    var input: ImageData[static_input_shape, input_type, static_data_layout]
    var filter: ImageData[
        static_filter_shape, filter_type, static_filter_layout
    ]
    var pad_h: StaticIntTuple[2]
    var pad_w: StaticIntTuple[2]
    var stride: StaticIntTuple[2]
    var dilation: StaticIntTuple[2]
    var num_groups: Int

    # Derived params.
    var output_shape: ImageShape
    var input_shape: ImageShape
    var filter_shape: ImageShape

    @staticmethod
    fn run(
        output: ImageData[static_output_shape, output_type, static_data_layout],
        input: ImageData[static_input_shape, input_type, static_data_layout],
        filter: ImageData[
            static_filter_shape, filter_type, static_filter_layout
        ],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        num_groups: Int,
    ):
        """Interface function to run a convolution op on the given input and
        filter tensor and stores the result in the given output tensor.

        Args:
            output: Pre-allocated output tensor space.
            input: Batched image input to the conv2d operator.
            filter: Filters to apply in the conv2d operator.
            pad_h: Padding on the height dimension with
                assumed tuple def (PadOnLowerIdx, PadOnHigherIdx).
            pad_w: Padding on the width dimension with
                assumed tuple def (PadOnLowerIdx, PadOnHigherIdx).
            stride: Strides on height and width dimensions
                with assumed tuple def (StrideH, StrideW).
            dilation: Dilations on height and width
                dimensions with assumed tuple def (dilation_h, dilation_w).
            num_groups: The number of groups in the convolution.
        """
        # Create an instance of the convolution op.
        let naive2d_convolution = Naive2dConvolution[
            static_output_shape,
            static_filter_shape,
            static_input_shape,
            input_type,
            filter_type,
            output_type,
            static_data_layout,
            static_filter_layout,
        ](output, input, filter, pad_h, pad_w, stride, dilation, num_groups)

        # Run the actual loops and computations.
        naive2d_convolution._outer_loop()

    fn __init__(
        inout self,
        output: ImageData[static_output_shape, output_type, static_data_layout],
        input: ImageData[static_input_shape, input_type, static_data_layout],
        filter: ImageData[
            static_filter_shape, filter_type, static_filter_layout
        ],
        pad_h: StaticIntTuple[2],
        pad_w: StaticIntTuple[2],
        stride: StaticIntTuple[2],
        dilation: StaticIntTuple[2],
        num_groups: Int,
    ):
        """Constructor of a convolution op instance on the given input and
        filter tensor and stores the result in the give output tensor.

        Args:
            output: Pre-allocated output tensor space.
            input: Batched image input to the conv2d operator.
            filter: Filters to apply in the conv2d operator.
            pad_h: Padding on the height dimension with assu-
                med tuple def (PadOnLowerIdx, PadOnHigherIdx).
            pad_w: Padding on the width dimension with assum-
                ed tuple def (PadOnLowerIdx, PadOnHigherIdx).
            stride: Strides on height and width dimensions
                with assumed tuple def (StrideH, StrideW).
            dilation: Dilations on height and width dimensi-
                ons with assumed tuple def (dilation_h, dilation_w).
            num_groups: The number of groups in the convolution.
        """
        # Register input/output buffers and parameters.
        self.output = output
        self.input = input
        self.filter = filter
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride = stride
        self.dilation = dilation
        self.num_groups = num_groups

        # Derive layout agnostic shape information.
        self.output_shape = ImageShape.__init__[
            static_output_shape, output_type, static_data_layout
        ](output)
        self.input_shape = ImageShape.__init__[
            static_input_shape, input_type, static_data_layout
        ](input)
        self.filter_shape = ImageShape.__init__[
            static_filter_shape, filter_type, static_filter_layout
        ](filter)

    fn _outer_loop(self):
        """Implementation of the outermost loop of a convolution operator with
        loops covering the iteration space of batch, filter count, height and wi-
        dth dimensions.
        """
        # Iterate on output batch dimension.
        for no_idx in range(self.output_shape.N):
            # Iterate on filter dimension.
            for f_idx in range(self.output_shape.C):
                # Iterate on output H dimension.
                for ho_idx in range(self.output_shape.H):
                    # Iterate on output W dimension.
                    for wo_idx in range(self.output_shape.W):
                        # Compute the result value at this specific output posit-
                        #  ion.
                        self._compute_poInt__(
                            StaticIntTuple[4](no_idx, f_idx, ho_idx, wo_idx)
                        )

    fn _compute_poInt__(
        self,
        # Output index [N,C,H,W]
        output_idx: StaticIntTuple[4],
    ):
        """Implementation of the inner loop computation of a conv2d operator
        producing a single scalar value at the given output tensor index.
            Args:
                output_index(StaticIntTuple): Index vector specifying which
            value of the output tensor to produce.
        """
        # Initialize the result of this point.
        var value: SIMD[output_type, 1] = 0

        # Extract the H and W size of the input image.
        let image_bound = StaticIntTuple[2](
            self.input_shape.H, self.input_shape.W
        )
        let C_per_group = self.input_shape.C // self.num_groups
        let F_per_group = self.output_shape.C // self.num_groups
        let group_idx = output_idx[1] // F_per_group

        # Iterate on filter height dimension.
        for r_idx in range(self.filter_shape.H):
            # Iterate on filter width dimension.
            for s_idx in range(self.filter_shape.W):
                # Compute input access index, on the H and W dimension.
                let input_image_index = (
                    # Output HxW with striding.
                    (
                        StaticIntTuple[2](
                            output_idx[2],
                            output_idx[3],
                        )
                        * self.stride
                    )
                    +
                    # Filter RxS with dilation.
                    (Index(r_idx, s_idx) * self.dilation)
                    -
                    # Padding offset, using the left padding only here.
                    Index(self.pad_h[0], self.pad_w[0])
                )

                if (
                    # Check that the current image index is within valid range
                    #  on the input image data tensor.
                    Index(0, 0) <= input_image_index
                    and input_image_index < image_bound
                ):
                    # Iterate on channels dimension.
                    for c_idx in range(
                        C_per_group * group_idx, C_per_group * (group_idx + 1)
                    ):
                        # Accumulate product of input data filter data.
                        let input_val = self.input[
                            output_idx[0],  # N
                            c_idx,
                            input_image_index[0],  # H
                            input_image_index[1],  # W
                        ]
                        let filter_val = self.filter[
                            output_idx[1],
                            c_idx % C_per_group,
                            r_idx,
                            s_idx,  # F  # C  # R  # S
                        ]
                        value += (
                            input_val.cast[output_type]()
                            * filter_val.cast[output_type]()
                        )

        # Store the computed output at the given output position..
        self.output[
            output_idx[0],
            output_idx[1],
            output_idx[2],
            output_idx[3],
        ] = value


# ===----------------------------------------------------------------------=== #
# Im2Col convolution:
# ===----------------------------------------------------------------------=== #


@always_inline
fn _m_to_n_ho_wo_nhwc(m: Int, conv_shape: ConvShape) -> StaticIntTuple[3]:
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
    let n = m // (conv_shape.out_h * conv_shape.out_w)
    let ho = (m % (conv_shape.out_h * conv_shape.out_w)) // conv_shape.out_w
    let wo = m % conv_shape.out_w
    return Index(n, ho, wo)


@always_inline
fn _k_to_r_s_c_nhwc(k: Int, conv_shape: ConvShape) -> StaticIntTuple[3]:
    """Converts post-im2col k dimension index to pre-im2col coordinates on
    (R, S, C) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            The translated 3d indices in (R, S, C) format.
    TODO(Fixel): This utility should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    let r = k // (conv_shape.s * conv_shape.c)
    let s = (k // conv_shape.c) % conv_shape.s
    let c = k % conv_shape.c
    return Index(r, s, c)


@always_inline
fn _ho_wo_to_hi_wi(
    ho_wo: StaticIntTuple[2],
    r_s: StaticIntTuple[2],
    conv_shape: ConvShape,
) -> StaticIntTuple[2]:
    """Converts index on the output images to index on the input images.
        Args:
            ho_wo (StaticIntTuple): Index on output image in (Hout, Wout).
            r_s (StaticIntTuple): Index on filter dimensions in (R, S).
            conv_shape (ConvShape): convolution dimension description.

        Returns (StaticIntTuple):
            Index on input image in (H, W).

        Returns (StaticIntTuple):
            The translated 3d indices in (R, S, C) format.
    TODO(Fixel): This utility should be generalized into a conv util
    class with some additional layout agnostic logic.
    """
    let pad_left = Index(
        conv_shape.pad_h[0],
        conv_shape.pad_w[0],
    )
    return ho_wo * conv_shape.stride - pad_left + r_s * conv_shape.dilation


# ===----------------------------------------------------------------------=== #
# Direct Convolution                                                           #
# ===----------------------------------------------------------------------=== #

alias elementwise_epilogue_type = fn (Int, Int, Int, Int, Int) escaping -> None


@value
struct ConvDirectNHWC[
    filter_rank: Int,
    shape_input: DimList,
    shape_filter: DimList,
    shape_output: DimList,
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

    var output: NDBuffer[4, shape_output, output_type]
    var input: NDBuffer[4, shape_input, input_type]
    var filter: NDBuffer[filter_rank, shape_filter, filter_type]

    var conv_shape: ConvShape

    # If n, ho, wo dimensions are merged (for no padding), the first three
    # dimensions are the offsets and sizes in (n_ho_wo, c, f) iteration space.
    # If they are not merged (for non-zero padding), the following denotes
    # (n, c, f, ho). Prioritize partitioning batch size (n).
    var partition_offsets: StaticIntTuple[4]
    var partition_sizes: StaticIntTuple[4]

    var cf_tile_size: StaticIntTuple[2]

    var elementwise_epilogue_fn: elementwise_epilogue_type

    # If shapes and attributes are known at compile time
    alias fully_static = conv_attr.all_known() and shape_input.all_known[
        4
    ]() and shape_output.all_known[4]() and shape_filter.all_known[
        filter_rank
    ]()

    @staticmethod
    fn run(
        output: NDBuffer[4, shape_output, output_type],
        input: NDBuffer[4, shape_input, input_type],
        filter: NDBuffer[filter_rank, shape_filter, filter_type],
        conv_shape: ConvShape,
    ) raises:
        fn direct_null_elementwise_epilogue(
            n: Int, ho: Int, wo: Int, f_offset: Int, f_size: Int
        ) escaping:
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
        output: NDBuffer[4, shape_output, output_type],
        input: NDBuffer[4, shape_input, input_type],
        filter: NDBuffer[filter_rank, shape_filter, filter_type],
        conv_shape: ConvShape,
        elementwise_epilogue_fn: elementwise_epilogue_type,
    ) raises:
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_shape = get_micro_kernel_shape[
            shape_output.at[2](),
            shape_output.at[3](),
            conv_attr,
            simd_size,
        ]()
        alias micro_kernel_height = micro_kernel_shape[0]
        alias micro_kernel_width = micro_kernel_shape[1]
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        let cf_tile_size = get_conv_tile_shape[filter_type, micro_kernel_width](
            conv_shape
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

        # Number of partitions in n_ho_wo, c, f dimensions.
        let num_threads = Runtime().parallelism_level()
        let max_num_tasks = get_conv_num_tasks(num_threads, conv_shape)
        let num_partitions = get_conv_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](max_num_tasks, conv_shape)
        let num_tasks = num_partitions[0] * num_partitions[1] * num_partitions[
            2
        ] * num_partitions[3]

        if num_partitions[1] > 1 and conv_shape.num_groups > 1:
            trap("can't partition on C when num_groups > 1")

        if num_partitions[2] > 1 and conv_shape.num_groups > 1:
            trap("can't partition on F when num_groups > 1")

        # Wrap the pointer inside NDBuffer so it can be properly captured by async closure.
        var output_ptr = output.data
        let output_size = conv_shape.n * conv_shape.out_h * conv_shape.out_w * conv_shape.f
        let scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = DTypePointer[output_type].alloc(scratch_size)
        let output_scratch = Buffer[Dim(), output_type](
            output_ptr, scratch_size
        )

        @parameter
        @always_inline
        fn task_func(task_id: Int):
            let task_id_f = task_id % num_partitions[2]
            var quotient = task_id // num_partitions[2]
            let task_id_c = quotient % num_partitions[1]
            quotient = quotient // num_partitions[1]
            let task_id_howo = quotient % num_partitions[3]
            let task_id_n = quotient // num_partitions[3]

            let n_range = partition_work(
                task_id_n, num_partitions[0], conv_shape.n, 1
            )

            let c_range = partition_work(
                task_id_c, num_partitions[1], conv_shape.c, 1
            )
            let f_range = partition_work(
                task_id_f,
                num_partitions[2],
                conv_shape.f,
                micro_kernel_f_size,
            )

            let has_padding = conv_shape.pad_h != Index(
                0, 0
            ) or conv_shape.pad_w != Index(0, 0)

            # Merge wo and ho loops when there is no padding.
            # Otherwise the partition granularity is a row.
            let work_unit = 1 if has_padding else micro_kernel_height
            let work_load = conv_shape.out_h if has_padding else conv_shape.out_h * conv_shape.out_w
            let howo_range = partition_work(
                task_id_howo, num_partitions[3], work_load, work_unit
            )

            # Short circuit when a task gets no work. This could happen when
            # the previous tasks get more work due to alignment requirement.
            if (
                n_range[1] <= 0
                or c_range[1] <= 0
                or f_range[1] <= 0
                or howo_range[1] <= 0
            ):
                return

            let task_tile_size = Index(
                min(cf_tile_size[0], c_range[1]), cf_tile_size[1]
            )

            let task_output = NDBuffer[4, shape_output, output_type](
                output_scratch.data.offset(task_id_c * output_size),
                Index(
                    conv_shape.n,
                    conv_shape.out_h,
                    conv_shape.out_w,
                    conv_shape.f,
                ),
            )

            let instance = ConvDirectNHWC[
                filter_rank,
                shape_input,
                shape_filter,
                shape_output,
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
                Index(n_range[0], c_range[0], f_range[0], howo_range[0]),
                Index(n_range[1], c_range[1], f_range[1], howo_range[1]),
                task_tile_size,
                elementwise_epilogue_fn,
            )
            instance._n_loop()

        if num_partitions[1] > 1:
            sync_parallelize[task_func](num_tasks)

            # Reduce from the output scratch buffer to the actual output.
            @parameter
            @always_inline
            fn reduce_task(tid: Int):
                # Use all threads in reduction.
                let nhowo = conv_shape.n * conv_shape.out_w * conv_shape.out_h
                let reduce_range = partition_work(tid, num_threads, nhowo, 1)

                @parameter
                @always_inline
                fn sum[width: Int](offset: Int):
                    let tid_output_offset = reduce_range[
                        0
                    ] * conv_shape.f + offset
                    var vec = output_scratch.data.offset(
                        tid_output_offset
                    ).simd_load[width]()
                    # The number of partitions here is typically small.
                    # There may not be much benefit from unrolling the reduction axis.
                    # Only unroll the last dimension.
                    for i in range(1, num_partitions[1]):
                        vec += output_scratch.data.offset(
                            tid_output_offset + i * output_size
                        ).simd_load[width]()
                    output.data.offset(tid_output_offset).simd_store[width](vec)

                vectorize_unroll[simd_size, 4, sum](
                    reduce_range[1] * conv_shape.f
                )

                @parameter
                if elementwise_epilogue_enabled:
                    for m in range(
                        reduce_range[0], reduce_range[0] + reduce_range[1]
                    ):
                        let nhowo = _m_to_n_ho_wo_nhwc(m, conv_shape)
                        elementwise_epilogue_fn(
                            nhowo[0], nhowo[1], nhowo[2], 0, conv_shape.f
                        )

            # NOTE: synchronous, so use of locally allocated output_ptr is safe.
            sync_parallelize[reduce_task](num_threads)
            output_ptr.free()
        else:
            # Use sync to work around #12624
            sync_parallelize[task_func](num_tasks)

    fn _n_loop(self):
        """Loop over the batch size.
        This is the outermost loop and is used with padding."""

        @always_inline
        @parameter
        fn body[has_padding: Bool]():
            for n in range(
                self.partition_offsets[0],
                self.partition_offsets[0] + self.partition_sizes[0],
            ):
                self._c_tile_loop[has_padding](n, self.cf_tile_size[0])

        unswitch[body](
            self.conv_shape.pad_h != Index(0, 0)
            or self.conv_shape.pad_w != Index(0, 0)
        )

    fn _c_tile_loop[padded: Bool](self, n: Int, tile_size: Int):
        """Loop over C tiles."""

        @always_inline
        @parameter
        fn c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
            # Only apply static shape optimizations to shapes with padding since
            # there is a fast path for pointwise (no padding) conv with strides.
            # Grouped conv logic has not been plumbed into static specialized funcs yet.
            @parameter
            if self.fully_static and padded and conv_attr.num_groups == Dim(1):
                self._f_tile_loop_static[False](n, c_tile_offset, c_tile_size)
            else:
                self._f_tile_loop[padded, False](n, c_tile_offset, c_tile_size)

        # Can't fuse epilogue inside conv if C is partitioned
        if self.partition_sizes[1] < self.conv_shape.c:
            tile[c_tile_iteration](
                self.partition_offsets[1],
                self.partition_offsets[1] + self.partition_sizes[1],
                tile_size,
            )
        # C is not partitioned, fuse epilogue in the last C tile.
        else:
            for g in range(self.conv_shape.num_groups):
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
                if self.fully_static and padded:
                    self._f_tile_loop_static[True](
                        n,
                        c_start + c_round_by_tile,
                        c_round_by_tile_residual,
                    )
                else:
                    self._f_tile_loop[padded, True](
                        n,
                        c_start + c_round_by_tile,
                        c_round_by_tile_residual,
                    )

    fn _f_tile_loop[
        padded: Bool, last_c_tile: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        alias micro_kernel_width = get_direct_conv_micro_kernel_width()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            @parameter
            if padded:
                self._h_loop[
                    micro_kernel_height, size // simd_size, False, last_c_tile
                ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)
            else:
                self._ho_wo_tile_loop[size, False, last_c_tile](
                    n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

        let group_idx = self.conv_shape.c_to_group(c_tile_offset)
        let f_per_group = self.conv_shape.f_per_group()

        # When num_groups > 1 we can assume that F has not been partitioned between
        # threads.
        let f_group_offset = group_idx * f_per_group if self.conv_shape.num_groups > 1 else self.partition_offsets[
            2
        ]
        let f_group_end_align_simd = (
            f_group_offset + align_down(f_per_group, simd_size)
        ) if self.conv_shape.num_groups > 1 else align_down(
            self.partition_offsets[2] + self.partition_sizes[2], simd_size
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
            f_group_offset,
            f_group_end_align_simd,
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
        )

        # If this is the last partition in F and it's not a multiple of simd_size.
        # The partition is aligned by micro_kernel_f_size, so only the last
        # partition is possible to have residual.
        let residual = align_down_residual(f_per_group, simd_size)
        if (
            self.partition_offsets[2] + self.partition_sizes[2]
            == self.conv_shape.f  # always true for grouped conv since no partitioning in F
            and residual > 0
        ):

            @parameter
            if padded:
                self._h_loop[micro_kernel_height, 1, True, last_c_tile](
                    n,
                    f_group_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )
            else:
                self._ho_wo_tile_loop[simd_size, True, last_c_tile](
                    n,
                    f_group_end_align_simd,
                    simd_size,
                    c_tile_offset,
                    c_tile_size,
                )

    fn _ho_wo_tile_loop[
        micro_kernel_f_size: Int, has_residual: Bool, last_c_tile: Bool
    ](
        self,
        n: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
    ):
        """The N, HO, WO dimensions are fused and traversed with the micro
        kernel height as the step.
        Note that the micro kernel height changes for residual blocks."""
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias micro_kernel_width = micro_kernel_f_size // simd_size

        @always_inline
        @parameter
        fn ho_wo_iteration[ho_wo_tile_size: Int](ho_wo: Int):
            @always_inline
            @parameter
            fn body[c_fully_cached: Bool]():
                self._inner_loops[
                    ho_wo_tile_size,  # micro kernel height
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
                    ho_wo,
                )

            # c_fully_cached means the C dimension is fully covered in the
            # cache tile.
            unswitch[body](self.conv_shape.c == c_tile_size)

        # After the loop can't be stepped with micro_kernel_height,
        # it will step by 5, 4, 3, 2, 1. This works with micro_kernel_height > 6
        # but maybe not very efficient.
        tile[
            ho_wo_iteration,
            VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),
        ](
            self.partition_offsets[3],
            self.partition_offsets[3] + self.partition_sizes[3],
        )

    @always_inline
    fn is_new_c_accum(self, c_idx: Int) -> Bool:
        # returns true when processing first C in a group or first C in a C partition
        if self.conv_shape.num_groups > 1:
            return self.conv_shape.c_in_group(c_idx) == 0
        return c_idx == self.partition_offsets[1]

    fn _inner_loops[
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
        ho_wo: Int,
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

        # Set input base offsets, corresponding to r=s=0
        @always_inline
        @parameter
        fn set_input_base_offsets[idx: Int]():
            # Global wo, ho index.
            let ho = (ho_wo + idx) // self.conv_shape.out_w
            let wo = (ho_wo + idx) % self.conv_shape.out_w
            # Translate ho, wo to hi, wi/range
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
            let w = wo * self.conv_shape.stride[1] - self.conv_shape.pad_w[0]
            input_base_offsets.simd_store[1](
                idx,
                c_tile_offset
                + self.conv_shape.c
                * (w + self.conv_shape.w * (h + self.conv_shape.h * n)),
            )

        unroll[micro_kernel_height, set_input_base_offsets]()

        alias alignment = alignof[SIMD[output_type, simd_size]]()
        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ].aligned_stack_allocation[alignment]()

        let n_ho_wo = n * self.conv_shape.out_h * self.conv_shape.out_w + ho_wo
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
            ](
                self.output.data.offset(
                    n_ho_wo * self.conv_shape.f + f_tile_offset
                ),
                output_micro_tile,
            )

        var filter_ptr: DTypePointer[filter_type] = self.filter.data

        @parameter
        if filter_packed:
            filter_ptr = _get_group_filter_base(
                self.filter,
                self.conv_shape.c_to_group(c_tile_offset),
                self.conv_shape.f,
                self.conv_shape.num_groups,
            )
            filter_ptr = filter_ptr.offset(
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.r
                * self.conv_shape.s
                * self.conv_shape.c_per_group()
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )

        for r in range(self.conv_shape.r):
            for s in range(self.conv_shape.s):
                let input_offset = self.conv_shape.c * (
                    s + self.conv_shape.w * r
                )

                # Unpacked version. For each (r, s), we first offset the
                # filter pointer by (r, s) plus c_tile_offset. Later for
                # each c, we access micro_kernel_f_size contiguous elements.
                # These contiguous segments are strided by F.
                @parameter
                if not filter_packed:
                    filter_ptr = self.filter.data.offset(
                        (s + r * self.conv_shape.s)
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
        ](
            output_micro_tile,
            self.output.data.offset(
                n_ho_wo * self.conv_shape.f + f_tile_offset
            ),
        )

        @parameter
        if elementwise_epilogue_enabled and last_c_tile:
            for m in range(ho_wo, ho_wo + micro_kernel_height):
                # If has residual, the tile size has been extended to a simd_size.
                # Here needs to use the real bound F.
                let f_tile_size_bounded = self.conv_shape.f - f_tile_offset if has_residual else f_tile_size
                # The micro tile may cover points in different rows/images.
                # Convert the 1D index back to (n, ho, wo).
                self.elementwise_epilogue_fn(
                    n,
                    m // self.conv_shape.out_w,
                    m % self.conv_shape.out_w,
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
            if shape_output.at[3]().has_value():
                alias F = shape_output.get[3]()
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
            if shape_output.at[3]().has_value():
                alias F = shape_output.get[3]()
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

        @parameter
        if has_neon():
            accumulate_neon[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                prefetch_offset= -1,  # Don't prefetch with neon.
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
        else:
            accumulate_x86_simd[
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

    @always_inline
    fn _h_loop[
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

        # Divide each row into three part:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        let left_pad_impact_end = div_ceil(
            self.conv_shape.pad_w[0], self.conv_shape.stride[1]
        )
        let right_pad_impact_start = (
            self.conv_shape.w
            + self.conv_shape.pad_w[0]
            - self.conv_shape.s * self.conv_shape.dilation[1]
        ) // self.conv_shape.stride[1] + 1

        var filter_base: DTypePointer[filter_type]

        @parameter
        if filter_packed:
            filter_base = _get_group_filter_base(
                self.filter,
                self.conv_shape.f_to_group(f_tile_offset),
                self.conv_shape.f,
                self.conv_shape.num_groups,
            )
            filter_base = filter_base.offset(
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.c_per_group()
                * self.conv_shape.r
                * self.conv_shape.s
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )
        else:
            filter_base = self.filter.data.offset(
                c_tile_offset * self.conv_shape.f + f_tile_offset
            )

        let input_curr_image = self.input.data.offset(
            n * self.conv_shape.w * self.conv_shape.h * self.conv_shape.c
        )
        let output_curr_image = self.output.data.offset(
            n
            * self.conv_shape.out_w
            * self.conv_shape.out_h
            * self.conv_shape.f
        )

        # Temporary fix for #23189, edge case where padding is larger than input.
        # Use 1 x width kernel for all output points.
        if left_pad_impact_end >= right_pad_impact_start:
            for ho in range(
                self.partition_offsets[3],
                self.partition_offsets[3] + self.partition_sizes[3],
            ):
                let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[
                    0
                ]
                # Point to (n, 0, ho, c_tile_offset) mapped in input
                var input_base = input_curr_image.offset(
                    c_tile_offset
                    + self.conv_shape.c
                    * (-self.conv_shape.pad_w[0] + self.conv_shape.w * h)
                )
                # Point to (n, 0, ho, f_tile_offset) mapped in input
                var output_base = output_curr_image.offset(
                    f_tile_offset
                    + self.conv_shape.f * self.conv_shape.out_w * ho
                )

                for wo in range(self.conv_shape.out_w):
                    self._inner_loops_padding[
                        1,
                        micro_kernel_width,
                        simd_size,
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
                        wo,
                    )
                    input_base = input_base.offset(
                        self.conv_shape.stride[1] * self.conv_shape.c
                    )
                    output_base = output_base.offset(self.conv_shape.f)
            return

        for ho in range(
            self.partition_offsets[3],
            self.partition_offsets[3] + self.partition_sizes[3],
        ):
            let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
            # Point to (n, 0, ho, c_tile_offset) mapped in input
            var input_base = input_curr_image.offset(
                c_tile_offset
                + self.conv_shape.c
                * (-self.conv_shape.pad_w[0] + self.conv_shape.w * h)
            )
            # Point to (n, 0, ho, f_tile_offset) in output
            var output_base = output_curr_image.offset(
                f_tile_offset + self.conv_shape.f * self.conv_shape.out_w * ho
            )

            @parameter
            @always_inline
            fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                self._inner_loops_padding[
                    height,
                    micro_kernel_width,
                    simd_size,
                    effected_by_padding,
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
                    height * self.conv_shape.stride[1] * self.conv_shape.c,
                )
                output_base = output_base.offset(height * self.conv_shape.f)

            tile_middle_unswitch_boundaries[
                work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.out_w,
            )

    @always_inline
    fn _inner_loops_padding[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        w_padding_impact: Bool,
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
        """Inner loop computation with padding
        Given input (ho, wo), this kernel accumulates over the stencil RxS.
        """
        constrained[
            not has_residual or (has_residual and micro_kernel_width == 1),
            "Use Height x 1 kernel for residual in F.",
        ]()

        constrained[
            not w_padding_impact
            or (w_padding_impact and micro_kernel_height == 1),
            "USE 1 x width kernel on boundary",
        ]()

        alias micro_kernel_f_size = micro_kernel_width * simd_size

        # Shift in input when shifting 1 in filter S dimension.
        let input_shift = self.conv_shape.dilation[1] * self.conv_shape.c
        # WO dimension stride mapped in input.
        let wo_stride_in_input = self.conv_shape.stride[1] * self.conv_shape.c

        # Filter stride in S dimension
        let filter_S_stride: Int

        @parameter
        if filter_packed:  # FRSCf layout
            filter_S_stride = (
                self.conv_shape.c_per_group()
            ) * micro_kernel_f_size
        else:  # RSCF layout
            filter_S_stride = self.conv_shape.c * self.conv_shape.f

        # Filter stride in F dimension in FRSCf
        let filter_F_stride: Int

        @parameter
        if filter_packed:  # FRSCf layout
            filter_F_stride = (
                self.conv_shape.r * self.conv_shape.s * filter_S_stride
            )
        else:
            filter_F_stride = micro_kernel_f_size

        # This will be all lifted to simd registers for FMA unless the micro
        # kernel is too large that spills named registers.
        alias alignment = alignof[SIMD[output_type, simd_size]]()
        let output_micro_tile = NDBuffer[
            2,
            DimList(micro_kernel_height, micro_kernel_width * simd_size),
            output_type,
        ].aligned_stack_allocation[alignment]()

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

        # Shift in input H when shifting 1 in filter stencil' R dimension.
        var h_shift = 0
        # h index in input image
        let h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]
        for r in range(self.conv_shape.r):
            # Skip if row falls in padding.
            if h + h_shift < 0 or h + h_shift >= self.conv_shape.h:
                h_shift += self.conv_shape.dilation[0]
                continue

            var input_ptr = input_base.offset(
                h_shift * self.conv_shape.c * self.conv_shape.w
            )
            var filter_ptr = filter_base.offset(
                r * self.conv_shape.s * filter_S_stride
            )

            var w = wo * self.conv_shape.stride[1] - self.conv_shape.pad_w[0]
            for s in range(self.conv_shape.s):
                # Skip neighbor points in padding, if current point is
                # effected by padding.
                @parameter
                if w_padding_impact:
                    if w < 0 or w >= self.conv_shape.w:
                        w += self.conv_shape.dilation[1]
                        filter_ptr = filter_ptr.offset(filter_S_stride)
                        input_ptr = input_ptr.offset(input_shift)
                        continue

                self._accumulate[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                    # prefetch offset, default to 4 for now
                    4,
                ](
                    c_tile_size,
                    wo_stride_in_input,
                    input_ptr,
                    filter_ptr,
                    output_micro_tile.data,
                )

                w += self.conv_shape.dilation[1]
                filter_ptr = filter_ptr.offset(filter_S_stride)
                input_ptr = input_ptr.offset(input_shift)

            h_shift += self.conv_shape.dilation[0]

        # Store the micro tile
        self._store_output_micro_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            has_residual,
        ](output_micro_tile, output_base)

        # Apply elmentwise epilogue to the
        @parameter
        if elementwise_epilogue_enabled and last_c_tile:
            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            let f_tile_size_bounded = self.conv_shape.f - f_tile_offset if has_residual else f_tile_size
            for wo_idx in range(wo, wo + micro_kernel_height):
                self.elementwise_epilogue_fn(
                    n, ho, wo_idx, f_tile_offset, f_tile_size_bounded
                )

    fn _f_tile_loop_static[
        last_c_tile: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        alias WO = shape_output.get[2]()  # NHWC
        alias F = shape_output.get[3]()  # NHWC
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_shape = get_micro_kernel_shape[
            WO, F, conv_attr, simd_size
        ]()
        alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size

        let f_round_by_simd = (
            (self.partition_offsets[2] + self.partition_sizes[2]) // simd_size
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
            self.partition_offsets[2],
            f_round_by_simd,
            VariadicList[Int](micro_kernel_f_size, simd_size),
            simd_size,
        )

        let residual = F - f_round_by_simd
        if (
            self.partition_offsets[2] + self.partition_sizes[2] == F
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

        alias H = shape_input.get[1]()  # NHWC
        alias W = shape_input.get[2]()  # NHWC
        alias C = shape_input.get[3]()  # NHWC
        alias R = shape_filter.get[1]()  # FRSCf
        alias S = shape_filter.get[2]()  # FRSCf
        alias HO = shape_output.get[1]()  # NHWC
        alias WO = shape_output.get[2]()  # NHWC
        alias F = shape_output.get[3]()  # NHWC

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
            self.partition_offsets[3],
            self.partition_offsets[3] + self.partition_sizes[3],
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

        alias R = shape_filter.get[1]()  # FRSCf
        alias S = shape_filter.get[2]()  # FRSCf
        alias C = shape_input.get[3]()  # NHWC
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

        alias W = shape_input.get[2]()  # NHWC
        alias H = shape_input.get[1]()  # NHWC
        alias WO = shape_output.get[2]()  # NHWC
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
        alias F = shape_output.get[3]()  # NHWC

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
    filter_type: DType,
    single_thread_blocking_override: Bool,
](
    filter_buf: NDBuffer[4, DimList.create_unknown[4](), filter_type],
    num_groups: Int,
) -> StaticIntTuple[5]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
        filter_type: Type of the filter.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        filter_buf: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """
    # ASSUME input has layout RSCF
    let R = filter_buf.dim(0)
    let S = filter_buf.dim(1)
    let C = filter_buf.dim(2)
    let F = filter_buf.dim(3)
    return pack_filter_shape_impl[filter_type](R, S, C, F, num_groups)


@always_inline
fn pack_conv_filter_shape[
    filter_type: DType,
    WO: Int,
    single_thread_blocking_override: Bool,
](
    filter_buf: NDBuffer[4, DimList.create_unknown[4](), filter_type],
    num_groups: Int,
) -> StaticIntTuple[5]:
    """
    Compute the output shape of convolution filter packing.

    Parameters:
        filter_type: Type of the filter.
        WO: Width dimension of the convolution output.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        filter_buf: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """
    # TODO specialize via `WO`
    return pack_conv_filter_shape[filter_type, single_thread_blocking_override](
        filter_buf, num_groups
    )


@always_inline
fn _get_group_filter_base[
    rank: Int, type: DType, dims: DimList
](
    filter: NDBuffer[rank, dims, type], group_idx: Int, f: Int, num_groups: Int
) -> DTypePointer[type]:
    # Each group is zero padded to the nearest multiple of
    # div_ceil(F_per_group, micro_kernel_width)*R*S*C*micro_kernel_width
    # Within a group the residual filters are ragged so normal NDBuffer
    # indexing cannot be used.
    let shape = filter.get_shape()
    let micro_kernel_f_size = shape[rank - 1]
    return filter.data.offset(
        _compute_ndbuffer_offset(
            filter,
            StaticIntTuple[rank](
                group_idx
                * div_ceil(
                    f // num_groups,
                    micro_kernel_f_size,
                ),
                0,
                0,
                0,
                0,
            ),
        )
    )


@always_inline
fn pack_filter[
    type: DType,
](
    filter: NDBuffer[4, DimList.create_unknown[4](), type],
    packed_filter: NDBuffer[5, DimList.create_unknown[5](), type],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.
    Use the default micro kernel size for dynamic shapes."""

    alias simd_size = simdwidthof[type]()
    alias micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    pack_filter[type, simd_size, micro_kernel_f_size](
        filter, packed_filter, num_groups
    )


@always_inline
fn pack_filter[
    type: DType,
    simd_size: Int,
    micro_kernel_f_size: Int,
](
    filter: NDBuffer[4, DimList.create_unknown[4](), type],
    packed_filter: NDBuffer[5, DimList.create_unknown[5](), type],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.

    Parameters:
        type: Filter data type.
        simd_size: Can differ from the simd size of the input type.
        micro_kernel_f_size: The size of the last dimension in FRSCf, which is
            equals the size of the micro kernel's F dimension.

    Args:
        filter: Filter in RSCF layout.
        packed_filter: Packed filter in FRScf layout. Here,
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
    constrained[simd_size <= simdwidthof[type]()]()

    let R = filter.dim[0]()
    let S = filter.dim[1]()
    let C = filter.dim[2]()
    let F = filter.dim[3]()
    let F_per_group = F // num_groups
    packed_filter.zero()

    # For the last filters in a group that are leftover after dividing the filters
    # per group by the microkernel width, the inner dim has length simd_width.
    # This makes the packed format "ragged" - you cannot use normal NDBuffer
    # indexing to get the value corresponding to a set of coordinates.
    # Each group is zero padded to the nearest multiple of
    # div_ceil(F_per_group, micro_kernel_width)*R*S*C*micro_kernel_width.
    for g in range(num_groups):
        let group_start = _get_group_filter_base(
            packed_filter, g, F, num_groups
        )

        @always_inline
        @parameter
        fn pack[f_tile_size: Int](f_tile_start: Int):
            var packed_filter_ptr = group_start.offset(f_tile_start * R * S * C)
            for r in range(R):
                for s in range(S):
                    for c in range(C):
                        let filter_ptr = filter.data.offset(
                            _compute_ndbuffer_offset(
                                filter,
                                StaticIntTuple[4](
                                    r, s, c, g * F_per_group + f_tile_start
                                ),
                            )
                        )

                        @always_inline
                        @parameter
                        fn body[idx: Int]():
                            let filter_vec = filter_ptr.offset(
                                idx * simd_size
                            ).simd_load[simd_size]()
                            packed_filter_ptr.offset(
                                idx * simd_size
                            ).simd_store[simd_size](filter_vec)

                        unroll[f_tile_size // simd_size, body]()

                        packed_filter_ptr = packed_filter_ptr.offset(
                            f_tile_size
                        )

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
                packed_filter, g, F, num_groups
            )
            var packed_filter_ptr = group_start.offset(
                F_round_by_simd * R * S * C
            )
            for r in range(R):
                for s in range(S):
                    for c in range(C):
                        let filter_ptr = filter.data.offset(
                            _compute_ndbuffer_offset(
                                filter,
                                StaticIntTuple[4](
                                    r, s, c, g * F_per_group + F_round_by_simd
                                ),
                            )
                        )
                        # Load remainder elements and pad with zero to
                        # to fill a simd vector.
                        let filter_vec = partial_simd_load[simd_size](
                            filter_ptr, 0, residual, 0.0
                        )
                        packed_filter_ptr.simd_store(filter_vec)
                        # Hence, packed filter is incremented by simd_size
                        packed_filter_ptr = packed_filter_ptr.offset(simd_size)


@always_inline
fn pack_conv_filter[
    type: DType,
](
    filter: NDBuffer[4, DimList.create_unknown[4](), type],
    packed_filter: NDBuffer[5, DimList.create_unknown[5](), type],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.

    Args:
        filter: Filter in RSCF layout.
        packed_filter: Packed filter in FRScf layout. Here,
            F       - the index of continuous segments in micro kernel.
            R, S, C - original R, S, C.
            f       - the index within a continuous segments.
        num_groups: The number of groups in the convolution.
    """
    pack_filter(filter, packed_filter, num_groups)


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
) -> StaticIntTuple[input_rank]:
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

    # TODO(#17512)
    debug_assert(input_rank == 4, "input rank must be 4")
    debug_assert(input_rank == filter_rank, "input rank must match filter rank")
    debug_assert(
        strides_buf.dim(0) == input_rank - 2
        and dilations_buf.dim(0) == input_rank - 2,
        "strides and dilations size must be input rank - 2",
    )
    debug_assert(
        paddings_buf.dim(0) == 2 * (input_rank - 2),
        "paddings size must be 2 * (input rank - 2)",
    )

    # Assume input has layout NHWC
    let batch_size = input_buf.dim(0)
    let input_channels = input_buf.dim(3)
    # Assume filter has layout RSCF
    let filter_channels = filter_buf.dim(2)
    let num_groups = int(num_groups_buf[0])
    let output_channels = filter_buf.dim(3)

    # TODO(#17512)
    debug_assert(
        input_channels == (num_groups * filter_channels),
        "input channels and groups times filter channels must match",
    )
    debug_assert(
        (output_channels % num_groups) == 0,
        "output_channels must be divisible by the number of groups",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[input_rank]()
    output_shape[0] = batch_size
    output_shape[1] = get_sliding_window_out_dim(
        input_buf.dim(1),
        filter_buf.dim(0),
        int(dilations_buf[0]),
        int(strides_buf[0]),
        int(paddings_buf[0] + paddings_buf[1]),
    )
    output_shape[2] = get_sliding_window_out_dim(
        input_buf.dim(2),
        filter_buf.dim(1),
        int(dilations_buf[1]),
        int(strides_buf[1]),
        int(paddings_buf[2] + paddings_buf[3]),
    )
    output_shape[3] = output_channels

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
    ) escaping:
        alias simd_size = simdwidthof[output_type]()

        @always_inline
        @parameter
        fn body[width: Int](idx: Int):
            let coords = Index(n, ho, wo, f_offset + idx)
            let vec = output.simd_load[width](coords)
            epilogue_wrapper[output_type, width](coords, vec)

        vectorize[simd_size, body](f_size)

    ConvDirectNHWC[
        filter_rank,
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
