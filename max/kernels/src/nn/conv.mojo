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

from collections import Optional, Dict
from math import align_down, ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
comptime OpaquePointer = LegacyUnsafePointer[
    mut=True, NoneType, origin=MutAnyOrigin
]

from os import abort
from ffi import _get_global_or_null, external_call
from sys.info import align_of, simd_width_of

from _cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionMode_t,
    cudnnConvolutionStruct,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionGroupCount,
    cudnnSetConvolutionMathType,
)
from _cudnn.infer import (
    cudnnContext,
    cudnnConvolutionFwdAlgo_t,
    cudnnCreate,
    cudnnCreateFilterDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnDataType_t,
    cudnnDestroy,
    cudnnDestroyFilterDescriptor,
    cudnnDestroyTensorDescriptor,
    cudnnFilterStruct,
    cudnnMathType_t,
    cudnnSetFilter4dDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorStruct,
)
from algorithm import (
    elementwise,
    sync_parallelize,
    tile,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)
from buffer.buffer import (
    # LayoutTensor,
    partial_simd_load,
    partial_simd_store,
    prod_dims,
)
from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import CUDA
from gpu import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor, RuntimeLayout, IntTuple, UNKNOWN_VALUE
from linalg.accumulate import _Accumulator
from linalg.utils import partition_work
from runtime.asyncrt import parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from sys import has_nvidia_gpu_accelerator
from sys.info import _accelerator_arch
from gpu.host.info import B200
from utils.index import Index, IndexList
from utils.numerics import get_accum_type


from .conv_utils import (
    ConvInfoStatic,
    ConvPartition,
    ConvShape,
    align_down_residual,
    elementwise_epilogue_type,
    elementwise_simd_epilogue_type,
    get_conv_num_partitions,
    get_conv_shape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
    get_partition,
    reorder_padding,
)
from .shapes import get_sliding_window_out_dim
from nn.pad_gpu import pad_constant as pad_constant_gpu


@fieldwise_init
struct Naive2dConvolution[
    output_type: DType,
    input_type: DType,
    filter_type: DType,
](ImplicitlyCopyable):
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: UnsafePointer[Scalar[Self.output_type]]
    var input: UnsafePointer[Scalar[Self.input_type]]
    var filter: UnsafePointer[Scalar[Self.filter_type]]
    var pad_d: IndexList[2]
    var pad_h: IndexList[2]
    var pad_w: IndexList[2]
    var stride: IndexList[3]
    var dilation: IndexList[3]
    var num_groups: Int

    # Derived params.
    var output_shape: IndexList[5]  # NDHWC layout.
    var input_shape: IndexList[5]  # NDHWC layout.
    var filter_shape: IndexList[5]  # QRSCF layout.

    @staticmethod
    fn run(
        output: UnsafePointer[Scalar[Self.output_type]],
        input: UnsafePointer[Scalar[Self.input_type]],
        filter: UnsafePointer[Scalar[Self.filter_type]],
        output_shape: IndexList[5],
        input_shape: IndexList[5],
        filter_shape: IndexList[5],
        pad_d: IndexList[2],
        pad_h: IndexList[2],
        pad_w: IndexList[2],
        stride: IndexList[3],
        dilation: IndexList[3],
        num_groups: Int,
    ):
        # Create an instance of the convolution op.
        var naive2d_convolution = Naive2dConvolution[
            Self.output_type, Self.input_type, Self.filter_type
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
        out self,
        output: UnsafePointer[Scalar[Self.output_type]],
        input: UnsafePointer[Scalar[Self.input_type]],
        filter: UnsafePointer[Scalar[Self.filter_type]],
        output_shape: IndexList[5],
        input_shape: IndexList[5],
        filter_shape: IndexList[5],
        pad_d: IndexList[2],
        pad_h: IndexList[2],
        pad_w: IndexList[2],
        stride: IndexList[3],
        dilation: IndexList[3],
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
        var value: Scalar[Self.output_type] = 0

        # Input dims.
        var D = self.input_shape[1]
        var H = self.input_shape[2]
        var W = self.input_shape[3]
        var C = self.input_shape[4]
        var image_bound = Index(D, H, W)
        var C_per_group = C // self.num_groups

        # Filter dims.
        var Q = self.filter_shape[0]
        var R = self.filter_shape[1]
        var S = self.filter_shape[2]

        # Output dims.
        var DO = self.output_shape[1]
        var HO = self.output_shape[2]
        var WO = self.output_shape[3]
        var F = self.output_shape[4]

        var g = f // (F // self.num_groups)

        for q in range(Q):
            for r in range(R):
                for s in range(S):
                    # Compute input access index, on the H and W dimension.
                    var dhw = (
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
                            var input_val = self.input[
                                c
                                + C
                                * (dhw[2] + W * (dhw[1] + H * (dhw[0] + D * n)))
                            ]
                            var c_in_group = c % C_per_group
                            var filter_val = self.filter[
                                f
                                + F
                                * (
                                    c_in_group
                                    + C_per_group * (s + S * (r + R * q))
                                )
                            ]
                            value += (
                                input_val.cast[Self.output_type]()
                                * filter_val.cast[Self.output_type]()
                            )

        # Store the computed output at the given output position..
        self.output.store(f + F * (wo + WO * (ho + HO * (do + DO * n))), value)


# ===----------------------------------------------------------------------=== #
# Direct convolution helpers
# ===----------------------------------------------------------------------=== #


@always_inline
fn _m_to_n_ho_wo_nhwc(m: Int, HO: Int, WO: Int) -> IndexList[3]:
    """Converts post-im2col m dimension index to pre-im2col coordinates on
    (N, Hout, Wout) dimensions.
        Args:
            m (Int): Index on M dimension.
            conv_shape (ConvShape): convolution dimension description.

        Returns (IndexList):
            The translated 3d indices in (N, Hout, Wout) format.
    TODO(Fixel): This utility should be generalized into a im2col util
    class with some additional layout agnostic logic.
    """
    var n = m // (HO * WO)
    var ho = (m % (HO * WO)) // WO
    var wo = m % WO
    return Index(n, ho, wo)


# Reduce helper when the input channel dimension is partitioned.
@always_inline
fn _reduce_output[
    dtype: DType,
    //,
    simd_size: Int,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    scratch: UnsafePointer[Scalar[dtype]],
    output: UnsafePointer[Scalar[dtype]],
    N: Int,
    output_space_dims: IndexList,
    F: Int,
    num_partitions: Int,
    num_threads: Int,
):
    var num_rows = N * output_space_dims.flattened_length()
    var buf_size = num_rows * F

    # Reduce from the output scratch buffer to the actual output.
    @parameter
    @always_inline
    fn reduce_task(tid: Int):
        # Use all threads in reduction.
        var reduce_range = partition_work(tid, num_threads, num_rows, 1)

        @always_inline
        fn sum[width: Int](offset: Int) unified {mut}:
            var tid_output_offset = reduce_range[0] * F + offset
            var vec = scratch.load[width=width](tid_output_offset)
            # The number of partitions here is typically small.
            # There may not be much benefit from unrolling the reduction axis.
            # Only unroll the last dimension.
            for i in range(1, num_partitions):
                vec += scratch.load[width=width](
                    tid_output_offset + i * buf_size
                )
            output.store(tid_output_offset, vec)

        vectorize[simd_size, unroll_factor=4](reduce_range[1] * F, sum)

        comptime if elementwise_epilogue:
            comptime epilogue = elementwise_epilogue.value()
            for m in range(reduce_range[0], reduce_range[0] + reduce_range[1]):
                var nhowo = _m_to_n_ho_wo_nhwc(
                    m, output_space_dims[0], output_space_dims[1]
                )
                epilogue(Index(nhowo[0], nhowo[1], nhowo[2], 0), F)

    # NOTE: _synchronous, so use of locally allocated output_ptr is safe.
    sync_parallelize[reduce_task](num_threads)


# ===----------------------------------------------------------------------=== #
# Direct Convolution Entry Point                                               #
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ConvDirectNHWC[
    input_mut: Bool,
    filter_mut: Bool,
    conv_attr_rank: Int,
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_origin: Origin[mut=input_mut],
    filter_origin: Origin[mut=filter_mut],
    output_origin: MutOrigin,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    conv_attr: ConvInfoStatic[conv_attr_rank],
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](ImplicitlyCopyable):
    """Implement the outer loops for direct convolution.
    Collapse N, HO, WO into one dimension n_ho_wo. Tile n_ho_wo, C, and F.
    The tile factor for C and F are chosen by a heuristic prioritizing C.
    n_ho_wo is tiled by micro kernel's height.

    If n_ho_wo is large enough to spill LLC, we may need to tile n_ho_wo as the
    outer most loop with a factor fit in LLC.

    Assume F is divisible at least by simd_size.
    """

    var output: LayoutTensor[
        Self.output_type, Self.output_layout, Self.output_origin
    ]
    var input: LayoutTensor[
        Self.input_type, Self.input_layout, Self.input_origin
    ]
    var filter: LayoutTensor[
        Self.filter_type, Self.filter_layout, Self.filter_origin
    ]

    var conv_shape: ConvShape[Self.conv_attr_rank]

    # Support partition in 4 dims: (n, c, f, ho_or_howo). If the input is
    # padded, the output spatial dims are merged into one as howo. If not
    # padded, only ho is partitioned for now.
    var partition: ConvPartition

    var cf_tile_size: IndexList[2]

    # If shapes and attributes are known at compile time
    comptime packed_and_fully_static = Self.conv_attr.all_known() and Self.input_layout.shape.all_known[
        1, Self.input_layout.rank()
    ]() and Self.output_layout.shape.all_known[
        1, Self.output_layout.rank()
    ]() and Self.filter_layout.shape.all_known() and Self.filter_packed

    @staticmethod
    fn run(
        output: LayoutTensor[
            Self.output_type, Self.output_layout, Self.output_origin
        ],
        input: LayoutTensor[
            Self.input_type, Self.input_layout, Self.input_origin
        ],
        filter: LayoutTensor[
            Self.filter_type, Self.filter_layout, Self.filter_origin
        ],
        conv_shape: ConvShape[Self.conv_attr_rank],
    ) raises:
        comptime assert Self.conv_attr_rank == Self.input_layout.rank() - 2
        comptime simd_size = simd_width_of[Self.output_type]()
        # TODO: extend to 1d/3d.
        comptime WO = Int(
            Self.output_layout.shape[output.rank - 2]
        ) if input.rank == 4 else UNKNOWN_VALUE
        comptime F = Int(Self.output_layout.shape[output.rank - 1])
        comptime micro_kernel_shape = get_micro_kernel_shape[
            Self.conv_attr_rank,
            WO,
            F,
            Self.conv_attr,
            simd_size,
        ]()
        comptime micro_kernel_height = micro_kernel_shape[0]
        comptime micro_kernel_width = micro_kernel_shape[1]
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var cf_tile_size = get_conv_tile_shape[Self.filter_type](
            conv_shape.c,
            conv_shape.filter_window_flat_size(),
            micro_kernel_width,
        )

        comptime if Self.conv_attr.num_groups != UNKNOWN_VALUE:
            comptime assert (
                Self.filter_packed or Self.conv_attr.num_groups == 1
            ), (
                "if number of conv groups is statically known, conv filter"
                " must be prepacked when num_groups > 1"
            )

        if conv_shape.num_groups > 1 and not Self.filter_packed:
            raise Error("grouped conv requires packed filter")
        if conv_shape.c % conv_shape.num_groups != 0:
            raise Error("channel count must be divisible by group count")
        if conv_shape.f % conv_shape.num_groups != 0:
            raise Error("filter count must be divisible by group count")

        # Number of partitions in n, ho_wo, c, f dimensions.
        var num_threads = parallelism_level()
        var num_partitions = get_conv_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_threads, conv_shape)
        var num_tasks = num_partitions.flattened_length()

        # Wrap the pointer inside LayoutTensor so it can be properly captured by async closure.
        var output_ptr = output.ptr
        var output_size = output.size()
        var scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = UnsafePointer[Scalar[Self.output_type]].alloc(
                scratch_size
            )
        var output_scratch = LayoutTensor[
            Self.output_type, Layout.row_major(UNKNOWN_VALUE)
        ](
            output_ptr,
            RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
                Index(scratch_size)
            ),
        )

        @__copy_capture(
            num_partitions, cf_tile_size, output_scratch, output_size
        )
        @parameter
        @always_inline
        fn task_func(task_id: Int):
            var partition = get_partition(
                task_id,
                num_partitions,
                conv_shape,
                micro_kernel_height,
                micro_kernel_f_size,
            )

            if partition.empty():
                return

            var task_tile_size = Index(
                min(cf_tile_size[0], partition.c_size), cf_tile_size[1]
            )

            # TODO: Need to have a more robust way to compute task_id_c
            var task_id_c = (task_id // num_partitions[2]) % num_partitions[1]
            var task_output = LayoutTensor[
                Self.output_type, Self.output_layout, Self.output_origin
            ](
                output_scratch.ptr + task_id_c * output_size,
                RuntimeLayout[Self.output_layout].row_major(
                    output.runtime_layout.shape.value.canonicalize()
                ),
            )

            var instance = ConvDirectNHWC[
                Self.input_layout,
                Self.filter_layout,
                Self.output_layout,
                Self.input_origin,
                Self.filter_origin,
                Self.output_origin,
                Self.input_type,
                Self.filter_type,
                Self.output_type,
                Self.filter_packed,
                Self.conv_attr,
                Self.elementwise_epilogue,
            ](
                task_output,
                input,
                filter,
                conv_shape,
                partition,
                task_tile_size,
            )
            instance._batch_group_loop()

        if num_partitions[1] > 1:
            sync_parallelize[task_func](num_tasks)

            # Reduce from the output scratch buffer to the actual output.
            _reduce_output[
                simd_size,
                # Only support channel partition for 2D shapes (ResNet).
                elementwise_epilogue = Self.elementwise_epilogue if input.rank
                == 4 else None,
            ](
                output_scratch.ptr,
                output.ptr,
                conv_shape.n,
                conv_shape.output_space_dims(),
                conv_shape.f,
                num_partitions[1],
                num_threads,
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
                var n = ng // self.conv_shape.num_groups
                var g = ng % self.conv_shape.num_groups
                self._c_tile_loop[padded](n, g, self.cf_tile_size[0])

        unswitch[body](self.conv_shape.padded())

    fn _c_tile_loop[padded: Bool](self, n: Int, g: Int, tile_size: Int):
        """Loop over C tiles."""

        # TODO: Extend to 1D/3D.
        # fmt: off
        comptime apply_static_shape_optimization = \
            self.packed_and_fully_static \
            and padded \
            and Self.conv_attr.num_groups == 1 \
            and Self.input_layout.rank() == 4
        # fmt: on

        @always_inline
        @parameter
        fn c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
            # Only apply static shape optimizations to shapes with padding since
            # there is a fast path for pointwise (no padding) conv with strides.
            # Grouped conv logic has not been plumbed into static specialized funcs yet.
            comptime if apply_static_shape_optimization:
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
            var c_start = g * self.conv_shape.c_per_group()
            var c_round_by_tile = align_down(
                (self.conv_shape.c_per_group() - 1), tile_size
            )
            var c_round_by_tile_residual = (
                self.conv_shape.c_per_group() - c_round_by_tile
            )
            tile[c_tile_iteration](
                c_start,
                c_start + c_round_by_tile,
                tile_size,
            )

            # Update the last c tile with fusion
            comptime if apply_static_shape_optimization:
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
        comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
        comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # TODO: Extend the merged loop to support 1d and 3d.
        # For now, only merge HO and WO dims for 2D conv w/o padding.
        comptime merge_output_space_loops = (
            not padded
        ) and Self.input_layout.rank() == 4

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            comptime if not merge_output_space_loops:
                self.output_space_loop[
                    micro_kernel_height, size // simd_size, False, last_c_tile
                ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)
            else:
                self.output_space_flat_loop[size, False, last_c_tile](
                    n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size
                )

        var f_per_group = self.conv_shape.f_per_group()

        # The partition heuristic sees F_per_group and may partition it.
        # The partition's F_offset should be added to the group's F offset to
        # get the actually offset in output's F dim.
        var group_f_offset = g * f_per_group + self.partition.f_offset

        var group_f_end_align_simd = group_f_offset + align_down(
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
        var residual = align_down_residual(f_per_group, simd_size)
        if (
            self.partition.f_offset + self.partition.f_size == f_per_group
            and residual > 0
        ):
            comptime if not merge_output_space_loops:
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
        comptime assert not has_residual or (
            has_residual and micro_kernel_width == 1
        ), "Use Height x 1 kernel for residual in F."

        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # Base input offsets.
        var input_base_stack = InlineArray[Int32, micro_kernel_height](
            uninitialized=True
        )
        var input_base_offsets = LayoutTensor[
            DType.int32, Layout.row_major(micro_kernel_height)
        ](input_base_stack)

        comptime for i in range(micro_kernel_height):
            input_base_offsets[i] = Int32(
                self.conv_shape.output_flat_coord_to_input_offset(
                    n, output_flat_coord + i
                )
                + c_tile_offset
            )

        comptime alignment = align_of[SIMD[Self.output_type, simd_size]]()

        var acc = _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ]()

        var output_offset = (
            self.conv_shape.f
            * (n * self.conv_shape.output_image_flat_size() + output_flat_coord)
            + f_tile_offset
        )

        if self.is_new_c_accum(c_tile_offset):
            acc.init(0)
        else:
            acc.load[partial_load=has_residual](
                self.output.ptr + output_offset,
                self.conv_shape.f,
                self.conv_shape.f_per_group() % simd_size,
            )
        var filter_ptr: UnsafePointer[
            Scalar[Self.filter_type]
        ] = self.filter.ptr

        comptime if Self.filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter,
                self.conv_shape.c_to_group(c_tile_offset),  # group index
                self.conv_shape.f_per_group(),
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr + (
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
                var input_offset = self.conv_shape.c * (
                    s + self.conv_shape.w() * r
                )

                # Unpacked version. For each (r, s), we first offset the
                # filter pointer by (r, s) plus c_tile_offset. Later for
                # each c, we access micro_kernel_f_size contiguous elements.
                # These contiguous segments are strided by F.
                comptime if not Self.filter_packed:
                    filter_ptr = self.filter.ptr + (
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
                    has_residual and not Self.filter_packed,
                    prefetch_offset=4,
                ](
                    input_base_offsets,
                    input_offset,
                    c_tile_size,
                    self.input.ptr,
                    filter_ptr,
                    acc,
                )

                # Shift C*f to get the next point in stencil (s+1) for FRSCf layout.
                if Self.filter_packed:
                    filter_ptr = filter_ptr + (
                        self.conv_shape.c_per_group() * micro_kernel_f_size
                    )

        acc.store[partial_store=has_residual](
            self.output.ptr + output_offset,
            self.conv_shape.f,
            self.conv_shape.f_per_group() % simd_size,
        )

        comptime if Self.elementwise_epilogue.__bool__() and last_c_tile.__bool__():
            comptime epilogue = Self.elementwise_epilogue.value()

            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            var f_tile_size_bounded: Int

            comptime if has_residual:
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
                epilogue(
                    Index(
                        n,
                        m // self.conv_shape.wo(),
                        m % self.conv_shape.wo(),
                        f_tile_offset,
                    ),
                    f_tile_size_bounded,
                )

    @always_inline
    fn _init_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
    ](
        self,
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
        ],
    ):
        """Initialize a micro tile to zero.
        Arguments:
            n_ho_wo: offset of micro tile in fused (n, ho, wo) dimension.
            f: offset of micro tile in F dimension.
            output_micro_tile: micro_kernel_height * micro_kernel_width simd vectors.
        """

        comptime for idx0 in range(micro_kernel_height):
            comptime for idx1 in range(micro_kernel_width):
                output_micro_tile.store[width=simd_size](
                    Index(idx0, idx1 * simd_size),
                    SIMD[Self.output_type, simd_size](0.0),
                )

    @always_inline
    fn _load_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_base: UnsafePointer[Scalar[Self.output_type]],
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
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

        comptime for i in range(micro_kernel_height):
            comptime for j in range(micro_kernel_width):
                comptime if has_residual:
                    var residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    output_micro_tile.store[width=simd_size](
                        Index(i, j * simd_size),
                        partial_simd_load[simd_size](
                            output_ptr + j * simd_size, 0, residual, 0.0
                        ),
                    )
                else:
                    output_micro_tile.store[width=simd_size](
                        Index(i, j * simd_size),
                        (output_ptr + j * simd_size).load[width=simd_size](),
                    )

            comptime if (
                Self.output_layout.shape[Self.output_layout.rank() - 1]
                != UNKNOWN_VALUE
            ):
                comptime F = Int(
                    Self.output_layout.shape[Self.output_layout.rank() - 1]
                )
                output_ptr = output_ptr + F
            else:
                output_ptr = output_ptr + self.conv_shape.f

    @always_inline
    fn _store_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
        ],
        output_base: UnsafePointer[Scalar[Self.output_type]],
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

        comptime for i in range(micro_kernel_height):
            comptime for j in range(micro_kernel_width):
                var output_vec = output_micro_tile.load[width=simd_size](
                    Index(i, j * simd_size)
                )

                comptime if has_residual:
                    var residual = align_down_residual(
                        self.conv_shape.f_per_group(), simd_size
                    )
                    partial_simd_store[simd_size](
                        output_ptr + j * simd_size,
                        0,
                        residual,
                        output_vec,
                    )
                else:
                    output_ptr.store(j * simd_size, output_vec)

            comptime if (
                Self.output_layout.shape[Self.output_layout.rank() - 1]
                != UNKNOWN_VALUE
            ):
                comptime F = Int(
                    Self.output_layout.shape[Self.output_layout.rank() - 1]
                )
                output_ptr = output_ptr + F
            else:
                output_ptr = output_ptr + self.conv_shape.f

    @always_inline
    fn _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: LayoutTensor[
            DType.int32, Layout.row_major(micro_kernel_height)
        ],
        input_offset: Int,
        c_tile_size: Int,
        input: UnsafePointer[Scalar[Self.input_type]],
        filter: UnsafePointer[Scalar[Self.filter_type]],
        mut acc: _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
        ],
    ):
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var F = self.output.dim[3]()
        var filter_stride = micro_kernel_f_size if Self.filter_packed else F

        acc.accumulate[
            prefetch_offset=prefetch_offset,
            partial_load_b = has_residual and not Self.filter_packed,
        ](
            c_tile_size,
            input,
            input_base_offsets,
            input_offset,
            filter,
            filter_stride,
            F % simd_size,
        )

    @always_inline
    fn _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
        row_start: Int,
        row_stop: Int,
    ](
        self,
        c_tile_size: Int,
        input_stride: Int,
        input_base: UnsafePointer[Scalar[Self.input_type]],
        filter_base: UnsafePointer[Scalar[Self.filter_type]],
        mut acc_in: _Accumulator[
            Self.output_type, micro_kernel_height, micro_kernel_width, simd_size
        ],
    ):
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        var F = self.output.dim[3]()
        var filter_stride = micro_kernel_f_size if Self.filter_packed else F

        # NOTE: To avoid initial load and final store after accumulation, this
        # function is rewritten to use a subset of storage in acc_in for rows
        # in range [row_start, row_stop].
        var acc = _Accumulator[
            Self.output_type,
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            row_start,
            row_stop,
        ](acc_in._storage)

        acc.accumulate[
            prefetch_offset=prefetch_offset,
            partial_load_b = has_residual and not Self.filter_packed,
        ](
            c_tile_size,
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
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_height = get_direct_conv_micro_kernel_height()
        comptime micro_kernel_width = micro_kernel_f_size // simd_size

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
        tile[iteration, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)](
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        )

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
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        # Current group index.
        var g = self.conv_shape.f_to_group(f_tile_offset)

        # Filter pointer to the current cf tile offset location.
        var filter_ptr: UnsafePointer[Scalar[Self.filter_type]]

        comptime if Self.filter_packed:
            # Move the pointer to the current group's start.
            filter_ptr = _get_group_filter_base(
                self.filter, g, self.conv_shape.f_per_group()
            )
            # Move the pointer to (c_tile_offset, f_tile_offset) mapped in
            # current group.
            filter_ptr = filter_ptr + (
                # Jump over f_tile_offset in current group.
                self.conv_shape.f_in_group(f_tile_offset)
                * self.conv_shape.c_per_group()
                * self.conv_shape.filter_window_flat_size()
                # Jump over c_tile_offset in current group.
                + self.conv_shape.c_in_group(c_tile_offset)
                * micro_kernel_f_size
            )
        else:
            filter_ptr = self.filter.ptr + (
                c_tile_offset * self.conv_shape.f + f_tile_offset
            )

        # Pointer to input and output of the current sample (batch dim).
        # fmt: off
        var input_ptr  = self.input.ptr + c_tile_offset \
                       + self.conv_shape.input_image_flat_size() \
                       * self.conv_shape.c * n

        var output_ptr = self.output.ptr + f_tile_offset \
                       + self.conv_shape.output_image_flat_size() \
                       * self.conv_shape.f * n
        # fmt: on

        # Divide each row into three part:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        var left_pad_impact_end = ceildiv(
            self.conv_shape.pad_w[0],
            self.conv_shape.stride[comptime (Self.input_layout.rank() - 3)],
        )
        var right_pad_impact_start = (
            self.conv_shape.w()
            + self.conv_shape.pad_w[0]
            - self.conv_shape.s()
            * self.conv_shape.dilation[comptime (Self.input_layout.rank() - 3)]
        ) // self.conv_shape.stride[comptime (Self.input_layout.rank() - 3)] + 1

        comptime if Self.input_layout.rank() == 3:
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
        elif Self.input_layout.rank() == 4:
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
        elif Self.input_layout.rank() == 5:
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

    fn output_space_loop_1d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[Scalar[output_dt]],
        input: UnsafePointer[Scalar[input_dt]],
        filter: UnsafePointer[Scalar[filter_dt]],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        # Offset by -pad_w because s loop starts from the leftmost neighbor
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
                Self.filter_packed,
                effected_by_padding,
                has_residual,
                last_c_tile,
                elementwise_epilogue = Self.elementwise_epilogue,
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
            )

            input_base = input_base + (
                height * self.conv_shape.stride[0] * self.conv_shape.c
            )
            output_base = output_base + height * self.conv_shape.f

        tile_middle_unswitch_boundaries[
            work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
        ](
            0,
            left_pad_impact_end,
            right_pad_impact_start,
            self.conv_shape.wo(),
        )
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = input_base
        _ = output_base

    fn output_space_loop_2d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[Scalar[output_dt]],
        input: UnsafePointer[Scalar[input_dt]],
        filter: UnsafePointer[Scalar[filter_dt]],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            var h = ho * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]

            # Points input to the start of the row.
            # Offset by -pad_w because s loop starts from the leftmost neighbor
            # in padding. The kernel skip the padding point and increment the
            # pointer.
            var input_base = input + self.conv_shape.c * (
                -self.conv_shape.pad_w[0] + self.conv_shape.w() * h
            )

            # Points output to the start of the row
            var output_base = (
                output + self.conv_shape.f * self.conv_shape.wo() * ho
            )

            @parameter
            @always_inline
            fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                conv2d_update_wo_tile[
                    height,
                    micro_kernel_width,
                    simd_size,
                    Self.filter_packed,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
                    elementwise_epilogue = Self.elementwise_epilogue,
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
                )

                input_base = input_base + (
                    height * self.conv_shape.stride[1] * self.conv_shape.c
                )
                output_base = output_base + height * self.conv_shape.f

            tile_middle_unswitch_boundaries[
                work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.wo(),
            )
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = input_base
            _ = output_base

    fn output_space_loop_3d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[Scalar[output_dt]],
        input: UnsafePointer[Scalar[input_dt]],
        filter: UnsafePointer[Scalar[filter_dt]],
        n: Int,
        first_c_tile_in_group: Bool,
        c_tile_size: Int,
        f_tile_offset: Int,
        f_tile_size: Int,
        left_pad_impact_end: Int,
        right_pad_impact_start: Int,
    ):
        comptime simd_size = simd_width_of[Self.output_type]()

        for do in range(0, self.conv_shape.do()):
            var d = do * self.conv_shape.stride[0] - self.conv_shape.pad_d[0]

            for ho in range(
                self.partition.ho_or_howo_offset,
                self.partition.ho_or_howo_offset
                + self.partition.ho_or_howo_size,
            ):
                # fmt: off
                var h = ho * self.conv_shape.stride[1] - self.conv_shape.pad_h[0]
                # fmt: on

                # Points input to the start of the row.
                # Offset by -pad_w because s loop starts from the leftmost neighbor
                # in padding. The kernel skip the padding point and increment the
                # pointer.
                var input_base = input + self.conv_shape.c * (
                    -self.conv_shape.pad_w[0]
                    + self.conv_shape.w() * (h + self.conv_shape.h() * d)
                )

                # Points output to the start of the row
                var output_base = (
                    output
                    + self.conv_shape.f
                    * self.conv_shape.wo()
                    * (ho + self.conv_shape.ho() * do)
                )

                @parameter
                @always_inline
                fn work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                    conv3d_update_wo_tile[
                        height,
                        micro_kernel_width,
                        simd_size,
                        Self.filter_packed,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
                        elementwise_epilogue = Self.elementwise_epilogue,
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
                    )

                    input_base = input_base + (
                        height * self.conv_shape.stride[2] * self.conv_shape.c
                    )
                    output_base = output_base + height * self.conv_shape.f

                tile_middle_unswitch_boundaries[
                    work_fn,
                    VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.wo(),
                )
                # TODO(MOCO-2074): Suppress false positive unused var warning.
                _ = input_base
                _ = output_base

    fn _f_tile_loop_static[
        last_c_tile: Bool
    ](self, n: Int, c_tile_offset: Int, c_tile_size: Int):
        comptime assert Self.conv_attr_rank == Self.input_layout.rank() - 2
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        comptime F = Int(Self.output_layout.shape[3])  # NHWC
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_shape = get_micro_kernel_shape[
            Self.conv_attr_rank, WO, F, Self.conv_attr, simd_size
        ]()
        comptime micro_kernel_f_size = micro_kernel_shape[1] * simd_size

        var f_round_by_simd = (
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

        var residual = F - f_round_by_simd
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
        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        comptime H = Int(Self.input_layout.shape[1])  # NHWC
        comptime W = Int(Self.input_layout.shape[2])  # NHWC
        comptime C = Int(Self.input_layout.shape[3])  # NHWC
        comptime R = Int(Self.filter_layout.shape[1])  # FRSCf
        comptime S = Int(Self.filter_layout.shape[2])  # FRSCf
        comptime HO = Int(Self.output_layout.shape[1])  # NHWC
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        comptime F = Int(Self.output_layout.shape[3])  # NHWC

        var filter_base: UnsafePointer[Scalar[Self.filter_type]]

        comptime if Self.filter_packed:
            filter_base = self.filter.ptr + (
                f_tile_offset * C * R * S + c_tile_offset * micro_kernel_f_size
            )
        else:
            filter_base = self.filter.ptr + (c_tile_offset * F + f_tile_offset)

        var input_curr_image = self.input.ptr + n * W * H * C
        var output_curr_image = self.output.ptr + n * WO * HO * F
        var conv_attr_dyn = materialize[Self.conv_attr]()

        for ho in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            var h = ho * conv_attr_dyn.strides()[0] - conv_attr_dyn.pad_bottom()
            # Point to (n, 0, ho, c_tile_offset) mapped in input
            var input_base = input_curr_image + (
                c_tile_offset + C * (-conv_attr_dyn.pad_left() + W * h)
            )
            # Point to (n, 0, ho, f_tile_offset) mapped in input
            var output_base = output_curr_image + (f_tile_offset + F * WO * ho)

            # The entire row fits in one micro kernel.
            comptime if WO <= micro_kernel_height:
                self._inner_loops_static[
                    WO,
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
                comptime micro_kernel_height_lbound = min(
                    micro_kernel_height, WO // 2
                )
                comptime micro_kernel_height_rbound = min(
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
                input_base = input_base + (
                    micro_kernel_height_lbound * conv_attr_dyn.strides()[1] * C
                )
                output_base = output_base + micro_kernel_height_lbound * F

                # Update middle points if any. They aren't effected by padding.
                @__copy_capture(filter_base)
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
                    input_base = input_base + (
                        height * conv_attr_dyn.strides()[1] * C
                    )
                    output_base = output_base + height * F

                # Middle points are the points not updated by micro kernels
                # on left or right boundary
                comptime num_middle_points = WO - micro_kernel_height_lbound - micro_kernel_height_rbound
                # `tile` can't handle zero tile size.
                comptime micro_kernel_height_middle = num_middle_points % micro_kernel_height if num_middle_points % micro_kernel_height > 0 else 1
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
        input_base: UnsafePointer[
            Scalar[Self.input_type]
        ],  # points to (ho, wo) mapped in input
        filter_base: UnsafePointer[
            Scalar[Self.filter_type]
        ],  # point to filter in cf tile
        output_base: UnsafePointer[
            Scalar[Self.output_type]
        ],  # point to (ho, wo) in output
        f_tile_offset: Int,
        f_tile_size: Int,
        c_tile_offset: Int,
        c_tile_size: Int,
        n: Int,  # batch Index
        ho: Int,  # index in output height
        wo: Int,  # index in output width
    ):
        comptime if micro_kernel_height == 0:
            return

        comptime simd_size = simd_width_of[Self.output_type]()
        comptime micro_kernel_f_size = micro_kernel_width * simd_size

        comptime R = Int(Self.filter_layout.shape[1])  # FRSCf
        comptime S = Int(Self.filter_layout.shape[2])  # FRSCf
        comptime C = Int(Self.input_layout.shape[3])  # NHWC
        comptime s_stride_in_input = Self.conv_attr.dilations()[1] * C
        comptime wo_stride_in_input = Self.conv_attr.strides()[1] * C
        comptime filter_S_stride = C * micro_kernel_f_size
        comptime filter_F_stride = R * S * filter_S_stride

        comptime output_tile_layout = Layout.row_major(
            micro_kernel_height, micro_kernel_width * simd_size
        )
        var output_tile_stack = InlineArray[
            Scalar[Self.output_type], output_tile_layout.size()
        ](uninitialized=True)
        var output_micro_tile = LayoutTensor[
            Self.output_type,
            output_tile_layout,
        ](output_tile_stack)

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

        var acc = _Accumulator[
            Self.output_type, micro_kernel_height, micro_kernel_width, simd_size
        ]()
        acc.load(output_micro_tile.ptr, micro_kernel_width * simd_size)

        comptime W = Int(Self.input_layout.shape[2])  # NHWC
        comptime H = Int(Self.input_layout.shape[1])  # NHWC
        comptime WO = Int(Self.output_layout.shape[2])  # NHWC
        # Shift in input H when shifting 1 in filter stencil' R dimension.
        var h_shift = 0
        var conv_attr_dyn = materialize[Self.conv_attr]()
        # h index in input image
        var h = ho * conv_attr_dyn.strides()[0] - conv_attr_dyn.pad_bottom()
        for r in range(R):
            # Skip if row falls in padding.
            if h + h_shift < 0 or h + h_shift >= H:
                h_shift += conv_attr_dyn.dilations()[0]
                continue

            var input_ptr = input_base + h_shift * C * W
            var filter_ptr = filter_base + r * S * filter_S_stride
            var w = wo * conv_attr_dyn.strides()[1] - conv_attr_dyn.pad_left()

            comptime for s in range(S):
                # Adjustment of micro kernel height for left padding
                # The first left_adjust x micro_kernel_width registers are
                # ignored because they fall in padding.
                comptime left_adjust = max(
                    ceildiv(
                        Self.conv_attr.pad_left()
                        - s * Self.conv_attr.dilations()[1],
                        Self.conv_attr.strides()[1],
                    ),
                    0,
                ) if padded_left else 0
                # Adjustment of micro kernel height for right padding
                # The last left_adjust x micro_kernel_width registers are ignored.
                # fmt: off
                comptime right_adjust = max(
                    WO - 1 - (W - 1 + Self.conv_attr.pad_left() - s * Self.conv_attr.dilations()[1])
                             // Self.conv_attr.strides()[1],
                    0,
                ) if padded_right else 0
                # fmt: on

                # Revised calculation of tile_height to avoid cases of tile_height<=0.
                comptime tile_height = micro_kernel_height - left_adjust - right_adjust

                comptime if tile_height > 0:
                    self._accumulate[
                        micro_kernel_height,
                        micro_kernel_width,
                        simd_size,
                        has_residual,
                        # prefetch offset, default to 4 for now
                        4,
                        left_adjust,
                        left_adjust + tile_height,
                    ](
                        c_tile_size,
                        wo_stride_in_input,
                        input_ptr,
                        filter_ptr,
                        acc,
                    )

                filter_ptr = filter_ptr + filter_S_stride
                input_ptr = input_ptr + s_stride_in_input

            h_shift += conv_attr_dyn.dilations()[0]

        acc.store(output_micro_tile.ptr, micro_kernel_width * simd_size)
        # Store the micro tile
        self._store_output_micro_tile[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            has_residual,
        ](output_micro_tile, output_base)

        # Apply elmentwise epilogue to the
        comptime F = Int(Self.output_layout.shape[3])  # NHWC

        comptime if Self.elementwise_epilogue.__bool__() and last_c_tile.__bool__():
            comptime epilogue = Self.elementwise_epilogue.value()
            # If has residual, the tile size has been extended to a simd_size.
            # Here needs to use the real bound F.
            var f_tile_size_bounded = (
                F - f_tile_offset if has_residual else f_tile_size
            )
            for wo_idx in range(wo, wo + micro_kernel_height):
                epilogue(
                    Index(n, ho, wo_idx, f_tile_offset), f_tile_size_bounded
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
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    S: Int,
    mut acc: _Accumulator,
    input: UnsafePointer[Scalar[input_dt]],
    input_stride: Int,
    input_stride_to_nbr: Int,
    filter: UnsafePointer[Scalar[filter_dt]],
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
        input_dt: DType of input.
        filter_dt: DType of filter.

    Args:
        c_tile_size: Tile size in input channel.
        S: Filter window width.
        acc: Pointer to register tile accumulator.
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

        var input_ptr = input + s * input_stride_to_nbr
        var filter_ptr = filter + s * filter_stride_to_nbr

        # When effected by padding, we update 1 output point a time.
        # Skip this point's neighbor if it's in padding.
        comptime if effected_by_padding:
            comptime assert (
                micro_kernel_height == 1
            ), "The tile must only have 1 point when effected bypadding."
            var w_nbr = w + s * dilation
            if w_nbr < 0 or w_nbr >= W:
                continue

        # Accumulat in output registers.
        acc.accumulate[prefetch_offset=4, partial_load_b=partial_load_filter](
            c_tile_size,
            input_ptr,
            input_stride,
            filter_ptr,
            filter_stride,
            partial_load_filter_size,
        )


fn conv1d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[Scalar[output_dt]],
    input: UnsafePointer[Scalar[input_dt]],
    filter: UnsafePointer[Scalar[filter_dt]],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape,
    n: Int,
    wo: Int,
):
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride when s increments by 1
    var input_stride_by_s = conv_shape.dilation[0] * conv_shape.c

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # SCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var w = wo * conv_shape.stride[0] - conv_shape.pad_w[0]

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
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
        acc,
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
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elementwise epilogue if necessary
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()
        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(wo, wo + micro_kernel_height):
            epilogue(Index(n, wo_idx, f_tile_offset), f_tile_size_bounded)


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
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    RS: IndexList[2],
    mut acc: _Accumulator,
    input: UnsafePointer[Scalar[input_dt]],
    input_stride: Int,
    input_stride_to_nbr: IndexList[2],
    filter: UnsafePointer[Scalar[filter_dt]],
    filter_stride: Int,
    filter_stride_to_nbr: IndexList[2],
    partial_load_filter_size: Int,
    hw: IndexList[2],
    HW: IndexList[2],
    dilation: IndexList[2],
):
    for r in range(RS[0]):
        # Skip the row if it falls into padding.
        var h_nbr = hw[0] + r * dilation[0]
        if h_nbr < 0 or h_nbr >= HW[0]:
            continue

        var input_ptr = input + r * input_stride_to_nbr[0]
        var filter_ptr = filter + r * filter_stride_to_nbr[0]

        accumulate_wo_tile_1d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            RS[1],
            acc,
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


fn conv2d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[Scalar[output_dt]],
    input: UnsafePointer[Scalar[input_dt]],
    filter: UnsafePointer[Scalar[filter_dt]],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[2],
    n: Int,
    howo: IndexList[2],
):
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (R, S).
    var input_stride_by_s = conv_shape.dilation[1] * conv_shape.c
    var input_stride_by_r = (
        conv_shape.dilation[0] * conv_shape.w() * conv_shape.c
    )

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var hw = Index(
        howo[0] * conv_shape.stride[0] - conv_shape.pad_h[0],
        howo[1] * conv_shape.stride[1] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
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
        acc,
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
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    # if elementwise_epilogue_enabled and last_c_tile:
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()

        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(howo[1], howo[1] + micro_kernel_height):
            # elementwise_epilogue_fn[4](
            epilogue(
                Index(n, howo[0], wo_idx, f_tile_offset), f_tile_size_bounded
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
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    QRS: IndexList[3],
    mut acc: _Accumulator,
    input: UnsafePointer[Scalar[input_dt]],
    input_stride: Int,
    input_stride_to_nbr: IndexList[3],
    filter: UnsafePointer[Scalar[filter_dt]],
    filter_stride: Int,
    filter_stride_to_nbr: IndexList[3],
    partial_load_filter_size: Int,
    dhw: IndexList[3],
    DHW: IndexList[3],
    dilation: IndexList[3],
):
    for q in range(QRS[0]):
        var d_nbr = dhw[0] + q * dilation[0]
        if d_nbr < 0 or d_nbr >= DHW[0]:
            continue

        var input_ptr = input + q * input_stride_to_nbr[0]
        var filter_ptr = filter + q * filter_stride_to_nbr[0]

        accumulate_wo_tile_2d[
            micro_kernel_height,
            micro_kernel_width,
            simd_size,
            partial_load_filter,
            effected_by_padding,
        ](
            c_tile_size,
            Index(QRS[1], QRS[2]),
            acc,
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


fn conv3d_update_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    filter_packed: Bool,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    output: UnsafePointer[Scalar[output_dt]],
    input: UnsafePointer[Scalar[input_dt]],
    filter: UnsafePointer[Scalar[filter_dt]],
    first_c_tile: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[3],
    n: Int,
    dohowo: IndexList[3],
):
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Input stride to neighbor point in the filter window (Q, R, S).
    # fmt: off
    var input_stride_by_s = conv_shape.dilation[2] * conv_shape.c
    var input_stride_by_r = conv_shape.dilation[1] * conv_shape.w() * conv_shape.c
    var input_stride_by_q = conv_shape.dilation[0] * conv_shape.w() * conv_shape.h() * conv_shape.c
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s: Int

    comptime if filter_packed:  # FRSCf layout
        filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    else:  # RSCF layout
        filter_stride_by_s = conv_shape.c * conv_shape.f

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s
    var filter_stride_by_q = conv_shape.r() * filter_stride_by_r

    # Filter stride in F dimension in FRSCf
    var filter_stride = micro_kernel_f_size if filter_packed else conv_shape.f

    # Input coordinates
    var dhw = Index(
        dohowo[0] * conv_shape.stride[0] - conv_shape.pad_d[0],
        dohowo[1] * conv_shape.stride[1] - conv_shape.pad_h[0],
        dohowo[2] * conv_shape.stride[2] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    if first_c_tile:
        acc.init(0)
    else:
        acc.load[partial_load=has_residual](
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
        acc,
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
    acc.store[partial_store=has_residual](
        output,
        conv_shape.f,
        conv_shape.f_per_group() % simd_size,
    )

    # Apply elmentwise epilogue to the
    comptime if elementwise_epilogue.__bool__() and last_c_tile.__bool__():
        comptime epilogue = elementwise_epilogue.value()

        # If has residual, the tile size has been extended to a simd_size.
        # Here needs to use the real bound F.
        var f_tile_size_bounded: Int

        comptime if has_residual:
            f_tile_size_bounded = (
                conv_shape.f_per_group() - conv_shape.f_in_group(f_tile_offset)
            )
        else:
            f_tile_size_bounded = f_tile_size

        for wo_idx in range(dohowo[2], dohowo[2] + micro_kernel_height):
            epilogue(
                Index(n, dohowo[0], dohowo[1], wo_idx, f_tile_offset),
                f_tile_size_bounded,
            )


# ===----------------------------------------------------------------------=== #
# Direct Convolution Filter Packing                                            #
# ===----------------------------------------------------------------------=== #


@always_inline
fn pack_filter_shape_impl[
    filter_type: DType
](Q: Int, R: Int, S: Int, C: Int, F: Int, num_groups: Int) -> IndexList[6]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Args:
        Q: Original Q filter dimension.
        R: Original R filter dimension.
        S: Original S filter dimension.
        C: Original C filter dimension.
        F: Original F filter dimension.
        num_groups: Number of groups in the convolution.

    Returns:
        The output shape.
    """
    comptime simd_size = simd_width_of[filter_type]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    var F_per_group = F // num_groups

    var output_shape = IndexList[6]()
    output_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    output_shape[1] = Q
    output_shape[2] = R
    output_shape[3] = S
    output_shape[4] = C
    output_shape[5] = micro_kernel_f_size

    return output_shape


@always_inline
fn pack_conv_filter_shape[
    single_thread_blocking_override: Bool,
](filter: LayoutTensor, num_groups: Int) -> IndexList[filter.rank + 1]:
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

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime micro_kernel_width = get_direct_conv_micro_kernel_width()
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # Filter is in RSCF layout. The last dim is F no matter it's 1d, 2d, or 3d.
    var F = filter.dim[filter.rank - 1]()

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    var F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = IndexList[filter.rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.rank] = micro_kernel_f_size

    comptime for i in range(filter.rank - 1):
        packed_shape[i + 1] = filter.dim[i]()

    return packed_shape


@always_inline
fn pack_filter_shape[
    filter_type: DType,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
    strides: DimList,
    dilations: DimList,
    paddings: DimList,
    num_groups: Int,
    single_thread_blocking_override: Bool,
](filter: LayoutTensor) -> IndexList[filter.rank + 1]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Returns:
        The output shape.
    """
    comptime simd_size = simd_width_of[filter_type]()

    var F = filter.dim[filter.rank - 1]()  # RSCF layout

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    var F_per_group = F // num_groups

    comptime conv_attr = ConvInfoStatic[filter.rank - 2](
        pad=reorder_padding[filter.rank - 2](IntTuple(paddings)),
        stride=IntTuple(strides),
        dilation=IntTuple(dilations),
        num_groups=num_groups,
    )

    # TODO: extend to 1D/3D.
    comptime WO = output_shape.at[
        2
    ]().get() if filter.rank == 4 and output_shape.at[
        2
    ]().has_value() else UNKNOWN_VALUE
    comptime F_NHWC = output_shape.at[
        filter.rank - 1
    ]().get() if output_shape.at[
        filter.rank - 1
    ]().has_value() else UNKNOWN_VALUE
    comptime micro_kernel_shape = get_micro_kernel_shape[
        filter.rank - 2,
        WO,
        F_NHWC,
        conv_attr,
        simd_size,
    ]()

    comptime micro_kernel_width = micro_kernel_shape[1]
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # FSCf/FRSCf/FQRSCf layout.
    var packed_shape = IndexList[filter.rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.rank] = micro_kernel_f_size

    comptime for i in range(filter.rank - 1):
        packed_shape[i + 1] = filter.dim[i]()

    return packed_shape


@always_inline
fn _get_group_filter_base(
    packed_filter: LayoutTensor, group_idx: Int, f_per_group: Int
) -> UnsafePointer[
    Scalar[packed_filter.dtype], address_space = packed_filter.address_space
]:
    """Returns the pointer of the input group's start in the packed filter."""
    # Each group is zero padded to
    #     ceildiv(F_per_group, micro_kernel_width)
    #   * filter_window_size
    #   * C
    #   * micro_kernel_f_width
    # Output pointer points to the start of the current group.

    var micro_kernel_f_size = packed_filter.dim[packed_filter.rank - 1]()
    comptime rank = packed_filter.rank

    var filter_window_size = 1

    # The packed filter has layout e.x. FRSCf. The [1, rank-2) dims are filter
    # window sizes.
    comptime for i in range(rank - 3):
        filter_window_size *= packed_filter.dim[i + 1]()

    # Size of one group's packed filter.
    # fmt: off
    var group_size = ceildiv(f_per_group , micro_kernel_f_size) \
                   * filter_window_size * packed_filter.dim[rank-2]() \
                   * micro_kernel_f_size
    # fmt: on

    return packed_filter.ptr + group_idx * group_size


@always_inline
fn pack_filter(
    filter: LayoutTensor,
    packed_filter: LayoutTensor[mut=True, ...],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.
    Use the default micro kernel size for dynamic shapes."""

    comptime assert (
        filter.dtype == packed_filter.dtype
    ), "Type mismatch between the filter and the packed filter."

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime f_size_default = get_direct_conv_micro_kernel_width() * simd_size

    comptime if packed_filter.layout.shape[
        packed_filter.rank - 1
    ] != UNKNOWN_VALUE:
        comptime f_size = Int(
            packed_filter.layout.shape[packed_filter.rank - 1]
        )
        pack_filter[simd_size, f_size](filter, packed_filter, num_groups)
    else:
        pack_filter[simd_size, f_size_default](
            filter, packed_filter, num_groups
        )


@always_inline
fn pack_filter[
    simd_size: Int,
    micro_kernel_f_size: Int,  # 64
](
    filter: LayoutTensor,
    packed_filter: LayoutTensor[mut=True, ...],
    num_groups: Int,
):
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

    F is first broken down to segments of size micro_kernel_f_size, then the
    remainder is further divided by simd_size. The last residual elements if
    any is padded with zero to fill simd_size.
    """

    # The micro kernel should be multiple of simd_size in F dimension.
    comptime assert micro_kernel_f_size % simd_size == 0

    # The input simd size should not exceed filter type's simd size.
    # E.x. we can pack int8 filter based on int32 simd size.
    comptime assert simd_size <= simd_width_of[filter.dtype]()

    # Product of filter dims upto (rank - 1).
    var outer_dims_prod = 1

    comptime for i in range(filter.rank - 1):
        outer_dims_prod *= filter.dim[i]()

    var F = filter.dim[filter.rank - 1]()
    var F_per_group = F // num_groups

    _ = packed_filter.fill(0)

    # Each group is zero padded to
    #
    #                   ceildiv(F_per_group, micro_kernel_f_size)
    #                 * outer_dims_prod
    #                 * micro_kernel_f_size.
    #
    # There can be a remainder: F_per_group % micro_kernel_f_size. That's further
    # tiled by simd_size. The elements beyond the remainder is set to 0. E.x.
    # micro_kernel_f_size = 8, simd_size = 2, 21 values in total, follows
    #
    #                       |--------|--------|--|--|-0|00|

    for g in range(num_groups):
        var group_start = _get_group_filter_base(packed_filter, g, F_per_group)

        @always_inline
        @__copy_capture(group_start, F_per_group, F)
        @parameter
        fn pack[f_tile_size: Int](f_tile_start: Int):
            var packed_filter_ptr = group_start + f_tile_start * outer_dims_prod

            for row in range(outer_dims_prod):
                var filter_ptr = (
                    filter.ptr + row * F + g * F_per_group + f_tile_start
                )

                comptime for i in range(f_tile_size // simd_size):
                    packed_filter_ptr.store(
                        i * simd_size,
                        filter_ptr.load[width=simd_size](i * simd_size).cast[
                            packed_filter.dtype
                        ](),
                    )

                packed_filter_ptr += f_tile_size

        # If F % simd_size != 0, the following won't touch the remainder.
        tile[pack, VariadicList[Int](micro_kernel_f_size, simd_size)](
            0, F_per_group
        )

    # Check the remainder if any
    var F_round_by_simd = align_down(F_per_group, simd_size)
    var residual = F_per_group - F_round_by_simd

    # Handle the remainder if any
    if residual > 0:
        for g in range(num_groups):
            var group_start = _get_group_filter_base(
                packed_filter, g, F_per_group
            )
            var packed_filter_ptr = (
                group_start + F_round_by_simd * outer_dims_prod
            )

            for row in range(outer_dims_prod):
                var filter_ptr = (
                    filter.ptr + row * F + g * F_per_group + F_round_by_simd
                )

                # Load remainder elements and pad with zero to
                # to fill a simd vector.
                var filter_vec = partial_simd_load[simd_size](
                    filter_ptr, 0, residual, 0
                ).cast[packed_filter.dtype]()
                packed_filter_ptr.store(filter_vec)

                # Hence, packed filter is incremented by simd_size
                packed_filter_ptr = packed_filter_ptr + simd_size


@always_inline
fn conv_shape[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: LayoutTensor[
        input_type, address_space = AddressSpace.GENERIC, ...
    ],
    filter_buf: LayoutTensor[
        filter_type, address_space = AddressSpace.GENERIC, ...
    ],
    strides_buf: LayoutTensor[
        strides_type, address_space = AddressSpace.GENERIC, ...
    ],
    dilations_buf: LayoutTensor[
        dilations_type, address_space = AddressSpace.GENERIC, ...
    ],
    paddings_buf: LayoutTensor[
        paddings_type, address_space = AddressSpace.GENERIC, ...
    ],
    num_groups_scalar: Scalar,
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `conv` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.
        single_thread_blocking_override: If True, then the operation is run
          ssynchronouslysing a single thread.

    Args:
        input_buf: The input tensor.
        filter_buf: The filter tensor.
        strides_buf: The strides tensor.
        dilations_buf: The dilations tensor.
        paddings_buf: The paddings tensor.
        num_groups_scalar: The num_groups scalar.

    Returns:
        The output shape.
    """
    comptime assert strides_buf.rank == 1
    comptime assert dilations_buf.rank == 1
    comptime assert paddings_buf.rank == 1

    if input_buf.rank < 3:
        raise Error("[convolution] requires (input_rank >= 3)")
    if input_buf.rank != filter_buf.rank:
        raise Error("[convolution] requires (input_rank == filter_rank)")
    if (
        strides_buf.dim(0) != input_buf.rank - 2
        or dilations_buf.dim(0) != input_buf.rank - 2
    ):
        raise Error(
            "[convolution] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if paddings_buf.dim(0) != 2 * (input_buf.rank - 2):
        raise Error(
            "[convolution] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume
    # - input and output have layout [batch_size, ...spatial_dims..., input_channels]
    # - filter has layout [...spatial_dims..., filter_channels, output_channels]
    var batch_size = input_buf.dim(0)
    var input_channels = input_buf.dim(input_buf.rank - 1)
    var filter_channels = filter_buf.dim(input_buf.rank - 2)
    var output_channels = filter_buf.dim(input_buf.rank - 1)
    var num_groups = Int(num_groups_scalar)

    if input_channels != (num_groups * filter_channels):
        raise Error(
            "[convolution] requires (input_channels == num_groups *"
            " filter_channels)"
        )
    if (output_channels % num_groups) != 0:
        raise Error(
            "[convolution] output_channels must be divisible by num_groups"
        )

    var output_shape = IndexList[input_buf.rank]()
    output_shape[0] = batch_size
    output_shape[input_buf.rank - 1] = output_channels

    comptime for i in range(1, input_buf.rank - 1):
        var input_spatial_dim = input_buf.dim(i)
        var filter_spatial_dim = filter_buf.dim(i - 1)

        var output_spatial_dim = get_sliding_window_out_dim(
            input_spatial_dim,
            filter_spatial_dim,
            Int(dilations_buf[i - 1]),
            Int(strides_buf[i - 1]),
            Int(paddings_buf[2 * i - 2] + paddings_buf[2 * i - 1]),
        )

        if output_spatial_dim <= 0:
            raise Error("[convolution] output spatial dim must be positive")

        output_shape[i] = output_spatial_dim

    return output_shape


fn conv_nhwc_direct[
    conv_info_rank: Int,
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    conv_info_static: ConvInfoStatic[conv_info_rank],
    lambdas_have_fusion: Bool,
    elementwise_lambda: elementwise_simd_epilogue_type,
](
    input: LayoutTensor[input_type, input_layout],
    filter: LayoutTensor[filter_type, filter_layout],
    output: LayoutTensor[mut=True, output_type, output_layout],
    stride: IndexList[conv_info_rank],
    dilation: IndexList[conv_info_rank],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    comptime assert conv_info_rank == input_layout.rank() - 2
    comptime assert (
        input_type == filter_type and input_type == output_type
    ), "conv input/output/filter types must be the same."
    comptime assert (filter_packed and filter.rank == input.rank + 1) or (
        not filter_packed and filter.rank == input.rank
    ), "Filter and input ranks mismatch."

    @always_inline
    @parameter
    fn description_fn() -> String:
        return ";".join(
            Span(
                [
                    trace_arg("input", input.runtime_layout.shape.value),
                    trace_arg("filter", filter.runtime_layout.shape.value),
                    trace_arg("output", output.runtime_layout.shape.value),
                    "group=" + String(num_groups),
                    "stride=" + "x".join(Span([stride])),
                    "padding_h=" + "x".join(Span([pad_h])),
                    "padding_w=" + "x".join(Span([pad_w])),
                ]
            )
        )

    with Trace[TraceLevel.OP, target = StaticString("cpu")](
        "conv",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        var conv_shape = get_conv_shape[conv_info_rank, filter_packed](
            output,
            input,
            filter,
            stride,
            dilation,
            pad_d,
            pad_h,
            pad_w,
            num_groups,
        )

        # The closure updates a row segment of the output.
        @always_inline
        @parameter
        fn elementwise_epilogue[
            rank: Int
        ](coords: IndexList[rank], f_size: Int):
            comptime simd_size = simd_width_of[output_type]()

            @always_inline
            fn body[width: Int](idx: Int) unified {mut}:
                # Coordinates of the current index.
                var curr_coords = rebind[IndexList[input.rank]](coords)
                curr_coords[input.rank - 1] += idx

                var vec = output.load[width=width](curr_coords)
                elementwise_lambda(curr_coords, vec)

            vectorize[simd_size](f_size, body)

        ConvDirectNHWC[
            input_layout,
            filter_layout,
            output_layout,
            input.origin,
            filter.origin,
            output.origin,
            input_type,
            filter_type,
            output_type,
            filter_packed,
            conv_info_static,
            Optional[elementwise_epilogue_type](
                elementwise_epilogue
            ) if lambdas_have_fusion else None,
        ].run(
            output,
            input,
            filter,
            conv_shape,
        )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using cuDNN                                                  #
# ===----------------------------------------------------------------------=== #


fn conv2d_gpu_naive_nhwc_rscf[
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type],
](
    input: LayoutTensor[input_type, input_layout, MutAnyOrigin],
    filter: LayoutTensor[filter_type, filter_layout, MutAnyOrigin],
    output: LayoutTensor[output_type, output_layout, MutAnyOrigin],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
):
    var N = input.dim[0]()
    var H = input.dim[1]()
    var W = input.dim[2]()
    var C_in = input.dim[3]()  # channel_in
    var R = filter.dim[0]()
    var S = filter.dim[1]()
    var H_out = output.dim[1]()
    var W_out = output.dim[2]()
    var C_out = output.dim[3]()  # channel_out or #F
    var pad_h = padding[0]
    var pad_w = padding[1]
    var stride_h = stride[0]
    var stride_w = stride[1]
    var dil_h = dilation[0]
    var dil_w = dilation[1]

    var n = block_idx.z
    var h = block_idx.y * block_dim.y + thread_idx.y
    var w = block_idx.x * block_dim.x + thread_idx.x

    if h >= UInt(H_out) or w >= UInt(W_out):
        return

    for co in range(C_out):
        comptime accum_type = get_accum_type[output_type]()
        var value = Scalar[accum_type](0)
        for r in range(R):
            for s in range(S):
                var h_in = h * UInt(stride_h) - UInt(pad_h) + UInt(r * dil_h)
                var w_in = w * UInt(stride_w) - UInt(pad_w) + UInt(s * dil_w)
                if 0 <= Int(h_in) < H and 0 <= Int(w_in) < W:
                    for ci in range(C_in):
                        value += (
                            input.load[width=1](
                                IndexList[4](Int(n), Int(h_in), Int(w_in), ci)
                            ).cast[accum_type]()
                            * filter.load[width=1](
                                IndexList[4](r, s, ci, co)
                            ).cast[accum_type]()
                        )

        comptime if maybe_epilogue_func:
            comptime epilogue_func = maybe_epilogue_func.value()
            epilogue_func(
                IndexList[4](Int(n), Int(h), Int(w), co),
                value.cast[output_type](),
            )
        else:
            output.store(
                IndexList[4](Int(n), Int(h), Int(w), co),
                value.cast[output_type](),
            )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using cuDNN                                                  #
# ===----------------------------------------------------------------------=== #


@always_inline
fn check_cudnn_error(stat: cudnnStatus_t):
    if stat != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        print(stat)


struct CuDNNConvMeta(ImplicitlyCopyable, RegisterPassable):
    var ptr_handle: UnsafePointer[cudnnContext]
    var ptr_input_desc: UnsafePointer[cudnnTensorStruct]
    var ptr_filter_desc: UnsafePointer[cudnnFilterStruct]
    var ptr_conv_desc: UnsafePointer[cudnnConvolutionStruct]
    var ptr_output_desc: UnsafePointer[cudnnTensorStruct]

    fn __init__(out self) raises:
        self.ptr_handle = UnsafePointer[cudnnContext]()
        check_cudnn_error(cudnnCreate(UnsafePointer(to=self.ptr_handle)))

        self.ptr_input_desc = UnsafePointer[cudnnTensorStruct]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_input_desc))
        )

        self.ptr_filter_desc = UnsafePointer[cudnnFilterStruct]()
        check_cudnn_error(
            cudnnCreateFilterDescriptor(UnsafePointer(to=self.ptr_filter_desc))
        )

        self.ptr_conv_desc = UnsafePointer[cudnnConvolutionStruct]()
        check_cudnn_error(
            cudnnCreateConvolutionDescriptor(
                UnsafePointer(to=self.ptr_conv_desc)
            )
        )

        self.ptr_output_desc = UnsafePointer[cudnnTensorStruct]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_output_desc))
        )

    fn __del__(deinit self):
        try:
            check_cudnn_error(
                cudnnDestroyTensorDescriptor(self.ptr_output_desc)
            )
            check_cudnn_error(
                cudnnDestroyConvolutionDescriptor(self.ptr_conv_desc)
            )
            check_cudnn_error(
                cudnnDestroyFilterDescriptor(self.ptr_filter_desc)
            )
            check_cudnn_error(cudnnDestroyTensorDescriptor(self.ptr_input_desc))
            check_cudnn_error(cudnnDestroy(self.ptr_handle))
        except e:
            abort(String(e))


fn _get_cudnn_meta(ctx: DeviceContext) raises -> UnsafePointer[CuDNNConvMeta]:
    """Get the cuDNN metadata with proper device context management.

    If the metadata is not found for this device, create a new one and insert
    it into the global cache keyed by device ID.

    IMPORTANT: this function _must_ be called with `ctx`'s CUcontext active via:

    ```mojo
    from gpu.host import DeviceContext
    var ctx = DeviceContext()
    with ctx.push_context():
        ptr_meta = _get_cudnn_meta(ctx)
    ```

    This is to satisfy the stateful `cudnn*` API calls.

    Args:
        ctx: The device context.

    Returns:
        The cuDNN metadata.
    """
    # Key the cuDNN metadata cache on the device ID.
    var cache_key = "CUDA_CUDNN_META_CACHE" + String(ctx.id())

    # Get or create the per-device cache dictionary.
    if ptr_meta := _get_global_or_null(cache_key).bitcast[CuDNNConvMeta]():
        check_cudnn_error(
            cudnnSetStream(ptr_meta[].ptr_handle, CUDA(ctx.stream()))
        )
        return ptr_meta

    ptr_meta = UnsafePointer[CuDNNConvMeta].alloc(1)
    ptr_meta.init_pointee_move(CuDNNConvMeta())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        ptr_meta.bitcast[NoneType](),
    )

    return ptr_meta


fn get_cudnn_dtype[dtype: DType]() raises -> cudnnDataType_t:
    """Map Mojo DType to cuDNN data type.

    Support only floating point dtypes for now.
    """

    comptime if dtype == DType.float32:
        return cudnnDataType_t.CUDNN_DATA_FLOAT
    elif dtype == DType.float16:
        return cudnnDataType_t.CUDNN_DATA_HALF
    elif dtype == DType.bfloat16:
        return cudnnDataType_t.CUDNN_DATA_BFLOAT16
    else:
        raise Error("unsupported dtype", dtype, "for cuDNN")


fn _conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, ...],
    filter: LayoutTensor[filter_type, ...],
    output: LayoutTensor[output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    var ptr_meta = _get_cudnn_meta(ctx)

    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            ptr_meta[].ptr_input_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[input_type](),
            Int16(input.dim[0]()),
            Int16(input.dim[3]()),
            Int16(input.dim[1]()),
            Int16(input.dim[2]()),
        )
    )

    check_cudnn_error(
        cudnnSetFilter4dDescriptor(
            ptr_meta[].ptr_filter_desc,
            get_cudnn_dtype[filter_type](),
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            Int16(filter.dim[0]()),
            Int16(filter.dim[1]()),
            Int16(filter.dim[2]()),
            Int16(filter.dim[3]()),
        )
    )

    check_cudnn_error(
        cudnnSetConvolution2dDescriptor(
            ptr_meta[].ptr_conv_desc,
            Int16(padding[0]),
            Int16(padding[1]),
            Int16(stride[0]),
            Int16(stride[1]),
            Int16(dilation[0]),
            Int16(dilation[1]),
            cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
            # cuDNN 8+ requires float32 accumulation when the I/O tensors are
            # bfloat16.
            # Note that this is correct for float16, bfloat16, and float32 but
            # would have to be adjusted for other input dtypes, such as int8.
            cudnnDataType_t.CUDNN_DATA_FLOAT,
        )
    )

    check_cudnn_error(
        cudnnSetConvolutionGroupCount(
            ptr_meta[].ptr_conv_desc, Int16(num_groups)
        )
    )

    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            ptr_meta[].ptr_output_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[output_type](),
            Int16(output.dim[0]()),
            Int16(output.dim[3]()),
            Int16(output.dim[1]()),
            Int16(output.dim[2]()),
        )
    )

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    check_cudnn_error(
        cudnnSetConvolutionMathType(
            ptr_meta[].ptr_conv_desc,
            cudnnMathType_t.CUDNN_DEFAULT_MATH,  # this is the line that enables tf32
        )
    )
    # to disable tf32, run export NVIDIA_TF32_OVERRIDE=0 in the environment
    algo = (
        cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    )
    var workspace_size_var = 0
    var workspace_size_ptr = UnsafePointer(to=workspace_size_var)
    check_cudnn_error(
        cudnnGetConvolutionForwardWorkspaceSize(
            ptr_meta[].ptr_handle,
            ptr_meta[].ptr_input_desc,
            ptr_meta[].ptr_filter_desc,
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].ptr_output_desc,
            algo,
            workspace_size_ptr,
        )
    )
    var workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](
        workspace_size_var
    )
    check_cudnn_error(
        cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            rebind[OpaquePointer](input.ptr.bitcast[NoneType]()),
            ptr_meta[].ptr_filter_desc,
            rebind[OpaquePointer](filter.ptr.bitcast[NoneType]()),
            ptr_meta[].ptr_conv_desc,
            algo,
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            workspace_size_var,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            rebind[OpaquePointer](output.ptr.bitcast[NoneType]()),
        )
    )
    _ = workspace_buffer^


fn conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, ...],
    filter: LayoutTensor[filter_type, ...],
    output: LayoutTensor[output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    # Set `ctx`'s CUcontext as current to satisfy cudnn's stateful API.
    with ctx.push_context() as ctx:
        _conv_cudnn(
            input, filter, output, stride, dilation, padding, num_groups, ctx
        )


fn conv_gpu[
    conv_rank: Int,
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    filter_is_fcrs: Bool = False,
](
    input: LayoutTensor[input_type, input_layout, MutAnyOrigin],
    filter: LayoutTensor[filter_type, filter_layout, MutAnyOrigin],
    output: LayoutTensor[mut=True, output_type, output_layout, MutAnyOrigin],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    padding: IndexList[2 * conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime assert conv_rank == input.rank - 2

    var has_asymmetric_padding = False
    var pad_before = IndexList[conv_rank](0)

    comptime for i in range(conv_rank):
        pad_before[i] = padding[2 * i]
        var after = padding[2 * i + 1]
        if pad_before[i] != after:
            has_asymmetric_padding = True

    if has_asymmetric_padding:
        # Pre-pad on GPU so downstream kernels (including cuDNN) can assume symmetric padding.
        comptime full_rank = input_layout.rank()
        var paddings_tensor = LayoutTensor[
            DType.int, Layout(2 * full_rank), MutAnyOrigin
        ].stack_allocation()

        comptime for axis in range(full_rank):
            paddings_tensor[2 * axis] = 0
            paddings_tensor[2 * axis + 1] = 0

        comptime for i in range(conv_rank):
            comptime SIMDInt = Scalar[DType.int]

            var axis = i + 1  # skip batch axis
            paddings_tensor[2 * axis] = SIMDInt(padding[2 * i])  # before
            paddings_tensor[2 * axis + 1] = SIMDInt(padding[2 * i + 1])  # after

        var input_shape = rebind[IndexList[full_rank]](
            input.runtime_layout.shape.value.canonicalize()
        )
        var padded_shape = IndexList[full_rank]()

        comptime for axis in range(full_rank):
            var before = 0
            var after = 0
            if axis > 0 and axis < full_rank - 1:
                var spatial_idx = axis - 1
                before = padding[2 * spatial_idx]
                after = padding[2 * spatial_idx + 1]
            padded_shape[axis] = input_shape[axis] + before + after

        var padded_elements = padded_shape.flattened_length()
        var tmp_buffer = ctx.enqueue_create_buffer[input_type](padded_elements)
        var padded_device_buffer = tmp_buffer.unsafe_ptr()
        var zero_scalar = Scalar[input_type](0)

        pad_constant_gpu[full_rank, input_type, DType.int](
            padded_device_buffer,
            padded_shape,
            input.ptr,
            input_shape,
            paddings_tensor.ptr,
            zero_scalar,
            ctx,
        )

        var padded_input = LayoutTensor[
            input_type,
            Layout.row_major[full_rank](),
            MutAnyOrigin,
        ](
            padded_device_buffer,
            RuntimeLayout[Layout.row_major[full_rank]()].row_major(
                padded_shape
            ),
        )

        var zero_padding = IndexList[2 * conv_rank](0)

        conv_gpu[
            Layout.row_major[full_rank](),
            filter_layout,
            output_layout,
            input_type,
            filter_type,
            output_type,
            maybe_epilogue_func,
            filter_is_fcrs,
        ](
            padded_input,
            filter,
            output,
            stride,
            dilation,
            zero_padding,
            num_groups,
            ctx,
        )

        return

    # We can now use pad_before (which is now confirmed equal to pad_after) as
    # the symmetric padding.
    var symmetric_padding = pad_before

    comptime block_size = 16

    comptime conv_gpu_n = conv2d_gpu_naive_nhwc_rscf[
        input_layout,
        filter_layout,
        output_layout,
        input_type,
        filter_type,
        output_type,
        block_size,
        maybe_epilogue_func,
    ]

    comptime conv_gpu_3d = conv3d_gpu_naive_ndhwc_qrscf[
        input_layout,
        filter_layout,
        output_layout,
        input_type,
        filter_type,
        output_type,
        block_size,
        maybe_epilogue_func,
    ]
    var grid_dim_y = ceildiv(
        output.dim[1](), block_size
    )  # height for 2d and depth for 3d
    var grid_dim_z = input.dim[0]()  # n for both

    comptime if input.rank == 4:
        # Try SM100 structured conv2d on Blackwell GPUs (4-7x faster than cuDNN)
        comptime _is_sm100 = ctx.default_device_info == B200
        comptime _is_supported_dtype = input_type == DType.bfloat16

        comptime if _is_sm100 and _is_supported_dtype and not maybe_epilogue_func:
            from nn.conv_sm100.dispatch import (
                dispatch_sm100_conv2d,
            )

            # SM100 dispatch: stride=1, dilation=1, groups=1,
            # and channels aligned to 64 (TMA tile K alignment)
            var s = rebind[IndexList[2]](stride)
            var d = rebind[IndexList[2]](dilation)
            var in_c = input.dim[input.rank - 1]()
            var out_c = output.dim[output.rank - 1]()
            if (
                s[0] == 1
                and s[1] == 1
                and d[0] == 1
                and d[1] == 1
                and num_groups == 1
                and in_c % 64 == 0
                and out_c % 128 == 0
            ):
                dispatch_sm100_conv2d[
                    input_layout,
                    filter_layout,
                    output_layout,
                    input_type,
                    filter_type,
                    output_type,
                    filter_is_fcrs,
                ](
                    input,
                    filter,
                    output,
                    rebind[IndexList[2]](symmetric_padding),
                    ctx,
                )
                return

        # Fallback paths for non-SM100, unsupported dtypes, or constraints
        comptime if filter_is_fcrs:
            comptime if maybe_epilogue_func:
                comptime epilogue = maybe_epilogue_func.value()
                var output_tmp_data = ctx.enqueue_create_buffer[output_type](
                    output.size()
                )

                var output_tmp = output
                output_tmp.ptr = output_tmp_data.unsafe_ptr()

                conv_cudnn[input_type, filter_type, output_type](
                    LayoutTensor[
                        input_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        input.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            input.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    LayoutTensor[
                        filter_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        filter.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            filter.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    LayoutTensor[
                        output_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        output_tmp.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            output_tmp.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

                @parameter
                @__copy_capture(output_tmp)
                @always_inline
                fn epilogue_wrapper[
                    _width: Int, _rank: Int, alignment: Int = 1
                ](coords: IndexList[_rank]):
                    comptime align = align_of[SIMD[output_type, _width]]()
                    vec = output_tmp.load[width=_width](
                        rebind[IndexList[4]](coords)
                    )
                    epilogue(coords, vec)

                elementwise[
                    epilogue_wrapper, simd_width_of[output_type](), target="gpu"
                ](output.runtime_layout.shape.value.canonicalize(), ctx)

                _ = output_tmp_data^

            else:
                conv_cudnn[input_type, filter_type, output_type](
                    LayoutTensor[
                        input_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        input.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            input.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    LayoutTensor[
                        filter_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        filter.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            filter.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    LayoutTensor[
                        output_type, Layout.row_major[4](), MutAnyOrigin
                    ](
                        output.ptr,
                        RuntimeLayout[Layout.row_major[4]()].row_major(
                            output.runtime_layout.shape.value.canonicalize(),
                        ),
                    ),
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

        else:
            var grid_dim_x = ceildiv(
                output.dim[2](), block_size
            )  # w / block size for 2d
            ctx.enqueue_function[conv_gpu_n, conv_gpu_n](
                input,
                filter,
                output,
                stride,
                dilation,
                symmetric_padding,
                grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                block_dim=(block_size, block_size),
            )

    elif input.rank == 5:
        var grid_dim_x = ceildiv(
            output.dim[2]() * output.dim[3](), block_size
        )  # h * w / block size for 3d
        ctx.enqueue_function[conv_gpu_3d, conv_gpu_3d](
            input,
            filter,
            output,
            stride,
            dilation,
            symmetric_padding,
            grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
            block_dim=(block_size, block_size),
        )


fn conv3d_gpu_naive_ndhwc_qrscf[
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    block_size: Int,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type],
](
    input: LayoutTensor[input_type, input_layout, MutAnyOrigin],
    filter: LayoutTensor[filter_type, filter_layout, MutAnyOrigin],
    output: LayoutTensor[output_type, output_layout, MutAnyOrigin],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
):
    var N = input.dim[0]()
    var D = input.dim[1]()  # depth
    var H = input.dim[2]()
    var W = input.dim[3]()
    var C_in = input.dim[4]()  # channel_input

    var Q = filter.dim[0]()
    var R = filter.dim[1]()
    var S = filter.dim[2]()

    var D_out = output.dim[1]()  # depth
    var H_out = output.dim[2]()
    var W_out = output.dim[3]()
    var C_out = output.dim[4]()  # channel_output

    var pad_d = padding[0]
    var pad_h = padding[1]
    var pad_w = padding[2]

    var stride_d = stride[0]
    var stride_h = stride[1]
    var stride_w = stride[2]

    var dil_d = dilation[0]
    var dil_h = dilation[1]
    var dil_w = dilation[2]

    var n = block_idx.z  # batch dimension (unchanged)
    # calculate the linear thread id in x-dimension (width*height)
    var x_thread_id = block_idx.x * block_dim.x + thread_idx.x

    # map back to separate height and width
    var h_out_idx = x_thread_id // UInt(W_out)  # integer division to get height
    var w_out_idx = x_thread_id % UInt(W_out)  # modulo to get width

    # calculate depth from y-dimension
    var d_out_idx = block_idx.y * block_dim.y + thread_idx.y

    # bounds check
    if (
        n >= UInt(N)
        or d_out_idx >= UInt(D_out)
        or h_out_idx >= UInt(H_out)
        or w_out_idx >= UInt(W_out)
    ):
        return

    # ============= convolution =============
    for co in range(C_out):
        comptime accum_type = get_accum_type[output_type]()
        var value = Scalar[accum_type](0)

        for q in range(Q):
            for r in range(R):
                for s in range(S):
                    var d_in = Int(
                        d_out_idx * UInt(stride_d)
                        + UInt(q * dil_d)
                        - UInt(pad_d)
                    )
                    var h_in = Int(
                        h_out_idx * UInt(stride_h)
                        + UInt(r * dil_h)
                        - UInt(pad_h)
                    )
                    var w_in = Int(
                        w_out_idx * UInt(stride_w)
                        + UInt(s * dil_w)
                        - UInt(pad_w)
                    )

                    # check all input bounds bro
                    if 0 <= d_in < D and 0 <= h_in < H and 0 <= w_in < W:
                        for ci in range(C_in):
                            value += (
                                input.load[width=1](
                                    IndexList[5](Int(n), d_in, h_in, w_in, ci)
                                ).cast[accum_type]()
                                * filter.load[width=1](
                                    IndexList[5](q, r, s, ci, co)
                                ).cast[accum_type]()
                            )

        comptime if maybe_epilogue_func:
            comptime epilogue_func = maybe_epilogue_func.value()
            epilogue_func(
                IndexList[5](
                    Int(n), Int(d_out_idx), Int(h_out_idx), Int(w_out_idx), co
                ),
                value.cast[output_type](),
            )
        else:
            output.store(
                IndexList[5](
                    Int(n), Int(d_out_idx), Int(h_out_idx), Int(w_out_idx), co
                ),
                value.cast[output_type](),
            )
