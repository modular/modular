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

from std.collections import Optional
from std.math import align_down, ceildiv


from std.os import abort
from std.ffi import _get_global_or_null, external_call
from std.sys.info import align_of, simd_width_of

from _cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionFwdAlgoPerfStruct,
    cudnnConvolutionMode_t,
    cudnnConvolutionStruct,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnFindConvolutionForwardAlgorithmEx,
    cudnnGetConvolutionForwardWorkspaceSize,
    cudnnSetConvolution2dDescriptor,
    cudnnSetConvolutionGroupCount,
    cudnnSetConvolutionMathType,
    cudnnSetConvolutionNdDescriptor,
    cudnnGetConvolutionForwardAlgorithm_v7,
    cudnnConvolutionFwdAlgoPerf_t,
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
    cudnnSetFilterNdDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnSetTensorNdDescriptorEx,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorStruct,
)
from _miopen.miopen import (
    miopenCreate,
    miopenSetStream,
    miopenCreateTensorDescriptor,
    miopenSet4dTensorDescriptorEx,
    miopenCreateConvolutionDescriptor,
    miopenInitConvolutionNdDescriptor,
    miopenSetConvolutionGroupCount,
    miopenConvolutionForwardGetWorkSpaceSize,
    miopenFindConvolutionForwardAlgorithm,
    miopenConvolutionForward,
)
from _miopen.types import (
    Handle as MIOpenHandle,
    TensorDescriptor as MIOpenTensorDescriptor,
    ConvolutionDescriptor as MIOpenConvolutionDescriptor,
    DataType as MIOpenDataType,
    ConvolutionMode,
    ConvFwdAlgorithm,
    ConvAlgoPerf,
)
from _miopen.utils import check_error as check_miopen_error
from std.algorithm import (
    elementwise,
    sync_parallelize,
    tile,
    tile_middle_unswitch_boundaries,
    unswitch,
    vectorize,
)
from buffer.buffer import (
    partial_simd_load,
    partial_simd_store,
)
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA
from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from layout import (
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
    stack_allocation as tt_stack_allocation,
)
from linalg.accumulate import _Accumulator
from linalg.utils import partition_work
from std.runtime.asyncrt import parallelism_level
from std.runtime.tracing import Trace, TraceLevel, trace_arg

from std.sys import has_amd_gpu_accelerator, has_amd_rdna_gpu_accelerator
from std.gpu.host.info import _is_sm10x_gpu
from std.gpu.host._amdgpu_hip import HIP
from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type


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
from nn.shapes import get_sliding_window_out_dim
from nn.pad_gpu import pad_constant as pad_constant_gpu
from layout import lt_to_tt


@fieldwise_init
struct Naive2dConvolution[
    output_origin: Origin[mut=True],
    input_origin: Origin[mut=False],
    filter_origin: Origin[mut=False],
    //,
    output_type: DType,
    input_type: DType,
    filter_type: DType,
](ImplicitlyCopyable):
    """Struct wrapper for naive 2d convolution implementation."""

    # Input params.
    var output: UnsafePointer[Scalar[Self.output_type], Self.output_origin]
    var input: UnsafePointer[Scalar[Self.input_type], Self.input_origin]
    var filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin]
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
    def run(
        output: UnsafePointer[Scalar[Self.output_type], Self.output_origin],
        input: UnsafePointer[Scalar[Self.input_type], Self.input_origin],
        filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin],
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

    def __init__(
        out self,
        output: UnsafePointer[Scalar[Self.output_type], Self.output_origin],
        input: UnsafePointer[Scalar[Self.input_type], Self.input_origin],
        filter: UnsafePointer[Scalar[Self.filter_type], Self.filter_origin],
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

    def _outer_loop(self):
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

    def _compute_point(self, n: Int, do: Int, ho: Int, wo: Int, f: Int):
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
def _m_to_n_ho_wo_nhwc(m: Int, HO: Int, WO: Int) -> IndexList[3]:
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
    var n, rem = divmod(m, HO * WO)
    var ho, wo = divmod(rem, WO)
    return Index(n, ho, wo)


# Reduce helper when the input channel dimension is partitioned.
@always_inline
def _reduce_output[
    dtype: DType,
    //,
    simd_size: Int,
    elementwise_epilogue: Optional[elementwise_epilogue_type] = None,
](
    scratch: UnsafePointer[mut=False, Scalar[dtype], _],
    output: UnsafePointer[mut=True, Scalar[dtype], _],
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
    def reduce_task(tid: Int):
        # Use all threads in reduction.
        var reduce_range = partition_work(tid, num_threads, num_rows, 1)

        @always_inline
        def sum[width: Int](offset: Int) unified {mut}:
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
    conv_attr_rank: Int,
    input_origin: Origin[mut=False],
    filter_origin: Origin[mut=False],
    output_origin: Origin[mut=True],
    //,
    input_layout: Layout,
    filter_layout: Layout,
    output_layout: Layout,
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
    def run(
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

        # Safety: the scratch pointer below will alias the output_ptr, so cast to MutAnyOrigin
        # here to turn off the check.
        var output_ptr = output.ptr.unsafe_origin_cast[MutAnyOrigin]()
        var output_size = output.size()
        var scratch_size = num_partitions[1] * output_size
        if num_partitions[1] > 1:
            output_ptr = alloc[Scalar[Self.output_type]](scratch_size)
        # Wrap the pointer inside LayoutTensor so it can be properly captured by async closure.
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
        def task_func(task_id: Int):
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
                Self.output_type, Self.output_layout, AnyOrigin[mut=True]
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
                elementwise_epilogue=Self.elementwise_epilogue if input.rank
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

    def _batch_group_loop(self):
        """Loop over the batch and group dimensions. The two dimension are
        merged and partitioned for parallelism."""

        @always_inline
        @parameter
        def body[padded: Bool]():
            for ng in range(
                self.partition.ng_offset,
                self.partition.ng_offset + self.partition.ng_size,
            ):
                var n, g = divmod(ng, self.conv_shape.num_groups)
                self._c_tile_loop[padded](n, g, self.cf_tile_size[0])

        unswitch[body](self.conv_shape.padded())

    def _c_tile_loop[padded: Bool](self, n: Int, g: Int, tile_size: Int):
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
        def c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
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

    def _f_tile_loop[
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
        def f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
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
            [micro_kernel_f_size, simd_size],
            simd_size,
            f_tile_iteration,
        ](
            group_f_offset,
            group_f_end_align_simd,
            micro_kernel_f_size,
            simd_size,
            primary_cleanup_tile=simd_size,
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
    def is_new_c_accum(self, c_idx: Int) -> Bool:
        # returns true when processing first C in a group or first C in a C partition
        if self.conv_shape.num_groups > 1:
            return self.conv_shape.c_in_group(c_idx) == 0
        return c_idx == self.partition.c_offset

    def update_output_tile_no_padding[
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
        var input_base_offsets = TileTensor(
            input_base_stack.unsafe_ptr(), row_major[micro_kernel_height]()
        )

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
            Scalar[Self.filter_type], Self.filter_origin
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
                var ho, wo = divmod(m, self.conv_shape.wo())
                epilogue(
                    Index(
                        n,
                        ho,
                        wo,
                        f_tile_offset,
                    ),
                    f_tile_size_bounded,
                )

    @always_inline
    def _init_output_micro_tile[
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
            _,
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
    def _load_output_micro_tile[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
    ](
        self,
        output_base: UnsafePointer[Scalar[Self.output_type], ...],
        output_micro_tile: LayoutTensor[
            mut=True,
            Self.output_type,
            Layout.row_major(
                micro_kernel_height, micro_kernel_width * simd_size
            ),
            _,
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
    def _store_output_micro_tile[
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
            _,
        ],
        output_base: UnsafePointer[mut=True, Scalar[Self.output_type], ...],
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
    def _accumulate[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        simd_size: Int,
        has_residual: Bool,
        prefetch_offset: Int,
    ](
        self,
        input_base_offsets: TileTensor[DType.int32, ...],
        input_offset: Int,
        c_tile_size: Int,
        input: UnsafePointer[Scalar[Self.input_type], ...],
        filter: UnsafePointer[Scalar[Self.filter_type], ...],
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
            partial_load_b=has_residual and not Self.filter_packed,
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
    def _accumulate[
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
        input_base: UnsafePointer[Scalar[Self.input_type], ...],
        filter_base: UnsafePointer[Scalar[Self.filter_type], ...],
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
            partial_load_b=has_residual and not Self.filter_packed,
        ](
            c_tile_size,
            input_base,
            input_stride,
            filter_base,
            filter_stride,
            F % simd_size,
        )

    def output_space_flat_loop[
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
        def iteration[tile_size: Int](output_flat_coord: Int):
            @always_inline
            @parameter
            def body[c_fully_cached: Bool]():
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
        tile[iteration, [micro_kernel_height, 5, 4, 3, 2, 1]](
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        )

    def output_space_loop[
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
        # Use ImmutAnyOrigin to detach from self's filter_origin for aliasing.
        var filter_ptr: UnsafePointer[Scalar[Self.filter_type], ImmutAnyOrigin]

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
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
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
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
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
                # Safety: turn off mutable aliasing pointer check
                output_ptr.unsafe_origin_cast[AnyOrigin[mut=True]](),
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

    def output_space_loop_1d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
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
        def work_fn[height: Int, effected_by_padding: Bool](wo: Int):
            conv1d_update_wo_tile[
                height,
                micro_kernel_width,
                simd_size,
                Self.filter_packed,
                effected_by_padding,
                has_residual,
                last_c_tile,
                elementwise_epilogue=Self.elementwise_epilogue,
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
            work_fn, [micro_kernel_height, 5, 4, 3, 2, 1]
        ](
            0,
            left_pad_impact_end,
            right_pad_impact_start,
            self.conv_shape.wo(),
        )
        # TODO(MOCO-2074): Suppress false positive unused var warning.
        _ = input_base
        _ = output_base

    def output_space_loop_2d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
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
            def work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                conv2d_update_wo_tile[
                    height,
                    micro_kernel_width,
                    simd_size,
                    Self.filter_packed,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
                    elementwise_epilogue=Self.elementwise_epilogue,
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
                work_fn, [micro_kernel_height, 5, 4, 3, 2, 1]
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.wo(),
            )
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = input_base
            _ = output_base

    def output_space_loop_3d[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        has_residual: Bool,
        last_c_tile: Bool,
        output_dt: DType,
        input_dt: DType,
        filter_dt: DType,
    ](
        self,
        output: UnsafePointer[mut=True, Scalar[output_dt], ...],
        input: UnsafePointer[mut=False, Scalar[input_dt], ...],
        filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
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
                def work_fn[height: Int, effected_by_padding: Bool](wo: Int):
                    conv3d_update_wo_tile[
                        height,
                        micro_kernel_width,
                        simd_size,
                        Self.filter_packed,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
                        elementwise_epilogue=Self.elementwise_epilogue,
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
                    [micro_kernel_height, 5, 4, 3, 2, 1],
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.wo(),
                )
                # TODO(MOCO-2074): Suppress false positive unused var warning.
                _ = input_base
                _ = output_base

    def _f_tile_loop_static[
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
        def f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            self._h_loop_static[
                micro_kernel_shape[0],
                size // simd_size,
                False,
                last_c_tile,
            ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)

        tile[
            [micro_kernel_f_size, simd_size],
            simd_size,
            f_tile_iteration,
        ](
            self.partition.f_offset,
            f_round_by_simd,
            micro_kernel_f_size,
            simd_size,
            primary_cleanup_tile=simd_size,
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
    def _h_loop_static[
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

        var filter_base: UnsafePointer[Scalar[Self.filter_type], ImmutAnyOrigin]

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
                    # Safety: turn off mutable aliasing pointer check
                    output_base.unsafe_origin_cast[AnyOrigin[mut=True]](),
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
                    # Safety: turn off mutable aliasing pointer check
                    output_base.unsafe_origin_cast[AnyOrigin[mut=True]](),
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
                def update_middle[height: Int](wo: Int):
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
                        # Safety: turn off mutable aliasing pointer check
                        output_base.unsafe_origin_cast[AnyOrigin[mut=True]](),
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
                    [micro_kernel_height, micro_kernel_height_middle],
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
                    # Safety: turn off mutable aliasing pointer check
                    output_base.unsafe_origin_cast[AnyOrigin[mut=True]](),
                    f_tile_offset,
                    f_tile_size,
                    c_tile_offset,
                    c_tile_size,
                    n,
                    ho,
                    WO - micro_kernel_height_rbound,  # offset in wo dimension
                )

    @always_inline
    def _inner_loops_static[
        micro_kernel_height: Int,
        micro_kernel_width: Int,
        padded_left: Bool,
        padded_right: Bool,
        has_residual: Bool,
        last_c_tile: Bool,
    ](
        self,
        input_base: UnsafePointer[
            mut=False, Scalar[Self.input_type], ...
        ],  # points to (ho, wo) mapped in input
        filter_base: UnsafePointer[
            mut=False, Scalar[Self.filter_type], ...
        ],  # point to filter in cf tile
        output_base: UnsafePointer[
            mut=True, Scalar[Self.output_type], ...
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
def accumulate_wo_tile_1d[
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
    input: UnsafePointer[Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: Int,
    filter: UnsafePointer[Scalar[filter_dt], ...],
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


def conv1d_update_wo_tile[
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
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
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
def accumulate_wo_tile_2d[
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
    input: UnsafePointer[Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: IndexList[2],
    filter: UnsafePointer[Scalar[filter_dt], ...],
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


def conv2d_update_wo_tile[
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
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[Scalar[input_dt], ...],
    filter: UnsafePointer[Scalar[filter_dt], ...],
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
def accumulate_wo_tile_3d[
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
    input: UnsafePointer[Scalar[input_dt], ...],
    input_stride: Int,
    input_stride_to_nbr: IndexList[3],
    filter: UnsafePointer[Scalar[filter_dt], ...],
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


def conv3d_update_wo_tile[
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
    output: UnsafePointer[mut=True, Scalar[output_dt], ...],
    input: UnsafePointer[mut=False, Scalar[input_dt], ...],
    filter: UnsafePointer[mut=False, Scalar[filter_dt], ...],
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
def pack_filter_shape_impl[
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

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
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
def pack_conv_filter_shape(
    filter: TileTensor, num_groups: Int
) -> IndexList[filter.flat_rank + 1]:
    """
    Compute the output shape of convolution filter packing.

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
    var F = Int(filter.dim[filter.flat_rank - 1]())

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = IndexList[filter.flat_rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.flat_rank] = micro_kernel_f_size

    comptime for i in range(filter.flat_rank - 1):
        packed_shape[i + 1] = Int(filter.dim[i]())

    return packed_shape


@always_inline
def pack_filter_shape[
    filter_type: DType,
    input_shape: IntTuple,
    filter_shape: IntTuple,
    output_shape: IntTuple,
    strides: IntTuple,
    dilations: IntTuple,
    paddings: IntTuple,
    num_groups: Int,
](filter: TileTensor) -> IndexList[filter.flat_rank + 1]:
    """
    Compute the shape of packed filter. The packed layout is FRSCf.
    shape_ref should be allocated with size 5 outside this kernel.

    Returns:
        The output shape.
    """
    comptime simd_size = simd_width_of[filter_type]()

    var F = Int(filter.dim[filter.flat_rank - 1]())  # RSCF layout

    assert (
        F % num_groups == 0
    ), "number of filters F must be divisible by number of groups"
    var F_per_group = F // num_groups

    comptime conv_attr = ConvInfoStatic[filter.flat_rank - 2](
        pad=reorder_padding[filter.flat_rank - 2](IntTuple(paddings)),
        stride=IntTuple(strides),
        dilation=IntTuple(dilations),
        num_groups=num_groups,
    )

    # TODO: extend to 1D/3D.
    comptime WO = output_shape[
        2
    ].value() if filter.flat_rank == 4 and output_shape[
        2
    ].value() != UNKNOWN_VALUE else UNKNOWN_VALUE
    comptime F_NHWC = output_shape[
        filter.flat_rank - 1
    ].value() if output_shape[
        filter.flat_rank - 1
    ].value() != UNKNOWN_VALUE else UNKNOWN_VALUE
    comptime micro_kernel_shape = get_micro_kernel_shape[
        filter.flat_rank - 2,
        WO,
        F_NHWC,
        conv_attr,
        simd_size,
    ]()

    comptime micro_kernel_width = micro_kernel_shape[1]
    comptime micro_kernel_f_size = micro_kernel_width * simd_size

    # FSCf/FRSCf/FQRSCf layout.
    var packed_shape = IndexList[filter.flat_rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.flat_rank] = micro_kernel_f_size

    comptime for i in range(filter.flat_rank - 1):
        packed_shape[i + 1] = Int(filter.dim[i]())

    return packed_shape


@always_inline
def _get_group_filter_base(
    packed_filter: LayoutTensor, group_idx: Int, f_per_group: Int
) -> UnsafePointer[
    Scalar[packed_filter.dtype],
    packed_filter.origin,
    address_space=packed_filter.address_space,
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
def pack_filter(
    filter: TileTensor,
    packed_filter: TileTensor[mut=True, ...],
    num_groups: Int,
):
    """This packs the filter form RSCF to FRSCf.
    Use the default micro kernel size for dynamic shapes."""

    comptime assert (
        filter.dtype == packed_filter.dtype
    ), "Type mismatch between the filter and the packed filter."

    # Bridge to LayoutTensor for legacy Layout shape access and fill().
    var filter_lt = filter.to_layout_tensor()
    var packed_filter_lt = packed_filter.to_layout_tensor()

    comptime simd_size = simd_width_of[filter.dtype]()
    comptime f_size_default = get_direct_conv_micro_kernel_width() * simd_size

    comptime if packed_filter_lt.layout.shape[
        packed_filter_lt.rank - 1
    ] != UNKNOWN_VALUE:
        comptime f_size = Int(
            packed_filter_lt.layout.shape[packed_filter_lt.rank - 1]
        )
        pack_filter_lt[simd_size, f_size](
            filter_lt, packed_filter_lt, num_groups
        )
    else:
        pack_filter_lt[simd_size, f_size_default](
            filter_lt, packed_filter_lt, num_groups
        )


@always_inline
def pack_filter_lt[
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
        def pack[f_tile_size: Int](f_tile_start: Int):
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
        tile[pack, [micro_kernel_f_size, simd_size]](0, F_per_group)

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
def conv_shape[
    input_type: DType,
    filter_type: DType,
    strides_type: DType,
    dilations_type: DType,
    paddings_type: DType,
](
    input_buf: TileTensor[input_type, address_space=AddressSpace.GENERIC, ...],
    filter_buf: TileTensor[
        filter_type, address_space=AddressSpace.GENERIC, ...
    ],
    strides_buf: TileTensor[
        strides_type, address_space=AddressSpace.GENERIC, ...
    ],
    dilations_buf: TileTensor[
        dilations_type, address_space=AddressSpace.GENERIC, ...
    ],
    paddings_buf: TileTensor[
        paddings_type, address_space=AddressSpace.GENERIC, ...
    ],
    num_groups_scalar: Scalar,
) raises -> IndexList[input_buf.flat_rank]:
    """
    Compute the output shape of a `conv` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        filter_type: Type of the filter tensor.
        strides_type: Type of the strides tensor.
        dilations_type: Type of the dilations tensor.
        paddings_type: Type of the paddings tensor.

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
    # Bridge to LayoutTensor for runtime dim access.
    var input_lt = input_buf.to_layout_tensor()
    var filter_lt = filter_buf.to_layout_tensor()
    var strides_lt = strides_buf.to_layout_tensor()
    var dilations_lt = dilations_buf.to_layout_tensor()
    var paddings_lt = paddings_buf.to_layout_tensor()

    comptime assert strides_buf.flat_rank == 1
    comptime assert dilations_buf.flat_rank == 1
    comptime assert paddings_buf.flat_rank == 1

    if input_lt.rank < 3:
        raise Error("[convolution] requires (input_rank >= 3)")
    if input_lt.rank != filter_lt.rank:
        raise Error("[convolution] requires (input_rank == filter_rank)")
    if (
        strides_lt.dim(0) != input_lt.rank - 2
        or dilations_lt.dim(0) != input_lt.rank - 2
    ):
        raise Error(
            "[convolution] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if paddings_lt.dim(0) != 2 * (input_lt.rank - 2):
        raise Error(
            "[convolution] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume
    # - input and output have layout [batch_size, ...spatial_dims..., input_channels]
    # - filter has layout [...spatial_dims..., filter_channels, output_channels]
    var batch_size = input_lt.dim(0)
    var input_channels = input_lt.dim(input_lt.rank - 1)
    var filter_channels = filter_lt.dim(input_lt.rank - 2)
    var output_channels = filter_lt.dim(input_lt.rank - 1)
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

    var output_shape = IndexList[input_lt.rank]()
    output_shape[0] = batch_size
    output_shape[input_lt.rank - 1] = output_channels

    comptime for i in range(1, input_lt.rank - 1):
        var input_spatial_dim = input_lt.dim(i)
        var filter_spatial_dim = filter_lt.dim(i - 1)

        var output_spatial_dim = get_sliding_window_out_dim(
            input_spatial_dim,
            filter_spatial_dim,
            Int(dilations_lt[i - 1]),
            Int(strides_lt[i - 1]),
            Int(paddings_lt[2 * i - 2] + paddings_lt[2 * i - 1]),
        )

        if output_spatial_dim <= 0:
            raise Error("[convolution] output spatial dim must be positive")

        output_shape[i] = output_spatial_dim

    comptime assert (
        input_buf.flat_rank == input_lt.rank
    ), "TileTensor flat_rank must match LayoutTensor rank for rebind safety"
    return rebind[IndexList[input_buf.flat_rank]](output_shape)


def conv_nhwc_direct[
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
    input: TileTensor[input_type, address_space=AddressSpace.GENERIC, ...],
    filter: TileTensor[filter_type, address_space=AddressSpace.GENERIC, ...],
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    stride: IndexList[conv_info_rank],
    dilation: IndexList[conv_info_rank],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    # Construct LayoutTensors with explicit Layouts passed by the caller,
    # using the TileTensor's pointer and runtime shape. The Layouts must come
    # from ManagedTensorSlice.to_layout_tensor() (via the caller) so that
    # ConvDirectNHWC gets the same compile-time shape/stride info as it did
    # before the TileTensor migration.
    comptime ILT = LayoutTensor[input_type, input_layout, MutAnyOrigin]
    comptime FLT = LayoutTensor[filter_type, filter_layout, MutAnyOrigin]
    comptime OLT = LayoutTensor[output_type, output_layout, AnyOrigin[mut=True]]
    var input_lt = ILT(
        UnsafePointer[Scalar[input_type], MutAnyOrigin](
            unsafe_from_address=Int(input.ptr)
        ),
        ILT.RuntimeLayoutType.row_major(
            coord_to_index_list(input.layout.shape_coord()).cast[
                ILT.layout_int_type
            ]()
        ),
    )
    var filter_lt = FLT(
        UnsafePointer[Scalar[filter_type], MutAnyOrigin](
            unsafe_from_address=Int(filter.ptr)
        ),
        FLT.RuntimeLayoutType.row_major(
            coord_to_index_list(filter.layout.shape_coord()).cast[
                FLT.layout_int_type
            ]()
        ),
    )
    var output_lt = OLT(
        UnsafePointer[Scalar[output_type], AnyOrigin[mut=True]](
            unsafe_from_address=Int(output.ptr)
        ),
        OLT.RuntimeLayoutType.row_major(
            coord_to_index_list(output.layout.shape_coord()).cast[
                OLT.layout_int_type
            ]()
        ),
    )

    comptime assert conv_info_rank == input_layout.rank() - 2
    comptime assert (
        input_type == filter_type and input_type == output_type
    ), "conv input/output/filter types must be the same."
    comptime assert (filter_packed and filter_lt.rank == input_lt.rank + 1) or (
        not filter_packed and filter_lt.rank == input_lt.rank
    ), "Filter and input ranks mismatch."

    @always_inline
    @parameter
    def description_fn() -> String:
        return ";".join(
            Span(
                [
                    trace_arg("input", input_lt.runtime_layout.shape.value),
                    trace_arg("filter", filter_lt.runtime_layout.shape.value),
                    trace_arg("output", output_lt.runtime_layout.shape.value),
                    "group=" + String(num_groups),
                    "stride=" + "x".join(Span([stride])),
                    "padding_h=" + "x".join(Span([pad_h])),
                    "padding_w=" + "x".join(Span([pad_w])),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("cpu")](
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
        def elementwise_epilogue[
            rank: Int
        ](coords: IndexList[rank], f_size: Int):
            comptime simd_size = simd_width_of[output_type]()

            @always_inline
            def body[width: Int](idx: Int) unified {mut}:
                # Coordinates of the current index.
                var curr_coords = rebind[IndexList[input_lt.rank]](coords)
                curr_coords[input_lt.rank - 1] += idx

                var vec = output_lt.load[width=width](curr_coords)
                elementwise_lambda(curr_coords, vec)

            vectorize[simd_size](f_size, body)

        ConvDirectNHWC[
            input_layout,
            filter_layout,
            output_layout,
            input_type,
            filter_type,
            output_type,
            filter_packed,
            conv_info_static,
            Optional[elementwise_epilogue_type](
                elementwise_epilogue
            ) if lambdas_have_fusion else None,
        ].run(
            output_lt,
            input_lt,
            filter_lt,
            conv_shape,
        )


# ===----------------------------------------------------------------------=== #
# GPU Convolution using cuDNN                                                  #
# ===----------------------------------------------------------------------=== #


def conv2d_gpu_naive_nhwc_rscf[
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
    num_groups: Int,
):
    var N = input.dim[0]()
    var H = input.dim[1]()
    var W = input.dim[2]()
    var C_in = input.dim[3]()  # channel_in
    var R = filter.dim[0]()
    var S = filter.dim[1]()
    var C_per_group = filter.dim[2]()  # C_in / num_groups
    var H_out = output.dim[1]()
    var W_out = output.dim[2]()
    var C_out = output.dim[3]()  # channel_out or #F
    var F_per_group = C_out // num_groups
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
        var g = co // F_per_group
        var ci_base = g * C_per_group
        for r in range(R):
            for s in range(S):
                var h_in = h * UInt(stride_h) - UInt(pad_h) + UInt(r * dil_h)
                var w_in = w * UInt(stride_w) - UInt(pad_w) + UInt(s * dil_w)
                if 0 <= Int(h_in) < H and 0 <= Int(w_in) < W:
                    for ci in range(C_per_group):
                        value += (
                            input.load[width=1](
                                IndexList[4](
                                    Int(n), Int(h_in), Int(w_in), ci_base + ci
                                )
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
def check_cudnn_error(stat: cudnnStatus_t):
    if stat != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        print(stat)


struct CuDNNConvMeta(ImplicitlyCopyable, RegisterPassable):
    var ptr_handle: UnsafePointer[cudnnContext, AnyOrigin[mut=True]]
    var ptr_input_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]
    var ptr_filter_desc: UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]]
    var ptr_conv_desc: UnsafePointer[
        cudnnConvolutionStruct, AnyOrigin[mut=True]
    ]
    var ptr_output_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    def __init__(out self) raises:
        self.ptr_handle = UnsafePointer[cudnnContext, AnyOrigin[mut=True]]()
        check_cudnn_error(cudnnCreate(UnsafePointer(to=self.ptr_handle)))

        self.ptr_input_desc = UnsafePointer[
            cudnnTensorStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_input_desc))
        )

        self.ptr_filter_desc = UnsafePointer[
            cudnnFilterStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateFilterDescriptor(UnsafePointer(to=self.ptr_filter_desc))
        )

        self.ptr_conv_desc = UnsafePointer[
            cudnnConvolutionStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateConvolutionDescriptor(
                UnsafePointer(to=self.ptr_conv_desc)
            )
        )

        self.ptr_output_desc = UnsafePointer[
            cudnnTensorStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_output_desc))
        )

    def __del__(deinit self):
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


def _get_cudnn_meta(
    ctx: DeviceContext,
) raises -> UnsafePointer[CuDNNConvMeta, AnyOrigin[mut=True]]:
    """Get the cuDNN metadata with proper device context management.

    If the metadata is not found for this device, create a new one and insert
    it into the global cache keyed by device ID.

    IMPORTANT: this function _must_ be called with `ctx`'s CUcontext active via:

    ```mojo
    from std.gpu.host import DeviceContext
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
    if ptr_meta := _get_global_or_null(cache_key):
        var ptr = ptr_meta.unsafe_value().bitcast[CuDNNConvMeta]()
        check_cudnn_error(cudnnSetStream(ptr[].ptr_handle, CUDA(ctx.stream())))
        return ptr

    var new_ptr_meta = alloc[CuDNNConvMeta](1)
    new_ptr_meta.init_pointee_move(CuDNNConvMeta())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    return new_ptr_meta


def get_cudnn_dtype[dtype: DType]() raises -> cudnnDataType_t:
    """Map Mojo DType to cuDNN data type.

    Support only floating point dtypes for now.

    Raises:
        If the dtype is not supported by cuDNN.
    """

    comptime if dtype == DType.float32:
        return cudnnDataType_t.CUDNN_DATA_FLOAT
    elif dtype == DType.float16:
        return cudnnDataType_t.CUDNN_DATA_HALF
    elif dtype == DType.bfloat16:
        return cudnnDataType_t.CUDNN_DATA_BFLOAT16
    else:
        raise Error("unsupported dtype", dtype, "for cuDNN")


struct CachedCuDNNMetaNHWCFull(ImplicitlyCopyable):
    var ptr_handle: UnsafePointer[cudnnContext, AnyOrigin[mut=True]]
    var ptr_input_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]
    var ptr_filter_desc: UnsafePointer[cudnnFilterStruct, AnyOrigin[mut=True]]
    var ptr_conv_desc: UnsafePointer[
        cudnnConvolutionStruct, AnyOrigin[mut=True]
    ]
    var ptr_output_desc: UnsafePointer[cudnnTensorStruct, AnyOrigin[mut=True]]

    # Workspace size cache (actual buffer is allocated per-call via ctx)
    var workspace_size: Int

    # Algo Cache
    var best_algo: cudnnConvolutionFwdAlgo_t

    # Cache key fields
    var is_set: Bool
    var in_dtype: DType
    var in_: Tuple[Int, Int, Int, Int]
    var filt: Tuple[Int, Int, Int, Int]
    var out: Tuple[Int, Int, Int, Int]

    var pad: Tuple[Int, Int]
    var stride: Tuple[Int, Int]
    var dil: Tuple[Int, Int]

    def __init__(out self) raises:
        self.ptr_handle = UnsafePointer[cudnnContext, AnyOrigin[mut=True]]()
        check_cudnn_error(cudnnCreate(UnsafePointer(to=self.ptr_handle)))

        self.ptr_input_desc = UnsafePointer[
            cudnnTensorStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_input_desc))
        )

        self.ptr_filter_desc = UnsafePointer[
            cudnnFilterStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateFilterDescriptor(UnsafePointer(to=self.ptr_filter_desc))
        )

        self.ptr_conv_desc = UnsafePointer[
            cudnnConvolutionStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateConvolutionDescriptor(
                UnsafePointer(to=self.ptr_conv_desc)
            )
        )

        self.ptr_output_desc = UnsafePointer[
            cudnnTensorStruct, AnyOrigin[mut=True]
        ]()
        check_cudnn_error(
            cudnnCreateTensorDescriptor(UnsafePointer(to=self.ptr_output_desc))
        )

        self.workspace_size = 0
        self.best_algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        )

        self.is_set = False
        self.in_dtype = DType.invalid
        self.in_ = (0, 0, 0, 0)
        self.filt = (0, 0, 0, 0)
        self.out = (0, 0, 0, 0)
        self.pad = (0, 0)
        self.stride = (0, 0)
        self.dil = (0, 0)


def _get_cached_cudnn_meta_nhwc_full(
    ctx: DeviceContext,
) raises -> UnsafePointer[CachedCuDNNMetaNHWCFull, AnyOrigin[mut=True]]:
    var cache_key = "CUDA_CUDNN_CACHED_META_NHWC_FULL_" + String(ctx.id())

    if ptr_meta := _get_global_or_null(cache_key):
        var ptr = ptr_meta.unsafe_value().bitcast[CachedCuDNNMetaNHWCFull]()
        check_cudnn_error(cudnnSetStream(ptr[].ptr_handle, CUDA(ctx.stream())))
        return ptr

    var new_ptr_meta = alloc[CachedCuDNNMetaNHWCFull](1)
    new_ptr_meta.init_pointee_move(CachedCuDNNMetaNHWCFull())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    check_cudnn_error(
        cudnnSetStream(new_ptr_meta[].ptr_handle, CUDA(ctx.stream()))
    )

    return new_ptr_meta


def _conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[output_type, ...],
    stride_list: IndexList[2],
    dilation_list: IndexList[2],
    padding_list: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    # Use the optimized cached metadata implementation
    var ptr_meta = _get_cached_cudnn_meta_nhwc_full(ctx)

    # Input shape: NHWC
    var in_: Tuple[Int, Int, Int, Int] = (
        Int(input.dim[0]()),
        Int(input.dim[1]()),
        Int(input.dim[2]()),
        Int(input.dim[3]()),
    )

    # Filter shape: FCRS (K, C, R, S)
    var filt: Tuple[Int, Int, Int, Int] = (
        Int(filter.dim[0]()),
        Int(filter.dim[1]()),
        Int(filter.dim[2]()),
        Int(filter.dim[3]()),
    )

    # Output shape: NHWC
    var out: Tuple[Int, Int, Int, Int] = (
        Int(output.dim[0]()),
        Int(output.dim[1]()),
        Int(output.dim[2]()),
        Int(output.dim[3]()),
    )

    var pad: Tuple[Int, Int] = (padding_list[0], padding_list[1])
    var stride: Tuple[Int, Int] = (stride_list[0], stride_list[1])
    var dil: Tuple[Int, Int] = (dilation_list[0], dilation_list[1])

    var params_match = ptr_meta[].is_set

    if params_match:
        if ptr_meta[].in_dtype != input_type:
            params_match = False
        elif ptr_meta[].in_ != in_:
            params_match = False
        elif ptr_meta[].filt != filt:
            params_match = False
        elif ptr_meta[].out != out:
            params_match = False
        elif ptr_meta[].pad != pad:
            params_match = False
        elif ptr_meta[].stride != stride:
            params_match = False
        elif ptr_meta[].dil != dil:
            params_match = False

    if not params_match:
        # Update Input Descriptor (NHWC)
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                ptr_meta[].ptr_input_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[input_type](),
                Int16(in_[0]),
                Int16(in_[3]),
                Int16(in_[1]),
                Int16(in_[2]),
            )
        )

        # Update Filter Descriptor (NCHW for filter)
        check_cudnn_error(
            cudnnSetFilter4dDescriptor(
                ptr_meta[].ptr_filter_desc,
                get_cudnn_dtype[filter_type](),
                cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
                Int16(filt[0]),
                Int16(filt[1]),
                Int16(filt[2]),
                Int16(filt[3]),
            )
        )

        # Update Conv Descriptor
        check_cudnn_error(
            cudnnSetConvolution2dDescriptor(
                ptr_meta[].ptr_conv_desc,
                Int16(pad[0]),
                Int16(pad[1]),
                Int16(stride[0]),
                Int16(stride[1]),
                Int16(dil[0]),
                Int16(dil[1]),
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
            )
        )

        check_cudnn_error(
            cudnnSetConvolutionGroupCount(
                ptr_meta[].ptr_conv_desc, Int16(num_groups)
            )
        )

        # Update Output Descriptor (NHWC)
        check_cudnn_error(
            cudnnSetTensor4dDescriptor(
                ptr_meta[].ptr_output_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[output_type](),
                Int16(out[0]),
                Int16(out[3]),
                Int16(out[1]),
                Int16(out[2]),
            )
        )

        # Use ALLOW_CONVERSION only for half-precision types to enable tensor
        # core acceleration. For float32, use DEFAULT_MATH to avoid incorrect
        # results on some GPU architectures (e.g., B200).
        comptime if input_type == DType.float16 or input_type == DType.bfloat16:
            check_cudnn_error(
                cudnnSetConvolutionMathType(
                    ptr_meta[].ptr_conv_desc,
                    cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
                )
            )

        # Algorithm Autotuning.
        # The Mojo binding cudnnConvolutionFwdAlgoPerfStruct has incorrect
        # layout (Int8 enums vs C's 4-byte int enums). We bypass it by
        # allocating a raw 48-byte buffer matching the C ABI layout and
        # reading the algo Int32 at offset 0.
        var perf_buf = alloc[UInt8](48)
        var requested_count: Int16 = 1
        var returned_count: Int16 = 0

        check_cudnn_error(
            cudnnGetConvolutionForwardAlgorithm_v7(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                requested_count,
                UnsafePointer(to=returned_count),
                perf_buf.bitcast[cudnnConvolutionFwdAlgoPerf_t](),
            )
        )

        if returned_count > 0:
            # Read algo enum (C int32) at byte offset 0
            var algo_val = perf_buf.bitcast[Int32]()[]
            ptr_meta[].best_algo = cudnnConvolutionFwdAlgo_t(Int(algo_val))
        else:
            ptr_meta[].best_algo = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            )

        perf_buf.free()

        # Query workspace size
        var ws_size: Int = 0
        check_cudnn_error(
            cudnnGetConvolutionForwardWorkspaceSize(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                ptr_meta[].best_algo,
                UnsafePointer(to=ws_size),
            )
        )
        ptr_meta[].workspace_size = ws_size

        # Update Cache State
        ptr_meta[].is_set = True
        ptr_meta[].in_dtype = input_type
        ptr_meta[].in_ = in_
        ptr_meta[].filt = filt
        ptr_meta[].out = out
        ptr_meta[].pad = pad
        ptr_meta[].stride = stride
        ptr_meta[].dil = dil

    # Allocate workspace per-call using ctx (runtime-managed buffer)
    var workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](
        ptr_meta[].workspace_size
    )

    var alpha: Float32 = 1.0
    var beta: Float32 = 0.0

    check_cudnn_error(
        cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].best_algo,
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            ptr_meta[].workspace_size,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
        )
    )
    _ = workspace_buffer^


def conv_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[output_type, ...],
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


# ===----------------------------------------------------------------------=== #
# GPU Convolution using MIOpen (AMD)                                           #
# ===----------------------------------------------------------------------=== #


struct CachedMIOpenMeta(Movable):
    var handle: MIOpenHandle
    var input_desc: MIOpenTensorDescriptor
    var filter_desc: MIOpenTensorDescriptor
    var output_desc: MIOpenTensorDescriptor
    var conv_desc: MIOpenConvolutionDescriptor

    # Legacy API: cached algorithm and workspace size from FindAlgorithm
    var algo: ConvFwdAlgorithm
    var workspace_size: UInt64

    # Cache key fields
    var is_set: Bool
    var in_dtype: DType
    var in_: Tuple[Int, Int, Int, Int]
    var filt: Tuple[Int, Int, Int, Int]
    var out: Tuple[Int, Int, Int, Int]
    var pad: Tuple[Int, Int]
    var stride: Tuple[Int, Int]
    var dil: Tuple[Int, Int]
    var filter_is_fcrs: Bool

    def __init__(out self) raises:
        self.handle = MIOpenHandle()
        check_miopen_error(
            miopenCreate(UnsafePointer(to=self.handle).bitcast[NoneType]())
        )

        self.input_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.input_desc).bitcast[NoneType]()
            )
        )

        self.filter_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.filter_desc).bitcast[NoneType]()
            )
        )

        self.output_desc = MIOpenTensorDescriptor()
        check_miopen_error(
            miopenCreateTensorDescriptor(
                UnsafePointer(to=self.output_desc).bitcast[NoneType]()
            )
        )

        self.conv_desc = MIOpenConvolutionDescriptor()
        check_miopen_error(
            miopenCreateConvolutionDescriptor(
                UnsafePointer(to=self.conv_desc).bitcast[NoneType]()
            )
        )

        self.algo = ConvFwdAlgorithm(0)
        self.workspace_size = 0

        self.is_set = False
        self.in_dtype = DType.invalid
        self.in_ = (0, 0, 0, 0)
        self.filt = (0, 0, 0, 0)
        self.out = (0, 0, 0, 0)
        self.pad = (0, 0)
        self.stride = (0, 0)
        self.dil = (0, 0)
        self.filter_is_fcrs = False


def _get_cached_miopen_meta(
    ctx: DeviceContext,
) raises -> UnsafePointer[CachedMIOpenMeta, AnyOrigin[mut=True]]:
    var cache_key = "MIOPEN_CACHED_META_" + String(ctx.id())

    if ptr_meta := _get_global_or_null(cache_key):
        var ptr = ptr_meta.unsafe_value().bitcast[CachedMIOpenMeta]()
        check_miopen_error(miopenSetStream(ptr[].handle, HIP(ctx.stream())))
        return ptr

    var new_ptr_meta = alloc[CachedMIOpenMeta](1)
    new_ptr_meta.init_pointee_move(CachedMIOpenMeta())

    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(cache_key),
        new_ptr_meta.bitcast[NoneType](),
    )

    check_miopen_error(
        miopenSetStream(new_ptr_meta[].handle, HIP(ctx.stream()))
    )

    return new_ptr_meta


def _miopen_algo_find_and_forward[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    ptr_meta: UnsafePointer[CachedMIOpenMeta, AnyOrigin[mut=True]],
    input: TileTensor[input_type, ...],
    filter_ptr: OpaquePointer,
    filt: Tuple[Int, Int, Int, Int],
    output: TileTensor[output_type, ...],
    in_shape: Tuple[Int, Int, Int, Int],
    out_shape: Tuple[Int, Int, Int, Int],
    pad: Tuple[Int, Int],
    stride: Tuple[Int, Int],
    dil: Tuple[Int, Int],
    num_groups: Int,
    params_match: Bool,
    filter_is_fcrs: Bool,
    ctx: DeviceContext,
) raises:
    """Set up descriptors, find algorithm, and run forward convolution.

    Uses NHWC strides for input/output and appropriate strides for
    the filter physical layout, avoiding NHWC↔NCHW transposes.
    Dimensions are passed in NCHW logical order with strides describing
    the actual physical memory layout (following PyTorch's approach).
    """
    # in_shape is NHWC: (N, H, W, C)
    var in_N = in_shape[0]
    var in_H = in_shape[1]
    var in_W = in_shape[2]
    var in_C = in_shape[3]

    var out_N = out_shape[0]
    var out_H = out_shape[1]
    var out_W = out_shape[2]
    var out_C = out_shape[3]

    if not params_match:
        # Input descriptor: NCHW logical dims with NHWC physical strides
        check_miopen_error(
            miopenSet4dTensorDescriptorEx(
                ptr_meta[].input_desc,
                MIOpenDataType(input_type),
                Int32(in_N),
                Int32(in_C),
                Int32(in_H),
                Int32(in_W),
                Int32(in_H * in_W * in_C),  # N stride
                Int32(1),  # C stride (channels-last)
                Int32(in_W * in_C),  # H stride
                Int32(in_C),  # W stride
            )
        )

        # Filter descriptor: NHWC strides (matching input/output layout).
        # Filter data must be in FRSC physical layout for NHWC strides.
        var f_F: Int
        var f_C: Int
        var f_R: Int
        var f_S: Int
        if filter_is_fcrs:
            f_F = filt[0]
            f_C = filt[1]
            f_R = filt[2]
            f_S = filt[3]
        else:
            f_F = filt[3]
            f_C = filt[2]
            f_R = filt[0]
            f_S = filt[1]
        check_miopen_error(
            miopenSet4dTensorDescriptorEx(
                ptr_meta[].filter_desc,
                MIOpenDataType(filter_type),
                Int32(f_F),
                Int32(f_C),
                Int32(f_R),
                Int32(f_S),
                Int32(f_R * f_S * f_C),  # F stride (NHWC: channels-last)
                Int32(1),  # C stride
                Int32(f_S * f_C),  # R stride
                Int32(f_C),  # S stride
            )
        )

        # Output descriptor: NCHW logical dims with NHWC physical strides
        check_miopen_error(
            miopenSet4dTensorDescriptorEx(
                ptr_meta[].output_desc,
                MIOpenDataType(output_type),
                Int32(out_N),
                Int32(out_C),
                Int32(out_H),
                Int32(out_W),
                Int32(out_H * out_W * out_C),  # N stride
                Int32(1),  # C stride (channels-last)
                Int32(out_W * out_C),  # H stride
                Int32(out_C),  # W stride
            )
        )

        # Convolution descriptor (stack-allocated to avoid leak on raise)
        var conv_pad: InlineArray[Int32, 2] = [Int32(pad[0]), Int32(pad[1])]
        var conv_stride: InlineArray[Int32, 2] = [
            Int32(stride[0]),
            Int32(stride[1]),
        ]
        var conv_dilation: InlineArray[Int32, 2] = [
            Int32(dil[0]),
            Int32(dil[1]),
        ]
        check_miopen_error(
            miopenInitConvolutionNdDescriptor(
                ptr_meta[].conv_desc,
                Int32(2),
                conv_pad.unsafe_ptr().bitcast[NoneType](),
                conv_stride.unsafe_ptr().bitcast[NoneType](),
                conv_dilation.unsafe_ptr().bitcast[NoneType](),
                ConvolutionMode.CONVOLUTION,
            )
        )

        if num_groups > 1:
            check_miopen_error(
                miopenSetConvolutionGroupCount(
                    ptr_meta[].conv_desc, Int32(num_groups)
                )
            )

        # Get workspace size
        var ws_size: UInt64 = 0
        check_miopen_error(
            miopenConvolutionForwardGetWorkSpaceSize(
                ptr_meta[].handle,
                ptr_meta[].filter_desc,
                ptr_meta[].input_desc,
                ptr_meta[].conv_desc,
                ptr_meta[].output_desc,
                UnsafePointer(to=ws_size).bitcast[NoneType](),
            )
        )

        var find_ws = ctx.enqueue_create_buffer[DType.uint8](Int(ws_size))

        var perf = ConvAlgoPerf()
        var returned_count: Int32 = 0
        check_miopen_error(
            miopenFindConvolutionForwardAlgorithm(
                ptr_meta[].handle,
                ptr_meta[].input_desc,
                input.ptr.bitcast[NoneType](),
                ptr_meta[].filter_desc,
                filter_ptr.bitcast[NoneType](),
                ptr_meta[].conv_desc,
                ptr_meta[].output_desc,
                output.ptr.bitcast[NoneType](),
                Int32(1),
                UnsafePointer(to=returned_count).bitcast[NoneType](),
                UnsafePointer(to=perf).bitcast[NoneType](),
                find_ws.unsafe_ptr().bitcast[NoneType](),
                ws_size,
                False,  # non-exhaustive search
            )
        )
        _ = find_ws^

        if returned_count == 0:
            raise Error("MIOpen: no algorithm found for convolution")

        ptr_meta[].algo = ConvFwdAlgorithm(perf.fwd_algo)
        ptr_meta[].workspace_size = perf.memory

        # Update cache state
        ptr_meta[].is_set = True
        ptr_meta[].in_dtype = input_type
        ptr_meta[].in_ = in_shape
        ptr_meta[].filt = filt
        ptr_meta[].out = out_shape
        ptr_meta[].pad = pad
        ptr_meta[].stride = stride
        ptr_meta[].dil = dil
        ptr_meta[].filter_is_fcrs = filter_is_fcrs

    # Run forward convolution
    var fwd_ws_size = Int(ptr_meta[].workspace_size)
    var workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](fwd_ws_size)

    var alpha = Float32(1.0)
    var beta = Float32(0.0)
    check_miopen_error(
        miopenConvolutionForward(
            ptr_meta[].handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].filter_desc,
            filter_ptr.bitcast[NoneType](),
            ptr_meta[].conv_desc,
            ptr_meta[].algo,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].output_desc,
            output.ptr.bitcast[NoneType](),
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            ptr_meta[].workspace_size,
        )
    )

    _ = workspace_buffer^


def _conv_miopen[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool = False,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[output_type, ...],
    stride_list: IndexList[2],
    dilation_list: IndexList[2],
    padding_list: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    var ptr_meta = _get_cached_miopen_meta(ctx)

    # Bridge to LayoutTensor for filter transpose closures that capture it.
    var filter_lt = filter.to_layout_tensor()

    # Input shape: NHWC
    var in_: Tuple[Int, Int, Int, Int] = (
        Int(input.dim[0]()),
        Int(input.dim[1]()),
        Int(input.dim[2]()),
        Int(input.dim[3]()),
    )

    # Filter shape depends on filter_is_fcrs
    var filt: Tuple[Int, Int, Int, Int] = (
        Int(filter.dim[0]()),
        Int(filter.dim[1]()),
        Int(filter.dim[2]()),
        Int(filter.dim[3]()),
    )

    # Output shape: NHWC
    var out: Tuple[Int, Int, Int, Int] = (
        Int(output.dim[0]()),
        Int(output.dim[1]()),
        Int(output.dim[2]()),
        Int(output.dim[3]()),
    )

    var pad: Tuple[Int, Int] = (padding_list[0], padding_list[1])
    var stride: Tuple[Int, Int] = (stride_list[0], stride_list[1])
    var dil: Tuple[Int, Int] = (dilation_list[0], dilation_list[1])

    var params_match = ptr_meta[].is_set

    if params_match:
        if ptr_meta[].in_dtype != input_type:
            params_match = False
        elif ptr_meta[].in_ != in_:
            params_match = False
        elif ptr_meta[].filt != filt:
            params_match = False
        elif ptr_meta[].out != out:
            params_match = False
        elif ptr_meta[].pad != pad:
            params_match = False
        elif ptr_meta[].stride != stride:
            params_match = False
        elif ptr_meta[].dil != dil:
            params_match = False
        elif ptr_meta[].filter_is_fcrs != filter_is_fcrs:
            params_match = False

    # MIOpen needs all tensors in the same layout. Since input/output use
    # NHWC strides, the filter must also be NHWC (FRSC physical layout).
    # Transpose RSCF→FRSC or FCRS→FRSC on GPU. This is a small weight
    # tensor — the cost is negligible compared to the conv itself.
    var filter_size = filter.num_elements()
    var filter_frsc_buf = ctx.enqueue_create_buffer[filter_type](filter_size)
    var filter_frsc_ptr = filter_frsc_buf.unsafe_ptr()

    comptime if filter_is_fcrs:
        # FCRS [F,C,R,S] -> FRSC [F,R,S,C]
        var F_dim = Int(filter.dim[0]())
        var C_dim = Int(filter.dim[1]())
        var R_dim = Int(filter.dim[2]())
        var S_dim = Int(filter.dim[3]())

        @parameter
        @__copy_capture(filter_lt, filter_frsc_ptr, F_dim, C_dim, R_dim, S_dim)
        @always_inline
        def transpose_fcrs_to_frsc[
            _width: Int, _rank: Int, alignment: Int = 1
        ](coords: IndexList[_rank]):
            var f = coords[0]
            var r = coords[1]
            var s = coords[2]
            var c = coords[3]
            var val = filter_lt.load[width=_width](IndexList[4](f, c, r, s))
            var out_idx = (
                f * R_dim * S_dim * C_dim + r * S_dim * C_dim + s * C_dim + c
            )
            filter_frsc_ptr.store(out_idx, val)

        elementwise[transpose_fcrs_to_frsc, 1, target="gpu"](
            IndexList[4](F_dim, R_dim, S_dim, C_dim), ctx
        )
    else:
        # RSCF [R,S,C,F] -> FRSC [F,R,S,C]
        var R_dim = Int(filter.dim[0]())
        var S_dim = Int(filter.dim[1]())
        var C_dim = Int(filter.dim[2]())
        var F_dim = Int(filter.dim[3]())

        @parameter
        @__copy_capture(filter_lt, filter_frsc_ptr, R_dim, S_dim, C_dim, F_dim)
        @always_inline
        def transpose_rscf_to_frsc[
            _width: Int, _rank: Int, alignment: Int = 1
        ](coords: IndexList[_rank]):
            var f = coords[0]
            var r = coords[1]
            var s = coords[2]
            var c = coords[3]
            var val = filter_lt.load[width=_width](IndexList[4](r, s, c, f))
            var out_idx = (
                f * R_dim * S_dim * C_dim + r * S_dim * C_dim + s * C_dim + c
            )
            filter_frsc_ptr.store(out_idx, val)

        elementwise[transpose_rscf_to_frsc, 1, target="gpu"](
            IndexList[4](F_dim, R_dim, S_dim, C_dim), ctx
        )

    _miopen_algo_find_and_forward[input_type, filter_type, output_type](
        ptr_meta,
        input,
        filter_frsc_ptr.bitcast[NoneType](),
        filt,
        output,
        in_,
        out,
        pad,
        stride,
        dil,
        num_groups,
        params_match,
        filter_is_fcrs,
        ctx,
    )
    _ = filter_frsc_buf^


def conv_miopen[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_is_fcrs: Bool = False,
](
    input: TileTensor[input_type, ...],
    filter: TileTensor[filter_type, ...],
    output: TileTensor[output_type, ...],
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    _conv_miopen[filter_is_fcrs=filter_is_fcrs](
        input, filter, output, stride, dilation, padding, num_groups, ctx
    )


def conv_gpu[
    conv_rank: Int,
    //,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    maybe_epilogue_func: Optional[elementwise_simd_epilogue_type] = None,
    filter_is_fcrs: Bool = False,
    has_residual: Bool = False,
](
    input: TileTensor[input_type, address_space=AddressSpace.GENERIC, ...],
    filter: TileTensor[filter_type, address_space=AddressSpace.GENERIC, ...],
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    stride: IndexList[conv_rank],
    dilation: IndexList[conv_rank],
    padding: IndexList[2 * conv_rank],
    num_groups: Int,
    ctx: DeviceContext,
    source_ptr: UnsafePointer[
        Scalar[output_type], MutAnyOrigin
    ] = UnsafePointer[Scalar[output_type], MutAnyOrigin](),
    beta: Float32 = 0.0,
) raises:
    # Bridge to LayoutTensor for internal GPU kernel dispatch and cuDNN/MIOpen
    # which require Layout type parameters.
    var input_lt = input.to_layout_tensor()
    var filter_lt = filter.to_layout_tensor()
    var output_lt = output.to_layout_tensor()

    comptime input_layout = input_lt.layout
    comptime filter_layout = filter_lt.layout
    comptime output_layout = output_lt.layout

    comptime assert conv_rank == input_lt.rank - 2

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
        var paddings_tensor = tt_stack_allocation[dtype=DType.int](
            row_major[2 * full_rank]()
        )

        comptime for axis in range(full_rank):
            paddings_tensor[2 * axis] = 0
            paddings_tensor[2 * axis + 1] = 0

        comptime for i in range(conv_rank):
            comptime SIMDInt = Scalar[DType.int]

            var axis = i + 1  # skip batch axis
            paddings_tensor[2 * axis] = SIMDInt(padding[2 * i])  # before
            paddings_tensor[2 * axis + 1] = SIMDInt(padding[2 * i + 1])  # after

        var input_shape = rebind[IndexList[full_rank]](
            input_lt.runtime_layout.shape.value.canonicalize()
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

        # Construct padded input as LayoutTensor, then bridge to TileTensor
        # for the recursive call. Using LayoutTensor here because full_rank
        # is variable and row_major(Coord) requires a fixed-rank tuple.
        var padded_input_lt = LayoutTensor[
            input_type,
            Layout.row_major[full_rank](),
            MutAnyOrigin,
        ](
            padded_device_buffer,
            RuntimeLayout[Layout.row_major[full_rank]()].row_major(
                padded_shape
            ),
        )
        var padded_input_tt = lt_to_tt(padded_input_lt)

        var zero_padding = IndexList[2 * conv_rank](0)

        conv_gpu[
            input_type,
            filter_type,
            output_type,
            maybe_epilogue_func,
            filter_is_fcrs,
            has_residual,
        ](
            padded_input_tt,
            filter,
            output,
            stride,
            dilation,
            zero_padding,
            num_groups,
            ctx,
            source_ptr,
            beta,
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
        output_lt.dim[1](), block_size
    )  # height for 2d and depth for 3d
    var grid_dim_z = input_lt.dim[0]()  # n for both

    comptime if input_lt.rank == 4:
        # Try SM100 structured conv2d on Blackwell GPUs (4-7x faster than cuDNN)
        comptime _is_sm100 = _is_sm10x_gpu(ctx.default_device_info)
        comptime _is_supported_dtype = input_type == DType.bfloat16

        comptime if _is_sm100 and _is_supported_dtype:
            from nn.conv.gpu.nvidia.sm100.dispatch import (
                dispatch_sm100_conv2d,
            )
            from linalg.utils import elementwise_epilogue_type

            # SM100 dispatch: stride=1, dilation=1, groups=1,
            # and channels aligned to 64 (TMA tile K alignment)
            var s = rebind[IndexList[2]](stride)
            var d = rebind[IndexList[2]](dilation)
            var in_c = input_lt.dim[input_lt.rank - 1]()
            var out_c = output_lt.dim[output_lt.rank - 1]()
            if (
                s[0] == 1
                and s[1] == 1
                and d[0] == 1
                and d[1] == 1
                and num_groups == 1
                and in_c % 64 == 0
                and out_c % 128 == 0
            ):

                @parameter
                @always_inline
                def _sm100_dispatch[
                    _epilogue: Optional[elementwise_epilogue_type] = None,
                ]() raises:
                    dispatch_sm100_conv2d[
                        input_type,
                        filter_type,
                        output_type,
                        filter_is_fcrs,
                        elementwise_lambda_fn=_epilogue,
                        has_residual=has_residual,
                    ](
                        input,
                        filter,
                        output,
                        rebind[IndexList[2]](symmetric_padding),
                        ctx,
                        source_ptr,
                        beta,
                    )

                comptime if maybe_epilogue_func:
                    # Wrap the 4D NHWC epilogue into a 2D GEMM-space
                    # void epilogue for the SM100 kernel. The kernel
                    # calls this with (m, n) coords where
                    # m = batch*H_out*W_out + h*W_out + w, n = channel.
                    comptime epilogue = maybe_epilogue_func.value()
                    var out_h = output_lt.dim[1]()
                    var out_w = output_lt.dim[2]()
                    var hw = out_h * out_w

                    @parameter
                    @always_inline
                    @__copy_capture(hw, out_w)
                    def sm100_void_epilogue[
                        _dtype: DType,
                        _width: Int,
                        *,
                        alignment: Int = 1,
                    ](coords_2d: IndexList[2], val: SIMD[_dtype, _width],):
                        var m = coords_2d[0]
                        var n = coords_2d[1]
                        var batch_idx: Int
                        var rem: Int
                        var h_idx: Int
                        var w_idx: Int
                        batch_idx, rem = divmod(m, hw)
                        h_idx, w_idx = divmod(rem, out_w)
                        epilogue(
                            IndexList[4](batch_idx, h_idx, w_idx, n),
                            rebind[SIMD[output_type, _width]](val),
                        )

                    _sm100_dispatch[
                        Optional[elementwise_epilogue_type](sm100_void_epilogue)
                    ]()
                else:
                    _sm100_dispatch[]()
                return

        # AMD RDNA 3+ dispatch: im2col + WMMA matmul for supported shapes.
        comptime if has_amd_rdna_gpu_accelerator() and input_type in (
            DType.bfloat16,
            DType.float16,
        ):
            from nn.conv.gpu.amd.rdna.dispatch import dispatch_rdna_conv2d

            if dispatch_rdna_conv2d[
                input_type,
                filter_type,
                output_type,
                filter_is_fcrs,
                maybe_epilogue_func=maybe_epilogue_func,
            ](
                input,
                filter,
                output,
                rebind[IndexList[2]](stride),
                rebind[IndexList[2]](dilation),
                rebind[IndexList[2]](symmetric_padding),
                num_groups,
                ctx,
            ):
                return

        # AMD GPU path: use MIOpen for conv2d.
        # Note: MIOpen's miopenConvolution mode (0) is cross-correlation
        # (standard DNN conv). The old CROSS_CORRELATION (1) was actually
        # miopenTranspose (deconvolution), causing x/y descriptor swaps.
        comptime if has_amd_gpu_accelerator():
            comptime if maybe_epilogue_func:
                # MIOpen doesn't support epilogues. Compute to temp buffer,
                # then apply epilogue.
                comptime epilogue = maybe_epilogue_func.value()
                var output_tmp_data = ctx.enqueue_create_buffer[output_type](
                    output_lt.size()
                )
                var output_tmp_lt = LayoutTensor[
                    output_type, output_layout, MutAnyOrigin
                ](
                    output_tmp_data.unsafe_ptr(),
                    output_lt.runtime_layout,
                )
                var output_tmp_tt = lt_to_tt(output_tmp_lt)
                _conv_miopen[filter_is_fcrs=filter_is_fcrs](
                    input,
                    filter,
                    output_tmp_tt,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

                @parameter
                @__copy_capture(output_tmp_lt)
                @always_inline
                def amd_miopen_epilogue[
                    _width: Int, _rank: Int, alignment: Int = 1
                ](coords: IndexList[_rank]):
                    var vec = output_tmp_lt.load[width=_width](
                        rebind[IndexList[4]](coords)
                    )
                    epilogue(coords, vec)

                elementwise[
                    amd_miopen_epilogue,
                    simd_width_of[output_type](),
                    target="gpu",
                ](
                    output_lt.runtime_layout.shape.value.canonicalize(),
                    ctx,
                )
                _ = output_tmp_data^
            else:
                _conv_miopen[filter_is_fcrs=filter_is_fcrs](
                    input,
                    filter,
                    output,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )
            return

        # Fallback paths for non-SM100, unsupported dtypes, or constraints
        comptime if filter_is_fcrs:
            # Construct row-major TileTensors for cuDNN (shared by both
            # epilogue and non-epilogue paths).
            var _in_s = input_lt.runtime_layout.shape.value.canonicalize()
            var input_rm = TileTensor(
                input.ptr,
                row_major(
                    (
                        Idx(_in_s[0]),
                        Idx(_in_s[1]),
                        Idx(_in_s[2]),
                        Idx(_in_s[3]),
                    )
                ),
            )
            var _filt_s = filter_lt.runtime_layout.shape.value.canonicalize()
            var filter_rm = TileTensor(
                filter.ptr,
                row_major(
                    (
                        Idx(_filt_s[0]),
                        Idx(_filt_s[1]),
                        Idx(_filt_s[2]),
                        Idx(_filt_s[3]),
                    )
                ),
            )

            comptime if maybe_epilogue_func:
                comptime epilogue = maybe_epilogue_func.value()
                var output_tmp_data = ctx.enqueue_create_buffer[output_type](
                    output_lt.size()
                )

                var output_tmp_lt = LayoutTensor[
                    output_type, output_layout, MutAnyOrigin
                ](
                    output_tmp_data.unsafe_ptr(),
                    output_lt.runtime_layout,
                )

                var _out_tmp_s = (
                    output_tmp_lt.runtime_layout.shape.value.canonicalize()
                )
                var output_tmp_rm = TileTensor(
                    output_tmp_lt.ptr.unsafe_origin_cast[MutAnyOrigin](),
                    row_major(
                        (
                            Idx(_out_tmp_s[0]),
                            Idx(_out_tmp_s[1]),
                            Idx(_out_tmp_s[2]),
                            Idx(_out_tmp_s[3]),
                        )
                    ),
                )

                conv_cudnn[input_type, filter_type, output_type](
                    input_rm,
                    filter_rm,
                    output_tmp_rm,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

                @parameter
                @__copy_capture(output_tmp_lt)
                @always_inline
                def epilogue_wrapper[
                    _width: Int, _rank: Int, alignment: Int = 1
                ](coords: IndexList[_rank]):
                    comptime align = align_of[SIMD[output_type, _width]]()
                    vec = output_tmp_lt.load[width=_width](
                        rebind[IndexList[4]](coords)
                    )
                    epilogue(coords, vec)

                elementwise[
                    epilogue_wrapper, simd_width_of[output_type](), target="gpu"
                ](output_lt.runtime_layout.shape.value.canonicalize(), ctx)

                _ = output_tmp_data^

            else:
                var _out_s = output_lt.runtime_layout.shape.value.canonicalize()
                var output_rm = TileTensor(
                    output.ptr,
                    row_major(
                        (
                            Idx(_out_s[0]),
                            Idx(_out_s[1]),
                            Idx(_out_s[2]),
                            Idx(_out_s[3]),
                        )
                    ),
                )

                conv_cudnn[input_type, filter_type, output_type](
                    input_rm,
                    filter_rm,
                    output_rm,
                    rebind[IndexList[2]](stride),
                    rebind[IndexList[2]](dilation),
                    rebind[IndexList[2]](symmetric_padding),
                    num_groups,
                    ctx,
                )

        else:
            var grid_dim_x = ceildiv(
                output_lt.dim[2](), block_size
            )  # w / block size for 2d
            ctx.enqueue_function[conv_gpu_n, conv_gpu_n](
                input_lt,
                filter_lt,
                output_lt,
                stride,
                dilation,
                symmetric_padding,
                num_groups,
                grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                block_dim=(block_size, block_size),
            )

    elif input_lt.rank == 5:
        comptime if filter_is_fcrs:
            conv3d_cudnn[input_type, filter_type, output_type](
                input_lt,
                filter_lt,
                output_lt,
                rebind[IndexList[3]](stride),
                rebind[IndexList[3]](dilation),
                rebind[IndexList[3]](symmetric_padding),
                num_groups,
                ctx,
            )
        else:
            var grid_dim_x = ceildiv(
                output_lt.dim[2]() * output_lt.dim[3](), block_size
            )  # h * w / block size for 3d
            ctx.enqueue_function[conv_gpu_3d, conv_gpu_3d](
                input_lt,
                filter_lt,
                output_lt,
                stride,
                dilation,
                symmetric_padding,
                num_groups,
                grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                block_dim=(block_size, block_size),
            )


def conv3d_gpu_naive_ndhwc_qrscf[
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
    num_groups: Int,
):
    var N = input.dim[0]()
    var D = input.dim[1]()  # depth
    var H = input.dim[2]()
    var W = input.dim[3]()
    var C_in = input.dim[4]()  # channel_input

    var Q = filter.dim[0]()
    var R = filter.dim[1]()
    var S = filter.dim[2]()
    var C_per_group = filter.dim[3]()  # C_in / num_groups

    var D_out = output.dim[1]()  # depth
    var H_out = output.dim[2]()
    var W_out = output.dim[3]()
    var C_out = output.dim[4]()  # channel_output
    var F_per_group = C_out // num_groups

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
    var h_out_idx, w_out_idx = divmod(x_thread_id, UInt(W_out))

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
        var g = co // F_per_group
        var ci_base = g * C_per_group

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

                    if 0 <= d_in < D and 0 <= h_in < H and 0 <= w_in < W:
                        for ci in range(C_per_group):
                            value += (
                                input.load[width=1](
                                    IndexList[5](
                                        Int(n), d_in, h_in, w_in, ci_base + ci
                                    )
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


# ===----------------------------------------------------------------------=== #
# GPU 3D Convolution using cuDNN (Nd APIs)                                     #
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _Conv3dAlgoCacheEntry(Copyable, Movable):
    """Cached cuDNN algorithm selection result for a conv3d shape."""

    var algo_value: Int8
    var workspace_size: Int

    def algo(self) -> cudnnConvolutionFwdAlgo_t:
        return rebind[cudnnConvolutionFwdAlgo_t](self.algo_value)


def _conv3d_cudnn_depth_tiled[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, ...],
    filter: LayoutTensor[filter_type, ...],
    output: LayoutTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """Depth-tiled cuDNN 3D convolution for tensors exceeding INT32_MAX elements.

    Splits the computation along the depth dimension (dim[1] in NDHWC) into
    tiles small enough for cuDNN's internal Int32 stride calculations.
    Each tile uses a separate set of cuDNN descriptors.
    """
    comptime INT32_MAX_VAL = 2147483647
    comptime FIND_WS_CAP = 256 * 1024 * 1024

    var N = input.dim[0]()
    var D_in = input.dim[1]()
    var H = input.dim[2]()
    var W = input.dim[3]()
    var C = input.dim[4]()

    var K_d = filter.dim[2]()  # kernel depth (Q in FCQRS)
    var F_out = filter.dim[0]()  # output channels
    var D_out = output.dim[1]()
    var H_out = output.dim[2]()
    var W_out = output.dim[3]()

    var eff_k = (K_d - 1) * dilation[0] + 1  # effective kernel depth

    # Calculate max input depth per tile.
    var per_frame_in = N * H * W * C
    var max_d_in = INT32_MAX_VAL // per_frame_in

    # Also ensure output elements per tile fit in INT32.
    var per_frame_out = N * H_out * W_out * F_out
    var max_d_out = INT32_MAX_VAL // per_frame_out
    # Output frames from max_d_in input frames:
    var tile_d_out_from_in = (max_d_in + 2 * padding[0] - eff_k) // stride[
        0
    ] + 1
    var tile_d_out = min(tile_d_out_from_in, max_d_out)
    if tile_d_out < 1:
        raise "conv3d: tensor too large even for single-frame tiling"

    # Input depth needed for tile_d_out output frames.
    var tile_d_in = (tile_d_out - 1) * stride[0] + eff_k - 2 * padding[0]

    # Strides (in elements) along the depth dimension.
    var in_d_stride = H * W * C  # elements per depth frame
    var out_d_stride = H_out * W_out * F_out

    var ptr_meta = _get_cudnn_meta(ctx)

    # Descriptor arrays (reused across tiles).
    var input_dims = alloc[Int32](5)
    var output_dims = alloc[Int32](5)
    var filter_dims = alloc[Int32](5)
    var pad_a = alloc[Int32](3)
    var stride_a = alloc[Int32](3)
    var dilation_a = alloc[Int32](3)

    # Filter dims (constant across tiles).
    filter_dims[0] = Int32(filter.dim[0]())
    filter_dims[1] = Int32(filter.dim[1]())
    filter_dims[2] = Int32(filter.dim[2]())
    filter_dims[3] = Int32(filter.dim[3]())
    filter_dims[4] = Int32(filter.dim[4]())

    check_cudnn_error(
        cudnnSetFilterNdDescriptor(
            ptr_meta[].ptr_filter_desc,
            get_cudnn_dtype[filter_type](),
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            Int16(5),
            filter_dims.bitcast[NoneType](),
        )
    )

    # Convolution params (constant except padding for first tile).
    stride_a[0] = Int32(stride[0])
    stride_a[1] = Int32(stride[1])
    stride_a[2] = Int32(stride[2])
    dilation_a[0] = Int32(dilation[0])
    dilation_a[1] = Int32(dilation[1])
    dilation_a[2] = Int32(dilation[2])

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    var d_out_start = 0
    while d_out_start < D_out:
        var this_d_out = min(tile_d_out, D_out - d_out_start)

        # Determine input range for this output tile.
        # First tile gets front padding, last tile gets back padding.
        var d_in_start: Int
        var this_d_in: Int
        var tile_pad_front: Int
        var tile_pad_back: Int

        if d_out_start == 0:
            # First tile: include front padding.
            tile_pad_front = padding[0]
            d_in_start = 0
            this_d_in = (
                (this_d_out - 1) * stride[0] + eff_k - 2 * tile_pad_front
            )
            # Adjust: no need for more input than available
            if this_d_in > D_in:
                this_d_in = D_in
            tile_pad_back = 0
        else:
            tile_pad_front = 0
            # For stride=1: input frame for output d is at d (with padding=0)
            d_in_start = d_out_start * stride[0] - padding[0]
            if d_in_start < 0:
                tile_pad_front = -d_in_start
                d_in_start = 0
            this_d_in = (this_d_out - 1) * stride[0] + eff_k - tile_pad_front
            # Check if we need back padding
            if d_in_start + this_d_in > D_in:
                tile_pad_back = d_in_start + this_d_in - D_in
                this_d_in = D_in - d_in_start
            else:
                tile_pad_back = 0

        # --- Set up tile descriptors ---
        # Input tile: [N, this_d_in, H, W, C]
        input_dims[0] = Int32(N)
        input_dims[1] = Int32(C)
        input_dims[2] = Int32(this_d_in)
        input_dims[3] = Int32(H)
        input_dims[4] = Int32(W)

        check_cudnn_error(
            cudnnSetTensorNdDescriptorEx(
                ptr_meta[].ptr_input_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[input_type](),
                Int16(5),
                input_dims.bitcast[NoneType](),
            )
        )

        # Output tile: [N, this_d_out, H_out, W_out, F]
        output_dims[0] = Int32(N)
        output_dims[1] = Int32(F_out)
        output_dims[2] = Int32(this_d_out)
        output_dims[3] = Int32(H_out)
        output_dims[4] = Int32(W_out)

        check_cudnn_error(
            cudnnSetTensorNdDescriptorEx(
                ptr_meta[].ptr_output_desc,
                cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
                get_cudnn_dtype[output_type](),
                Int16(5),
                output_dims.bitcast[NoneType](),
            )
        )

        # Convolution with tile-specific depth padding.
        pad_a[0] = Int32(tile_pad_front)
        pad_a[1] = Int32(padding[1])
        pad_a[2] = Int32(padding[2])

        check_cudnn_error(
            cudnnSetConvolutionNdDescriptor(
                ptr_meta[].ptr_conv_desc,
                Int16(3),
                pad_a.bitcast[NoneType](),
                stride_a.bitcast[NoneType](),
                dilation_a.bitcast[NoneType](),
                cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
            )
        )
        check_cudnn_error(
            cudnnSetConvolutionGroupCount(
                ptr_meta[].ptr_conv_desc, Int16(num_groups)
            )
        )
        check_cudnn_error(
            cudnnSetConvolutionMathType(
                ptr_meta[].ptr_conv_desc,
                cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
            )
        )

        # --- Algorithm selection (use GetWorkspaceSize for PRECOMP_GEMM) ---
        var algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        )
        var ws_size: Int = 0
        var ws_st = cudnnGetConvolutionForwardWorkspaceSize(
            ptr_meta[].ptr_handle,
            ptr_meta[].ptr_input_desc,
            ptr_meta[].ptr_filter_desc,
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].ptr_output_desc,
            algo,
            UnsafePointer(to=ws_size),
        )
        if ws_st != cudnnStatus_t.CUDNN_STATUS_SUCCESS or ws_size > FIND_WS_CAP:
            # Fall back to IMPLICIT_GEMM (no workspace needed).
            algo = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            )
            ws_size = 0

        # --- Execute tile ---
        var workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](ws_size)

        # Compute pointer offsets for input and output tiles.
        var in_offset = d_in_start * in_d_stride
        var out_offset = d_out_start * out_d_stride
        var in_ptr = input.ptr + in_offset
        var out_ptr = output.ptr + out_offset

        var fwd_status = cudnnConvolutionForward(
            ptr_meta[].ptr_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            ptr_meta[].ptr_input_desc,
            in_ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            algo,
            workspace_buffer.unsafe_ptr().bitcast[NoneType](),
            ws_size,
            UnsafePointer(to=beta).bitcast[NoneType](),
            ptr_meta[].ptr_output_desc,
            out_ptr.bitcast[NoneType](),
        )
        _ = workspace_buffer^

        if fwd_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
            input_dims.free()
            output_dims.free()
            filter_dims.free()
            pad_a.free()
            stride_a.free()
            dilation_a.free()
            ctx.synchronize()
            raise String("conv3d tiled forward failed: ", fwd_status)

        d_out_start += this_d_out

    # Clean up.
    input_dims.free()
    output_dims.free()
    filter_dims.free()
    pad_a.free()
    stride_a.free()
    dilation_a.free()


def _conv3d_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, ...],
    filter: LayoutTensor[filter_type, ...],
    output: LayoutTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    """cuDNN 3D convolution using Nd descriptor APIs.

    Expects:
      - input:  NDHWC layout [N, D, H, W, C]
      - filter: FCQRS layout [F, C/groups, Q, R, S]
      - output: NDHWC layout [N, D_out, H_out, W_out, F]

    Algorithm selection is cached per unique shape+params combination so that
    the expensive FindEx search only runs once per shape.

    When the total number of elements exceeds INT32_MAX (~2.1B), cuDNN's
    internal stride calculations overflow. In this case we tile along the
    depth (D) dimension, processing each tile with a separate cuDNN call.
    """
    comptime FIND_WS_CAP = 256 * 1024 * 1024
    comptime INT32_MAX_VAL = 2147483647

    # --- Check if depth tiling is needed (INT32 stride overflow) ---
    var total_in = (
        input.dim[0]()
        * input.dim[1]()
        * input.dim[2]()
        * input.dim[3]()
        * input.dim[4]()
    )
    if total_in > INT32_MAX_VAL:
        _conv3d_cudnn_depth_tiled(
            input,
            filter,
            output,
            stride,
            dilation,
            padding,
            num_groups,
            ctx,
        )
        return

    var ptr_meta = _get_cudnn_meta(ctx)

    # --- Set up cuDNN descriptors (required every call — shared state) ---
    # Input: NDHWC in memory, described as NHWC format with dims [N,C,D,H,W].
    var input_dims = alloc[Int32](5)
    input_dims[0] = Int32(input.dim[0]())  # N
    input_dims[1] = Int32(input.dim[4]())  # C
    input_dims[2] = Int32(input.dim[1]())  # D
    input_dims[3] = Int32(input.dim[2]())  # H
    input_dims[4] = Int32(input.dim[3]())  # W

    check_cudnn_error(
        cudnnSetTensorNdDescriptorEx(
            ptr_meta[].ptr_input_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[input_type](),
            Int16(5),
            input_dims.bitcast[NoneType](),
        )
    )

    # Filter: FCQRS layout [F, C/groups, Q, R, S], described as NCHW format.
    var filter_dims = alloc[Int32](5)
    filter_dims[0] = Int32(filter.dim[0]())  # F (out_channels)
    filter_dims[1] = Int32(filter.dim[1]())  # C (in_channels / groups)
    filter_dims[2] = Int32(filter.dim[2]())  # Q (depth)
    filter_dims[3] = Int32(filter.dim[3]())  # R (height)
    filter_dims[4] = Int32(filter.dim[4]())  # S (width)

    check_cudnn_error(
        cudnnSetFilterNdDescriptor(
            ptr_meta[].ptr_filter_desc,
            get_cudnn_dtype[filter_type](),
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            Int16(5),
            filter_dims.bitcast[NoneType](),
        )
    )

    # Convolution: 3 spatial dimensions.
    var pad_a = alloc[Int32](3)
    pad_a[0] = Int32(padding[0])
    pad_a[1] = Int32(padding[1])
    pad_a[2] = Int32(padding[2])

    var stride_a = alloc[Int32](3)
    stride_a[0] = Int32(stride[0])
    stride_a[1] = Int32(stride[1])
    stride_a[2] = Int32(stride[2])

    var dilation_a = alloc[Int32](3)
    dilation_a[0] = Int32(dilation[0])
    dilation_a[1] = Int32(dilation[1])
    dilation_a[2] = Int32(dilation[2])

    check_cudnn_error(
        cudnnSetConvolutionNdDescriptor(
            ptr_meta[].ptr_conv_desc,
            Int16(3),
            pad_a.bitcast[NoneType](),
            stride_a.bitcast[NoneType](),
            dilation_a.bitcast[NoneType](),
            cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
        )
    )

    check_cudnn_error(
        cudnnSetConvolutionGroupCount(
            ptr_meta[].ptr_conv_desc, Int16(num_groups)
        )
    )

    # Output: NDHWC in memory, described as NHWC format with dims [N,C,D,H,W].
    var output_dims = alloc[Int32](5)
    output_dims[0] = Int32(output.dim[0]())  # N
    output_dims[1] = Int32(output.dim[4]())  # C (out_channels)
    output_dims[2] = Int32(output.dim[1]())  # D_out
    output_dims[3] = Int32(output.dim[2]())  # H_out
    output_dims[4] = Int32(output.dim[3]())  # W_out

    check_cudnn_error(
        cudnnSetTensorNdDescriptorEx(
            ptr_meta[].ptr_output_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NHWC,
            get_cudnn_dtype[output_type](),
            Int16(5),
            output_dims.bitcast[NoneType](),
        )
    )

    # Allow tensor-op math with automatic type conversion — required for
    # bfloat16 3D convolutions on modern cuDNN (matches PR #5988 approach).
    check_cudnn_error(
        cudnnSetConvolutionMathType(
            ptr_meta[].ptr_conv_desc,
            cudnnMathType_t.CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION,
        )
    )

    # --- Algorithm selection (cached per shape) ---
    var cache_key = String(
        "CONV3D_ALGO_",
        ctx.id(),
        "_",
        input.dim[0](),
        "_",
        input.dim[4](),
        "_",
        input.dim[1](),
        "_",
        input.dim[2](),
        "_",
        input.dim[3](),
        "_F",
        filter.dim[0](),
        "_",
        filter.dim[1](),
        "_",
        filter.dim[2](),
        "_",
        filter.dim[3](),
        "_",
        filter.dim[4](),
        "_p",
        padding[0],
        "_",
        padding[1],
        "_",
        padding[2],
        "_s",
        stride[0],
        "_",
        stride[1],
        "_",
        stride[2],
        "_d",
        dilation[0],
        "_",
        dilation[1],
        "_",
        dilation[2],
        "_g",
        num_groups,
    )

    var algo: cudnnConvolutionFwdAlgo_t
    var workspace_size_var: Int

    if ptr_cached := _get_global_or_null(cache_key):
        # Cache hit — reuse previously selected algorithm.
        var entry = ptr_cached.unsafe_value().bitcast[_Conv3dAlgoCacheEntry]()
        algo = entry[].algo()
        workspace_size_var = entry[].workspace_size
    else:
        # Cache miss — run FindEx to find the fastest algorithm.
        var find_ws = ctx.enqueue_create_buffer[DType.uint8](FIND_WS_CAP)

        # CRITICAL: The Mojo cudnnConvolutionFwdAlgoPerfStruct uses Int8 for
        # enum fields, but the C struct uses int (4 bytes). This causes a
        # size mismatch: Mojo struct = ~32 bytes, C struct = 48 bytes.
        # Allocating with the Mojo struct size would cause a buffer overflow
        # when cuDNN writes 8 * 48 = 384 bytes. We allocate raw bytes with
        # the correct C struct size and read fields at proper offsets.
        comptime C_PERF_STRUCT_SIZE = 48  # sizeof(cudnnConvolutionFwdAlgoPerf_t)
        comptime MAX_ALGOS = 8
        var perf_bytes = alloc[UInt8](MAX_ALGOS * C_PERF_STRUCT_SIZE)

        # returned_algo_count is int* in C (4 bytes), not Int16*.
        # Use Int32 and bitcast the pointer.
        var returned_count_i32 = Int32(0)

        var find_status = cudnnFindConvolutionForwardAlgorithmEx(
            ptr_meta[].ptr_handle,
            ptr_meta[].ptr_input_desc,
            input.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_filter_desc,
            filter.ptr.bitcast[NoneType](),
            ptr_meta[].ptr_conv_desc,
            ptr_meta[].ptr_output_desc,
            output.ptr.bitcast[NoneType](),
            Int16(MAX_ALGOS),
            UnsafePointer(to=returned_count_i32).bitcast[Int16](),
            perf_bytes.bitcast[cudnnConvolutionFwdAlgoPerfStruct](),
            find_ws.unsafe_ptr().bitcast[NoneType](),
            FIND_WS_CAP,
        )
        _ = find_ws^

        # Read the returned count (C int at offset 0 of returned_count_i32).
        var returned_count = Int(returned_count_i32)

        # Pick the fastest successful algorithm within workspace cap.
        # Read fields from raw bytes at correct C struct offsets:
        #   offset  0: algo (int32)
        #   offset  4: status (int32)
        #   offset  8: time (float32)
        #   offset 16: memory (size_t / int64)
        algo = (
            cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        )
        workspace_size_var = 0

        var find_status_val = rebind[Int8](find_status)
        if find_status_val == 0:  # CUDNN_STATUS_SUCCESS
            for i in range(returned_count):
                var base = perf_bytes + i * C_PERF_STRUCT_SIZE
                var algo_val = base.bitcast[Int32]()[]  # offset 0
                var status_val = (base + 4).bitcast[Int32]()[]  # offset 4
                var memory_val = (base + 16).bitcast[Int]()[]  # offset 16
                if status_val == 0 and memory_val <= FIND_WS_CAP:
                    algo = rebind[cudnnConvolutionFwdAlgo_t](Int8(algo_val))
                    workspace_size_var = memory_val
                    break
        else:
            print(
                "conv3d FindEx FAILED: status=",
                Int(find_status_val),
                " input=[N=",
                input.dim[0](),
                " C=",
                input.dim[4](),
                " D=",
                input.dim[1](),
                " H=",
                input.dim[2](),
                " W=",
                input.dim[3](),
                "]",
            )
        perf_bytes.free()

        # Fallback: if FindEx found nothing useful, try PRECOMP_GEMM via
        # workspace size query (cheaper than FindEx).
        if (
            algo
            == cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            and workspace_size_var == 0
        ):
            var precomp = (
                cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            )
            var ws_size: Int = 0
            var ws_st = cudnnGetConvolutionForwardWorkspaceSize(
                ptr_meta[].ptr_handle,
                ptr_meta[].ptr_input_desc,
                ptr_meta[].ptr_filter_desc,
                ptr_meta[].ptr_conv_desc,
                ptr_meta[].ptr_output_desc,
                precomp,
                UnsafePointer(to=ws_size),
            )
            if (
                ws_st == cudnnStatus_t.CUDNN_STATUS_SUCCESS
                and ws_size <= FIND_WS_CAP
            ):
                algo = precomp
                workspace_size_var = ws_size

        # Store result in global cache.
        var ptr_entry = alloc[_Conv3dAlgoCacheEntry](1)
        ptr_entry.init_pointee_move(
            _Conv3dAlgoCacheEntry(
                algo_value=rebind[Int8](algo),
                workspace_size=workspace_size_var,
            )
        )
        external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
            StringSlice(cache_key),
            ptr_entry.bitcast[NoneType](),
        )

    # --- Execute convolution with cached/selected algorithm ---
    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    var workspace_buffer = ctx.enqueue_create_buffer[DType.uint8](
        workspace_size_var
    )
    var fwd_status = cudnnConvolutionForward(
        ptr_meta[].ptr_handle,
        UnsafePointer(to=alpha).bitcast[NoneType](),
        ptr_meta[].ptr_input_desc,
        input.ptr.bitcast[NoneType](),
        ptr_meta[].ptr_filter_desc,
        filter.ptr.bitcast[NoneType](),
        ptr_meta[].ptr_conv_desc,
        algo,
        workspace_buffer.unsafe_ptr().bitcast[NoneType](),
        workspace_size_var,
        UnsafePointer(to=beta).bitcast[NoneType](),
        ptr_meta[].ptr_output_desc,
        output.ptr.bitcast[NoneType](),
    )
    # Free workspace BEFORE sync to release the buffer back to the pool.
    _ = workspace_buffer^

    # Free temporary descriptor arrays.
    input_dims.free()
    filter_dims.free()
    pad_a.free()
    stride_a.free()
    dilation_a.free()
    output_dims.free()

    if fwd_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        # Synchronize device to flush any pending GPU operations and free
        # temporary cuDNN allocations, preventing VRAM accumulation.
        print("conv3d FORWARD FAILED: ", fwd_status, " algo=", algo)
        ctx.synchronize()
        raise String("cudnnConvolutionForward failed: ", fwd_status)


def conv3d_cudnn[
    input_type: DType,
    filter_type: DType,
    output_type: DType,
](
    input: LayoutTensor[input_type, ...],
    filter: LayoutTensor[filter_type, ...],
    output: LayoutTensor[output_type, ...],
    stride: IndexList[3],
    dilation: IndexList[3],
    padding: IndexList[3],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    # Set `ctx`'s CUcontext as current to satisfy cudnn's stateful API.
    with ctx.push_context() as ctx:
        _conv3d_cudnn(
            input, filter, output, stride, dilation, padding, num_groups, ctx
        )
