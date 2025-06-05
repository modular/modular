# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from collections.string import StaticString
from math import align_down, ceildiv
from sys.info import simdwidthof
from nn.conv import check_cudnn_error

from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import CUDA
from gpu._cudnn.cnn_infer import (
    cudnnConvolutionForward,
    cudnnConvolutionMode_t,
    cudnnConvolutionStruct,
    cudnnCreateConvolutionDescriptor,
    cudnnDestroyConvolutionDescriptor,
    cudnnDestroyFilterDescriptor,
    cudnnSetConvolution2dDescriptor,
    # Back-prop data helpers
    cudnnConvolutionBackwardData,
    cudnnGetConvolutionBackwardDataWorkspaceSize,
)
from gpu._cudnn.infer import (
    cudnnContext,
    cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionBwdDataAlgo_t,
    cudnnCreate,
    cudnnCreateFilterDescriptor,
    cudnnCreateTensorDescriptor,
    cudnnDataType_t,
    cudnnDestroy,
    cudnnDestroyTensorDescriptor,
    cudnnFilterStruct,
    cudnnSetFilter4dDescriptor,
    cudnnSetStream,
    cudnnSetTensor4dDescriptor,
    cudnnStatus_t,
    cudnnTensorFormat_t,
    cudnnTensorStruct,
)

from algorithm import (
    sync_parallelize,
    tile,
    tile_middle_unswitch_boundaries,
    vectorize,
)
from buffer.buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from linalg.accumulate import _Accumulator
from linalg.utils import partition_work
from memory import UnsafePointer
from register import register_internal
from runtime.asyncrt import parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList

from .conv_utils import (
    ConvInfoStatic,
    ConvPartition,
    ConvShape,
    align_down_residual,
    elementwise_epilogue_type,
    get_conv_num_tasks,
    get_conv_shape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)

# TODO: All attributes, except for groups and auto_pad, are supported.
#       - Kernel assumes groups = 1.
#       - For auto_pad, need to set `AutoPadMode.NOTSET` (default).
#       Only remaining issue is handling of optional attributes & setting defaults,
#       and the associated logic (e.g., if one attribute is specified another is
#       ignored).
#       Specifically, in the ONNX spec for ConvTranspose
#       (https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose)
#       - dilations (optional): if not provided, create (1,1) as default.
#       - kernel_shape (optional): if not provided, obtain from argument-provided kernel.
#       - strides (optional): if not provided, create (1,1) as default.
#       - output_shape (optional): if specified, pads values are ignored.
#       modular/Kernels/test/test_convtranspose.mojo provides examples of calls.
#       StarGAN, CycleGAN-and-pix2pix, Mask-RCNN are covered by this version.


@always_inline
fn conv_transpose_naive[
    type: DType,
](
    output: NDBuffer[type, 5, MutableAnyOrigin],
    input: NDBuffer[type, 5, MutableAnyOrigin],
    filter: NDBuffer[type, 5, MutableAnyOrigin],
    stride: IndexList[3],
    dilation: IndexList[3],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
):
    """
    Implements the ConvTranspose operator from the MO spec.

    Parameters:
        type: Type of the input, output, and kernel tensors.

    Args:
        output: Output data tensor that contains the result of the convolution.
        input: Input data tensor from previous layer, with size of (N x H x W x C),
               where N is the batch size, C is the number of channels, and H and
               W are the height and width.
        filter: The weight (kernel) tensor, with size of (kH x kW x M/groups x C),
                where C is the number of channels, kH and kW are the height and
                width of the kernel, and M is the number of feature maps.
        stride: Stride along each spatial axis.
        dilation: Dilation value along each spatial axis of the filter.
        pad_d: Padding in depth dimension.
        pad_h: Padding in height dimension.
        pad_w: Padding in width dimension.
    """

    var N = input.dim[0]()  # Number of images (num. batches)

    var D = input.dim[1]()
    var H = input.dim[2]()
    var W = input.dim[3]()

    var C = input.dim[4]()  # Number of input channels

    var Q = filter.dim[0]()
    var R = filter.dim[1]()
    var S = filter.dim[2]()

    var DO = output.dim[1]()
    var HO = output.dim[2]()
    var WO = output.dim[3]()

    var F = output.dim[4]()

    # Initialize output to zero
    output.zero()

    @always_inline
    fn accumulate_output_point(n: Int, d: Int, h: Int, w: Int, c: Int, f: Int):
        var do = d * stride[0] - pad_d[0]
        var ho = h * stride[1] - pad_h[0]
        var wo = w * stride[2] - pad_w[0]

        for q in range(Q):
            var do_nbr = do + q * dilation[0]
            if do_nbr < 0 or do_nbr >= DO:
                continue

            for r in range(R):
                var ho_nbr = ho + r * dilation[1]
                if ho_nbr < 0 or ho_nbr >= HO:
                    continue

                for s in range(S):
                    var wo_nbr = wo + s * dilation[2]
                    if wo_nbr < 0 or wo_nbr >= WO:
                        continue

                    var output_val = output[n, do_nbr, ho_nbr, wo_nbr, f]
                    var input_val = input[n, d, h, w, c]
                    var filter_val = filter[q, r, s, f, c]

                    output[Index(n, do_nbr, ho_nbr, wo_nbr, f)] = (
                        output_val + input_val * filter_val
                    )

    for n in range(N):
        for c in range(C):
            for f in range(F):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            accumulate_output_point(n, d, h, w, c, f)


@always_inline
fn conv_transpose_shape[
    input_rank: Int,
    kernel_rank: Int,
    type: DType,
    strides_type: DType,
    dilations_type: DType,
    pads_type: DType,
    output_pads_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, input_rank],
    kernel: NDBuffer[type, kernel_rank],
    strides: NDBuffer[strides_type, 1],
    dilations: NDBuffer[dilations_type, 1],
    pads: NDBuffer[pads_type, 1],
    output_pads: NDBuffer[output_pads_type, 1],
) raises -> IndexList[input_rank]:
    """
    Compute the output shape of a `conv-transpose` operation, and assert the
    inputs are compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        kernel_rank: Rank of the kernel tensor.
        type: Element type of the input and kernel tensor.
        strides_type: Element type of the strides tensor.
        dilations_type: Element type of the dilations tensor.
        pads_type: Element type of the pads tensor.
        output_pads_type: Element type of the output_pads tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        kernel: The kernel tensor.
        strides: The strides tensor.
        dilations: The dilations tensor.
        pads: The paddings tensor.
        output_pads: The output paddings tensor.

    Returns:
        The output shape.
    """

    if input_rank != 4 and input_rank != 5:
        raise Error("[conv_transpose] requires (input_rank == 4 or 5)")
    if input_rank != kernel_rank:
        raise Error("[conv_transpose] requires (input_rank == kernel_rank)")
    if strides.dim(0) != input_rank - 2 or dilations.dim(0) != input_rank - 2:
        raise Error(
            "[conv_transpose] requires (len(strides) == len(dilations) =="
            " input_rank - 2)"
        )
    if pads.dim(0) != 2 * (input_rank - 2):
        raise Error(
            "[conv_transpose] requires (len(paddings) == 2 * (input rank - 2))"
        )

    # Assume input has channel last layout, NHWC or NDHWC.
    var batch_size = input.dim(0)
    var output_shape = IndexList[input_rank]()
    # Assume kernel has layout RSFC or QRSFC. The output channel is F because
    # this is a convolution transpose shape function (inverse of regular
    # convolution).
    var output_channels = kernel.dim(input_rank - 2)

    # Check input and kernel channels
    if input.dim(input_rank - 1) != kernel.dim(input_rank - 1):
        raise Error(
            "[conv_transpose] requires input channel to match output channel in"
            " kernel"
        )

    # Compute output shape
    output_shape[0] = batch_size
    output_shape[input_rank - 1] = output_channels

    @parameter
    @always_inline
    fn compute_output_spatial_dim(
        input_spatial_dim: Int,
        kernel_spatial_dim: Int,
        stride: Int,
        dilation: Int,
        pre_pad: Int,
        post_pad: Int,
        out_pad: Int,
    ) -> Int:
        return (
            stride * (input_spatial_dim - 1)
            + out_pad
            + ((kernel_spatial_dim - 1) * dilation + 1)
            - pre_pad
            - post_pad
        )

    # Compute the spatial dims
    @parameter
    for i in range(input_rank - 2):
        var input_spatial_dim = input.dim(i + 1)
        var kernel_spatial_dim = kernel.dim(i)
        var stride = Int(strides[i])
        var dilation = Int(dilations[i])
        var pre_pad = Int(pads[2 * i])
        var post_pad = Int(pads[2 * i + 1])
        var out_pad = Int(output_pads[i])
        var output_spatial_dim = compute_output_spatial_dim(
            input_spatial_dim,
            kernel_spatial_dim,
            stride,
            dilation,
            pre_pad,
            post_pad,
            out_pad,
        )
        if output_spatial_dim <= 0:
            raise Error("[conv_transpose] output spatial dim must be positive")
        output_shape[i + 1] = output_spatial_dim

    return output_shape


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Helpers                                        #
# ===----------------------------------------------------------------------=== #


fn get_num_partitions[
    micro_kernel_height: Int, micro_kernel_f_size: Int
](num_threads: Int, conv_shape: ConvShape) -> IndexList[4]:
    """Partition the worload in (batch&group, C, F, H) dimensions.
    HOWO is the combination of HO and WO dimensions.
    The actual number of tasks are the product of return num_partitions.
    """

    var max_num_tasks = get_conv_num_tasks(num_threads, conv_shape)

    var num_batches_and_groups = conv_shape.n * conv_shape.num_groups

    var max_f_tasks = ceildiv(conv_shape.f, micro_kernel_f_size)

    var num_partitions = IndexList[4](1)

    num_partitions[0] = min(max_num_tasks, num_batches_and_groups)
    num_partitions[2] = min(max_f_tasks, max_num_tasks // num_partitions[0])

    return num_partitions


fn get_partition(
    task_id: Int,
    num_partitions: IndexList[4],
    conv_shape: ConvShape,
    micro_kernel_height: Int,
    micro_kernel_f_size: Int,
) -> ConvPartition:
    var task_id_f = task_id % num_partitions[2]
    var quotient = task_id // num_partitions[2]
    var task_id_howo = quotient % num_partitions[3]
    var task_id_ng = quotient // num_partitions[3]

    var ng_range = partition_work(
        task_id_ng, num_partitions[0], conv_shape.n * conv_shape.num_groups, 1
    )

    var c_range = Index(0, conv_shape.c)

    var f_range = partition_work(
        task_id_f,
        num_partitions[2],
        conv_shape.f // conv_shape.num_groups,
        micro_kernel_f_size,
    )

    var howo_range = partition_work(
        task_id_howo, num_partitions[3], conv_shape.h(), 1
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


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Entry Point                                    #
# ===----------------------------------------------------------------------=== #


@value
struct ConvTransposedPacked[
    input_mut: Bool,
    filter_mut: Bool, //,
    input_rank: Int,
    filter_rank: Int,
    output_rank: Int,
    input_origin: Origin[input_mut],
    filter_origin: Origin[filter_mut],
    output_origin: MutableOrigin,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    conv_attr: ConvInfoStatic[input_rank - 2],
    elementwise_epilogue: OptionalReg[elementwise_epilogue_type] = None,
]:
    var output: NDBuffer[output_type, output_rank, output_origin, output_shape]
    var input: NDBuffer[input_type, input_rank, input_origin, input_shape]
    var filter: NDBuffer[filter_type, filter_rank, filter_origin, filter_shape]

    var conv_shape: ConvShape[input_rank - 2]

    # Support partition in 4 dims: (n, c, f, ho_or_howo). If the input is
    # padded, the output spacial dims are merged into one as howo. If not
    # padded, only ho is partitioned for now.
    var partition: ConvPartition

    var cf_tile_size: IndexList[2]

    @staticmethod
    fn run(
        output: NDBuffer[output_type, output_rank, output_origin, output_shape],
        input: NDBuffer[input_type, input_rank, input_origin, input_shape],
        filter: NDBuffer[filter_type, filter_rank, filter_origin, filter_shape],
        conv_shape: ConvShape[input_rank - 2],
    ) raises:
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_shape = get_micro_kernel_shape[
            input_rank - 2,
            output_shape.at[output_rank - 2](),  # WO
            output_shape.at[output_rank - 1](),  # F
            conv_attr,
            simd_size,
        ]()
        alias micro_kernel_height = micro_kernel_shape[0]
        alias micro_kernel_width = micro_kernel_shape[1]
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        var cf_tile_size = get_conv_tile_shape[filter_type](
            conv_shape.c,
            conv_shape.filter_window_flat_size(),
            micro_kernel_width,
        )

        @parameter
        if conv_attr.num_groups:
            constrained[
                conv_attr.num_groups == Dim(1),
                "Don't support grouped transposed conv for now.",
            ]()

        # Number of partitions in n, ho_wo, c, f dimensions.
        var num_threads = parallelism_level()
        var num_partitions = get_num_partitions[
            micro_kernel_height, micro_kernel_f_size
        ](num_threads, conv_shape)
        var num_tasks = num_partitions.flattened_length()

        @__copy_capture(num_partitions, cf_tile_size)
        @parameter
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

            var instance = ConvTransposedPacked[
                input_rank,
                filter_rank,
                output_rank,
                input_origin,
                filter_origin,
                output_origin,
                input_shape,
                filter_shape,
                output_shape,
                input_type,
                filter_type,
                output_type,
                conv_attr,
                elementwise_epilogue=elementwise_epilogue,
            ](
                output,
                input,
                filter,
                conv_shape,
                partition,
                task_tile_size,
            )
            instance._batch_group_loop()

        sync_parallelize[task_func](num_tasks)

    @always_inline
    fn _zero_output(self, n: Int, g: Int):
        """Zero the output buffer."""
        alias simd_size = simdwidthof[output_type]()

        var f_offset = (
            g * self.conv_shape.f_per_group() + self.partition.f_offset
        )
        var num_rows = self.conv_shape.output_image_flat_size()
        var output_ptr = (
            self.output.data + n * num_rows * self.conv_shape.f + f_offset
        )

        for _ in range(num_rows):

            @always_inline
            @parameter
            fn zero[width: Int](offset: Int):
                output_ptr.store(offset, SIMD[output_type, width](0))

            vectorize[zero, simd_size](self.partition.f_size)

            output_ptr += self.conv_shape.f

    fn _batch_group_loop(self):
        """Loop over the batch and group dimensions. Only support groups = 1 for now.
        """

        for ng in range(
            self.partition.ng_offset,
            self.partition.ng_offset + self.partition.ng_size,
        ):
            var n = ng // self.conv_shape.num_groups
            var g = ng % self.conv_shape.num_groups

            # Initialize the output buffer for current batch and group.
            self._zero_output(n, g)

            # ConvTransposed computation
            self._c_tile_loop(n, g, self.cf_tile_size[0])

            # Epilogue. Avoid putting it after register tiling for now because
            # input row i may update output row i-1. It's hard to tell if a row
            # has been updated lastly especially with stride and dilation.
            self.apply_epilogue(n, g)

    fn _c_tile_loop(self, n: Int, g: Int, c_tile_size: Int):
        """Loop over C tiles."""

        @always_inline
        @parameter
        fn c_tile_iteration(c_tile_offset: Int, c_tile_size: Int):
            self._f_tile_loop[False](n, g, c_tile_offset, c_tile_size)

        var c_offset = g * self.conv_shape.c_per_group()
        var c_round_by_tile = align_down(
            (self.conv_shape.c_per_group() - 1), c_tile_size
        )

        # Update c tiles before the last one.
        for c_tile_offset in range(c_offset, c_round_by_tile, c_tile_size):
            self._f_tile_loop[False](n, g, c_tile_offset, c_tile_size)

        # Update the last c tile with fusion
        var c_round_by_tile_residual = (
            self.conv_shape.c_per_group() - c_round_by_tile
        )
        self._f_tile_loop[True](
            n,
            g,
            c_offset + c_round_by_tile,
            c_round_by_tile_residual,
        )

    fn _f_tile_loop[
        last_c_tile: Bool
    ](self, n: Int, g: Int, c_tile_offset: Int, c_tile_size: Int):
        """Loop over F tiles."""
        alias micro_kernel_width = get_direct_conv_micro_kernel_width()
        alias micro_kernel_height = get_direct_conv_micro_kernel_height()
        alias simd_size = simdwidthof[output_type]()
        alias micro_kernel_f_size = micro_kernel_width * simd_size

        @always_inline
        @parameter
        fn f_tile_iteration[size: Int](f_tile_offset: Int, f_tile_size: Int):
            self.input_space_loop[
                micro_kernel_height, size // simd_size, False, last_c_tile
            ](n, f_tile_offset, f_tile_size, c_tile_offset, c_tile_size)

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
            self.input_space_loop[micro_kernel_height, 1, True, last_c_tile](
                n,
                group_f_end_align_simd,
                simd_size,
                c_tile_offset,
                c_tile_size,
            )

    @always_inline
    fn input_space_loop[
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
        var g = self.conv_shape.f_to_group(f_tile_offset)

        # Filter pointer to the current cf tile offset location.
        var filter_ptr: UnsafePointer[Scalar[filter_type]]

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
            + self.conv_shape.c_in_group(c_tile_offset) * micro_kernel_f_size
        )

        # Pointer to input and output of the current sample (batch dim).
        # fmt: off
        var input_ptr  = self.input.data + c_tile_offset \
                       + self.conv_shape.input_image_flat_size() \
                       * self.conv_shape.c * n

        var output_ptr = self.output.data + f_tile_offset \
                       + self.conv_shape.output_image_flat_size() \
                       * self.conv_shape.f * n

        # fmt: on

        # Divide each input row into three parts:
        # [0, left_pad_impact_end)
        # [left_pad_impact_end, right_pad_impact_start)
        # [right_pad_impact_start, WO)
        var left_pad_impact_end = ceildiv(
            self.conv_shape.pad_w[0], self.conv_shape.stride[input_rank - 3]
        )
        var right_pad_impact_start = (
            self.conv_shape.wo()
            + self.conv_shape.pad_w[0]
            - self.conv_shape.s() * self.conv_shape.dilation[input_rank - 3]
        ) // self.conv_shape.stride[input_rank - 3] + 1
        # print("pad effect", left_pad_impact_end, right_pad_impact_start)

        @parameter
        if input_rank == 4:
            self.input_space_loop_2d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr,
                input_ptr,
                filter_ptr,
                n,
                # self.is_new_c_accum(c_tile_offset),
                False,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )
        elif input_rank == 5:
            self.input_space_loop_3d[
                micro_kernel_height,
                micro_kernel_width,
                has_residual,
                last_c_tile,
            ](
                output_ptr,
                input_ptr,
                filter_ptr,
                n,
                # self.is_new_c_accum(c_tile_offset),
                False,
                c_tile_size,
                f_tile_offset,
                f_tile_size,
                left_pad_impact_end,
                right_pad_impact_start,
            )

    @always_inline
    fn input_space_loop_2d[
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
        alias simd_size = simdwidthof[output_type]()

        for h in range(
            self.partition.ho_or_howo_offset,
            self.partition.ho_or_howo_offset + self.partition.ho_or_howo_size,
        ):
            # Compute the row index as if there is no padding. This index may be
            # < 0 while the actual output ho index is within [0, ho). In the
            # inner loops, `ho_nbr = ho + r * dilation` where r within [0, R)
            # can tell if a row is in padding i.e. ho_nbr < 0  or ho_nbr > wo-1.
            var ho = h * self.conv_shape.stride[0] - self.conv_shape.pad_h[0]

            var input_base = input + self.conv_shape.c * self.conv_shape.w() * h

            # Points output to the start of the row
            var output_base = output + self.conv_shape.f * (
                -self.conv_shape.pad_w[0] + self.conv_shape.wo() * ho
            )

            @parameter
            @always_inline
            fn work_fn[height: Int, effected_by_padding: Bool](w: Int):
                update_w_tile_2d[
                    height,
                    micro_kernel_width,
                    simd_size,
                    effected_by_padding,
                    has_residual,
                    last_c_tile,
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
                    Index(h, w),
                )

                input_base = input_base.offset(
                    height * self.conv_shape.c,
                )
                output_base = output_base.offset(
                    height * self.conv_shape.stride[1] * self.conv_shape.f
                )

            tile_middle_unswitch_boundaries[
                work_fn, VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1)
            ](
                0,
                left_pad_impact_end,
                right_pad_impact_start,
                self.conv_shape.w(),
            )

    @always_inline
    fn input_space_loop_3d[
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
        alias simd_size = simdwidthof[output_type]()

        for d in range(self.conv_shape.d()):
            var do = d * self.conv_shape.stride[0] - self.conv_shape.pad_d[0]

            for h in range(
                self.partition.ho_or_howo_offset,
                self.partition.ho_or_howo_offset
                + self.partition.ho_or_howo_size,
            ):
                # fmt: off
                var ho = h * self.conv_shape.stride[1] - self.conv_shape.pad_h[0]
                # fmt: on

                var input_base = (
                    input
                    + self.conv_shape.c
                    * self.conv_shape.w()
                    * (h + d * self.conv_shape.h())
                )

                var output_base = output + self.conv_shape.f * (
                    -self.conv_shape.pad_w[0]
                    + self.conv_shape.wo() * (ho + self.conv_shape.ho() * do)
                )

                @parameter
                @always_inline
                fn work_fn[height: Int, effected_by_padding: Bool](w: Int):
                    update_w_tile_3d[
                        height,
                        micro_kernel_width,
                        simd_size,
                        effected_by_padding,
                        has_residual,
                        last_c_tile,
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
                        Index(d, h, w),
                    )

                    input_base = input_base.offset(
                        height * self.conv_shape.c,
                    )
                    output_base = output_base.offset(
                        height * self.conv_shape.stride[2] * self.conv_shape.f
                    )

                tile_middle_unswitch_boundaries[
                    work_fn,
                    VariadicList[Int](micro_kernel_height, 5, 4, 3, 2, 1),
                ](
                    0,
                    left_pad_impact_end,
                    right_pad_impact_start,
                    self.conv_shape.w(),
                )

    @always_inline
    fn apply_epilogue(self, n: Int, g: Int):
        alias simd_size = simdwidthof[output_type]()

        var f_offset = (
            g * self.conv_shape.f_per_group() + self.partition.f_offset
        )
        var num_rows = self.conv_shape.output_image_flat_size()
        var output_base = (
            self.output.data + n * num_rows * self.conv_shape.f + f_offset
        )

        @parameter
        if elementwise_epilogue:
            alias epilogue = elementwise_epilogue.value()

            var output_ptr = output_base

            @parameter
            if input_rank == 4:  # 2D ConvTransposed.
                for ho in range(self.conv_shape.ho()):
                    for wo in range(self.conv_shape.wo()):
                        epilogue(
                            Index(n, ho, wo, f_offset), self.partition.f_size
                        )
                        output_ptr += self.conv_shape.f

            elif input_rank == 5:  # 3D ConvTransposed.
                for do in range(self.conv_shape.do()):
                    for ho in range(self.conv_shape.ho()):
                        for wo in range(self.conv_shape.wo()):
                            epilogue(
                                Index(n, do, ho, wo, f_offset),
                                self.partition.f_size,
                            )
                            output_ptr += self.conv_shape.f


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution 2D Resigter Tiling
# ===----------------------------------------------------------------------=== #


@always_inline
fn update_w_tile_2d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    output: UnsafePointer[Scalar[output_dt]],
    input: UnsafePointer[Scalar[input_dt]],
    filter: UnsafePointer[Scalar[filter_dt]],
    _init_output: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[2],
    n: Int,
    hw: IndexList[2],
):
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Output stride to neighbor point in the filter window (R, S).
    # fmt: off
    var output_stride_by_s = conv_shape.dilation[1] * conv_shape.f
    var output_stride_by_r = conv_shape.dilation[0] * conv_shape.wo() * conv_shape.f
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size

    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s

    # Filter stride in F dimension in FRSCf or RSFC
    var filter_stride = micro_kernel_f_size

    # Output coordinates
    var howo = Index(
        hw[0] * conv_shape.stride[0] - conv_shape.pad_h[0],
        hw[1] * conv_shape.stride[1] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.

    for r in range(conv_shape.r()):
        # Skip the row if it falls into padding.
        var ho_nbr = howo[0] + r * conv_shape.dilation[0]
        if ho_nbr < 0 or ho_nbr >= conv_shape.ho():
            continue

        for s in range(conv_shape.s()):
            var output_ptr = (
                output + r * output_stride_by_r + s * output_stride_by_s
            )
            var filter_ptr = (
                filter + r * filter_stride_by_r + s * filter_stride_by_s
            )

            @parameter
            if effected_by_padding:
                constrained[
                    micro_kernel_height == 1,
                    "The tile must only have 1 point when effected bypadding.",
                ]()
                var wo_nbr = howo[1] + s * conv_shape.dilation[1]
                if wo_nbr < 0 or wo_nbr >= conv_shape.wo():
                    continue

            accumulate_wo_tile[
                micro_kernel_height,
                micro_kernel_width,
                simd_size,
                has_residual,
            ](
                c_tile_size,
                output_ptr,
                conv_shape.f * conv_shape.stride[1],
                input,
                conv_shape.c,
                filter_ptr,
                filter_stride,
                conv_shape.f_per_group() % simd_size,
            )


@always_inline
fn update_w_tile_3d[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    effected_by_padding: Bool,
    has_residual: Bool,
    last_c_tile: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    output: UnsafePointer[Scalar[output_dt]],
    input: UnsafePointer[Scalar[input_dt]],
    filter: UnsafePointer[Scalar[filter_dt]],
    _init_output: Bool,
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    conv_shape: ConvShape[3],
    n: Int,
    hw: IndexList[3],
):
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Output stride to neighbor point in the filter window (R, S).
    # fmt: off
    var output_stride_by_s = conv_shape.dilation[2] * conv_shape.f
    var output_stride_by_r = conv_shape.dilation[1] * conv_shape.wo() * conv_shape.f
    var output_stride_by_q = conv_shape.dilation[0] * conv_shape.wo() * conv_shape.ho() * conv_shape.f
    # fmt: on

    # Filter stride when s increments by 1.
    var filter_stride_by_s = conv_shape.c_per_group() * micro_kernel_f_size
    var filter_stride_by_r = conv_shape.s() * filter_stride_by_s
    var filter_stride_by_q = conv_shape.r() * filter_stride_by_r

    # Filter stride in F dimension in FRSCf or RSFC
    var filter_stride = micro_kernel_f_size

    # Output coordinates
    var howo = Index(
        hw[0] * conv_shape.stride[0] - conv_shape.pad_d[0],
        hw[1] * conv_shape.stride[1] - conv_shape.pad_h[0],
        hw[2] * conv_shape.stride[2] - conv_shape.pad_w[0],
    )

    # This will be all lifted to simd registers for FMA unless the micro
    # kernel is too large that spills named registers.

    for q in range(conv_shape.q()):
        var do_nbr = howo[0] + q * conv_shape.dilation[0]
        if do_nbr < 0 or do_nbr >= conv_shape.do():
            continue

        for r in range(conv_shape.r()):
            var ho_nbr = howo[1] + r * conv_shape.dilation[1]
            if ho_nbr < 0 or ho_nbr >= conv_shape.ho():
                continue

            for s in range(conv_shape.s()):
                var output_ptr = output.offset(
                    q * output_stride_by_q
                    + r * output_stride_by_r
                    + s * output_stride_by_s
                )
                var filter_ptr = filter.offset(
                    q * filter_stride_by_q
                    + r * filter_stride_by_r
                    + s * filter_stride_by_s
                )

                @parameter
                if effected_by_padding:
                    constrained[micro_kernel_height == 1]()
                    var wo_nbr = howo[2] + s * conv_shape.dilation[2]
                    if wo_nbr < 0 or wo_nbr >= conv_shape.wo():
                        continue

                accumulate_wo_tile[
                    micro_kernel_height,
                    micro_kernel_width,
                    simd_size,
                    has_residual,
                ](
                    c_tile_size,
                    output_ptr,
                    conv_shape.f * conv_shape.stride[2],
                    input,
                    conv_shape.c,
                    filter_ptr,
                    filter_stride,
                    conv_shape.f_per_group() % simd_size,
                )


@always_inline
fn accumulate_wo_tile[
    micro_kernel_height: Int,
    micro_kernel_width: Int,
    simd_size: Int,
    partial_load: Bool,
    output_dt: DType,
    input_dt: DType,
    filter_dt: DType,
](
    c_tile_size: Int,
    output: UnsafePointer[Scalar[output_dt]],
    output_stride: Int,
    input: UnsafePointer[Scalar[input_dt]],
    input_stride: Int,
    filter: UnsafePointer[Scalar[filter_dt]],
    filter_stride: Int,
    partial_load_size: Int,
):
    var acc = _Accumulator[
        output_dt, micro_kernel_height, micro_kernel_width, simd_size
    ]()

    acc.load[partial_load=partial_load](
        output,
        output_stride,
        partial_load_size,
    )

    acc.accumulate[prefetch_offset=4, partial_load_b=partial_load](
        c_tile_size,
        input,
        input_stride,
        filter,
        filter_stride,
        partial_load_size,
    )

    acc.store[partial_store=partial_load](
        output,
        output_stride,
        partial_load_size,
    )


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution Filter Packing                                 #
# ===----------------------------------------------------------------------=== #


@always_inline
fn _get_group_filter_base(
    packed_filter: NDBuffer, group_idx: Int, f_per_group: Int
) -> UnsafePointer[
    Scalar[packed_filter.type], address_space = packed_filter.address_space
]:
    # TODO: support groups > 1.
    return packed_filter.data


@always_inline
fn pack_filter_shape(
    filter: NDBuffer, num_groups: Int
) -> IndexList[filter.rank + 1]:
    """
    Compute the output shape of transposed convolution filter packing.

    Args:
        filter: The filter to be packed.
        num_groups: The number of groups in the convolution.

    Returns:
        The output shape.
    """

    alias simd_size = simdwidthof[filter.type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Filter is in RSFC layout. The 2nd last dim is F.
    var F = filter.dim[filter.rank - 2]()

    debug_assert(
        F % num_groups == 0,
        "number of filters F must be divisible by number of groups",
    )
    var F_per_group = F // num_groups

    # FRSCf layout.
    var packed_shape = IndexList[filter.rank + 1]()
    packed_shape[0] = num_groups * ceildiv(F_per_group, micro_kernel_f_size)
    packed_shape[filter.rank] = micro_kernel_f_size
    # Input channel
    packed_shape[filter.rank - 1] = filter.dim[filter.rank - 1]()

    @parameter
    for i in range(filter.rank - 2):
        packed_shape[i + 1] = filter.dim[i]()

    return packed_shape


@always_inline
fn pack_filter(
    filter: NDBuffer,
    packed_filter: NDBuffer[mut=True, *_, **_],
    num_groups: Int,
):
    """This packs the filter form RSFC to FRSCf."""

    alias simd_size = simdwidthof[filter.type]()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size = micro_kernel_width * simd_size

    # Product of filter window dims.
    var window_dims_prod = 1

    @parameter
    for i in range(filter.rank - 2):
        window_dims_prod *= filter.dim[i]()

    var C = filter.dim[filter.rank - 1]()
    var F = filter.dim[filter.rank - 2]()
    var F_per_group = F // num_groups

    packed_filter.zero()

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
        @__copy_capture(group_start, C, F_per_group, F)
        @parameter
        fn pack[f_tile_size: Int](f_tile_start: Int):
            var packed_filter_ptr = (
                group_start + f_tile_start * window_dims_prod * C
            )

            # Consider a point in filter window as a neighbor to input point.
            for nbr in range(window_dims_prod):
                var filter_ptr = (
                    filter.data
                    + nbr * F * C
                    + (g * F_per_group + f_tile_start) * C
                )

                for _ in range(C):
                    for f in range(f_tile_size):
                        packed_filter_ptr.store(
                            f,
                            filter_ptr.load(f * C).cast[
                                packed_filter_ptr.type.dtype
                            ](),
                        )

                    # FRSCf layout. Increment by `f` in c loop.
                    packed_filter_ptr += f_tile_size
                    # RSFC layout. Increment by 1 in c loop.
                    filter_ptr += 1

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
                group_start + F_round_by_simd * window_dims_prod * C
            )

            for nbr in range(window_dims_prod):
                var filter_ptr = (
                    filter.data
                    + nbr * F * C
                    + (g * F_per_group + F_round_by_simd) * C
                )

                for _ in range(C):
                    for f in range(residual):
                        packed_filter_ptr.store(
                            f,
                            filter_ptr.load(f * C).cast[
                                packed_filter_ptr.type.dtype
                            ](),
                        )

                    filter_ptr += 1
                    packed_filter_ptr = packed_filter_ptr + simd_size


# ===----------------------------------------------------------------------=== #
# Direct Transposed Convolution API to MOGG                                    #
# ===----------------------------------------------------------------------=== #


fn conv_transposed[
    input_rank: Int,
    filter_rank: Int,
    input_shape: DimList,
    filter_shape: DimList,
    output_shape: DimList,
    input_type: DType,
    filter_type: DType,
    output_type: DType,
    filter_packed: Bool,
    filter_is_cfrs: Bool,
    lambdas_have_fusion: Bool,
    elementwise_lambda: fn[type: DType, rank: Int, width: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
](
    output: NDBuffer[mut=True, output_type, input_rank, _, output_shape],
    input: NDBuffer[input_type, input_rank, _, input_shape],
    filter: NDBuffer[mut=True, filter_type, filter_rank, _, filter_shape],
    stride: IndexList[input_rank - 2],
    dilation: IndexList[input_rank - 2],
    pad_d: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        # fmt: off
        return String(
            trace_arg("input", input),
            ";", trace_arg("filter", filter),
            ";", trace_arg("output", output),
            ";group=1",
            ";stride=", stride,
            ";padding_d=", Index(0, 0),
            ";padding_h=", pad_h,
            ";padding_w=", pad_w,
        )
        # fmt: on

    constrained[not filter_is_cfrs, "Filter layout CFRS is not supported"]()

    with Trace[TraceLevel.OP, target = StaticString("cpu")](
        "conv_transposed",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        alias packed_filter_rank = filter_rank if filter_packed else filter_rank + 1

        var packed_filter_ptr = filter.data
        var packed_filter_shape: IndexList[packed_filter_rank]

        # If filter is not packed, we have to pack it before the kernel.
        @parameter
        if not filter_packed:
            # Only support single group.
            packed_filter_shape = rebind[IndexList[packed_filter_rank]](
                pack_filter_shape(filter, 1)
            )
            packed_filter_ptr = UnsafePointer[Scalar[filter.type]].alloc(
                packed_filter_shape.flattened_length()
            )
        else:
            packed_filter_shape = rebind[IndexList[packed_filter_rank]](
                filter.get_shape()
            )

        var packed_filter = NDBuffer[filter_type, packed_filter_rank](
            packed_filter_ptr, packed_filter_shape
        )

        @parameter
        if not filter_packed:
            pack_filter(filter, packed_filter, 1)

        alias conv_attr = ConvInfoStatic[input_rank - 2]()

        var conv_shape = get_conv_shape[input_rank - 2, True](
            output,
            input,
            packed_filter,
            stride,
            dilation,
            pad_d,
            pad_h,
            pad_w,
            1,
        )

        # The closure updates a row segment of the output.
        @always_inline
        @parameter
        fn elementwise_epilogue[
            rank: Int
        ](coords: IndexList[rank], f_size: Int):
            alias simd_size = simdwidthof[output_type]()

            @always_inline
            @parameter
            fn body[width: Int](idx: Int):
                # Cooridates of the current index.
                var curr_coords = rebind[IndexList[input_rank]](coords)
                curr_coords[input_rank - 1] += idx

                var vec = output.load[width=width](curr_coords)
                elementwise_lambda(curr_coords, vec)

            vectorize[body, simd_size](f_size)

        ConvTransposedPacked[
            input_rank,
            packed_filter_rank,
            input_rank,
            input.origin,
            packed_filter.origin,
            output.origin,
            input_shape,
            DimList.create_unknown[packed_filter_rank](),
            output_shape,
            input_type,
            filter_type,
            output_type,
            conv_attr,
            OptionalReg[elementwise_epilogue_type](
                elementwise_epilogue
            ) if lambdas_have_fusion else None,
        ].run(output, input, packed_filter, conv_shape)

        @parameter
        if not filter_packed:
            packed_filter_ptr.free()


# ===----------------------------------------------------------------------=== #
# cuDNN Convolution Backward Data (i.e., Transposed Convolution) Helper        #
# ===----------------------------------------------------------------------=== #


fn conv_backward_data_cudnn[
    filter_dim: DimList,
    diff_dim: DimList,
    grad_dim: DimList,
    data_type: DType,
](
    filter: UnsafePointer[Scalar[data_type]],  # w
    diff: UnsafePointer[Scalar[data_type]],  # dy
    grad: UnsafePointer[Scalar[data_type]],  # dx (output)
    stride: IndexList[2],
    dilation: IndexList[2],
    padding: IndexList[2],
    ctx: DeviceContext,
) raises:
    """
    Thin wrapper around `cudnnConvolutionBackwardData` for a 2-D NHWC layout.

    Arguments mirror `conv_cudnn` for ease of use:
      • `filter`  -> pointer to RSCK filter (interpreted as K,C,H,W in NCHW).
      • `diff`    -> pointer to output gradient (dy) in NHWC order.
      • `grad`    -> pointer to input gradient buffer (dx) in NHWC order.
      • `stride`/`dilation`/`padding` -> spatial parameters (H,W order).
      • `ctx`     -> device context that provides the CUDA stream.
    """

    var cudnn_handle = UnsafePointer[cudnnContext]()
    check_cudnn_error(cudnnCreate(UnsafePointer(to=cudnn_handle)))

    # basically, vibes are that a cuda handle is the gateway to using cudnn
    # we want all the work from that handle to be done on a separate stream
    # than the main stream, otherwise, everyhting goes on main stream and
    # slows down the whole thing. binding handle to stream unclocks parallelism, and now
    # 2 handles , wth 2 separate functions, can work at same time.
    check_cudnn_error(cudnnSetStream(cudnn_handle, CUDA(ctx.stream())))

    # ---------------- Tensor / filter descriptors -------------------------
    var filter_desc = UnsafePointer[cudnnFilterStruct]()
    check_cudnn_error(
        cudnnCreateFilterDescriptor(UnsafePointer(to=filter_desc))
    )
    check_cudnn_error(
        cudnnSetFilter4dDescriptor(
            filter_desc,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            filter_dim.get[0](),  # K (out channels)
            filter_dim.get[1](),  # C (in channels)
            filter_dim.get[2](),  # H (kernel height)
            filter_dim.get[3](),  # W (kernel width)
        )
    )

    var dy_desc = UnsafePointer[cudnnTensorStruct]()
    check_cudnn_error(cudnnCreateTensorDescriptor(UnsafePointer(to=dy_desc)))
    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            dy_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            diff_dim.get[0](),  # N
            diff_dim.get[1](),  # C_in
            diff_dim.get[2](),  # H_out
            diff_dim.get[3](),  # W_out
        )
    )

    var dx_desc = UnsafePointer[cudnnTensorStruct]()
    check_cudnn_error(cudnnCreateTensorDescriptor(UnsafePointer(to=dx_desc)))
    check_cudnn_error(
        cudnnSetTensor4dDescriptor(
            dx_desc,
            cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
            grad_dim.get[0](),  # N
            grad_dim.get[1](),  # C_out
            grad_dim.get[2](),  # H
            grad_dim.get[3](),  # W
        )
    )

    var conv_desc = UnsafePointer[cudnnConvolutionStruct]()
    check_cudnn_error(
        cudnnCreateConvolutionDescriptor(UnsafePointer(to=conv_desc))
    )
    check_cudnn_error(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            padding[0],
            padding[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION,
            cudnnDataType_t.CUDNN_DATA_FLOAT,
        )
    )

    # ---------------- Algorithm & workspace -------------------------------
    var algo = cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0

    # For now, use no workspace since UnsafePointer.alloc() only allocates host memory,
    var workspace_bytes = Int(0)
    var workspace_ptr = UnsafePointer[Int8]()

    var alpha = Float32(1.0)
    var beta = Float32(0.0)

    check_cudnn_error(
        cudnnConvolutionBackwardData(
            cudnn_handle,
            UnsafePointer(to=alpha).bitcast[NoneType](),
            filter_desc,
            filter.bitcast[NoneType](),
            dy_desc,
            diff.bitcast[NoneType](),
            conv_desc,
            algo,
            workspace_ptr.bitcast[NoneType](),
            workspace_bytes,
            UnsafePointer(to=beta).bitcast[NoneType](),
            dx_desc,
            grad.bitcast[NoneType](),
        )
    )

    # ---------------- Cleanup ---------------------------------------------
    if workspace_ptr:
        workspace_ptr.free()
    check_cudnn_error(cudnnDestroyTensorDescriptor(dx_desc))
    check_cudnn_error(cudnnDestroyConvolutionDescriptor(conv_desc))
    check_cudnn_error(cudnnDestroyTensorDescriptor(dy_desc))
    check_cudnn_error(cudnnDestroyFilterDescriptor(filter_desc))
    check_cudnn_error(cudnnDestroy(cudnn_handle))
