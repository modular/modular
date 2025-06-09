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

from collections.string import StaticString
from math import align_down, ceildiv, exp, exp2, log
from sys import alignof, is_amd_gpu, is_nvidia_gpu, simdwidthof

import gpu.warp as warp
from algorithm import sync_parallelize, vectorize
from algorithm._gpu.reduction import block_reduce, row_reduce
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _reduce_generator,
)
from bit import log2_floor
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import WARP_SIZE, barrier, block_idx, grid_dim, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.host import DeviceAttribute, DeviceContext
from gpu.host.info import is_cpu, is_gpu
from gpu.memory import AddressSpace
from layout._utils import idx2crd
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from layout.tensor_core import get_fragment_size
from memory import UnsafePointer, stack_allocation
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils import IndexList, StaticTuple
from utils.index import Index, product
from utils.numerics import get_accum_type, min_or_neg_inf

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn reduce_add_simd[
    simd_width: Int,
    step_simd_width: Int,
    type: DType,
](
    mut scalar: Scalar[type],
    mut vector: SIMD[type, simd_width],
    val: SIMD[type, step_simd_width],
):
    """This functions adds val to either the scalar value or the vector value
    depending on the step_simd_width. This is useful when the simd_width varies
    between iterations as in vectorize.
    """

    @parameter
    if step_simd_width == 1:
        # When the step_simd_width is 1, then we add to the scalar value.
        scalar += val[0]
    else:
        # When the step_simd_Width is the same as the simd_width, then we add to
        # the vector value.
        vector += rebind[SIMD[type, simd_width]](val)


@always_inline
fn sub(x: SIMD, y: __type_of(x)) -> __type_of(x):
    return x - y


@always_inline
fn mul(x: SIMD, y: __type_of(x)) -> __type_of(x):
    return x * y


@always_inline
fn identity(x: SIMD) -> __type_of(x):
    return x


@always_inline
fn reciprocal(x: SIMD) -> __type_of(x):
    return 1 / x


@always_inline
fn _exp_concrete(x: SIMD) -> __type_of(x):
    """The concrete implementation of the exp function.

    This is a helper function that is used to provide a concrete implementation
    of the exp function. This is necessary because exp uses the _Expable trait
    and mojo cannot disambiguate between the different exp functions otherwise.
    """
    return exp(x)


@always_inline
fn _exp2_concrete(x: SIMD) -> __type_of(x):
    """The concrete implementation of the exp2 function."""
    return exp2(x)


# ===-----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===-----------------------------------------------------------------------===#


fn _softmax_2_pass_step1[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](input: NDBuffer[type, 1, _, buffer_size]) -> StaticTuple[Scalar[type], 2]:
    # STEP 1: find the runningMax and runningSum in each batch.
    #   runningMax = -∞
    #   runningSum = 0
    #   STAGE 1:
    #   for i = 0 to N do
    #     newMax = max(runningMax, Input[i])
    #     runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
    #     runningMax = newMax
    #   end for
    #   return runningMax, runningSum

    var running_max_vec = SIMD[type, simd_width](min_or_neg_inf[type]())
    var running_sum_vec = SIMD[type, simd_width](0)

    # TODO: Because vectorize cannot currently capture values from outside
    # scope, we therefore replicate the logic of Functional.vectorize here.
    # In the future (once we have non-isolated-from-above regions) we can
    # just reuse the Functional.vectorize code.
    var length = len(input)
    var vector_end = align_down(length, simd_width)

    for i in range(0, vector_end, simd_width):
        var simd_elem = input.load[width=simd_width](i)
        var new_max_vec = SIMD[type, simd_width](
            max(running_max_vec, simd_elem).reduce_max()
        )
        running_sum_vec = running_sum_vec * exp(
            running_max_vec - new_max_vec
        ) + exp(simd_elem - new_max_vec)
        running_max_vec = new_max_vec

    var running_max = running_max_vec.reduce_max()
    var running_sum = running_sum_vec.reduce_add()

    for i in range(vector_end, length):
        var elem = input[i]
        var new_max = max(running_max, elem)
        running_sum = running_sum * exp(running_max - new_max) + exp(
            elem - new_max
        )
        running_max = new_max

    return StaticTuple[Scalar[type], 2](running_max[0], running_sum[0])


fn _softmax_2_pass_step2[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
](
    output: NDBuffer[mut=True, type, 1, _, buffer_size],
    input: NDBuffer[type, 1, _, buffer_size],
    running_max: Scalar[type],
    running_sum: Scalar[type],
):
    # Step 2:
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    @always_inline
    @parameter
    fn _step_2[simd_width: Int](idx: Int):
        var running_max_simd = SIMD[type, simd_width](running_max)
        var running_sum_simd = SIMD[type, simd_width](running_sum)
        var input_val = input.load[width=simd_width](idx)
        output.store[width=simd_width](
            idx,
            exp(input_val - running_max_simd) / running_sum_simd,
        )

    vectorize[_step_2, simd_width, unroll_factor=unroll_factor](len(output))


fn softmax_2_pass[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](
    output: NDBuffer[mut=True, type, 1, _, buffer_size],
    input: NDBuffer[type, 1, _, buffer_size],
):
    """Performs an unbatched softmax on an input tensor using the two-pass
    online algorithm.

    The unbatched two-pass online softmax is described in "Online
    normalizer calculation for softmax" (https://arxiv.org/abs/1805.02867) and
    "A full-stack search technique for domain optimized deep learning
    accelerators" (https://dl.acm.org/doi/abs/10.1145/3503222.3507767) and is
    defined as:

    procedure SoftmaxUnbatched(InputInput)
      runningMax = -∞
      runningSum = 0
      STAGE 1:
      for i = 0 to N do
        newMax = max(runningMax, Input[i])
        runningSum = runningSum*exp(runningMax-newMax) + exp(Input[i]-newMax)
        runningMax = newMax
      end for
      for i = 0 to N do
        Output[i] = exp(Input[i] - runningMax) / runningSum
      end for

    Parameters:
        simd_width: The simd_width to use in vectorization.
        buffer_size: The size of the input and output buffers.
        type: The type of the input and output buffers.

    Args:
        output: The output buffer in which to store the softmax values.
        input: The input buffer used to compute the softmax.
    """

    var running_info = _softmax_2_pass_step1[simd_width, buffer_size, type](
        input
    )

    var running_max = running_info[0]
    var running_sum = running_info[1]

    alias unroll_factor = 8  # TODO: search
    _softmax_2_pass_step2[simd_width, unroll_factor, buffer_size, type](
        output, input, running_max, running_sum
    )


# ===-----------------------------------------------------------------------===#
# Softmax 3 Pass
# ===-----------------------------------------------------------------------===#


fn _softmax_3_pass_step_2[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [_] -> SIMD[
        type, _simd_width
    ],
    pre_update_func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
    post_update_func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
](
    output: NDBuffer[mut=True, type, 1, _, buffer_size],
    max_val: Scalar[type],
) -> Scalar[type]:
    # STEP 2: compute for each batch
    # for i = 0 to N do
    #   Output[i] = pre_update_func(Input[i] - max_val)
    #   accum += post_update_func(Output[i])
    # end for
    alias outer_simd_width = simd_width

    var accum_scalar: Scalar[type] = 0
    var accum_simd: SIMD[type, outer_simd_width] = 0

    @always_inline
    @parameter
    fn step_2[simd_width: Int](idx: Int):
        var vin = input_fn_1d[simd_width](idx)
        var elem = vin - SIMD[type, simd_width](max_val)

        elem = pre_update_func[type, simd_width](elem)
        output.store[width=simd_width](idx, elem)
        elem = post_update_func[type, simd_width](elem)
        reduce_add_simd[outer_simd_width, simd_width, type](
            accum_scalar, accum_simd, elem
        )

    vectorize[step_2, simd_width, unroll_factor=unroll_factor](len(output))
    # Reduce the values from both the scalar and vector accum.
    return accum_scalar + accum_simd.reduce_add()


fn _softmax_3_pass_step_3[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
    accum_proc_func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
    accum_apply_func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
](output: NDBuffer[mut=True, type, 1, _, buffer_size], accum: Scalar[type]):
    # STEP 3: normalize each batch
    # accum = accum_proc_func(accum)
    # for i = 0 to N do
    #   accum_apply_func(Output[b, i], accum)
    # end for
    var accum_proc = accum_proc_func[type, 1](accum)

    @always_inline
    @__copy_capture(accum_proc)
    @parameter
    fn step_3[simd_width: Int](idx: Int):
        var accum_simd = SIMD[type, simd_width](accum_proc)
        var elem = output.load[width=simd_width](idx)
        elem = accum_apply_func[type, simd_width](elem, accum_simd)
        output.store[width=simd_width](idx, elem)

    vectorize[step_3, simd_width, unroll_factor=unroll_factor](len(output))


fn _softmax_3_pass_base[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [_] -> SIMD[
        type, _simd_width
    ],
    step2_pre_update_func: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) -> SIMD[type, width],
    step2_post_update_func: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) -> SIMD[type, width],
    step3_accum_proc_func: fn[type: DType, width: Int] (
        SIMD[type, width]
    ) -> SIMD[type, width],
    step3_accum_apply_func: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
](output: NDBuffer[mut=True, type, 1, _, buffer_size]) raises:
    """Performs an unbatched three-pass softmax. The actual behavior of each
    step can be different between the (regular) softmax and logsoftmax.

    Parameters:
        simd_width: The simd_width to use in vectorization.
        buffer_size: The size of the input and output buffers.
        type: The type of the input and output buffers.
        input_fn_1d: The elementwise input lambda.
        step2_pre_update_func: Pre update function.
        step2_post_update_func: Post update function.
        step3_accum_proc_func: Pre accumulation function.
        step3_accum_apply_func: Post accumulation function.

    Args:
        output: The output buffer in which to store the softmax values.
    """
    # STEP 1 - Calculate max
    # Allocate buffer for max_val
    var max_buff = NDBuffer[type, 1, MutableAnyOrigin, 1].stack_allocation()

    # Use _reduce_generator to fuse input lambda with max-reduction
    # Reduce function
    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return max(v1, v2)

    # Input function
    # Translate given input lambda from 1d to Nd because _reduce_generator
    # needs Nd.
    @parameter
    @always_inline
    fn input_fn[
        _type: DType, _width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[_type, _width]:
        constrained[_rank == 1]()
        return rebind[SIMD[_type, _width]](input_fn_1d[_width](coords[0]))

    # Output function
    @parameter
    @always_inline
    fn output_fn[
        _type: DType, _width: Int, _rank: Int
    ](coords: IndexList[_rank], val: SIMD[_type, _width]):
        constrained[_rank == 1]()
        max_buff[0] = val.reduce_max().cast[type]()

    # Generate fused input-reduction
    _reduce_generator[
        input_fn,
        output_fn,
        reduce_impl,
        single_thread_blocking_override=True,
    ](
        IndexList[1](len(output)),
        init=Scalar[type].MIN,
        reduce_dim=0,
    )

    var max_val = max_buff[0]

    # STEP 2
    alias unroll_factor = 8  # TODO: search
    var accum = _softmax_3_pass_step_2[
        simd_width,
        unroll_factor,
        buffer_size,
        type,
        input_fn_1d,
        step2_pre_update_func,
        step2_post_update_func,
    ](output, max_val)

    # STEP 3
    _softmax_3_pass_step_3[
        simd_width,
        unroll_factor,
        buffer_size,
        type,
        step3_accum_proc_func,
        step3_accum_apply_func,
    ](output, accum)


fn softmax_3_pass[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    origins: OriginSet,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [origins] -> SIMD[
        type, _simd_width
    ],
](output: NDBuffer[mut=True, type, 1, _, buffer_size]) raises:
    """Performs an unbatched softmax on an input tensor using the three-pass
    algorithm.

    The unbatched three-pass softmax is defined as:
    procedure SoftmaxUnbatched(InputInput)
      maxVal = -∞
      denom = 0
      STEP 1: find the max value in each batch
      for i = 0 to N do
        maxVal = max(maxVal, Input[b, i])
      end for
      STEP 2: compute the exponential for each batch
      for i = 0 to N do
        Output[b, i] = exp(Input[b, i] - maxVal)
        denom += Output[b, i]
      end for
      STEP 3: normalize each batch
      for i = 0 to N do
        Output[b, i] /= denom
      end for

    Parameters:
        simd_width: The simd_width to use in vectorization.
        buffer_size: The size of the input and output buffers.
        type: The type of the input and output buffers.
        origins: The OriginSet of captured arguments by the input_fn_1d.
        input_fn_1d: The elementwise input lambda.

    Args:
        output: The output buffer in which to store the softmax values.
    """
    _softmax_3_pass_base[
        simd_width,
        buffer_size,
        type,
        input_fn_1d,
        exp,
        identity,
        reciprocal,
        mul,
    ](output)


# ===-----------------------------------------------------------------------===#
# LogSoftmax
# ===-----------------------------------------------------------------------===#


fn logsoftmax[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    origins: OriginSet,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [origins] -> SIMD[
        type, _simd_width
    ],
](output: NDBuffer[mut=True, type, 1, _, buffer_size]) raises:
    """Performs an unbatched logsoftmax on an input tensor using the three-pass
    algorithm.

    The unbatched three-pass softmax is defined as:
    procedure SoftmaxUnbatched(InputInput)
      maxVal = -∞
      denom = 0
      STEP 1: find the max value in each batch
      for i = 0 to N do
        maxVal = max(maxVal, Input[b, i])
      end for
      STEP 2: compute the sum of exponential of each batch
      for i = 0 to N do
        Output[b, i] = Input[b, i] - maxVal
        accum += exp(Output[b, i])
      end for
      STEP 3: normalize each batch
      for i = 0 to N do
        Output[b, i] -= log(accum)
      end for

    Parameters:
        simd_width: The simd_width to use in vectorization.
        buffer_size: The size of the input and output buffers.
        type: The type of the input and output buffers.
        origins: The OriginSet of captured arguments by the input_fn_1d.
        input_fn_1d: The elementwise input lambda.

    Args:
        output: The output buffer in which to store the softmax values.
    """
    _softmax_3_pass_base[
        simd_width, buffer_size, type, input_fn_1d, identity, exp, log, sub
    ](output)


fn logsoftmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        _
    ] -> SIMD[type, _simd_width],
](
    shape: IndexList[rank],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
) raises:
    # TODO: Add rowwise generator to de-duplicate partitioning logic between
    # softmax and logsoftmax
    if axis != rank - 1:
        raise Error("logsoftmax not supported on non-inner axis yet")

    if shape.flattened_length() == 0:
        return

    var inner_dim = output.dim[rank - 1]()
    var outer_dim = product[rank](shape, rank - 1)
    var num_workers = min(parallelism_level(), outer_dim)
    var chunk_size = ceildiv(outer_dim, num_workers)

    @parameter
    @__copy_capture(chunk_size, outer_dim, inner_dim)
    @always_inline
    fn task_func(task_id: Int) raises:
        var start_offset = task_id * chunk_size
        var end_offset = min((task_id + 1) * chunk_size, outer_dim)
        for i in range(start_offset, end_offset):
            var buffer_offset = i * inner_dim
            var output_buffer_view = NDBuffer[type, 1](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index(i, shape, rank - 1)

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lambda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            logsoftmax[simd_width, Dim(), type, __origin_of(), input_fn_1d](
                output_buffer_view
            )
            _ = indices

    sync_parallelize[task_func](num_workers)


fn logsoftmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
](
    input: NDBuffer[type, rank, _, static_shape],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
) raises:
    @parameter
    @always_inline
    fn input_fn[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        return input.load[width=_simd_width](rebind[IndexList[rank]](coords))

    logsoftmax[type, simd_width, rank, static_shape, input_fn](
        input.get_shape(), output, axis
    )


# ===-----------------------------------------------------------------------===#
# Softmax
# ===-----------------------------------------------------------------------===#


fn _softmax_cpu[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    origins: OriginSet,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        origins
    ] -> SIMD[type, _simd_width],
](
    shape: IndexList[rank],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
) raises:
    # TODO: Add rowwise generator to de-duplicate partitioning logic between
    # softmax and logsoftmax
    if axis != rank - 1:
        raise Error("softmax not supported on non-inner axis yet")

    if shape.flattened_length() == 0:
        return

    var inner_dim = output.dim[rank - 1]()
    var outer_dim = product[rank](shape, rank - 1)
    var num_workers = min(parallelism_level(), outer_dim)
    var chunk_size = ceildiv(outer_dim, num_workers)

    @__copy_capture(chunk_size, inner_dim, outer_dim)
    @parameter
    @always_inline
    fn task_func(task_id: Int) raises:
        var start_offset = task_id * chunk_size
        var end_offset = min((task_id + 1) * chunk_size, outer_dim)
        for i in range(start_offset, end_offset):
            var buffer_offset = i * inner_dim
            var output_buffer_view = NDBuffer[type, 1](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index(i, shape, rank - 1)

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lambda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            softmax_3_pass[simd_width, Dim(), type, __origin_of(), input_fn_1d](
                output_buffer_view
            )
            _ = indices

    sync_parallelize[task_func](num_workers)


# Softmax (no input lambda)
fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
](
    input: NDBuffer[type, rank, _, static_shape],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
) raises:
    @parameter
    @always_inline
    fn input_fn[
        _simd_width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[type, _simd_width]:
        return input.load[width=_simd_width](rebind[IndexList[rank]](coords))

    softmax[type, simd_width, rank, static_shape, input_fn](
        input.get_shape(), output, axis
    )


fn softmax_kernel[
    BLOCK_SIZE: Int,
    input_fn: fn[_type: DType, _simd_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing [_] -> SIMD[_type, _simd_width],
    type: DType,
    rank: Int,
    accum_type: DType = get_accum_type[type](),
](shape: IndexList[rank], output: NDBuffer[type, rank, MutableAnyOrigin]):
    alias axis = rank - 1

    var row_size = UInt(shape[axis])
    var num_rows = UInt(shape.flattened_length()) // row_size

    var max_buf = NDBuffer[
        accum_type, 1, MutableAnyOrigin, 1, address_space = AddressSpace.SHARED
    ].stack_allocation()
    var exp_sum_buf = NDBuffer[
        accum_type, 1, MutableAnyOrigin, 1, address_space = AddressSpace.SHARED
    ].stack_allocation()

    @parameter
    @always_inline
    fn _max[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return max(x, y)

    @parameter
    @always_inline
    fn _sum[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    var tid = thread_idx.x

    # grid stride loop over rows
    # each block reduces a row, which is convenient because it requires no partial
    # reductions across blocks
    for row_idx in range(block_idx.x, num_rows, grid_dim.x):
        # Step 1: compute max in row
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )
        var row_max = row_reduce[
            BLOCK_SIZE,
            input_fn,
            _max,
            type,
            1,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, Scalar[type].MIN, Int(row_size))

        if tid == 0:
            max_buf[0] = row_max
        barrier()

        row_max = max_buf[0]

        # Step 2: out[i] = exp(in[i] - max) and compute sum of out[i]
        var exp_sum = Scalar[accum_type](0)
        for row_offset in range(tid, row_size, UInt(BLOCK_SIZE)):
            row_coords[axis] = Int(row_offset)

            # loads from input_fn twice
            var val = exp(
                input_fn[type, 1, rank](row_coords).cast[accum_type]() - row_max
            )

            # TODO we're writing to and reading from global memory twice
            # we can reduce the amount of reads by keeping values local here.
            output[row_coords] = val.cast[type]()
            exp_sum += val

        var block_exp_sum = block_reduce[BLOCK_SIZE, _sum](exp_sum, 0)
        if tid == 0:
            exp_sum_buf[0] = block_exp_sum
        barrier()

        # Step 3: Normalize output
        var block_exp_sum_recip = 1 / exp_sum_buf[0]
        for row_offset in range(tid, row_size, UInt(BLOCK_SIZE)):
            row_coords[axis] = Int(row_offset)
            output[row_coords] *= block_exp_sum_recip.cast[type]()


fn _softmax_gpu[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        _
    ] -> SIMD[type, _simd_width],
](
    shape: IndexList[rank],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
    ctx: DeviceContext,
) raises:
    if axis != rank - 1:
        raise Error("softmax not supported on non-inner axis yet")

    @always_inline
    @parameter
    fn input_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_fn[width, rank](idx))

    alias BLOCK_SIZE = 128
    var num_rows = shape.flattened_length() // shape[axis]
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    alias sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)
    ctx.enqueue_function[
        softmax_kernel[BLOCK_SIZE, input_fn_wrapper, type, rank]
    ](shape, output, grid_dim=num_blocks, block_dim=BLOCK_SIZE)


fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        _
    ] -> SIMD[type, _simd_width],
    target: StaticString = "cpu",
](
    shape: IndexList[rank],
    output: NDBuffer[mut=True, type, rank, _, static_shape],
    axis: Int,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    @parameter
    fn trace_information() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP, target=target](
        "softmax",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
    ):

        @parameter
        if is_cpu[target]():
            _softmax_cpu[
                type, simd_width, rank, static_shape, __origin_of(), input_fn
            ](shape, output, axis)
        elif is_gpu[target]():
            _softmax_gpu[type, simd_width, rank, static_shape, input_fn](
                shape,
                output,
                axis,
                context.get_device_context(),
            )
        else:
            constrained[False, "unsupported target ", target]()


# ===----------------------------------------------------------------------=== #
# Online softmax in flash attention.
# ===----------------------------------------------------------------------=== #


fn _online_softmax_kernel[
    WM: Int,
    WN: Int,
    type: DType,
    layout: Layout,
    fragment_transpose: Bool = False,
](
    input: LayoutTensor[type, layout, MutableAnyOrigin],
    output: LayoutTensor[type, layout, MutableAnyOrigin],
):
    """This is only for online softmax validation, NOT a general kernel."""

    constrained[
        not fragment_transpose or (fragment_transpose and is_amd_gpu()),
        "fragment_transpose must be False on NVIDIA",
    ]()

    alias mma_shape = IndexList[3](16, 8, 8) if is_nvidia_gpu() else IndexList[
        3
    ](16, 16, 16)
    alias num_seqs = input.shape[0]()
    alias seqlen = input.shape[1]()

    constrained[
        WM == num_seqs, "Only consider WM equal to number of rows in test."
    ]()

    alias num_m_mmas = WM // mma_shape[0]
    alias num_n_mmas = WN // mma_shape[1]

    # TODO: This is a temporary hack, hopefully we can come up with a better way.
    alias mma_fragment_groups = 2 if is_nvidia_gpu() else 1

    # Each 16x8 mma tile has two 8x8 units and corresponds to 8x4 thread layout
    # in a single warp.
    alias num_mma_units = num_m_mmas * num_n_mmas * mma_fragment_groups
    alias score_layout_by_mma_unit = Layout.row_major(
        num_m_mmas * mma_fragment_groups, num_n_mmas
    )
    alias warp_layout = Layout.row_major(8, 4) if is_nvidia_gpu() else (
        Layout.col_major(16, 4) if fragment_transpose else Layout.row_major(
            4, 16
        )
    )

    # Only consider 2 iterations in this test. The number of warps is based on
    # half sequence length.
    alias num_rowwise_warps = seqlen // 2 // WN
    alias block_layout_by_warp = Layout.row_major(1, num_rowwise_warps)

    alias frag_size = get_fragment_size[mma_shape]()[2]

    var warp_id = get_warp_id()
    var lane = lane_id()

    # If we do more than 2 iterations, the first N - 2 iterations won't be
    # corrected with the right rowmax.
    var input_warp_tile0 = input.tile[WM, WN](0, Int(warp_id))
    var input_warp_tile1 = input.tile[WM, WN](
        0, Int(warp_id) + num_rowwise_warps
    )

    var output_warp_tile0 = output.tile[WM, WN](0, Int(warp_id))
    var output_warp_tile1 = output.tile[WM, WN](
        0, Int(warp_id) + num_rowwise_warps
    )

    var p = LayoutTensor[
        type,
        Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    alias fragment_layout = Layout.row_major(1, 2) if is_nvidia_gpu() else (
        Layout.row_major(1, 4) if fragment_transpose else Layout.row_major(4, 1)
    )
    alias simdwidth_row = fragment_layout.shape[0].value()
    alias simdwidth_col = fragment_layout.shape[1].value()

    @parameter
    if is_nvidia_gpu():
        p.vectorize[1, 2]().transpose().copy_from(
            input_warp_tile0.vectorize[1, 2]().distribute[warp_layout](lane)
        )
    else:
        p.vectorize[1, 4]().copy_from(
            input_warp_tile0.vectorize[
                simdwidth_row, simdwidth_col
            ]().distribute[warp_layout](lane)
        )

    var p_vecs = p.reshape[
        Layout.row_major(num_mma_units, frag_size // mma_fragment_groups)
    ]().vectorize[1, frag_size // mma_fragment_groups]()

    var o = (
        LayoutTensor[
            type,
            Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )
    var o_vecs = o.reshape[
        Layout.row_major(num_mma_units, frag_size // mma_fragment_groups)
    ]().vectorize[1, frag_size // mma_fragment_groups]()

    alias frag_num_rows = 2 if is_nvidia_gpu() else (
        1 if fragment_transpose else 4
    )
    alias row_alignment = alignof[SIMD[type, simdwidthof[type]()]]()
    var rowmax = stack_allocation[
        num_m_mmas * frag_num_rows, type, alignment=row_alignment
    ]()
    var rowsum = stack_allocation[
        num_m_mmas * frag_num_rows, type, alignment=row_alignment
    ]()

    var warp_scratch = LayoutTensor[
        type,
        Layout.row_major(2 * num_rowwise_warps, WM),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    @parameter
    for i in range(0, frag_num_rows * num_m_mmas, frag_num_rows):
        rowmax.store(i, SIMD[type, frag_num_rows](min_or_neg_inf[type]()))
        rowsum.store(i, SIMD[type, frag_num_rows](0))

    _online_softmax_iter_for_mma_output[
        type,
        score_layout_by_mma_unit,
        block_layout_by_warp,
        warp_layout,
        fragment_layout=fragment_layout,
    ](o_vecs, p_vecs, warp_scratch, rowmax, rowsum)

    # P has the softmax numerator for the first half, save it in q.
    o.copy_from(p)

    @parameter
    if is_nvidia_gpu():
        p.vectorize[1, 2]().transpose().copy_from(
            input_warp_tile1.vectorize[1, 2]().distribute[warp_layout](lane)
        )
    else:
        p.vectorize[1, 4]().copy_from(
            input_warp_tile1.vectorize[
                simdwidth_row, simdwidth_col
            ]().distribute[warp_layout](lane)
        )

    _online_softmax_iter_for_mma_output[
        type,
        score_layout_by_mma_unit,
        block_layout_by_warp,
        warp_layout,
        fragment_layout=fragment_layout,
    ](o_vecs, p_vecs, warp_scratch, rowmax, rowsum)

    # o, p has the correct softmax numerator for the 1st and 2nd half.
    # rowsum has the correct sum. Ready for correction.

    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(frag_size // mma_fragment_groups):

                @parameter
                if is_nvidia_gpu():
                    p[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                    p[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                        2 * m_mma + 1
                    ]
                    o[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                    o[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                        2 * m_mma + 1
                    ]
                else:
                    var rowsum_tensor = (
                        tb[type]()
                        .row_major[num_m_mmas, frag_num_rows]()
                        .view(rowsum)
                    )
                    p[n_mma * num_m_mmas + m_mma, i] /= rowsum_tensor[
                        m_mma, 0 if fragment_transpose else i
                    ]
                    o[n_mma * num_m_mmas + m_mma, i] /= rowsum_tensor[
                        m_mma, 0 if fragment_transpose else i
                    ]

    @parameter
    if is_nvidia_gpu():
        output_warp_tile0.vectorize[1, 2]().distribute[warp_layout](
            lane
        ).copy_from(o.vectorize[1, 2]().transpose())
        output_warp_tile1.vectorize[1, 2]().distribute[warp_layout](
            lane
        ).copy_from(p.vectorize[1, 2]().transpose())
    else:
        output_warp_tile0.vectorize[simdwidth_row, simdwidth_col]().distribute[
            warp_layout
        ](lane).copy_from(o.vectorize[1, 4]())
        output_warp_tile1.vectorize[simdwidth_row, simdwidth_col]().distribute[
            warp_layout
        ](lane).copy_from(p.vectorize[1, 4]())


@always_inline
fn _online_softmax_iter_for_mma_output[
    type: DType,
    score_layout_by_mma_unit: Layout,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    use_exp2: Bool = False,
    warp_split_k: Bool = False,
    fragment_layout: Layout = Layout.row_major(
        1, 2
    ) if is_nvidia_gpu() else Layout.row_major(4, 1),
](
    output_reg_tile: LayoutTensor[type, *_, **_],
    score_reg_tile: LayoutTensor[type, *_, **_],
    warp_scratch: LayoutTensor[type, *_, **_],
    rowmax: UnsafePointer[Scalar[type], **_],
    rowsum: UnsafePointer[Scalar[type], **_],
):
    alias num_colwise_warps = block_layout_by_warp.shape[0].value()
    alias num_rowwise_warps = block_layout_by_warp.shape[1].value()

    var tid = thread_idx.x
    var lane = lane_id()
    var warp_x = warp.broadcast(tid // WARP_SIZE) % UInt(num_rowwise_warps)

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    alias frag_type = score_reg_tile.element_type
    alias frag_num_rows = fragment_layout.shape[0].value()
    alias frag_num_cols = fragment_layout.shape[1].value()

    alias frag_is_row_vector = frag_num_rows == 1

    # Number of mma unit tiles in the score matrix.
    # 2*num_m_mmas
    alias num_colwise_tiles = score_layout_by_mma_unit.shape[0].value()
    # num_n_mmas
    alias num_rowwise_tiles = score_layout_by_mma_unit.shape[1].value()
    # The online softmax attributes for each thread's elements (fragments).
    alias num_rows_per_thread = num_colwise_tiles * frag_num_rows

    var score_frag_rowmax = (
        tb[type]().row_major[num_colwise_tiles, frag_num_rows]().local().alloc()
    )
    var score_frag_rowsum = (
        tb[type]().row_major[num_colwise_tiles, frag_num_rows]().local().alloc()
    )
    var correction = (
        tb[type]().row_major[num_colwise_tiles, frag_num_rows]().local().alloc()
    )

    var rowmax_tensor = (
        tb[type]().row_major[num_colwise_tiles, frag_num_rows]().view(rowmax)
    )
    var rowsum_tensor = (
        tb[type]().row_major[num_colwise_tiles, frag_num_rows]().view(rowsum)
    )

    # Initialize local max with the running max, and local sum with zero.
    @parameter
    for col_tile in range(num_colwise_tiles):

        @parameter
        for row in range(frag_num_rows):
            score_frag_rowmax[col_tile, row] = rowmax_tensor[col_tile, row]
            score_frag_rowsum[col_tile, row] = 0

    alias num_shuffles_per_row = log2_floor(warp_layout.shape[1].value())

    alias num_rowwise_lanes = UInt32(warp_layout.shape[1].value())
    alias num_colwise_lanes = UInt32(warp_layout.shape[0].value())
    alias rowwise_lanes_stride = UInt32(warp_layout.stride[1].value())

    alias exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    # Online softmax
    @parameter
    for col_tile in range(num_colwise_tiles):

        @parameter
        for row_tile in range(num_rowwise_tiles):
            alias tile_id = col_tile + row_tile * num_colwise_tiles

            # Assume this is a rowwise vector for now see above constraint.
            var frag = score_reg_tile[tile_id, 0]

            @parameter
            for row in range(frag_num_rows):

                @parameter
                for col in range(frag_num_cols):
                    score_frag_rowmax[col_tile, row] = max(
                        score_frag_rowmax[col_tile, row],
                        frag[col if frag_is_row_vector else row],
                    )

        @parameter
        if warp_split_k:
            # HACK: this makes a test failure go away for some reason
            barrier()

        # Every four threads have elements on the same row.
        # Reduce max for T0-T3, T4-T7, etc for nvidia
        #                T0-T15, T16-T31, etc for amd
        @parameter
        for row in range(frag_num_rows):
            score_frag_rowmax[
                col_tile, row
            ] = warp.lane_group_max_and_broadcast[
                Int(num_rowwise_lanes), stride = Int(rowwise_lanes_stride)
            ](
                score_frag_rowmax[col_tile, row]
            )

    var coords = idx2crd[warp_layout](lane)
    var lane_contains_first_column = coords[1] == 0
    var lane_row = coords[0]

    # If a row is split across multiple warps, communicate via shared memory
    # to achieve the rowwise max.
    @parameter
    if num_rowwise_warps > 1 and not warp_split_k:
        # Write per warp rowmax to shared memory.
        if lane_contains_first_column:

            @parameter
            for col_tile in range(num_colwise_tiles):

                @parameter
                for row in range(frag_num_rows):
                    var score_row_idx = (
                        col_tile * num_colwise_lanes * frag_num_rows
                        + lane_row * frag_num_rows
                        + row
                    )

                    # warp scratch has layout row_major(num_warps, num_rows). The
                    # "score_row_idx" is the idx-th row in the score matrix.
                    warp_scratch[
                        Int(warp_x), Int(score_row_idx)
                    ] = score_frag_rowmax[col_tile, row][0]

        barrier()

        # Reduce the warpwise rowmax.
        if lane_contains_first_column:

            @parameter
            for col_tile in range(num_colwise_tiles):

                @parameter
                for row in range(frag_num_rows):
                    var score_row_idx = (
                        col_tile * num_colwise_lanes * frag_num_rows
                        + lane_row * frag_num_rows
                        + row
                    )

                    @parameter
                    for row_warp in range(num_rowwise_warps):
                        score_frag_rowmax[col_tile, row] = max(
                            rebind[Scalar[type]](
                                score_frag_rowmax[col_tile, row]
                            ),
                            rebind[Scalar[type]](
                                warp_scratch[row_warp, Int(score_row_idx)]
                            ),
                        )

    # TODO: We can let all threads read shared memory in the above so that
    # we don't need to use warp shuffling.
    @parameter
    for col_tile in range(num_colwise_tiles):
        # Broadcast to 4 threads in the same row.
        @parameter
        if num_rowwise_warps > 1 and not warp_split_k:

            @parameter
            for row in range(frag_num_rows):
                score_frag_rowmax[
                    col_tile, row
                ] = warp.lane_group_max_and_broadcast[
                    Int(num_rowwise_lanes), stride = Int(rowwise_lanes_stride)
                ](
                    score_frag_rowmax[col_tile, row]
                )

        # Corrention since previous max may be updated.
        @parameter
        for row in range(frag_num_rows):
            correction[col_tile, row] = exp_function(
                rowmax_tensor[col_tile, row] - score_frag_rowmax[col_tile, row]
            )

        # Softmax numerator based on mma results.
        @parameter
        for row_tile in range(num_rowwise_tiles):
            alias tile_id = col_tile + num_colwise_tiles * row_tile

            @parameter
            if frag_is_row_vector:
                score_reg_tile[tile_id, 0] = exp_function(
                    score_reg_tile[tile_id, 0]
                    - rebind[frag_type](
                        SIMD[type, frag_num_cols](
                            score_frag_rowmax[col_tile, 0][0]
                        )
                    )
                )
            else:

                @parameter
                for row in range(frag_num_rows):
                    score_reg_tile[tile_id, 0][row] = exp_function(
                        score_reg_tile[tile_id, 0][row]
                        - score_frag_rowmax[col_tile, row][0]
                    )

        # Sum softmax numerator from a thread's fragments.
        @parameter
        for row_tile in range(num_rowwise_tiles):
            alias tile_id = col_tile + num_colwise_tiles * row_tile
            var frag = score_reg_tile[tile_id, 0]

            @parameter
            for row in range(frag_num_rows):

                @parameter
                for col in range(frag_num_cols):
                    score_frag_rowsum[col_tile, row] += frag[
                        col if frag_is_row_vector else row
                    ]

        @parameter
        for row in range(frag_num_rows):
            score_frag_rowsum[
                col_tile, row
            ] = warp.lane_group_sum_and_broadcast[
                Int(num_rowwise_lanes), stride = Int(rowwise_lanes_stride)
            ](
                score_frag_rowsum[col_tile, row]
            )

    # Reduce rowsum via shared memory.

    @parameter
    if num_rowwise_warps > 1 and not warp_split_k:
        # Write per warp rowmax to shared memory.
        if lane_contains_first_column:

            @parameter
            for col_tile in range(num_colwise_tiles):

                @parameter
                for row in range(frag_num_rows):
                    # Each thread handle two rows in the mma output.
                    var score_row_idx = (
                        col_tile * num_colwise_lanes * frag_num_rows
                        + lane_row * frag_num_rows
                        + row
                    )

                    warp_scratch[
                        warp_x + num_rowwise_warps, Int(score_row_idx)
                    ] = score_frag_rowsum[col_tile, row][0]

        # Guard writing warp_scratch
        barrier()

        # Reduce the warpwise rowsum.
        if lane_contains_first_column:

            @parameter
            for col_tile in range(num_colwise_tiles):

                @parameter
                for row in range(frag_num_rows):
                    var score_row_idx = (
                        col_tile * num_colwise_lanes * frag_num_rows
                        + lane_row * frag_num_rows
                        + row
                    )

                    score_frag_rowsum[col_tile, row] = 0

                    # Reduce rowmax. Warps in the same row do the same reduction.
                    @parameter
                    for row_warp in range(num_rowwise_warps):
                        score_frag_rowsum[col_tile, row] += rebind[
                            Scalar[type]
                        ](
                            warp_scratch[
                                row_warp + num_rowwise_warps, Int(score_row_idx)
                            ]
                        )

            # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.

        @parameter
        for col_tile in range(num_colwise_tiles):

            @parameter
            for row in range(frag_num_rows):
                # Broadcast to 4 threads in the same row.
                score_frag_rowsum[
                    col_tile, row
                ] = warp.lane_group_max_and_broadcast[
                    Int(num_rowwise_lanes), stride = Int(rowwise_lanes_stride)
                ](
                    score_frag_rowsum[col_tile, row]
                )

    alias num_output_replications = output_reg_tile.layout.shape[0].value() // (
        num_colwise_tiles * num_rowwise_tiles
    )
    # if num_output_replications != 1, then `warp_split_k` and it must equal `num_warps_n`.
    # FIXME: require `warp_split_k` when delaying inter-warp communication.
    constrained[
        num_output_replications == 1
        or num_output_replications % num_rowwise_warps == 0
        # or (warp_split_k and num_output_replications == num_rowwise_warps)
    ]()

    # if num_output_replications
    @parameter
    for k in range(num_output_replications):
        # Correct previous result
        @parameter
        for col_tile in range(num_colwise_tiles):

            @parameter
            for row_tile in range(num_rowwise_tiles):
                alias tile_id = col_tile + row_tile * num_colwise_tiles + k * num_colwise_tiles * num_rowwise_tiles

                alias output_frag_type = __type_of(output_reg_tile).element_type

                @parameter
                if frag_is_row_vector:
                    output_reg_tile[tile_id, 0] = output_reg_tile[
                        tile_id, 0
                    ] * output_frag_type(correction[col_tile, 0][0])
                else:

                    @parameter
                    for row in range(frag_num_rows):
                        output_reg_tile[tile_id, 0][row] = (
                            output_reg_tile[tile_id, 0][row]
                            * correction[col_tile, row][0]
                        )

    # Save current rowmax and rowsum
    @parameter
    for col_tile in range(num_colwise_tiles):

        @parameter
        for row in range(frag_num_rows):
            rowmax_tensor[col_tile, row] = score_frag_rowmax[col_tile, row]
            rowsum_tensor[col_tile, row] = (
                rowsum_tensor[col_tile, row] * correction[col_tile, row]
                + score_frag_rowsum[col_tile, row]
            )


# This performs a reduction after warp-level split-K for mha
# See `_online_softmax_iter_for_mma_output_split_warp` for
# the implementation of the online component that
# accumulates into separate tiles.
# `output_reg_tile` is `num_warps_n * num_m_mmas * num_n_mmas` rows.
# This performs the reduction, accumulating the `num_warps_n`
# row blocks of size `num_m_mmas * num_n_mmas` into the first row.
#
# This performns:
# m_i_x = -Inf
# for k in range(0, K): # across warps
#   m_i_x = max(m_i_x, m_i_k_{T_c-1})
# O_i_x = 0
# l_i_x_x_x 0
# for k in range(0, K): # across warps
#   c_k_x = exp(m_i_k_{T_c-1} - m_i_x)
#   O_i_x += O_i_k_{T_c-1} * c_k_x
#   l_i_x += l_i_k_{T_c-1} * c_k_x
#
# O_i = diag(l_i_x)^(-1) @ O_i_x
#
# Note that the `for k` loops are across warps (k is the index into
# the `num_warps_n` rowwise warps).
@always_inline
fn _online_softmax_iter_for_mma_output_split_warp_reduce[
    output_layout: Layout, //,
    type: DType,
    score_layout_by_mma_unit: Layout,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    WM: UInt,
    WN: UInt,
    /,
    use_exp2: Bool = False,
](
    output_reg_tile: LayoutTensor[
        type,
        output_layout,
        *_,
        address_space = AddressSpace.LOCAL, **_,
    ],
    warp_scratch: LayoutTensor[
        type, *_, address_space = AddressSpace.SHARED, **_
    ],
    o_smem_ptr_base: UnsafePointer[
        Scalar[type], address_space = AddressSpace.SHARED, **_
    ],
    rowmax: UnsafePointer[Scalar[type], **_],
    rowsum: UnsafePointer[Scalar[type], **_],
):
    # Here, we use naming conventions aligning with MHA's
    alias num_m_mmas = score_layout_by_mma_unit.shape[0].value()
    alias num_n_mmas = score_layout_by_mma_unit.shape[1].value()
    alias num_warps_m = block_layout_by_warp.shape[0].value()
    alias num_warps_n = block_layout_by_warp.shape[1].value()
    alias num_lanes_m = UInt32(warp_layout.shape[0].value())
    alias num_lanes_n = UInt32(warp_layout.shape[1].value())

    @parameter
    if num_warps_n == 1:
        return
    # Note that MHA cut the frag size in half:
    # var output_reg_vecs = output_reg_tile.tile[
    #     num_warps_n * num_m_mmas * num_n_mmas, p_frag_size // 2
    # ](0, 0).vectorize[1, p_frag_size // 2]()
    alias frag_size = output_reg_tile.element_layout.size()
    constrained[
        WM * WN == (2 * frag_size) * WARP_SIZE * num_m_mmas * num_n_mmas
    ]()
    # alias num_m_mmas = WM // MMA_M
    # alias num_n_mmas = WN // MMA_N
    # alias frag_size = MMA_M * MMA_N // WARP_SIZE
    #

    var tid = thread_idx.x
    var lane = lane_id()
    var warp_y, warp_x = divmod(tid // WARP_SIZE, UInt(num_warps_n))

    alias fragment_layout = Layout.row_major(
        1, 2
    ) if is_nvidia_gpu() else Layout.row_major(4, 1)
    alias frag_num_rows = fragment_layout.shape[0].value()

    # Write output reg to smem
    # Each warp has `num_warps_n` output register tiles
    # P(A @ B) @ C
    # `P(A @ B)` is a a `num_warps_m` x `num_warps_n` grid of warp tiles.
    # `C` is partitioned into a `num_warps_n` x `num_warps_n` grid of warp tiles
    #
    # When we don't `split_k_warp`, `P(A @ B)` is copied to smem, so that a warp tile
    # for `D = P(A @ B) @ C` can iterate across all columns of `P(A @ B)`.
    #
    # However, with `split_k_warp`, we skip this copy to smem.
    # Instead, for each `num_warps_n`, they calculate a row of `D`,
    # corresponding to their local columns `P(A @ B)`/rows `C`.
    # We must then perform the reduction afterwards.
    # First, each warp writes the parts other warps need to smem.
    #
    # o_smem is implicitly partitioned into a 5d array:
    # num_warps_m x num_warps_n x (num_warps_n - 1) x
    #    (num_m_mmas * num_n_mmas) x frag_size
    # The axis are:
    # 0. warp_m: No communication across `warps_m` is needed, so we offset the
    #    smem ptr immediately rather than representing this explicitly.
    # 1. warp_n: currently local to a warp, corresponding to axis 0 of
    #    `output_reg_tile`. We iterate across this when writing, and keep it
    #    constant when reducing.
    # 2. warp_n - 1: the other warp_n - 1 column tiles of the answer. We keep it
    #    constant when writing, and iterate across it when reducing.
    # 3-4. ((WM*WN)//frag_size) x frag_size: the two trailing dimensions of
    #    output_reg_tile
    alias warp_tile_size = WM * WN  # ((WM*WN)//frag_size) x frag_size
    alias row_warp_tile_size = (num_warps_n - 1) * warp_tile_size
    # Makes sure arithmetic is optimized away when `num_warps_m == 1`.
    var o_smem_ptr = (
        o_smem_ptr_base
        + warp_y * (num_warps_n - 1) * row_warp_tile_size if num_warps_m
        > 1 else o_smem_ptr_base
    )

    # NOTE: we must ensure that `output_reg_tile` is only ever indexed by constants.
    var out_reg_tile = output_reg_tile.tile[num_m_mmas * num_n_mmas, 1](0, 0)

    alias o_smem_layout = Layout.row_major(
        WM * WN // (2 * frag_size), frag_size
    )

    alias exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    var interwarp_frag_rowmax = (
        tb[type]().row_major[num_m_mmas, frag_num_rows]().local().alloc()
    )
    var interwarp_frag_rowsum = (
        tb[type]().row_major[num_m_mmas, frag_num_rows]().local().alloc()
    )
    var correction = (
        tb[type]().row_major[num_m_mmas, frag_num_rows]().local().alloc()
    )
    var rowmax_tensor = (
        tb[type]().row_major[num_m_mmas, frag_num_rows]().view(rowmax)
    )
    var rowsum_tensor = (
        tb[type]().row_major[num_m_mmas, frag_num_rows]().view(rowsum)
    )
    # corrections across warps
    # Write per warp rowmax to shared memory.
    if lane % num_lanes_n == 0:

        @parameter
        for col_tile in range(num_m_mmas):

            @parameter
            for row in range(frag_num_rows):
                var score_row_idx = (
                    col_tile * num_lanes_m
                    + (lane // num_lanes_n) * frag_num_rows
                    + row
                )
                # warp scratch has layout row_major(num_warps, num_rows). The
                # "score_row_idx" is the idx-th row in the score matrix.
                warp_scratch[
                    Int(warp_x) + num_warps_n, Int(score_row_idx)
                ] = rowmax_tensor[col_tile, row][0]

    barrier()

    # Reduce the warpwise rowmax.
    if lane % num_lanes_n == 0:

        @parameter
        for col_tile in range(num_m_mmas):

            @parameter
            for row in range(frag_num_rows):
                var score_row_idx = (
                    col_tile * num_lanes_m
                    + (lane // num_lanes_n) * frag_num_rows
                    + row
                )

                interwarp_frag_rowmax[col_tile, row] = rebind[Scalar[type]](
                    warp_scratch[num_warps_n, Int(score_row_idx)]
                )

                @parameter
                for row_warp in range(1, num_warps_n):
                    interwarp_frag_rowmax[col_tile, row] = max(
                        rebind[Scalar[type]](
                            interwarp_frag_rowmax[col_tile, row]
                        ),
                        rebind[Scalar[type]](
                            warp_scratch[
                                row_warp + num_warps_n, Int(score_row_idx)
                            ]
                        ),
                    )

    @parameter
    for col_tile in range(num_m_mmas):
        # Broadcast to 4 threads in the same row.
        @parameter
        if num_warps_n > 1:

            @parameter
            for row in range(frag_num_rows):
                interwarp_frag_rowmax[
                    col_tile, row
                ] = warp.lane_group_max_and_broadcast[Int(num_lanes_n)](
                    interwarp_frag_rowmax[col_tile, row]
                )

        # Corrention since previous max may be updated.
        @parameter
        for row in range(frag_num_rows):
            correction[col_tile, row] = exp_function(
                rowmax_tensor[col_tile, row]
                - interwarp_frag_rowmax[col_tile, row]
            )

    if lane % num_lanes_n == 0:

        @parameter
        for col_tile in range(num_m_mmas):

            @parameter
            for row in range(frag_num_rows):
                var score_row_idx = (
                    col_tile * num_lanes_m
                    + (lane // num_lanes_n) * frag_num_rows
                    + row
                )
                var c = rebind[Scalar[type]](correction[col_tile, row])
                warp_scratch[Int(warp_x), Int(score_row_idx)] = (
                    0.0 if c == 0.0 else rowsum_tensor[col_tile, row][0] * c
                )

    barrier()

    # Reduce the warpwise rowsum.
    if lane % num_lanes_n == 0:

        @parameter
        for col_tile in range(num_m_mmas):

            @parameter
            for row in range(frag_num_rows):
                var score_row_idx = (
                    col_tile * num_lanes_m
                    + (lane // num_lanes_n) * frag_num_rows
                    + row
                )
                interwarp_frag_rowsum[col_tile, row] = rebind[Scalar[type]](
                    warp_scratch[0, Int(score_row_idx)]
                )

                # Reduce rowmax. Warps in the same row do the same reduction.
                @parameter
                for row_warp in range(1, num_warps_n):
                    interwarp_frag_rowsum[col_tile, row] += rebind[
                        Scalar[type]
                    ](warp_scratch[row_warp, Int(score_row_idx)])

        # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.

    @parameter
    for col_tile in range(num_m_mmas):

        @parameter
        for row in range(frag_num_rows):
            # Broadcast to 4 threads in the same row.
            interwarp_frag_rowsum[
                col_tile, row
            ] = warp.lane_group_max_and_broadcast[
                # interwarp_frag_rowsum[col_tile, row] = lane_group_sum_and_broadcast[
                Int(num_lanes_n)
            ](
                interwarp_frag_rowsum[col_tile, row]
            )

    var output = output_reg_tile.split[num_warps_n, axis=0]()

    @parameter
    for col_tile in range(num_m_mmas):

        @parameter
        for row in range(frag_num_rows):
            # correction[col_tile, row] /= interwarp_frag_rowsum[col_tile, row]
            rowsum_tensor[col_tile, row] = interwarp_frag_rowsum[col_tile, row]

    # var ort00 = output_reg_tile[0,0]
    # scale output reg
    @parameter
    for col_tile in range(num_m_mmas):

        @parameter
        for row_tile in range(num_n_mmas):
            alias tile_id = col_tile + row_tile * num_m_mmas
            alias output_frag_type = __type_of(output_reg_tile).element_type

            @parameter
            for row in range(frag_num_rows):
                var c = correction[col_tile, row][0]

                @parameter
                for warp_tile in range(num_warps_n):
                    output[warp_tile][tile_id, 0] = (
                        0.0 if c == 0.0 else output[warp_tile][tile_id, 0] * c
                    )

    # reduce
    @parameter
    for warp_n in range(num_warps_n):
        var reg_tile = output_reg_tile.tile[num_m_mmas * num_n_mmas, 1](
            warp_n, 0
        )
        if warp_n == warp_x:

            @parameter
            if warp_n > 0:
                # we want `output_reg_tile[0,:,:]` to be the real output reg tile.
                out_reg_tile.copy_from(reg_tile)
        else:
            # copy output reg tile to smem
            # Example smem row, col when `num_warps_n = 4`:
            # -----------------------------------
            # | N\X |   0  |   1  |   2  |   3  |
            # |  0  |      | 0, 0 | 0, 1 | 0, 2 |
            # |  1  | 1, 0 |      | 1, 1 | 1, 2 |
            # |  2  | 2, 0 | 2, 1 |      | 2, 2 |
            # |  3  | 3, 0 | 3, 1 | 3, 2 |      |
            # -----------------------------------
            # `N\X` refer to `warp_n`, `warp_x`
            alias row = warp_n
            var col = warp_x - (1 if warp_x > warp_n else 0)
            var o_smem_ptr_write = (
                o_smem_ptr + (row * (num_warps_n - 1) + col) * warp_tile_size
            )
            var o_smem_write = (
                LayoutTensor[
                    type,
                    o_smem_layout,
                    address_space = AddressSpace.SHARED,
                ](o_smem_ptr_write)
                .vectorize[1, frag_size]()
                .distribute[Layout.row_major(WARP_SIZE, 1)](lane)
            )
            # after distribute and vectorize, the shape should be
            # WM * WN // (2*frag_size * WARP_SIZE), 1
            # Note that we have
            # frag_size = MMA_M * MMA_N // (2*WARP_SIZE)
            # num_m_mmas = WM // MMA_M
            # num_n_mmas = WN // MMA_N
            # so (because 2*WARP_SIZE*frag_size == MMA_M * MMA_N):
            # WM * WN // (2*frag_size * WARP_SIZE) = WM * WN // (MMA_M * MMA_N)
            #   = num_m_mmas * num_n_mmas
            # thus the shape of `o_smem_write` matches that of `reg_tile`.
            o_smem_write.copy_from(reg_tile)

    barrier()

    # Perform the reduction.
    @parameter
    for warp_n in range(num_warps_n - 1):
        var row = warp_x
        alias col = warp_n
        var o_smem_ptr_reduce = (
            o_smem_ptr + (row * (num_warps_n - 1) + col) * warp_tile_size
        )
        var o_smem_reduce = (
            LayoutTensor[
                type,
                o_smem_layout,
                address_space = AddressSpace.SHARED,
            ](o_smem_ptr_reduce)
            .vectorize[1, frag_size]()
            .distribute[Layout.row_major(WARP_SIZE, 1)](lane)
        )

        @parameter
        for i in range(o_smem_reduce.layout.size()):
            out_reg_tile[i] += rebind[SIMD[type, frag_size]](o_smem_reduce[i])


@always_inline
fn _rowmax_online_softmax[
    type: DType,
    reg_tile_layout: Layout,
    row_accum_layout: Layout,
    fragment_layout: Layout,
    accum_frag_layout: Layout, //,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    use_exp2: Bool,
](
    out score_frag_rowmax: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
    score_reg_tile: LayoutTensor[
        type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=fragment_layout,
    ],
    rowmax_tensor: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
    init_rowmax: Bool = False,
):
    alias num_colwise_warps = block_layout_by_warp.shape[0].value()
    alias num_rowwise_warps = block_layout_by_warp.shape[1].value()
    constrained[
        num_rowwise_warps == 1,
        "FIXME: add support for num_rowwise_warps>1, required by deepseek",
    ]()

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    alias frag_size = fragment_layout.size()
    # alias frag_num_rows = fragment_layout.shape[0].value() # sm90 1
    alias frag_num_cols = fragment_layout.shape[1].value()  # sm90 2
    alias frag_num_rows = accum_frag_layout.size()
    constrained[frag_num_rows == fragment_layout.shape[0].value()]()

    alias num_colwise_tiles = reg_tile_layout[0].size()
    alias num_rowwise_tiles = reg_tile_layout[1].size()
    # The online softmax attributes for each thread's elements (fragments).
    constrained[
        rowmax_tensor.element_layout.size() == frag_num_rows,
        (
            "`rowmax_tensor` and `rowsum_tensor` should be vectorized for AMD,"
            " where `frag_num_rows > 1`. This simplifies the implementation."
        ),
    ]()
    score_frag_rowmax = __type_of(rowmax_tensor).stack_allocation()

    alias num_rowwise_lanes = UInt32(warp_layout.shape[1].value())

    alias exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    # Online softmax
    @parameter
    for col_tile in range(num_colwise_tiles):
        # Initialize local max with the running max.
        score_frag_rowmax._set[col_tile](
            score_reg_tile._get[col_tile, 0]().reduce_max[frag_num_rows]()
        )

        @parameter
        for row_tile in range(1, num_rowwise_tiles):
            score_frag_rowmax._set[col_tile](
                max(
                    score_frag_rowmax._get[col_tile](),
                    score_reg_tile._get[col_tile, row_tile]().reduce_max[
                        frag_num_rows
                    ](),
                )
            )
    if not init_rowmax:

        @parameter
        for col_tile in range(num_colwise_tiles):
            score_frag_rowmax._set[col_tile](
                max(
                    score_frag_rowmax._get[col_tile](),
                    rowmax_tensor._get[col_tile](),
                )
            )

    @parameter
    for col_tile in range(num_colwise_tiles):
        # Every four threads have elements on the same row.
        # Reduce max for T0-T3, T4-T7, etc for nvidia
        #                T0-T15, T16-T31, etc for amd
        score_frag_rowmax._set[col_tile](
            warp.lane_group_max_and_broadcast[Int(num_rowwise_lanes)](
                score_frag_rowmax._get[col_tile]()
            )
        )

        # Softmax numerator based on mma results.
        @parameter
        for row_tile in range(num_rowwise_tiles):
            score_reg_tile._set[col_tile, row_tile](
                exp_function(
                    score_reg_tile._get[col_tile, row_tile]()
                    - score_frag_rowmax._get[col_tile, size=frag_size]()
                )
            )


@always_inline
fn _rowsum[
    type: DType,
    reg_tile_layout: Layout,
    fragment_layout: Layout, //,
    warp_layout: Layout,
](
    score_reg_tile: LayoutTensor[
        type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=fragment_layout,
    ],
    out score_frag_rowsum: LayoutTensor[
        type,
        Layout.row_major(reg_tile_layout[0].size()),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout = Layout.row_major(fragment_layout.shape[0].value()),
    ],
):
    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    alias frag_num_rows = score_frag_rowsum.element_layout.size()

    alias num_colwise_tiles = reg_tile_layout[0].size()
    alias num_rowwise_tiles = reg_tile_layout[1].size()
    # The online softmax attributes for each thread's elements (fragments).
    alias num_rows_per_thread = num_colwise_tiles * frag_num_rows

    score_frag_rowsum = __type_of(score_frag_rowsum).stack_allocation()

    # Initialize local max with the running max, and local sum with zero.
    @parameter
    for col_tile in range(num_colwise_tiles):
        score_frag_rowsum._set[col_tile](
            score_reg_tile._get[col_tile, 0]().reduce_add[frag_num_rows]()
        )

    alias num_rowwise_lanes = UInt32(warp_layout.shape[1].value())

    @parameter
    for row_tile in range(1, num_rowwise_tiles):

        @parameter
        for col_tile in range(num_colwise_tiles):
            score_frag_rowsum._set[col_tile](
                score_frag_rowsum._get[col_tile]()
                + score_reg_tile._get[col_tile, row_tile]().reduce_add[
                    frag_num_rows
                ]()
            )

    @parameter
    for col_tile in range(num_colwise_tiles):
        score_frag_rowsum._set[col_tile](
            warp.lane_group_sum_and_broadcast[Int(num_rowwise_lanes)](
                score_frag_rowsum._get[col_tile]()
            )
        )


@always_inline
fn _online_softmax_correction[
    type: DType,
    row_accum_layout: Layout,
    accum_frag_layout: Layout, //,
    use_exp2: Bool,
](
    rowmax_tensor: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
    score_frag_rowmax: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
):
    alias num_colwise_tiles = row_accum_layout.size()
    alias exp_function = _exp2_concrete if use_exp2 else _exp_concrete

    @parameter
    for col_tile in range(num_colwise_tiles):
        # Corrention since previous max may be updated.
        sfr = score_frag_rowmax._get[col_tile]()
        score_frag_rowmax._set[col_tile](
            exp_function(rowmax_tensor._get[col_tile]() - sfr)
        )
        rowmax_tensor._set[col_tile](sfr)


@always_inline
fn _online_softmax_iter_for_mma_output_sm90[
    type: DType,
    reg_tile_layout: Layout,
    row_accum_layout: Layout,
    fragment_layout: Layout,
    accum_frag_layout: Layout, //,
    block_layout_by_warp: Layout,
    warp_layout: Layout,
    use_exp2: Bool,
](
    out correction: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
    score_reg_tile: LayoutTensor[
        type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=fragment_layout,
    ],
    rowmax_tensor: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
    rowsum_tensor: LayoutTensor[
        type,
        row_accum_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=accum_frag_layout,
    ],
):
    alias num_colwise_warps = block_layout_by_warp.shape[0].value()
    alias num_rowwise_warps = block_layout_by_warp.shape[1].value()
    constrained[
        num_rowwise_warps == 1,
        "FIXME: add support for num_rowwise_warps>1, required by deepseek",
    ]()

    # Assume p_reg_tile has been properly vectorized. The element layout
    # represents number elements per thread in a row or column
    # Each mma fragment is a 2D tile e.g. (1, x) for nvidia and (x, 1) for AMD.

    # TODO: fragment_layout should ideally be inferred from the shape of output_reg_tile or score_reg_tile
    # alias frag_num_rows = fragment_layout.shape[0].value() # sm90 1
    alias frag_num_cols = fragment_layout.shape[1].value()  # sm90 2
    alias frag_num_rows = accum_frag_layout.size()
    constrained[frag_num_rows == fragment_layout.shape[0].value()]()

    alias num_colwise_tiles = reg_tile_layout[0].size()
    alias num_rowwise_tiles = reg_tile_layout[1].size()
    # The online softmax attributes for each thread's elements (fragments).
    alias num_rows_per_thread = num_colwise_tiles * frag_num_rows

    constrained[
        rowmax_tensor.element_layout.size() == frag_num_rows,
        (
            "`rowmax_tensor` and `rowsum_tensor` should be vectorized for AMD,"
            " where `frag_num_rows > 1`. This simplifies the implementation."
        ),
    ]()

    # Initialize local max with the running max, and local sum with zero.

    alias is_nvidia: Bool = is_nvidia_gpu()
    # this is basically M in mma shape, but for nvidia we absorb the factor
    # of 2 in num_m_mma so we use 8 here for nvidia
    alias num_colwise_lanes = UInt32(
        warp_layout.shape[0].value()
    ) if is_nvidia else UInt32(16)
    alias num_rowwise_lanes = UInt32(warp_layout.shape[1].value())

    # Online softmax; correction is initially used as `score_frag_rowmax`
    correction = _rowmax_online_softmax[
        block_layout_by_warp, warp_layout, use_exp2=use_exp2
    ](score_reg_tile, rowmax_tensor)

    score_frag_rowsum = _rowsum[warp_layout](score_reg_tile)

    _online_softmax_correction[use_exp2=use_exp2](rowmax_tensor, correction)

    @parameter
    for col_tile in range(num_colwise_tiles):
        # Save current rowmax and rowsum
        rowsum_tensor._set[col_tile](
            rowsum_tensor._get[col_tile]() * correction._get[col_tile]()
            + rebind[SIMD[type, accum_frag_layout.size()]](
                score_frag_rowsum._get[col_tile]()
            )
        )
