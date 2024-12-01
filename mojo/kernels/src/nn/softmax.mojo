# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import align_down, align_up, ceildiv, exp, log
from os import abort
from sys import alignof, simdwidthof

from algorithm import sync_parallelize, vectorize
from algorithm._gpu.reduction import block_reduce, row_reduce
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _reduce_generator,
)
from buffer import Buffer, NDBuffer
from buffer.dimlist import Dim, DimList
from builtin.uint import _temp_uint_from_int
from gpu import WARP_SIZE, BlockIdx, GridDim, ThreadIdx, barrier, lane_id
from gpu.host import DeviceAttribute, DeviceContext
from gpu.memory import AddressSpace
from gpu.shuffle import shuffle_up, shuffle_xor, warp_broadcast
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.tensor_core import get_fragment_size
from memory import UnsafePointer, stack_allocation
from runtime.asyncrt import MojoCallContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils import IndexList, StaticTuple
from utils.index import Index, product
from utils.numerics import get_accum_type, min_or_neg_inf

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===----------------------------------------------------------------------===#


fn _softmax_2_pass_step1[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](input: Buffer[type, buffer_size]) -> StaticTuple[Scalar[type], 2]:
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
    output: Buffer[type, buffer_size],
    input: Buffer[type, buffer_size],
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
](output: Buffer[type, buffer_size], input: Buffer[type, buffer_size]):
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


# ===----------------------------------------------------------------------===#
# Softmax 3 Pass
# ===----------------------------------------------------------------------===#


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
](output: Buffer[type, buffer_size], max_val: Scalar[type],) -> Scalar[type]:
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
](output: Buffer[type, buffer_size], accum: Scalar[type]):
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
](output: Buffer[type, buffer_size]) raises:
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
    var max_buff = Buffer[type, 1].stack_allocation()

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
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [_] -> SIMD[
        type, _simd_width
    ],
](output: Buffer[type, buffer_size]) raises:
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


# ===----------------------------------------------------------------------===#
# LogSoftmax
# ===----------------------------------------------------------------------===#


fn logsoftmax[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing [_] -> SIMD[
        type, _simd_width
    ],
](output: Buffer[type, buffer_size]) raises:
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
    output: NDBuffer[type, rank, static_shape],
    axis: Int,
) raises:
    # TODO: Add rowwise generator to de-duplicate partioning logic between
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
            var output_buffer_view = Buffer[type](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index(i, shape, rank - 1)

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lamda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            logsoftmax[simd_width, Dim(), type, input_fn_1d](output_buffer_view)
            _ = indices

    sync_parallelize[task_func](num_workers)


fn logsoftmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
](
    input: NDBuffer[type, rank, static_shape],
    output: NDBuffer[type, rank, static_shape],
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


# ===----------------------------------------------------------------------===#
# Softmax
# ===----------------------------------------------------------------------===#


fn _softmax_cpu[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        _
    ] -> SIMD[type, _simd_width],
](
    shape: IndexList[rank],
    output: NDBuffer[type, rank, static_shape],
    axis: Int,
) raises:
    # TODO: Add rowwise generator to de-duplicate partioning logic between
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
            var output_buffer_view = Buffer[type](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index(i, shape, rank - 1)

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lamda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            softmax_3_pass[simd_width, Dim(), type, input_fn_1d](
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
    input: NDBuffer[type, rank, static_shape],
    output: NDBuffer[type, rank, static_shape],
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
](shape: IndexList[rank], output: NDBuffer[type, rank], axis: Int,):
    var row_size = _temp_uint_from_int(shape[axis])
    var num_rows = _temp_uint_from_int(shape.flattened_length()) // row_size

    var max_buf = Buffer[
        accum_type, 1, address_space = AddressSpace.SHARED
    ].stack_allocation()
    var exp_sum_buf = Buffer[
        accum_type, 1, address_space = AddressSpace.SHARED
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

    var row_size_padded = align_up(row_size, _temp_uint_from_int(BLOCK_SIZE))

    # grid stride loop over rows
    # each block reduces a row, which is convenient because it requires no partial
    # reductions across blocks
    for row_idx in range(
        BlockIdx.x,
        num_rows,
        GridDim.x,
    ):
        # Step 1: compute max in row
        var row_coords = _get_nd_indices_from_flat_index(
            int(row_idx), shape, axis
        )
        var row_max = row_reduce[
            BLOCK_SIZE,
            input_fn,
            _max,
            type,
            1,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, Scalar[type].MIN, int(row_size))

        if ThreadIdx.x == 0:
            max_buf[0] = row_max
        barrier()

        # Step 2: out[i] = exp(in[i] - max) and compute sum of out[i]
        var exp_sum = Scalar[accum_type](0)
        for offset_in_row in range(
            UInt(0), row_size_padded, _temp_uint_from_int(BLOCK_SIZE)
        ):
            var idx_in_padded_row = ThreadIdx.x + offset_in_row
            if idx_in_padded_row >= row_size:
                break

            row_coords[axis] = int(idx_in_padded_row)

            # loads from input_fn twice
            var val = exp(
                input_fn[type, 1, rank](row_coords).cast[accum_type]()
                - max_buf[0]
            )

            # TODO we're writing to and reading from global memory twice
            # we can reduce the amount of reads by keeping values local here.
            output[row_coords] = val.cast[type]()
            exp_sum += val

        var block_exp_sum = block_reduce[BLOCK_SIZE, _sum](exp_sum, 0)
        if ThreadIdx.x == 0:
            exp_sum_buf[0] = block_exp_sum
        barrier()

        # Step 3: Normalize output
        var block_exp_sum_recip = 1 / exp_sum_buf[0]
        for offset_in_row in range(
            UInt(0), row_size_padded, _temp_uint_from_int(BLOCK_SIZE)
        ):
            var idx_in_padded_row = ThreadIdx.x + offset_in_row
            if idx_in_padded_row >= row_size:
                break

            row_coords[axis] = int(idx_in_padded_row)
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
    output: NDBuffer[type, rank, static_shape],
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
    var func = ctx.compile_function[
        softmax_kernel[
            BLOCK_SIZE,
            input_fn_wrapper,
            type,
            rank,
        ]
    ]()

    var num_rows = shape.flattened_length() // shape[axis]
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    alias sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    ctx.enqueue_function(
        func,
        shape,
        output,
        axis,
        grid_dim=(num_blocks,),
        block_dim=(BLOCK_SIZE,),
    )


fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (IndexList[_rank]) capturing [
        _
    ] -> SIMD[type, _simd_width],
    target: StringLiteral = "cpu",
](
    shape: IndexList[rank],
    output: NDBuffer[type, rank, static_shape],
    axis: Int,
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    @parameter
    fn trace_information() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP, target=target](
        "softmax",
        Trace[TraceLevel.OP]._get_detail_str[trace_information](),
    ):

        @parameter
        if target == "cpu":
            _softmax_cpu[type, simd_width, rank, static_shape, input_fn](
                shape, output, axis
            )
        elif "cuda" in target:
            _softmax_gpu[type, simd_width, rank, static_shape, input_fn](
                shape,
                output,
                axis,
                context.get_device_context(),
            )
        else:
            constrained[False, "unsupported target " + target]()


# ===----------------------------------------------------------------------=== #
# Online softmax in flash attention.
# ===----------------------------------------------------------------------=== #


fn _online_softmax_kernel[
    WM: Int,
    WN: Int,
    type: DType,
    layout: Layout,
](input: LayoutTensor[type, layout], output: LayoutTensor[type, layout]):
    """This is only for online softmax validation, NOT a general kernel."""

    alias mma_shape = IndexList[3](16, 8, 8)
    alias num_seqs = input.shape[0]()
    alias seqlen = input.shape[1]()

    constrained[
        WM == num_seqs, "Only consider WM equal to number of rows in test."
    ]()

    alias num_m_mmas = WM // mma_shape[0]
    alias num_n_mmas = WN // mma_shape[1]

    # Only consider 2 iterations in this test. The number of warps is based on
    # half sequence length.
    alias num_rowwise_warps = seqlen // 2 // WN

    alias frag_size = get_fragment_size[mma_shape]()[2]

    var warp_id: UInt = warp_broadcast(ThreadIdx.x // WARP_SIZE)
    var lane = lane_id()

    # If we do more than 2 iterations, the first N - 2 iterations won't be
    # corrected with the right rowmax.
    var input_warp_tile0 = input.tile[WM, WN](0, int(warp_id))
    var input_warp_tile1 = input.tile[WM, WN](
        0, int(warp_id) + num_rowwise_warps
    )

    var output_warp_tile0 = output.tile[WM, WN](0, int(warp_id))
    var output_warp_tile1 = output.tile[WM, WN](
        0, int(warp_id) + num_rowwise_warps
    )

    var p = LayoutTensor[
        type,
        Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    p.vectorize[1, 2]().transpose().copy_from(
        input_warp_tile0.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
            lane
        )
    )

    var o = LayoutTensor[
        type,
        Layout.row_major(num_m_mmas * num_n_mmas, frag_size),
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0.0)

    alias row_alignment = alignof[SIMD[type, simdwidthof[type]()]]()
    var rowmax = stack_allocation[
        num_m_mmas * 2, type, alignment=row_alignment
    ]()
    var rowsum = stack_allocation[
        num_m_mmas * 2, type, alignment=row_alignment
    ]()

    var warp_scratch = LayoutTensor[
        type,
        Layout.row_major(2 * num_rowwise_warps, WM),
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    @parameter
    for i in range(0, 2 * num_m_mmas, 2):
        rowmax.store(i, SIMD[type, 2](min_or_neg_inf[type]()))
        rowsum.store(i, SIMD[type, 2](0))

    _online_softmax_iter_for_mma_output[
        num_m_mmas, num_n_mmas, num_rowwise_warps, mma_shape, type
    ](o, p, warp_scratch, rowmax, rowsum)

    # P has the softmax numerator for the first half, save it in q.
    o.copy_from(p)
    p.vectorize[1, 2]().transpose().copy_from(
        input_warp_tile1.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
            lane
        )
    )

    _online_softmax_iter_for_mma_output[
        num_m_mmas, num_n_mmas, num_rowwise_warps, mma_shape, type
    ](o, p, warp_scratch, rowmax, rowsum)

    # o, p has the correct softmax numerator for the 1st and 2nd half.
    # rowsum has the correct sum. Ready for correction.
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(frag_size // 2):
                p[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                p[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                    2 * m_mma + 1
                ]
                o[n_mma * num_m_mmas + m_mma, i] /= rowsum[2 * m_mma]
                o[n_mma * num_m_mmas + m_mma, i + frag_size // 2] /= rowsum[
                    2 * m_mma + 1
                ]

    output_warp_tile0.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane
    ).copy_from(o.vectorize[1, 2]().transpose())
    output_warp_tile1.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane
    ).copy_from(p.vectorize[1, 2]().transpose())


@always_inline
fn _online_softmax_iter_for_mma_output[
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_rowwise_warps: Int,
    mma_shape: IndexList[3],
    type: DType,
](
    output_reg_tile: LayoutTensor[type, *_, **_],
    p_reg_tile: LayoutTensor[type, *_, **_],
    warp_scratch: LayoutTensor[type, *_, **_],
    rowmax: UnsafePointer[Scalar[type], **_],
    rowsum: UnsafePointer[Scalar[type], **_],
):
    constrained[num_m_mmas * num_n_mmas == p_reg_tile.shape[0]()]()

    var tid = ThreadIdx.x
    var lane = lane_id()
    var warp_x = warp_broadcast(tid // WARP_SIZE) % UInt(num_rowwise_warps)

    alias MMA_M = mma_shape[0]
    alias MMA_N = mma_shape[1]
    alias p_frag_simdwidth = 2
    alias p_frag_size: UInt = p_reg_tile.shape[1]()

    # Sum of fragment elements in the same row.
    # MMA output has two sub-matrices. Each thread's fragments are on two rows.
    alias frag_alignment = alignof[SIMD[type, simdwidthof[type]()]]()
    var p_frag_rowmax = stack_allocation[
        num_m_mmas * 2, type, alignment=frag_alignment
    ]()
    var p_frag_rowsum = stack_allocation[
        num_m_mmas * 2, type, alignment=frag_alignment
    ]()
    var correction = stack_allocation[
        num_m_mmas * 2, type, alignment=frag_alignment
    ]()

    @parameter
    for i in range(0, 2 * num_m_mmas, 2):
        p_frag_rowmax.store(i, rowmax.load[width=2]())
        p_frag_rowsum.store(i, SIMD[type, 2](0))

    # Online softmax
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(int(p_frag_size // 2)):
                var curr = SIMD[type, 2](
                    rebind[Scalar[type]](
                        p_reg_tile[n_mma * num_m_mmas + m_mma, i]
                    ),
                    rebind[Scalar[type]](
                        p_reg_tile[
                            n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                        ]
                    ),
                )
                p_frag_rowmax.store[alignment = p_frag_rowmax.alignment](
                    2 * m_mma,
                    max(
                        p_frag_rowmax.load[
                            width=2, alignment = p_frag_rowmax.alignment
                        ](2 * m_mma),
                        curr,
                    ),
                )

        # Every four threads have elements on the same row.
        # Reduce max for T0-T3, T4-T7, etc
        # TODO: Consider using 2 registers instead of 2 * m_mma.
        @parameter
        for i in range(2):

            @parameter
            for j in reversed(range(1, 3)):
                p_frag_rowmax[2 * m_mma + i] = max(
                    p_frag_rowmax[2 * m_mma + i],
                    shuffle_xor(p_frag_rowmax[2 * m_mma + i], j),
                )

    # If a row is split across multiple warps, communicate via shared memory
    # to achieve the rowwise max.
    @parameter
    if num_rowwise_warps > 1:
        # Write per warp rowmax to shared memory.
        if lane % 4 == 0:

            @parameter
            for m_mma0 in range(num_m_mmas):
                alias m_mma = UInt(m_mma0)
                # Each thread handle two rows in the mma output.
                var row0 = m_mma * MMA_M + lane // (MMA_N // p_frag_simdwidth)
                var row1 = row0 + MMA_M // 2
                warp_scratch[int(warp_x), row0] = p_frag_rowmax[2 * m_mma]
                warp_scratch[int(warp_x), row1] = p_frag_rowmax[2 * m_mma + 1]

        barrier()

        # Reduce the warpwise rowmax.
        if lane % 4 == 0:

            @parameter
            for m_mma in range(num_m_mmas):
                var row0 = m_mma * MMA_M + lane // (MMA_N // p_frag_simdwidth)
                var row1 = row0 + MMA_M // 2

                # Reduce rowmax. Warps in the same row do the same reduction.
                @parameter
                for w in range(num_rowwise_warps):
                    var curr = SIMD[type, 2](
                        rebind[Scalar[type]](warp_scratch[w, row0]),
                        rebind[Scalar[type]](warp_scratch[w, row1]),
                    )
                    p_frag_rowmax.store(
                        2 * m_mma,
                        max(
                            p_frag_rowmax.load[
                                width=2, alignment = p_frag_rowmax.alignment
                            ](2 * m_mma),
                            curr,
                        ),
                    )

    @parameter
    for m_mma in range(num_m_mmas):
        # Broadcast to 4 threads in the same row.
        @parameter
        if num_rowwise_warps > 1:

            @parameter
            for shift in range(1, 3):

                @parameter
                for i in range(2):
                    p_frag_rowmax[2 * m_mma + i] = max(
                        p_frag_rowmax[2 * m_mma + i],
                        shuffle_xor(p_frag_rowmax[2 * m_mma + i], shift),
                    )

        # Corrention since previous max may be updated.
        correction.store[alignment = correction.alignment](
            2 * m_mma,
            exp(
                rowmax.load[width=2, alignment = rowmax.alignment](2 * m_mma)
                - p_frag_rowmax.load[width=2, alignment = rowmax.alignment](
                    2 * m_mma
                )
            ),
        )

        # Softmax numerator based on mma results.
        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(int(p_frag_size // 2)):
                p_reg_tile[n_mma * num_m_mmas + m_mma, i] = exp(
                    p_reg_tile[n_mma * num_m_mmas + m_mma, i]
                    - p_frag_rowmax[2 * m_mma]
                )
                p_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] = exp(
                    p_reg_tile[n_mma * num_m_mmas + m_mma, i + p_frag_size // 2]
                    - p_frag_rowmax[2 * m_mma + 1]
                )

        p_frag_rowsum.store[alignment = p_frag_rowsum.alignment](
            2 * m_mma, SIMD[type, 2](0.0)
        )

        # Sum softmax numerator from a thread's fragments.
        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(int(p_frag_size // 2)):
                p_frag_rowsum[2 * m_mma] += rebind[Scalar[type]](
                    p_reg_tile[n_mma * num_m_mmas + m_mma, i]
                )
                p_frag_rowsum[2 * m_mma + 1] += rebind[Scalar[type]](
                    p_reg_tile[n_mma * num_m_mmas + m_mma, i + p_frag_size // 2]
                )

        # Sum numerator within a warp.
        @parameter
        for shift in range(1, 3):

            @parameter
            for i in range(2):
                p_frag_rowsum[2 * m_mma + i] += shuffle_xor(
                    p_frag_rowsum[2 * m_mma + i], shift
                )

    @parameter
    if num_rowwise_warps > 1:
        # Write per warp rowmax to shared memory.
        if lane % 4 == 0:

            @parameter
            for m_mma0 in range(num_m_mmas):
                alias m_mma = UInt(m_mma0)
                # Each thread handle two rows in the mma output.
                var row0 = m_mma * MMA_M + lane // (MMA_N // p_frag_simdwidth)
                var row1 = row0 + MMA_M // 2
                warp_scratch[warp_x + num_rowwise_warps, row0] = p_frag_rowsum[
                    2 * m_mma
                ]
                warp_scratch[warp_x + num_rowwise_warps, row1] = p_frag_rowsum[
                    2 * m_mma + 1
                ]
        # Guard writing warp_scratch
        barrier()

        # Reduce the warpwise rowsum.
        if lane % 4 == 0:

            @parameter
            for m_mma0 in range(num_m_mmas):
                alias m_mma = UInt(m_mma0)
                var row0 = m_mma * MMA_M + lane // (MMA_N // p_frag_simdwidth)
                var row1 = row0 + MMA_M // 2
                p_frag_rowsum[2 * m_mma] = 0.0
                p_frag_rowsum[2 * m_mma + 1] = 0.0

                # Reduce rowmax. Warps in the same row do the same reduction.
                @parameter
                for w in range(num_rowwise_warps):
                    p_frag_rowsum[2 * m_mma] += rebind[Scalar[type]](
                        warp_scratch[w + num_rowwise_warps, row0]
                    )

                    p_frag_rowsum[2 * m_mma + 1] += rebind[Scalar[type]](
                        warp_scratch[w + num_rowwise_warps, row1]
                    )

        # Broadcast to 4 threads in the same row e.g. T0 -> T0-T3.
        # TODO: Use Shuffle up
        @parameter
        for m_mma in range(num_m_mmas):
            # Broadcast to 4 threads in the same row.
            @parameter
            for shift in range(1, 3):

                @parameter
                for i in range(2):
                    p_frag_rowsum[2 * m_mma + i] = max(
                        p_frag_rowsum[2 * m_mma + i],
                        shuffle_xor(p_frag_rowsum[2 * m_mma + i], shift),
                    )

    # Correct previous result
    @parameter
    for m_mma in range(num_m_mmas):

        @parameter
        for n_mma in range(num_n_mmas):

            @parameter
            for i in range(int(p_frag_size // 2)):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= correction[
                    2 * m_mma
                ]
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= correction[2 * m_mma + 1]

    # Save current rowmax and rowsum
    @parameter
    for m_mma in range(num_m_mmas):
        rowmax.store(2 * m_mma, p_frag_rowmax.load[width=2](2 * m_mma))
        rowsum.store(
            2 * m_mma,
            rowsum.load[width=2](2 * m_mma)
            * correction.load[width=2](2 * m_mma)
            + p_frag_rowsum.load[width=2](2 * m_mma),
        )
