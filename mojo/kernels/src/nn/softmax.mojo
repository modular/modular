# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil, exp, identity, log, max, min, mul, reciprocal, sub
from math.limit import min_or_neginf, neginf

from algorithm import async_parallelize, vectorize_unroll
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _reduce_generator,
)
from memory.buffer import Buffer, NDBuffer
from runtime.llcl import OutputChainPtr
from runtime.tracing import TraceLevel

from utils.index import product
from utils.list import Dim, DimList
from utils.static_tuple import StaticTuple

from math import align_up

from algorithm.gpu.reduction import row_reduce, block_reduce

from gpu import (
    ThreadIdx,
    BlockIdx,
    BlockDim,
    GridDim,
    barrier,
)

from gpu.host import (
    Stream,
    Function,
    Device,
    DeviceAttribute,
)
from gpu.memory import AddressSpace

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn reduce_add_simd[
    simd_width: Int,
    step_simd_width: Int,
    type: DType,
](
    inout scalar: SIMD[type, 1],
    inout vector: SIMD[type, simd_width],
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


# ===----------------------------------------------------------------------===#
# Softmax 2 Pass
# ===----------------------------------------------------------------------===#


fn _softmax_2_pass_step1[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](input: Buffer[buffer_size, type]) -> StaticTuple[
    2, __mlir_type[`!pop.scalar<`, type.value, `>`]
]:
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

    var running_max_vec = SIMD[type, simd_width].splat(neginf[type]())
    var running_sum_vec = SIMD[type, simd_width].splat(0)

    # TODO: Because vectorize cannot currently capture values from outside
    # scope, we therefore replicate the logic of Functional.vectorize here.
    # In the future (once we have non-isolated-from-above regions) we can
    # just reuse the Functional.vectorize code.
    let len = input.__len__()
    let vector_end = (len // simd_width) * simd_width

    for i in range(0, vector_end, simd_width):
        let simd_elem = input.simd_load[simd_width](i)
        let new_max_vec = SIMD[type, simd_width].splat(
            running_max_vec.max(simd_elem).reduce_max()
        )
        running_sum_vec = running_sum_vec * exp(
            running_max_vec - new_max_vec
        ) + exp(simd_elem - new_max_vec)
        running_max_vec = new_max_vec

    var running_max = running_max_vec.reduce_max()
    var running_sum = running_sum_vec.reduce_add()

    for i in range(vector_end, len):
        let elem = input[i]
        let new_max = running_max.max(elem)
        running_sum = running_sum * exp(running_max - new_max) + exp(
            elem - new_max
        )
        running_max = new_max

    return StaticTuple[2, __mlir_type[`!pop.scalar<`, type.value, `>`]](
        running_max[0].value, running_sum[0].value
    )


fn _softmax_2_pass_step2[
    simd_width: Int,
    unroll_factor: Int,
    buffer_size: Dim,
    type: DType,
](
    output: Buffer[buffer_size, type],
    input: Buffer[buffer_size, type],
    running_max: SIMD[type, 1],
    running_sum: SIMD[type, 1],
):
    # Step 2:
    #   for i = 0 to N do
    #     Output[i] = exp(Input[i] - runningMax) / runningSum
    #   end for

    @always_inline
    @parameter
    fn _step_2[simd_width: Int](idx: Int):
        let running_max_simd = SIMD[type, simd_width].splat(running_max)
        let running_sum_simd = SIMD[type, simd_width].splat(running_sum)
        let input_val = input.simd_load[simd_width](idx)
        output.simd_store[simd_width](
            idx,
            exp(input_val - running_max_simd) / running_sum_simd,
        )

    vectorize_unroll[simd_width, unroll_factor, _step_2](output.__len__())


fn softmax_2_pass[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
](output: Buffer[buffer_size, type], input: Buffer[buffer_size, type]):
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

    let running_info = _softmax_2_pass_step1[simd_width, buffer_size, type](
        input
    )

    let running_max = running_info[0]
    let running_sum = running_info[1]

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
    input_fn_1d: fn[_simd_width: Int] (Int) capturing -> SIMD[
        type, _simd_width
    ],
    pre_update_func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
    post_update_func: fn[type: DType, width: Int] (SIMD[type, width]) -> SIMD[
        type, width
    ],
](output: Buffer[buffer_size, type], max_val: SIMD[type, 1],) -> SIMD[type, 1]:
    # STEP 2: compute for each batch
    # for i = 0 to N do
    #   Output[i] = pre_update_func(Input[i] - max_val)
    #   accum += post_update_func(Output[i])
    # end for
    alias outer_simd_width = simd_width

    var accum_scalar: SIMD[type, 1] = 0
    var accum_simd: SIMD[type, outer_simd_width] = 0

    @always_inline
    @parameter
    fn step_2[simd_width: Int](idx: Int):
        let vin = input_fn_1d[simd_width](idx)
        var elem = vin - SIMD[type, simd_width].splat(max_val)

        elem = pre_update_func[type, simd_width](elem)
        output.simd_store[simd_width](idx, elem)
        elem = post_update_func[type, simd_width](elem)
        reduce_add_simd[outer_simd_width, simd_width, type](
            accum_scalar, accum_simd, elem
        )

    vectorize_unroll[simd_width, unroll_factor, step_2](output.__len__())
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
](output: Buffer[buffer_size, type], accum: SIMD[type, 1]):
    # STEP 3: normalize each batch
    # accum = accum_proc_func(accum)
    # for i = 0 to N do
    #   accum_apply_func(Output[b, i], accum)
    # end for
    let accum_proc = accum_proc_func[type, 1](accum)

    @always_inline
    @parameter
    fn step_3[simd_width: Int](idx: Int):
        let accum_simd = SIMD[type, simd_width].splat(accum_proc)
        var elem = output.simd_load[simd_width](idx)
        elem = accum_apply_func[type, simd_width](elem, accum_simd)
        output.simd_store[simd_width](idx, elem)

    vectorize_unroll[simd_width, unroll_factor, step_3](output.__len__())


fn _softmax_3_pass_base[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing -> SIMD[
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
](output: Buffer[buffer_size, type], out_chain: OutputChainPtr):
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
        out_chain: The output-chain pointer.
    """
    # STEP 1 - Calculate max
    # Allocate buffer for max_val
    let max_buff = Buffer[1, type].stack_allocation()

    # Use _reduce_generator to fuse input lambda with max-reduction
    # Reduce function
    @always_inline
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
    ](coords: StaticIntTuple[_rank]) -> SIMD[_type, _width]:
        constrained[_rank == 1]()
        return rebind[SIMD[_type, _width]](input_fn_1d[_width](coords[0]))

    # Output function
    @parameter
    @always_inline
    fn output_fn[
        _type: DType, _width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[_type, _width]):
        constrained[_rank == 1]()
        max_buff[0] = val.reduce_max().cast[type]()

    # Generate fused input-reduction
    _reduce_generator[type, 1, True, input_fn, output_fn, reduce_impl](
        StaticIntTuple[1](output.__len__()), min_or_neginf[type](), 0, out_chain
    )

    let max_val = max_buff[0]

    # STEP 2
    alias unroll_factor = 8  # TODO: search
    let accum = _softmax_3_pass_step_2[
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
    input_fn_1d: fn[_simd_width: Int] (Int) capturing -> SIMD[
        type, _simd_width
    ],
](output: Buffer[buffer_size, type], out_chain: OutputChainPtr):
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
        out_chain: The output-chain pointer.
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
    ](output, out_chain)


# ===----------------------------------------------------------------------===#
# LogSoftmax
# ===----------------------------------------------------------------------===#


fn logsoftmax[
    simd_width: Int,
    buffer_size: Dim,
    type: DType,
    input_fn_1d: fn[_simd_width: Int] (Int) capturing -> SIMD[
        type, _simd_width
    ],
](output: Buffer[buffer_size, type], out_chain: OutputChainPtr):
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
        out_chain: The output-chain pointer.
    """
    _softmax_3_pass_base[
        simd_width, buffer_size, type, input_fn_1d, identity, exp, log, sub
    ](output, out_chain)


fn logsoftmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, static_shape, type],
    axis: Int,
    out_chain: OutputChainPtr,
):
    # TODO: Add rowwise generator to de-duplicate partioning logic between
    # softmax and logsoftmax
    if axis != rank - 1:
        out_chain.mark_error("logsoftmax not supported on non-inner axis yet")
        return

    if shape.flattened_length() == 0:
        return out_chain.mark_ready()

    let inner_dim = output.dim[rank - 1]()
    let outer_dim = product[rank](shape, rank - 1)
    let num_workers = min(
        out_chain.get_runtime().parallelism_level(), outer_dim
    )
    let chunk_size = div_ceil(outer_dim, num_workers)

    @parameter
    @always_inline
    fn task_func(task_id: Int):
        let start_offset = task_id * chunk_size
        let end_offset = min((task_id + 1) * chunk_size, outer_dim)
        for i in range(start_offset, end_offset):
            let buffer_offset = i * inner_dim
            let output_buffer_view = Buffer[Dim(), type](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index[rank](
                i, shape, rank - 1
            )

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lamda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            logsoftmax[simd_width, Dim(), type, input_fn_1d](
                output_buffer_view, out_chain
            )

    async_parallelize[task_func](out_chain, num_workers)


fn logsoftmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
](
    input: NDBuffer[rank, static_shape, type],
    output: NDBuffer[rank, static_shape, type],
    axis: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    @always_inline
    fn input_fn[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return input.simd_load[_simd_width](
            rebind[StaticIntTuple[rank]](coords)
        )

    logsoftmax[type, simd_width, rank, static_shape, input_fn](
        input.get_shape(), output, axis, out_chain
    )


# ===----------------------------------------------------------------------===#
# Softmax
# ===----------------------------------------------------------------------===#


@adaptive
fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, static_shape, type],
    axis: Int,
    out_chain: OutputChainPtr,
):
    constrained[target == "cpu", "cpu softmax"]()
    # TODO: Add rowwise generator to de-duplicate partioning logic between
    # softmax and logsoftmax
    if axis != rank - 1:
        return out_chain.mark_error(
            "softmax not supported on non-inner axis yet"
        )

    @always_inline
    @parameter
    fn trace_information() -> String:
        return String("shape=") + String("x").join(shape)

    out_chain.trace[TraceLevel.OP, trace_information]("mojo.softmax")

    if shape.flattened_length() == 0:
        return out_chain.mark_ready()

    let inner_dim = output.dim[rank - 1]()
    let outer_dim = product[rank](shape, rank - 1)
    let num_workers = min(
        out_chain.get_runtime().parallelism_level(), outer_dim
    )
    let chunk_size = div_ceil(outer_dim, num_workers)

    @parameter
    @always_inline
    fn task_func(task_id: Int):
        let start_offset = task_id * chunk_size
        let end_offset = min((task_id + 1) * chunk_size, outer_dim)
        for i in range(start_offset, end_offset):
            let buffer_offset = i * inner_dim
            let output_buffer_view = Buffer[Dim(), type](
                output.data.offset(buffer_offset), inner_dim
            )
            var indices = _get_nd_indices_from_flat_index[rank](
                i, shape, rank - 1
            )

            @parameter
            @always_inline
            # Given input lambda accepts N-dimensional coordinates, but the
            # softmax base routines operate on 1D buffers. Here we wrap the
            # given input lamda with some 1d-to-Nd translation logic.
            fn input_fn_1d[_width: Int](idx: Int) -> SIMD[type, _width]:
                indices[rank - 1] = idx
                return input_fn[_width, rank](indices)

            softmax_3_pass[simd_width, Dim(), type, input_fn_1d](
                output_buffer_view, out_chain
            )

    async_parallelize[task_func](out_chain, num_workers)


# Softmax (no input lambda)
fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
](
    input: NDBuffer[rank, static_shape, type],
    output: NDBuffer[rank, static_shape, type],
    axis: Int,
    out_chain: OutputChainPtr,
):
    @parameter
    @always_inline
    fn input_fn[
        _simd_width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, _simd_width]:
        return input.simd_load[_simd_width](
            rebind[StaticIntTuple[rank]](coords)
        )

    softmax[type, simd_width, rank, static_shape, input_fn](
        input.get_shape(), output, axis, out_chain
    )


fn softmax_kernel[
    BLOCK_SIZE: Int,
    input_fn: fn[_type: DType, _simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[_type, _simd_width],
    type: DType,
    rank: Int,
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    axis: Int,
):
    let row_size = shape[axis]
    let num_rows = shape.flattened_length() // row_size

    let max_buf = Buffer[1, type, AddressSpace.SHARED].stack_allocation()
    let exp_sum_buf = Buffer[1, type, AddressSpace.SHARED].stack_allocation()

    @parameter
    @always_inline
    @closure
    fn _max[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return max(x, y)

    @parameter
    @always_inline
    @closure
    fn _sum[
        type: DType, width: Int
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    let row_size_padded = align_up(row_size, BLOCK_SIZE)

    # grid stride loop over rows
    # each block reduces a row, which is convenient because it requires no partial
    # reductions across blocks
    for row_idx in range(
        BlockIdx.x(),
        num_rows,
        GridDim.x(),
    ):
        # Step 1: compute max in row
        var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)
        let row_max = row_reduce[BLOCK_SIZE, input_fn, _max](
            row_coords, axis, min_or_neginf[type](), row_size
        )
        if ThreadIdx.x() == 0:
            max_buf[0] = row_max
        barrier()

        # Step 2: out[i] = exp(in[i] - max) and compute sum of out[i]
        var exp_sum = SIMD[type, 1](0)
        for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
            let idx_in_padded_row = ThreadIdx.x() + offset_in_row
            if idx_in_padded_row >= row_size:
                break

            row_coords[axis] = idx_in_padded_row
            # loads from input_fn twice
            let val = exp(input_fn[type, 1, rank](row_coords) - max_buf[0])
            output[row_coords] = val
            exp_sum += val

        let block_exp_sum = block_reduce[BLOCK_SIZE, _sum](exp_sum, 0)
        if ThreadIdx.x() == 0:
            exp_sum_buf[0] = block_exp_sum
        barrier()

        # Step 3: Normalize output
        let block_exp_sum_recip = 1 / exp_sum_buf[0]
        for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
            let idx_in_padded_row = ThreadIdx.x() + offset_in_row
            if idx_in_padded_row >= row_size:
                break

            row_coords[axis] = idx_in_padded_row
            output[row_coords] *= block_exp_sum_recip


@adaptive
fn softmax[
    type: DType,
    simd_width: Int,
    rank: Int,
    static_shape: DimList,
    input_fn: fn[_simd_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _simd_width],
    target: StringLiteral,
](
    shape: StaticIntTuple[rank],
    output: NDBuffer[rank, static_shape, type],
    axis: Int,
    out_chain: OutputChainPtr,
):
    if axis != rank - 1:
        return out_chain.mark_error(
            "softmax not supported on non-inner axis yet"
        )

    constrained[target == "cuda", "cuda softmax"]()

    @always_inline
    fn input_fn_wrapper[
        _type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](input_fn[width, rank](idx))

    alias BLOCK_SIZE = 128
    try:
        # HACK HACK HACK (#25473)
        let stream = out_chain.get_cuda_stream() if out_chain else Stream[
            is_borrowed=True
        ]()

        let func = Function[
            fn (
                StaticIntTuple[rank],
                NDBuffer[rank, DimList.create_unknown[rank](), type],
                Int,
            ) capturing -> None, softmax_kernel[
                BLOCK_SIZE,
                input_fn_wrapper,
                type,
                rank,
            ]
        ]()

        let num_rows = shape.flattened_length() // shape[axis]
        let sm_count = Device()._query(DeviceAttribute.MULTIPROCESSOR_COUNT)
        alias sm_overprovision_factor = 32  # tunable
        let num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

        func((num_blocks,), (BLOCK_SIZE,), shape, output, axis, stream=stream)
    except e:
        out_chain.mark_error(e)
