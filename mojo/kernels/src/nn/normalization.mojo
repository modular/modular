# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv, sqrt

from algorithm import map_reduce, mean, variance, vectorize
from algorithm.functional import sync_parallelize
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _simd_sum,
    _simd_sum_elementwise,
)
from buffer import Buffer, NDBuffer
from buffer.list import DimList
from register import mogg_register
from runtime.llcl import parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg
from gpu.host.device_context import DeviceBuffer, DeviceContext
from utils.index import StaticTuple, StaticIntTuple, Index
from gpu import (
    BlockIdx,
    ThreadIdx,
    lane_id,
    BlockDim,
    barrier,
    syncwarp,
    WARP_SIZE,
)
from gpu.shuffle import _static_log2, shuffle_down, shuffle_idx
from gpu.memory import AddressSpace
from .reshape import reshape


# using numerically stable Welford online algorithm to compute single pass mean and variance
fn welford_update[
    type: DType
](
    val: Scalar[type],
    inout mean: Scalar[type],
    inout m2: Scalar[type],
    inout count: Scalar[type],
):
    count += 1
    var d1 = val - mean
    mean += d1 / count
    var d2 = val - mean
    m2 += d1 * d2


fn welford_combine[
    type: DType
](
    mean: Scalar[type],
    m2: Scalar[type],
    count: Scalar[type],
    inout res_mean: Scalar[type],
    inout res_m2: Scalar[type],
    inout res_count: Scalar[type],
):
    if count == 0:
        return
    var x_count = count + res_count
    var m = count / x_count
    var delta = mean - res_mean
    res_mean += delta * m
    res_m2 += m2 + delta * delta * res_count * m
    res_count = x_count


fn welford_warp_reduce[
    type: DType
](
    thread_mean: Scalar[type],
    thread_m2: Scalar[type],
    thread_count: Scalar[type],
    inout res_mean: Scalar[type],
    inout res_m2: Scalar[type],
    inout res_count: Scalar[type],
):
    res_mean = thread_mean
    res_m2 = thread_m2
    res_count = thread_count

    alias limit = _static_log2[WARP_SIZE]()

    @parameter
    for mask in reversed(range(limit)):
        var mean = shuffle_down(res_mean, 1 << mask)
        var m2 = shuffle_down(res_m2, 1 << mask)
        var count = shuffle_down(res_count, 1 << mask)
        welford_combine[type](mean, m2, count, res_mean, res_m2, res_count)


fn welford_warp_all_reduce[
    type: DType
](
    thread_mean: Scalar[type],
    thread_m2: Scalar[type],
    thread_count: Scalar[type],
    inout res_mean: Scalar[type],
    inout res_m2: Scalar[type],
    inout res_count: Scalar[type],
):
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, res_mean, res_m2, res_count
    )
    # broadcasting res from warp lane_id 0 to all in a warp
    res_mean = shuffle_idx(res_mean, 0)
    res_m2 = shuffle_idx(res_m2, 0)
    res_count = shuffle_idx(res_count, 0)


fn welford_block_all_reduce[
    type: DType
](
    thread_mean: Scalar[type],
    thread_m2: Scalar[type],
    thread_count: Scalar[type],
    inout res_mean: Scalar[type],
    inout res_m2: Scalar[type],
    inout res_count: Scalar[type],
):
    var mean_shared = stack_allocation[
        WARP_SIZE, type, address_space = AddressSpace.SHARED
    ]()
    var m2_shared = stack_allocation[
        WARP_SIZE, type, address_space = AddressSpace.SHARED
    ]()
    var count_shared = stack_allocation[
        WARP_SIZE, type, address_space = AddressSpace.SHARED
    ]()
    var mean_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()
    var count_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()

    var warp_idx = ThreadIdx.x() // WARP_SIZE
    var lane_idx = lane_id()
    var warp_mean = Scalar[type]()
    var warp_m2 = Scalar[type]()
    var warp_count = Scalar[type]()
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, warp_mean, warp_m2, warp_count
    )
    barrier()

    if lane_idx == 0:
        mean_shared[warp_idx] = warp_mean
        m2_shared[warp_idx] = warp_m2
        count_shared[warp_idx] = warp_count
    barrier()

    if warp_idx == 0:
        if ThreadIdx.x() < (BlockDim.x() // WARP_SIZE):
            warp_mean = Scalar.load(mean_shared, lane_idx)
            warp_m2 = Scalar.load(m2_shared, lane_idx)
            warp_count = Scalar.load(count_shared, lane_idx)
        else:
            warp_mean = Scalar[type]()
            warp_m2 = Scalar[type]()
            warp_count = Scalar[type]()
        syncwarp()
        var block_mean = Scalar[type]()
        var block_m2 = Scalar[type]()
        var block_count = Scalar[type]()
        welford_warp_reduce(
            warp_mean, warp_m2, warp_count, block_mean, block_m2, block_count
        )
        if lane_idx == 0:
            mean_broadcast[0] = block_mean
            m2_broadcast[0] = block_m2
            count_broadcast[0] = block_count

    barrier()

    welford_combine[type](
        mean_broadcast[0],
        m2_broadcast[0],
        count_broadcast[0],
        res_mean,
        res_m2,
        res_count,
    )


fn layer_norm_gpu_warp_tiling[
    type: DType, rank: Int, simd_width: Int, shape: DimList
](
    data: NDBuffer[type, 2, shape],
    gamma: NDBuffer[type, 1, DimList(1)],
    beta: NDBuffer[type, 1, DimList(1)],
    epsilon: NDBuffer[type, 1, DimList(1)],
):
    alias align = alignof[SIMD[type, simd_width]]()
    var num_rows = data.dim[0]()
    var num_cols = data.dim[1]()
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    var vec_data = SIMD[type, simd_width]()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    var idx = tid * simd_width
    var thread_mean = Scalar[type]()
    var thread_m2 = Scalar[type]()
    var thread_count = Scalar[type]()

    # To utilize simd vector load
    vec_data = data.load[width=simd_width, alignment=align](Index(row, idx))

    # every thread computes its own simd width of mean and variance
    for i in range(simd_width):
        welford_update[type](vec_data[i], thread_mean, thread_m2, thread_count)

    # a whole block computes part of the row main and variance and broadcasts to
    # threadIdx 0 to update the final row mean and variance
    welford_block_all_reduce(
        thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
    )

    var row_var = max((row_m2 / row_count), 0.0)

    var norm_factor = 1 / sqrt(row_var + epsilon[0])
    var norm_val = (vec_data - row_mean) * norm_factor * gamma[0] + beta[0]
    data.store[width=simd_width, alignment=align](Index(row, idx), norm_val)


fn layer_norm_gpu_block[
    type: DType, rank: Int, simd_width: Int, shape: DimList
](
    data: NDBuffer[type, 2, shape],
    gamma: NDBuffer[type, 1, DimList(1)],
    beta: NDBuffer[type, 1, DimList(1)],
    epsilon: NDBuffer[type, 1, DimList(1)],
):
    alias align = alignof[SIMD[type, simd_width]]()
    var num_rows = data.dim[0]()
    var num_cols = data.dim[1]()
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    # Every block has a single row to process
    for x in range(tid * simd_width, num_cols, BlockDim.x() * simd_width):
        var thread_mean = Scalar[type]()
        var thread_m2 = Scalar[type]()
        var thread_count = Scalar[type]()

        # To utilize simd vector load
        var vec = data.load[width=simd_width, alignment=align](Index(row, x))

        # every thread computes its own simd width of mean and variance
        @parameter
        for i in range(simd_width):
            welford_update[type](vec[i], thread_mean, thread_m2, thread_count)

        # a whole block computes part of the row main and variance and broadcasts to
        # threadIdx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

    var row_var = max((row_m2 / row_count), 0.0)

    # need a pass again to perform in place normalization
    for x in range(tid * simd_width, num_cols, BlockDim.x() * simd_width):
        var out_val = data.load[width=simd_width, alignment=align](
            Index(row, x)
        )
        var norm_factor = 1 / sqrt(row_var + epsilon[0])

        var norm_val = (out_val - row_mean) * norm_factor * gamma[0] + beta[0]
        data.store[width=simd_width, alignment=align](Index(row, x), norm_val)


fn layer_norm[
    simd_width: Int,
    type: DType,
    input_fn: fn[mytype: DType, width: Int] (Int, Int) capturing -> SIMD[
        mytype, width
    ],
    shape: DimList,
    inner_dim: DimList,
](
    out_buf: NDBuffer[type, 2, shape],
    gamma_buf: NDBuffer[type, 1, inner_dim],
    beta_buf: NDBuffer[type, 1, inner_dim],
    eps: Scalar[type],
):
    """Computes layernorm(elementwise_fn(x)) across the last dimension of x, where layernorm is
    defined as $(x-mean(x))/(sqrt(var(x)+eps)*gamma + beta$.

    Currently performs 3 passes over the input data. This can be reduced to 2 by
    fusing the add, mean, and variance loops using Welford's algorithm.

    Parameters:
        simd_width: The vector width for the computation.
        type: The x and out buffers' elements dtype.
        input_fn: Function called to generate an input value.
        shape: The x and out buffers' shape.
        inner_dim: The shape of gamma_buf and beta_buf.

    Args:
        out_buf: The output buffer.
        gamma_buf: The gamma value to use in the layernorm calculation.
        beta_buf: The beta value to use in the layernorm calculation.
        eps: The eps value to use in the layernorm calculation.
    """

    var m = out_buf.dim[0]()
    var n = out_buf.dim[1]()  # contiguous

    for i in range(m):
        var start_coord = StaticIntTuple[2](i, 0)
        var out_slice = Buffer[type, shape.at[1]()](
            out_buf._offset(start_coord), n
        )

        @__copy_capture(n)
        @parameter
        fn input_gen_wrapper[
            return_type: DType, simd_width: Int
        ](idx: Int) -> SIMD[return_type, simd_width]:
            return input_fn[return_type, simd_width](idx, i)

        var sum_val = map_reduce[
            simd_width,
            shape.at[1](),
            type,
            type,
            input_gen_wrapper,
            _simd_sum_elementwise,
            _simd_sum,
        ](out_slice, 0)

        @__copy_capture(sum_val, n)
        @parameter
        fn _sum_to_mean() -> Scalar[type]:
            @parameter
            if type.is_integral():
                return sum_val // n
            return sum_val / n

        var mean_val = _sum_to_mean()

        var var_val = variance(out_slice, mean_val, 0)  # use biased estimator

        var norm_factor = 1 / sqrt(var_val + eps)

        @__copy_capture(out_slice, norm_factor, mean_val)
        @parameter
        fn _normalize[simd_width: Int](idx: Int):
            var out_val = out_slice.load[width=simd_width](idx)
            var norm_val = (out_val - mean_val) * norm_factor * gamma_buf.load[
                width=simd_width
            ](idx) + beta_buf.load[width=simd_width](idx)
            out_slice.store(idx, norm_val)

        vectorize[_normalize, simd_width](n)


fn layer_norm[
    type: DType,
    input_0_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    rank: Int,
](
    shape: StaticIntTuple[rank],
    gamma: NDBuffer[type, 1],
    beta: NDBuffer[type, 1],
    epsilon: NDBuffer[type, 1],
    output: NDBuffer[type, rank],
):
    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP](
        "mojo.layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ) as t:
        var eps = epsilon[0]

        alias simd_width = simdwidthof[type]()

        var last_dim = shape[rank - 1]
        var prod_all_but_last_dim = shape.flattened_length() // last_dim
        var flat_shape = StaticIntTuple[2](prod_all_but_last_dim, last_dim)

        var output_buf = reshape[rank, 2, type, True](output, flat_shape)

        var num_workers = min(parallelism_level(), prod_all_but_last_dim)
        var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

        @__copy_capture(
            chunk_size, prod_all_but_last_dim, last_dim, output_buf, eps
        )
        @parameter
        fn task_func(thread_id: Int):
            var num_rows = min(
                chunk_size, prod_all_but_last_dim - thread_id * chunk_size
            )
            var row_idx = thread_id * chunk_size
            var thread_starting_coord = StaticIntTuple[2](row_idx, 0)
            var per_thread_dims = DimList(num_rows, last_dim)
            var output_buf_view = NDBuffer[type, 2](
                output_buf._offset(thread_starting_coord), per_thread_dims
            )

            @__copy_capture(row_idx, eps)
            @parameter
            @always_inline
            # Translate given 2d index back to original Nd tensor
            fn input_fn_2d[
                return_type: DType, simd_width: Int
            ](idx: Int, row: Int) -> SIMD[return_type, simd_width]:
                var indices = _get_nd_indices_from_flat_index[rank](
                    row_idx + row, shape, rank - 1
                )
                indices[rank - 1] = idx
                var input_val = input_0_fn[simd_width, rank](indices)
                return input_val.cast[return_type]()

            layer_norm[simd_width, type, input_fn_2d](
                output_buf_view, gamma, beta, eps
            )

        sync_parallelize[task_func](num_workers)


@mogg_register("layer_norm_shape")
@always_inline
fn layer_norm_shape[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    gamma: NDBuffer[type, 1, DimList(1)],
    beta: NDBuffer[type, 1, DimList(1)],
    epsilon: NDBuffer[type, 1, DimList(1)],
) -> StaticIntTuple[rank]:
    """
    Compute the output shape of a `layer_norm` operation.

    Parameters:
        type: Type of the input tensors.
        rank: Rank of the input tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        gamma: The tensor for gamma coefficient.
        beta: The tensor for beta coefficient.
        epsilon: The tensor for epsilon coefficient.

    Returns:
        The output shape.
    """
    return input.get_shape()
