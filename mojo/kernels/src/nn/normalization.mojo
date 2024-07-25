# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv, rsqrt
from collections import OptionalReg
from algorithm import map_reduce, mean, variance, vectorize
from algorithm.functional import sync_parallelize
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _simd_sum,
    _simd_sum_elementwise,
)
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from register import mogg_register
from runtime.asyncrt import parallelism_level
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
from gpu.host._compile import _get_nvptx_target
from gpu.shuffle import _static_log2, shuffle_down, shuffle_idx
from gpu.memory import AddressSpace
from .reshape import reshape
from runtime.asyncrt import MojoCallContextPtr


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


fn layer_norm_gpu_warp_tiling_vector[
    type: DType,
    simd_width: Int,
    rank: Int,
    input_func: fn[width: Int, _r: Int] (StaticIntTuple[_r]) capturing -> SIMD[
        type, width
    ],
    gamma_fn: fn[_width: Int, _r: Int] (StaticIntTuple[_r]) capturing -> SIMD[
        type, _width
    ],
](data: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type],):
    alias align = alignof[SIMD[type, simd_width]]()
    var num_rows = data.dim[0]()
    var num_cols = data.dim[1]()
    var tid: UInt = ThreadIdx.x()
    var row: UInt = BlockIdx.x()

    var vec_data = SIMD[type, simd_width]()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    var idx: UInt = tid * UInt(simd_width.value)
    var thread_mean = Scalar[type]()
    var thread_m2 = Scalar[type]()
    var thread_count = Scalar[type]()

    # To utilize simd vector load
    if idx < UInt(num_cols.value):
        vec_data = input_func[simd_width, rank](
            StaticIntTuple[rank](row.value, idx.value)
        )
        # every thread computes its own simd width of mean and variance
        for i in range(simd_width):
            welford_update(vec_data[i], thread_mean, thread_m2, thread_count)

    # a whole block computes part of the row main and variance and broadcasts to
    # threadIdx 0 to update the final row mean and variance
    welford_block_all_reduce(
        thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
    )

    var row_var = max((row_m2 / (row_count - 1)), 0.0)

    var norm_factor = rsqrt(row_var + epsilon)
    if idx < UInt(num_cols.value):
        var gamma_val = gamma_fn[simd_width, 1](StaticIntTuple[1](idx))
        var norm_val = (
            vec_data - row_mean
        ) * norm_factor * gamma_val + beta.load[
            width=simd_width, alignment=align
        ](
            Index(idx)
        )
        data.store[width=simd_width, alignment=align](Index(row, idx), norm_val)


fn layer_norm_gpu_warp_tiling_scalar[
    type: DType,
    simd_width: Int,
    rank: Int,
    input_func: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    gamma_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
](data: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type],):
    var num_rows = data.dim[0]()
    var num_cols = data.dim[1]()
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    var vec_data = Scalar[type]()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    var thread_mean = Scalar[type]()
    var thread_m2 = Scalar[type]()
    var thread_count = Scalar[type]()

    # To utilize simd vector load
    if tid < num_cols:
        vec_data = input_func[1, rank](StaticIntTuple[rank](int(row), int(tid)))
        welford_update(vec_data, thread_mean, thread_m2, thread_count)

    # a whole block computes part of the row main and variance and broadcasts to
    # threadIdx 0 to update the final row mean and variance
    welford_block_all_reduce(
        thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
    )

    var row_var = max((row_m2 / (row_count - 1)), 0.0)
    var norm_factor = rsqrt(row_var + epsilon)
    if tid < num_cols:
        var norm_val = (vec_data - row_mean) * norm_factor * gamma_fn[1, 1](
            StaticIntTuple[1](tid.value)
        ) + beta.load(tid.value)
        data.store(Index(row, tid), norm_val)


fn layer_norm_gpu_block_vector[
    type: DType,
    simd_width: Int,
    rank: Int,
    input_func: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    gamma_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
](data: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type],):
    alias align = alignof[SIMD[type, simd_width]]()
    var num_cols: UInt = data.dim[1]().value
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // UInt(simd_width.value), BlockDim.x())):
        var thread_mean = Scalar[type]()
        var thread_m2 = Scalar[type]()
        var thread_count = Scalar[type]()

        # To utilize simd vector load
        var vec_data = SIMD[type, simd_width]()
        var offset = x * BlockDim.x() * simd_width + tid * simd_width

        if offset < num_cols:
            vec_data = input_func[simd_width, rank](
                StaticIntTuple[rank](row.value, offset.value)
            )

            @parameter
            for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        # a whole block computes part of the row main and variance and broadcasts to
        # threadIdx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

    var row_var = max((row_m2 / (row_count - 1)), 0.0)

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols // simd_width, BlockDim.x())):
        var offset = x * BlockDim.x() * simd_width + tid * simd_width
        var norm_factor = rsqrt(row_var + epsilon)

        if offset < num_cols:
            var gamma_val = gamma_fn[simd_width, 1](StaticIntTuple[1](offset))
            var beta_val = beta.load[width=simd_width, alignment=align](
                Index(offset)
            )

            var vec_data = input_func[simd_width, rank](
                StaticIntTuple[rank](row.value, offset.value)
            )
            var norm_val = (
                (vec_data - row_mean) * norm_factor * gamma_val
            ) + beta_val
            data.store[width=simd_width, alignment=align](
                Index(row, offset), norm_val
            )


fn layer_norm_gpu_block_scalar[
    type: DType,
    simd_width: Int,
    rank: Int,
    input_func: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    gamma_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
](data: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type],):
    var num_rows = data.dim[0]()
    var num_cols = data.dim[1]()
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[type]()
    var row_m2 = Scalar[type]()
    var row_count = Scalar[type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols, BlockDim.x())):
        var thread_mean = Scalar[type]()
        var thread_m2 = Scalar[type]()
        var thread_count = Scalar[type]()

        # To utilize simd vector load
        var vec_data = Scalar[type]()
        var offset = x * BlockDim.x() + tid

        if offset < num_cols:
            vec_data = input_func[1, rank](
                StaticIntTuple[rank](row.value, offset.value)
            )
            welford_update(vec_data, thread_mean, thread_m2, thread_count)

        # a whole block computes part of the row main and variance and broadcasts to
        # threadIdx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

    var row_var = max((row_m2 / (row_count - 1)), 0.0)

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols, BlockDim.x())):
        var vec_data = Scalar[type]()
        var offset = x * BlockDim.x() + tid
        var norm_factor = rsqrt(row_var + epsilon)

        if offset < num_cols:
            vec_data = input_func[1, rank](
                StaticIntTuple[rank](row.value, offset.value)
            )
            var norm_val = (vec_data - row_mean) * norm_factor * gamma_fn[1, 1](
                StaticIntTuple[1](offset.value)
            ) + beta.load(offset.value)
            data.store(Index(row, offset), norm_val)


fn layer_norm_gpu[
    type: DType,
    input_0_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    input_1_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    rank: Int,
](
    shape: StaticIntTuple[rank],
    beta: NDBuffer[type, 1],
    epsilon: Scalar[type],
    output: NDBuffer[type, rank],
    ctx: DeviceContext,
) raises:
    constrained[rank == 2, "unsupported gpu layer_norm rank"]()

    var rows = output.dim[0]()
    var cols = output.dim[1]()

    alias simd_width = simdwidthof[type, target = _get_nvptx_target()]()
    alias max_warps_per_block = 32

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    # When the number of columns are less enough that they can be placed in
    # registers we do warp tiling which is a single pass to do mean/var computation
    # and normalization.
    if cols <= (WARP_SIZE * simd_width * max_warps_per_block):
        if cols % simd_width == 0:
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_warp_tiling_vector[
                    type, simd_width, rank, input_0_fn, input_1_fn
                ]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_warp_tiling_scalar[
                    type, simd_width, rank, input_0_fn, input_1_fn
                ]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
    else:
        if cols % simd_width == 0:
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_block_vector[
                    type, simd_width, rank, input_0_fn, input_1_fn
                ]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_block_scalar[
                    type, simd_width, rank, input_0_fn, input_1_fn
                ]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


fn layer_norm_cpu[
    simd_width: Int,
    type: DType,
    input_fn: fn[mytype: DType, width: Int] (Int, Int) capturing -> SIMD[
        mytype, width
    ],
    gamma_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
](out_buf: NDBuffer[type, 2, _], beta: NDBuffer[type, 1], eps: Scalar[type],):
    """Computes layernorm(elementwise_fn(x)) across the last dimension of x, where layernorm is
    defined as $(x-mean(x))/(sqrt(var(x)+eps)*gamma_fn + beta$.

    Currently performs 3 passes over the input data. This can be reduced to 2 by
    fusing the add, mean, and variance loops using Welford's algorithm.

    Parameters:
        simd_width: The vector width for the computation.
        type: The x and out buffers' elements dtype.
        input_fn: Function called to generate an input value.
        gamma_fn: Function called to generate a gamma value.

    Args:
        out_buf: The output buffer.
        beta: The beta value to use in the layernorm calculation.
        eps: The eps value to use in the layernorm calculation.
    """

    var m = out_buf.dim[0]()
    var n = out_buf.dim[1]()  # contiguous

    for i in range(m):
        var start_coord = StaticIntTuple[2](i, 0)
        var out_slice = Buffer[type, out_buf.shape.at[1]()](
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
            out_buf.shape.at[1](),
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

        var norm_factor = rsqrt(var_val + eps)

        @__copy_capture(out_slice, norm_factor, mean_val)
        @parameter
        fn _normalize[simd_width: Int](idx: Int):
            var out_val = out_slice.load[width=simd_width](idx)
            var gamma_val = gamma_fn[simd_width, 1](StaticIntTuple[1](idx))
            var norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_val + beta.load[width=simd_width](idx)
            out_slice.store(idx, norm_val)

        vectorize[_normalize, simd_width](n)


fn layer_norm_cpu[
    type: DType,
    input_0_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    gamma_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    rank: Int,
](
    shape: StaticIntTuple[rank],
    beta: NDBuffer[type, 1],
    epsilon: Scalar[type],
    output: NDBuffer[type, rank],
):
    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP](
        "mojo.layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        alias simd_width = simdwidthof[type]()

        var last_dim = shape[rank - 1]
        var prod_all_but_last_dim = shape.flattened_length() // last_dim
        var flat_shape = StaticIntTuple[2](prod_all_but_last_dim, last_dim)

        var output_buf = reshape[rank, 2, type, True](output, flat_shape)

        var num_workers = min(parallelism_level(), prod_all_but_last_dim)
        var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

        @__copy_capture(
            chunk_size, prod_all_but_last_dim, last_dim, output_buf, epsilon
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

            @__copy_capture(row_idx, epsilon)
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

            layer_norm_cpu[simd_width, type, input_fn_2d, gamma_fn](
                output_buf_view, beta, epsilon
            )

        sync_parallelize[task_func](num_workers)


fn layer_norm[
    type: DType,
    input_0_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    input_1_fn: fn[_width: Int, _rank: Int] (
        StaticIntTuple[_rank]
    ) capturing -> SIMD[type, _width],
    rank: Int,
    target: StringLiteral = "cpu",
](
    shape: StaticIntTuple[rank],
    gamma_shape: StaticIntTuple[1],
    beta: NDBuffer[type, 1],
    epsilon: Scalar[type],
    output: NDBuffer[type, rank],
    ctx: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    constrained[target in ("cpu", "cuda"), "unsupported target"]()

    # Note: we only support reduction along the last dimension
    if gamma_shape[0] != shape[rank - 1]:
        ctx.set_to_error("Gamma size does not match dimension of reduction.")
        return

    if beta.dynamic_shape[0] != shape[rank - 1]:
        ctx.set_to_error("Beta size does not match dimension of reduction.")
        return

    if output.dynamic_shape != shape:
        ctx.set_to_error("Input and output buffers are not same shape")
        return

    @parameter
    if target == "cpu":
        layer_norm_cpu[type, input_0_fn, input_1_fn](
            shape, beta, epsilon, output
        )
    else:
        layer_norm_gpu[type, input_0_fn, input_1_fn, rank](
            shape, beta, epsilon, output, ctx.get_cuda_device()
        )


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
    epsilon: Scalar[type],
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
