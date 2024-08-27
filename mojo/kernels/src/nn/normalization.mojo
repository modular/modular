# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceildiv, isqrt
from sys.info import alignof, simdwidthof

from algorithm import map_reduce, mean, variance, vectorize
from algorithm.functional import sync_parallelize
from algorithm.reduction import (
    _get_nd_indices_from_flat_index,
    _simd_sum,
    _simd_sum_elementwise,
)
from buffer import Buffer, NDBuffer
from buffer.dimlist import DimList
from gpu import (
    WARP_SIZE,
    BlockDim,
    BlockIdx,
    ThreadIdx,
    barrier,
    lane_id,
    syncwarp,
)
from gpu.host._compile import _get_nvptx_target
from gpu.host.device_context import DeviceContext
from gpu.memory import AddressSpace
from gpu.shuffle import _static_log2, shuffle_down, shuffle_idx, warp_reduce_add
from memory import stack_allocation
from register import mogg_register
from runtime.asyncrt import MojoCallContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, StaticIntTuple, StaticTuple
from utils.numerics import get_accum_type

from .reshape import reshape


@always_inline
fn block_reduce[type: DType](val: Scalar[type]) -> Scalar[type]:
    alias max_warps_per_block = 32
    var m2_shared = stack_allocation[
        max_warps_per_block, type, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()

    var tid = ThreadIdx.x()
    for i in range(tid, max_warps_per_block, BlockDim.x()):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp_reduce_add(val)

    var warp_id = tid // WARP_SIZE
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0:
        var block_m2 = warp_reduce_add(m2_shared[lane_idx])
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


fn rms_norm_gpu_warp_tiling[
    type: DType,
    simd_width: Int,
](data: NDBuffer[type, 2], gamma: NDBuffer[type, 1], epsilon: Scalar[type]):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols = data.dim[1]()
    var tid: UInt = ThreadIdx.x()
    var row: UInt = BlockIdx.x()

    var vec_data = SIMD[accum_type, simd_width]()

    var idx: UInt = tid * simd_width
    var thread_m2 = Scalar[accum_type](0)

    # To utilize simd vector load
    if idx < num_cols:
        vec_data = data.load[width=simd_width, alignment=align](
            Index(row, idx)
        ).cast[accum_type]()
        thread_m2 = (vec_data**2).reduce_add()

    var row_m2 = block_reduce(thread_m2)
    var norm_factor = isqrt((row_m2 / num_cols) + epsilon.cast[accum_type]())

    if idx < num_cols:
        var gamma_val = gamma.load[width=simd_width, alignment=align](
            Index(idx)
        )
        var norm_val = vec_data * norm_factor * gamma_val.cast[accum_type]()
        data.store[alignment=align](Index(row, idx), norm_val.cast[type]())


fn rms_norm_gpu_block[
    type: DType,
    simd_width: Int,
](data: NDBuffer[type, 2], gamma: NDBuffer[type, 1], epsilon: Scalar[type]):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols = data.dim[1]()
    var tid: UInt = ThreadIdx.x()
    var row: UInt = BlockIdx.x()
    var thread_m2 = Scalar[accum_type](0)

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, BlockDim.x())):
        var offset = x * BlockDim.x() * simd_width + tid * simd_width
        if offset < num_cols:
            var vec_data = data.load[width=simd_width, alignment=align](
                Index(row, offset)
            ).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce(thread_m2)
    var norm_factor = isqrt((row_m2 / num_cols) + epsilon.cast[accum_type]())

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols // simd_width, BlockDim.x())):
        var offset = x * BlockDim.x() * simd_width + tid * simd_width

        if offset < num_cols:
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                Index(offset)
            )

            var vec_data = data.load[width=simd_width, alignment=align](
                Index(row, offset)
            ).cast[accum_type]()
            var norm_val = (
                vec_data * norm_factor * gamma_val.cast[accum_type]()
            )
            data.store[alignment=align](
                Index(row, offset), norm_val.cast[type]()
            )


# using numerically stable Welford online algorithm to compute single pass mean and variance
fn welford_update[
    type: DType, //
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
    type: DType, //
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
    type: DType, //
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
        welford_combine(mean, m2, count, res_mean, res_m2, res_count)


fn welford_warp_all_reduce[
    type: DType, //
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
    type: DType, //
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
            warp_mean = mean_shared[lane_idx]
            warp_m2 = m2_shared[lane_idx]
            warp_count = count_shared[lane_idx]
        else:
            warp_mean = Scalar[type](0)
            warp_m2 = Scalar[type](0)
            warp_count = Scalar[type](0)
        syncwarp()
        var block_mean = Scalar[type](0)
        var block_m2 = Scalar[type](0)
        var block_count = Scalar[type](0)
        welford_warp_reduce(
            warp_mean, warp_m2, warp_count, block_mean, block_m2, block_count
        )
        if lane_idx == 0:
            mean_broadcast[0] = block_mean
            m2_broadcast[0] = block_m2
            count_broadcast[0] = block_count

    barrier()

    welford_combine(
        mean_broadcast[0],
        m2_broadcast[0],
        count_broadcast[0],
        res_mean,
        res_m2,
        res_count,
    )


fn layer_norm_gpu_warp_tiling[
    type: DType,
    simd_width: UInt,
    input_func: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    gamma_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
](data: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type]):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols = data.dim[1]()
    var tid: UInt = ThreadIdx.x()
    var row: UInt = BlockIdx.x()

    var vec_data = SIMD[accum_type, simd_width]()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    var idx: UInt = tid * simd_width
    var thread_mean = Scalar[accum_type]()
    var thread_m2 = Scalar[accum_type]()
    var thread_count = Scalar[accum_type]()

    if idx < num_cols:
        vec_data = input_func[simd_width](Index(row, idx)).cast[accum_type]()

        # every thread computes its own simd width of mean and variance
        @parameter
        for i in range(int(simd_width)):
            welford_update(vec_data[i], thread_mean, thread_m2, thread_count)

    # a whole block computes part of the row main and variance and broadcasts to
    # threadIdx 0 to update the final row mean and variance
    welford_block_all_reduce(
        thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
    )

    var row_var = max((row_m2 / (row_count - 1)), 0.0)
    var norm_factor = isqrt(row_var + epsilon.cast[accum_type]())

    if idx < num_cols:
        var gamma_val = gamma_fn[simd_width](Index(idx))
        var beta_val = beta.load[width=simd_width, alignment=align](Index(idx))
        var norm_val = (vec_data - row_mean) * norm_factor * gamma_val.cast[
            accum_type
        ]() + beta_val.cast[accum_type]()
        data.store[alignment=align](Index(row, idx), norm_val.cast[type]())


fn layer_norm_gpu_block[
    type: DType,
    simd_width: UInt,
    input_func: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    gamma_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
](output: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Scalar[type]):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols: UInt = output.dim[1]()
    var tid = ThreadIdx.x()
    var row = BlockIdx.x()

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, BlockDim.x())):
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        var offset = x * BlockDim.x() * simd_width + tid * simd_width

        if offset < num_cols:
            var vec_data = input_func[simd_width](Index(row, offset)).cast[
                accum_type
            ]()

            @parameter
            for i in range(int(simd_width)):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        # a whole block computes part of the row main and variance and broadcasts to
        # threadIdx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

    var row_var = max(row_m2 / (row_count - 1), 0)
    var norm_factor = isqrt(row_var + epsilon.cast[accum_type]())

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols // simd_width, BlockDim.x())):
        var offset = x * BlockDim.x() * simd_width + tid * simd_width

        if offset < num_cols:
            var gamma_val = gamma_fn[simd_width](Index(offset))
            var beta_val = beta.load[width=simd_width, alignment=align](offset)

            var vec_data = input_func[simd_width](Index(row, offset)).cast[
                accum_type
            ]()
            var norm_val = (
                (vec_data - row_mean)
                * norm_factor
                * gamma_val.cast[accum_type]()
            ) + beta_val.cast[accum_type]()
            output.store[alignment=align](
                Index(row, offset), norm_val.cast[type]()
            )


fn layer_norm_reshape[
    type: DType, rank: Int, output_rank: Int
](
    shape: StaticIntTuple[rank],
    buf: NDBuffer[type, rank],
) -> NDBuffer[
    type, output_rank
]:
    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    var new_shape = StaticIntTuple[output_rank](prod_all_but_last_dim, last_dim)
    var output_rs = reshape[output_rank](buf, new_shape)
    return output_rs


fn rms_norm_gpu[
    type: DType,
    rank: Int,
](
    shape: StaticIntTuple[rank],
    gamma: NDBuffer[type, 1],
    epsilon: Scalar[type],
    output: NDBuffer[type, rank],
    ctx: DeviceContext,
) raises:
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    alias rank_rs = 2
    var output_rs = layer_norm_reshape[type, rank, rank_rs](shape, output)
    var rows = output_rs.dim[0]()
    var cols = output_rs.dim[1]()

    alias simd_width = simdwidthof[type, target = _get_nvptx_target()]()
    alias max_warps_per_block = 32

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    if cols % simd_width == 0:
        # When the number of columns are less enough that they can be placed in
        # registers we do warp tiling which is a single pass to do mean/var
        # computation and normalization.
        if cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            var gpu_func = ctx.compile_function[
                rms_norm_gpu_warp_tiling[type, simd_width]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output_rs,
                gamma,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            var gpu_func = ctx.compile_function[
                rms_norm_gpu_block[type, simd_width]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output_rs,
                gamma,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
    else:
        var gpu_func = ctx.compile_function[rms_norm_gpu_block[type, 1]]()
        ctx.enqueue_function(
            gpu_func,
            output_rs,
            gamma,
            epsilon,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )


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
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    alias rank_rs = 2
    var output_rs = layer_norm_reshape[type, rank, rank_rs](shape, output)
    var rows = output_rs.dim[0]()
    var cols = output_rs.dim[1]()

    alias simd_width = simdwidthof[type, target = _get_nvptx_target()]()
    alias max_warps_per_block = 32

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    if cols % simd_width == 0:
        # When the number of columns are less enough that they can be placed in
        # registers we do warp tiling which is a single pass to do mean/var
        # computation and normalization.
        if cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_warp_tiling[
                    type, simd_width, input_0_fn, input_1_fn
                ]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output_rs,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            var gpu_func = ctx.compile_function[
                layer_norm_gpu_block[type, simd_width, input_0_fn, input_1_fn]
            ]()
            ctx.enqueue_function(
                gpu_func,
                output_rs,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
    else:
        var gpu_func = ctx.compile_function[
            layer_norm_gpu_block[type, 1, input_0_fn, input_1_fn]
        ]()
        ctx.enqueue_function(
            gpu_func,
            output_rs,
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

        var norm_factor = isqrt(var_val + eps)

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

        var output_buf = reshape[2](output, flat_shape)

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
    elif "cuda" in target:
        layer_norm_gpu[type, input_0_fn, input_1_fn, rank](
            shape, beta, epsilon, output, ctx.get_device_context()
        )
    else:
        constrained[False, "unsupported target " + target]()


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
