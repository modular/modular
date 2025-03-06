# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import align_down, ceildiv, isqrt
from sys.info import alignof, simdwidthof

from algorithm import map_reduce, mean, variance, vectorize
from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    sync_parallelize,
)
from algorithm.reduction import _simd_sum, _simd_sum_elementwise
from bit import log2_floor
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    syncwarp,
    thread_idx,
)
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu, is_gpu
from gpu.memory import AddressSpace
import gpu.warp as warp
from memory import stack_allocation
from register import register_internal
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type

from .reshape import reshape


@always_inline
fn block_reduce[
    type: DType, max_warps_per_block: Int
](val: Scalar[type]) -> Scalar[type]:
    var m2_shared = stack_allocation[
        max_warps_per_block, type, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, type, address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    for i in range(tid, max_warps_per_block, block_dim.x):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp.sum(val)

    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0 and lane_idx < max_warps_per_block:
        var block_m2 = warp.lane_group_sum[nthreads=max_warps_per_block](
            m2_shared[lane_idx]
        )
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


# using numerically stable Welford online algorithm to compute single pass mean and variance
fn welford_update[
    type: DType, //
](
    val: Scalar[type],
    mut mean: Scalar[type],
    mut m2: Scalar[type],
    mut count: Scalar[type],
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
    mut res_mean: Scalar[type],
    mut res_m2: Scalar[type],
    mut res_count: Scalar[type],
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
    mut res_mean: Scalar[type],
    mut res_m2: Scalar[type],
    mut res_count: Scalar[type],
):
    res_mean = thread_mean
    res_m2 = thread_m2
    res_count = thread_count

    alias limit = log2_floor(WARP_SIZE)

    @parameter
    for mask in reversed(range(limit)):
        var mean = warp.shuffle_down(res_mean, 1 << mask)
        var m2 = warp.shuffle_down(res_m2, 1 << mask)
        var count = warp.shuffle_down(res_count, 1 << mask)
        welford_combine(mean, m2, count, res_mean, res_m2, res_count)


fn welford_warp_all_reduce[
    type: DType, //
](
    thread_mean: Scalar[type],
    thread_m2: Scalar[type],
    thread_count: Scalar[type],
    mut res_mean: Scalar[type],
    mut res_m2: Scalar[type],
    mut res_count: Scalar[type],
):
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, res_mean, res_m2, res_count
    )
    # broadcasting res from warp lane_id 0 to all in a warp
    res_mean = warp.broadcast(res_mean)
    res_m2 = warp.broadcast(res_m2)
    res_count = warp.broadcast(res_count)


fn welford_block_all_reduce[
    type: DType, //
](
    thread_mean: Scalar[type],
    thread_m2: Scalar[type],
    thread_count: Scalar[type],
    mut res_mean: Scalar[type],
    mut res_m2: Scalar[type],
    mut res_count: Scalar[type],
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

    var warp_idx = thread_idx.x // WARP_SIZE
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
        if thread_idx.x < (block_dim.x // WARP_SIZE):
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
    type: DType, //,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        type, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
](output: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Float32):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols = output.dim[1]()
    var tid: UInt = thread_idx.x
    var row: UInt = block_idx.x

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
        vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

        # every thread computes its own simd width of mean and variance
        @parameter
        for i in range(Int(simd_width)):
            welford_update(vec_data[i], thread_mean, thread_m2, thread_count)

    # a whole block computes part of the row main and variance and broadcasts to
    # thread_idx 0 to update the final row mean and variance
    welford_block_all_reduce(
        thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
    )

    var row_var = max(row_m2 / row_count, 0.0)
    var norm_factor = isqrt(row_var + epsilon.cast[accum_type]())

    if idx < num_cols:
        var gamma_val = gamma_fn[simd_width](Index(idx))
        var beta_val = beta.load[width=simd_width, alignment=align](Index(idx))
        var norm_val = (vec_data - row_mean) * norm_factor * gamma_val.cast[
            accum_type
        ]() + beta_val.cast[accum_type]()
        output.store[alignment=align](Index(row, idx), norm_val.cast[type]())


fn layer_norm_gpu_block[
    type: DType, //,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        type, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
](output: NDBuffer[type, 2], beta: NDBuffer[type, 1], epsilon: Float32):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var num_cols: UInt = output.dim[1]()
    var tid = thread_idx.x
    var row = block_idx.x

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()

            @parameter
            for i in range(Int(simd_width)):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        # a whole block computes part of the row main and variance and broadcasts to
        # thread_idx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

    var row_var = max(row_m2 / row_count, 0)
    var norm_factor = isqrt(row_var + epsilon.cast[accum_type]())

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var gamma_val = gamma_fn[simd_width](Index(offset))
            var beta_val = beta.load[width=simd_width, alignment=align](offset)

            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val = (
                (vec_data - row_mean)
                * norm_factor
                * gamma_val.cast[accum_type]()
            ) + beta_val.cast[accum_type]()
            output.store[alignment=align](
                Index(row, offset), norm_val.cast[type]()
            )


fn layer_norm_reshape[
    type: DType, rank: Int, //, output_rank: Int
](
    shape: IndexList[rank, **_],
    buf: NDBuffer[type, rank, *_],
    out result: NDBuffer[type, output_rank],
):
    @parameter
    if rank == output_rank:
        return rebind[__type_of(result)](buf)

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    var new_shape = IndexList[output_rank](prod_all_but_last_dim, last_dim)
    var output_rs = reshape[output_rank](buf, new_shape)
    return output_rs


fn layer_norm_gpu[
    type: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
](
    shape: IndexList[rank, **_],
    beta: NDBuffer[type, 1],
    epsilon: Float32,
    output: NDBuffer[type, rank, *_],
    *,
    ctx: DeviceContext,
) raises:
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    alias rank_rs = 2
    var output_rs = layer_norm_reshape[rank_rs](shape, output)
    var rows = output_rs.dim[0]()
    var cols = output_rs.dim[1]()

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[type, simd_width]:
        # Translate given 2d index back to original Nd tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    alias simd_width = simdwidthof[type, target = _get_gpu_target()]()
    alias max_warps_per_block = ctx.device_info.max_thread_block_size // WARP_SIZE

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
            ctx.enqueue_function[
                layer_norm_gpu_warp_tiling[simd_width, input_fn_2d, gamma_fn]
            ](
                output_rs,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            ctx.enqueue_function[
                layer_norm_gpu_block[simd_width, input_fn_2d, gamma_fn]
            ](
                output_rs,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
    else:
        ctx.enqueue_function[layer_norm_gpu_block[1, input_fn_2d, gamma_fn]](
            output_rs,
            beta,
            epsilon,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )


@always_inline
fn _sum_to_mean[type: DType, //](sum_val: Scalar[type], n: Int) -> Scalar[type]:
    @parameter
    if type.is_integral():
        return sum_val // n
    return sum_val / n


fn layer_norm_cpu[
    type: DType, //,
    input_fn: fn[width: Int] (Int, Int) capturing -> SIMD[type, width],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
](
    out_buf: NDBuffer[type, 2, _], beta: NDBuffer[type, 1], epsilon: Float32
) raises:
    """Computes layernorm(elementwise_fn(x)) across the last dimension of x, where layernorm is
    defined as $(x-mean(x))/(sqrt(var(x)+eps)*gamma_fn + beta$.

    Currently performs 3 passes over the input data. This can be reduced to 2 by
    fusing the add, mean, and variance loops using Welford's algorithm.

    Parameters:
        type: The x and out buffers' elements dtype.
        input_fn: Function called to generate an input value.
        gamma_fn: Function called to generate a gamma value.

    Args:
        out_buf: The output buffer.
        beta: The beta value to use in the layernorm calculation.
        epsilon: The eps value to use in the layernorm calculation.
    """
    alias simd_width = simdwidthof[type]()

    var num_rows = out_buf.dim[0]()
    var num_cols = out_buf.dim[1]()

    for row in range(num_rows):
        var out_slice = NDBuffer[type, 1, out_buf.shape.at[1]()](
            out_buf._offset(Index(row, 0)), num_cols
        )

        @__copy_capture(row)
        @parameter
        fn input_gen_wrapper[
            type: DType, simd_width: Int
        ](col: Int) -> SIMD[type, simd_width]:
            return input_fn[simd_width](row, col).cast[type]()

        var sum_val = map_reduce[
            simd_width,
            out_buf.shape.at[1](),
            type,
            type,
            __origin_of(),
            input_gen_wrapper,
            __origin_of(),
            _simd_sum_elementwise,
            _simd_sum,
        ](out_slice, 0)

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var var_val = variance(out_slice, mean_val, 0)  # use biased estimator
        var norm_factor = isqrt(var_val + epsilon.cast[type]())

        @__copy_capture(out_slice, norm_factor, mean_val)
        @parameter
        fn _normalize[simd_width: Int](col: Int):
            var out_val = out_slice.load[width=simd_width](col)
            var gamma_val = gamma_fn[simd_width, 1](col)
            var norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_val + beta.load[width=simd_width](col)
            out_slice.store(col, norm_val)

        vectorize[_normalize, simd_width](num_cols)


fn layer_norm_cpu[
    type: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
](
    shape: IndexList[rank, **_],
    beta: NDBuffer[type, 1],
    epsilon: Float32,
    output: NDBuffer[type, rank, *_],
):
    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    var flat_shape = Index(prod_all_but_last_dim, last_dim)

    var output_buf = reshape[2](output, flat_shape)

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(
        chunk_size, prod_all_but_last_dim, last_dim, output_buf, epsilon
    )
    @parameter
    fn task_func(thread_id: Int) raises:
        var num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        var row_idx = thread_id * chunk_size
        var thread_starting_coord = Index(row_idx, 0)
        var per_thread_dims = DimList(num_rows, last_dim)
        var output_buf_view = NDBuffer[type, 2](
            output_buf._offset(thread_starting_coord), per_thread_dims
        )

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) -> SIMD[type, simd_width]:
            # Translate given 2d index back to original Nd tensor
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width](indices.canonicalize())

        layer_norm_cpu[input_fn_2d, gamma_fn](output_buf_view, beta, epsilon)

    sync_parallelize[task_func](num_workers)


@register_internal("mo.layer_norm")
@always_inline
fn layer_norm[
    type: DType,
    rank: Int,
    input_0_fn: fn[_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, _width],
    input_1_fn: fn[_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[type, _width],
    /,
    target: StringLiteral = "cpu",
](
    shape: IndexList[rank],
    gamma_shape: IndexList[1],
    beta: NDBuffer[type, 1],
    epsilon: Float32,
    output: NDBuffer[type, rank, *_],
    ctx: DeviceContextPtr,
) raises:
    # Note: we only support reduction along the last dimension
    if gamma_shape[0] != shape[rank - 1]:
        raise Error("Gamma size does not match dimension of reduction.")

    if beta.dynamic_shape[0] != shape[rank - 1]:
        raise Error("Beta size does not match dimension of reduction.")

    if output.dynamic_shape.canonicalize() != shape.canonicalize():
        raise Error("Input and output buffers are not same shape")

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP](
        "layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):

        @parameter
        if is_cpu[target]():
            layer_norm_cpu[input_0_fn, input_1_fn](
                shape.canonicalize(), beta, epsilon, output
            )
        elif is_gpu[target]():
            layer_norm_gpu[input_0_fn, input_1_fn](
                shape.canonicalize(),
                beta,
                epsilon,
                output,
                ctx=ctx.get_device_context(),
            )
        else:
            constrained[False, "unsupported target " + target]()


@always_inline
fn layer_norm_shape[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    gamma: NDBuffer[type, 1, DimList(1)],
    beta: NDBuffer[type, 1, DimList(1)],
    epsilon: Float32,
) -> IndexList[rank]:
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


fn rms_norm_gpu_warp_tiling[
    type: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        type, width
    ],
    output_fn: fn[width: Int] (
        row: Int, col: Int, val: SIMD[type, width]
    ) capturing -> None,
](gamma: NDBuffer[type, 1], epsilon: Float32, num_cols: Int):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var tid: UInt = thread_idx.x
    var row: UInt = block_idx.x

    var vec_data = SIMD[accum_type, simd_width]()

    var idx: UInt = tid * simd_width
    var thread_m2 = Scalar[accum_type](0)

    # To utilize simd vector load
    if idx < num_cols:
        vec_data = input_fn[simd_width](row, idx).cast[accum_type]()
        thread_m2 = (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = isqrt((row_m2 / num_cols) + epsilon.cast[accum_type]())

    if idx < num_cols:
        var gamma_val = gamma.load[width=simd_width, alignment=align](
            Index(idx)
        )
        var norm_val = vec_data * norm_factor * gamma_val.cast[accum_type]()
        output_fn(row, idx, norm_val.cast[type]())


fn rms_norm_gpu_block[
    type: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        type, width
    ],
    output_fn: fn[width: Int] (
        row: Int, col: Int, val: SIMD[type, width]
    ) capturing -> None,
](gamma: NDBuffer[type, 1], epsilon: Float32, num_cols: Int):
    alias align = alignof[SIMD[type, simd_width]]()
    alias accum_type = get_accum_type[type]()

    var tid: UInt = thread_idx.x
    var row: UInt = block_idx.x
    var thread_m2 = Scalar[accum_type](0)

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width
        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = isqrt((row_m2 / num_cols) + epsilon.cast[accum_type]())

    # need a pass again to perform in place normalization
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                Index(offset)
            )

            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val = (
                vec_data * norm_factor * gamma_val.cast[accum_type]()
            )
            output_fn(row, offset, norm_val.cast[type]())


fn rms_norm_gpu[
    type: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_fn: fn[width: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
](
    shape: IndexList[rank, **_],
    gamma: NDBuffer[type, 1],
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    @parameter
    @always_inline
    fn output_fn_2d[
        simd_width: Int
    ](row: Int, col: Int, val: SIMD[type, simd_width]) -> None:
        # Translate given 2d index back to original Nd tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn(indices.canonicalize(), val)

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[type, simd_width]:
        # Translate given 2d index back to original Nd tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    alias simd_width = simdwidthof[type, target = _get_gpu_target()]()
    alias max_warps_per_block = ctx.device_info.max_thread_block_size // WARP_SIZE

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
            ctx.enqueue_function[
                rms_norm_gpu_warp_tiling[
                    simd_width, max_warps_per_block, input_fn_2d, output_fn_2d
                ]
            ](
                gamma,
                epsilon,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
        else:
            ctx.enqueue_function[
                rms_norm_gpu_block[
                    simd_width, max_warps_per_block, input_fn_2d, output_fn_2d
                ]
            ](
                gamma,
                epsilon,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
    else:
        ctx.enqueue_function[
            rms_norm_gpu_block[
                1, max_warps_per_block, input_fn_2d, output_fn_2d
            ]
        ](
            gamma,
            epsilon,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )


fn rms_norm_cpu[
    type: DType, //,
    input_fn: fn[width: Int] (Int, Int) capturing -> SIMD[type, width],
    output_fn: fn[width: Int] (Int, Int, SIMD[type, width]) capturing -> None,
](gamma: NDBuffer[type, 1], epsilon: Float32, out_shape: IndexList[2]):
    alias simd_width = simdwidthof[type]()

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    var simd_loop_end = align_down(num_cols, simd_width)

    for row in range(num_rows):
        var sum_simd = SIMD[type, simd_width]()
        for col in range(0, simd_loop_end, simd_width):
            sum_simd += input_fn[simd_width](row, col) ** 2

        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            sum_val += input_fn[1](row, col) ** 2

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = isqrt(mean_val + epsilon.cast[type]())

        @__copy_capture(norm_factor)
        @parameter
        fn _normalize[simd_width: Int](col: Int):
            var input_val = input_fn[simd_width](row, col)
            var gamma_val = gamma.load[width=simd_width](col)
            var norm_val = input_val * norm_factor * gamma_val
            output_fn[simd_width](row, col, norm_val)

        vectorize[_normalize, simd_width](num_cols)


fn rms_norm_cpu[
    type: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_fn: fn[width: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
](shape: IndexList[rank], gamma: NDBuffer[type, 1], epsilon: Float32):
    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(chunk_size, prod_all_but_last_dim, last_dim, epsilon)
    @parameter
    fn task_func(thread_id: Int):
        var num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        var row_idx = thread_id * chunk_size

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn output_fn_2d[
            simd_width: Int
        ](row: Int, col: Int, val: SIMD[type, simd_width]) -> None:
            # Translate given 2d index back to the original Nd tensor.
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            output_fn[simd_width](indices, val)

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) -> SIMD[type, simd_width]:
            # Translate given 2d index back to the original Nd tensor.
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width, rank](indices)

        rms_norm_cpu[input_fn_2d, output_fn_2d](
            gamma, epsilon, out_shape=IndexList[2](num_rows, last_dim)
        )

    sync_parallelize[task_func](num_workers)


@always_inline
fn _rms_norm_impl[
    type: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    output_fn: fn[width: Int] (
        IndexList[rank], SIMD[type, width]
    ) capturing -> None,
    /,
    target: StringLiteral = "cpu",
](
    shape: IndexList[rank],
    gamma: NDBuffer[type, 1],
    epsilon: Float32,
    ctx: DeviceContextPtr,
) raises:
    # Note: we only support reduction along the last dimension
    if gamma.dynamic_shape[0] != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.dynamic_shape[0])
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @parameter
    if is_cpu[target]():
        rms_norm_cpu[input_0_fn, output_fn](shape, gamma, epsilon)
    elif is_gpu[target]():
        rms_norm_gpu[input_0_fn, output_fn](
            shape, gamma, epsilon, ctx.get_device_context()
        )
    else:
        constrained[False, "unsupported target " + target]()


@register_internal("rms_norm")
@always_inline
fn rms_norm[
    type: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        type, width
    ],
    /,
    target: StringLiteral = "cpu",
](
    shape: IndexList[rank],
    gamma: NDBuffer[type, 1],
    epsilon: Float32,
    output: NDBuffer[type, rank],
    ctx: DeviceContextPtr,
) raises:
    if output.dynamic_shape.canonicalize() != shape.canonicalize():
        raise Error("Input and output buffers are not same shape")

    alias align = simdwidthof[type]()

    @always_inline
    @__copy_capture(output)
    @parameter
    fn identity_output_fn[
        width: Int
    ](idx: IndexList[rank], val: SIMD[type, width]) -> None:
        output.store(idx, val)

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, type)

    with Trace[TraceLevel.OP](
        "rms_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
    ):
        _rms_norm_impl[
            type, rank, input_0_fn, identity_output_fn, target=target
        ](shape, gamma, epsilon, ctx)


@always_inline
fn rms_norm_shape[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    gamma: NDBuffer[type, 1],
    epsilon: Float32,
) -> IndexList[rank]:
    return input.get_shape()
