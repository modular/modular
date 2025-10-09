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

from math import align_down, ceildiv, rsqrt
from sys.info import align_of, simd_width_of, size_of

import gpu.warp as warp
from algorithm import map_reduce, mean, variance, vectorize
from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    sync_parallelize,
)
from algorithm.reduction import _simd_sum, _simd_sum_elementwise
from bit import log2_floor
from gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    syncwarp,
    thread_idx,
    warp_id,
)
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from gpu.host.info import is_cpu, is_gpu
from gpu.memory import AddressSpace, external_memory
from layout import (
    UNKNOWN_VALUE,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout.int_tuple import fill_like
from memory import stack_allocation
from register import register_internal
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, trace_arg

from utils.index import Index, IndexList
from utils.numerics import get_accum_type

from .reshape import reshape


@always_inline
fn block_reduce[
    dtype: DType, max_warps_per_block: Int
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = stack_allocation[
        max_warps_per_block, dtype, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, dtype, address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    for i in range(tid, max_warps_per_block, block_dim.x):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp.sum(val)

    var warp_id = warp.broadcast(tid // UInt(WARP_SIZE))
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0 and lane_idx < UInt(max_warps_per_block):
        var block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            m2_shared[lane_idx]
        )
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


# using numerically stable Welford online algorithm to compute single pass mean and variance
fn welford_update[
    dtype: DType, //
](
    val: Scalar[dtype],
    mut mean: Scalar[dtype],
    mut m2: Scalar[dtype],
    mut count: Scalar[dtype],
):
    count += 1
    var d1 = val - mean
    mean += d1 / count
    var d2 = val - mean
    m2 += d1 * d2


fn welford_combine[
    dtype: DType, //
](
    mean: Scalar[dtype],
    m2: Scalar[dtype],
    count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
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
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
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
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    welford_warp_reduce(
        thread_mean, thread_m2, thread_count, res_mean, res_m2, res_count
    )
    # broadcasting res from warp lane_id 0 to all in a warp
    res_mean = warp.broadcast(res_mean)
    res_m2 = warp.broadcast(res_m2)
    res_count = warp.broadcast(res_count)


fn welford_block_all_reduce[
    dtype: DType, //
](
    thread_mean: Scalar[dtype],
    thread_m2: Scalar[dtype],
    thread_count: Scalar[dtype],
    mut res_mean: Scalar[dtype],
    mut res_m2: Scalar[dtype],
    mut res_count: Scalar[dtype],
):
    var mean_shared = stack_allocation[
        WARP_SIZE, dtype, address_space = AddressSpace.SHARED
    ]()
    var m2_shared = stack_allocation[
        WARP_SIZE, dtype, address_space = AddressSpace.SHARED
    ]()
    var count_shared = stack_allocation[
        WARP_SIZE, dtype, address_space = AddressSpace.SHARED
    ]()
    var mean_broadcast = stack_allocation[
        1, dtype, address_space = AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, dtype, address_space = AddressSpace.SHARED
    ]()
    var count_broadcast = stack_allocation[
        1, dtype, address_space = AddressSpace.SHARED
    ]()

    var warp_idx = warp_id()
    var lane_idx = lane_id()
    var warp_mean = Scalar[dtype]()
    var warp_m2 = Scalar[dtype]()
    var warp_count = Scalar[dtype]()
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
        if thread_idx.x < UInt(block_dim.x // UInt(WARP_SIZE)):
            warp_mean = mean_shared[lane_idx]
            warp_m2 = m2_shared[lane_idx]
            warp_count = count_shared[lane_idx]
        else:
            warp_mean = Scalar[dtype](0)
            warp_m2 = Scalar[dtype](0)
            warp_count = Scalar[dtype](0)
        syncwarp()
        var block_mean = Scalar[dtype](0)
        var block_m2 = Scalar[dtype](0)
        var block_count = Scalar[dtype](0)
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
    mut: Bool,
    origin: Origin[mut],
    layout: Layout,
    dtype: DType, //,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
):
    constrained[beta.rank == 1, "beta must have rank 1"]()
    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var num_cols = shape[1]
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

    with PDL():
        if idx < UInt(num_cols):
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

            # every thread computes its own simd width of mean and variance
            @parameter
            for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        # a whole block computes part of the row main and variance and broadcasts to
        # thread_idx 0 to update the final row mean and variance
        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

        var row_var = max(row_m2 / row_count, 0.0)
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx < UInt(num_cols):
            var gamma_val = gamma_fn[simd_width](Index(idx))
            var beta_idx = beta.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
            )
            var beta_val = beta.ptr.load[width=simd_width, alignment=align](
                beta_idx
            )
            var norm_val = (vec_data - row_mean) * norm_factor * gamma_val.cast[
                accum_type
            ]() + beta_val.cast[accum_type]()
            output_fn[simd_width, align](row, idx, norm_val.cast[dtype]())


fn layer_norm_gpu_block[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout,
    dtype: DType, //,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
):
    constrained[beta.rank == 1, "beta must have rank 1"]()
    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var num_cols: UInt = UInt(shape[1])
    var tid = thread_idx.x
    var row = block_idx.x

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    with PDL():
        # Every block has a single row to process
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var thread_mean = Scalar[accum_type]()
            var thread_m2 = Scalar[accum_type]()
            var thread_count = Scalar[accum_type]()

            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                @parameter
                for i in range(simd_width):
                    welford_update(
                        vec_data[i], thread_mean, thread_m2, thread_count
                    )

            # a whole block computes part of the row main and variance and broadcasts to
            # thread_idx 0 to update the final row mean and variance
            welford_block_all_reduce(
                thread_mean,
                thread_m2,
                thread_count,
                row_mean,
                row_m2,
                row_count,
            )

        var row_var = max(row_m2 / row_count, 0)
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        # need a pass again to perform in place normalization
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var gamma_val = gamma_fn[simd_width](Index(offset))
                var beta_offset = beta.runtime_layout(
                    RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset)
                )
                var beta_val = beta.ptr.load[width=simd_width, alignment=align](
                    beta_offset
                )

                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                var norm_val = (
                    (vec_data - row_mean)
                    * norm_factor
                    * gamma_val.cast[accum_type]()
                ) + beta_val.cast[accum_type]()
                output_fn[simd_width, align](
                    row, offset, norm_val.cast[dtype]()
                )


fn layer_norm_reshape[
    rank: Int, //, output_rank: Int
](shape: IndexList[rank, **_],) -> IndexList[output_rank]:
    @parameter
    if rank == output_rank:
        return rebind[IndexList[output_rank]](shape)

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    return IndexList[output_rank](prod_all_but_last_dim, last_dim)


fn layer_norm_gpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank, **_],
    beta: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    *,
    ctx: DeviceContext,
) raises:
    constrained[beta.rank == 1, "beta must have rank 1"]()
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    alias rank_rs = 2
    var flattened_shape = layer_norm_reshape[rank_rs](shape)
    var rows = flattened_shape[0]
    var cols = flattened_shape[1]

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @parameter
    @always_inline
    fn output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]):
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, rank, alignment](indices.canonicalize(), val)

    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    if cols % simd_width == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        if cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            alias kernel = layer_norm_gpu_warp_tiling[
                mut = beta.mut,
                origin = beta.origin,
                layout = beta.layout,
                UInt(simd_width),
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
        else:
            alias kernel = layer_norm_gpu_block[
                mut = beta.mut,
                origin = beta.origin,
                layout = beta.layout,
                UInt(simd_width),
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
    else:
        alias kernel = layer_norm_gpu_block[
            mut = beta.mut,
            origin = beta.origin,
            layout = beta.layout,
            1,
            input_fn_2d,
            gamma_fn,
            output_fn_2d,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            flattened_shape,
            beta,
            epsilon,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(),
        )


@always_inline
fn _sum_to_mean[
    dtype: DType, //
](sum_val: Scalar[dtype], n: Int) -> Scalar[dtype]:
    @parameter
    if dtype.is_integral():
        return sum_val // n
    return sum_val / n


fn layer_norm_cpu[
    dtype: DType, //,
    input_fn: fn[width: Int] (Int, Int) capturing -> SIMD[dtype, width],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    num_rows: Int,
    num_cols: Int,
    beta: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
) raises:
    """Computes layernorm(elementwise_fn(x)) across the last dimension of x, where layernorm is
    defined as $(x-mean(x))/(sqrt(var(x)+eps)*gamma_fn + beta$.

    Currently performs 3 passes over the input data. This can be reduced to 2 by
    fusing the add, mean, and variance loops using Welford's algorithm.

    Parameters:
        dtype: The x and out buffers' elements dtype.
        input_fn: Function called to generate an input value.
        gamma_fn: Function called to generate a gamma value.
        output_fn: Function called to store the output value.

    Args:
        num_rows: The number of rows in the input tensor.
        num_cols: The number of columns in the input tensor.
        beta: The beta value to use in the layernorm calculation.
        epsilon: The eps value to use in the layernorm calculation.
    """
    constrained[beta.rank == 1, "beta must have rank 1"]()
    alias simd_width = simd_width_of[dtype]()

    for row in range(num_rows):

        @always_inline
        @parameter
        @__copy_capture(row)
        fn output_fn_1d[
            dtype_: DType, simd_width: Int, alignment: Int
        ](idx: Int, val: SIMD[dtype_, simd_width]):
            output_fn[simd_width, alignment](
                row, idx, rebind[SIMD[dtype, simd_width]](val)
            )

        @__copy_capture(row)
        @parameter
        fn input_gen_wrapper[
            dtype: DType, simd_width: Int
        ](col: Int) -> SIMD[dtype, simd_width]:
            return input_fn[simd_width](row, col).cast[dtype]()

        var sum_val = map_reduce[
            simd_width,
            dtype,
            dtype,
            __origin_of(),
            input_gen_wrapper,
            __origin_of(),
            _simd_sum_elementwise,
            _simd_sum,
            output_fn_1d,
        ](num_cols, 0)

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var var_val = variance[dtype, input_gen_wrapper](
            num_cols, mean_val, 0
        )  # use biased estimator
        var norm_factor = rsqrt(var_val + epsilon)

        @__copy_capture(norm_factor, mean_val, row)
        @parameter
        fn _normalize[simd_width: Int](col: Int):
            var out_val = input_fn[simd_width](row, col)
            var gamma_val = gamma_fn[simd_width, 1](col)
            var beta_col = beta.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](col)
            )

            var norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_val + beta.ptr.load[width=simd_width](
                beta_col
            )
            output_fn[simd_width, 1](
                row, col, rebind[SIMD[dtype, simd_width]](norm_val)
            )

        vectorize[_normalize, simd_width](num_cols)


fn layer_norm_cpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, rank: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank],
    beta: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
):
    constrained[beta.rank == 1, "beta must have rank 1"]()
    var last_dim = shape[rank - 1]

    var prod_all_but_last_dim = 1

    @parameter
    for i in range(rank - 1):
        prod_all_but_last_dim *= shape[i]

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(chunk_size, prod_all_but_last_dim, last_dim, epsilon)
    @parameter
    fn task_func(thread_id: Int) raises:
        var row_idx = thread_id * chunk_size
        var chunk_rows = min(chunk_size, prod_all_but_last_dim - row_idx)

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
            # Translate a given 2D index back to the original n-D tensor
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width](indices.canonicalize())

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn output_fn_2d[
            simd_width: Int, alignment: Int
        ](row: Int, col: Int, val: SIMD[dtype, simd_width]):
            # Translate a given 2D index back to the original n-D tensor
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            output_fn[simd_width, rank, alignment](indices.canonicalize(), val)

        layer_norm_cpu[input_fn_2d, gamma_fn, output_fn_2d](
            chunk_rows, shape[rank - 1], beta, epsilon
        )

    sync_parallelize[task_func](num_workers)


@always_inline
fn layer_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: fn[_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    input_1_fn: fn[_width: Int, _rank: Int] (
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    output_0_fn: fn[width: Int, rank: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[rank],
    gamma_shape: IndexList[1],
    beta: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    constrained[beta.rank == 1, "beta must have rank 1"]()
    # Note: we only support reduction along the last dimension
    if gamma_shape[0] != shape[rank - 1]:
        raise Error("Gamma size does not match dimension of reduction.")

    if beta.runtime_layout.shape.value[0] != shape[rank - 1]:
        raise Error("Beta size does not match dimension of reduction.")

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):

        @parameter
        if is_cpu[target]():
            layer_norm_cpu[input_0_fn, input_1_fn, output_0_fn](
                shape.canonicalize(),
                beta,
                epsilon,
            )
        elif is_gpu[target]():
            layer_norm_gpu[input_0_fn, input_1_fn, output_0_fn](
                shape.canonicalize(),
                beta,
                epsilon,
                ctx=ctx.get_device_context(),
            )
        else:
            constrained[False, "unsupported target " + target]()


@always_inline
fn layer_norm_shape[
    dtype: DType,
    single_thread_blocking_override: Bool,
](
    input: LayoutTensor[dtype, **_],
    gamma: LayoutTensor[dtype, Layout.row_major(1)],
    beta: LayoutTensor[dtype, Layout.row_major(1)],
    epsilon: Scalar[dtype],
) -> IndexList[input.rank]:
    """
    Compute the output shape of a `layer_norm` operation.

    Parameters:
        dtype: Type of the input tensors.
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
    return rebind[IndexList[input.rank]](
        input.runtime_layout.shape.value.canonicalize()
    )


@always_inline
fn _rms_norm_warp_tiling_subkernel[
    dtype: DType,
    simd_width: Int,
    accum_type: DType, //,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    rows_per_warp: Int = 1,
](
    row: Int,
    idx: Int,
    vec_data: SIMD[accum_type, simd_width],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[accum_type],
    weight_offset: Scalar[accum_type],
    num_cols: Int,
) -> SIMD[dtype, simd_width]:
    constrained[gamma.rank == 1, "gamma must have rank 1"]()
    alias align = align_of[SIMD[dtype, simd_width]]()

    # To utilize simd vector load.
    var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()

    @parameter
    if rows_per_warp == 2:
        # Each half warp handles reduction for one row.
        row_m2 = warp.lane_group_sum_and_broadcast[num_lanes = WARP_SIZE // 2](
            thread_m2
        )
    else:
        row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

    var norm_factor = rsqrt((row_m2 / num_cols) + epsilon)
    var norm_val: SIMD[dtype, simd_width] = 0
    if idx < num_cols:
        var gamma_idx = gamma.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
        )
        var gamma_val = gamma.ptr.load[width=simd_width, alignment=align](
            gamma_idx
        )

        @parameter
        if multiply_before_cast:
            var gamma_accum = gamma_val.cast[accum_type]() + weight_offset
            norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
        else:
            norm_val = (vec_data * norm_factor).cast[dtype]() * (
                gamma_val + weight_offset.cast[dtype]()
            )

    return norm_val


fn rms_norm_gpu_warp_tiling_128[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout,
    dtype: DType, //,
    simd_width: Int,
    warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
    num_cols: Int,
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()
    alias half_warp_size = WARP_SIZE // 2
    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    # Each warp handles 2 rows, so total rows per block is warps_per_block * 2
    var block_row = block_idx.x * UInt(warps_per_block * 2)
    var warp_id = tid // UInt(WARP_SIZE)
    var sub_warp_id = (tid % UInt(WARP_SIZE)) // UInt(half_warp_size)
    # Each warp handles 2 rows, offset by the block's base row
    var row = block_row + (warp_id * 2) + sub_warp_id
    var local_tid = tid % UInt(half_warp_size)
    var idx = local_tid * UInt(simd_width)

    with PDL():
        if row < UInt(num_rows) and idx < UInt(num_cols):
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block, multiply_before_cast, rows_per_warp=2
        ](
            row,
            idx,
            vec_data,
            gamma,
            eps_accum,
            weight_offset_accum,
            num_cols,
        )
        if row < UInt(num_rows) and idx < UInt(num_cols):
            output_fn[simd_width, align](row, idx, norm_val)


fn rms_norm_gpu_warp_tiling[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout,
    dtype: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * UInt(simd_width)

    with PDL():
        if idx < UInt(num_cols):
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

        var norm_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            row,
            idx,
            vec_data,
            gamma,
            eps_accum,
            weight_offset_accum,
            num_cols,
        )
        if idx < UInt(num_cols):
            output_fn[simd_width, align](row, idx, norm_val)


@always_inline
fn _rms_norm_gpu_block_subkernel[
    dtype: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var thread_m2 = Scalar[accum_type](0)
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * UInt(simd_width)
        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = rsqrt((row_m2 / num_cols) + eps_accum)

    # Need a pass again to perform in place normalization.
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * UInt(simd_width)

        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val: SIMD[dtype, simd_width]
            var gamma_offset = gamma.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset)
            )

            var gamma_val = gamma.ptr.load[width=simd_width, alignment=align](
                gamma_offset
            )

            if multiply_before_cast:
                var gamma_accum = (
                    gamma_val.cast[accum_type]() + weight_offset_accum
                )
                norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
            else:
                norm_val = (vec_data * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset
                )

            output_fn[simd_width, align](row, offset, norm_val)


fn rms_norm_gpu_block[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout,
    dtype: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    with PDL():
        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            input_fn,
            output_fn,
            multiply_before_cast,
        ](gamma, epsilon, weight_offset, num_cols)


fn rms_norm_gpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, **_],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    constrained[gamma.rank == 1, "gamma must have rank 1"]()
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
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    if cols % simd_width == 0:
        # When the number of columns are less enough that they can be placed in
        # registers we do warp tiling which is a single pass to do mean/var
        # computation and normalization.
        if cols <= 128 and dtype == DType.bfloat16:
            # Experimentally determined to be the best - tapers off at 2.
            alias warps_per_block = 2
            # Each warp handles 2 rows, so total rows per block is warps_per_block * 2.
            block_dim = warps_per_block * WARP_SIZE
            grid_dim = ceildiv(rows, warps_per_block * 2)

            alias kernel = rms_norm_gpu_warp_tiling_128[
                mut = gamma.mut,
                origin = gamma.origin,
                layout = gamma.layout,
                simd_width,
                warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                rows,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
        elif cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            alias kernel = rms_norm_gpu_warp_tiling[
                mut = gamma.mut,
                origin = gamma.origin,
                layout = gamma.layout,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
        else:
            alias kernel = rms_norm_gpu_block[
                mut = gamma.mut,
                origin = gamma.origin,
                layout = gamma.layout,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
    else:
        alias kernel = rms_norm_gpu_block[
            mut = gamma.mut,
            origin = gamma.origin,
            layout = gamma.layout,
            1,
            max_warps_per_block,
            input_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            gamma,
            epsilon,
            weight_offset,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(),
        )


fn rms_norm_cpu[
    dtype: DType, //,
    input_fn: fn[width: Int] (Int, Int) capturing -> SIMD[dtype, width],
    output_fn: fn[width: Int, alignment: Int] (
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    out_shape: IndexList[2],
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    alias simd_width = simd_width_of[dtype]()

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    var simd_loop_end = align_down(num_cols, simd_width)
    alias intermediate_type = get_accum_type[dtype]()

    # PyTorch converts the input to float32 before computing the RMS norm
    # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
    for row in range(num_rows):
        var sum_simd = SIMD[intermediate_type, simd_width]()
        for col in range(0, simd_loop_end, simd_width):
            sum_simd += (
                input_fn[simd_width](row, col).cast[intermediate_type]() ** 2
            )

        var sum_val = sum_simd.reduce_add()
        for col in range(simd_loop_end, num_cols):
            sum_val += input_fn[1](row, col).cast[intermediate_type]() ** 2

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var norm_factor = rsqrt(mean_val + epsilon.cast[intermediate_type]())

        @__copy_capture(norm_factor, weight_offset)
        @parameter
        fn _normalize[simd_width: Int](col: Int):
            var input_val = input_fn[simd_width](row, col).cast[
                intermediate_type
            ]()
            var gamma_col = gamma.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](col)
            )

            var gamma_val = gamma.ptr.load[width=simd_width](gamma_col)
            var norm_val: SIMD[dtype, simd_width]

            if multiply_before_cast:
                var gamma_offset = gamma_val + weight_offset
                norm_val = (input_val * norm_factor).cast[
                    dtype
                ]() * gamma_offset
            else:
                norm_val = (input_val * norm_factor).cast[dtype]() * (
                    gamma_val + weight_offset
                )

            output_fn[simd_width, 1](row, col, norm_val)

        vectorize[_normalize, simd_width](num_cols)


fn rms_norm_cpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
):
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(
        chunk_size, prod_all_but_last_dim, last_dim, epsilon, weight_offset
    )
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
            simd_width: Int, alignment: Int
        ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            output_fn[simd_width, alignment](indices, val)

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        fn input_fn_2d[
            simd_width: Int
        ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
            # Translate a given 2D index back to the original n-D tensor.
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width, rank](indices)

        rms_norm_cpu[
            input_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ](
            gamma,
            epsilon,
            weight_offset,
            out_shape=IndexList[2](num_rows, last_dim),
        )

    sync_parallelize[task_func](num_workers)


@always_inline
fn _rms_norm_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    # Note: we only support reduction along the last dimension
    if gamma.runtime_layout.shape.value[0] != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.runtime_layout.shape.value[0])
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @parameter
    if is_cpu[target]():
        rms_norm_cpu[
            input_0_fn, output_fn, multiply_before_cast=multiply_before_cast
        ](shape, gamma, epsilon, weight_offset)
    elif is_gpu[target]():
        rms_norm_gpu[
            input_0_fn, output_fn, multiply_before_cast=multiply_before_cast
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx.get_device_context(),
        )
    else:
        constrained[False, "unsupported target " + target]()


fn rms_norm_fused_residual_add_gpu_warp_tiling[
    mut1: Bool,
    origin1: Origin[mut1],
    layout1: Layout,
    mut2: Bool,
    origin2: Origin[mut2],
    layout2: Layout,
    dtype: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma1: LayoutTensor[dtype, layout1, origin1],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, layout2, origin2],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    num_cols: Int,
):
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var eps_accum1 = epsilon1.cast[accum_type]()
    var weight_offset_accum1 = weight_offset1.cast[accum_type]()
    var eps_accum2 = epsilon2.cast[accum_type]()
    var weight_offset_accum2 = weight_offset2.cast[accum_type]()

    var vec_data = SIMD[dtype, simd_width](0)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * UInt(simd_width)

    with PDL():
        if idx < UInt(num_cols):
            vec_data = input_fn[simd_width](row, idx)

        var norm1_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            row,
            idx,
            vec_data.cast[accum_type](),
            gamma1,
            eps_accum1,
            weight_offset_accum1,
            num_cols,
        )

        if idx < UInt(num_cols):
            norm1_val += residual_input_fn[simd_width](row, idx)
            output_residual_fn[simd_width, align](row, idx, norm1_val)

        var norm2_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            row,
            idx,
            norm1_val.cast[accum_type](),
            gamma2,
            eps_accum2,
            weight_offset_accum2,
            num_cols,
        )

        if idx < UInt(num_cols):
            output_fn[simd_width, align](row, idx, norm2_val)


fn rms_norm_fused_residual_add_gpu_block[
    mut1: Bool,
    origin1: Origin[mut1],
    layout1: Layout,
    mut2: Bool,
    origin2: Origin[mut2],
    layout2: Layout,
    dtype: DType, //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int] (
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma1: LayoutTensor[dtype, layout1, origin1],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, layout2, origin2],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    num_cols: Int,
):
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

    var shared_mem = external_memory[
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
        alignment = align_of[SIMD[dtype, simd_width]](),
        name="intermediate_shared_memory",
    ]()
    with PDL():

        @parameter
        @always_inline
        @__copy_capture(shared_mem)
        fn stage1_output_fn[
            width: Int, alignment: Int
        ](row: Int, col: Int, val: SIMD[dtype, width]):
            residual_val = residual_input_fn[width](row, col)
            var residual_add_val = residual_val + val
            output_residual_fn[width, alignment](row, col, residual_add_val)

            shared_mem.store[width=width, alignment=alignment](
                col, residual_add_val
            )

        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            input_fn,
            stage1_output_fn,
            multiply_before_cast=multiply_before_cast,
        ](gamma1, epsilon1, weight_offset1, num_cols)

        barrier()

        @parameter
        @always_inline
        @__copy_capture(shared_mem)
        fn stage2_input_fn[
            width: Int
        ](row: Int, col: Int) -> SIMD[dtype, width]:
            return shared_mem.load[width=width](col)

        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            stage2_input_fn,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](gamma2, epsilon2, weight_offset2, num_cols)


fn rms_norm_fused_residual_add_gpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_residual_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, **_],
    gamma1: LayoutTensor[dtype, **_],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, **_],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

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
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    fn output_residual_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @parameter
    @always_inline
    fn residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width](indices.canonicalize())

    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

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
            alias kernel = rms_norm_fused_residual_add_gpu_warp_tiling[
                mut1 = gamma1.mut,
                origin1 = gamma1.origin,
                layout1 = gamma1.layout,
                mut2 = gamma2.mut,
                origin2 = gamma2.origin,
                layout2 = gamma2.layout,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                residual_input_fn_2d,
                output_fn_2d,
                output_residual_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
        else:
            var shared_mem_size = (
                ceildiv(cols, simd_width) * simd_width * size_of[dtype]()
            )

            alias kernel = rms_norm_fused_residual_add_gpu_block[
                mut1 = gamma1.mut,
                origin1 = gamma1.origin,
                layout1 = gamma1.layout,
                mut2 = gamma2.mut,
                origin2 = gamma2.origin,
                layout2 = gamma2.layout,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                residual_input_fn_2d,
                output_fn_2d,
                output_residual_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
                shared_mem_bytes=shared_mem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    shared_mem_size
                ),
            )

    else:
        var shared_mem_size = Int(cols * size_of[dtype]())

        alias kernel = rms_norm_fused_residual_add_gpu_block[
            mut1 = gamma1.mut,
            origin1 = gamma1.origin,
            layout1 = gamma1.layout,
            mut2 = gamma2.mut,
            origin2 = gamma2.origin,
            layout2 = gamma2.layout,
            1,
            max_warps_per_block,
            input_fn_2d,
            residual_input_fn_2d,
            output_fn_2d,
            output_residual_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            gamma1,
            epsilon1,
            weight_offset1,
            gamma2,
            epsilon2,
            weight_offset2,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(),
            shared_mem_bytes=shared_mem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                ctx.default_device_info.shared_memory_per_multiprocessor - 4096
            ),
        )


fn rms_norm_fused_residual_add_cpu[
    dtype: DType,
    rank: Int, //,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: fn[width: Int, rank: Int] (
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: fn[width: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: LayoutTensor[dtype, **_],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, **_],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
) raises:
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

    var intermediate_buffer_ptr = UnsafePointer[Scalar[dtype]].alloc(
        shape.flattened_length()
    )
    var intermediate_buffer = LayoutTensor[dtype, Layout.row_major[rank]()](
        intermediate_buffer_ptr,
        RuntimeLayout[Layout.row_major[rank]()].row_major(shape),
    )

    @parameter
    @always_inline
    @__copy_capture(intermediate_buffer)
    fn intermediate_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var residual_val = residual_input_fn[width](idx)

        var residual_add_val = val + residual_val
        output_residual_fn[width, alignment](idx, residual_add_val)
        var intermediate_idx = intermediate_buffer.runtime_layout(
            RuntimeTuple[
                fill_like(intermediate_buffer.layout.shape, UNKNOWN_VALUE)
            ](idx)
        )
        intermediate_buffer.ptr.store[width=width, alignment=alignment](
            intermediate_idx, residual_add_val
        )

    rms_norm_cpu[
        input_0_fn,
        intermediate_output_fn,
        multiply_before_cast=multiply_before_cast,
    ](shape, gamma1, epsilon1, weight_offset1)

    @parameter
    @always_inline
    @__copy_capture(intermediate_buffer)
    fn intermediate_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        var intermediate_idx = intermediate_buffer.runtime_layout(
            RuntimeTuple[
                fill_like(intermediate_buffer.layout.shape, UNKNOWN_VALUE)
            ](idx)
        )
        return intermediate_buffer.ptr.load[width=width](intermediate_idx)

    rms_norm_cpu[
        intermediate_input_fn,
        output_0_fn,
        multiply_before_cast=multiply_before_cast,
    ](shape, gamma2, epsilon2, weight_offset2)

    intermediate_buffer_ptr.free()


@register_internal("rms_norm")
@always_inline
fn rms_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: fn[width: Int, rank: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        _rms_norm_impl[
            dtype,
            rank,
            input_0_fn,
            output_fn_wrapper,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](shape, gamma, epsilon, weight_offset, ctx)


fn _rms_norm_fused_residual_add_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: LayoutTensor[dtype, **_],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, **_],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

    # Note: we only support reduction along the last dimension
    if gamma1.runtime_layout.shape.value[0] != shape[rank - 1]:
        raise Error(
            "Gamma1 size "
            + String(gamma1.runtime_layout.shape.value[0])
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if gamma2.runtime_layout.shape.value[0] != shape[rank - 1]:
        raise Error(
            "Gamma2 size "
            + String(gamma2.runtime_layout.shape.value[0])
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    @parameter
    if is_gpu[target]():
        rms_norm_fused_residual_add_gpu[
            input_0_fn,
            input_1_fn,
            output_residual_fn,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma1,
            epsilon1,
            weight_offset1,
            gamma2,
            epsilon2,
            weight_offset2,
            ctx.get_device_context(),
        )
    else:
        rms_norm_fused_residual_add_cpu[
            input_0_fn,
            input_1_fn,
            output_residual_fn,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma1,
            epsilon1,
            weight_offset1,
            gamma2,
            epsilon2,
            weight_offset2,
        )


@register_internal("rms_norm_fused_residual_add")
@always_inline
fn rms_norm_fused_residual_add[
    dtype: DType,
    rank: Int, //,
    input_0_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: fn[width: Int, rank: Int, alignment: Int] (
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: fn[width: Int, rank: Int, alignment: Int] (
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: LayoutTensor[dtype, **_],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: LayoutTensor[dtype, **_],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    constrained[gamma1.rank == 1, "gamma1 must have rank 1"]()
    constrained[gamma2.rank == 1, "gamma2 must have rank 1"]()

    @always_inline
    @parameter
    fn output_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    fn output_residual_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_residual_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_residual_add",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        _rms_norm_fused_residual_add_impl[
            dtype,
            rank,
            input_0_fn,
            input_1_fn,
            output_fn_wrapper,
            output_residual_fn_wrapper,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma1,
            epsilon1,
            weight_offset1,
            gamma2,
            epsilon2,
            weight_offset2,
            ctx,
        )


@always_inline
fn rms_norm_shape[
    dtype: DType,
    single_thread_blocking_override: Bool,
](
    input: LayoutTensor[dtype, **_],
    gamma: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
) -> IndexList[input.rank]:
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    return rebind[IndexList[input.rank]](
        input.runtime_layout.shape.value.canonicalize()
    )


fn group_norm_reshape[
    dtype: DType,
    rank: Int,
](
    shape: IndexList[rank, **_],
    buf: LayoutTensor[dtype, **_],
    channels_per_group: Int,
    spatial: Int,
    out result: LayoutTensor[
        dtype,
        Layout.row_major[2](),
        buf.origin,
        address_space = buf.address_space,
    ],
):
    """
    Reshapes an input buffer for group normalization by flattening all
    dimensions except the group dimension. Returns a 2D buffer of shape
    (num_groups * N, group_size), where group_size is the product of
    channels_per_group and spatial.
    """
    constrained[buf.rank == rank, "buf.rank must equal rank"]()
    var group_size = channels_per_group * spatial
    var prod_all_but_group_dim = shape.flattened_length() // group_size
    var new_shape = IndexList[2](prod_all_but_group_dim, group_size)
    var reshaped = reshape[2](buf, new_shape)
    return reshaped


fn group_norm_gpu_warp_tiling[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout, //,
    dtype: DType,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
](
    output: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    num_groups: Int,
    channels_per_group: Int,
    spatial: Int,
):
    constrained[output.rank == 2, "output.rank must be 2"]()
    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var idx = tid * simd_width

    var vec_data = SIMD[accum_type, simd_width]()
    var group_size = channels_per_group * spatial

    var row = block_idx.x
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    var thread_mean = Scalar[accum_type]()
    var thread_m2 = Scalar[accum_type]()
    var thread_count = Scalar[accum_type]()

    with PDL():
        if idx + simd_width <= UInt(group_size):
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

            @parameter
            for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx + simd_width <= UInt(group_size):
            var g = row % UInt(num_groups)
            var c_base = g * UInt(channels_per_group)
            var norm_val = SIMD[accum_type, simd_width]()
            for i in range(simd_width):
                var offset = (idx + i) // UInt(spatial)
                var c = c_base + offset
                var gamma_val = gamma_fn[1](Index(c))
                var beta_val = beta_fn[1](Index(c))
                norm_val[i] = (
                    vec_data[i] - row_mean
                ) * norm_factor * gamma_val.cast[accum_type]() + beta_val.cast[
                    accum_type
                ]()

            var output_idx = output.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](
                    Index(row, idx)
                )
            )
            output.ptr.store[alignment=align](
                output_idx, norm_val.cast[dtype]()
            )


fn group_norm_gpu_block[
    mut: Bool,
    origin: Origin[mut],
    layout: Layout, //,
    dtype: DType,
    simd_width: UInt,
    input_fn: fn[width: Int] (row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
](
    output: LayoutTensor[dtype, layout, origin],
    epsilon: Scalar[dtype],
    num_groups: Int,
    channels_per_group: Int,
    spatial: Int,
):
    constrained[output.rank == 2, "output.rank must be 2"]()
    alias align = align_of[SIMD[dtype, simd_width]]()
    alias accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var group_size = channels_per_group * spatial

    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                @parameter
                for i in range(simd_width):
                    welford_update(
                        vec_data[i], thread_mean, thread_m2, thread_count
                    )

        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        for x in range(ceildiv(group_size // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < group_size:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var g = row % UInt(num_groups)
                var c_base = g * UInt(channels_per_group)

                var norm_val = SIMD[accum_type, simd_width]()
                for i in range(simd_width):
                    var offset_c = (offset + i) // spatial
                    var c = c_base + UInt(offset_c)
                    var gamma_val = gamma_fn[1](Index(c))
                    var beta_val = beta_fn[1](Index(c))
                    norm_val[i] = (
                        vec_data[i] - row_mean
                    ) * norm_factor * gamma_val.cast[
                        accum_type
                    ]() + beta_val.cast[
                        accum_type
                    ]()

                var output_row_offset = output.runtime_layout(
                    RuntimeTuple[IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)](
                        Index(row, offset)
                    )
                )
                output.ptr.store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


fn group_norm_gpu[
    dtype: DType,
    rank: Int, //,
    input_fn: fn[width: Int, rank: Int] (IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
](
    shape: IndexList[rank, **_],
    epsilon: Scalar[dtype],
    output: LayoutTensor[mut=True, dtype, **_],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    constrained[output.rank == rank, "output.rank must be the same as rank"]()
    alias accum_type = get_accum_type[dtype]()

    var N = shape[0]
    var C = shape[1]

    var spatial = shape.flattened_length() // (N * C)
    var channels_per_group = C // num_groups

    var output_rs = group_norm_reshape[dtype, rank](
        shape, output, channels_per_group, spatial
    )

    var num_rows = output_rs.dim[0]()
    var num_cols = output_rs.dim[1]()

    @parameter
    @always_inline
    @__copy_capture(shape, num_groups, channels_per_group)
    fn input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) capturing -> SIMD[dtype, simd_width]:
        var n = row // num_groups
        var g = row % num_groups
        var c = g * channels_per_group

        var indices = IndexList[rank]()  # placeholder to satisfy compiler

        @parameter
        if rank == 4:
            var inner_volume = shape[2] * shape[3]
            c += col // inner_volume
            var hw = col % inner_volume
            var h = hw // shape[3]
            var w = hw % shape[3]
            indices = IndexList[rank](n, c, h, w)

        elif rank == 3:
            var inner_volume = shape[2]
            c += col // inner_volume
            var l = col % inner_volume
            indices = IndexList[rank](n, c, l)

        return input_fn[simd_width, rank](indices)

    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    if num_cols < simd_width:
        raise Error(
            "group_norm_gpu requires num_cols >= simd_width; got num_cols="
            + String(num_cols)
            + " and simd_width="
            + String(simd_width)
        )

    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = num_rows
    var block_dim = min(
        ceildiv(ceildiv(num_cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    if num_cols % simd_width == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        if num_cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            alias kernel = group_norm_gpu_warp_tiling[
                mut = output_rs.mut,
                origin = output_rs.origin,
                layout = output_rs.layout,
                dtype=dtype,
                simd_width = UInt(simd_width),
                input_fn=input_fn_2d,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                output_rs,
                epsilon,
                num_groups,
                channels_per_group,
                spatial,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
        else:
            alias kernel = group_norm_gpu_block[
                mut = output_rs.mut,
                origin = output_rs.origin,
                layout = output_rs.layout,
                dtype=dtype,
                simd_width = UInt(simd_width),
                input_fn=input_fn_2d,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ]
            ctx.enqueue_function_checked[kernel, kernel](
                output_rs,
                epsilon,
                num_groups,
                channels_per_group,
                spatial,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(),
            )
    else:
        alias kernel = group_norm_gpu_block[
            mut = output_rs.mut,
            origin = output_rs.origin,
            layout = output_rs.layout,
            dtype=dtype,
            simd_width=1,
            input_fn=input_fn_2d,
            gamma_fn=gamma_fn,
            beta_fn=beta_fn,
        ]
        ctx.enqueue_function_checked[kernel, kernel](
            output_rs,
            epsilon,
            num_groups,
            channels_per_group,
            spatial,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(),
        )


@always_inline
fn group_norm[
    dtype: DType,
    rank: Int,
    input_fn: fn[width: Int, _rank: Int] (IndexList[_rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: fn[width: Int] (IndexList[1]) capturing -> SIMD[dtype, width],
    /,
    target: StaticString = "gpu",
](
    shape: IndexList[rank],
    epsilon: Scalar[dtype],
    groups: Int32,
    output: LayoutTensor[mut=True, dtype, **_],
    ctx: DeviceContextPtr,
) raises:
    constrained[output.rank == rank, "output.rank must be the same as rank"]()
    constrained[
        rank > 2 and rank < 5, "group_norm requires input rank of 3 or 4"
    ]()
    constrained[
        is_gpu[target](), "group_norm only supports GPU targets at this point"
    ]()

    if shape.canonicalize() != rebind[IndexList[rank]](
        output.runtime_layout.shape.value.canonicalize()
    ):
        raise Error(
            "Input/output shape mismatch: input = {shape}, output ="
            " {output.dynamic_shape}"
        )

    var num_groups: Int = Int(groups[0])

    var C = shape[1]
    if C % num_groups != 0:
        raise Error(
            "Invalid num_groups: channels (C = {C}) must be divisible by"
            " num_groups = {num_groups}"
        )

    @always_inline
    @parameter
    fn description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "group_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        group_norm_gpu[
            dtype=dtype,
            rank=rank,
            input_fn=input_fn,
            gamma_fn=gamma_fn,
            beta_fn=beta_fn,
        ](
            shape,
            epsilon,
            output,
            num_groups,
            ctx=ctx.get_device_context(),
        )


@always_inline
fn group_norm_shape[
    dtype: DType,
    single_thread_blocking_override: Bool,
](
    input: LayoutTensor[dtype, **_],
    gamma: LayoutTensor[dtype, **_],
    beta: LayoutTensor[dtype, **_],
    epsilon: Scalar[dtype],
    num_groups: Int32,
) -> IndexList[input.rank]:
    constrained[beta.rank == 1, "beta must have rank 1"]()
    constrained[gamma.rank == 1, "gamma must have rank 1"]()

    return rebind[IndexList[input.rank]](
        input.runtime_layout.shape.value.canonicalize()
    )
