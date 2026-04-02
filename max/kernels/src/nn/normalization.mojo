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

from std.math import align_down, ceildiv, clamp, rsqrt
from std.sys.info import align_of, simd_width_of, size_of

import std.gpu.primitives.warp as warp
from std.algorithm import map_reduce, mean, variance, vectorize
from std.algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    sync_parallelize,
)
from std.algorithm.reduction import _simd_sum, _simd_sum_elementwise
from std.bit import log2_floor
from std.gpu import (
    WARP_SIZE,
    barrier,
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    lane_id_uint as lane_id,
    syncwarp,
    thread_idx_uint as thread_idx,
    warp_id_uint as warp_id,
)
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.memory import external_memory
from std.sys.info import has_apple_gpu_accelerator, is_apple_gpu
from std.gpu.primitives import block
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    Coord,
    CoordLike,
    Idx,
    RuntimeInt,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.memory import stack_allocation
from std.runtime.asyncrt import DeviceContextPtr, parallelism_level
from std.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from std.utils.numerics import get_accum_type, max_finite, min_finite
from comm.rms_norm_fp8 import rms_norm_fused_fp8
from std.gpu.primitives.grid_controls import PDLLevel
from .reshape import reshape

comptime _APPLE_STATIC_SHMEM_MAX_BYTES = 32 * 1024
"""Maximum number of bytes that can be used on Apple GPUs (32K)."""

comptime _APPLE_STATIC_SHMEM_MAX_COUNT[
    T: AnyType
] = _APPLE_STATIC_SHMEM_MAX_BYTES // size_of[T]()
"""Maximum number of elements of type T that can fit in Apple's
static shared memory which is 32k."""


@always_inline
def block_reduce[
    dtype: DType, max_warps_per_block: Int
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = stack_allocation[
        max_warps_per_block, dtype, address_space=AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()

    var warp_m2 = warp.sum(val)

    var warp_id = warp_id[broadcast=True]()
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0:
        var block_m2 = Scalar[dtype](0)

        # Only read lanes corresponding to active warps to avoid
        # reading uninitialized shared memory.
        if lane_idx < block_dim.x // UInt(WARP_SIZE):
            block_m2 = m2_shared[lane_idx]

        # On some GPUs, the warp-level reduction implicitly requires all lanes
        # to participate in the reduction. Otherwise, we would get deadlocks.
        block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](block_m2)

        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


@always_inline
def block_reduce_dual_sum[
    dtype: DType, max_warps_per_block: Int
](val0: Scalar[dtype], val1: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """Combined block reduction for two sums using only 2 barriers."""
    var shared0 = stack_allocation[
        max_warps_per_block, dtype, address_space=AddressSpace.SHARED
    ]()
    var shared1 = stack_allocation[
        max_warps_per_block, dtype, address_space=AddressSpace.SHARED
    ]()
    var broadcast0 = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()
    var broadcast1 = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()

    var warp_sum0 = warp.sum(val0)
    var warp_sum1 = warp.sum(val1)

    var warp_id = warp_id()
    var lane_idx = lane_id()

    if lane_idx == 0:
        shared0[warp_id] = warp_sum0
        shared1[warp_id] = warp_sum1
    barrier()

    if warp_id == 0:
        var block_sum0 = Scalar[dtype](0)
        var block_sum1 = Scalar[dtype](0)

        if lane_idx < block_dim.x // UInt(WARP_SIZE):
            block_sum0 = shared0[lane_idx]
            block_sum1 = shared1[lane_idx]

        block_sum0 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum0
        )
        block_sum1 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            block_sum1
        )

        if lane_idx == 0:
            broadcast0[0] = block_sum0
            broadcast1[0] = block_sum1
    barrier()
    return (broadcast0[0], broadcast1[0])


# using numerically stable Welford online algorithm to compute single pass mean and variance
def welford_update[
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


def welford_combine[
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


def welford_warp_reduce[
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

    comptime limit = log2_floor(WARP_SIZE)

    comptime for mask in reversed(range(limit)):
        var mean = warp.shuffle_down(res_mean, UInt32(1 << mask))
        var m2 = warp.shuffle_down(res_m2, UInt32(1 << mask))
        var count = warp.shuffle_down(res_count, UInt32(1 << mask))
        welford_combine(mean, m2, count, res_mean, res_m2, res_count)


def welford_block_all_reduce[
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
        WARP_SIZE, dtype, address_space=AddressSpace.SHARED
    ]()
    var m2_shared = stack_allocation[
        WARP_SIZE, dtype, address_space=AddressSpace.SHARED
    ]()
    var count_shared = stack_allocation[
        WARP_SIZE, dtype, address_space=AddressSpace.SHARED
    ]()
    var mean_broadcast = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()
    var count_broadcast = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
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
        if thread_idx.x < block_dim.x // UInt(WARP_SIZE):
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


def layer_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: UInt,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    comptime align = align_of[SIMD[dtype, Int(simd_width)]]()
    comptime accum_type = get_accum_type[dtype]()

    var num_cols = shape[1]
    var tid: UInt = thread_idx.x
    var row: UInt = block_idx.x

    var vec_data = SIMD[accum_type, Int(simd_width)]()

    var idx: UInt = tid * simd_width

    with PDL():
        var row_mean: Scalar[accum_type]
        var row_var: Scalar[accum_type]

        if idx < UInt(num_cols):
            vec_data = input_fn[Int(simd_width)](Int(row), Int(idx)).cast[
                accum_type
            ]()

        var thread_sum = vec_data.reduce_add()
        var n = Scalar[accum_type](num_cols)

        comptime if accum_type != dtype:
            # Higher-precision accumulation (e.g. bf16→f32): single-pass
            # dual reduction (2 barriers). E[X^2]-E[X]^2 is stable
            # because accum_type has enough headroom.
            var thread_sum_sq = (vec_data**2).reduce_add()
            var reduced = block_reduce_dual_sum[
                max_warps_per_block=max_warps_per_block
            ](thread_sum, thread_sum_sq)
            row_mean = reduced[0] / n
            row_var = max(reduced[1] / n - row_mean * row_mean, 0.0)
        else:
            # Same-precision accumulation (e.g. f32→f32): two-pass
            # centered variance (4 barriers) for numerical stability.
            var total_sum = block_reduce[
                max_warps_per_block=max_warps_per_block
            ](thread_sum)
            row_mean = total_sum / n
            var thread_centered_sq = Scalar[accum_type](0)
            if idx < UInt(num_cols):
                thread_centered_sq = ((vec_data - row_mean) ** 2).reduce_add()
            var total_centered_sq = block_reduce[
                max_warps_per_block=max_warps_per_block
            ](thread_centered_sq)
            row_var = max(total_centered_sq / n, 0.0)

        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx < UInt(num_cols):
            var gamma_val = gamma_fn[Int(simd_width)](Index(idx))
            var beta_val = beta.load[width=Int(simd_width)](Coord(Idx(idx)))
            var norm_val = (vec_data - row_mean) * norm_factor * gamma_val.cast[
                accum_type
            ]() + beta_val.cast[accum_type]()
            output_fn[Int(simd_width), align](
                Int(row), Int(idx), norm_val.cast[dtype]()
            )


def layer_norm_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: UInt,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    comptime align = align_of[SIMD[dtype, Int(simd_width)]]()
    comptime accum_type = get_accum_type[dtype]()

    var num_cols = UInt(shape[1])
    var tid = thread_idx.x
    var row = block_idx.x

    # To store final row mean, mean of squares and the element count
    var row_mean = Scalar[accum_type]()
    var row_m2 = Scalar[accum_type]()
    var row_count = Scalar[accum_type]()
    var thread_mean = Scalar[accum_type]()
    var thread_m2 = Scalar[accum_type]()
    var thread_count = Scalar[accum_type]()

    with PDL():
        # First pass: compute per-tile mean and m2 using SIMD reductions,
        # then combine via Welford for numerical stability.
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var vec_data = input_fn[Int(simd_width)](
                    Int(row), Int(offset)
                ).cast[accum_type]()

                # SIMD-optimized per-tile statistics.
                var tile_sum = vec_data.reduce_add()
                var tile_count = Scalar[accum_type](Int(simd_width))
                var tile_mean = tile_sum / tile_count
                var tile_m2 = ((vec_data - tile_mean) ** 2).reduce_add()
                welford_combine(
                    tile_mean,
                    tile_m2,
                    tile_count,
                    thread_mean,
                    thread_m2,
                    thread_count,
                )

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

        # Second pass: normalize.
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var gamma_val = gamma_fn[Int(simd_width)](Index(offset))
                var beta_offset = beta.layout(Idx(offset))
                var beta_val = beta.ptr.load[
                    width=Int(simd_width), alignment=align
                ](beta_offset)

                var vec_data = input_fn[Int(simd_width)](
                    Int(row), Int(offset)
                ).cast[accum_type]()
                var norm_val = (
                    (vec_data - row_mean)
                    * norm_factor
                    * gamma_val.cast[accum_type]()
                ) + beta_val.cast[accum_type]()
                output_fn[Int(simd_width), align](
                    Int(row), Int(offset), norm_val.cast[dtype]()
                )


def layer_norm_reshape[
    rank: Int, //, output_rank: Int
](shape: IndexList[rank, ...],) -> IndexList[output_rank]:
    comptime if rank == output_rank:
        return rebind[IndexList[output_rank]](shape)

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim
    return IndexList[output_rank](prod_all_but_last_dim, last_dim)


def layer_norm_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank, ...],
    beta: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    *,
    ctx: DeviceContext,
) raises:
    comptime assert beta.rank == 1, "beta must have rank 1"
    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    comptime rank_rs = 2
    var flattened_shape = layer_norm_reshape[rank_rs](shape)
    var rows = flattened_shape[0]
    var cols = flattened_shape[1]

    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]):
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, rank, alignment](indices.canonicalize(), val)

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

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
            comptime kernel = layer_norm_gpu_warp_tiling[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                UInt(simd_width),
                max_warps_per_block,
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel, kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        elif (
            cols <= (WARP_SIZE * (simd_width * 2) * max_warps_per_block)
            and cols % (simd_width * 2) == 0
        ):
            comptime kernel = layer_norm_gpu_warp_tiling[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                UInt(simd_width * 2),
                max_warps_per_block,
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel, kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime kernel = layer_norm_gpu_block[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                UInt(simd_width),
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel, kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
    else:
        comptime kernel = layer_norm_gpu_block[
            mut=beta.mut,
            LayoutType=beta.LayoutType,
            origin=beta.origin,
            1,
            input_fn_2d,
            gamma_fn,
            output_fn_2d,
        ]
        ctx.enqueue_function[kernel, kernel](
            flattened_shape,
            beta,
            epsilon,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel(1)),
        )


@always_inline
def _sum_to_mean[
    dtype: DType, //
](sum_val: Scalar[dtype], n: Int) -> Scalar[dtype]:
    comptime if dtype.is_integral():
        return sum_val // Scalar[dtype](n)
    return sum_val / Scalar[dtype](n)


def layer_norm_cpu[
    dtype: DType,
    //,
    input_fn: def[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    num_rows: Int,
    num_cols: Int,
    beta: TileTensor[dtype, ...],
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
    comptime assert beta.rank == 1, "beta must have rank 1"
    comptime simd_width = simd_width_of[dtype]()

    for var row in range(num_rows):

        @always_inline
        @parameter
        @__copy_capture(row)
        def output_fn_1d[
            dtype_: DType, simd_width: Int, alignment: Int
        ](idx: Int, val: SIMD[dtype_, simd_width]):
            output_fn[simd_width, alignment](
                row, idx, rebind[SIMD[dtype, simd_width]](val)
            )

        @__copy_capture(row)
        @parameter
        def input_gen_wrapper[
            dtype: DType, simd_width: Int
        ](col: Int) -> SIMD[dtype, simd_width]:
            return input_fn[simd_width](row, col).cast[dtype]()

        var sum_val = map_reduce[
            simd_width,
            dtype,
            dtype,
            origin_of()._mlir_origin,
            input_gen_wrapper,
            origin_of()._mlir_origin,
            _simd_sum_elementwise,
            _simd_sum,
            output_fn_1d,
        ](num_cols, 0)

        var mean_val = _sum_to_mean(sum_val, num_cols)
        var var_val = variance[dtype, input_gen_wrapper](
            num_cols, mean_val, 0
        )  # use biased estimator
        var norm_factor = rsqrt(var_val + epsilon)

        def _normalize[simd_width: Int](col: Int) unified {mut}:
            var out_val = input_fn[simd_width](row, col)
            var gamma_val = gamma_fn[simd_width, 1](Index(col))
            var beta_col = beta.layout(Idx(col))

            var norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_val + beta.ptr.load[width=simd_width](
                beta_col
            )
            output_fn[simd_width, 1](
                row, col, rebind[SIMD[dtype, simd_width]](norm_val)
            )

        vectorize[simd_width](num_cols, _normalize)


def layer_norm_cpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank],
    beta: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    var last_dim = shape[rank - 1]

    var prod_all_but_last_dim = 1

    comptime for i in range(rank - 1):
        prod_all_but_last_dim *= shape[i]

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(chunk_size, prod_all_but_last_dim, last_dim, epsilon)
    @parameter
    def task_func(thread_id: Int) raises:
        var row_idx = thread_id * chunk_size
        var chunk_rows = min(chunk_size, prod_all_but_last_dim - row_idx)

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        def input_fn_2d[
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
        def output_fn_2d[
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
def layer_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: def[_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    input_1_fn: def[_width: Int, _rank: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    output_0_fn: def[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[rank],
    gamma_shape: IndexList[1],
    beta: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert beta.rank == 1, "beta must have rank 1"
    # Note: we only support reduction along the last dimension
    if gamma_shape[0] != shape[rank - 1]:
        raise Error("Gamma size does not match dimension of reduction.")

    if beta.layout.shape[0]().value() != shape[rank - 1]:
        raise Error("Beta size does not match dimension of reduction.")

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.get_device_context().id()),
    ):
        comptime if is_cpu[target]():
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
            comptime assert False, "unsupported target " + target


@always_inline
def layer_norm_shape[
    dtype: DType
](
    input: TileTensor[dtype, ...],
    gamma: TileTensor[dtype, ...],
    beta: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
) -> IndexList[input.rank]:
    """
    Compute the output shape of a `layer_norm` operation.

    Parameters:
        dtype: Type of the input tensors.

    Args:
        input: The input tensor.
        gamma: The tensor for gamma coefficient.
        beta: The tensor for beta coefficient.
        epsilon: The tensor for epsilon coefficient.

    Returns:
        The output shape.
    """
    comptime assert gamma.flat_rank == 1 and gamma.static_shape[0] == 1
    comptime assert beta.rank == 1 and beta.static_shape[0] == 1

    return rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )


@always_inline
def _rms_norm_warp_tiling_subkernel[
    dtype: DType,
    simd_width: Int,
    accum_type: DType,
    //,
    max_warps_per_block: Int,
    multiply_before_cast: Bool,
    rows_per_warp: Int = 1,
](
    row: Int,
    idx: Int,
    vec_data: SIMD[accum_type, simd_width],
    gamma_val: SIMD[dtype, simd_width],
    epsilon: Scalar[accum_type],
    weight_offset: Scalar[accum_type],
    num_cols: Int,
) -> SIMD[dtype, simd_width]:
    # To utilize simd vector load.
    var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()

    comptime if rows_per_warp == 2:
        # Each half warp handles reduction for one row.
        row_m2 = warp.lane_group_sum[num_lanes=WARP_SIZE // 2](thread_m2)
    else:
        row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

    var norm_factor = rsqrt((row_m2 / Scalar[accum_type](num_cols)) + epsilon)
    var norm_val: SIMD[dtype, simd_width] = 0
    if idx < num_cols:
        comptime if multiply_before_cast:
            var gamma_accum = gamma_val.cast[accum_type]() + weight_offset
            norm_val = (vec_data * norm_factor * gamma_accum).cast[dtype]()
        else:
            norm_val = (vec_data * norm_factor).cast[dtype]() * (
                gamma_val + weight_offset.cast[dtype]()
            )

    return norm_val


def rms_norm_gpu_warp_tiling_128[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_rows: Int,
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1
    comptime half_warp_size = WARP_SIZE // 2
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

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
        var gamma_val = SIMD[dtype, simd_width](0)
        if row < UInt(num_rows) and idx < UInt(num_cols):
            vec_data = input_fn[simd_width](Int(row), Int(idx)).cast[
                accum_type
            ]()
            # Prefetch gamma before reduction to overlap load with compute.
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(Idx(idx))
            )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block, multiply_before_cast, rows_per_warp=2
        ](
            Int(row),
            Int(idx),
            vec_data,
            gamma_val,
            eps_accum,
            weight_offset_accum,
            num_cols,
        )
        if row < UInt(num_rows) and idx < UInt(num_cols):
            output_fn[simd_width, align](Int(row), Int(idx), norm_val)


def rms_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * UInt(simd_width)

    with PDL():
        var gamma_val = SIMD[dtype, simd_width](0)
        if idx < UInt(num_cols):
            vec_data = input_fn[simd_width](Int(row), Int(idx)).cast[
                accum_type
            ]()
            # Prefetch gamma before reduction to overlap load with compute.
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(Idx(idx))
            )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            Int(row),
            Int(idx),
            vec_data,
            gamma_val,
            eps_accum,
            weight_offset_accum,
            num_cols,
        )
        if idx < UInt(num_cols):
            output_fn[simd_width, align](Int(row), Int(idx), norm_val)


@always_inline
def _rms_norm_gpu_block_subkernel[
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var thread_m2 = Scalar[accum_type](0)
    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    # Every block has a single row to process
    for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
        var offset = x * Int(block_dim.x) * simd_width + Int(
            tid * UInt(simd_width)
        )
        if offset < num_cols:
            var vec_data = input_fn[simd_width](Int(row), offset).cast[
                accum_type
            ]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = rsqrt((row_m2 / Scalar[accum_type](num_cols)) + eps_accum)

    # Need a pass again to perform in place normalization.
    for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
        var offset = x * Int(block_dim.x) * simd_width + Int(
            tid * UInt(simd_width)
        )

        if offset < num_cols:
            var vec_data = input_fn[simd_width](Int(row), offset).cast[
                accum_type
            ]()
            var norm_val: SIMD[dtype, simd_width]
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(Idx(offset))
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

            output_fn[simd_width, align](Int(row), offset, norm_val)


def rms_norm_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    with PDL():
        _rms_norm_gpu_block_subkernel[
            simd_width,
            max_warps_per_block,
            input_fn,
            output_fn,
            multiply_before_cast,
        ](gamma, epsilon, weight_offset, num_cols)


def rms_norm_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel(1),
](
    shape: IndexList[rank, ...],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    if rank == 0:
        return

    # Derive the number of columns from the `gamma` input as this value may be
    # statically known.
    var cols = Int(gamma.dim[0]())

    if cols == 0:
        return

    var rows = shape.flattened_length() // cols

    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

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
            comptime warps_per_block = 2
            # Each warp handles 2 rows, so total rows per block is warps_per_block * 2.
            block_dim = warps_per_block * WARP_SIZE
            grid_dim = ceildiv(rows, warps_per_block * 2)

            comptime kernel = rms_norm_gpu_warp_tiling_128[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                rows,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        elif cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            comptime kernel = rms_norm_gpu_warp_tiling[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        elif (
            cols <= (WARP_SIZE * (simd_width * 2) * max_warps_per_block)
            and cols % (simd_width * 2) == 0
        ):
            comptime kernel = rms_norm_gpu_warp_tiling[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width * 2,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        else:
            comptime kernel = rms_norm_gpu_block[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                output_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
    else:
        comptime kernel = rms_norm_gpu_block[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            1,
            max_warps_per_block,
            input_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel, kernel](
            gamma,
            epsilon,
            weight_offset,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def rms_norm_cpu[
    dtype: DType,
    //,
    input_fn: def[width: Int](Int, Int) capturing -> SIMD[dtype, width],
    output_fn: def[width: Int, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    out_shape: IndexList[2],
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime assert gamma.flat_rank >= 1

    comptime simd_width = simd_width_of[dtype]()

    var num_rows = out_shape[0]
    var num_cols = out_shape[1]

    var simd_loop_end = align_down(num_cols, simd_width)
    comptime intermediate_type = get_accum_type[dtype]()

    # PyTorch converts the input to float32 before computing the RMS norm
    # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
    for var row in range(num_rows):
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

        def _normalize[simd_width: Int](col: Int) unified {mut}:
            var input_val = input_fn[simd_width](row, col).cast[
                intermediate_type
            ]()
            var gamma_val = gamma.load[width=simd_width, alignment=1](
                Coord(Idx(col))
            )
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

        vectorize[simd_width](num_cols, _normalize)


def rms_norm_cpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    var num_workers = min(parallelism_level(), prod_all_but_last_dim)
    var chunk_size = ceildiv(prod_all_but_last_dim, num_workers)

    @__copy_capture(
        chunk_size, prod_all_but_last_dim, last_dim, epsilon, weight_offset
    )
    @parameter
    def task_func(thread_id: Int):
        var num_rows = min(
            chunk_size, prod_all_but_last_dim - thread_id * chunk_size
        )
        var row_idx = thread_id * chunk_size

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        def output_fn_2d[
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
        def input_fn_2d[
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
def _rms_norm_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    # Note: we only support reduction along the last dimension
    if gamma.layout.shape[0]().value() != shape[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    comptime if is_cpu[target]():
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
        comptime assert False, "unsupported target " + target


def rms_norm_fused_residual_add_gpu_warp_tiling[
    mut1: Bool,
    LayoutType1: TensorLayout,
    origin1: Origin[mut=mut1],
    mut2: Bool,
    LayoutType2: TensorLayout,
    origin2: Origin[mut=mut2],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma1: TileTensor[dtype, LayoutType1, origin1],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, LayoutType2, origin2],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma1.flat_rank == 1, "gamma1 must have flat_rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"
    comptime assert gamma2.flat_rank == 1, "gamma2 must have flat_rank 1"
    comptime assert gamma1.flat_rank >= 1
    comptime assert gamma2.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum1 = epsilon1.cast[accum_type]()
    var weight_offset_accum1 = weight_offset1.cast[accum_type]()
    var eps_accum2 = epsilon2.cast[accum_type]()
    var weight_offset_accum2 = weight_offset2.cast[accum_type]()

    var vec_data = SIMD[dtype, simd_width](0)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * UInt(simd_width)

    with PDL():
        var gamma1_val = SIMD[dtype, simd_width](0)
        if idx < UInt(num_cols):
            vec_data = input_fn[simd_width](Int(row), Int(idx))
            # Prefetch gamma1 before reduction to overlap load with compute.
            gamma1_val = gamma1.load[width=simd_width, alignment=align](
                Coord(Idx(idx))
            )

        var norm1_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            Int(row),
            Int(idx),
            vec_data.cast[accum_type](),
            gamma1_val,
            eps_accum1,
            weight_offset_accum1,
            num_cols,
        )

        var gamma2_val = SIMD[dtype, simd_width](0)
        if idx < UInt(num_cols):
            norm1_val += residual_input_fn[simd_width](Int(row), Int(idx))
            output_residual_fn[simd_width, align](Int(row), Int(idx), norm1_val)
            # Prefetch gamma2 before second reduction.
            gamma2_val = gamma2.load[width=simd_width, alignment=align](
                Coord(Idx(idx))
            )

        var norm2_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            Int(row),
            Int(idx),
            norm1_val.cast[accum_type](),
            gamma2_val,
            eps_accum2,
            weight_offset_accum2,
            num_cols,
        )

        if idx < UInt(num_cols):
            output_fn[simd_width, align](Int(row), Int(idx), norm2_val)


def rms_norm_fused_residual_add_gpu_block[
    mut1: Bool,
    LayoutType1: TensorLayout,
    origin1: Origin[mut=mut1],
    mut2: Bool,
    LayoutType2: TensorLayout,
    origin2: Origin[mut=mut2],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma1: TileTensor[dtype, LayoutType1, origin1],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, LayoutType2, origin2],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    num_cols: Int,
):
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    # Fused 3-pass implementation:
    #   Pass 1: Read input from global → accumulate m2 for stage 1
    #   Pass 2: Re-read input → normalize with gamma1 → add residual →
    #           write residual output → accumulate m2 for stage 2 →
    #           write to shmem
    #   Pass 3: Read from shmem → normalize with gamma2 → write final output
    #
    # This saves 1 barrier and 1 shmem read pass vs the prior approach of
    # calling _rms_norm_gpu_block_subkernel twice with an explicit barrier
    # between them (5 barriers + 4 data passes → 4 barriers + 3 data passes).
    # The first barrier inside block_reduce for m2_2 synchronizes the shmem
    # writes from Pass 2, so no extra barrier is needed.

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var shared_mem = stack_allocation[
        _APPLE_STATIC_SHMEM_MAX_COUNT[Scalar[dtype]],
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]() if comptime (is_apple_gpu()) else external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=align_of[SIMD[dtype, simd_width]](),
        name="intermediate_shared_memory",
    ]()

    with PDL():
        var tid = thread_idx.x
        var row = block_idx.x
        var eps_accum1 = epsilon1.cast[accum_type]()
        var weight_offset_accum1 = weight_offset1.cast[accum_type]()
        var eps_accum2 = epsilon2.cast[accum_type]()
        var weight_offset_accum2 = weight_offset2.cast[accum_type]()

        # Pass 1: Accumulate sum-of-squares for stage 1 from global input.
        var thread_m2_1 = Scalar[accum_type](0)
        for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
            var offset = x * Int(block_dim.x) * simd_width + Int(
                tid * UInt(simd_width)
            )
            if offset < num_cols:
                var vec_data = input_fn[simd_width](Int(row), offset).cast[
                    accum_type
                ]()
                thread_m2_1 += (vec_data**2).reduce_add()

        var row_m2_1 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2_1
        )
        var norm_factor1 = rsqrt(
            (row_m2_1 / Scalar[accum_type](num_cols)) + eps_accum1
        )

        # Pass 2: Re-read input, normalize with gamma1, add residual,
        # write residual output, accumulate m2 for stage 2, write to shmem.
        var thread_m2_2 = Scalar[accum_type](0)
        for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
            var offset = x * Int(block_dim.x) * simd_width + Int(
                tid * UInt(simd_width)
            )
            if offset < num_cols:
                var vec_data = input_fn[simd_width](Int(row), offset).cast[
                    accum_type
                ]()
                var gamma1_val = gamma1.load[width=simd_width, alignment=align](
                    Coord(Idx(offset))
                )

                var norm1_val: SIMD[dtype, simd_width]

                if multiply_before_cast:
                    var gamma1_accum = (
                        gamma1_val.cast[accum_type]() + weight_offset_accum1
                    )
                    norm1_val = (vec_data * norm_factor1 * gamma1_accum).cast[
                        dtype
                    ]()
                else:
                    norm1_val = (vec_data * norm_factor1).cast[dtype]() * (
                        gamma1_val + weight_offset1
                    )

                var residual_val = residual_input_fn[simd_width](
                    Int(row), offset
                )
                var residual_add_val = norm1_val + residual_val
                output_residual_fn[simd_width, align](
                    Int(row), offset, residual_add_val
                )

                # Accumulate for stage 2.
                var residual_accum = residual_add_val.cast[accum_type]()
                thread_m2_2 += (residual_accum**2).reduce_add()

                # Store to shmem for stage 2 normalize pass.
                shared_mem.store[width=simd_width, alignment=align](
                    offset, residual_add_val
                )

        # The first barrier inside block_reduce synchronizes shmem writes.
        var row_m2_2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2_2
        )
        var norm_factor2 = rsqrt(
            (row_m2_2 / Scalar[accum_type](num_cols)) + eps_accum2
        )

        # Pass 3: Read from shmem, normalize with gamma2, write final output.
        for x in range(ceildiv(num_cols // simd_width, Int(block_dim.x))):
            var offset = x * Int(block_dim.x) * simd_width + Int(
                tid * UInt(simd_width)
            )
            if offset < num_cols:
                var stage2_input = shared_mem.load[width=simd_width](
                    offset
                ).cast[accum_type]()
                var gamma2_val = gamma2.load[width=simd_width, alignment=align](
                    Coord(Idx(offset))
                )

                var norm2_val: SIMD[dtype, simd_width]

                if multiply_before_cast:
                    var gamma2_accum = (
                        gamma2_val.cast[accum_type]() + weight_offset_accum2
                    )
                    norm2_val = (
                        stage2_input * norm_factor2 * gamma2_accum
                    ).cast[dtype]()
                else:
                    norm2_val = (stage2_input * norm_factor2).cast[dtype]() * (
                        gamma2_val + weight_offset2
                    )

                output_fn[simd_width, align](Int(row), offset, norm2_val)


def rms_norm_fused_residual_add_gpu_block_no_shmem[
    mut1: Bool,
    LayoutType1: TensorLayout,
    origin1: Origin[mut=mut1],
    mut2: Bool,
    LayoutType2: TensorLayout,
    origin2: Origin[mut=mut2],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma1: TileTensor[dtype, LayoutType1, origin1],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, LayoutType2, origin2],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    num_rows: Int,
    num_cols: Int,
):
    """RMS norm fused with residual add, without shared memory reductions.

    Each warp independently processes one row using only warp-level
    reductions (`warp.sum`), avoiding all shared memory usage. Multiple
    rows are processed per block (one row per warp). Intermediate results
    between stages are recomputed instead of being stored in shared memory,
    trading extra global memory reads for zero shared memory usage.

    This is particularly useful on Apple GPUs where shared memory capacity
    is limited.
    """
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma1.flat_rank == 1, "gamma1 must have flat_rank 1"
    comptime assert gamma1.flat_rank >= 1
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"
    comptime assert gamma2.flat_rank == 1, "gamma2 must have flat_rank 1"
    comptime assert gamma2.flat_rank >= 1

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum1 = epsilon1.cast[accum_type]()
    var weight_offset_accum1 = weight_offset1.cast[accum_type]()
    var eps_accum2 = epsilon2.cast[accum_type]()
    var weight_offset_accum2 = weight_offset2.cast[accum_type]()

    var wid = warp_id[broadcast=True]()
    var lid = lane_id()
    var row = block_idx.x * UInt(max_warps_per_block) + wid

    if row >= UInt(num_rows):
        return

    with PDL():
        # ---- Stage 1: First RMS norm ----
        # Pass 1: Accumulate sum-of-squares for stage 1.
        var thread_m2_1 = Scalar[accum_type](0)
        for x in range(ceildiv(num_cols // simd_width, WARP_SIZE)):
            var offset = x * WARP_SIZE * simd_width + Int(
                lid * UInt(simd_width)
            )
            if offset < num_cols:
                var vec_data = input_fn[simd_width](Int(row), offset).cast[
                    accum_type
                ]()
                thread_m2_1 += (vec_data**2).reduce_add()

        var row_m2_1 = warp.sum(thread_m2_1)
        var norm_factor1 = rsqrt(
            (row_m2_1 / Scalar[accum_type](num_cols)) + eps_accum1
        )

        # Pass 2: Normalize with gamma1, add residual, write output_residual,
        # and accumulate sum-of-squares for stage 2.
        var thread_m2_2 = Scalar[accum_type](0)
        for x in range(ceildiv(num_cols // simd_width, WARP_SIZE)):
            var offset = x * WARP_SIZE * simd_width + Int(
                lid * UInt(simd_width)
            )
            if offset < num_cols:
                var vec_data = input_fn[simd_width](Int(row), offset).cast[
                    accum_type
                ]()
                var gamma1_val = gamma1.load[width=simd_width, alignment=align](
                    Coord(Idx(offset))
                )

                var norm1_val: SIMD[dtype, simd_width]

                comptime if multiply_before_cast:
                    var gamma1_accum = (
                        gamma1_val.cast[accum_type]() + weight_offset_accum1
                    )
                    norm1_val = (vec_data * norm_factor1 * gamma1_accum).cast[
                        dtype
                    ]()
                else:
                    norm1_val = (vec_data * norm_factor1).cast[dtype]() * (
                        gamma1_val + weight_offset1
                    )

                var residual_val = residual_input_fn[simd_width](
                    Int(row), offset
                )
                var residual_add_val = norm1_val + residual_val
                output_residual_fn[simd_width, align](
                    Int(row), offset, residual_add_val
                )

                # Accumulate for stage 2.
                var residual_add_accum = residual_add_val.cast[accum_type]()
                thread_m2_2 += (residual_add_accum**2).reduce_add()

        # ---- Stage 2: Second RMS norm ----
        var row_m2_2 = warp.sum(thread_m2_2)
        var norm_factor2 = rsqrt(
            (row_m2_2 / Scalar[accum_type](num_cols)) + eps_accum2
        )

        # Pass 3: Recompute stage 1 output (input norm + residual add),
        # then normalize with gamma2 and write final output.
        for x in range(ceildiv(num_cols // simd_width, WARP_SIZE)):
            var offset = x * WARP_SIZE * simd_width + Int(
                lid * UInt(simd_width)
            )
            if offset < num_cols:
                # Recompute the residual-added value from stage 1.
                var vec_data = input_fn[simd_width](Int(row), offset).cast[
                    accum_type
                ]()
                var gamma1_val = gamma1.load[width=simd_width, alignment=align](
                    Coord(Idx(offset))
                )

                var norm1_val: SIMD[dtype, simd_width]

                comptime if multiply_before_cast:
                    var gamma1_accum = (
                        gamma1_val.cast[accum_type]() + weight_offset_accum1
                    )
                    norm1_val = (vec_data * norm_factor1 * gamma1_accum).cast[
                        dtype
                    ]()
                else:
                    norm1_val = (vec_data * norm_factor1).cast[dtype]() * (
                        gamma1_val + weight_offset1
                    )

                var residual_val = residual_input_fn[simd_width](
                    Int(row), offset
                )
                var stage2_input = (norm1_val + residual_val).cast[accum_type]()

                var gamma2_val = gamma2.load[width=simd_width, alignment=align](
                    Coord(Idx(offset))
                )

                var norm2_val: SIMD[dtype, simd_width]

                comptime if multiply_before_cast:
                    var gamma2_accum = (
                        gamma2_val.cast[accum_type]() + weight_offset_accum2
                    )
                    norm2_val = (
                        stage2_input * norm_factor2 * gamma2_accum
                    ).cast[dtype]()
                else:
                    norm2_val = (stage2_input * norm_factor2).cast[dtype]() * (
                        gamma2_val + weight_offset2
                    )

                output_fn[simd_width, align](Int(row), offset, norm2_val)


def rms_norm_fused_residual_add_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_residual_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, ...],
    gamma1: TileTensor[dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    if rank == 0:
        return

    var last_dim = shape[rank - 1]

    if last_dim == 0:
        return

    var rows = shape.flattened_length() // last_dim
    var cols = last_dim

    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    def output_residual_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_residual_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width](indices.canonicalize())

    @parameter
    @always_inline
    def residual_input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return residual_input_fn[simd_width](indices.canonicalize())

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

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
            comptime kernel = rms_norm_fused_residual_add_gpu_warp_tiling[
                mut1=gamma1.mut,
                LayoutType1=gamma1.LayoutType,
                origin1=gamma1.origin,
                mut2=gamma2.mut,
                LayoutType2=gamma2.LayoutType,
                origin2=gamma2.origin,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                residual_input_fn_2d,
                output_fn_2d,
                output_residual_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            comptime if has_apple_gpu_accelerator():
                # On Apple GPUs, use the no-shmem variant to avoid shared
                # memory limitations. Each warp handles one row
                # independently using only warp-level reductions.
                comptime no_shmem_kernel = rms_norm_fused_residual_add_gpu_block_no_shmem[
                    mut1=gamma1.mut,
                    LayoutType1=gamma1.LayoutType,
                    origin1=gamma1.origin,
                    mut2=gamma2.mut,
                    LayoutType2=gamma2.LayoutType,
                    origin2=gamma2.origin,
                    simd_width,
                    max_warps_per_block,
                    input_fn_2d,
                    residual_input_fn_2d,
                    output_fn_2d,
                    output_residual_fn_2d,
                    multiply_before_cast=multiply_before_cast,
                ]
                ctx.enqueue_function[no_shmem_kernel, no_shmem_kernel](
                    gamma1,
                    epsilon1,
                    weight_offset1,
                    gamma2,
                    epsilon2,
                    weight_offset2,
                    rows,
                    cols,
                    grid_dim=ceildiv(rows, max_warps_per_block),
                    block_dim=WARP_SIZE * max_warps_per_block,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
            else:
                var shared_mem_size = (
                    ceildiv(cols, simd_width) * simd_width * size_of[dtype]()
                )

                comptime kernel = rms_norm_fused_residual_add_gpu_block[
                    mut1=gamma1.mut,
                    LayoutType1=gamma1.LayoutType,
                    origin1=gamma1.origin,
                    mut2=gamma2.mut,
                    LayoutType2=gamma2.LayoutType,
                    origin2=gamma2.origin,
                    simd_width,
                    max_warps_per_block,
                    input_fn_2d,
                    residual_input_fn_2d,
                    output_fn_2d,
                    output_residual_fn_2d,
                    multiply_before_cast=multiply_before_cast,
                ]
                ctx.enqueue_function[kernel, kernel](
                    gamma1,
                    epsilon1,
                    weight_offset1,
                    gamma2,
                    epsilon2,
                    weight_offset2,
                    cols,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                    shared_mem_bytes=shared_mem_size,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(shared_mem_size)
                    ),
                )

    else:
        comptime if has_apple_gpu_accelerator():
            # On Apple GPUs, use the no-shmem variant with simd_width=1
            # for non-aligned column counts.
            comptime no_shmem_kernel = rms_norm_fused_residual_add_gpu_block_no_shmem[
                mut1=gamma1.mut,
                LayoutType1=gamma1.LayoutType,
                origin1=gamma1.origin,
                mut2=gamma2.mut,
                LayoutType2=gamma2.LayoutType,
                origin2=gamma2.origin,
                1,
                max_warps_per_block,
                input_fn_2d,
                residual_input_fn_2d,
                output_fn_2d,
                output_residual_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]

            ctx.enqueue_function[no_shmem_kernel, no_shmem_kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                rows,
                cols,
                grid_dim=ceildiv(rows, max_warps_per_block),
                block_dim=WARP_SIZE * max_warps_per_block,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            var shared_mem_size = cols * size_of[dtype]()

            comptime kernel = rms_norm_fused_residual_add_gpu_block[
                mut1=gamma1.mut,
                LayoutType1=gamma1.LayoutType,
                origin1=gamma1.origin,
                mut2=gamma2.mut,
                LayoutType2=gamma2.LayoutType,
                origin2=gamma2.origin,
                1,
                max_warps_per_block,
                input_fn_2d,
                residual_input_fn_2d,
                output_fn_2d,
                output_residual_fn_2d,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel, kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
                shared_mem_bytes=shared_mem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(
                        ctx.default_device_info.shared_memory_per_multiprocessor
                        - 4096
                    )
                ),
            )


def rms_norm_fused_residual_add_cpu[
    dtype: DType,
    rank: Int,
    //,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    residual_input_fn: def[width: Int, rank: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_0_fn: def[width: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    var intermediate_buffer_ptr = alloc[Scalar[dtype]](shape.flattened_length())
    var intermediate_buffer = TileTensor(
        intermediate_buffer_ptr,
        row_major(Coord(shape)),
    )

    @parameter
    @always_inline
    @__copy_capture(intermediate_buffer)
    def intermediate_output_fn[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var residual_val = residual_input_fn[width](idx)

        var residual_add_val = val + residual_val
        output_residual_fn[width, alignment](idx, residual_add_val)
        var intermediate_idx = intermediate_buffer.layout(Coord(idx))
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
    def intermediate_input_fn[
        width: Int, rank_: Int
    ](idx: IndexList[rank_]) -> SIMD[dtype, width]:
        var intermediate_idx = intermediate_buffer.layout(Coord(idx))
        return intermediate_buffer.ptr.load[width=width](intermediate_idx)

    rms_norm_cpu[
        intermediate_input_fn,
        output_0_fn,
        multiply_before_cast=multiply_before_cast,
    ](shape, gamma2, epsilon2, weight_offset2)

    intermediate_buffer_ptr.free()


@always_inline
def rms_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: def[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma: TileTensor[dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    @always_inline
    @parameter
    def output_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    def description_fn() -> String:
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


def _rms_norm_fused_residual_add_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    # Note: we only support reduction along the last dimension
    if gamma1.layout.shape[0]().value() != shape[rank - 1]:
        raise Error(
            "Gamma1 size "
            + String(gamma1.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if gamma2.layout.shape[0]().value() != shape[rank - 1]:
        raise Error(
            "Gamma2 size "
            + String(gamma2.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if shape.flattened_length() == 0:
        # Nothing to do.
        return

    comptime if is_gpu[target]():
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


@always_inline
def rms_norm_fused_residual_add[
    dtype: DType,
    rank: Int,
    //,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_0_fn: def[width: Int, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContextPtr,
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    @always_inline
    @parameter
    def output_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    def output_residual_fn_wrapper[
        width: Int, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_residual_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    def description_fn() -> String:
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


def group_norm_reshape[
    dtype: DType,
    rank: Int,
](
    shape: IndexList[rank, ...],
    buf: TileTensor[dtype, ...],
    channels_per_group: Int,
    spatial: Int,
    out result: TileTensor[
        dtype,
        Layout[
            shape_types=DynamicCoord[DType.int64, 2].element_types,
            stride_types=DynamicCoord[DType.int64, 2].element_types,
        ],
        buf.origin,
        address_space=buf.address_space,
    ],
):
    """
    Reshapes an input buffer for group normalization by flattening all
    dimensions except the group dimension. Returns a 2D buffer of shape
    (num_groups * N, group_size), where group_size is the product of
    channels_per_group and spatial.
    """
    comptime assert buf.rank == rank, "buf.rank must equal rank"
    var group_size = channels_per_group * spatial
    var prod_all_but_group_dim = shape.flattened_length() // group_size
    var new_shape = IndexList[2](prod_all_but_group_dim, group_size)
    var reshaped = reshape[2](buf, new_shape)
    result = {
        reshaped.ptr,
        reshaped.layout,
    }


def group_norm_gpu_warp_tiling[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
    num_groups: Int,
    channels_per_group: Int,
    spatial: Int,
):
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var idx = Int(thread_idx.x) * simd_width

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
        if idx + simd_width <= group_size:
            vec_data = input_fn[simd_width](Int(row), idx).cast[accum_type]()

            comptime for i in range(simd_width):
                welford_update(
                    vec_data[i], thread_mean, thread_m2, thread_count
                )

        welford_block_all_reduce(
            thread_mean, thread_m2, thread_count, row_mean, row_m2, row_count
        )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx + simd_width <= group_size:
            var g = row % UInt(num_groups)
            var c_base = g * UInt(channels_per_group)
            var norm_val = SIMD[accum_type, simd_width]()
            for i in range(simd_width):
                var offset = (idx + i) // spatial
                var c = c_base + UInt(offset)
                var gamma_val = gamma_fn[1](Index(c))
                var beta_val = beta_fn[1](Index(c))
                norm_val[i] = (
                    vec_data[i] - row_mean
                ) * norm_factor * gamma_val.cast[accum_type]() + beta_val.cast[
                    accum_type
                ]()

            var output_idx = output.layout(Coord(Idx(row), Idx(idx)))
            output.ptr.store[alignment=align](
                output_idx, norm_val.cast[dtype]()
            )


def group_norm_gpu_block[
    LayoutType: TensorLayout,
    origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: UInt,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
    num_groups: Int,
    channels_per_group: Int,
    spatial: Int,
):
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, Int(simd_width)]]()
    comptime accum_type = get_accum_type[dtype]()

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

        for x in range(
            ceildiv(group_size // Int(simd_width), Int(block_dim.x))
        ):
            var offset = x * Int(block_dim.x) * Int(simd_width) + Int(
                tid * simd_width
            )
            if offset < group_size:
                var vec_data = input_fn[Int(simd_width)](Int(row), offset).cast[
                    accum_type
                ]()

                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[Int(i)], thread_mean, thread_m2, thread_count
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

        for x in range(
            ceildiv(group_size // Int(simd_width), Int(block_dim.x))
        ):
            var offset = x * Int(block_dim.x) * Int(simd_width) + Int(
                tid * simd_width
            )
            if offset < group_size:
                var vec_data = input_fn[Int(simd_width)](Int(row), offset).cast[
                    accum_type
                ]()

                var g = row % UInt(num_groups)
                var c_base = g * UInt(channels_per_group)

                var norm_val = SIMD[accum_type, Int(simd_width)]()
                for i in range(simd_width):
                    var offset_c = (offset + Int(i)) // spatial
                    var c = c_base + UInt(offset_c)
                    var gamma_val = gamma_fn[1](Index(c))
                    var beta_val = beta_fn[1](Index(c))
                    norm_val[Int(i)] = (
                        vec_data[Int(i)] - row_mean
                    ) * norm_factor * gamma_val.cast[
                        accum_type
                    ]() + beta_val.cast[
                        accum_type
                    ]()

                var output_row_offset = output.layout(
                    Coord(Idx(row), Idx(offset))
                )
                output.ptr.store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


def group_norm_gpu_multi_block_stats[
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: UInt,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
](
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    num_splits: Int,
    group_size: Int,
):
    """Multi-block stats kernel: computes partial Welford statistics per split.

    Grid: num_rows * num_splits blocks. Each block handles one split of one
    group and writes partial (mean, m2, count) to the stats buffer.
    Stats layout: stats[block_idx * 3 + {0,1,2}] = {mean, m2, count}.
    """
    comptime accum_type = get_accum_type[dtype]()

    var block_id = Int(block_idx.x)
    var row, split_id = divmod(block_id, num_splits)
    var tid = thread_idx.x

    # Compute chunk boundaries (each split handles a contiguous chunk,
    # aligned to simd_width).
    var total_simd_elems = group_size // Int(simd_width)
    var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
    var chunk_start = split_id * chunk_simd_size * Int(simd_width)
    var chunk_end = min(
        chunk_start + chunk_simd_size * Int(simd_width), group_size
    )
    var chunk_iters = ceildiv(chunk_simd_size, Int(block_dim.x))

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(chunk_iters):
            var offset = (
                chunk_start
                + x * Int(block_dim.x) * Int(simd_width)
                + Int(tid * simd_width)
            )
            if offset < chunk_end:
                var vec_data = input_fn[Int(simd_width)](row, offset).cast[
                    accum_type
                ]()
                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[Int(i)],
                        thread_mean,
                        thread_m2,
                        thread_count,
                    )

        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        welford_block_all_reduce(
            thread_mean,
            thread_m2,
            thread_count,
            row_mean,
            row_m2,
            row_count,
        )

        # Thread 0 writes partial stats to the global stats buffer.
        if tid == 0:
            var base_idx = block_id * 3
            stats.store(Coord(Idx(base_idx)), row_mean)
            stats.store(Coord(Idx(base_idx + 1)), row_m2)
            stats.store(Coord(Idx(base_idx + 2)), row_count)


def group_norm_gpu_multi_block_norm[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: UInt,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
](
    output: TileTensor[dtype, OutputLayoutType, output_origin],
    stats: TileTensor[get_accum_type[dtype](), StatsLayoutType, stats_origin],
    epsilon: Scalar[dtype],
    num_groups: Int,
    channels_per_group: Int,
    spatial: Int,
    num_splits: Int,
    group_size: Int,
):
    """Multi-block normalize kernel: reduces partial stats and normalizes.

    Grid: num_rows * num_splits blocks. Each block reads all partial stats
    for its group, reduces to final mean/variance, then normalizes its
    chunk of elements.
    """
    comptime assert output.rank == 2, "output.rank must be 2"
    comptime align = align_of[SIMD[dtype, Int(simd_width)]]()
    comptime accum_type = get_accum_type[dtype]()

    var block_id = Int(block_idx.x)
    var row, split_id = divmod(block_id, num_splits)
    var tid = thread_idx.x

    # Same chunk boundaries as stats kernel.
    var total_simd_elems = group_size // Int(simd_width)
    var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
    var chunk_start = split_id * chunk_simd_size * Int(simd_width)
    var chunk_end = min(
        chunk_start + chunk_simd_size * Int(simd_width), group_size
    )
    var chunk_iters = ceildiv(chunk_simd_size, Int(block_dim.x))

    with PDL():
        # Reduce all partial stats for this group (num_splits is small,
        # typically 4-16, so this loop is cheap).
        var row_mean = Scalar[accum_type]()
        var row_m2 = Scalar[accum_type]()
        var row_count = Scalar[accum_type]()
        var stats_row_base = row * num_splits * 3
        for s in range(num_splits):
            var base_idx = stats_row_base + s * 3
            welford_combine(
                stats.load[width=1](Coord(Idx(base_idx))),
                stats.load[width=1](Coord(Idx(base_idx + 1))),
                stats.load[width=1](Coord(Idx(base_idx + 2))),
                row_mean,
                row_m2,
                row_count,
            )

        var row_var = row_m2 / row_count
        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        var g = row % num_groups
        var c_base = g * channels_per_group

        for x in range(chunk_iters):
            var offset = (
                chunk_start
                + x * Int(block_dim.x) * Int(simd_width)
                + Int(tid * simd_width)
            )
            if offset < chunk_end:
                var vec_data = input_fn[Int(simd_width)](row, offset).cast[
                    accum_type
                ]()

                var norm_val = SIMD[accum_type, Int(simd_width)]()

                # Vectorized gamma/beta: when all SIMD elements share the
                # same channel (common case for large spatial dims), load
                # gamma/beta once and broadcast.
                var c_first = c_base + offset // spatial
                var c_last = c_base + (offset + Int(simd_width) - 1) // spatial
                if c_first == c_last:
                    var gamma_val = gamma_fn[1](Index(c_first)).cast[
                        accum_type
                    ]()
                    var beta_val = beta_fn[1](Index(c_first)).cast[accum_type]()
                    norm_val = (
                        vec_data - row_mean
                    ) * norm_factor * gamma_val + beta_val
                else:
                    for i in range(simd_width):
                        var c = c_base + (offset + Int(i)) // spatial
                        var gamma_val = gamma_fn[1](Index(c))
                        var beta_val = beta_fn[1](Index(c))
                        norm_val[Int(i)] = (
                            vec_data[Int(i)] - row_mean
                        ) * norm_factor * gamma_val.cast[
                            accum_type
                        ]() + beta_val.cast[
                            accum_type
                        ]()

                var output_row_offset = output.layout(
                    Coord(Idx(row), Idx(offset))
                )
                output.ptr.store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


def group_norm_gpu[
    dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
](
    shape: IndexList[rank, ...],
    epsilon: Scalar[dtype],
    output: TileTensor[mut=True, dtype, ...],
    num_groups: Int,
    ctx: DeviceContext,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime accum_type = get_accum_type[dtype]()

    var N = shape[0]
    var C = shape[1]

    var spatial = shape.flattened_length() // (N * C)
    var channels_per_group = C // num_groups

    var output_rs = group_norm_reshape[dtype, rank](
        shape, output, channels_per_group, spatial
    )

    comptime OutputLinearIdxType = Scalar[output_rs.linear_idx_type]

    var num_rows = output_rs.dim[0]()
    var num_cols = output_rs.dim[1]()

    @parameter
    @always_inline
    @__copy_capture(shape, num_groups, channels_per_group)
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) capturing -> SIMD[dtype, simd_width]:
        var n, g = divmod(row, num_groups)
        var c = g * channels_per_group

        var indices = IndexList[rank]()  # placeholder to satisfy compiler

        comptime if rank == 4:
            var inner_volume = shape[2] * shape[3]
            var c_offset, hw = divmod(col, inner_volume)
            c += c_offset
            var h, w = divmod(hw, shape[3])
            indices = IndexList[rank](n, c, h, w)

        elif rank == 3:
            var inner_volume = shape[2]
            var c_offset, l = divmod(col, inner_volume)
            c += c_offset
            indices = IndexList[rank](n, c, l)

        return input_fn[simd_width, rank](indices)

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    if num_cols < OutputLinearIdxType(simd_width):
        raise Error(
            "group_norm_gpu requires num_cols >= simd_width; got num_cols="
            + String(num_cols)
            + " and simd_width="
            + String(simd_width)
        )

    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = num_rows
    var block_dim = min(
        ceildiv(
            ceildiv(num_cols, OutputLinearIdxType(simd_width)),
            OutputLinearIdxType(WARP_SIZE),
        )
        * OutputLinearIdxType(WARP_SIZE),
        OutputLinearIdxType(WARP_SIZE * max_warps_per_block),
    )

    if num_cols % OutputLinearIdxType(simd_width) == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        if num_cols <= OutputLinearIdxType(
            WARP_SIZE * simd_width * max_warps_per_block
        ):
            comptime kernel = group_norm_gpu_warp_tiling[
                LayoutType=output_rs.LayoutType,
                origin=output_rs.origin,
                dtype=dtype,
                simd_width=simd_width,
                input_fn=input_fn_2d,
                gamma_fn=gamma_fn,
                beta_fn=beta_fn,
            ]
            ctx.enqueue_function[kernel, kernel](
                output_rs,
                epsilon,
                num_groups,
                channels_per_group,
                spatial,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel(1)),
            )
        else:
            # Use multi-block reduction when the grid is too small for
            # good GPU occupancy.  Each group is split across num_splits
            # blocks so that more SMs are active.
            comptime desired_min_grid = 256
            var num_splits = 1
            if Int(num_rows) < desired_min_grid:
                num_splits = min(ceildiv(desired_min_grid, Int(num_rows)), 32)
                # Ensure each split has enough work (≥ 1 SIMD iter per
                # thread at block_dim threads).
                var group_size = Int(num_cols)
                var max_useful_splits = max(
                    1,
                    group_size
                    // (Int(simd_width) * WARP_SIZE * max_warps_per_block),
                )
                num_splits = min(num_splits, max_useful_splits)

            if num_splits > 1:
                var group_size = Int(num_cols)

                # Allocate a small buffer for partial Welford statistics:
                # 3 values (mean, m2, count) per (row, split).
                var stats_size = Int(num_rows) * num_splits * 3
                var stats_buf = ctx.enqueue_create_buffer[accum_type](
                    stats_size
                )
                var stats = TileTensor(
                    stats_buf.unsafe_ptr(),
                    row_major(Idx(stats_size)),
                )

                # Compute block_dim based on per-split chunk size.
                # Cap at 256 threads: both kernels capture closures
                # (input_fn_2d with its coordinate computation chain,
                # gamma_fn, beta_fn) that cause high register pressure,
                # especially for bfloat16 (simd_width=8).  256 threads
                # keeps total register usage within GPU limits while
                # each thread processes more elements per iteration.
                comptime mb_max_block_dim = min(
                    256, WARP_SIZE * max_warps_per_block
                )
                var total_simd_elems = group_size // simd_width
                var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
                var mb_block_dim = min(
                    ceildiv(chunk_simd_size, WARP_SIZE) * WARP_SIZE,
                    mb_max_block_dim,
                )
                var mb_grid_dim = Int(num_rows) * num_splits

                # Kernel 1: compute partial Welford stats per split.
                comptime stats_kernel = group_norm_gpu_multi_block_stats[
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=UInt(simd_width),
                    input_fn=input_fn_2d,
                ]
                ctx.enqueue_function[stats_kernel, stats_kernel](
                    stats,
                    num_splits,
                    group_size,
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )

                # Kernel 2: reduce stats and normalize each chunk.
                comptime norm_kernel = group_norm_gpu_multi_block_norm[
                    OutputLayoutType=output_rs.LayoutType,
                    output_origin=output_rs.origin,
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=UInt(simd_width),
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[norm_kernel, norm_kernel](
                    output_rs,
                    stats,
                    epsilon,
                    num_groups,
                    channels_per_group,
                    spatial,
                    num_splits,
                    group_size,
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )

                _ = stats_buf^
            else:
                comptime kernel = group_norm_gpu_block[
                    LayoutType=output_rs.LayoutType,
                    origin=output_rs.origin,
                    dtype=dtype,
                    simd_width=UInt(simd_width),
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[kernel, kernel](
                    output_rs,
                    epsilon,
                    num_groups,
                    channels_per_group,
                    spatial,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(PDLLevel(1)),
                )
    else:
        comptime kernel = group_norm_gpu_block[
            LayoutType=output_rs.LayoutType,
            origin=output_rs.origin,
            dtype=dtype,
            simd_width=1,
            input_fn=input_fn_2d,
            gamma_fn=gamma_fn,
            beta_fn=beta_fn,
        ]
        ctx.enqueue_function[kernel, kernel](
            output_rs,
            epsilon,
            num_groups,
            channels_per_group,
            spatial,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel(1)),
        )


@always_inline
def group_norm[
    dtype: DType,
    rank: Int,
    input_fn: def[width: Int, _rank: Int](IndexList[_rank]) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    beta_fn: def[width: Int](IndexList[1]) capturing -> SIMD[dtype, width],
    /,
    target: StaticString = "gpu",
](
    shape: IndexList[rank],
    epsilon: Scalar[dtype],
    groups: Int32,
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContextPtr,
) raises:
    comptime assert output.rank == rank, "output.rank must be the same as rank"
    comptime assert (
        rank > 2 and rank < 5
    ), "group_norm requires input rank of 3 or 4"
    comptime assert is_gpu[
        target
    ](), "group_norm only supports GPU targets at this point"

    if shape.canonicalize() != rebind[IndexList[rank]](
        coord_to_index_list(output.layout.shape_coord())
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
    def description_fn() -> String:
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
