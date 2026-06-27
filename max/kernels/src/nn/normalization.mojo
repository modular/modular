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
from std.math.uutils import umod, ufloordiv, uceildiv
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
    syncwarp,
    barrier,
    thread_idx,
    block_dim,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from std.gpu.host.info import is_cpu, is_gpu
from std.gpu.memory import external_memory
from std.sys.info import is_apple_gpu
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
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.coord import DynamicCoord
from layout.tile_layout import Layout
from std.memory import stack_allocation
from std.runtime.asyncrt import parallelism_level
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
        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
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

        if lane_idx < ufloordiv(block_dim.x, WARP_SIZE):
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
        if thread_idx.x < ufloordiv(block_dim.x, WARP_SIZE):
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


@__name(t"layer_norm_gpu_warp_tiling_{dtype}")
def layer_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var num_cols = shape[1]
    var tid = thread_idx.x
    var row = block_idx.x

    var vec_data = SIMD[accum_type, simd_width]()

    var idx: Int = tid * simd_width

    with PDL():
        var row_mean: Scalar[accum_type]
        var row_var: Scalar[accum_type]

        if idx < num_cols:
            vec_data = input_fn[simd_width, align](row, idx).cast[accum_type]()

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
            if idx < num_cols:
                thread_centered_sq = ((vec_data - row_mean) ** 2).reduce_add()
            var total_centered_sq = block_reduce[
                max_warps_per_block=max_warps_per_block
            ](thread_centered_sq)
            row_var = max(total_centered_sq / n, 0.0)

        var norm_factor = rsqrt(row_var + epsilon.cast[accum_type]())

        if idx < num_cols:
            var gamma_val = gamma_fn[simd_width, 1, align](Index(idx))
            var beta_val = beta.load[width=simd_width](Coord(idx))
            var norm_val = (vec_data - row_mean) * norm_factor * gamma_val.cast[
                accum_type
            ]() + beta_val.cast[accum_type]()
            output_fn[simd_width, align](row, idx, norm_val.cast[dtype]())


@__name(t"layer_norm_gpu_block_{dtype}")
def layer_norm_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    input_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[2],
    beta: TileTensor[dtype, LayoutType, origin],
    epsilon: Scalar[dtype],
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var num_cols = shape[1]
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
        for x in range(ceildiv(ufloordiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var vec_data = input_fn[simd_width, align](row, offset).cast[
                    accum_type
                ]()

                # SIMD-optimized per-tile statistics.
                var tile_sum = vec_data.reduce_add()
                var tile_count = Scalar[accum_type](simd_width)
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
        for x in range(ceildiv(ufloordiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width

            if offset < num_cols:
                var gamma_val = gamma_fn[simd_width, 1, align](Index(offset))
                var beta_offset = beta.layout(offset)
                var beta_val = beta.raw_load[width=simd_width, alignment=align](
                    beta_offset
                )

                var vec_data = input_fn[simd_width, align](row, offset).cast[
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
    input_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank, ...],
    beta: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    *,
    ctx: DeviceContext,
) raises:
    comptime assert beta.rank == 1, "beta must have rank 1"
    if rank == 0:
        return

    comptime rank_rs = 2
    var flattened_shape = layer_norm_reshape[rank_rs](shape)
    var rows = flattened_shape[0]
    var cols = flattened_shape[1]

    if rows == 0 or cols == 0:
        return

    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank, alignment](indices.canonicalize())

    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: SIMDSize, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]):
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, rank, alignment](indices.canonicalize(), val)

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows

    @parameter
    @always_inline
    def warp_tiling_block_dim(sw: Int) -> Int:
        return min(
            ceildiv(ceildiv(cols, sw), WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )

    if cols % simd_width == 0:
        # When the number of columns is small enough that they can be placed in
        # registers, we do warp tiling, which is a single pass to do mean/var
        # computation and normalization.
        #
        # Prefer the simd_width*2 specialization when cols is large enough
        # that the baseline Path A would saturate at least half of
        # max_warps_per_block. At that point the inter-warp barrier in
        # block_reduce_dual_sum dominates, and halving the warp count
        # (e.g. 24 -> 12 at cols=6144, bf16) is a net win. Below that
        # threshold (e.g. cols <= 3072 for bf16), Path A's wider thread
        # parallelism amortises memory latency better than Path B's wider
        # per-thread SIMD.
        if cols % (simd_width * 2) == 0 and (
            WARP_SIZE * simd_width * (max_warps_per_block // 2)
        ) <= cols <= (WARP_SIZE * (simd_width * 2) * max_warps_per_block):
            comptime kernel = layer_norm_gpu_warp_tiling[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                simd_width * 2,
                max_warps_per_block,
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=warp_tiling_block_dim(simd_width * 2),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        elif cols <= (WARP_SIZE * simd_width * max_warps_per_block):
            comptime kernel = layer_norm_gpu_warp_tiling[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                simd_width,
                max_warps_per_block,
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=warp_tiling_block_dim(simd_width),
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
        else:
            comptime kernel = layer_norm_gpu_block[
                mut=beta.mut,
                LayoutType=beta.LayoutType,
                origin=beta.origin,
                simd_width,
                input_fn_2d,
                gamma_fn,
                output_fn_2d,
            ]
            ctx.enqueue_function[kernel](
                flattened_shape,
                beta,
                epsilon,
                grid_dim=grid_dim,
                block_dim=warp_tiling_block_dim(simd_width),
                attributes=pdl_launch_attributes(PDLLevel.ON),
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
        ctx.enqueue_function[kernel](
            flattened_shape,
            beta,
            epsilon,
            grid_dim=grid_dim,
            block_dim=warp_tiling_block_dim(1),
            attributes=pdl_launch_attributes(PDLLevel.ON),
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
    input_fn: def[width: Int, alignment: Int](Int, Int) capturing -> SIMD[
        dtype, width
    ],
    gamma_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
](
    num_rows: Int,
    num_cols: Int,
    beta: TileTensor[mut=False, dtype, ...],
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
            dtype_: DType, simd_width: SIMDSize, alignment: Int
        ](idx: Int, val: SIMD[dtype_, simd_width]):
            output_fn[simd_width, alignment](
                row, idx, rebind[SIMD[dtype, simd_width]](val)
            )

        @__copy_capture(row)
        @parameter
        def input_gen_wrapper[
            dtype: DType, simd_width: Int
        ](col: Int) -> SIMD[dtype, simd_width]:
            return input_fn[simd_width, alignment=1](row, col).cast[dtype]()

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

        def _normalize[simd_width: Int](col: Int) {beta, mut}:
            var out_val = input_fn[simd_width, 1](row, col)
            var gamma_val = gamma_fn[simd_width, 1, 1](Index(col))
            var beta_col = beta.layout(col)

            var norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_val + beta.raw_load[width=simd_width](
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
    input_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    gamma_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
](
    shape: IndexList[rank],
    beta: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert beta.rank == 1, "beta must have rank 1"
    var last_dim = shape[rank - 1]

    var prod_all_but_last_dim = 1

    comptime for i in range(rank - 1):
        prod_all_but_last_dim *= shape[i]

    var num_workers = min(parallelism_level(ctx), prod_all_but_last_dim)
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
            simd_width: Int, alignment: Int
        ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
            # Translate a given 2D index back to the original n-D tensor
            var indices = _get_start_indices_of_nth_subvolume(
                row_idx + row, shape
            )
            indices[rank - 1] = col
            return input_fn[simd_width, rank, alignment](indices.canonicalize())

        @__copy_capture(row_idx)
        @parameter
        @always_inline
        def output_fn_2d[
            simd_width: SIMDSize, alignment: Int
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

    sync_parallelize[task_func](num_workers, ctx)


@always_inline
def layer_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: def[_width: Int, _rank: Int, alignment: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    input_1_fn: def[_width: Int, _rank: Int, alignment: Int](
        IndexList[_rank]
    ) capturing -> SIMD[dtype, _width],
    output_0_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
](
    shape: IndexList[rank],
    gamma_shape: IndexList[1],
    beta: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime assert beta.rank == 1, "beta must have rank 1"
    # Note: we only support reduction along the last dimension
    if gamma_shape[0] != shape[rank - 1]:
        raise Error("Gamma size does not match dimension of reduction.")

    if Int(beta.layout.shape[0]().value()) != shape[rank - 1]:
        raise Error("Beta size does not match dimension of reduction.")

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "layer_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        comptime if is_cpu[target]():
            layer_norm_cpu[input_0_fn, input_1_fn, output_0_fn](
                shape.canonicalize(),
                beta,
                epsilon,
                Optional[DeviceContext](ctx),
            )
        elif is_gpu[target]():
            layer_norm_gpu[input_0_fn, input_1_fn, output_0_fn](
                shape.canonicalize(),
                beta,
                epsilon,
                ctx=ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


@always_inline
def layer_norm_shape[
    dtype: DType
](
    input: TileTensor[mut=False, dtype, ...],
    gamma: TileTensor[mut=False, dtype, ...],
    beta: TileTensor[mut=False, dtype, ...],
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
    simd_width: SIMDSize,
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


@__name(t"rms_norm_gpu_warp_tiling_128_{dtype}_{multiply_before_cast}")
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
    output_fn: def[width: SIMDSize, alignment: Int](
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
    var block_row = block_idx.x * warps_per_block * 2
    var warp_id = ufloordiv(tid, WARP_SIZE)
    var sub_warp_id = ufloordiv(umod(tid, WARP_SIZE), half_warp_size)
    # Each warp handles 2 rows, offset by the block's base row
    var row = block_row + warp_id * 2 + Int(sub_warp_id)
    var local_tid = umod(tid, half_warp_size)
    var idx = local_tid * simd_width

    with PDL():
        var gamma_val = SIMD[dtype, simd_width](0)
        if row < num_rows and idx < num_cols:
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()
            # Prefetch gamma before reduction to overlap load with compute.
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        var norm_val = _rms_norm_warp_tiling_subkernel[
            warps_per_block, multiply_before_cast, rows_per_warp=2
        ](
            row,
            idx,
            vec_data,
            gamma_val,
            eps_accum,
            weight_offset_accum,
            num_cols,
        )
        if row < num_rows and idx < num_cols:
            output_fn[simd_width, align](row, idx, norm_val)


# Barrier-free, SMEM-free warp-per-row RMSNorm: one warp owns a full row,
# `rows_per_block` warps per block, `warp.sum` reduction (no block barrier).
#
# `single_pass` selects the load strategy:
#   * single_pass=True: cache each lane's `chunks` exact-fit SIMD vectors in
#     registers across the reduction and normalize from registers (input read
#     ONCE). Requires exact fit (`chunks * WARP_SIZE * simd_width == num_cols`).
#     Fastest for narrow f32/bf16 rows of 1..4 vectors/lane (1.2-1.4x over
#     two-pass); beyond that the register cache spills.
#   * single_pass=False: two passes -- accumulate mean-of-squares, then reload
#     from L2 and normalize. Handles ragged tails and wider rows. `chunks` is
#     unused (pass any value, e.g. 1).
@__name(
    t"rms_norm_gpu_warp_per_row_{dtype}_{single_pass}_{chunks}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_per_row[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    //,
    simd_width: Int,
    rows_per_block: Int,
    single_pass: Bool,
    chunks: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, alignment: Int](
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

    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = thread_idx.x
    var warp_in_block = ufloordiv(tid, WARP_SIZE)
    var lane = umod(tid, WARP_SIZE)
    var row = block_idx.x * rows_per_block + Int(warp_in_block)

    comptime stride = WARP_SIZE * simd_width

    comptime if single_pass:
        # Register-cached input chunks (accum precision), carried across the warp
        # reduction so the normalize pass needs no re-read.
        var vec_data = InlineArray[SIMD[accum_type, simd_width], chunks](
            fill=SIMD[accum_type, simd_width](0)
        )

        with PDL():
            # Single load pass: cache each chunk and accumulate mean-of-squares.
            var thread_m2 = Scalar[accum_type](0)
            if row < num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    vec_data[c] = input_fn[simd_width](row, col).cast[
                        accum_type
                    ]()
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
            )

            # Normalize from the cached registers (no reload).
            if row < num_rows:
                comptime for c in range(chunks):
                    var col = Int(lane) * simd_width + c * stride
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (
                            vec_data[c] * norm_factor * gamma_accum
                        ).cast[dtype]()
                    else:
                        norm_val = (vec_data[c] * norm_factor).cast[dtype]() * (
                            gamma_val + weight_offset
                        )
                    output_fn[simd_width, align](row, col, norm_val)
    else:
        with PDL():
            # Pass 1: accumulate the per-thread mean-of-squares scalar only.
            var thread_m2 = Scalar[accum_type](0)
            if row < num_rows:
                var col = Int(lane) * simd_width
                while col < num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    thread_m2 += (v**2).reduce_add()
                    col += stride

            # Barrier-free, SMEM-free warp reduction (shuffle butterfly).
            var row_m2 = warp.sum(thread_m2)
            var norm_factor = rsqrt(
                (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
            )

            # Pass 2: reload from L2 and normalize.
            if row < num_rows:
                var col = Int(lane) * simd_width
                while col < num_cols:
                    var v = input_fn[simd_width](row, col).cast[accum_type]()
                    var gamma_val = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    var norm_val: SIMD[dtype, simd_width]
                    comptime if multiply_before_cast:
                        var gamma_accum = (
                            gamma_val.cast[accum_type]() + weight_offset_accum
                        )
                        norm_val = (v * norm_factor * gamma_accum).cast[dtype]()
                    else:
                        norm_val = (v * norm_factor).cast[dtype]() * (
                            gamma_val + weight_offset
                        )
                    output_fn[simd_width, align](row, col, norm_val)
                    col += stride


# Static-divisor row -> n-D start-index decomposition.
#
# Functionally identical to `_get_start_indices_of_nth_subvolume[1](n, shape)`
# (the `subvolume_rank == 1` row-translation: dim `rank-1` is the column, set by
# the caller; dims `1..rank-2` are decomposed; dim 0 is the final quotient), but
# the shape arrives as the `Coord` *type* `ShapeCoord` rather than a runtime
# `IndexList`. For dims whose extent is statically known in the `Coord` type
# (`ParamListType[i].is_static_value`), the divisor becomes a compile-time
# literal, so the `divmod` strength-reduces to a magic-multiply + shift
# (verified: SASS `IMAD.HI`/`SHF`, no `IDIV`/`MUFU.RCP`) instead of the runtime
# Newton-reciprocal divide that an `IndexList` divisor forces. Dynamic dims fall
# back to the runtime value carried in `shape`.
#
# The row index `n` stays runtime, so the *result* indices are runtime; only the
# DIVISORS fold. The `Coord` is never passed to the device (a static-typed
# `Coord` is not `DevicePassable`); its static dims live in the type and dynamic
# dims (if any) are read from the captured runtime `shape` value.
@always_inline
def _get_row_start_indices_static[
    ShapeCoord: CoordLike, rank: Int
](n: Int, runtime_shape: IndexList[rank]) -> IndexList[rank]:
    var res = IndexList[rank]()

    # Match `_get_start_indices_of_nth_subvolume`'s fast paths so behavior is
    # bit-identical for the shapes that reach it.
    comptime if rank == 2:
        res[0] = n
        return res

    var curr = n

    comptime for i in reversed(range(1, rank - 1)):
        comptime ElemT = ShapeCoord.ParamListType[i]
        comptime if ElemT.is_static_value:
            # Compile-time divisor -> magic-multiply + shift (no `IDIV`).
            comptime divisor = ElemT.static_value
            res[i] = umod(curr, divisor)
            curr = ufloordiv(curr, divisor)
        else:
            # Dynamic dim: read the divisor from the runtime `IndexList` (which,
            # unlike a static-typed `Coord`, is `DevicePassable` and transfers as
            # a closure capture). This path emits a runtime divide, same as the
            # `_get_start_indices_of_nth_subvolume` baseline.
            var divisor = runtime_shape[i]
            res[i] = umod(curr, divisor)
            curr = ufloordiv(curr, divisor)

    res[0] = curr
    return res


# SM100 (B200) primary target; portable (only uses `block_reduce`, warp
# shuffle, and a per-thread grid-stride column loop — no arch intrinsics).
#
# Warp-tiling RMSNorm tuned to cut instruction overhead. Three structural
# choices vs the old single-chunk-per-thread form (which was instruction-issue
# bound on B200, not bandwidth bound -- high SM issue but low L2/HBM
# throughput):
#
#   1. The rank-N row -> base-coords translation
#      (`_get_start_indices_of_nth_subvolume`) is hoisted and run ONCE per
#      thread, then reused for every chunk's load AND store. The old 2D
#      wrappers ran that divmod chain twice per thread (once in `input_fn_2d`,
#      once in `output_fn_2d`); we now take the rank-N `input_fn`/`output_fn`
#      directly and only mutate `base[rank - 1]` per chunk.
#   2. Each thread processes `chunks_per_thread` independent vector chunks
#      (unrolled LDGs cached in registers), so the launch uses fewer threads
#      per block -> fewer warps -> a cheaper two-barrier `block_reduce`, and
#      the loads pipeline (ILP). Data stays in registers between the reduction
#      and the normalize pass (still a single global read).
#   3. When `exact_fit` (block_dim * simd_width * chunks_per_thread == cols),
#      every thread is fully active, so the per-chunk `col < num_cols` guards
#      (the ISETP/SEL bloat) are dropped at comptime. A guarded variant
#      (`exact_fit=False`) handles ragged tails.
@__name(
    t"rms_norm_gpu_warp_tiling_{dtype}_{chunks_per_thread}_{exact_fit}_{multiply_before_cast}"
)
def rms_norm_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    dtype: DType,
    rank: Int,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    chunks_per_thread: Int,
    exact_fit: Bool,
    input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank],
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

    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var bdim = Int(block_dim.x)

    # Hoist the rank-N row translation ONCE; reuse the base for load and store.
    var base = _get_start_indices_of_nth_subvolume(row, shape)

    # Per-chunk register-cached input (in accum precision) and gamma weights,
    # carried across the reduction so the normalize pass needs no re-read.
    var vec_data = InlineArray[SIMD[accum_type, simd_width], chunks_per_thread](
        fill=SIMD[accum_type, simd_width](0)
    )
    var gamma_val = InlineArray[SIMD[dtype, simd_width], chunks_per_thread](
        fill=SIMD[dtype, simd_width](0)
    )

    with PDL():
        var thread_m2 = Scalar[accum_type](0)

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width
            comptime if exact_fit:
                base[rank - 1] = col
                vec_data[c] = input_fn[simd_width](base.canonicalize()).cast[
                    accum_type
                ]()
                gamma_val[c] = gamma.load[width=simd_width, alignment=align](
                    Coord(col)
                )
                thread_m2 += (vec_data[c] ** 2).reduce_add()
            else:
                if col < num_cols:
                    base[rank - 1] = col
                    vec_data[c] = input_fn[simd_width](
                        base.canonicalize()
                    ).cast[accum_type]()
                    gamma_val[c] = gamma.load[
                        width=simd_width, alignment=align
                    ](Coord(col))
                    thread_m2 += (vec_data[c] ** 2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
        )

        comptime for c in range(chunks_per_thread):
            var col = (c * bdim + tid) * simd_width

            @always_inline
            @parameter
            def _normalize() -> SIMD[dtype, simd_width]:
                comptime if multiply_before_cast:
                    var gamma_accum = (
                        gamma_val[c].cast[accum_type]() + weight_offset_accum
                    )
                    return (vec_data[c] * norm_factor * gamma_accum).cast[
                        dtype
                    ]()
                else:
                    return (vec_data[c] * norm_factor).cast[dtype]() * (
                        gamma_val[c] + weight_offset
                    )

            comptime if exact_fit:
                base[rank - 1] = col
                output_fn[simd_width, align](base.canonicalize(), _normalize())
            else:
                if col < num_cols:
                    base[rank - 1] = col
                    output_fn[simd_width, align](
                        base.canonicalize(), _normalize()
                    )


@always_inline
def _rms_norm_gpu_block_subkernel[
    dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[mut=False, dtype, ...],
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
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width
        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            thread_m2 += (vec_data**2).reduce_add()

    var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_m2
    )
    var norm_factor = rsqrt((row_m2 / Scalar[accum_type](num_cols)) + eps_accum)

    # Need a pass again to perform in place normalization.
    for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
        var offset = x * block_dim.x * simd_width + tid * simd_width

        if offset < num_cols:
            var vec_data = input_fn[simd_width](row, offset).cast[accum_type]()
            var norm_val: SIMD[dtype, simd_width]
            var gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(offset)
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


# ===----------------------------------------------------------------------=== #
# Row mean-of-squares reduction
# ===----------------------------------------------------------------------=== #
#
# SM100 (B200) primary target; portable to SM90 / CDNA4 / Apple (only uses
# `block_reduce` + grid-stride column loop, no arch-specific intrinsics).
#
# Computes, for an input `x` of shape `[..., N]` flattened to `[M, N]`:
#     out[m] = sum_n(accum(x[m, n])^2) / N        (accumulated in `accum_type`)
# and writes a `[M, 1]`-shaped `out_dtype` result (one value per row).
#
# This mirrors the reduction half of `_rms_norm_warp_tiling_subkernel`
# (`var thread_m2 = (vec_data**2).reduce_add()` then `block_reduce`, then
# `row_m2 / num_cols`) WITHOUT applying the norm and WITHOUT a `gamma`
# weight. Like the rms-norm block path it launches exactly one block per row
# (`grid_dim = rows`), so small-M decode does not over-provision the grid.


@__name(t"row_mean_of_squares_gpu_block_{in_dtype}_{out_dtype}")
def row_mean_of_squares_gpu_block[
    in_dtype: DType,
    out_dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    output_fn: def(row: Int, val: Scalar[out_dtype]) capturing -> None,
](num_cols: Int):
    comptime accum_type = get_accum_type[in_dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    var thread_m2 = Scalar[accum_type](0)

    with PDL():
        # Each block owns a single row; threads grid-stride across the columns.
        for x in range(ceildiv(ceildiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                thread_m2 += (vec_data**2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

        if tid == 0:
            var mean = row_m2 / Scalar[accum_type](num_cols)
            output_fn(Int(row), mean.cast[out_dtype]())


@__name(t"row_mean_of_squares_qk_gpu_block_{in_dtype}_{out_dtype}")
def row_mean_of_squares_qk_gpu_block[
    in_dtype: DType,
    out_dtype: DType,
    out_mut: Bool,
    out_layout: TensorLayout,
    out_origin: Origin[mut=out_mut],
    q_layout: TensorLayout,
    q_origin: Origin,
    k_layout: TensorLayout,
    k_origin: Origin,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
](
    output: TileTensor[out_dtype, out_layout, out_origin],
    q: TileTensor[in_dtype, q_layout, q_origin],
    k: TileTensor[in_dtype, k_layout, k_origin],
    q_cols: Int,
    k_cols: Int,
) where out_mut:
    """Fused per-row mean of squares for Q and K in a single launch.

    The grid is 2D: `block_idx.x` selects the row and `block_idx.y` selects the
    operand (0 = Q, 1 = K). Each block owns one (row, operand) reduction and
    writes column `block_idx.y` of the `[rows, 2]` output. This replaces two
    `row_mean_of_squares` launches plus a concat with one launch. All operands
    (`q [M, Nq]`, `k [M, Nk]`, and the `[M, 2]` output) are passed directly as
    `TileTensor`s and loaded/stored in-kernel.
    """
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime accum_type = get_accum_type[in_dtype]()

    var tid = thread_idx.x
    var row = block_idx.x
    # block_idx.y is uniform across the block, so this branch never diverges.
    var is_k = block_idx.y == 1
    var num_cols = k_cols if is_k else q_cols
    var thread_m2 = Scalar[accum_type](0)

    with PDL():
        # Each block owns a single (row, operand); threads grid-stride the cols.
        for x in range(ceildiv(ceildiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var vec_data: SIMD[in_dtype, simd_width]
                if is_k:
                    vec_data = k.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    )
                else:
                    vec_data = q.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    )
                thread_m2 += (vec_data.cast[accum_type]() ** 2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )

        if tid == 0:
            var mean = row_m2 / Scalar[accum_type](num_cols)
            output.store[width=1](
                Coord(Index(Int(row), Int(block_idx.y))),
                mean.cast[out_dtype](),
            )


@__name(t"rms_norm_gpu_block_{dtype}_{multiply_before_cast}")
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
    output_fn: def[width: SIMDSize, alignment: Int](
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
    //,
    rank: Int,
    input_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    # Boundary `IndexList` -> `Coord` migration (mirror of softmax PR #88203):
    # the public `shape` arrives as a `Coord` (statically-known outer dims are
    # encoded in its type), then is materialized to a runtime `IndexList` once.
    # All existing runtime arithmetic and the IndexList-form GPU kernels run on
    # `shape_il`; the public n-D lambdas are Coord-form and re-wrapped to the
    # internal IndexList interface the kernels expect.
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    if rank == 0:
        return

    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))

    # Internal IndexList-form adapters: the warp-tiling / block GPU kernels
    # consume `def[width, rank](IndexList[rank])` lambdas (they build the n-D
    # index in-kernel), so wrap the Coord-form public lambdas back to that shape.
    @parameter
    @always_inline
    def input_fn_il[
        simd_width: Int, _rank: Int
    ](indices: IndexList[_rank]) -> SIMD[dtype, simd_width]:
        return input_fn[simd_width](Coord(rebind[IndexList[rank]](indices)))

    @parameter
    @always_inline
    def output_fn_il[
        simd_width: SIMDSize, alignment: Int
    ](indices: IndexList[rank], val: SIMD[dtype, simd_width]) -> None:
        output_fn[simd_width, alignment](Coord(indices), val)

    # Derive the number of columns from the `gamma` input as this value may be
    # statically known.
    var cols = Int(gamma.dim[0]())

    if cols == 0:
        return

    var rows = shape_il.flattened_length() // cols

    @parameter
    # `@__copy_capture(shape_il)` is required: these wrappers are embedded into
    # GPU kernels as `capturing` closures, and a captured *local* `var` (unlike
    # a function parameter, which the pre-migration code captured directly) is
    # not carried to the device without an explicit copy-capture. Without it the
    # rank-N `_get_start_indices_of_nth_subvolume` divmod reads garbage outer
    # dims on device (rank-2 is unaffected since its outer translation is
    # trivial; rank>=3 produces wrong results / launch failures).
    @__copy_capture(shape_il)
    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: SIMDSize, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape_il)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](Coord(indices), val)

    @__copy_capture(shape_il)
    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape_il)
        indices[rank - 1] = col
        return input_fn[simd_width](Coord(indices))

    # Static-divisor variants of the 2D wrappers. Identical translation, but the
    # row -> n-D decomposition divides by the statically-known outer dims carried
    # in `type_of(shape)` (a `Coord` whose inner dims are `ComptimeInt` on the
    # static-shape path), so the per-row `divmod` strength-reduces to magic-multiply
    # + shift instead of the runtime Newton-reciprocal `IDIV` that the
    # `shape_il`-form wrappers above emit.
    comptime ShapeCoordType = type_of(shape)

    @__copy_capture(shape_il)
    @parameter
    @always_inline
    def output_fn_2d_static[
        simd_width: SIMDSize, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        var indices = _get_row_start_indices_static[ShapeCoordType, rank](
            row, shape_il
        )
        indices[rank - 1] = col
        output_fn[simd_width, alignment](Coord(indices), val)

    @__copy_capture(shape_il)
    @parameter
    @always_inline
    def input_fn_2d_static[
        simd_width: Int
    ](row: Int, col: Int) -> SIMD[dtype, simd_width]:
        var indices = _get_row_start_indices_static[ShapeCoordType, rank](
            row, shape_il
        )
        indices[rank - 1] = col
        return input_fn[simd_width](Coord(indices))

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    comptime sm_version = ctx.default_device_info.version
    comptime sm_count = ctx.default_device_info.sm_count
    comptime warp_per_row_rows_per_block = 8
    var warp_per_row_min_grid = 3 * sm_count
    var warp_per_row_region = (
        ceildiv(rows, warp_per_row_rows_per_block) >= warp_per_row_min_grid
    )
    # Conservative floor: below one block per SM, chunking can't raise
    # occupancy and only reassociates the mean-of-squares reduction (the real
    # crossover is higher and width-dependent; this sits safely below it).
    var enough_rows_to_chunk = rows >= sm_count

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    # Warp-tiling launch geometry for a given per-thread (`eff_simd`, `chunks`):
    # one block per row, `_wt_threads_per_block` threads (rounded to whole
    # warps, capped at the device max), each owning `eff_simd * chunks` columns.
    # Single source of truth for both the launcher and the warp-per-row gate
    # below.
    @parameter
    @always_inline
    def _wt_threads_per_block[eff_simd: Int, chunks: Int]() -> Int:
        var threads = ceildiv(ceildiv(cols, eff_simd), chunks)
        return min(
            ceildiv(threads, WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )

    # `exact` means every thread is fully active (the block tiles the row with
    # no ragged tail), so the unguarded kernel can be used.
    @parameter
    @always_inline
    def _wt_exact[eff_simd: Int, chunks: Int]() -> Bool:
        return (
            _wt_threads_per_block[eff_simd, chunks]() * eff_simd * chunks
        ) == cols

    # Within the warp-tiling column range, warp-tiling (register-cached chunks)
    # beats the barrier-free warp-per-row kernel on SM100 (B200) whenever it
    # tiles the row exactly *and* the row is wide enough to amortize the
    # inter-warp block reduction: a measured 1.1-1.4x at 2048..8192 cols.
    # Warp-per-row wins in two cases, both of which must stay on it:
    #   1. ragged tail (e.g. 8192x2880, cols not a clean multiple of the
    #      per-thread tile -> wasted threads + per-chunk bounds guards), and
    #   2. narrow rows (cols <= 1024), where the one-warp-per-row kernel has no
    #      inter-warp barrier to pay and warp-tiling's block reduce dominates
    #      (measured: 8192x256 1.9x, 8192x512 1.7x, 8192x1024 1.3x slower under
    #      warp-tiling).
    # So prefer warp-tiling only for exact-fit rows past the narrow-row floor;
    # everything else keeps warp-per-row. Uses the native-width chunk count the
    # dispatch below would pick (1 up to one warp-row, 2 up to two, else 4) and
    # the shared geometry.
    #
    # This is only consulted to gate warp-per-row, which itself only runs for
    # `cols <= WARP_SIZE * simd_width * max_warps_per_block`. Within that range
    # the dispatch always launches at native `simd_width` with these same chunk
    # counts, so the gate matches the kernel that runs. The wider-row branch
    # below (`simd_width * 2`) lies entirely past the warp-per-row bound, so the
    # native-width value computed here is never read against it.
    #
    # `warp_tiling_min_cols` is the narrow-row floor: warp-tiling is preferred
    # only above it. 1024 = WARP_SIZE * simd_width * 4 (= one max-width
    # per-thread tile across a single warp) is the measured crossover on B200
    # bf16 -- the largest exact-fit width that still loses to warp-per-row.
    var warp_tiling_min_cols = WARP_SIZE * simd_width * 4
    var warp_tiling_exact_fit: Bool
    if cols <= (WARP_SIZE * simd_width):
        warp_tiling_exact_fit = _wt_exact[simd_width, 1]()
    elif cols <= (WARP_SIZE * simd_width * 2):
        warp_tiling_exact_fit = _wt_exact[simd_width, 2]()
    else:
        warp_tiling_exact_fit = _wt_exact[simd_width, 4]()
    var warp_tiling_exact = warp_tiling_exact_fit and (
        cols > warp_tiling_min_cols
    )

    # Launch the multi-chunk warp-tiling kernel. `exact_fit` (every thread
    # fully active, no ragged tail) is decided at runtime and selects the
    # unguarded instantiation.
    @parameter
    @always_inline
    def _launch_warp_tiling[eff_simd: Int, chunks: Int]() raises:
        var threads_per_block = _wt_threads_per_block[eff_simd, chunks]()
        var exact = _wt_exact[eff_simd, chunks]()

        @parameter
        @always_inline
        def _enqueue[exact_fit: Bool]() raises:
            comptime kernel = rms_norm_gpu_warp_tiling[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                eff_simd,
                max_warps_per_block,
                chunks,
                exact_fit,
                input_fn_il,
                output_fn_il,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel](
                shape_il.canonicalize(),
                gamma,
                epsilon,
                weight_offset,
                cols,
                grid_dim=rows,
                block_dim=threads_per_block,
                attributes=pdl_launch_attributes(pdl_level),
            )

        if exact:
            _enqueue[True]()
        else:
            _enqueue[False]()

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
                input_fn_2d_static,
                output_fn_2d_static,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel](
                gamma,
                epsilon,
                weight_offset,
                rows,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(pdl_level),
            )
        elif (
            cols >= 128
            and cols <= (WARP_SIZE * simd_width * max_warps_per_block)
            and warp_per_row_region
            and not warp_tiling_exact
        ):
            comptime rows_per_block = warp_per_row_rows_per_block
            block_dim = rows_per_block * WARP_SIZE
            grid_dim = ceildiv(rows, rows_per_block)

            # Single-pass register-cached warp-per-row for narrow, exact-fit
            # rows. Keeps the 8-rows/block high-occupancy geometry of the
            # two-pass kernel but loads each lane's row chunks ONCE (cached in
            # registers across the `warp.sum`) instead of reloading + resquaring
            # in pass 2.
            comptime sp_stride = WARP_SIZE * simd_width
            var sp_chunks = cols // sp_stride
            if (
                dtype in (DType.float32, DType.bfloat16)
                and (cols % sp_stride == 0)
                and (sp_chunks >= 1)
                and (sp_chunks <= 4)
            ):
                comptime for cc in range(1, 5):
                    if sp_chunks == cc:
                        comptime kernel = rms_norm_gpu_warp_per_row[
                            mut=gamma.mut,
                            LayoutType=gamma.LayoutType,
                            origin=gamma.origin,
                            simd_width,
                            rows_per_block,
                            True,
                            cc,
                            input_fn_2d_static,
                            output_fn_2d_static,
                            multiply_before_cast=multiply_before_cast,
                        ]
                        ctx.enqueue_function[kernel](
                            gamma,
                            epsilon,
                            weight_offset,
                            rows,
                            cols,
                            grid_dim=grid_dim,
                            block_dim=block_dim,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
            else:
                comptime kernel = rms_norm_gpu_warp_per_row[
                    mut=gamma.mut,
                    LayoutType=gamma.LayoutType,
                    origin=gamma.origin,
                    simd_width,
                    rows_per_block,
                    False,
                    1,
                    input_fn_2d_static,
                    output_fn_2d_static,
                    multiply_before_cast=multiply_before_cast,
                ]
                ctx.enqueue_function[kernel](
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
            # CDNA4 (MI355X): when there are enough rows to keep the GPU busy,
            # use a 2x-wider per-thread SIMD so each row's block needs half the
            # warps. This halves the inter-warp block reduction cost and
            # doubles blocks-per-CU, lifting achieved HBM bandwidth by ~15-30%
            # on prefill-sized shapes. It is gated on row count because the
            # smaller block lowers total occupancy when rows are few (a net loss
            # below ~8x the CU count).
            comptime sw_wide = simd_width * 2
            comptime widen_ok = sm_version == "CDNA4"
            var enough_rows = rows >= 8 * sm_count
            if widen_ok and enough_rows and cols % sw_wide == 0:
                _launch_warp_tiling[sw_wide, 1]()
            elif not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width, 1]()
            else:
                # Narrow rows: a single full-width pass keeps the block small;
                # split into independent chunks (ILP + smaller block_reduce)
                # once there are enough columns to fill ~2+ chunks/thread.
                if cols <= (WARP_SIZE * simd_width):
                    _launch_warp_tiling[simd_width, 1]()
                elif cols <= (WARP_SIZE * simd_width * 2):
                    _launch_warp_tiling[simd_width, 2]()
                else:
                    _launch_warp_tiling[simd_width, 4]()
        elif (
            cols <= (WARP_SIZE * (simd_width * 2) * max_warps_per_block)
            and cols % (simd_width * 2) == 0
        ):
            # Wider rows: double the vector width and, for high-row launches,
            # split into chunks.
            if not enough_rows_to_chunk:
                _launch_warp_tiling[simd_width * 2, 1]()
            elif cols <= (WARP_SIZE * simd_width * 2 * 2):
                _launch_warp_tiling[simd_width * 2, 2]()
            else:
                _launch_warp_tiling[simd_width * 2, 4]()
        else:
            comptime kernel = rms_norm_gpu_block[
                mut=gamma.mut,
                LayoutType=gamma.LayoutType,
                origin=gamma.origin,
                simd_width,
                max_warps_per_block,
                input_fn_2d_static,
                output_fn_2d_static,
                multiply_before_cast=multiply_before_cast,
            ]
            ctx.enqueue_function[kernel](
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
            input_fn_2d_static,
            output_fn_2d_static,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
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
    output_fn: def[width: SIMDSize, alignment: Int](
        Int, Int, SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[mut=False, dtype, ...],
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

        def _normalize[simd_width: Int](col: Int) {gamma, weight_offset, mut}:
            var input_val = input_fn[simd_width](row, col).cast[
                intermediate_type
            ]()
            var gamma_val = gamma.load[width=simd_width, alignment=1](
                Coord(col)
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
    output_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank],
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: Optional[DeviceContext] = None,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var last_dim = shape[rank - 1]
    var prod_all_but_last_dim = shape.flattened_length() // last_dim

    var num_workers = min(parallelism_level(ctx), prod_all_but_last_dim)
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
            simd_width: SIMDSize, alignment: Int
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

    sync_parallelize[task_func](num_workers, ctx)


@always_inline
def _rms_norm_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_fn: def[width: SIMDSize, alignment: Int](
        Coord, SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    # Boundary `IndexList` -> `Coord` migration (softmax PR #88203 form). The
    # public n-D lambdas + `shape` are `Coord`; `shape_il` materializes the
    # runtime `IndexList` once for the rank-check, the empty-tensor guard, and
    # the IndexList-form CPU path. Callers whose lambdas need runtime index
    # subscripts (`kv_cache.mojo`) wrap their IndexList-form lambdas to
    # `Coord`-form at the call site (see `coord_to_index_list`).
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    var shape_il = rebind[IndexList[rank]](coord_to_index_list(shape))

    # Note: we only support reduction along the last dimension
    if Int(gamma.layout.shape[0]().value()) != shape_il[rank - 1]:
        raise Error(
            "Gamma size "
            + String(gamma.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(shape_il[rank - 1])
            + "."
        )

    if shape_il.flattened_length() == 0:
        # Nothing to do.
        return

    comptime if is_cpu[target]():
        # The CPU path consumes n-D `IndexList`-form lambdas; wrap the Coord
        # public lambdas back to that interface.
        @parameter
        @always_inline
        def input_fn_il[
            width: Int, _rank: Int
        ](indices: IndexList[_rank]) -> SIMD[dtype, width]:
            return input_0_fn[width](Coord(rebind[IndexList[rank]](indices)))

        @parameter
        @always_inline
        def output_fn_il[
            width: SIMDSize, alignment: Int
        ](indices: IndexList[rank], val: SIMD[dtype, width]) -> None:
            output_fn[width, alignment](Coord(indices), val)

        rms_norm_cpu[
            input_fn_il, output_fn_il, multiply_before_cast=multiply_before_cast
        ](
            shape_il,
            gamma,
            epsilon,
            weight_offset,
            Optional[DeviceContext](ctx),
        )
    elif is_gpu[target]():
        rms_norm_gpu[
            rank,
            input_0_fn,
            output_fn,
            multiply_before_cast=multiply_before_cast,
        ](
            shape,
            gamma,
            epsilon,
            weight_offset,
            ctx,
        )
    else:
        comptime assert False, "unsupported target " + target


@__name(
    t"rms_norm_fused_residual_add_gpu_warp_tiling_{dtype}_{multiply_before_cast}",
)
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
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDSize, alignment: Int](
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
    var idx = tid * simd_width

    with PDL():
        var gamma1_val = SIMD[dtype, simd_width](0)
        if idx < num_cols:
            vec_data = input_fn[simd_width](row, idx)
            # Prefetch gamma1 before reduction to overlap load with compute.
            gamma1_val = gamma1.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        var norm1_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            row,
            idx,
            vec_data.cast[accum_type](),
            gamma1_val,
            eps_accum1,
            weight_offset_accum1,
            num_cols,
        )

        var gamma2_val = SIMD[dtype, simd_width](0)
        if idx < num_cols:
            norm1_val += residual_input_fn[simd_width](row, idx)
            output_residual_fn[simd_width, align](row, idx, norm1_val)
            # Prefetch gamma2 before second reduction.
            gamma2_val = gamma2.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        var norm2_val = _rms_norm_warp_tiling_subkernel[
            max_warps_per_block, multiply_before_cast
        ](
            row,
            idx,
            norm1_val.cast[accum_type](),
            gamma2_val,
            eps_accum2,
            weight_offset_accum2,
            num_cols,
        )

        if idx < num_cols:
            output_fn[simd_width, align](row, idx, norm2_val)


@__name(
    t"rms_norm_fused_residual_add_gpu_block_{dtype}_{multiply_before_cast}",
)
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
    output_fn: def[width: SIMDSize, alignment: Int](
        row: Int, col: Int, val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDSize, alignment: Int](
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
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var vec_data = input_fn[simd_width](row, offset).cast[
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
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                var gamma1_val = gamma1.load[width=simd_width, alignment=align](
                    Coord(offset)
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

                var residual_val = residual_input_fn[simd_width](row, offset)
                var residual_add_val = norm1_val + residual_val
                output_residual_fn[simd_width, align](
                    row, offset, residual_add_val
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
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var stage2_input = shared_mem.load[width=simd_width](
                    offset
                ).cast[accum_type]()
                var gamma2_val = gamma2.load[width=simd_width, alignment=align](
                    Coord(offset)
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

                output_fn[simd_width, align](row, offset, norm2_val)


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
    output_residual_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    shape: IndexList[rank, ...],
    gamma1: TileTensor[mut=False, dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[mut=False, dtype, ...],
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
        simd_width: SIMDSize, alignment: Int
    ](row: Int, col: Int, val: SIMD[dtype, simd_width]) -> None:
        # Translate a given 2D index back to the original n-D tensor
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    def output_residual_fn_2d[
        simd_width: SIMDSize, alignment: Int
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
            ctx.enqueue_function[kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel.ON),
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
            ctx.enqueue_function[kernel](
                gamma1,
                epsilon1,
                weight_offset1,
                gamma2,
                epsilon2,
                weight_offset2,
                cols,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel.ON),
                shared_mem_bytes=shared_mem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(shared_mem_size)
                ),
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
        ctx.enqueue_function[kernel](
            gamma1,
            epsilon1,
            weight_offset1,
            gamma2,
            epsilon2,
            weight_offset2,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel.ON),
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
    output_0_fn: def[width: SIMDSize, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[mut=False, dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[mut=False, dtype, ...],
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
        width: SIMDSize, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var residual_val = residual_input_fn[width](idx)

        var residual_add_val = val + residual_val
        output_residual_fn[width, alignment](idx, residual_add_val)
        var intermediate_idx = intermediate_buffer.layout(Coord(idx))
        intermediate_buffer.raw_store[width=width, alignment=alignment](
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
        return intermediate_buffer.raw_load[width=width](intermediate_idx)

    rms_norm_cpu[
        intermediate_input_fn,
        output_0_fn,
        multiply_before_cast=multiply_before_cast,
    ](shape, gamma2, epsilon2, weight_offset2)

    intermediate_buffer_ptr.free()


# ===-----------------------------------------------------------------------===#
# RMS Norm + RoPE (fused)
# ===-----------------------------------------------------------------------===#


def _rms_norm_rope_gpu_warp_tiling[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[input_dtype, width],
    cos_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    sin_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[output_dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[input_dtype, LayoutType, origin],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime align = align_of[SIMD[input_dtype, simd_width]]()
    comptime out_align = align_of[SIMD[output_dtype, simd_width]]()
    comptime accum_type = get_accum_type[input_dtype]()

    # Shared memory to store raw (un-normalized) input values so threads can
    # read the "paired" element (col ± half_cols) for RoPE and re-normalize
    # it on-the-fly.  Storing raw input avoids the risk of exceeding the
    # shared-memory budget that could arise from storing normalized values in
    # a different precision.
    var shared_input = external_memory[
        Scalar[input_dtype],
        address_space=AddressSpace.SHARED,
        alignment=align_of[SIMD[input_dtype, simd_width]](),
        name="rms_norm_rope_normed",
    ]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var vec_data = SIMD[accum_type, simd_width](0)
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * simd_width

    with PDL():
        var gamma_val = SIMD[input_dtype, simd_width](0)
        if idx < num_cols:
            vec_data = input_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        # Compute RMS norm factor.
        var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()
        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
        )

        # Compute normed value in registers and store raw input to shared
        # memory for paired-element access during the RoPE step.
        # `norm_val` is rounded to `output_dtype` here (before RoPE) so the
        # split/mul see the same precision the unfused graph produced when it
        # cast the RMSNorm result to the output dtype prior to RoPE. When
        # input_dtype == output_dtype this is identical to the original kernel.
        var norm_val: SIMD[output_dtype, simd_width] = 0
        if idx < num_cols:
            comptime if multiply_before_cast:
                var gamma_accum = (
                    gamma_val.cast[accum_type]() + weight_offset_accum
                )
                norm_val = (vec_data * norm_factor * gamma_accum).cast[
                    output_dtype
                ]()
            else:
                norm_val = (vec_data * norm_factor).cast[output_dtype]() * (
                    gamma_val + weight_offset
                ).cast[output_dtype]()
            shared_input.store[alignment=align](
                idx, vec_data.cast[input_dtype]()
            )

        barrier()

        # Apply RoPE: result[col] = normed[col]*cos[col] + rotated[col]*sin[col]
        # where rotated[col] = -normed[col+half] for col < half, else normed[col-half].
        # The paired normed value is recomputed from the raw input in shared memory.
        if idx < num_cols:
            var half_cols = num_cols // 2
            var cos_val = cos_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            var sin_val = sin_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            var normed_col = norm_val.cast[accum_type]()

            # Since half_cols % simd_width == 0 (guaranteed by dispatch), all
            # simd_width elements belong entirely to one half.
            var paired_idx = (
                idx + half_cols if idx < half_cols else idx - half_cols
            )
            var paired_raw = shared_input.load[
                width=simd_width, alignment=align
            ](paired_idx).cast[accum_type]()
            var paired_gamma = gamma.load[width=simd_width, alignment=align](
                Coord(paired_idx)
            )
            var paired_normed: SIMD[accum_type, simd_width]
            # Cast through output_dtype to reproduce the same rounding that
            # would occur if the paired normed value had been stored and
            # reloaded from a typed buffer (matching the unfused graph, which
            # rounds the normed value to the output dtype before RoPE).
            comptime if multiply_before_cast:
                var paired_gamma_accum = (
                    paired_gamma.cast[accum_type]() + weight_offset_accum
                )
                paired_normed = (
                    (paired_raw * norm_factor * paired_gamma_accum)
                    .cast[output_dtype]()
                    .cast[accum_type]()
                )
            else:
                paired_normed = (
                    (paired_raw * norm_factor).cast[output_dtype]()
                    * (paired_gamma + weight_offset).cast[output_dtype]()
                ).cast[accum_type]()

            var rotated: SIMD[accum_type, simd_width]
            if idx < half_cols:
                rotated = -paired_normed
            else:
                rotated = paired_normed

            var result = (normed_col * cos_val + rotated * sin_val).cast[
                output_dtype
            ]()
            output_fn[alignment=out_align](row, idx, result)


def _rms_norm_rope_gpu_warp_tiling_128[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    //,
    simd_width: Int,
    warps_per_block: Int,
    input_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[input_dtype, width],
    cos_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    sin_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[output_dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[input_dtype, LayoutType, origin],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    num_rows: Int,
    num_cols: Int,
):
    """Runs the fused RMSNorm+RoPE kernel optimized for small column counts (cols <= 128).

    Packs two rows per warp by splitting each 32-thread warp into two 16-thread
    half-warps, one per row.  Half-warp reductions via `lane_group_sum()`
    replace the full `block_reduce()`, and `syncwarp()` provides the
    shared-memory fence (no cross-warp shmem access occurs, so `__syncthreads`
    is not required).  Shared memory layout: `[warp_id * 2 + sub_warp_id][col]`,
    total size = `warps_per_block * 2 * num_cols * sizeof(input_dtype)`.

    Parameters:
        mut: Whether the gamma tensor's origin is mutable.
        LayoutType: Memory layout type of the gamma tensor.
        origin: Origin of the gamma tensor's memory.
        input_dtype: Data type of the input and output tensors.
        cos_sin_dtype: Data type of the cosine and sine tables.
        simd_width: Number of elements per SIMD vector per thread.
        warps_per_block: Number of warps per thread block; each warp handles
            two rows via two 16-thread half-warps.
        input_fn: Callback that loads `simd_width` input elements at
            `(row, col)`.
        cos_fn: Callback that loads `simd_width` cosine values at `(row, col)`.
        sin_fn: Callback that loads `simd_width` sine values at `(row, col)`.
        output_fn: Callback that stores `simd_width` result elements at
            `(row, col)` with the given alignment.
        multiply_before_cast: If `True`, gamma scaling is applied in
            accumulation precision before casting back to `input_dtype`.

    Args:
        gamma: RMS normalization scale weights with shape `[num_cols]`.
        epsilon: Small constant added to the variance for numerical stability.
        weight_offset: Additive offset applied to `gamma` before scaling.
        num_rows: Total number of rows to process.
        num_cols: Hidden dimension size; must be <= 128 and satisfy
            `cols % (2 * simd_width) == 0`.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime half_warp_size = WARP_SIZE // 2
    comptime align = align_of[SIMD[input_dtype, simd_width]]()
    comptime out_align = align_of[SIMD[output_dtype, simd_width]]()
    comptime accum_type = get_accum_type[input_dtype]()

    # Raw input is stored here so each half-warp can read the paired column
    # (col ± half_cols) during the RoPE step without a second global load.
    var shared_input = external_memory[
        Scalar[input_dtype],
        address_space=AddressSpace.SHARED,
        alignment=align_of[SIMD[input_dtype, simd_width]](),
        name="rms_norm_rope_normed_128",
    ]()

    var eps_accum = epsilon.cast[accum_type]()
    var weight_offset_accum = weight_offset.cast[accum_type]()

    var tid = thread_idx.x
    var local_warp_id = ufloordiv(tid, WARP_SIZE)
    var sub_warp_id = ufloordiv(umod(tid, WARP_SIZE), half_warp_size)
    var local_tid = umod(tid, half_warp_size)
    var idx = local_tid * simd_width

    # Each warp handles 2 rows; total rows per block = warps_per_block * 2.
    var block_row = block_idx.x * warps_per_block * 2
    var row = block_row + local_warp_id * 2 + Int(sub_warp_id)

    with PDL():
        var vec_data = SIMD[accum_type, simd_width](0)
        var gamma_val = SIMD[input_dtype, simd_width](0)

        if row < num_rows and idx < num_cols:
            vec_data = input_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            gamma_val = gamma.load[width=simd_width, alignment=align](
                Coord(idx)
            )

        # Half-warp reduction: each 16-thread group independently reduces
        # its own row's sum-of-squares.
        var thread_m2: Scalar[accum_type] = (vec_data**2).reduce_add()
        var row_m2 = warp.lane_group_sum[num_lanes=half_warp_size](thread_m2)
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
        )

        # Compute normed value and write raw input to this row's shmem slot.
        # `norm_val` is rounded to `output_dtype` (before RoPE) to match the
        # unfused graph; identical to the original when in/out dtypes match.
        var norm_val: SIMD[output_dtype, simd_width] = 0
        var shmem_row_offset = (local_warp_id * 2 + Int(sub_warp_id)) * num_cols

        if row < num_rows and idx < num_cols:
            comptime if multiply_before_cast:
                var gamma_accum = (
                    gamma_val.cast[accum_type]() + weight_offset_accum
                )
                norm_val = (vec_data * norm_factor * gamma_accum).cast[
                    output_dtype
                ]()
            else:
                norm_val = (vec_data * norm_factor).cast[output_dtype]() * (
                    gamma_val + weight_offset
                ).cast[output_dtype]()
            shared_input.store[alignment=align](
                shmem_row_offset + idx, vec_data.cast[input_dtype]()
            )

        # syncwarp() is sufficient: each warp's shmem section is disjoint from
        # every other warp's, so only intra-warp ordering is required.
        syncwarp()

        # Apply RoPE using the paired element re-normalized from shared memory.
        if row < num_rows and idx < num_cols:
            var half_cols = num_cols // 2
            var cos_val = cos_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            var sin_val = sin_fn[simd_width, alignment=align](row, idx).cast[
                accum_type
            ]()
            var normed_col = norm_val.cast[accum_type]()

            # paired_idx is in the same row's shmem slot (within [0, num_cols)).
            var paired_idx = (
                idx + half_cols if idx < half_cols else idx - half_cols
            )
            var paired_raw = shared_input.load[
                width=simd_width, alignment=align
            ](shmem_row_offset + paired_idx).cast[accum_type]()
            var paired_gamma = gamma.load[width=simd_width, alignment=align](
                Coord(paired_idx)
            )
            var paired_normed: SIMD[accum_type, simd_width]
            comptime if multiply_before_cast:
                var paired_gamma_accum = (
                    paired_gamma.cast[accum_type]() + weight_offset_accum
                )
                paired_normed = (
                    (paired_raw * norm_factor * paired_gamma_accum)
                    .cast[output_dtype]()
                    .cast[accum_type]()
                )
            else:
                paired_normed = (
                    (paired_raw * norm_factor).cast[output_dtype]()
                    * (paired_gamma + weight_offset).cast[output_dtype]()
                ).cast[accum_type]()

            var rotated: SIMD[accum_type, simd_width]
            if idx < half_cols:
                rotated = -paired_normed
            else:
                rotated = paired_normed

            var result = (normed_col * cos_val + rotated * sin_val).cast[
                output_dtype
            ]()
            output_fn[alignment=out_align](row, idx, result)


def _rms_norm_rope_gpu_block[
    mut: Bool,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    //,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[input_dtype, width],
    cos_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    sin_fn: def[width: Int, alignment: Int](
        row: Int, col: Int
    ) capturing -> SIMD[cos_sin_dtype, width],
    output_fn: def[width: Int, alignment: Int](
        row: Int, col: Int, val: SIMD[output_dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
](
    gamma: TileTensor[input_dtype, LayoutType, origin],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    num_cols: Int,
):
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    comptime align = align_of[SIMD[input_dtype, simd_width]]()
    comptime out_align = align_of[SIMD[output_dtype, simd_width]]()
    comptime accum_type = get_accum_type[input_dtype]()

    with PDL():
        var tid = thread_idx.x
        var row = block_idx.x
        var eps_accum = epsilon.cast[accum_type]()
        var weight_offset_accum = weight_offset.cast[accum_type]()

        # Pass 1: Compute sum-of-squares for norm factor.
        var thread_m2 = Scalar[accum_type](0)
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var v = input_fn[simd_width, alignment=align](row, offset).cast[
                    accum_type
                ]()
                thread_m2 += (v**2).reduce_add()

        var row_m2 = block_reduce[max_warps_per_block=max_warps_per_block](
            thread_m2
        )
        var norm_factor = rsqrt(
            (row_m2 / Scalar[accum_type](num_cols)) + eps_accum
        )

        # Pass 2: Load own and paired raw inputs from global memory, normalize
        # both on-the-fly using the shared norm_factor, apply RoPE, and write
        # output.  No shared memory is required: the paired element is fetched
        # directly from global memory, avoiding any shared-memory size concern.
        var half_cols = num_cols // 2
        for x in range(ceildiv(num_cols // simd_width, block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                var v = input_fn[simd_width, alignment=align](row, offset).cast[
                    accum_type
                ]()
                var gamma_val = gamma.load[width=simd_width, alignment=align](
                    Coord(offset)
                )
                var norm_val: SIMD[accum_type, simd_width]
                # Cast through output_dtype to reproduce the rounding that would
                # occur if the normed value were stored and reloaded from a typed
                # buffer (matching the unfused graph, which rounds the normed
                # value to the output dtype before RoPE). Identical to the
                # original kernel when input_dtype == output_dtype.
                comptime if multiply_before_cast:
                    var gamma_accum = (
                        gamma_val.cast[accum_type]() + weight_offset_accum
                    )
                    norm_val = (
                        (v * norm_factor * gamma_accum)
                        .cast[output_dtype]()
                        .cast[accum_type]()
                    )
                else:
                    norm_val = (
                        (v * norm_factor).cast[output_dtype]()
                        * (gamma_val + weight_offset).cast[output_dtype]()
                    ).cast[accum_type]()

                var paired_offset = (
                    offset + half_cols if offset
                    < half_cols else offset - half_cols
                )
                var paired_v = input_fn[simd_width, alignment=align](
                    row, paired_offset
                ).cast[accum_type]()
                var paired_gamma_val = gamma.load[
                    width=simd_width, alignment=align
                ](Coord(paired_offset))
                var paired_norm_val: SIMD[accum_type, simd_width]
                comptime if multiply_before_cast:
                    var paired_gamma_accum = (
                        paired_gamma_val.cast[accum_type]()
                        + weight_offset_accum
                    )
                    paired_norm_val = (
                        (paired_v * norm_factor * paired_gamma_accum)
                        .cast[output_dtype]()
                        .cast[accum_type]()
                    )
                else:
                    paired_norm_val = (
                        (paired_v * norm_factor).cast[output_dtype]()
                        * (paired_gamma_val + weight_offset).cast[
                            output_dtype
                        ]()
                    ).cast[accum_type]()

                var rotated: SIMD[accum_type, simd_width]
                if offset < half_cols:
                    rotated = -paired_norm_val
                else:
                    rotated = paired_norm_val

                var cos_val = cos_fn[simd_width, alignment=align](
                    row, offset
                ).cast[accum_type]()
                var sin_val = sin_fn[simd_width, alignment=align](
                    row, offset
                ).cast[accum_type]()
                var result = (norm_val * cos_val + rotated * sin_val).cast[
                    output_dtype
                ]()
                output_fn[alignment=out_align](row, offset, result)


def rms_norm_rope_gpu[
    input_dtype: DType,
    output_dtype: DType,
    cos_sin_dtype: DType,
    rank: Int,
    //,
    input_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[input_dtype, width],
    cos_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[cos_sin_dtype, width],
    sin_fn: def[width: Int, rank: Int, alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[cos_sin_dtype, width],
    output_fn: def[width: Int, alignment: Int](
        IndexList[rank], SIMD[output_dtype, width]
    ) capturing -> None,
    multiply_before_cast: Bool,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    shape: IndexList[rank, ...],
    gamma: TileTensor[mut=False, input_dtype, ...],
    epsilon: Scalar[input_dtype],
    weight_offset: Scalar[input_dtype],
    cos_vals: TileTensor[mut=False, cos_sin_dtype, ...],
    sin_vals: TileTensor[mut=False, cos_sin_dtype, ...],
    ctx: DeviceContext,
) raises:
    """Fused RMS normalization followed by Rotary Position Embedding (RoPE) for GPU.

    Computes:
      normed = rms_norm(input, gamma, epsilon, weight_offset)
      x1, x2 = split(normed, axis=-1)          # halves along last dim
      rotated = concat(-x2, x1, axis=-1)
      output = normed * cos_vals + rotated * sin_vals

    The last dimension must be a known even number.
    """
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"
    if rank == 0:
        return

    var cols = Int(gamma.dim[0]())
    if cols == 0:
        return

    var rows = shape.flattened_length() // cols

    @parameter
    @always_inline
    def output_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int, val: SIMD[output_dtype, simd_width]) -> None:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        output_fn[simd_width, alignment](indices.canonicalize(), val)

    @parameter
    @always_inline
    def input_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[input_dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_fn[simd_width, rank, alignment](indices.canonicalize())

    @parameter
    @always_inline
    def cos_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[cos_sin_dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return cos_fn[simd_width, rank, alignment](indices.canonicalize())

    @parameter
    @always_inline
    def sin_fn_2d[
        simd_width: Int, alignment: Int
    ](row: Int, col: Int) -> SIMD[cos_sin_dtype, simd_width]:
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return sin_fn[simd_width, rank, alignment](indices.canonicalize())

    comptime simd_width = simd_width_of[input_dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    var shared_mem_size = (
        ceildiv(cols, simd_width) * simd_width * size_of[input_dtype]()
    )

    # The paired SIMD load from shared memory (at col ± half_cols) is aligned
    # only if half_cols is a multiple of simd_width, i.e. cols % (2*simd_width) == 0.
    if (
        input_dtype == DType.bfloat16
        and cols <= 128
        and cols % (2 * simd_width) == 0
    ):
        # Two rows per warp: pack two half-warps per warp, each handling one
        # row.  Mirrors the rms_norm_gpu_warp_tiling_128 optimization.
        # warps_per_block=2 was found optimal empirically (matches rms_norm_gpu).
        comptime warps_per_block = 2
        block_dim = warps_per_block * WARP_SIZE
        grid_dim = ceildiv(rows, warps_per_block * 2)
        shared_mem_size = warps_per_block * 2 * cols * size_of[input_dtype]()

        comptime kernel = _rms_norm_rope_gpu_warp_tiling_128[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            simd_width,
            warps_per_block,
            input_fn_2d,
            cos_fn_2d,
            sin_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            epsilon,
            weight_offset,
            rows,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
            shared_mem_bytes=shared_mem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(shared_mem_size)
            ),
        )
    elif cols % (2 * simd_width) == 0 and cols <= (
        WARP_SIZE * simd_width * max_warps_per_block
    ):
        # Warp-tiling path: all elements fit in registers; use shmem only for
        # the RoPE cross-half access.
        comptime kernel = _rms_norm_rope_gpu_warp_tiling[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            simd_width,
            max_warps_per_block,
            input_fn_2d,
            cos_fn_2d,
            sin_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            epsilon,
            weight_offset,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
            shared_mem_bytes=shared_mem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(shared_mem_size)
            ),
        )
    elif cols % (2 * simd_width) == 0:
        # Block path, aligned: large rows that exceed warp-tiling capacity.
        # No dynamic shared memory needed: paired elements are re-loaded from
        # global memory and normalized on-the-fly.
        comptime kernel = _rms_norm_rope_gpu_block[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            simd_width,
            max_warps_per_block,
            input_fn_2d,
            cos_fn_2d,
            sin_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            epsilon,
            weight_offset,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # Block path, not aligned: fall back to simd_width=1.
        comptime kernel = _rms_norm_rope_gpu_block[
            mut=gamma.mut,
            LayoutType=gamma.LayoutType,
            origin=gamma.origin,
            1,
            max_warps_per_block,
            input_fn_2d,
            cos_fn_2d,
            sin_fn_2d,
            output_fn_2d,
            multiply_before_cast=multiply_before_cast,
        ]
        ctx.enqueue_function[kernel](
            gamma,
            epsilon,
            weight_offset,
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


@always_inline
def rms_norm[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int](Coord) capturing -> SIMD[dtype, width],
    output_0_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: Coord,
    gamma: TileTensor[mut=False, dtype, ...],
    epsilon: Scalar[dtype],
    weight_offset: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    # Boundary `IndexList` -> `Coord` migration (softmax PR #88203 form).
    # `input_0_fn` and `shape` are `Coord`; `output_0_fn` keeps its n-D
    # `IndexList` form for source compatibility and is wrapped to the
    # `Coord`-form `_rms_norm_impl` expects. `shape_il` materializes the
    # runtime `IndexList` once for the trace string.
    comptime assert gamma.flat_rank == 1, "gamma must have rank 1"

    @always_inline
    @parameter
    def output_fn_wrapper[
        width: SIMDSize, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](
            rebind[IndexList[rank]](coord_to_index_list(coords)), val
        )

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg(
            "input",
            rebind[IndexList[rank]](coord_to_index_list(shape)),
            dtype,
        )

    with Trace[TraceLevel.OP, target=target](
        "rms_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        _rms_norm_impl[
            dtype,
            rank,
            input_0_fn,
            output_fn_wrapper,
            target=target,
            multiply_before_cast=multiply_before_cast,
        ](shape, gamma, epsilon, weight_offset, ctx)


# ===----------------------------------------------------------------------=== #
# Row mean-of-squares: dispatcher + CPU/GPU entry points
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares_gpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    output_fn: def(row: Int, val: Scalar[out_dtype]) capturing -> None,
    pdl_level: PDLLevel = PDLLevel.ON,
](rows: Int, cols: Int, ctx: DeviceContext) raises:
    """Launches the GPU mean-of-squares reduction: one block per row.

    SM100 (B200) primary target; uses only `block_reduce` so it is portable.
    """
    if rows == 0 or cols == 0:
        return

    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    # One block per row keeps the grid tiny for small-M decode (e.g. M=16).
    var grid_dim = rows

    if cols % simd_width == 0:
        # Vectorized loads: clamp threads to what is needed for the row, but
        # never exceed the device's max threads-per-block.
        var block_dim = min(
            ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = row_mean_of_squares_gpu_block[
            simd_width, max_warps_per_block, input_fn, output_fn
        ]
        ctx.enqueue_function[kernel](
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # General N (incl. non-multiple of vector width): scalar loads.
        var block_dim = min(
            ceildiv(cols, WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = row_mean_of_squares_gpu_block[
            1, max_warps_per_block, input_fn, output_fn
        ]
        ctx.enqueue_function[kernel](
            cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def row_mean_of_squares_cpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    input_fn: def[width: Int](row: Int, col: Int) capturing -> SIMD[
        in_dtype, width
    ],
    output_fn: def(row: Int, val: Scalar[out_dtype]) capturing -> None,
](rows: Int, cols: Int):
    """Naive CPU reference path (also used as a correctness oracle)."""
    comptime accum_type = get_accum_type[in_dtype]()
    for r in range(rows):
        var acc = Scalar[accum_type](0)
        for c in range(cols):
            var v = input_fn[1](r, c)[0].cast[accum_type]()
            acc += v * v
        output_fn(r, (acc / Scalar[accum_type](cols)).cast[out_dtype]())


def row_mean_of_squares[
    in_dtype: DType,
    out_dtype: DType,
    rank: Int,
    //,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        in_dtype, width
    ],
    output_0_fn: def(row: Int, val: Scalar[out_dtype]) capturing -> None,
    /,
    target: StaticString = "cpu",
](shape: IndexList[rank], ctx: DeviceContext) raises:
    """Per-row mean of squares over the last axis, accumulated in `accum_type`.

    For input flattened to `[M, N]`, computes `out[m] = sum_n(x[m,n]^2) / N`
    and invokes `output_0_fn(m, ...)` once per row with an `out_dtype` scalar.

    Parameters:
        in_dtype: Element type of the input (e.g. `bfloat16` or `float32`).
        out_dtype: Element type of the per-row result (typically `float32`).
        rank: Rank of the logical input shape.
        input_0_fn: Loads `width` contiguous input elements at a 2D `[row, col]`
            position re-expressed as an n-D index.
        output_0_fn: Receives `(row, value)` once per row.
        target: `"cpu"` or a GPU target string.

    Args:
        shape: Logical input shape. Reduction runs over the last axis.
        ctx: Device context (ignored on CPU).
    """
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    @always_inline
    @parameter
    def input_fn_2d[width: Int](row: Int, col: Int) -> SIMD[in_dtype, width]:
        # Translate a 2D [row, col] index back to the original n-D tensor.
        var indices = _get_start_indices_of_nth_subvolume(row, shape)
        indices[rank - 1] = col
        return input_0_fn[width](indices.canonicalize())

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("input", shape, in_dtype)

    with Trace[TraceLevel.OP, target=target](
        "row_mean_of_squares",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        if shape.flattened_length() == 0:
            return

        comptime if is_cpu[target]():
            row_mean_of_squares_cpu[input_fn_2d, output_0_fn](rows, cols)
        elif is_gpu[target]():
            row_mean_of_squares_gpu[input_fn_2d, output_0_fn](rows, cols, ctx)
        else:
            comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# Fused Q/K row mean-of-squares: dispatcher + CPU/GPU entry points
# ===----------------------------------------------------------------------=== #


def row_mean_of_squares_qk_gpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    output: TileTensor[mut=True, out_dtype, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Launches the fused Q/K mean-of-squares reduction: one launch, grid (rows, 2).

    `block_idx.y` selects Q (0) or K (1). Block dim is sized for the wider of
    the two operands; the narrower operand simply leaves trailing threads idle.
    """
    if rows == 0 or (q_cols == 0 and k_cols == 0):
        return

    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    # 2D grid: x = row, y = operand (0 = Q, 1 = K). One block per (row, operand)
    # keeps the grid tiny for small-M decode (e.g. M=16 -> 32 blocks).
    var grid_dim = (rows, 2)
    var max_cols = max(q_cols, k_cols)

    if q_cols % simd_width == 0 and k_cols % simd_width == 0:
        # Vectorized loads; size threads for the wider operand.
        var block_dim = min(
            ceildiv(ceildiv(max_cols, simd_width), WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = row_mean_of_squares_qk_gpu_block[
            in_dtype=q.dtype,
            out_dtype=output.dtype,
            out_mut=output.mut,
            out_layout=output.LayoutType,
            out_origin=output.origin,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            simd_width=simd_width,
            max_warps_per_block=max_warps_per_block,
        ]
        ctx.enqueue_function[kernel](
            output,
            q,
            k,
            q_cols,
            k_cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # General N (incl. non-multiple of vector width): scalar loads.
        var block_dim = min(
            ceildiv(max_cols, WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = row_mean_of_squares_qk_gpu_block[
            in_dtype=q.dtype,
            out_dtype=output.dtype,
            out_mut=output.mut,
            out_layout=output.LayoutType,
            out_origin=output.origin,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            simd_width=1,
            max_warps_per_block=max_warps_per_block,
        ]
        ctx.enqueue_function[kernel](
            output,
            q,
            k,
            q_cols,
            k_cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def row_mean_of_squares_qk_cpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
](
    output: TileTensor[mut=True, out_dtype, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
):
    """Naive CPU reference path (also used as a correctness oracle)."""
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime accum_type = get_accum_type[in_dtype]()
    for r in range(rows):
        var accq = Scalar[accum_type](0)
        for c in range(q_cols):
            var v = q.load[width=1](Coord(Index(r, c)))[0].cast[accum_type]()
            accq += v * v
        output.store[width=1](
            Coord(Index(r, 0)),
            (accq / Scalar[accum_type](q_cols)).cast[out_dtype](),
        )

        var acck = Scalar[accum_type](0)
        for c in range(k_cols):
            var v = k.load[width=1](Coord(Index(r, c)))[0].cast[accum_type]()
            acck += v * v
        output.store[width=1](
            Coord(Index(r, 1)),
            (acck / Scalar[accum_type](k_cols)).cast[out_dtype](),
        )


def row_mean_of_squares_qk[
    in_dtype: DType,
    out_dtype: DType,
    //,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, out_dtype, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Fused per-row mean of squares for two operands Q and K.

    Computes `out[m, 0] = sum_n(q[m,n]^2) / q_cols` and
    `out[m, 1] = sum_n(k[m,n]^2) / k_cols`, accumulated in `accum_type`. Q and K
    share the leading rows dimension but may have different column counts. This
    is a single-launch fusion of two `row_mean_of_squares` reductions, used for
    cross-head QK-RMSNorm statistics under tensor parallelism.

    All operands (`q [M, Nq]`, `k [M, Nk]`, and the `[M, 2]` output) are passed
    directly as `TileTensor`s and loaded/stored in-kernel.

    Parameters:
        in_dtype: Element type of both inputs (e.g. `bfloat16` or `float32`).
        out_dtype: Element type of the per-row result (typically `float32`).
        target: `"cpu"` or a GPU target string.

    Args:
        output: Per-row result, shape `[M, 2]` (col 0 = mean(q^2),
            col 1 = mean(k^2)).
        q: Q activations, shape `[M, Nq]`.
        k: K activations, shape `[M, Nk]`.
        rows: Shared leading dimension of Q and K.
        q_cols: Number of columns reduced for Q.
        k_cols: Number of columns reduced for K.
        ctx: Device context (ignored on CPU).
    """

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("qk", IndexList[2](rows, q_cols + k_cols), in_dtype)

    with Trace[TraceLevel.OP, target=target](
        "row_mean_of_squares_qk",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        if rows == 0:
            return

        comptime if is_cpu[target]():
            row_mean_of_squares_qk_cpu(output, q, k, rows, q_cols, k_cols)
        elif is_gpu[target]():
            row_mean_of_squares_qk_gpu(output, q, k, rows, q_cols, k_cols, ctx)
        else:
            comptime assert False, "unsupported target " + target


# ===----------------------------------------------------------------------=== #
# Fused Q/K RMSNorm apply: kernel + CPU/GPU entry points + dispatcher
# ===----------------------------------------------------------------------=== #
#
# SM100 (B200) primary target; portable to SM90 / CDNA4 / Apple (pure
# elementwise apply, no arch-specific intrinsics, no block reduction).
#
# Given the already-reduced per-row statistics `qk_var` of shape `[M, 2]`
# (col 0 = mean(q^2), col 1 = mean(k^2), float32), and per-column float32
# scales `gamma_q [Nq]` / `gamma_k [Nk]`, this applies the QK-RMSNorm scale to
# Q `[M, Nq]` and K `[M, Nk]` in a SINGLE launch:
#
#     rs_q = rsqrt(qk_var[m, 0] + epsilon)            # float32
#     q_out[m, c] = cast( (cast(q[m,c], f32) * rs_q) * gamma_q[c], out_dtype )
#     rs_k = rsqrt(qk_var[m, 1] + epsilon)
#     k_out[m, c] = cast( (cast(k[m,c], f32) * rs_k) * gamma_k[c], out_dtype )
#
# The grouping is `((x_f32 * rs) * gamma)` then cast (NOT `x_f32 * (rs*gamma)`)
# to bit-match the unfused graph it replaces. This fuses the ~7 tiny
# elementwise/View kernels of the QK-norm apply chain into one launch, keeping
# the grid tiny for small-M decode (grid = (rows, 2); e.g. M=16 -> 32 blocks).


@__name(t"apply_qk_rms_norm_gpu_block_{in_dtype}_{out_dtype}")
def apply_qk_rms_norm_gpu_block[
    in_dtype: DType,
    out_dtype: DType,
    q_out_mut: Bool,
    q_out_layout: TensorLayout,
    q_out_origin: Origin[mut=q_out_mut],
    k_out_mut: Bool,
    k_out_layout: TensorLayout,
    k_out_origin: Origin[mut=k_out_mut],
    gamma_q_mut: Bool,
    gamma_q_layout: TensorLayout,
    gamma_q_origin: Origin[mut=gamma_q_mut],
    gamma_k_mut: Bool,
    gamma_k_layout: TensorLayout,
    gamma_k_origin: Origin[mut=gamma_k_mut],
    var_mut: Bool,
    var_layout: TensorLayout,
    var_origin: Origin[mut=var_mut],
    q_layout: TensorLayout,
    q_origin: Origin,
    k_layout: TensorLayout,
    k_origin: Origin,
    //,
    simd_width: Int,
](
    q_out: TileTensor[out_dtype, q_out_layout, q_out_origin],
    k_out: TileTensor[out_dtype, k_out_layout, k_out_origin],
    gamma_q: TileTensor[DType.float32, gamma_q_layout, gamma_q_origin],
    gamma_k: TileTensor[DType.float32, gamma_k_layout, gamma_k_origin],
    qk_var: TileTensor[DType.float32, var_layout, var_origin],
    q: TileTensor[in_dtype, q_layout, q_origin],
    k: TileTensor[in_dtype, k_layout, k_origin],
    epsilon: Scalar[DType.float32],
    q_cols: Int,
    k_cols: Int,
) where (q_out_mut and k_out_mut):
    """Fused per-element QK-RMSNorm apply for Q and K in a single launch.

    The grid is 2D: `block_idx.x` selects the row and `block_idx.y` selects the
    operand (0 = Q, 1 = K). Each block owns one (row, operand) and threads
    grid-stride across that operand's columns, applying `((x * rs) * gamma)`.
    All operands (`q [M, Nq]`, `k [M, Nk]`, `gamma_q [Nq]`, `gamma_k [Nk]`,
    `qk_var [M, 2]`, and the outputs `q_out [M, Nq]` / `k_out [M, Nk]`) are
    loaded/stored directly from their `TileTensor`s, matching the in-file
    rms_norm `gamma.load[...]` idiom.
    """
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    comptime align = align_of[SIMD[DType.float32, simd_width]]()

    var tid = thread_idx.x
    var row = block_idx.x
    # block_idx.y is uniform across the block, so this branch never diverges.
    var is_k = block_idx.y == 1
    var num_cols = k_cols if is_k else q_cols

    with PDL():
        # rsqrt of the (already cross-rank reduced) per-row mean of squares.
        var rs = rsqrt(
            qk_var.load[width=1](Coord(Index(Int(row), Int(block_idx.y))))
            + epsilon
        )

        # Each block owns a single (row, operand); threads grid-stride the cols.
        for x in range(ceildiv(ceildiv(num_cols, simd_width), block_dim.x)):
            var offset = x * block_dim.x * simd_width + tid * simd_width
            if offset < num_cols:
                if is_k:
                    var xf = k.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    ).cast[DType.float32]()
                    var g = gamma_k.load[width=simd_width, alignment=align](
                        Coord(offset)
                    )
                    k_out.store[width=simd_width](
                        Coord(Index(Int(row), offset)),
                        ((xf * rs) * g).cast[out_dtype](),
                    )
                else:
                    var xf = q.load[width=simd_width](
                        Coord(Index(Int(row), offset))
                    ).cast[DType.float32]()
                    var g = gamma_q.load[width=simd_width, alignment=align](
                        Coord(offset)
                    )
                    q_out.store[width=simd_width](
                        Coord(Index(Int(row), offset)),
                        ((xf * rs) * g).cast[out_dtype](),
                    )


def apply_qk_rms_norm_gpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
    pdl_level: PDLLevel = PDLLevel.ON,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, DType.float32, ...],
    gamma_k: TileTensor[mut=False, DType.float32, ...],
    qk_var: TileTensor[mut=False, DType.float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[DType.float32],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Launches the fused Q/K RMSNorm apply: one launch, grid (rows, 2).

    `block_idx.y` selects Q (0) or K (1). Block dim is sized for the wider of
    the two operands; the narrower operand simply leaves trailing threads idle.
    """
    if rows == 0 or (q_cols == 0 and k_cols == 0):
        return

    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE

    # 2D grid: x = row, y = operand (0 = Q, 1 = K). One block per (row, operand)
    # keeps the grid tiny for small-M decode (e.g. M=16 -> 32 blocks).
    var grid_dim = (rows, 2)
    var max_cols = max(q_cols, k_cols)

    if q_cols % simd_width == 0 and k_cols % simd_width == 0:
        # Vectorized loads/stores; size threads for the wider operand.
        var block_dim = min(
            ceildiv(ceildiv(max_cols, simd_width), WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            simd_width=simd_width,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            q_cols,
            k_cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )
    else:
        # General N (incl. non-multiple of vector width): scalar loads/stores.
        var block_dim = min(
            ceildiv(max_cols, WARP_SIZE) * WARP_SIZE,
            WARP_SIZE * max_warps_per_block,
        )
        comptime kernel = apply_qk_rms_norm_gpu_block[
            in_dtype=q.dtype,
            out_dtype=out_dtype,
            q_out_mut=q_out.mut,
            q_out_layout=q_out.LayoutType,
            q_out_origin=q_out.origin,
            k_out_mut=k_out.mut,
            k_out_layout=k_out.LayoutType,
            k_out_origin=k_out.origin,
            gamma_q_mut=gamma_q.mut,
            gamma_q_layout=gamma_q.LayoutType,
            gamma_q_origin=gamma_q.origin,
            gamma_k_mut=gamma_k.mut,
            gamma_k_layout=gamma_k.LayoutType,
            gamma_k_origin=gamma_k.origin,
            var_mut=qk_var.mut,
            var_layout=qk_var.LayoutType,
            var_origin=qk_var.origin,
            q_layout=q.LayoutType,
            q_origin=q.origin,
            k_layout=k.LayoutType,
            k_origin=k.origin,
            simd_width=1,
        ]
        ctx.enqueue_function[kernel](
            q_out,
            k_out,
            gamma_q,
            gamma_k,
            qk_var,
            q,
            k,
            epsilon,
            q_cols,
            k_cols,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(pdl_level),
        )


def apply_qk_rms_norm_cpu[
    in_dtype: DType,
    out_dtype: DType,
    //,
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, DType.float32, ...],
    gamma_k: TileTensor[mut=False, DType.float32, ...],
    qk_var: TileTensor[mut=False, DType.float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[DType.float32],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
):
    """Naive CPU reference path (also used as a correctness oracle)."""
    comptime assert q.flat_rank == 2, "q must have rank 2"
    comptime assert k.flat_rank == 2, "k must have rank 2"
    for r in range(rows):
        var rs_q = rsqrt(qk_var.load[width=1](Coord(Index(r, 0))) + epsilon)
        for c in range(q_cols):
            var xf = q.load[width=1](Coord(Index(r, c)))[0].cast[
                DType.float32
            ]()
            var g = gamma_q.load[width=1](Coord(c))
            q_out.store[width=1](
                Coord(Index(r, c)), ((xf * rs_q) * g).cast[out_dtype]()
            )

        var rs_k = rsqrt(qk_var.load[width=1](Coord(Index(r, 1))) + epsilon)
        for c in range(k_cols):
            var xf = k.load[width=1](Coord(Index(r, c)))[0].cast[
                DType.float32
            ]()
            var g = gamma_k.load[width=1](Coord(c))
            k_out.store[width=1](
                Coord(Index(r, c)), ((xf * rs_k) * g).cast[out_dtype]()
            )


def apply_qk_rms_norm[
    in_dtype: DType,
    out_dtype: DType,
    //,
    target: StaticString = "cpu",
](
    q_out: TileTensor[mut=True, out_dtype, ...],
    k_out: TileTensor[mut=True, out_dtype, ...],
    gamma_q: TileTensor[mut=False, DType.float32, ...],
    gamma_k: TileTensor[mut=False, DType.float32, ...],
    qk_var: TileTensor[mut=False, DType.float32, ...],
    q: TileTensor[mut=False, in_dtype, ...],
    k: TileTensor[mut=False, in_dtype, ...],
    epsilon: Scalar[DType.float32],
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    ctx: DeviceContext,
) raises:
    """Fused per-element QK-RMSNorm apply for two operands Q and K.

    Given the already cross-rank-reduced per-row statistics `qk_var [M, 2]`
    (col 0 = mean(q^2), col 1 = mean(k^2), float32) and per-column float32
    scales `gamma_q [Nq]` / `gamma_k [Nk]`, applies in a single launch:

    `q_out[m,c] = cast((cast(q[m,c], f32) * rsqrt(qk_var[m,0] + eps)) * gamma_q[c], out_dtype)`
    and likewise for K with column 1. The grouping `((x * rs) * gamma)` then
    cast matches the unfused graph this replaces for bit-accuracy. This fuses
    the QK-RMSNorm apply chain (~7 tiny elementwise/View kernels) into one
    launch, used for cross-head QK-RMSNorm under tensor parallelism.

    All operands (`q` / `k` activations, the outputs `q_out` / `k_out`, and the
    `gamma_q` / `gamma_k` / `qk_var` inputs) are passed directly as
    `TileTensor`s and loaded/stored in-kernel, matching the in-file rms_norm
    `gamma.load[...]` idiom.

    Parameters:
        in_dtype: Element type of both activation inputs (`bfloat16` or
            `float32`).
        out_dtype: Element type of the outputs (typically equal to `in_dtype`).
        target: `"cpu"` or a GPU target string.

    Args:
        q_out: Scaled Q output, shape `[M, Nq]`.
        k_out: Scaled K output, shape `[M, Nk]`.
        gamma_q: Per-column float32 Q scales, shape `[Nq]`.
        gamma_k: Per-column float32 K scales, shape `[Nk]`.
        qk_var: Per-row float32 statistics, shape `[M, 2]` (col 0 = mean(q^2),
            col 1 = mean(k^2)).
        q: Q activations, shape `[M, Nq]`.
        k: K activations, shape `[M, Nk]`.
        epsilon: RMSNorm epsilon, added to the variance before `rsqrt`.
        rows: Shared leading dimension of Q and K.
        q_cols: Number of columns of Q.
        k_cols: Number of columns of K.
        ctx: Device context (ignored on CPU).
    """

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("qk", IndexList[2](rows, q_cols + k_cols), in_dtype)

    with Trace[TraceLevel.OP, target=target](
        "apply_qk_rms_norm",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
    ):
        if rows == 0:
            return

        comptime if is_cpu[target]():
            apply_qk_rms_norm_cpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
            )
        elif is_gpu[target]():
            apply_qk_rms_norm_gpu(
                q_out,
                k_out,
                gamma_q,
                gamma_k,
                qk_var,
                q,
                k,
                epsilon,
                rows,
                q_cols,
                k_cols,
                ctx,
            )
        else:
            comptime assert False, "unsupported target " + target


def _rms_norm_fused_residual_add_impl[
    dtype: DType,
    rank: Int,
    input_0_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    input_1_fn: def[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    output_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDSize, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[mut=False, dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[mut=False, dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    # Note: we only support reduction along the last dimension
    if Int(gamma1.layout.shape[0]().value()) != shape[rank - 1]:
        raise Error(
            "Gamma1 size "
            + String(gamma1.layout.shape[0]().value())
            + " does not match dimension of reduction "
            + String(shape[rank - 1])
            + "."
        )

    if Int(gamma2.layout.shape[0]().value()) != shape[rank - 1]:
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
            ctx,
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
    output_0_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        idx: IndexList[rank], val: SIMD[dtype, width]
    ) capturing -> None,
    output_residual_fn: def[width: SIMDSize, rank: Int, alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    /,
    target: StaticString = "cpu",
    multiply_before_cast: Bool = True,
](
    shape: IndexList[rank],
    gamma1: TileTensor[mut=False, dtype, ...],
    epsilon1: Scalar[dtype],
    weight_offset1: Scalar[dtype],
    gamma2: TileTensor[mut=False, dtype, ...],
    epsilon2: Scalar[dtype],
    weight_offset2: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime assert gamma1.rank == 1, "gamma1 must have rank 1"
    comptime assert gamma2.rank == 1, "gamma2 must have rank 1"

    @always_inline
    @parameter
    def output_fn_wrapper[
        width: SIMDSize, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_0_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    def output_residual_fn_wrapper[
        width: SIMDSize, alignment: Int
    ](idx: IndexList[rank], val: SIMD[dtype, width]) -> None:
        output_residual_fn[width, rank, alignment](idx, val)

    @always_inline
    @parameter
    def description_fn() -> String:
        return trace_arg("input", shape, dtype)

    with Trace[TraceLevel.OP, target=target](
        "rms_norm_fused_residual_add",
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=Int(ctx.id()),
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


@__name(t"group_norm_gpu_warp_tiling_{dtype}")
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

    var idx = thread_idx.x * simd_width

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
            vec_data = input_fn[simd_width](row, idx).cast[accum_type]()

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
            var g = umod(row, num_groups)
            var c_base = g * channels_per_group
            var norm_val = SIMD[accum_type, simd_width]()
            for i in range(simd_width):
                var offset = (idx + i) // spatial
                var c = c_base + offset
                var gamma_val = gamma_fn[1](Index(c))
                var beta_val = beta_fn[1](Index(c))
                norm_val[i] = (
                    vec_data[i] - row_mean
                ) * norm_factor * gamma_val.cast[accum_type]() + beta_val.cast[
                    accum_type
                ]()

            var output_idx = output.layout(Coord(row, idx))
            output.raw_store[alignment=align](
                output_idx, norm_val.cast[dtype]()
            )


@__name(t"group_norm_gpu_block_{dtype}")
def group_norm_gpu_block[
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

                comptime for i in range(simd_width):
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

                var g = umod(row, num_groups)
                var c_base = g * channels_per_group

                var norm_val = SIMD[accum_type, simd_width]()
                for i in range(simd_width):
                    var offset_c = (offset + i) // spatial
                    var c = c_base + offset_c
                    var gamma_val = gamma_fn[1](Index(c))
                    var beta_val = beta_fn[1](Index(c))
                    norm_val[i] = (
                        vec_data[i] - row_mean
                    ) * norm_factor * gamma_val.cast[
                        accum_type
                    ]() + beta_val.cast[
                        accum_type
                    ]()

                var output_row_offset = output.layout(Coord(row, offset))
                output.raw_store[alignment=align](
                    output_row_offset, norm_val.cast[dtype]()
                )


@__name(t"group_norm_gpu_multi_block_stats_{dtype}")
def group_norm_gpu_multi_block_stats[
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
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

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, num_splits)
    var tid = thread_idx.x

    # Compute chunk boundaries (each split handles a contiguous chunk,
    # aligned to simd_width).
    var total_simd_elems = group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

    with PDL():
        var thread_mean = Scalar[accum_type]()
        var thread_m2 = Scalar[accum_type]()
        var thread_count = Scalar[accum_type]()

        for x in range(chunk_iters):
            var offset = (
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()
                comptime for i in range(simd_width):
                    welford_update(
                        vec_data[i],
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
            stats.store(Coord(base_idx), row_mean)
            stats.store(Coord(base_idx + 1), row_m2)
            stats.store(Coord(base_idx + 2), row_count)


@__name(t"group_norm_gpu_multi_block_norm_{dtype}")
def group_norm_gpu_multi_block_norm[
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    StatsLayoutType: TensorLayout,
    stats_origin: MutOrigin,
    //,
    dtype: DType,
    simd_width: Int,
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
    comptime align = align_of[SIMD[dtype, simd_width]]()
    comptime accum_type = get_accum_type[dtype]()

    var block_id = block_idx.x
    var row, split_id = divmod(block_id, num_splits)
    var tid = thread_idx.x

    # Same chunk boundaries as stats kernel.
    var total_simd_elems = group_size // simd_width
    var chunk_simd_size = ceildiv(total_simd_elems, num_splits)
    var chunk_start = split_id * chunk_simd_size * simd_width
    var chunk_end = min(chunk_start + chunk_simd_size * simd_width, group_size)
    var chunk_iters = ceildiv(chunk_simd_size, block_dim.x)

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
                stats.load[width=1](Coord(base_idx)),
                stats.load[width=1](Coord(base_idx + 1)),
                stats.load[width=1](Coord(base_idx + 2)),
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
                chunk_start + x * block_dim.x * simd_width + tid * simd_width
            )
            if offset < chunk_end:
                var vec_data = input_fn[simd_width](row, offset).cast[
                    accum_type
                ]()

                var norm_val = SIMD[accum_type, simd_width]()

                # Vectorized gamma/beta: when all SIMD elements share the
                # same channel (common case for large spatial dims), load
                # gamma/beta once and broadcast.
                var c_first = c_base + offset // spatial
                var c_last = c_base + (offset + simd_width - 1) // spatial
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
                        var c = c_base + (offset + i) // spatial
                        var gamma_val = gamma_fn[1](Index(c))
                        var beta_val = beta_fn[1](Index(c))
                        norm_val[i] = (
                            vec_data[i] - row_mean
                        ) * norm_factor * gamma_val.cast[
                            accum_type
                        ]() + beta_val.cast[
                            accum_type
                        ]()

                var output_row_offset = output.layout(Coord(row, offset))
                output.raw_store[alignment=align](
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

    # Zero-sized input (e.g. a ``(B, C, 0, 0)`` tensor flowing through a
    # diffusion VAE encoder for the text-to-image placeholder): nothing
    # to normalize.  The output buffer is pre-allocated zero-element by
    # the caller and the kernel's ``num_cols < simd_width`` misalignment
    # check below would otherwise abort.  Early-return is correct because
    # mean/var of an empty group has no defined value and the downstream
    # readers also have zero spatial dims.
    if num_rows == OutputLinearIdxType(0) or num_cols == OutputLinearIdxType(0):
        return

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

            # Guard against c_offset boundary straddling.  A view-fused
            # NHWC→NCHW transpose generates a strided_load(stride=C) for
            # the W-dimension.  That load is correct within a single c_offset
            # region (consecutive-w with stride C maps to adjacent NHWC
            # addresses), but it reads wrong elements when the simd_width
            # window crosses a c_offset boundary at a multiple of
            # inner_volume=H*W.  This happens when H*W % simd_width != 0 and
            # the thread's starting column lands near a boundary.  Fall back
            # to element-wise scalar loads in that case so each element's
            # full (n, c, h, w) index is recomputed independently.
            if (col + simd_width - 1) // inner_volume != c_offset:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, hw_i = divmod(cur_col, inner_volume)
                    var h_i, w_i = divmod(hw_i, shape[3])
                    result[i] = input_fn[1, rank](
                        IndexList[rank](
                            n, g * channels_per_group + c_off, h_i, w_i
                        )
                    )[0]
                return result

        elif rank == 3:
            var inner_volume = shape[2]
            var c_offset, l = divmod(col, inner_volume)
            c += c_offset
            indices = IndexList[rank](n, c, l)

            if (col + simd_width - 1) // inner_volume != c_offset:
                var result = SIMD[dtype, simd_width]()
                for i in range(simd_width):
                    var cur_col = col + i
                    var c_off, l_i = divmod(cur_col, inner_volume)
                    result[i] = input_fn[1, rank](
                        IndexList[rank](n, g * channels_per_group + c_off, l_i)
                    )[0]
                return result

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
            ctx.enqueue_function[kernel](
                output_rs,
                epsilon,
                num_groups,
                channels_per_group,
                spatial,
                grid_dim=grid_dim,
                block_dim=block_dim,
                attributes=pdl_launch_attributes(PDLLevel.ON),
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
                    stats_buf,
                    row_major(stats_size),
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
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                ]
                ctx.enqueue_function[stats_kernel](
                    stats,
                    num_splits,
                    group_size,
                    grid_dim=mb_grid_dim,
                    block_dim=mb_block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                # Kernel 2: reduce stats and normalize each chunk.
                comptime norm_kernel = group_norm_gpu_multi_block_norm[
                    OutputLayoutType=output_rs.LayoutType,
                    output_origin=output_rs.origin,
                    StatsLayoutType=stats.LayoutType,
                    stats_origin=stats.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[norm_kernel](
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
                    attributes=pdl_launch_attributes(PDLLevel.ON),
                )

                _ = stats_buf^
            else:
                comptime kernel = group_norm_gpu_block[
                    LayoutType=output_rs.LayoutType,
                    origin=output_rs.origin,
                    dtype=dtype,
                    simd_width=simd_width,
                    input_fn=input_fn_2d,
                    gamma_fn=gamma_fn,
                    beta_fn=beta_fn,
                ]
                ctx.enqueue_function[kernel](
                    output_rs,
                    epsilon,
                    num_groups,
                    channels_per_group,
                    spatial,
                    grid_dim=grid_dim,
                    block_dim=block_dim,
                    attributes=pdl_launch_attributes(PDLLevel.ON),
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
        ctx.enqueue_function[kernel](
            output_rs,
            epsilon,
            num_groups,
            channels_per_group,
            spatial,
            grid_dim=grid_dim,
            block_dim=block_dim,
            attributes=pdl_launch_attributes(PDLLevel.ON),
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
    ctx: DeviceContext,
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
        task_id=Int(ctx.id()),
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
            ctx=ctx,
        )
