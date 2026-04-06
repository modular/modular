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

from std.math import align_up
from std.math.uutils import udivmod

from std.algorithm.reduction import _get_nd_indices_from_flat_index
from std.gpu.primitives.block import broadcast
from std.gpu.host import DeviceContext
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    grid_dim_uint as grid_dim,
    global_idx_uint as global_idx,
    lane_id_uint as lane_id,
    thread_idx_uint as thread_idx,
    warp_id_uint as warp_id,
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
    AddressSpace,
)
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.primitives import warp
from std.gpu.primitives.grid_controls import (
    pdl_launch_attributes,
)  # @doc_hidden
from std.memory import bitcast, stack_allocation
from std.os.atomic import Atomic

from std.utils import IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple
from std.sys import get_defined_bool, get_defined_int
from std.sys.info import simd_width_of


@always_inline
def block_reduce[
    BLOCK_SIZE: Int,
    reduce_fn: def[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width], init: Scalar[dtype]) -> Scalar[dtype]:
    """Performs a block-level reduction of a single SIMD value across all
    threads in a GPU thread block using warp-level primitives and shared memory.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.

    Args:
        val: The per-thread SIMD value to reduce.
        init: The identity value for the reduction.

    Returns:
        The reduced scalar result (valid on thread 0).
    """
    comptime num_reductions = 1

    @always_inline
    @parameter
    def reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_fn(lhs, rhs)

    var val_tup = StaticTuple[SIMD[dtype, simd_width], num_reductions](val)
    var init_tup = StaticTuple[Scalar[dtype], num_reductions](init)

    return block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_wrapper,
        dtype,
        simd_width,
    ](val_tup, init_tup)[0]


@always_inline
def block_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    """Performs a block-level reduction of multiple fused SIMD values across all
    threads in a GPU thread block using warp shuffles and shared memory.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        num_reductions: The number of fused reductions to perform.
        reduce_fn: The binary reduction function, parameterized by reduction
          index.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.

    Args:
        val: The per-thread SIMD values to reduce, one per reduction.
        init: The identity values for each reduction.

    Returns:
        The reduced scalar results (valid on thread 0).
    """
    comptime assert (
        BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"

    @always_inline
    @parameter
    def do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        comptime for i in range(num_reductions):

            @always_inline
            @parameter
            def reduce_wrapper[
                dtype: DType, width: Int
            ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
                dtype, width
            ]:
                return reduce_fn[dtype, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        (BLOCK_SIZE // WARP_SIZE) * num_reductions * simd_width,
        dtype,
        address_space=AddressSpace.SHARED,
    ]()

    var warp = warp_id()

    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:
        comptime for i in range(num_reductions):
            # bank conflict for sub 4 byte data elems
            shared.store(
                (Int(warp) * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var last_accum = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

    if thread_idx.x < (block_dim.x // UInt(WARP_SIZE)):
        comptime for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * Int(lane_id()) + i) * simd_width
            )
    else:
        comptime for i in range(num_reductions):
            last_accum[i] = init[i]

    var result_packed = do_warp_reduce(last_accum)
    var result = StaticTuple[Scalar[dtype], num_reductions]()

    comptime for i in range(num_reductions):
        result[i] = result_packed[i].reduce[
            reduce_fn[dtype, reduction_idx=i, ...]
        ]()

    return result


@always_inline
def block_reduce_warp0_epilogue[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    """Block reduction with a warp-0 epilogue over only live warp partials."""
    comptime assert (
        BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"
    comptime warp_partial_count = BLOCK_SIZE // WARP_SIZE

    @always_inline
    @parameter
    def do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        comptime for i in range(num_reductions):

            @always_inline
            @parameter
            def reduce_wrapper[
                dtype: DType, width: Int
            ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
                dtype, width
            ]:
                return reduce_fn[dtype, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        warp_partial_count * num_reductions * simd_width,
        dtype,
        address_space=AddressSpace.SHARED,
    ]()

    var warp_idx = warp_id()
    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:
        comptime for i in range(num_reductions):
            shared.store(
                (Int(warp_idx) * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var last_accum = StaticTuple[SIMD[dtype, simd_width], num_reductions]()
    comptime for i in range(num_reductions):
        last_accum[i] = init[i]

    if warp_idx == 0 and lane_id() < UInt(warp_partial_count):
        comptime for i in range(num_reductions):
            last_accum[i] = shared.load[width=simd_width](
                (num_reductions * Int(lane_id()) + i) * simd_width
            )

    var result_packed = StaticTuple[SIMD[dtype, simd_width], num_reductions]()
    comptime for i in range(num_reductions):

        @always_inline
        @parameter
        def reduce_wrapper[
            dtype: DType, width: Int
        ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
            dtype, width
        ]:
            return reduce_fn[dtype, width, i](lhs, rhs)

        comptime if warp_partial_count == 1:
            result_packed[i] = last_accum[i]
        elif (
            warp_partial_count == 2
            or warp_partial_count == 4
            or warp_partial_count == 8
            or warp_partial_count == 16
            or warp_partial_count == 32
        ):
            result_packed[i] = warp.lane_group_reduce[
                warp.shuffle_down,
                reduce_wrapper,
                num_lanes=warp_partial_count,
            ](last_accum[i])
        else:
            result_packed[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](
                last_accum[i]
            )

    var result = StaticTuple[Scalar[dtype], num_reductions]()

    comptime for i in range(num_reductions):
        result[i] = result_packed[i].reduce[
            reduce_fn[dtype, reduction_idx=i, ...]
        ]()

    return result


def block_reduce_thread0_serial_epilogue[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
](
    val: StaticTuple[SIMD[dtype, simd_width], num_reductions],
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[dtype], num_reductions]:
    """Block reduction with a thread-0 serial epilogue over live warp partials."""
    comptime assert (
        BLOCK_SIZE % WARP_SIZE == 0
    ), "block size must be a multiple of the warp size"
    comptime warp_partial_count = BLOCK_SIZE // WARP_SIZE

    @always_inline
    @parameter
    def do_warp_reduce(
        val: StaticTuple[SIMD[dtype, simd_width], num_reductions]
    ) -> StaticTuple[SIMD[dtype, simd_width], num_reductions]:
        var result = StaticTuple[SIMD[dtype, simd_width], num_reductions]()

        comptime for i in range(num_reductions):

            @always_inline
            @parameter
            def reduce_wrapper[
                dtype: DType, width: Int
            ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
                dtype, width
            ]:
                return reduce_fn[dtype, width, i](lhs, rhs)

            result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](val[i])

        return result

    var shared = stack_allocation[
        warp_partial_count * num_reductions * simd_width,
        dtype,
        address_space=AddressSpace.SHARED,
    ]()

    var warp_idx = warp_id()
    var warp_accum = do_warp_reduce(val)

    if lane_id() == 0:
        comptime for i in range(num_reductions):
            shared.store(
                (Int(warp_idx) * num_reductions + i) * simd_width,
                warp_accum[i],
            )

    barrier()

    var result = StaticTuple[Scalar[dtype], num_reductions]()
    comptime for i in range(num_reductions):
        result[i] = init[i]

    if thread_idx.x == 0:
        comptime for i in range(num_reductions):
            var packed_accum = SIMD[dtype, simd_width](init[i])
            comptime for partial_idx in range(warp_partial_count):
                packed_accum = reduce_fn[dtype, simd_width, i](
                    shared.load[width=simd_width](
                        (partial_idx * num_reductions + i) * simd_width
                    ),
                    packed_accum,
                )
            result[i] = packed_accum.reduce[
                reduce_fn[dtype, reduction_idx=i, ...]
            ]()

    return result


@always_inline
def row_reduce[
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: Scalar[dtype],
    row_size: Int,
) -> Scalar[accum_type]:
    """Reduces a single row along the given axis using block-level cooperative
    reduction. Delegates to the multi-reduction `row_reduce` overload with
    `num_reductions=1`.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the input elements.
        simd_width: The SIMD vector width.
        rank: The tensor rank.
        accum_type: The accumulator data type (defaults to widened type).

    Args:
        row_coords: The ND coordinates identifying the row.
        axis: The axis along which to reduce.
        init: The identity value for the reduction.
        row_size: The number of elements in the row.

    Returns:
        The reduced scalar result for the row.
    """
    comptime num_reductions = 1

    @always_inline
    @parameter
    def reduce_wrapper[
        dtype: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            reduction_idx < num_reductions
        ), "invalid reduction index"
        return reduce_fn(lhs, rhs)

    var init_tup = StaticTuple[Scalar[dtype], num_reductions](init)

    return row_reduce[
        BLOCK_SIZE,
        num_reductions,
        input_fn,
        reduce_wrapper,
        dtype,
        simd_width,
        rank,
        accum_type=accum_type,
    ](row_coords, axis, init_tup, row_size)[0]


@always_inline
def row_reduce[
    BLOCK_SIZE: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    row_size: Int,
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row along the given axis with multiple fused reductions using
    cooperative SIMD-width reads across threads, a `block_reduce`, and scalar
    tail handling.

    Parameters:
        BLOCK_SIZE: The number of threads per block.
        num_reductions: The number of fused reductions to perform.
        input_fn: The lambda to load input elements.
        reduce_fn: The binary reduction function parameterized by reduction
          index.
        dtype: The data type of the input elements.
        simd_width: The SIMD vector width.
        rank: The tensor rank.
        accum_type: The accumulator data type (defaults to widened type).

    Args:
        row_coords: The ND coordinates identifying the row.
        axis: The axis along which to reduce.
        init: The identity values for each reduction.
        row_size: The number of elements in the row.

    Returns:
        The reduced scalar results, one per fused reduction.
    """
    var num_tail_values = row_size % simd_width
    var rounded_row_size = row_size - num_tail_values
    var row_size_padded = align_up(row_size // simd_width, BLOCK_SIZE)

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var tid: UInt = thread_idx.x
    for offset_in_row in range(0, row_size_padded, BLOCK_SIZE):
        var idx_in_padded_row = (tid + UInt(offset_in_row)) * UInt(simd_width)

        if idx_in_padded_row >= UInt(rounded_row_size):
            break

        row_coords[axis] = Int(idx_in_padded_row)
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime for i in range(num_reductions):
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]
    var scalar_accum = block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)

    # handle trailing values
    for idx_in_padded_row in range(rounded_row_size, row_size):
        row_coords[axis] = idx_in_padded_row
        var val = input_fn[dtype, 1, rank](row_coords).cast[accum_type]()

        comptime for i in range(num_reductions):
            scalar_accum[i] = reduce_fn[accum_type, 1, i](val, scalar_accum[i])

    return scalar_accum


def row_reduce_fixed_turns[
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row with a compile-time fixed number of SIMD turns.

    This keeps the existing scalarized width-1 `block_reduce` lowering, but
    removes the padded loop and scalar tail bookkeeping when the row geometry
    is known exactly at dispatch time.
    """
    comptime assert FIXED_TURNS > 0, "fixed-turn reductions must do work"

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var base_idx = Int(thread_idx.x) * simd_width
    comptime for turn in range(FIXED_TURNS):
        row_coords[axis] = base_idx + turn * BLOCK_SIZE * simd_width
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime for i in range(num_reductions):
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]

    return block_reduce[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)


def row_reduce_fixed_turns_warp0_epilogue[
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Fixed-turn reduction with a warp-0 block epilogue."""
    comptime assert FIXED_TURNS > 0, "fixed-turn reductions must do work"

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    var base_idx = Int(thread_idx.x) * simd_width
    comptime for turn in range(FIXED_TURNS):
        row_coords[axis] = base_idx + turn * BLOCK_SIZE * simd_width
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime for i in range(num_reductions):
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]

    return block_reduce_thread0_serial_epilogue[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)


def row_reduce_fixed_turns_ilp2[
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row with fixed turns and two independent SIMD chains."""
    comptime assert FIXED_TURNS > 1, "ilp2 fixed-turn reductions need >= 2 turns"

    var accum_even = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum_odd = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum_even[i] = init_cast[i]
        accum_odd[i] = init_cast[i]

    var base_idx = Int(thread_idx.x) * simd_width
    comptime for turn in range(FIXED_TURNS):
        row_coords[axis] = base_idx + turn * BLOCK_SIZE * simd_width
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime if turn % 2 == 0:
            comptime for i in range(num_reductions):
                accum_even[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum_even[i]
                )
        else:
            comptime for i in range(num_reductions):
                accum_odd[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum_odd[i]
                )

    comptime for i in range(num_reductions):
        accum_even[i] = reduce_fn[accum_type, simd_width, i](
            accum_odd[i], accum_even[i]
        )

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum_even[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]

    return block_reduce_warp0_epilogue[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)


@always_inline
def widen_bf16_pairs_to_f32x8[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width]) -> SIMD[accum_type, simd_width]:
    """Widen a bf16x8 vector as four bf16x2 pairs before rejoining."""
    comptime assert simd_width == 8, "bf16 pair widen helper expects width 8"

    var pair01 = val.slice[2]().cast[accum_type]()
    var pair23 = val.slice[2, offset=2]().cast[accum_type]()
    var pair45 = val.slice[2, offset=4]().cast[accum_type]()
    var pair67 = val.slice[2, offset=6]().cast[accum_type]()
    var lower = pair01.join(pair23)
    var upper = pair45.join(pair67)

    return rebind[SIMD[accum_type, simd_width]](lower.join(upper))


@always_inline
def split_bf16x8_to_f32x4_halves[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width]) -> StaticTuple[SIMD[accum_type, 4], 2]:
    """Split a bf16x8 vector into lower/upper f32x4 halves."""
    comptime assert simd_width == 8, "bf16 half split helper expects width 8"

    var halves = StaticTuple[SIMD[accum_type, 4], 2]()
    halves[0] = val.slice[4]().cast[accum_type]()
    halves[1] = val.slice[4, offset=4]().cast[accum_type]()
    return halves


def split_bf16x8_to_f32x2_pairs[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width]) -> StaticTuple[SIMD[accum_type, 2], 4]:
    """Split a bf16x8 vector into four bf16x2-derived f32x2 pairs."""
    comptime assert simd_width == 8, "bf16 pair split helper expects width 8"

    var pairs = StaticTuple[SIMD[accum_type, 2], 4]()
    pairs[0] = val.slice[2]().cast[accum_type]()
    pairs[1] = val.slice[2, offset=2]().cast[accum_type]()
    pairs[2] = val.slice[2, offset=4]().cast[accum_type]()
    pairs[3] = val.slice[2, offset=6]().cast[accum_type]()
    return pairs


@always_inline
def split_bf16x8_u32_pairs_to_f32x2_pairs[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](val: SIMD[dtype, simd_width]) -> StaticTuple[SIMD[accum_type, 2], 4]:
    """Split bf16x8 via explicit u32 pair bitcasts before widening to f32x2."""
    comptime assert dtype == DType.bfloat16, "u32 pair helper expects bf16"
    comptime assert simd_width == 8, "u32 pair helper expects width 8"

    var packed = bitcast[DType.uint32, 4](val)
    var pairs = StaticTuple[SIMD[accum_type, 2], 4]()
    comptime for pair_idx in range(4):
        pairs[pair_idx] = bitcast[dtype, 2](packed[pair_idx]).cast[
            accum_type
        ]()
    return pairs


def row_reduce_fixed_turns_ilp3[
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row with fixed turns and three independent SIMD chains."""
    comptime assert FIXED_TURNS > 2, "ilp3 fixed-turn reductions need >= 3 turns"
    # The live 3072 medium-tail path lands here with one fused reduction and
    # exactly three SIMD turns. In that case, each ILP chain only receives one
    # vector, so we can skip the tuple-of-accumulators setup and fold the three
    # vectors directly before the existing block epilogue.
    comptime if num_reductions == 1 and FIXED_TURNS == 3:
        var base_idx = Int(thread_idx.x) * simd_width
        comptime if dtype == DType.bfloat16 and simd_width == 8:
            comptime half4_blockfold_3072 = get_defined_bool[
                "half4_blockfold_3072", True
            ]()
            comptime if half4_blockfold_3072:
                @always_inline
                @parameter
                def duplicate_reduce_fn[
                    reduce_dtype: DType, width: Int, reduction_idx: Int
                ](
                    lhs: SIMD[reduce_dtype, width],
                    rhs: SIMD[reduce_dtype, width],
                    ) -> SIMD[reduce_dtype, width]:
                    return reduce_fn[reduce_dtype, width, 0](lhs, rhs)

                row_coords[axis] = base_idx
                var halves0 = split_bf16x8_to_f32x4_halves[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + BLOCK_SIZE * simd_width
                var halves1 = split_bf16x8_to_f32x4_halves[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + 2 * BLOCK_SIZE * simd_width
                var halves2 = split_bf16x8_to_f32x4_halves[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))

                var lower_accum = duplicate_reduce_fn[accum_type, 4, 0](
                    halves1[0], halves0[0]
                )
                lower_accum = duplicate_reduce_fn[accum_type, 4, 0](
                    halves2[0], lower_accum
                )

                var upper_accum = duplicate_reduce_fn[accum_type, 4, 0](
                    halves1[1], halves0[1]
                )
                upper_accum = duplicate_reduce_fn[accum_type, 4, 0](
                    halves2[1], upper_accum
                )

                # Scalarize each half before the block fold because the
                # live warp shuffle helpers do not support f32x4 lane-group
                # reduce.
                var scalar_vals = StaticTuple[SIMD[accum_type, 1], 2]()
                var scalar_init = StaticTuple[Scalar[accum_type], 2]()
                scalar_vals[0] = lower_accum.reduce[
                    duplicate_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                scalar_vals[1] = upper_accum.reduce[
                    duplicate_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                scalar_init[0] = init[0].cast[accum_type]()
                scalar_init[1] = scalar_init[0]

                var block_scalars = block_reduce_warp0_epilogue[
                    BLOCK_SIZE,
                    2,
                    duplicate_reduce_fn,
                    accum_type,
                    1,
                ](scalar_vals, scalar_init)

                var result = StaticTuple[
                    Scalar[accum_type], num_reductions
                ]()
                var combined = duplicate_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[1]),
                    SIMD[accum_type, 1](block_scalars[0]),
                )
                result[0] = combined.reduce[
                    duplicate_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                return result

            # Spell the bf16 widen as four bf16x2 casts rejoined into f32x8 so
            # the exact 3072 path can perturb only the bf16 use-def graph.
            row_coords[axis] = base_idx
            var val0 = widen_bf16_pairs_to_f32x8[dtype, accum_type, simd_width](
                input_fn[dtype, simd_width, rank](row_coords)
            )

            row_coords[axis] = base_idx + BLOCK_SIZE * simd_width
            var val1 = widen_bf16_pairs_to_f32x8[dtype, accum_type, simd_width](
                input_fn[dtype, simd_width, rank](row_coords)
            )

            row_coords[axis] = base_idx + 2 * BLOCK_SIZE * simd_width
            var val2 = widen_bf16_pairs_to_f32x8[dtype, accum_type, simd_width](
                input_fn[dtype, simd_width, rank](row_coords)
            )

            var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
            var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
            var accum01 = reduce_fn[accum_type, simd_width, 0](val1, val0)
            var accum = reduce_fn[accum_type, simd_width, 0](val2, accum01)

            scalar_vals[0] = accum.reduce[
                reduce_fn[accum_type, reduction_idx=0, ...]
            ]()
            scalar_init[0] = init[0].cast[accum_type]()

            return block_reduce_warp0_epilogue[
                BLOCK_SIZE,
                num_reductions,
                reduce_fn,
                accum_type,
                1,
            ](scalar_vals, scalar_init)

        row_coords[axis] = base_idx
        var val0 = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        row_coords[axis] = base_idx + BLOCK_SIZE * simd_width
        var val1 = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        row_coords[axis] = base_idx + 2 * BLOCK_SIZE * simd_width
        var val2 = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        var accum01 = reduce_fn[accum_type, simd_width, 0](val1, val0)
        var accum = reduce_fn[accum_type, simd_width, 0](val2, accum01)

        var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
        var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()

        scalar_vals[0] = accum.reduce[
            reduce_fn[accum_type, reduction_idx=0, ...]
        ]()
        scalar_init[0] = init[0].cast[accum_type]()

        return block_reduce_warp0_epilogue[
            BLOCK_SIZE,
            num_reductions,
            reduce_fn,
            accum_type,
            1,
        ](scalar_vals, scalar_init)

    var accum0 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum1 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum2 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum0[i] = init_cast[i]
        accum1[i] = init_cast[i]
        accum2[i] = init_cast[i]

    var base_idx = Int(thread_idx.x) * simd_width
    comptime for turn in range(FIXED_TURNS):
        row_coords[axis] = base_idx + turn * BLOCK_SIZE * simd_width
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime if turn % 3 == 0:
            comptime for i in range(num_reductions):
                accum0[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum0[i]
                )
        elif turn % 3 == 1:
            comptime for i in range(num_reductions):
                accum1[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum1[i]
                )
        else:
            comptime for i in range(num_reductions):
                accum2[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum2[i]
                )

    comptime for i in range(num_reductions):
        var accum01 = reduce_fn[accum_type, simd_width, i](
            accum1[i], accum0[i]
        )
        accum0[i] = reduce_fn[accum_type, simd_width, i](accum2[i], accum01)

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum0[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]

    return block_reduce_warp0_epilogue[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)


def row_reduce_fixed_turns_ilp4[
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    reduce_fn: def[dtype: DType, width: Int, reduction_idx: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing[_] -> SIMD[dtype, width],
    dtype: DType,
    simd_width: Int,
    rank: Int,
    accum_type: DType = get_accum_type[dtype](),
](
    mut row_coords: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
) -> StaticTuple[Scalar[accum_type], num_reductions]:
    """Reduces a row with fixed turns and four independent SIMD chains."""
    comptime assert FIXED_TURNS > 3, "ilp4 fixed-turn reductions need >= 4 turns"
    # The exact 4096 path also lands here with one fused reduction and exactly
    # four SIMD turns. A bf16-only front-half rewrite can therefore replace the
    # tuple-of-accumulators setup with a narrower pair-wise accumulation graph.
    comptime if num_reductions == 1 and FIXED_TURNS == 4:
        var base_idx = Int(thread_idx.x) * simd_width
        comptime if dtype == DType.bfloat16 and simd_width == 8:
            comptime dual_half_load_4096 = get_defined_bool[
                "dual_half_load_4096", False
            ]()
            comptime pair2_frontload_4096 = get_defined_bool[
                "pair2_frontload_4096", False
            ]()
            comptime u32_pair_bitcast_frontload_4096 = get_defined_bool[
                "u32_pair_bitcast_frontload_4096", False
            ]()
            comptime if dual_half_load_4096:
                @always_inline
                @parameter
                def dual_half_reduce_fn[
                    reduce_dtype: DType, width: Int, reduction_idx: Int
                ](
                    lhs: SIMD[reduce_dtype, width],
                    rhs: SIMD[reduce_dtype, width],
                ) -> SIMD[reduce_dtype, width]:
                    return reduce_fn[reduce_dtype, width, 0](lhs, rhs)

                comptime turn_stride = BLOCK_SIZE * simd_width

                row_coords[axis] = base_idx
                var lower0 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()
                row_coords[axis] = base_idx + 4
                var upper0 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()

                row_coords[axis] = base_idx + turn_stride
                var lower1 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()
                row_coords[axis] = base_idx + turn_stride + 4
                var upper1 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()

                row_coords[axis] = base_idx + 2 * turn_stride
                var lower2 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()
                row_coords[axis] = base_idx + 2 * turn_stride + 4
                var upper2 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()

                row_coords[axis] = base_idx + 3 * turn_stride
                var lower3 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()
                row_coords[axis] = base_idx + 3 * turn_stride + 4
                var upper3 = input_fn[dtype, 4, rank](row_coords).cast[
                    accum_type
                ]()

                var lower01 = dual_half_reduce_fn[accum_type, 4, 0](
                    lower1, lower0
                )
                var lower23 = dual_half_reduce_fn[accum_type, 4, 0](
                    lower3, lower2
                )
                var lower_accum = dual_half_reduce_fn[accum_type, 4, 0](
                    lower23, lower01
                )

                var upper01 = dual_half_reduce_fn[accum_type, 4, 0](
                    upper1, upper0
                )
                var upper23 = dual_half_reduce_fn[accum_type, 4, 0](
                    upper3, upper2
                )
                var upper_accum = dual_half_reduce_fn[accum_type, 4, 0](
                    upper23, upper01
                )

                var scalar_vals = StaticTuple[SIMD[accum_type, 1], 2]()
                var scalar_init = StaticTuple[Scalar[accum_type], 2]()
                scalar_vals[0] = lower_accum.reduce[
                    dual_half_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                scalar_vals[1] = upper_accum.reduce[
                    dual_half_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                scalar_init[0] = init[0].cast[accum_type]()
                scalar_init[1] = scalar_init[0]

                var block_scalars = block_reduce_warp0_epilogue[
                    BLOCK_SIZE,
                    2,
                    dual_half_reduce_fn,
                    accum_type,
                    1,
                ](scalar_vals, scalar_init)

                var result = StaticTuple[Scalar[accum_type], num_reductions]()
                var combined = dual_half_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[1]),
                    SIMD[accum_type, 1](block_scalars[0]),
                )
                result[0] = combined.reduce[
                    dual_half_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                return result
            elif u32_pair_bitcast_frontload_4096:
                @always_inline
                @parameter
                def u32_pair_reduce_fn[
                    reduce_dtype: DType, width: Int, reduction_idx: Int
                ](
                    lhs: SIMD[reduce_dtype, width],
                    rhs: SIMD[reduce_dtype, width],
                ) -> SIMD[reduce_dtype, width]:
                    return reduce_fn[reduce_dtype, width, 0](lhs, rhs)

                row_coords[axis] = base_idx
                var pairs0 = split_bf16x8_u32_pairs_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + BLOCK_SIZE * simd_width
                var pairs1 = split_bf16x8_u32_pairs_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + 2 * BLOCK_SIZE * simd_width
                var pairs2 = split_bf16x8_u32_pairs_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + 3 * BLOCK_SIZE * simd_width
                var pairs3 = split_bf16x8_u32_pairs_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))

                var pair_accum = StaticTuple[SIMD[accum_type, 2], 4]()
                comptime for pair_idx in range(4):
                    var accum02 = u32_pair_reduce_fn[accum_type, 2, 0](
                        pairs2[pair_idx], pairs0[pair_idx]
                    )
                    var accum13 = u32_pair_reduce_fn[accum_type, 2, 0](
                        pairs3[pair_idx], pairs1[pair_idx]
                    )
                    pair_accum[pair_idx] = u32_pair_reduce_fn[
                        accum_type, 2, 0
                    ](accum13, accum02)

                var scalar_vals = StaticTuple[SIMD[accum_type, 1], 4]()
                var scalar_init = StaticTuple[Scalar[accum_type], 4]()
                comptime for pair_idx in range(4):
                    scalar_vals[pair_idx] = pair_accum[pair_idx].reduce[
                        u32_pair_reduce_fn[
                            accum_type, reduction_idx=0, ...
                        ]
                    ]()
                    scalar_init[pair_idx] = init[0].cast[accum_type]()

                var block_scalars = block_reduce_warp0_epilogue[
                    BLOCK_SIZE,
                    4,
                    u32_pair_reduce_fn,
                    accum_type,
                    1,
                ](scalar_vals, scalar_init)

                var result = StaticTuple[Scalar[accum_type], num_reductions]()
                var combined01 = u32_pair_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[1]),
                    SIMD[accum_type, 1](block_scalars[0]),
                )
                var combined23 = u32_pair_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[3]),
                    SIMD[accum_type, 1](block_scalars[2]),
                )
                var combined = u32_pair_reduce_fn[accum_type, 1, 0](
                    combined23, combined01
                )
                result[0] = combined.reduce[
                    u32_pair_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                return result
            elif pair2_frontload_4096:
                @always_inline
                @parameter
                def duplicate_reduce_fn[
                    reduce_dtype: DType, width: Int, reduction_idx: Int
                ](
                    lhs: SIMD[reduce_dtype, width],
                    rhs: SIMD[reduce_dtype, width],
                ) -> SIMD[reduce_dtype, width]:
                    return reduce_fn[reduce_dtype, width, 0](lhs, rhs)

                row_coords[axis] = base_idx
                var pairs0 = split_bf16x8_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + BLOCK_SIZE * simd_width
                var pairs1 = split_bf16x8_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + 2 * BLOCK_SIZE * simd_width
                var pairs2 = split_bf16x8_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))
                row_coords[axis] = base_idx + 3 * BLOCK_SIZE * simd_width
                var pairs3 = split_bf16x8_to_f32x2_pairs[
                    dtype, accum_type, simd_width
                ](input_fn[dtype, simd_width, rank](row_coords))

                var pair_accum = StaticTuple[SIMD[accum_type, 2], 4]()
                comptime for pair_idx in range(4):
                    var accum02 = duplicate_reduce_fn[accum_type, 2, 0](
                        pairs2[pair_idx], pairs0[pair_idx]
                    )
                    var accum13 = duplicate_reduce_fn[accum_type, 2, 0](
                        pairs3[pair_idx], pairs1[pair_idx]
                    )
                    pair_accum[pair_idx] = duplicate_reduce_fn[
                        accum_type, 2, 0
                    ](accum13, accum02)

                var scalar_vals = StaticTuple[SIMD[accum_type, 1], 4]()
                var scalar_init = StaticTuple[Scalar[accum_type], 4]()
                comptime for pair_idx in range(4):
                    scalar_vals[pair_idx] = pair_accum[pair_idx].reduce[
                        duplicate_reduce_fn[
                            accum_type, reduction_idx=0, ...
                        ]
                    ]()
                    scalar_init[pair_idx] = init[0].cast[accum_type]()

                var block_scalars = block_reduce_warp0_epilogue[
                    BLOCK_SIZE,
                    4,
                    duplicate_reduce_fn,
                    accum_type,
                    1,
                ](scalar_vals, scalar_init)

                var result = StaticTuple[Scalar[accum_type], num_reductions]()
                var combined01 = duplicate_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[1]),
                    SIMD[accum_type, 1](block_scalars[0]),
                )
                var combined23 = duplicate_reduce_fn[accum_type, 1, 0](
                    SIMD[accum_type, 1](block_scalars[3]),
                    SIMD[accum_type, 1](block_scalars[2]),
                )
                var combined = duplicate_reduce_fn[accum_type, 1, 0](
                    combined23, combined01
                )
                result[0] = combined.reduce[
                    duplicate_reduce_fn[accum_type, reduction_idx=0, ...]
                ]()
                return result

    var accum0 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum1 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum2 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var accum3 = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum0[i] = init_cast[i]
        accum1[i] = init_cast[i]
        accum2[i] = init_cast[i]
        accum3[i] = init_cast[i]

    var base_idx = Int(thread_idx.x) * simd_width
    comptime for turn in range(FIXED_TURNS):
        row_coords[axis] = base_idx + turn * BLOCK_SIZE * simd_width
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime if turn % 4 == 0:
            comptime for i in range(num_reductions):
                accum0[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum0[i]
                )
        elif turn % 4 == 1:
            comptime for i in range(num_reductions):
                accum1[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum1[i]
                )
        elif turn % 4 == 2:
            comptime for i in range(num_reductions):
                accum2[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum2[i]
                )
        else:
            comptime for i in range(num_reductions):
                accum3[i] = reduce_fn[accum_type, simd_width, i](
                    val, accum3[i]
                )

    comptime for i in range(num_reductions):
        var accum01 = reduce_fn[accum_type, simd_width, i](
            accum1[i], accum0[i]
        )
        var accum23 = reduce_fn[accum_type, simd_width, i](
            accum3[i], accum2[i]
        )
        accum0[i] = reduce_fn[accum_type, simd_width, i](accum23, accum01)

    var scalar_vals = StaticTuple[SIMD[accum_type, 1], num_reductions]()
    var scalar_init = StaticTuple[Scalar[accum_type], num_reductions]()
    comptime for i in range(num_reductions):
        scalar_vals[i] = accum0[i].reduce[
            reduce_fn[accum_type, reduction_idx=i, ...]
        ]()
        scalar_init[i] = init_cast[i]

    return block_reduce_warp0_epilogue[
        BLOCK_SIZE,
        num_reductions,
        reduce_fn,
        accum_type,
        1,
    ](scalar_vals, scalar_init)

@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def reduce_kernel_fixed_turns_ilp2[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """Fixed-turn reduction with split accumulation chains for extra ILP."""
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce_fixed_turns_ilp2[
            BLOCK_SIZE,
            FIXED_TURNS,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def reduce_kernel_fixed_turns_ilp3[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """Fixed-turn reduction with three independent accumulation chains."""
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce_fixed_turns_ilp3[
            BLOCK_SIZE,
            FIXED_TURNS,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


def reduce_kernel_fixed_turns_ilp4[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """Fixed-turn reduction with four independent accumulation chains."""
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce_fixed_turns_ilp4[
            BLOCK_SIZE,
            FIXED_TURNS,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


def reduce_kernel_fixed_turns[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """Block-cooperative reduction with a fixed compile-time row geometry."""
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce_fixed_turns[
            BLOCK_SIZE,
            FIXED_TURNS,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


def reduce_kernel_fixed_turns_warp0_epilogue[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    FIXED_TURNS: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """Fixed-turn reduction with a warp-0 block epilogue."""
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce_fixed_turns_warp0_epilogue[
            BLOCK_SIZE,
            FIXED_TURNS,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel that reduces rows along a given axis. Each block reduces one
    row at a time using `row_reduce` and writes the result via `output_fn`.
    Uses a grid-stride loop to handle more rows than blocks.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    # grid stride loop over rows
    # each block reduces a row, which requires no partial reductions
    for row_idx in range(block_idx.x, UInt(num_rows), grid_dim.x):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var row_accum = row_reduce[
            BLOCK_SIZE,
            num_reductions,
            input_fn,
            reduce_fn,
            dtype,
            simd_width,
            rank,
            accum_type=accum_type,
        ](row_coords, axis, init, row_size)

        if thread_idx.x == 0:
            var row_accum_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                row_accum_cast[i] = row_accum[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


def small_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel optimized for rows smaller than the warp size. Each warp
    reduces an entire row independently, allowing multiple rows to be reduced
    per block without shared-memory synchronization.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    comptime warps_per_block = BLOCK_SIZE // WARP_SIZE

    # grid stride loop over rows
    # each block reduces as many rows as warps,
    # No need to partial reduction because this is the degenerated case of
    # rows smaller than warp size
    #
    for row_idx in range(
        block_idx.x * UInt(warps_per_block),
        UInt(num_rows),
        grid_dim.x * UInt(warps_per_block),
    ):
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx) + Int(warp_id()), shape, axis
        )

        # One row per warp, warp collectively reads from global
        if warp_id() < UInt(warps_per_block):
            var val = InlineArray[SIMD[accum_type, simd_width], num_reductions](
                fill=0
            )

            comptime for i in range(num_reductions):
                val[i] = init[i].cast[accum_type]()

            if lane_id() < UInt(row_size):
                row_coords[axis] = Int(lane_id())
                var t = input_fn[dtype, simd_width, rank](row_coords).cast[
                    accum_type
                ]()

                val = type_of(val)(fill=t)
            else:
                comptime for i in range(num_reductions):
                    val[i] = init[i].cast[accum_type]()

            var result = InlineArray[
                SIMD[accum_type, simd_width], num_reductions
            ](fill=0)

            comptime for i in range(num_reductions):

                @always_inline
                @parameter
                def reduce_wrapper[
                    dtype: DType, width: Int
                ](
                    x: SIMD[dtype, width], y: SIMD[dtype, width]
                ) capturing -> SIMD[dtype, width]:
                    return reduce_fn[dtype, width, i](x, y)

                result[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](
                    val[i]
                )

            if lane_id() == 0:
                var row_accum_cast = StaticTuple[
                    Scalar[dtype], num_reductions
                ]()

                comptime for i in range(num_reductions):
                    row_accum_cast[i] = result[i][0].cast[dtype]()

                row_coords[axis] = 0
                output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def warp_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int = 1,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    comptime warps_per_block = BLOCK_SIZE // WARP_SIZE
    comptime VEC_STRIDE = WARP_SIZE * simd_width

    for row_idx in range(
        block_idx.x * UInt(warps_per_block),
        UInt(num_rows),
        grid_dim.x * UInt(warps_per_block),
    ):
        var my_row = Int(row_idx) + Int(warp_id())
        if UInt(my_row) >= UInt(num_rows):
            continue

        var row_coords = _get_nd_indices_from_flat_index(my_row, shape, axis)

        if warp_id() < UInt(warps_per_block):
            var accum = InlineArray[SIMD[accum_type, simd_width], num_reductions](
                uninitialized=True
            )
            comptime for i in range(num_reductions):
                accum[i] = SIMD[accum_type, simd_width](init[i].cast[accum_type]())

            var lid = Int(lane_id())
            var vec_col = lid * simd_width
            var vec_limit = row_size - (simd_width - 1)
            while vec_col < vec_limit:
                row_coords[axis] = vec_col
                var v = input_fn[dtype, simd_width, rank](row_coords).cast[accum_type]()
                comptime for i in range(num_reductions):
                    accum[i] = reduce_fn[accum_type, simd_width, i](accum[i], v)
                vec_col += VEC_STRIDE

            var scalar_accum = InlineArray[Scalar[accum_type], num_reductions](
                uninitialized=True
            )
            comptime for i in range(num_reductions):
                scalar_accum[i] = accum[i].reduce[
                    reduce_fn[accum_type, reduction_idx=i, ...]
                ]()

            var tail_col = (row_size // VEC_STRIDE) * VEC_STRIDE + lid
            while tail_col < row_size:
                row_coords[axis] = tail_col
                var v = input_fn[dtype, 1, rank](row_coords).cast[accum_type]()
                comptime for i in range(num_reductions):
                    scalar_accum[i] = reduce_fn[accum_type, 1, i](scalar_accum[i], v)
                tail_col += WARP_SIZE

            comptime for i in range(num_reductions):

                @always_inline
                @parameter
                def reduce_wrapper[
                    _dtype: DType, width: Int
                ](
                    x: SIMD[_dtype, width], y: SIMD[_dtype, width]
                ) capturing -> SIMD[_dtype, width]:
                    return reduce_fn[_dtype, width, i](x, y)

                scalar_accum[i] = warp.reduce[warp.shuffle_down, reduce_wrapper](
                    scalar_accum[i]
                )

            if lane_id() == 0:
                var row_accum_cast = StaticTuple[
                    Scalar[dtype], num_reductions
                ]()
                comptime for i in range(num_reductions):
                    row_accum_cast[i] = scalar_accum[i].cast[dtype]()
                row_coords[axis] = 0
                output_fn[dtype, 1, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def twophase_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](
    shape: IndexList[rank],
    init: StaticTuple[Scalar[dtype], num_reductions],
    partials: UnsafePointer[Scalar[accum_type], MutAnyOrigin],
    counters: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    blocks_per_row: Int,
):
    """GPU kernel for reductions when there are too few rows to saturate the
    device at one block per row. Assigns multiple blocks per row and uses a
    two-phase approach: each block reduces a chunk via cooperative block-level
    reduction, then the last block to finish (detected via a per-row atomic
    counter) reduces all partial results for its row.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
        partials: Global memory buffer for per-block partial results.
            Size: grid_dim.x * num_reductions elements of accum_type.
        counters: Global memory buffer for per-row atomic completion counters.
            Size: num_rows elements of int32, zero-initialized.
        blocks_per_row: The number of blocks assigned to each row.
    """
    comptime assert (
        simd_width == 1
    ), "twophase_reduce_kernel only currently supports simd_width == 1"
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    var row_idx, block_in_row = udivmod(Int(block_idx.x), blocks_per_row)

    if row_idx >= num_rows:
        return

    var row_coords = _get_nd_indices_from_flat_index(row_idx, shape, axis)

    # --- Phase 1: Each block reduces its portion of the row ---
    # Threads are striped across ALL blocks for this row to coalesce reads.
    var row_tid = block_in_row * BLOCK_SIZE + Int(thread_idx.x)
    var row_total_threads = blocks_per_row * BLOCK_SIZE

    var accum = StaticTuple[SIMD[accum_type, simd_width], num_reductions]()
    var init_cast = StaticTuple[Scalar[accum_type], num_reductions]()

    comptime for i in range(num_reductions):
        init_cast[i] = init[i].cast[accum_type]()
        accum[i] = init_cast[i]

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    for elem_idx in range(row_tid, row_size, row_total_threads):
        row_coords[axis] = elem_idx
        var val = input_fn[dtype, simd_width, rank](row_coords).cast[
            accum_type
        ]()

        comptime for i in range(num_reductions):
            accum[i] = reduce_fn[accum_type, simd_width, i](val, accum[i])

    var partial = block_reduce[
        BLOCK_SIZE, num_reductions, reduce_fn, accum_type, simd_width
    ](accum, init_cast)

    # Thread 0 writes partial result for this block and signals completion.
    var is_last_block: Scalar[DType.bool] = False
    if thread_idx.x == 0:
        var base = Int(block_idx.x) * num_reductions
        comptime for i in range(num_reductions):
            partials[base + i] = partial[i]

        var finished = Atomic[DType.int32].fetch_add(
            counters + row_idx, Int32(1)
        )
        is_last_block = finished == Int32(blocks_per_row - 1)

    # --- Phase 2: Last block reduces all partials for this row ---
    # Broadcast is_last_block from thread 0 to all threads via shared memory
    # so the entire block can participate cooperatively.
    is_last_block = broadcast[block_size=BLOCK_SIZE](is_last_block)
    if is_last_block:
        # Each thread loads a stripe of the partials and reduces locally.
        var thread_accum = StaticTuple[Scalar[accum_type], num_reductions]()

        comptime for i in range(num_reductions):
            thread_accum[i] = init_cast[i]

        var row_base = row_idx * blocks_per_row * num_reductions
        for b in range(Int(thread_idx.x), blocks_per_row, BLOCK_SIZE):
            comptime for i in range(num_reductions):
                thread_accum[i] = reduce_fn[accum_type, 1, i](
                    thread_accum[i],
                    partials[row_base + b * num_reductions + i],
                )

        # Note this is currently no-op since we insist simd_width==1
        var accum_simd = StaticTuple[
            SIMD[accum_type, simd_width], num_reductions
        ]()
        comptime for i in range(num_reductions):
            accum_simd[i] = thread_accum[i]

        var final_result = block_reduce[
            BLOCK_SIZE, num_reductions, reduce_fn, accum_type, simd_width
        ](accum_simd, init_cast)

        if thread_idx.x == 0:
            var result_cast = StaticTuple[Scalar[dtype], num_reductions]()

            comptime for i in range(num_reductions):
                result_cast[i] = final_result[i].cast[dtype]()

            row_coords[axis] = 0
            output_fn[dtype, 1, rank](row_coords, result_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
def saturated_reduce_kernel[
    rank: Int,
    axis: Int,
    num_reductions: Int,
    BLOCK_SIZE: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    dtype: DType,
    simd_width: Int,
    accum_type: DType = get_accum_type[dtype](),
    pdl_level: PDLLevel = PDLLevel(),
](shape: IndexList[rank], init: StaticTuple[Scalar[dtype], num_reductions],):
    """GPU kernel for reductions when the device is saturated with enough rows.
    Each thread independently reduces an entire row using SIMD packing,
    avoiding shared-memory synchronization entirely. Used when reducing along
    a non-contiguous axis.

    Parameters:
        rank: The tensor rank.
        axis: The axis along which to reduce.
        num_reductions: The number of fused reductions to perform.
        BLOCK_SIZE: The number of threads per block.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        dtype: The data type of the elements.
        simd_width: The SIMD vector width.
        accum_type: The accumulator data type.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the input tensor.
        init: The identity values for each reduction.
    """
    var row_size = shape[axis]
    var num_rows = shape.flattened_length() // row_size

    comptime if pdl_level == PDLLevel.OVERLAP_AT_BEGINNING:
        launch_dependent_grids()

    comptime if pdl_level > PDLLevel.OFF:
        wait_on_dependent_grids()

    var global_dim_x = grid_dim.x * block_dim.x
    # Loop over rows
    for row_idx in range(
        global_idx.x * UInt(simd_width),
        UInt(num_rows),
        global_dim_x * UInt(simd_width),
    ):
        # Reduce the whole row
        var row_coords = _get_nd_indices_from_flat_index(
            Int(row_idx), shape, axis
        )

        var val = InlineArray[SIMD[accum_type, simd_width], num_reductions](
            uninitialized=True
        )

        comptime for i in range(num_reductions):
            val[i] = init[i].cast[accum_type]()

        for val_idx in range(row_size):
            row_coords[axis] = val_idx
            var t = input_fn[dtype, simd_width, rank](row_coords).cast[
                accum_type
            ]()

            comptime for i in range(num_reductions):
                val[i] = reduce_fn[reduction_idx=i](val[i], t)

        # Cast to output type
        var row_accum_cast = StaticTuple[
            SIMD[dtype, simd_width], num_reductions
        ]()

        comptime for i in range(num_reductions):
            row_accum_cast[i] = rebind[SIMD[dtype, simd_width]](
                val[i].cast[dtype]()
            )

        # Write output
        row_coords[axis] = 0
        output_fn[dtype, simd_width, rank](row_coords, row_accum_cast)

    comptime if pdl_level == PDLLevel.OVERLAP_AT_END:
        launch_dependent_grids()


def reduce_launch[
    num_reductions: Int,
    input_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_fn: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    rank: Int,
    dtype: DType,
    pdl_level: PDLLevel = PDLLevel(),
](
    shape: IndexList[rank],
    axis: Int,
    init: StaticTuple[Scalar[dtype], num_reductions],
    ctx: DeviceContext,
) raises:
    """Selects and launches the appropriate GPU reduction kernel based on the
    tensor shape, axis, and device saturation level.

    Three-tier dispatch:
    1. Thread-saturated (many rows, non-contiguous axis): one row per thread
       via `saturated_reduce_kernel`.
    2. Block-saturated (enough rows to fill SMs at one block per row):
       `reduce_kernel` or `small_reduce_kernel`.
    3. Under-saturated (too few rows to fill the device): multiple blocks per
       row via `twophase_reduce_kernel` with a two-phase atomic finish.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        input_fn: The lambda to load input elements.
        output_fn: The lambda to store output elements.
        reduce_fn: The binary reduction function.
        rank: The tensor rank.
        dtype: The data type of the elements.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the input tensor.
        axis: The axis along which to reduce.
        init: The identity values for each reduction.
        ctx: The device context for GPU execution.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime register_width = 32
    comptime sm_count = ctx.default_device_info.sm_count

    comptime packing_factor = 1

    var num_rows = shape.flattened_length() // shape[axis] // packing_factor
    comptime sm_overprovision_factor = 32  # tunable
    var num_blocks = min(num_rows, sm_overprovision_factor * sm_count)

    # Do not launch gpu kernels with grid_dim = 0
    if num_blocks == 0:
        return

    # 256 is a proxy for BLOCK_SIZE, because having BLOCK_SIZE affect kernel
    # selection is likely to confound autotuning.
    comptime num_persistent_threads = 256 * sm_count

    # enough rows that each thread can handle a whole row.
    var thread_saturated: Bool = num_rows >= num_persistent_threads
    # enough rows that each block/SM can handle a whole row.
    var block_saturated: Bool = num_rows >= sm_count
    # enough work to justify two-phase kernel
    comptime unsaturated_block_size = 128
    var more_values_than_threads: Bool = shape[axis] > unsaturated_block_size

    # This assumes row-major layout:
    var reduce_contig_dim: Bool = axis == rank - 1

    # --- Tier 1: Thread-saturated, non-contiguous axis ---
    # Each thread handles a whole row. SIMD packing across adjacent rows.
    if thread_saturated and not reduce_contig_dim:
        comptime BLOCK_SIZE = get_defined_int["MOJO_REDUCTION_BLOCK_SIZE", 64 if dtype == DType.float16 else 32]()
        comptime native_packing = simd_width_of[dtype, get_gpu_target()]()
        comptime narrow_packing = 4 if dtype == DType.float16 else native_packing
        if shape[axis] >= 8 or (dtype == DType.float16 and shape[axis] >= 6):
            comptime for ax in range(rank):
                if axis == ax:
                    comptime kernel = saturated_reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        narrow_packing,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=(min(num_rows, 128 * sm_count) if num_rows > 2000000 else num_blocks),
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(pdl_level),
                    )
        else:
            comptime for ax in range(rank):
                if axis == ax:
                    comptime kernel = saturated_reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        native_packing,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=(min(num_rows, 128 * sm_count) if num_rows > 2000000 else num_blocks),
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(pdl_level),
                    )

    # --- Tier 3: Under-saturated ---
    # Too few rows to fill the device. Assign multiple blocks per row.
    # The canonical case here is a complete reduction (i.e. output size = 1)
    # Only use twophase when there are more_values_than_threads to
    # justify the overhead. Otherwise fall through to standard kernels.
    elif not block_saturated and more_values_than_threads:
        comptime BLOCK_SIZE = get_defined_int[
            "MOJO_REDUCTION_BLOCK_SIZE", unsaturated_block_size
        ]()

        # Round down to avoid a second wave
        var target_blocks = sm_count * sm_overprovision_factor
        var blocks_per_row = target_blocks // num_rows
        var total_blocks = num_rows * blocks_per_row
        comptime _accum_type = get_accum_type[dtype]()
        var partials_buf = ctx.enqueue_create_buffer[_accum_type](
            total_blocks * num_reductions
        )
        var counter_buf = ctx.enqueue_create_buffer[DType.int32](num_rows)
        ctx.enqueue_memset(counter_buf, Int32(0))

        comptime for ax in range(rank):
            if axis == ax:
                comptime kernel = twophase_reduce_kernel[
                    rank,
                    ax,
                    num_reductions,
                    BLOCK_SIZE,
                    input_fn,
                    output_fn,
                    reduce_fn,
                    dtype,
                    packing_factor,
                    pdl_level=pdl_level,
                ]
                ctx.enqueue_function[kernel, kernel](
                    shape,
                    init,
                    partials_buf.unsafe_ptr(),
                    counter_buf.unsafe_ptr(),
                    blocks_per_row,
                    grid_dim=total_blocks,
                    block_dim=BLOCK_SIZE,
                    attributes=pdl_launch_attributes(pdl_level),
                )

        _ = partials_buf
        _ = counter_buf

    # --- Tier 2: Block-saturated ---
    # Enough rows for one block per row. Standard cooperative reduction.
    else:
        comptime BLOCK_SIZE = get_defined_int[
            "MOJO_REDUCTION_BLOCK_SIZE", 128
        ]()
        comptime contig_simd = simd_width_of[dtype, get_gpu_target()]()
        if shape[axis] < WARP_SIZE:
            comptime for ax in range(rank):
                if axis == ax:
                    comptime kernel = small_reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        packing_factor,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(pdl_level),
                    )
        elif reduce_contig_dim and shape[axis] >= WARP_SIZE and (thread_saturated or (block_saturated and num_rows >= 8 * sm_count)):
            comptime for ax in range(rank):
                if axis == ax:
                    if not thread_saturated and shape[axis] <= 256:
                        comptime warp_vec_width = 4 if (
                            dtype == DType.float16 or dtype == DType.bfloat16
                        ) else simd_width_of[dtype, get_gpu_target()]()
                        comptime WARP_BLOCK = 128
                        comptime kernel = warp_reduce_kernel[
                            rank,
                            ax,
                            num_reductions,
                            WARP_BLOCK,
                            input_fn,
                            output_fn,
                            reduce_fn,
                            dtype,
                            warp_vec_width,
                            pdl_level=pdl_level,
                        ]
                        comptime warps_per_block = WARP_BLOCK // WARP_SIZE
                        var warp_grid = min((num_rows + warps_per_block - 1) // warps_per_block, 128 * sm_count)
                        ctx.enqueue_function[kernel, kernel](
                            shape,
                            init,
                            grid_dim=warp_grid,
                            block_dim=WARP_BLOCK,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
                    else:
                        comptime warp_vec_width = 4 if (
                            dtype == DType.float16
                        ) else simd_width_of[dtype, get_gpu_target()]()
                        comptime WARP_BLOCK = 256
                        comptime kernel = warp_reduce_kernel[
                            rank,
                            ax,
                            num_reductions,
                            WARP_BLOCK,
                            input_fn,
                            output_fn,
                            reduce_fn,
                            dtype,
                            warp_vec_width,
                            pdl_level=pdl_level,
                        ]
                        comptime warps_per_block = WARP_BLOCK // WARP_SIZE
                        var warp_grid = min((num_rows + warps_per_block - 1) // warps_per_block, 128 * sm_count)
                        ctx.enqueue_function[kernel, kernel](
                            shape,
                            init,
                            grid_dim=warp_grid,
                            block_dim=WARP_BLOCK,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
        elif (
            reduce_contig_dim
            and (dtype == DType.bfloat16 or dtype == DType.float16)
            and shape[axis] == 3072
            and block_saturated
            and not thread_saturated
        ):
            comptime bf16_plain_fixedturn_3072 = get_defined_bool[
                "bf16_plain_fixedturn_3072", False
            ]()
            comptime bf16_warp_kernel_3072 = get_defined_bool[
                "bf16_warp_kernel_3072", False
            ]()
            comptime assert (
                3072 % (BLOCK_SIZE * contig_simd) == 0
            ), "3072 ILP path requires exact block geometry"
            comptime fixed_turns = 3072 // (BLOCK_SIZE * contig_simd)
            comptime for ax in range(rank):
                if axis == ax:
                    comptime if dtype == DType.bfloat16 and bf16_warp_kernel_3072:
                        # Round-local A/B: reroute only the exact bf16 medium-tail
                        # shape through the existing long-row warp kernel.
                        comptime warp_vec_width = simd_width_of[
                            dtype, get_gpu_target()
                        ]()
                        comptime WARP_BLOCK = 256
                        comptime kernel = warp_reduce_kernel[
                            rank,
                            ax,
                            num_reductions,
                            WARP_BLOCK,
                            input_fn,
                            output_fn,
                            reduce_fn,
                            dtype,
                            warp_vec_width,
                            pdl_level=pdl_level,
                        ]
                        comptime warps_per_block = WARP_BLOCK // WARP_SIZE
                        var warp_grid = min(
                            (num_rows + warps_per_block - 1) // warps_per_block,
                            128 * sm_count,
                        )
                        ctx.enqueue_function[kernel, kernel](
                            shape,
                            init,
                            grid_dim=warp_grid,
                            block_dim=WARP_BLOCK,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
                    else:
                        comptime if (
                            dtype == DType.bfloat16 and bf16_plain_fixedturn_3072
                        ):
                            # Round-local A/B: compare the generic fixed-turn
                            # helper against the live bf16-only ILP3 path.
                            comptime kernel = reduce_kernel_fixed_turns[
                                rank,
                                ax,
                                num_reductions,
                                BLOCK_SIZE,
                                fixed_turns,
                                input_fn,
                                output_fn,
                                reduce_fn,
                                dtype,
                                contig_simd,
                                pdl_level=pdl_level,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                shape,
                                init,
                                grid_dim=num_blocks,
                                block_dim=BLOCK_SIZE,
                                attributes=pdl_launch_attributes(pdl_level),
                            )
                        else:
                            comptime kernel = reduce_kernel_fixed_turns_ilp3[
                                rank,
                                ax,
                                num_reductions,
                                BLOCK_SIZE,
                                fixed_turns,
                                input_fn,
                                output_fn,
                                reduce_fn,
                                dtype,
                                contig_simd,
                                pdl_level=pdl_level,
                            ]
                            ctx.enqueue_function[kernel, kernel](
                                shape,
                                init,
                                grid_dim=num_blocks,
                                block_dim=BLOCK_SIZE,
                                attributes=pdl_launch_attributes(pdl_level),
                            )
        elif (
            reduce_contig_dim
            and shape[axis] == 4096
            and block_saturated
            and not thread_saturated
            and num_rows <= 2 * sm_count
        ):
            comptime block256_ilp2_kernel_4096 = get_defined_bool[
                "block256_ilp2_kernel_4096", False
            ]()
            comptime for ax in range(rank):
                if axis == ax:
                    comptime if block256_ilp2_kernel_4096:
                        comptime wide_block_size = 256
                        comptime assert (
                            4096 % (wide_block_size * contig_simd) == 0
                        ), "4096 block256 ILP2 path requires exact block geometry"
                        comptime wide_fixed_turns = 4096 // (
                            wide_block_size * contig_simd
                        )
                        comptime kernel = reduce_kernel_fixed_turns_ilp2[
                            rank,
                            ax,
                            num_reductions,
                            wide_block_size,
                            wide_fixed_turns,
                            input_fn,
                            output_fn,
                            reduce_fn,
                            dtype,
                            contig_simd,
                            pdl_level=pdl_level,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            shape,
                            init,
                            grid_dim=num_blocks,
                            block_dim=wide_block_size,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
                    else:
                        comptime assert (
                            4096 % (BLOCK_SIZE * contig_simd) == 0
                        ), "4096 fixed-turn path requires exact block geometry"
                        comptime fixed_turns = 4096 // (
                            BLOCK_SIZE * contig_simd
                        )
                        comptime kernel = reduce_kernel_fixed_turns_ilp4[
                            rank,
                            ax,
                            num_reductions,
                            BLOCK_SIZE,
                            fixed_turns,
                            input_fn,
                            output_fn,
                            reduce_fn,
                            dtype,
                            contig_simd,
                            pdl_level=pdl_level,
                        ]
                        ctx.enqueue_function[kernel, kernel](
                            shape,
                            init,
                            grid_dim=num_blocks,
                            block_dim=BLOCK_SIZE,
                            attributes=pdl_launch_attributes(pdl_level),
                        )
        else:
            comptime for ax in range(rank):
                if axis == ax:
                    comptime is_contig = (ax == rank - 1)
                    comptime reduce_simd = contig_simd if is_contig else packing_factor
                    comptime kernel = reduce_kernel[
                        rank,
                        ax,
                        num_reductions,
                        BLOCK_SIZE,
                        input_fn,
                        output_fn,
                        reduce_fn,
                        dtype,
                        reduce_simd,
                        pdl_level=pdl_level,
                    ]
                    ctx.enqueue_function[kernel, kernel](
                        shape,
                        init,
                        grid_dim=num_blocks,
                        block_dim=BLOCK_SIZE,
                        attributes=pdl_launch_attributes(pdl_level),
                    )


@always_inline
def _reduce_generator_gpu[
    num_reductions: Int,
    init_type: DType,
    input_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank]
    ) capturing[_] -> SIMD[dtype, width],
    output_0_fn: def[dtype: DType, width: Int, rank: Int](
        IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
    ) capturing[_] -> None,
    reduce_function: def[ty: DType, width: Int, reduction_idx: Int](
        SIMD[ty, width], SIMD[ty, width]
    ) capturing[_] -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
](
    shape: IndexList[_, element_type=DType.int64],
    init: StaticTuple[Scalar[init_type], num_reductions],
    reduce_dim: Int,
    ctx: DeviceContext,
) raises:
    """Reduce the given tensor using the given reduction function on GPU. The
    num_reductions parameter enables callers to execute fused reductions. The
    reduce_0_fn and output_0_fn should be implemented in a way which routes
    between the fused reduction methods using their reduction_idx parameter.

    Parameters:
        num_reductions: The number of fused reductions to perform.
        init_type: The initial accumulator value for each reduction.
        input_0_fn: The lambda to use to access the incoming tensor.
        output_0_fn: The lambda to use to storing to the output tensor.
        reduce_function: The lambda implementing the reduction.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.
        pdl_level: The PDL level controlling kernel overlap behavior.

    Args:
        shape: The shape of the tensor we are reducing.
        init: The value to start the reduction from.
        reduce_dim: The dimension we are reducing.
        ctx: The pointer to DeviceContext.

    Raises:
        If the GPU kernel launch fails.
    """

    var reduce_dim_normalized = (
        len(shape) + reduce_dim
    ) if reduce_dim < 0 else reduce_dim

    reduce_launch[
        num_reductions,
        input_0_fn,
        output_0_fn,
        reduce_function,
        shape.size,
        init_type,
        pdl_level,
    ](shape, reduce_dim_normalized, init, ctx)
