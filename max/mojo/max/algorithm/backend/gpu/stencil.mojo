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
"""GPU implementation of stencil computation."""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.math.uutils import udivmod
from std.utils.coord import Coord, CoordLike, DynamicCoord


# ===-----------------------------------------------------------------------===#
# stencil GPU implementation
# ===-----------------------------------------------------------------------===#


def _stencil_impl_gpu[
    shape_types: TypeList[Trait=CoordLike, ...],
    input_shape_types: TypeList[Trait=CoordLike, ...],
    //,
    rank: Int,
    stencil_rank: Int,
    simd_width: Int,
    dtype: DType,
    MapFnType: ImplicitlyCopyable
    & RegisterPassable
    & def(DynamicCoord[DType.int, stencil_rank]) -> Tuple[
        DynamicCoord[DType.int, stencil_rank],
        DynamicCoord[DType.int, stencil_rank],
    ],
    MapStridesType: ImplicitlyCopyable
    & RegisterPassable
    & def(dim: Int) -> Int,
    LoadFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: Int, dtype: DType](Coord) -> SIMD[dtype, simd_width],
    ComputeInitFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: Int]() -> SIMD[dtype, simd_width],
    ComputeFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: SIMDLength](
        Coord, SIMD[dtype, simd_width], SIMD[dtype, simd_width]
    ) -> SIMD[dtype, simd_width],
    ComputeFinalizeFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[simd_width: SIMDLength](Coord, SIMD[dtype, simd_width]) -> None,
](
    ctx: DeviceContext,
    shape: Coord[*shape_types],
    input_shape: Coord[*input_shape_types],
    map_func: MapFnType,
    map_strides_func: MapStridesType,
    load_func: LoadFnType,
    compute_init_func: ComputeInitFnType,
    compute_func: ComputeFnType,
    compute_finalize_func: ComputeFinalizeFnType,
) raises:
    """(Naive implementation) Computes stencil operation in parallel on GPU.

    Parameters:
        shape_types: The element types of the `shape` coordinate.
        input_shape_types: The element types of the `input_shape` coordinate.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice. Must be 2: the stencil
            covers the spatial axes 1 and 2.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        MapFnType: Maps a point in the output domain to input co-domain bounds.
        MapStridesType: Returns the stride for each dimension.
        LoadFnType: Loads a SIMD vector from input.
        ComputeInitFnType: Initializes the stencil accumulator.
        ComputeFnType: Processes the value computed for each stencil point.
        ComputeFinalizeFnType: Finalizes the output value from the stencil result.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.
        map_func: Closure mapping output points to input co-domain bounds.
        map_strides_func: Closure returning the stride for a given dimension.
        load_func: Closure loading a SIMD vector from input.
        compute_init_func: Closure initializing the stencil accumulator.
        compute_func: Closure processing each stencil point.
        compute_finalize_func: Closure finalizing the output value.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert rank == 4, "Only stencil of rank-4 supported"
    comptime assert (
        type_of(shape).rank == rank and type_of(input_shape).rank == rank
    ), "shape and input_shape must have rank `rank`"
    comptime assert (
        stencil_rank == 2
    ), "Only stencils over the spatial axes [1, 2] are supported"

    # A vector spans `simd_width` adjacent channels, so it must not run past the
    # end of the channel axis: the channel count has to be a multiple of the
    # requested width.
    if simd_width > 1 and Int(shape[3]) % simd_width != 0:
        raise Error(
            "stencil_gpu: channel count",
            Int(shape[3]),
            "is not a multiple of the vector width",
            simd_width,
        )

    var channel_blocks = ceildiv(Int(shape[3]), simd_width)
    comptime block_x = 32
    comptime block_y = 32
    var grid_dim = (
        ceildiv(Int(shape[2]), block_x),  # width
        ceildiv(Int(shape[1]), block_y),  # height
        Int(shape[0]) * channel_blocks,  # batch_size * channel blocks
    )

    # GPU kernel implementation
    @inline(.always)
    def stencil_kernel() {
        var shape,
        var channel_blocks,
        var input_shape,
        var map_func,
        var map_strides_func,
        var load_func,
        var compute_init_func,
        var compute_func,
        var compute_finalize_func,
    }:
        # Get thread indices
        var tid_x = thread_idx.x
        var tid_y = thread_idx.y
        var bid_x = block_idx.x
        var bid_y = block_idx.y
        var bid_z = block_idx.z

        # Calculate global indices
        var x = bid_x * block_x + tid_x
        var y = bid_y * block_y + tid_y

        # Channels are the innermost (contiguous) axis, so a thread owns
        # `simd_width` adjacent channels of a single output point. The channel
        # count is guaranteed to be a multiple of `simd_width` by the caller.
        var batch_idx, channel_block = udivmod(bid_z, channel_blocks)
        var channel = channel_block * simd_width

        # Early exit if outside bounds
        if x >= Int(shape[2]) or y >= Int(shape[1]):
            return

        var indices = Coord(batch_idx, y, x, channel)

        # Process stencil for this point
        var bounds = map_func(
            rebind[DynamicCoord[DType.int, stencil_rank]](Coord(y, x))
        )
        var lower_i = Int(bounds[0][0])
        var lower_j = Int(bounds[0][1])
        var upper_i = Int(bounds[1][0])
        var upper_j = Int(bounds[1][1])
        var step_i = map_strides_func(0)
        var step_j = map_strides_func(1)
        var result = compute_init_func[simd_width]()
        var input_height = Int(input_shape[1])
        var input_width = Int(input_shape[2])

        # Handle boundary conditions
        if lower_i < 0:
            lower_i += ceildiv(-lower_i, step_i) * step_i
        if lower_j < 0:
            lower_j += ceildiv(-lower_j, step_j) * step_j

        # Process stencil window
        for i in range(lower_i, min(input_height, upper_i), step_i):
            for j in range(lower_j, min(input_width, upper_j), step_j):
                var point_idx = Coord(indices[0], i, j, indices[3])
                var val = load_func[simd_width, dtype](point_idx)
                result = compute_func[simd_width](point_idx, result, val)

        compute_finalize_func[simd_width](indices, result)

    # Compile and launch kernel
    ctx.enqueue_closure(
        stencil_kernel, grid_dim=grid_dim, block_dim=(block_x, block_y, 1)
    )
