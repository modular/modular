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

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.math.uutils import udivmod
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# stencil GPU implementation
# ===-----------------------------------------------------------------------===#


def _stencil_impl_gpu[
    shape_element_type: DType,
    input_shape_element_type: DType,
    //,
    rank: Int,
    stencil_rank: Int,
    stencil_axis: IndexList[stencil_rank, ...],
    simd_width: Int,
    dtype: DType,
    map_fn: def(IndexList[stencil_rank, ...]) capturing[_] -> Tuple[
        IndexList[stencil_rank],
        IndexList[stencil_rank],
    ],
    map_strides: def(dim: Int) capturing[_] -> Int,
    load_fn: def[simd_width: Int, dtype: DType](IndexList[rank, ...]) capturing[
        _
    ] -> SIMD[dtype, simd_width],
    compute_init_fn: def[simd_width: Int]() capturing[_] -> SIMD[
        dtype, simd_width
    ],
    compute_fn: def[simd_width: Int](
        IndexList[rank, ...],
        SIMD[dtype, simd_width],
        SIMD[dtype, simd_width],
    ) capturing[_] -> SIMD[dtype, simd_width],
    compute_finalize_fn: def[simd_width: Int](
        IndexList[rank, ...], SIMD[dtype, simd_width]
    ) capturing[_] -> None,
](
    ctx: DeviceContext,
    shape: IndexList[rank, element_type=shape_element_type],
    input_shape: IndexList[rank, element_type=input_shape_element_type],
) raises:
    """(Naive implementation) Computes stencil operation in parallel on GPU.

    Parameters:
        shape_element_type: The element dtype of the shape.
        input_shape_element_type: The element dtype of the input shape.
        rank: Input and output domain rank.
        stencil_rank: Rank of stencil subdomain slice.
        stencil_axis: Stencil subdomain axes.
        simd_width: The SIMD vector width to use.
        dtype: The input and output data dtype.
        map_fn: A function that a point in the output domain to the input co-domain.
        map_strides: A function that returns the stride for the dim.
        load_fn: A function that loads a vector of simd_width from input.
        compute_init_fn: A function that initializes vector compute over the stencil.
        compute_fn: A function the process the value computed for each point in the stencil.
        compute_finalize_fn: A function that finalizes the computation of a point in the output domain given a stencil.

    Args:
        ctx: The DeviceContext to use for GPU execution.
        shape: The shape of the output buffer.
        input_shape: The shape of the input buffer.

    Raises:
        If the GPU kernel launch fails.
    """
    comptime assert rank == 4, "Only stencil of rank-4 supported"
    comptime assert (
        stencil_axis[0] == 1 and stencil_axis[1] == 2
    ), "Only stencil spatial axes [1, 2] are supported"

    # GPU kernel implementation
    @always_inline
    @parameter
    def stencil_kernel():
        # Get thread indices
        var tid_x = thread_idx.x
        var tid_y = thread_idx.y
        var bid_x = block_idx.x
        var bid_y = block_idx.y
        var bid_z = block_idx.z

        # Calculate global indices
        var x = bid_x * block_dim.x + tid_x
        var y = bid_y * block_dim.y + tid_y

        # Calculate batch and channel from bid_z
        var batch_idx, channel = udivmod(bid_z, shape[3])

        # Early exit if outside bounds
        if x >= shape[2] or y >= shape[1]:
            return

        # Create output point indices with computed batch and channel
        var indices = IndexList[rank, element_type=shape_element_type](
            batch_idx, y, x, channel
        )

        # Process stencil for this point
        var stencil_indices = IndexList[
            stencil_rank, element_type=stencil_axis.element_type
        ](indices[stencil_axis[0]], indices[stencil_axis[1]])
        var bounds = map_fn(stencil_indices)
        var lower_bound = bounds[0]
        var upper_bound = bounds[1]
        var step_i = map_strides(0)
        var step_j = map_strides(1)
        var result = compute_init_fn[simd_width]()
        var input_height = input_shape[1]
        var input_width = input_shape[2]

        # Handle boundary conditions
        if lower_bound[0] < 0:
            var mul_i = ceildiv(-lower_bound[0], step_i)
            lower_bound[0] = lower_bound[0] + mul_i * step_i
        if lower_bound[1] < 0:
            var mul_j = ceildiv(-lower_bound[1], step_j)
            lower_bound[1] = lower_bound[1] + mul_j * step_j

        # Process stencil window
        for i in range(
            lower_bound[0],
            min(input_height, upper_bound[0]),
            step_i,
        ):
            for j in range(
                lower_bound[1],
                min(input_width, upper_bound[1]),
                step_j,
            ):
                var point_idx = IndexList[
                    rank, element_type=shape_element_type
                ](indices[0], i, j, indices[3])
                var val = load_fn[simd_width, dtype](point_idx)
                result = compute_fn[simd_width](point_idx, result, val)

        compute_finalize_fn[simd_width](indices, result)

    # Calculate grid and block dimensions
    var block_dim = (32, 32, 1)
    var grid_dim = (
        ceildiv(shape[2], block_dim[0]),  # width
        ceildiv(shape[1], block_dim[1]),  # height
        shape[0] * shape[3],  # batch_size * num_channels
    )

    # Compile and launch kernel
    ctx.enqueue_function[stencil_kernel, stencil_kernel](
        grid_dim=grid_dim, block_dim=block_dim
    )
