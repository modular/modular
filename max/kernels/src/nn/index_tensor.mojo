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
"""Implements numpy-style advanced tensor indexing (getitem and setitem) for CPU and GPU."""

from std.math import ceildiv
from std.sys import simd_width_of
from std.sys.info import CompilationTarget

from nn.reshape import reshape
from max.algorithm import elementwise, sync_parallelize
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.host.info import is_cpu
from layout import Coord, TileTensor, coord_to_index_list, row_major
from max.runtime.asyncrt import parallelism_level

from std.utils import IndexList


@inline(.always)
def index_tensor_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    indices_buf: TileTensor[mut=False, indices_type, ...],
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `index_tensor` operation, and assert the
    inputs are compatible.

    Parameters:
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        batch_dims: Batch dimensions.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.

    Returns:
        The output shape.
    """

    # TODO: Revisit when we generalize (e.g.res[: indA] vs. res[:, indA, indB]).
    if input_buf.rank <= 1 or indices_buf.rank <= 1:
        raise Error("[index_tensor] input_rank and indices_rank must be >= 2")
    if batch_dims + indices_buf.rank != input_buf.rank:
        raise Error(
            "Sum of batch_dims and indices_rank needs to equal input_rank"
        )

    # Since we pass indices without the batch_dims dimensions (since they do
    # not need to be materialized), we need to construct the indices_shape as
    # follows for the purposes of calculating the index.tensor shape:
    comptime combined_indices_rank = batch_dims + indices_buf.rank
    var indices_shape = IndexList[combined_indices_rank]()

    comptime for i in range(batch_dims):
        indices_shape[i] = Int(input_buf.dim[i]())

    comptime for i in range(indices_buf.rank):
        indices_shape[batch_dims + i] = Int(indices_buf.dim[i]())

    var index_size = indices_shape[combined_indices_rank - 1]
    # TODO: Revisit when we generalize (see above TODO).
    if index_size < 2 or input_buf.rank - batch_dims < index_size:
        raise Error(
            "[index_tensor] index size must be within range [2, input_rank -"
            " batch_dims]"
        )
    # TODO: Revisit keeping when we generalize.
    if batch_dims >= combined_indices_rank:
        raise Error(
            "[index_tensor] requires (batch_dims < indices_rank + batch_dims)"
        )

    # compute and return the output shape
    var output_shape = IndexList[output_rank]()
    var next_out_dim = 0

    var input_shape = coord_to_index_list(input_buf.layout.shape_coord())

    comptime for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    comptime for i in range(batch_dims, combined_indices_rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    if indices_shape[combined_indices_rank - 1] == input_buf.rank - batch_dims:
        return output_shape

    # TODO: Revisit cases where/if this applies for generalized index_tensor.
    for i in range(
        batch_dims + indices_shape[combined_indices_rank - 1],
        len(input_shape),
    ):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===-----------------------------------------------------------------------===#
# index_tensor
# ===-----------------------------------------------------------------------===#

# TODO:
# Need to limit to cases where : is in consecutive dimensions starting from 0th.
# (so it does NOT work with non-contiguous case).
# This needs to get the TWO indices as part of the OP itself (?)
#   See if it makes sense to leave like this to be generic and lowering can
#   deal with subcases (e.g., N-D indices).
# FOLLOW-UP: Revisit all constrained and raises in dimensions and values of indices.
#        When we support more general cases like below.
# FOLLOW-UP: See if it works with 2D indices case.
# FOLLOW-UP: See example with [:, indA] indexing.
# FOLLOW-UP: Simplify if not needed to be that complex.
# Note: We could have used original gather_nd but then would need to materialize
# an unneeded huge index tensor (would broadcast to : dimension(s)).
# Note: Currently, the `_index_tensor_1d` is retained as the CPU implementation
# (see PR #38365). The `_index_tensor_impl` is introduced as the gpu implementation.
# We intend to merge `index_tensor` with the `gather_nd` operations in the future.


def index_tensor[
    dtype: DType,
    indices_type: DType,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
) raises:
    """
    Index_tensor operation; based on modified implementation of gather_nd.

    Parameters:
        dtype: Type of data tensor.
        indices_type: Type of indices tensor.
        batch_dims: Number of batch dimensions. The gather of indexing
                    starts from dimension of data[batch_dims:].
        target: The target architecture to execute on.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank >= 1. All index values are expected
                 to be within bounds [-s, s-1] along axis of size s. It is an
                 error if any of the index values are out of bounds.
        output: Tensor of rank data_rank + indices_rank - indices_shape[-1] - 1 - b.
        ctx: The device context as prepared by the graph compiler.

    """

    comptime if is_cpu[target]():
        return _index_tensor_1d[
            batch_dims,
            target=target,
        ](data, indices, output, Optional(ctx.assert_cpu_target()))
    else:
        return _index_tensor_impl[
            batch_dims,
            target=target,
        ](data, indices, output, ctx)


# Note: this is an extremely specialized version of the kernel that only handles
# the [:, :, x, y] case where x and y are 1D tensors.
# Batch dims refer to the number of sliced dimensions at the beginning
def _index_tensor_1d[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: Optional[DeviceContext.CPU] = None,
):
    comptime assert (
        data.flat_rank >= 2 and indices.flat_rank == 2
    ), "Constraint: data_rank >= 2 and indices_rank == 2"
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...) loads below.
    comptime assert indices.flat_rank >= 2

    var last_index_dim = Int(indices.dim(indices.rank - 1))

    assert (
        last_index_dim + batch_dims == data.rank
    ), "kernel doesn't support slicing after specified dims"

    var data_shape = coord_to_index_list(data.layout.shape_coord())
    var batch_volume: Int = 1

    comptime for i in range(batch_dims):
        batch_volume *= data_shape[i]

    # Flatten data to array of shape (batch_dim_size, data.shape[batch_dims:])
    comptime reshaped_data_rank = data.rank - batch_dims + 1
    var reshaped_data_tuple = IndexList[reshaped_data_rank]()

    reshaped_data_tuple[0] = batch_volume
    var counter = 1
    for i in range(batch_dims, data.rank):
        reshaped_data_tuple[counter] = data_shape[i]
        counter += 1

    var reshaped_data = reshape[reshaped_data_rank](
        data.make_dynamic[.int64](),
        reshaped_data_tuple,
    )

    # TODO: Find a heuristic to replace the magic number
    #       to also take into account the data size per line.
    comptime MIN_LINES = 32
    var num_threads = parallelism_level(ctx)
    var num_tasks = min(
        ceildiv(
            batch_volume,
            MIN_LINES,
        ),
        num_threads,
    )
    var work_per_thread = ceildiv(batch_volume, num_tasks)

    def calc_batch_dim(
        task_id: Int,
    ) {var work_per_thread, var batch_volume, var last_index_dim, imm}:
        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var work_start = task_id * work_per_thread
        var work_end = min((task_id + 1) * work_per_thread, batch_volume)
        var output_flat = output.reshape(row_major(output.num_elements()))

        for i in range(work_start, work_end):
            for j in range(Int(indices.dim(0))):
                var data_coord = IndexList[reshaped_data_rank]()
                data_coord[0] = i
                for k in range(last_index_dim):
                    data_coord[k + 1] = Int(indices.load((j, k)))

                var rd_coord = Coord(data_coord)
                output_flat.store(
                    (i * Int(indices.dim(0)) + j,),
                    reshaped_data.load(rd_coord),
                )

    sync_parallelize(calc_batch_dim, num_tasks, ctx)


def _index_tensor_impl[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime assert (
        data.flat_rank >= 2 and indices.flat_rank >= 2
    ), "Constraint: data_rank >= 2 and indices_rank >= 2"

    # This is modeled as an elementwise function mapping an index in the
    # output to an index in the input
    def index_tensor_elementwise_fn[
        simd_width: Int, alignment: Int = 1
    ](output_idx_arg: Coord) {var}:
        var output_idx = IndexList[output.rank]()
        comptime for i in range(output.rank):
            output_idx[i] = Int(output_idx_arg[i])
        var data_idx = IndexList[data.rank]()
        var indices_idx = IndexList[indices.rank]()
        var indices_last_dim = Int(indices.dim[indices.rank - 1]())

        # Fill in the known dimensions in our batch_dim
        comptime for i in range(batch_dims):
            data_idx[i] = output_idx[i]

        # Start filling in the index into the indices buffer
        comptime for i in range(0, indices.rank - 1):
            indices_idx[i] = output_idx[batch_dims + i]

        # walk the last dimensions, which are the slices we're gathering
        for i in range(indices_last_dim):
            indices_idx[indices.rank - 1] = i
            var coord = Coord(indices_idx)
            data_idx[batch_dims + i] = Int(indices.load(coord))

        # fill in the last slices in the input
        var num_tail_elems = data.rank - batch_dims - indices_last_dim
        var output_start = output.rank - num_tail_elems
        var src_start = indices_last_dim + batch_dims
        for i in range(0, num_tail_elems):
            data_idx[src_start + i] = output_idx[output_start + i]

        var data_coord = Coord(data_idx)
        var out_coord = Coord(output_idx)
        output.store[alignment=1](
            out_coord, data.load[width=simd_width, alignment=1](data_coord)
        )

    comptime target_simd_width = (
        simd_width_of[dtype, target=CompilationTarget.current()]() if is_cpu[
            target
        ]() else simd_width_of[dtype, target=get_gpu_target()]()
    )

    # Only use SIMD if:
    #   - the input data is contiguous
    #   - the slices at the end of the input are not scalars
    #   - the last dimension of the slices are evenly divisible by simd_width
    var slice_rank = (
        Scalar[indices.linear_idx_type](data.rank - batch_dims)
        - indices.dim[indices.rank - 1]()
    )
    var slice_last_dim = output.dim[output.rank - 1]() if slice_rank > 0 else 1

    comptime assert data.rank > 0
    var use_simd = (
        data.static_stride[data.rank - 1] == 1
        and (slice_last_dim % Scalar[output.linear_idx_type](target_simd_width))
        == 0
    )

    comptime if is_cpu[target]():
        var cpu_ctx = DeviceContext(api="cpu")
        if use_simd:
            elementwise[
                target_simd_width,
                target=target,
            ](index_tensor_elementwise_fn, output.layout.shape_coord(), cpu_ctx)
        else:
            elementwise[
                1,
                target=target,
            ](index_tensor_elementwise_fn, output.layout.shape_coord(), cpu_ctx)
    else:
        assert Bool(ctx), "Must provide DeviceContext if executing on GPU."
        var cuda_ctx = ctx.value()
        if use_simd:
            elementwise[
                target_simd_width,
                target=target,
            ](
                index_tensor_elementwise_fn,
                output.layout.shape_coord(),
                cuda_ctx,
            )
        else:
            elementwise[
                1,
                target=target,
            ](
                index_tensor_elementwise_fn,
                output.layout.shape_coord(),
                cuda_ctx,
            )
