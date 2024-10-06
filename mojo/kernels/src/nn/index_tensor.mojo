# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv

from algorithm import sync_parallelize
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer, memcpy, memset_zero
from nn.gather_scatter import normalize_neg_index
from runtime.asyncrt import parallelism_level

from utils import Index, IndexList


@always_inline
fn index_tensor_shape[
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
    single_thread_blocking_override: Bool = True,
](
    input_buf: NDBuffer[input_type, input_rank],
    indices_buf: NDBuffer[indices_type, indices_rank],
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `index_tensor` operation, and assert the
    inputs are compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        indices_rank: Rank of the indices tensor.
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        batch_dims: Batch dimensions.
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.

    Returns:
        The output shape.
    """

    # TODO: Revisit when we generalize (e.g.res[: indA] vs. res[:, indA, indB]).
    if input_rank <= 1 or indices_rank <= 1:
        raise Error("[index_tensor] input_rank and indices_rank must be >= 2")
    if batch_dims + indices_rank != input_rank:
        raise Error(
            "Sum of batch_dims and indices_rank needs to equal input_rank"
        )

    # Since we pass indices without the batch_dims dimensions (since they do
    # not need to be materialized), we need to construct the indices_shape as
    # follows for the purposes of calculating the index.tensor shape:
    alias combined_indices_rank = batch_dims + indices_rank
    var indices_shape = IndexList[combined_indices_rank]()

    @parameter
    for i in range(batch_dims):
        indices_shape[i] = input_buf.get_shape()[i]

    @parameter
    for i in range(indices_rank):
        indices_shape[batch_dims + i] = indices_buf.get_shape()[i]

    var index_size = indices_shape[combined_indices_rank - 1]
    # TODO: Revisit when we generalize (see above TODO).
    if index_size < 2 or input_rank - batch_dims < index_size:
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

    var input_shape = input_buf.get_shape()

    @parameter
    for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    @parameter
    for i in range(batch_dims, combined_indices_rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    if indices_shape[combined_indices_rank - 1] == input_rank - batch_dims:
        return output_shape

    # TODO: Revisit cases where/if this applies for generalized index_tensor.
    for i in range(
        batch_dims + indices_shape[combined_indices_rank - 1],
        len(input_shape),
    ):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===----------------------------------------------------------------------===#
# index_tensor
# ===----------------------------------------------------------------------===#

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
# an unneded huge index tensor (would broadcast to : dimension(s)).


# Note: this is an extremely specialized version of the kernel that only handles
# the [:, :, x, y] case where x and y are are 1D tensors.
# Batch dims refer to the number of sliced dimensions at the beginning
fn index_tensor_1d[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
):
    constrained[
        data_rank >= 2 and indices_rank == 2,
        "Constraint: data_rank >= 2 and indices_rank == 2",
    ]()

    var last_index_dim = indices.get_shape()[indices_rank - 1]

    debug_assert(
        last_index_dim + batch_dims == data_rank,
        "kernel doesn't support slicing after specified dims",
    )

    var data_shape = data.get_shape()
    var batch_volume: Int = 1

    @parameter
    for i in range(batch_dims):
        batch_volume *= data_shape[i]

    # Flatten data to array of shape (batch_dim_size, data.shape[batch_dims:])
    alias reshaped_data_rank = data_rank - batch_dims + 1
    var reshaped_data_tuple = IndexList[reshaped_data_rank]()

    reshaped_data_tuple[0] = batch_volume
    var counter = 1
    for i in range(batch_dims, data_rank):
        reshaped_data_tuple[counter] = data_shape[i]
        counter += 1

    var reshaped_data = reshape.reshape[reshaped_data_rank](
        data.make_dims_unknown(), reshaped_data_tuple
    )

    # TODO: Find a heuristic to replace the magic number
    #       to also take into account the data size per line.
    alias MIN_LINES = 32
    var num_threads = parallelism_level()
    var num_tasks = min(
        ceildiv(
            batch_volume,
            MIN_LINES,
        ),
        num_threads,
    )
    var work_per_thread = ceildiv(batch_volume, num_tasks)

    @__copy_capture(work_per_thread, batch_volume, last_index_dim)
    @parameter
    fn calc_batch_dim(task_id: Int):
        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var work_start = task_id * work_per_thread
        var work_end = min((task_id + 1) * work_per_thread, batch_volume)

        for i in range(work_start, work_end):
            for j in range(indices.get_shape()[0]):
                var data_coord = IndexList[reshaped_data_rank]()
                data_coord[0] = i
                for k in range(last_index_dim):
                    data_coord[k + 1] = int(
                        indices[IndexList[indices_rank](j, k)]
                    )

                output.data[i * indices.get_shape()[0] + j] = reshaped_data[
                    data_coord
                ]

    sync_parallelize[calc_batch_dim](num_tasks)


fn index_tensor[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
):
    """
    Index_tensor operation; based on modified implementation of gather_nd.

    Parameters:
        type: Type of data tensor.
        indices_type: Type of indices tensor.
        data_rank: Rank of data tensor (data_rank >= 1).
        indices_rank: Rank of indices tensor (indices_rank >= 1).
        output_rank: Rank of output tensor.
        batch_dims: Number of batch dimensions. The gather of indexing
                    starts from dimension of data[batch_dims:].

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank >= 1. All index values are expected
                 to be within bounds [-s, s-1] along axis of size s. It is an
                 error if any of the index values are out of bounds.
        output: Tensor of rank data_rank + indices_rank - indices_shape[-1] - 1 - b.

    """

    constrained[
        data_rank >= 2 and indices_rank >= 2,
        "Constraint: data_rank >= 2 and indices_rank >= 2",
    ]()

    # Since we pass indices without the batch_dims dimensions (since they do
    # not need to be materialized), we need to construct the indices_shape as
    # follows for the purposes of calculating the index.tensor shape:
    alias combined_indices_rank = batch_dims + indices_rank
    var indices_shape = IndexList[combined_indices_rank]()

    @parameter
    for i in range(batch_dims):
        indices_shape[i] = data.get_shape()[i]

    @parameter
    for i in range(indices_rank):
        indices_shape[batch_dims + i] = indices.get_shape()[i]
    debug_assert(
        2 <= indices_shape[combined_indices_rank - 1] <= data_rank - batch_dims,
        "Constraint: 2 <= indices_shape[-1] <= data_rank - batch_dims",
    )

    # The number of elements in the batch_dims for data/indices array.
    # E.g., if batch_dims = 2 (always is the outermost dimensions), and the
    #       dimensions of data are [2,3,...], then batch_dims_size = 6
    var batch_dims_size = 1
    for i in range(batch_dims):
        batch_dims_size = batch_dims_size * indices_shape[i]

    var last_shape_of_indices = indices_shape[combined_indices_rank - 1]

    # Number of elements is equal to the product of the number of elements on
    # each dimension of indices, and this needs to be multiplied by the implied
    # indices in the batch dimensions.
    var num_elems = indices.num_elements()
    for i in range(batch_dims):
        num_elems *= data.get_shape()[i]

    # Conceptually reshape indices array, as 3D array. All batch_dims_size
    # elements go to the outermost dimension, and elements of amount equal to
    # indices.shape[-1] go to the innermost.
    # Equivalent to numpy:
    # reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])
    # Note: In normal gather_nd, we would construct a reshaped_indices NDBuffer
    #       with the shape below. In the case of index_tensor, we don't need to
    #       it since the first batch_dims dimensions are the same and we don't
    #       want to waste memory.
    var reshaped_indices_shape = IndexList[3](
        batch_dims_size,
        num_elems // (batch_dims_size * last_shape_of_indices),
        last_shape_of_indices,
    )

    var data_shape = data.get_shape()

    # Flatten data to array of shape (batch_dim_size, data.shape[batch_dims:])
    alias reshaped_data_rank = data_rank - batch_dims + 1
    var reshaped_data_tuple = IndexList[reshaped_data_rank]()
    # Calculate the dimensions of reshaped_data.
    reshaped_data_tuple[0] = batch_dims_size
    var counter = 1
    for i in range(batch_dims, data_rank):
        reshaped_data_tuple[counter] = data_shape[i]
        counter += 1

    # Do the actual reshaping.
    var reshaped_data = reshape.reshape[reshaped_data_rank](
        data.make_dims_unknown(), reshaped_data_tuple
    )
    var reshaped_data_shape = reshaped_data.get_shape()

    # idx[] stores the index from where to gather the requested elements.
    var idx_ptr = UnsafePointer[Scalar[DType.index]].alloc(
        reshaped_indices_shape[2]
    )
    var idx = NDBuffer[DType.index, 1](
        idx_ptr, Index(reshaped_indices_shape[2])
    )

    # Depending on r_minus_m = data_rank - last_shape_of_indices - batch_dims,
    # we will be copying (gather):
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    var r_minus_m = data_rank - last_shape_of_indices - batch_dims
    # Calculate how many elements to copy (this is from the innermost
    # dimensions, and is continuous memory locations).
    var count_copy = 1
    for i in range(r_minus_m):
        count_copy = (
            count_copy * reshaped_data_shape[reshaped_data_rank - 1 - i]
        )
    # Stores the full index on reshaped_data, where to copy from.
    # It is constructed within the nested loop below.
    var start_tensor = NDBuffer[
        DType.index,
        1,
        DimList(reshaped_data_rank),
    ]().stack_allocation()
    # Zeroing here to avoid doing it selectively within the nested loop below.
    memset_zero(start_tensor.data, reshaped_data_rank)

    var output_buffer_copy_ind = 0
    for batch_dim in range(reshaped_indices_shape[0]):
        for outer_dim in range(reshaped_indices_shape[1]):
            # Construct the tuple (all dimensions except outermost, which is
            # the batches dimension - recall all batch dimensions are reshaped
            # into one - the outermost).
            for constr in range(reshaped_indices_shape[2]):
                var input_ax_dim = reshaped_data.get_shape()[constr + 1]
                # Note here that the batch_dim index is implied; since we have
                # ':' on batch_dims, the SAME indices[outer_dim, constr] is
                # reused.
                var idx_on_axis = indices[outer_dim, constr]
                idx[constr] = int(
                    normalize_neg_index(idx_on_axis, input_ax_dim)
                )

            # Construct the full index on reshaped_data, where to copy from.
            start_tensor[0] = batch_dim
            var start_index = 1
            for dim in range(len(idx)):
                start_tensor[start_index] = idx[dim]
                start_index = start_index + 1

            # Calculate the input_offset from where to copy the data.
            var input_offset = 0
            for i in range(reshaped_data_rank):
                input_offset = input_offset + reshaped_data.stride(i) * int(
                    start_tensor[i]
                )
            # Calculate the output_offset where to copy the data.
            var output_offset = output_buffer_copy_ind * (count_copy)
            output_buffer_copy_ind = output_buffer_copy_ind + 1

            # Perform the actual copy of element/slice/sheet/cuboid/etc.
            memcpy(
                output.data + output_offset,
                reshaped_data.data + input_offset,
                count_copy,
            )
    idx_ptr.free()
