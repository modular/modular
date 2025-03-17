# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Optional
from math import ceildiv
from sys import simdwidthof
from sys.info import _current_target
from collections.string import StaticString

from algorithm import elementwise, sync_parallelize
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from memory import UnsafePointer, memcpy, memset_zero
from nn.gather_scatter import normalize_neg_index
from runtime.asyncrt import DeviceContextPtr, parallelism_level

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
# an unneded huge index tensor (would broadcast to : dimension(s)).
# Note: Currently, the `_index_tensor_1d` is retained as the CPU implemetation
# (see PR #38365). The `_index_tensor_impl` is introduced as the gpu implemetation.
# We intend to merge `index_tensor` with the `gather_nd` operations in the future.


fn index_tensor[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: DeviceContextPtr,
) raises:
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
        target: The target architecture to execute on.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank >= 1. All index values are expected
                 to be within bounds [-s, s-1] along axis of size s. It is an
                 error if any of the index values are out of bounds.
        output: Tensor of rank data_rank + indices_rank - indices_shape[-1] - 1 - b.
        ctx: The DeviceContextPtr as prepared by the graph compiler.

    """

    @parameter
    if is_cpu[target]():
        return _index_tensor_1d[
            batch_dims,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](data, indices, output)
    else:
        return _index_tensor_impl[
            batch_dims,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](data, indices, output, ctx.get_device_context())


# Note: this is an extremely specialized version of the kernel that only handles
# the [:, :, x, y] case where x and y are are 1D tensors.
# Batch dims refer to the number of sliced dimensions at the beginning
fn _index_tensor_1d[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int, //,
    batch_dims: Int,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: Optional[DeviceContext] = None,
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
                    data_coord[k + 1] = Int(
                        indices[IndexList[indices_rank](j, k)]
                    )

                output.data[i * indices.get_shape()[0] + j] = reshaped_data[
                    data_coord
                ]

    sync_parallelize[calc_batch_dim](num_tasks)


fn _index_tensor_impl[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int, //,
    batch_dims: Int,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
    ctx: Optional[DeviceContext] = None,
) raises:
    constrained[
        data_rank >= 2 and indices_rank >= 2,
        "Constraint: data_rank >= 2 and indices_rank >= 2",
    ]()

    # This is modeled as an elementwise function mapping an index in the
    # output to an index in the input
    @parameter
    fn index_tensor_elementwise_fn[
        simd_width: Int, rank: Int
    ](output_idx_arg: IndexList[rank]) capturing -> None:
        var output_idx = rebind[IndexList[output_rank]](output_idx_arg)
        var data_idx = IndexList[data_rank]()
        var indices_idx = IndexList[indices_rank]()
        var indices_last_dim = indices.dim[indices_rank - 1]()

        # Fill in the known dimensions in our batch_dim
        @parameter
        for i in range(batch_dims):
            data_idx[i] = output_idx[i]

        # Start filling in the index into the indices buffer
        @parameter
        for i in range(0, indices_rank - 1):
            indices_idx[i] = output_idx[batch_dims + i]

        # walk the last dimensions, which are the slices we're gathering
        for i in range(indices_last_dim):
            indices_idx[indices_rank - 1] = i
            data_idx[batch_dims + i] = Int(indices[indices_idx])

        # fill in the last slices in the input
        num_tail_elems = data_rank - batch_dims - indices_last_dim
        output_start = output_rank - num_tail_elems
        src_start = indices_last_dim + batch_dims
        for i in range(0, num_tail_elems):
            data_idx[src_start + i] = output_idx[output_start + i]

        output.store[width=simd_width](
            output_idx, data.load[width=simd_width](data_idx)
        )

    alias compile_target = _current_target() if is_cpu[
        target
    ]() else _get_gpu_target()
    alias target_simd_width = simdwidthof[type, target=compile_target]()

    # Only use SIMD if:
    #   - the input data is contiguous
    #   - the slices at the end of the input are not scalars
    #   - the last dimension of the slices are evenly divisible by simd_width
    var slice_rank = data_rank - batch_dims - indices.dim[indices_rank - 1]()
    var slice_last_dim = output.dim[output_rank - 1]() if slice_rank > 0 else 1

    var use_simd = data.stride[data_rank - 1]() == 1 and (
        slice_last_dim % target_simd_width
    ) == 0

    @parameter
    if is_cpu[target]():
        if use_simd:
            elementwise[
                index_tensor_elementwise_fn,
                target_simd_width,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output.get_shape())
        else:
            elementwise[
                index_tensor_elementwise_fn,
                1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output.get_shape())
    else:
        debug_assert(
            Bool(ctx), "Must provide DeviceContext if executing on GPU."
        )
        var cuda_ctx = ctx.value()
        if use_simd:
            elementwise[
                index_tensor_elementwise_fn,
                target_simd_width,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output.get_shape(), cuda_ctx)
        else:
            elementwise[
                index_tensor_elementwise_fn,
                1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output.get_shape(), cuda_ctx)
