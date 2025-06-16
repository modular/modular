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

from collections import Optional
from collections.string import StaticString
from math import ceildiv
from sys import simdwidthof
from sys.info import _current_target

from algorithm import elementwise, sync_parallelize
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from memory import UnsafePointer, memcpy, memset_zero
from nn.gather_scatter import normalize_neg_index
from runtime.asyncrt import DeviceContextPtr, parallelism_level

from utils import Index, IndexList, StaticTuple


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
# an unneeded huge index tensor (would broadcast to : dimension(s)).
# Note: Currently, the `_index_tensor_1d` is retained as the CPU implementation
# (see PR #38365). The `_index_tensor_impl` is introduced as the gpu implementation.
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
    output: NDBuffer[mut=True, type, output_rank],
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
# the [:, :, x, y] case where x and y are 1D tensors.
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
    output: NDBuffer[mut=True, type, output_rank],
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
    output: NDBuffer[mut=True, type, output_rank],
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

    var use_simd = (
        data.stride[data_rank - 1]() == 1
        and (slice_last_dim % target_simd_width) == 0
    )

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


# ===-----------------------------------------------------------------------===#
# Advanced Indexing
# ===-----------------------------------------------------------------------===#
@always_inline
fn _advanced_indexing_use_simd[
    start_axis: Int, num_index_tensors: Int, input_rank: Int
](read_strides: IndexList, write_strides: IndexList) -> Bool:
    """Return whether we can use vectorized loads/stores for advanced indexing

    Parameters:
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        input_rank: The rank of the tensor being indexed.

    Args:
        read_strides: The stride of the tensor being read from in advanced indexing.
            In `getitem` this is `input_tensor`, in `setitem` it is `update_tensor`.
        write_strides: The strides of the tensor being written to in advanced indexing.
            In `getitem` this is `out_tensor`, in `setitem` it is `input_tensor`
    """
    # We can vectorize the assignment only if:
    # - The tensors we are reading and writing to are contiguous in inner dimension
    # - We are not directly indexing the inner dimension of input
    alias inner_dim_not_indexed = (start_axis + num_index_tensors - 1) < (
        input_rank - 1
    )
    var read_contiguous = read_strides[read_strides.size - 1] == 1
    var write_contiguous = write_strides[write_strides.size - 1] == 1
    return inner_dim_not_indexed and read_contiguous and write_contiguous


@always_inline
fn advanced_indexing_getitem[
    input_rank: Int,
    index_rank: Int,
    input_type: DType,
    index_type: DType, //,
    start_axis: Int,
    num_index_tensors: Int,
    target: StaticString,
    single_thread_blocking_override: Bool,
    trace_description: StaticString,
    input_tensor_fn: fn[width: Int] (IndexList[input_rank]) capturing -> SIMD[
        input_type, width
    ],
    indices_fn: fn[indices_index: Int] (
        IndexList[index_rank]
    ) capturing -> SIMD[index_type, 1],
](
    out_tensor: NDBuffer[
        mut=True, input_type, input_rank + index_rank - num_index_tensors
    ],
    in_tensor_strides: IndexList[input_rank],
    ctx: DeviceContextPtr,
) raises:
    """Implement basic numpy-style advanced indexing.

    This is designed to be fused with other view-producing operations to
    implement full numpy-indexing semantics.

    This assumes the dimensions in `input_tensor` not indexed by index tensors
    are ":", ie selecting all indices along the slice. For example in numpy:

    ```
    # rank(indices1) == 3
    # rank(indices2) == 3
    out_tensor = input_tensor[:, :, :, indices1, indices2, :, :]
    ```

    We calculate the following for all valid valued indexing variables:

    ```
    out_tensor[a, b, c, i, j, k, d, e] = input_tensor[
        a, b, c,
        indices1[i, j, k],
        indices2[i, j, k],
        d, e
    ]
    ```

    In this example `start_axis = 3` and `num_index_tensors = 2`.

    Parameters:
        input_rank: The rank of the input tensor.
        index_rank: The rank of the indexing tensors.
        input_type: The dtype of the input tensor.
        index_type: The dtype of the indexing tensors.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        target: The target architecture to operation on.
        single_thread_blocking_override: If True, then the operation is run
            synchronously using a single thread.
        trace_description: For profiling, the trace name the operation will
            appear under.
        input_tensor_fn: Fusion lambda for the input tensor.
        indices_fn: Fusion lambda for the indices tensors.

    Args:
        out_tensor: The output tensor to write to.
        in_tensor_strides: The strides of the input tensor.
        ctx: The DeviceContextPtr as prepared by the graph compiler.

    TODO(GEX-1951): Support boolean tensor mask support
    TODO(GEX-1952): Support non-contiguous indexing tensor case
    TODO(GEX-1953): Support fusion (especially view-fusion)
    """
    # Do not support boolean masks at this time.
    constrained[index_type != DType.bool]()

    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, out_tensor_rank: Int
    ](output_index: IndexList[out_tensor_rank]) capturing:
        input_index = IndexList[input_rank]()

        # Find the associated output index from input index
        @parameter
        for input_dim in range(input_rank):

            @parameter
            if input_dim < start_axis:
                input_index[input_dim] = output_index[input_dim]
            elif input_dim >= start_axis + num_index_tensors:
                input_index[input_dim] = output_index[
                    input_dim - num_index_tensors + index_rank
                ]
            else:
                alias index_tensor_offset = input_dim - start_axis
                var index_tensor_indices = IndexList[index_rank]()

                @parameter
                for offset in range(index_rank):
                    index_tensor_indices[offset] = output_index[
                        offset + start_axis
                    ]
                input_index[input_dim] = Int(
                    indices_fn[index_tensor_offset](index_tensor_indices)
                )

        out_tensor.store[width=width](
            rebind[IndexList[out_tensor.rank]](output_index),
            input_tensor_fn[width=width](input_index),
        )

    alias compile_target = _current_target() if is_cpu[
        target
    ]() else _get_gpu_target()
    alias target_simd_width = simdwidthof[input_type, target=compile_target]()
    var use_simd = _advanced_indexing_use_simd[
        start_axis, num_index_tensors, input_rank
    ](read_strides=in_tensor_strides, write_strides=out_tensor.get_strides())
    if use_simd:
        elementwise[
            elementwise_fn_wrapper,
            target_simd_width,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
            _trace_description=trace_description,
        ](out_tensor.get_shape(), ctx)
    else:
        elementwise[
            elementwise_fn_wrapper,
            1,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
            _trace_description=trace_description,
        ](out_tensor.get_shape(), ctx)


@always_inline
fn advanced_indexing_getitem_shape[
    input_rank: Int,
    index_rank: Int, //,
    start_axis: Int,
    num_index_tensors: Int,
](
    input_shape: IndexList[input_rank],
    index_shape: IndexList[index_rank],
) -> IndexList[input_rank + index_rank - num_index_tensors]:
    """Calculate the output shape from advanced indexing.

    Parameters:
        input_rank: The rank of the input tensor.
        index_rank: The rank of the indexing tensors.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.

    Args:
        input_shape: The shape of the input tensor in the operation.
        index_shape: The shape of the indexing tensors in the operation.
    """
    alias output_rank = input_rank + index_rank - num_index_tensors
    var answer = IndexList[output_rank]()

    @parameter
    for i in range(output_rank):
        if i < start_axis:
            answer[i] = input_shape[i]
        elif i >= start_axis + index_rank:
            answer[i] = input_shape[i - index_rank + num_index_tensors]
        else:
            answer[i] = index_shape[i - start_axis]

    return answer


@always_inline
fn advanced_indexing_setitem_inplace[
    input_rank: Int,
    index_rank: Int,
    updates_rank: Int,
    input_type: DType,
    index_type: DType, //,
    start_axis: Int,
    num_index_tensors: Int,
    target: StaticString,
    single_thread_blocking_override: Bool,
    trace_description: StaticString,
    updates_tensor_fn: fn[width: Int] (
        IndexList[updates_rank]
    ) capturing -> SIMD[input_type, width],
    indices_fn: fn[indices_index: Int] (
        IndexList[index_rank]
    ) capturing -> SIMD[index_type, 1],
](
    input_tensor: NDBuffer[mut=True, type=input_type, rank=input_rank],
    index_tensor_shape: IndexList[index_rank, **_],
    updates_tensor_strides: IndexList[updates_rank],
    ctx: DeviceContextPtr,
) raises:
    """Implement basic numpy-style advanced indexing with assignment.

    This is designed to be fused with other view-producing operations to
    implement full numpy-indexing semantics.

    This assumes the dimensions in `input_tensor` not indexed by index tensors
    are ":", ie selecting all indices along the slice. For example in numpy:

    ```
    # rank(indices1) == 2
    # rank(indices2) == 2
    # rank(updates) == 2
    input_tensor[:, :, :, indices1, indices2, :, :] = updates
    ```

    We calculate the following for all valid valued indexing variables:

    ```
    input_tensor[
        a, b, c,
        indices1[i, j],
        indices2[i, j],
        d, e
    ] = updates[i, j]
    ```

    In this example `start_axis = 3` and `num_index_tensors = 2`.

    In terms of implementation details, our strategy is to iterate over
    all indices over a common iteration range. The idea is we can map
    indices in this range to the write location in `input_tensor` as well
    as the data location in `updates`. An update can illustrate how this is
    possible best:

    Imagine the `input_tensor` shape is [A, B, C, D] and we have indexing
    tensors I1 and I2 with shape [M, N, K]. Assume I1 and I2 are applied
    to dimensions 1 and 2.

    I claim an appropriate common iteration range is then (A, M, N, K, D).
    Note we expect `updates` to be the shape [A, M, N, K, D]. We will show
    this by providing the mappings into `updates` and `input_tensor`:

    Consider an arbitrary set of indices in this range (a, m, n, k, d):
        - The index into `updates` is (a, m, n, k, d).
        - The index into `input_tensor` is (a, I1[m, n, k], I2[m, n, k], d).

    Parameters:
        input_rank: The rank of the input tensor.
        index_rank: The rank of the indexing tensors.
        updates_rank: The rank of the updates tensor.
        input_type: The dtype of the input tensor.
        index_type: The dtype of the indexing tensors.
        start_axis: The first dimension in input where the indexing tensors
            are applied. It is assumed the indexing tensors are applied in
            consecutive dimensions.
        num_index_tensors: The number of indexing tensors.
        target: The target architecture to operation on.
        single_thread_blocking_override: If True, then the operation is run
            synchronously using a single thread.
        trace_description: For profiling, the trace name the operation will
            appear under.
        updates_tensor_fn: Fusion lambda for the update tensor.
        indices_fn: Fusion lambda for the indices tensors.

    Args:
        input_tensor: The input tensor being indexed into and modified in-place.
        index_tensor_shape: The shape of each index tensor.
        updates_tensor_strides: The strides of the update tensor.
        ctx: The DeviceContextPtr as prepared by the graph compiler.

    TODO(GEX-1951): Support boolean tensor mask support
    TODO(GEX-1952): Support non-contiguous indexing tensor case
    TODO(GEX-1953): Support fusion (especially view-fusion)
    TODO(GEX-1954): Unify getitem and setitem using generic views.
                    (Requires non-strided view functions).
    """

    # First calculate
    alias iteration_rank = input_rank + index_rank - num_index_tensors
    constrained[iteration_rank == updates_rank]()
    var iteration_shape = IndexList[iteration_rank]()

    # Find the common iteration space
    @parameter
    for i in range(iteration_rank):

        @parameter
        if i < start_axis:
            iteration_shape[i] = input_tensor.get_shape()[i]
        elif i >= start_axis + index_rank:
            iteration_shape[i] = input_tensor.get_shape()[
                i - index_rank + num_index_tensors
            ]
        else:
            iteration_shape[i] = index_tensor_shape[i - start_axis]

    @parameter
    @always_inline
    fn elementwise_fn_wrapper[
        width: Int, iteration_rank: Int
    ](iteration_indices: IndexList[iteration_rank]) capturing:
        var index_tensor_indices = IndexList[index_rank]()

        # Find the index into the indexing tensors from the common index
        @parameter
        for i in range(index_rank):
            index_tensor_indices[i] = iteration_indices[i + start_axis]

        # Find the index into the inputs from the common index
        var input_tensor_indices = IndexList[input_rank]()

        @parameter
        for i in range(input_rank):

            @parameter
            if i < start_axis:
                input_tensor_indices[i] = iteration_indices[i]
            elif i >= start_axis + num_index_tensors:
                input_tensor_indices[i] = iteration_indices[
                    i - num_index_tensors + index_rank
                ]
            else:
                alias index_tensor_offset = i - start_axis
                input_tensor_indices[i] = Int(
                    indices_fn[index_tensor_offset](index_tensor_indices)
                )

        input_tensor.store[width=width](
            input_tensor_indices,
            updates_tensor_fn[width=width](
                rebind[IndexList[updates_rank]](iteration_indices)
            ),
        )

    # We can vectorize the assignment only if we are
    # not indexing in the last dimension of input.
    alias last_indexed_dim = start_axis + num_index_tensors - 1
    alias compile_target = _current_target() if is_cpu[
        target
    ]() else _get_gpu_target()
    alias target_simd_width = simdwidthof[input_type, target=compile_target]()
    var use_simd = _advanced_indexing_use_simd[
        start_axis, num_index_tensors, input_rank
    ](
        read_strides=updates_tensor_strides,
        write_strides=input_tensor.get_strides(),
    )
    if use_simd:
        elementwise[
            elementwise_fn_wrapper,
            target_simd_width,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
            _trace_description=trace_description,
        ](iteration_shape, ctx)
    else:
        elementwise[
            elementwise_fn_wrapper,
            1,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
            _trace_description=trace_description,
        ](iteration_shape, ctx)
