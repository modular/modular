# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil, max, min
from sys.info import sizeof, has_neon
from sys.intrinsics import PrefetchOptions

from algorithm import (
    async_parallelize,
    elementwise,
    unroll,
    vectorize,
    vectorize_unroll,
)
from algorithm.functional import _elementwise_impl, tile
from memory import memset_zero, stack_allocation
from memory.buffer import Buffer, NDBuffer, prod_dims
from MOGG import reshape
from runtime.llcl import OutputChainPtr, OwningOutputChainPtr
from runtime.tracing import TraceLevel

from utils.index import StaticIntTuple
from utils.list import Dim, DimList
from utils.optional_param import OptionalParamInt
from utils.optional import Optional


@always_inline
fn normalize_index[type: DType](idx: SIMD[type, 1], dim_size: Int) -> Int:
    """Indices passed to gather and scatter ops may be negative. This performs
    a normalization so that they can be used to index into a buffer.

    Returns val + dim if val < 0 else val
    """
    debug_assert(
        -dim_size <= idx.to_int() < dim_size,
        "indices must be in range [-dim_size, dim_size)",
    )
    constrained[
        type.is_integral(),
        "normalize_index expects index to be an integral type",
    ]()
    return idx.to_int() + dim_size if idx < 0 else idx.to_int()


@always_inline
fn gather_reduce[
    output_rank: Int,
    output_shape: DimList,
    input_rank: Int,
    input_shape: DimList,
    indices_rank: Int,
    indices_shape: DimList,
    type: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
](
    output: NDBuffer[output_rank, output_shape, type],
    input: NDBuffer[input_rank, input_shape, type],
    indices: NDBuffer[
        indices_rank,
        indices_shape,
        DType.int32,
    ],
    reduce_init: SIMD[type, 1],
    out_chain: OutputChainPtr,
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output accross axis 1 to produce output[i, k].

    The motivating use-case for this is multi-hot embeddings in recommender models.
    This provides similar functionality to Torch's EmbeddingBag layer. In that
    context, i is the batch dimension, j is the multi-hot dimension, and k is
    the embedding dimension.
    """
    constrained[input_rank == 2]()
    constrained[indices_rank == 2]()
    constrained[gather_axis == 0]()
    constrained[reduce_axis == 1]()

    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.size() == 0 or indices.size() == 0:
        return

    # TODO: find a heuristic to replace the magic number.
    # This is about 4x larger than the default in gather, which makes sense
    # since this kernel performs far fewer writes
    alias MIN_TASK_COPY_SIZE = 64 * 100 * 32 * 4  # bytes
    let num_threads = out_chain.get_runtime().parallelism_level()
    let num_tasks = min(
        div_ceil(
            indices.dim[0]()
            * indices.dim[1]()
            * input.dim[1]()
            * sizeof[type](),
            MIN_TASK_COPY_SIZE,
        ),
        num_threads,
    )

    let num_chunks_per_task = div_ceil(indices.dim[0](), num_tasks)

    var output_2d_dims = StaticIntTuple[2](output.dim[0](), output.dim[1]())

    @parameter
    if output_rank == 3:
        output_2d_dims[1] = output.dim[2]()

    let output_bind = NDBuffer[2, DimList.create_unknown[2](), type](
        output.data, output_2d_dims
    )
    let input_bind = rebind[NDBuffer[2, DimList.create_unknown[2](), type]](
        input
    )
    let indices_bind = rebind[
        NDBuffer[indices_rank, indices_shape, DType.int32]
    ](indices)

    let gather_axis_size = input.get_shape()[gather_axis]

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        alias prefetch_offset = -1

        let output = output_bind
        let input = input_bind
        let indices = indices_bind
        let row_size = output.dim[1]()

        # need to reduce on an entire 2D slice at a time, otherwise multiple
        # threads will try to accumulate in the same buffer simultaneously
        let start_slice = task_id * num_chunks_per_task
        let end_slice = min(
            (task_id + 1) * num_chunks_per_task, indices.dim[0]()
        )
        for i in range(start_slice, end_slice):

            @always_inline
            @parameter
            fn _accum_in_place[simd_width: Int](k: Int):
                var accum = SIMD[type, simd_width](reduce_init)
                for j in range(indices.dim[1]()):
                    """Computes output[i,k] = reduction over j of (input[indices[i,j],k])
                    for j in range [0,indices.dim[1])"""
                    let idx: Int

                    @parameter
                    if has_neon():  # TODO(#24060): remove this branch
                        idx = indices[i, j].value
                    else:
                        idx = normalize_index(indices[i, j], gather_axis_size)

                    # min so that we don't read beyond end of indices
                    @parameter
                    if prefetch_offset > 0:
                        let clamped_prefetch_offset = min(
                            prefetch_offset,
                            indices.dim[0]() * indices.dim[1]()
                            - (i * indices.dim[1]() + j)
                            - 1,
                        )
                        let next_idx_ptr = indices._offset(
                            StaticIntTuple[indices_rank](i, j)
                        ) + clamped_prefetch_offset
                        input.prefetch[
                            PrefetchOptions()
                            .for_read()
                            .high_locality()
                            .to_data_cache()
                        ](next_idx_ptr.load().to_int(), 0)

                    let in_idx = StaticIntTuple[2](idx, k)

                    let gather_chunk = input.simd_load[simd_width](in_idx)
                    accum = reduce_fn[type, simd_width](accum, gather_chunk)

                let out_idx = StaticIntTuple[2](i, k)
                output.simd_store[simd_width](out_idx, accum)

            # TODO(#24060): remove this branch
            alias tile_sizes = VariadicList[Int](
                2 * simd_width, 1
            ) if has_neon() else VariadicList[Int](
                8 * simd_width, 4 * simd_width, 2 * simd_width, simd_width, 1
            )
            tile[
                _accum_in_place,
                tile_sizes,
            ](0, row_size)

    async_parallelize[task_func](out_chain, num_tasks)


fn gather[
    output_rank: Int,
    input_rank: Int,
    indices_rank: Int,
    type: DType,
    indices_type: DType,
    axis: Int,
    simd_width: Int,
](
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    input: NDBuffer[input_rank, DimList.create_unknown[input_rank](), type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    out_chain: OutputChainPtr,
):
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    alias prefetch_offset = 12  # TODO: search

    let end_indices_ptr = indices.flatten().data.offset(indices.size())

    @parameter
    @always_inline
    fn prefetch_fn[
        _input_rank: Int, _indices_rank: Int
    ](
        _input_coords: StaticIntTuple[_input_rank],
        _indices_coords: StaticIntTuple[_indices_rank],
    ):
        let __input_coords = _input_coords
        var input_coords = rebind[StaticIntTuple[input_rank]](__input_coords)
        let indices_coords = rebind[StaticIntTuple[indices_rank]](
            _indices_coords
        )

        @parameter
        if prefetch_offset > 0:
            let indices_ptr = indices._offset(indices_coords)
            let indices_remaining = (
                end_indices_ptr.__as_index() - indices_ptr.__as_index()
            ) // sizeof[indices_type]()
            # assumes that indices are layed out in row major order
            let next_idx_ptr = indices._offset(indices_coords) + min(
                indices_remaining - 1, prefetch_offset
            )
            input_coords[axis] = normalize_index(
                next_idx_ptr.load(), input.get_shape()[axis]
            )
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](input_coords)

    @parameter
    @always_inline
    fn input_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return input.simd_load[width](
            rebind[StaticIntTuple[input_rank]](coords)
        )

    @parameter
    @always_inline
    fn indices_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[indices_type, width]:
        return indices.simd_load[width](
            rebind[StaticIntTuple[indices_rank]](coords)
        )

    @parameter
    @always_inline
    fn output_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[type, width]):
        output.simd_store[width](
            rebind[StaticIntTuple[output_rank]](coords),
            rebind[SIMD[type, width]](val),
        )

    gather[
        type,
        input_rank,
        indices_type,
        indices_rank,
        output_rank,
        simd_width,
        False,  # single_thread_blocking_override
        input_fn,
        indices_fn,
        output_fn,
        prefetch_fn,
        Dim(axis),
    ](
        OptionalParamInt[axis](axis),
        input.dynamic_shape,
        indices.dynamic_shape,
        output.dynamic_shape,
        out_chain,
    )


@always_inline
fn gather[
    type: DType,
    input_rank: Int,
    indices_type: DType,
    indices_rank: Int,
    output_rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    indices_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[indices_type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    prefetch_fn: fn[input_rank: Int, indices_rank: Int] (
        StaticIntTuple[input_rank], StaticIntTuple[indices_rank]
    ) capturing -> None,
    axis_static: Dim,
](
    axis: OptionalParamInt[axis_static],
    input_shape: StaticIntTuple[input_rank],
    indices_shape: StaticIntTuple[indices_rank],
    output_shape: StaticIntTuple[output_rank],
    out_chain: OutputChainPtr,
):
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    # Disable error checking in trivial kernels.
    @parameter
    if not single_thread_blocking_override:
        if axis.get() < 0:
            return out_chain.mark_error(
                "gather kernel does not support negative axis"
            )

        # The output shape has the same shape as the input, with the indexed-axis
        # replaced by the shape of the indices
        for i in range(axis.get()):
            if output_shape[i] != input_shape[i]:
                return out_chain.mark_error(
                    "gather: output_shape[0:axis] does not match"
                    " input_shape[0:axis]"
                )
        for i in range(axis.get(), axis.get() + indices_rank):
            if output_shape[i] != indices_shape[i - axis.get()]:
                return out_chain.mark_error(
                    "gather: output_shape[axis:axis+indices_rank] does not"
                    " match indices_shape"
                )
        for i in range(axis.get() + indices_rank, output_rank):
            if output_shape[i] != input_shape[i - indices_rank + 1]:
                return out_chain.mark_error(
                    "gather: output_shape[axis + indices_rank:] does not match"
                    " input_shape[axis:]"
                )

        if axis.get() >= input_rank:
            return out_chain.mark_error(
                "gather: axis must be less than input rank"
            )

    out_chain.trace[TraceLevel.OP]("mojo.gather")

    # Short-circuit for trivial cases, and to avoid divide-by-zero
    let indices_len = indices_shape.flattened_length()
    if input_shape.flattened_length() == 0 or indices_len == 0:

        @parameter
        if not single_thread_blocking_override:
            out_chain.mark_ready()
        return

    @parameter
    @always_inline
    fn gather_lambda[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        # out_coords consists of 3 chunks:
        #   out_coords[0:axis] = input coords[0:axis]
        #   out_coords[axis:axis+indices_rank] = indices_coords
        #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
        # and input_coords[axis] = indices[indices_coords]
        # Get the gather indices.
        var indices_index = StaticIntTuple[indices_rank]()

        # Get the indices of the index.
        @always_inline
        @parameter
        fn indices_get[unrolled_i: Int]():
            indices_index[unrolled_i] = idx[unrolled_i + axis.get()]

        unroll[indices_rank, indices_get]()

        # The index we are gathering.
        let data_index = indices_fn[1, indices_rank](indices_index)

        # Update the indices with the new data index.
        var data_indices = StaticIntTuple[input_rank]()

        let skip_factor = indices_rank - 1

        # Build the indices for the input. We have replaced in index in 'axis'
        # with an index from the indices tensor.
        @always_inline
        @parameter
        fn input_indices_get[unrolled_i: Int]():
            if unrolled_i == axis.get():
                data_indices[unrolled_i] = normalize_index(
                    data_index, input_shape[axis.get()]
                )
            elif unrolled_i > axis.get():
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[unrolled_i] = idx[unrolled_i + skip_factor]
            else:
                data_indices[unrolled_i] = idx[unrolled_i]

        unroll[input_rank, input_indices_get]()

        # Load the the data.
        prefetch_fn[input_rank, indices_rank](data_indices, indices_index)
        let data = input_fn[simd_width, input_rank](data_indices)

        # Store it to the original index.
        output_fn[simd_width, rank](idx, data)

    # If we are gathering on the last dimension then we have to be scalar.
    if axis.get() == input_rank - 1:
        _elementwise_impl[
            output_rank,
            1,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output_shape,
            out_chain,
        )
    else:
        _elementwise_impl[
            output_rank,
            simd_width,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output_shape,
            out_chain,
        )


# ===----------------------------------------------------------------------===#
# scatter_nd op
# ===----------------------------------------------------------------------===#


@always_inline
fn scatter_nd_generator[
    output_type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    updates_rank: Int,
    single_thread_blocking_override: Bool,
    /,
    reduce_fn: Optional[
        fn[
            type: DType, width: Int
        ] (SIMD[type, width], SIMD[type, width]) capturing -> SIMD[type, width]
    ] = None,
](
    data: NDBuffer[data_rank, DimList.create_unknown[data_rank](), output_type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    output: NDBuffer[
        data_rank, DimList.create_unknown[data_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        output_type: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
        data_rank: Rank of input (data) tensor (data_rank >= 1).
        indices_rank: Rank of input (data) tensor (indices_rank >= 1).
        updates_rank: Rank of updates tensor (updates_rank = data_rank +
                      indices_rank - indices_shape[-1] - 1).
        single_thread_blocking_override: Whether this function can block.
        reduce_fn: Reduction function to apply: none (default), add, mul, max,
                   min.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank containing indices for the scatter
                 operation.
        updates: Tensor containing values to update output tensor based on
                 indices tensor.
        output: Tensor of rank data_rank, shaped the same as data tensor.
        out_chain: The OutputChainPtr used to mark competion or error of the task.
    """
    if data.get_shape() != output.get_shape():
        return out_chain.mark_error(
            "Input and output shapes in scatter_nd must be the same."
        )

    if (
        len(updates.get_shape())
        != data_rank + indices_rank - indices.get_shape()[indices_rank - 1] - 1
    ):
        return out_chain.mark_error(
            "updates rank must be: data_rank + indices_rank -"
            " indices_shape[-1] - 1"
        )

    let output_flat = output.flatten()
    let data_flat = data.flatten()
    let updates_flat = updates.flatten()
    memcpy(output_flat, data_flat)

    let data_shape = data.get_shape()
    let indices_shape = indices.get_shape()
    let last_shape_of_indices = indices_shape[indices_rank - 1]

    # Depending on r_minus_m = data_rank - last_shape_of_indices,
    # we will be copying (gather):
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    let r_minus_m = data_rank - last_shape_of_indices
    # Calculate how many elements to copy (this is from the innermost
    # dimensions, and is continuous memory locations).
    var count_copy = 1
    for i in range(r_minus_m):
        count_copy = count_copy * data_shape[data_rank - 1 - i]

    # Stores the full index on output, where to copy updates to.
    let output_index_tensor = NDBuffer[
        1,
        DimList(data_rank),
        DType.index,
    ]().stack_allocation()
    # Zeroing here to avoid doing it selectively within the nested loop below.
    output_index_tensor.zero()

    # Stores the full index on updates, where to copy from.
    let updates_index_tensor = NDBuffer[
        1,
        DimList(updates_rank),
        DType.index,
    ]().stack_allocation()
    # Zeroing here to avoid doing it selectively within the nested loop below.
    updates_index_tensor.zero()

    @parameter
    fn update_func[
        simd_width: Int, _rank: Int
    ](_indices_coords: StaticIntTuple[_rank]):
        let indices_coords = rebind[StaticIntTuple[_rank]](_indices_coords)

        # Construct the full index on updates tensor, i.e., where to copy from.
        for dim in range(_rank):
            updates_index_tensor[dim] = indices_coords[dim]

        # Construct the output_index_tensor whose elements contain the indices
        # for each dimension of the output, i.e., where to copy updates to.
        # As part of that we need to construct the indices_index, which is the
        # index to the indices tensor, where we get the elements for the
        # output_index_tensor from.
        var indices_index = StaticIntTuple[indices_rank]()
        for dim in range(last_shape_of_indices):
            # Size of current dimension on data.
            # Used to compare to index on this dimension (idx_on_axis).
            let input_ax_dim = data_shape[dim]

            for i in range(_rank):
                indices_index[i] = indices_coords[i]
            indices_index[indices_rank - 1] = dim

            let idx_on_axis = indices[indices_index]
            let pos_idx_on_axis = normalize_index(idx_on_axis, input_ax_dim)
            output_index_tensor[dim] = pos_idx_on_axis

        # Calculate the updates_offset from where to copy the updates.
        var updates_offset = 0

        @unroll
        for i in range(updates_rank):
            updates_offset = (
                updates_offset
                + updates.stride(i) * updates_index_tensor[i].to_int()
            )

        # Calculate the output_offset to where to copy the updates.
        var output_offset = 0

        @unroll
        for i in range(data_rank):
            output_offset = (
                output_offset
                + output.stride(i) * output_index_tensor[i].to_int()
            )

        # Perform the actual copy of element/slice/sheet/cuboid/etc.
        # Also handling any reduction operation reduce_fn.
        @parameter
        if reduce_fn:
            alias reduction_fn = reduce_fn.value()

            @parameter
            @always_inline
            fn reduce_updates[simd_width: Int](idx: Int):
                output_flat.simd_store[simd_width](
                    output_offset + idx,
                    reduction_fn(
                        output_flat.simd_load[simd_width](output_offset + idx),
                        updates_flat.simd_load[simd_width](
                            updates_offset + idx
                        ),
                    ),
                )

            vectorize[simdwidthof[output_type](), reduce_updates](count_copy)
        else:

            @parameter
            @always_inline
            fn copy_updates[simd_width: Int](idx: Int):
                output_flat.simd_store[simd_width](
                    output_offset + idx,
                    updates_flat.simd_load[simd_width](updates_offset + idx),
                )

            vectorize[simdwidthof[output_type](), copy_updates](count_copy)

    # TODO: SEE: simd_width > 1
    var iter_shape = StaticIntTuple[indices_rank - 1]()
    for i in range(len(indices.get_shape()) - 1):
        iter_shape[i] = indices.get_shape()[i]

    # Execute `elementwise()` synchronously because parametric closures capture
    # variables by reference without extending their lifetimes.
    let new_out_chain = OwningOutputChainPtr(out_chain.get_runtime())
    elementwise[indices_rank - 1, 1, update_func](
        iter_shape, new_out_chain.borrow()
    )
    new_out_chain.wait()

    # Avoid prematurely marking `out_chain` as ready if this kernel is fused.
    @parameter
    if not single_thread_blocking_override:
        out_chain.mark_ready()


@always_inline
fn scatter_nd[
    output_type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    updates_rank: Int,
    single_thread_blocking_override: Bool,
](
    data: NDBuffer[data_rank, DimList.create_unknown[data_rank](), output_type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), output_type
    ],
    output: NDBuffer[
        data_rank, DimList.create_unknown[data_rank](), output_type
    ],
    out_chain: OutputChainPtr,
):
    """Scatter_nd operation without any reduction."""

    scatter_nd_generator[
        output_type,
        indices_type,
        data_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        reduce_fn=None,
    ](data, indices, updates, output, out_chain)


# ===----------------------------------------------------------------------===#
# Gather Shape
# ===----------------------------------------------------------------------===#


@always_inline
fn gather_shape[
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    indices_buf: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        indices_rank: Rank of the indices tensor.
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """

    constrained[
        output_rank == input_rank + indices_rank - 1,
        "output rank must equal (input_rank + indices_rank - 1)",
    ]()

    # extract hyper parameter
    var axis = axis_buf[0].to_int()
    if axis < 0:
        axis += input_rank
    # TODO(#17512)
    debug_assert(
        0 <= axis and axis < input_rank,
        "normalized axis must be within range [0, input_rank)",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[output_rank]()
    var next_out_dim = 0

    let input_shape = input_buf.get_shape()
    for i in range(axis):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    let indices_shape = indices_buf.get_shape()
    for i in range(indices_rank):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(axis + 1, input_rank):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===----------------------------------------------------------------------===#
# Scatter Elements
# ===----------------------------------------------------------------------===#


@always_inline
fn scatter_elements[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing -> SIMD[type, width],
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[rank, DimList.create_unknown[rank](), indices_type],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    _axis: Int,
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    """
    Implements ONNX ScatterElements op which is equivalent to Pytorch scatter.
    """
    constrained[
        indices_type == DType.int32 or indices_type == DType.int64,
        "indices in scatter_elements must be int32 or int64",
    ]()

    if input.get_shape() != output.get_shape():
        return out_chain.mark_error(
            "input and output shape in scatter_elements must be the same"
        )

    if indices.get_shape() != updates.get_shape():
        return out_chain.mark_error(
            "inidices and updates shape in scatter_elements must be the same"
        )

    if not (-rank <= _axis < rank):
        return out_chain.mark_error(
            "axis in scatter_elements must be in the range [-rank, rank)"
        )

    let axis = _axis if _axis >= 0 else _axis + rank

    # TODO: multithread
    memcpy(output.flatten(), input.flatten())

    let input_ax_dim = input.get_shape()[axis]

    @parameter
    fn update_func[
        simd_width: Int, _rank: Int
    ](_indices_coords: StaticIntTuple[_rank]):
        let indices_coords = rebind[StaticIntTuple[rank]](_indices_coords)
        let idx_on_axis = indices[indices_coords]
        var output_coords = indices_coords
        output_coords[axis] = normalize_index(idx_on_axis, input_ax_dim)
        let curr = output[output_coords]
        output[output_coords] = reduce_fn[input_type, 1](
            curr, updates[indices_coords]
        )

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[rank, 1, update_func](indices.get_shape(), out_chain)


@always_inline
fn scatter_elements_shape[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    updates: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[rank, DimList.create_unknown[rank](), indices_type],
    axis: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[rank]:
    """
    Compute the output shape of a `scatter_elements` operation, and assert the
    inputs are compatible.

    Parameters:
        rank: Rank of the input tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input: The input tensor.
        updates: The input tensor.
        indices: The indices tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # Check axis
    let axis_int = axis[0].to_int()
    # TODO(#17512)
    debug_assert(
        -rank <= axis_int and axis_int < rank,
        "axis must be within range [-input_rank, input_rank)",
    )

    # Check individual dimensions
    for axis in range(rank):
        let input_dim = input.dim(axis)
        let indices_dim = indices.dim(axis)
        let updates_dim = updates.dim(axis)
        # TODO(#17512)
        debug_assert(
            indices_dim == updates_dim,
            "indices and updates must have the same shape",
        )
        # TODO(#17512)
        debug_assert(
            indices_dim < input_dim,
            "indices and updates must have smaller shape than input",
        )

    # Return output shape
    return input.get_shape()


# ===----------------------------------------------------------------------===#
# Gather Elements
# ===----------------------------------------------------------------------===#


@always_inline
fn gather_elements[
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    indices: NDBuffer[rank, DimList.create_unknown[rank](), indices_type],
    _axis: Int,
    output: NDBuffer[rank, DimList.create_unknown[rank](), input_type],
    out_chain: OutputChainPtr,
):
    """
    Implements ONNX GatherElements op which is equivalent to Pytorch gather.
    """
    constrained[
        indices_type == DType.int32 or indices_type == DType.int64,
        "indices in gather_elements must be int32 or int64",
    ]()

    if indices.get_shape() != output.get_shape():
        return out_chain.mark_error(
            "indices and output shape in gather_elements must be the same"
        )

    if not (-rank <= _axis < rank):
        return out_chain.mark_error(
            "axis in gather_elements must be in the range [-rank, rank)"
        )

    let axis = _axis if _axis >= 0 else _axis + rank

    let input_ax_dim = input.get_shape()[axis]

    @parameter
    fn gather_func[
        simd_width: Int, _rank: Int
    ](_output_coords: StaticIntTuple[_rank]):
        let output_coords = rebind[StaticIntTuple[rank]](_output_coords)
        let idx_on_axis = indices[output_coords]
        var input_coords = output_coords
        input_coords[axis] = normalize_index(idx_on_axis, input_ax_dim)
        output[output_coords] = input[input_coords]

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[rank, 1, gather_func](output.get_shape(), out_chain)


# ===----------------------------------------------------------------------===#
# gather_nd shape
# ===----------------------------------------------------------------------===#


fn gather_nd_shape[
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    indices_buf: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
) -> StaticIntTuple[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        indices_rank: Rank of the indices tensor.
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
    constrained[
        input_rank >= 1 and indices_rank >= 1,
        "Constraint: data_rank >= 1 and indices_rank >= 1",
    ]()

    let indices_shape = indices_buf.get_shape()
    debug_assert(
        1 <= indices_shape[indices_rank - 1] <= input_rank - batch_dims,
        "Constraint: 1 <= indices_shape[-1] <= input_rank - batch_dims",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[output_rank]()
    var next_out_dim = 0

    let input_shape = input_buf.get_shape()

    for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(batch_dims, indices_rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    if indices_shape[indices_rank - 1] == input_rank - batch_dims:
        return output_shape

    for i in range(
        batch_dims + indices_shape[indices_rank - 1], len(input_shape)
    ):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===----------------------------------------------------------------------===#
# GatherND
# ===----------------------------------------------------------------------===#


fn gather_nd[
    type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    batch_dims: Int,
](
    data: NDBuffer[data_rank, DimList.create_unknown[data_rank](), type],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), indices_type
    ],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
):
    """
    GatherND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND.
    Based on reference implementation: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gathernd.py.

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
        data_rank >= 1 and indices_rank >= 1,
        "Constraint: data_rank >= 1 and indices_rank >= 1",
    ]()

    let indices_shape = indices.get_shape()
    debug_assert(
        1 <= indices_shape[indices_rank - 1] <= data_rank - batch_dims,
        "Constraint: 1 <= indices_shape[-1] <= data_rank - batch_dims",
    )

    # The number of elements in the batch_dims for data/indices array.
    # E.g., if batch_dims = 2 (always is the outermost dimensions), and the
    #       dimensions of data are [2,3,...], then batch_dims_size = 6
    var batch_dims_size = 1
    for i in range(batch_dims):
        batch_dims_size = batch_dims_size * indices_shape[i]

    let last_shape_of_indices = indices_shape[indices_rank - 1]
    let num_elems = indices.num_elements()
    # Reshape indices array, as 3D array. All batch_dims_size elements go to the
    # outermost dimension, and elements of amount equal to indices.shape[-1] go
    # to the innermost.
    # Equivalent to numpy:
    # reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])
    let reshaped_indices = reshape[indices_rank, 3, indices_type, True](
        indices.make_dims_unknown(),
        StaticIntTuple[3](
            batch_dims_size,
            num_elems // (batch_dims_size * last_shape_of_indices),
            last_shape_of_indices,
        ),
    )

    let reshaped_indices_shape = reshaped_indices.get_shape()
    let data_shape = data.get_shape()

    # Flatten data to array of shape (batch_dim_size, data.shape[batch_dims:])
    alias reshaped_data_rank = data_rank + 1 - batch_dims
    var reshaped_data_tuple = StaticIntTuple[reshaped_data_rank]()
    # Calculate the dimensions of reshaped_data.
    reshaped_data_tuple[0] = batch_dims_size
    var counter = 1
    for i in range(batch_dims, data_rank):
        reshaped_data_tuple[counter] = data_shape[i]
        counter += 1
    # Do the actual reshaping.
    let reshaped_data = reshape[data_rank, reshaped_data_rank, type, True](
        data.make_dims_unknown(), reshaped_data_tuple
    )

    let reshaped_data_shape = reshaped_data.get_shape()

    # idx[] stores the index from where to gather the requested elements.
    let idx_ptr = DTypePointer[DType.index].alloc(reshaped_indices_shape[2])
    let idx = NDBuffer[1, DimList.create_unknown[1](), DType.index](
        idx_ptr, reshaped_indices_shape[2]
    )

    # Depending on r_minus_m = data_rank - last_shape_of_indices - batch_dims,
    # we will be copying (gather):
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    let r_minus_m = data_rank - last_shape_of_indices - batch_dims
    # Calculate how many elements to copy (this is from the innermost
    # dimensions, and is continuous memory locations).
    var count_copy = 1
    for i in range(r_minus_m):
        count_copy = (
            count_copy * reshaped_data_shape[reshaped_data_rank - 1 - i]
        )

    # Stores the full index on reshaped_data, where to copy from.
    # It is constructed within the nested loop below.
    let start_tensor = NDBuffer[
        1,
        DimList(reshaped_data_rank),
        DType.index,
    ]().stack_allocation()
    # Zeroing here to avoid doing it selectively within the nested loop below.
    memset_zero[DType.index](start_tensor.data, reshaped_data_rank)

    var output_buffer_copy_ind = 0
    for batch_dim in range(reshaped_indices_shape[0]):
        for outer_dim in range(reshaped_indices_shape[1]):
            # Construct the tuple (all dimensions except outermost, which is
            # the batches dimension - recall all batch dimensions are reshaped
            # into one - the outermost).
            for constr in range(reshaped_indices_shape[2]):
                let input_ax_dim = reshaped_data.get_shape()[constr + 1]
                let idx_on_axis = reshaped_indices[batch_dim, outer_dim, constr]
                idx[constr] = normalize_index(idx_on_axis, input_ax_dim)

            # Construct the full index on reshaped_data, where to copy from.
            start_tensor[0] = batch_dim
            var start_index = 1
            for dim in range(idx.__len__()):
                start_tensor[start_index] = idx[dim]
                start_index = start_index + 1

            # Calculate the input_offset from where to copy the data.
            var input_offset = 0
            for i in range(reshaped_data_rank):
                input_offset = (
                    input_offset
                    + reshaped_data.stride(i) * start_tensor[i].to_int()
                )

            # Calculate the output_offset where to copy the data.
            let output_offset = output_buffer_copy_ind * (count_copy)
            output_buffer_copy_ind = output_buffer_copy_ind + 1

            # Perform the actual copy of element/slice/sheet/cuboid/etc.
            memcpy[type](
                output.data + output_offset,
                reshaped_data.data + input_offset,
                count_copy,
            )
    idx_ptr.free()
