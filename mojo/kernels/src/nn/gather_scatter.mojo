# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import Buffer, NDBuffer, prod_dims
from algorithm import (
    vectorize,
    vectorize_unroll,
    async_parallelize,
    unroll,
    elementwise,
)
from algorithm.functional import _elementwise_impl
from Index import StaticIntTuple
from sys.intrinsics import PrefetchOptions
from runtime.llcl import OutputChainPtr
from List import DimList, Dim
from math import div_ceil
from math import min
from sys.info import sizeof
from Tracing import TraceLevel
from OptionalParam import OptionalParamInt

## gather_reduce_2D_axis_1
@adaptive
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

    @always_inline
    @parameter
    fn task_func(task_id: Int):
        alias unroll_factor = 2
        alias prefetch_offset = -1
        alias usimd_width = unroll_factor * simd_width

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
                    let idx = indices[i, j].value

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

            vectorize[usimd_width, _accum_in_place](row_size)

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
            ) // sizeof[type]()
            # assumes that indices are layed out in row major order
            let next_idx_ptr = indices._offset(indices_coords) + min(
                indices_remaining, prefetch_offset
            )
            input_coords[axis] = next_idx_ptr.load().to_int()
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](input_coords)

    @parameter
    @always_inline
    fn input_fn[
        _type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](
            input.simd_load[width](rebind[StaticIntTuple[input_rank]](coords))
        )

    @parameter
    @always_inline
    fn indices_fn[
        _type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[_type, width]:
        return rebind[SIMD[_type, width]](
            indices.simd_load[width](
                rebind[StaticIntTuple[indices_rank]](coords)
            )
        )

    @parameter
    @always_inline
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[_type, width]):
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
        output,
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
    input_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    indices_fn: fn[type: DType, width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    output_fn: fn[type: DType, width: Int, rank: Int] (
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
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
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
            if output.dim(i) != input_shape[i]:
                return out_chain.mark_error(
                    "gather: output_shape[0:axis] does not match"
                    " input_shape[0:axis]"
                )
        for i in range(axis.get(), axis.get() + indices_rank):
            if output.dim(i) != indices_shape[i - axis.get()]:
                return out_chain.mark_error(
                    "gather: output_shape[axis:axis+indices_rank] does not"
                    " match indices_shape"
                )
        for i in range(axis.get() + indices_rank, output_rank):
            if output.dim(i) != input_shape[i - indices_rank + 1]:
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
        let data_index = indices_fn[indices_type, 1, indices_rank](
            indices_index
        ).to_int()

        # Update the indices with the new data index.
        var data_indices = StaticIntTuple[input_rank]()

        let skip_factor = indices_rank - 1

        # Build the indices for the input. We have replaced in index in 'axis'
        # with an index from the indices tensor.
        @always_inline
        @parameter
        fn input_indices_get[unrolled_i: Int]():
            indices_index[unrolled_i] = idx[unrolled_i + axis.get()]
            if unrolled_i == axis.get():
                data_indices[unrolled_i] = data_index
            elif unrolled_i > axis.get():
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[unrolled_i] = idx[unrolled_i + skip_factor]
            else:
                data_indices[unrolled_i] = idx[unrolled_i]

        unroll[input_rank, input_indices_get]()

        # Load the the data.
        prefetch_fn[input_rank, indices_rank](data_indices, indices_index)
        let data = input_fn[type, simd_width, input_rank](data_indices)

        # Store it to the original index.
        output_fn[type, simd_width, rank](idx, data)

    # If we are gathering on the last dimension then we have to be scalar.
    if axis.get() == input_rank - 1:
        _elementwise_impl[
            output_rank,
            1,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )
    else:
        _elementwise_impl[
            output_rank,
            simd_width,
            single_thread_blocking_override,
            gather_lambda,
        ](
            output.dynamic_shape,
            out_chain,
        )


# ===----------------------------------------------------------------------===#
# ScatterNd
# ===----------------------------------------------------------------------===#


@always_inline
fn scatter_nd[
    type: DType,
    updates_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    single_thread_blocking_override: Bool,
](
    updates: NDBuffer[
        updates_rank, DimList.create_unknown[updates_rank](), type
    ],
    indices: NDBuffer[
        indices_rank, DimList.create_unknown[indices_rank](), DType.int32
    ],
    output: NDBuffer[output_rank, DimList.create_unknown[output_rank](), type],
    out_chain: OutputChainPtr,
):
    """A specialized scatter for updates[i0][i1][i2][i3], indices[i0][i1][1],
    output[o0][i2][i3]. TODO(#15445): Make this general."""

    constrained[type.is_float64(), "scatter_nd only supports F64 currently"]()
    constrained[
        updates_rank == 4, "scatter_nd updates rank must be 4 currently"
    ]()
    constrained[
        indices_rank == 3, "scatter_nd indices rank must be 3 currently"
    ]()
    constrained[
        output_rank == 3, "scatter_nd output rank must be 3 currently"
    ]()

    let output_shape = output.get_shape()
    let updates_shape = updates.get_shape()
    let indices_shape = indices.get_shape()

    @parameter
    if not single_thread_blocking_override:
        if indices_shape[indices_rank - 1] != 1:
            return out_chain.mark_error("unsupported indices shape")

        if (
            updates_shape[0] != indices_shape[0]
            or updates_shape[1] != indices_shape[1]
        ):
            return out_chain.mark_error(
                "updates and index shape prefix mismatch"
            )

    let N = updates_shape[0] * updates_shape[1]
    let D = updates_shape[2] * updates_shape[3]

    output.zero()

    let output_1d = Buffer[Dim(), type](output.data, output.num_elements())
    let indices_1d = Buffer[Dim(), DType.int32](
        indices.data, indices.num_elements()
    )
    let updates_1d = Buffer[Dim(), type](updates.data, updates.num_elements())

    for n in range(N):
        let index = indices_1d[n].to_int()
        for d in range(D):
            output_1d[index * D + d] = updates_1d[n * D + d]

    @parameter
    if not single_thread_blocking_override:
        out_chain.mark_ready()


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
