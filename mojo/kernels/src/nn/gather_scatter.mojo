# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import align_down, ceildiv
from os import abort
from sys import has_neon, simdwidthof, sizeof
from sys.intrinsics import PrefetchOptions

from algorithm import elementwise, parallel_memcpy, sync_parallelize
from algorithm.functional import tile
from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceBuffer, DeviceContext
from memory import memcpy, memset_zero, stack_allocation, UnsafePointer
from register import mogg_register, mogg_register_shape_func
from runtime.asyncrt import MojoCallContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel

from utils import StaticIntTuple, StaticTuple, unroll

from .reshape import reshape


@always_inline
fn normalize_neg_index[
    type: DType, width: Int, out_type: DType = DType.index
](idx: SIMD[type, width], dim_size: Int) -> SIMD[out_type, width]:
    """Indices passed to gather and scatter ops may be negative. This performs
    a normalization so that they can be used to index into a buffer.

    Returns val + dim if val < 0 else val
    """

    debug_assert(
        (
            (
                -SIMD[out_type, width](dim_size) <= idx.cast[out_type]()
            ).reduce_and()
            and (
                idx.cast[out_type]() < SIMD[out_type, width](dim_size)
            ).reduce_and()
        ),
        "indices must be in range [-dim_size, dim_size)",
    )
    constrained[
        type.is_integral(),
        "normalize_neg_index expects index to be an integral type",
    ]()
    return (idx < 0).select(
        idx.cast[out_type]() + dim_size, idx.cast[out_type]()
    )


@value
@register_passable("trivial")
struct Axis(Intable, Indexer):
    var axis: Int

    @always_inline
    fn __init__[
        type: DType
    ](inout self, axis_unnormalized: Scalar[type], rank: Int):
        self.axis = int(normalize_neg_index(axis_unnormalized, rank))

    @always_inline
    fn __int__(self) -> Int:
        return self.axis

    @always_inline
    fn __index__(self) -> Int:
        return int(self)


@always_inline
fn gather_reduce[
    type: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) -> SIMD[type, width],
    output_rank: Int,
    output_shape: DimList,
    input_rank: Int,
    input_shape: DimList,
    indices_rank: Int,
    indices_shape: DimList,
](
    output: NDBuffer[type, output_rank, output_shape],
    input: NDBuffer[type, input_rank, input_shape],
    indices: NDBuffer[
        DType.int32,
        indices_rank,
        indices_shape,
    ],
    reduce_init: Scalar[type],
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
    var num_threads = parallelism_level()
    var num_tasks = min(
        ceildiv(
            indices.dim[0]()
            * indices.dim[1]()
            * input.dim[1]()
            * sizeof[type](),
            MIN_TASK_COPY_SIZE,
        ),
        num_threads,
    )

    var out_vecs_per_thread = ceildiv(indices.dim[0](), num_tasks)

    var output_2d_dims = StaticIntTuple[2](output.dim[0](), output.dim[1]())

    @parameter
    if output_rank == 3:
        output_2d_dims[1] = output.dim[2]()

    var output_bind = NDBuffer[type, 2](output.data, output_2d_dims)
    var input_bind = rebind[NDBuffer[type, 2]](input)
    var indices_bind = rebind[
        NDBuffer[DType.int32, indices_rank, indices_shape]
    ](indices)

    var gather_axis_size = input.get_shape()[gather_axis]

    @always_inline
    @__copy_capture(
        output_bind,
        input_bind,
        indices_bind,
        out_vecs_per_thread,
        gather_axis_size,
    )
    @parameter
    fn task_func(task_id: Int):
        alias prefetch_offset = -1

        var output = output_bind
        var input = input_bind
        var indices = indices_bind
        var row_size = output.dim[1]()

        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var out_vec_start = task_id * out_vecs_per_thread
        var out_vec_end = min(
            (task_id + 1) * out_vecs_per_thread, indices.dim[0]()
        )

        # For multi-hot embeddings reduction, k is the embedding dim and j is the multi-hot dim
        alias k_tile_sizes = VariadicList[Int](
            2 * simd_width, 1
        ) if has_neon() else VariadicList[Int](
            8 * simd_width, 4 * simd_width, 2 * simd_width, simd_width, 1
        )
        # unroll the j loop on neon because it benefits from vectorized
        # blend instructions and avoids conditional flag dependencies
        # does not appear to help on other archs
        alias j_tile_size = 4 if has_neon() else 1

        for i in range(out_vec_start, out_vec_end):

            @always_inline
            @__copy_capture(input, indices, output)
            @parameter
            fn gather_k_tile[simd_width: Int](k: Int):
                @always_inline
                @parameter
                fn reduce_j_tile[
                    unroll_factor: Int
                ](
                    accums: StaticTuple[SIMD[type, simd_width], unroll_factor],
                    j: Int,
                ) -> StaticTuple[SIMD[type, simd_width], unroll_factor]:
                    var out = accums
                    var idxs = normalize_neg_index(
                        indices.load[width=unroll_factor](i, j),
                        gather_axis_size,
                    )

                    @parameter
                    for unroll_idx in range(0, unroll_factor):
                        var gather_chunk = input.load[width=simd_width](
                            int(idxs[unroll_idx]), k
                        )
                        out[unroll_idx] = reduce_fn[type, simd_width](
                            accums[unroll_idx], gather_chunk
                        )
                    return out

                var j_residual_start = align_down(indices.dim[1](), j_tile_size)
                var accums = StaticTuple[SIMD[type, simd_width], j_tile_size](
                    reduce_init
                )
                for j in range(0, j_residual_start, j_tile_size):
                    accums = reduce_j_tile[j_tile_size](accums, j)

                var accum = SIMD[type, simd_width](reduce_init)

                # TODO: use tree reduction here by generalizing simd reduce method
                @parameter
                for unroll_idx in range(j_tile_size):
                    accum = reduce_fn(accum, accums[unroll_idx])

                for j in range(j_residual_start, indices.dim[1](), 1):
                    accum = reduce_j_tile[1](
                        StaticTuple[SIMD[type, simd_width], 1](accum), j
                    )[0]

                var out_idx = StaticIntTuple[2](i, k)
                output.store[width=simd_width](out_idx, accum)

            tile[
                gather_k_tile,
                k_tile_sizes,
            ](0, row_size)

    sync_parallelize[task_func](num_tasks)


# TODO: Delete / for testing purposes (test_gather.mojo)
fn gather[
    axis: Int,
    output_rank: Int,
    input_rank: Int,
    indices_rank: Int,
    type: DType,
    indices_type: DType,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[type, output_rank],
    input: NDBuffer[type, input_rank],
    indices: NDBuffer[indices_type, indices_rank],
    context: DeviceContext,
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    alias prefetch_offset = 12  # TODO: search

    var end_indices_ptr = indices.flatten().data.offset(indices.size())

    @parameter
    @__copy_capture(end_indices_ptr)
    @always_inline
    fn prefetch_fn[
        _input_rank: Int, _indices_rank: Int
    ](
        _input_coords: StaticIntTuple[_input_rank],
        _indices_coords: StaticIntTuple[_indices_rank],
    ):
        var __input_coords = _input_coords
        var input_coords = rebind[StaticIntTuple[input_rank]](__input_coords)
        var indices_coords = rebind[StaticIntTuple[indices_rank]](
            _indices_coords
        )

        @parameter
        if prefetch_offset > 0:
            var indices_ptr = indices._offset(indices_coords)
            var indices_remaining = (
                int(end_indices_ptr) - int(indices_ptr)
            ) // sizeof[indices_type]()
            # assumes that indices are layed out in row major order
            var next_idx_ptr = indices._offset(indices_coords) + min(
                indices_remaining - 1, prefetch_offset
            )
            input_coords[axis] = int(
                normalize_neg_index(
                    next_idx_ptr.load(),
                    input.get_shape()[axis],
                )
            )
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](input_coords)

    @parameter
    @always_inline
    fn input_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return input.load[width=width](
            rebind[StaticIntTuple[input_rank]](coords)
        )

    @parameter
    @always_inline
    fn indices_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[indices_type, width]:
        return indices.load[width=width](
            rebind[StaticIntTuple[indices_rank]](coords)
        )

    @parameter
    @always_inline
    fn output_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[type, width]):
        output.store[width=width](
            rebind[StaticIntTuple[output_rank]](coords),
            rebind[SIMD[type, width]](val),
        )

    gather[
        type,
        indices_type,
        input_fn,
        indices_fn,
        output_fn,
        prefetch_fn=prefetch_fn,
        target=target,
    ](
        axis,
        input.dynamic_shape,
        indices.dynamic_shape,
        output.dynamic_shape,
        context=context,
    )


fn gather[
    axis: Int,
    output_rank: Int,
    input_rank: Int,
    indices_rank: Int,
    type: DType,
    indices_type: DType,
    target: StringLiteral = "cpu",
](
    output: NDBuffer[type, output_rank],
    input: NDBuffer[type, input_rank],
    indices: NDBuffer[indices_type, indices_rank],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    alias prefetch_offset = 12  # TODO: search

    var end_indices_ptr = indices.flatten().data.offset(indices.size())

    @parameter
    @__copy_capture(end_indices_ptr)
    @always_inline
    fn prefetch_fn[
        _input_rank: Int, _indices_rank: Int
    ](
        _input_coords: StaticIntTuple[_input_rank],
        _indices_coords: StaticIntTuple[_indices_rank],
    ):
        var __input_coords = _input_coords
        var input_coords = rebind[StaticIntTuple[input_rank]](__input_coords)
        var indices_coords = rebind[StaticIntTuple[indices_rank]](
            _indices_coords
        )

        @parameter
        if prefetch_offset > 0:
            var indices_ptr = indices._offset(indices_coords)
            var indices_remaining = (
                int(end_indices_ptr) - int(indices_ptr)
            ) // sizeof[indices_type]()
            # assumes that indices are layed out in row major order
            var next_idx_ptr = indices._offset(indices_coords) + min(
                indices_remaining - 1, prefetch_offset
            )
            input_coords[axis] = int(
                normalize_neg_index(
                    next_idx_ptr.load(),
                    input.get_shape()[axis],
                )
            )
            input.prefetch[
                PrefetchOptions().for_read().high_locality().to_data_cache()
            ](input_coords)

    @parameter
    @always_inline
    fn input_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
        return input.load[width=width](
            rebind[StaticIntTuple[input_rank]](coords)
        )

    @parameter
    @always_inline
    fn indices_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank]) -> SIMD[indices_type, width]:
        return indices.load[width=width](
            rebind[StaticIntTuple[indices_rank]](coords)
        )

    @parameter
    @always_inline
    fn output_fn[
        width: Int, _rank: Int
    ](coords: StaticIntTuple[_rank], val: SIMD[type, width]):
        output.store[width=width](
            rebind[StaticIntTuple[output_rank]](coords),
            rebind[SIMD[type, width]](val),
        )

    gather[
        type,
        indices_type,
        input_fn,
        indices_fn,
        output_fn,
        prefetch_fn=prefetch_fn,
        target=target,
    ](
        axis,
        input.dynamic_shape,
        indices.dynamic_shape,
        output.dynamic_shape,
        context=context,
    )


fn gather_guards[
    input_rank: Int, indices_rank: Int, output_rank: Int
](
    axis: Axis,
    input_shape: StaticIntTuple[input_rank],
    indices_shape: StaticIntTuple[indices_rank],
    output_shape: StaticIntTuple[output_rank],
) raises -> None:
    if int(axis) < 0:
        raise Error("gather kernel does not support negative axis")
    for i in range(axis):
        if output_shape[i] != input_shape[i]:
            raise Error(
                "gather: output_shape[0:axis] does not match"
                " input_shape[0:axis]"
            )
    for i in range(axis, int(axis) + indices_rank):
        if output_shape[i] != indices_shape[i - int(axis)]:
            raise Error(
                "gather: output_shape[axis:axis+indices_rank] does not"
                " match indices_shape"
            )
    for i in range(int(axis) + indices_rank, output_rank):
        if output_shape[i] != input_shape[i - indices_rank + 1]:
            raise Error(
                "gather: output_shape[axis + indices_rank:] does not match"
                " input_shape[axis:]"
            )
    if int(axis) >= input_rank:
        raise Error("gather: axis must be less than input rank")


@always_inline
fn gather_elementwise_fn_wrapper[
    type: DType,
    input_rank: Int,
    indices_type: DType,
    indices_rank: Int,
    output_rank: Int,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    indices_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[indices_type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    coords_rank: Int,
    simd_width: Int,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ] (
            StaticIntTuple[input_rank], StaticIntTuple[indices_rank]
        ) capturing -> None
    ] = None,
](
    axis: Axis,
    input_shape: StaticIntTuple[input_rank],
    indices_shape: StaticIntTuple[indices_rank],
    output_shape: StaticIntTuple[output_rank],
    coords: StaticIntTuple[coords_rank],
):
    @parameter
    @always_inline
    fn gather_elementwise_fn[
        simd_width: Int, rank: Int
    ](idx: StaticIntTuple[rank]):
        # out_coords consists of 3 chunks:
        #   out_coords[0:axis] = input coords[0:axis]
        #   out_coords[axis:axis+indices_rank] = indices_coords
        #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
        # and input_coords[axis] = indices[indices_coords]
        # Get the gather indices.
        var indices_index = StaticIntTuple[indices_rank]()

        # Get the indices of the index.
        @parameter
        for i in range(indices_rank):
            indices_index[i] = idx[i + int(axis)]

        # The index we are gathering.
        var data_index = indices_fn[1, indices_rank](indices_index)

        # Update the indices with the new data index.
        var data_indices = StaticIntTuple[input_rank]()

        var skip_factor = indices_rank - 1

        # Build the indices for the input. We have replaced in index in 'axis'
        # with an index from the indices tensor.
        @parameter
        for i in range(input_rank):
            if i == int(axis):
                data_indices[i] = int(
                    normalize_neg_index(data_index, input_shape[axis])
                )
            elif i > int(axis):
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[i] = idx[i + skip_factor]
            else:
                data_indices[i] = idx[i]

        # Load the the data.
        @parameter
        if prefetch_fn:
            alias func = prefetch_fn.value()
            func[input_rank, indices_rank](data_indices, indices_index)
        var data = input_fn[simd_width, input_rank](data_indices)

        # Store it to the original index.
        output_fn[simd_width, rank](idx, data)

    gather_elementwise_fn[simd_width](coords)


# TODO: Delete / for testing purposes (test_gather.mojo)
@always_inline
fn gather[
    type: DType,
    indices_type: DType,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    indices_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[indices_type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ] (
            StaticIntTuple[input_rank], StaticIntTuple[indices_rank]
        ) capturing -> None
    ] = None,
    target: StringLiteral = "cpu",
    single_thread_blocking_override: Bool = False,
](
    axis: Axis,
    input_shape: StaticIntTuple[input_rank],
    indices_shape: StaticIntTuple[indices_rank],
    output_shape: StaticIntTuple[output_rank],
    context: DeviceContext,
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """
    gather_guards(axis, input_shape, indices_shape, output_shape)
    with Trace[TraceLevel.OP, target=target]("gather"):
        if (
            input_shape.flattened_length() == 0
            or indices_shape.flattened_length() == 0
        ):
            return

        @parameter
        @always_inline
        fn gather_elementwise_fn[
            simd_width: Int, rank: Int
        ](idx: StaticIntTuple[rank]):
            gather_elementwise_fn_wrapper[
                type,
                input_rank,
                indices_type,
                indices_rank,
                output_rank,
                input_fn,
                indices_fn,
                output_fn,
                simd_width=simd_width,
                prefetch_fn=prefetch_fn,
            ](axis, input_shape, indices_shape, output_shape, idx)

        # If we are gathering on the last dimension then we have to be scalar.
        if int(axis) == input_rank - 1:
            elementwise[
                gather_elementwise_fn,
                simd_width=1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)
        else:
            elementwise[
                gather_elementwise_fn,
                simd_width = simdwidthof[type](),
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)


@always_inline
fn gather[
    type: DType,
    indices_type: DType,
    input_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[type, width],
    indices_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank]
    ) capturing -> SIMD[indices_type, width],
    output_fn: fn[width: Int, rank: Int] (
        StaticIntTuple[rank], SIMD[type, width]
    ) capturing -> None,
    input_rank: Int,
    indices_rank: Int,
    output_rank: Int,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ] (
            StaticIntTuple[input_rank], StaticIntTuple[indices_rank]
        ) capturing -> None
    ] = None,
    target: StringLiteral = "cpu",
    single_thread_blocking_override: Bool = False,
](
    axis: Axis,
    input_shape: StaticIntTuple[input_rank],
    indices_shape: StaticIntTuple[indices_rank],
    output_shape: StaticIntTuple[output_rank],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """
    gather_guards(axis, input_shape, indices_shape, output_shape)
    with Trace[TraceLevel.OP, target=target]("gather"):
        if (
            input_shape.flattened_length() == 0
            or indices_shape.flattened_length() == 0
        ):
            return

        @parameter
        @always_inline
        fn gather_elementwise_fn[
            simd_width: Int, rank: Int
        ](idx: StaticIntTuple[rank]):
            gather_elementwise_fn_wrapper[
                type,
                input_rank,
                indices_type,
                indices_rank,
                output_rank,
                input_fn,
                indices_fn,
                output_fn,
                simd_width=simd_width,
                prefetch_fn=prefetch_fn,
            ](axis, input_shape, indices_shape, output_shape, idx)

        # If we are gathering on the last dimension then we have to be scalar.
        if int(axis) == input_rank - 1:
            elementwise[
                gather_elementwise_fn,
                simd_width=1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)
        else:
            elementwise[
                gather_elementwise_fn,
                simd_width = simdwidthof[type](),
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)


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
    target: StringLiteral = "cpu",
    /,
    reduce_fn: OptionalReg[
        fn[
            type: DType, width: Int
        ] (SIMD[type, width], SIMD[type, width]) capturing -> SIMD[type, width]
    ] = None,
](
    data: NDBuffer[output_type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    updates: NDBuffer[output_type, updates_rank],
    output: NDBuffer[output_type, data_rank],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        output_type: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
        data_rank: Rank of input (data) tensor (data_rank >= 1).
        indices_rank: Rank of input (data) tensor (indices_rank >= 1).
        updates_rank: Rank of updates tensor (updates_rank = data_rank +
                      indices_rank - indices_shape[-1] - 1).
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        target: Target cpu or cuda.
        reduce_fn: Reduction function to apply: none (default), add, mul, max,
                   min.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank containing indices for the scatter
                 operation.
        updates: Tensor containing values to update output tensor based on
                 indices tensor.
        output: Tensor of rank data_rank, shaped the same as data tensor.
        context: Pointer to DeviceContext.
    """
    if data.get_shape() != output.get_shape():
        raise Error("Input and output shapes in scatter_nd must be the same.")

    if (
        len(updates.get_shape())
        != data_rank + indices_rank - indices.get_shape()[indices_rank - 1] - 1
    ):
        raise Error(
            "updates rank must be: data_rank + indices_rank -"
            " indices_shape[-1] - 1"
        )

    var output_flat = output.flatten()
    var data_flat = data.flatten()
    var updates_flat = updates.flatten()

    var data_shape = data.get_shape()
    var indices_shape = indices.get_shape()
    var last_shape_of_indices = indices_shape[indices_rank - 1]

    # Depending on r_minus_m = data_rank - last_shape_of_indices,
    # we will be copying (gather):
    #   element (r_minus_m = 0),
    #   row (r_minus_m = 1),
    #   sheet (r_minus_m = 2),
    #   cuboid (r_minus_m = 3), etc.
    var r_minus_m = data_rank - last_shape_of_indices

    @parameter
    if "cuda" in target:
        try:
            # TODO: Does it matter if output.data or output_flat.data (and data)?
            var ctx = context.get_device_context()
            # TODO: Owning = True or False?
            var outp = DeviceBuffer(
                ctx,
                output.data,
                data.num_elements(),
                owning=False,
            )
            var inp = DeviceBuffer(
                ctx, data.data, data.num_elements(), owning=False
            )
            ctx.enqueue_copy_device_to_device(
                outp,
                inp,
            )

        except e:
            abort(e)

    @parameter
    if "cuda" not in target:
        memcpy(output_flat.data, data_flat.data, len(output_flat))

    @__copy_capture(
        r_minus_m, data_shape, last_shape_of_indices, output_flat, updates_flat
    )
    @parameter
    fn update_func[
        simd_width: Int, _rank: Int
    ](_indices_coords: StaticIntTuple[_rank]):
        # Calculate how many elements to copy (this is from the innermost
        # dimensions, and is continuous memory locations).
        var count_copy = 1
        for i in range(r_minus_m):
            count_copy = count_copy * data_shape[data_rank - 1 - i]
        var indices_coords = rebind[StaticIntTuple[_rank]](_indices_coords)

        # Stores the full index on output, where to copy updates to.
        # Zeroing here to avoid doing it selectively within the nested loop below.
        var output_index_tensor = StaticIntTuple[data_rank](0)

        # Stores the full index on updates, where to copy from.
        # Zeroing here to avoid doing it selectively within the nested loop below.
        var updates_index_tensor = StaticIntTuple[updates_rank](0)

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
            var input_ax_dim = data_shape[dim]

            for i in range(_rank):
                indices_index[i] = indices_coords[i]
            indices_index[indices_rank - 1] = dim

            var idx_on_axis = indices[indices_index]
            var pos_idx_on_axis = int(
                normalize_neg_index(idx_on_axis, input_ax_dim)
            )
            output_index_tensor[dim] = pos_idx_on_axis

        # Calculate the updates_offset from where to copy the updates.
        var updates_offset = 0

        for i in range(updates_rank):
            updates_offset = (
                updates_offset + updates.stride(i) * updates_index_tensor[i]
            )

        # Calculate the output_offset to where to copy the updates.
        var output_offset = 0

        for i in range(data_rank):
            output_offset = (
                output_offset + output.stride(i) * output_index_tensor[i]
            )

        # Perform the actual copy of element/slice/sheet/cuboid/etc.
        # Also handling any reduction operation reduce_fn.
        @parameter
        if reduce_fn:
            alias reduction_fn = reduce_fn.value()

            for i in range(count_copy):
                output_flat[output_offset + i] = reduction_fn[output_type, 1](
                    output_flat[output_offset + i],
                    updates_flat[updates_offset + i],
                )

        else:
            for i in range(count_copy):
                output_flat[output_offset + i] = updates_flat[
                    updates_offset + i
                ]

    # TODO: SEE: simd_width > 1
    var iter_shape = StaticIntTuple[indices_rank - 1]()
    for i in range(len(indices.get_shape()) - 1):
        iter_shape[i] = indices.get_shape()[i]

    elementwise[
        update_func,
        simd_width=1,
        use_blocking_impl=single_thread_blocking_override,
        target=target,
    ](iter_shape, context)


@always_inline
fn scatter_nd[
    output_type: DType,
    indices_type: DType,
    data_rank: Int,
    indices_rank: Int,
    updates_rank: Int,
    single_thread_blocking_override: Bool,
    target: StringLiteral = "cpu",
](
    data: NDBuffer[output_type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    updates: NDBuffer[output_type, updates_rank],
    output: NDBuffer[output_type, data_rank],
    context: MojoCallContextPtr = MojoCallContextPtr(),
) raises:
    """Scatter_nd operation without any reduction."""

    scatter_nd_generator[
        output_type,
        indices_type,
        data_rank,
        indices_rank,
        updates_rank,
        single_thread_blocking_override,
        target,
        reduce_fn=None,
    ](data, indices, updates, output, context)


@mogg_register("scatter_nd_shape")
@always_inline
fn scatter_nd_shape[
    input_rank: Int,
    updates_rank: Int,
    indices_rank: Int,
    input_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[input_type, input_rank],
    updates: NDBuffer[input_type, updates_rank],
    indices: NDBuffer[indices_type, indices_rank],
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `scatter_nd` operation, and assert the
    inputs are compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        updates_rank: Rank of the updates tensor.
        indices_rank: Rank of the indices tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        updates: The input tensor.
        indices: The indices tensor.

    Returns:
        The output shape.
    """

    if indices_rank < 1:
        raise Error("[scatter_nd] indices cannot be a scalar")

    var num_sliced_dims = indices.dim(indices_rank - 1)
    if num_sliced_dims > input_rank:
        raise Error(
            "[scatter_nd] cannot slice more dimensions than what input has"
        )

    if indices_rank - 1 + input_rank - num_sliced_dims != updates_rank:
        raise Error(
            "[scatter_nd] requires (updates_rank == indices_rank - 1 +"
            " input_rank - num_sliced_dims)"
        )

    @parameter
    for i in range(indices_rank - 1):
        if indices.dim(i) != updates.dim(i):
            raise Error(
                "[scatter_nd] batch dimensions of indices and updates don't"
                " match"
            )

    for i in range(input_rank - num_sliced_dims):
        if input.dim(i + num_sliced_dims) != updates.dim(i + indices_rank - 1):
            raise Error(
                "[scatter_nd] updated dimensions of input and updates don't"
                " match"
            )

    return input.get_shape()


# ===----------------------------------------------------------------------===#
# Gather Shape
# ===----------------------------------------------------------------------===#


@mogg_register_shape_func("mo.gather")
@always_inline
fn gather_shape[
    output_rank: Int,
    input_rank: Int,
    indices_rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool = False,
](
    input_buf: NDBuffer[input_type, input_rank],
    indices_buf: NDBuffer[indices_type, indices_rank],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
        output_rank: Rank of the output tensor.
        input_rank: Rank of the input tensor.
        indices_rank: Rank of the indices tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.
        axis_buf: The axis tensor.

    Returns:
        The output shape.
    """
    if output_rank != input_rank + indices_rank - 1:
        raise Error(
            "[gather] requires (output_rank == input_rank + indices_rank - 1)"
        )

    # extract hyper parameter
    var axis = int(axis_buf[0])
    if axis < 0:
        axis += input_rank
    if axis < 0 or input_rank <= axis:
        raise Error(
            "[gather] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = StaticIntTuple[output_rank]()

    var input_shape = input_buf.get_shape()
    var indices_shape = indices_buf.get_shape()

    # NOTE it's written this way instead of 3 separate for-loops because
    # currently KGEN unrolling only works for strictly static bounds, but `axis`
    # only becomes static after inlining `axis_buf`.
    @parameter
    for out_dim in range(output_rank):
        if out_dim < axis:
            output_shape[out_dim] = input_shape[out_dim]
        elif out_dim < axis + indices_rank:
            output_shape[out_dim] = indices_shape[out_dim - axis]
        else:
            output_shape[out_dim] = input_shape[out_dim - indices_rank + 1]

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
    input: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    updates: NDBuffer[input_type, rank],
    _axis: Int,
    output: NDBuffer[input_type, rank],
) raises:
    """
    Implements ONNX ScatterElements op which is equivalent to Pytorch scatter.
    """
    constrained[
        indices_type is DType.int32 or indices_type is DType.int64,
        "indices in scatter_elements must be int32 or int64",
    ]()

    if input.get_shape() != output.get_shape():
        raise Error(
            "input and output shape in scatter_elements must be the same"
        )

    if indices.get_shape() != updates.get_shape():
        raise Error(
            "inidices and updates shape in scatter_elements must be the same"
        )

    if not (-rank <= _axis < rank):
        raise Error(
            "axis in scatter_elements must be in the range [-rank, rank)"
        )

    var axis = _axis if _axis >= 0 else _axis + rank

    # Do serial or parallel memcpy depending on output size.
    parallel_memcpy(output.data, input.data, output.size())

    var input_ax_dim = input.get_shape()[axis]

    @__copy_capture(axis, input_ax_dim)
    @parameter
    fn update_func[
        simd_width: Int, _rank: Int
    ](_indices_coords: StaticIntTuple[_rank]):
        var indices_coords = rebind[StaticIntTuple[rank]](_indices_coords)
        var idx_on_axis = indices[indices_coords]
        var output_coords = indices_coords
        output_coords[axis] = int(
            normalize_neg_index(idx_on_axis, input_ax_dim)
        )
        var curr = output[output_coords]
        output[output_coords] = reduce_fn[input_type, 1](
            curr, updates[indices_coords]
        )

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[update_func, 1](indices.get_shape())


@mogg_register("scatter_shape")
@always_inline
fn scatter_elements_shape[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[input_type, rank],
    updates: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    axis: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[rank]:
    """
    Compute the output shape of a `scatter_elements` operation, and assert the
    inputs are compatible.

    Parameters:
        rank: Rank of the input tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        axis_type: Type of the axis tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        updates: The input tensor.
        indices: The indices tensor.
        axis: The axis tensor.

    Returns:
        The output shape.
    """

    # Normalize and check axis
    var axis_int = int(axis[0])
    if axis_int < 0:
        axis_int += rank
    if axis_int < 0 or rank <= axis_int:
        raise Error(
            "[scatter] normalized axis must be within range [0, input_rank)"
        )

    # Check individual dimensions
    @parameter
    for axis in range(rank):
        var input_dim = input.dim(axis)
        var indices_dim = indices.dim(axis)
        var updates_dim = updates.dim(axis)
        if indices_dim != updates_dim:
            raise Error(
                "[scatter] indices and updates must have the same shape"
            )
        if indices_dim > input_dim:
            raise Error(
                "[scatter] indices shape cannot be bigger than input shape"
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
    input: NDBuffer[input_type, rank],
    indices: NDBuffer[indices_type, rank],
    _axis: Int,
    output: NDBuffer[input_type, rank],
) raises:
    """
    Implements ONNX GatherElements op which is equivalent to Pytorch gather.
    """
    constrained[
        indices_type is DType.int32 or indices_type is DType.int64,
        "indices in gather_elements must be int32 or int64",
    ]()

    if indices.get_shape() != output.get_shape():
        raise Error(
            "indices and output shape in gather_elements must be the same"
        )

    if not (-rank <= _axis < rank):
        raise Error(
            "axis in gather_elements must be in the range [-rank, rank)"
        )

    var axis = _axis if _axis >= 0 else _axis + rank

    var input_ax_dim = input.get_shape()[axis]

    @__copy_capture(input_ax_dim, axis)
    @parameter
    fn gather_func[
        simd_width: Int, _rank: Int
    ](_output_coords: StaticIntTuple[_rank]):
        var output_coords = rebind[StaticIntTuple[rank]](_output_coords)
        var idx_on_axis = indices[output_coords]
        var input_coords = output_coords
        input_coords[axis] = int(normalize_neg_index(idx_on_axis, input_ax_dim))
        output[output_coords] = input[input_coords]

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[gather_func, 1](output.get_shape())


# ===----------------------------------------------------------------------===#
# gather_nd shape
# ===----------------------------------------------------------------------===#


@mogg_register("gather_nd_shape")
@always_inline
fn gather_nd_shape[
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
) raises -> StaticIntTuple[output_rank]:
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
        single_thread_blocking_override: If True, then reduction is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.

    Returns:
        The output shape.
    """
    if input_rank < 1 or indices_rank < 1:
        raise Error("[gather_nd] input_rank and indices_rank must be >= 1")

    var indices_shape = indices_buf.get_shape()
    var index_size = indices_shape[indices_rank - 1]
    if index_size < 1 or input_rank - batch_dims < index_size:
        raise Error(
            "[gather_nd] index size must be within range [1, input_rank -"
            " batch_dims]"
        )
    if batch_dims >= indices_rank:
        raise Error("[gather_nd] requires (batch_dims < indices_rank)")

    # compute and return the output shape
    var output_shape = StaticIntTuple[output_rank]()
    var next_out_dim = 0

    var input_shape = input_buf.get_shape()

    @parameter
    for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    @parameter
    for i in range(batch_dims, indices_rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(batch_dims + index_size, input_rank):
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
    data: NDBuffer[type, data_rank],
    indices: NDBuffer[indices_type, indices_rank],
    output: NDBuffer[type, output_rank],
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

    var indices_shape = indices.get_shape()
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

    var last_shape_of_indices = indices_shape[indices_rank - 1]
    var num_elems = indices.num_elements()
    # Reshape indices array, as 3D array. All batch_dims_size elements go to the
    # outermost dimension, and elements of amount equal to indices.shape[-1] go
    # to the innermost.
    # Equivalent to numpy:
    # reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])
    var reshaped_indices = reshape[3](
        indices.make_dims_unknown(),
        StaticIntTuple[3](
            batch_dims_size,
            num_elems // (batch_dims_size * last_shape_of_indices),
            last_shape_of_indices,
        ),
    )

    var reshaped_indices_shape = reshaped_indices.get_shape()
    var data_shape = data.get_shape()

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
    var reshaped_data = reshape[reshaped_data_rank](
        data.make_dims_unknown(), reshaped_data_tuple
    )

    var reshaped_data_shape = reshaped_data.get_shape()

    # idx[] stores the index from where to gather the requested elements.
    var idx_ptr = UnsafePointer[Scalar[DType.index]].alloc(
        reshaped_indices_shape[2]
    )
    var idx = NDBuffer[DType.index, 1](idx_ptr, reshaped_indices_shape[2])

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
                var idx_on_axis = reshaped_indices[batch_dim, outer_dim, constr]
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
