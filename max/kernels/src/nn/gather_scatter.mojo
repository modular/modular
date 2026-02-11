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

from collections.string.string_slice import get_static_string
from math import align_down, ceildiv
from sys import simd_width_of, size_of
from sys.info import CompilationTarget, _current_target
from sys.intrinsics import PrefetchOptions

from algorithm import elementwise, parallel_memcpy, sync_parallelize
from algorithm.functional import tile
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from gpu.host.info import is_cpu, is_gpu
from layout import UNKNOWN_VALUE
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from memory import memcpy
from runtime.asyncrt import DeviceContextPtr, parallelism_level
from runtime.tracing import Trace, TraceLevel, get_safe_task_id
from tensor import ManagedTensorSlice

from utils import Index, IndexList, StaticTuple
from collections import OptionalReg


@always_inline
fn _unsafe_normalize_neg_index(idx: Int, dim_size: Int) -> Int:
    return idx + dim_size if idx < 0 else idx


@always_inline
fn _unsafe_normalize_neg_index[
    dtype: DType, width: Int, out_type: DType = DType.int
](idx: SIMD[dtype, width], dim_size: Int) -> SIMD[out_type, width]:
    return idx.lt(0).select(
        idx.cast[out_type]() + Scalar[out_type](dim_size),
        idx.cast[out_type](),
    )


@always_inline
fn normalize_neg_index(idx: Int, dim_size: Int) raises -> Int:
    """Indices passed to gather and scatter ops may be negative. This performs
    a normalization so that they can be used to index into a buffer.

    Returns val + dim if val < 0 else val
    """
    if -dim_size <= idx < dim_size:
        return _unsafe_normalize_neg_index(idx, dim_size)

    raise Error("indices must be in range [-dim_size, dim_size)")


@always_inline
fn normalize_neg_index[
    dtype: DType, width: Int, out_type: DType = DType.int
](idx: SIMD[dtype, width], dim_size: Int) raises -> SIMD[out_type, width]:
    """Indices passed to gather and scatter ops may be negative. This performs
    a normalization so that they can be used to index into a buffer.

    Returns val + dim if val < 0 else val
    """
    comptime assert (
        dtype.is_integral()
    ), "normalize_neg_index expects index to be an integral dtype"

    var indices = idx.cast[out_type]()
    var bounds = SIMD[out_type, width](dim_size)
    if all(indices.ge(-bounds) & indices.lt(bounds)):
        return _unsafe_normalize_neg_index[out_type=out_type](idx, dim_size)

    raise Error("indices must be in range [-dim_size, dim_size)")


struct Axis(Indexer, Intable, TrivialRegisterPassable):
    var axis: Int

    @always_inline
    fn __init__(out self, axis: Int):
        self.axis = axis

    @always_inline
    fn __init__(out self, axis: Int, rank: Int) raises:
        self.axis = normalize_neg_index(axis, rank)

    @always_inline
    fn __int__(self) -> Int:
        return self.axis

    @always_inline("nodebug")
    fn __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.axis._mlir_value


@always_inline
fn gather_reduce[
    dtype: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: fn[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    indices: TileTensor[DType.int32, ...],
    reduce_init: Scalar[dtype],
):
    """Computes output[i, j, k] = input[indices[i, j], k] and simultaneously
    reduces the output across axis 1 to produce output[i, k].

    The motivating use-case for this is multi-hot embeddings in recommender models.
    This provides similar functionality to Torch's EmbeddingBag layer. In that
    context, i is the batch dimension, j is the multi-hot dimension, and k is
    the embedding dimension.
    """
    comptime assert input.flat_rank == 2
    comptime assert indices.flat_rank == 2
    comptime assert gather_axis == 0
    comptime assert reduce_axis == 1

    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.numel() == 0 or indices.numel() == 0:
        return

    # TODO: find a heuristic to replace the magic number.
    # This is about 4x larger than the default in gather, which makes sense
    # since this kernel performs far fewer writes
    comptime MIN_TASK_COPY_SIZE = 64 * 100 * 32 * 4  # bytes
    var num_threads = parallelism_level()
    var num_tasks = min(
        ceildiv(
            Int(indices.dim[0]())
            * Int(indices.dim[1]())
            * Int(input.dim[1]())
            * size_of[dtype](),
            MIN_TASK_COPY_SIZE,
        ),
        num_threads,
    )

    var out_vecs_per_thread = ceildiv(Int(indices.dim[0]()), num_tasks)

    var output_2d_dims = IndexList[2](
        Int(output.dim[0]()), Int(output.dim[1]())
    )

    @parameter
    if output.flat_rank == 3:
        output_2d_dims[1] = Int(output.dim[2]())

    var output_bind = TileTensor(output.ptr, row_major(Coord(output_2d_dims)))
    var input_bind = TileTensor(
        input.ptr,
        input.layout.make_dynamic[DType.int64](),
    )

    var gather_axis_size = Int(input.dim(gather_axis))

    @always_inline
    @__copy_capture(
        output_bind,
        input_bind,
        indices,
        out_vecs_per_thread,
        gather_axis_size,
    )
    @parameter
    fn task_func(task_id: Int):
        comptime prefetch_offset = -1

        var output = output_bind
        var input = input_bind
        comptime assert output.flat_rank == 2
        comptime assert input.flat_rank == 2
        var row_size = Int(output.dim[1]())

        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var out_vec_start = task_id * out_vecs_per_thread
        var out_vec_end = min(
            (task_id + 1) * out_vecs_per_thread, Int(indices.dim[0]())
        )

        # For multi-hot embeddings reduction, k is the embedding dim and j is the multi-hot dim
        comptime k_tile_sizes = VariadicList[Int](
            2 * simd_width, 1
        ) if CompilationTarget.has_neon() else VariadicList[Int](
            8 * simd_width, 4 * simd_width, 2 * simd_width, simd_width, 1
        )
        # unroll the j loop on neon because it benefits from vectorized
        # blend instructions and avoids conditional flag dependencies
        # does not appear to help on other archs
        comptime j_tile_size = 4 if CompilationTarget.has_neon() else 1

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
                    accums: StaticTuple[SIMD[dtype, simd_width], unroll_factor],
                    j: Int,
                ) -> StaticTuple[SIMD[dtype, simd_width], unroll_factor]:
                    var out = accums
                    var idxs = _unsafe_normalize_neg_index(
                        indices.load[width=unroll_factor](
                            Coord(Idx(i), Idx(j))
                        ),
                        gather_axis_size,
                    )

                    @parameter
                    for unroll_idx in range(0, unroll_factor):
                        var gather_chunk = input.load[width=simd_width](
                            (Idx(Int(idxs[unroll_idx])), Idx(k))
                        )
                        out[unroll_idx] = reduce_fn[dtype, simd_width](
                            accums[unroll_idx], gather_chunk
                        )
                    return out

                var j_residual_start = align_down(
                    Int(indices.dim[1]()), j_tile_size
                )
                var accums = StaticTuple[SIMD[dtype, simd_width], j_tile_size](
                    reduce_init
                )
                for j in range(0, j_residual_start, j_tile_size):
                    accums = reduce_j_tile[j_tile_size](accums, j)

                var accum = SIMD[dtype, simd_width](reduce_init)

                # TODO: use tree reduction here by generalizing simd reduce method
                @parameter
                for unroll_idx in range(j_tile_size):
                    accum = reduce_fn(accum, accums[unroll_idx])

                for j in range(j_residual_start, indices.dim[1](), 1):
                    accum = reduce_j_tile[1](
                        StaticTuple[SIMD[dtype, simd_width], 1](accum), j
                    )[0]

                var out_idx = Coord(Idx(i), Idx(k))
                output.store[width=simd_width](out_idx, accum)

            tile[
                gather_k_tile,
                k_tile_sizes,
            ](0, row_size)
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = i

    sync_parallelize[task_func](num_tasks)


# TODO: Delete / for testing purposes (test_gather.mojo)
fn gather[
    dtype: DType,
    indices_type: DType,
    //,
    *,
    axis: Int,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    indices: TileTensor[indices_type, ...],
    *,
    context: DeviceContext,
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    comptime prefetch_offset = 12  # TODO: search

    var end_indices_ptr = indices.ptr + indices.numel()

    @parameter
    @__copy_capture(end_indices_ptr)
    @always_inline
    fn prefetch_fn[
        _input_rank: Int, _indices_rank: Int
    ](
        _input_coords: IndexList[_input_rank],
        _indices_coords: IndexList[_indices_rank],
    ):
        var __input_coords = _input_coords
        var input_coords = Coord(__input_coords)
        var indices_coords = Coord(_indices_coords)
        comptime assert indices_coords.rank == indices.rank
        comptime assert input_coords.rank == input.rank
        comptime assert indices_coords.flat_rank == indices.flat_rank
        comptime assert input_coords.flat_rank == input.flat_rank

        @parameter
        if prefetch_offset > 0:
            var indices_ptr = indices.ptr_at_offset(indices_coords)
            var indices_remaining = (
                Int(end_indices_ptr) - Int(indices_ptr)
            ) // size_of[indices_type]()
            # assumes that indices are laid out in row major order
            var next_idx_ptr = indices_ptr + min(
                indices_remaining - 1, prefetch_offset
            )
            input_coords[axis] = rebind[input_coords.element_types[axis]](
                Int64(
                    _unsafe_normalize_neg_index(
                        next_idx_ptr.load(),
                        Int(input.dim[axis]()),
                    )
                )
            )
            input.prefetch(input_coords)

    @parameter
    @always_inline
    fn input_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank]) -> SIMD[dtype, width]:
        var coords = Coord(index)
        comptime assert coords.flat_rank == input.flat_rank
        return input.load[width=width](coords)

    @parameter
    @always_inline
    fn indices_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank]) -> SIMD[indices_type, width]:
        var coords = Coord(index)
        comptime assert coords.flat_rank == indices.flat_rank
        return indices.load[width=width](coords)

    @parameter
    @always_inline
    fn output_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank], val: SIMD[dtype, width]):
        var coords = Coord(index)
        comptime assert coords.flat_rank == output.flat_rank
        output.store[width=width](coords, rebind[SIMD[dtype, width]](val))

    gather[
        dtype=dtype,
        indices_type=indices_type,
        input_fn=input_fn,
        indices_fn=indices_fn,
        output_fn=output_fn,
        prefetch_fn=prefetch_fn,
        target=target,
    ](
        Axis(axis),
        coord_to_index_list(input.layout.shape_coord()),
        coord_to_index_list(indices.layout.shape_coord()),
        coord_to_index_list(output.layout.shape_coord()),
        context=context,
    )


fn gather[
    dtype: DType,
    indices_type: DType,
    //,
    *,
    axis: Int,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    indices: TileTensor[indices_type, ...],
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    comptime prefetch_offset = 12  # TODO: search

    var end_indices_ptr = indices.ptr + indices.numel()

    @parameter
    @__copy_capture(end_indices_ptr)
    @always_inline
    fn prefetch_fn[
        _input_rank: Int, _indices_rank: Int
    ](
        _input_coords: IndexList[_input_rank],
        _indices_coords: IndexList[_indices_rank],
    ):
        var __input_coords = _input_coords
        var input_coords = Coord(__input_coords)
        var indices_coords = Coord(_indices_coords)
        comptime assert indices_coords.rank == indices.rank
        comptime assert input_coords.rank == input.rank
        comptime assert indices_coords.flat_rank == indices.flat_rank
        comptime assert input_coords.flat_rank == input.flat_rank

        @parameter
        if prefetch_offset > 0:
            var indices_ptr = indices.ptr_at_offset(indices_coords)
            var indices_remaining = (
                Int(end_indices_ptr) - Int(indices_ptr)
            ) // size_of[indices_type]()
            # assumes that indices are laid out in row major order
            var next_idx_ptr = indices_ptr + min(
                indices_remaining - 1, prefetch_offset
            )
            input_coords[axis] = rebind[input_coords.element_types[axis]](
                Int64(
                    _unsafe_normalize_neg_index(
                        next_idx_ptr.load(),
                        Int(input.dim[axis]()),
                    )
                )
            )
            input.prefetch(input_coords)

    @parameter
    @always_inline
    fn input_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank]) -> SIMD[dtype, width]:
        var coords = Coord(index)
        comptime assert coords.flat_rank == input.flat_rank
        return input.load[width=width](coords)

    @parameter
    @always_inline
    fn indices_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank]) -> SIMD[indices_type, width]:
        var coords = Coord(index)
        comptime assert coords.flat_rank == indices.flat_rank
        return indices.load[width=width](coords)

    @parameter
    @always_inline
    fn output_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank], val: SIMD[dtype, width]):
        var coords = Coord(index)
        comptime assert coords.flat_rank == output.flat_rank
        output.store[width=width](coords, rebind[SIMD[dtype, width]](val))

    gather[
        dtype=dtype,
        indices_type=indices_type,
        input_fn=input_fn,
        indices_fn=indices_fn,
        output_fn=output_fn,
        prefetch_fn=prefetch_fn,
        target=target,
    ](
        Axis(axis),
        coord_to_index_list(input.layout.shape_coord()),
        coord_to_index_list(indices.layout.shape_coord()),
        coord_to_index_list(output.layout.shape_coord()),
        context=context,
    )


fn gather_guards(
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
) raises -> None:
    if Int(axis) < 0:
        raise Error("gather kernel does not support negative axis")
    for i in range(axis):
        if output_shape[i] != input_shape[i]:
            raise Error(
                "gather: output_shape[0:axis] does not match"
                " input_shape[0:axis]"
            )
    for i in range(axis, Int(axis) + indices_shape.size):
        if output_shape[i] != indices_shape[i - Int(axis)]:
            raise Error(
                "gather: output_shape[axis:axis+indices_rank] does not"
                " match indices_shape"
            )
    for i in range(Int(axis) + indices_shape.size, output_shape.size):
        if output_shape[i] != input_shape[i - indices_shape.size + 1]:
            raise Error(
                "gather: output_shape[axis + indices_rank:] does not match"
                " input_shape[axis:]"
            )
    if Int(axis) >= input_shape.size:
        raise Error("gather: axis must be less than input rank")


comptime error_index_fn_type = fn(Int) capturing -> None


@always_inline
fn gather_elementwise_fn_wrapper[
    *,
    dtype: DType,
    indices_type: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    indices_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        indices_type, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    simd_width: Int,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ](IndexList[input_rank], IndexList[indices_rank]) capturing -> None
    ] = None,
    error_index_fn: Optional[error_index_fn_type] = None,
](
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
    coords: IndexList,
):
    @parameter
    @always_inline
    fn gather_elementwise_fn[
        simd_width: Int, rank: Int
    ](idx: IndexList[rank, ...]):
        # out_coords consists of 3 chunks:
        #   out_coords[0:axis] = input coords[0:axis]
        #   out_coords[axis:axis+indices_rank] = indices_coords
        #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
        # and input_coords[axis] = indices[indices_coords]
        # Get the gather indices.
        var indices_index = IndexList[indices_shape.size]()

        # Get the indices of the index.
        @parameter
        for i in range(indices_shape.size):
            indices_index[i] = idx[i + Int(axis)]

        # The index we are gathering.
        var data_index = indices_fn[1, indices_shape.size](indices_index)

        # Update the indices with the new data index.
        var data_indices = IndexList[input_shape.size]()

        var skip_factor = indices_shape.size - 1

        # Build the indices for the input. We have replaced in index in 'axis'
        # with an index from the indices tensor.
        @parameter
        for i in range(input_shape.size):
            if i == Int(axis):
                var normalized_idx = _unsafe_normalize_neg_index(
                    data_index, input_shape[axis]
                )
                data_indices[i] = Int(normalized_idx)

                # Do a real bounds check and provide a nice message on CPU
                # (where error_index_fn is provided). Use debug_assert to
                # validate normalized index is within bounds on GPU and trap,
                # as more detailed checking is costly on GPU.
                @parameter
                if error_index_fn:
                    if not (0 <= Int(normalized_idx) < input_shape[axis]):
                        comptime error_index_func = error_index_fn.value()
                        # Store the invalid index for debugging
                        error_index_func(Int(data_index))
                        return  # Early return on bounds error

                debug_assert[assert_mode="safe"](
                    0 <= Int(normalized_idx) < input_shape[axis],
                    (
                        "Gather index out of bounds. Run on CPU for more"
                        " detailed error checking."
                    ),
                )
            elif i > Int(axis):
                # Skip over any extra indices dimensions. These are essentially new dimensions.
                data_indices[i] = idx[i + skip_factor]
            else:
                data_indices[i] = idx[i]

        # Load the data.
        @parameter
        if prefetch_fn:
            comptime func = prefetch_fn.value()
            func[input_shape.size, indices_shape.size](
                data_indices, indices_index
            )
        var data = input_fn[simd_width, input_shape.size](data_indices)

        # Store it to the original index.
        output_fn[simd_width, rank](idx.canonicalize(), data)

    gather_elementwise_fn[simd_width](coords)


# TODO: Delete / for testing purposes (test_gather.mojo)
@always_inline
fn gather[
    *,
    dtype: DType,
    indices_type: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    indices_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        indices_type, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ](IndexList[input_rank], IndexList[indices_rank]) capturing -> None
    ] = None,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
    *,
    context: DeviceContext,
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """
    gather_guards(axis, input_shape, indices_shape, output_shape)
    with Trace[TraceLevel.OP, target=target](
        "gather", task_id=get_safe_task_id(context)
    ):
        if (
            input_shape.flattened_length() == 0
            or indices_shape.flattened_length() == 0
        ):
            return

        # Create an error reporting location since we cannot raise from an elementwise lambda.
        var error_index: Int = -1  # Initialize invalid index

        @parameter
        @always_inline
        fn error_index_fn(val: Int):
            error_index = val

        comptime error_fn = Optional[error_index_fn_type](
            error_index_fn
        ) if is_cpu[target]() else None

        @parameter
        @always_inline
        fn gather_elementwise_fn[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            gather_elementwise_fn_wrapper[
                dtype=dtype,
                indices_type=indices_type,
                input_fn=input_fn,
                indices_fn=indices_fn,
                output_fn=output_fn,
                simd_width=simd_width,
                prefetch_fn=prefetch_fn,
                error_index_fn=error_fn,
            ](
                axis,
                input_shape.canonicalize(),
                indices_shape.canonicalize(),
                output_shape.canonicalize(),
                idx,
            )

        # If we are gathering on the last dimension then we have to be scalar.
        if Int(axis) == input_shape.size - 1:
            elementwise[
                gather_elementwise_fn,
                simd_width=1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](
                output_shape.canonicalize(),
                context,
            )
        else:
            elementwise[
                gather_elementwise_fn,
                simd_width = simd_width_of[dtype](),
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](
                output_shape.canonicalize(),
                context,
            )

        # Check for bounds errors after elementwise operation completes (CPU only)
        @parameter
        if is_cpu[target]():
            if error_index != -1:
                var invalid_index = error_index
                raise Error(
                    String(
                        "gather index {} is out of bounds for axis {} with"
                        " size {}"
                    ).format(invalid_index, Int(axis), input_shape[axis])
                )


@always_inline
fn gather[
    *,
    dtype: DType,
    indices_type: DType,
    input_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        dtype, width
    ],
    indices_fn: fn[width: Int, rank: Int](IndexList[rank]) capturing -> SIMD[
        indices_type, width
    ],
    output_fn: fn[width: Int, rank: Int](
        IndexList[rank], SIMD[dtype, width]
    ) capturing -> None,
    prefetch_fn: OptionalReg[
        fn[
            input_rank: Int, indices_rank: Int
        ](IndexList[input_rank], IndexList[indices_rank]) capturing -> None
    ] = None,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
    *,
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """
    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()

    gather_guards(axis, input_shape, indices_shape, output_shape)
    with Trace[TraceLevel.OP, target=target](
        "gather", task_id=get_safe_task_id(context)
    ):
        if (
            input_shape.flattened_length() == 0
            or indices_shape.flattened_length() == 0
        ):
            return

        # Create an error reporting location since we cannot raise from an elementwise lambda.
        var error_index: Int = -1  # Initialize invalid index

        @parameter
        @always_inline
        fn error_index_fn(val: Int):
            error_index = val

        comptime error_fn = Optional[error_index_fn_type](
            error_index_fn
        ) if is_cpu[target]() else None

        @parameter
        @always_inline
        fn gather_elementwise_fn[
            simd_width: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]):
            gather_elementwise_fn_wrapper[
                dtype=dtype,
                indices_type=indices_type,
                input_fn=input_fn,
                indices_fn=indices_fn,
                output_fn=output_fn,
                simd_width=simd_width,
                prefetch_fn=prefetch_fn,
                error_index_fn=error_fn,
            ](
                axis,
                input_shape.canonicalize(),
                indices_shape.canonicalize(),
                output_shape.canonicalize(),
                idx,
            )

        # If we are gathering on the last dimension then we have to be scalar.
        if Int(axis) == input_shape.size - 1:
            elementwise[
                gather_elementwise_fn,
                simd_width=1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)
        else:
            elementwise[
                gather_elementwise_fn,
                simd_width = simd_width_of[dtype, target=compile_target](),
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](output_shape, context)

        # Check for bounds errors after elementwise operation completes (CPU only)
        @parameter
        if is_cpu[target]():
            if error_index != -1:
                var invalid_index = error_index
                raise Error(
                    String(
                        "gather index {} is out of bounds for axis {} with"
                        " size {}"
                    ).format(invalid_index, Int(axis), input_shape[axis])
                )


# ===-----------------------------------------------------------------------===#
# scatter_nd op
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ScatterOobIndexStrategy(Equatable, ImplicitlyCopyable, Writable):
    """Valid indices are within the range [-dim_size, dim_size). Indices which
    fall outside of that can be handled using different strategies. Note that
    negative indices are allowed in order to support negative relative indexing.
       Eg: x[-1] == x[dim_size - 1].
    """

    var _value: Int32

    comptime UNDEFINED = Self(0)
    """Users must not pass in invalid indices. If passed, the scatter method may
    raise an error or return undefined results. Today, the scatter_nd kernel uses
    `_unsafe_normalize_neg_index` which will render the output contents invalid."""

    comptime SKIP = Self(1)
    """Users may pass in indices outside of the range [-dim_size, dim_size). In
    which case the corresponding update will be skipped."""


@always_inline
fn scatter_nd_generator[
    output_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
    oob_index_strategy: ScatterOobIndexStrategy = ScatterOobIndexStrategy.UNDEFINED,
    target: StaticString = "cpu",
    reduce_fn: OptionalReg[
        fn[
            dtype: DType, width: Int
        ](SIMD[dtype, width], SIMD[dtype, width]) capturing -> SIMD[
            dtype, width
        ]
    ] = None,
    *,
    _trace_description: StaticString = "scatter_nd",
](
    data: TileTensor[output_type, address_space = AddressSpace.GENERIC, ...],
    indices: TileTensor[
        indices_type, address_space = AddressSpace.GENERIC, ...
    ],
    updates: TileTensor[output_type, address_space = AddressSpace.GENERIC, ...],
    output: TileTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        output_type: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.
        oob_index_strategy: Strategy to handle out of bounds indices.
        target: Target cpu or cuda.
        reduce_fn: Reduction function to apply: none (default), add, mul, max,
                   min.
        _trace_description: A description of the function, used for profiling and tracing.

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank containing indices for the scatter
                 operation.
        updates: Tensor containing values to update output tensor based on
                 indices tensor.
        output: Tensor of rank data_rank, shaped the same as data tensor.
        context: Pointer to DeviceContext.
    """
    with Trace[TraceLevel.OP, target=target](
        _trace_description, task_id=get_safe_task_id(context)
    ):
        if rebind[IndexList[data.rank]](
            coord_to_index_list(data.layout.shape_coord())
        ) != rebind[IndexList[data.rank]](
            coord_to_index_list(output.layout.shape_coord())
        ):
            raise Error(
                "Input and output shapes in scatter_nd must be the same."
            )

        if (
            updates.rank
            != data.rank
            + indices.rank
            - Int(indices.dim[indices.rank - 1]())
            - 1
        ):
            raise Error(
                "updates rank must be: data_rank + indices_rank -"
                " indices_shape[-1] - 1"
            )

        var output_flat = TileTensor(output.ptr, row_major(Idx(output.numel())))
        var data_flat = TileTensor(data.ptr, row_major(Idx(data.numel())))

        # Always copy input to output first.
        @parameter
        if is_gpu[target]():
            # TODO: Does it matter if output.data or output_flat.data (and data)?
            var ctx = context.get_device_context()
            # TODO: Owning = True or False?
            var outp = DeviceBuffer(
                ctx,
                output.ptr,
                data.numel(),
                owning=False,
            )
            var inp = DeviceBuffer(ctx, data.ptr, data.numel(), owning=False)
            ctx.enqueue_copy(
                outp,
                inp,
            )

        @parameter
        if is_cpu[target]():
            memcpy(
                dest=output_flat.ptr,
                src=data_flat.ptr,
                count=output_flat.numel(),
            )

        if updates.numel() == 0:
            # Nothing to update.
            return

        var updates_flat = TileTensor(
            updates.ptr, row_major(Idx(updates.numel()))
        )

        var data_shape = coord_to_index_list(data.layout.shape_coord())
        var indices_shape = coord_to_index_list(indices.layout.shape_coord())
        var last_shape_of_indices = indices_shape[indices.rank - 1]

        # Depending on r_minus_m = data_rank - last_shape_of_indices,
        # we will be copying (gather):
        #   element (r_minus_m = 0),
        #   row (r_minus_m = 1),
        #   sheet (r_minus_m = 2),
        #   cuboid (r_minus_m = 3), etc.
        var r_minus_m = data.rank - last_shape_of_indices

        @__copy_capture(
            r_minus_m,
            data_shape,
            last_shape_of_indices,
            output_flat,
            updates_flat,
        )
        @parameter
        fn update_func[
            simd_width: Int,
            _rank: Int,
            alignment: Int = 1,
        ](_indices_coords: IndexList[_rank]):
            # Calculate how many elements to copy (this is from the innermost
            # dimensions, and is continuous memory locations).
            var count_copy = 1
            for i in range(r_minus_m):
                count_copy = count_copy * data_shape[data.rank - 1 - i]
            var indices_coords = rebind[IndexList[_rank]](_indices_coords)

            # Stores the full index on output, where to copy updates to.
            # Zeroing here to avoid doing it selectively within the nested loop below.
            var output_index_tensor = IndexList[data.rank](0)

            # Stores the full index on updates, where to copy from.
            # Zeroing here to avoid doing it selectively within the nested loop below.
            var updates_index_tensor = IndexList[updates.rank](0)

            # Construct the full index on updates tensor, i.e., where to copy from.
            for dim in range(_rank):
                updates_index_tensor[dim] = indices_coords[dim]

            # Construct the output_index_tensor whose elements contain the indices
            # for each dimension of the output, i.e., where to copy updates to.
            # As part of that we need to construct the indices_index, which is the
            # index to the indices tensor, where we get the elements for the
            # output_index_tensor from.
            var indices_index = IndexList[indices.rank]()
            for dim in range(last_shape_of_indices):
                # Size of current dimension on data.
                # Used to compare to index on this dimension (idx_on_axis).
                var input_ax_dim = data_shape[dim]

                for i in range(_rank):
                    indices_index[i] = indices_coords[i]
                indices_index[indices.rank - 1] = dim

                var indices_coord = Coord(indices_index)
                comptime assert indices_coord.flat_rank == indices.flat_rank
                var idx_on_axis = indices.load[width=1](indices_coord)

                @parameter
                if oob_index_strategy == ScatterOobIndexStrategy.SKIP:
                    # Quit if the index falls outside of [-input_ax_dim, input_ax_dim)
                    if idx_on_axis < Scalar[indices_type](
                        -input_ax_dim
                    ) or idx_on_axis >= Scalar[indices_type](input_ax_dim):
                        return

                output_index_tensor[dim] = Int(
                    _unsafe_normalize_neg_index(idx_on_axis, input_ax_dim)
                )

            # Calculate the updates_offset from where to copy the updates.
            var updates_offset = 0

            for i in range(updates.rank):
                updates_offset = (
                    updates_offset
                    + Int(updates.dynamic_stride(i)) * updates_index_tensor[i]
                )

            # Calculate the output_offset to where to copy the updates.
            var output_offset = 0

            for i in range(data.rank):
                output_offset = (
                    output_offset
                    + Int(output.dynamic_stride(i)) * output_index_tensor[i]
                )

            # Perform the actual copy of element/slice/sheet/cuboid/etc.
            # Also handling any reduction operation reduce_fn.
            @parameter
            if reduce_fn:
                comptime reduction_fn = reduce_fn.value()

                for i in range(count_copy):
                    output_flat[output_offset + i] = reduction_fn[
                        output_type, 1
                    ](
                        output_flat.load[width=1](
                            Coord(Idx(output_offset + i))
                        ),
                        updates_flat.load[width=1](
                            Coord(Idx(updates_offset + i))
                        ),
                    )

            else:
                for i in range(count_copy):
                    output_flat[output_offset + i] = updates_flat[
                        updates_offset + i
                    ]

        # TODO: SEE: simd_width > 1
        var iter_shape = IndexList[indices.rank - 1]()

        @parameter
        for i in range(indices.rank - 1):
            iter_shape[i] = Int(indices.dim[i]())

        comptime trace_description_str = get_static_string[
            "elementwise_impl_" + _trace_description
        ]()

        elementwise[
            update_func,
            simd_width=1,
            use_blocking_impl=single_thread_blocking_override,
            target=target,
            _trace_description=trace_description_str,
        ](iter_shape, context)


@always_inline
fn scatter_nd[
    output_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
    target: StaticString = "cpu",
](
    data: TileTensor[output_type, address_space = AddressSpace.GENERIC, ...],
    indices: TileTensor[
        indices_type, address_space = AddressSpace.GENERIC, ...
    ],
    updates: TileTensor[output_type, address_space = AddressSpace.GENERIC, ...],
    output: TileTensor[
        mut=True, output_type, address_space = AddressSpace.GENERIC, ...
    ],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    """Scatter_nd operation without any reduction."""
    scatter_nd_generator[
        output_type,
        indices_type,
        single_thread_blocking_override,
        oob_index_strategy = ScatterOobIndexStrategy.UNDEFINED,
        target=target,
        reduce_fn=None,
    ](data, indices, updates, output, context)


@always_inline
fn scatter_nd_shape[
    input_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool,
](
    input: TileTensor[input_type, ...],
    updates: TileTensor[input_type, ...],
    indices: TileTensor[indices_type, ...],
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a `scatter_nd` operation, and assert the
    inputs are compatible.

    Parameters:
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

    if indices.rank < 1:
        raise Error("[scatter_nd] indices cannot be a scalar")

    var num_sliced_dims = Int(indices.dim(indices.rank - 1))
    if num_sliced_dims > input.rank:
        raise Error(
            "[scatter_nd] cannot slice more dimensions than what input has"
        )

    if indices.rank - 1 + input.rank - num_sliced_dims != updates.rank:
        raise Error(
            "[scatter_nd] requires (updates_rank == indices_rank - 1 +"
            " input_rank - num_sliced_dims)"
        )

    @parameter
    for i in range(indices.rank - 1):
        if Int(indices.dim(i)) != Int(updates.dim(i)):
            raise Error(
                "[scatter_nd] batch dimensions of indices and updates don't"
                " match"
            )

    for i in range(input.rank - num_sliced_dims):
        if Int(input.dim(i + num_sliced_dims)) != Int(
            updates.dim(i + indices.rank - 1)
        ):
            raise Error(
                "[scatter_nd] updated dimensions of input and updates don't"
                " match"
            )

    return rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )


# ===-----------------------------------------------------------------------===#
# Gather Shape
# ===-----------------------------------------------------------------------===#


@always_inline
fn gather_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    single_thread_blocking_override: Bool = False,
](
    input_buf: TileTensor[input_type, ...],
    indices_buf: TileTensor[indices_type, ...],
    axis: Int,
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        indices_buf: The indices tensor.
        axis: The axis.

    Returns:
        The output shape.
    """
    if output_rank != input_buf.rank + indices_buf.rank - 1:
        raise Error(
            "[gather] requires (output_rank == input_rank + indices_rank - 1)"
        )

    # extract hyper parameter
    var normalized_axis = normalize_neg_index(axis, input_buf.rank)

    # compute and return the output shape
    var output_shape = IndexList[output_rank]()

    var input_shape = coord_to_index_list(input_buf.layout.shape_coord())
    var indices_shape = coord_to_index_list(indices_buf.layout.shape_coord())

    # NOTE it's written this way instead of 3 separate for-loops because
    # currently KGEN unrolling only works for strictly static bounds.
    @parameter
    for out_dim in range(output_rank):
        if out_dim < normalized_axis:
            output_shape[out_dim] = input_shape[out_dim]
        elif out_dim < normalized_axis + indices_buf.rank:
            output_shape[out_dim] = indices_shape[out_dim - normalized_axis]
        else:
            output_shape[out_dim] = input_shape[out_dim - indices_buf.rank + 1]

    return output_shape


# ===-----------------------------------------------------------------------===#
# Scatter Elements
# ===-----------------------------------------------------------------------===#


@always_inline
fn scatter_elements[
    reduce_fn: fn[dtype: DType, width: Int](
        SIMD[dtype, width], SIMD[dtype, width]
    ) capturing -> SIMD[dtype, width],
    rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input: ManagedTensorSlice[dtype=input_type, rank=rank],
    indices: ManagedTensorSlice[dtype=indices_type, rank=rank],
    updates: ManagedTensorSlice[dtype=input_type, rank=rank],
    _axis: Int,
    output: ManagedTensorSlice[dtype=input_type, rank=rank],
) raises:
    """
    Implements ONNX ScatterElements op which is equivalent to Pytorch scatter.
    """
    comptime assert (
        indices_type == DType.int32 or indices_type == DType.int64
    ), "indices in scatter_elements must be int32 or int64"

    if input.shape() != output.shape():
        raise Error(
            "input and output shape in scatter_elements must be the same"
        )

    if indices.shape() != updates.shape():
        raise Error(
            "indices and updates shape in scatter_elements must be the same"
        )

    if not (-rank <= _axis < rank):
        raise Error(
            "axis in scatter_elements must be in the range [-rank, rank)"
        )

    var axis = _axis if _axis >= 0 else _axis + rank

    # Do serial or parallel memcpy depending on output size.
    parallel_memcpy(
        dest=output.unsafe_ptr(), src=input.unsafe_ptr(), count=output.size()
    )

    var input_ax_dim = input.dim_size(axis)

    @__copy_capture(axis, input_ax_dim)
    @parameter
    fn update_func[
        simd_width: Int, _rank: Int, alignment: Int = 1
    ](_indices_coords: IndexList[_rank]):
        var indices_coords = rebind[IndexList[rank]](_indices_coords)
        var idx_on_axis = indices[indices_coords]
        var output_coords = indices_coords
        output_coords[axis] = Int(
            _unsafe_normalize_neg_index(idx_on_axis, input_ax_dim)
        )
        var curr = output[output_coords]
        output[output_coords] = reduce_fn[input_type, 1](
            curr, updates[indices_coords]
        )

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[update_func, 1](indices.shape())


@always_inline
fn scatter_elements_shape[
    input_type: DType,
    indices_type: DType,
    //,
    *,
    single_thread_blocking_override: Bool,
](
    input: TileTensor[input_type, ...],
    updates: TileTensor[input_type, ...],
    indices: TileTensor[indices_type, ...],
    axis: Int,
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a `scatter_elements` operation, and assert the
    inputs are compatible.

    Parameters:
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input: The input tensor.
        updates: The input tensor.
        indices: The indices tensor.
        axis: The axis.

    Returns:
        The output shape.
    """

    # Normalize and check axis
    _ = normalize_neg_index(axis, input.rank)

    # Check individual dimensions
    @parameter
    for axis in range(input.rank):
        var input_dim = Int(input.dim(axis))
        var indices_dim = Int(indices.dim(axis))
        var updates_dim = Int(updates.dim(axis))
        if indices_dim != updates_dim:
            raise Error(
                "[scatter] indices and updates must have the same shape"
            )
        if indices_dim > input_dim:
            raise Error(
                "[scatter] indices shape cannot be bigger than input shape"
            )

    # Return output shape
    return rebind[IndexList[input.rank]](
        coord_to_index_list(input.layout.shape_coord())
    )


# ===-----------------------------------------------------------------------===#
# Gather Elements
# ===-----------------------------------------------------------------------===#


@always_inline
fn gather_elements[
    input_type: DType,
    indices_type: DType,
](
    input: TileTensor[input_type, ...],
    indices: TileTensor[indices_type, ...],
    _axis: Int,
    output: TileTensor[mut=True, input_type, ...],
) raises:
    """
    Implements ONNX GatherElements op which is equivalent to Pytorch gather.
    """
    comptime assert (
        indices_type == DType.int32 or indices_type == DType.int64
    ), "indices in gather_elements must be int32 or int64"

    if rebind[IndexList[input.rank]](
        coord_to_index_list(indices.layout.shape_coord())
    ) != rebind[IndexList[input.rank]](
        coord_to_index_list(output.layout.shape_coord())
    ):
        raise Error(
            "indices and output shape in gather_elements must be the same"
        )

    if not (-input.rank <= _axis < input.rank):
        raise Error(
            "axis in gather_elements must be in the range [-rank, rank)"
        )

    var axis = normalize_neg_index(_axis, input.rank)

    var input_ax_dim = Int(input.dim(axis))

    @__copy_capture(input_ax_dim, axis)
    @parameter
    fn gather_func[
        simd_width: Int, _rank: Int, alignment: Int = 1
    ](_output_coords: IndexList[_rank]):
        var output_coords = Coord(_output_coords)
        comptime assert output_coords.flat_rank == indices.flat_rank
        comptime assert output_coords.flat_rank == output.flat_rank
        var idx_on_axis = indices.load[width=1](output_coords)
        var input_idx = _output_coords
        input_idx[axis] = Int(
            _unsafe_normalize_neg_index(idx_on_axis, input_ax_dim)
        )
        var input_coords = Coord(input_idx)
        comptime assert input_coords.flat_rank == input.flat_rank
        output.store(output_coords, input.load[width=1](input_coords))

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[gather_func, 1](
        coord_to_index_list(output.layout.shape_coord())
    )


# ===-----------------------------------------------------------------------===#
# gather_nd shape
# ===-----------------------------------------------------------------------===#


@always_inline
fn gather_nd_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
    single_thread_blocking_override: Bool = True,
](
    input_buf: TileTensor[input_type, ...],
    indices_buf: TileTensor[indices_type, ...],
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
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
    if input_buf.rank < 1 or indices_buf.rank < 1:
        raise Error("[gather_nd] input_rank and indices_rank must be >= 1")

    var indices_shape = coord_to_index_list(indices_buf.layout.shape_coord())
    var index_size = indices_shape[indices_buf.rank - 1]
    if index_size < 1 or input_buf.rank - batch_dims < index_size:
        raise Error(
            "[gather_nd] index size must be within range [1, input_rank -"
            " batch_dims]"
        )
    if batch_dims >= indices_buf.rank:
        raise Error("[gather_nd] requires (batch_dims < indices_rank)")

    # compute and return the output shape
    var output_shape = IndexList[output_rank]()
    var next_out_dim = 0

    var input_shape = coord_to_index_list(input_buf.layout.shape_coord())

    @parameter
    for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    @parameter
    for i in range(batch_dims, indices_buf.rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(batch_dims + index_size, input_buf.rank):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===-----------------------------------------------------------------------===#
# GatherND
# ===-----------------------------------------------------------------------===#


fn gather_nd[
    dtype: DType,
    indices_type: DType,
    batch_dims: Int,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    data: TileTensor[dtype, ...],
    indices: TileTensor[indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContextPtr,
) raises:
    """
    GatherND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND.
    Based on reference implementation: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gathernd.py.

    Parameters:
        dtype: Type of data tensor.
        indices_type: Type of indices tensor.
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
        return _gather_nd_impl[
            batch_dims,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](data, indices, output)
    else:
        return _gather_nd_impl[
            batch_dims,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](data, indices, output, ctx.get_device_context())


fn _gather_nd_impl[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
    single_thread_blocking_override: Bool = False,
](
    data: TileTensor[dtype, ...],
    indices: TileTensor[indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime assert (
        data.rank >= 1 and indices.rank >= 1
    ), "Constraint: data_rank >= 1 and indices_rank >= 1"

    var indices_shape = coord_to_index_list(indices.layout.shape_coord())
    debug_assert(
        1 <= indices_shape[indices.rank - 1] <= data.rank - batch_dims,
        "Constraint: 1 <= indices_shape[-1] <= data_rank - batch_dims",
    )

    # This is modeled as an elementwise function mapping an index in the
    # output to an index in the input
    @parameter
    fn gather_nd_elementwise_fn[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](output_idx_arg: IndexList[rank]):
        var output_idx = rebind[IndexList[output.rank]](output_idx_arg)
        var data_idx = IndexList[data.rank]()
        var indices_idx = IndexList[indices.rank]()
        var indices_last_dim = Int(indices.dim[indices.rank - 1]())

        # Fill in the known dimensions in our batch_dim
        @parameter
        for i in range(batch_dims):
            data_idx[i] = output_idx[i]

        # Start filling in the index into the indices buffer
        @parameter
        for i in range(0, indices.rank - 1):
            indices_idx[i] = output_idx[i]

        # walk the last dimensions, which are the slices we're gathering
        for i in range(indices_last_dim):
            indices_idx[indices.rank - 1] = i
            var indices_coord = Coord(indices_idx)
            comptime assert indices_coord.flat_rank == indices.flat_rank
            data_idx[batch_dims + i] = Int(indices.load[width=1](indices_coord))

        # fill in the last slices in the input
        num_tail_elems = data.rank - batch_dims - indices_last_dim
        output_start = output.rank - num_tail_elems
        src_start = indices_last_dim + batch_dims
        for i in range(0, num_tail_elems):
            data_idx[src_start + i] = output_idx[output_start + i]

        @parameter
        for i in range(data.rank):
            debug_assert(
                data_idx[i] >= 0 and data_idx[i] < Int(data.dim[i]()),
                "data index out of bounds",
            )

        @parameter
        for i in range(output.rank):
            debug_assert(
                output_idx[i] >= 0 and output_idx[i] < Int(output.dim[i]()),
                "output index out of bounds",
            )

        var data_coord = Coord(data_idx)
        var output_coord = Coord(output_idx)
        comptime assert data_coord.flat_rank == data.flat_rank
        comptime assert output_coord.flat_rank == output.flat_rank
        output.store[width=simd_width](
            output_coord, data.load[width=simd_width](data_coord)
        )

    comptime compile_target = _current_target() if is_cpu[
        target
    ]() else get_gpu_target()
    comptime target_simd_width = simd_width_of[dtype, target=compile_target]()

    # Only use SIMD if:
    #   - the input data is contiguous
    #   - the slices at the end of the input are not scalars
    #   - the last dimension of the slices are evenly divisible by simd_width
    var slice_rank = (
        data.rank - batch_dims - Int(indices.dim[indices.rank - 1]())
    )
    var slice_last_dim = (
        Int(output.dim[output.rank - 1]()) if slice_rank > 0 else 1
    )

    comptime assert data.rank - 1 != UNKNOWN_VALUE
    var use_simd = (
        data.dynamic_stride(data.rank - 1) == 1
        and (slice_last_dim % target_simd_width) == 0
    )

    @parameter
    if is_cpu[target]():
        if use_simd:
            elementwise[
                gather_nd_elementwise_fn,
                target_simd_width,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](coord_to_index_list(output.layout.shape_coord()))
        else:
            elementwise[
                gather_nd_elementwise_fn,
                1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](coord_to_index_list(output.layout.shape_coord()))
    else:
        debug_assert(
            Bool(ctx), "Must provide DeviceContext if executing on GPU."
        )
        var cuda_ctx = ctx.value()
        if use_simd:
            elementwise[
                gather_nd_elementwise_fn,
                target_simd_width,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](coord_to_index_list(output.layout.shape_coord()), cuda_ctx)
        else:
            elementwise[
                gather_nd_elementwise_fn,
                1,
                use_blocking_impl=single_thread_blocking_override,
                target=target,
            ](coord_to_index_list(output.layout.shape_coord()), cuda_ctx)


# ===-----------------------------------------------------------------------===#
# ScatterSetConstant
# ===-----------------------------------------------------------------------===#


fn scatter_set_constant[
    data_type: DType,
    index_type: DType,
    //,
    target: StaticString,
    single_thread_blocking_override: Bool = False,
](
    data: TileTensor[mut=True, data_type, ...],
    indices: TileTensor[index_type, ...],
    fill_value: Scalar[data_type],
    ctx: DeviceContextPtr,
) raises:
    """
    Scatter the fill_value into the data at the specified indices.

    Example:
        Suppose we have a 3x3 matrix `data` initialized to zeros:

        data = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

        And `indices` is a 2D tensor with shape [2, 2]:

        indices = [[0, 1],
                   [2, 0]]

        If `fill_value` is 5, after calling `scatter_set_constant`, `data` will be:

        data = [[0, 5, 0],
                [0, 0, 0],
                [5, 0, 0]]

    Arguments:
        data: The data to scatter the updates into.
        indices: The indices to scatter the updates into.
        fill_value: The value to fill the data with.
        ctx: The device context.
    """
    comptime assert (
        index_type.is_integral()
    ), "index_type must be an integer dtype"
    comptime assert data.flat_rank == 2, "scatter_set: data must have rank 2"
    comptime assert (
        indices.flat_rank == 2
    ), "scatter_set: indices must have rank 2"
    debug_assert(
        Int(indices.dim[1]()) == 2,
        "scatter_set: indices must have shape [total_seq_len, 2]",
    )

    @always_inline
    @parameter
    fn scatter_set_constant_fn[
        width: Int, rank_: Int, alignment: Int = 1
    ](idx: IndexList[rank_]):
        comptime assert rank_ == 1, "scatter_set_constant_fn: rank must be 1"

        data[Int(indices[idx[0], 0]), Int(indices[idx[0], 1])] = fill_value

    var dispatch_shape = IndexList[1](Int(indices.dim[0]()))
    elementwise[
        func=scatter_set_constant_fn,
        simd_width=1,
        target=target,
        use_blocking_impl=single_thread_blocking_override,
        _trace_description="scatter_set_constant",
    ](dispatch_shape, ctx)
