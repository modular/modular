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

from std.collections.string.string_slice import get_static_string
from std.math import align_down, ceildiv, iota
from std.sys import align_of, simd_width_of, size_of
from std.sys.info import CompilationTarget, _current_target

from std.algorithm import elementwise, parallel_memcpy, sync_parallelize
from std.algorithm.functional import tile
from std.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.gpu.host.info import is_cpu, is_gpu
from layout import (
    Coord,
    Idx,
    TileTensor,
    UNKNOWN_VALUE,
    coord_to_index_list,
    row_major,
)
from std.memory import memcpy
from std.runtime.asyncrt import parallelism_level
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id
from extensibility import ManagedTensorSlice

from std.utils import IndexList, StaticTuple
from std.collections import OptionalReg


@always_inline
def _unsafe_normalize_neg_index(idx: Int, dim_size: Int) -> Int:
    return idx + dim_size if idx < 0 else idx


@always_inline
def _unsafe_normalize_neg_index[
    dtype: DType, width: SIMDSize, out_type: DType = DType.int
](idx: SIMD[dtype, width], dim_size: Int) -> SIMD[out_type, width]:
    return idx.lt(0).select(
        idx.cast[out_type]() + Scalar[out_type](dim_size),
        idx.cast[out_type](),
    )


@always_inline
def normalize_neg_index(idx: Int, dim_size: Int) raises -> Int:
    """Indices passed to gather and scatter ops may be negative. This performs
    a normalization so that they can be used to index into a buffer.

    Returns val + dim if val < 0 else val

    Raises:
        If the index is out of range [-dim_size, dim_size).
    """
    if -dim_size <= idx < dim_size:
        return _unsafe_normalize_neg_index(idx, dim_size)

    raise Error("indices must be in range [-dim_size, dim_size)")


@always_inline
def normalize_neg_index[
    dtype: DType, width: SIMDSize, out_type: DType = DType.int
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
    def __init__(out self, axis: Int):
        self.axis = axis

    @always_inline
    def __init__(out self, axis: Int, rank: Int) raises:
        self.axis = normalize_neg_index(axis, rank)

    @always_inline
    def __int__(self) -> Int:
        return self.axis

    @doc_hidden
    @always_inline("nodebug")
    def __mlir_index__(self) -> __mlir_type.index:
        """Convert to index.

        Returns:
            The corresponding __mlir_type.index value.
        """
        return self.axis.__mlir_index__()


@always_inline
def gather_reduce[
    dtype: DType,
    gather_axis: Int,
    reduce_axis: Int,
    simd_width: Int,
    reduce_fn: def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) thin -> SIMD[dtype, width],
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, DType.int32, ...],
    reduce_init: Scalar[dtype],
    ctx: Optional[DeviceContext] = None,
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
    # For the compiler to index below
    comptime assert indices.flat_rank >= 2
    comptime assert input.flat_rank >= 2

    # Short-circuit for trivial cases, and to avoid divide-by-zero
    if input.num_elements() == 0 or indices.num_elements() == 0:
        return

    # TODO: find a heuristic to replace the magic number.
    # This is about 4x larger than the default in gather, which makes sense
    # since this kernel performs far fewer writes
    comptime MIN_TASK_COPY_SIZE = 64 * 100 * 32 * 4  # bytes
    var num_threads = parallelism_level(ctx)
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

    comptime if output.flat_rank == 3:
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
    def task_func(task_id: Int):
        comptime prefetch_offset = -1

        var output = output_bind
        var input = input_bind
        comptime assert output.flat_rank == 2
        comptime assert input.flat_rank == 2
        comptime assert input.flat_rank >= 2
        var row_size = Int(output.dim[1]())

        # each thread gets a chunk of output embedding vectors to avoid inter-thread reduction
        var out_vec_start = task_id * out_vecs_per_thread
        var out_vec_end = min(
            (task_id + 1) * out_vecs_per_thread, Int(indices.dim[0]())
        )

        # For multi-hot embeddings reduction, k is the embedding dim and j is the multi-hot dim
        comptime k_tile_sizes = [
            2 * simd_width,
            1,
        ] if CompilationTarget.has_neon() else [
            8 * simd_width,
            4 * simd_width,
            2 * simd_width,
            simd_width,
            1,
        ]
        # unroll the j loop on neon because it benefits from vectorized
        # blend instructions and avoids conditional flag dependencies
        # does not appear to help on other archs
        comptime j_tile_size = 4 if CompilationTarget.has_neon() else 1

        for i in range(out_vec_start, out_vec_end):

            @always_inline
            @__copy_capture(input, indices, output)
            @parameter
            def gather_k_tile[simd_width: Int](k: Int):
                @always_inline
                @parameter
                def reduce_j_tile[
                    unroll_factor: Int
                ](
                    accums: StaticTuple[SIMD[dtype, simd_width], unroll_factor],
                    j: Int,
                ) -> StaticTuple[SIMD[dtype, simd_width], unroll_factor]:
                    var out = accums
                    var idxs = _unsafe_normalize_neg_index(
                        indices.load[width=unroll_factor](Coord(i, j)),
                        gather_axis_size,
                    )

                    comptime for unroll_idx in range(0, unroll_factor):
                        var gather_chunk = input.load[
                            width=simd_width, alignment=1
                        ]((Int(idxs[unroll_idx]), k))
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
                comptime for unroll_idx in range(j_tile_size):
                    accum = reduce_fn(accum, accums[unroll_idx])

                for j in range(j_residual_start, Int(indices.dim[1]()), 1):
                    accum = reduce_j_tile[1](
                        StaticTuple[SIMD[dtype, simd_width], 1](accum), j
                    )[0]

                var out_idx = Coord(i, k)
                output.store[width=simd_width, alignment=1](out_idx, accum)

            tile[
                gather_k_tile,
                k_tile_sizes,
            ](0, row_size)
            # TODO(MOCO-2074): Suppress false positive unused var warning.
            _ = i

    sync_parallelize[task_func](num_tasks, ctx)


# TODO: Delete / for testing purposes (test_gather.mojo)
def gather[
    dtype: DType,
    indices_type: DType,
    //,
    *,
    axis: Int,
    target: StaticString = "cpu",
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    *,
    context: DeviceContext,
) raises:
    """Gather operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather.

    Note that this is NOT the same as the default PyTorch gather (which is equivalent to
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#gatherelements).
    """

    comptime prefetch_offset = 12  # TODO: search

    var end_indices_ptr = indices.ptr + indices.num_elements()

    @parameter
    @__copy_capture(end_indices_ptr)
    @always_inline
    def prefetch_fn[
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

        comptime if prefetch_offset > 0:
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

    @always_inline
    def input_fn[
        width: Int, _rank: Int, element_alignment: Int
    ](index: IndexList[_rank]) {var input} -> SIMD[dtype, width]:
        var coords = Coord(index)
        comptime assert input.flat_rank >= coords.flat_rank
        return input.load[
            width=width, alignment=element_alignment * align_of[dtype]()
        ](coords)

    @always_inline
    def indices_fn[
        width: Int, _rank: Int
    ](index: IndexList[_rank]) {var indices} -> SIMD[indices_type, width]:
        var coords = Coord(index)
        comptime assert indices.flat_rank >= coords.flat_rank
        return indices.load[width=width, alignment=align_of[indices_type]()](
            coords
        )

    @always_inline
    def output_fn[
        width: SIMDSize, _rank: Int, element_alignment: Int
    ](index: IndexList[_rank], val: SIMD[dtype, width]) {var output}:
        var coords = Coord(index)
        comptime assert output.flat_rank >= coords.flat_rank
        output.store[
            width=width, alignment=element_alignment * align_of[dtype]()
        ](coords, rebind[SIMD[dtype, width]](val))

    gather[
        dtype,
        indices_type,
        prefetch_fn=prefetch_fn,
        target=target,
    ](
        Axis(axis),
        coord_to_index_list(input.layout.shape_coord()),
        coord_to_index_list(indices.layout.shape_coord()),
        coord_to_index_list(output.layout.shape_coord()),
        input_fn=input_fn,
        indices_fn=indices_fn,
        output_fn=output_fn,
        context=context,
    )


def gather_guards(
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
    for i in range(Int(axis), Int(axis) + indices_shape.size):
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


@always_inline
def gather_elementwise_fn_wrapper[
    dtype: DType,
    indices_type: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int, element_alignment: Int](
        IndexList[rank]
    ) -> SIMD[dtype, width],
    IndicesFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[indices_type, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDSize, rank: Int, element_alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    *,
    simd_width: Int,
    prefetch_fn: OptionalReg[
        def[
            input_rank: Int, indices_rank: Int
        ](IndexList[input_rank], IndexList[indices_rank]) capturing -> None
    ] = None,
    target: StaticString = "cpu",
    element_alignment: Int = 1,
](
    input_fn: InputFnType,
    indices_fn: IndicesFnType,
    output_fn: OutputFnType,
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
    coords: IndexList,
    error_index_ptr: OptionalReg[UnsafePointer[Int, MutAnyOrigin]] = None,
):
    # out_coords consists of 3 chunks:
    #   out_coords[0:axis] = input coords[0:axis]
    #   out_coords[axis:axis+indices_rank] = indices_coords
    #   out_coords[axis + indices_rank:] = input_coords[axis + 1:]
    # and input_coords[axis] = indices[indices_coords]
    # Get the gather indices.
    var indices_index = IndexList[indices_shape.size]()

    # Get the indices of the index.
    comptime for i in range(indices_shape.size):
        indices_index[i] = coords[i + Int(axis)]

    # The index we are gathering.
    var data_index = indices_fn[1, indices_shape.size](indices_index)

    # Update the indices with the new data index.
    var data_indices = IndexList[input_shape.size]()

    var skip_factor = indices_shape.size - 1

    # Build the indices for the input. We have replaced in index in 'axis'
    # with an index from the indices tensor.
    comptime for i in range(input_shape.size):
        if i == Int(axis):
            var normalized_coords = _unsafe_normalize_neg_index(
                data_index, input_shape[axis]
            )
            data_indices[i] = Int(normalized_coords)

            # Do a real bounds check and provide a nice message on CPU.
            # Use debug_assert to validate normalized index is within bounds
            # on GPU and trap, as more detailed checking is costly on GPU.
            comptime if is_cpu[target]():
                if not (0 <= Int(normalized_coords) < input_shape[axis]):
                    error_index_ptr.value()[] = Int(data_index)
                    return  # Early return on bounds error

            debug_assert[assert_mode="safe"](
                0 <= Int(normalized_coords) < input_shape[axis],
                (
                    "Gather index out of bounds. Run on CPU for more"
                    " detailed error checking."
                ),
            )
        elif i > Int(axis):
            # Skip over any extra indices dimensions. These are essentially new dimensions.
            data_indices[i] = coords[i + skip_factor]
        else:
            data_indices[i] = coords[i]

    # Load the data.
    comptime if prefetch_fn:
        comptime func = prefetch_fn.value()
        func[input_shape.size, indices_shape.size](data_indices, indices_index)
    var data = input_fn[simd_width, input_shape.size, element_alignment](
        data_indices
    )

    # Store it to the original index.
    output_fn[simd_width, coords.size, element_alignment](
        coords.canonicalize(), data
    )


# TODO: Delete / for testing purposes (test_gather.mojo)
@always_inline
def gather[
    dtype: DType,
    indices_type: DType,
    InputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int, element_alignment: Int](
        IndexList[rank]
    ) -> SIMD[dtype, width],
    IndicesFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int](IndexList[rank]) -> SIMD[indices_type, width],
    OutputFnType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: SIMDSize, rank: Int, element_alignment: Int](
        IndexList[rank], SIMD[dtype, width]
    ) -> None,
    *,
    prefetch_fn: OptionalReg[
        def[
            input_rank: Int, indices_rank: Int
        ](IndexList[input_rank], IndexList[indices_rank]) capturing -> None
    ] = None,
    target: StaticString = "cpu",
](
    axis: Axis,
    input_shape: IndexList,
    indices_shape: IndexList,
    output_shape: IndexList,
    *,
    input_fn: InputFnType,
    indices_fn: IndicesFnType,
    output_fn: OutputFnType,
    context: DeviceContext,
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
        var error_index: Int = -1
        var error_index_ptr = OptionalReg[UnsafePointer[Int, MutAnyOrigin]](
            None
        )
        comptime if is_cpu[target]():
            error_index_ptr = OptionalReg[UnsafePointer[Int, MutAnyOrigin]](
                UnsafePointer[Int, MutAnyOrigin](to=error_index)
            )

        @always_inline
        def gather_elementwise_fn[
            simd_width: Int, alignment: Int = 1
        ](idx: Coord) {
            var axis,
            var input_shape,
            var indices_shape,
            var output_shape,
            var input_fn,
            var indices_fn,
            var output_fn,
            var error_index_ptr,
        }:
            gather_elementwise_fn_wrapper[
                dtype,
                indices_type,
                simd_width=simd_width,
                prefetch_fn=prefetch_fn,
                target=target,
                element_alignment=alignment,
            ](
                input_fn,
                indices_fn,
                output_fn,
                axis,
                input_shape.canonicalize(),
                indices_shape.canonicalize(),
                output_shape.canonicalize(),
                coord_to_index_list(idx),
                error_index_ptr,
            )

        # If we are gathering on the last dimension then we have to be scalar.
        if Int(axis) == input_shape.size - 1:
            elementwise[
                simd_width=1,
                target=target,
                _trace_description="gather",
            ](
                gather_elementwise_fn,
                Coord(output_shape),
                context,
            )
        else:
            elementwise[
                simd_width=simd_width_of[dtype, target=compile_target](),
                target=target,
                _trace_description="gather",
            ](
                gather_elementwise_fn,
                Coord(output_shape),
                context,
            )

        # Check for bounds errors after elementwise operation completes (CPU only)
        comptime if is_cpu[target]():
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
def scatter_nd_generator[
    output_type: DType,
    indices_type: DType,
    //,
    oob_index_strategy: ScatterOobIndexStrategy = ScatterOobIndexStrategy.UNDEFINED,
    target: StaticString = "cpu",
    reduce_fn: OptionalReg[
        def[
            dtype: DType, width: SIMDSize
        ](SIMD[dtype, width], SIMD[dtype, width]) thin -> SIMD[dtype, width]
    ] = None,
    *,
    _trace_description: StaticString = "scatter_nd",
](
    data: TileTensor[
        mut=False, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    indices: TileTensor[
        mut=False, indices_type, address_space=AddressSpace.GENERIC, ...
    ],
    updates: TileTensor[
        mut=False, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    """
    Implements ONNX ScatterND operation as defined in https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND.

    Parameters:
        output_type: Type of data, updates, and output tensors.
        indices_type: Type of the indices tensor.
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

        var output_flat = TileTensor(
            output.ptr, row_major(output.num_elements())
        )
        var data_flat = TileTensor(data.ptr, row_major(data.num_elements()))

        # Always copy input to output first.
        comptime if is_gpu[target]():
            # TODO: Does it matter if output.data or output_flat.data (and data)?
            var ctx = context
            # TODO: Owning = True or False?
            var outp = DeviceBuffer(
                ctx,
                output.ptr,
                data.num_elements(),
                owning=False,
            )
            var inp = DeviceBuffer(
                ctx, data.ptr, data.num_elements(), owning=False
            )
            ctx.enqueue_copy(
                outp,
                inp,
            )

        comptime if is_cpu[target]():
            memcpy(
                dest=output_flat.ptr,
                src=data_flat.ptr,
                count=output_flat.num_elements(),
            )

        if updates.num_elements() == 0:
            # Nothing to update.
            return

        var updates_flat = TileTensor(
            updates.ptr, row_major(updates.num_elements())
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

        @always_inline
        def update_func[
            simd_width: Int,
            alignment: Int = 1,
        ](_indices_coords: Coord) {
            var r_minus_m,
            var data,
            var data_shape,
            var last_shape_of_indices,
            var output_flat,
            var updates_flat,
            var indices,
            var updates,
            var output,
        }:
            # Calculate how many elements to copy (this is from the innermost
            # dimensions, and is continuous memory locations).
            var count_copy = 1
            for i in range(r_minus_m):
                count_copy = count_copy * data_shape[data.rank - 1 - i]
            var indices_coords = coord_to_index_list(_indices_coords)

            # Stores the full index on output, where to copy updates to.
            # Zeroing here to avoid doing it selectively within the nested loop below.
            var output_index_tensor = IndexList[data.rank](0)

            # Stores the full index on updates, where to copy from.
            # Zeroing here to avoid doing it selectively within the nested loop below.
            var updates_index_tensor = IndexList[updates.rank](0)

            # Construct the full index on updates tensor, i.e., where to copy from.
            for dim in range(_indices_coords.rank):
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

                for i in range(_indices_coords.rank):
                    indices_index[i] = indices_coords[i]
                indices_index[indices.rank - 1] = dim

                var indices_coord = Coord(indices_index)
                var idx_on_axis = indices.load[width=1](indices_coord)

                comptime if oob_index_strategy == ScatterOobIndexStrategy.SKIP:
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
            comptime if reduce_fn:
                comptime reduction_fn = reduce_fn.value()

                for i in range(count_copy):
                    output_flat[output_offset + i] = reduction_fn[
                        output_type, 1
                    ](
                        output_flat.load[width=1](Coord(output_offset + i)),
                        updates_flat.load[width=1](Coord(updates_offset + i)),
                    )

            else:
                for i in range(count_copy):
                    output_flat[output_offset + i] = updates_flat[
                        updates_offset + i
                    ]

        @always_inline
        def update_element_func[
            simd_width: Int,
            alignment: Int = 1,
        ](_coords: Coord) {
            var data_shape,
            var last_shape_of_indices,
            var output_flat,
            var updates_flat,
            var indices,
            var updates,
            var output,
        }:
            # One update element per invocation: the leading coordinates
            # select the index row, the last coordinate selects the element
            # within the row's slice. This exposes rows x slice_elems
            # parallelism, where iterating over index rows alone caps
            # parallelism at the row count and leaves each thread serially
            # copying an entire slice.
            var coords = coord_to_index_list(_coords)
            var elem = coords[indices.rank - 1]

            # Index into the indices tensor naming this update row.
            var indices_index = IndexList[indices.rank]()
            for i in range(indices.rank - 1):
                indices_index[i] = coords[i]

            # Base offset on output addressed by this index row.
            var output_base = 0
            for dim in range(last_shape_of_indices):
                # Size of current dimension on data.
                # Used to compare to index on this dimension (idx_on_axis).
                var input_ax_dim = data_shape[dim]
                indices_index[indices.rank - 1] = dim
                var idx_on_axis = indices.load[width=1](Coord(indices_index))

                comptime if oob_index_strategy == ScatterOobIndexStrategy.SKIP:
                    # Quit if the index falls outside of [-input_ax_dim, input_ax_dim)
                    if idx_on_axis < Scalar[indices_type](
                        -input_ax_dim
                    ) or idx_on_axis >= Scalar[indices_type](input_ax_dim):
                        return

                output_base = output_base + Int(
                    output.dynamic_stride(dim)
                ) * Int(_unsafe_normalize_neg_index(idx_on_axis, input_ax_dim))

            # Base offset on updates for this row; the copied slice occupies
            # the trailing dimensions contiguously. Both `updates_base + elem`
            # and `output_base + elem` below treat `elem` as a flat offset into
            # the trailing slice, so this path requires `updates` and `output`
            # to be row-major contiguous in their slice dimensions (the leading
            # row dimensions may be strided). A strided slice would silently
            # produce wrong results.
            var updates_base = 0
            for i in range(indices.rank - 1):
                updates_base = (
                    updates_base + Int(updates.dynamic_stride(i)) * coords[i]
                )

            # The launch below only selects simd_width > 1 when slice_elems,
            # the row strides, and the base pointers are all multiples of the
            # vector width, so these accesses stay aligned.
            comptime access_alignment = align_of[
                SIMD[output_type, simd_width], target=get_gpu_target()
            ]()
            var update_vec = updates_flat.load[
                width=simd_width, alignment=access_alignment
            ](Coord(updates_base + elem))

            comptime if reduce_fn:
                comptime reduction_fn = reduce_fn.value()
                update_vec = reduction_fn[output_type, simd_width](
                    output_flat.load[
                        width=simd_width, alignment=access_alignment
                    ](Coord(output_base + elem)),
                    update_vec,
                )

            output_flat.store[alignment=access_alignment](
                Coord(output_base + elem), update_vec
            )

        comptime trace_description_str = get_static_string[
            "elementwise_impl_" + _trace_description
        ]()

        comptime if is_gpu[target]():
            # Iterate over indices.shape[:-1] x slice_elems, one update
            # element per thread. The CPU path below iterates over index rows
            # with a contiguous inner copy per row, which is the right shape
            # for CPU but caps GPU parallelism at the row count.
            var slice_elems = 1
            for i in range(r_minus_m):
                slice_elems = slice_elems * data_shape[data.rank - 1 - i]

            var iter_shape = IndexList[indices.rank]()
            comptime for i in range(indices.rank - 1):
                iter_shape[i] = Int(indices.dim[i]())
            iter_shape[indices.rank - 1] = slice_elems

            # Vectorize across the slice when every access is provably
            # aligned: each thread touches `base + elem` where elem is a
            # multiple of the vector width (elementwise packs the last
            # iteration dim), so the bases must also be multiples of the
            # width — slice_elems, every stride feeding a base offset, and
            # the raw pointers all have to be pack-divisible.
            comptime pack = simd_width_of[
                output_type, target=get_gpu_target()
            ]()
            comptime vector_alignment = align_of[
                SIMD[output_type, pack], target=get_gpu_target()
            ]()

            var vector_safe = (
                slice_elems % pack == 0
                and Int(output.ptr) % vector_alignment == 0
                and Int(updates.ptr) % vector_alignment == 0
            )
            for dim in range(last_shape_of_indices):
                vector_safe = vector_safe and (
                    Int(output.dynamic_stride(dim)) % pack == 0
                )
            for i in range(indices.rank - 1):
                vector_safe = vector_safe and (
                    Int(updates.dynamic_stride(i)) % pack == 0
                )

            if vector_safe:
                elementwise[
                    simd_width=pack,
                    target=target,
                    _trace_description=trace_description_str,
                ](update_element_func, Coord(iter_shape), context)
            else:
                elementwise[
                    simd_width=1,
                    target=target,
                    _trace_description=trace_description_str,
                ](update_element_func, Coord(iter_shape), context)
        else:
            # Iterate over indices.shape[:-1], i.e. one update vector per
            # index row.
            var iter_shape = IndexList[indices.rank - 1]()
            comptime for i in range(indices.rank - 1):
                iter_shape[i] = Int(indices.dim[i]())

            elementwise[
                simd_width=1,
                target=target,
                _trace_description=trace_description_str,
            ](update_func, Coord(iter_shape), context)


@always_inline
def scatter_nd[
    output_type: DType,
    indices_type: DType,
    //,
    target: StaticString = "cpu",
](
    data: TileTensor[
        mut=False, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    indices: TileTensor[
        mut=False, indices_type, address_space=AddressSpace.GENERIC, ...
    ],
    updates: TileTensor[
        mut=False, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    output: TileTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    context: DeviceContext,
) raises:
    """Scatter_nd operation without any reduction."""
    scatter_nd_generator[target=target](data, indices, updates, output, context)


@always_inline
def scatter_nd_shape[
    input_type: DType,
    indices_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    updates: TileTensor[mut=False, input_type, ...],
    indices: TileTensor[mut=False, indices_type, ...],
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a `scatter_nd` operation, and assert the
    inputs are compatible.

    Parameters:
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.

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

    comptime for i in range(indices.rank - 1):
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
def gather_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    indices_buf: TileTensor[mut=False, indices_type, ...],
    axis: Int,
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

    Parameters:
        output_rank: Rank of the output tensor.
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.

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
    comptime for out_dim in range(output_rank):
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
def scatter_elements[
    rank: Int,
    input_type: DType,
    indices_type: DType,
    *,
    ReduceFn: ImplicitlyCopyable
    & RegisterPassable
    & def[dtype: DType, width: SIMDSize](
        SIMD[dtype, width], SIMD[dtype, width]
    ) -> SIMD[dtype, width],
](
    input: ManagedTensorSlice[dtype=input_type, rank=rank, ...],
    indices: ManagedTensorSlice[dtype=indices_type, rank=rank, ...],
    updates: ManagedTensorSlice[dtype=input_type, rank=rank, ...],
    _axis: Int,
    output: ManagedTensorSlice[dtype=input_type, rank=rank, ...],
    ctx: DeviceContext,
    reduce_fn: ReduceFn,
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

    def update_func[
        simd_width: Int, alignment: Int = 1
    ](indices_coords: Coord) {var}:
        var idx_on_axis = indices.to_tile_tensor()[indices_coords]
        var output_coords = coord_to_index_list(indices_coords)
        output_coords[axis] = Int(
            _unsafe_normalize_neg_index(idx_on_axis, input_ax_dim)
        )
        var curr = output.to_tile_tensor()[Coord(output_coords)]
        output.to_tile_tensor()[Coord(output_coords)] = reduce_fn[
            dtype=input_type, width=1
        ](curr, updates.to_tile_tensor()[indices_coords])

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[1](update_func, indices.shape_coord(), ctx)


@always_inline
def scatter_elements_shape[
    input_type: DType,
    indices_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    updates: TileTensor[mut=False, input_type, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    axis: Int,
) raises -> IndexList[input.rank]:
    """
    Compute the output shape of a `scatter_elements` operation, and assert the
    inputs are compatible.

    Parameters:
        input_type: Type of the input tensor.
        indices_type: Type of the indices tensor.

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
    comptime for axis in range(input.rank):
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
def gather_elements[
    input_type: DType,
    indices_type: DType,
](
    input: TileTensor[mut=False, input_type, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    _axis: Int,
    output: TileTensor[mut=True, input_type, ...],
    ctx: DeviceContext,
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

    def gather_func[
        simd_width: Int, alignment: Int = 1
    ](output_coords: Coord) {var}:
        comptime assert indices.flat_rank >= output_coords.flat_rank
        comptime assert output.flat_rank >= output_coords.flat_rank
        var idx_on_axis = indices.load[width=1](output_coords)
        var input_idx = coord_to_index_list(output_coords)
        input_idx[axis] = Int(
            _unsafe_normalize_neg_index(idx_on_axis, input_ax_dim)
        )
        var input_coords = Coord(input_idx)
        comptime assert input.flat_rank >= input_coords.flat_rank
        output.store(output_coords, input.load[width=1](input_coords))

    # cannot use simd_width > 1 here because consecutive updates are not contiguous
    elementwise[1](gather_func, output.layout.shape_coord(), ctx)


# ===-----------------------------------------------------------------------===#
# gather_nd shape
# ===-----------------------------------------------------------------------===#


@always_inline
def gather_nd_shape[
    output_rank: Int,
    input_type: DType,
    indices_type: DType,
    batch_dims: Int,
](
    input_buf: TileTensor[mut=False, input_type, ...],
    indices_buf: TileTensor[mut=False, indices_type, ...],
) raises -> IndexList[output_rank]:
    """
    Compute the output shape of a `gather` operation, and assert the inputs are
    compatible.

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

    comptime for i in range(batch_dims):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    comptime for i in range(batch_dims, indices_buf.rank - 1):
        output_shape[next_out_dim] = indices_shape[i]
        next_out_dim += 1

    for i in range(batch_dims + index_size, input_buf.rank):
        output_shape[next_out_dim] = input_shape[i]
        next_out_dim += 1

    return output_shape


# ===-----------------------------------------------------------------------===#
# GatherND
# ===-----------------------------------------------------------------------===#


def gather_nd[
    dtype: DType,
    indices_type: DType,
    //,
    batch_dims: Int,
    target: StaticString = "cpu",
](
    data: TileTensor[mut=False, dtype, ...],
    indices: TileTensor[mut=False, indices_type, ...],
    output: TileTensor[mut=True, dtype, ...],
    ctx: DeviceContext,
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

    Args:
        data: Tensor of rank data_rank >= 1.
        indices: Tensor of rank indices_rank >= 1. All index values are expected
                 to be within bounds [-s, s-1] along axis of size s. It is an
                 error if any of the index values are out of bounds.
        output: Tensor of rank data_rank + indices_rank - indices_shape[-1] - 1 - b.
        ctx: The device context as prepared by the graph compiler.

    """
    comptime assert (
        data.rank >= 1 and indices.rank >= 1
    ), "Constraint: data_rank >= 1 and indices_rank >= 1"

    var indices_shape = coord_to_index_list(indices.layout.shape_coord())
    assert (
        1 <= indices_shape[indices.rank - 1] <= data.rank - batch_dims
    ), "Constraint: 1 <= indices_shape[-1] <= data_rank - batch_dims"

    # This is modeled as an elementwise function mapping an index in the
    # output to an index in the input
    def gather_nd_elementwise_fn[
        simd_width: Int, alignment: Int = 1
    ](output_idx_arg: Coord) {var}:
        var output_idx = coord_to_index_list(output_idx_arg)
        var data_idx = IndexList[data.rank]()
        var indices_idx = IndexList[indices.rank]()
        var indices_last_dim = Int(indices.dim[indices.rank - 1]())

        # Fill in the known dimensions in our batch_dim
        comptime for i in range(batch_dims):
            data_idx[i] = output_idx[i]

        # Start filling in the index into the indices buffer
        comptime for i in range(0, indices.rank - 1):
            indices_idx[i] = output_idx[i]

        # walk the last dimensions, which are the slices we're gathering
        for i in range(indices_last_dim):
            indices_idx[indices.rank - 1] = i
            var indices_coord = Coord(indices_idx)
            data_idx[batch_dims + i] = Int(indices.load[width=1](indices_coord))

        # fill in the last slices in the input
        num_tail_elems = data.rank - batch_dims - indices_last_dim
        output_start = output.rank - num_tail_elems
        src_start = indices_last_dim + batch_dims
        for i in range(0, num_tail_elems):
            data_idx[src_start + i] = output_idx[output_start + i]

        comptime for i in range(data.rank):
            assert data_idx[i] >= 0 and data_idx[i] < Int(
                data.dim[i]()
            ), "data index out of bounds"

        comptime for i in range(output.rank):
            assert Int(output_idx[i].value()) >= 0 and Int(
                output_idx[i].value()
            ) < Int(output.dim[i]()), "output index out of bounds"

        var data_coord = Coord(data_idx)
        var output_coord = Coord(output_idx)
        output.store[width=simd_width, alignment=1](
            output_coord, data.load[width=simd_width, alignment=1](data_coord)
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

    if use_simd:
        elementwise[
            target_simd_width,
            target=target,
            _trace_description="gather_nd",
        ](gather_nd_elementwise_fn, output.layout.shape_coord(), ctx)
    else:
        elementwise[
            1,
            target=target,
            _trace_description="gather_nd",
        ](gather_nd_elementwise_fn, output.layout.shape_coord(), ctx)


# ===-----------------------------------------------------------------------===#
# ScatterSetConstant
# ===-----------------------------------------------------------------------===#


def scatter_set_constant[
    data_type: DType,
    index_type: DType,
    //,
    target: StaticString,
](
    data: TileTensor[mut=True, data_type, ...],
    indices: TileTensor[mut=False, index_type, ...],
    fill_value: Scalar[data_type],
    ctx: DeviceContext,
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
    # An inner dimension other than 2 would make the elementwise body read
    # indices[i, 1] out of bounds, scattering to a garbage location. Always
    # raise (not just assert) since this is a once-per-launch host check.
    if Int(indices.dim[1]()) != 2:
        raise Error("scatter_set: indices must have shape [total_seq_len, 2]")

    @always_inline
    def scatter_set_constant_fn[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var}:
        comptime assert idx.rank == 1, "scatter_set_constant_fn: rank must be 1"

        data[Int(indices[idx[0], 0]), Int(indices[idx[0], 1])] = fill_value

    var dispatch_shape = Coord(Int(indices.dim[0]()))
    elementwise[
        simd_width=1,
        target=target,
        _trace_description="scatter_set_constant",
    ](scatter_set_constant_fn, dispatch_shape, ctx)


def apply_packed_bitmask[
    dtype: DType,
    //,
    target: StaticString,
](
    output: TileTensor[mut=True, dtype, ...],
    logits: TileTensor[dtype, ...],
    packed: TileTensor[DType.int32, ...],
    fill_value: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """Apply a packed-int32 grammar bitmask to logits in a single fused pass.

    Unpacks a packed bitmask (1 bit per token, 32 tokens per `int32` word) and
    masks `logits` with it without ever materializing a bool tensor: for each
    `(b, v)`, the token is kept when bit `v % 32` of word `packed[b, v // 32]`
    is set, otherwise `output[b, v]` is set to `fill_value` (the masked-out
    sentinel, e.g. a large negative number). This replaces a CPU unpack +
    `ops.where` in constrained decoding.

    Args:
        output: Masked logits, shape `[batch, vocab]`.
        logits: Input logits, shape `[batch, vocab]`.
        packed: Packed `int32` bitmask, shape `[batch, ceil(vocab / 32)]`. A set
            bit means the token is grammar-valid. Extra trailing bits beyond
            `vocab` (32-bit alignment padding from llguidance) are never read.
        fill_value: Value written for masked-out (grammar-invalid) tokens.
        ctx: The device context.
    """
    comptime assert output.flat_rank == 2, "apply_packed_bitmask: output rank 2"
    comptime assert logits.flat_rank == 2, "apply_packed_bitmask: logits rank 2"
    comptime assert packed.flat_rank == 2, "apply_packed_bitmask: packed rank 2"

    @always_inline
    def apply_packed_bitmask_fn[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var}:
        comptime assert idx.rank == 2, "apply_packed_bitmask_fn: rank must be 2"
        # A `width`-wide block can straddle a 32-bit word boundary (elementwise
        # may emit an unaligned tail block), but spans at most two consecutive
        # words for `width <= 32`, so resolve each lane's word individually.
        comptime assert (
            width <= 32
        ), "apply_packed_bitmask: simd_width must be <= 32"
        var b = Int(idx[0].value())
        var v = Int(idx[1].value())
        var tok = Int32(v) + iota[DType.int32, width]()
        var base = v >> 5
        var w0 = SIMD[DType.int32, width](packed[b, base][0])
        # Second word only feeds the spilled lanes; clamp the index so the
        # no-spill case never loads out of bounds.
        var last_word = Int(packed.dim[1]()) - 1
        var w1 = SIMD[DType.int32, width](
            packed[b, min(base + 1, last_word)][0]
        )
        var word = (tok >> 5).ne(Int32(base)).select(w1, w0)
        var keep = ((word >> (tok & 31)) & 1).ne(0)
        var values = logits.load[width=width]((b, v))
        var filled = SIMD[dtype, width](fill_value)
        output.store((b, v), keep.select(values, filled))

    comptime simd_width = simd_width_of[dtype]()
    var dispatch_shape = Coord(Int(output.dim[0]()), Int(output.dim[1]()))
    elementwise[
        simd_width=simd_width,
        target=target,
        _trace_description="apply_packed_bitmask",
    ](apply_packed_bitmask_fn, dispatch_shape, ctx)
