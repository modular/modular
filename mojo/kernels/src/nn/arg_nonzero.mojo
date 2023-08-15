# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer
from Functional import (
    vectorize_unroll,
    async_parallelize,
    _get_start_indices_of_nth_subvolume,
)
from Index import StaticIntTuple
from List import Dim, DimList
from LLCL import OutputChainPtr
from Range import range
from Tracing import Trace, TraceLevel


# ===----------------------------------------------------------------------===#
# where
# ===----------------------------------------------------------------------===#


@always_inline
fn where[
    type: DType,
    output_type: DType,
    rank: Int,
](
    input_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
    output_buffer: NDBuffer[2, DimList.create_unknown[2](), output_type],
    out_chain: OutputChainPtr,
):
    """Gather the indices of all non-zero elements in input buffer storing
    the indices in the output_buffer.

    Parameters:
        type: The element type
        output_type: The integer type to store the indices in.
        rank: The rank of the tensor.

    Args:
        input_buffer: The tensor to count the non-zeros in.
        output_buffer: The indices of all non-zero elements.
        out_chain: The our chain to attach results to.
    """

    out_chain.trace[TraceLevel.OP]("mojo.where")

    let numel = input_buffer.dynamic_shape.flattened_length()
    if numel == 0:
        out_chain.mark_ready()
        return

    var j: Int = 0
    for i in range(numel):
        let indices = _get_start_indices_of_nth_subvolume[rank, 0](
            i, input_buffer.dynamic_shape
        )
        if input_buffer[indices] != 0:
            var out_indices = StaticIntTuple[2]()
            out_indices[0] = j
            j += 1

            # Write each of the output values to the where output.
            @unroll
            for k in range(rank):
                out_indices[1] = k
                output_buffer[out_indices] = indices[k]

    out_chain.mark_ready()


# Where has the shape 2D shape [NumNonZeros, InputRank]
@always_inline
fn where_shape[
    type: DType,
    rank: Int,
    single_thread_blocking_override: Bool,
](
    input_buffer: NDBuffer[rank, DimList.create_unknown[rank](), type],
) -> StaticIntTuple[2]:
    """Return [NumNonZeros, InputRank] where NumNonZeros are the number of
    non-zero elements in the input.

    Parameters:
        type: The element type
        output_type: The integer type to store the indices in.
        single_thread_blocking_override: This op can block

    Args:
        input_buffer: The tensor to count the non-zeros in.

    Returns:
        Shape of the where kernel for this input [NumNonZeros, InputRank]
    """

    var shape = StaticIntTuple[2]()
    shape[1] = rank

    let numel = input_buffer.dynamic_shape.flattened_length()

    var j: Int = 0
    for i in range(numel):
        let indices = _get_start_indices_of_nth_subvolume[rank, 0](
            i, input_buffer.dynamic_shape
        )
        if input_buffer[indices] != 0:
            j += 1

    shape[0] = j
    return shape
