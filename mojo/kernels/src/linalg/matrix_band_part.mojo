# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module implements matrix band part functions."""

from algorithm.functional import _elementwise_impl
from memory.buffer import NDBuffer
from runtime.llcl import OutputChainPtr
from runtime.tracing import TraceLevel

from utils.index import Index, StaticIntTuple
from utils.list import DimList


@always_inline
fn matrix_band_part[
    type: DType,
    int_type: DType,
    cond_type: DType,
    rank: Int,
    simd_width: Int,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    num_lower: NDBuffer[1, DimList.create_unknown[1](), int_type],
    num_upper: NDBuffer[1, DimList.create_unknown[1](), int_type],
    exclude_buf: NDBuffer[1, DimList.create_unknown[1](), cond_type],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    let lower_diagonal_index = num_lower[0].to_int()
    let upper_diagonal_index = num_upper[0].to_int()
    let exclude = exclude_buf[0] != 0

    constrained[rank >= 2, "Matrix band only supports rank >=2"]()

    @parameter
    @always_inline
    fn func[
        simd_width: Int, inner_rank: Int
    ](index: StaticIntTuple[inner_rank]):
        let idx = rebind[StaticIntTuple[rank]](index)

        let row = idx[rank - 2]
        let col = idx[rank - 1]

        var in_band = (
            lower_diagonal_index < 0 or (row - col) <= lower_diagonal_index
        ) and (upper_diagonal_index < 0 or (col - row) <= upper_diagonal_index)
        if exclude:
            in_band = not in_band

        if in_band:
            output[idx] = input[idx]
        else:
            output[idx] = 0

    _elementwise_impl[rank, 1, single_thread_blocking_override, func](
        input.dynamic_shape,
        out_chain,
    )
