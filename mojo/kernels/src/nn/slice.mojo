# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, NDBuffer
from Functional import elementwise
from List import Dim, DimList
from LLCL import OutputChainPtr
from Math import div_ceil
from Index import StaticIntTuple
from Int import Int
from Range import range
from DType import DType
from TypeUtilities import rebind


fn slice_as_view[
    type: DType, index_type: DType, rank: Int
](
    tensor: NDBuffer[rank, DimList[rank].create_unknown(), type],
    starts: Buffer[Dim(), index_type],
    ends: Buffer[Dim(), index_type],
    steps: Buffer[Dim(), index_type],
) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:

    var new_shape: StaticIntTuple[rank.__as_mlir_index()]
    var new_stride: StaticIntTuple[rank.__as_mlir_index()]

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.data

    for i in range(rank):
        var start = starts[i].to_int()
        var stop = ends[i].to_int()
        let step = steps[i].to_int()

        if start < 0:
            start = start + tensor.dim(i)

        if stop < 0:
            stop = stop + tensor.dim(i)

        # Allow start and stop to truncate like numpy and torch allow.
        if start < 0:
            start = 0
        elif start >= tensor.dim(i):
            start = tensor.dim(i) - 1

        if stop < 0:
            stop = -1
        elif stop >= tensor.dim(i) and step > 0:
            stop = tensor.dim(i)
        elif stop >= tensor.dim(i) and step < 0:
            stop = tensor.dim(i) - 1

        let new_offset = start * tensor.stride(i)
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = tensor.stride(i) * step

        # If the steps are positive we traverse from start, if negative from stop.
        if step > 0:
            new_shape[i] = div_ceil(stop - start, step)
        else:
            new_shape[i] = div_ceil(start - stop, -step)

    # Create the new view
    return NDBuffer[rank, DimList[rank].create_unknown(), type](
        new_data, new_shape, tensor.dynamic_dtype, new_stride
    )


fn slice_as_copy[
    type: DType, index_type: DType, in_rank: Int
](
    output: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    tensor: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    start: Buffer[Dim(), index_type],
    end: Buffer[Dim(), index_type],
    step: Buffer[Dim(), index_type],
    out_chain: OutputChainPtr,
):

    # Apply slice to the tensor
    var sliced = slice_as_view[type, index_type, in_rank](
        tensor, start, end, step
    )

    # Copy lambda sliced view into output buffer.
    @always_inline
    fn copy[
        simd_width: Int, rank: __mlir_type.index
    ](idx: StaticIntTuple[rank]):
        let index = rebind[StaticIntTuple[in_rank.__as_mlir_index()]](idx)
        output.simd_store[simd_width](
            index, sliced.simd_load[simd_width](index)
        )

    # Invoke copy.
    elementwise[in_rank.__as_mlir_index(), 1, 1, copy](
        output.dynamic_shape, out_chain
    )
