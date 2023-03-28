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
from IO import print


fn slice_as_view[
    type: DType, rank: Int
](
    tensor: NDBuffer[rank, DimList[rank].create_unknown(), type],
    starts: NDBuffer[1, DimList[1].create_unknown(), DType.index],
    stops: NDBuffer[1, DimList[1].create_unknown(), DType.index],
    steps: NDBuffer[1, DimList[1].create_unknown(), DType.index],
) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:

    var new_shape: StaticIntTuple[rank.__as_mlir_index()]
    var new_stride: StaticIntTuple[rank.__as_mlir_index()]

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.data

    for i in range(rank):
        var start = Int(starts.data.offset(i).load().value)
        var stop = Int(stops.data.offset(i).load().value)
        let step = Int(steps.data.offset(i).load().value)

        if start < 0:
            start = start + tensor.dim(i)

        if stop < 0:
            stop = stop + tensor.dim(i)

        # Offset the pointer to refect the new starting point.
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
        new_data,
        rebind[StaticIntTuple[rank.__as_mlir_index()]](new_shape),
        tensor.dynamic_dtype,
        rebind[StaticIntTuple[rank.__as_mlir_index()]]((new_stride)),
    )


fn slice_as_copy[
    type: DType, in_rank: Int
](
    output: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    tensor: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    start: NDBuffer[1, DimList[1].create_unknown(), DType.index],
    stop: NDBuffer[1, DimList[1].create_unknown(), DType.index],
    step: NDBuffer[1, DimList[1].create_unknown(), DType.index],
    chain: OutputChainPtr,
):

    # Apply slice to the tensor
    var sliced = slice_as_view[type, in_rank](tensor, start, stop, step)

    # Copy lambda sliced view into output buffer.
    @always_inline
    fn copy[
        simd_width: Int, rank: __mlir_type.index
    ](idx: StaticIntTuple[rank]):
        var index = rebind[StaticIntTuple[in_rank.__as_mlir_index()]](idx)

        var in1 = sliced.simd_load[simd_width](index)
        output.simd_store[simd_width](index, in1)

    # Invoke copy.
    elementwise[in_rank.__as_mlir_index(), 1, 1, copy](
        output.dynamic_shape, chain
    )
