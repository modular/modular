# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import Buffer, NDBuffer


@always_inline
fn cumsum[
    rank: Int, size: Int, type: DType, exclusive: Int, reverse: Int, axis: Int
](output: Buffer[size, type], input: Buffer[size, type]):
    """
    Implements the CumSum operator from the ONNX spec - entry point for 1D input.
    Computes cumulative sum of the input elements.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        rank: Rank of the input and output tensors.
        size: The size of the input and output tensors.
        type: Type of the input and output tensors.
        exclusive: If set to 1, return exclusive sum (top element not included).
        reverse: If set to 1, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation (0 for 1D input).

    Args:
        output: The output tensor.
        input: The input tensor.
    """
    let input_B = NDBuffer[rank, DimList.create_unknown[rank](), type](
        input.data, StaticIntTuple[rank](size)
    )
    let output_B = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, StaticIntTuple[rank](size)
    )

    debug_assert(
        axis == 0,
        "Axis value must be 0 for 1D tensor.",
    )
    cumsum[rank, size, type, exclusive, reverse, axis](output_B, input_B)


@always_inline
fn cumsum[
    rank: Int, size: Int, type: DType, exclusive: Int, reverse: Int, axis: Int
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
):
    """
    Implements the CumSum operator from the ONNX spec:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
    Computes cumulative sum of the input elements along the given axis.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        rank: Rank of the input and output tensors.
        size: The size of the input and output tensors.
        type: Type of the input and output tensors.
        exclusive: If set to 1, return exclusive sum (top element not included).
        reverse: If set to 1, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.

    Args:
        output: The output tensor.
        input: The input tensor.
    """

    debug_assert(
        -rank <= axis < rank,
        "Axis value must be in range [-rank, rank)",
    )

    alias axis_pos = axis if axis >= 0 else axis + rank

    let shape = input.get_shape()

    var inner = 1
    var outer = 1
    var depth = 1
    for i in range(rank):
        if i < axis_pos:
            inner *= shape[i]
        elif i > axis_pos:
            outer *= shape[i]
        else:
            depth = shape[i]

    let output_data = output.flatten()
    let input_data = input.flatten()

    for outer_index in range(outer):
        let outer_index_adj: Int

        @parameter
        if reverse:
            outer_index_adj = (outer - 1) - outer_index
        else:
            outer_index_adj = outer_index

        for inner_index in range(inner):
            var accumulator: SIMD[type, 1] = 0
            let inner_index_adj: Int

            @parameter
            if reverse:
                inner_index_adj = (inner - 1) - inner_index
            else:
                inner_index_adj = inner_index

            for depth_index in range(depth):
                let depth_index_adj: Int

                @parameter
                if reverse:
                    depth_index_adj = (depth - 1) - depth_index
                else:
                    depth_index_adj = depth_index

                let index = outer_index_adj + inner_index_adj * depth * outer + depth_index_adj * outer

                if exclusive:
                    output_data[index] = accumulator
                    accumulator = accumulator + input_data[index]
                else:
                    accumulator = accumulator + input_data[index]
                    output_data[index] = accumulator
