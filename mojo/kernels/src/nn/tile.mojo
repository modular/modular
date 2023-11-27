# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from runtime.llcl import OutputChainPtr, OwningOutputChainPtr, Runtime

# TODO: This implementation supports up to 4 dimensions.

# Note: ONNX spec specifies that `tile` behaves like Numpy's tile, but without
#       broadcast. This means that `repeats` is a 1D int64 tensor of the SAME
#       length as input's rank (unlike Numpy that allows `repeats` to have more
#       or less elements than the input's rank).


@always_inline
fn tile[
    rank: Int, type: DType, rank_repeats: Int, type_repeats: DType
](
    input: NDBuffer[
        rank,
        DimList.create_unknown[rank](),
        type,
    ],
    repeats: NDBuffer[
        rank_repeats,
        DimList.create_unknown[rank_repeats](),
        type_repeats,
    ],
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    """
    Implements the `Tile` operator from the ONNX spec. This behaves like Numpy
    tile, but without broadcast.

    Parameters:
        rank: Rank of the input and output tensors.
        type: Type of the input and output tensors.
        rank_repeats: Rank of the repeats tensor.
        type_repeats: Type of the repeats tensor.

    Args:
        input: The input tensor. Currently <= 4 dimensions are supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        output: The output tensor. Has the same dimensions and type as input.
        out_chain: The OutputChainPtr used to mark competion or error of the task.
    """

    if rank > 4:
        return out_chain.mark_error(
            "Currently only inputs of up to three dimensions are supported."
        )

    if rank_repeats != 1 or type_repeats != DType.int64:
        return out_chain.mark_error(
            "Rank of repeats tensor needs to be one-dimensional and of int64"
            " type."
        )

    if rank != repeats.dim(0):
        return out_chain.mark_error(
            "Length of repeats tensor should be equal to the rank of the input"
            " tensor."
        )

    var num_dp_input = 1
    var num_depth_input = 1
    var num_rows_input = 1

    @parameter
    if rank == 4:
        num_dp_input = input.dim(rank - 4)

    @parameter
    if rank >= 3:
        num_depth_input = input.dim(rank - 3)

    @parameter
    if rank >= 2:
        num_rows_input = input.dim(rank - 2)
    let num_cols_input = input.dim(rank - 1)

    let num_cols_output = output.dim(rank - 1)
    let repeats_len = repeats.dim(0)

    # Initializes output by first copying in the original input to the
    # appropriate output elements, and then handles tiling across the column
    # (last) dimension.
    # e.g., for:
    #   input:  [[1, 2, 3],
    #            [4, 5, 6]]
    #   and repeats = [2,2], the below will handle:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [X, X, X, X, X, X],
    #            [X, X, X, X, X, X]]
    #   where 'X' denotes parts of the output that are not yet calculated.
    #   These (i.e., for dimensions beyond the innermost) are handled later.
    for dp in range(num_dp_input):
        for d in range(num_depth_input):
            for r in range(num_rows_input):
                # print(dp, d, r)
                let input_src_index = dp * num_depth_input * num_rows_input * num_cols_input + d * num_rows_input * num_cols_input + r * num_cols_input
                let output_src_index = dp * num_depth_input * int(
                    repeats[repeats_len - 3]
                ) * num_rows_input * int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * int(
                    repeats[repeats_len - 1]
                ) + d * num_rows_input * int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * int(
                    repeats[repeats_len - 1]
                ) + r * num_cols_input * int(
                    repeats[repeats_len - 1]
                )
                let output_src_stride = num_cols_input
                let count = output_src_stride
                for rep in range(int(repeats[repeats_len - 1])):
                    let src_ptr = input.data.offset(input_src_index)
                    let dst_ptr = output.data.offset(
                        output_src_index + rep * output_src_stride
                    )
                    memcpy[type](dst_ptr, src_ptr, count)

    # Handles tiling across the second lowest dimension (if tensor rank >= 2).
    # Continuing with the example above, this will handle the 'X's, which
    # correspond to the first '2' in the repeats = [2, 2] tensor.
    # The result is:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6]]
    # Moving from the inner to the outermost dimension, we can memcpy to
    # replicate contiguous memory areas (representing a dimension to be tiled).
    @parameter
    if rank >= 2:
        let src_index_stride = num_rows_input * num_cols_input * int(
            repeats[repeats_len - 1]
        )
        let count = src_index_stride
        for dp in range(num_dp_input):
            for d in range(num_depth_input):
                let src_index = dp * num_depth_input * int(
                    repeats[repeats_len - 3]
                ) * num_rows_input * int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * int(
                    repeats[repeats_len - 1]
                ) + d * num_rows_input * int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * int(
                    repeats[repeats_len - 1]
                )
                for rep in range(int(repeats[repeats_len - 2]) - 1):
                    let src_ptr = output.data.offset(src_index)
                    let dst_ptr = output.data.offset(
                        src_index + (rep + 1) * src_index_stride
                    )
                    memcpy[type](dst_ptr, src_ptr, count)

    # Handles tiling across the third dimension from the end (if tensor rank >= 3)
    @parameter
    if rank >= 3:
        let src_index_stride = num_depth_input * int(
            repeats[repeats_len - 2]
        ) * num_rows_input * num_cols_input * int(repeats[repeats_len - 1])
        let count = src_index_stride
        for dp in range(num_dp_input):
            let src_index = dp * num_depth_input * int(
                repeats[repeats_len - 3]
            ) * num_rows_input * int(
                repeats[repeats_len - 2]
            ) * num_cols_input * int(
                repeats[repeats_len - 1]
            )
            for rep in range(int(repeats[repeats_len - 3]) - 1):
                let src_ptr = output.data.offset(src_index)
                let dst_ptr = output.data.offset(
                    src_index + (rep + 1) * src_index_stride
                )
                memcpy[type](dst_ptr, src_ptr, count)

    # Handles tiling across the fourth dimension from the end(if tensor rank >= 3)
    @parameter
    if rank == 4:
        let src_index_stride = num_dp_input * int(
            repeats[repeats.dim(0) - 3]
        ) * num_depth_input * int(
            repeats[repeats.dim(0) - 2]
        ) * num_rows_input * num_cols_input * int(
            repeats[repeats.dim(0) - 1]
        )
        let count = src_index_stride
        let src_index = 0
        for rep in range(int(repeats[repeats.dim(0) - 4]) - 1):
            let src_ptr = output.data.offset(src_index)
            let dst_ptr = output.data.offset(
                src_index + (rep + 1) * src_index_stride
            )
            memcpy[type](dst_ptr, src_ptr, count)

    out_chain.mark_ready()


@always_inline
fn tile_shape[
    input_rank: Int,
    repeats_rank: Int,
    input_type: DType,
    repeats_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    repeats_buf: NDBuffer[
        repeats_rank, DimList.create_unknown[repeats_rank](), repeats_type
    ],
) -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `tile` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor (can be any shape).
        repeats_rank: Rank of the repeats tensor (must be 1).
        input_type: Type of the input tensor.
        repeats_type: Type of the repeats tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The input tensor.
        repeats_buf: The repeats tensor.

    Returns:
        The output shape.
    """

    # TODO(#17512)
    debug_assert(repeats_rank == 1, "repeats rank must be 1")
    debug_assert(
        repeats_buf.dim(0) == input_rank, "repeats length must match input rank"
    )

    # Compute and return the output shape.
    var output_shape = StaticIntTuple[input_rank]()
    for i in range(input_rank):
        output_shape[i] = input_buf.dim(i) * int(repeats_buf[i])

    return output_shape
