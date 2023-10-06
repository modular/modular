# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import NDBuffer
from runtime.llcl import Runtime, OutputChainPtr, OwningOutputChainPtr

# TODO: This implementation supports up to 3 dimensions.

# Note: ONNX spec specifies that `tile` behaves like Numpy's tile, but without
#       broadcast. This means that `repeats` is a 1D int64 tensor of the SAME
#       length as input's rank (unlike Numpy that allows `repeats` to have more
#       or less elements than the input's rank).


@always_inline
fn tile[
    rank: Int, type: DType, rank_repeats: Int, type_repeats: DType
](
    output: NDBuffer[rank, DimList.create_unknown[rank](), type],
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
        output: The output tensor. Has the same dimensions and type as input.
        input: The input tensor. Currently <= 3 dimensions are supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        out_chain: The OutputChainPtr used to mark competion or error of the task.
    """

    if rank > 3:
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

    var num_depth_input = 1
    var num_rows_input = 1

    @parameter
    if rank == 3:
        num_depth_input = input.dim(rank - 3)

    @parameter
    if rank >= 2:
        num_rows_input = input.dim(rank - 2)
    let num_cols_input = input.dim(rank - 1)
    let num_cols_output = output.dim(rank - 1)
    let repeats_len = repeats.dim(0)

    # Initializes output by first copying in the original input to the
    # appropriate output elements, and then handles tiling across the column
    # dimension.
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
    for d in range(num_depth_input):
        for r in range(num_rows_input):
            for rep in range(repeats[repeats_len - 1].to_int()):
                let next = repeats[
                    repeats_len - 2
                ].to_int() * num_rows_input * num_cols_output
                let src_ptr = input.data.offset(
                    d * num_rows_input * num_cols_input + r * num_cols_input
                )
                let dst_ptr = output.data.offset(
                    d * next + r * num_cols_output + rep * num_cols_input
                )
                let count = num_cols_input
                memcpy[type](dst_ptr, src_ptr, count)

    # Handles tiling across the rest dimensions (i.e., beyond the innermost).
    # Continuing with the example above, this will handle the 'X's, which
    # correspond to the first '2' in the repeats = [2, 2] tensor.
    # The result is:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6]]
    # Moving from the inner to the outermost dimension, we can memcpy to
    # replicate contiguous memory areas (representing a dimension to be tiled).
    for dim in range(1, rank):
        var inp_dims = 1
        var rep_dims = 1
        # inp_dims, rep_dims used in calculating count (num of elements to copy).
        for i in range(dim + 1):
            inp_dims *= input.dim(rank - i - 1)
        for i in range(dim):
            rep_dims *= repeats[rank - i - 1].to_int()
        # Number of copies (inner_reps) depends on next (higher) dimension.
        # For the last dimension, it is 1 (cannot get from input.dim(), since it
        # does not exist).
        # For the rest, it is input.dim(rank - (dim + 2))
        var inner_reps = 1
        if dim < rank - 1:
            inner_reps = input.dim(rank - dim - 2)
        for d in range(inner_reps):
            for rep in range(repeats[repeats_len - (dim + 1)].to_int() - 1):
                # count depends on dimension copied. e.g., for dim = 1:
                # count = input.dim(1)*input.dim(2)*repeats[2]
                let count = inp_dims * rep_dims
                let src_offset = d * count * repeats[repeats_len - (dim + 1)]
                let dst_offset = src_offset + (rep + 1) * count
                let src_ptr = output.data.offset(src_offset.to_int())
                let dst_ptr = output.data.offset(dst_offset.to_int())
                memcpy[type](dst_ptr, src_ptr, count)

    out_chain.mark_ready()
