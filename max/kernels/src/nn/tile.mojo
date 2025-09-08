# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from layout import LayoutTensor
from memory import memcpy

from utils import IndexList

# TODO: This implementation supports up to 4 dimensions.

# Note: ONNX spec specifies that `tile` behaves like Numpy's tile, but without
#       broadcast. This means that `repeats` is a 1D int64 tensor of the SAME
#       length as input's rank (unlike Numpy that allows `repeats` to have more
#       or less elements than the input's rank).


@always_inline
fn tile[
    dtype: DType, type_repeats: DType
](
    input: LayoutTensor[dtype, address_space = AddressSpace.GENERIC, **_],
    repeats: LayoutTensor[
        type_repeats, address_space = AddressSpace.GENERIC, **_
    ],
    output: LayoutTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, **_
    ],
) raises:
    """
    Implements the `Tile` operator from the ONNX spec. This behaves like Numpy
    tile, but without broadcast.

    Parameters:
        dtype: Type of the input and output tensors.
        type_repeats: Type of the repeats tensor.

    Args:
        input: The input tensor. Currently <= 4 dimensions are supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        output: The output tensor. Has the same dimensions and type as input.
    """
    constrained[
        input.rank == output.rank, "input and output must have the same rank"
    ]()

    if input.rank > 4:
        raise Error(
            "Currently only inputs of up to three dimensions are supported."
        )

    if repeats.rank != 1 or type_repeats != DType.int64:
        raise Error(
            "Rank of repeats tensor needs to be one-dimensional and of int64"
            " type."
        )

    if input.rank != Int(repeats.runtime_layout.shape[0]):
        raise Error(
            "Length of repeats tensor should be equal to the rank of the input"
            " tensor."
        )

    var num_dp_input = 1
    var num_depth_input = 1
    var num_rows_input = 1

    @parameter
    if input.rank == 4:
        num_dp_input = Int(input.runtime_layout.shape[input.rank - 4])

    @parameter
    if input.rank >= 3:
        num_depth_input = Int(input.runtime_layout.shape[input.rank - 3])

    @parameter
    if input.rank >= 2:
        num_rows_input = Int(input.runtime_layout.shape[input.rank - 2])
    var num_cols_input = Int(input.runtime_layout.shape[input.rank - 1])

    var repeats_len = Int(repeats.runtime_layout.shape[0])

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
                var input_src_index = (
                    dp * num_depth_input * num_rows_input * num_cols_input
                    + d * num_rows_input * num_cols_input
                    + r * num_cols_input
                )
                var output_src_index = (
                    dp
                    * num_depth_input
                    * Int(repeats[repeats_len - 3])
                    * num_rows_input
                    * Int(repeats[repeats_len - 2])
                    * num_cols_input
                    * Int(repeats[repeats_len - 1])
                    + d
                    * num_rows_input
                    * Int(repeats[repeats_len - 2])
                    * num_cols_input
                    * Int(repeats[repeats_len - 1])
                    + r * num_cols_input * Int(repeats[repeats_len - 1])
                )
                var output_src_stride = num_cols_input
                var count = output_src_stride
                for rep in range(repeats[repeats_len - 1]):
                    var src_ptr = input.ptr.offset(input_src_index)
                    var dst_ptr = output.ptr.offset(
                        output_src_index + rep * output_src_stride
                    )
                    memcpy(dst_ptr, src_ptr, count)

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
    if input.rank >= 2:
        var src_index_stride = (
            num_rows_input * num_cols_input * Int(repeats[repeats_len - 1])
        )
        var count = src_index_stride
        for dp in range(num_dp_input):
            for d in range(num_depth_input):
                var src_index = dp * num_depth_input * Int(
                    repeats[repeats_len - 3]
                ) * num_rows_input * Int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * Int(
                    repeats[repeats_len - 1]
                ) + d * num_rows_input * Int(
                    repeats[repeats_len - 2]
                ) * num_cols_input * Int(
                    repeats[repeats_len - 1]
                )
                for rep in range(repeats[repeats_len - 2] - 1):
                    var src_ptr = output.ptr.offset(src_index)
                    var dst_ptr = output.ptr.offset(
                        src_index + (rep + 1) * src_index_stride
                    )
                    memcpy(dst_ptr, src_ptr, count)

    # Handles tiling across the third dimension from the end (if tensor rank >= 3)
    @parameter
    if input.rank >= 3:
        var src_index_stride = (
            num_depth_input
            * Int(repeats[repeats_len - 2])
            * num_rows_input
            * num_cols_input
            * Int(repeats[repeats_len - 1])
        )
        var count = src_index_stride
        for dp in range(num_dp_input):
            var src_index = (
                dp
                * num_depth_input
                * Int(repeats[repeats_len - 3])
                * num_rows_input
                * Int(repeats[repeats_len - 2])
                * num_cols_input
                * Int(repeats[repeats_len - 1])
            )
            for rep in range(repeats[repeats_len - 3] - 1):
                var src_ptr = output.ptr.offset(src_index)
                var dst_ptr = output.ptr.offset(
                    src_index + (rep + 1) * src_index_stride
                )
                memcpy(dst_ptr, src_ptr, count)

    # Handles tiling across the fourth dimension from the end(if tensor rank >= 3)
    @parameter
    if input.rank == 4:
        var src_index_stride = (
            num_dp_input
            * Int(repeats[Int(repeats.runtime_layout.shape[0]) - 3])
            * num_depth_input
            * Int(repeats[Int(repeats.runtime_layout.shape[0]) - 2])
            * num_rows_input
            * num_cols_input
            * Int(repeats[Int(repeats.runtime_layout.shape[0]) - 1])
        )
        var count = src_index_stride
        var src_index = 0
        for rep in range(
            Int(repeats[Int(repeats.runtime_layout.shape[0]) - 4] - 1)
        ):
            var src_ptr = output.ptr.offset(src_index)
            var dst_ptr = output.ptr.offset(
                src_index + (rep + 1) * src_index_stride
            )
            memcpy(dst_ptr, src_ptr, count)


@always_inline
fn tile_shape[
    input_type: DType,
    repeats_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: LayoutTensor[input_type, **_],
    repeats_buf: LayoutTensor[repeats_type, **_],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `tile` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        repeats_type: Type of the repeats tensor.
        single_thread_blocking_override: If True, then the operation is run
          synchronously using a single thread.

    Args:
        input_buf: The input tensor.
        repeats_buf: The repeats tensor.

    Returns:
        The output shape.
    """
    constrained[repeats_buf.rank == 1, "repeats_buf must be of rank 1"]()

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::TileOp` prevents testing this with static rank.
    if Int(repeats_buf.runtime_layout.shape[0]) != input_buf.rank:
        raise Error("[tile] requires (len(repeats) == input_rank)")

    # Compute and return the output shape.
    var output_shape = IndexList[input_buf.rank]()

    @parameter
    for i in range(input_buf.rank):
        output_shape[i] = Int(input_buf.runtime_layout.shape[i]) * Int(
            repeats_buf[i]
        )

    return output_shape
