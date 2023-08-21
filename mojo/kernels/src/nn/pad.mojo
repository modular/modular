# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------===#
# pad
# ===----------------------------------------------------------------------===#

from memory.buffer import Buffer, NDBuffer
from utils.index import StaticIntTuple
from utils.list import Dim, DimList
from memory import memcpy
from memory.unsafe import DTypePointer

# TODO Refactor -- we should decide on and put them into a more common file
from Transpose import _fill_strides
from sys.info import sizeof


fn _fill[
    type: DType
](dst: DTypePointer[type], value: SIMD[type, 1], count: Int):
    _ = Buffer[Dim(), type](dst, count).fill(value)


fn pad[
    rank: Int,
    output_shape: DimList,
    input_shape: DimList,
    type: DType,
    paddings_type: DType,
    constant_type: DType,
](
    output: NDBuffer[rank, output_shape, type],
    input: NDBuffer[rank, input_shape, type],
    paddings: DTypePointer[paddings_type],
    constant: SIMD[constant_type, 1],
):
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        input: The input buffer.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.

    Example:
        let input_shape = (X, Y, Z)
        let paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
    """

    let input_strides_buf = Buffer[rank, DType.index].stack_allocation()
    let output_strides_buf = Buffer[rank, DType.index].stack_allocation()
    _fill_strides(input, input_strides_buf)
    _fill_strides(output, output_strides_buf)

    alias init_axis = 0
    _pad_impl(
        init_axis,
        output,
        input.data,
        paddings,
        constant.cast[type](),
        output_strides_buf.data,
        input_strides_buf.data,
        0,  # output_offset
        0,  # input_offset
        False,  # pad_with_constant
    )


fn _pad_impl[
    rank: Int,
    output_shape: DimList,
    type: DType,
    paddings_type: DType,
](
    axis: Int,
    output: NDBuffer[rank, output_shape, type],
    input: DTypePointer[type],
    paddings: DTypePointer[paddings_type],
    constant: SIMD[type, 1],
    output_strides: DTypePointer[DType.index],
    input_strides: DTypePointer[DType.index],
    output_offset: Int,
    input_offset: Int,
    pad_with_constant: Bool,
):
    """
    Fill axis ∈ [axis, rank) in `output` with values from `input`, and edges
    padded with `constant` based on `paddings`.

    Args:
        axis: The axis to operate on.
        output: The output buffer.
        input: The input buffer.
        paddings: The (before, after) padding sizes for each axis.
        constant: the constant to pad output with.
        output_strides: the stride at each output axis.
        input_strides: the stride at each input axis.
        output_offset: The offset at which output data starts.
        input_offset: The offset at which input data starts.
        pad_with_constant: whether to always pad remaining region with constant.
    """

    let axis_dim = output.dim(axis)
    let pre_pad = paddings.load(2 * axis).to_int()
    let post_pad = paddings.load(2 * axis + 1).to_int()
    let non_pad = axis_dim - pre_pad - post_pad

    if axis + 1 == rank:
        # pointers
        let pre_pad_start_ptr = output.data.offset(output_offset)
        let non_pad_start_ptr = pre_pad_start_ptr.offset(pre_pad)
        let post_pad_start_ptr = non_pad_start_ptr.offset(non_pad)
        let input_start_ptr = input.offset(input_offset)

        # setting values
        if pad_with_constant:
            _fill(pre_pad_start_ptr, constant.value, axis_dim)
            return

        _fill(pre_pad_start_ptr, constant.value, pre_pad)
        memcpy(non_pad_start_ptr, input_start_ptr, non_pad)
        _fill(post_pad_start_ptr, constant.value, post_pad)
        return

    debug_assert(axis + 1 < rank, "axis is not within range")

    let input_axis_stride = input_strides.load(axis)[0].value
    let output_axis_stride = output_strides.load(axis)[0].value

    var next_input_offset: Int = input_offset.value
    var next_output_offset: Int = output_offset.value
    for i in range(axis_dim):
        let is_within_padding = (i < pre_pad) or (pre_pad + non_pad <= i)
        let next_pad_with_constant = pad_with_constant or is_within_padding
        _pad_impl(
            axis + 1,
            output,
            input,
            paddings,
            constant,
            output_strides,
            input_strides,
            next_output_offset,
            next_input_offset,
            next_pad_with_constant,
        )
        if not is_within_padding:
            next_input_offset += input_axis_stride
        next_output_offset += output_axis_stride


@always_inline
fn pad_shape[
    input_rank: Int,
    input_type: DType,
    paddings_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    paddings_buf: NDBuffer[2, DimList.create_unknown[2](), paddings_type],
) -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `pad` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Rank of the input tensor.
        input_type: Type of the input tensor.
        paddings_type: Type of the padding tensor.
        single_thread_blocking_override: Whether this function can block.

    Args:
        input_buf: The tensor to pad.
        paddings_buf: The paddings tensor, of shape (input_rank, 2).

    Returns:
        The output shape.
    """

    # TODO(#17512)
    debug_assert(
        paddings_buf.dim(0) == input_rank and paddings_buf.dim(1) == 2,
        "paddings shape must be (input_rank, 2)",
    )

    # compute and return the output shape
    var output_shape = StaticIntTuple[input_rank]()
    for axis in range(input_rank):
        let pre_pad = paddings_buf[axis, 0].to_int()
        let post_pad = paddings_buf[axis, 1].to_int()
        output_shape[axis] = pre_pad + input_buf.dim(axis) + post_pad

    return output_shape
