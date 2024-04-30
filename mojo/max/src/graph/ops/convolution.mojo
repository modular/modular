# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops commonly used in convolutional networks."""


from tensor import Tensor, TensorShape
from ..error import error


def conv2d(
    input: Symbol,
    filter: Symbol,
    stride: (Int, Int) = (1, 1),
    dilation: (Int, Int) = (1, 1),
    padding: (Int, Int, Int, Int) = (0, 0, 0, 0),
    groups: Int = 1,
) -> Symbol:
    """Computes the 2-D convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 2-D convolution, with the following layout assumptions:
    - input has NHWC layout, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2-D
    convolution, dim1 here repesents H and dim2 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    This op currently only supports strides and padding on the input.

    Args:
        input: An NHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (height, width, in_channels / num_groups, out_channels).
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
                parallel convolutions. The number of input and output
                channels must both be divisible by the number of groups.

    Returns:
        A symbolic tensor value with the convolution applied.
    """
    var filter_type = filter.tensor_type()
    if filter_type.rank() != 4:
        raise error("filter for a 2-D convolution must be rank 4")

    g = input.graph()
    stride_constant = g.constant(
        Tensor[DType.int64](TensorShape(2), stride[0], stride[1])
    )
    dilation_constant = g.constant(
        Tensor[DType.int64](TensorShape(2), dilation[0], dilation[1])
    )
    padding_constant = g.constant(
        Tensor[DType.int64](
            TensorShape(4), padding[0], padding[1], padding[2], padding[3]
        )
    )

    var output_type = input.tensor_type()
    var input_height = output_type.dims[1]
    var filter_height = filter_type.dims[0]
    if input_height.is_dynamic() or filter_height.is_dynamic():
        output_type.dims[1] = Dim.dynamic()
    elif input_height.is_symbolic():
        raise error("Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[1] = Dim.static(
            input_height.num_elements()
            - filter_height.num_elements()
            + 1
            + padding[0]
            + padding[1]
        )
    var input_width = output_type.dims[2]
    var filter_width = filter_type.dims[1]
    if input_width.is_dynamic() or filter_width.is_dynamic():
        output_type.dims[2] = Dim.dynamic()
    elif input_width.is_symbolic():
        raise error("Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[2] = Dim.static(
            input_width.num_elements()
            - filter_width.num_elements()
            + 1
            + padding[2]
            + padding[3]
        )
    output_type.dims[3] = filter_type.dims[3]

    return g.op(
        "mo.conv",
        List[Symbol](
            input,
            filter,
            stride_constant,
            dilation_constant,
            padding_constant,
            g.scalar[DType.int64](groups),
        ),
        output_type,
    )


def conv3d(
    input: Symbol,
    filter: Symbol,
    stride: (Int, Int, Int) = (1, 1, 1),
    dilation: (Int, Int, Int) = (1, 1, 1),
    padding: (Int, Int, Int, Int, Int, Int) = (0, 0, 0, 0, 0, 0),
    groups: Int = 1,
) -> Symbol:
    """Computes the 3-D convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 3-D convolution, with the following layout assumptions:
    - input has NDHWC layout, i.e.,
      (batch_size, depth, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (depth, height, width, in_channels / num_groups, out_channels)

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2-D
    convolution, dim1 here repesents H and dim2 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    This op currently only supports strides and padding on the input.

    Args:
        input: An NDHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (height, depth, width, in_channels / num_groups, out_channels).
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
                parallel convolutions. The number of input and output
                channels must both be divisible by the number of groups.

    Returns:
        A symbolic tensor value with the convolution applied.
    """
    var filter_type = filter.tensor_type()
    if filter_type.rank() != 5:
        raise error("filter for a 3-D convolution must be rank 5")

    g = input.graph()
    stride_constant = g.constant(
        Tensor[DType.int64](TensorShape(3), stride[0], stride[1], stride[2])
    )
    dilation_constant = g.constant(
        Tensor[DType.int64](
            TensorShape(3), dilation[0], dilation[1], dilation[2]
        )
    )
    padding_constant = g.constant(
        Tensor[DType.int64](
            TensorShape(6),
            padding[0],
            padding[1],
            padding[2],
            padding[3],
            padding[4],
            padding[5],
        )
    )

    var output_type = input.tensor_type()
    var input_depth = output_type.dims[1]
    var filter_depth = filter_type.dims[0]
    if input_depth.is_dynamic() or filter_depth.is_dynamic():
        output_type.dims[1] = Dim.dynamic()
    elif input_depth.is_symbolic():
        raise error("Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[1] = Dim.static(
            input_depth.num_elements()
            - filter_depth.num_elements()
            + 1
            + padding[0]
            + padding[1]
        )
    var input_height = output_type.dims[2]
    var filter_height = filter_type.dims[1]
    if input_height.is_dynamic() or filter_height.is_dynamic():
        output_type.dims[2] = Dim.dynamic()
    elif input_height.is_symbolic():
        raise error("Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[2] = Dim.static(
            input_height.num_elements()
            - filter_height.num_elements()
            + 1
            + padding[2]
            + padding[3]
        )
    var input_width = output_type.dims[3]
    var filter_width = filter_type.dims[2]
    if input_width.is_dynamic() or filter_width.is_dynamic():
        output_type.dims[3] = Dim.dynamic()
    elif input_width.is_symbolic():
        raise error("Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[3] = Dim.static(
            input_width.num_elements()
            - filter_width.num_elements()
            + 1
            + padding[4]
            + padding[5]
        )
    output_type.dims[4] = filter_type.dims[4]

    return g.op(
        "mo.conv",
        List[Symbol](
            input,
            filter,
            stride_constant,
            dilation_constant,
            padding_constant,
            g.scalar[DType.int64](groups),
        ),
        output_type,
    )
