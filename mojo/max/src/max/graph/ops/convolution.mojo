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
"""Ops commonly used in convolutional networks."""

from _mlir import Identifier, NamedAttribute
from _mlir.builtin_attributes import BoolAttr
from max.tensor import Tensor, TensorShape

from ..error import error


def _padded_dimensions(
    g: Graph,
    input: TensorType,
    filter_shape: (Dim, Dim),
    padding: (Int, Int, Int, Int),
    stride: (Int, Int),
) -> TensorType:
    """Computes the final dimensions of a rank-4 tensor after a 2-D convolution.

    When a 2-D filter is applied to a tensor, the final height and width
    dimensions are inset by the size of the filter, minus one, and then grown
    by the padding applied before and after each dimension.

    Args:
        g: The Graph instance.
        input: The shape of the rank-4 input tensor.
        filter_shape: The shape of the filter being applied.
        padding: The padding to apply to the height and width of the input
                 tensor. Values are provided in (before_height, after_height,
                 before_width, after_width) ordering.
        stride: The stride of the filter, in (height, width) ordering.

    Returns:
        The resized shape of the tensor.
    """
    output_type = input
    input_height = input.dims[1]
    filter_height = filter_shape[0]
    if input_height.is_dynamic() or filter_height.is_dynamic():
        output_type.dims[1] = Dim.dynamic()
    elif input_height.is_symbolic():
        raise error(
            g, "padding an input with named dimensions is not yet supported"
        )
    else:
        output_type.dims[1] = Dim.static(
            (
                input_height.num_elements()
                - filter_height.num_elements()
                + padding[0]
                + padding[1]
                + stride[0]
            )
            / stride[0]
        )
    input_width = input.dims[2]
    filter_width = filter_shape[1]
    if input_width.is_dynamic() or filter_width.is_dynamic():
        output_type.dims[2] = Dim.dynamic()
    elif input_width.is_symbolic():
        raise error(
            g, "padding an input with named dimensions is not yet supported"
        )
    else:
        output_type.dims[2] = Dim.static(
            (
                input_width.num_elements()
                - filter_width.num_elements()
                + padding[2]
                + padding[3]
                + stride[1]
            )
            / stride[1]
        )
    return output_type


def avg_pool(
    input: Symbol,
    filter_shape: (Int, Int),
    stride: (Int, Int) = (1, 1),
    dilation: (Int, Int) = (1, 1),
    padding: (Int, Int, Int, Int) = (0, 0, 0, 0),
    count_boundary: Bool = True,
) -> Symbol:
    """Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Args:
        input: The rank-4 input tensor to perform the pooling upon, in NHWC layout.
        filter_shape: The shape of the pooling filter.
        stride: The stride of the pooling operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        count_boundary: Whether to include the zero-padding in the averaging calculation.

    Returns:
        A symbolic tensor value with the pooling applied.
    """
    g = input.graph()
    filter_constant = g.constant(
        Tensor[DType.int64](TensorShape(2), filter_shape[0], filter_shape[1])
    )
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

    output_type = _padded_dimensions(
        g,
        input.tensor_type(),
        filter_shape=(Dim.static(filter_shape[0]), Dim.static(filter_shape[1])),
        padding=padding,
        stride=stride,
    )

    attributes = List[NamedAttribute]()
    attributes.append(
        NamedAttribute(
            name=Identifier(g._context(), "count_boundary"),
            attr=BoolAttr(g._context(), count_boundary),
        )
    )

    return g.op(
        "rmo.mo.avg_pool",
        List[Symbol](
            input,
            filter_constant,
            stride_constant,
            dilation_constant,
            padding_constant,
        ),
        output_type,
        attributes,
    )


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
    g = input.graph()
    filter_type = filter.tensor_type()
    if filter_type.rank() != 4:
        raise error(g, "filter for a 2-D convolution must be rank 4")

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

    output_type = _padded_dimensions(
        g,
        input.tensor_type(),
        filter_shape=(filter_type.dims[0], filter_type.dims[1]),
        padding=padding,
        stride=stride,
    )
    output_type.dims[3] = filter_type.dims[3]

    return g.op(
        "rmo.mo.conv",
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
    g = input.graph()
    filter_type = filter.tensor_type()
    if filter_type.rank() != 5:
        raise error(g, "filter for a 3-D convolution must be rank 5")

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

    output_type = input.tensor_type()
    input_depth = output_type.dims[1]
    filter_depth = filter_type.dims[0]
    if input_depth.is_dynamic() or filter_depth.is_dynamic():
        output_type.dims[1] = Dim.dynamic()
    elif input_depth.is_symbolic():
        raise error(g, "Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[1] = Dim.static(
            input_depth.num_elements()
            - filter_depth.num_elements()
            + 1
            + padding[0]
            + padding[1]
        )
    input_height = output_type.dims[2]
    filter_height = filter_type.dims[1]
    if input_height.is_dynamic() or filter_height.is_dynamic():
        output_type.dims[2] = Dim.dynamic()
    elif input_height.is_symbolic():
        raise error(g, "Convolution doesn't yet support symbolic dimensions")
    else:
        output_type.dims[2] = Dim.static(
            input_height.num_elements()
            - filter_height.num_elements()
            + 1
            + padding[2]
            + padding[3]
        )
    input_width = output_type.dims[3]
    filter_width = filter_type.dims[2]
    if input_width.is_dynamic() or filter_width.is_dynamic():
        output_type.dims[3] = Dim.dynamic()
    elif input_width.is_symbolic():
        raise error(g, "Convolution doesn't yet support symbolic dimensions")
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
        "rmo.mo.conv",
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


def max_pool(
    input: Symbol,
    filter_shape: (Int, Int),
    stride: (Int, Int) = (1, 1),
    dilation: (Int, Int) = (1, 1),
    padding: (Int, Int, Int, Int) = (0, 0, 0, 0),
) -> Symbol:
    """Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here repesents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Args:
        input: The input tensor to perform the pooling upon.
        filter_shape: The shape of the pooling filter.
        stride: The stride of the pooling operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.

    Returns:
        A symbolic tensor value with the pooling applied.
    """
    g = input.graph()
    filter_constant = g.constant(
        Tensor[DType.int64](TensorShape(2), filter_shape[0], filter_shape[1])
    )
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

    output_type = _padded_dimensions(
        g,
        input.tensor_type(),
        filter_shape=(Dim.static(filter_shape[0]), Dim.static(filter_shape[1])),
        padding=padding,
        stride=stride,
    )

    return g.op(
        "rmo.mo.max_pool",
        List[Symbol](
            input,
            filter_constant,
            stride_constant,
            dilation_constant,
            padding_constant,
        ),
        output_type,
    )
