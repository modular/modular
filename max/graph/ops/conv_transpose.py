# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for conv2d."""

from typing import Optional

from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph, TensorType
from ..type import Shape
from ..value import TensorValue, TensorValueLike


def conv2d_transpose(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    output_paddings: tuple[int, int] = (0, 0),
    bias: Optional[TensorValueLike] = None,
) -> TensorValue:
    """Computes the 2-D deconvolution of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports the transpose (gradient) of convolution, with the following layout assumptions:
    (note the `out_channel` is w.r.t. the original convolution)

    - input `x` has NHWC layout, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (kernel_height, kernel_width, out_channels, in_channels)
    - bias has shape (out_channels,)

    The padding values are expected to take the form in the form [[0, 0], [pad_top, pad_bottom],
    [pad_left, pad_right], [0, 0]].

    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.


    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here repesents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    Args:
        input: An NHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (height, width, out_channels, in_channels).
        stride: The stride of the sliding window for each dimension of input.
            If a single value is given it is replicated in the H and W dimension.
            By default the N and C dimensions are set to 0.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        output_paddings: this argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.
        bias: tensor of shape (out_channels,)

    Returns:
        A symbolic tensor value with the convolution applied.
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D deconvolution must be rank 1 with shape (out_channels,)"
            )

    if x.rank != 4:
        raise ValueError(
            "input to a 2-D deconvolution must be rank 4 with shape (batch_size,"
            " height, width, in_channels)"
        )

    if filter.rank != 4:
        raise ValueError(
            "filter for a 2-D deconvolution must be rank 4 with shape (height,"
            " width, out_channels, in_channels)"
        )
    if output_paddings[0] >= stride[0] or output_paddings[1] >= stride[1]:
        raise ValueError(
            f"output padding must be smaller than either stride or dilation, but got output_padding = {output_paddings}"
        )

    output_height = (
        stride[0] * (x.shape[1] - 1)
        + output_paddings[0]
        + ((filter.shape[0] - 1) * dilation[0] + 1)
        - padding[0]
        - padding[1]
    )
    output_width = (
        stride[1] * (x.shape[2] - 1)
        + output_paddings[1]
        + ((filter.shape[1] - 1) * dilation[1] + 1)
        - padding[2]
        - padding[3]
    )
    output_shape = [x.shape[0], output_height, output_width, filter.shape[2]]

    out = Graph.current._add_op(
        rmo.mo_conv_transpose,
        result=TensorType(x.dtype, output_shape, x.device).to_mlir(),
        input=x,
        filter=filter,
        strides=TensorValue(Shape(stride)),
        dilations=TensorValue(Shape(dilation)),
        paddings=TensorValue(Shape(padding)),
        output_paddings=TensorValue(Shape(output_paddings)),
    )[0].tensor

    if bias is not None:
        return Graph.current._add_op(rmo.add, out, bias)[0].tensor
    return out
