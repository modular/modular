# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Op implementation for conv2d."""

from max._core.dialects import builtin, rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import ConvInputLayout, FilterLayout, Shape
from ..value import TensorValue, TensorValueLike
from .elementwise import add
from .validation import assert_same_device

_si64 = builtin.IntegerType(64, builtin.SignednessSemantics.signed)


def conv2d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    bias: TensorValueLike | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.RSCF,
) -> TensorValue:
    """Computes the 2-D convolution product of the input with the given filter, bias, strides, dilations, paddings, and groups.

    This uses the following layout assumptions:

    - The input has channels-last (NHWC) layout, meaning
      ``(batch_size, height, width, in_channels)``.
    - The filter has RSCF layout, meaning
      ``(height, width, in_channels / num_groups, out_channels)``.
    - The bias has shape ``(out_channels,)``.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2-D
    convolution, dim1 here represents H and dim2 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    .. code-block:: text

        input = [
          [1, 2, 3],
          [4, 5, 6]
        ]
        # Shape is 2x3

        padded_input = [
          [0, 0, 1, 2, 3, 0],
          [0, 0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0, 0]
        ]
        # Shape is 3x6

    This op currently only supports strides and padding on the input.

    Convolving a 2x2 input with an all-ones 2x2 filter sums the window:

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("conv2d_example") as graph:
            # NHWC input: batch 1, 2x2 spatial, 1 channel.
            x = ops.constant(
                [[[[1.0], [2.0]], [[3.0], [4.0]]]],
                DType.float32,
                device=device,
            )
            # RSCF filter: 2x2, 1 in-channel, 1 out-channel, all ones.
            filter = ops.constant(
                [[[[1.0]], [[1.0]]], [[[1.0]], [[1.0]]]],
                DType.float32,
                device=device,
            )
            graph.output(ops.conv2d(x, filter))

        model = InferenceSession().load(graph)
        result = model.execute()[0]

    .. invisible-code-block: python

        import numpy as np

        assert np.allclose(result.to_numpy(), [[[[10.0]]]])

    Args:
        x: An NHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout,
            ``(height, width, in_channels / num_groups, out_channels)``.
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
            parallel convolutions. The number of input and output channels
            must both be divisible by the number of groups.
        bias: An optional 1-D bias of shape ``(out_channels,)``.
        input_layout: The layout of the input tensor. Defaults to NHWC.
        filter_layout: The layout of the filter tensor. Defaults to RSCF.

    Returns:
        A ``TensorValue`` representing the result of the convolution, with
        shape ``(batch_size, height_out, width_out, out_channels)``.

    Raises:
        ValueError: If ``x`` isn't rank 4, ``filter`` isn't rank 4, ``bias`` is
            given and isn't rank 1, or ``x`` and ``filter`` aren't on the same
            device.
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape"
                " (out_channels,)"
            )

    if x.rank != 4:
        raise ValueError(
            "input to a 2-D convolution must be rank 4 with shape (batch_size,"
            " height, width, in_channels)"
        )

    if filter.rank != 4:
        raise ValueError(
            "filter for a 2-D convolution must be rank 4 with shape (height,"
            " width, in_channels / num_groups, out_channels)"
        )

    assert_same_device(x=x, filter=filter)

    conv_output = Graph.current._add_op_generated(
        rmo.ConvOp,
        input=x,
        filter=filter._with_layout(filter_layout),
        strides=Shape(stride),
        dilations=Shape(dilation),
        paddings=Shape(padding),
        num_groups=builtin.IntegerAttr(_si64, groups),
        input_layout=input_layout,
    )[0].tensor

    if bias is not None:
        return add(conv_output, bias)
    return conv_output


def conv3d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    groups: int = 1,
    bias: TensorValueLike | None = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.QRSCF,
) -> TensorValue:
    """Computes the 3-D convolution product of the input with the given filter, bias, strides, dilations, paddings, and groups.

    This uses the following layout assumptions:

    - The input has channels-last (NDHWC) layout, meaning
      ``(batch_size, depth, height, width, in_channels)``.
    - The filter has QRSCF layout, meaning
      ``(depth, height, width, in_channels / num_groups, out_channels)``.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 3-D
    convolution, dim1 here represents D, dim2 represents H and dim3 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    .. code-block:: text

        input = [
          [1, 2, 3],
          [4, 5, 6]
        ]
        # Shape is 2x3

        padded_input = [
          [0, 0, 1, 2, 3, 0],
          [0, 0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0, 0]
        ]
        # Shape is 3x6

    This op currently only supports strides and padding on the input.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, ops

        device = DeviceRef.CPU()
        input_type = TensorType(DType.float32, [1, 4, 4, 4, 1], device=device)
        filter_type = TensorType(DType.float32, [2, 2, 2, 1, 1], device=device)
        with Graph(
            "conv3d", input_types=[input_type, filter_type]
        ) as graph:
            x, filter = graph.inputs
            # Output shape is (1, 3, 3, 3, 1).
            graph.output(ops.conv3d(x.tensor, filter.tensor))

    Args:
        x: An NDHWC input tensor to perform the convolution upon.
        filter: The convolution filter in QRSCF layout,
            ``(depth, height, width, in_channels / num_groups, out_channels)``.
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
            parallel convolutions. The number of input and output channels
            must both be divisible by the number of groups.
        bias: An optional 1-D bias of shape ``(out_channels,)``.
        input_layout: The layout of the input tensor. Defaults to NDHWC.
        filter_layout: The layout of the filter tensor. Defaults to QRSCF.

    Returns:
        A ``TensorValue`` representing the result of the convolution, with
        shape ``(batch_size, depth_out, height_out, width_out, out_channels)``.

    Raises:
        ValueError: If ``x`` isn't rank 5, ``filter`` isn't rank 5, or ``bias``
            is given and isn't rank 1.
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape"
                " (out_channels,)"
            )

    if x.rank != 5:
        raise ValueError(
            "input to a 3-D convolution must be rank 5 with shape (batch_size,"
            " depth, height, width, in_channels)"
        )

    if filter.rank != 5:
        raise ValueError(
            "filter for a 3-D convolution must be rank 5 with shape (depth,"
            " height, width, in_channels / num_groups, out_channels)"
        )

    conv_output = Graph.current._add_op_generated(
        rmo.ConvOp,
        input=x,
        filter=filter._with_layout(filter_layout),
        strides=Shape(stride),
        dilations=Shape(dilation),
        paddings=Shape(padding),
        num_groups=builtin.IntegerAttr(_si64, groups),
        input_layout=input_layout,
    )[0].tensor

    if bias is not None:
        return add(conv_output, bias)
    return conv_output
