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
"""Op implementation for pooling (max, avg, etc)."""

from __future__ import annotations

from max.mlir.dialects import rmo

from ..dim import DimLike
from ..graph import Graph
from ..shape import Shape
from ..value import TensorValue, TensorValueLike


def avg_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_boundary: bool = True,
) -> TensorValue:
    """Perform a 2D average pooling operation on the input tensor.

    Applies a 2D average pooling operation to the input tensor with layout
    ``[N, H, W, C]``. The pooling operation slides a window of size
    ``kernel_size`` over the spatial dimensions and computes the average
    value within each window.

    Args:
        input: The input tensor with shape ``[N, H, W, C]``.
        kernel_size: The height and width of the sliding window.
        stride: The stride of the sliding window. Can be a single integer
            applied to both spatial dimensions or a tuple ``(stride_h,
            stride_w)``. Defaults to 1.
        dilation: The spacing between kernel elements. Can be a single
            integer or a tuple ``(dilation_h, dilation_w)``. Defaults to 1.
        padding: Zero-padding added to both sides of each spatial dimension.
            Can be a single integer or a tuple ``(pad_h, pad_w)``.
            Defaults to 0.
        ceil_mode: If ``True``, uses ceil instead of floor when computing
            the output spatial shape. Defaults to ``False``.
        count_boundary: If ``True``, includes padding elements in the
            divisor when computing the average. Defaults to ``True``.

    Returns:
        A symbolic tensor with the average pooling applied, with shape
        ``[N, H_out, W_out, C]``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op(
        rmo.avg_pool,
        input=input,
        filter_shape=Shape(kernel_size).to_mlir(),
        strides=Shape(stride).to_mlir(),
        dilations=Shape(dilation).to_mlir(),
        paddings=Shape(_padding).to_mlir(),
        ceil_mode=ceil_mode,
        count_boundary=count_boundary,
    )[0].tensor


def max_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
) -> TensorValue:
    """Perform a 2D max pooling operation on the input tensor.

    Applies a 2D max pooling operation to the input tensor with layout
    ``[N, H, W, C]``. The pooling operation slides a window of size
    ``kernel_size`` over the spatial dimensions and selects the maximum
    value within each window.

    Args:
        input: The input tensor with shape ``[N, H, W, C]``.
        kernel_size: The height and width of the sliding window.
        stride: The stride of the sliding window. Can be a single integer
            applied to both spatial dimensions or a tuple ``(stride_h,
            stride_w)``. Defaults to 1.
        dilation: The spacing between kernel elements. Can be a single
            integer or a tuple ``(dilation_h, dilation_w)``. Defaults to 1.
        padding: Zero-padding added to both sides of each spatial dimension.
            Can be a single integer or a tuple ``(pad_h, pad_w)``.
            Defaults to 0.
        ceil_mode: If ``True``, uses ceil instead of floor when computing
            the output spatial shape. Defaults to ``False``.

    Returns:
        A symbolic tensor with the max pooling applied, with shape
        ``[N, H_out, W_out, C]``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op(
        rmo.max_pool,
        input=input,
        filter_shape=Shape(kernel_size).to_mlir(),
        strides=Shape(stride).to_mlir(),
        dilations=Shape(dilation).to_mlir(),
        paddings=Shape(_padding).to_mlir(),
        ceil_mode=ceil_mode,
    )[0].tensor
