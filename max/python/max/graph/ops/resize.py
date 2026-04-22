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
"""Op implementation for resize operations."""

from enum import Enum

from max._core.dialects import builtin, kgen, mo, rmo

from ..graph import Graph
from ..type import Shape, ShapeLike, TensorType
from ..value import TensorValue, TensorValueLike
from .shape_to_tensor import shape_to_tensor


class InterpolationMode(Enum):
    """Interpolation modes for image resize operations.

    This enum defines the available interpolation methods that can be used
    when resizing tensors.
    """

    NEAREST = "nearest"
    """Nearest neighbor interpolation."""
    BILINEAR = "bilinear"
    """Bilinear (linear) interpolation."""
    BICUBIC = "bicubic"
    """Bicubic interpolation."""

    def __str__(self) -> str:
        """Return the string representation of the interpolation mode."""
        return self.value


def resize_linear(
    input: TensorValueLike,
    size: ShapeLike,
    coordinate_transform_mode: int = 0,
    antialias: bool = False,
) -> TensorValue:
    """Resize a tensor using linear (bilinear) interpolation.

    Produces an output tensor whose spatial dimensions are given by ``size``
    using separable 1-D linear filters.  The operation maps output coordinates
    back to input coordinates according to ``coordinate_transform_mode``.

    Args:
        input: The input symbolic tensor to resize.
        size: Desired output shape.  Must have the same rank as ``input``.
        coordinate_transform_mode: How to map an output coordinate to an input
            coordinate.  Allowed values:

            - ``0`` -- ``half_pixel`` (default): shifts by 0.5 before scaling,
              consistent with most deep-learning frameworks.
            - ``1`` -- ``align_corners``: aligns the corner pixels of input and
              output so that the first and last coordinates are preserved
              exactly.
            - ``2`` -- ``asymmetric``: no shift; equivalent to floor-dividing
              coordinates by the scale factor.
            - ``3`` -- ``half_pixel_1D``: like ``half_pixel`` but only applied
              to the last spatial dimension.
        antialias: When ``True``, applies an antialiasing filter when the
            output is smaller than the input (i.e. when downscaling), which
            reduces aliasing artifacts by widening the tent filter support by
            ``1 / scale``.  Has no effect when upscaling.

    Returns:
        A new symbolic tensor with shape ``size`` and the same dtype as
        ``input``.

    Raises:
        ValueError: If ``coordinate_transform_mode`` is not 0-3, or if
            ``size`` has a different rank than ``input``.
    """
    if coordinate_transform_mode not in (0, 1, 2, 3):
        raise ValueError(
            f"coordinate_transform_mode must be 0-3, got"
            f" {coordinate_transform_mode}"
        )

    input = TensorValue(input)
    size = Shape(size)

    if len(size) != input.rank:
        raise ValueError(
            f"size must have the same rank as input ({input.rank}), got"
            f" {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeLinearOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        mo.CoordinateTransformModeAttr(
            mo.CoordinateTransformMode(coordinate_transform_mode)
        ),
        builtin.BoolAttr(antialias),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize_nearest(
    input: TensorValueLike,
    size: ShapeLike,
    coordinate_transform_mode: int = 0,
    round_mode: int = 0,
) -> TensorValue:
    """Resize a tensor using nearest-neighbor interpolation.

    Produces an output tensor whose dimensions are given by ``size`` by
    selecting the nearest input sample for each output coordinate.

    Args:
        input: The input symbolic tensor to resize.
        size: Desired output shape.  Must have the same rank as ``input``.
        coordinate_transform_mode: How to map an output coordinate to an input
            coordinate.  Allowed values:

            - ``0`` -- ``half_pixel`` (default).
            - ``1`` -- ``align_corners``.
            - ``2`` -- ``asymmetric``.
            - ``3`` -- ``half_pixel_1D``.
        round_mode: How to round the mapped coordinate to select the nearest
            input sample.  Allowed values:

            - ``0`` -- ``HalfDown`` (default): ``ceil(x - 0.5)``.
            - ``1`` -- ``HalfUp``: ``floor(x + 0.5)``.
            - ``2`` -- ``Floor``: ``floor(x)``.
            - ``3`` -- ``Ceil``: ``ceil(x)``.

    Returns:
        A new symbolic tensor with shape ``size`` and the same dtype as
        ``input``.

    Raises:
        ValueError: If ``coordinate_transform_mode`` is not 0-3,
            ``round_mode`` is not 0-3, or ``size`` has a different rank
            than ``input``.
    """
    if coordinate_transform_mode not in (0, 1, 2, 3):
        raise ValueError(
            f"coordinate_transform_mode must be 0-3, got"
            f" {coordinate_transform_mode}"
        )
    if round_mode not in (0, 1, 2, 3):
        raise ValueError(f"round_mode must be 0-3, got {round_mode}")

    input = TensorValue(input)
    size = Shape(size)

    if len(size) != input.rank:
        raise ValueError(
            f"size must have the same rank as input ({input.rank}), got"
            f" {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeNearestOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        mo.CoordinateTransformModeAttr(
            mo.CoordinateTransformMode(coordinate_transform_mode)
        ),
        builtin.IntegerAttr(builtin.IntegerType(64), round_mode),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize_bicubic(
    input: TensorValueLike,
    size: ShapeLike,
) -> TensorValue:
    """Resize a tensor using bicubic interpolation.

    Produces an output tensor whose dimensions are given by ``size`` using
    a 4x4-pixel Catmull-Rom (a=-0.75) cubic convolution filter with
    half_pixel coordinate mapping.  Input must be rank-4 NCHW.

    Args:
        input: The input symbolic tensor to resize.  Must have rank 4.
        size: Desired output shape of length 4 (N, C, H, W).

    Returns:
        A new symbolic tensor with shape ``size`` and the same dtype as
        ``input``.

    Raises:
        ValueError: If ``input`` doesn't have rank 4 or ``size`` has a
            different length.
    """
    input = TensorValue(input)
    size = Shape(size)

    if input.rank != 4:
        raise ValueError(
            f"resize_bicubic requires rank-4 (NCHW) input, got rank"
            f" {input.rank}"
        )
    if len(size) != 4:
        raise ValueError(
            f"size must have 4 elements for NCHW format, got {len(size)}"
        )

    result_type = TensorType(dtype=input.dtype, shape=size, device=input.device)

    return Graph.current._add_op_generated(
        rmo.MoResizeBicubicOp,
        result_type.to_mlir(),
        input,
        shape_to_tensor(size),
        kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def resize(
    input: TensorValueLike,
    shape: ShapeLike,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> TensorValue:
    """Resize the input tensor to the given shape.

    This function resizes a tensor using the specified interpolation method.
    The tensor is expected to have NCHW format (batch, channels, height, width).

    Args:
        input: The input tensor to resize. Must have rank 4 in NCHW format.
        shape: Desired output shape of length 4 corresponding to (N, C, H, W).
        interpolation: Desired interpolation enum defined by
            :class:`InterpolationMode`.  Defaults to
            :attr:`InterpolationMode.BILINEAR`.

    Returns:
        A resized tensor with the shape specified by the shape argument.

    Raises:
        ValueError: If the input doesn't have rank 4, shape has wrong number
            of elements, or unsupported interpolation mode is specified.
    """
    input = TensorValue(input)
    shape = Shape(shape)

    if input.rank != 4:
        raise ValueError(
            f"Input tensor must have rank 4 (NCHW format) for resize"
            f" operation, but got rank {input.rank}"
        )

    if len(shape) != 4:
        raise ValueError(
            f"shape must have 4 elements for NCHW format"
            f" (batch, channels, height, width), but got {len(shape)} elements"
        )

    if interpolation == InterpolationMode.NEAREST:
        return resize_nearest(input, shape)

    if interpolation == InterpolationMode.BILINEAR:
        return resize_linear(input, shape)

    # BICUBIC (only remaining option after NEAREST and BILINEAR).
    return resize_bicubic(input, shape)
