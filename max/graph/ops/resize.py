# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for resize operations."""

from enum import Enum

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Shape, ShapeLike, TensorType
from ..value import TensorValue, TensorValueLike
from .shape_to_tensor import shape_to_tensor


class InterpolationMode(Enum):
    """Interpolation modes for image resize operations.

    This enum defines the available interpolation methods that can be used
    when resizing tensors. Currently only BICUBIC is implemented, with
    BILINEAR and NEAREST planned for future support.
    """

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"

    def __str__(self) -> str:
        """Return the string representation of the interpolation mode."""
        return self.value


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
        interpolation: Desired interpolation enum defined by InterpolationMode.
            Default is InterpolationMode.BILINEAR. Currently only BICUBIC is
            supported.

    Returns:
        A resized tensor with the shape specified by the shape argument.

    Raises:
        ValueError: If the input doesn't have rank 4, shape has wrong number
            of elements, or unsupported interpolation mode is specified.
        NotImplementedError: If single integer size or non-BICUBIC interpolation
            mode is specified.
    """
    input = TensorValue(input)
    shape = Shape(shape)

    if input.rank != 4:
        raise ValueError(
            f"Input tensor must have rank 4 (NCHW format) for resize operation, "
            f"but got rank {input.rank}"
        )

    if len(shape) != 4:
        raise ValueError(
            f"shape must have 4 elements for NCHW format (batch, channels, height, width), "
            f"but got {len(shape)} elements"
        )

    if interpolation != InterpolationMode.BICUBIC:
        raise NotImplementedError(
            f"Interpolation mode {interpolation} is not yet supported. "
            "Currently only InterpolationMode.BICUBIC is available."
        )

    # NOTE: half_pixel is the default coordinate transform mode.
    # This matches the behavior of torchvision and other libraries.

    # Create the result type with the new shape.
    result_type = TensorType(
        dtype=input.dtype, shape=shape, device=input.device
    )

    # Stage bicubic resize op.
    return Graph.current._add_op(
        rmo.mo_resize_bicubic,
        result_type.to_mlir(),
        input,
        shape_to_tensor(shape),
    )[0].tensor
