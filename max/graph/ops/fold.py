# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for fold."""

from __future__ import annotations

from ..type import DimLike, Shape, StaticDim
from ..value import TensorType, TensorValue, TensorValueLike
from .custom import custom
from .shape_to_tensor import shape_to_tensor


def fold(
    input: TensorValueLike,
    output_size: tuple[DimLike, DimLike],
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> TensorValue:
    """Combines an array of sliding blocks into a larger containing tensor.

    The input tensor must have shape ``(N, C * kernel_sizes, L)`` where ``N`` is
    the batch dimension, ``C`` is the number of channels, ``kernel_sizes`` is
    the product of the kernel sizes, and ``L`` is the number of local blocks.

    The resulting output tensor will have shape
    ``(N, C, output_shape[0], output_shape[1])``.

    ``L``, the number of blocks, must be equivalent to:
    ``prod((output_size[d] + 2 * padding[d] - dilation[d] * (kernel_size[d] - 1) - 1) / stride[d] + 1)``

    where ``d`` is over all spatial dimensions.

    Args:
        input: The 3D tensor to fold with shape ``(N, C * kernel sizes, L)``.
        output_size: Spacial dimensions of the output tensor. Must be a tuple of two ints.
        kernel_size: The size of the sliding blocks. Must be a tuple of two ints.
        stride: The stride of the sliding blocks in the input dimension
            (can be an int or a tuple of two ints).
        dilation: The spacing between the kernel elements.
            (can be an int or a tuple of two ints).
        padding: 0-paddings to be added on both sides of the inputs.
            (can be an int or a tuple of two ints).

    Returns:
        The folded 4D tensor with shape ``(N, C, output_shape[0], output_shape[1])``.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        padding = (padding, padding)

    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        channels = input.shape[1] // (kernel_size[0] * kernel_size[1])
        output_shape = Shape(
            [input.shape[0], channels, output_size[0], output_size[1]]
        )
    else:
        output_shape = Shape(
            [input.shape[0], "channels", output_size[0], output_size[1]]
        )

    # Run early shape checks if the shapes are statically known.
    if isinstance(kernel_size[0], int) and isinstance(kernel_size[1], int):
        if (
            isinstance(input.shape[1], StaticDim)
            and int(input.shape[1]) % (kernel_size[0] * kernel_size[1]) != 0
        ):
            raise ValueError(
                f"Dim 1 of the input tensor ({input.shape[1]}) must be a multiple "
                "of the product of the total kernel size"
                f" ({kernel_size[0]} * {kernel_size[1]})."
            )

        if (
            isinstance(input.shape[2], StaticDim)
            and isinstance(output_size[0], int)
            and isinstance(output_size[1], int)
        ):
            L = 1
            for n, (o, k) in enumerate(zip(output_size, kernel_size)):
                L_d = int(
                    (int(o) + 2 * padding[n] - dilation[n] * (int(k) - 1) - 1)
                    // stride[n]
                    + 1
                )
                L *= L_d
            if int(input.shape[2]) != L:
                raise ValueError(
                    f"Last dimension of input tensor ({input.shape[2]}) must match "
                    f"the calculated number of blocks ({L})."
                )

    parameters: dict[str, int] = {
        "stride_h": stride[0],
        "stride_w": stride[1],
        "dilation_h": dilation[0],
        "dilation_w": dilation[1],
        "padding_h": padding[0],
        "padding_w": padding[1],
    }

    return custom(
        "fold",
        input.device,
        [
            input,
            shape_to_tensor(output_size),
            shape_to_tensor(kernel_size),
        ],
        [TensorType(input.dtype, output_shape, input.device)],
        parameters=parameters,
    )[0].tensor
