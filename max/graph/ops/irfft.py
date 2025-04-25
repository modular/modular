# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Op implementation for irfft."""

from __future__ import annotations

from enum import Enum
from typing import Any

from max.dtype import DType

from ..type import DeviceKind, Dim, StaticDim, TensorType
from ..value import TensorValue
from .concat import concat
from .constant import constant
from .custom import custom
from .elementwise import sqrt
from .pad import pad
from .transpose import transpose
from .unsqueeze import unsqueeze


class Normalization(Enum):
    BACKWARD = "backward"
    ORTHO = "ortho"
    FORWARD = "forward"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, Normalization):
            return self.value == other.value
        return False


def _process_input_signal(input_tensor: TensorValue, n: int, axis: int):
    """Resizes input tensor to the required signal size."""
    axis = axis % input_tensor.rank
    axis_dim = input_tensor.shape[axis]
    if not isinstance(axis_dim, StaticDim):
        raise ValueError(f"Axis dimension must be static, got {axis_dim}.")
    output_shape = list(input_tensor.shape)
    output_shape[axis] = Dim(n)

    required_axis_dim = n // 2 + 1
    if axis_dim > required_axis_dim:
        # Slice the input tensor to the new size.
        index = [slice(None)] * input_tensor.rank
        index[axis] = slice(0, required_axis_dim)
        input_tensor = input_tensor[tuple(index)]
    elif axis_dim < required_axis_dim:
        # Pad the input tensor with zeros to the new size.
        paddings = [0] * 2 * input_tensor.rank
        paddings[axis * 2 + 1] = required_axis_dim - int(axis_dim)
        input_tensor = pad(input_tensor, paddings=paddings)
    return input_tensor


def irfft(
    input_tensor: TensorValue,
    n: int | None = None,
    axis: int = -1,
    normalization: Normalization | str = Normalization.BACKWARD,
):
    """Compute the inverse real FFT of the input tensor.

    Args:
        input_tensor: The input tensor to compute the inverse real FFT of.
        n: The size of the output tensor. Must be an int, and cannot be a
            symbolic Tensor. The input tensor will be padded or truncated to
            `n // 2 + 1` along the specified axis.
        axis: The axis to compute the inverse real FFT of.
        normalization: The normalization to apply to the output tensor.
            Can be "backward", "ortho", or "forward". When "backward", the
            output is divided by `n`. When "ortho", the output is divided by
            `sqrt(n)`. When "forward", no normalization is applied.

    Returns:
        The inverse real FFT of the input tensor. The shape of the output tensor
        is the same as the shape of the input tensor, except for the axis that
        the inverse real FFT is computed over, which is replaced by `n`.
    """
    if not input_tensor.dtype == DType.float32:
        raise ValueError(
            f"Input tensor must be of type float32, got {input_tensor.dtype}."
        )
    if input_tensor.device.device_type != DeviceKind.GPU:
        raise ValueError("IRFFT is currently only supported on GPU.")

    if not n:
        n = 2 * (int(input_tensor.shape[axis]) - 1)
    input_tensor = _process_input_signal(input_tensor, n, axis)

    # Transpose the input tensor so that the axis that the inverse real FFT is
    # computed over is the last axis.
    orig_axis = axis % input_tensor.rank
    if orig_axis != input_tensor.rank - 1:
        input_tensor = transpose(input_tensor, orig_axis, -1)

    output_shape = list(input_tensor.shape)
    output_shape[-1] = Dim(n)

    # Convert input tensor to a complex tensor.
    # Since we don't support complex tensors, we represent the complex values as
    # interleaved real and imaginary parts.
    x_re = unsqueeze(input_tensor, -1)
    x_im = constant(0, input_tensor.dtype, input_tensor.device).broadcast_to(
        x_re.shape
    )
    x = concat([x_re, x_im], -1).reshape(
        (*input_tensor.shape[:-1], input_tensor.shape[-1] * 2)
    )
    irfft_out = custom(
        "irfft",
        [x],
        [
            TensorType(
                dtype=input_tensor.dtype,
                shape=output_shape,
                device=input_tensor.device,
            )
        ],
        {"n": n},
    )[0].tensor

    if normalization == Normalization.BACKWARD:
        irfft_out /= n
    elif normalization == Normalization.ORTHO:
        irfft_out /= sqrt(constant(n, input_tensor.dtype, input_tensor.device))
    elif normalization == Normalization.FORWARD:
        pass
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    # Transpose the output tensor back to the original axis.
    if orig_axis != input_tensor.rank - 1:
        irfft_out = transpose(irfft_out, -1, orig_axis)
    return irfft_out
