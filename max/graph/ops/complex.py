# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Complex ops."""

from ..type import StaticDim
from ..value import TensorValue, TensorValueLike
from .reshape import reshape


def as_interleaved_complex(x: TensorValueLike) -> TensorValue:
    """Reshapes the input symbolic tensor as complex from alternating (real, imag).

    Args:
        interleaved: A symbolic tensor representing complex numbers as
                     alternating pairs of (real, imag) real-valued numbers. Its last
                     dimension must have an even size.

    Returns:
        A symbolic tensor representing the complex-valued tensor, but with the
        values pulled out as complex numbers. The result has the same dimensions
        for all dimensions except the last dimension, which is halved,
        and then a final dimension of size 2 representing the complex value.
    """
    g = TensorValue(x)
    shape = g.shape
    last = shape[-1]
    if not isinstance(last, StaticDim):
        raise TypeError("The last dimension must be static.")
    if last.dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    new_shape = shape[:-1] + [last.dim // 2, 2]
    return reshape(g, new_shape)
