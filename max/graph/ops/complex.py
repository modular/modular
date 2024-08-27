# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Complex ops."""

from ..value import TensorValue, ValueLike
from ..type import StaticDim
from .reshape import reshape


def as_interleaved_complex(x: ValueLike) -> TensorValue:
    g = TensorValue(x)
    shape = g.shape
    last = shape[-1]
    if not isinstance(last, StaticDim):
        raise TypeError("The last dimension must be static.")
    if last.dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    new_shape = shape[:-1] + [last.dim // 2, 2]
    return reshape(g, new_shape)
