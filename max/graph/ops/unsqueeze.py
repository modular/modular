# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for squeeze."""

from ..value import TensorValue, ValueLike
from .reshape import reshape


def unsqueeze(x: ValueLike, axis: int) -> TensorValue:
    x = TensorValue(x)
    rank = x.rank
    if axis < 0:
        axis += rank + 1
    if not 0 <= axis <= rank:
        raise ValueError(f"unsqueeze axis out of bounds: {axis=}, {rank=}")

    shape = x.shape
    new_shape = shape[:axis] + [1] + shape[axis:]
    return reshape(x, new_shape)
