# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for squeeze."""

from ..graph_value import GraphValue, ValueLike
from ..type import Shape

from .reshape import reshape


def squeeze(x: ValueLike, axis: int) -> GraphValue:
    v = GraphValue(x)
    # TODO (MSDK-655): Probably want to add rmo.mo_squeeze_shape here
    shape = Shape(v.shape)
    if shape[axis] != 1:
        raise ValueError(f"Squeeze dim must be 1, got {axis=}, {shape=}")
    shape.pop(axis)
    return reshape(v, shape)
