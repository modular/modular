# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""

import array

from max._core import graph as _graph
from max.dtype import DType
from max.graph import DeviceRef, TensorType


def test_array_attr(mlir_context) -> None:
    """Tests array attribute creation."""
    buffer = array.array("f", [42, 3.14])

    array_attr = _graph.array_attr(
        "foo",
        buffer,
        TensorType(DType.float32, (2,), device=DeviceRef.CPU()).to_mlir(),
    )
    assert "dense_array" in str(array_attr)
