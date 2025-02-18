# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Custom ops tests."""

from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def test_inplace_custom() -> None:
    """Tests that returning the result of an `inplace_custom` op works."""
    M = 42
    N = 37
    graph = Graph(
        "foo",
        forward=lambda x: ops.inplace_custom("foo", values=[x]),
        input_types=[
            BufferType(
                dtype=DType.float32, shape=(M, N), device=DeviceRef.GPU()
            )
        ],
    )


def test_custom() -> None:
    """Tests that returning the result of an `custom` op works."""
    M = 42
    N = 37
    tensor_type = TensorType(
        dtype=DType.float32, shape=(M, N), device=DeviceRef.GPU()
    )
    graph = Graph(
        "foo",
        forward=lambda x: ops.custom(
            "foo", values=[x], out_types=[tensor_type]
        ),
        input_types=[tensor_type],
    )
