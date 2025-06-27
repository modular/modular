# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.rebind."""

import pytest
from hypothesis import assume, given
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType


def test_rebind() -> None:
    """Builds a simple graph with a reshape and checks the IR."""
    with Graph(
        "rebind",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.CPU()
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels", "other"],
                device=DeviceRef.CPU(),
            ),
        ],
    ) as graph:
        rebind_to_existing_names = graph.inputs[0].rebind(("batch", "channels"))
        assert rebind_to_existing_names.shape == ["batch", "channels"]

        rebind_to_const = graph.inputs[1].rebind((3, 10))
        assert rebind_to_const.shape == [3, 10]

        rebind_to_new_names = graph.inputs[0].rebind(
            ("notbatch", "notchannels")
        )
        assert rebind_to_new_names.shape == ["notbatch", "notchannels"]

        rebind_expression = (
            graph.inputs[2]
            .reshape(("batch", -1))
            .rebind(("batch", "expression"))
        )
        assert rebind_expression.shape == ["batch", "expression"]

        graph.output(
            rebind_to_existing_names,
            rebind_to_const,
            rebind_to_new_names,
            rebind_expression,
        )


@given(input_type=..., shape=...)
def test_rebind__incorrect_rank(input_type: TensorType, shape: Shape) -> None:
    assume(input_type.rank != shape.rank)
    with Graph("rebind", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].rebind(shape)
