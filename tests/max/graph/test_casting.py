# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys

import pytest
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops
from max.graph.type import shape


def test_reshape() -> None:
    """Builds a simple graph with a reshape and checks the IR."""
    with Graph(
        "simple_reshape",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5]),
            TensorType(dtype=DType.float32, shape=["batch", "channels"]),
        ],
    ) as graph:
        # TODO(MSDK-662): Add shape checks to these test cases once we can get the shape of a value.

        static_reshape = graph.inputs[0].reshape((3, 10))
        static_reshape_neg_one = graph.inputs[0].reshape((2, -1))
        assert static_reshape_neg_one.shape == shape((2, 15))

        symbolic_reshape = graph.inputs[1].reshape(("channels", "batch"))
        symbolic_reshape_neg_one = graph.inputs[1].reshape(("channels", -1))
        assert symbolic_reshape_neg_one.shape == shape(("channels", "batch"))

        graph.output(
            static_reshape,
            static_reshape_neg_one,
            symbolic_reshape,
            symbolic_reshape_neg_one,
        )

        graph._mlir_op.verify()
