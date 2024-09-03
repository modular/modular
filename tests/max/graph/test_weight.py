# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import re

import numpy as np
import pytest
from max.dtype import DType
from max.graph import Graph, Weight, ops


def test_add_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "random_weight",
            dtype=DType.int64,
            shape=[5, 10],
        )

        w2 = Weight(
            "scalar_float",
            dtype=DType.float32,
            shape=[1],
        )

        graph.output(graph.add_weight(w), graph.add_weight(w2))
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*!mo.tensor<\[5, 10\], si64", gen_mlir
        )
        assert re.search(
            r"mo.constant.external.*!mo.tensor<\[1\], f32", gen_mlir
        )


def test_add_same_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "w",
            dtype=DType.float32,
            shape=[],
        )
        value = graph.add_weight(w)

        # Adding the same Weight is fine, and should return the previously
        # created Value.
        value2 = graph.add_weight(w)
        assert value is value2

        # Test that adding a different Weight with the same name fails.
        w2 = Weight(
            "w",
            dtype=DType.float32,
            shape=[],
        )

        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(w2)


def test_weight_is_value_like() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        w = Weight(
            "w",
            dtype=DType.float32,
            shape=[],
        )
        constant = ops.constant(np.array(1, dtype=np.float32))
        graph.output(constant + w)
        gen_mlir = str(graph._mlir_op)
        assert re.search(
            r"mo.constant.external.*!mo.tensor<\[\], f32", gen_mlir
        )
