# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from max.dtype import DType
from max.graph import Graph, Weight, ops


def test_add_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        with NamedTemporaryFile() as f:
            weight.tofile(f.name)

            w = Weight(
                "random_weight",
                dtype=DType.int64,
                shape=weight_shape,
                filepath=f.name,
            )

            w2 = Weight(
                "scalar_float",
                dtype=DType.float32,
                shape=[1],
                filepath=f.name,
            )

            graph.output(w, w2)
            gen_mlir = str(graph._mlir_op)
            assert (
                "dense_resource<random_weight> : tensor<5x10xsi64>" in gen_mlir
            )
            assert "dense_resource<scalar_float> : tensor<1xf32>" in gen_mlir


def test_add_same_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        with NamedTemporaryFile() as f:
            w = Weight(
                "w",
                dtype=DType.float32,
                shape=[],
                filepath=f.name,
            )
            value = graph.add_weight(w)

            # Adding the same Weight is fine, and should return the previously
            # created value.
            value2 = graph.add_weight(w)
            assert value is value2

            # Test that adding a different Weight with the same name fails.
            w2 = Weight(
                "w",
                dtype=DType.float32,
                shape=[],
                filepath=f.name,
            )

            with pytest.raises(ValueError, match="already exists"):
                graph.add_weight(w2)


def test_weight_is_value_like() -> None:
    with Graph("graph_with_weights", input_types=()) as graph:
        with NamedTemporaryFile() as f:
            w = Weight(
                "w",
                dtype=DType.float32,
                shape=[],
                filepath=f.name,
            )
            constant = ops.constant(np.array(1, dtype=np.float32))
            graph.output(constant + w)
            gen_mlir = str(graph._mlir_op)
            assert "dense_resource<w> : tensor<f32>" in gen_mlir
