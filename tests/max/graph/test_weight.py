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
from max.graph import Graph


def test_add_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        weight_shape = [5, 10]
        weight = np.random.uniform(1, 100, size=weight_shape).astype(np.int64)
        with NamedTemporaryFile() as f:
            weight.tofile(f.name)

            w = graph.add_weight(
                "random_weight",
                dtype=DType.int64,
                shape=weight_shape,
                filepath=f.name,
            )

            w2 = graph.add_weight(
                "defaults",
                filepath=f.name,
            )

            graph.output(w.value, w2.value)
            gen_mlir = str(graph._mlir_op)
            print(gen_mlir)
            assert (
                "dense_resource<random_weight> : tensor<5x10xsi64>" in gen_mlir
            )
            assert "dense_resource<defaults> : tensor<1xf32>" in gen_mlir


def test_add_same_weight() -> None:
    """Tests adding weights to the graph."""
    with Graph("graph_with_weights", input_types=()) as graph:
        with NamedTemporaryFile() as f:
            graph.add_weight(
                "w",
                filepath=f.name,
            )
            with pytest.raises(ValueError, match="already exists"):
                _ = graph.add_weight(
                    "w",
                    filepath=f.name,
                )
