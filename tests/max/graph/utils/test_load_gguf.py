# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external GGUF weights into Max Graph."""

import pytest
from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import GGUFWeights


def test_load_gguf(testdata_directory) -> None:
    weights = GGUFWeights(testdata_directory / "example_data.gguf")
    with Graph("test_load_gguf") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 6
        assert data["a"].type == TensorType(DType.int32, [5, 2])
        assert data["b"].type == TensorType(DType.float64, [1, 2, 3])
        assert data["c"].type == TensorType(DType.float32, [])
        assert data["fancy/name"].type == TensorType(DType.int64, [3])
        assert data["bf16"].type == TensorType(DType.bfloat16, [2])
        assert data["quantized"].type == TensorType(DType.uint8, [2, 144])


def test_load_using_prefix(testdata_directory) -> None:
    weights = GGUFWeights(testdata_directory / "example_data.gguf")
    with Graph("test_load_gguf_by_prefix") as graph:
        a = graph.add_weight(weights.a.allocate())
        assert a.type == TensorType(DType.int32, [5, 2])
        b = graph.add_weight(weights["b"].allocate())
        assert b.type == TensorType(DType.float64, [1, 2, 3])


def test_load_same_weight(testdata_directory) -> None:
    weights = GGUFWeights(testdata_directory / "example_data.gguf")
    with Graph("test_load_gguf_same_weight") as graph:
        graph.add_weight(weights.a.allocate())
        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(weights["a"].allocate())
