# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external safetensor weights into Max Graph."""

import pytest
from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights


def test_load_safetensors_one(testdata_directory) -> None:
    weights = SafetensorWeights(
        [testdata_directory / "example_data_1.safetensors"]
    )
    with Graph("test_load_safetensors1") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 5
        assert data["1.a"].type == TensorType(DType.int32, [5, 2])
        assert data["1.b"].type == TensorType(DType.float64, [1, 2, 3])
        assert data["1.c"].type == TensorType(DType.float32, [])
        assert data["1.fancy/name"].type == TensorType(DType.int64, [3])
        assert data["1.bf16"].type == TensorType(DType.bfloat16, [2])


def test_load_safetensors_multi(testdata_directory) -> None:
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_multi") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 10
        assert data["1.a"].type == TensorType(DType.int32, [5, 2])
        assert data["1.b"].type == TensorType(DType.float64, [1, 2, 3])
        assert data["1.c"].type == TensorType(DType.float32, [])
        assert data["1.fancy/name"].type == TensorType(DType.int64, [3])
        assert data["1.bf16"].type == TensorType(DType.bfloat16, [2])
        assert data["2.a"].type == TensorType(DType.int32, [5, 2])
        assert data["2.b"].type == TensorType(DType.float64, [1, 2, 3])
        assert data["2.c"].type == TensorType(DType.float32, [])
        assert data["2.fancy/name"].type == TensorType(DType.int64, [3])
        assert data["2.bf16"].type == TensorType(DType.bfloat16, [2])


def test_load_using_prefix(testdata_directory) -> None:
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_by_prefix") as graph:
        a = graph.add_weight(weights[1].a.allocate())
        assert a.type == TensorType(DType.int32, [5, 2])
        b = graph.add_weight(weights["1.b"].allocate())
        assert b.type == TensorType(DType.float64, [1, 2, 3])


def test_load_same_weight(testdata_directory) -> None:
    weights = SafetensorWeights(
        [
            testdata_directory / "example_data_1.safetensors",
            testdata_directory / "example_data_2.safetensors",
        ]
    )
    with Graph("test_load_safetensors_same_weight") as graph:
        a = graph.add_weight(weights[1].a.allocate())
        with pytest.raises(ValueError, match="already exists"):
            a2 = graph.add_weight(weights[1]["a"].allocate())
