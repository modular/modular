# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external PyTorch weights into Max Graph."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.graph.weights import PytorchWeights


def test_load_pytorch(testdata_directory) -> None:
    # Loads the values saved in gen_external_checkpoints.py.
    weights = PytorchWeights(testdata_directory / "example_data.pt")
    with Graph("test_load_pytorch") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 5
        assert data["a"].type == TensorType(
            DType.int32, [5, 2], device=DeviceRef.CPU()
        )
        assert data["b"].type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )
        assert data["c"].type == TensorType(
            DType.float32, [], device=DeviceRef.CPU()
        )
        assert data["fancy/name"].type == TensorType(
            DType.int64, [3], device=DeviceRef.CPU()
        )
        assert data["bf16"].type == TensorType(
            DType.bfloat16, [2], device=DeviceRef.CPU()
        )


def test_load_using_prefix(testdata_directory) -> None:
    weights = PytorchWeights(testdata_directory / "example_data.pt")
    with Graph("test_load_pytorch_by_prefix") as graph:
        a = graph.add_weight(weights.a.allocate())
        assert a.type == TensorType(DType.int32, [5, 2], device=DeviceRef.CPU())
        b = graph.add_weight(weights["b"].allocate())
        assert b.type == TensorType(
            DType.float64, [1, 2, 3], device=DeviceRef.CPU()
        )


def test_load_same_weight(testdata_directory) -> None:
    weights = PytorchWeights(testdata_directory / "example_data.pt")
    with Graph("test_load_pytorch_same_weight") as graph:
        graph.add_weight(weights.a.allocate())
        with pytest.raises(ValueError, match="already exists"):
            graph.add_weight(weights["a"].allocate())
