# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external safetensor weights into Max Graph."""

from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.weights import SafetensorWeights


def test_load_safetensors(testdata_directory) -> None:
    weights = SafetensorWeights(testdata_directory / "example_data.safetensors")
    with Graph("test_load_safetensors") as graph:
        data = {
            key: graph.add_weight(weight.allocate())
            for key, weight in weights.items()
        }
        assert len(data) == 5
        assert data["a"].type == TensorType(DType.int32, [5, 2])
        assert data["b"].type == TensorType(DType.float64, [1, 2, 3])
        assert data["c"].type == TensorType(DType.float32, [])
        assert data["fancy/name"].type == TensorType(DType.int64, [3])
        assert data["bf16"].type == TensorType(DType.bfloat16, [2])
