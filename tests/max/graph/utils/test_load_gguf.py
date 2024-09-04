# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external GGUF weights into Max Graph."""

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
