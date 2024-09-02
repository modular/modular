# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test loading external GGUF weights into Max Graph."""

from max.dtype import DType
from max.graph import Graph, TensorType, Value
from max.graph.utils import load_gguf


def test_load_gguf(testdata_directory) -> None:
    weights = load_gguf(testdata_directory / "example_data.gguf")
    with Graph("test_load_gguf") as graph:
        values = {key: Value(weight) for key, weight in weights.items()}
        assert len(values) == 6
        assert values["a"].type == TensorType(DType.int32, [5, 2])
        assert values["b"].type == TensorType(DType.float64, [1, 2, 3])
        assert values["c"].type == TensorType(DType.float32, [])
        assert values["fancy/name"].type == TensorType(DType.int64, [3])
        assert values["bf16"].type == TensorType(DType.bfloat16, [2])
        assert values["quantized"].type == TensorType(DType.uint8, [2, 144])
